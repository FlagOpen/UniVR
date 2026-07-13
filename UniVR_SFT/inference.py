# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inference Script

Supports two input modes:
1. "parquet" mode: reads pre-tokenized data from parquet files (cfg.parquet_path / cfg.PARQUET_CONFIGS)
2. "manual"  mode: reads user-specified prompts and reference image paths from cfg.prompts
                   Each prompt is a dict {"prompt": str, "reference_image": str | list[str]}.
                   When use_image=False, cfg.prompts may be a list of plain strings.

Mode is controlled by cfg.input_mode (or --mode on CLI).
"""

import argparse
import os
import io
import json
import torch
import importlib.util
import importlib.machinery
import sys
import os.path as osp
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from src.utils.model_utils import build_emu3p5_vllm
from src.utils.vllm_generation_utils import generate, generate_batch
from src.utils.generation_utils import multimodal_decode, decode_image
from src.utils.input_utils import build_image


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Script (parquet or manual prompts)")
    parser.add_argument("--cfg", required=True, type=str, help="Path to the config file")
    parser.add_argument("--tensor-parallel-size", default=8, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.7, type=float)
    parser.add_argument("--seed", default=6666, type=int)
    parser.add_argument("--guidance-scale", default=None, type=float,
                        help="Override guidance_scale from config. <=1.0 disables CFG for 2x speedup")
    parser.add_argument("--batch-size", default=16, type=int,
                        help="Batch size for inference. >1 enables batch processing (needs more VRAM)")

    # Data parallelism
    parser.add_argument("--num-parts", default=1, type=int, help="Total number of parts to split data into")
    parser.add_argument("--part-idx", default=0, type=int, help="Index of the current part (0-based)")

    # Sample size (parquet mode)
    parser.add_argument("--sample-size", default=1000, type=int, help="Number of samples to process (parquet mode)")

    # Mode selection
    parser.add_argument("--mode", default=None, choices=["parquet", "manual"],
                        help="Override cfg.input_mode. 'parquet' uses parquet files; 'manual' uses cfg.prompts.")

    # Multi-parquet support
    parser.add_argument("--parquet-paths", nargs="+", type=str, default=None,
                        help="One or more parquet file paths to process. Overrides cfg.parquet_path if provided.")

    # VQ decode device
    parser.add_argument("--vq-decode-device", default="cuda:0", type=str,
                        help="Device for VQ image decoding.")

    return parser.parse_args()


def load_config(cfg_path):
    """Load configuration from Python file."""
    cfg_path = Path(cfg_path).resolve()
    if str(cfg_path.parent) not in sys.path:
        sys.path.append(str(cfg_path.parent))

    loader = importlib.machinery.SourceFileLoader(cfg_path.stem, str(cfg_path))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(mod)
    return mod


# ====================== Data processing ======================

def process_data_parquet(cfg, seed=6666, sample_size=20, parquet_path=None):
    """
    Read a parquet file, sample rows, and format them for model inference.

    Expected new-format columns:
      - vlm_task_instruction:  task instruction text
      - problem_images:        list of dicts with [caption, frame_filename, image_tokens]
      - answer_images:         list of dicts with [caption, frame_filename, image_tokens]
      - problem_image_bytes:   ndarray of raw image bytes (for visualization, optional)
      - answer_image_bytes:    ndarray of raw image bytes (for visualization, optional)
      - height, width:         token grid dimensions

    Legacy format (problem_frames / answer_frames with image_bytes in frame dicts) is also supported.
    """
    if parquet_path is None:
        parquet_path = cfg.parquet_path
    print(f"[INFO] Loading parquet from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
        print(f"[INFO] Sampled {sample_size} rows with seed {seed}")
    else:
        print(f"[INFO] Dataframe has {len(df)} rows, using all.")

    has_new_cols = 'problem_images' in df.columns
    problem_col = 'problem_images' if has_new_cols else 'problem_frames'
    answer_col = 'answer_images' if has_new_cols else 'answer_frames'
    has_toplevel_bytes = 'problem_image_bytes' in df.columns and 'answer_image_bytes' in df.columns

    prompts = []
    for _, row in df.iterrows():
        question = row.get('vlm_task_instruction', None)
        if question is None or (isinstance(question, str) and question.strip() == ''):
            question = row.get('global_summary', '')

        if answer_col not in row.index and 'frames' in row.index:
            row_frames = row['frames']
            problem_frames = row_frames[:1]
            frames = row_frames[1:]
        else:
            frames = row[answer_col]
            problem_frames = row[problem_col]

        if isinstance(frames, np.ndarray):
            frames = frames.tolist()
        if isinstance(problem_frames, np.ndarray):
            problem_frames = problem_frames.tolist()

        frames.sort(key=lambda x: x.get('frame_filename', ''))

        image_str = ""
        problem_images_str = []
        h = row.get('height', 24)
        w = row.get('width', 24)
        for frame in problem_frames:
            tokens_bytes = frame['image_tokens']
            one_image_str = cfg.format_image_from_bytes(tokens_bytes, h, w)
            image_str += one_image_str
            problem_images_str.append(one_image_str)

        gt_answer_image_bytes = []
        gt_problem_image_bytes = []
        if has_toplevel_bytes:
            aib = row.get('answer_image_bytes', None)
            if aib is not None:
                gt_answer_image_bytes = [b for b in (aib.tolist() if isinstance(aib, np.ndarray) else aib) if b is not None]
            pib = row.get('problem_image_bytes', None)
            if pib is not None:
                gt_problem_image_bytes = [b for b in (pib.tolist() if isinstance(pib, np.ndarray) else pib) if b is not None]
        else:
            for frame in frames:
                if isinstance(frame, dict) and frame.get('image_bytes') is not None:
                    gt_answer_image_bytes.append(frame['image_bytes'])
            for frame in problem_frames:
                if isinstance(frame, dict) and frame.get('image_bytes') is not None:
                    gt_problem_image_bytes.append(frame['image_bytes'])

        prompts.append({
            "prompt": question,
            "image_str": image_str,
            "problem_images_str": problem_images_str,
            "gt_answer_image_bytes": gt_answer_image_bytes,
            "gt_problem_image_bytes": gt_problem_image_bytes,
        })

    return prompts


def process_data_manual(cfg, tokenizer, vq_model):
    """
    Build prompts directly from cfg.prompts. Each item is either:
      - dict {"prompt": str, "reference_image": str | list[str]}, when cfg.use_image is True
      - plain string, when cfg.use_image is False

    Each image is loaded from disk and tokenized via the VQ model on the fly.
    """
    use_image = getattr(cfg, 'use_image', True)
    raw_prompts = getattr(cfg, 'prompts', None)
    if not raw_prompts:
        raise ValueError("cfg.prompts is empty or undefined; manual mode requires cfg.prompts.")

    prompts = []
    for item in raw_prompts:
        if use_image:
            if not isinstance(item, dict):
                raise ValueError(f"Manual mode with use_image=True expects dict items, got: {type(item)}")
            question = item["prompt"]
            ref_imgs = item.get("reference_image", None)
            if isinstance(ref_imgs, str):
                ref_imgs = [ref_imgs]
            elif ref_imgs is None:
                ref_imgs = []

            image_str = ""
            problem_images_str = []
            gt_problem_image_bytes = []
            for img_path in ref_imgs:
                if not osp.isfile(img_path):
                    raise FileNotFoundError(f"Reference image not found: {img_path}")
                pil_img = Image.open(img_path).convert("RGB")
                one_image_str = build_image(pil_img, cfg, tokenizer, vq_model)
                image_str += one_image_str
                problem_images_str.append(one_image_str)
                with open(img_path, "rb") as fb:
                    gt_problem_image_bytes.append(fb.read())

            prompts.append({
                "prompt": question,
                "image_str": image_str,
                "problem_images_str": problem_images_str,
                "gt_answer_image_bytes": [],
                "gt_problem_image_bytes": gt_problem_image_bytes,
            })
        else:
            question = item if isinstance(item, str) else item.get("prompt", "")
            prompts.append({
                "prompt": question,
                "image_str": "",
                "problem_images_str": [],
                "gt_answer_image_bytes": [],
                "gt_problem_image_bytes": [],
            })
    return prompts


# ====================== Prompt template plumbing ======================

def prepare_batch_inputs(cfg, tokenizer, prompts_with_ids):
    """Tokenize prompts for batched vLLM generation."""
    batch_data = []
    for name, question_data in prompts_with_ids:
        if isinstance(question_data, dict):
            question = question_data["prompt"]
        else:
            question = question_data

        reference_images_to_decode = []
        if isinstance(question_data, dict) and "problem_images_str" in question_data:
            reference_images_to_decode = question_data["problem_images_str"]

        gt_answer_image_bytes = []
        gt_problem_image_bytes = []
        if isinstance(question_data, dict):
            gt_answer_image_bytes = question_data.get("gt_answer_image_bytes", [])
            gt_problem_image_bytes = question_data.get("gt_problem_image_bytes", [])

        prompt = cfg.template.format(question=question)
        image_str = question_data.get("image_str", "") if isinstance(question_data, dict) else ""
        prompt = prompt.replace("<|IMAGE|>", image_str)
        unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
            BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], dtype=input_ids.dtype)
            input_ids = torch.cat([BOS, input_ids], dim=1)

        unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

        batch_data.append({
            "name": name,
            "question": question,
            "input_ids": input_ids,
            "unconditional_ids": unconditional_ids,
            "reference_images_to_decode": reference_images_to_decode,
            "gt_answer_image_bytes": gt_answer_image_bytes,
            "gt_problem_image_bytes": gt_problem_image_bytes,
        })
    return batch_data


# ====================== Output saving ======================

def save_multimodal_output(mm_out, question, save_dir, name, tokenizer=None, vq_model=None,
                           reference_images_to_decode=None,
                           gt_answer_image_bytes=None, gt_problem_image_bytes=None):
    """
    Save multimodal output (generated images + text). Also persists input reference images
    (either from raw bytes or decoded from token strings) for visualization.
    """
    os.makedirs(save_dir, exist_ok=True)

    generated_images = []
    reference_images = []

    question_path = osp.join(save_dir, f"{name}_question.txt")
    with open(question_path, "w", encoding="utf-8") as f:
        f.write(question)

    img_idx = 0
    for t, c in mm_out:
        if t == "image":
            img_path = osp.join(save_dir, f"{name}_generated_{img_idx:03d}.png")
            c.save(img_path)
            generated_images.append(img_path)
            img_idx += 1
        elif t in ["text", "global_cot", "image_cot"]:
            text_path = osp.join(save_dir, f"{name}_{t}_{img_idx:03d}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(str(c))

    if gt_answer_image_bytes:
        for gt_idx, img_bytes in enumerate(gt_answer_image_bytes):
            Image.open(io.BytesIO(img_bytes)).save(osp.join(save_dir, f"{name}_gt_answer_{gt_idx:03d}.png"))

    if gt_problem_image_bytes:
        for gt_idx, img_bytes in enumerate(gt_problem_image_bytes):
            Image.open(io.BytesIO(img_bytes)).save(osp.join(save_dir, f"{name}_gt_problem_{gt_idx:03d}.png"))

    if reference_images_to_decode and tokenizer and vq_model:
        for ref_idx, img_str in enumerate(reference_images_to_decode):
            img = decode_image(img_str, tokenizer, vq_model)
            if img is not None:
                img_path = osp.join(save_dir, f"{name}_reference_image_{ref_idx:03d}.png")
                img.save(img_path)
                reference_images.append(img_path)

    return generated_images, reference_images


def save_raw_output(raw_string, save_dir, name):
    """[DEBUG] Save the raw decoded model output for inspection (visual tokens compacted)."""
    import re as _re

    os.makedirs(save_dir, exist_ok=True)

    def _compact_visual_tokens(s):
        def _replacer(m):
            tokens = _re.findall(r'<\|visual token (\d+)\|>', m.group(0))
            return f'[...{len(tokens)} visual tokens...]'
        return _re.sub(r'(?:<\|visual token \d+\|>\s*){2,}', _replacer, s)

    compact_raw = _compact_visual_tokens(raw_string)
    raw_path = osp.join(save_dir, f"{name}_raw_output.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(compact_raw)

    analysis_lines = [
        f"=== Raw Output Analysis for {name} ===",
        f"Total raw string length: {len(raw_string)} chars",
        "",
    ]
    img_pattern = _re.compile(r'<\|image start\|>(.*?)<\|image end\|>', _re.DOTALL)
    img_blocks = list(img_pattern.finditer(raw_string))
    analysis_lines.append(f"Total image blocks found: {len(img_blocks)}")
    analysis_lines.append("")
    for i, m in enumerate(img_blocks):
        block = m.group(1)
        header_match = _re.match(r'(.*?)<\|image token\|>', block, _re.DOTALL)
        resolution_str = header_match.group(1).strip() if header_match else "NOT FOUND"
        vis_tokens = _re.findall(r'<\|visual token (\d+)\|>', block)
        eol_count = block.count('<|extra_200|>')
        n_vis = len(vis_tokens)
        n_rows = eol_count
        n_cols = (n_vis // n_rows) if n_rows > 0 else n_vis
        analysis_lines += [
            f"--- Image #{i} ---",
            f"  Resolution header: '{resolution_str}'",
            f"  Visual tokens: {n_vis}",
            f"  EOL tokens: {eol_count}",
            f"  Inferred grid: {n_rows} rows x {n_cols} cols",
            f"  Inferred pixel size: {n_rows*16} x {n_cols*16}",
            "",
        ]

    text_segments = img_pattern.split(raw_string)
    text_only = [s.strip() for s in text_segments if s and '<|visual token' not in s and len(s.strip()) > 0]
    if text_only:
        analysis_lines.append(f"=== Text segments ({len(text_only)}) ===")
        for j, t in enumerate(text_only):
            display = t[:500] + '...' if len(t) > 500 else t
            analysis_lines.append(f"  [{j}] {display}")
            analysis_lines.append("")

    with open(osp.join(save_dir, f"{name}_raw_analysis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(analysis_lines))


# ====================== Inference loop ======================

def inference_and_save(cfg, model, tokenizer, vq_model, prompts_with_ids, batch_size=1):
    """Run inference on a list of prompts and save outputs."""
    save_path = cfg.save_path
    os.makedirs(f"{save_path}/images", exist_ok=True)
    results = {}

    if batch_size > 1:
        pending = []
        for name, question_data in prompts_with_ids:
            img_dir = f"{save_path}/images/{name}"
            if osp.exists(img_dir) and len(list(Path(img_dir).glob("*_generated_*.png"))) > 0:
                print(f"[WARNING] Result {name} already exists, skipping inference", flush=True)
                results[name] = img_dir
                continue
            pending.append((name, question_data))

        print(f"[INFO] Starting batch inference for {len(pending)} prompts (batch_size={batch_size})")

        for batch_start in tqdm(range(0, len(pending), batch_size), desc="Batch Inference"):
            batch_end = min(batch_start + batch_size, len(pending))
            batch_prompts = pending[batch_start:batch_end]

            torch.cuda.empty_cache()
            batch_data = prepare_batch_inputs(cfg, tokenizer, batch_prompts)
            batch_input_ids = [d["input_ids"] for d in batch_data]
            batch_unconditional_ids = [d["unconditional_ids"] for d in batch_data]

            try:
                results_list = generate_batch(cfg, model, tokenizer, batch_input_ids, batch_unconditional_ids)
                if len(results_list) != len(batch_data):
                    print(f"[WARNING] Results count ({len(results_list)}) != input count ({len(batch_data)})")
                    results_list = results_list[:len(batch_data)]

                for data, result_tokens in zip(batch_data, results_list):
                    result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                    img_dir = f"{save_path}/images/{data['name']}"
                    save_raw_output(result, img_dir, data['name'])
                    mm_out = multimodal_decode(result, tokenizer, vq_model)
                    gen_imgs, _ = save_multimodal_output(
                        mm_out, data["question"], img_dir, data["name"],
                        tokenizer, vq_model, data["reference_images_to_decode"],
                        gt_answer_image_bytes=data.get("gt_answer_image_bytes"),
                        gt_problem_image_bytes=data.get("gt_problem_image_bytes"),
                    )
                    results[data["name"]] = img_dir
                    print(f"[INFO] Saved {len(gen_imgs)} generated images for {data['name']}")

            except Exception as e:
                print(f"[ERROR] Batch generation failed: {e}")
                import traceback
                traceback.print_exc()
                for data in batch_data:
                    try:
                        for result_tokens in generate(cfg, model, tokenizer, data["input_ids"], data["unconditional_ids"]):
                            result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                            img_dir = f"{save_path}/images/{data['name']}"
                            save_raw_output(result, img_dir, data['name'])
                            mm_out = multimodal_decode(result, tokenizer, vq_model)
                            gen_imgs, _ = save_multimodal_output(
                                mm_out, data["question"], img_dir, data["name"],
                                tokenizer, vq_model, data["reference_images_to_decode"],
                                gt_answer_image_bytes=data.get("gt_answer_image_bytes"),
                                gt_problem_image_bytes=data.get("gt_problem_image_bytes"),
                            )
                            results[data["name"]] = img_dir
                    except Exception as e2:
                        print(f"[ERROR] Single inference also failed for {data['name']}: {e2}")
    else:
        print(f"[INFO] Starting single-sample inference for {len(prompts_with_ids)} prompts")

        for name, question_data in tqdm(prompts_with_ids, desc="Inference"):
            img_dir = f"{save_path}/images/{name}"
            if osp.exists(img_dir) and len(list(Path(img_dir).glob("*_generated_*.png"))) > 0:
                print(f"[WARNING] Result {name} already exists, skipping inference", flush=True)
                results[name] = img_dir
                continue

            torch.cuda.empty_cache()

            if isinstance(question_data, dict):
                question = question_data["prompt"]
            else:
                question = question_data

            reference_images_to_decode = []
            if isinstance(question_data, dict) and "problem_images_str" in question_data:
                reference_images_to_decode = question_data["problem_images_str"]

            gt_answer_image_bytes = []
            gt_problem_image_bytes = []
            if isinstance(question_data, dict):
                gt_answer_image_bytes = question_data.get("gt_answer_image_bytes", [])
                gt_problem_image_bytes = question_data.get("gt_problem_image_bytes", [])

            prompt = cfg.template.format(question=question)
            image_str = question_data.get("image_str", "") if isinstance(question_data, dict) else ""
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = cfg.unc_prompt.replace("<|IMAGE|>", image_str)

            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
                BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], dtype=input_ids.dtype)
                input_ids = torch.cat([BOS, input_ids], dim=1)

            unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

            try:
                for result_tokens in generate(cfg, model, tokenizer, input_ids, unconditional_ids):
                    result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                    save_raw_output(result, img_dir, name)
                    mm_out = multimodal_decode(result, tokenizer, vq_model)
                    gen_imgs, _ = save_multimodal_output(
                        mm_out, question, img_dir, name,
                        tokenizer, vq_model, reference_images_to_decode,
                        gt_answer_image_bytes=gt_answer_image_bytes,
                        gt_problem_image_bytes=gt_problem_image_bytes,
                    )
                    results[name] = img_dir
                    print(f"[INFO] Saved {len(gen_imgs)} generated images for {name}")

            except Exception as e:
                print(f"[ERROR] Failed to generate for {name}: {e}")
                import traceback
                traceback.print_exc()

    return results


# ====================== Main ======================

def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    # Determine input mode: CLI overrides cfg.input_mode (default 'parquet')
    mode = args.mode or getattr(cfg, 'input_mode', 'parquet')
    if mode not in ("parquet", "manual"):
        raise ValueError(f"Invalid input mode: {mode!r} (expected 'parquet' or 'manual')")
    print(f"[INFO] Input mode: {mode}")

    base_save_path = getattr(cfg, 'save_path', None) or osp.join(osp.dirname(args.cfg), "outputs")
    os.makedirs(base_save_path, exist_ok=True)
    print(f"[INFO] Base output directory: {base_save_path}")

    # ---- Load model ----
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else getattr(cfg, 'classifier_free_guidance', 3.0)
    print(f"[INFO] Using guidance_scale={guidance_scale} (CFG {'enabled' if guidance_scale > 1.0 else 'disabled'})")
    if args.guidance_scale is not None:
        cfg.classifier_free_guidance = args.guidance_scale

    batch_size = args.batch_size
    max_num_seqs = max(2, batch_size * 2) if guidance_scale > 1.0 else max(1, batch_size)

    print(f"[INFO] Loading model...")
    model, tokenizer, vq_model = build_emu3p5_vllm(
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.vq_path,
        vq_type=cfg.vq_type,
        vq_device=cfg.vq_device,
        seed=cfg.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        guidance_scale=guidance_scale,
        max_num_seqs=max_num_seqs,
        **getattr(cfg, "diffusion_decoder_kwargs", {}),
    )

    cfg.special_token_ids = {k: tokenizer.encode(v)[0] for k, v in cfg.special_tokens.items()}

    # ---- Dispatch by mode ----
    if mode == "manual":
        # Build prompts now while VQ is on its original device (needed for image encoding).
        print(f"[INFO] Building prompts from cfg.prompts ({len(getattr(cfg, 'prompts', []))} items)...")
        prompts = process_data_manual(cfg, tokenizer, vq_model)

        # After encoding, optionally move VQ model to the decode device.
        vq_decode_device = args.vq_decode_device
        if vq_decode_device != cfg.vq_device:
            print(f"[INFO] Moving VQ model from {cfg.vq_device} to {vq_decode_device} for decoding...")
            vq_model = vq_model.to(vq_decode_device)
            torch.cuda.empty_cache()

        cfg.save_path = base_save_path
        os.makedirs(cfg.save_path, exist_ok=True)
        print(f"[INFO] Output directory: {cfg.save_path}")

        prompts_with_ids = [(f"{idx:03d}", p) for idx, p in enumerate(prompts)]
        total = len(prompts_with_ids)
        chunk_size = (total + args.num_parts - 1) // args.num_parts
        start = args.part_idx * chunk_size
        end = min(start + chunk_size, total)
        my_prompts = prompts_with_ids[start:end]
        print(f"Worker {args.part_idx}/{args.num_parts} assigned {len(my_prompts)} prompts (indices {start}-{end-1})")

        if my_prompts:
            inference_and_save(cfg, model, tokenizer, vq_model, my_prompts, batch_size=args.batch_size)
        else:
            print("Nothing to do for inference.")
        print(f"[INFO] Worker {args.part_idx} finished.")
        return

    # ---- Parquet mode ----
    # Move VQ model to decode device now (no encoding needed for parquet input).
    vq_decode_device = args.vq_decode_device
    if vq_decode_device != cfg.vq_device:
        print(f"[INFO] Moving VQ model from {cfg.vq_device} to {vq_decode_device} for decoding...")
        vq_model = vq_model.to(vq_decode_device)
        torch.cuda.empty_cache()

    parquet_task_types = {}
    parquet_resolutions = {}

    if args.parquet_paths:
        parquet_paths = args.parquet_paths
    else:
        pq_configs = getattr(cfg, 'PARQUET_CONFIGS', None)
        if pq_configs and len(pq_configs) > 0:
            parquet_paths = [pc['path'] for pc in pq_configs]
            for pc in pq_configs:
                parquet_task_types[pc['path']] = pc.get('task_type', None)
                pc_th = pc.get('target_height', None)
                pc_tw = pc.get('target_width', None)
                if pc_th is not None and pc_tw is not None:
                    parquet_resolutions[pc['path']] = (pc_th, pc_tw)
            print(f"[INFO] Using PARQUET_CONFIGS from config ({len(pq_configs)} entries)")
        else:
            parquet_paths = [cfg.parquet_path]

    print(f"[INFO] Parquet files to process: {len(parquet_paths)}")
    for pp in parquet_paths:
        print(f"  - {pp}")

    for pq_idx, parquet_path in enumerate(parquet_paths):
        parquet_name = Path(parquet_path).stem
        print(f"\n{'='*60}")
        print(f"[{pq_idx+1}/{len(parquet_paths)}] Processing: {parquet_name}")
        print(f"  Path: {parquet_path}")
        print(f"{'='*60}")

        pq_task_type = parquet_task_types.get(parquet_path) or getattr(cfg, 'task_type', 'howto')
        cfg.unc_prompt, cfg.template = cfg.build_unc_and_template(
            pq_task_type, getattr(cfg, 'use_image', True)
        )
        print(f"[INFO] Using template for task_type='{pq_task_type}'")

        cfg.save_path = osp.join(base_save_path, parquet_name) if len(parquet_paths) > 1 else base_save_path
        os.makedirs(cfg.save_path, exist_ok=True)
        print(f"[INFO] Output directory: {cfg.save_path}")

        if parquet_path in parquet_resolutions:
            pq_th, pq_tw = parquet_resolutions[parquet_path]
            cfg.target_height = pq_th
            cfg.target_width = pq_tw
            print(f"[INFO] Per-parquet resolution: target_height={pq_th}, target_width={pq_tw}")
        else:
            cfg.target_height = getattr(cfg, '_default_target_height', getattr(cfg, 'target_height', None))
            cfg.target_width = getattr(cfg, '_default_target_width', getattr(cfg, 'target_width', None))

        prompts = process_data_parquet(cfg, seed=args.seed, sample_size=args.sample_size, parquet_path=parquet_path)
        prompts_with_ids = [(f"{idx:03d}", p) for idx, p in enumerate(prompts)]

        total = len(prompts_with_ids)
        chunk_size = (total + args.num_parts - 1) // args.num_parts
        start = args.part_idx * chunk_size
        end = min(start + chunk_size, total)
        my_prompts = prompts_with_ids[start:end]
        print(f"Worker {args.part_idx}/{args.num_parts} assigned {len(my_prompts)} prompts (indices {start}-{end-1})")

        if my_prompts:
            inference_and_save(cfg, model, tokenizer, vq_model, my_prompts, batch_size=args.batch_size)
        else:
            print(f"Nothing to do for inference on {parquet_name}.")

    print(f"[INFO] Worker {args.part_idx} finished.")


if __name__ == "__main__":
    main()
