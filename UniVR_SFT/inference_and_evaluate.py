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
Integrated Inference and Evaluation Script

This script combines:
1. Model inference from parquet data (from inference_vllm_dist.py)
2. Image generation and storage
3. Evaluation using Qwen3-VL API (from evaluate_emu_output.py)
4. Score recording and averaging

Reference task images for airplane/boat are supported.
"""

import argparse
import os
import re
import json
import base64
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
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utils
from src.utils.model_utils import build_emu3p5_vllm
from src.utils.vllm_generation_utils import generate, generate_batch
from src.utils.generation_utils import multimodal_decode, decode_image
# Try to import natsort for natural sorting
try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

# ============== Evaluation Configuration ==============
API_BASE = "http://<your-vlm-server>/v1"
API_KEY = "sk-abc123"
MODEL_NAME = "qwen3vl"

# Agibot folding task IDs
FOLDING_TASK_IDS = {361, 362, 444, 681, 658, 616, 599, 570, 561, 555, 520, 510, 477}


def get_task_category(task_name="", task_id="", question_text="", eval_source="", dataset_source=""):
    """
    Determine the task category based on task_name, task_id, question_text, eval_source, or dataset_source.
    Returns a category string or None if the task has no specific criteria.
    Supports both Agibot and EgoDex tasks.
    """
    # --- Zebra-CoT / VR-Bench reasoning tasks ---
    es_lower = str(eval_source).lower()
    tn_lower = str(task_name).lower().strip()
    ds_lower = str(dataset_source).lower().strip()

    # Count task (Zebra-CoT)
    if "multi-hop objects counting" in tn_lower or "objects counting" in tn_lower:
        return "zebra_count"
    # Jigsaw task (Zebra-CoT)
    if "visual jigsaw" in tn_lower or "jigsaw" in tn_lower:
        return "zebra_jigsaw"
    # Search task (Zebra-CoT / ThinkMorph)
    if "visual search" in tn_lower or "visual_search" in tn_lower:
        return "zebra_search"
    # Maze / Trapfield task (VR-Bench)
    if tn_lower in ("maze", "trapfield") or "vr-bench" in ds_lower:
        if tn_lower == "trapfield" or "trapfield" in ds_lower:
            return "vrbench_trapfield"
        return "vrbench_maze"

    # --- EgoDex tasks: match by eval_source or task_name ---

    # Direct match on EgoDex eval_source or task_name
    if "egodex" in es_lower or str(dataset_source).upper() == "EGODEX":
        if "assemble_disassemble_legos" in tn_lower or "assemble_disassemble_legos" in es_lower:
            return "egodex_legos"
        if "build_unstack_lego" in tn_lower or "build_unstack_lego" in es_lower:
            return "egodex_legos"
        if "assemble_disassemble_soft_legos" in tn_lower or "assemble_disassemble_soft_legos" in es_lower:
            return "egodex_legos"
        if "basic_fold" == tn_lower or "egodex_basic_fold" in es_lower:
            return "egodex_basic_fold"
        if "fold_unfold_paper_basic" in tn_lower or "fold_unfold_paper_basic" in es_lower:
            return "egodex_fold_paper"
        if "insert_remove_bookshelf" in tn_lower or "insert_remove_bookshelf" in es_lower:
            return "egodex_bookshelf"
        if "stack_unstack_bowls" in tn_lower or "stack_unstack_bowls" in es_lower:
            return "egodex_bowls"
        # Other EgoDex tasks: use generic
        return "egodex_generic"

    # --- Agibot tasks ---
    # Try to parse task_id as int for folding check
    try:
        tid = int(task_id)
    except (ValueError, TypeError):
        tid = None

    if tid is not None and tid in FOLDING_TASK_IDS:
        return "folding"

    # Combine task_name and question_text for keyword matching
    combined = (str(task_name) + " " + str(question_text)).lower()

    # Order matters: more specific patterns first
    if "pickup" in combined and "supermarket" in combined:
        return "pickup_supermarket"
    if ("produce section" in combined or "produce area" in combined) and "pickup" in combined:
        return "pickup_supermarket"
    if "open" in combined and "fridge" in combined and ("get food" in combined or "take" in combined):
        return "open_fridge"
    if "toast bread" in combined or ("toast" in combined and "bread" in combined and "toaster" in combined):
        return "toast_bread"
    if "take toast" in combined or "take" in combined and "toaster" in combined:
        return "take_toast"
    if "sort" in combined and "warehouse" in combined:
        return "sort_warehouse"
    if "fold" in combined or "unfold" in combined or "flatten" in combined or "stack" in combined:
        return "folding"
    if "clear" in combined and ("countertop" in combined or "counter" in combined) and "waste" in combined:
        return "clear_countertop"
    if ("dispose" in combined or "trash" in combined or "waste" in combined or "garbage" in combined) and ("desk" in combined or "table" in combined or "countertop" in combined):
        return "clear_countertop"
    if "boil water" in combined and "kettle" in combined:
        return "boil_water"
    if "open" in combined and "red wine" in combined:
        return "open_wine"
    if "inner pot" in combined and "rice cooker" in combined:
        return "rice_cooker"
    if "rice" in combined and "cooker" in combined and "pot" in combined:
        return "rice_cooker"
    if "wardrobe" in combined and "hang" in combined and "clothes" in combined:
        return "hang_wardrobe"
    if "open" in combined and "wardrobe" in combined and "hang" in combined:
        return "hang_wardrobe"
    if "swipe" in combined and ("card" in combined or "toy" in combined):
        return "swipe_cards"
    if "hang" in combined and "hanger" in combined:
        return "hang_hanger"
    if "hang clothes" in combined and "hanger" in combined:
        return "hang_hanger"

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Inference and Evaluation Script")
    parser.add_argument("--cfg", required=True, type=str, help="Path to the config file")
    parser.add_argument("--tensor-parallel-size", default=8, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.7, type=float)
    parser.add_argument("--seed", default=6666, type=int)
    parser.add_argument("--guidance-scale", default=None, type=float,
                        help="Override guidance_scale from config. Set to 1.0 or below to disable CFG for 2x speedup")
    parser.add_argument("--batch-size", default=16, type=int,
                        help="Batch size for inference. Set >1 for batch processing (requires more VRAM)")
    
    # Distribution args
    parser.add_argument("--num-parts", default=1, type=int, help="Total number of parts to split data into")
    parser.add_argument("--part-idx", default=0, type=int, help="Index of the current part (0-based)")
    
    # Sample size
    parser.add_argument("--sample-size", default=1000, type=int, help="Number of samples to process")
    
    # Evaluation options
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference and only run evaluation on existing outputs")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip evaluation and only run inference")
    
    # API configuration
    parser.add_argument("--api-base", default=API_BASE, type=str, help="Qwen3-VL API base URL")
    parser.add_argument("--api-key", default=API_KEY, type=str, help="API key")
    parser.add_argument("--model-name", default=MODEL_NAME, type=str, help="Model name for evaluation")
    parser.add_argument("--eval-workers", default=64, type=int,
                        help="Number of parallel workers for evaluation API requests")
    
    # Multi-parquet support
    parser.add_argument("--parquet-paths", nargs="+", type=str, default=None,
                        help="One or more parquet file paths to process. Overrides cfg.parquet_path if provided.")
    
    # VQ decode device (use CPU to avoid OOM when vLLM occupies GPU memory)
    parser.add_argument("--vq-decode-device", default="cuda:0", type=str,
                        help="Device for VQ image decoding. Default 'cpu' to avoid GPU OOM from vLLM memory reservation. "
                             "Set to 'cuda:0' etc. only if GPU has enough free memory.")
    
    args = parser.parse_args()
    return args


def load_config(cfg_path):
    """Load configuration from Python file."""
    cfg_path = Path(cfg_path).resolve()
    if str(cfg_path.parent) not in sys.path:
        sys.path.append(str(cfg_path.parent))
    
    loader = importlib.machinery.SourceFileLoader(cfg_path.stem, str(cfg_path))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(mod)
    return mod


def process_data(cfg, seed=6666, sample_size=20, parquet_path=None):
    """
    Reads parquet from parquet_path (or cfg.parquet_path), samples `sample_size` items with `seed`,
    and formats them for model inference.
    
    New parquet format columns:
      - vlm_task_instruction: task instruction text
      - problem_images: list of dicts with keys [caption, frame_filename, image_tokens]
      - answer_images:  list of dicts with keys [caption, frame_filename, image_tokens]
      - problem_image_bytes: ndarray of raw image bytes (for GT visualization)
      - answer_image_bytes:  ndarray of raw image bytes (for GT visualization)
      - height, width: token grid dimensions
    
    Also supports legacy format (problem_frames/answer_frames, image_bytes inside frame dicts).
    """
    if parquet_path is None:
        parquet_path = cfg.parquet_path
    print(f"[INFO] Loading parquet from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    # Randomly sample
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
        print(f"[INFO] Sampled {sample_size} rows with seed {seed}")
    else:
        print(f"[INFO] Dataframe has {len(df)} rows, using all.")
    
    # Detect column names (support both old and new naming conventions)
    has_new_cols = 'problem_images' in df.columns
    problem_col = 'problem_images' if has_new_cols else 'problem_frames'
    answer_col = 'answer_images' if has_new_cols else 'answer_frames'
    if has_new_cols:
        print(f"[INFO] Using new column names: {problem_col}, {answer_col}")
    else:
        print(f"[INFO] Using legacy column names: {problem_col}, {answer_col}")
    
    # Check for new top-level image bytes columns
    has_toplevel_bytes = 'problem_image_bytes' in df.columns and 'answer_image_bytes' in df.columns
    if has_toplevel_bytes:
        print(f"[INFO] Top-level image bytes columns detected (problem_image_bytes, answer_image_bytes)")
        
    prompts = []
    
    for idx, row in df.iterrows():
        # --- Instruction: priority order: vlm_task_instruction → question → global_summary ---
        question = row.get('vlm_task_instruction', None)
        if question is None or (isinstance(question, str) and question.strip() == ''):
            question = row.get('question', None)
        if question is None or (isinstance(question, str) and question.strip() == ''):
            question = row.get('global_summary', '')
        
        # --- Get problem and answer frames ---
        if answer_col not in row.index and 'frames' in row.index:
            # Legacy fallback: split frames column
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
        
        # --- Build tokenized image string from problem_images (model input) ---
        image_str = ""
        problem_images_str = []

        h = row.get('height', 24)
        w = row.get('width', 24)

        for frame in problem_frames:
            tokens_bytes = frame.get('image_tokens', None) if isinstance(frame, dict) else None
            if tokens_bytes is None or (isinstance(tokens_bytes, (bytes, bytearray)) and len(tokens_bytes) == 0):
                # Frame has no VQ tokens (e.g. action100m was not VQ-encoded); skip silently.
                continue
            # Use per-frame resolution when available (VR-X datasets store frame_height/frame_width).
            # Cast to int: parquet may store these as float64 (e.g. 32.0).
            fh = int(frame.get('frame_height') or h) if isinstance(frame, dict) else int(h)
            fw = int(frame.get('frame_width')  or w) if isinstance(frame, dict) else int(w)
            one_image_str = cfg.format_image_from_bytes(tokens_bytes, fh, fw)
            image_str += one_image_str
            problem_images_str.append(one_image_str)

        # --- Extract GT image bytes for evaluation / visualization ---
        gt_answer_image_bytes = []
        gt_problem_image_bytes = []
        
        if has_toplevel_bytes:
            # New format: use top-level byte arrays
            aib = row.get('answer_image_bytes', None)
            if aib is not None:
                if isinstance(aib, np.ndarray):
                    gt_answer_image_bytes = [b for b in aib if b is not None]
                elif isinstance(aib, list):
                    gt_answer_image_bytes = [b for b in aib if b is not None]
            
            pib = row.get('problem_image_bytes', None)
            if pib is not None:
                if isinstance(pib, np.ndarray):
                    gt_problem_image_bytes = [b for b in pib if b is not None]
                elif isinstance(pib, list):
                    gt_problem_image_bytes = [b for b in pib if b is not None]
        else:
            # Legacy format: extract image_bytes from frame dicts
            for frame in frames:
                if isinstance(frame, dict) and 'image_bytes' in frame and frame['image_bytes'] is not None:
                    gt_answer_image_bytes.append(frame['image_bytes'])
            for frame in problem_frames:
                if isinstance(frame, dict) and 'image_bytes' in frame and frame['image_bytes'] is not None:
                    gt_problem_image_bytes.append(frame['image_bytes'])

        # Extract eval_source and dataset_source for per-task evaluation
        eval_source = row.get('eval_source', '')
        dataset_source = row.get('dataset_source', '')
        task_name = row.get('task_name', '')
        task_id = row.get('task_id', '')

        prompts.append({
            "prompt": question,
            "image_str": image_str,
            "problem_images_str": problem_images_str,
            "gt_answer_image_bytes": gt_answer_image_bytes,
            "gt_problem_image_bytes": gt_problem_image_bytes,
            "eval_source": str(eval_source),
            "dataset_source": str(dataset_source),
            "task_name": str(task_name),
            "task_id": str(task_id),
        })
        
    has_gt = any(len(p["gt_answer_image_bytes"]) > 0 for p in prompts)
    print(f"[INFO] GT image_bytes available: {has_gt} (from parquet)")
    
    # Print eval_source distribution
    eval_sources = [p['eval_source'] for p in prompts if p.get('eval_source')]
    if eval_sources:
        from collections import Counter
        src_counts = Counter(eval_sources)
        print(f"[INFO] Eval source distribution ({len(src_counts)} tasks):")
        for src, cnt in sorted(src_counts.items()):
            print(f"  {src}: {cnt} samples")
    
    return prompts


def _get_template_for_sample(cfg, question_data):
    """Get the appropriate template and unc_prompt for a sample.

    Routing priority:
      1. dataset_source  → DATASET_TASK_TYPE_MAP
      2. eval_source     → EVAL_SOURCE_TASK_TYPE_MAP  (handles same-dataset_source ambiguity,
                           e.g. Zebra-CoT used for both count and jigsaw tasks)
      3. cfg.template / cfg.unc_prompt  (config-level default)
    """
    if isinstance(question_data, dict):
        task_type_templates = getattr(cfg, 'task_type_templates', {})

        # 1. Try dataset_source
        ds = question_data.get('dataset_source', '')
        task_type_map = getattr(cfg, 'DATASET_TASK_TYPE_MAP', {})
        if ds and ds in task_type_map:
            tt = task_type_map[ds]
            if tt in task_type_templates:
                return task_type_templates[tt]['template'], task_type_templates[tt]['unc_prompt']

        # 2. Try eval_source (finer-grained, overrides dataset_source ambiguity)
        es = question_data.get('eval_source', '')
        eval_source_map = getattr(cfg, 'EVAL_SOURCE_TASK_TYPE_MAP', {})
        if es and es in eval_source_map:
            tt = eval_source_map[es]
            if tt in task_type_templates:
                return task_type_templates[tt]['template'], task_type_templates[tt]['unc_prompt']

    # 3. Config-level default
    return cfg.template, cfg.unc_prompt


def prepare_batch_inputs(cfg, tokenizer, prompts_with_ids):
    """Prepare tokenized inputs for a batch of prompts."""
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
            
        # Get per-sample template based on dataset_source
        tmpl, unc_tmpl = _get_template_for_sample(cfg, question_data)
        
        # Prepare prompt (only problem frame tokens in image_str)
        prompt = tmpl.format(question=question)
        image_str = question_data.get("image_str", "") if isinstance(question_data, dict) else ""
        prompt = prompt.replace("<|IMAGE|>", image_str)
        unc_prompt = unc_tmpl.replace("<|IMAGE|>", image_str)

        # Tokenize 
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
            "eval_source": question_data.get("eval_source", "") if isinstance(question_data, dict) else "",
            "dataset_source": question_data.get("dataset_source", "") if isinstance(question_data, dict) else "",
        })
    return batch_data


import io

def save_multimodal_output(mm_out, question, save_dir, name, tokenizer=None, vq_model=None, reference_images_to_decode=None, gt_answer_image_bytes=None, gt_problem_image_bytes=None):
    """
    Save multimodal output (images and text) to disk.
    Also saves GT reference images from parquet (image_bytes) if available.
    Returns paths of generated images.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    generated_images = []
    reference_images = []
    
    # Save question text
    question_path = osp.join(save_dir, f"{name}_question.txt")
    with open(question_path, "w", encoding="utf-8") as f:
        f.write(question)
    
    # Save generated images
    img_idx = 0
    for t, c in mm_out:
        if t == "image":
            img_path = osp.join(save_dir, f"{name}_generated_{img_idx:03d}.png")
            c.save(img_path)
            generated_images.append(img_path)
            img_idx += 1
        elif t in ["text", "global_cot", "image_cot"]:
            # Save text content as well
            text_path = osp.join(save_dir, f"{name}_{t}_{img_idx:03d}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(str(c))
    
    # Save GT answer images from parquet (image_bytes) if available
    if gt_answer_image_bytes:
        for gt_idx, img_bytes in enumerate(gt_answer_image_bytes):
            gt_img = Image.open(io.BytesIO(img_bytes))
            gt_path = osp.join(save_dir, f"{name}_gt_answer_{gt_idx:03d}.png")
            gt_img.save(gt_path)
    
    # Save GT problem images from parquet (image_bytes) if available
    if gt_problem_image_bytes:
        for gt_idx, img_bytes in enumerate(gt_problem_image_bytes):
            gt_img = Image.open(io.BytesIO(img_bytes))
            gt_path = osp.join(save_dir, f"{name}_gt_problem_{gt_idx:03d}.png")
            gt_img.save(gt_path)
    
    # Decode and save reference images from token strings if provided (legacy fallback)
    if reference_images_to_decode and tokenizer and vq_model:
        ref_idx = 0
        for img_str in reference_images_to_decode:
            img = decode_image(img_str, tokenizer, vq_model)
            if img is not None:
                img_path = osp.join(save_dir, f"{name}_reference_image_{ref_idx:03d}.png")
                img.save(img_path)
                reference_images.append(img_path)
                ref_idx += 1
    
    return generated_images, reference_images


def save_raw_output(raw_string, save_dir, name):
    """
    [DEBUG] Save the raw decoded output string from the model for inspection.
    Saves two files:
      1. {name}_raw_output.txt  – full raw string (visual tokens replaced with placeholders for readability)
      2. {name}_raw_analysis.txt – structured analysis: per-image resolution, token counts, text segments
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Save full raw string (replace long visual token sequences with compact placeholders) ---
    import re as _re

    def _compact_visual_tokens(s):
        """Replace consecutive visual tokens with a count placeholder for readability."""
        def _replacer(m):
            tokens = _re.findall(r'<\|visual token (\d+)\|>', m.group(0))
            return f'[...{len(tokens)} visual tokens...]'
        # Match sequences of visual tokens (possibly separated by whitespace)
        return _re.sub(r'(?:<\|visual token \d+\|>\s*){2,}', _replacer, s)

    compact_raw = _compact_visual_tokens(raw_string)
    raw_path = osp.join(save_dir, f"{name}_raw_output.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(compact_raw)

    # --- 2. Structured analysis ---
    analysis_lines = []
    analysis_lines.append(f"=== Raw Output Analysis for {name} ===")
    analysis_lines.append(f"Total raw string length: {len(raw_string)} chars")
    analysis_lines.append("")

    # Find all image blocks: <|image start|> ... <|image end|>
    img_pattern = _re.compile(
        r'<\|image start\|>(.*?)<\|image end\|>',
        _re.DOTALL
    )
    img_blocks = list(img_pattern.finditer(raw_string))
    analysis_lines.append(f"Total image blocks found: {len(img_blocks)}")
    analysis_lines.append("")

    for i, m in enumerate(img_blocks):
        block = m.group(1)
        # Extract resolution header: everything before <|image token|>
        header_match = _re.match(r'(.*?)<\|image token\|>', block, _re.DOTALL)
        resolution_str = header_match.group(1).strip() if header_match else "NOT FOUND"

        # Count visual tokens
        vis_tokens = _re.findall(r'<\|visual token (\d+)\|>', block)
        # Count EOL tokens
        eol_count = block.count('<|extra_200|>')
        # Infer rows and cols
        n_vis = len(vis_tokens)
        n_rows = eol_count  # each row ends with an EOL (last row may or may not)
        n_cols = (n_vis // n_rows) if n_rows > 0 else n_vis

        analysis_lines.append(f"--- Image #{i} ---")
        analysis_lines.append(f"  Resolution header: '{resolution_str}'")
        analysis_lines.append(f"  Visual tokens: {n_vis}")
        analysis_lines.append(f"  EOL tokens: {eol_count}")
        analysis_lines.append(f"  Inferred grid: {n_rows} rows x {n_cols} cols")
        analysis_lines.append(f"  Inferred pixel size: {n_rows*16} x {n_cols*16}")
        analysis_lines.append("")

    # Find text segments (between image blocks, or before first / after last)
    text_segments = img_pattern.split(raw_string)
    text_only = [s.strip() for s in text_segments if s and '<|visual token' not in s and len(s.strip()) > 0]
    if text_only:
        analysis_lines.append(f"=== Text segments ({len(text_only)}) ===")
        for j, t in enumerate(text_only):
            # Truncate very long text segments
            display = t[:500] + '...' if len(t) > 500 else t
            analysis_lines.append(f"  [{j}] {display}")
            analysis_lines.append("")

    analysis_path = osp.join(save_dir, f"{name}_raw_analysis.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(analysis_lines))

    print(f"[DEBUG] Raw output saved to {raw_path}")
    print(f"[DEBUG] Analysis saved to {analysis_path}")


def inference_and_save(cfg, model, tokenizer, vq_model, prompts_with_ids, batch_size=1):
    """
    Run inference and save outputs as images.
    Only problem frame tokens are passed to vLLM. Answer frames are saved as GT for evaluation only.
    Returns a dictionary mapping sample names to their output directories.
    """
    save_path = cfg.save_path
    os.makedirs(f"{save_path}/images", exist_ok=True)
    
    results = {}  # name -> img_dir
    results_meta = {}  # name -> {"img_dir": ..., "eval_source": ..., "dataset_source": ..., "task_name": ...}
    
    if batch_size > 1:
        # Batch inference
        pending = []
        for name, question_data in prompts_with_ids:
            img_dir = f"{save_path}/images/{name}"
            # Store metadata for all samples (including skipped)
            if isinstance(question_data, dict):
                results_meta[name] = {
                    "img_dir": img_dir,
                    "eval_source": question_data.get("eval_source", ""),
                    "dataset_source": question_data.get("dataset_source", ""),
                    "task_name": question_data.get("task_name", ""),
                    "task_id": question_data.get("task_id", ""),
                }
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
                results_list = generate_batch(
                    cfg, model, tokenizer, batch_input_ids, batch_unconditional_ids
                )
                
                if len(results_list) != len(batch_data):
                    print(f"[WARNING] Results count ({len(results_list)}) != input count ({len(batch_data)})")
                    results_list = results_list[:len(batch_data)]
                
                for i, (data, result_tokens) in enumerate(zip(batch_data, results_list)):
                    result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                    # [DEBUG] Save raw model output for inspection
                    _debug_dir = f"{save_path}/images/{data['name']}"
                    save_raw_output(result, _debug_dir, data['name'])
                    mm_out = multimodal_decode(result, tokenizer, vq_model)
                    
                    img_dir = f"{save_path}/images/{data['name']}"
                    gen_imgs, ref_imgs = save_multimodal_output(
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
                # Fallback to single inference
                for data in batch_data:
                    try:
                        for result_tokens in generate(cfg, model, tokenizer, data["input_ids"], data["unconditional_ids"]):
                            result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                            # [DEBUG] Save raw model output for inspection
                            _debug_dir = f"{save_path}/images/{data['name']}"
                            save_raw_output(result, _debug_dir, data['name'])
                            mm_out = multimodal_decode(result, tokenizer, vq_model)
                            
                            img_dir = f"{save_path}/images/{data['name']}"
                            gen_imgs, ref_imgs = save_multimodal_output(
                                mm_out, data["question"], img_dir, data["name"],
                                tokenizer, vq_model, data["reference_images_to_decode"],
                                gt_answer_image_bytes=data.get("gt_answer_image_bytes"),
                                gt_problem_image_bytes=data.get("gt_problem_image_bytes"),
                            )
                            results[data["name"]] = img_dir
                    except Exception as e2:
                        print(f"[ERROR] Single inference also failed for {data['name']}: {e2}")
    else:
        # Single-sample inference
        print(f"[INFO] Starting single-sample inference for {len(prompts_with_ids)} prompts")
        
        for name, question_data in tqdm(prompts_with_ids, desc="Inference"):
            img_dir = f"{save_path}/images/{name}"
            # Store metadata for all samples
            if isinstance(question_data, dict):
                results_meta[name] = {
                    "img_dir": img_dir,
                    "eval_source": question_data.get("eval_source", ""),
                    "dataset_source": question_data.get("dataset_source", ""),
                    "task_name": question_data.get("task_name", ""),
                    "task_id": question_data.get("task_id", ""),
                }
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
                
            # Get per-sample template based on dataset_source
            tmpl, unc_tmpl = _get_template_for_sample(cfg, question_data)
            
            # Prepare prompt (only problem frame tokens in image_str)
            prompt = tmpl.format(question=question)
            image_str = question_data.get("image_str", "") if isinstance(question_data, dict) else ""
            prompt = prompt.replace("<|IMAGE|>", image_str)
            unc_prompt = unc_tmpl.replace("<|IMAGE|>", image_str)

            # Tokenize 
            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
                BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], dtype=input_ids.dtype)
                input_ids = torch.cat([BOS, input_ids], dim=1)

            unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

            try:
                for result_tokens in generate(cfg, model, tokenizer, input_ids, unconditional_ids):
                    result = tokenizer.decode(result_tokens, skip_special_tokens=False)
                    # [DEBUG] Save raw model output for inspection
                    save_raw_output(result, img_dir, name)
                    mm_out = multimodal_decode(result, tokenizer, vq_model)
                    
                    gen_imgs, ref_imgs = save_multimodal_output(
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
    
    return results, results_meta


# ============== Evaluation Functions ==============

def encode_image_to_base64(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_evaluation_prompt(question_text, eval_source=None, task_name=None, task_id=None):
    """
    Constructs the evaluation textual prompt.
    Now supports per-task Agibot evaluation criteria and outputs TWO scores:
      - TASK_SCORE: task completion & logic
      - VISUAL_SCORE: visual consistency & quality
    """
    
    # --- Determine Agibot task category ---
    task_category = get_task_category(
        task_name=task_name or "",
        task_id=task_id or "",
        question_text=question_text,
        eval_source=eval_source or "",
    )

    # --- Build task-specific instruction (Agibot tasks) ---
    AGIBOT_TASK_CRITERIA = {
        "pickup_supermarket": (
            "\n**Task Background**: This is a robot task of picking up items in a supermarket.\n"
            "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm successfully grasps the specified item "
            "and places it correctly into the shopping cart at the bottom of the image. If the item is not grasped, the wrong item is grasped, "
            "or the item is not placed into the cart, the task completion score should be significantly reduced.\n"
        ),
        "open_fridge": (
            "\n**Task Background**: This is a robot task of taking food out of a refrigerator.\n"
            "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm correctly retrieves items from the refrigerator "
            "and places them on the table with quantity and shape consistent with the GT. Inconsistent quantity, obvious shape changes, "
            "or failure to retrieve should significantly reduce the score.\n"
        ),
        "toast_bread": (
            "\n**Task Background**: This is a robot task of toasting bread.\n"
            "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm correctly places bread slices into the toaster. "
            "The score should be reduced if bread slices do not enter the toaster or are placed in obviously incorrect positions.\n"
        ),
        "take_toast": (
            "\n**Task Background**: This is a robot task of taking bread slices out of a toaster.\n"
            "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm retrieves both bread slices from the toaster. "
            "Insufficient quantity or failure to retrieve should reduce the score.\n"
        ),
        "sort_warehouse": (
            "\n**Task Background**: This is a robot task of sorting items in a warehouse.\n"
            "**Task Completion Evaluation Focus**: Focus on judging whether the target items are correctly transferred to the designated location. "
            "Items not reaching the target location or being placed in the wrong location should reduce the score.\n"
        ),
        "folding": (
            "\n**Task Background**: This is a robot task involving folding/unfolding clothes.\n"
            "**Task Completion Evaluation Focus**: Focus on the shape change process of the clothes. The folding process does not need to exactly match the GT; "
            "as long as there is no obvious distortion and the clothes can be seen progressively folding or unfolding toward the target state, it is acceptable. "
            "The final shape should be basically consistent with the GT target state. If the clothes are severely distorted or the final shape differs greatly from the GT, the score should be reduced.\n"
        ),
        "clear_countertop": (
            "\n**Task Background**: This is a robot task of clearing waste from a countertop.\n"
            "**Task Completion Evaluation Focus**: Focus on whether all items on the countertop have been placed into the trash bin, "
            "and whether the trash bin lid is finally closed. Missing items or an unclosed trash bin should reduce the score.\n"
        ),
        "boil_water": (
            "\n**Task Background**: This is a robot task of boiling water with a kettle.\n"
            "**Task Completion Evaluation Focus**: Focus on whether the kettle is placed under the faucet to fill with water, "
            "whether the lid is closed after filling, and whether the kettle is then returned to its original position. "
            "Any missing step (e.g., water not collected, lid not closed, kettle not returned) should reduce the score.\n"
        ),
        "open_wine": (
            "\n**Task Background**: This is a robot task of opening red wine.\n"
            "**Task Completion Evaluation Focus**: Focus on whether the mouth of the wine bottle is aligned with the mouth of the decanter, "
            "and whether the red wine is successfully poured to fill the decanter. Misalignment of the bottle mouth or failure to pour should reduce the score.\n"
        ),
        "rice_cooker": (
            "\n**Task Background**: This is a robot task of placing the inner pot into a rice cooker for cooking.\n"
            "**Task Completion Evaluation Focus**: Focus on whether the pot is placed into the rice cooker by the robotic arm, "
            "and whether the robotic arm then presses the button to start cooking. Failure to place the pot or press the button should reduce the score.\n"
        ),
        "hang_wardrobe": (
            "\n**Task Background**: This is a robot task of opening a wardrobe and hanging clothes.\n"
            "**Task Completion Evaluation Focus**: Focus on whether the clothes held by the robotic arm are correctly hung in the wardrobe, "
            "and whether the wardrobe door is finally closed. Failure to hang the clothes properly should reduce the score.\n"
        ),
        "swipe_cards": (
            "\n**Task Background**: This is a robot task of swiping a toy card.\n"
            "**Task Completion Evaluation Focus**: Focus on whether the robotic arm grips the white card "
            "and touches the card-swiping position while maintaining the grip. Failure to grip the card or touch the swiping position should reduce the score.\n"
        ),
        "hang_hanger": (
            "\n**Task Background**: This is a robot task of hanging clothes on a hanger.\n"
            "**Task Completion Evaluation Focus**: It is not necessary to strictly follow the GT's hanging steps, "
            "but the hanger should be progressively attached to the clothes step by step, ultimately presenting a shape similar to the GT. "
            "If the final shape differs greatly from the GT or serious anomalies occur during the process, the score should be reduced.\n"
        ),
    }

    # --- EgoDex task-specific criteria ---
    EGODEX_TASK_CRITERIA = {
        "egodex_legos": (
            "\n**Task Background**: This is an assembly or disassembly task involving Lego/soft Lego bricks (hand-operated).\n"
            "**Task Completion Evaluation Focus**: Pay close attention to whether the color and quantity of the bricks have changed unexpectedly. "
            "Even if the disassembly or assembly is completed, if the quantity or color of the bricks changes anomalously (appearing out of nowhere, disappearing, or changing color), a low score should be given. "
            "The final shape does not need to exactly match the GT; for example, a different building method that still satisfies the task instruction is acceptable.\n"
        ),
        "egodex_basic_fold": (
            "\n**Task Background**: This is a basic folding task (hand-operated).\n"
            "**Task Completion Evaluation Focus**: Focus on the shape change process of the target object. "
            "The folding process does not need to exactly match the GT; "
            "as long as there is no obvious distortion and the target object can be seen progressively folding or unfolding toward the target state, it is acceptable.\n"
        ),
        "egodex_fold_paper": (
            "\n**Task Background**: This is a paper folding/unfolding task (hand-operated).\n"
            "**Task Completion Evaluation Focus**: Focus on whether the paper deformation is reasonable, "
            "whether the fold lines are clear, whether the paper shape changes are coherent, and whether the paper progressively advances toward the target state. "
            "Unreasonable deformation, tearing, or a final state that differs greatly from the target should reduce the score.\n"
        ),
        "egodex_bookshelf": (
            "\n**Task Background**: This is a task of inserting or removing books from a bookshelf (hand-operated).\n"
            "**Task Completion Evaluation Focus**: Focus on whether the books are correctly inserted into or removed from the bookshelf, "
            "and ensure that the quantity and shape of the books do not change unexpectedly (they cannot appear or disappear out of nowhere or be deformed).\n"
        ),
        "egodex_bowls": (
            "\n**Task Background**: This is a bowl stacking or unstacking task (hand-operated).\n"
            "**Task Completion Evaluation Focus**: While judging whether the task instruction is completed, "
            "pay close attention to whether the quantity and color of the bowls have changed unexpectedly (they cannot appear or disappear out of nowhere or change color).\n"
        ),
        "egodex_generic": (
            "\n**Task Background**: This is a tabletop manipulation task (hand-operated).\n"
            "**Task Completion Evaluation Focus**: Judge whether the generated image sequence correctly executes the task instruction; "
            "the quantity, color, and shape of items should remain reasonable throughout the operation without obvious anomalous changes.\n"
        ),
    }

    # --- Zebra-CoT & VR-Bench reasoning task criteria ---
    REASONING_TASK_CRITERIA = {
        "zebra_count": (
            "\n**Task Background**: This is a multi-step object counting task in 3D visual reasoning. The model needs to progressively add or remove specific shapes/objects in the scene according to the instruction.\n"
            "**Task Completion Evaluation Focus**:\n"
            "- Check whether the generated image sequence **progressively** adds or removes the corresponding shapes/objects as instructed.\n"
            "- Whether the type (shape, color) and quantity of objects added or removed at each step are consistent with the instruction.\n"
            "- Whether the quantity of objects remaining in the scene in the **final step** image is correct.\n"
            "- The specific placement positions of shapes/objects are not important and do not need to exactly match the GT positions.\n"
            "- If the quantity change in intermediate steps is obviously wrong (e.g., 3 should be added but only 1 is added), the score should be significantly reduced.\n"
        ),
        "zebra_jigsaw": (
            "\n**Task Background**: This is a jigsaw completion task in 2D visual reasoning. Given an image with one piece missing and several candidate puzzle pieces, the model needs to select the correct piece and complete the image.\n"
            "**Task Completion Evaluation Focus**:\n"
            "- The model may complete the missing position directly on the original image, or generate a new complete image; both approaches are acceptable.\n"
            "- Focus on the **degree of similarity** between the completed image and the GT image.\n"
            "- The content of the completed area should come from one of the given candidate options; if the correct option is selected, the completion result should be highly consistent with the GT image.\n"
            "- If the content of the completed area clearly does not belong to any given option (e.g., fabricated content), a low score should be given.\n"
            "- If the completion result is visually highly consistent with the GT image, a high score should be given.\n"
        ),
        "zebra_search": (
            "\n**Task Background**: This is a visual search task in 2D visual reasoning. The model needs to find the target region in a given image that matches the question description.\n"
            "**Task Completion Evaluation Focus**:\n"
            "- Refer to the GT image to determine the correct target region. The GT image may **draw a bounding box/annotation** on the original image to identify the target, or it may **crop out the target sub-region**.\n"
            "- The model-generated image **must not directly copy the original input image**. If the generated image is almost identical to the original input with no annotation or cropping, it means the model has not completed the search task and should receive a **low score**.\n"
            "- The model must **explicitly express** the target sub-region indicated by the GT image through some means (bounding box, annotation, cropping, etc.).\n"
            "- The key evaluation question is: does the model-generated image correctly **locate and highlight the target region indicated by the GT**?\n"
            "- If the model correctly locates the target region and highlights it in some way, a high score should be given.\n"
            "- If the model fails to locate the target or simply outputs the input image unchanged, a low score should be given.\n"
        ),
        "vrbench_maze": (
            "\n**Task Background**: This is a maze-solving task. The model needs to generate a series of images showing the path planning process from start to finish.\n"
            "**Task Completion Evaluation Focus**:\n"
            "- Please judge based on the specific criteria given in each maze problem.\n"
            "- Focus on checking whether the intermediate process and final result comply with the maze rules (e.g., no passing through walls, moving along legal paths).\n"
            "- Whether the moving object (e.g., green circle, golf ball, etc.) slides along the white/legal path and finally stops at the target position.\n"
            "- The moving object must not pass through black wall areas.\n"
            "- If the path obviously passes through walls or the target position is not reached in the end, the score should be significantly reduced.\n"
        ),
        "vrbench_trapfield": (
            "\n**Task Background**: This is a Trapfield navigation task. The model needs to generate a series of images showing the safe navigation process in a scene containing traps.\n"
            "**Task Completion Evaluation Focus**:\n"
            "- Please judge based on the specific criteria given in each problem.\n"
            "- Focus on checking whether the navigation process avoids all trap areas and successfully reaches the target position.\n"
            "- Whether each step of the intermediate process is reasonable and compliant with the rules.\n"
            "- If a trap is touched or the target is not reached, the score should be significantly reduced.\n"
        ),
    }

    task_specific_instruction = ""

    # Match task-specific criteria: Agibot > EgoDex > Reasoning > Craft > fallback
    if task_category and task_category in AGIBOT_TASK_CRITERIA:
        task_specific_instruction = AGIBOT_TASK_CRITERIA[task_category]
    elif task_category and task_category in EGODEX_TASK_CRITERIA:
        task_specific_instruction = EGODEX_TASK_CRITERIA[task_category]
    elif task_category and task_category in REASONING_TASK_CRITERIA:
        task_specific_instruction = REASONING_TASK_CRITERIA[task_category]

    # --- Build the prompt: always request TWO scores ---
    system_prompt = (
        "You are a reward model for multimodal generative models used in reinforcement learning training. "
        "Your task is to evaluate the quality of the generated images and provide quantitative reward scores.\n"
        f"{task_specific_instruction}"
        "Evaluation criteria (please score each separately):\n"
        "1. **Task Completion and Logical Coherence**: Whether the multiple images output by the model form a correct reasoning and planning process, accurately execute the task instruction, and help achieve the task goal. "
        "If GT reference images are provided, focus on comparing whether the generated steps are consistent with the GT steps."
    )
    has_specific_criteria = (
        (task_category and task_category in AGIBOT_TASK_CRITERIA) or
        (task_category and task_category in EGODEX_TASK_CRITERIA) or
        (task_category and task_category in REASONING_TASK_CRITERIA)
    )
    if has_specific_criteria:
        system_prompt += " Please pay special attention to the evaluation focus described in the task background above."
    else:
        system_prompt += " Focus on whether the generated sequence correctly and coherently executes the task instruction."
    system_prompt += (
        "\n2. **Visual Consistency**: Whether the multiple images output by the model have spatial consistency, with the visual content (objects, background, etc.) remaining coherent throughout the sequence, "
        "without significant sudden changes or artifacts, and can be regarded as key frames of a continuous action. Objects and people/robotic arms in the images must not show obvious distortion or deformation. "
        "Evaluate visual coherence and visual quality.\n\n"
        "Output requirements:\n"
        "First provide a brief analysis, then in the **last part** of your response, strictly output two score values between 0.0 and 1.0 (floating point) in the following format:\n"
        "[[TASK_SCORE: 0.xx]]\n"
        "[[VISUAL_SCORE: 0.xx]]\n"
        "Note: Both score lines above must be output; neither can be omitted."
    )

    user_instruction = f"Task Instruction (User Question):\n{question_text}"

    return system_prompt, user_instruction


def evaluate_single_sample(client, model_name, data_dir, eval_source=None, task_name=None, task_id=None):
    """
    Evaluate a single sample using Qwen3-VL.
    Returns (score, response_text, task_score, visual_score) or (None, error_message, None, None) on failure.
    The returned score is the average of task_score and visual_score.
    """
    data_dir = Path(data_dir)

    # 1. Read Question Text
    question_files = list(data_dir.glob("*_question.txt"))
    if question_files:
        with open(question_files[0], "r", encoding="utf-8") as f:
            question_text = f.read().strip()
    else:
        question_text = "N/A"

    # 2. Identify Images
    all_images = list(data_dir.glob("*.png"))
    reference_images = []
    generated_images = []
    gt_answer_images = []
    gt_problem_images = []

    for img_path in all_images:
        if "gt_answer_" in img_path.name:
            gt_answer_images.append(img_path)
        elif "gt_problem_" in img_path.name:
            gt_problem_images.append(img_path)
        elif "reference_image" in img_path.name:
            reference_images.append(img_path)
        elif "generated" in img_path.name:
            generated_images.append(img_path)

    reference_images = natsorted(reference_images)
    generated_images = natsorted(generated_images)
    gt_answer_images = natsorted(gt_answer_images)
    gt_problem_images = natsorted(gt_problem_images)

    if len(generated_images) == 0:
        return None, "No generated images found", None, None

    # 3. Construct Message Payload
    system_text, instruction_text = get_evaluation_prompt(
        question_text,
        eval_source=eval_source, task_name=task_name, task_id=task_id
    )

    content_parts = []
    content_parts.append({"type": "text", "text": system_text + "\n\n" + instruction_text})

    # A0. GT Answer Images (ground truth answer sequence from parquet)
    if gt_answer_images:
        content_parts.append({"type": "text", "text": "\n\nBelow is the Ground Truth answer image sequence for this sample (GT Answer Image Sequence):"})
        for img_path in gt_answer_images:
            base64_img = encode_image_to_base64(img_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}"
                }
            })

    # B. Reference Images (from model output)
    if reference_images:
        content_parts.append({"type": "text", "text": "\n\nBelow are the reference images from the model input (Input Reference Images):"})
        for img_path in reference_images:
            base64_img = encode_image_to_base64(img_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}"
                }
            })

    # C. Generated Images
    content_parts.append({"type": "text", "text": "\n\nBelow is the model-generated image sequence to be evaluated (Model Generated Sequence):"})
    for img_path in generated_images:
        base64_img = encode_image_to_base64(img_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_img}"
            }
        })
    
    # D. Final request
    content_parts.append({"type": "text", "text": "\n\nPlease provide a detailed evaluation of the generated image sequence according to the above criteria:"})
    # 4. Call API
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user", 
                    "content": content_parts
                }
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        
        content = response.choices[0].message.content
        print(content)
        # Extract two scores: TASK_SCORE and VISUAL_SCORE
        task_match = re.search(r'\[\[TASK_SCORE:\s*([0-9.]+)\]\]', content)
        visual_match = re.search(r'\[\[VISUAL_SCORE:\s*([0-9.]+)\]\]', content)

        # Fallback: try old single-score format for backward compatibility
        single_match = re.search(r'\[\[SCORE:\s*([0-9.]+)\]\]', content)

        task_score = None
        visual_score = None

        if task_match:
            try:
                task_score = max(0.0, min(1.0, float(task_match.group(1))))
            except ValueError:
                pass
        if visual_match:
            try:
                visual_score = max(0.0, min(1.0, float(visual_match.group(1))))
            except ValueError:
                pass

        if task_score is not None and visual_score is not None:
            avg_score = (task_score + visual_score) / 2.0
            return avg_score, content, task_score, visual_score
        elif task_score is not None and visual_score is None:
            # Only task score found, use it as both
            return task_score, content, task_score, task_score
        elif task_score is None and visual_score is not None:
            # Only visual score found, use it as both
            return visual_score, content, visual_score, visual_score
        elif single_match:
            # Fallback: old [[SCORE: ...]] format
            try:
                score = max(0.0, min(1.0, float(single_match.group(1))))
                return score, content, score, score
            except ValueError:
                return None, f"Could not parse score as float: {content}", None, None
        else:
            return None, f"Could not extract score from response: {content}", None, None
        
    except Exception as e:
        return None, f"API call failed: {str(e)}", None, None


def evaluate_all_samples(output_dirs, api_base, api_key, model_name, save_path, parquet_name=None, results_meta=None, eval_workers=8):
    """
    Evaluate all samples and compute average score.
    Also computes per-task (eval_source) score breakdown.
    Returns (scores_dict, average_score).

    Args:
        results_meta: dict mapping sample name -> {"eval_source": ..., "task_name": ..., etc.}
                      Used for per-task score grouping.
    """
    if results_meta is None:
        results_meta = {}
    
    scores = {}
    failed_samples = []
    
    parquet_label = f" [{parquet_name}]" if parquet_name else ""
    print(f"\n[INFO] Starting evaluation{parquet_label} for {len(output_dirs)} samples...")

    num_workers = max(1, min(int(eval_workers), len(output_dirs))) if output_dirs else 1
    print(f"[INFO] Evaluation requests are running in parallel (workers={num_workers})")

    def _eval_one(name, data_dir):
        # Use a per-thread client to avoid any cross-thread client-state issues
        client = OpenAI(api_key=api_key, base_url=api_base)
        meta = results_meta.get(name, {})
        eval_source = meta.get("eval_source", "")
        task_name_meta = meta.get("task_name", "")
        task_id_meta = meta.get("task_id", "")

        score, response, task_score, visual_score = evaluate_single_sample(
            client, model_name, data_dir,
            eval_source=eval_source, task_name=task_name_meta, task_id=task_id_meta
        )
        return name, data_dir, eval_source, task_name_meta, task_id_meta, score, response, task_score, visual_score

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_eval_one, name, data_dir) for name, data_dir in output_dirs.items()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                name, data_dir, eval_source, task_name_meta, task_id_meta, score, response, task_score, visual_score = future.result()
            except Exception as e:
                failed_samples.append(("unknown", f"Worker failed: {str(e)}"))
                continue

            if score is not None:
                scores[name] = {
                    "score": score,
                    "task_score": task_score,
                    "visual_score": visual_score,
                    "response": response,
                    "data_dir": str(data_dir),
                    "eval_source": eval_source,
                    "task_name": task_name_meta,
                    "task_id": task_id_meta,
                }
                print(f"  {name} [{eval_source}]: avg={score:.3f} (task={task_score:.3f}, visual={visual_score:.3f})")
            else:
                failed_samples.append((name, response))
                print(f"  {name} [{eval_source}]: FAILED - {response[:100]}...")
    
    # Calculate overall statistics
    if scores:
        score_values = [s["score"] for s in scores.values()]
        avg_score = np.mean(score_values)
        std_score = np.std(score_values)
        min_score = np.min(score_values)
        max_score = np.max(score_values)
    else:
        avg_score = std_score = min_score = max_score = 0.0
    
    # ============== Per-task (eval_source) score breakdown ==============
    per_task_scores = {}  # eval_source -> list of (score, task_score, visual_score)
    for name, s_info in scores.items():
        es = s_info.get("eval_source", "unknown")
        if not es:
            es = "unknown"
        if es not in per_task_scores:
            per_task_scores[es] = {"scores": [], "task_scores": [], "visual_scores": []}
        per_task_scores[es]["scores"].append(s_info["score"])
        per_task_scores[es]["task_scores"].append(s_info.get("task_score", s_info["score"]))
        per_task_scores[es]["visual_scores"].append(s_info.get("visual_score", s_info["score"]))
    
    per_task_stats = {}
    for es, sdict in sorted(per_task_scores.items()):
        svals = sdict["scores"]
        tvals = sdict["task_scores"]
        vvals = sdict["visual_scores"]
        per_task_stats[es] = {
            "count": len(svals),
            "avg_score": float(np.mean(svals)),
            "avg_task_score": float(np.mean(tvals)),
            "avg_visual_score": float(np.mean(vvals)),
            "std_score": float(np.std(svals)),
            "min_score": float(np.min(svals)),
            "max_score": float(np.max(svals)),
        }
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "parquet_name": parquet_name,
        "total_samples": len(output_dirs),
        "successful_evaluations": len(scores),
        "failed_evaluations": len(failed_samples),
        "statistics": {
            "average_score": float(avg_score),
            "std_score": float(std_score),
            "min_score": float(min_score),
            "max_score": float(max_score),
        },
        "per_task_statistics": per_task_stats,
        "scores": scores,
        "failed_samples": [{"name": n, "error": e} for n, e in failed_samples],
    }
    
    suffix = f"_{parquet_name}" if parquet_name else ""
    results_path = osp.join(save_path, f"evaluation_results{suffix}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results Summary{parquet_label}")
    print(f"{'='*60}")
    print(f"Total Samples: {len(output_dirs)}")
    print(f"Successful: {len(scores)}")
    print(f"Failed: {len(failed_samples)}")
    print(f"Average Score: {avg_score:.4f}")
    print(f"Std Score: {std_score:.4f}")
    print(f"Min Score: {min_score:.4f}")
    print(f"Max Score: {max_score:.4f}")
    
    # Print per-task breakdown
    if per_task_stats:
        print(f"\n{'─'*60}")
        print(f"Per-Task Score Breakdown:")
        print(f"{'─'*60}")
        print(f"{'Task (eval_source)':<45} {'Count':>6} {'Avg':>8} {'Task':>8} {'Visual':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"{'─'*45} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for es, stats in sorted(per_task_stats.items()):
            print(f"{es:<45} {stats['count']:>6} {stats['avg_score']:>8.4f} {stats.get('avg_task_score', 0):>8.4f} {stats.get('avg_visual_score', 0):>8.4f} {stats['std_score']:>8.4f} {stats['min_score']:>8.4f} {stats['max_score']:>8.4f}")
        print(f"{'─'*45} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        print(f"{'OVERALL':<45} {len(score_values):>6} {avg_score:>8.4f} {'':>8} {'':>8} {std_score:>8.4f} {min_score:>8.4f} {max_score:>8.4f}")
    
    print(f"\nResults saved to: {results_path}")
    print(f"{'='*60}\n")
    
    return scores, avg_score


def generate_comparison_html(output_dirs, save_path, scores=None, parquet_name=None):
    """
    Generate a self-contained HTML file with base64-embedded images
    for side-by-side comparison of generated images vs GT images.
    Can be downloaded and opened directly in a local browser.

    Features:
      - Summary table with scores at the top (clickable navigation)
      - Side-by-side GT Answer vs Generated comparison per sample
      - Click any image to zoom (modal lightbox)
      - Sort / filter controls by score
    """
    from html import escape as html_escape

    label = parquet_name or "results"
    html_path = osp.join(save_path, f"comparison_{label}.html")

    # ---- collect per-sample data ----
    samples_data = []
    sorted_names = natsorted(output_dirs.keys())

    for name in sorted_names:
        data_dir = Path(output_dirs[name])
        if not data_dir.exists():
            continue

        question_text = "N/A"
        q_files = list(data_dir.glob("*_question.txt"))
        if q_files:
            with open(q_files[0], "r", encoding="utf-8") as f:
                question_text = f.read().strip()

        all_pngs = list(data_dir.glob("*.png"))
        gt_problem = natsorted([p for p in all_pngs if "gt_problem_" in p.name])
        gt_answer  = natsorted([p for p in all_pngs if "gt_answer_"  in p.name])
        generated  = natsorted([p for p in all_pngs if "generated"   in p.name])
        reference  = natsorted([p for p in all_pngs if "reference_image" in p.name])

        # Collect model-generated text files (text, global_cot, image_cot)
        all_txts = list(data_dir.glob("*.txt"))
        gen_texts = []
        for tf in natsorted(all_txts):
            # Skip question, raw_output, raw_analysis files
            if "_question.txt" in tf.name or "_raw_output.txt" in tf.name or "_raw_analysis.txt" in tf.name:
                continue
            # Match text / global_cot / image_cot files
            if "_text_" in tf.name or "_global_cot_" in tf.name or "_image_cot_" in tf.name:
                with open(tf, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    # Determine label from filename
                    if "_global_cot_" in tf.name:
                        lbl = "Global CoT"
                    elif "_image_cot_" in tf.name:
                        lbl = "Image CoT"
                    else:
                        lbl = "Text"
                    gen_texts.append({"label": lbl, "content": content, "file": tf.name})

        score_val = None
        if scores and name in scores:
            score_val = scores[name]["score"] if isinstance(scores[name], dict) else scores[name]

        samples_data.append({
            "name": name,
            "question": question_text,
            "gt_problem": gt_problem,
            "gt_answer": gt_answer,
            "generated": generated,
            "reference": reference,
            "gen_texts": gen_texts,
            "score": score_val,
        })

    # ---- helpers ----
    def _img_tag(path, label_text):
        b64 = encode_image_to_base64(path)
        suffix = Path(path).suffix.lower()
        mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
        src = f"data:{mime};base64,{b64}"
        return (
            f'<div class="img-box">'
            f'<img src="{src}" onclick="openModal(this.src)" loading="lazy" />'
            f'<div class="img-label">{html_escape(label_text)}</div>'
            f'</div>'
        )

    def _render_row(img_paths, prefix):
        if not img_paths:
            return '<span class="empty">—</span>'
        return "\n".join(_img_tag(p, f"{prefix} {i}") for i, p in enumerate(img_paths))

    # ---- build sample cards ----
    sample_blocks = []
    for sd in samples_data:
        name = sd["name"]
        score_val = sd["score"]

        # score badge
        if score_val is not None:
            color = "#27ae60" if score_val >= 0.7 else ("#f39c12" if score_val >= 0.4 else "#e74c3c")
            score_badge = f'<span class="score-badge" style="background:{color};">Score: {score_val:.3f}</span>'
        else:
            score_badge = '<span class="score-badge" style="background:#95a5a6;">N/A</span>'

        # max columns = max image count among GT answer and generated
        n_gt  = len(sd["gt_answer"])
        n_gen = len(sd["generated"])
        max_cols = max(n_gt, n_gen, 1)

        # build per-step comparison grid: row 0 = GT answer, row 1 = Generated
        step_cells_gt  = []
        step_cells_gen = []
        for ci in range(max_cols):
            if ci < n_gt:
                step_cells_gt.append(_img_tag(sd["gt_answer"][ci], f"GT-A {ci}"))
            else:
                step_cells_gt.append('<div class="img-box placeholder"></div>')
            if ci < n_gen:
                step_cells_gen.append(_img_tag(sd["generated"][ci], f"Gen {ci}"))
            else:
                step_cells_gen.append('<div class="img-box placeholder"></div>')

        comparison_html = (
            '<table class="compare-table"><tr>'
            '<td class="row-label gt-label">GT Answer</td>'
            + "".join(f"<td>{c}</td>" for c in step_cells_gt)
            + '</tr><tr>'
            '<td class="row-label gen-label">Generated</td>'
            + "".join(f"<td>{c}</td>" for c in step_cells_gen)
            + '</tr></table>'
        )

        # optional GT problem & reference rows
        extra_sections = ""
        if sd["gt_problem"]:
            extra_sections += (
                '<div class="extra-row">'
                '<span class="extra-title">GT Problem (Input):</span>'
                f'<div class="img-row">{_render_row(sd["gt_problem"], "GT-P")}</div>'
                '</div>'
            )
        if sd["reference"]:
            extra_sections += (
                '<div class="extra-row">'
                '<span class="extra-title">Reference (Decoded Token):</span>'
                f'<div class="img-row">{_render_row(sd["reference"], "Ref")}</div>'
                '</div>'
            )

        # Model generated text section
        if sd["gen_texts"]:
            text_blocks = []
            for gt_item in sd["gen_texts"]:
                text_blocks.append(
                    f'<div class="gen-text-block">'
                    f'<span class="gen-text-label">{html_escape(gt_item["label"])}</span>'
                    f'<pre class="gen-text-content">{html_escape(gt_item["content"])}</pre>'
                    f'</div>'
                )
            extra_sections += (
                '<div class="extra-row">'
                '<span class="extra-title">Model Generated Text:</span>'
                '<div class="gen-text-container">' + "\n".join(text_blocks) + '</div>'
                '</div>'
            )

        score_data = f'data-score="{score_val:.4f}"' if score_val is not None else 'data-score="-1"'

        sample_blocks.append(f"""
<div class="sample-card" id="sample-{html_escape(name)}" {score_data}>
  <div class="sample-header">
    <span class="sample-name">#{html_escape(name)}</span>
    {score_badge}
  </div>
  <div class="instruction"><b>Instruction:</b> {html_escape(question_text[:600] if len(sd["question"]) > 600 else sd["question"])}</div>
  <div class="compare-section">
    <div class="compare-title">⬇ Step-by-step: GT Answer vs Generated</div>
    {comparison_html}
  </div>
  {extra_sections}
</div>
""")

    # ---- summary stats ----
    total = len(samples_data)
    score_vals = [sd["score"] for sd in samples_data if sd["score"] is not None]
    avg_score = np.mean(score_vals) if score_vals else 0.0
    std_score = np.std(score_vals) if score_vals else 0.0

    # ---- index table rows ----
    index_rows = []
    for sd in samples_data:
        s = sd["score"]
        s_str = f"{s:.3f}" if s is not None else "N/A"
        color = ""
        if s is not None:
            color = "#27ae60" if s >= 0.7 else ("#f39c12" if s >= 0.4 else "#e74c3c")
        index_rows.append(
            f'<tr><td><a href="#sample-{html_escape(sd["name"])}">{html_escape(sd["name"])}</a></td>'
            f'<td style="color:{color};font-weight:600;">{s_str}</td>'
            f'<td>{len(sd["gt_answer"])}</td><td>{len(sd["generated"])}</td>'
            f'<td class="q-cell">{html_escape(sd["question"][:120])}</td></tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GT vs Generated: {html_escape(label)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
         background: #f0f2f5; color: #333; padding: 16px; }}
  h1 {{ text-align: center; margin: 12px 0 4px; font-size: 22px; }}
  .top-bar {{ text-align: center; color: #666; font-size: 13px; margin-bottom: 12px; }}
  .stats {{ display: flex; justify-content: center; gap: 24px; margin-bottom: 14px; }}
  .stat-box {{ background: #fff; padding: 10px 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
              text-align: center; min-width: 110px; }}
  .stat-box .val {{ font-size: 22px; font-weight: 700; }}
  .stat-box .lbl {{ font-size: 12px; color: #888; }}
  /* Controls */
  .controls {{ display: flex; gap: 12px; justify-content: center; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }}
  .controls select, .controls input {{ padding: 6px 10px; border-radius: 6px; border: 1px solid #ccc; font-size: 13px; }}
  /* Index table */
  .index-wrap {{ max-height: 300px; overflow-y: auto; margin: 0 auto 18px; max-width: 960px;
                background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .index-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .index-table th {{ position: sticky; top: 0; background: #2c3e50; color: #ecf0f1; padding: 8px; text-align: left; }}
  .index-table td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
  .index-table tr:hover {{ background: #f7f9fb; }}
  .index-table a {{ color: #2980b9; text-decoration: none; }}
  .index-table a:hover {{ text-decoration: underline; }}
  .q-cell {{ max-width: 400px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  /* Sample cards */
  .sample-card {{ background: #fff; border-radius: 8px; margin-bottom: 20px;
                 box-shadow: 0 1px 4px rgba(0,0,0,.1); overflow: hidden; }}
  .sample-header {{ padding: 10px 18px; background: #2c3e50; color: #ecf0f1;
                   display: flex; align-items: center; justify-content: space-between; }}
  .sample-name {{ font-size: 15px; font-weight: 600; }}
  .score-badge {{ padding: 3px 12px; border-radius: 12px; color: #fff; font-weight: 600; font-size: 13px; }}
  .instruction {{ padding: 10px 18px; line-height: 1.55; border-bottom: 1px solid #eee; font-size: 13px;
                 max-height: 90px; overflow-y: auto; }}
  .compare-section {{ padding: 10px 18px; }}
  .compare-title {{ font-weight: 700; font-size: 14px; color: #2c3e50; margin-bottom: 8px; }}
  .compare-table {{ border-collapse: collapse; }}
  .compare-table td {{ padding: 4px 6px; vertical-align: top; }}
  .row-label {{ writing-mode: vertical-lr; text-align: center; font-weight: 700; font-size: 13px;
               padding: 4px 8px !important; white-space: nowrap; border-right: 2px solid #ddd; }}
  .gt-label {{ color: #2980b9; }}
  .gen-label {{ color: #8e44ad; }}
  .img-box {{ display: inline-block; text-align: center; }}
  .img-box img {{ max-width: 240px; max-height: 240px; border-radius: 4px; border: 1px solid #ddd;
                 cursor: pointer; transition: transform .15s; display: block; }}
  .img-box img:hover {{ transform: scale(1.05); box-shadow: 0 2px 8px rgba(0,0,0,.15); }}
  .img-label {{ font-size: 10px; color: #999; margin-top: 2px; }}
  .placeholder {{ width: 120px; height: 120px; }}
  .empty {{ color: #bbb; font-style: italic; }}
  .extra-row {{ padding: 8px 18px; border-top: 1px dashed #e0e0e0; }}
  .extra-title {{ font-weight: 600; font-size: 13px; color: #555; margin-right: 8px; }}
  .img-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
  /* Generated text */
  .gen-text-container {{ margin-top: 6px; }}
  .gen-text-block {{ margin-bottom: 8px; }}
  .gen-text-label {{ display: inline-block; background: #8e44ad; color: #fff; font-size: 11px; font-weight: 600;
                    padding: 2px 8px; border-radius: 4px; margin-bottom: 3px; }}
  .gen-text-content {{ background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px 14px;
                      font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;
                      max-height: 300px; overflow-y: auto; font-family: 'Segoe UI', sans-serif; color: #333; }}
  /* Modal */
  .modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                   background: rgba(0,0,0,.85); z-index: 9999; justify-content: center; align-items: center; cursor: zoom-out; }}
  .modal-overlay.active {{ display: flex; }}
  .modal-overlay img {{ max-width: 92vw; max-height: 92vh; border-radius: 6px; }}
  /* Back to top */
  .back-top {{ position: fixed; bottom: 24px; right: 24px; background: #2c3e50; color: #fff;
              border: none; border-radius: 50%; width: 40px; height: 40px; font-size: 20px;
              cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,.3); display: none; z-index: 100; }}
</style>
</head>
<body>

<h1>🔍 GT vs Generated Comparison</h1>
<div class="top-bar">{html_escape(label)} &mdash; {total} samples &mdash; Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="stats">
  <div class="stat-box"><div class="val">{total}</div><div class="lbl">Samples</div></div>
  <div class="stat-box"><div class="val">{len(score_vals)}</div><div class="lbl">Evaluated</div></div>
  <div class="stat-box"><div class="val" style="color:#2980b9;">{avg_score:.4f}</div><div class="lbl">Avg Score</div></div>
  <div class="stat-box"><div class="val">{std_score:.4f}</div><div class="lbl">Std</div></div>
</div>

<div class="controls">
  <label>Sort:
    <select id="sortSel" onchange="sortCards()">
      <option value="name-asc">Name ↑</option>
      <option value="name-desc">Name ↓</option>
      <option value="score-asc">Score ↑ (low first)</option>
      <option value="score-desc">Score ↓ (high first)</option>
    </select>
  </label>
  <label>Min Score: <input id="minScore" type="number" step="0.05" min="0" max="1" value="0" onchange="filterCards()" style="width:70px;"></label>
  <label>Max Score: <input id="maxScore" type="number" step="0.05" min="0" max="1" value="1" onchange="filterCards()" style="width:70px;"></label>
</div>

<details open style="max-width:960px;margin:0 auto 18px;">
  <summary style="cursor:pointer;font-weight:600;font-size:14px;margin-bottom:6px;">📋 Sample Index (click to navigate)</summary>
  <div class="index-wrap">
    <table class="index-table">
      <thead><tr><th>Name</th><th>Score</th><th>#GT</th><th>#Gen</th><th>Instruction</th></tr></thead>
      <tbody>
        {"".join(index_rows)}
      </tbody>
    </table>
  </div>
</details>

<div id="cards-container">
{"".join(sample_blocks)}
</div>

<!-- Modal for zoom -->
<div class="modal-overlay" id="imgModal" onclick="closeModal()">
  <img id="modalImg" src="" />
</div>

<button class="back-top" id="backTop" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑</button>

<script>
function openModal(src) {{
  document.getElementById('modalImg').src = src;
  document.getElementById('imgModal').classList.add('active');
}}
function closeModal() {{
  document.getElementById('imgModal').classList.remove('active');
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeModal();
}});

// Back to top button
window.addEventListener('scroll', function() {{
  document.getElementById('backTop').style.display = window.scrollY > 400 ? 'block' : 'none';
}});

// Sort
function sortCards() {{
  var mode = document.getElementById('sortSel').value;
  var container = document.getElementById('cards-container');
  var cards = Array.from(container.querySelectorAll('.sample-card'));
  cards.sort(function(a, b) {{
    if (mode === 'score-asc' || mode === 'score-desc') {{
      var sa = parseFloat(a.dataset.score), sb = parseFloat(b.dataset.score);
      return mode === 'score-asc' ? sa - sb : sb - sa;
    }}
    var na = a.id, nb = b.id;
    return mode === 'name-asc' ? na.localeCompare(nb) : nb.localeCompare(na);
  }});
  cards.forEach(function(c) {{ container.appendChild(c); }});
}}

// Filter
function filterCards() {{
  var lo = parseFloat(document.getElementById('minScore').value) || 0;
  var hi = parseFloat(document.getElementById('maxScore').value) || 1;
  var cards = document.querySelectorAll('.sample-card');
  cards.forEach(function(c) {{
    var s = parseFloat(c.dataset.score);
    if (s < 0) {{ c.style.display = ''; return; }}
    c.style.display = (s >= lo && s <= hi) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(html_path) / (1024 * 1024)
    print(f"[INFO] Comparison HTML saved to: {html_path} ({size_mb:.1f} MB)")
    return html_path


def main():
    args = parse_args()
    
    # 1. Load Config
    cfg = load_config(args.cfg)
    
    # 2. Determine parquet paths and their task_type mappings
    # parquet_task_types maps parquet_path -> task_type for template selection
    parquet_task_types = {}
    # parquet_resolutions maps parquet_path -> (target_height, target_width) in pixels
    parquet_resolutions = {}
    
    if args.parquet_paths:
        parquet_paths = args.parquet_paths
    else:
        # Use PARQUET_CONFIGS from config if available, otherwise fall back to single parquet_path
        pq_configs = getattr(cfg, 'PARQUET_CONFIGS', None)
        if pq_configs and len(pq_configs) > 0:
            parquet_paths = [pc['path'] for pc in pq_configs]
            for pc in pq_configs:
                parquet_task_types[pc['path']] = pc.get('task_type', None)
                # Per-parquet resolution override (pixel dimensions)
                pc_th = pc.get('target_height', None)
                pc_tw = pc.get('target_width', None)
                if pc_th is not None and pc_tw is not None:
                    parquet_resolutions[pc['path']] = (pc_th, pc_tw)
            print(f"[INFO] Using PARQUET_CONFIGS from config ({len(pq_configs)} entries)")
        else:
            parquet_paths = [cfg.parquet_path]
    
    # Set up base save path
    base_save_path = getattr(cfg, 'save_path', None) or osp.join(osp.dirname(args.cfg), "outputs")
    os.makedirs(base_save_path, exist_ok=True)
    print(f"[INFO] Base output directory: {base_save_path}")
    print(f"[INFO] Parquet files to process: {len(parquet_paths)}")
    for pp in parquet_paths:
        print(f"  - {pp}")
    
    # 3. Load model once (if inference is needed)
    model = tokenizer = vq_model = None
    if not args.skip_inference:
        guidance_scale = args.guidance_scale if args.guidance_scale is not None else getattr(cfg, 'classifier_free_guidance', 3.0)
        print(f"[INFO] Using guidance_scale={guidance_scale} (CFG {'enabled' if guidance_scale > 1.0 else 'disabled'})")
        
        if args.guidance_scale is not None:
            cfg.classifier_free_guidance = args.guidance_scale
        
        batch_size = args.batch_size
        if guidance_scale > 1.0:
            max_num_seqs = max(2, batch_size * 2)
        else:
            max_num_seqs = max(1, batch_size)
        
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
        
        # Setup special tokens
        cfg.special_token_ids = {}
        for k, v in cfg.special_tokens.items():
            cfg.special_token_ids[k] = tokenizer.encode(v)[0]
        
        # Move VQ model to decode device (CPU by default) to avoid GPU OOM
        # vLLM pre-allocates GPU memory for KV cache, leaving little room for VQ decoding
        vq_decode_device = args.vq_decode_device
        if vq_decode_device != cfg.vq_device:
            print(f"[INFO] Moving VQ model from {cfg.vq_device} to {vq_decode_device} for decoding...")
            vq_model = vq_model.to(vq_decode_device)
            torch.cuda.empty_cache()
            print(f"[INFO] VQ model moved to {vq_decode_device}. GPU cache cleared.")
        
        print(f"[INFO] Model loaded successfully.")
    
    # 4. Process each parquet file
    all_parquet_results = {}
    
    for pq_idx, parquet_path in enumerate(parquet_paths):
        parquet_name = Path(parquet_path).stem
        print(f"\n{'='*60}")
        print(f"[{pq_idx+1}/{len(parquet_paths)}] Processing: {parquet_name}")
        print(f"  Path: {parquet_path}")
        print(f"{'='*60}")
        
        # Switch template/unc_prompt based on per-parquet task_type
        pq_task_type = parquet_task_types.get(parquet_path, None)
        task_type_templates = getattr(cfg, 'task_type_templates', {})
        if pq_task_type and pq_task_type in task_type_templates:
            cfg.template = task_type_templates[pq_task_type]['template']
            cfg.unc_prompt = task_type_templates[pq_task_type]['unc_prompt']
            print(f"[INFO] Using template for task_type='{pq_task_type}'")
            print(f"[INFO]   template: {cfg.template[:120]}...")
        else:
            # Use default template from config
            default_unc, default_tmpl = cfg.build_unc_and_template(getattr(cfg, 'task_type', 'vla'), getattr(cfg, 'use_image', True))
            cfg.template = default_tmpl
            cfg.unc_prompt = default_unc
            print(f"[INFO] Using default template (task_type='{getattr(cfg, 'task_type', 'vla')}')")
        
        # Set per-parquet save path (subdirectory if multiple parquets)
        if len(parquet_paths) > 1:
            cfg.save_path = osp.join(base_save_path, parquet_name)
        else:
            cfg.save_path = base_save_path
        os.makedirs(cfg.save_path, exist_ok=True)
        print(f"[INFO] Output directory: {cfg.save_path}")
        
        # Per-parquet resolution override (for vLLM parse_customized_hw)
        if parquet_path in parquet_resolutions:
            pq_th, pq_tw = parquet_resolutions[parquet_path]
            cfg.target_height = pq_th
            cfg.target_width = pq_tw
            print(f"[INFO] Per-parquet resolution: target_height={pq_th}, target_width={pq_tw} pixels"
                  f" -> token grid {pq_th//16}*{pq_tw//16}")
        else:
            # Use global default from config (or None)
            cfg.target_height = getattr(cfg, '_default_target_height', getattr(cfg, 'target_height', None))
            cfg.target_width = getattr(cfg, '_default_target_width', getattr(cfg, 'target_width', None))
            if cfg.target_height:
                print(f"[INFO] Using global resolution: target_height={cfg.target_height}, target_width={cfg.target_width}")
            else:
                print(f"[WARN] No resolution constraint! Model will freely generate resolution tokens.")
        
        output_dirs = {}
        results_meta = {}  # name -> {eval_source, dataset_source, task_name, ...}
        
        if not args.skip_inference:
            # Process data from this parquet
            prompts = process_data(cfg, seed=args.seed, sample_size=args.sample_size, parquet_path=parquet_path)
            
            # Enumerate globally
            prompts_with_ids = [(f"{idx:03d}", p) for idx, p in enumerate(prompts)]
            
            # Filter for this worker (Data Parallelism)
            total = len(prompts_with_ids)
            chunk_size = (total + args.num_parts - 1) // args.num_parts
            start = args.part_idx * chunk_size
            end = min(start + chunk_size, total)
            
            my_prompts = prompts_with_ids[start:end]
            
            print(f"Worker {args.part_idx}/{args.num_parts} assigned {len(my_prompts)} prompts (indices {start}-{end-1})")
            
            if len(my_prompts) == 0:
                print(f"Nothing to do for inference on {parquet_name}.")
            else:
                output_dirs, results_meta = inference_and_save(
                    cfg, model, tokenizer, vq_model, my_prompts,
                    batch_size=args.batch_size,
                )
                print(f"[INFO] Inference completed for {parquet_name}. Generated outputs for {len(output_dirs)} samples.")
                
                # Save results_meta to disk for later use when skip_inference
                meta_path = osp.join(cfg.save_path, "results_meta.json")
                meta_serializable = {k: v for k, v in results_meta.items()}
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_serializable, f, ensure_ascii=False, indent=2)
                print(f"[INFO] Saved results_meta to {meta_path}")
        else:
            print(f"[INFO] Skipping inference for {parquet_name}, loading existing outputs...")
            images_dir = osp.join(cfg.save_path, "images")
            if osp.exists(images_dir):
                for name in os.listdir(images_dir):
                    name_dir = osp.join(images_dir, name)
                    if osp.isdir(name_dir):
                        output_dirs[name] = name_dir
            print(f"[INFO] Found {len(output_dirs)} existing output directories.")
            
            # Try to load results_meta from disk
            meta_path = osp.join(cfg.save_path, "results_meta.json")
            if osp.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    results_meta = json.load(f)
                print(f"[INFO] Loaded results_meta from {meta_path} ({len(results_meta)} entries)")
            else:
                print(f"[INFO] No results_meta.json found. Per-task breakdown may not be available.")
        
        # Run Evaluation for this parquet
        if not args.skip_evaluation and len(output_dirs) > 0:
            scores, avg_score = evaluate_all_samples(
                output_dirs,
                api_base=args.api_base,
                api_key=args.api_key,
                model_name=args.model_name,
                save_path=cfg.save_path,
                parquet_name=parquet_name,
                results_meta=results_meta,
                eval_workers=args.eval_workers,
            )
            all_parquet_results[parquet_name] = {
                "avg_score": avg_score,
                "num_samples": len(scores),
                "scores": scores,
            }
            print(f"[RESULT] {parquet_name}: Average Score = {avg_score:.4f} ({len(scores)} samples)")
        elif args.skip_evaluation:
            print(f"[INFO] Evaluation skipped for {parquet_name}.")
        else:
            print(f"[WARNING] No outputs to evaluate for {parquet_name}.")
        
        # Generate comparison HTML for this parquet
        if len(output_dirs) > 0:
            eval_scores = all_parquet_results.get(parquet_name, {}).get("scores", None)
            generate_comparison_html(
                output_dirs, cfg.save_path,
                scores=eval_scores, parquet_name=parquet_name,
            )
    
    # 5. Print final summary across all parquets (with per-task breakdown)
    if len(all_parquet_results) > 0:
        print(f"\n{'='*60}")
        print(f"{'='*60}")
        print(f"  FINAL SUMMARY - All Parquets")
        print(f"{'='*60}")
        all_score_values = []
        all_per_task = {}  # Aggregate per-task scores across all parquets
        
        for pname, presult in all_parquet_results.items():
            print(f"  {pname}: avg_score={presult['avg_score']:.4f} ({presult['num_samples']} samples)")
            for s_name, s_info in presult["scores"].items():
                score_val = s_info["score"] if isinstance(s_info, dict) else s_info
                all_score_values.append(score_val)
                es = s_info.get("eval_source", "unknown") if isinstance(s_info, dict) else "unknown"
                if not es:
                    es = "unknown"
                if es not in all_per_task:
                    all_per_task[es] = []
                all_per_task[es].append(score_val)
        
        if all_score_values:
            overall_avg = np.mean(all_score_values)
            print(f"  {'─'*40}")
            print(f"  Overall Average: {overall_avg:.4f} ({len(all_score_values)} total samples)")
        
        # Print per-task breakdown across all parquets
        if all_per_task and len(all_per_task) > 1:
            print(f"\n  {'─'*60}")
            print(f"  Per-Task Score Breakdown (All Parquets):")
            print(f"  {'─'*60}")
            print(f"  {'Task (eval_source)':<45} {'Count':>6} {'Avg':>8}")
            print(f"  {'─'*45} {'─'*6} {'─'*8}")
            for es, svals in sorted(all_per_task.items()):
                print(f"  {es:<45} {len(svals):>6} {np.mean(svals):>8.4f}")
            print(f"  {'─'*45} {'─'*6} {'─'*8}")
            if all_score_values:
                print(f"  {'OVERALL':<45} {len(all_score_values):>6} {overall_avg:>8.4f}")
        
        print(f"{'='*60}\n")
        
        # Save summary JSON with per-task breakdown
        summary = {
            "timestamp": datetime.now().isoformat(),
            "parquet_results": {},
            "per_task_results": {},
        }
        for pname, presult in all_parquet_results.items():
            summary["parquet_results"][pname] = {
                "avg_score": float(presult["avg_score"]),
                "num_samples": presult["num_samples"],
            }
        for es, svals in sorted(all_per_task.items()):
            summary["per_task_results"][es] = {
                "count": len(svals),
                "avg_score": float(np.mean(svals)),
                "std_score": float(np.std(svals)),
                "min_score": float(np.min(svals)),
                "max_score": float(np.max(svals)),
            }
        if all_score_values:
            summary["overall_avg_score"] = float(overall_avg)
        summary_path = osp.join(base_save_path, "evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Summary saved to: {summary_path}")
    
    print(f"[INFO] Worker {args.part_idx} finished.")


if __name__ == "__main__":
    main()
