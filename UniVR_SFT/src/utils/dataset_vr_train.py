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

import os
import io
import glob
import random
import traceback
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from src.utils.input_utils import format_image_string
from src.vision_tokenizer import build_vision_tokenizer

# ============================================================
# Constants
# ============================================================
VR_TRAIN_ROOT = "./datasets/VR-X"  # root directory containing all dataset subdirectories

# Dataset name → subdirectory under VR_TRAIN_ROOT
DATASET_DIRS = {
    "Action100":       "Action100",
    "Agibot":          "Agibot",
    "Bridge":          "Bridge",
    "Droid":           "Droid",   # default subdir
    "EgoDex":          "EgoDex",
    "Epic_Kitchen":    "Epic_Kitchen",
    "VideoCraftBench": "VideoCraftBench",
    "Zebra_Count":     "Zebra_Count",
    "Zebra_Robot":     "Zebra_Robot",
    "Zebra_Jigsaw":    "Zebra_Jigsaw",
    "Zebra_Search":    "Zebra_Search",
    "Maze":            "Maze",
    "Visual_Search":   "Visual_Search",
    "Spatial_Navigation":           "Spatial_Navigation",
    "Spatial_Navigation_maze":      "Spatial_Navigation_maze",
    "Spatial_Navigation_trapfield": "Spatial_Navigation_trapfield",
    "Visual_CoT_GQA": "Visual_CoT_GQA"
}

# Per-dataset user prompt templates
PROMPT_TEMPLATES = {
    "Action100":       "You are a helpful assistant for howto task. USER: Given the first frame, how to perform the following task? {summary}",
    "Agibot":          "You are a helpful assistant for vla task. USER: Given the first frame, how to perform the following task? {summary}",
    "Bridge":          "You are a helpful assistant for vla task. USER: Given the first frame, how to perform the following task? {summary}",
    "Droid":           "You are a helpful assistant for vla task. USER: Given the first frame, how to perform the following task? {summary}",
    "EgoDex":          "You are a helpful assistant for first-person perspective howto task. USER: Given the first frame, how to perform the following task? {summary}",
    "Epic_Kitchen":    "You are a helpful assistant for first-person perspective howto task. USER: Given the first frame, how to perform the following task? {summary}",
    "VideoCraftBench": "You are a helpful assistant for first-person perspective howto task. USER: Given the first frame, how to perform the following task? {summary}",
    "Zebra_Count":     "You are a helpful assistant for visual reasoning task. USER: Given the first frame, how to perform the following task? {summary}",
    "Zebra_Robot":     "You are a helpful assistant for vla task. USER: Given the first frame, how to perform the following task? {summary}",
    "Zebra_Jigsaw":    "You are a helpful assistant for visual reasoning task. USER: Given the first frame, how to perform the following task? {summary}",
    "Zebra_Search":    "You are a helpful assistant for visual reasoning task. USER: Given the first frame, how to perform the following task? {summary}",
    "Visual_Search":   "You are a helpful assistant for visual reasoning task. USER: Given the first frame, how to perform the following task? {summary}",
    "Spatial_Navigation":           "You are a helpful assistant for visual reasoning task. USER: Given the first frame, how to perform the following task? {summary}",
    "Spatial_Navigation_maze":      "You are a helpful assistant for visual reasoning task. USER: Given the first frame, how to perform the following task? {summary}",
    "Spatial_Navigation_trapfield": "You are a helpful assistant for visual reasoning task. USER: Given the first frame, how to perform the following task? {summary}",
    "Visual_CoT_GQA":  "You are a helpful assistant for visual reasoning task. USER: Given the first frame, how to perform the following task? {summary}",
}

# Unconditional prompt template for CFG (Classifier-Free Guidance) training
# With cfg_prob probability, the conditional prompt is replaced with this generic one
# (no task description / summary), while problem images are still included.
UNCOND_PROMPT_TEMPLATE = "You are a helpful assistant. USER: "

# Datasets that have per-frame captions (vlm_frame_captions) and support
# interleaved text-image format in the answer sequence.
INTERLEAVED_TEXT_DATASETS = {"Action100", "Agibot", "Droid", "EgoDex", "Epic_Kitchen", "Zebra_Robot", "Visual_Search", "Spatial_Navigation", "Spatial_Navigation_maze", "Spatial_Navigation_trapfield", "Zebra_Count", "VideoCraftBench", "Visual_CoT_GQA"}


# ============================================================
# Visualization utilities
# ============================================================
def decode_image_bytes(image_bytes: bytes) -> Optional[Image.Image]:
    """Decode raw RGB/JPEG bytes into a PIL Image."""
    if image_bytes is None:
        return None
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def visualize_sample(sample: dict, output_dir: str = "./vis_debug", sample_idx: int = 0):
    """
    Save problem/answer frame images from a raw parquet sample to disk for debugging.
    
    Args:
        sample: a single row dict from the parquet DataFrame
        output_dir: directory to save visualization images
        sample_idx: index for naming
    """
    os.makedirs(output_dir, exist_ok=True)
    ds_name = sample.get("dataset_source", "unknown")
    ep_id = sample.get("episode_id", sample_idx)
    
    prefix = f"{ds_name}_{ep_id}"
    saved = []

    # --- Problem image bytes ---
    problem_bytes_list = sample.get("problem_image_bytes", None)
    if problem_bytes_list is not None:
        for i, img_bytes in enumerate(problem_bytes_list):
            img = decode_image_bytes(img_bytes)
            if img is not None:
                path = os.path.join(output_dir, f"{prefix}_problem_{i:03d}.jpg")
                img.save(path)
                saved.append(path)

    # --- Answer image bytes ---
    answer_bytes_list = sample.get("answer_image_bytes", None)
    if answer_bytes_list is not None:
        for i, img_bytes in enumerate(answer_bytes_list):
            img = decode_image_bytes(img_bytes)
            if img is not None:
                path = os.path.join(output_dir, f"{prefix}_answer_{i:03d}.jpg")
                img.save(path)
                saved.append(path)

    return saved


def visualize_sample_grid(sample: dict, output_path: str = "./vis_debug/grid.jpg",
                          max_cols: int = 8, thumb_size: int = 256):
    """
    Create a single grid image showing all problem + answer frames side by side.
    Problem frames get a green border, answer frames get a blue border.
    
    Args:
        sample: a single row dict from the parquet DataFrame
        output_path: path to save the grid image
        max_cols: maximum columns in the grid
        thumb_size: thumbnail size for each frame
    """
    from PIL import ImageDraw
    
    frames = []  # list of (PIL.Image, label_str, color)
    
    problem_bytes = sample.get("problem_image_bytes", []) 
    answer_bytes = sample.get("answer_image_bytes", []) 
    
    for i, b in enumerate(problem_bytes):
        img = decode_image_bytes(b)
        if img:
            frames.append((img, f"P{i}", "green"))
    
    for i, b in enumerate(answer_bytes):
        img = decode_image_bytes(b)
        if img:
            frames.append((img, f"A{i}", "blue"))
    
    if not frames:
        print("No image bytes to visualize.")
        return None
    
    n = len(frames)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    
    cell_size = thumb_size + 4  # 2px border on each side
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size + 20 * rows), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    for idx, (img, label, color) in enumerate(frames):
        r, c = divmod(idx, cols)
        img_thumb = img.copy()
        img_thumb.thumbnail((thumb_size, thumb_size))
        x = c * cell_size + 2
        y = r * (cell_size + 20) + 2
        grid.paste(img_thumb, (x, y))
        # Draw border
        draw.rectangle([x - 2, y - 2, x + thumb_size + 1, y + thumb_size + 1], outline=color, width=2)
        # Draw label
        draw.text((x, y + thumb_size + 2), label, fill=color)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grid.save(output_path)
    print(f"Grid saved to {output_path}  ({len(frames)} frames)")
    return output_path


def decode_vq_tokens(vq_model, token_array: np.ndarray, h: int, w: int) -> Optional[Image.Image]:
    """
    Decode VQ token indices back into a PIL image using the VQ model.
    
    Args:
        vq_model: vision tokenizer model with decode_code method
        token_array: numpy array of shape (h, w) with token indices
        h: token grid height
        w: token grid width
    Returns:
        PIL Image or None on failure
    """
    if vq_model is None:
        return None
    try:
        indices = torch.from_numpy(token_array.flatten().astype(np.int64)).cuda()
        shape = (1, h, w, vq_model.quantize.e_dim)
        with torch.no_grad():
            decoded = vq_model.decode_code(indices, shape=shape)
        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1.0) / 2.0 * 255.0
        decoded = decoded.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        return Image.fromarray(decoded)
    except Exception as e:
        print(f"[decode_vq_tokens] Error: {e}")
        return None


# ============================================================
# Dataset Class
# ============================================================
class VRTrainDataset(Dataset):
    """
    Unified dataset for VR_train data.
    
    Args:
        dataset_cfg: dict mapping dataset name → config dict.
            Each config dict can have:
              - "max_samples": int, max samples to use (None = all)
              - "subdir": str, override subdirectory (e.g. "Droid" → "aligned")
              - "enabled": bool, default True
              - "prompt_template": str, override prompt template
        data_root: root directory for all datasets
        tokenizer: HuggingFace tokenizer
        max_images: max answer frames per sample (None = no limit, limited by max_length)
        max_length: max token sequence length
        use_vlm_caption: if True, prefer 'question' field (v3) over global_summary
        debug_mode: if True, print extra info during loading
    """

    def __init__(
        self,
        dataset_cfg: Dict[str, Dict] = None,
        data_root: str = VR_TRAIN_ROOT,
        tokenizer=None,
        max_images: Optional[int] = None,
        max_length: int = 11000,
        use_vlm_caption: bool = True,
        debug_mode: bool = False,
        vq_model=None,
        debug_vis_dir: str = "./vis_debug_tokens",
        debug_vis_max: int = 20,
        cfg_prob: float = 0.0,
        interleaved_text: bool = False,
    ):
        super().__init__()
        assert tokenizer is not None, "tokenizer is required"
        self.tokenizer = tokenizer
        self.max_images = max_images
        self.max_length = max_length
        self.use_vlm_caption = use_vlm_caption
        self.data_root = data_root
        self.debug_mode = debug_mode
        self.vq_model = vq_model
        self.debug_vis_dir = debug_vis_dir
        self.debug_vis_max = debug_vis_max
        self._debug_vis_count = 0
        self.cfg_prob = 0.0  # probability of using unconditional prompt for CFG training
        self.interleaved_text = interleaved_text  # use interleaved text-image format for qualified datasets

        # Special tokens
        self.bos_token = "<|extra_203|>"
        self.eos_token = "<|extra_204|>"
        self.bss_token = "<|extra_100|>"
        self.ess_token = "<|extra_101|>"
        self.bog_token = "<|extra_60|>"
        self.eog_token = "<|extra_61|>"
        self.boc_token = "<|extra_50|>"
        self.eoc_token = "<|extra_51|>"

        self.bos_id = self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.bss_id = self.tokenizer.convert_tokens_to_ids(self.bss_token)
        self.ess_id = self.tokenizer.convert_tokens_to_ids(self.ess_token)
        self.bog_id = self.tokenizer.convert_tokens_to_ids(self.bog_token)
        self.eog_id = self.tokenizer.convert_tokens_to_ids(self.eog_token)
        self.boc_id = self.tokenizer.convert_tokens_to_ids(self.boc_token)
        self.eoc_id = self.tokenizer.convert_tokens_to_ids(self.eoc_token)

        # ---- Load data ----
        # Each element: (dataset_name, sample_dict)
        self.samples: List[tuple] = []
        # Per-dataset counts for reporting
        self.dataset_counts: Dict[str, int] = {}
        
        if dataset_cfg is None:
            dataset_cfg = {}

        for ds_name, ds_conf in dataset_cfg.items():
            if not ds_conf.get("enabled", True):
                continue
            self._load_dataset(ds_name, ds_conf)

        print(f"\n{'='*60}")
        print(f"VRTrainDataset loaded: {len(self.samples)} total samples")
        for name, cnt in self.dataset_counts.items():
            print(f"  {name}: {cnt}")
        print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------
    @staticmethod
    def _load_parquet_task(filepath: str, max_samples_per_task: Optional[int] = None) -> List[dict]:
        """Load a single parquet file using pyarrow (fast) with optional per-task sampling."""
        table = pq.read_table(filepath)
        num_rows = table.num_rows
        if max_samples_per_task is not None and num_rows > max_samples_per_task:
            indices = sorted(random.sample(range(num_rows), max_samples_per_task))
            table = table.take(indices)
        # to_pydict → list-of-dicts is faster than to_pandas → to_dict("records")
        col_dict = table.to_pydict()
        keys = list(col_dict.keys())
        n = len(col_dict[keys[0]]) if keys else 0
        return [{k: col_dict[k][i] for k in keys} for i in range(n)]

    def _load_dataset(self, ds_name: str, ds_conf: dict):
        """Load a single dataset from parquet files.
        
        Sampling modes (controlled by ds_conf keys):
          1. "max_samples_per_task": int  — each parquet file (=task) contributes
             at most this many samples (randomly sampled). All tasks are covered.
             Optionally combined with "max_samples" to cap the grand total.
          2. "max_samples": int (without per-task) — legacy mode, reads files
             sequentially and stops when total reaches the limit.
        
        Uses pyarrow + ThreadPool for fast parallel loading.
        """
        subdir = ds_conf.get("subdir", DATASET_DIRS.get(ds_name, ds_name))
        max_samples = ds_conf.get("max_samples", None)
        max_samples_per_task = ds_conf.get("max_samples_per_task", None)
        num_workers = ds_conf.get("num_workers", 8)
        data_dir = os.path.join(self.data_root, subdir)

        if not os.path.isdir(data_dir):
            print(f"[WARNING] Dataset directory not found: {data_dir}")
            return

        # Find all parquet files
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not parquet_files:
            parquet_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))

        if not parquet_files:
            print(f"[WARNING] No parquet files found in {data_dir}")
            return

        import time as _time
        t0 = _time.time()
        print(f"Loading {ds_name} from {data_dir} ({len(parquet_files)} files, workers={num_workers})...")

        records = []

        if max_samples_per_task is not None:
            # ---- Per-task parallel sampling ----
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_to_file = {
                    pool.submit(self._load_parquet_task, f, max_samples_per_task): f
                    for f in parquet_files
                }
                for future in as_completed(future_to_file):
                    f = future_to_file[future]
                    try:
                        task_records = future.result()
                        records.extend(task_records)
                        if self.debug_mode:
                            task_name = os.path.splitext(os.path.basename(f))[0]
                            print(f"    {task_name}: {len(task_records)} samples")
                    except Exception as e:
                        print(f"  [ERROR] loading {f}: {e}")
            # Optionally cap the grand total
            if max_samples is not None and len(records) > max_samples:
                random.shuffle(records)
                records = records[:max_samples]
            elapsed = _time.time() - t0
            print(f"  {ds_name}: loaded {len(records)} samples "
                  f"({len(parquet_files)} tasks, ≤{max_samples_per_task}/task, {elapsed:.1f}s)")
        else:
            # ---- Legacy parallel loading ----
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_to_file = {
                    pool.submit(self._load_parquet_task, f, None): f
                    for f in parquet_files
                }
                for future in as_completed(future_to_file):
                    f = future_to_file[future]
                    try:
                        task_records = future.result()
                        records.extend(task_records)
                    except Exception as e:
                        print(f"  [ERROR] loading {f}: {e}")
                    # Early stop check (approximate, since parallel)
                    if max_samples is not None and len(records) >= max_samples:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
            # Apply max_samples
            if max_samples is not None and len(records) > max_samples:
                random.shuffle(records)
                records = records[:max_samples]
            elapsed = _time.time() - t0
            print(f"  {ds_name}: loaded {len(records)} samples ({elapsed:.1f}s)")

        # Repeat dataset if requested (e.g. "repeat": 3 to triple the samples)
        repeat = ds_conf.get("repeat", 1)
        if repeat > 1:
            records = records * repeat
            print(f"  {ds_name}: repeated {repeat}x → {len(records)} samples")

        # Tag each record with its dataset name
        for r in records:
            self.samples.append((ds_name, r))

        self.dataset_counts[ds_name] = len(records)

    # ----------------------------------------------------------
    # Length / Getitem
    # ----------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ds_name, sample = self.samples[idx]
        try:
            return self._build_training_sample(ds_name, sample, idx)
        except Exception as e:
            if self.debug_mode:
                traceback.print_exc()
            # Return a fallback: try the next sample
            print(f"[ERROR] idx={idx} ds={ds_name}: {e}")
            fallback_idx = (idx + 1) % len(self.samples)
            ds_name2, sample2 = self.samples[fallback_idx]
            return self._build_training_sample(ds_name2, sample2, fallback_idx)

    # ----------------------------------------------------------
    # Core: build a training sample
    # ----------------------------------------------------------
    def _get_summary(self, sample: dict) -> str:
        """Get the best available text description for this sample."""
        if self.use_vlm_caption:
            # v3: prefer 'question' field (was 'vlm_task_instruction' in v2)
            q = sample.get("question", None)
            if q and isinstance(q, str) and len(q.strip()) > 0:
                return q.strip()
            # v2 backward compat
            vlm_instr = sample.get("vlm_task_instruction", None)
            if vlm_instr and isinstance(vlm_instr, str) and len(vlm_instr.strip()) > 0:
                return vlm_instr.strip()
        # Fallback to global_summary
        gs = sample.get("global_summary", "")
        return gs.strip() if gs else ""

    def _get_vlm_frame_captions(self, sample: dict) -> Optional[List[str]]:
        """Get per-frame VLM captions if available."""
        captions = sample.get("vlm_frame_captions", None)
        if captions is not None and isinstance(captions, list) and len(captions) > 0:
            return captions
        return None

    def _get_interleaved_captions(self, sample: dict, n_problem: int, n_answer: int
                                  ) -> tuple:
        """
        Extract captions for interleaved text-image format.

        Returns:
            (problem_caption, answer_captions)
            - problem_caption: str, caption for the first problem image (used in BoG)
            - answer_captions: list[str], one caption per answer frame
        """
        # --- Problem image caption (for BoG) ---
        problem_images = sample.get("problem_images", [])
        if isinstance(problem_images, np.ndarray):
            problem_images = problem_images.tolist()
        problem_caption = ""
        if problem_images and isinstance(problem_images[0], dict):
            problem_caption = problem_images[0].get("caption", "") or ""

        # --- Per-frame answer captions ---
        # vlm_frame_captions maps 1:1 to answer_images (no problem frame caption)
        vlm_captions = sample.get("vlm_frame_captions", None)
        answer_captions = []
        if vlm_captions and isinstance(vlm_captions, list):
            answer_captions = list(vlm_captions)
        
        # Fallback: use caption field from answer_images structs
        if not answer_captions:
            answer_images = sample.get("answer_images", [])
            if isinstance(answer_images, np.ndarray):
                answer_images = answer_images.tolist()
            for aimg in answer_images:
                if isinstance(aimg, dict):
                    answer_captions.append(aimg.get("caption", "") or "")
                else:
                    answer_captions.append("")

        # Pad or truncate to n_answer
        while len(answer_captions) < n_answer:
            answer_captions.append("")
        answer_captions = answer_captions[:n_answer]

        return problem_caption, answer_captions

    def _build_training_sample(self, ds_name: str, sample: dict, idx: int) -> dict:
        """
        Construct input_ids, labels, attention_mask from a sample.
        
        Sequence structure:
        [BOS] user_prompt problem_images [ASSISTANT] [BSS] [BoG] summary [EoG] answer_images [ESS] [EOS]
        
        Labels: -100 for [BOS] user_prompt problem_images [ASSISTANT] [BSS], 
                real ids for the rest (summary + answer_images + ESS + EOS)
        """
        # --- CFG: with cfg_prob probability, use unconditional prompt ---
        is_unconditional = (self.cfg_prob > 0) and (random.random() < self.cfg_prob)

        # --- Text summary ---
        summary = self._get_summary(sample)
        if is_unconditional:
            # Unconditional prompt: generic text without task description/summary
            user_prompt = UNCOND_PROMPT_TEMPLATE
        else:
            prompt_template = PROMPT_TEMPLATES.get(ds_name, 
                "You are a helpful assistant. USER: {summary}")
            user_prompt = prompt_template.format(summary=summary)
        user_prompt_ids = self.tokenizer.encode(user_prompt, add_special_tokens=False)

        # --- Problem images (input) ---
        h = sample.get("height")
        w = sample.get("width")
        
        problem_images = sample.get("problem_images", [])
        if isinstance(problem_images, np.ndarray):
            problem_images = problem_images.tolist()
        
        problem_image_ids = []
        if h is not None and w is not None:
            for pf in problem_images:
                if "image_tokens" in pf and pf["image_tokens"] is not None:
                    try:
                        # Use per-frame h/w if available (Zebra), else sample-level (Agibot)
                        fh = pf.get("frame_height") or h
                        fw = pf.get("frame_width") or w
                        pf_tokens = np.frombuffer(pf["image_tokens"], dtype=np.int64).astype(np.int32).reshape(fh, fw)
                        pf_str = format_image_string(self.tokenizer, pf_tokens)
                        problem_image_ids.extend(self.tokenizer.encode(pf_str, add_special_tokens=False))
                    except Exception as e:
                        if self.debug_mode:
                            print(f"  [WARN] problem image decode error: {e}")

        # --- ASSISTANT prompt ---
        assistant_prompt = " ASSISTANT: "
        assistant_prompt_ids = self.tokenizer.encode(assistant_prompt, add_special_tokens=False)

        # --- Interleaved text mode check ---
        use_interleaved = (self.interleaved_text and ds_name in INTERLEAVED_TEXT_DATASETS
                          and not is_unconditional)

        # --- Summary (BoG ... EoG) ---
        if use_interleaved:
            # Use global_summary as the summary text for interleaved mode
            global_sum = (sample.get("global_summary") or "").strip()
            if not global_sum:
                global_sum = "Let's begin."
            summary_ids = ([self.bog_id]
                           + self.tokenizer.encode(global_sum, add_special_tokens=False)
                           + [self.eog_id])
        else:
            summary_ids = [self.bog_id] + self.tokenizer.encode("Let's begin.", add_special_tokens=False) + [self.eog_id]

        # --- Answer images (output) ---
        answer_images = sample.get("answer_images", [])
        if isinstance(answer_images, np.ndarray):
            answer_images = answer_images.tolist()

        if self.max_images is not None and len(answer_images) > self.max_images:
            answer_images = answer_images[:self.max_images]

        # --- DEBUG: visualize raw image bytes vs decoded VQ tokens ---
        if self.debug_mode and self._debug_vis_count < self.debug_vis_max:
            self._debug_visualize_comparison(
                ds_name=ds_name, sample=sample, idx=idx,
                problem_images=problem_images, answer_images=answer_images,
                h=h, w=w,
            )
            self._debug_vis_count += 1

        # --- Get per-frame captions for interleaved mode (before budget calc) ---
        answer_captions = []
        if use_interleaved:
            n_prob = len(problem_images)
            _, answer_captions = self._get_interleaved_captions(
                sample, n_prob, len(answer_images))

        # Pre-calculate length budget for answer frames
        fixed_overhead = (1 + len(user_prompt_ids) + len(problem_image_ids) + len(assistant_prompt_ids) +
                         1 + len(summary_ids) + 1 + 1)  # BOS, BSS, ESS, EOS

        if h is not None and w is not None and len(answer_images) > 0:
            # Use max per-frame token count for budget estimation
            # (conservative: ensures we don't exceed max_length even for largest frames)
            max_fh = max((f.get("frame_height") or h) for f in answer_images)
            max_fw = max((f.get("frame_width") or w) for f in answer_images)
            tokens_per_frame = max_fh * max_fw + max_fh + 2  # image tokens + EOL per row + BOI/EOI
            # In interleaved mode, add estimated caption overhead per frame
            if use_interleaved and answer_captions:
                avg_cap_len = sum(
                    len(self.tokenizer.encode(c, add_special_tokens=False))
                    for c in answer_captions if c
                ) / max(len(answer_captions), 1)
                tokens_per_frame += int(avg_cap_len) + 2  # +2 margin
            # Reserve budget for final answer text
            fa_overhead = 0
            if use_interleaved:
                fa_text = (sample.get("final_answer") or "").strip()
                if fa_text:
                    fa_overhead = len(self.tokenizer.encode(fa_text, add_special_tokens=False))
            available_tokens = self.max_length - fixed_overhead - fa_overhead
            max_answer_frames = max(1, available_tokens // tokens_per_frame)

            if len(answer_images) > max_answer_frames:
                if max_answer_frames == 1:
                    answer_images = [answer_images[-1]]
                    if use_interleaved and answer_captions:
                        answer_captions = [answer_captions[-1]]
                else:
                    # Keep last frame, uniformly sample from the rest
                    last_frame = answer_images[-1]
                    other_frames = answer_images[:-1]
                    num_to_sample = max_answer_frames - 1
                    if len(other_frames) <= num_to_sample:
                        sampled_indices = list(range(len(other_frames)))
                    else:
                        sampled_indices = np.linspace(0, len(other_frames) - 1, num_to_sample, dtype=int).tolist()
                    answer_images = [other_frames[i] for i in sampled_indices] + [last_frame]
                    if use_interleaved and answer_captions:
                        orig_captions = answer_captions
                        answer_captions = [orig_captions[i] for i in sampled_indices] + [orig_captions[-1]]

        interleaved_ids = []
        if h is not None and w is not None:
            for fi, frame in enumerate(answer_images):
                if "image_tokens" in frame and frame["image_tokens"] is not None:
                    try:
                        # Use per-frame h/w if available (Zebra), else sample-level (Agibot)
                        fh = frame.get("frame_height") or h
                        fw = frame.get("frame_width") or w
                        frame_tokens = np.frombuffer(frame["image_tokens"], dtype=np.int64).astype(np.int32).reshape(fh, fw)
                        frame_str = format_image_string(self.tokenizer, frame_tokens)

                        # Interleaved: prepend caption text before each answer image
                        if use_interleaved:
                            cap = answer_captions[fi] if fi < len(answer_captions) else ""
                            if cap:
                                cap_ids = self.tokenizer.encode(cap, add_special_tokens=False)
                                interleaved_ids.extend(cap_ids)

                        interleaved_ids.extend(self.tokenizer.encode(frame_str, add_special_tokens=False))
                    except Exception as e:
                        if self.debug_mode:
                            print(f"  [WARN] answer image decode error: {e}")

        # --- Final answer text (append after last answer image) ---
        final_answer_ids = []
        if use_interleaved:
            fa_text = (sample.get("final_answer") or "").strip()
            if fa_text:
                final_answer_ids = self.tokenizer.encode(fa_text, add_special_tokens=False)
                interleaved_ids.extend(final_answer_ids)

        # --- Assemble sequence ---
        input_ids = (
            [self.bos_id] + user_prompt_ids + problem_image_ids + assistant_prompt_ids +
            [self.bss_id] + summary_ids + interleaved_ids + [self.ess_id, self.eos_id]
        )
        # --- Labels (mask user/problem part) ---
        user_part_len = len(user_prompt_ids) + len(problem_image_ids) + len(assistant_prompt_ids)
        labels = [-100] * (1 + user_part_len + 1)  # BOS + user_prompt + problem + assistant + BSS
        labels.extend(summary_ids)
        labels.extend(interleaved_ids)
        labels.extend([self.ess_id, self.eos_id])

        # --- Padding / Truncation ---
        seq_len = len(input_ids)
        padding_length = self.max_length - seq_len

        if padding_length < 0:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            padding_length = 0

        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length

        attention_mask = [1] * min(seq_len, self.max_length) + [0] * padding_length

        # --- Sample info for logging ---
        ep_id = sample.get("episode_id", "?")
        n_prob = len(problem_images)
        n_ans = len(answer_images)
        actual_len = min(seq_len, self.max_length)
        sample_info = (f"dataset={ds_name} | idx={idx} | ep_id={ep_id} | "
                       f"n_problem={n_prob} | n_answer={n_ans} | "
                       f"seq_len={actual_len} | cfg_uncond={is_unconditional}")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sample_info": sample_info,
        }

    # ----------------------------------------------------------
    # Visualization helpers (for debug / data inspection)
    # ----------------------------------------------------------
    def _debug_visualize_comparison(
        self,
        ds_name: str,
        sample: dict,
        idx: int,
        problem_images: list,
        answer_images: list,
        h: Optional[int],
        w: Optional[int],
    ):
        """
        DEBUG: Save a side-by-side comparison of raw image bytes vs VQ-decoded images.
        
        For each frame, shows two rows:
          Row 1 ("bytes"): image decoded from problem_image_bytes / answer_image_bytes
          Row 2 ("VQ decoded"): image reconstructed from VQ tokens via vq_model
        
        Saved as a grid image to self.debug_vis_dir.
        If vq_model is None, only the raw bytes row is shown.
        """
        from PIL import ImageDraw, ImageFont

        output_dir = self.debug_vis_dir
        os.makedirs(output_dir, exist_ok=True)

        ep_id = sample.get("episode_id", idx)
        prefix = f"{ds_name}_{ep_id}"

        # Collect frames: list of (label, bytes_img_or_None, vq_img_or_None)
        pairs = []  # (label, raw_pil, vq_pil)

        # --- Problem frames ---
        problem_bytes_list = sample.get("problem_image_bytes", None)
        if problem_bytes_list is None:
            problem_bytes_list = []
        for i, pf in enumerate(problem_images):
            label = f"P{i}"
            # Raw bytes image
            raw_img = None
            if i < len(problem_bytes_list) and problem_bytes_list[i] is not None:
                raw_img = decode_image_bytes(problem_bytes_list[i])
            # VQ decoded image
            vq_img = None
            if (
                self.vq_model is not None
                and h is not None
                and w is not None
                and "image_tokens" in pf
                and pf["image_tokens"] is not None
            ):
                try:
                    pfh = pf.get("frame_height") or h
                    pfw = pf.get("frame_width") or w
                    tok = np.frombuffer(pf["image_tokens"], dtype=np.int64).astype(np.int32).reshape(pfh, pfw)
                    vq_img = decode_vq_tokens(self.vq_model, tok, pfh, pfw)
                except Exception as e:
                    print(f"  [DEBUG-VIS] VQ decode error for problem frame {i}: {e}")
            pairs.append((label, raw_img, vq_img))

        # --- Answer frames ---
        answer_bytes_list = sample.get("answer_image_bytes", None)
        if answer_bytes_list is None:
            answer_bytes_list = []
        for i, af in enumerate(answer_images):
            label = f"A{i}"
            raw_img = None
            if i < len(answer_bytes_list) and answer_bytes_list[i] is not None:
                raw_img = decode_image_bytes(answer_bytes_list[i])
            vq_img = None
            if (
                self.vq_model is not None
                and h is not None
                and w is not None
                and "image_tokens" in af
                and af["image_tokens"] is not None
            ):
                try:
                    afh = af.get("frame_height") or h
                    afw = af.get("frame_width") or w
                    tok = np.frombuffer(af["image_tokens"], dtype=np.int64).astype(np.int32).reshape(afh, afw)
                    vq_img = decode_vq_tokens(self.vq_model, tok, afh, afw)
                except Exception as e:
                    print(f"  [DEBUG-VIS] VQ decode error for answer frame {i}: {e}")
            pairs.append((label, raw_img, vq_img))

        if not pairs:
            return

        # --- Build comparison grid ---
        # Layout: each column = one frame, row 0 = raw bytes, row 1 = VQ decoded
        thumb_w, thumb_h = 256, 256
        label_height = 24
        row_header_w = 100  # width for row labels ("Bytes", "VQ Dec")
        has_vq = self.vq_model is not None
        n_rows = 2 if has_vq else 1
        n_cols = len(pairs)
        max_cols_per_row = 16

        # If too many columns, split into multiple grid rows
        grid_rows = (n_cols + max_cols_per_row - 1) // max_cols_per_row
        cols_in_grid = min(n_cols, max_cols_per_row)

        cell_w = thumb_w + 4
        cell_h = thumb_h + label_height + 4
        total_w = row_header_w + cols_in_grid * cell_w
        total_h = grid_rows * n_rows * cell_h + label_height

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Title
        title = f"{ds_name} | ep={ep_id} | idx={idx} | h={h} w={w} | P={len(problem_images)} A={len(answer_images)}"
        draw.text((4, 2), title, fill="black")

        for frame_idx, (label, raw_img, vq_img) in enumerate(pairs):
            grid_row_block = frame_idx // max_cols_per_row
            col_in_block = frame_idx % max_cols_per_row
            x_base = row_header_w + col_in_block * cell_w

            for row_idx, (row_label, img, border_color) in enumerate([
                ("Bytes", raw_img, "green"),
                ("VQ Dec", vq_img, "blue"),
            ]):
                if row_idx >= n_rows:
                    break
                y_base = label_height + (grid_row_block * n_rows + row_idx) * cell_h

                # Row header (only for first column in each block)
                if col_in_block == 0:
                    draw.text((4, y_base + thumb_h // 2), row_label, fill=border_color)

                if img is not None:
                    thumb = img.copy()
                    thumb.thumbnail((thumb_w, thumb_h))
                    px = x_base + 2
                    py = y_base + 2
                    canvas.paste(thumb, (px, py))
                    draw.rectangle(
                        [px - 1, py - 1, px + thumb.width, py + thumb.height],
                        outline=border_color, width=2,
                    )
                else:
                    # Draw placeholder
                    px = x_base + 2
                    py = y_base + 2
                    draw.rectangle([px, py, px + thumb_w - 1, py + thumb_h - 1], outline="gray", width=1)
                    draw.text((px + 10, py + thumb_h // 2 - 6), "N/A", fill="gray")

                # Column label (only on first row)
                if row_idx == 0:
                    draw.text((x_base + 2, y_base + thumb_h + 4), label, fill="black")

        out_path = os.path.join(output_dir, f"debug_compare_{prefix}_{idx}.jpg")
        canvas.save(out_path, quality=90)
        print(f"  [DEBUG-VIS] Saved comparison: {out_path}  ({len(pairs)} frames, vq={'yes' if has_vq else 'no'})")

    def visualize(self, idx: int, output_dir: str = "./vis_debug"):
        """
        Visualize problem/answer frames for sample at index `idx`.
        Saves individual frames, a grid image, and optionally a bytes-vs-VQ comparison.
        
        Returns:
            list of saved file paths
        """
        ds_name, sample = self.samples[idx]
        saved = visualize_sample(sample, output_dir=output_dir, sample_idx=idx)
        
        # Also create grid
        grid_path = os.path.join(output_dir, f"grid_{ds_name}_{idx}.jpg")
        visualize_sample_grid(sample, output_path=grid_path)
        saved.append(grid_path)

        # Also create bytes-vs-VQ comparison if vq_model is available or debug_mode
        if self.debug_mode or self.vq_model is not None:
            h = sample.get("height")
            w = sample.get("width")
            problem_images = sample.get("problem_images", [])
            answer_images = sample.get("answer_images", [])
            if isinstance(problem_images, np.ndarray):
                problem_images = problem_images.tolist()
            if isinstance(answer_images, np.ndarray):
                answer_images = answer_images.tolist()
            # Temporarily override debug_vis_dir to use the caller's output_dir
            old_vis_dir = self.debug_vis_dir
            self.debug_vis_dir = output_dir
            self._debug_visualize_comparison(
                ds_name=ds_name, sample=sample, idx=idx,
                problem_images=problem_images, answer_images=answer_images,
                h=h, w=w,
            )
            self.debug_vis_dir = old_vis_dir
        
        return saved

    def visualize_by_dataset(self, ds_name: str, num_samples: int = 5, output_dir: str = "./vis_debug"):
        """Visualize random samples from a specific dataset (with bytes-vs-VQ comparison)."""
        indices = [i for i, (name, _) in enumerate(self.samples) if name == ds_name]
        if not indices:
            print(f"No samples found for dataset: {ds_name}")
            return
        
        chosen = random.sample(indices, min(num_samples, len(indices)))
        for idx in chosen:
            self.visualize(idx, output_dir=output_dir)
        print(f"Visualized {len(chosen)} samples from {ds_name}")

    def get_raw_sample(self, idx: int) -> tuple:
        """Return (dataset_name, raw_sample_dict) for inspection."""
        return self.samples[idx]

    def print_sample_info(self, idx: int):
        """Print detailed info about a sample for debugging."""
        ds_name, sample = self.samples[idx]
        print(f"\n{'='*60}")
        print(f"Sample {idx} | Dataset: {ds_name}")
        print(f"{'='*60}")
        print(f"  episode_id:      {sample.get('episode_id', '?')}")
        print(f"  task_id:         {sample.get('task_id', '?')}")
        print(f"  task_name:       {sample.get('task_name', '?')}")
        print(f"  dataset_source:  {sample.get('dataset_source', '?')}")
        print(f"  height x width:  {sample.get('height', '?')} x {sample.get('width', '?')}")
        
        summary = self._get_summary(sample)
        print(f"  summary:         {summary[:120]}{'...' if len(summary) > 120 else ''}")
        
        q = sample.get("question", None)
        if q:
            print(f"  question:        {str(q)[:120]}{'...' if len(str(q)) > 120 else ''}")
        
        fa = sample.get("final_answer", None)
        if fa:
            print(f"  final_answer:    {str(fa)[:120]}{'...' if len(str(fa)) > 120 else ''}")
        
        captions = self._get_vlm_frame_captions(sample)
        if captions:
            print(f"  vlm_captions:    {len(captions)} captions")
            for i, c in enumerate(captions[:3]):
                print(f"    [{i}] {str(c)[:100]}")
            if len(captions) > 3:
                print(f"    ... and {len(captions) - 3} more")
        
        prob = sample.get("problem_images", []) 
        ans = sample.get("answer_images", []) 
        print(f"  problem_images:  {len(prob)}")
        print(f"  answer_images:   {len(ans)}")
        
        prob_bytes = sample.get("problem_image_bytes", None)
        ans_bytes = sample.get("answer_image_bytes", None)
        print(f"  problem_bytes:   {len(prob_bytes) if prob_bytes is not None else 0}")
        print(f"  answer_bytes:    {len(ans_bytes) if ans_bytes is not None else 0}")
        
        if prob_bytes:
            sw = sample.get("save_image_width", "?")
            sh = sample.get("save_image_height", "?")
            print(f"  saved_img_size:  {sw} x {sh}")
        print()

    def summary(self):
        """Print a summary of the loaded datasets."""
        print(f"\nVRTrainDataset Summary")
        print(f"{'='*50}")
        print(f"Total samples: {len(self.samples)}")
        print(f"Max length:    {self.max_length}")
        print(f"Max images:    {self.max_images}")
        print(f"Use VLM cap:   {self.use_vlm_caption}")
        print(f"{'─'*50}")
        for name, cnt in sorted(self.dataset_counts.items()):
            pct = cnt / len(self.samples) * 100 if len(self.samples) > 0 else 0
            print(f"  {name:20s}: {cnt:>8d}  ({pct:5.1f}%)")
        print(f"{'='*50}\n")


# ============================================================
# Main: Debug & Visualization
# ============================================================
if __name__ == "__main__":
    from transformers import AutoTokenizer

    workspace_path = Path("./")
    tokenizer_path = workspace_path / "src/tokenizer_emu3_ibq"

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        special_tokens_file=os.path.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"

    # ---- Configure which datasets to use and how many samples ----
    dataset_cfg = {
        "Action100":       {"max_samples": 20, "enabled": True},
        "Agibot":          {"max_samples_per_task": 5, "enabled": True},
        "Bridge":          {"max_samples": 5, "enabled": True},
        "Droid":           {"max_samples": 5, "subdir": "Droid/aligned", "enabled": True},
        "EgoDex":          {"max_samples_per_task": 5, "enabled": True},
        "Epic_Kitchen":    {"max_samples": 20, "enabled": True},
        "VideoCraftBench": {"max_samples": 5, "enabled": True},
        "Zebra_Count":     {"max_samples": 20, "enabled": True},
        "Zebra_Robot":     {"max_samples": 20, "enabled": True},
        "Zebra_Jigsaw":    {"max_samples": 5, "enabled": True},
        "Visual_Search":               {"max_samples": 5, "enabled": True},
        "Spatial_Navigation":          {"max_samples": 5, "enabled": True},
        "Spatial_Navigation_maze":     {"max_samples": 5, "enabled": True},
        "Spatial_Navigation_trapfield":{"max_samples": 5, "enabled": True},
        "Visual_CoT_GQA":  {"max_samples": 5, "enabled": True},
    }

    vq_model = None
    vq_model = build_vision_tokenizer('ibq', "./weights/Emu3.5-VisionTokenizer", device='cuda')
    ds = VRTrainDataset(
        dataset_cfg=dataset_cfg,
        tokenizer=tokenizer,
        max_images=None,
        max_length=11000,
        use_vlm_caption=True,
        debug_mode=True,
        vq_model=vq_model,
        interleaved_text=True,
    )

    ds.summary()

    # ---- Test a few samples from each dataset ----
    tested_datasets = set()
    for i in range(len(ds)):
        ds_name, _ = ds.samples[i]
        if ds_name in tested_datasets:
            continue
        tested_datasets.add(ds_name)

        print(f"\n--- Testing {ds_name} (idx={i}) ---")
        ds.print_sample_info(i)
        
        try:
            item = ds[i]
            print(f"  input_ids shape:    {item['input_ids'].shape}")
            print(f"  labels shape:       {item['labels'].shape}")
            print(f"  attention_mask shape:{item['attention_mask'].shape}")
            labels_list = item["labels"].tolist()
            masked = labels_list.count(-100)
            print(f"  masked labels:      {masked}")
            print(f"  valid labels:       {len(labels_list) - masked}")
            print(f"  sample_info:        {item['sample_info']}")

            # Decode input_ids back to text to show the raw prompt content
            input_ids_list = item["input_ids"].tolist()
            attn = item["attention_mask"].tolist()
            actual_len = sum(attn)
            valid_ids = input_ids_list[:actual_len]
            decoded_text = tokenizer.decode(valid_ids, skip_special_tokens=False)
            print(f"\n  === Decoded prompt text (first 2000 chars) ===")
            print(f"  {decoded_text[:2000]}")
            if len(decoded_text) > 2000:
                print(f"  ... (truncated, total {len(decoded_text)} chars)")
            print(f"  === End of decoded prompt ===\n")
        except Exception as e:
            traceback.print_exc()
            print(f"  [FAILED] {e}")

        if len(tested_datasets) == len(dataset_cfg):
            break

    # ================================================================
    # Interleaved text-image visualization → HTML
    # ================================================================
    import base64
    import html as html_lib
    from io import BytesIO

    def pil_to_base64(img: Image.Image, fmt="JPEG", quality=85) -> str:
        buf = BytesIO()
        img.save(buf, format=fmt, quality=quality)
        return base64.b64encode(buf.getvalue()).decode()

    NUM_SAMPLES_PER_DS = 10 # how many samples per dataset to visualize
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Interleaved Text-Image Debug</title>",
        "<style>",
        "body{font-family:monospace;background:#f8f8f8;margin:20px}",
        ".sample{border:2px solid #333;margin:20px 0;padding:15px;background:#fff;border-radius:8px}",
        ".sample h2{margin:0 0 10px 0;font-size:16px}",
        ".meta{color:#666;font-size:12px;margin-bottom:10px}",
        ".seq-block{display:flex;flex-wrap:wrap;align-items:flex-start;gap:6px}",
        ".token-group{display:inline-flex;flex-direction:column;align-items:center;"
        "border:1px solid #ddd;border-radius:4px;padding:4px;background:#fafafa;max-width:280px}",
        ".token-group.special{background:#e8f0fe;border-color:#4285f4;font-weight:bold;padding:4px 8px}",
        ".token-group.text{background:#fff9e6;border-color:#f9ab00;padding:4px 8px;max-width:400px;word-break:break-word}",
        ".token-group.image{background:#e6f4ea;border-color:#34a853}",
        ".token-group img{max-width:256px;max-height:256px;border-radius:4px}",
        ".caption{font-size:11px;color:#333;margin-top:2px;text-align:center;max-width:260px;word-break:break-word}",
        ".label{font-size:10px;color:#888;margin-bottom:2px}",
        ".sep{border-left:2px solid #ccc;height:60px;margin:0 4px}",
        "</style></head><body>",
        "<h1>Interleaved Text-Image Visualization (interleaved_text=True)</h1>",
    ]

    # Collect samples per dataset
    ds_sample_map = {}  # ds_name → list of indices
    for i, (dn, _) in enumerate(ds.samples):
        ds_sample_map.setdefault(dn, []).append(i)

    sample_count = 0
    for ds_name in sorted(ds_sample_map.keys()):
        indices = ds_sample_map[ds_name]
        chosen = random.sample(indices, min(NUM_SAMPLES_PER_DS, len(indices)))

        html_parts.append(f"<h2 style='margin-top:30px;border-bottom:2px solid #333;padding-bottom:5px'>"
                          f"Dataset: {html_lib.escape(ds_name)} ({len(indices)} total samples)</h2>")

        for sample_idx in chosen:
            ds_n, sample = ds.samples[sample_idx]
            h = sample.get("height")
            w = sample.get("width")
            ep_id = sample.get("episode_id", "?")

            # Check interleaved eligibility
            use_interleaved = ds_name in INTERLEAVED_TEXT_DATASETS

            # Get text fields
            summary = ds._get_summary(sample)
            problem_images = sample.get("problem_images", [])
            answer_images_raw = sample.get("answer_images", [])
            if isinstance(problem_images, np.ndarray):
                problem_images = problem_images.tolist()
            if isinstance(answer_images_raw, np.ndarray):
                answer_images_raw = answer_images_raw.tolist()

            # Limit answer images for display
            answer_images_vis = answer_images_raw[:20]

            # Get captions
            if use_interleaved:
                n_prob = len(problem_images)
                q_caption, ans_captions = ds._get_interleaved_captions(
                    sample, n_prob, len(answer_images_vis))
            else:
                q_caption = ""
                ans_captions = []

            html_parts.append(f"<div class='sample'>")
            html_parts.append(f"<h2>Sample #{sample_idx} | ep_id={html_lib.escape(str(ep_id))} | "
                              f"h={h} w={w} | P={len(problem_images)} A={len(answer_images_raw)}"
                              f"{' | INTERLEAVED' if use_interleaved else ''}</h2>")
            html_parts.append(f"<div class='meta'>Summary: {html_lib.escape(summary[:200])}</div>")

            html_parts.append("<div class='seq-block'>")

            # --- BOS ---
            html_parts.append("<div class='token-group special'><div class='label'>BOS</div>"
                              "&lt;|extra_203|&gt;</div>")

            # --- User prompt (abbreviated) ---
            prompt_text = summary[:120] + ("..." if len(summary) > 120 else "")
            html_parts.append(f"<div class='token-group text'><div class='label'>USER Prompt</div>"
                              f"{html_lib.escape(prompt_text)}</div>")

            # --- Problem images ---
            for pi, pf in enumerate(problem_images):
                img_html = ""
                if h and w and "image_tokens" in pf and pf["image_tokens"] is not None:
                    try:
                        fh = pf.get("frame_height") or h
                        fw = pf.get("frame_width") or w
                        tok = np.frombuffer(pf["image_tokens"], dtype=np.int64).astype(np.int32).reshape(fh, fw)
                        pil_img = decode_vq_tokens(vq_model, tok, fh, fw)
                        if pil_img:
                            b64 = pil_to_base64(pil_img)
                            img_html = f"<img src='data:image/jpeg;base64,{b64}'/>"
                    except Exception:
                        img_html = "<span style='color:red'>decode error</span>"
                cap_text = pf.get("caption", "") or "" if isinstance(pf, dict) else ""
                html_parts.append(f"<div class='token-group image'><div class='label'>Problem Image {pi}</div>"
                                  f"{img_html}"
                                  f"<div class='caption'>{html_lib.escape(cap_text[:150])}</div></div>")

            # --- ASSISTANT ---
            html_parts.append("<div class='token-group special'><div class='label'>ASSISTANT</div>"
                              "ASSISTANT:</div>")

            # --- BSS ---
            html_parts.append("<div class='token-group special'><div class='label'>BSS</div>"
                              "&lt;|extra_100|&gt;</div>")

            # --- BoG summary ---
            # Match training logic (L592): use global_summary, not q_caption
            global_sum = (sample.get("global_summary") or "").strip() if use_interleaved else ""
            bog_text = global_sum if global_sum else "Let's begin."
            html_parts.append(f"<div class='token-group text'><div class='label'>BoG..EoG</div>"
                              f"&lt;BoG&gt; {html_lib.escape(bog_text[:200])} &lt;EoG&gt;</div>")

            # --- Answer images (interleaved with captions) ---
            for ai, af in enumerate(answer_images_vis):
                # Caption before image (interleaved mode)
                if use_interleaved and ai < len(ans_captions) and ans_captions[ai]:
                    cap = ans_captions[ai]
                    html_parts.append(f"<div class='token-group text'><div class='label'>"
                                      f"Caption A{ai}</div>{html_lib.escape(cap[:200])}</div>")

                # Image
                img_html = ""
                if h and w and "image_tokens" in af and af["image_tokens"] is not None:
                    try:
                        fh = af.get("frame_height") or h
                        fw = af.get("frame_width") or w
                        tok = np.frombuffer(af["image_tokens"], dtype=np.int64).astype(np.int32).reshape(fh, fw)
                        pil_img = decode_vq_tokens(vq_model, tok, fh, fw)
                        if pil_img:
                            b64 = pil_to_base64(pil_img)
                            img_html = f"<img src='data:image/jpeg;base64,{b64}'/>"
                    except Exception:
                        img_html = "<span style='color:red'>decode error</span>"
                html_parts.append(f"<div class='token-group image'><div class='label'>Answer {ai}</div>"
                                  f"{img_html}</div>")

            # --- Final answer text (after last image) ---
            if use_interleaved:
                fa_text = (sample.get("final_answer") or "").strip()
                if fa_text:
                    html_parts.append(f"<div class='token-group text' style='background:#ffe0e0;border-color:#d93025'>"
                                      f"<div class='label'>Final Answer</div>"
                                      f"{html_lib.escape(fa_text[:300])}</div>")

            # --- ESS + EOS ---
            html_parts.append("<div class='token-group special'><div class='label'>ESS</div>"
                              "&lt;|extra_101|&gt;</div>")
            html_parts.append("<div class='token-group special'><div class='label'>EOS</div>"
                              "&lt;|extra_204|&gt;</div>")

            html_parts.append("</div>")  # seq-block
            html_parts.append("</div>")  # sample
            sample_count += 1

    html_parts.append(f"<p style='margin-top:30px;color:#666'>Total {sample_count} samples visualized.</p>")
    html_parts.append("</body></html>")

    vis_html_path = os.path.join(
        "./",
        "interleaved_debug_vis.html")
    with open(vis_html_path, "w") as f:
        f.write("\n".join(html_parts))
    print(f"\n[VIS] Saved interleaved visualization to {vis_html_path} ({sample_count} samples)")

    print("\nDone!")
