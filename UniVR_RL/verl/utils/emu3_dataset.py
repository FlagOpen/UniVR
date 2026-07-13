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
import glob
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import random
import torch
import datasets as hf_datasets_lib
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from . import torch_functional as VF
# import  torch_functional as VF

def format_image_string(image_tokens, boi_token, eoi_token, img_token, eol_token):
    """
    Format image tokens into Emu3 image string format.
    
    Args:
        image_tokens: 2D numpy array of shape (height, width) containing VQ token IDs
        boi_token: Beginning of Image token
        eoi_token: End of Image token
        img_token: Image token marker
        eol_token: End of Line token
        
    Returns:
        Formatted image string: <|image start|>{h}*{w}<|image token|>{visual_tokens}<|image end|>
    """
    image_string = ""
    h, w = image_tokens.shape
    for _h in range(h):
        row_string = ""
        for _w in range(w):
            row_string += "<|visual token {token_id:0>6d}|>".format(token_id=image_tokens[_h, _w])

        if _h < h - 1:
            row_string += eol_token
        image_string += row_string

    return "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}".format(
        image_start=boi_token,
        token_height=h,
        token_width=w,
        image_token=img_token,
        token_str=image_string,
        image_end=eoi_token,
    )


def emu3_collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for Emu3 pre-tokenized data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class Emu3RLHFDataset(Dataset):
    """
    Dataset for Emu3 models with pre-tokenized image data for RLHF training.
    
    Supports multiple data sources:
    - agibot:       'problem_images' + 'answer_images', sample-level h/w, robotic manipulation prompt
    - egodex:       'problem_images' + 'answer_images', sample-level h/w, dexterous manipulation prompt
    - videocraft:   'problem_frames'/'problem_images' + 'answer_frames'/'answer_images', task prediction prompt
    - epic_kitchen: 'problem_images' + 'answer_images', sample-level h/w, first-person cooking prompt
    - maze:         'problem_images' + 'answer_images', sample-level h/w, visual animation prompt
    - zebra_count:  'problem_images' + 'answer_images', sample-level h/w, 3D object counting prompt
    - zebra_jigsaw: 'problem_images' + 'answer_images', sample-level h/w, 2D jigsaw puzzle prompt
    - zebra_robot:  'problem_images' + 'answer_images', sample-level h/w, robot planning prompt
    - zebra_search: 'problem_images' + 'answer_images', sample-level h/w, visual search prompt
    
    data_path formats:
    - str: single parquet file/directory (auto-detect dataset type)
    - list[str]: multiple parquet files/directories (auto-detect per source)
    - dict: {"dataset_type": "path_or_list"} for explicit type specification
      e.g. {"agibot": "/path/to/agibot_dir",
            "egodex": "/path/to/egodex_dir",
            "videocraft": "/path/to/videocraft.parquet"}
    
    Directories are recursively scanned for .parquet files, allowing per-task
    organization (e.g. Agibot/task_327.parquet, Agibot/task_385.parquet, ...).
    
    Args:
        data_path: Path(s) to parquet dataset(s). See above for formats.
        tokenizer: Emu3 tokenizer with special tokens configured
        prompt_key: Key for custom prompt text (default: "prompt")
        answer_key: Key for ground truth answer (default: "global_summary")
        max_prompt_length: Maximum prompt length
        max_images: Maximum number of images per sample (for ground truth)
        truncation: Truncation strategy ("left", "right", "error")
        enable_cfg: Enable Classifier-Free Guidance by providing unconditional prompts
    """

    # Known dataset source types
    KNOWN_DATASET_TYPES = (
        "agibot", "egodex", "videocraft",
        "epic_kitchen", "maze",
        "zebra_count", "zebra_jigsaw", "zebra_robot", "zebra_search",
        "visual_search",
    )

    # Per-type user prompt templates ({task} will be replaced with vlm_task_instruction)
    PROMPT_TEMPLATES = {
        "agibot":        "You are a helpful assistant for vla task. USER: Given the first frame, how to perform the following task? {task}",
        "egodex":        "You are a helpful assistant for first-person perspective howto task. USER: Given the first frame, how to perform the following task? {task}",
        "videocraft":    "You are a helpful assistant for visual reasoning task. USER: Given the first frame, predict the following frames for the task: {task}",
        "epic_kitchen":  "You are a helpful assistant for first-person cooking task. USER: Given the first frame, how to perform the following task? {task}",
        "maze":          "You are a helpful assistant for visual reasoning task. USER: Given the first frame, predict the following frames for the task: {task}",
        "zebra_count":   "You are a helpful assistant for visual reasoning task. USER: Given the first frame, predict the following frames for the task: {task}",
        "zebra_jigsaw":  "You are a helpful assistant for visual reasoning task. USER: Given the first frame, predict the following frames for the task: {task}",
        "zebra_robot":   "You are a helpful assistant for visual reasoning task. USER: Given the first frame, predict the following frames for the task: {task}",
        "zebra_search":  "You are a helpful assistant for visual reasoning task. USER: Given the first frame, predict the following frames for the task: {task}",
        "visual_search": "You are a helpful assistant for visual search task. USER: Given the images, answer the following question: {task}",
    }

    # Default per-source sample limits (set to None to disable)
    DEFAULT_MAX_SAMPLES_PER_SOURCE = {}

    # Dataset types that carry per-frame captions and support interleaved text-image format
    INTERLEAVED_TEXT_DATASETS = {
        "agibot", "egodex", "epic_kitchen", "videocraft",
        "zebra_count", "zebra_jigsaw", "zebra_robot", "zebra_search",
        "visual_search",
    }

    def __init__(
        self,
        data_path: Union[str, list, dict],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        answer_key: str = "global_summary",
        input_ids_key: Optional[str] = None,
        max_prompt_length: int = 4096,
        max_images: Optional[int] = None,
        truncation: str = "right",
        format_prompt: Optional[str] = None,
        filter_overlong_prompts: bool = False,
        filter_overlong_prompts_workers: int = 8,
        enable_cfg: bool = False,
        max_samples_per_source: Optional[dict] = None,
        interleaved_text: bool = False,
    ):
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.max_prompt_length = max_prompt_length
        self.max_images = max_images
        self.truncation = truncation
        self.enable_cfg = enable_cfg
        self.interleaved_text = interleaved_text

        print(f"DEBUG: __init__ called. filter_overlong_prompts={filter_overlong_prompts}, interleaved_text={interleaved_text}, path={data_path}")

        # Emu3 special tokens
        self.bos_token = "<|extra_203|>"
        self.eos_token = "<|extra_204|>"
        self.bss_token = "<|extra_100|>"
        self.ess_token = "<|extra_101|>"
        self.bog_token = "<|extra_60|>"
        self.eog_token = "<|extra_61|>"
        self.boc_token = "<|extra_50|>"
        self.eoc_token = "<|extra_51|>"

        # Convert special tokens to IDs
        self.bos_id = self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.bss_id = self.tokenizer.convert_tokens_to_ids(self.bss_token)
        self.ess_id = self.tokenizer.convert_tokens_to_ids(self.ess_token)
        self.bog_id = self.tokenizer.convert_tokens_to_ids(self.bog_token)
        self.eog_id = self.tokenizer.convert_tokens_to_ids(self.eog_token)
        self.boc_id = self.tokenizer.convert_tokens_to_ids(self.boc_token)
        self.eoc_id = self.tokenizer.convert_tokens_to_ids(self.eoc_token)

        # Store image special tokens strings for worker process safe access
        self.boi_token = getattr(self.tokenizer, 'boi_token', '<|image start|>')
        self.eoi_token = getattr(self.tokenizer, 'eoi_token', '<|image end|>')
        self.img_token = getattr(self.tokenizer, 'img_token', '<|image token|>')
        self.eol_token = getattr(self.tokenizer, 'eol_token', '<|extra_200|>')

        # Merge default sample limits with user overrides
        self.max_samples_per_source = dict(self.DEFAULT_MAX_SAMPLES_PER_SOURCE)
        if max_samples_per_source is not None:
            self.max_samples_per_source.update(max_samples_per_source)

        # Load dataset(s) — supports str, list[str], dict
        # Each element is (source_type: str, dataset: HF Dataset)
        self._datasets: list[tuple[str, Any]] = self._load_multi_source(data_path)

        # Apply per-source random sampling
        self._datasets = self._apply_per_source_sampling(self._datasets)

        # Load prompt template (optional)
        self.format_prompt = None
        if format_prompt and os.path.exists(format_prompt):
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()
        
        if filter_overlong_prompts:
            total_before = sum(len(ds) for _, ds in self._datasets)
            print(f"[Emu3RLHFDataset] Filtering overlong prompts from {total_before} samples...")
            filtered = []
            for source_type, ds in self._datasets:
                n_before = len(ds)
                _st = source_type  # capture by value
                ds_filtered = ds.filter(
                    lambda sample, _source=_st: self._filter_overlong_prompts(sample, _source),
                    desc=f"Filtering {source_type}",
                    num_proc=filter_overlong_prompts_workers,
                    load_from_cache_file=False,
                )
                print(f"  [{source_type}] {n_before} -> {len(ds_filtered)} samples")
                filtered.append((source_type, ds_filtered))
            self._datasets = filtered

        # Precompute offsets for fast index lookup
        self._rebuild_offsets()

        total = sum(len(ds) for _, ds in self._datasets)
        print(f"[Emu3RLHFDataset] Loaded {total} samples from {data_path}")
        for stype, ds in self._datasets:
            print(f"  [{stype}] {len(ds)} samples")
        print(f"[Emu3RLHFDataset] Special tokens - BOS: {self.bos_id}, EOS: {self.eos_id}, BSS: {self.bss_id}, ESS: {self.ess_id}")

    def _rebuild_offsets(self):
        """Precompute cumulative offsets for global index -> (dataset_idx, local_idx) mapping."""
        self._offsets = []  # list of (cumulative_start, source_type, dataset)
        cumulative = 0
        for source_type, ds in self._datasets:
            self._offsets.append((cumulative, source_type, ds))
            cumulative += len(ds)
        self._total_len = cumulative

    # ------------------------------------------------------------------ #
    #  Multi-source data loading
    # ------------------------------------------------------------------ #
    def _load_single_path(self, path: str) -> "hf_datasets_lib.Dataset":
        """Load a single parquet file, directory, or glob pattern into a HuggingFace Dataset.
        
        Supports:
        - Single parquet file: "/path/to/file.parquet"
        - Directory (recursively finds all .parquet inside): "/path/to/dir/"
        - Glob pattern: "/path/to/dir/*.parquet" or "/path/to/dir/**/*.parquet"
        
        Falls back to pyarrow direct loading if HF datasets fails with nested
        data conversion errors (common with complex struct columns).
        """
        # Check if path contains glob wildcards
        if any(c in path for c in ["*", "?", "["]):
            parquet_files = sorted(glob.glob(path, recursive=True))
            if not parquet_files:
                raise FileNotFoundError(f"Glob pattern matched no files: {path}")
            print(f"  [glob] Loading {len(parquet_files)} parquet files matching: {path}")
            return self._load_parquet_files(parquet_files)

        if os.path.isdir(path):
            # Recursively find all parquet files in directory
            parquet_files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
            if not parquet_files:
                # Fallback: let HF try data_dir
                print(f"  [dir] No .parquet files found under {path}, trying data_dir fallback")
                return load_dataset("parquet", data_dir=path, split="train")
            print(f"  [dir] Found {len(parquet_files)} parquet files in: {path}")
            for f in parquet_files[:5]:
                print(f"        - {os.path.basename(f)}")
            if len(parquet_files) > 5:
                print(f"        ... and {len(parquet_files) - 5} more")
            return self._load_parquet_files(parquet_files)
        elif os.path.isfile(path):
            return self._load_parquet_files([path])
        else:
            return load_dataset(path, split="train")

    @staticmethod
    def _load_parquet_files(parquet_files: list) -> "hf_datasets_lib.Dataset":
        """Load parquet files with fallback for nested-struct Arrow errors.
        
        First tries HF ``load_dataset``; if that hits the common
        ``ArrowNotImplementedError: Nested data conversions not implemented for
        chunked array outputs`` error, falls back to loading via PyArrow directly.
        """
        try:
            return load_dataset("parquet", data_files=parquet_files, split="train")
        except Exception as e:
            if "Nested data conversions not implemented" in str(e) or "chunked array" in str(e):
                import pyarrow.parquet as pq
                print(f"  [fallback] HF load_dataset failed with Arrow nested-data error, "
                      f"loading {len(parquet_files)} file(s) via pyarrow directly...")
                tables = [pq.read_table(f) for f in parquet_files]
                if len(tables) == 1:
                    table = tables[0]
                else:
                    import pyarrow as pa
                    table = pa.concat_tables(tables, promote_options="default")
                return hf_datasets_lib.Dataset(table)
            raise

    def _detect_dataset_type(self, ds) -> str:
        """Auto-detect dataset type from schema/column names of first sample."""
        if len(ds) == 0:
            return "agibot"
        sample = ds[0]
        # Check for explicit dataset_source column
        ds_source = sample.get("dataset_source", "")
        if ds_source:
            ds_source_lower = ds_source.lower().replace("-", "_")
            if "agibot" in ds_source_lower:
                return "agibot"
            if "egodex" in ds_source_lower:
                return "egodex"
            if "videocraft" in ds_source_lower or "video_craftbench" in ds_source_lower:
                return "videocraft"
            if "epic" in ds_source_lower and "kitchen" in ds_source_lower:
                return "epic_kitchen"
            if "vr_bench" in ds_source_lower or "vr bench" in ds_source_lower.replace("_", " "):
                # VR-Bench → check task_name for sub-type
                task_name = sample.get("task_name", "").lower()
                if "maze" in task_name or "trapfield" in task_name:
                    return "maze"
                return "maze"  # default VR-Bench to maze
            if "visual_search" in ds_source_lower or "thinkmorph" in ds_source_lower:
                return "visual_search"
            if "zebra" in ds_source_lower:
                # Zebra-CoT → detect sub-type from task_name
                task_name = sample.get("task_name", "").lower()
                if "count" in task_name:
                    return "zebra_count"
                if "jigsaw" in task_name:
                    return "zebra_jigsaw"
                if "robot" in task_name:
                    return "zebra_robot"
                if "search" in task_name:
                    return "zebra_search"
                return "zebra_count"  # default Zebra-CoT fallback
        # Schema-based detection
        has_problem_images = "problem_images" in sample
        has_problem_frames = "problem_frames" in sample
        has_answer_frames = "answer_frames" in sample
        has_task_id = "task_id" in sample

        if has_problem_frames and has_answer_frames:
            return "videocraft"
        if has_task_id and has_problem_images:
            return "agibot"
        if has_problem_images:
            return "egodex"
        # Default fallback
        return "agibot"

    def _load_multi_source(self, data_path) -> list:
        """
        Load data from multiple sources.
        Returns a list of (source_type, HF_Dataset) tuples.
        
        data_path can be:
        - str: single path (auto-detect type)
        - list[str]: multiple paths (auto-detect type per path)
        - dict: {"dataset_type": path_or_list_of_paths}
        """
        # Convert OmegaConf DictConfig/ListConfig to plain Python objects
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf
            if isinstance(data_path, (DictConfig, ListConfig)):
                data_path = OmegaConf.to_container(data_path, resolve=True)
        except ImportError:
            pass

        all_datasets = []  # list of (source_type, dataset)

        if isinstance(data_path, str):
            # Single path — backward compatible
            ds = self._load_single_path(data_path)
            detected = self._detect_dataset_type(ds)
            print(f"[Emu3RLHFDataset] Auto-detected dataset type: {detected} for {data_path} ({len(ds)} samples)")
            all_datasets.append((detected, ds))

        elif isinstance(data_path, list):
            # List of paths — auto-detect each
            for p in data_path:
                ds = self._load_single_path(p)
                detected = self._detect_dataset_type(ds)
                print(f"[Emu3RLHFDataset] Auto-detected dataset type: {detected} for {p} ({len(ds)} samples)")
                all_datasets.append((detected, ds))

        elif isinstance(data_path, dict):
            # Dict: explicit type -> path(s)
            for dtype, paths in data_path.items():
                dtype = dtype.lower().replace("-", "_")
                if dtype not in self.KNOWN_DATASET_TYPES:
                    print(f"[Emu3RLHFDataset] Warning: unknown dataset type '{dtype}', treating as agibot")
                    dtype = "agibot"
                if isinstance(paths, str):
                    paths = [paths]
                for p in paths:
                    ds = self._load_single_path(p)
                    print(f"[Emu3RLHFDataset] Loaded {len(ds)} samples from {p} (type={dtype})")
                    all_datasets.append((dtype, ds))
        else:
            raise ValueError(f"data_path must be str, list[str], or dict, got {type(data_path)}")

        if len(all_datasets) == 0:
            raise ValueError("No datasets were loaded!")

        return all_datasets

    def _apply_per_source_sampling(self, datasets: list) -> list:
        """Apply per-source sampling: subsample (cap) or oversample (repeat) datasets.
        
        For each source type in ``max_samples_per_source``:
        - If target < actual count: randomly subsample to target (downsampling/cap)
        - If target > actual count: repeat the dataset to reach target (oversampling)
        - If target == actual count or target is None: no change
        
        Oversampling works by concatenating full copies of the dataset plus a
        random subset for the remainder, effectively repeating small datasets
        to balance against larger ones.
        """
        import datasets as hf_datasets
        
        sampled = []
        for source_type, ds in datasets:
            max_n = self.max_samples_per_source.get(source_type)
            if max_n is None or max_n <= 0 or len(ds) == max_n:
                sampled.append((source_type, ds))
                continue
                
            if len(ds) > max_n:
                # Downsampling: randomly pick max_n samples
                indices = list(range(len(ds)))
                random.seed(42)
                sampled_indices = sorted(random.sample(indices, max_n))
                ds_sampled = ds.select(sampled_indices)
                print(f"[Emu3RLHFDataset] Subsampled {source_type}: {len(ds)} -> {len(ds_sampled)} samples (target={max_n})")
                sampled.append((source_type, ds_sampled))
            else:
                # Oversampling: repeat dataset to reach max_n
                n_full_copies = max_n // len(ds)
                n_remainder = max_n % len(ds)
                
                copies = [ds] * n_full_copies
                if n_remainder > 0:
                    random.seed(42)
                    remainder_indices = sorted(random.sample(range(len(ds)), n_remainder))
                    copies.append(ds.select(remainder_indices))
                
                ds_oversampled = hf_datasets.concatenate_datasets(copies)
                print(f"[Emu3RLHFDataset] Oversampled {source_type}: {len(ds)} -> {len(ds_oversampled)} samples "
                      f"(target={max_n}, {n_full_copies}x full + {n_remainder} remainder)")
                sampled.append((source_type, ds_oversampled))
        return sampled

    def __len__(self):
        return self._total_len

    def _get_sample_and_source(self, index: int) -> tuple:
        """Map global index to (sample_dict, source_type) using precomputed offsets."""
        for i in range(len(self._offsets) - 1, -1, -1):
            cum_start, source_type, ds = self._offsets[i]
            if index >= cum_start:
                local_idx = index - cum_start
                return ds[local_idx], source_type
        raise IndexError(f"Index {index} out of range for dataset of size {self._total_len}")

    # ------------------------------------------------------------------ #
    #  Text helpers
    # ------------------------------------------------------------------ #
    def _get_summary(self, sample: dict) -> str:
        """Return the best prompt text: question > vlm_task_instruction > global_summary."""
        q = sample.get("question", None)
        if q and isinstance(q, str) and q.strip():
            return q.strip()
        vi = sample.get("vlm_task_instruction", None)
        if vi and isinstance(vi, str) and vi.strip():
            return vi.strip()
        gs = sample.get("global_summary", "") or ""
        return gs.strip()

    def _get_interleaved_captions(self, sample: dict, n_answer: int) -> tuple:
        """Return (problem_caption, answer_captions) for interleaved text-image mode.

        - problem_caption: caption of the first problem frame (for BoG context)
        - answer_captions: list of per-frame captions, length == n_answer
        """
        problem_images = sample.get("problem_images", [])
        if isinstance(problem_images, np.ndarray):
            problem_images = problem_images.tolist()
        problem_caption = ""
        if problem_images and isinstance(problem_images[0], dict):
            problem_caption = problem_images[0].get("caption", "") or ""

        # Try vlm_frame_captions first (maps 1:1 to answer frames)
        vlm_captions = sample.get("vlm_frame_captions", None)
        answer_captions = []
        if vlm_captions and isinstance(vlm_captions, (list, np.ndarray)):
            answer_captions = list(vlm_captions)

        # Fallback: per-frame caption field inside answer_images structs
        if not answer_captions:
            answer_images = sample.get("answer_images", [])
            if isinstance(answer_images, np.ndarray):
                answer_images = answer_images.tolist()
            for aimg in answer_images:
                if isinstance(aimg, dict):
                    answer_captions.append(aimg.get("caption", "") or "")
                else:
                    answer_captions.append("")

        # Pad / truncate to n_answer
        while len(answer_captions) < n_answer:
            answer_captions.append("")
        answer_captions = answer_captions[:n_answer]
        return problem_caption, answer_captions

    def _build_answer_ids(self, sample: dict, source_type: str,
                          answer_frames: list, sample_h, sample_w) -> list:
        """
        Assemble the answer token sequence after [BSS]:

          [BoG] summary_or_begin [EoG]
          (cap_1 img_1 cap_2 img_2 ... cap_N img_N)
          [final_answer_text]          ← interleaved mode only
          [ESS] [EOS]

        Returns a flat list of token IDs (does NOT include [BSS]).
        """
        use_interleaved = self.interleaved_text and source_type in self.INTERLEAVED_TEXT_DATASETS

        # BoG text
        if use_interleaved:
            bog_text = (sample.get("global_summary") or "").strip() or "Let's begin."
        else:
            bog_text = "Let's begin."
        summary_ids = ([self.bog_id]
                       + self.tokenizer.encode(bog_text, add_special_tokens=False)
                       + [self.eog_id])

        # Per-frame captions
        if use_interleaved:
            _, answer_captions = self._get_interleaved_captions(sample, len(answer_frames))
        else:
            answer_captions = [""] * len(answer_frames)

        # Encode answer frames (with optional caption prefix)
        interleaved_ids = []
        for fi, frame in enumerate(answer_frames):
            if "image_tokens" not in frame:
                continue
            fh = frame.get("frame_height") or frame.get("height") or sample_h
            fw = frame.get("frame_width") or frame.get("width") or sample_w
            _, fids = self._format_image_ids(frame["image_tokens"], fh, fw)
            if use_interleaved:
                cap = answer_captions[fi] if fi < len(answer_captions) else ""
                if cap:
                    interleaved_ids.extend(
                        self.tokenizer.encode(cap, add_special_tokens=False))
            interleaved_ids.extend(fids)

        # Final answer text (after last image, interleaved mode only)
        if use_interleaved:
            fa_text = (sample.get("final_answer")
                       or sample.get("text_reasoning_trace") or "")
            if fa_text and isinstance(fa_text, str) and fa_text.strip():
                interleaved_ids.extend(
                    self.tokenizer.encode(fa_text.strip(), add_special_tokens=False))

        return summary_ids + interleaved_ids + [self.ess_id, self.eos_id]

    def _get_problem_and_answer(self, sample):
        """
        Get problem (input) and answer (output) frame lists separately.
        Handles different schema formats.
        Returns: (problem_frames_list, answer_frames_list)
        """
        problem_frames = []
        answer_frames = []
        
        # Schema 1: problem_frames + answer_frames (videocraft)
        if "problem_frames" in sample and sample["problem_frames"] is not None:
            pf = sample["problem_frames"]
            problem_frames = pf.tolist() if isinstance(pf, np.ndarray) else list(pf)
        if "answer_frames" in sample and sample["answer_frames"] is not None:
            af = sample["answer_frames"]
            answer_frames = af.tolist() if isinstance(af, np.ndarray) else list(af)
        
        # Schema 2: problem_images + answer_images (agibot, egodex, new videocraft)
        if not problem_frames and "problem_images" in sample and sample["problem_images"] is not None:
            pi = sample["problem_images"]
            problem_frames = pi.tolist() if isinstance(pi, np.ndarray) else list(pi)
        if not answer_frames and "answer_images" in sample and sample["answer_images"] is not None:
            ai = sample["answer_images"]
            answer_frames = ai.tolist() if isinstance(ai, np.ndarray) else list(ai)
        
        return problem_frames, answer_frames

    def _get_dataset_source(self, sample) -> str:
        """Get the dataset source type for a sample."""
        return sample.get("__dataset_source__", "agibot")

    def _format_image_ids(self, image_tokens_bytes, h, w):
        """Convert raw image token bytes to formatted image string and token IDs."""
        tokens = np.frombuffer(image_tokens_bytes, dtype=np.int64).astype(np.int32).reshape(h, w)
        img_str = format_image_string(tokens, self.boi_token, self.eoi_token, self.img_token, self.eol_token)
        img_ids = self.tokenizer.encode(img_str, add_special_tokens=False)
        return img_str, img_ids

    # ------------------------------------------------------------------ #
    #  Build raw input IDs — dispatcher
    # ------------------------------------------------------------------ #
    def _build_raw_input_ids(self, sample, frames_data=None) -> tuple:
        """Build raw input IDs for a sample. Routes to per-dataset-type builder."""
        source = self._get_dataset_source(sample)
        if source == "videocraft":
            return self._build_raw_input_ids_videocraft(sample)
        elif source == "agibot":
            return self._build_raw_input_ids_agibot(sample)
        elif source == "egodex":
            return self._build_raw_input_ids_egodex(sample)
        elif source == "visual_search":
            return self._build_raw_input_ids_visual_search(sample)
        elif source in ("epic_kitchen", "maze", "zebra_count", "zebra_jigsaw", "zebra_robot", "zebra_search"):
            return self._build_raw_input_ids_standard(sample, source)
        else:
            # Default fallback to standard builder with agibot prompt
            return self._build_raw_input_ids_standard(sample, "agibot")

    # ------------------------------------------------------------------ #
    #  Shared problem-image encoder
    # ------------------------------------------------------------------ #
    def _encode_problem_images(self, sample, source_type: str) -> tuple:
        """Encode all problem images and return (problem_image_ids, first_image_string).

        For videocraft, tries problem_frames (per-frame h/w) before problem_images.
        For all others, uses problem_images with per-frame h/w falling back to sample h/w.
        """
        h = sample.get('height')
        w = sample.get('width')

        if source_type == "videocraft":
            frames = sample.get('problem_frames', [])
            if isinstance(frames, np.ndarray):
                frames = frames.tolist()
            use_frame_hw = bool(frames)
            if not frames:
                frames = sample.get('problem_images', [])
                if isinstance(frames, np.ndarray):
                    frames = frames.tolist()
                use_frame_hw = False
        else:
            frames = sample.get('problem_images', [])
            if isinstance(frames, np.ndarray):
                frames = frames.tolist()
            use_frame_hw = True  # per-frame h/w if present, fallback to sample h/w

        ids = []
        first_str = ""
        for pf in frames:
            if 'image_tokens' not in pf:
                continue
            if use_frame_hw:
                fh = pf.get('frame_height') or pf.get('height') or h
                fw = pf.get('frame_width') or pf.get('width') or w
            else:
                fh, fw = h, w
            img_str, img_ids = self._format_image_ids(pf['image_tokens'], fh, fw)
            ids.extend(img_ids)
            if not first_str:
                first_str = img_str
        return ids, first_str

    def _get_answer_frames(self, sample, source_type: str) -> list:
        """Return the answer frame list, applying max_images cap."""
        if source_type == "videocraft":
            frames = sample.get('answer_frames', [])
            if isinstance(frames, np.ndarray):
                frames = frames.tolist()
            if not frames:
                frames = sample.get('answer_images', [])
                if isinstance(frames, np.ndarray):
                    frames = frames.tolist()
        else:
            frames = sample.get('answer_images', [])
            if isinstance(frames, np.ndarray):
                frames = frames.tolist()

        if self.max_images is not None and len(frames) > self.max_images:
            frames = frames[:self.max_images]
        return frames

    # ------------------------------------------------------------------ #
    #  Standard builder  (agibot / egodex / epic_kitchen / zebra / etc.)
    # ------------------------------------------------------------------ #
    def _build_raw_input_ids_standard(self, sample, source_type: str) -> tuple:
        summary = self._get_summary(sample)
        prompt_template = self.PROMPT_TEMPLATES.get(
            source_type,
            "You are a helpful assistant for visual reasoning task. "
            "USER: Given the first frame, predict the following frames for the task: {task}",
        )
        user_prompt_ids = self.tokenizer.encode(
            prompt_template.format(task=summary), add_special_tokens=False)

        problem_image_ids, first_problem_image_string = self._encode_problem_images(sample, source_type)
        assistant_prompt_ids = self.tokenizer.encode(" ASSISTANT: ", add_special_tokens=False)

        answer_frames = self._get_answer_frames(sample, source_type)
        h, w = sample.get('height'), sample.get('width')
        answer_ids = self._build_answer_ids(sample, source_type, answer_frames, h, w)

        raw_input_ids = ([self.bos_id] + user_prompt_ids + problem_image_ids
                         + assistant_prompt_ids + [self.bss_id])
        return raw_input_ids, first_problem_image_string, answer_ids

    # ------------------------------------------------------------------ #
    #  VideoCraft  (may use problem_frames/answer_frames with per-frame h/w)
    # ------------------------------------------------------------------ #
    def _build_raw_input_ids_videocraft(self, sample) -> tuple:
        return self._build_raw_input_ids_standard(sample, "videocraft")

    # ------------------------------------------------------------------ #
    #  AgiBotWorld
    # ------------------------------------------------------------------ #
    def _build_raw_input_ids_agibot(self, sample) -> tuple:
        return self._build_raw_input_ids_standard(sample, "agibot")

    # ------------------------------------------------------------------ #
    #  EgoDex
    # ------------------------------------------------------------------ #
    def _build_raw_input_ids_egodex(self, sample) -> tuple:
        return self._build_raw_input_ids_standard(sample, "egodex")

    # ------------------------------------------------------------------ #
    #  Visual_Search  (prompt from 'question' column — handled by _get_summary)
    # ------------------------------------------------------------------ #
    def _build_raw_input_ids_visual_search(self, sample) -> tuple:
        return self._build_raw_input_ids_standard(sample, "visual_search")

    def _filter_overlong_prompts(self, sample: dict[str, Any], source_type: str = "agibot") -> bool:
        """Filter samples where prompt length exceeds max_prompt_length."""
        try:
            sample_with_source = dict(sample)
            sample_with_source["__dataset_source__"] = source_type
            raw_input_ids, problem_image_string, answer_ids = self._build_raw_input_ids(sample_with_source)
            return len(raw_input_ids) <= self.max_prompt_length
        except Exception as e:
            print(f"Error processing sample for filtering ({source_type}): {e}")
            return False

    def __getitem__(self, index):
        sample, source = self._get_sample_and_source(index)

        # Use _get_summary: question > vlm_task_instruction > global_summary
        global_summary = self._get_summary(sample)

        # Inject source type so dispatcher can route correctly
        sample_with_source = dict(sample)
        sample_with_source["__dataset_source__"] = source

        raw_input_ids, problem_image_string, answer_ids = self._build_raw_input_ids(sample_with_source)

        # Store raw prompt ids before padding (for vLLM)
        raw_prompt_ids = raw_input_ids.copy()
        
        # Determine sequence length for position ids
        seq_length = len(raw_input_ids)

        # Create tensors locally before post-processing
        input_ids = torch.tensor(raw_input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(seq_length, dtype=torch.long)

        # Use VF.postprocess_data for padding and truncation
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Handle raw_prompt_ids truncation for consistency if needed
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Input length {len(raw_prompt_ids)} exceeds max length {self.max_prompt_length}")

        # Prepare ground truth frames for reward computation
        gt_frames_info = []
        sample_h = sample.get('height')
        sample_w = sample.get('width')
        
        # Get answer frames based on dataset source
        _, answer_frame_list = self._get_problem_and_answer(sample)
        for frame in answer_frame_list:
            h = frame.get('frame_height') or frame.get('height') or sample_h
            w = frame.get('frame_width') or frame.get('width') or sample_w
            if 'image_tokens' in frame:
                gt_frames_info.append({
                    'image_tokens': frame['image_tokens'],
                    'height': h,
                    'width': w,
                })

        # Check for pre-decoded images (JPEG bytes)
        decoded_gt_images_bytes = []
        decoded_ref_images_bytes = []
        if 'decoded_answer_images' in sample and sample['decoded_answer_images'] is not None:
            dai = sample['decoded_answer_images']
            if isinstance(dai, np.ndarray):
                dai = dai.tolist()
            decoded_gt_images_bytes = [x for x in dai if x is not None]
        if 'decoded_problem_images' in sample and sample['decoded_problem_images'] is not None:
            dpi = sample['decoded_problem_images']
            if isinstance(dpi, np.ndarray):
                dpi = dpi.tolist()
            decoded_ref_images_bytes = [x for x in dpi if x is not None]

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "ground_truth": global_summary if isinstance(global_summary, str) else str(global_summary) if global_summary else "",
            "gt_frames_info": gt_frames_info,
            "decoded_gt_images_bytes": decoded_gt_images_bytes,
            "decoded_ref_images_bytes": decoded_ref_images_bytes,
            "dataset_source": source,
        }

        # Build unconditional prompt for CFG if enabled
        if self.enable_cfg:
            uncond_prompt = f"<|extra_203|>You are a helpful assistant. USER: {problem_image_string} ASSISTANT: <|extra_100|>"
            uncond_prompt_ids = self.tokenizer.encode(uncond_prompt, add_special_tokens=False)
            uncond_input_ids = [self.bos_id] + uncond_prompt_ids + [self.bss_id]
            output["uncond_prompt_ids"] = uncond_input_ids
        
        return output


def create_emu3_dataloader(
    config,
    tokenizer: PreTrainedTokenizer,
):
    """
    Create dataloaders for Emu3 RLHF training.
    
    Args:
        config: DataConfig with train_files, val_files, etc.
        tokenizer: Emu3 tokenizer with special tokens configured
        
    Returns:
        train_dataloader, val_dataloader
    """
    from torch.utils.data import RandomSampler, SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    train_dataset = Emu3RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        prompt_key=config.prompt_key,
        answer_key=getattr(config, 'answer_key', 'global_summary'),
        max_prompt_length=config.max_prompt_length,
        max_images=getattr(config, 'max_images', None),
        truncation="right",
        format_prompt=config.format_prompt,
        enable_cfg=getattr(config, 'enable_cfg', False),
        interleaved_text=getattr(config, 'interleaved_text', False),
    )

    if config.shuffle:
        train_generator = torch.Generator()
        train_generator.manual_seed(config.seed)
        sampler = RandomSampler(train_dataset, generator=train_generator)
    else:
        sampler = SequentialSampler(train_dataset)

    train_batch_size = config.mini_rollout_batch_size or config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=emu3_collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = Emu3RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        prompt_key=config.prompt_key,
        answer_key=getattr(config, 'answer_key', 'global_summary'),
        max_prompt_length=config.max_prompt_length,
        max_images=getattr(config, 'max_images', None),
        truncation="right",
        format_prompt=config.format_prompt,
        enable_cfg=getattr(config, 'enable_cfg', False),
        interleaved_text=getattr(config, 'interleaved_text', False),
    )

    val_batch_size = len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=emu3_collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":

    import argparse
    import pdb as _pdb

    parser = argparse.ArgumentParser(description="Debug emu3_dataset loading")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Single parquet file or directory path (auto-detect type)")
    parser.add_argument("--agibot", type=str, default=None, help="Path to Agibot parquet dir/file")
    parser.add_argument("--egodex", type=str, default=None, help="Path to EgoDex parquet dir/file")
    parser.add_argument("--videocraft", type=str, default=None, help="Path to VideoCraft parquet file")
    parser.add_argument("--epic_kitchen", type=str, default=None, help="Path to Epic Kitchen parquet dir/file")
    parser.add_argument("--maze", type=str, default=None, help="Path to Maze parquet dir/file")
    parser.add_argument("--zebra_count", type=str, default=None, help="Path to Zebra Count parquet dir/file")
    parser.add_argument("--zebra_jigsaw", type=str, default=None, help="Path to Zebra Jigsaw parquet dir/file")
    parser.add_argument("--zebra_robot", type=str, default=None, help="Path to Zebra Robot parquet dir/file")
    parser.add_argument("--zebra_search", type=str, default=None, help="Path to Zebra Search parquet dir/file")
    parser.add_argument("--tokenizer", type=str,
                        default="",
                        help="Path to Emu3 tokenizer")
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--filter_overlong", action="store_true", help="Filter overlong prompts")
    parser.add_argument("--show_sample", type=int, default=0, help="Index of sample to inspect")
    parser.add_argument("--pdb", action="store_true", help="Drop into pdb after loading")
    args = parser.parse_args()

    # Build data_path — check all known source CLI args
    _cli_sources = {
        "agibot": args.agibot, "egodex": args.egodex, "videocraft": args.videocraft,
        "epic_kitchen": args.epic_kitchen, "maze": args.maze,
        "zebra_count": args.zebra_count, "zebra_jigsaw": args.zebra_jigsaw,
        "zebra_robot": args.zebra_robot, "zebra_search": args.zebra_search,
    }
    _cli_specified = {k: v for k, v in _cli_sources.items() if v is not None}

    if _cli_specified:
        data_path = _cli_specified
    elif args.data_path:
        data_path = args.data_path
    else:
        # Default: load all sources from v2
        base = "./datasets/VR-X-RL"
        data_path = {
            "agibot": f"{base}/Agibot",
            "egodex": f"{base}/EgoDex",
            "videocraft": f"{base}/VideoCraftBench",
            "epic_kitchen": f"{base}/Epic_Kitchen",
            "maze": f"{base}/Maze",
            "zebra_count": f"{base}/Zebra_Count",
            "zebra_jigsaw": f"{base}/Zebra_Jigsaw",
            "zebra_robot": f"{base}/Zebra_Robot",
            "zebra_search": f"{base}/Zebra_Search",
        }
        print(f"[DEBUG] No --data_path specified, using default: {data_path}")
    
    from pathlib import Path
    workspace_path = Path("./")
    tokenizer_path = workspace_path / "workspace/Emu3.5_ori/src/tokenizer_emu3_ibq"

    # Load tokenizer
    from transformers import AutoTokenizer
    print(f"[DEBUG] Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Create dataset
    print(f"[DEBUG] Creating Emu3RLHFDataset with data_path={data_path}")
    dataset = Emu3RLHFDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_images=10,
        filter_overlong_prompts=False,
    )

    print("=" * 60)
    print(f"[DEBUG] Dataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Sources: {[(stype, len(ds)) for stype, ds in dataset._datasets]}")
    print("=" * 60)

    # Inspect a sample
    idx = args.show_sample
    for idx in range(10):
        print(f"\n[DEBUG] Fetching sample[{idx}]...")
        sample = dataset[idx]
        print(f"  Keys: {list(sample.keys())}")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  raw_prompt_ids length: {len(sample['raw_prompt_ids'])}")
        print(f"  dataset_source: {sample['dataset_source']}")
        print(f"  ground_truth: {sample['ground_truth'][:200]}...")
        print(f"  gt_frames_info count: {len(sample['gt_frames_info'])}")
        if sample['gt_frames_info']:
            f0 = sample['gt_frames_info'][0]
            print(f"  gt_frames_info[0]: h={f0['height']}, w={f0['width']}, tokens_bytes={len(f0['image_tokens'])} bytes")
        print(f"  decoded_gt_images_bytes count: {len(sample['decoded_gt_images_bytes'])}")
        print(f"  decoded_ref_images_bytes count: {len(sample['decoded_ref_images_bytes'])}")
    else:
        print(f"[DEBUG] Sample index {idx} out of range (dataset size={len(dataset)})")

    if args.pdb:
        print("\n[DEBUG] Dropping into pdb. Useful variables: dataset, tokenizer, sample")
        _pdb.set_trace()
