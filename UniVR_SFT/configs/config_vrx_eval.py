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

from pathlib import Path
from src.utils.logging_utils import setup_logger
import pandas as pd
import numpy as np
import re
import io
from PIL import Image

cfg_name = Path(__file__).stem

exp_name = "Emu3.5_ori"
save_path = f"./outputs/{exp_name}/vrx_eval"
save_to_proto = True
setup_logger(save_path)

# Model config
model_path = "/opt/tiger/Emu3.5"

vq_path = "./weights/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

hf_device = "auto"
vq_device = "cuda"
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = 3.0
max_new_tokens = 15000
image_area = 518400

seed = 6666

# Sampling params
sampling_params = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=1.0,
    image_top_k=10240,
    image_top_p=1.0,
    image_temperature=1.0,
    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,
    use_differential_sampling=True,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = (
    sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]
)

special_tokens = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)

# ============================================================
# Input mode
# ============================================================
input_mode = "parquet"
use_image = True

# ============================================================
# Parquet configs
# ============================================================
# Each entry: path, task_type (parquet-level default), label.
# Per-sample routing is done by DATASET_TASK_TYPE_MAP / EVAL_SOURCE_TASK_TYPE_MAP below.
PARQUET_CONFIGS = [
    {
        "path": "./datasets/VR-X-Eval/eval_combined.parquet",
        "task_type": "vla",   # parquet-level fallback; per-sample routing overrides this
        "label": "vrx_eval_combined",
    },
]

parquet_path = PARQUET_CONFIGS[0]["path"]
parquet_paths = [pc["path"] for pc in PARQUET_CONFIGS]

# ============================================================
# Per-sample template routing
# ============================================================
# Priority in _get_template_for_sample():
#   1. dataset_source  → DATASET_TASK_TYPE_MAP
#   2. eval_source     → EVAL_SOURCE_TASK_TYPE_MAP  (finer-grained, breaks ties)
#   3. cfg.template / cfg.unc_prompt  (config-level default)

# Maps dataset_source column value → task_type key in task_type_templates
DATASET_TASK_TYPE_MAP = {
    # Robot manipulation (Agibot)
    "AGIBOT-WORLD-BETA-FROM-JSON":  "vla",
    "AGIBOT-WORLD-ALPHA-FROM-JSON": "vla",
    # Egocentric hand manipulation (EgoDex)
    "EGODEX":                       "first-person perspective howto",
    # Instructional how-to videos (Action100M via VR-X)
    "ACTION100M":                   "howto",
    # Visual search
    "ThinkMorph-Visual_Search":     "visual reasoning",
    # Spatial navigation in trap-field grids (VR-Bench)
    "VR-Bench-trapfield":           "visual reasoning",
    # Zebra-CoT covers both count (editing) and jigsaw (puzzle); dataset_source alone
    # is ambiguous, so it is intentionally omitted here and handled by EVAL_SOURCE_TASK_TYPE_MAP.
}

# Maps eval_source column value → task_type key (used when dataset_source is ambiguous)
EVAL_SOURCE_TASK_TYPE_MAP = {
    "robot_eval_40task_800":        "vla",
    "EgoDex":                       "first-person perspective howto",
    "Action100":                     "howto",
    "search_eval_200":              "visual reasoning",
    "Spatial_Navigation_trapfield": "visual reasoning",
    "count_eval_200":               "visual reasoning",
    "Zebra_Jigsaw":                 "visual reasoning",
}


def _build_template(task_str: str, with_image: bool):
    """Return (template, unc_prompt) for the given task label."""
    task_str = task_str.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = (
            "<|extra_203|>You are a helpful assistant for %s task. "
            "USER: Given the first frame, how to perform the following task? "
            "{question}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
        )
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = (
            "<|extra_203|>You are a helpful assistant for %s task. "
            "USER: Given the first frame, how to perform the following task? "
            "{question} ASSISTANT: <|extra_100|>" % task_str
        )
    return tmpl, unc_p


# Pre-build templates for every task_type used above
_TASK_TYPES = [
    "vla",
    "first-person perspective howto",
    "visual reasoning",
    "howto"
]

task_type_templates = {}
for _tt in _TASK_TYPES:
    _tmpl, _unc = _build_template(_tt, use_image)
    task_type_templates[_tt] = {"template": _tmpl, "unc_prompt": _unc}

# ============================================================
# Config-level default template (parquet-level fallback = "vla")
# ============================================================
task_type = "vla"

def build_unc_and_template(task: str, with_image: bool):
    return _build_template(task, with_image)

unc_prompt, template = build_unc_and_template(task_type, use_image)

# ============================================================
# Image string helper (required by inference_and_evaluate.py)
# ============================================================
def format_image_from_bytes(tokens_bytes, height=24, width=32):
    tokens = np.frombuffer(tokens_bytes, dtype='<i8').reshape(height, width)

    image_string = ""
    start_fmt  = special_tokens["BOI"]
    end_fmt    = special_tokens["EOI"]
    img_tok    = special_tokens["IMG"]
    eol        = special_tokens["EOL"]

    for _h in range(height):
        row_string = ""
        for _w in range(width):
            row_string += "<|visual token {token_id:0>6d}|>".format(
                token_id=tokens[_h, _w]
            )
        if _h < height - 1:
            row_string += eol
        image_string += row_string

    return (
        "{image_start}{token_height}*{token_width}"
        "{image_token}{token_str}{image_end}".format(
            image_start=start_fmt,
            token_height=height,
            token_width=width,
            image_token=img_tok,
            token_str=image_string,
            image_end=end_fmt,
        )
    )

# ============================================================
# Manual prompts (unused when input_mode == "parquet")
# ============================================================
prompts = []

print(f"[config] {cfg_name} loaded | parquet: {parquet_path}")
print(f"[config] task_type_templates: {list(task_type_templates.keys())}")
