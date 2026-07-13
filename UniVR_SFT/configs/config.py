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

# exp_name = "emu3_lora_trapfieldP2_Craft_Robot_Count_Epic_Agi50k_lora128"
exp_name = ""
save_path = f"./outputs/{exp_name}/test_result"
save_to_proto = True
setup_logger(save_path)

# Model config

model_path = f"./outputs/{exp_name}/"
 
vq_path = "./weights/Emu3.5-VisionTokenizer"
tokenizer_path = "./src/tokenizer_emu3_ibq"
vq_type = "ibq"

hf_device = "auto"
vq_device = "cuda"
streaming = False
unconditional_type = "no_text"
# classifier_free_guidance = 3.0
classifier_free_guidance = 3.0
max_new_tokens = 15000
image_area = 518400

seed = 6666

# Sampling params
sampling_params = dict(
    use_cache=True,
    # text token sampling config
    text_top_k=1024,         
    text_top_p=0.9,         
    text_temperature=1.0,    

    # image token sampling config
    image_top_k=10240,      
    image_top_p=1.0,         
    image_temperature=1.0,  

    # general config
    top_k=131072,            # default topk (backward compatible)
    top_p=1.0,               # default top_p (backward compatible)
    temperature=1.0,         # default temperature (backward compatible)
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,

    # enable differential sampling
    use_differential_sampling=True,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]
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


# task_type = "visual reasoning"
task_type = "vla"
use_image = True

# ============== Input mode ==============
# "parquet": load pre-tokenized samples from PARQUET_CONFIGS / parquet_path (default).
# "manual" : use the `prompts` list below; each item is {"prompt", "reference_image"}.
input_mode = "manual"

# ============== Manual prompts (used when input_mode == "manual") ==============
# When use_image=True, each entry must be a dict with keys:
#   - "prompt":          the textual instruction
#   - "reference_image": a path (str) or list of paths (list[str]) to the input image(s)
# When use_image=False, entries may be plain strings.
_prompts_base = [
    {
        "prompt": "How to draw a cow?",
        "reference_image": "assets/Draw.png",
    },

    {
        "prompt": "How to Stir fry okra?",
        "reference_image": "assets/okra.png",
    },
    {
        "prompt": "Restock two bags of shrimp-flavored chips from the blue restocking bin into their designated spot on the snack shelf, aligning them with the existing shrimp chip packages. Finish this task in 4 steps.",
        "reference_image": "assets/robot_eval_02.jpg",
    },

    {
        "prompt": "Heat the plate of six cooked shrimp by placing it inside the orange microwave, closing the door, and starting the microwave. Finish this task in 6 steps.",
        "reference_image": "assets/robot_eval_07.jpg",
    },

    {
        "prompt": "Smoothly spread and flatten the dark green patterned tablecloth over the entire surface of the table, ensuring it is evenly laid out with no major wrinkles or folds. Finish this task in 5 steps.",
        "reference_image": "assets/robot_eval_08.jpg",
    },

    {
        "prompt": "Use both robotic arms to pick up the red rope from the table and tie it into a knot over the white gift box. Finish this task in 3 steps.",
        "reference_image": "assets/tie_rope_02.jpg",
    },
    {
        "prompt": "Fold the denim shorts in two sequential steps: first, fold the lower pant leg and waistband over the upper section, then fold the waistband down to meet the pant leg, resulting in a compact, neatly folded garment on the checkered surface. Finish this task in 2 steps.",
        "reference_image": "assets/fold_clothes_04.jpg",
    },
    {
        "prompt": "Hang the blue shirt onto the pink plastic hanger by grasping both collars and positioning the garment symmetrically on the hanger, ensuring it is fully suspended and centered. Finish this task in 7 steps.",
        "reference_image": "assets/hang_clothes_00.jpg",
    },

    {
        "prompt": "Hang a black T-shirt on a wooden hanger inside a closed wardrobe by opening the door, placing the garment on the internal rod, and closing the door afterward. Finish this task in 7 steps.",
        "reference_image": "assets/hang_clothes_03.jpg",
    }

]

if use_image:
    prompts = _prompts_base
else:
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in _prompts_base]

# ============== Multi-parquet evaluation configs ==============
# Each entry specifies: parquet path, task_type for template, and an optional label.
# The task_type here determines the prompt template wording.
PARQUET_CONFIGS = [
    {
        "path": "path to parquet",
        "task_type": "vla",
        "label": "action_eval",
    },

]

def build_unc_and_template(task: str, with_image: bool):
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        # unc_p = "<|extra_203|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: Given the first frame, how to perform the following task? {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: Given the first frame, how to perform the following task? {question} ASSISTANT: <|extra_100|>" % task_str
    return unc_p, tmpl


unc_prompt, template = build_unc_and_template(task_type, use_image)


# Image string helper
def format_image_from_bytes(tokens_bytes, height=24, width=32):
    # Tokens are int64 little endian
    tokens = np.frombuffer(tokens_bytes, dtype='<i8').reshape(height, width)
    
    image_string = ""
    start_fmt = special_tokens["BOI"]
    end_fmt = special_tokens["EOI"]
    img_tok = special_tokens["IMG"]
    eol = special_tokens["EOL"]

    for _h in range(height):
        row_string = ""
        for _w in range(width):
            row_string += "<|visual token {token_id:0>6d}|>".format(token_id=tokens[_h, _w])
        if _h < height - 1:
            row_string += eol
        image_string += row_string
    
    return "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}".format(
        image_start=start_fmt,
        token_height=height,
        token_width=width,
        image_token=img_tok,
        token_str=image_string,
        image_end=end_fmt,
    )


# Default parquet_path: use the first entry from PARQUET_CONFIGS (only relevant in parquet mode)
parquet_path = PARQUET_CONFIGS[0]["path"] if PARQUET_CONFIGS else None
parquet_paths = [pc["path"] for pc in PARQUET_CONFIGS]

if input_mode == "manual":
    print(f"Input mode: manual ({len(prompts)} prompts)")
else:
    print(f"Input mode: parquet")
    print(f"Default parquet: {parquet_path}")
    print(f"All configured parquets: {parquet_paths}")
