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
Emu3 Validation Reward Function for EasyR1 RL Training.

This module is used ONLY during validation to override the training reward function.
It provides single-sample (non-pairwise) reward computation:
- Format Reward: Validates image token format compliance
- VLM Reward: Uses Qwen3-VL to evaluate each generated image independently

NOTE: No pairwise (Pref-GRPO) reward is used here. Each sample is scored individually.
"""

import re
import base64
import io
import os
import sys
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

# Metadata
REWARD_NAME = "emu3"
REWARD_TYPE = "batch"

# NOTE: Vision tokenizer is NOT initialized in reward worker!
# Reward worker runs in a separate Ray actor without GPU access.
# Image decoding MUST happen in rollout worker (which has GPU).
# Decoded images are passed via `decoded_images` field in reward_inputs.


# ============== Image Parsing ==============

def parse_emu3_images(response: str) -> list[dict]:
    """
    Parse Emu3 image tokens from response string.
    
    Expected format: <|image start|>{h}*{w}<|image token|>{visual_tokens}<|image end|>
    where visual_tokens are like: <|visual token 000123|><|visual token 000456|>...
    with <|extra_200|> as EOL token between rows.
    
    Returns:
        List of dicts with 'height', 'width', 'token_ids' (2D numpy array)
    """
    # Pattern to match image blocks
    # <|image start|>24*32<|image token|>...<|image end|>
    image_pattern = re.compile(
        r'<\|image start\|>(\d+)\*(\d+)<\|image token\|>(.*?)<\|image end\|>',
        re.DOTALL
    )
    
    images = []
    for match in image_pattern.finditer(response):
        height = int(match.group(1))
        width = int(match.group(2))
        token_content = match.group(3)
        
        # Parse visual tokens
        # Split by EOL token <|extra_200|> to get rows
        eol_token = "<|extra_200|>"
        rows = token_content.split(eol_token)
        
        token_rows = []
        for row in rows:
            # Extract token IDs from each row
            token_ids = re.findall(r'<\|visual token (\d+)\|>', row)
            if token_ids:
                token_rows.append([int(tid) for tid in token_ids])
        
        if token_rows:
            # Validate dimensions
            try:
                token_array = np.array(token_rows, dtype=np.int32)
                actual_h, actual_w = token_array.shape
                
                images.append({
                    'height': height,
                    'width': width,
                    'actual_height': actual_h,
                    'actual_width': actual_w,
                    'token_ids': token_array,
                    'valid': (actual_h == height and actual_w == width),
                })
            except Exception as e:
                # Rows have inconsistent lengths
                images.append({
                    'height': height,
                    'width': width,
                    'actual_height': len(token_rows),
                    'actual_width': -1,  # inconsistent
                    'token_ids': None,
                    'valid': False,
                    'error': str(e),
                })
    
    return images


# NOTE: decode_image_tokens has been REMOVED from reward function.
# Image decoding happens in rollout worker (which has GPU access).
# Decoded images are passed via `decoded_images` field.


# ============== Format Reward ==============

def _check_orphan_image_tokens(response: str, valid_image_blocks: list[dict], tolerate_truncation: bool = True) -> tuple[bool, str, bool]:
    """
    Check if there are any orphan image tokens outside valid image blocks.
    
    Image-related tokens (<|image start|>, <|image token|>, <|image end|>, <|visual token xxx|>)
    must only appear within complete image blocks:
        <|image start|>{h}*{w}<|image token|>{visual_tokens}<|image end|>
    
    Args:
        response: The model response string
        valid_image_blocks: List of parsed valid image blocks
        tolerate_truncation: If True, tolerate incomplete image blocks at the END of response
                            (likely due to max_length truncation)
    
    Returns:
        (is_valid, error_message, has_truncation): 
            - is_valid: True if no orphan tokens found (or only truncated block at end)
            - error_message: Description of any issues found
            - has_truncation: True if there's an incomplete image block at the end
    """
    # First, remove all valid image blocks from response to check for orphans
    cleaned_response = response
    
    # Pattern for complete image blocks
    complete_block_pattern = re.compile(
        r'<\|image start\|>\d+\*\d+<\|image token\|>.*?<\|image end\|>',
        re.DOTALL
    )
    
    # Remove all complete image blocks
    cleaned_response = complete_block_pattern.sub('', cleaned_response)
    
    # DEBUG: Print cleaned response length and snippet
    print(f"[Emu3 Orphan Debug] Cleaned response length: {len(cleaned_response)}")
    if len(cleaned_response) > 0:
        print(f"[Emu3 Orphan Debug] Cleaned response first 300 chars: {cleaned_response[:300]}")
    
    # Check for truncated image block at the END of response
    # This happens when generation is cut off by max_length
    has_truncation = False
    truncation_info = ""
    
    if tolerate_truncation:
        # Pattern for incomplete image block at end (starts but doesn't end)
        # Case 1: <|image start|>H*W<|image token|>...visual tokens... (no <|image end|>)
        # Updated pattern to handle various token formats including spaces and any characters
        truncated_pattern = re.compile(
            r'<\|image start\|>\d+\*\d+<\|image token\|>(?:<\|[^>]+\|>)*$',
            re.DOTALL
        )
        truncated_match = truncated_pattern.search(cleaned_response)
        
        if truncated_match:
            has_truncation = True
            truncation_info = "Incomplete image block at end (likely truncated by max_length)"
            # Remove the truncated block from checking
            cleaned_response = cleaned_response[:truncated_match.start()]
            print(f"[Emu3 Orphan Debug] Found truncated block at end, removed")
        else:
            # Case 2: Just started with <|image start|> but no dimensions yet
            simple_truncated = re.compile(r'<\|image start\|>[^<]*$')
            simple_match = simple_truncated.search(cleaned_response)
            if simple_match:
                has_truncation = True
                truncation_info = "Truncated at image start"
                cleaned_response = cleaned_response[:simple_match.start()]
                print(f"[Emu3 Orphan Debug] Found simple truncated block, removed")
    
    # Now check for any remaining image-related tokens (true orphans in the middle)
    orphan_patterns = [
        (r'<\|image start\|>', 'image start'),
        (r'<\|image token\|>', 'image token'),
        (r'<\|image end\|>', 'image end'),
        (r'<\|visual token \d+\|>', 'visual token'),
    ]
    
    orphan_found = []
    for pattern, name in orphan_patterns:
        matches = re.findall(pattern, cleaned_response)
        if matches:
            orphan_found.append(f"{name}({len(matches)})")
            print(f"[Emu3 Orphan Debug] Found orphan {name}: {len(matches)} instances")
    
    if orphan_found:
        return False, f"Orphan image tokens found in middle of response: {', '.join(orphan_found)}", has_truncation
    
    return True, truncation_info, has_truncation


def format_reward(response: str, tolerate_truncation: bool = True) -> tuple[float, dict]:
    """
    Check if the response follows Emu3 image format requirements:
    1. At least one valid image in output
    2. All images have the same height*width dimensions
    3. Images follow correct format_image_string pattern
    4. All image-related tokens must appear within complete image blocks
       (no orphan <|image start|>, <|image token|>, <|image end|>, or <|visual token|>)
    
    Args:
        response: The model response string
        tolerate_truncation: If True, tolerate incomplete image blocks at the END of response
                            (likely due to max_length truncation). This allows the model to 
                            receive reward for complete images even if the last one was cut off.
    
    Returns:
        (score, details): score is 0.0 or 1.0, details contains diagnostic info
    """
    print()
    # DEBUG: Print response snippet to diagnose format issues
    print(f"[Emu3 Format Debug] Response length: {len(response)}")
    print(f"[Emu3 Format Debug] Response first 500 chars: {response[:500]}")
    print(f"[Emu3 Format Debug] Response last 500 chars: {response[-500:] if len(response) > 500 else response}")
    
    # Check if key tokens exist at all
    has_image_start = '<|image start|>' in response
    has_image_token = '<|image token|>' in response
    has_image_end = '<|image end|>' in response
    has_visual_token = '<|visual token' in response
    print(f"[Emu3 Format Debug] Token presence: image_start={has_image_start}, image_token={has_image_token}, image_end={has_image_end}, visual_token={has_visual_token}")
    
    images = parse_emu3_images(response)
    print(f"[Emu3 Format Debug] Parsed {len(images)} images")
    
    # DEBUG: Print each parsed image info
    for idx, img in enumerate(images):
        print(f"[Emu3 Format Debug] Image {idx}: height={img.get('height')}, width={img.get('width')}, "
              f"actual_h={img.get('actual_height')}, actual_w={img.get('actual_width')}, valid={img.get('valid')}")
    
    details = {
        'num_images': len(images),
        'valid_images': 0,
        'dimension_consistent': True,
        'dimensions': [],
        'has_orphan_tokens': False,
        'has_truncation': False,
    }
    
    # Check for orphan image tokens (tokens outside valid image blocks)
    # With tolerate_truncation=True, we allow incomplete blocks at the END
    is_clean, orphan_error, has_truncation = _check_orphan_image_tokens(response, images, tolerate_truncation)
    details['has_truncation'] = has_truncation
    
    # DEBUG: Print orphan check results
    print(f"[Emu3 Format Debug] Orphan check: is_clean={is_clean}, has_truncation={has_truncation}, error={orphan_error}")
    
    if not is_clean:
        details['has_orphan_tokens'] = True
        details['error'] = orphan_error
        return 0.0, details
    
    if has_truncation:
        details['truncation_info'] = orphan_error  # Contains truncation details
    
    if len(images) == 0:
        # No complete images found
        if has_truncation:
            details['error'] = 'Only truncated image found (no complete images)'
        else:
            details['error'] = 'No images found in response'
        return 0.0, details
    
    # Check each image validity
    valid_images = []
    dimensions = set()
    
    for img in images:
        if img.get('valid', False):
            valid_images.append(img)
            dim = (img['height'], img['width'])
            dimensions.add(dim)
            details['dimensions'].append(dim)
        else:
            details['dimensions'].append(
                f"invalid({img.get('height')}x{img.get('width')}, actual={img.get('actual_height')}x{img.get('actual_width')})"
            )
    
    details['valid_images'] = len(valid_images)
    
    if len(valid_images) == 0:
        details['error'] = 'No valid images (dimension mismatch or parsing error)'
        return 0.0, details
    
    # Check dimension consistency across all valid images
    if len(dimensions) > 1:
        details['dimension_consistent'] = False
        details['error'] = f'Inconsistent dimensions: {dimensions}'
        return 0.0, details
    
    return 1.0, details


# ============== Task Category & Task-Specific Criteria ==============
# (Mirrored from emu3.py — keep in sync)

def get_task_category(question_text: str = "", dataset_source: str = "") -> str | None:
    """Determine the task category from question text and dataset source."""
    ds = dataset_source.upper() if dataset_source else ""
    q = question_text.lower() if question_text else ""

    # ---------- EgoDex ----------
    if ds == "EGODEX" or "egodex" in ds.lower():
        if "assemble_disassemble_legos" in q or "build_unstack_lego" in q or "assemble_disassemble_soft_legos" in q:
            return "egodex_legos"
        if "basic_fold" in q:
            return "egodex_basic_fold"
        if "fold_unfold_paper" in q:
            return "egodex_fold_paper"
        if "insert_remove_bookshelf" in q:
            return "egodex_bookshelf"
        if "stack_unstack_bowls" in q:
            return "egodex_bowls"
        return "egodex_generic"

    # ---------- VideoCraft ----------
    if ds == "VIDEOCRAFT" or "videocraft" in ds.lower():
        if "airplane" in q or "paper airplane" in q:
            return "craft_airplane"
        if "boat" in q or "paper boat" in q:
            return "craft_boat"
        if "horse" in q:
            return "craft_horse"
        if "tower" in q:
            return "craft_tower"
        if "person" in q:
            return "craft_person"
        return "craft_generic"

    # ---------- Agibot ----------
    if "pickup" in q and "supermarket" in q:
        return "pickup_supermarket"
    if ("produce section" in q or "produce area" in q) and "pickup" in q:
        return "pickup_supermarket"
    if "open" in q and "fridge" in q and ("get food" in q or "take" in q):
        return "open_fridge"
    if "toast bread" in q or ("toast" in q and "bread" in q and "toaster" in q):
        return "toast_bread"
    if "take toast" in q or ("take" in q and "toaster" in q):
        return "take_toast"
    if "sort" in q and "warehouse" in q:
        return "sort_warehouse"
    if "fold" in q or "unfold" in q or "flatten" in q or "stack" in q:
        return "folding"
    if "clear" in q and ("countertop" in q or "counter" in q) and "waste" in q:
        return "clear_countertop"
    if ("dispose" in q or "trash" in q or "waste" in q or "garbage" in q) and ("desk" in q or "table" in q or "countertop" in q):
        return "clear_countertop"
    if "boil water" in q and "kettle" in q:
        return "boil_water"
    if "open" in q and "red wine" in q:
        return "open_wine"
    if "inner pot" in q and "rice cooker" in q:
        return "rice_cooker"
    if "rice" in q and "cooker" in q and "pot" in q:
        return "rice_cooker"
    if "wardrobe" in q and "hang" in q and "clothes" in q:
        return "hang_wardrobe"
    if "open" in q and "wardrobe" in q and "hang" in q:
        return "hang_wardrobe"
    if "swipe" in q and ("card" in q or "toy" in q):
        return "swipe_cards"
    if "hang" in q and "hanger" in q:
        return "hang_hanger"

    return None


_TASK_CRITERIA: dict[str, str] = {
    "pickup_supermarket": (
        "\n**Task Background**: This is a robot task for picking up items in a supermarket.\n"
        "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm successfully grasps the specified item "
        "and places it correctly into the shopping cart at the bottom of the image. If the item is not grasped, the wrong item is "
        "grasped, or it is not placed in the cart, the task completion score should be significantly reduced.\n"
    ),
    "open_fridge": (
        "\n**Task Background**: This is a robot task for taking food out of a refrigerator.\n"
        "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm correctly retrieves items from the "
        "refrigerator and places them on the table with the same quantity and shape as the GT. If the quantity is inconsistent, "
        "the shape has changed significantly, or the item was not successfully retrieved, the score should be significantly reduced.\n"
    ),
    "toast_bread": (
        "\n**Task Background**: This is a robot task for toasting bread.\n"
        "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm correctly places bread slices into "
        "the toaster. If bread slices do not enter the toaster or the placement position is clearly wrong, the score should be reduced.\n"
    ),
    "take_toast": (
        "\n**Task Background**: This is a robot task for taking toast slices out of the toaster.\n"
        "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm takes both toast slices out of the "
        "toaster. If the quantity is insufficient or retrieval fails, the score should be reduced.\n"
    ),
    "sort_warehouse": (
        "\n**Task Background**: This is a robot warehouse sorting task.\n"
        "**Task Completion Evaluation Focus**: Determine whether the target item has been correctly moved to the specified location. "
        "If the item does not reach the target location or is placed in the wrong location, the score should be reduced.\n"
    ),
    "folding": (
        "\n**Task Background**: This is a robot task involving folding/unfolding clothes.\n"
        "**Task Completion Evaluation Focus**: Focus on the shape change process of the clothing. It is not required that the "
        "folding process be exactly the same as the GT; as long as there is no obvious distortion and the clothing is shown "
        "being folded or unfolded step by step toward the target state, it is acceptable. The final shape should be basically "
        "consistent with the GT target state. If the clothing shows severe distortion or the final shape deviates too far from "
        "the GT, the score should be reduced.\n"
    ),
    "clear_countertop": (
        "\n**Task Background**: This is a robot task for clearing waste from a countertop.\n"
        "**Task Completion Evaluation Focus**: Focus on whether all items on the countertop have been placed into the trash bin "
        "and whether the trash bin lid is finally closed. If items are missed or the trash bin is not closed, the score should be reduced.\n"
    ),
    "boil_water": (
        "\n**Task Background**: This is a robot task for boiling water with a kettle.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the kettle is placed under the faucet to fill with water, whether "
        "the lid is closed after it is full, and whether it is placed back in its original position. If any step is missing "
        "(e.g., no water collected, lid not closed, not returned to original position), the score should be reduced.\n"
    ),
    "open_wine": (
        "\n**Task Background**: This is a robot task for opening red wine.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the wine bottle is aligned with the decanter opening and whether "
        "the wine successfully fills the decanter. If the bottle is not aligned or the wine is not poured in, the score should be reduced.\n"
    ),
    "rice_cooker": (
        "\n**Task Background**: This is a robot task for placing the inner pot into a rice cooker.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the pot is placed into the rice cooker by the robotic arm and "
        "whether the robotic arm presses the button to start cooking. If the pot is not placed in or the button is not pressed, "
        "the score should be reduced.\n"
    ),
    "hang_wardrobe": (
        "\n**Task Background**: This is a robot task for opening a wardrobe and hanging clothes.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the clothes held by the robotic arm are correctly hung inside the "
        "wardrobe and whether the wardrobe door is finally closed. If the clothes are not properly hung, the score should be reduced.\n"
    ),
    "swipe_cards": (
        "\n**Task Background**: This is a robot task for swiping toy cards.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the robotic arm grasps the white card and, while holding it, "
        "touches the card-swiping position. If the card is not grasped or the swiping position is not touched, the score should be reduced.\n"
    ),
    "hang_hanger": (
        "\n**Task Background**: This is a robot task for hanging clothes with a hanger.\n"
        "**Task Completion Evaluation Focus**: It is not required to strictly follow the GT hanging steps, but the hanger should "
        "be attached to the clothes step by step and the final shape should be similar to the GT. If the final shape deviates "
        "too far from the GT or there are serious anomalies during the process, the score should be reduced.\n"
    ),
    "egodex_legos": (
        "\n**Task Background**: This is a task involving assembly or disassembly of LEGO/soft LEGO bricks (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Focus on whether the color and quantity of bricks have mutated. Even if assembly "
        "or disassembly is completed, if the quantity and color of bricks change (appear out of nowhere, disappear, or change "
        "color), a low score should be given. It is not required that the final shape be exactly the same as the GT; different "
        "construction methods that still meet the task instruction requirements are acceptable.\n"
    ),
    "egodex_basic_fold": (
        "\n**Task Background**: This is a basic folding task (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Focus on the shape change process of the target object. It is not required that "
        "the folding process be exactly the same as the GT; as long as there is no obvious distortion and the target object is "
        "shown being folded or unfolded step by step toward the target state, it is acceptable.\n"
    ),
    "egodex_fold_paper": (
        "\n**Task Background**: This is a paper folding/unfolding task (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Focus on whether the paper deformation is reasonable, whether the creases are "
        "clear, whether the paper shape change is continuous, and whether it is progressing toward the target state step by step. "
        "If the paper shows unreasonable deformation, tearing, or deviates too far from the target state, the score should be reduced.\n"
    ),
    "egodex_bookshelf": (
        "\n**Task Background**: This is a task for inserting or removing books from a bookshelf (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Focus on whether the books are correctly placed into or taken out of the bookshelf, "
        "while the quantity and shape of the books must not change (cannot appear out of nowhere, disappear, or change shape).\n"
    ),
    "egodex_bowls": (
        "\n**Task Background**: This is a bowl stacking or unstacking task (hand-operated).\n"
        "**Task Completion Evaluation Focus**: While judging whether the task instruction is completed, focus on whether the "
        "quantity and color of the bowls have mutated (cannot appear out of nowhere, disappear, or change color).\n"
    ),
    "egodex_generic": (
        "\n**Task Background**: This is a tabletop manipulation task (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Determine whether the generated image sequence correctly executes the task "
        "instruction; the quantity, color, and shape of items during the operation should remain reasonable without obvious mutations.\n"
    ),
    "craft_airplane": (
        "\n**Task Background**: This is a paper airplane folding task. "
        "The reference step images show the correct sequence of paper airplane folding steps. "
        "Compare whether the generated image sequence follows the correct folding steps and order.\n"
    ),
    "craft_boat": (
        "\n**Task Background**: This is a paper boat folding task. "
        "The reference step images show the correct sequence of paper boat folding steps. "
        "Compare whether the generated image sequence follows the correct folding steps and order.\n"
    ),
    "craft_horse": (
        "\n**Task Background**: This is a task for building a horse with building blocks. "
        "The reference step images show the correct sequence of building steps. "
        "Compare whether the generated image sequence follows the correct building steps and order.\n"
    ),
    "craft_tower": (
        "\n**Task Background**: This is a task for building a tower with building blocks. "
        "The reference step images show the correct sequence of building steps. "
        "Compare whether the generated image sequence follows the correct building steps and order.\n"
    ),
    "craft_person": (
        "\n**Task Background**: This is a task for building a person figure with building blocks. "
        "The reference step images show the correct sequence of building steps. "
        "Compare whether the generated image sequence follows the correct building steps and order.\n"
    ),
    "craft_generic": (
        "\n**Task Background**: This is a handicraft making task. "
        "Compare whether the generated image sequence follows the correct making steps and order.\n"
    ),
}


def _get_task_specific_instruction(question_text: str, dataset_source: str = "") -> str:
    """Return task-specific evaluation instruction, or empty string if no match."""
    category = get_task_category(question_text, dataset_source)
    return _TASK_CRITERIA.get(category, "") if category else ""


# ============== VLM Reward (Qwen3-VL Evaluation) ==========================

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_vlm_evaluation_prompt(question_text: str, dataset_source: str = "") -> tuple[str, str]:
    """
    Construct the evaluation prompt for VLM reward model.
    Aligned with inference_and_evaluate_v3.py: dual-score (TASK_SCORE + VISUAL_SCORE).
    
    Args:
        question_text: The task instruction/question
        dataset_source: Dataset source identifier (e.g. "agibot", "egodex", "videocraft")
    
    Returns:
        (system_prompt, user_instruction)
    """
    category = get_task_category(question_text, dataset_source)
    task_instruction = _TASK_CRITERIA.get(category, "") if category else ""

    # Determine if this is an Agibot or EgoDex task (has specific criteria)
    _AGIBOT_CATEGORIES = {
        "pickup_supermarket", "open_fridge", "toast_bread", "take_toast",
        "sort_warehouse", "folding", "clear_countertop", "boil_water",
        "open_wine", "rice_cooker", "hang_wardrobe", "swipe_cards", "hang_hanger",
    }
    _EGODEX_CATEGORIES = {
        "egodex_legos", "egodex_basic_fold", "egodex_fold_paper",
        "egodex_bookshelf", "egodex_bowls", "egodex_generic",
    }
    has_specific_criteria = (
        (category is not None and category in _AGIBOT_CATEGORIES) or
        (category is not None and category in _EGODEX_CATEGORIES)
    )

    system_prompt = (
        "You are a reward model for multimodal generative models used in reinforcement learning training. "
        "Your task is to evaluate the quality of the generated images and provide quantitative reward scores.\n"
        f"{task_instruction}"
        "Evaluation criteria (please score each separately):\n"
        "1. **Task Completion and Logical Coherence**: Can the multiple generated images form a correct reasoning and planning "
        "process, accurately execute the task instruction, and help achieve the task goal? "
        "If GT reference images are provided, focus on comparing whether the generated steps are consistent with the GT steps."
    )
    if has_specific_criteria:
        system_prompt += " Please pay special attention to the evaluation focus described in the task background above."
    else:
        system_prompt += (
            " For origami tasks, focus on whether the paper shape changes follow the sequence shown in the reference steps, "
            "and whether the shape and fold marks of the paper at each step are similar to the reference images."
        )
    system_prompt += (
        "\n2. **Visual Consistency**: Do the multiple generated images have spatial consistency, with appearance content "
        "(objects, background, etc.) remaining coherent throughout the sequence, without significant abrupt changes or "
        "artifacts, viewable as key frames of continuous actions? Objects and people/robotic arms in the images must not "
        "show obvious distortion or deformation. Evaluate visual coherence and visual quality.\n\n"
        "Output requirements:\n"
        "First provide a brief analysis, then in the **last part** of your response, strictly output two score values between "
        "0.0 and 1.0 (floating point) in the following format:\n"
        "[[TASK_SCORE: 0.xx]]\n"
        "[[VISUAL_SCORE: 0.xx]]\n"
        "Note: Both score lines above must be output; neither can be omitted."
    )

    user_instruction = f"Task Instruction (User Question):\n{question_text}"
    
    return system_prompt, user_instruction


def call_vlm_reward(
    question_text: str,
    reference_images: list[Image.Image],
    generated_images: list[Image.Image],
    gt_images: list[Image.Image] = None,
    dataset_source: str = "",
    api_base: str = "http://<your-vlm-server>/v1",
    api_key: str = "sk-abc123",
    model_name: str = "qwen3vl",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = 60.0,
) -> tuple[float, str, float, float]:
    """
    Call VLM (Qwen3-VL) to evaluate generated images.
    Aligned with inference_and_evaluate_v3.py: returns (avg_score, explanation, task_score, visual_score).
    
    Args:
        question_text: The task instruction/question
        reference_images: Reference images from input (can be empty)
        generated_images: Model generated images to evaluate
        gt_images: Ground truth images for reference (can be empty)
        api_base: vLLM API base URL
        api_key: API key
        model_name: Model name for the API
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        timeout: Request timeout in seconds
        
    Returns:
        (avg_score, explanation, task_score, visual_score): avg_score is average of task and visual scores
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[Emu3 Reward] Warning: openai package not installed, VLM reward disabled")
        return 0.5, "openai package not installed", 0.5, 0.5
    
    if not generated_images:
        return 0.0, "No generated images to evaluate", 0.0, 0.0
    
    try:
        client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        
        system_text, instruction_text = get_vlm_evaluation_prompt(question_text, dataset_source)
        
        # Construct message content
        content_parts = []
        
        # System/Instruction context
        content_parts.append({"type": "text", "text": system_text + "\n\n" + instruction_text})
        
        # Ground Truth Images FIRST (aligned with v3: GT before Reference)
        if gt_images:
            content_parts.append({"type": "text", "text": "\n\nBelow is the Ground Truth answer image sequence for this sample (GT Answer Image Sequence):"})
            for img in gt_images:
                base64_img = encode_image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })

        # Reference Images (if any)
        if reference_images:
            content_parts.append({"type": "text", "text": "\n\nBelow are the reference images from the model input (Input Reference Images):"})
            for img in reference_images:
                base64_img = encode_image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })

        # Generated Images
        content_parts.append({"type": "text", "text": "\n\nBelow is the model-generated image sequence to be evaluated (Model Generated Sequence):"})
        for img in generated_images:
            base64_img = encode_image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })

        # Final request
        content_parts.append({"type": "text", "text": "\n\nPlease provide a detailed evaluation of the generated image sequence according to the above criteria:"})
        
        # Call API
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content_parts}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content
        
        # Extract two scores: TASK_SCORE and VISUAL_SCORE (aligned with v3)
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
            return task_score, content, task_score, task_score
        elif task_score is None and visual_score is not None:
            return visual_score, content, visual_score, visual_score
        elif single_match:
            try:
                score = max(0.0, min(1.0, float(single_match.group(1))))
                return score, content, score, score
            except ValueError:
                return 0.3, f"Could not parse score as float: {content[:500]}", 0.3, 0.3
        else:
            return 0.3, f"Could not extract score from response: {content[:500]}", 0.3, 0.3
            
    except Exception as e:
        print(f"[Emu3 Reward] api_key:{api_key}, api_base: {api_base}, VLM API call failed: {e}")
        return 0.5, f"API error: {str(e)}", 0.5, 0.5


# ============== Main Compute Score Function (Single-Sample Only) ==============


def _normalize_images(images) -> list[Image.Image]:
    """Convert a list of images (possibly np.ndarray) to list of PIL.Image."""
    if images is None:
        return []
    result = []
    for img in images:
        if isinstance(img, np.ndarray):
            result.append(Image.fromarray(img))
        elif isinstance(img, Image.Image):
            result.append(img)
        else:
            print(f"[Emu3 Val Reward] Warning: Unknown image type {type(img)}, skipping")
    return result


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.3,
    vlm_weight: float = 0.7,
    vlm_api_base: str = "http://<your-vlm-server>/v1",
    vlm_api_key: str = "sk-abc123",
    vlm_model_name: str = "qwen3vl",
    enable_vlm_reward: bool = True,
    max_vlm_workers: int = 4,
    # Legacy parameters (not used, kept for backward compatibility)
    vq_path: str = "",
    vq_type: str = "ibq",
    vq_device: str = "cuda:0",
    **kwargs,  # absorb any extra kwargs (e.g. pref_grpo params from old config)
) -> list[dict[str, float]]:
    """
    Compute reward scores for a batch of Emu3 responses (Validation-only, single-sample mode).

    Each sample is scored independently by VLM. No pairwise (Pref-GRPO) comparison is used.

    Args:
        reward_inputs: List of dicts, each containing:
            - response: The model's response string (with Emu3 image tokens)
            - ground_truth: The expected answer/task description
            - decoded_images: List of pre-decoded PIL Images from rollout worker
            - decoded_reference_images: List of reference images from input (optional)
            - decoded_gt_images: List of decoded Ground Truth images (optional)
        format_weight: Weight for format reward
        vlm_weight: Weight for VLM reward
    """
    print(f"[Emu3 Val Reward] ========== compute_score called with {len(reward_inputs)} samples ==========")
    print(f"[Emu3 Val Reward] Mode: Single-sample absolute VLM evaluation (validation only)")

    # Normalize weights
    total_weight = format_weight + vlm_weight
    format_weight = format_weight / total_weight
    vlm_weight = vlm_weight / total_weight

    # Check if we have pre-decoded images from rollout worker
    num_samples_with_images = sum(
        1 for ri in reward_inputs
        if ri.get("decoded_images") is not None and len(ri.get("decoded_images", [])) > 0
    )
    print(f"[Emu3 Val Reward] Samples with decoded_images: {num_samples_with_images}/{len(reward_inputs)}")

    if enable_vlm_reward and num_samples_with_images == 0:
        print("[Emu3 Val Reward] WARNING: No pre-decoded images from rollout worker!")
        print("[Emu3 Val Reward] Please set 'enable_image_decode_for_reward: true' in rollout config.")

    scores = []

    # Statistics
    format_pass_count = 0
    format_fail_count = 0
    no_image_count = 0

    # ============ First pass: compute format rewards, extract images ============
    sample_gen_images = []
    sample_ref_images = []
    sample_gt_images = []

    for i, reward_input in enumerate(reward_inputs):
        response = reward_input.get("response", "")
        pre_decoded_images = reward_input.get("decoded_images", None)
        pre_decoded_reference_images = reward_input.get("decoded_reference_images", None)
        pre_decoded_gt_images = reward_input.get("decoded_gt_images", None)

        # Compute format reward
        format_score, format_details = format_reward(response)

        images_data = parse_emu3_images(response)
        valid_images = [img for img in images_data if img.get('valid', False)]

        if format_score > 0:
            format_pass_count += 1
        else:
            format_fail_count += 1
            if len(images_data) == 0:
                no_image_count += 1

        score_entry = {
            "format": format_score,
            "vlm": 0.5,  # Default VLM score
            "vlm_task_score": 0.5,  # Default task score
            "vlm_visual_score": 0.5,  # Default visual score
            "num_images": len(valid_images),
            "format_details": format_details,
        }
        scores.append(score_entry)

        gen_images = _normalize_images(pre_decoded_images)
        ref_images = _normalize_images(pre_decoded_reference_images)
        gt_images = _normalize_images(pre_decoded_gt_images)
        sample_gen_images.append(gen_images)
        sample_ref_images.append(ref_images)
        sample_gt_images.append(gt_images)

    print(f"[Emu3 Val Reward] Format: {format_pass_count} passed, {format_fail_count} failed ({no_image_count} no images)")

    # ============ Second pass: Single-sample VLM evaluation ============
    vlm_tasks = []

    for i, reward_input in enumerate(reward_inputs):
        if enable_vlm_reward and scores[i]["format"] > 0 and len(sample_gen_images[i]) > 0:
            question = reward_input.get("ground_truth", "")
            if not isinstance(question, str):
                question = str(question)
            ds_source = reward_input.get("dataset_source", "")
            vlm_tasks.append((i, question, sample_ref_images[i], sample_gen_images[i], sample_gt_images[i], ds_source))
        elif scores[i]["format"] > 0 and len(sample_gen_images[i]) == 0:
            print(f"[Emu3 Val Reward] Sample {i}: Format passed but no decoded images for VLM reward")

    print(f"[Emu3 Val Reward] VLM tasks prepared: {len(vlm_tasks)}")

    if vlm_tasks:
        print(f"[Emu3 Val Reward] Running VLM evaluation for {len(vlm_tasks)} samples...")

        def evaluate_single(task):
            idx, question, ref_imgs, gen_imgs, gt_imgs, ds_src = task
            score, explanation, task_score, visual_score = call_vlm_reward(
                question_text=question,
                reference_images=ref_imgs,
                generated_images=gen_imgs,
                gt_images=gt_imgs,
                dataset_source=ds_src,
                api_base=vlm_api_base,
                api_key=vlm_api_key,
                model_name=vlm_model_name,
            )
            return idx, score, explanation, task_score, visual_score

        with ThreadPoolExecutor(max_workers=max_vlm_workers) as executor:
            futures = [executor.submit(evaluate_single, task) for task in vlm_tasks]
            for future in as_completed(futures):
                try:
                    idx, vlm_score, explanation, task_score, visual_score = future.result()
                    scores[idx]["vlm"] = vlm_score
                    scores[idx]["vlm_task_score"] = task_score
                    scores[idx]["vlm_visual_score"] = visual_score
                    scores[idx]["vlm_explanation"] = explanation[:200] if explanation else ""
                except Exception as e:
                    print(f"[Emu3 Val Reward] VLM task failed: {e}")

    # Compute overall scores
    for score_entry in scores:
        format_score = score_entry["format"]
        vlm_score = score_entry["vlm"]
        task_score = score_entry.get("vlm_task_score", vlm_score)
        visual_score = score_entry.get("vlm_visual_score", vlm_score)

        if format_score == 0:
            overall = 0.0
        else:
            # overall = format_weight * format_score + vlm_weight * vlm_score
            overall = vlm_score
        score_entry["overall"] = overall
        score_entry["task_score"] = task_score
        score_entry["visual_score"] = visual_score
        score_entry.pop("format_details", None)
        score_entry.pop("vlm_explanation", None)
        print(f"[Emu3 Val Reward] Sample score - Format: {format_score}, VLM: {vlm_score} (task={task_score:.3f}, visual={visual_score:.3f}), Overall: {overall}")

    print(f"[Emu3 Val Reward] ========== compute_score finished, returning {len(scores)} scores ==========")
    return scores


if __name__ == "__main__":
    # Example usage for testing
    pass
