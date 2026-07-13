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
Emu3 Reward Function for EasyR1 RL Training.

This module provides reward computation for Emu3.5 image generation:
- Format Reward: Validates image token format compliance
- VLM Reward: Uses Qwen3-VL to evaluate generated image quality
"""

import logging
import random
import re
import base64
import datetime
import io
import os
import sys
import requests as _requests_module
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Control verbosity via environment variable EMU3_LOG_LEVEL.
#   export EMU3_LOG_LEVEL=WARNING   # suppress INFO/DEBUG (quiet training)
#   export EMU3_LOG_LEVEL=INFO      # default – show batch-level summaries
#   export EMU3_LOG_LEVEL=DEBUG     # verbose – show per-sample details
# ---------------------------------------------------------------------------
logger = logging.getLogger("emu3_reward")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
_log_level_str = os.environ.get("EMU3_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _log_level_str, logging.INFO))

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
    logger.debug(f"[Emu3 Orphan Debug] Cleaned response length: {len(cleaned_response)}")
    if len(cleaned_response) > 0:
        logger.debug(f"[Emu3 Orphan Debug] Cleaned response first 300 chars: {cleaned_response[:300]}")
    
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
            logger.debug(f"[Emu3 Orphan Debug] Found truncated block at end, removed")
        else:
            # Case 2: Just started with <|image start|> but no dimensions yet
            simple_truncated = re.compile(r'<\|image start\|>[^<]*$')
            simple_match = simple_truncated.search(cleaned_response)
            if simple_match:
                has_truncation = True
                truncation_info = "Truncated at image start"
                cleaned_response = cleaned_response[:simple_match.start()]
                logger.debug(f"[Emu3 Orphan Debug] Found simple truncated block, removed")
    
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
            logger.debug(f"[Emu3 Orphan Debug] Found orphan {name}: {len(matches)} instances")
    
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
    logger.debug("")
    # DEBUG: Print response snippet to diagnose format issues
    logger.debug(f"[Emu3 Format Debug] Response length: {len(response)}")
    logger.debug(f"[Emu3 Format Debug] Response first 500 chars: {response[:500]}")
    logger.debug(f"[Emu3 Format Debug] Response last 500 chars: {response[-500:] if len(response) > 500 else response}")
    
    # Check if key tokens exist at all
    has_image_start = '<|image start|>' in response
    has_image_token = '<|image token|>' in response
    has_image_end = '<|image end|>' in response
    has_visual_token = '<|visual token' in response
    logger.debug(f"[Emu3 Format Debug] Token presence: image_start={has_image_start}, image_token={has_image_token}, image_end={has_image_end}, visual_token={has_visual_token}")
    
    images = parse_emu3_images(response)
    logger.debug(f"[Emu3 Format Debug] Parsed {len(images)} images")
    
    # DEBUG: Print each parsed image info
    for idx, img in enumerate(images):
        logger.debug(f"[Emu3 Format Debug] Image {idx}: height={img.get('height')}, width={img.get('width')}, "
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
    logger.debug(f"[Emu3 Format Debug] Orphan check: is_clean={is_clean}, has_truncation={has_truncation}, error={orphan_error}")
    
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


# ============== VLM Reward (Qwen3-VL Evaluation) ==============

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_vlm_evaluation_prompt(question_text: str, dataset_source: str = "") -> tuple[str, str]:
    """
    Construct the evaluation prompt for VLM reward model.
    Aligned with inference_and_evaluate_v3.py: outputs TWO scores (TASK_SCORE, VISUAL_SCORE).

    Returns:
        (system_prompt, user_instruction)
    """
    system_prompt = (
        "You are a reward model for multimodal generative models used in reinforcement learning training. "
        "Your task is to evaluate the quality of the generated images and provide quantitative reward scores.\n"
        "Evaluation criteria (please score each separately):\n"
        "1. **Task Completion and Logical Coherence**: Can the multiple generated images form a correct reasoning and planning "
        "process, accurately execute the task instruction, and achieve the task goal? Focus on:\n"
        "   - Whether the sequence progresses step by step toward the task goal (e.g., object manipulation, "
        "folding/construction steps, navigation, visual reasoning, etc.).\n"
        "   - Whether the key actions described in the task instruction (grasping, placing, folding, navigating, "
        "locating, assembling, etc.) are correctly performed.\n"
        "   - Whether objects, items, and scene elements maintain consistent quantity, color, and shape throughout the "
        "sequence — no unexpected mutations, appearances, or disappearances.\n"
        "   - Whether physical interactions are plausible and state transitions between frames are smooth and coherent.\n"
        "   - Whether the final state of the sequence matches the expected task outcome.\n"
        "   - If GT reference images are provided, compare whether the generated steps are aligned with the GT steps, "
        "especially whether the final frame matches the GT target state.\n"
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
    gt_images: list[Image.Image] | None = None,
    dataset_source: str = "",
    api_base: str = "http://<your-vlm-server>/v1",
    api_key: str = "sk-abc123",
    model_name: str = "qwen3vl",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = 200.0,
) -> tuple[float, str, float, float]:
    """
    Call VLM (Qwen3-VL) to evaluate generated images.
    
    Args:
        question_text: The task instruction/question
        reference_images: Reference images from input (can be empty)
        generated_images: Model generated images to evaluate
        api_base: vLLM API base URL
        api_key: API key
        model_name: Model name for the API
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        timeout: Request timeout in seconds
        
    Returns:
        (avg_score, explanation, task_score, visual_score)
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[Emu3 Reward] Warning: openai package not installed, VLM reward disabled")
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
        
        # A. GT Answer Images (ground truth sequence) — shown first, like v3
        if gt_images:
            content_parts.append({"type": "text", "text": "\n\nBelow is the Ground Truth answer image sequence for this sample (GT Answer Image Sequence):"})
            for img in gt_images:
                base64_img = encode_image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })

        # B. Reference Images (from model input)
        if reference_images:
            content_parts.append({"type": "text", "text": "\n\nBelow are the reference images from the model input (Input Reference Images):"})
            for img in reference_images:
                base64_img = encode_image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })

        # C. Generated Images
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
        elif task_score is not None:
            return task_score, content, task_score, task_score
        elif visual_score is not None:
            return visual_score, content, visual_score, visual_score
        elif single_match:
            try:
                score = max(0.0, min(1.0, float(single_match.group(1))))
                return score, content, score, score
            except ValueError:
                return 0.3, f"Could not parse score: {content[:500]}", 0.3, 0.3
        else:
            return 0.3, f"Score not found in response: {content[:500]}", 0.3, 0.3
            
    except Exception as e:
        logger.warning(f"[Emu3 Reward] api_key:{api_key}, api_base: {api_base}, VLM API call failed: {e}")
        return 0.5, f"API error: {str(e)}", 0.5, 0.5


# ============== HPSv3 Aesthetic Reward ==============

def call_hpsv3_reward(
    prompt_text: str,
    images: list[Image.Image],
    api_base: str = "http://localhost:8866",
    timeout: float = 120.0,
) -> tuple[float, str]:
    """
    Call HPSv3 API to get aesthetic scores for generated images.
    
    HPSv3 is deployed as a separate FastAPI service (serve_hpsv3_api.py)
    on a different GPU server to avoid OOM on the training node.
    
    Args:
        prompt_text: The task instruction / text prompt
        images: List of PIL Images to evaluate
        api_base: HPSv3 API server URL (e.g. http://<gpu_server>:8866)
        timeout: Request timeout in seconds
        
    Returns:
        (mean_score, detail_str): mean aesthetic score across all images, detail info
    """
    if not images:
        return 0.0, "No images to evaluate"
    
    try:
        # Encode all images to base64
        images_b64 = []
        prompts = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            images_b64.append(b64)
            prompts.append(prompt_text)
        
        # Call HPSv3 API
        resp = _requests_module.post(
            f"{api_base}/score",
            json={"prompts": prompts, "images_base64": images_b64},
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        
        scores = result.get("scores", [])
        if not scores:
            return 0.0, "HPSv3 returned empty scores"
        
        mean_score = sum(scores) / len(scores)
        detail = f"HPSv3 per-image scores: {[f'{s:.4f}' for s in scores]}, mean={mean_score:.4f}, elapsed={result.get('elapsed_ms', 0):.0f}ms"
        logger.debug(f"[HPSv3 Reward] {detail}")
        return mean_score, detail
        
    except Exception as e:
        logger.warning(f"[HPSv3 Reward] API call failed: {e}")
        return 0.0, f"HPSv3 API error: {str(e)}"


def call_hpsv3_reward_batch(
    prompt_texts: list[str],
    images_batch: list[list[Image.Image]],
    api_base: str = "http://localhost:8866",
    timeout: float = 120.0,
    max_workers: int = 4,
) -> list[tuple[float, str]]:
    """
    Batch call HPSv3 API for multiple samples in parallel.
    
    Flattens all images into a single API call for efficiency,
    then maps scores back to per-sample means.
    
    Args:
        prompt_texts: List of text prompts, one per sample
        images_batch: List of image lists, one list per sample
        api_base: HPSv3 API URL
        timeout: Request timeout
        max_workers: Max parallel workers for fallback
        
    Returns:
        List of (mean_score, detail) tuples, one per sample
    """
    if not images_batch:
        return []
    
    # Flatten into a single batch for one API call (more efficient)
    flat_prompts = []
    flat_images_b64 = []
    sample_sizes = []  # number of images per sample
    
    for prompt, imgs in zip(prompt_texts, images_batch):
        n = len(imgs)
        sample_sizes.append(n)
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            flat_images_b64.append(b64)
            flat_prompts.append(prompt)
    
    total_images = len(flat_prompts)
    if total_images == 0:
        return [(0.0, "No images") for _ in images_batch]
    
    try:
        logger.info(f"[HPSv3 Reward] Sending batch of {total_images} images to HPSv3 API...")
        resp = _requests_module.post(
            f"{api_base}/score",
            json={"prompts": flat_prompts, "images_base64": flat_images_b64},
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        all_scores = result.get("scores", [])
        
        if len(all_scores) != total_images:
            logger.warning(f"[HPSv3 Reward] WARNING: Expected {total_images} scores, got {len(all_scores)}")
            return [(0.0, "Score count mismatch") for _ in images_batch]
        
        # Map flat scores back to per-sample means
        results = []
        offset = 0
        for i, n in enumerate(sample_sizes):
            if n == 0:
                results.append((0.0, "No images for this sample"))
            else:
                sample_scores = all_scores[offset:offset + n]
                mean_s = sum(sample_scores) / len(sample_scores)
                detail = f"HPSv3 scores: {[f'{s:.4f}' for s in sample_scores]}, mean={mean_s:.4f}"
                results.append((mean_s, detail))
            offset += n
        
        elapsed = result.get('elapsed_ms', 0)
        logger.info(f"[HPSv3 Reward] Batch scored {total_images} images in {elapsed:.0f}ms")
        return results
        
    except Exception as e:
        logger.warning(f"[HPSv3 Reward] Batch API call failed: {e}, falling back to per-sample calls")
        # Fallback: call per-sample with threading
        def _single(args):
            prompt, imgs = args
            return call_hpsv3_reward(prompt, imgs, api_base=api_base, timeout=timeout)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_single, (p, imgs)) for p, imgs in zip(prompt_texts, images_batch)]
            return [f.result() for f in futures]


# ============== Sampled-Frames VLM Comparison Reward ==============

def get_vlm_sampled_frame_prompt(question_text: str, dataset_source: str = "") -> tuple[str, str]:
    """
    Construct the evaluation prompt for comparing a single generated frame
    with the corresponding GT frame using VLM.

    Aligned with v3: uses the same 2-criterion format (TASK_SCORE + VISUAL_SCORE).

    Returns:
        (system_prompt, user_instruction)
    """
    system_prompt = (
        "You are an image similarity evaluation model for reward computation in reinforcement learning training.\n"
        "Your task is to compare the [Generated Frame] with the [Corresponding Ground Truth Frame] "
        "and evaluate whether the generated result successfully achieves the task goal state.\n"
        "Evaluation criteria (please score each separately):\n"
        "1. **Task Completion and Goal Consistency**: Does the generated frame present the same goal state as the "
        "GT corresponding frame? Focus on:\n"
        "   - Whether key task elements (object positions, poses, operation progress, spatial relationships, etc.) "
        "match the GT frame.\n"
        "   - Whether objects, items, and scene elements maintain consistent quantity, color, and shape compared to "
        "the GT — no unexpected mutations, appearances, or disappearances.\n"
        "   - Whether the depicted action or state (grasping, placing, folding, navigating, assembling, etc.) "
        "is consistent with the task instruction and the GT goal state.\n"
        "   - Whether physical plausibility is maintained (no impossible poses, coherent interactions).\n"
        "\n2. **Visual Consistency**: The visual quality of the generated frame — whether the shape, color, and size "
        "of objects are consistent with the GT, and whether there are artifacts, distortions, or warping.\n\n"
        "Output requirements:\n"
        "First provide a brief comparative analysis, then in the **last part** of your response, strictly output two "
        "score values between 0.0 and 1.0 (floating point) in the following format:\n"
        "[[TASK_SCORE: 0.xx]]\n"
        "[[VISUAL_SCORE: 0.xx]]\n"
        "Note: Both score lines above must be output; neither can be omitted."
    )

    user_instruction = f"Task Instruction (User Question):\n{question_text}"

    return system_prompt, user_instruction


def call_vlm_frame_reward(
    question_text: str,
    reference_images: list[Image.Image],
    gen_frame: Image.Image,
    gt_frame: Image.Image,
    dataset_source: str = "",
    api_base: str = "http://<your-vlm-server>/v1",
    api_key: str = "sk-abc123",
    model_name: str = "qwen3vl",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = 200.0,
) -> tuple[float, str, float, float]:
    """
    Call VLM to compare a single generated frame with the corresponding GT frame.
    
    Args:
        question_text: The task instruction/question
        reference_images: Reference images from input (can be empty)
        gen_frame: A single generated frame (PIL Image)
        gt_frame: The corresponding GT frame (PIL Image)
        api_base: vLLM API base URL
        api_key: API key
        model_name: Model name for the API
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        timeout: Request timeout in seconds
        
    Returns:
        (avg_score, explanation, task_score, visual_score)
    """
    if gen_frame is None:
        return 0.0, "No generated frame", 0.0, 0.0
    if gt_frame is None:
        return 0.5, "No GT frame for comparison", 0.5, 0.5
    
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[Frame Reward] Warning: openai package not installed")
        return 0.5, "openai package not installed", 0.5, 0.5
    
    try:
        client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        
        system_text, instruction_text = get_vlm_sampled_frame_prompt(question_text, dataset_source)
        
        content_parts = []
        content_parts.append({"type": "text", "text": system_text + "\n\n" + instruction_text})
        
        # A. GT frame (shown first, like v3)
        content_parts.append({"type": "text", "text": "\n\nBelow is the Ground Truth corresponding frame for this sample (representing the task goal state):"})
        gt_b64 = encode_image_to_base64(gt_frame)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{gt_b64}"}
        })

        # B. Reference images (from model input)
        if reference_images:
            content_parts.append({"type": "text", "text": "\n\nBelow are the reference images from the model input (Input Reference Images):"})
            for img in reference_images:
                base64_img = encode_image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })

        # Generated frame
        content_parts.append({"type": "text", "text": "\n\nBelow is the model-generated corresponding frame (to be evaluated):"})
        gen_b64 = encode_image_to_base64(gen_frame)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{gen_b64}"}
        })

        content_parts.append({"type": "text", "text": "\n\nPlease compare the generated frame with the GT corresponding frame and evaluate the degree of goal state matching:"})
        
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
        elif task_score is not None:
            return task_score, content, task_score, task_score
        elif visual_score is not None:
            return visual_score, content, visual_score, visual_score
        elif single_match:
            try:
                score = max(0.0, min(1.0, float(single_match.group(1))))
                return score, content, score, score
            except ValueError:
                return 0.3, f"Could not parse score: {content[:500]}", 0.3, 0.3
        else:
            return 0.3, f"Score not found in response: {content[:500]}", 0.3, 0.3
            
    except Exception as e:
        logger.warning(f"[Frame Reward] VLM API call failed: {e}")
        return 0.5, f"API error: {str(e)}", 0.5, 0.5


def call_vlm_sampled_frames_reward(
    question_text: str,
    reference_images: list[Image.Image],
    generated_images: list[Image.Image],
    gt_images: list[Image.Image],
    sample_ratio: float = 0.5,
    forced_indices: list[int] | None = None,
    dataset_source: str = "",
    api_base: str = "http://<your-vlm-server>/v1",
    api_key: str = "sk-abc123",
    model_name: str = "qwen3vl",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = 200.0,
    max_vlm_workers: int = 8,
) -> tuple[float, str, list[int], list[float]]:
    """
    Sample frames from generated and GT sequences, compute per-frame VLM reward,
    and return the average.
    
    If ``forced_indices`` is provided, those exact frame indices are used instead
    of random sampling.  This allows all samples in the same Pref-GRPO group to
    be evaluated on the **same** set of frames for a fair comparison.
    
    For each sampled frame index, the generated frame and the corresponding GT frame
    (same index) are compared by VLM. The final reward is the mean of all per-frame scores.
    
    Args:
        question_text: The task instruction/question
        reference_images: Reference images from input (can be empty)
        generated_images: Model generated image sequence
        gt_images: Ground truth image sequence
        sample_ratio: Fraction of frames to sample (default 0.5 = 50%)
        forced_indices: If provided, use these exact frame indices instead of
            random sampling.  Indices that exceed the available frame count are
            silently dropped.
        api_base: vLLM API base URL
        api_key: API key
        model_name: Model name for the API
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        timeout: Request timeout in seconds
        max_vlm_workers: Max parallel VLM workers for per-frame evaluation
        
    Returns:
        (mean_score, detail, sampled_indices, per_frame_scores):
            - mean_score: average reward across sampled frames
            - detail: text description
            - sampled_indices: list of frame indices that were sampled
            - per_frame_scores: per-frame scores aligned with sampled_indices
    """
    if not generated_images:
        return 0.0, "No generated images", [], []
    if not gt_images:
        return 0.5, "No GT images for frame comparison", [], []
    
    # Use the minimum of gen/gt lengths so we always have aligned pairs
    num_frames = min(len(generated_images), len(gt_images))
    if num_frames == 0:
        return 0.0, "No aligned frames", [], []
    
    if forced_indices is not None:
        # Use caller-provided indices, dropping any out-of-range
        sampled_indices = sorted([i for i in forced_indices if i < num_frames])
        if not sampled_indices:
            return 0.0, "All forced_indices out of range", [], []
    else:
        # Determine how many frames to sample (at least 1)
        num_to_sample = max(1, int(num_frames * sample_ratio))
        # Random sample without replacement
        all_indices = list(range(num_frames))
        sampled_indices = sorted(random.sample(all_indices, num_to_sample))
    
    logger.debug(f"[SampledFrames] Sampling {len(sampled_indices)}/{num_frames} frames: {sampled_indices} (forced={'yes' if forced_indices is not None else 'no'})")
    
    # Evaluate each sampled frame in parallel
    per_frame_scores = [0.5] * len(sampled_indices)  # default
    
    def _eval_frame(args):
        local_idx, frame_idx = args
        sc, expl, _ts, _vs = call_vlm_frame_reward(
            question_text=question_text,
            reference_images=reference_images,
            gen_frame=generated_images[frame_idx],
            gt_frame=gt_images[frame_idx],
            dataset_source=dataset_source,
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return local_idx, sc, expl
    
    tasks = [(local_idx, frame_idx) for local_idx, frame_idx in enumerate(sampled_indices)]
    
    with ThreadPoolExecutor(max_workers=max_vlm_workers) as executor:
        futures = [executor.submit(_eval_frame, t) for t in tasks]
        for future in as_completed(futures):
            try:
                local_idx, sc, expl = future.result()
                per_frame_scores[local_idx] = sc
            except Exception as e:
                logger.debug(f"[Frame Reward] Frame eval failed: {e}")
    
    mean_score = sum(per_frame_scores) / len(per_frame_scores)
    detail = (f"Sampled {len(sampled_indices)}/{num_frames} frames, "
              f"indices={sampled_indices}, per_frame_scores={[f'{s:.3f}' for s in per_frame_scores]}, "
              f"mean={mean_score:.4f}")
    logger.debug(f"[Frame Reward] {detail}")
    
    return mean_score, detail, sampled_indices, per_frame_scores


# ============== Image Count Reward ==============

def image_count_reward(gen_images: list, gt_images: list) -> tuple[float, dict]:
    """
    Reward for matching the number of generated images to the GT answer image count.
    
    Score is 1.0 if the number of generated images equals the number of GT images,
    0.0 otherwise.
    
    Args:
        gen_images: List of generated PIL Images
        gt_images: List of GT PIL Images
        
    Returns:
        (score, details): score is 0.0 or 1.0, details contains diagnostic info
    """
    gen_count = len(gen_images) if gen_images else 0
    gt_count = len(gt_images) if gt_images else 0
    
    match = (gen_count == gt_count)
    score = 1.0 if match else 0.0
    
    details = {
        'gen_count': gen_count,
        'gt_count': gt_count,
        'match': match,
    }
    
    logger.debug(f"[Image Count Reward] gen={gen_count}, gt={gt_count}, match={match}, score={score}")
    return score, details


# ============== Pref-GRPO: Pairwise Preference Reward ==============

def get_vlm_pairwise_prompt(question_text: str, dataset_source: str = "") -> tuple[str, str]:
    """
    Construct the pairwise comparison prompt for PPRM (Pairwise Preference Reward Model).

    Instead of scoring a single image sequence absolutely, we compare two image sequences
    and ask the VLM to pick the better one. This produces more stable reward signals.

    Returns:
        (system_prompt, user_instruction)
    """
    system_prompt = (
        "You are a Pairwise Preference Reward Model (PPRM) for comparative evaluation of image generation quality "
        "in reinforcement learning training.\n"
        "Your task is to compare two image sequences generated from the same task instruction and determine which is better.\n"
        "You may receive a Ground Truth reference image sequence representing the ideal output for this task; "
        "please use it as an important reference for your judgment.\n"
        "Evaluation criteria:\n"
        "1. **Task Completion and Logical Coherence**: Which image sequence more accurately executes the task instruction "
        "and forms a more reasonable reasoning and planning process? Focus on:\n"
        "   - Which sequence better progresses step by step toward the task goal (object manipulation, "
        "folding/construction steps, navigation, visual reasoning, etc.).\n"
        "   - Which sequence correctly performs the key actions described in the task (grasping, placing, folding, "
        "navigating, locating, assembling, etc.).\n"
        "   - Which sequence better maintains consistent quantity, color, and shape of objects and scene elements "
        "throughout — fewer unexpected mutations, appearances, or disappearances.\n"
        "   - Which sequence shows more plausible physical interactions and smoother state transitions between frames.\n"
        "   - Which sequence's final state is closer to the expected task outcome.\n"
        "   - If GT reference images are provided, compare whether the generated steps are aligned with the GT steps, "
        "especially whether the last frame is similar to the last frame of the Ground Truth.\n"
        "\n2. **Visual Consistency**: Which image sequence is better in terms of spatial consistency and appearance coherence, "
        "with fewer artifacts, and without obvious distortion or deformation of objects and people/robotic arms.\n\n"
        "Output requirements:\n"
        "First provide a brief comparative analysis, then on the **last line** of your response, strictly output your choice "
        "in the following format:\n"
        "- If Sequence A is better, output: [[PREFER: A]]\n"
        "- If Sequence B is better, output: [[PREFER: B]]\n"
        "- If both are roughly equal in quality, output: [[PREFER: TIE]]\n"
    )

    user_instruction = f"Task Instruction (User Question):\n{question_text}"

    return system_prompt, user_instruction


def call_vlm_pairwise(
    question_text: str,
    reference_images: list[Image.Image],
    images_a: list[Image.Image],
    images_b: list[Image.Image],
    gt_images: list[Image.Image] = None,
    dataset_source: str = "",
    api_base: str = "http://<your-vlm-server>/v1",
    api_key: str = "sk-abc123",
    model_name: str = "qwen3vl",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: float = 2000.0,
) -> tuple[str, str]:
    """
    Call VLM to compare two image sequences and determine which is preferred.
    
    Args:
        question_text: The task instruction/question
        reference_images: Reference images from input (can be empty)
        images_a: First image sequence
        images_b: Second image sequence
        gt_images: Ground truth image sequence for reference (can be empty)
        
    Returns:
        (preference, explanation): preference is 'A', 'B', or 'TIE'
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "TIE", "openai package not installed"
    
    if not images_a or not images_b:
        # If one is empty, the other wins
        if images_a and not images_b:
            return "A", "B has no images"
        elif images_b and not images_a:
            return "B", "A has no images"
        return "TIE", "Both have no images"
    
    try:
        client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        
        system_text, instruction_text = get_vlm_pairwise_prompt(question_text, dataset_source)
        
        content_parts = []
        content_parts.append({"type": "text", "text": system_text + "\n\n" + instruction_text})
        
        # A. Ground Truth Images (shown first, like v3)
        if gt_images:
            content_parts.append({"type": "text", "text": "\n\nBelow is the Ground Truth answer image sequence for this sample (GT Answer Image Sequence):"})
            for img in gt_images:
                base64_img = encode_image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })

        # B. Reference Images (from model input)
        if reference_images:
            content_parts.append({"type": "text", "text": "\n\nBelow are the reference images from the model input (Input Reference Images):"})
            for img in reference_images:
                base64_img = encode_image_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })

        # Sequence A
        content_parts.append({"type": "text", "text": "\n\nBelow are the images generated by Sequence A (Sequence A):"})
        for img in images_a:
            base64_img = encode_image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })

        # Sequence B
        content_parts.append({"type": "text", "text": "\n\nBelow are the images generated by Sequence B (Sequence B):"})
        for img in images_b:
            base64_img = encode_image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })

        content_parts.append({"type": "text", "text": "\n\nPlease compare the two image sequences and determine which is better:"})
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content_parts}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content
        
        # Extract preference
        match = re.search(r'\[\[PREFER:\s*(A|B|TIE)\]\]', content, re.IGNORECASE)
        if match:
            preference = match.group(1).upper()
            return preference, content
        else:
            return "TIE", f"Preference not found in response: {content[:500]}"
            
    except Exception as e:
        logger.warning(f"[Emu3 Pref-GRPO] Pairwise VLM call failed: {e}")
        return "TIE", f"API error: {str(e)}"


def _img_to_base64(img: Image.Image, max_size: int = 256, quality: int = 75) -> str:
    """Convert a PIL Image to a base64-encoded JPEG data URI (thumbnail)."""
    try:
        thumb = img.copy()
        thumb.thumbnail((max_size, max_size), Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BILINEAR)
        if thumb.mode == "RGBA":
            thumb = thumb.convert("RGB")
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def _score_color(value: float, low: float = 0.0, high: float = 1.0) -> str:
    """Return a CSS color string (red→yellow→green) based on score in [low, high]."""
    ratio = max(0.0, min(1.0, (value - low) / (high - low) if high > low else 0.5))
    # red (0) → orange (0.25) → yellow (0.5) → green (1.0)
    if ratio < 0.5:
        r, g = 220, int(80 + 340 * ratio)
    else:
        r, g = int(220 - 280 * (ratio - 0.5)), 200
    return f"rgb({r},{g},60)"


def _save_pref_grpo_visualization(
    uid_to_indices: dict[str, list[int]],
    scores: list[dict],
    sample_gen_images: list[list[Image.Image]],
    sample_ref_images: list[list[Image.Image]],
    sample_gt_images: list[list[Image.Image]],
    reward_inputs: list[dict],
    save_dir: str = "./debug_outputs/pref_grpo_vis",
    max_groups: int = 50,
):
    """
    Save a self-contained HTML visualization for ALL Pref-GRPO groups.

    Called AFTER all rewards (format, win_rate, sampled_frames, dinov2, hpsv3,
    image_count, overall) have been computed, so every score is available.

    The HTML file embeds all images as base64 thumbnails, shows:
    - Group-level info (question / uid / ref images / GT images)
    - Per-sample reward breakdown (format, win_rate, sampled_frames, dinov2,
      hpsv3, image_count, overall) with color-coded score badges
    - Generated image sequence per sample, with sampled-frame indices
      highlighted and their per-frame VLM scores annotated
    - Samples sorted by overall score (best first) within each group

    Output:
        {save_dir}/batch_{timestamp}.html
    """

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(save_dir, f"batch_{timestamp}.html")

    # ---- CSS ----
    css = """
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #222; padding: 20px; }
      h1 { text-align: center; margin-bottom: 10px; }
      .meta { text-align: center; color: #666; margin-bottom: 24px; font-size: 14px; }
      .group-card { background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    margin-bottom: 28px; padding: 20px; }
      .group-header { display: flex; justify-content: space-between; align-items: center;
                      border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 14px; }
      .group-title { font-size: 18px; font-weight: 700; }
      .group-info { font-size: 13px; color: #888; }
      .question { background: #f0f4ff; border-left: 4px solid #4a7dff; padding: 8px 12px;
                  margin-bottom: 14px; font-size: 13px; white-space: pre-wrap; word-break: break-word;
                  max-height: 120px; overflow-y: auto; }
      .img-section { margin-bottom: 14px; }
      .img-section-title { font-size: 13px; font-weight: 600; color: #555; margin-bottom: 6px; }
      .img-row { display: flex; flex-wrap: wrap; gap: 4px; align-items: flex-end; }
      .img-cell { text-align: center; }
      .img-cell img { border-radius: 4px; border: 2px solid #ddd; display: block; }
      .img-cell .idx-label { font-size: 10px; color: #999; margin-top: 2px; }
      .sample-card { background: #fafbfc; border: 1px solid #e8e8e8; border-radius: 8px;
                     padding: 12px; margin-bottom: 12px; }
      .sample-header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
                       margin-bottom: 8px; }
      .sample-rank { font-size: 20px; font-weight: 800; color: #4a7dff; min-width: 36px; }
      .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
               font-size: 12px; font-weight: 600; color: #fff; }
      .score-table { font-size: 12px; border-collapse: collapse; margin-bottom: 8px; }
      .score-table td, .score-table th { padding: 3px 10px; border: 1px solid #e0e0e0; }
      .score-table th { background: #f0f0f0; font-weight: 600; }
      .frame-box { position: relative; display: inline-block; }
      .frame-box img { display: block; }
      .frame-box.sampled img { border-color: #ff6b35 !important; border-width: 3px !important; }
      .frame-score-tag { position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
                         background: rgba(0,0,0,0.72); color: #fff; font-size: 10px; font-weight: 700;
                         padding: 1px 5px; border-radius: 4px; white-space: nowrap; }
      .legend { display: flex; gap: 16px; font-size: 12px; color: #666; margin-bottom: 6px; }
      .legend span { display: inline-flex; align-items: center; gap: 4px; }
      .legend .dot { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
      .collapsible { cursor: pointer; user-select: none; }
      .collapsible::before { content: '▶ '; font-size: 12px; }
      .collapsible.open::before { content: '▼ '; }
      .collapse-content { display: none; }
      .collapse-content.show { display: block; }
    </style>
    """

    # ---- JS for collapsible sections ----
    js = """
    <script>
    function toggleCollapse(el) {
      el.classList.toggle('open');
      var content = el.nextElementSibling;
      content.classList.toggle('show');
    }
    </script>
    """

    # ---- Build HTML body ----
    num_groups = len(uid_to_indices)
    total_samples = sum(len(v) for v in uid_to_indices.values())
    groups_html_parts = []
    saved_count = 0

    for uid, indices in uid_to_indices.items():
        if saved_count >= max_groups:
            break

        uid_short = uid[:12].replace("/", "_")
        G = len(indices)
        question = reward_inputs[indices[0]].get("ground_truth", "") if indices else ""
        if not isinstance(question, str):
            question = str(question)

        # ---- Reference images (shared) ----
        ref_imgs = sample_ref_images[indices[0]] if indices and sample_ref_images else []
        ref_html = ""
        if ref_imgs:
            cells = []
            for j, img in enumerate(ref_imgs):
                uri = _img_to_base64(img)
                if uri:
                    cells.append(f'<div class="img-cell"><img src="{uri}" width="120"><div class="idx-label">ref_{j}</div></div>')
            if cells:
                ref_html = f'<div class="img-section"><div class="img-section-title">Reference Images ({len(ref_imgs)})</div><div class="img-row">{"".join(cells)}</div></div>'

        # ---- GT images (shared) ----
        gt_imgs = sample_gt_images[indices[0]] if indices and sample_gt_images else []
        gt_html = ""
        if gt_imgs:
            cells = []
            for j, img in enumerate(gt_imgs):
                uri = _img_to_base64(img)
                if uri:
                    cells.append(f'<div class="img-cell"><img src="{uri}" width="120"><div class="idx-label">gt_{j}</div></div>')
            if cells:
                gt_html = f'<div class="img-section"><div class="img-section-title">Ground Truth Images ({len(gt_imgs)})</div><div class="img-row">{"".join(cells)}</div></div>'

        # ---- Per-sample cards (sorted by overall, best first) ----
        sorted_enum = sorted(enumerate(indices), key=lambda x: scores[x[1]].get("overall", 0.0), reverse=True)
        samples_html_parts = []

        for rank, (k, idx) in enumerate(sorted_enum):
            s = scores[idx]
            gen_imgs = sample_gen_images[idx] if idx < len(sample_gen_images) else []

            # Retrieve sampled-frame info
            sampled_indices = s.get("sampled_frames_indices", []) or []
            per_frame_scores = s.get("sampled_frames_per_frame_scores", []) or []
            sampled_set = set(sampled_indices)
            # Build index -> per-frame score mapping
            idx_to_ff_score = {}
            for si, sf in zip(sampled_indices, per_frame_scores):
                idx_to_ff_score[si] = sf

            # Score badge helper
            def _badge(label, val, fmt=".3f", lo=0.0, hi=1.0):
                color = _score_color(val, lo, hi)
                return f'<span class="badge" style="background:{color}">{label} {val:{fmt}}</span>'

            overall = s.get("overall", 0.0)
            badges = " ".join([
                _badge("Overall", overall),
                _badge("WinRate", s.get("win_rate", 0.5)),
                _badge("Format", s.get("format", 0), ".1f"),
                _badge("SampledFrames", s.get("sampled_frames", 0.5)),
                _badge("DINOv2", s.get("dinov2", 0), ".4f"),
                _badge("HPSv3", s.get("hpsv3", 0), ".4f"),
                _badge("ImgCount", s.get("image_count", 0), ".1f"),
            ])

            # Score detail table
            score_keys = [
                ("overall", "Overall", ".4f"),
                ("format", "Format", ".1f"),
                ("win_rate", "Win Rate", ".3f"),
                ("sampled_frames", "Sampled Frames (avg)", ".3f"),
                ("dinov2", "DINOv2", ".4f"),
                ("hpsv3", "HPSv3", ".4f"),
                ("image_count", "Image Count", ".1f"),
                ("vlm", "VLM (abs)", ".3f"),
                ("num_images", "Num Images", "d"),
            ]
            table_rows = ""
            for key, label, fmt in score_keys:
                val = s.get(key, 0)
                if isinstance(val, float):
                    val_str = f"{val:{fmt}}"
                else:
                    val_str = str(val)
                table_rows += f"<tr><th>{label}</th><td>{val_str}</td></tr>"

            # Sampled-frame detail rows
            if sampled_indices:
                frame_detail_cells = []
                for si, sf in zip(sampled_indices, per_frame_scores):
                    c = _score_color(sf)
                    frame_detail_cells.append(f'<span class="badge" style="background:{c};font-size:11px">F{si}: {sf:.2f}</span>')
                frame_detail_str = " ".join(frame_detail_cells)
                table_rows += f'<tr><th>Sampled Frames</th><td style="line-height:1.8">{frame_detail_str}</td></tr>'

            # Generated images with sampled-frame highlighting
            gen_cells = []
            for j, img in enumerate(gen_imgs):
                uri = _img_to_base64(img)
                if not uri:
                    continue
                is_sampled = j in sampled_set
                cls = "frame-box sampled" if is_sampled else "frame-box"
                score_tag = ""
                if is_sampled and j in idx_to_ff_score:
                    fscore = idx_to_ff_score[j]
                    score_tag = f'<div class="frame-score-tag">{fscore:.2f}</div>'
                border_col = "#ff6b35" if is_sampled else "#ddd"
                gen_cells.append(
                    f'<div class="{cls}">'
                    f'<img src="{uri}" width="120" style="border:2px solid {border_col};border-radius:4px;">'
                    f'{score_tag}'
                    f'<div style="font-size:10px;color:#999;text-align:center;margin-top:2px;">gen_{j}</div>'
                    f'</div>'
                )

            gen_html = f'<div class="img-row" style="gap:4px;flex-wrap:wrap;">{"".join(gen_cells)}</div>' if gen_cells else '<div style="color:#aaa;font-size:13px;">No generated images</div>'

            sample_html = f"""
            <div class="sample-card" style="border-left:4px solid {_score_color(overall)}">
              <div class="sample-header">
                <span class="sample-rank">#{rank+1}</span>
                <span style="font-size:13px;color:#666;">Sample {k} (idx={idx})</span>
                {badges}
              </div>
              <div class="collapsible" onclick="toggleCollapse(this)">Score Details & Sampled-Frame Breakdown</div>
              <div class="collapse-content">
                <table class="score-table" style="margin:6px 0">{table_rows}</table>
              </div>
              <div class="img-section" style="margin-top:6px;">
                <div class="legend">
                  <span><span class="dot" style="background:#ff6b35;"></span> Sampled frame (VLM scored)</span>
                  <span><span class="dot" style="background:#ddd;"></span> Not sampled</span>
                </div>
                {gen_html}
              </div>
            </div>
            """
            samples_html_parts.append(sample_html)

        # ---- Assemble group card ----
        group_html = f"""
        <div class="group-card">
          <div class="group-header">
            <div class="group-title">Group: {uid_short}</div>
            <div class="group-info">{G} samples &nbsp;|&nbsp; UID: {uid[:32]}{'...' if len(uid) > 32 else ''}</div>
          </div>
          <div class="question"><b>Question / Prompt:</b><br>{question[:1000]}{'...' if len(question) > 1000 else ''}</div>
          {ref_html}
          {gt_html}
          <hr style="border:none;border-top:1px dashed #ddd;margin:12px 0;">
          {"".join(samples_html_parts)}
        </div>
        """
        groups_html_parts.append(group_html)
        saved_count += 1

    # ---- Full HTML ----
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pref-GRPO Reward Visualization — {timestamp}</title>
{css}
</head>
<body>
<h1>🏆 Pref-GRPO Reward Visualization</h1>
<div class="meta">Generated: {timestamp} &nbsp;|&nbsp; Groups: {saved_count}/{num_groups} &nbsp;|&nbsp; Total Samples: {total_samples}</div>
{"".join(groups_html_parts)}
{js}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"[Pref-GRPO Vis] Saved HTML visualization ({saved_count} groups) to {html_path}")
    return html_path


def compute_pref_grpo_win_rates(
    group_indices: list[int],
    group_images: list[list[Image.Image]],
    group_ref_images: list[list[Image.Image]],
    group_gt_images: list[list[Image.Image]],
    question_text: str,
    dataset_source: str = "",
    vlm_api_base: str = "",
    vlm_api_key: str = "",
    vlm_model_name: str = "",
    max_vlm_workers: int = 4,
) -> dict[int, float]:
    """
    Compute Pref-GRPO win rates for a group of samples from the same prompt.
    
    For G images sampled from policy for prompt c, enumerate all pairs (i, j)
    and use PPRM to determine preferred image. Win rate for image i:
        w_i = (1/(G-1)) * sum_{j!=i} I(x_i > x_j)
    
    Args:
        group_indices: Original indices into the scores list
        group_images: List of decoded image lists, one per sample in group
        group_ref_images: List of reference image lists, one per sample
        question_text: The task instruction
        vlm_api_base: VLM API base URL
        vlm_api_key: API key
        vlm_model_name: Model name
        max_vlm_workers: Max parallel workers
        
    Returns:
        Dict mapping original_index -> win_rate (0.0 to 1.0)
    """
    G = len(group_indices)
    if G <= 1:
        # Single sample, win rate = 0.5 (neutral)
        return {group_indices[0]: 0.5}
    
    # Use first sample's reference/GT images (they should all be the same within a group)
    ref_images = group_ref_images[0] if group_ref_images else []
    gt_images = group_gt_images[0] if group_gt_images and len(group_gt_images) > 0 else []
    
    # Enumerate all pairs and prepare VLM tasks
    pair_tasks = []  # (local_i, local_j)
    for i in range(G):
        for j in range(i + 1, G):
            # Only compare if both have images
            if group_images[i] and group_images[j]:
                pair_tasks.append((i, j))
    
    if not pair_tasks:
        # No valid pairs, all get 0.5
        return {idx: 0.5 for idx in group_indices}
    
    logger.info(f"[Pref-GRPO] Group size={G}, pairs to compare={len(pair_tasks)}")
    
    # Track wins for each local index
    wins = {i: 0.0 for i in range(G)}
    comparisons = {i: 0 for i in range(G)}
    
    def compare_pair(pair):
        local_i, local_j = pair
        preference, explanation = call_vlm_pairwise(
            question_text=question_text,
            reference_images=ref_images,
            images_a=group_images[local_i],
            images_b=group_images[local_j],
            gt_images=gt_images,
            dataset_source=dataset_source,
            api_base=vlm_api_base,
            api_key=vlm_api_key,
            model_name=vlm_model_name,
        )
        return local_i, local_j, preference
    
    # Parallel pairwise comparisons
    with ThreadPoolExecutor(max_workers=max_vlm_workers) as executor:
        futures = [executor.submit(compare_pair, pair) for pair in pair_tasks]
        for future in as_completed(futures):
            try:
                local_i, local_j, preference = future.result()
                comparisons[local_i] += 1
                comparisons[local_j] += 1
                if preference == "A":
                    wins[local_i] += 1.0
                elif preference == "B":
                    wins[local_j] += 1.0
                else:  # TIE
                    wins[local_i] += 0.5
                    wins[local_j] += 0.5
            except Exception as e:
                logger.warning(f"[Pref-GRPO] Pair comparison failed: {e}")
    
    # Compute win rates: w_i = wins_i / (G - 1)
    win_rates = {}
    for local_i in range(G):
        if comparisons[local_i] > 0:
            win_rates[group_indices[local_i]] = wins[local_i] / comparisons[local_i]
        else:
            win_rates[group_indices[local_i]] = 0.5  # No comparisons, neutral
    
    logger.info(f"[Pref-GRPO] Win rates: {win_rates}")
    
    # Save visualization for debugging
    # try:
    #     _save_pref_grpo_visualization(
    #         group_indices=group_indices,
    #         group_images=group_images,
    #         group_ref_images=group_ref_images,
    #         group_gt_images=group_gt_images,
    #         question_text=question_text,
    #         win_rates=win_rates,
    #     )
    # except Exception as e:
    #     print(f"[Pref-GRPO Vis] Failed to save visualization: {e}")
    
    return win_rates


# ============== DINOv2 Feature Similarity Reward ==============

class DINOv2FeatureExtractor:
    """Singleton DINOv2 feature extractor with configurable pooling."""
    _instance = None
    _model = None
    _processor = None
    _device = None

    @classmethod
    def get_instance(cls, model_name="facebook/dinov2-base", device="cpu"):
        if cls._instance is None:
            cls._instance = cls()
            logger.info(f"[DINOv2] Loading model {model_name} on {device}...")
            cls._processor = AutoImageProcessor.from_pretrained(model_name)
            cls._model = AutoModel.from_pretrained(model_name).to(device)
            cls._model.eval()
            cls._device = device
            logger.info(f"[DINOv2] Model loaded.")
        return cls._instance

    @classmethod
    def extract_features(cls, images: list[Image.Image], pooling: str = "avg_patch"):
        """
        Extract L2-normalized DINOv2 features from images.

        Args:
            images: List of PIL images.
            pooling: Feature pooling strategy.
                - "cls":       CLS token only. Captures high-level semantics but
                               discards spatial structure.
                - "avg_patch": Global average of all *patch* tokens (excluding CLS).
                               Preserves both semantic and structural information.
                - "avg_all":   Average of CLS + patch tokens.

        Returns:
            L2-normalized tensor of shape ``[batch, hidden_dim]``.
        """
        if not images:
            return None

        try:
            inputs = cls._processor(images=images, return_tensors="pt").to(cls._device)
            with torch.no_grad():
                outputs = cls._model(**inputs)

            # last_hidden_state: [batch, 1 + num_patches, hidden_dim]
            last_hidden_states = outputs.last_hidden_state

            if pooling == "cls":
                features = last_hidden_states[:, 0, :]            # [batch, D]
            elif pooling == "avg_patch":
                features = last_hidden_states[:, 1:, :].mean(dim=1)  # [batch, D]
            elif pooling == "avg_all":
                features = last_hidden_states.mean(dim=1)            # [batch, D]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

            # Always L2-normalize so that downstream metrics have a stable scale.
            # After normalization ||f||=1, so ||f1-f2||^2 = 2(1-cos(f1,f2)) ∈ [0,4].
            features = F.normalize(features, p=2, dim=-1)
            return features
        except Exception as e:
            logger.warning(f"[DINOv2] Feature extraction failed: {e}")
            return None


def _dinov2_score(feat_a: torch.Tensor, feat_b: torch.Tensor,
                  metric: str = "gaussian_rbf", rbf_sigma: float = 3.0) -> float:
    """
    Compute a [0, 1] similarity score between two L2-normalized feature vectors.

    Metrics:
        cosine:        dot(f1, f2)  ∈ [-1, 1], clamped to [0, 1].
        gaussian_rbf:  exp(-σ · ||f1 - f2||²).  On L2-normalised features
                       ||f1-f2||² = 2(1-cos θ) ∈ [0, 4], so the output
                       smoothly maps to (0, 1] and is more sensitive to
                       small structural differences than raw cosine.
                       σ controls sharpness (higher → more peaked).
    """
    if metric == "cosine":
        sim = (feat_a * feat_b).sum().item()
        return max(0.0, min(1.0, sim))
    elif metric == "gaussian_rbf":
        sq_dist = ((feat_a - feat_b) ** 2).sum().item()  # ∈ [0, 4] for L2-normed
        return float(np.exp(-rbf_sigma * sq_dist))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_dinov2_similarity(
    gen_image: Image.Image,
    gt_image: Image.Image,
    model_name: str = "facebook/dinov2-base",
    device: str = "cpu",
    pooling: str = "avg_patch",
    metric: str = "gaussian_rbf",
    rbf_sigma: float = 3.0,
) -> float:
    """
    Compute DINOv2 feature similarity between two images.

    Args:
        pooling: "cls" | "avg_patch" | "avg_all"
        metric:  "cosine" | "gaussian_rbf"
        rbf_sigma: Bandwidth for Gaussian RBF (only used when metric="gaussian_rbf")
    """
    DINOv2FeatureExtractor.get_instance(model_name, device)

    feats = DINOv2FeatureExtractor.extract_features([gen_image, gt_image], pooling=pooling)
    if feats is None or feats.shape[0] != 2:
        return 0.0

    return _dinov2_score(feats[0], feats[1], metric=metric, rbf_sigma=rbf_sigma)


def compute_dinov2_similarity_batch(
    gen_images: list[Image.Image],
    gt_images: list[Image.Image],
    model_name: str = "facebook/dinov2-base",
    device: str = "cuda",
    pooling: str = "avg_patch",
    metric: str = "gaussian_rbf",
    rbf_sigma: float = 3.0,
) -> list[float]:
    """
    Batch compute DINOv2 feature similarity between paired (gen, gt) images.

    Args:
        pooling: "cls" | "avg_patch" | "avg_all"
        metric:  "cosine" | "gaussian_rbf"
        rbf_sigma: Bandwidth for Gaussian RBF
    """
    if not gen_images or not gt_images or len(gen_images) != len(gt_images):
        return []

    DINOv2FeatureExtractor.get_instance(model_name, device)

    # Interleave: [gen0, gt0, gen1, gt1, ...]
    batch_imgs = []
    for g, t in zip(gen_images, gt_images):
        batch_imgs.append(g)
        batch_imgs.append(t)

    feats = DINOv2FeatureExtractor.extract_features(batch_imgs, pooling=pooling)
    if feats is None:
        return [0.0] * len(gen_images)

    scores = []
    for i in range(0, len(feats), 2):
        scores.append(_dinov2_score(feats[i], feats[i + 1], metric=metric, rbf_sigma=rbf_sigma))

    return scores


# ============== Main Compute Score Function ==============

def compute_score(
    reward_inputs: list[dict[str, Any]], 
    format_weight: float = 0.1,
    vlm_weight: float = 0.7,
    vlm_api_base: str = "http://<your-vlm-server>/v1",
    vlm_api_key: str = "sk-abc123",
    vlm_model_name: str = "qwen3vl",
    enable_vlm_reward: bool = True,
    max_vlm_workers: int = 8,
    # HPSv3 Aesthetic Reward parameters
    enable_hpsv3_reward: bool = False,
    hpsv3_api_base: str = "http://localhost:8866",
    hpsv3_weight: float = 0.4,
    hpsv3_timeout: float = 120.0,
    # Final-frame (sampled frames) VLM comparison parameters
    enable_sampled_frames_reward: bool = True,
    sampled_frames_weight: float = 1.0,
    sampled_frames_ratio: float = 0.5,   # fraction of frames to sample (default 50%)
    sampled_frames_api_base: str = "",  # defaults to vlm_api_base if empty
    sampled_frames_api_key: str = "",   # defaults to vlm_api_key if empty
    sampled_frames_model_name: str = "",  # defaults to vlm_model_name if empty
    # DINOv2 Feature Similarity Reward parameters
    enable_dinov2_reward: bool = False,
    dinov2_weight: float = 1.0,
    dinov2_model_name: str = "",
    dinov2_device: str = "cuda",
    dinov2_pooling: str = "avg_patch",       # "cls" | "avg_patch" | "avg_all"
    dinov2_metric: str = "gaussian_rbf",     # "cosine" | "gaussian_rbf"
    dinov2_rbf_sigma: float = 3.0,           # RBF bandwidth (higher = more peaked)
    # Image count reward parameters
    enable_image_count_reward: bool = True,
    image_count_weight: float = 0.2,
    # Pref-GRPO parameters
    enable_pref_grpo: bool = True,
    pref_grpo_winrate_weight: float = 1.0,
    pref_grpo_hpsv3_weight: float = 0.0,
    # Debug mode for sanity-checking the reward pipeline
    # ""              = normal mode (use actual generated images)
    # "gt_as_gen"     = replace generated images with GT images (expect score ≈ 1.0)
    # "gt_first_frame" = replace generated images with GT[0] repeated N times (expect lower score)
    debug_mode: str = "",
    # Legacy parameters (not used, kept for backward compatibility)
    vq_path: str = "",
    vq_type: str = "ibq",
    vq_device: str = "cuda:0",
) -> list[dict[str, float]]:
    """
    Compute reward scores for a batch of Emu3 responses.
    
    Supports two modes:
    1. Absolute VLM reward (default): Each sample is scored independently by VLM
    2. Pref-GRPO (enable_pref_grpo=True): Samples from the same prompt (same uid) are 
       compared pairwise using VLM, and win rates are used as rewards instead of absolute scores.
       
       Pref-GRPO reference: "Instead of relying on absolute reward scores, PREF-GRPO evaluates 
       relative preferences among generated images, mirroring the human process of assessing 
       two comparable images."
       
       Win rate formula: w_i = (1/(G-1)) * sum_{j!=i} I(x_i > x_j)
       Advantage: A_i_t = (w_i - mean(w)) / std(w)
    
    Args:
        reward_inputs: List of dicts, each containing:
            - response: The model's response string (with Emu3 image tokens)
            - ground_truth: The expected answer/task description
            - uid: Unique ID grouping samples from the same prompt (REQUIRED for Pref-GRPO)
            - decoded_images: List of pre-decoded PIL Images from rollout worker
            - decoded_reference_images: List of reference images from input (optional)
            - decoded_gt_images: List of GT images for pairwise comparison (optional)
        format_weight: Weight for format reward (absolute mode)
        vlm_weight: Weight for VLM reward (absolute mode)
        enable_hpsv3_reward: Enable HPSv3 aesthetic reward (requires HPSv3 API server)
        hpsv3_api_base: HPSv3 API server URL (e.g. http://<gpu_server>:8866)
        hpsv3_weight: Weight for HPSv3 reward (absolute mode, added to format+vlm)
        hpsv3_timeout: HPSv3 API request timeout in seconds
        enable_sampled_frames_reward: Enable sampled-frame VLM comparison reward (randomly samples 50% frames)
        sampled_frames_weight: Weight for sampled-frame reward (shared across pref/non-pref modes)
        sampled_frames_ratio: Fraction of frames to sample for comparison (default 0.5)
        sampled_frames_api_base: API base for frame VLM (defaults to vlm_api_base if empty)
        sampled_frames_api_key: API key for frame VLM (defaults to vlm_api_key if empty)
        sampled_frames_model_name: Model name for frame VLM (defaults to vlm_model_name if empty)
        enable_dinov2_reward: Enable DINOv2 feature similarity reward between last gen and GT frames
        dinov2_weight: Weight for DINOv2 reward (shared across pref/non-pref modes)
        dinov2_model_name: DINOv2 model name (default: facebook/dinov2-base)
        dinov2_device: Device for DINOv2 model (cpu/cuda)
        dinov2_pooling: Feature pooling - "avg_patch" (recommended), "cls", or "avg_all"
        dinov2_metric: Similarity metric - "gaussian_rbf" (recommended) or "cosine"
        dinov2_rbf_sigma: Gaussian RBF bandwidth (default 3.0, higher = more peaked)
        enable_image_count_reward: Enable image count reward (match gen count to GT count)
        image_count_weight: Weight for image count reward (shared across pref/non-pref modes)
        enable_pref_grpo: Enable Pref-GRPO pairwise preference reward mode
        pref_grpo_winrate_weight: Weight for win rate reward in Pref-GRPO mode
        pref_grpo_hpsv3_weight: Weight for HPSv3 aesthetic reward in Pref-GRPO mode
    """
    import time as _time_module
    _reward_total_start = _time_module.time()
    _reward_timing = {}  # phase -> elapsed seconds

    logger.info(f"[Emu3 Reward] ========== compute_score called with {len(reward_inputs)} samples ==========")
    logger.info(f"[Emu3 Reward] Mode: {'Pref-GRPO' if enable_pref_grpo else 'Absolute VLM'}, HPSv3: {'enabled' if enable_hpsv3_reward else 'disabled'}, SampledFrames: {'enabled' if enable_sampled_frames_reward else 'disabled'}, DINOv2: {'enabled' if enable_dinov2_reward else 'disabled'}, ImageCount: {'enabled' if enable_image_count_reward else 'disabled'}")
    
    # Resolve final-frame API defaults
    ff_api_base = sampled_frames_api_base if sampled_frames_api_base else vlm_api_base
    ff_api_key = sampled_frames_api_key if sampled_frames_api_key else vlm_api_key
    ff_model_name = sampled_frames_model_name if sampled_frames_model_name else vlm_model_name
    
    # Normalize weights (include hpsv3_weight, sampled_frames_weight, image_count_weight if enabled)
    _ff_w = sampled_frames_weight if enable_sampled_frames_reward else 0.0
    _hp_w = hpsv3_weight if (enable_hpsv3_reward and hpsv3_weight > 0) else 0.0
    _dino_w = dinov2_weight if (enable_dinov2_reward and dinov2_weight > 0) else 0.0
    _ic_w = image_count_weight if enable_image_count_reward else 0.0
    total_weight = format_weight + vlm_weight + _hp_w + _ff_w + _dino_w + _ic_w
    if total_weight > 0:
        format_weight = format_weight / total_weight
        vlm_weight = vlm_weight / total_weight
        hpsv3_weight = _hp_w / total_weight
        sampled_frames_weight = _ff_w / total_weight
        dinov2_weight = _dino_w / total_weight
        image_count_weight = _ic_w / total_weight
    
    # Check if we have pre-decoded images from rollout worker
    # This is REQUIRED because reward worker runs in Ray actor without GPU
    num_samples_with_images = sum(
        1 for ri in reward_inputs 
        if ri.get("decoded_images") is not None and len(ri.get("decoded_images", [])) > 0
    )
    logger.info(f"[Emu3 Reward] Samples with decoded_images: {num_samples_with_images}/{len(reward_inputs)}")
    
    if enable_vlm_reward and num_samples_with_images == 0:
        logger.warning("[Emu3 Reward] WARNING: No pre-decoded images from rollout worker!")
        logger.info("[Emu3 Reward] Please set 'enable_image_decode_for_reward: true' in rollout config.")
        logger.info("[Emu3 Reward] VLM reward will use default score 0.5 for samples without images.")
    
    scores = []
    
    # Statistics for debugging
    format_pass_count = 0
    format_fail_count = 0
    no_image_count = 0
    
    _format_start = _time_module.time()
    # ============ First pass: compute format rewards for all samples ============
    # Also extract and normalize decoded images for later use
    sample_gen_images = []   # list of list[PIL.Image] per sample
    sample_ref_images = []   # list of list[PIL.Image] per sample
    sample_gt_images = []    # list of list[PIL.Image] per sample (for Pref-GRPO GT comparison)
    sample_uids = []         # uid per sample (for Pref-GRPO grouping)
    sample_dataset_sources = []  # dataset_source per sample (for task-specific prompts)
    sample_precomputed_dinov2 = []  # pre-computed DINOv2 scores from rollout worker (or None)
    
    
    for i, reward_input in enumerate(reward_inputs):
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        pre_decoded_images = reward_input.get("decoded_images", None)
        pre_decoded_reference_images = reward_input.get("decoded_reference_images", None)
        pre_decoded_gt_images = reward_input.get("decoded_gt_images", None)
        uid = reward_input.get("uid", f"unknown_{i}")
        dataset_source = reward_input.get("dataset_source", "")
        
        sample_uids.append(uid)
        sample_dataset_sources.append(dataset_source)
        
        # Compute format reward
        format_score, format_details = format_reward(response)
        
        # Parse images for counting
        images_data = parse_emu3_images(response)
        valid_images = [img for img in images_data if img.get('valid', False)]
        
        # Track statistics
        if format_score > 0:
            format_pass_count += 1
        else:
            format_fail_count += 1
            if len(images_data) == 0:
                no_image_count += 1
        
        # Initialize score entry
        score_entry = {
            "format": format_score,
            "vlm": 0.5,  # Default VLM score
            "hpsv3": 0.0,  # Default HPSv3 aesthetic score
            "sampled_frames": 0.5,  # Default final-frame comparison score
            "dinov2": 0.0,  # Default DINOv2 score
            "image_count": 0.0,  # Default image count reward
            "win_rate": 0.5,  # Default win rate for Pref-GRPO
            "num_images": len(valid_images),
            "format_details": format_details,
        }
        scores.append(score_entry)
        
        # Normalize decoded images to PIL.Image
        gen_images = _normalize_images(pre_decoded_images)
        ref_images = _normalize_images(pre_decoded_reference_images)
        gt_images = _normalize_images(pre_decoded_gt_images)
        
        sample_gen_images.append(gen_images)
        sample_ref_images.append(ref_images)
        sample_gt_images.append(gt_images)
        sample_precomputed_dinov2.append(reward_input.get("precomputed_dinov2_score", None))
    
    _reward_timing['format_check'] = _time_module.time() - _format_start
    logger.info(f"[Emu3 Reward] Format check results: {format_pass_count} passed, {format_fail_count} failed ({no_image_count} with no images) | Time: {_reward_timing['format_check']:.2f}s")
    
    # ============ Debug mode: override generated images for sanity checking ============
    if debug_mode:
        logger.warning(f"[Emu3 Reward] ⚠️  DEBUG MODE ACTIVE: '{debug_mode}' — generated images will be REPLACED")
        _debug_replaced = 0
        for i in range(len(reward_inputs)):
            gt_imgs = sample_gt_images[i]
            if not gt_imgs:
                continue
            if debug_mode == "gt_as_gen":
                # Replace gen images with GT images directly → expect near-perfect scores
                sample_gen_images[i] = list(gt_imgs)  # shallow copy
            elif debug_mode == "gt_first_frame":
                # Replace gen images with GT[0] repeated to match GT count → test frame diversity penalty
                sample_gen_images[i] = [gt_imgs[0]] * len(gt_imgs)
            else:
                logger.warning(f"[Emu3 Reward] Unknown debug_mode='{debug_mode}', ignoring")
                break
            # Also fix format score & num_images so debug samples pass format gate
            scores[i]["format"] = 1.0
            scores[i]["num_images"] = len(sample_gen_images[i])
            _debug_replaced += 1
        logger.warning(f"[Emu3 Reward] ⚠️  DEBUG: replaced gen_images for {_debug_replaced}/{len(reward_inputs)} samples (mode='{debug_mode}')")
    
    _vlm_start = _time_module.time()
    # ============ Second pass: VLM evaluation (mode-dependent) ============
    if enable_pref_grpo:
        # ==================== Pref-GRPO Mode ====================
        # Group samples by uid, then do pairwise comparisons within each group
        from collections import defaultdict as _defaultdict
        uid_to_indices = _defaultdict(list)
        for i, uid in enumerate(sample_uids):
            uid_to_indices[uid].append(i)
        
        num_groups = len(uid_to_indices)
        logger.info(f"[Pref-GRPO] Found {num_groups} groups from {len(reward_inputs)} samples")
        for uid, indices in uid_to_indices.items():
            logger.debug(f"[Pref-GRPO]   Group uid={uid[:8]}...: {len(indices)} samples")
        
        # ---- Collect ALL pairwise tasks across ALL groups, then run in one thread pool ----
        # This maximizes VLM concurrency instead of processing groups sequentially.
        all_win_rates = {}
        
        # Per-group bookkeeping: valid_indices list and per-index win/comparison counters
        group_valid_indices = {}   # uid -> list of valid original indices
        wins = {}                  # original_index -> float (win count)
        comparisons = {}           # original_index -> int (comparison count)
        all_pair_tasks = []        # (orig_i, orig_j, question, ref_images, gt_images)
        
        for uid, indices in uid_to_indices.items():
            valid_indices = [i for i in indices if scores[i]["format"] > 0 and len(sample_gen_images[i]) > 0]
            invalid_indices = [i for i in indices if i not in valid_indices]
            
            # Invalid samples get win_rate = 0.0
            for i in invalid_indices:
                all_win_rates[i] = 0.0
            
            if len(valid_indices) <= 1:
                for i in valid_indices:
                    all_win_rates[i] = 0.5
                if valid_indices:
                    logger.debug(f"[Pref-GRPO] Group uid={uid[:8]}...: Only {len(valid_indices)} valid sample(s), using neutral win rate")
                continue
            
            group_valid_indices[uid] = valid_indices
            
            # Shared reference/GT images for this group
            ref_images = sample_ref_images[valid_indices[0]] if sample_ref_images else []
            gt_images_group = sample_gt_images[valid_indices[0]] if sample_gt_images and len(sample_gt_images) > 0 else []
            question = reward_inputs[valid_indices[0]].get("ground_truth", "")
            if not isinstance(question, str):
                question = str(question)
            ds_source = sample_dataset_sources[valid_indices[0]] if sample_dataset_sources else ""
            
            # Initialize counters for valid indices
            for i in valid_indices:
                wins[i] = 0.0
                comparisons[i] = 0
            
            # Enumerate all pairs within this group (using original indices)
            for a_pos in range(len(valid_indices)):
                for b_pos in range(a_pos + 1, len(valid_indices)):
                    orig_i = valid_indices[a_pos]
                    orig_j = valid_indices[b_pos]
                    if sample_gen_images[orig_i] and sample_gen_images[orig_j]:
                        all_pair_tasks.append((orig_i, orig_j, question, ref_images, gt_images_group, ds_source))
        
        total_pairs = len(all_pair_tasks)
        total_groups_with_pairs = len(group_valid_indices)
        logger.info(f"[Pref-GRPO] Collected {total_pairs} pairwise tasks across {total_groups_with_pairs} groups, launching with max_workers={max_vlm_workers}")
        
        if all_pair_tasks:
            def _compare_pair_global(task):
                orig_i, orig_j, question, ref_imgs, gt_imgs, ds_src = task
                preference, explanation = call_vlm_pairwise(
                    question_text=question,
                    reference_images=ref_imgs,
                    images_a=sample_gen_images[orig_i],
                    images_b=sample_gen_images[orig_j],
                    gt_images=gt_imgs,
                    dataset_source=ds_src,
                    api_base=vlm_api_base,
                    api_key=vlm_api_key,
                    model_name=vlm_model_name,
                )
                return orig_i, orig_j, preference
            
            with ThreadPoolExecutor(max_workers=max_vlm_workers) as executor:
                futures = [executor.submit(_compare_pair_global, t) for t in all_pair_tasks]
                for future in as_completed(futures):
                    try:
                        orig_i, orig_j, preference = future.result()
                        comparisons[orig_i] += 1
                        comparisons[orig_j] += 1
                        if preference == "A":
                            wins[orig_i] += 1.0
                        elif preference == "B":
                            wins[orig_j] += 1.0
                        else:  # TIE
                            wins[orig_i] += 0.5
                            wins[orig_j] += 0.5
                    except Exception as e:
                        logger.warning(f"[Pref-GRPO] Pair comparison failed: {e}")
        
        # Compute win rates from aggregated wins/comparisons
        for uid, valid_indices in group_valid_indices.items():
            for i in valid_indices:
                if comparisons.get(i, 0) > 0:
                    all_win_rates[i] = wins[i] / comparisons[i]
                else:
                    all_win_rates[i] = 0.5
            
            group_wr = {i: all_win_rates[i] for i in valid_indices}
            logger.debug(f"[Pref-GRPO] Group uid={uid[:8]}... win rates: {group_wr}")
        
        # Apply win rates to scores
        for i in range(len(scores)):
            scores[i]["win_rate"] = all_win_rates.get(i, 0.5)
        
        # ---- HPSv3 aesthetic scoring (Pref-GRPO mode) ----
        if enable_hpsv3_reward and pref_grpo_hpsv3_weight > 0:
            hpsv3_indices = [i for i in range(len(scores)) if scores[i]["format"] > 0 and len(sample_gen_images[i]) > 0]
            if hpsv3_indices:
                hpsv3_prompts = []
                hpsv3_images = []
                for i in hpsv3_indices:
                    q = reward_inputs[i].get("ground_truth", "")
                    hpsv3_prompts.append(str(q) if not isinstance(q, str) else q)
                    hpsv3_images.append(sample_gen_images[i])
                
                hpsv3_results = call_hpsv3_reward_batch(
                    hpsv3_prompts, hpsv3_images,
                    api_base=hpsv3_api_base, timeout=hpsv3_timeout,
                )
                for idx, (h_score, h_detail) in zip(hpsv3_indices, hpsv3_results):
                    scores[idx]["hpsv3"] = h_score
                logger.info(f"[Pref-GRPO] HPSv3 scored {len(hpsv3_indices)} samples")
        
        # ---- Sampled-frame VLM comparison (Pref-GRPO mode) ----
        # Strategy: within each uid group, all samples that match the expected
        # image count share the SAME randomly-chosen frame indices so that
        # their per-frame VLM scores are directly comparable.
        # Samples that do NOT match the expected image count get score = 0.0
        # (they are already penalised heavily via image_count reward, and
        # comparing frames at mismatched indices would be meaningless).
        if enable_sampled_frames_reward and sampled_frames_weight > 0:
            ff_tasks = []  # (index, question, ref_images, gen_images, gt_images, forced_indices)
            sf_penalised = 0  # count of samples penalised for wrong image count

            for uid, indices in uid_to_indices.items():
                # Determine expected frame count from GT of this group
                gt_imgs_group = sample_gt_images[indices[0]] if indices and len(sample_gt_images) > 0 else []
                gt_count = len(gt_imgs_group)

                if gt_count == 0:
                    continue  # no GT → skip entire group

                # Pre-compute shared sampled indices for this group
                num_to_sample = max(1, int(gt_count * sampled_frames_ratio))
                shared_indices = sorted(random.sample(range(gt_count), min(num_to_sample, gt_count)))

                for i in indices:
                    # Only score samples with correct format AND matching image count
                    gen_count = len(sample_gen_images[i])
                    if scores[i]["format"] <= 0 or gen_count == 0:
                        continue  # already scored 0 via format
                    if gen_count != gt_count:
                        # Image-count mismatch → sampled_frames = 0.0, record indices for vis
                        scores[i]["sampled_frames"] = 0.0
                        scores[i]["sampled_frames_indices"] = shared_indices
                        scores[i]["sampled_frames_per_frame_scores"] = [0.0] * len(shared_indices)
                        sf_penalised += 1
                        continue
                    
                    q = reward_inputs[i].get("ground_truth", "")
                    if not isinstance(q, str):
                        q = str(q)
                    ff_tasks.append((i, q, sample_ref_images[i], sample_gen_images[i],
                                     sample_gt_images[i], shared_indices, sample_dataset_sources[i]))

            if ff_tasks:
                logger.info(f"[Pref-GRPO] Running sampled-frame VLM comparison for {len(ff_tasks)} samples "
                            f"(ratio={sampled_frames_ratio}, penalised={sf_penalised} for img-count mismatch)...")
                
                def _ff_eval(task):
                    idx, q, ref, gen, gt, forced_idx, ds_src = task
                    sc, detail, sampled_idx, per_frame = call_vlm_sampled_frames_reward(
                        question_text=q,
                        reference_images=ref,
                        generated_images=gen,
                        gt_images=gt,
                        sample_ratio=sampled_frames_ratio,
                        forced_indices=forced_idx,
                        dataset_source=ds_src,
                        api_base=ff_api_base,
                        api_key=ff_api_key,
                        model_name=ff_model_name,
                        max_vlm_workers=max(1, max_vlm_workers // max(1, len(ff_tasks))),
                    )
                    return idx, sc, detail, sampled_idx, per_frame
                
                with ThreadPoolExecutor(max_workers=max_vlm_workers) as executor:
                    futures = [executor.submit(_ff_eval, t) for t in ff_tasks]
                    for future in as_completed(futures):
                        try:
                            idx, ff_score, ff_detail, sampled_idx, per_frame = future.result()
                            scores[idx]["sampled_frames"] = ff_score
                            scores[idx]["sampled_frames_indices"] = sampled_idx
                            scores[idx]["sampled_frames_per_frame_scores"] = per_frame
                        except Exception as e:
                            logger.warning(f"[Pref-GRPO] Sampled-frame task failed: {e}")
                
                logger.info(f"[Pref-GRPO] Sampled-frame scored {len(ff_tasks)} samples (penalised {sf_penalised})")
        
        # ---- Image Count Reward (Pref-GRPO mode) ----
        if enable_image_count_reward and image_count_weight > 0:
            ic_count = 0
            for i in range(len(scores)):
                if scores[i]["format"] > 0:
                    ic_score, ic_details = image_count_reward(sample_gen_images[i], sample_gt_images[i])
                    scores[i]["image_count"] = ic_score
                    ic_count += 1
            logger.info(f"[Pref-GRPO] Image count reward computed for {ic_count} samples")
        
        # ---- DINOv2 Feature Similarity (Pref-GRPO mode) ----
        if enable_dinov2_reward and dinov2_weight > 0:
            # Check if pre-computed scores from rollout worker are available
            _has_precomputed = any(s is not None for s in sample_precomputed_dinov2)
            
            if _has_precomputed:
                # Use pre-computed DINOv2 scores from rollout worker (already computed on GPU)
                _used = 0
                for i in range(len(scores)):
                    if sample_precomputed_dinov2[i] is not None:
                        scores[i]["dinov2"] = float(sample_precomputed_dinov2[i])
                        _used += 1
                logger.info(f"[Pref-GRPO] Using {_used} pre-computed DINOv2 scores from rollout worker (GPU)")
            else:
                # Fallback: compute DINOv2 locally (requires GPU in reward worker)
                dino_gen_imgs = []
                dino_gt_imgs = []
                dino_indices = []
                
                for i in range(len(scores)):
                    if scores[i]["format"] > 0 and len(sample_gen_images[i]) > 0 and len(sample_gt_images[i]) > 0:
                        dino_indices.append(i)
                        dino_gen_imgs.append(sample_gen_images[i][-1])  # Last frame
                        dino_gt_imgs.append(sample_gt_images[i][-1])    # Last frame
                
                if dino_indices:
                    logger.info(f"[Pref-GRPO] Computing DINOv2 scores locally for {len(dino_indices)} samples (pooling={dinov2_pooling}, metric={dinov2_metric}, sigma={dinov2_rbf_sigma})...")
                    logger.warning(f"[Pref-GRPO] WARNING: Computing DINOv2 in reward worker. For better performance, set 'enable_dinov2_in_rollout: true' in rollout config.")
                    dino_scores = compute_dinov2_similarity_batch(
                        dino_gen_imgs, dino_gt_imgs,
                        model_name=dinov2_model_name, device=dinov2_device,
                        pooling=dinov2_pooling, metric=dinov2_metric, rbf_sigma=dinov2_rbf_sigma,
                    )
                    for idx, score in zip(dino_indices, dino_scores):
                        scores[idx]["dinov2"] = score
                    logger.info(f"[Pref-GRPO] DINOv2 scored {len(dino_indices)} samples")

        _reward_timing['pref_grpo_vlm'] = _time_module.time() - _vlm_start
        logger.info(f"[Emu3 Reward] Pref-GRPO VLM evaluation time: {_reward_timing['pref_grpo_vlm']:.2f}s")

        # Compute overall scores using unified weights
        _pref_hp_w = pref_grpo_hpsv3_weight if (enable_hpsv3_reward and pref_grpo_hpsv3_weight > 0) else 0.0
        _pref_ic_w = image_count_weight if enable_image_count_reward else 0.0
        _pref_ff_w = sampled_frames_weight if enable_sampled_frames_reward else 0.0
        _pref_dino_w = dinov2_weight if (enable_dinov2_reward and dinov2_weight > 0) else 0.0
        total_w = format_weight + pref_grpo_winrate_weight + _pref_hp_w + _pref_ff_w + _pref_dino_w + _pref_ic_w
        if total_w > 0:
            fmt_w = format_weight / total_w
            wr_w = pref_grpo_winrate_weight / total_w
            hp_w = _pref_hp_w / total_w
            ff_w = _pref_ff_w / total_w
            dino_w = _pref_dino_w / total_w
            ic_w = _pref_ic_w / total_w
        else:
            fmt_w, wr_w, hp_w, ff_w, dino_w, ic_w = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for score_entry in scores:
            format_score = score_entry["format"]
            win_rate = score_entry["win_rate"]
            hpsv3_score = score_entry["hpsv3"]
            ff_score = score_entry["sampled_frames"]
            dino_score = score_entry["dinov2"]
            ic_score = score_entry["image_count"]
            gl_combined = 0.0

            if format_score == 0:
                overall = 0.0
            elif ic_score == 0:
                # Image count mismatch → zero out all other scores
                score_entry["win_rate"] = 0.0
                score_entry["hpsv3"] = 0.0
                score_entry["sampled_frames"] = 0.0
                score_entry["dinov2"] = 0.0
                win_rate = 0.0
                hpsv3_score = 0.0
                ff_score = 0.0
                dino_score = 0.0
                overall = 0.0
                logger.debug(f"[Pref-GRPO] Image count mismatch → all scores zeroed")
            else:
                gl_combined = win_rate - 2.0 * abs(win_rate - ff_score)
                overall = fmt_w * format_score + (wr_w + ff_w) * gl_combined + hp_w * hpsv3_score + dino_w * dino_score + ic_w * ic_score

            score_entry["overall"] = overall
            score_entry.pop("format_details", None)
            logger.debug(f"[Pref-GRPO] Sample score - Format: {format_score}, WinRate: {win_rate:.3f}, HPSv3: {hpsv3_score:.4f}, SampledFrames: {ff_score:.3f}, DINOv2: {dino_score:.3f}, ImageCount: {ic_score:.1f}, GL_combined(Rg-2|Rg-Rl|): {gl_combined:.3f}, Overall: {overall:.3f}")
        
        # Save visualization for ALL groups with ALL reward scores
        # NOTE: must be called BEFORE popping list-type keys below
        try:
            _save_pref_grpo_visualization(
                uid_to_indices=uid_to_indices,
                scores=scores,
                sample_gen_images=sample_gen_images,
                sample_ref_images=sample_ref_images,
                sample_gt_images=sample_gt_images,
                reward_inputs=reward_inputs,
            )
        except Exception as e:
            logger.warning(f"[Pref-GRPO Vis] Failed to save visualization: {e}")

        # Remove non-scalar keys that would break np.mean in reduce_metrics
        for score_entry in scores:
            score_entry.pop("sampled_frames_indices", None)
            score_entry.pop("sampled_frames_per_frame_scores", None)

        _reward_total_elapsed = _time_module.time() - _reward_total_start
        _reward_timing['total'] = _reward_total_elapsed
        logger.info(f"\n{'='*80}")
        logger.info(f"[Emu3 Reward Timing Summary] Total: {_reward_total_elapsed:.2f}s")
        _timing_parts = [f"{k}={v:.2f}s" for k, v in _reward_timing.items() if k != 'total']
        logger.info(f"[Emu3 Reward Timing Summary] Breakdown: {' | '.join(_timing_parts)}")
        logger.info(f"{'='*80}")
        logger.info(f"[Emu3 Reward] ========== compute_score finished, returning {len(scores)} scores ==========")
        return scores

    else:
        # ==================== Absolute VLM Reward Mode ====================
        vlm_tasks = []  # (index, question, ref_images, gen_images)
        
        for i, reward_input in enumerate(reward_inputs):
            if enable_vlm_reward and scores[i]["format"] > 0 and len(sample_gen_images[i]) > 0:
                question = reward_input.get("ground_truth", "")
                if not isinstance(question, str):
                    question = str(question)
                vlm_tasks.append((i, question, sample_ref_images[i], sample_gen_images[i], sample_gt_images[i], sample_dataset_sources[i]))
            elif scores[i]["format"] > 0 and len(sample_gen_images[i]) == 0:
                logger.debug(f"[Emu3 Reward] Sample {i}: Format passed but no decoded images for VLM reward")
        
        logger.info(f"[Emu3 Reward] VLM tasks prepared: {len(vlm_tasks)}")
        
        _vlm_eval_start = _time_module.time()
        if vlm_tasks:
            logger.info(f"[Emu3 Reward] Running VLM evaluation for {len(vlm_tasks)} samples...")
            
            def evaluate_single(task):
                idx, question, ref_imgs, gen_imgs, gt_imgs, ds_src = task
                avg_score, explanation, task_score, visual_score = call_vlm_reward(
                    question_text=question,
                    reference_images=ref_imgs,
                    generated_images=gen_imgs,
                    gt_images=gt_imgs,
                    dataset_source=ds_src,
                    api_base=vlm_api_base,
                    api_key=vlm_api_key,
                    model_name=vlm_model_name,
                )
                return idx, avg_score, explanation, task_score, visual_score
            
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
                        logger.warning(f"[Emu3 Reward] VLM task failed: {e}")
        
        # ---- HPSv3 aesthetic scoring (Absolute mode) ----
        if enable_hpsv3_reward and hpsv3_weight > 0:
            hpsv3_indices = [i for i in range(len(scores)) if scores[i]["format"] > 0 and len(sample_gen_images[i]) > 0]
            if hpsv3_indices:
                hpsv3_prompts = []
                hpsv3_images = []
                for i in hpsv3_indices:
                    q = reward_inputs[i].get("ground_truth", "")
                    hpsv3_prompts.append(str(q) if not isinstance(q, str) else q)
                    hpsv3_images.append(sample_gen_images[i])
                
                hpsv3_results = call_hpsv3_reward_batch(
                    hpsv3_prompts, hpsv3_images,
                    api_base=hpsv3_api_base, timeout=hpsv3_timeout,
                )
                for idx, (h_score, h_detail) in zip(hpsv3_indices, hpsv3_results):
                    scores[idx]["hpsv3"] = h_score
                logger.info(f"[Emu3 Reward] HPSv3 scored {len(hpsv3_indices)} samples")
        
        # ---- Sampled-frame VLM comparison (Absolute mode) ----
        if enable_sampled_frames_reward and sampled_frames_weight > 0:
            ff_tasks = []  # (index, question, ref_images, gen_images, gt_images)
            for i in range(len(scores)):
                if scores[i]["format"] > 0 and len(sample_gen_images[i]) > 0 and len(sample_gt_images[i]) > 0:
                    q = reward_inputs[i].get("ground_truth", "")
                    if not isinstance(q, str):
                        q = str(q)
                    ff_tasks.append((i, q, sample_ref_images[i], sample_gen_images[i], sample_gt_images[i], sample_dataset_sources[i]))
            
            if ff_tasks:
                logger.info(f"[Emu3 Reward] Running sampled-frame VLM comparison for {len(ff_tasks)} samples (ratio={sampled_frames_ratio})...")
                
                def _ff_eval_abs(task):
                    idx, q, ref, gen, gt, ds_src = task
                    sc, detail, sampled_idx, per_frame = call_vlm_sampled_frames_reward(
                        question_text=q,
                        reference_images=ref,
                        generated_images=gen,
                        gt_images=gt,
                        sample_ratio=sampled_frames_ratio,
                        dataset_source=ds_src,
                        api_base=ff_api_base,
                        api_key=ff_api_key,
                        model_name=ff_model_name,
                        max_vlm_workers=max(1, max_vlm_workers // max(1, len(ff_tasks))),
                    )
                    return idx, sc, detail, sampled_idx, per_frame
                
                with ThreadPoolExecutor(max_workers=max_vlm_workers) as executor:
                    futures = [executor.submit(_ff_eval_abs, t) for t in ff_tasks]
                    for future in as_completed(futures):
                        try:
                            idx, ff_score, ff_detail, sampled_idx, per_frame = future.result()
                            scores[idx]["sampled_frames"] = ff_score
                            scores[idx]["sampled_frames_indices"] = sampled_idx
                            scores[idx]["sampled_frames_per_frame_scores"] = per_frame
                        except Exception as e:
                            logger.warning(f"[Emu3 Reward] Sampled-frame task failed: {e}")
                
                logger.info(f"[Emu3 Reward] Sampled-frame scored {len(ff_tasks)} samples")
        
        # ---- Image Count Reward (Absolute mode) ----
        if enable_image_count_reward and image_count_weight > 0:
            ic_count = 0
            for i in range(len(scores)):
                if scores[i]["format"] > 0:
                    ic_score, ic_details = image_count_reward(sample_gen_images[i], sample_gt_images[i])
                    scores[i]["image_count"] = ic_score
                    ic_count += 1
            logger.info(f"[Emu3 Reward] Image count reward computed for {ic_count} samples")
        
        # ---- DINOv2 Feature Similarity (Absolute mode) ----
        if enable_dinov2_reward and dinov2_weight > 0:
            # Check if pre-computed scores from rollout worker are available
            _has_precomputed = any(s is not None for s in sample_precomputed_dinov2)
            
            if _has_precomputed:
                # Use pre-computed DINOv2 scores from rollout worker (already computed on GPU)
                _used = 0
                for i in range(len(scores)):
                    if sample_precomputed_dinov2[i] is not None:
                        scores[i]["dinov2"] = float(sample_precomputed_dinov2[i])
                        _used += 1
                logger.info(f"[Emu3 Reward] Using {_used} pre-computed DINOv2 scores from rollout worker (GPU)")
            else:
                # Fallback: compute DINOv2 locally (requires GPU in reward worker)
                dino_gen_imgs = []
                dino_gt_imgs = []
                dino_indices = []
                
                for i in range(len(scores)):
                    if scores[i]["format"] > 0 and len(sample_gen_images[i]) > 0 and len(sample_gt_images[i]) > 0:
                        dino_indices.append(i)
                        dino_gen_imgs.append(sample_gen_images[i][-1])  # Last frame
                        dino_gt_imgs.append(sample_gt_images[i][-1])    # Last frame
                
                if dino_indices:
                    logger.info(f"[Emu3 Reward] Computing DINOv2 scores locally for {len(dino_indices)} samples (pooling={dinov2_pooling}, metric={dinov2_metric}, sigma={dinov2_rbf_sigma})...")
                    logger.warning(f"[Emu3 Reward] WARNING: Computing DINOv2 in reward worker. For better performance, set 'enable_dinov2_in_rollout: true' in rollout config.")
                    dino_scores = compute_dinov2_similarity_batch(
                        dino_gen_imgs, dino_gt_imgs,
                        model_name=dinov2_model_name, device=dinov2_device,
                        pooling=dinov2_pooling, metric=dinov2_metric, rbf_sigma=dinov2_rbf_sigma,
                    )
                    for idx, score in zip(dino_indices, dino_scores):
                        scores[idx]["dinov2"] = score
                    logger.info(f"[Emu3 Reward] DINOv2 scored {len(dino_indices)} samples")

        _reward_timing['absolute_vlm'] = _time_module.time() - _vlm_eval_start
        logger.info(f"[Emu3 Reward] Absolute VLM evaluation time: {_reward_timing['absolute_vlm']:.2f}s")

        # Compute overall scores
        for score_entry in scores:
            format_score = score_entry["format"]
            vlm_score = score_entry["vlm"]
            hpsv3_score = score_entry["hpsv3"]
            ff_score = score_entry["sampled_frames"]
            dino_score = score_entry["dinov2"]
            ic_score = score_entry["image_count"]
            
            if format_score == 0:
                overall = 0.0
            elif ic_score == 0:
                # Image count mismatch → zero out all other scores
                score_entry["vlm"] = 0.0
                score_entry["hpsv3"] = 0.0
                score_entry["sampled_frames"] = 0.0
                score_entry["dinov2"] = 0.0
                vlm_score = 0.0
                hpsv3_score = 0.0
                ff_score = 0.0
                dino_score = 0.0
                overall = 0.0
                logger.debug(f"[Emu3 Reward] Image count mismatch → all scores zeroed")
            else:
                gl_combined = vlm_score - 2.0 * abs(vlm_score - ff_score)
                overall = format_weight * format_score + (vlm_weight + sampled_frames_weight) * gl_combined + hpsv3_weight * hpsv3_score + dinov2_weight * dino_score + image_count_weight * ic_score

            score_entry["overall"] = overall
            score_entry.pop("format_details", None)
            score_entry.pop("vlm_explanation", None)
            # Remove non-scalar keys that would break np.mean in reduce_metrics
            score_entry.pop("sampled_frames_indices", None)
            score_entry.pop("sampled_frames_per_frame_scores", None)
            logger.debug(f"[Emu3 Reward] Sample score - Format: {format_score}, VLM: {vlm_score}, HPSv3: {hpsv3_score:.4f}, SampledFrames: {ff_score:.3f}, DINOv2: {dino_score:.3f}, ImageCount: {ic_score:.1f}, GL_combined(Rg-2|Rg-Rl|): {gl_combined:.3f}, Overall: {overall}")
    
    _reward_total_elapsed = _time_module.time() - _reward_total_start
    _reward_timing['total'] = _reward_total_elapsed
    logger.info(f"\n{'='*80}")
    logger.info(f"[Emu3 Reward Timing Summary] Total: {_reward_total_elapsed:.2f}s")
    _timing_parts = [f"{k}={v:.2f}s" for k, v in _reward_timing.items() if k != 'total']
    logger.info(f"[Emu3 Reward Timing Summary] Breakdown: {' | '.join(_timing_parts)}")
    logger.info(f"{'='*80}")
    logger.info(f"[Emu3 Reward] ========== compute_score finished, returning {len(scores)} scores ==========")
    return scores


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
            logger.warning(f"[Emu3 Reward] Warning: Unknown image type {type(img)}, skipping")
    return result


if __name__ == "__main__":
    # Example usage for testing
    test_response = "<|extra_60|>Let's begin.<|extra_61|><|image start|>32*32<|image token|><|visual token 027913|><|visual token 088473|><|visual token 054821|><|visual token 048384|><|visual token 007808|><|visual token 044303|><|visual token 099858|><|visual token 011332|><|visual token 055887|><|visual token 032108|><|visual token 083909|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 125661|><|visual token 011332|><|visual token 125661|><|visual token 051964|><|visual token 022673|><|visual token 113201|><|visual token 087578|><|visual token 090820|><|extra_200|><|visual token 016747|><|visual token 044285|><|visual token 124640|><|visual token 000176|><|visual token 016365|><|visual token 001345|><|visual token 012484|><|visual token 089162|><|visual token 026492|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 020504|><|visual token 026492|><|visual token 020504|><|visual token 026492|><|visual token 092827|><|visual token 037873|><|visual token 077042|><|visual token 076157|><|visual token 013651|><|extra_200|><|visual token 021708|><|visual token 021115|><|visual token 112970|><|visual token 101487|><|visual token 022384|><|visual token 083324|><|visual token 010331|><|visual token 075986|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 124308|><|visual token 098758|><|visual token 109485|><|visual token 000099|><|visual token 112476|><|visual token 041476|><|visual token 048200|><|visual token 078908|><|visual token 094092|><|extra_200|><|visual token 059965|><|visual token 108993|><|visual token 085877|><|visual token 057360|><|visual token 122584|><|visual token 036690|><|visual token 124671|><|visual token 018767|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 095708|><|visual token 054014|><|visual token 036832|><|visual token 098111|><|visual token 099932|><|visual token 007694|><|visual token 072417|><|visual token 033009|><|extra_200|><|visual token 118115|><|visual token 121206|><|visual token 032772|><|visual token 017727|><|visual token 006419|><|visual token 107459|><|visual token 066066|><|visual token 122694|><|visual token 069821|><|visual token 013912|><|visual token 123185|><|visual token 123185|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 123185|><|visual token 123185|><|visual token 013912|><|visual token 043970|><|visual token 123185|><|visual token 123185|><|visual token 108832|><|visual token 063312|><|visual token 091378|><|visual token 013912|><|visual token 113298|><|visual token 106786|><|visual token 012336|><|visual token 046565|><|extra_200|><|visual token 053732|><|visual token 083161|><|visual token 091876|><|visual token 021753|><|visual token 055557|><|visual token 019848|><|visual token 074827|><|visual token 077277|><|visual token 073290|><|visual token 081590|><|visual token 030728|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 030728|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 030728|><|visual token 067729|><|visual token 106808|><|visual token 095273|><|visual token 064060|><|visual token 020838|><|visual token 048757|><|visual token 022376|><|visual token 074582|><|visual token 067322|><|visual token 075357|><|visual token 086832|><|extra_200|><|visual token 110776|><|visual token 108618|><|visual token 028407|><|visual token 020368|><|visual token 020531|><|visual token 054884|><|visual token 091213|><|visual token 057554|><|visual token 055352|><|visual token 017540|><|visual token 070057|><|visual token 129391|><|visual token 020875|><|visual token 129391|><|visual token 020875|><|visual token 020875|><|visual token 129391|><|visual token 020875|><|visual token 020875|><|visual token 105169|><|visual token 020542|><|visual token 082965|><|visual token 105169|><|visual token 039689|><|visual token 013470|><|visual token 107943|><|visual token 017391|><|visual token 054883|><|visual token 126639|><|visual token 089700|><|visual token 117859|><|visual token 068408|><|extra_200|><|visual token 004011|><|visual token 045244|><|visual token 079743|><|visual token 040510|><|visual token 033186|><|visual token 008736|><|visual token 050228|><|visual token 054662|><|visual token 052056|><|visual token 009381|><|visual token 031613|><|visual token 052422|><|visual token 096127|><|visual token 128470|><|visual token 117054|><|visual token 003287|><|visual token 015156|><|visual token 114780|><|visual token 025601|><|visual token 025164|><|visual token 067983|><|visual token 012083|><|visual token 023342|><|visual token 092787|><|visual token 057418|><|visual token 001304|><|visual token 082838|><|visual token 046564|><|visual token 066914|><|visual token 041665|><|visual token 102036|><|visual token 076681|><|extra_200|><|visual token 003381|><|visual token 108618|><|visual token 096640|><|visual token 123469|><|visual token 025862|><|visual token 047431|><|visual token 072375|><|visual token 025552|><|visual token 044923|><|visual token 059755|><|visual token 017540|><|visual token 103930|><|visual token 056221|><|visual token 013050|><|visual token 003704|><|visual token 058953|><|visual token 084739|><|visual token 067779|><|visual token 045921|><|visual token 120855|><|visual token 005725|><|visual token 113500|><|visual token 065469|><|visual token 061320|><|visual token 062359|><|visual token 019599|><|visual token 076308|><|visual token 091672|><|visual token 071394|><|visual token 035376|><|visual token 011678|><|visual token 067275|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 073996|><|visual token 070387|><|visual token 028361|><|visual token 036237|><|visual token 013568|><|visual token 059201|><|visual token 058785|><|visual token 054979|><|visual token 067493|><|visual token 061004|><|visual token 004822|><|visual token 056221|><|visual token 081188|><|visual token 079181|><|visual token 090784|><|visual token 020812|><|visual token 066066|><|visual token 067941|><|visual token 096041|><|visual token 035518|><|visual token 103338|><|visual token 096456|><|visual token 102291|><|visual token 070333|><|visual token 108061|><|visual token 032994|><|visual token 122829|><|visual token 121019|><|visual token 085502|><|visual token 020489|><|extra_200|><|visual token 091453|><|visual token 002085|><|visual token 117333|><|visual token 011670|><|visual token 080453|><|visual token 113382|><|visual token 067282|><|visual token 100546|><|visual token 025056|><|visual token 008816|><|visual token 053293|><|visual token 056456|><|visual token 036163|><|visual token 110516|><|visual token 094858|><|visual token 050079|><|visual token 016410|><|visual token 085319|><|visual token 071250|><|visual token 009912|><|visual token 105057|><|visual token 037909|><|visual token 113555|><|visual token 009961|><|visual token 040751|><|visual token 018011|><|visual token 102633|><|visual token 030282|><|visual token 105176|><|visual token 046477|><|visual token 118283|><|visual token 123512|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 088941|><|visual token 105510|><|visual token 008027|><|visual token 061763|><|visual token 052552|><|visual token 047691|><|visual token 021827|><|visual token 046363|><|visual token 004287|><|visual token 095120|><|visual token 098883|><|visual token 015482|><|visual token 064189|><|visual token 088688|><|visual token 110384|><|visual token 034516|><|visual token 067177|><|visual token 103225|><|visual token 082993|><|visual token 101194|><|visual token 096699|><|visual token 002926|><|visual token 097231|><|visual token 018212|><|visual token 039414|><|visual token 005658|><|visual token 052771|><|visual token 007660|><|visual token 124941|><|visual token 065394|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 001126|><|visual token 117935|><|visual token 004231|><|visual token 035040|><|visual token 116818|><|visual token 062434|><|visual token 077036|><|visual token 127252|><|visual token 017327|><|visual token 122413|><|visual token 034466|><|visual token 085285|><|visual token 091973|><|visual token 037766|><|visual token 052162|><|visual token 058666|><|visual token 057959|><|visual token 085656|><|visual token 106047|><|visual token 129160|><|visual token 083862|><|visual token 029532|><|visual token 127044|><|visual token 008146|><|visual token 082226|><|visual token 128936|><|visual token 089743|><|visual token 056817|><|visual token 024968|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 033720|><|visual token 084150|><|visual token 125916|><|visual token 102698|><|visual token 027992|><|visual token 072573|><|visual token 082645|><|visual token 083187|><|visual token 001822|><|visual token 082806|><|visual token 085250|><|visual token 091790|><|visual token 121372|><|visual token 130527|><|visual token 087905|><|visual token 092838|><|visual token 084627|><|visual token 102034|><|visual token 129067|><|visual token 001468|><|visual token 120205|><|visual token 052328|><|visual token 056230|><|visual token 050050|><|visual token 016693|><|visual token 024779|><|visual token 034424|><|visual token 052983|><|visual token 093146|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 107389|><|visual token 125860|><|visual token 080151|><|visual token 110023|><|visual token 027937|><|visual token 060787|><|visual token 053261|><|visual token 076640|><|visual token 003445|><|visual token 098169|><|visual token 027155|><|visual token 003001|><|visual token 116150|><|visual token 072197|><|visual token 037347|><|visual token 095994|><|visual token 015621|><|visual token 073553|><|visual token 102956|><|visual token 128145|><|visual token 127523|><|visual token 107170|><|visual token 038028|><|visual token 065300|><|visual token 007025|><|visual token 050683|><|visual token 082867|><|visual token 054041|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 113177|><|visual token 089507|><|visual token 105517|><|visual token 088455|><|visual token 095804|><|visual token 038837|><|visual token 121546|><|visual token 028846|><|visual token 055144|><|visual token 053742|><|visual token 048650|><|visual token 114138|><|visual token 116563|><|visual token 128916|><|visual token 064538|><|visual token 041574|><|visual token 052483|><|visual token 113514|><|visual token 128612|><|visual token 017924|><|visual token 102042|><|visual token 124141|><|visual token 086281|><|visual token 082085|><|visual token 116515|><|visual token 005378|><|visual token 036760|><|visual token 082867|><|visual token 054041|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 096440|><|visual token 063812|><|visual token 093134|><|visual token 070748|><|visual token 038647|><|visual token 038034|><|visual token 018543|><|visual token 026402|><|visual token 083823|><|visual token 113758|><|visual token 087635|><|visual token 088595|><|visual token 126199|><|visual token 127892|><|visual token 024666|><|visual token 033310|><|visual token 112009|><|visual token 013235|><|visual token 085114|><|visual token 028968|><|visual token 014579|><|visual token 066843|><|visual token 042474|><|visual token 072008|><|visual token 053994|><|visual token 007025|><|visual token 000025|><|visual token 082867|><|visual token 054041|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 104787|><|visual token 072580|><|visual token 122163|><|visual token 037874|><|visual token 102133|><|visual token 111340|><|visual token 055057|><|visual token 051586|><|visual token 120079|><|visual token 043866|><|visual token 034657|><|visual token 097319|><|visual token 027156|><|visual token 079957|><|visual token 120911|><|visual token 122887|><|visual token 097749|><|visual token 012239|><|visual token 024878|><|visual token 073569|><|visual token 094098|><|visual token 074464|><|visual token 127001|><|visual token 016286|><|visual token 053994|><|visual token 007025|><|visual token 000025|><|visual token 082867|><|visual token 054041|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 007054|><|visual token 089507|><|visual token 105850|><|visual token 098461|><|visual token 038034|><|visual token 098822|><|visual token 029822|><|visual token 023009|><|visual token 074992|><|visual token 009098|><|visual token 103930|><|visual token 010758|><|visual token 128182|><|visual token 046071|><|visual token 097655|><|visual token 107709|><|visual token 119543|><|visual token 015986|><|visual token 060485|><|visual token 096160|><|visual token 011386|><|visual token 118228|><|visual token 125673|><|visual token 112711|><|visual token 053994|><|visual token 067831|><|visual token 000025|><|visual token 118008|><|visual token 054041|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 033720|><|visual token 047009|><|visual token 058339|><|visual token 037543|><|visual token 080218|><|visual token 126612|><|visual token 051586|><|visual token 065759|><|visual token 123816|><|visual token 021109|><|visual token 088002|><|visual token 062079|><|visual token 058953|><|visual token 023761|><|visual token 092000|><|visual token 012734|><|visual token 107035|><|visual token 050391|><|visual token 098818|><|visual token 011411|><|visual token 067815|><|visual token 074840|><|visual token 083809|><|visual token 122337|><|visual token 053994|><|visual token 007025|><|visual token 000025|><|visual token 118008|><|visual token 054041|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 064045|><|visual token 029328|><|visual token 087793|><|visual token 012592|><|visual token 117114|><|visual token 076127|><|visual token 113521|><|visual token 114455|><|visual token 018787|><|visual token 069107|><|visual token 089995|><|visual token 049918|><|visual token 035244|><|visual token 103930|><|visual token 087365|><|visual token 129935|><|visual token 085327|><|visual token 081346|><|visual token 081801|><|visual token 000467|><|visual token 081030|><|visual token 120057|><|visual token 129935|><|visual token 013650|><|visual token 053994|><|visual token 007025|><|visual token 000025|><|visual token 118008|><|visual token 054041|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 073996|><|visual token 123094|><|visual token 049412|><|visual token 040262|><|visual token 006009|><|visual token 108964|><|visual token 052521|><|visual token 083546|><|visual token 095166|><|visual token 059827|><|visual token 045135|><|visual token 092007|><|visual token 102413|><|visual token 035147|><|visual token 006900|><|visual token 035375|><|visual token 068564|><|visual token 008436|><|visual token 123650|><|visual token 108675|><|visual token 018568|><|visual token 032022|><|visual token 030932|><|visual token 050832|><|visual token 006797|><|visual token 053994|><|visual token 007025|><|visual token 000025|><|visual token 118008|><|visual token 054041|><|extra_200|><|visual token 093483|><|visual token 002270|><|visual token 117333|><|visual token 124246|><|visual token 004898|><|visual token 089192|><|visual token 071753|><|visual token 003210|><|visual token 103327|><|visual token 072176|><|visual token 105692|><|visual token 090775|><|visual token 025441|><|visual token 048549|><|visual token 111986|><|visual token 027162|><|visual token 028475|><|visual token 054903|><|visual token 031029|><|visual token 040187|><|visual token 075745|><|visual token 093874|><|visual token 092317|><|visual token 042873|><|visual token 023168|><|visual token 089175|><|visual token 052404|><|visual token 053994|><|visual token 007025|><|visual token 000025|><|visual token 119436|><|visual token 054041|><|extra_200|><|visual token 128739|><|visual token 073293|><|visual token 088941|><|visual token 102263|><|visual token 128631|><|visual token 092575|><|visual token 071963|><|visual token 059343|><|visual token 075549|><|visual token 102128|><|visual token 008064|><|visual token 056750|><|visual token 006010|><|visual token 089504|><|visual token 110321|><|visual token 119203|><|visual token 101865|><|visual token 025749|><|visual token 082493|><|visual token 011398|><|visual token 083965|><|visual token 025648|><|visual token 019350|><|visual token 035409|><|visual token 013338|><|visual token 035225|><|visual token 016286|><|visual token 070587|><|visual token 027020|><|visual token 036760|><|visual token 127688|><|visual token 001883|><|extra_200|><|visual token 099348|><|visual token 053358|><|visual token 091872|><|visual token 024031|><|visual token 100087|><|visual token 065188|><|visual token 092202|><|visual token 108740|><|visual token 014536|><|visual token 081615|><|visual token 102258|><|visual token 046923|><|visual token 115288|><|visual token 113312|><|visual token 125236|><|visual token 051063|><|visual token 037371|><|visual token 084691|><|visual token 031987|><|visual token 052509|><|visual token 009176|><|visual token 008702|><|visual token 067815|><|visual token 102163|><|visual token 119124|><|visual token 116500|><|visual token 021185|><|visual token 129401|><|visual token 037152|><|visual token 081055|><|visual token 128436|><|visual token 006393|><|extra_200|><|visual token 078134|><|visual token 063941|><|visual token 049524|><|visual token 087866|><|visual token 079452|><|visual token 047617|><|visual token 089619|><|visual token 017007|><|visual token 047617|><|visual token 103636|><|visual token 055778|><|visual token 113450|><|visual token 038626|><|visual token 011770|><|visual token 003508|><|visual token 070203|><|visual token 104787|><|visual token 056709|><|visual token 106823|><|visual token 014818|><|visual token 094540|><|visual token 114565|><|visual token 113500|><|visual token 008711|><|visual token 015241|><|visual token 054082|><|visual token 113234|><|visual token 012147|><|visual token 068202|><|visual token 101019|><|visual token 103070|><|visual token 010888|><|extra_200|><|visual token 008607|><|visual token 089635|><|visual token 051176|><|visual token 034756|><|visual token 051235|><|visual token 118196|><|visual token 055554|><|visual token 025855|><|visual token 074736|><|visual token 045260|><|visual token 108789|><|visual token 093874|><|visual token 076791|><|visual token 083382|><|visual token 028417|><|visual token 085678|><|visual token 043287|><|visual token 002742|><|visual token 061653|><|visual token 049209|><|visual token 080772|><|visual token 019498|><|visual token 114703|><|visual token 105573|><|visual token 081251|><|visual token 113923|><|visual token 119258|><|visual token 007519|><|visual token 082420|><|visual token 064719|><|visual token 030828|><|visual token 048336|><|extra_200|><|visual token 005874|><|visual token 057124|><|visual token 008148|><|visual token 111606|><|visual token 037129|><|visual token 127451|><|visual token 117908|><|visual token 034229|><|visual token 034659|><|visual token 097285|><|visual token 013597|><|visual token 127198|><|visual token 085774|><|visual token 029024|><|visual token 104195|><|visual token 020157|><|visual token 114006|><|visual token 028377|><|visual token 034578|><|visual token 045037|><|visual token 078323|><|visual token 077207|><|visual token 099609|><|visual token 011332|><|visual token 070850|><|visual token 074385|><|visual token 086796|><|visual token 064349|><|visual token 098461|><|visual token 026137|><|visual token 060795|><|visual token 046622|><|extra_200|><|visual token 028244|><|visual token 071157|><|visual token 080068|><|visual token 017524|><|visual token 014494|><|visual token 110479|><|visual token 068091|><|visual token 112127|><|visual token 120126|><|visual token 118794|><|visual token 043769|><|visual token 062830|><|visual token 124754|><|visual token 123847|><|visual token 048348|><|visual token 030048|><|visual token 042007|><|visual token 113977|><|visual token 054847|><|visual token 072145|><|visual token 034812|><|visual token 113517|><|visual token 111851|><|visual token 001764|><|visual token 007936|><|visual token 019919|><|visual token 115215|><|visual token 060046|><|visual token 056490|><|visual token 100446|><|visual token 091651|><|visual token 124384|><|extra_200|><|visual token 101617|><|visual token 112302|><|visual token 086429|><|visual token 110642|><|visual token 054071|><|visual token 026568|><|visual token 013538|><|visual token 070594|><|visual token 102316|><|visual token 041721|><|visual token 048597|><|visual token 128296|><|visual token 097136|><|visual token 121317|><|visual token 059468|><|visual token 090307|><|visual token 084431|><|visual token 124013|><|visual token 105362|><|visual token 007453|><|visual token 129940|><|visual token 082933|><|visual token 060421|><|visual token 118502|><|visual token 016394|><|visual token 071352|><|visual token 094122|><|visual token 076959|><|visual token 057271|><|visual token 081510|><|visual token 015161|><|visual token 044370|><|extra_200|><|visual token 074243|><|visual token 078082|><|visual token 031353|><|visual token 094654|><|visual token 009266|><|visual token 050955|><|visual token 030295|><|visual token 089301|><|visual token 022578|><|visual token 128885|><|visual token 101022|><|visual token 036811|><|visual token 089301|><|visual token 089301|><|visual token 101022|><|visual token 021145|><|visual token 097059|><|visual token 047190|><|visual token 094426|><|visual token 065833|><|visual token 117276|><|visual token 020211|><|visual token 039963|><|visual token 089301|><|visual token 121295|><|visual token 022578|><|visual token 103762|><|visual token 121667|><|visual token 084264|><|visual token 107815|><|visual token 011843|><|visual token 108348|><|extra_200|><|visual token 064945|><|visual token 105304|><|visual token 080052|><|visual token 019822|><|visual token 068945|><|visual token 018787|><|visual token 129558|><|visual token 087108|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 013555|><|visual token 013555|><|visual token 080266|><|visual token 087326|><|visual token 006021|><|visual token 080395|><|visual token 108539|><|visual token 129558|><|visual token 111561|><|visual token 111821|><|visual token 004015|><|visual token 024398|><|visual token 024417|><|visual token 095772|><|visual token 066488|><|visual token 098635|><|visual token 104023|><|image end|><|image start|>32*32<|image token|><|visual token 027913|><|visual token 088473|><|visual token 054821|><|visual token 048384|><|visual token 007808|><|visual token 044303|><|visual token 099858|><|visual token 011332|><|visual token 055887|><|visual token 032108|><|visual token 083909|><|visual token 083909|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 125661|><|visual token 011332|><|visual token 125661|><|visual token 051964|><|visual token 022673|><|visual token 113201|><|visual token 087578|><|visual token 090820|><|extra_200|><|visual token 016747|><|visual token 044285|><|visual token 124640|><|visual token 000176|><|visual token 016365|><|visual token 001345|><|visual token 012484|><|visual token 089162|><|visual token 026492|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 020504|><|visual token 026492|><|visual token 020504|><|visual token 026492|><|visual token 092827|><|visual token 037873|><|visual token 077042|><|visual token 076157|><|visual token 116734|><|extra_200|><|visual token 021708|><|visual token 021115|><|visual token 112970|><|visual token 101487|><|visual token 022384|><|visual token 083324|><|visual token 010331|><|visual token 075986|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 124308|><|visual token 098758|><|visual token 109485|><|visual token 003445|><|visual token 112476|><|visual token 041476|><|visual token 048200|><|visual token 078908|><|visual token 094092|><|extra_200|><|visual token 059965|><|visual token 108993|><|visual token 085877|><|visual token 057360|><|visual token 122584|><|visual token 036690|><|visual token 124671|><|visual token 018767|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 095708|><|visual token 054014|><|visual token 036832|><|visual token 098111|><|visual token 099932|><|visual token 007694|><|visual token 072417|><|visual token 033009|><|extra_200|><|visual token 118115|><|visual token 121206|><|visual token 032772|><|visual token 017727|><|visual token 006419|><|visual token 107459|><|visual token 066066|><|visual token 122694|><|visual token 069821|><|visual token 013912|><|visual token 123185|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 123185|><|visual token 013912|><|visual token 043970|><|visual token 013912|><|visual token 123185|><|visual token 108832|><|visual token 025665|><|visual token 091378|><|visual token 013912|><|visual token 113298|><|visual token 106786|><|visual token 102446|><|visual token 092800|><|extra_200|><|visual token 053732|><|visual token 083161|><|visual token 091876|><|visual token 021753|><|visual token 055557|><|visual token 121496|><|visual token 074827|><|visual token 077277|><|visual token 073290|><|visual token 081590|><|visual token 030728|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 030728|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 030728|><|visual token 067729|><|visual token 106808|><|visual token 095273|><|visual token 064060|><|visual token 098533|><|visual token 029744|><|visual token 029328|><|visual token 117071|><|visual token 067322|><|visual token 075357|><|visual token 086832|><|extra_200|><|visual token 110776|><|visual token 108618|><|visual token 028407|><|visual token 020368|><|visual token 020531|><|visual token 023849|><|visual token 091213|><|visual token 057554|><|visual token 055352|><|visual token 017540|><|visual token 070057|><|visual token 129391|><|visual token 020875|><|visual token 129391|><|visual token 020875|><|visual token 020875|><|visual token 129391|><|visual token 061004|><|visual token 020875|><|visual token 105169|><|visual token 020542|><|visual token 082965|><|visual token 105169|><|visual token 039689|><|visual token 119369|><|visual token 107943|><|visual token 029250|><|visual token 054094|><|visual token 020095|><|visual token 102724|><|visual token 117859|><|visual token 065361|><|extra_200|><|visual token 004011|><|visual token 045244|><|visual token 079743|><|visual token 040510|><|visual token 033186|><|visual token 008736|><|visual token 050228|><|visual token 054662|><|visual token 039142|><|visual token 009381|><|visual token 031613|><|visual token 052422|><|visual token 096127|><|visual token 128470|><|visual token 117054|><|visual token 003287|><|visual token 015156|><|visual token 114780|><|visual token 025601|><|visual token 025164|><|visual token 097897|><|visual token 012083|><|visual token 023342|><|visual token 074827|><|visual token 023079|><|visual token 001304|><|visual token 082838|><|visual token 046564|><|visual token 066914|><|visual token 041665|><|visual token 022960|><|visual token 076681|><|extra_200|><|visual token 003381|><|visual token 108618|><|visual token 096640|><|visual token 123469|><|visual token 025862|><|visual token 047431|><|visual token 072375|><|visual token 025552|><|visual token 071239|><|visual token 059755|><|visual token 087629|><|visual token 037431|><|visual token 056221|><|visual token 013050|><|visual token 003704|><|visual token 058953|><|visual token 084739|><|visual token 067779|><|visual token 045921|><|visual token 021060|><|visual token 122684|><|visual token 113500|><|visual token 065469|><|visual token 061320|><|visual token 062359|><|visual token 019599|><|visual token 001225|><|visual token 091672|><|visual token 071394|><|visual token 128456|><|visual token 008345|><|visual token 078555|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 073996|><|visual token 070387|><|visual token 028361|><|visual token 032270|><|visual token 013568|><|visual token 059201|><|visual token 075381|><|visual token 054979|><|visual token 067493|><|visual token 061004|><|visual token 094718|><|visual token 056221|><|visual token 081188|><|visual token 079181|><|visual token 090784|><|visual token 020812|><|visual token 066066|><|visual token 120911|><|visual token 096041|><|visual token 035518|><|visual token 103338|><|visual token 073090|><|visual token 066320|><|visual token 070333|><|visual token 108061|><|visual token 032994|><|visual token 079655|><|visual token 093748|><|visual token 047849|><|visual token 085476|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 011670|><|visual token 080453|><|visual token 113382|><|visual token 067282|><|visual token 100546|><|visual token 025056|><|visual token 008816|><|visual token 017301|><|visual token 056456|><|visual token 036163|><|visual token 110516|><|visual token 049836|><|visual token 050079|><|visual token 073006|><|visual token 085319|><|visual token 071250|><|visual token 009912|><|visual token 105057|><|visual token 037909|><|visual token 113555|><|visual token 009961|><|visual token 040751|><|visual token 018011|><|visual token 102633|><|visual token 030282|><|visual token 020217|><|visual token 003183|><|visual token 049559|><|visual token 021777|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 088941|><|visual token 105510|><|visual token 008027|><|visual token 061763|><|visual token 052552|><|visual token 104315|><|visual token 069688|><|visual token 062242|><|visual token 004287|><|visual token 072640|><|visual token 127044|><|visual token 015482|><|visual token 064189|><|visual token 088688|><|visual token 110415|><|visual token 034516|><|visual token 067177|><|visual token 103225|><|visual token 082993|><|visual token 101194|><|visual token 117138|><|visual token 002926|><|visual token 097231|><|visual token 096692|><|visual token 014672|><|visual token 064084|><|visual token 085807|><|visual token 040317|><|visual token 000671|><|visual token 040476|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 073996|><|visual token 001126|><|visual token 111997|><|visual token 004231|><|visual token 035040|><|visual token 116818|><|visual token 071386|><|visual token 000963|><|visual token 127252|><|visual token 017327|><|visual token 122413|><|visual token 034466|><|visual token 085285|><|visual token 091973|><|visual token 037766|><|visual token 052162|><|visual token 058666|><|visual token 057959|><|visual token 085656|><|visual token 106047|><|visual token 129160|><|visual token 083862|><|visual token 054925|><|visual token 094288|><|visual token 121538|><|visual token 055557|><|visual token 041460|><|visual token 128279|><|visual token 066957|><|visual token 098243|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 033720|><|visual token 029562|><|visual token 031132|><|visual token 102698|><|visual token 027992|><|visual token 072573|><|visual token 082645|><|visual token 083187|><|visual token 001822|><|visual token 082806|><|visual token 085250|><|visual token 091790|><|visual token 121372|><|visual token 130527|><|visual token 087905|><|visual token 092838|><|visual token 084627|><|visual token 102034|><|visual token 054184|><|visual token 001468|><|visual token 037629|><|visual token 108610|><|visual token 110443|><|visual token 071428|><|visual token 081697|><|visual token 019212|><|visual token 086568|><|visual token 034554|><|visual token 001781|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 107389|><|visual token 125860|><|visual token 088837|><|visual token 110023|><|visual token 027937|><|visual token 060787|><|visual token 053261|><|visual token 076640|><|visual token 003445|><|visual token 098169|><|visual token 027155|><|visual token 003001|><|visual token 116150|><|visual token 072197|><|visual token 037347|><|visual token 095994|><|visual token 015621|><|visual token 073553|><|visual token 014625|><|visual token 055639|><|visual token 058501|><|visual token 003661|><|visual token 041281|><|visual token 010822|><|visual token 098564|><|visual token 064776|><|visual token 002842|><|visual token 115423|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 113177|><|visual token 089507|><|visual token 105517|><|visual token 124377|><|visual token 095804|><|visual token 038837|><|visual token 034685|><|visual token 028846|><|visual token 006073|><|visual token 053742|><|visual token 048650|><|visual token 114138|><|visual token 116563|><|visual token 128916|><|visual token 064538|><|visual token 041574|><|visual token 103959|><|visual token 095304|><|visual token 128612|><|visual token 072057|><|visual token 011385|><|visual token 029342|><|visual token 003395|><|visual token 048447|><|visual token 046781|><|visual token 076493|><|visual token 045060|><|visual token 047450|><|visual token 095387|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 096440|><|visual token 063812|><|visual token 093134|><|visual token 070748|><|visual token 038647|><|visual token 038034|><|visual token 018543|><|visual token 026402|><|visual token 080218|><|visual token 113758|><|visual token 087635|><|visual token 088595|><|visual token 126199|><|visual token 127892|><|visual token 038673|><|visual token 120698|><|visual token 115288|><|visual token 053901|><|visual token 072216|><|visual token 028968|><|visual token 026325|><|visual token 111812|><|visual token 017524|><|visual token 123219|><|visual token 078213|><|visual token 107810|><|visual token 062068|><|visual token 005304|><|visual token 111768|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 113177|><|visual token 072580|><|visual token 122163|><|visual token 037874|><|visual token 084575|><|visual token 111340|><|visual token 055057|><|visual token 051586|><|visual token 015074|><|visual token 072485|><|visual token 034657|><|visual token 117296|><|visual token 024223|><|visual token 056221|><|visual token 051305|><|visual token 078373|><|visual token 123123|><|visual token 097257|><|visual token 064737|><|visual token 003300|><|visual token 116166|><|visual token 007054|><|visual token 055171|><|visual token 063206|><|visual token 051193|><|visual token 079941|><|visual token 067275|><|visual token 057015|><|visual token 069743|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 007054|><|visual token 089507|><|visual token 105850|><|visual token 126247|><|visual token 038034|><|visual token 098822|><|visual token 029822|><|visual token 023009|><|visual token 074992|><|visual token 009098|><|visual token 103930|><|visual token 010758|><|visual token 074568|><|visual token 090784|><|visual token 020812|><|visual token 113298|><|visual token 012533|><|visual token 030011|><|visual token 010970|><|visual token 119100|><|visual token 067354|><|visual token 042033|><|visual token 042560|><|visual token 119968|><|visual token 009357|><|visual token 092870|><|visual token 086001|><|visual token 050805|><|visual token 089652|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 033720|><|visual token 047009|><|visual token 058339|><|visual token 037543|><|visual token 080218|><|visual token 126612|><|visual token 051586|><|visual token 027020|><|visual token 123816|><|visual token 013534|><|visual token 110376|><|visual token 110376|><|visual token 045169|><|visual token 021060|><|visual token 073621|><|visual token 001764|><|visual token 018660|><|visual token 117550|><|visual token 086902|><|visual token 095008|><|visual token 009151|><|visual token 051855|><|visual token 028234|><|visual token 083261|><|visual token 038442|><|visual token 067806|><|visual token 071551|><|visual token 112569|><|visual token 105362|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 064045|><|visual token 029328|><|visual token 087793|><|visual token 012592|><|visual token 117114|><|visual token 076127|><|visual token 039471|><|visual token 114455|><|visual token 018787|><|visual token 069107|><|visual token 089995|><|visual token 031628|><|visual token 054903|><|visual token 079090|><|visual token 098196|><|visual token 085587|><|visual token 011513|><|visual token 073905|><|visual token 020867|><|visual token 069948|><|visual token 076238|><|visual token 089074|><|visual token 107905|><|visual token 049498|><|visual token 109175|><|visual token 068064|><|visual token 006975|><|visual token 054218|><|visual token 071237|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 088838|><|visual token 049412|><|visual token 040262|><|visual token 006009|><|visual token 108964|><|visual token 052521|><|visual token 083546|><|visual token 095166|><|visual token 034212|><|visual token 045135|><|visual token 083771|><|visual token 089944|><|visual token 006633|><|visual token 122025|><|visual token 125831|><|visual token 123816|><|visual token 079854|><|visual token 073190|><|visual token 029853|><|visual token 118644|><|visual token 056611|><|visual token 019203|><|visual token 089683|><|visual token 041775|><|visual token 106874|><|visual token 010909|><|visual token 113977|><|visual token 093799|><|visual token 091523|><|extra_200|><|visual token 093483|><|visual token 002270|><|visual token 117333|><|visual token 124246|><|visual token 004898|><|visual token 089192|><|visual token 071753|><|visual token 003210|><|visual token 031385|><|visual token 072176|><|visual token 105692|><|visual token 029973|><|visual token 093490|><|visual token 048549|><|visual token 009098|><|visual token 081067|><|visual token 107347|><|visual token 067652|><|visual token 027783|><|visual token 109355|><|visual token 111165|><|visual token 073764|><|visual token 041937|><|visual token 124649|><|visual token 105666|><|visual token 002564|><|visual token 118450|><|visual token 121733|><|visual token 108329|><|visual token 129080|><|visual token 038576|><|visual token 066737|><|extra_200|><|visual token 128739|><|visual token 073293|><|visual token 088941|><|visual token 102263|><|visual token 128631|><|visual token 092575|><|visual token 071963|><|visual token 093421|><|visual token 075549|><|visual token 102128|><|visual token 079068|><|visual token 056750|><|visual token 086941|><|visual token 089504|><|visual token 027541|><|visual token 048572|><|visual token 012428|><|visual token 067779|><|visual token 021161|><|visual token 089993|><|visual token 038900|><|visual token 089113|><|visual token 038100|><|visual token 116365|><|visual token 105076|><|visual token 127023|><|visual token 088006|><|visual token 030617|><|visual token 097235|><|visual token 095643|><|visual token 115245|><|visual token 060896|><|extra_200|><|visual token 099348|><|visual token 053358|><|visual token 091872|><|visual token 024031|><|visual token 100087|><|visual token 065188|><|visual token 092202|><|visual token 108740|><|visual token 014536|><|visual token 081615|><|visual token 102258|><|visual token 046923|><|visual token 115288|><|visual token 108477|><|visual token 125236|><|visual token 051063|><|visual token 014808|><|visual token 121139|><|visual token 125751|><|visual token 090041|><|visual token 039347|><|visual token 124790|><|visual token 105882|><|visual token 026520|><|visual token 094653|><|visual token 029744|><|visual token 084531|><|visual token 049789|><|visual token 021145|><|visual token 127994|><|visual token 014297|><|visual token 075555|><|extra_200|><|visual token 078134|><|visual token 084303|><|visual token 049524|><|visual token 087866|><|visual token 079452|><|visual token 047617|><|visual token 089619|><|visual token 017007|><|visual token 047617|><|visual token 103636|><|visual token 055778|><|visual token 054180|><|visual token 074808|><|visual token 011770|><|visual token 003508|><|visual token 070203|><|visual token 104787|><|visual token 056709|><|visual token 097301|><|visual token 063463|><|visual token 046687|><|visual token 061218|><|visual token 007456|><|visual token 091803|><|visual token 115141|><|visual token 063159|><|visual token 053460|><|visual token 022748|><|visual token 000662|><|visual token 101019|><|visual token 103070|><|visual token 090708|><|extra_200|><|visual token 008607|><|visual token 089635|><|visual token 051176|><|visual token 034756|><|visual token 051235|><|visual token 118196|><|visual token 055554|><|visual token 025855|><|visual token 074736|><|visual token 045260|><|visual token 045037|><|visual token 093874|><|visual token 076791|><|visual token 083382|><|visual token 028417|><|visual token 085678|><|visual token 111102|><|visual token 008397|><|visual token 061653|><|visual token 009030|><|visual token 032474|><|visual token 008021|><|visual token 037465|><|visual token 013611|><|visual token 062703|><|visual token 079387|><|visual token 073105|><|visual token 007519|><|visual token 082420|><|visual token 064719|><|visual token 102019|><|visual token 048336|><|extra_200|><|visual token 005874|><|visual token 057124|><|visual token 008148|><|visual token 111606|><|visual token 037129|><|visual token 127451|><|visual token 090307|><|visual token 034229|><|visual token 034659|><|visual token 097285|><|visual token 013597|><|visual token 127198|><|visual token 085774|><|visual token 029024|><|visual token 031040|><|visual token 020157|><|visual token 094257|><|visual token 107430|><|visual token 086458|><|visual token 117114|><|visual token 052060|><|visual token 072680|><|visual token 069677|><|visual token 090968|><|visual token 067770|><|visual token 005874|><|visual token 086796|><|visual token 064349|><|visual token 098461|><|visual token 026137|><|visual token 060795|><|visual token 046622|><|extra_200|><|visual token 028244|><|visual token 071157|><|visual token 080068|><|visual token 017524|><|visual token 014494|><|visual token 110479|><|visual token 068091|><|visual token 112127|><|visual token 120126|><|visual token 118794|><|visual token 043769|><|visual token 120571|><|visual token 007694|><|visual token 075801|><|visual token 049343|><|visual token 030048|><|visual token 042007|><|visual token 115696|><|visual token 085753|><|visual token 057263|><|visual token 072145|><|visual token 042228|><|visual token 113517|><|visual token 019713|><|visual token 111703|><|visual token 085265|><|visual token 041807|><|visual token 023249|><|visual token 060046|><|visual token 056490|><|visual token 108332|><|visual token 123508|><|extra_200|><|visual token 101617|><|visual token 112302|><|visual token 086429|><|visual token 110642|><|visual token 054071|><|visual token 026568|><|visual token 013538|><|visual token 070594|><|visual token 099782|><|visual token 041721|><|visual token 048597|><|visual token 128296|><|visual token 097136|><|visual token 121317|><|visual token 091634|><|visual token 020542|><|visual token 007529|><|visual token 082930|><|visual token 086851|><|visual token 041588|><|visual token 110613|><|visual token 089144|><|visual token 112727|><|visual token 064071|><|visual token 088270|><|visual token 099584|><|visual token 038757|><|visual token 020211|><|visual token 009004|><|visual token 039575|><|visual token 081627|><|visual token 057368|><|extra_200|><|visual token 074243|><|visual token 078082|><|visual token 031353|><|visual token 094654|><|visual token 009266|><|visual token 050955|><|visual token 030295|><|visual token 089301|><|visual token 128885|><|visual token 128885|><|visual token 101022|><|visual token 036811|><|visual token 089301|><|visual token 022874|><|visual token 101022|><|visual token 048414|><|visual token 054985|><|visual token 020349|><|visual token 112956|><|visual token 041332|><|visual token 071432|><|visual token 020463|><|visual token 018071|><|visual token 025177|><|visual token 016957|><|visual token 094056|><|visual token 030295|><|visual token 103762|><|visual token 034278|><|visual token 122815|><|visual token 120726|><|visual token 100499|><|extra_200|><|visual token 064945|><|visual token 105304|><|visual token 057130|><|visual token 019822|><|visual token 068945|><|visual token 018787|><|visual token 129558|><|visual token 087108|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 013555|><|visual token 122307|><|visual token 073247|><|visual token 122017|><|visual token 094980|><|visual token 049648|><|visual token 033612|><|visual token 117674|><|visual token 056991|><|visual token 082148|><|visual token 104283|><|visual token 007914|><|visual token 000731|><|visual token 095273|><|visual token 098572|><|visual token 095103|><|visual token 027426|><|visual token 078267|><|visual token 027845|><|visual token 035028|><|image end|><|image start|>32*32<|image token|><|visual token 027913|><|visual token 088473|><|visual token 054821|><|visual token 048384|><|visual token 007808|><|visual token 044303|><|visual token 099858|><|visual token 011332|><|visual token 055887|><|visual token 032108|><|visual token 083909|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 083909|><|visual token 032108|><|visual token 032108|><|visual token 083909|><|visual token 125661|><|visual token 011332|><|visual token 125661|><|visual token 051964|><|visual token 022673|><|visual token 113201|><|visual token 082326|><|visual token 090820|><|extra_200|><|visual token 016747|><|visual token 044285|><|visual token 124640|><|visual token 000176|><|visual token 016365|><|visual token 001345|><|visual token 012484|><|visual token 089162|><|visual token 026492|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 020504|><|visual token 026492|><|visual token 020504|><|visual token 026492|><|visual token 092827|><|visual token 037873|><|visual token 077042|><|visual token 076157|><|visual token 013651|><|extra_200|><|visual token 021708|><|visual token 021115|><|visual token 112970|><|visual token 101487|><|visual token 022384|><|visual token 083324|><|visual token 010331|><|visual token 075986|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 033450|><|visual token 124308|><|visual token 098758|><|visual token 109485|><|visual token 034087|><|visual token 086278|><|visual token 118225|><|visual token 048200|><|visual token 078908|><|visual token 094092|><|extra_200|><|visual token 059965|><|visual token 108993|><|visual token 085877|><|visual token 057360|><|visual token 122584|><|visual token 036690|><|visual token 124671|><|visual token 018767|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 054014|><|visual token 126749|><|visual token 062882|><|visual token 117050|><|visual token 052317|><|visual token 078555|><|visual token 001626|><|visual token 033009|><|extra_200|><|visual token 118115|><|visual token 121206|><|visual token 032772|><|visual token 017727|><|visual token 006419|><|visual token 107459|><|visual token 066066|><|visual token 122694|><|visual token 069821|><|visual token 013912|><|visual token 123185|><|visual token 013912|><|visual token 123185|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 013912|><|visual token 123185|><|visual token 123185|><|visual token 123185|><|visual token 123185|><|visual token 123185|><|visual token 123185|><|visual token 013912|><|visual token 021060|><|visual token 049279|><|visual token 013912|><|visual token 113298|><|visual token 099566|><|visual token 106823|><|visual token 126243|><|extra_200|><|visual token 053732|><|visual token 083161|><|visual token 091876|><|visual token 021753|><|visual token 055557|><|visual token 019848|><|visual token 074827|><|visual token 077277|><|visual token 088530|><|visual token 081590|><|visual token 030728|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 051162|><|visual token 106808|><|visual token 030728|><|visual token 036562|><|visual token 106808|><|visual token 036562|><|visual token 053338|><|visual token 114892|><|visual token 020838|><|visual token 007439|><|visual token 092919|><|visual token 113070|><|visual token 109491|><|visual token 125253|><|visual token 061924|><|extra_200|><|visual token 110776|><|visual token 108618|><|visual token 028407|><|visual token 020368|><|visual token 020531|><|visual token 054884|><|visual token 091213|><|visual token 057554|><|visual token 055352|><|visual token 017540|><|visual token 129391|><|visual token 129391|><|visual token 129391|><|visual token 129391|><|visual token 020875|><|visual token 020875|><|visual token 061004|><|visual token 020875|><|visual token 105169|><|visual token 072246|><|visual token 108770|><|visual token 017661|><|visual token 068520|><|visual token 108539|><|visual token 016079|><|visual token 041665|><|visual token 108509|><|visual token 015161|><|visual token 090073|><|visual token 110703|><|visual token 105076|><|visual token 095392|><|extra_200|><|visual token 004011|><|visual token 045244|><|visual token 079743|><|visual token 040510|><|visual token 033186|><|visual token 008736|><|visual token 050228|><|visual token 054662|><|visual token 039142|><|visual token 009381|><|visual token 031613|><|visual token 052422|><|visual token 096127|><|visual token 128470|><|visual token 117054|><|visual token 012701|><|visual token 031328|><|visual token 053311|><|visual token 126413|><|visual token 083268|><|visual token 000508|><|visual token 029539|><|visual token 030752|><|visual token 069851|><|visual token 071790|><|visual token 022102|><|visual token 054325|><|visual token 081136|><|visual token 081364|><|visual token 012437|><|visual token 041665|><|visual token 129416|><|visual token 126002|><|extra_200|><|visual token 093483|><|visual token 108618|><|visual token 096640|><|visual token 123469|><|visual token 025862|><|visual token 047431|><|visual token 053149|><|visual token 025552|><|visual token 071239|><|visual token 059755|><|visual token 017540|><|visual token 103930|><|visual token 056221|><|visual token 049641|><|visual token 002729|><|visual token 088457|><|visual token 027541|><|visual token 056221|><|visual token 010527|><|visual token 095643|><|visual token 123238|><|visual token 006407|><|visual token 041677|><|visual token 105705|><|visual token 004100|><|visual token 108576|><|visual token 053755|><|visual token 091368|><|visual token 057987|><|visual token 033510|><|visual token 099522|><|visual token 079980|><|visual token 078963|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 091440|><|visual token 070387|><|visual token 028361|><|visual token 036237|><|visual token 013568|><|visual token 106179|><|visual token 058785|><|visual token 054979|><|visual token 067493|><|visual token 061004|><|visual token 094718|><|visual token 056221|><|visual token 021109|><|visual token 079181|><|visual token 107709|><|visual token 091071|><|visual token 096582|><|visual token 040939|><|visual token 106793|><|visual token 019665|><|visual token 116636|><|visual token 107877|><|visual token 015677|><|visual token 073561|><|visual token 101127|><|visual token 041205|><|visual token 057475|><|visual token 036601|><|visual token 007879|><|visual token 113845|><|visual token 121992|><|extra_200|><|visual token 091453|><|visual token 002085|><|visual token 117333|><|visual token 011670|><|visual token 080453|><|visual token 113382|><|visual token 067282|><|visual token 100546|><|visual token 025056|><|visual token 061320|><|visual token 017301|><|visual token 056456|><|visual token 049641|><|visual token 110516|><|visual token 007914|><|visual token 050079|><|visual token 054129|><|visual token 011759|><|visual token 030093|><|visual token 115022|><|visual token 115621|><|visual token 050676|><|visual token 086902|><|visual token 069441|><|visual token 045435|><|visual token 031903|><|visual token 045362|><|visual token 045362|><|visual token 130759|><|visual token 062540|><|visual token 100551|><|visual token 003661|><|visual token 028291|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 088941|><|visual token 105510|><|visual token 008027|><|visual token 061763|><|visual token 052552|><|visual token 104315|><|visual token 069688|><|visual token 046363|><|visual token 004287|><|visual token 095120|><|visual token 127044|><|visual token 015482|><|visual token 064189|><|visual token 029328|><|visual token 032285|><|visual token 034516|><|visual token 057136|><|visual token 001702|><|visual token 096765|><|visual token 042713|><|visual token 116808|><|visual token 069415|><|visual token 015719|><|visual token 108265|><|visual token 050312|><|visual token 017827|><|visual token 087225|><|visual token 084079|><|visual token 009560|><|visual token 014934|><|visual token 091306|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 001126|><|visual token 117935|><|visual token 004231|><|visual token 035040|><|visual token 116818|><|visual token 062434|><|visual token 000963|><|visual token 090237|><|visual token 115974|><|visual token 122413|><|visual token 034466|><|visual token 085285|><|visual token 006028|><|visual token 022793|><|visual token 099444|><|visual token 081800|><|visual token 061320|><|visual token 129137|><|visual token 015059|><|visual token 007383|><|visual token 065109|><|visual token 073024|><|visual token 087653|><|visual token 041588|><|visual token 024566|><|visual token 052562|><|visual token 072186|><|visual token 070141|><|visual token 111812|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 033720|><|visual token 084150|><|visual token 031132|><|visual token 102698|><|visual token 027992|><|visual token 072573|><|visual token 082645|><|visual token 083187|><|visual token 001822|><|visual token 082806|><|visual token 085250|><|visual token 091790|><|visual token 121372|><|visual token 130527|><|visual token 087905|><|visual token 092838|><|visual token 058192|><|visual token 106597|><|visual token 109223|><|visual token 089118|><|visual token 081191|><|visual token 097347|><|visual token 060133|><|visual token 048199|><|visual token 112466|><|visual token 024728|><|visual token 053047|><|visual token 007985|><|visual token 094351|><|visual token 105468|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 107389|><|visual token 125860|><|visual token 080151|><|visual token 110023|><|visual token 027937|><|visual token 060787|><|visual token 053261|><|visual token 076640|><|visual token 119823|><|visual token 098169|><|visual token 027155|><|visual token 003001|><|visual token 116150|><|visual token 090264|><|visual token 086073|><|visual token 095994|><|visual token 099777|><|visual token 073553|><|visual token 079984|><|visual token 093286|><|visual token 121545|><|visual token 048095|><|visual token 100623|><|visual token 002270|><|visual token 045244|><|visual token 041566|><|visual token 062440|><|visual token 063380|><|visual token 015033|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 113177|><|visual token 089507|><|visual token 105517|><|visual token 124377|><|visual token 095804|><|visual token 038837|><|visual token 034685|><|visual token 028846|><|visual token 006073|><|visual token 053742|><|visual token 068287|><|visual token 114138|><|visual token 116563|><|visual token 128916|><|visual token 064538|><|visual token 088158|><|visual token 107392|><|visual token 054748|><|visual token 021302|><|visual token 129985|><|visual token 002620|><|visual token 020044|><|visual token 013513|><|visual token 052652|><|visual token 061717|><|visual token 113048|><|visual token 060267|><|visual token 084216|><|visual token 048685|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 096440|><|visual token 063812|><|visual token 093134|><|visual token 070748|><|visual token 020531|><|visual token 038034|><|visual token 018543|><|visual token 026402|><|visual token 080218|><|visual token 015677|><|visual token 087635|><|visual token 088595|><|visual token 126199|><|visual token 127892|><|visual token 038673|><|visual token 033310|><|visual token 129982|><|visual token 094019|><|visual token 118973|><|visual token 097580|><|visual token 071716|><|visual token 052763|><|visual token 100692|><|visual token 095791|><|visual token 016877|><|visual token 044771|><|visual token 075733|><|visual token 017615|><|visual token 115696|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 104787|><|visual token 072580|><|visual token 008816|><|visual token 037874|><|visual token 084575|><|visual token 111340|><|visual token 055057|><|visual token 051586|><|visual token 082993|><|visual token 068286|><|visual token 034657|><|visual token 119799|><|visual token 024223|><|visual token 089995|><|visual token 051305|><|visual token 008223|><|visual token 060575|><|visual token 101675|><|visual token 025425|><|visual token 112520|><|visual token 032536|><|visual token 071645|><|visual token 082116|><|visual token 034205|><|visual token 056872|><|visual token 117581|><|visual token 123512|><|visual token 063953|><|visual token 002819|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 089507|><|visual token 105850|><|visual token 098461|><|visual token 038034|><|visual token 098822|><|visual token 029822|><|visual token 023009|><|visual token 074992|><|visual token 111986|><|visual token 103930|><|visual token 010758|><|visual token 128182|><|visual token 107709|><|visual token 013605|><|visual token 015709|><|visual token 088511|><|visual token 030968|><|visual token 080147|><|visual token 001357|><|visual token 082111|><|visual token 014845|><|visual token 057778|><|visual token 018204|><|visual token 021645|><|visual token 117766|><|visual token 037000|><|visual token 111394|><|visual token 032762|><|visual token 063050|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 033720|><|visual token 063223|><|visual token 058339|><|visual token 037543|><|visual token 080218|><|visual token 126612|><|visual token 051586|><|visual token 027020|><|visual token 123816|><|visual token 021109|><|visual token 110376|><|visual token 110376|><|visual token 058953|><|visual token 091071|><|visual token 105417|><|visual token 061051|><|visual token 060414|><|visual token 058902|><|visual token 041765|><|visual token 024591|><|visual token 015487|><|visual token 107467|><|visual token 003651|><|visual token 128284|><|visual token 055888|><|visual token 042630|><|visual token 105176|><|visual token 014724|><|visual token 042515|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 064045|><|visual token 029328|><|visual token 087793|><|visual token 012592|><|visual token 117114|><|visual token 076127|><|visual token 059175|><|visual token 114455|><|visual token 018787|><|visual token 069107|><|visual token 089995|><|visual token 021109|><|visual token 054903|><|visual token 101337|><|visual token 032765|><|visual token 111118|><|visual token 063375|><|visual token 107846|><|visual token 084283|><|visual token 051531|><|visual token 033492|><|visual token 001233|><|visual token 074054|><|visual token 010331|><|visual token 007786|><|visual token 074582|><|visual token 028846|><|visual token 120652|><|visual token 115844|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 088838|><|visual token 049412|><|visual token 040262|><|visual token 006009|><|visual token 108964|><|visual token 019519|><|visual token 083546|><|visual token 095166|><|visual token 034212|><|visual token 045135|><|visual token 083771|><|visual token 089944|><|visual token 035147|><|visual token 107709|><|visual token 038694|><|visual token 119817|><|visual token 064907|><|visual token 084165|><|visual token 084675|><|visual token 038640|><|visual token 033769|><|visual token 129530|><|visual token 125499|><|visual token 013534|><|visual token 104360|><|visual token 057026|><|visual token 067831|><|visual token 076509|><|visual token 020364|><|visual token 013236|><|extra_200|><|visual token 093483|><|visual token 002270|><|visual token 117333|><|visual token 124246|><|visual token 004898|><|visual token 070589|><|visual token 071753|><|visual token 055353|><|visual token 031385|><|visual token 072176|><|visual token 105692|><|visual token 090775|><|visual token 067779|><|visual token 048549|><|visual token 111986|><|visual token 013605|><|visual token 129434|><|visual token 100764|><|visual token 074682|><|visual token 095414|><|visual token 062100|><|visual token 096376|><|visual token 049953|><|visual token 087837|><|visual token 097756|><|visual token 063005|><|visual token 079104|><|visual token 090413|><|visual token 115209|><|visual token 066770|><|visual token 086454|><|visual token 039532|><|visual token 009644|><|extra_200|><|visual token 128739|><|visual token 073293|><|visual token 088941|><|visual token 102263|><|visual token 128631|><|visual token 092575|><|visual token 071963|><|visual token 059343|><|visual token 075549|><|visual token 102128|><|visual token 008064|><|visual token 056750|><|visual token 086941|><|visual token 089504|><|visual token 094418|><|visual token 049641|><|visual token 101865|><|visual token 031103|><|visual token 067717|><|visual token 112218|><|visual token 083965|><|visual token 040501|><|visual token 104986|><|visual token 087248|><|visual token 015241|><|visual token 021109|><|visual token 108832|><|visual token 027997|><|visual token 003072|><|visual token 063102|><|visual token 006830|><|visual token 054041|><|extra_200|><|visual token 099348|><|visual token 053358|><|visual token 091872|><|visual token 024031|><|visual token 100087|><|visual token 065188|><|visual token 092202|><|visual token 108740|><|visual token 014536|><|visual token 081615|><|visual token 102258|><|visual token 046923|><|visual token 115288|><|visual token 108477|><|visual token 125236|><|visual token 051063|><|visual token 037371|><|visual token 084691|><|visual token 039089|><|visual token 075821|><|visual token 009176|><|visual token 121972|><|visual token 112351|><|visual token 049703|><|visual token 083051|><|visual token 032922|><|visual token 065862|><|visual token 094470|><|visual token 002921|><|visual token 119749|><|visual token 058383|><|visual token 124070|><|visual token 039019|><|extra_200|><|visual token 078134|><|visual token 084303|><|visual token 049524|><|visual token 087866|><|visual token 079452|><|visual token 116261|><|visual token 089619|><|visual token 017007|><|visual token 047617|><|visual token 103636|><|visual token 055778|><|visual token 054180|><|visual token 074808|><|visual token 011770|><|visual token 003508|><|visual token 070203|><|visual token 104787|><|visual token 040388|><|visual token 068914|><|visual token 066524|><|visual token 097255|><|visual token 025104|><|visual token 086212|><|visual token 097124|><|visual token 028900|><|visual token 066531|><|visual token 076798|><|visual token 040409|><|visual token 093680|><|visual token 090183|><|visual token 053612|><|visual token 011283|><|extra_200|><|visual token 008607|><|visual token 089635|><|visual token 051176|><|visual token 034756|><|visual token 051235|><|visual token 118196|><|visual token 055554|><|visual token 064333|><|visual token 074736|><|visual token 045260|><|visual token 045037|><|visual token 055170|><|visual token 076791|><|visual token 083382|><|visual token 009967|><|visual token 085678|><|visual token 111102|><|visual token 002742|><|visual token 061653|><|visual token 130092|><|visual token 080772|><|visual token 019498|><|visual token 003072|><|visual token 105573|><|visual token 112610|><|visual token 005883|><|visual token 073105|><|visual token 007519|><|visual token 033171|><|visual token 055388|><|visual token 123080|><|visual token 101757|><|extra_200|><|visual token 005874|><|visual token 057124|><|visual token 008148|><|visual token 111606|><|visual token 037129|><|visual token 127451|><|visual token 117908|><|visual token 034229|><|visual token 034659|><|visual token 097285|><|visual token 013597|><|visual token 127198|><|visual token 085774|><|visual token 029024|><|visual token 031040|><|visual token 103962|><|visual token 012228|><|visual token 119691|><|visual token 086458|><|visual token 076431|><|visual token 082632|><|visual token 072680|><|visual token 113927|><|visual token 017616|><|visual token 054883|><|visual token 085319|><|visual token 024449|><|visual token 064349|><|visual token 102698|><|visual token 013470|><|visual token 016926|><|visual token 076498|><|extra_200|><|visual token 028244|><|visual token 071157|><|visual token 080068|><|visual token 017524|><|visual token 014494|><|visual token 110479|><|visual token 068091|><|visual token 112127|><|visual token 120126|><|visual token 118794|><|visual token 093539|><|visual token 062830|><|visual token 007694|><|visual token 073931|><|visual token 003000|><|visual token 030048|><|visual token 042007|><|visual token 113977|><|visual token 054847|><|visual token 088452|><|visual token 034812|><|visual token 113517|><|visual token 092977|><|visual token 042486|><|visual token 097319|><|visual token 129451|><|visual token 023155|><|visual token 060046|><|visual token 056490|><|visual token 099087|><|visual token 091686|><|visual token 079428|><|extra_200|><|visual token 101617|><|visual token 112302|><|visual token 086429|><|visual token 110642|><|visual token 054071|><|visual token 026568|><|visual token 013538|><|visual token 070594|><|visual token 099782|><|visual token 041721|><|visual token 062502|><|visual token 128296|><|visual token 097136|><|visual token 121317|><|visual token 059468|><|visual token 029375|><|visual token 093833|><|visual token 088393|><|visual token 106813|><|visual token 053950|><|visual token 021489|><|visual token 082933|><|visual token 102679|><|visual token 021280|><|visual token 020083|><|visual token 068996|><|visual token 001095|><|visual token 076959|><|visual token 057271|><|visual token 114834|><|visual token 108737|><|visual token 129290|><|extra_200|><|visual token 074243|><|visual token 078082|><|visual token 031353|><|visual token 094654|><|visual token 009266|><|visual token 050955|><|visual token 030295|><|visual token 089301|><|visual token 128885|><|visual token 022578|><|visual token 101022|><|visual token 036811|><|visual token 089301|><|visual token 089301|><|visual token 101022|><|visual token 021145|><|visual token 097059|><|visual token 129428|><|visual token 072905|><|visual token 124749|><|visual token 117276|><|visual token 020211|><|visual token 039963|><|visual token 089301|><|visual token 026492|><|visual token 022578|><|visual token 103762|><|visual token 121667|><|visual token 084264|><|visual token 118449|><|visual token 048564|><|visual token 033179|><|extra_200|><|visual token 064945|><|visual token 105304|><|visual token 057130|><|visual token 019822|><|visual token 068945|><|visual token 018787|><|visual token 129558|><|visual token 087108|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 013555|><|visual token 013555|><|visual token 005072|><|visual token 087326|><|visual token 006021|><|visual token 080395|><|visual token 108539|><|visual token 129558|><|visual token 013555|><|visual token 111821|><|visual token 004015|><|visual token 024398|><|visual token 024417|><|visual token 095772|><|visual token 066488|><|visual token 098635|><|visual token 104023|><|image end|><|image start|>32*32<|image token|><|visual token 027913|><|visual token 088473|><|visual token 054821|><|visual token 048384|><|visual token 007808|><|visual token 044303|><|visual token 099858|><|visual token 011332|><|visual token 055887|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 083909|><|visual token 125661|><|visual token 099429|><|visual token 125661|><|visual token 051964|><|visual token 022673|><|visual token 113201|><|visual token 016257|><|visual token 090820|><|extra_200|><|visual token 016747|><|visual token 044285|><|visual token 124640|><|visual token 000176|><|visual token 016365|><|visual token 001345|><|visual token 012484|><|visual token 089162|><|visual token 026492|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 077095|><|visual token 026492|><|visual token 034174|><|visual token 026492|><|visual token 092827|><|visual token 037873|><|visual token 109533|><|visual token 076157|><|visual token 116734|><|extra_200|><|visual token 021708|><|visual token 021115|><|visual token 112970|><|visual token 101487|><|visual token 022384|><|visual token 083324|><|visual token 010331|><|visual token 075986|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 109485|><|visual token 043016|><|visual token 086278|><|visual token 041476|><|visual token 048200|><|visual token 075897|><|visual token 094092|><|extra_200|><|visual token 059965|><|visual token 108993|><|visual token 085877|><|visual token 057360|><|visual token 122584|><|visual token 036690|><|visual token 097565|><|visual token 018767|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126452|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 096325|><|visual token 126749|><|visual token 126749|><|visual token 117050|><|visual token 126749|><|visual token 126749|><|visual token 117050|><|visual token 054014|><|visual token 036832|><|visual token 117050|><|visual token 053272|><|visual token 007694|><|visual token 072417|><|visual token 045597|><|extra_200|><|visual token 118115|><|visual token 121206|><|visual token 032772|><|visual token 017727|><|visual token 006419|><|visual token 107459|><|visual token 085133|><|visual token 122694|><|visual token 069821|><|visual token 069821|><|visual token 035244|><|visual token 069821|><|visual token 047190|><|visual token 013912|><|visual token 069821|><|visual token 021823|><|visual token 047190|><|visual token 083268|><|visual token 013522|><|visual token 065111|><|visual token 055964|><|visual token 055964|><|visual token 043970|><|visual token 035244|><|visual token 013912|><|visual token 090592|><|visual token 091378|><|visual token 108832|><|visual token 065293|><|visual token 106786|><|visual token 067275|><|visual token 092800|><|extra_200|><|visual token 053732|><|visual token 083161|><|visual token 091876|><|visual token 021753|><|visual token 065176|><|visual token 121496|><|visual token 074827|><|visual token 038747|><|visual token 035710|><|visual token 130939|><|visual token 020425|><|visual token 123289|><|visual token 051255|><|visual token 071480|><|visual token 067729|><|visual token 078691|><|visual token 088276|><|visual token 108509|><|visual token 007309|><|visual token 027774|><|visual token 107212|><|visual token 006740|><|visual token 129451|><|visual token 122017|><|visual token 039581|><|visual token 124702|><|visual token 099103|><|visual token 104629|><|visual token 099566|><|visual token 006853|><|visual token 075357|><|visual token 068189|><|extra_200|><|visual token 110776|><|visual token 108618|><|visual token 028407|><|visual token 009974|><|visual token 020531|><|visual token 023849|><|visual token 091213|><|visual token 127193|><|visual token 021855|><|visual token 045964|><|visual token 107943|><|visual token 054586|><|visual token 033233|><|visual token 057049|><|visual token 113220|><|visual token 025419|><|visual token 087088|><|visual token 084257|><|visual token 123641|><|visual token 060598|><|visual token 107615|><|visual token 092636|><|visual token 109078|><|visual token 037363|><|visual token 057469|><|visual token 081796|><|visual token 069009|><|visual token 067275|><|visual token 063585|><|visual token 023324|><|visual token 117859|><|visual token 095413|><|extra_200|><|visual token 004011|><|visual token 045244|><|visual token 079743|><|visual token 128935|><|visual token 033186|><|visual token 049632|><|visual token 050228|><|visual token 054662|><|visual token 034520|><|visual token 024855|><|visual token 109688|><|visual token 114631|><|visual token 078462|><|visual token 053298|><|visual token 085016|><|visual token 079991|><|visual token 092429|><|visual token 062751|><|visual token 054351|><|visual token 085016|><|visual token 120571|><|visual token 110824|><|visual token 120571|><|visual token 113860|><|visual token 006541|><|visual token 124684|><|visual token 000654|><|visual token 120571|><|visual token 035462|><|visual token 081796|><|visual token 095785|><|visual token 008834|><|visual token 091866|><|extra_200|><|visual token 093483|><|visual token 108618|><|visual token 096640|><|visual token 050292|><|visual token 111829|><|visual token 047431|><|visual token 013330|><|visual token 128492|><|visual token 046596|><|visual token 057293|><|visual token 027877|><|visual token 008465|><|visual token 085641|><|visual token 036280|><|visual token 050108|><|visual token 072346|><|visual token 021824|><|visual token 103470|><|visual token 102793|><|visual token 080722|><|visual token 030076|><|visual token 050104|><|visual token 082558|><|visual token 046801|><|visual token 003661|><|visual token 002741|><|visual token 037038|><|visual token 019875|><|visual token 005186|><|visual token 079915|><|visual token 021432|><|visual token 080363|><|visual token 125798|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 091440|><|visual token 070387|><|visual token 028361|><|visual token 007823|><|visual token 070838|><|visual token 122024|><|visual token 116234|><|visual token 121819|><|visual token 071272|><|visual token 088751|><|visual token 023265|><|visual token 082171|><|visual token 069223|><|visual token 112080|><|visual token 069180|><|visual token 085197|><|visual token 096689|><|visual token 004454|><|visual token 084299|><|visual token 062572|><|visual token 052247|><|visual token 055475|><|visual token 086792|><|visual token 005341|><|visual token 128902|><|visual token 093346|><|visual token 029767|><|visual token 030230|><|visual token 092151|><|visual token 034875|><|visual token 051291|><|extra_200|><|visual token 091453|><|visual token 002085|><|visual token 117333|><|visual token 011670|><|visual token 054502|><|visual token 053698|><|visual token 129262|><|visual token 067357|><|visual token 075213|><|visual token 104753|><|visual token 006512|><|visual token 006512|><|visual token 051822|><|visual token 059636|><|visual token 028443|><|visual token 036557|><|visual token 119816|><|visual token 127599|><|visual token 016551|><|visual token 006436|><|visual token 091160|><|visual token 108979|><|visual token 069447|><|visual token 044082|><|visual token 056611|><|visual token 024566|><|visual token 072263|><|visual token 126912|><|visual token 022413|><|visual token 072227|><|visual token 070148|><|visual token 049652|><|visual token 070914|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 088941|><|visual token 025076|><|visual token 008027|><|visual token 061763|><|visual token 110609|><|visual token 016291|><|visual token 034854|><|visual token 003434|><|visual token 005444|><|visual token 076373|><|visual token 036469|><|visual token 039898|><|visual token 123023|><|visual token 121998|><|visual token 079396|><|visual token 128084|><|visual token 084560|><|visual token 127115|><|visual token 044474|><|visual token 118220|><|visual token 110033|><|visual token 017957|><|visual token 115590|><|visual token 119387|><|visual token 073399|><|visual token 049345|><|visual token 060326|><|visual token 023576|><|visual token 026498|><|visual token 068630|><|visual token 071965|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 073996|><|visual token 021931|><|visual token 117935|><|visual token 009912|><|visual token 035040|><|visual token 021353|><|visual token 024249|><|visual token 047364|><|visual token 026242|><|visual token 039248|><|visual token 096312|><|visual token 119133|><|visual token 056257|><|visual token 003651|><|visual token 054779|><|visual token 102436|><|visual token 058254|><|visual token 106514|><|visual token 130049|><|visual token 072623|><|visual token 110359|><|visual token 025264|><|visual token 011482|><|visual token 092426|><|visual token 107482|><|visual token 084639|><|visual token 012402|><|visual token 041574|><|visual token 103059|><|visual token 020553|><|visual token 115689|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 042624|><|visual token 084150|><|visual token 076151|><|visual token 034560|><|visual token 027992|><|visual token 017346|><|visual token 035411|><|visual token 065992|><|visual token 050158|><|visual token 099094|><|visual token 057042|><|visual token 046551|><|visual token 085421|><|visual token 110070|><|visual token 112377|><|visual token 120051|><|visual token 068134|><|visual token 015478|><|visual token 058909|><|visual token 104558|><|visual token 025894|><|visual token 051586|><|visual token 012715|><|visual token 105076|><|visual token 051654|><|visual token 092884|><|visual token 048389|><|visual token 129144|><|visual token 065222|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 072580|><|visual token 127786|><|visual token 088837|><|visual token 017377|><|visual token 020662|><|visual token 101680|><|visual token 053261|><|visual token 015409|><|visual token 067983|><|visual token 036925|><|visual token 107486|><|visual token 068911|><|visual token 070621|><|visual token 113425|><|visual token 060919|><|visual token 021851|><|visual token 052928|><|visual token 016081|><|visual token 040913|><|visual token 088130|><|visual token 061431|><|visual token 012003|><|visual token 073563|><|visual token 107312|><|visual token 052492|><|visual token 121432|><|visual token 041627|><|visual token 031550|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 072117|><|visual token 035594|><|visual token 086938|><|visual token 024110|><|visual token 095804|><|visual token 038837|><|visual token 021356|><|visual token 028846|><|visual token 075080|><|visual token 054051|><|visual token 120002|><|visual token 110566|><|visual token 065872|><|visual token 048348|><|visual token 007371|><|visual token 012567|><|visual token 049726|><|visual token 119055|><|visual token 024011|><|visual token 036663|><|visual token 068897|><|visual token 025188|><|visual token 115650|><|visual token 094858|><|visual token 086005|><|visual token 119203|><|visual token 097452|><|visual token 033899|><|visual token 085437|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 042624|><|visual token 063812|><|visual token 046991|><|visual token 105400|><|visual token 038890|><|visual token 038034|><|visual token 129160|><|visual token 026402|><|visual token 045173|><|visual token 036299|><|visual token 040896|><|visual token 114453|><|visual token 126345|><|visual token 077375|><|visual token 003892|><|visual token 062129|><|visual token 043128|><|visual token 066238|><|visual token 019735|><|visual token 044812|><|visual token 028612|><|visual token 065550|><|visual token 031767|><|visual token 113462|><|visual token 080416|><|visual token 079232|><|visual token 046565|><|visual token 044812|><|visual token 112066|><|visual token 021231|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 113177|><|visual token 072580|><|visual token 008816|><|visual token 037874|><|visual token 067305|><|visual token 062970|><|visual token 055057|><|visual token 051586|><|visual token 015074|><|visual token 012080|><|visual token 048182|><|visual token 030729|><|visual token 039312|><|visual token 075467|><|visual token 111606|><|visual token 066066|><|visual token 102508|><|visual token 126546|><|visual token 124728|><|visual token 114432|><|visual token 075912|><|visual token 094392|><|visual token 104345|><|visual token 092971|><|visual token 128550|><|visual token 071730|><|visual token 014145|><|visual token 129793|><|visual token 044905|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 072117|><|visual token 089507|><|visual token 065469|><|visual token 126247|><|visual token 038034|><|visual token 001779|><|visual token 029822|><|visual token 023009|><|visual token 074992|><|visual token 126976|><|visual token 038113|><|visual token 059471|><|visual token 048692|><|visual token 007243|><|visual token 043476|><|visual token 053427|><|visual token 060022|><|visual token 043759|><|visual token 059544|><|visual token 041914|><|visual token 097768|><|visual token 130642|><|visual token 112727|><|visual token 058254|><|visual token 112776|><|visual token 050692|><|visual token 036993|><|visual token 116705|><|visual token 024698|><|visual token 033931|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 033720|><|visual token 027009|><|visual token 058339|><|visual token 037543|><|visual token 080218|><|visual token 126612|><|visual token 051586|><|visual token 027020|><|visual token 053259|><|visual token 083665|><|visual token 093113|><|visual token 002031|><|visual token 028784|><|visual token 012433|><|visual token 116515|><|visual token 065558|><|visual token 050943|><|visual token 023009|><|visual token 070009|><|visual token 060764|><|visual token 126199|><|visual token 014724|><|visual token 026387|><|visual token 032324|><|visual token 086227|><|visual token 058074|><|visual token 024344|><|visual token 001005|><|visual token 054041|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 045819|><|visual token 029328|><|visual token 087793|><|visual token 012592|><|visual token 098822|><|visual token 076127|><|visual token 059175|><|visual token 114455|><|visual token 046349|><|visual token 032897|><|visual token 104912|><|visual token 020378|><|visual token 014335|><|visual token 072568|><|visual token 089383|><|visual token 059671|><|visual token 102865|><|visual token 099636|><|visual token 122609|><|visual token 051531|><|visual token 084732|><|visual token 124151|><|visual token 052311|><|visual token 106655|><|visual token 092276|><|visual token 074376|><|visual token 038736|><|visual token 116777|><|visual token 086938|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 088838|><|visual token 049412|><|visual token 116355|><|visual token 006009|><|visual token 108964|><|visual token 019519|><|visual token 083546|><|visual token 095166|><|visual token 110514|><|visual token 045135|><|visual token 083771|><|visual token 020598|><|visual token 064060|><|visual token 078905|><|visual token 115527|><|visual token 113070|><|visual token 102436|><|visual token 084165|><|visual token 084675|><|visual token 093612|><|visual token 045190|><|visual token 017143|><|visual token 092301|><|visual token 095881|><|visual token 068479|><|visual token 067941|><|visual token 023276|><|visual token 066253|><|visual token 032777|><|extra_200|><|visual token 093483|><|visual token 002270|><|visual token 117333|><|visual token 124246|><|visual token 004898|><|visual token 089192|><|visual token 071753|><|visual token 003210|><|visual token 031385|><|visual token 072176|><|visual token 105692|><|visual token 088457|><|visual token 026011|><|visual token 002841|><|visual token 111986|><|visual token 039313|><|visual token 124414|><|visual token 037558|><|visual token 101889|><|visual token 107308|><|visual token 013490|><|visual token 096376|><|visual token 049953|><|visual token 087837|><|visual token 097756|><|visual token 051305|><|visual token 096783|><|visual token 052810|><|visual token 078808|><|visual token 090307|><|visual token 027888|><|visual token 035705|><|extra_200|><|visual token 128739|><|visual token 073293|><|visual token 088941|><|visual token 102263|><|visual token 128631|><|visual token 092575|><|visual token 071963|><|visual token 059343|><|visual token 075549|><|visual token 102128|><|visual token 008064|><|visual token 056750|><|visual token 079104|><|visual token 004071|><|visual token 030873|><|visual token 035244|><|visual token 059866|><|visual token 048597|><|visual token 019563|><|visual token 112218|><|visual token 083965|><|visual token 040501|><|visual token 104986|><|visual token 095414|><|visual token 045765|><|visual token 013534|><|visual token 067652|><|visual token 027997|><|visual token 059866|><|visual token 110627|><|visual token 008687|><|visual token 056478|><|extra_200|><|visual token 099348|><|visual token 082451|><|visual token 091872|><|visual token 024031|><|visual token 126951|><|visual token 065188|><|visual token 092202|><|visual token 108740|><|visual token 047290|><|visual token 081615|><|visual token 087512|><|visual token 036981|><|visual token 126243|><|visual token 108477|><|visual token 101545|><|visual token 051063|><|visual token 092126|><|visual token 124636|><|visual token 039089|><|visual token 044843|><|visual token 009176|><|visual token 121972|><|visual token 112351|><|visual token 107035|><|visual token 079122|><|visual token 085897|><|visual token 065862|><|visual token 094470|><|visual token 068383|><|visual token 058383|><|visual token 106563|><|visual token 085001|><|extra_200|><|visual token 078134|><|visual token 084303|><|visual token 049524|><|visual token 087866|><|visual token 079452|><|visual token 116261|><|visual token 112315|><|visual token 017007|><|visual token 047617|><|visual token 129851|><|visual token 055778|><|visual token 054180|><|visual token 074808|><|visual token 011770|><|visual token 003508|><|visual token 070203|><|visual token 104787|><|visual token 031198|><|visual token 068914|><|visual token 066524|><|visual token 097255|><|visual token 025104|><|visual token 019921|><|visual token 128137|><|visual token 108047|><|visual token 027156|><|visual token 042792|><|visual token 061350|><|visual token 066786|><|visual token 071130|><|visual token 071875|><|visual token 048158|><|extra_200|><|visual token 008607|><|visual token 089635|><|visual token 051176|><|visual token 034756|><|visual token 051235|><|visual token 118196|><|visual token 055554|><|visual token 064333|><|visual token 074736|><|visual token 045260|><|visual token 096215|><|visual token 093874|><|visual token 076791|><|visual token 083382|><|visual token 009967|><|visual token 085678|><|visual token 093755|><|visual token 002742|><|visual token 061653|><|visual token 115730|><|visual token 124852|><|visual token 019498|><|visual token 003072|><|visual token 053306|><|visual token 130160|><|visual token 087372|><|visual token 038054|><|visual token 064269|><|visual token 044332|><|visual token 055388|><|visual token 013971|><|visual token 035338|><|extra_200|><|visual token 005874|><|visual token 057124|><|visual token 008148|><|visual token 111606|><|visual token 037129|><|visual token 127451|><|visual token 015411|><|visual token 034229|><|visual token 034659|><|visual token 097285|><|visual token 013597|><|visual token 127198|><|visual token 085774|><|visual token 029024|><|visual token 031040|><|visual token 103962|><|visual token 126204|><|visual token 018543|><|visual token 086458|><|visual token 029094|><|visual token 033670|><|visual token 127223|><|visual token 113927|><|visual token 088577|><|visual token 021060|><|visual token 085319|><|visual token 100484|><|visual token 035745|><|visual token 085662|><|visual token 120571|><|visual token 096719|><|visual token 040056|><|extra_200|><|visual token 028244|><|visual token 071157|><|visual token 080068|><|visual token 017524|><|visual token 014494|><|visual token 110479|><|visual token 068091|><|visual token 112127|><|visual token 120126|><|visual token 118794|><|visual token 043769|><|visual token 062830|><|visual token 124754|><|visual token 073931|><|visual token 049343|><|visual token 030048|><|visual token 042007|><|visual token 113977|><|visual token 054847|><|visual token 072145|><|visual token 034812|><|visual token 113517|><|visual token 092977|><|visual token 036239|><|visual token 027979|><|visual token 012734|><|visual token 053629|><|visual token 092997|><|visual token 123120|><|visual token 107943|><|visual token 096859|><|visual token 059638|><|extra_200|><|visual token 101617|><|visual token 112302|><|visual token 086429|><|visual token 110642|><|visual token 054071|><|visual token 026568|><|visual token 013538|><|visual token 070594|><|visual token 099782|><|visual token 041721|><|visual token 128296|><|visual token 124749|><|visual token 063626|><|visual token 121317|><|visual token 059468|><|visual token 029375|><|visual token 084431|><|visual token 056187|><|visual token 116122|><|visual token 053950|><|visual token 089144|><|visual token 083695|><|visual token 067264|><|visual token 011786|><|visual token 015151|><|visual token 126835|><|visual token 043481|><|visual token 104149|><|visual token 116058|><|visual token 037924|><|visual token 105905|><|visual token 120688|><|extra_200|><|visual token 074243|><|visual token 078082|><|visual token 031353|><|visual token 094654|><|visual token 009266|><|visual token 050955|><|visual token 030295|><|visual token 089301|><|visual token 022578|><|visual token 128885|><|visual token 101022|><|visual token 036811|><|visual token 089301|><|visual token 101022|><|visual token 123185|><|visual token 055734|><|visual token 070587|><|visual token 106182|><|visual token 012518|><|visual token 117450|><|visual token 110342|><|visual token 083637|><|visual token 011355|><|visual token 021145|><|visual token 063431|><|visual token 017669|><|visual token 046518|><|visual token 128806|><|visual token 071811|><|visual token 041890|><|visual token 025494|><|visual token 117674|><|extra_200|><|visual token 064945|><|visual token 105304|><|visual token 057130|><|visual token 019822|><|visual token 068945|><|visual token 018787|><|visual token 129558|><|visual token 087108|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 078955|><|visual token 090285|><|visual token 013555|><|visual token 013555|><|visual token 090285|><|visual token 091973|><|visual token 093574|><|visual token 059443|><|visual token 091882|><|visual token 128118|><|visual token 029480|><|visual token 026628|><|visual token 007167|><|visual token 049567|><|visual token 052689|><|visual token 059238|><|visual token 116705|><|visual token 118691|><|visual token 124377|><|visual token 098635|><|visual token 030979|><|image end|><|image start|>32*32<|image token|><|visual token 027913|><|visual token 088473|><|visual token 054821|><|visual token 048384|><|visual token 007808|><|visual token 044303|><|visual token 099858|><|visual token 011332|><|visual token 055887|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 083909|><|visual token 125661|><|visual token 099429|><|visual token 125661|><|visual token 051964|><|visual token 083956|><|visual token 113201|><|visual token 082326|><|visual token 090820|><|extra_200|><|visual token 016747|><|visual token 044285|><|visual token 124640|><|visual token 000176|><|visual token 016365|><|visual token 001345|><|visual token 012484|><|visual token 089162|><|visual token 026492|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 020504|><|visual token 026492|><|visual token 034174|><|visual token 026492|><|visual token 092827|><|visual token 037873|><|visual token 077042|><|visual token 076157|><|visual token 116734|><|extra_200|><|visual token 021708|><|visual token 021115|><|visual token 112970|><|visual token 101487|><|visual token 022384|><|visual token 083324|><|visual token 010331|><|visual token 075986|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 033450|><|visual token 109485|><|visual token 002429|><|visual token 112476|><|visual token 040333|><|visual token 048200|><|visual token 078908|><|visual token 094092|><|extra_200|><|visual token 059965|><|visual token 108993|><|visual token 085877|><|visual token 057360|><|visual token 122584|><|visual token 111340|><|visual token 097565|><|visual token 018767|><|visual token 022122|><|visual token 126452|><|visual token 126749|><|visual token 126749|><|visual token 126452|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126749|><|visual token 126452|><|visual token 126749|><|visual token 069914|><|visual token 117050|><|visual token 126749|><|visual token 117050|><|visual token 117050|><|visual token 054014|><|visual token 084885|><|visual token 117050|><|visual token 053272|><|visual token 007694|><|visual token 072417|><|visual token 033009|><|extra_200|><|visual token 118115|><|visual token 121206|><|visual token 032772|><|visual token 017727|><|visual token 006419|><|visual token 063371|><|visual token 006374|><|visual token 088577|><|visual token 047190|><|visual token 121427|><|visual token 043970|><|visual token 058860|><|visual token 013912|><|visual token 013912|><|visual token 114843|><|visual token 108832|><|visual token 121427|><|visual token 121427|><|visual token 065111|><|visual token 055964|><|visual token 013912|><|visual token 055964|><|visual token 039699|><|visual token 121427|><|visual token 121427|><|visual token 043970|><|visual token 066271|><|visual token 097059|><|visual token 125644|><|visual token 069821|><|visual token 012336|><|visual token 092800|><|extra_200|><|visual token 053732|><|visual token 083161|><|visual token 091876|><|visual token 058843|><|visual token 064884|><|visual token 050864|><|visual token 120565|><|visual token 072763|><|visual token 051356|><|visual token 013605|><|visual token 005658|><|visual token 073309|><|visual token 123816|><|visual token 128194|><|visual token 051866|><|visual token 071968|><|visual token 122307|><|visual token 129451|><|visual token 117693|><|visual token 076198|><|visual token 123870|><|visual token 062129|><|visual token 129451|><|visual token 115650|><|visual token 056478|><|visual token 089773|><|visual token 105753|><|visual token 067487|><|visual token 037191|><|visual token 027441|><|visual token 075357|><|visual token 068189|><|extra_200|><|visual token 110776|><|visual token 108618|><|visual token 028407|><|visual token 040156|><|visual token 038478|><|visual token 089996|><|visual token 127540|><|visual token 103442|><|visual token 051773|><|visual token 103475|><|visual token 004646|><|visual token 103812|><|visual token 058630|><|visual token 099888|><|visual token 057049|><|visual token 032717|><|visual token 100056|><|visual token 054607|><|visual token 060598|><|visual token 076153|><|visual token 048430|><|visual token 112971|><|visual token 048845|><|visual token 081796|><|visual token 102683|><|visual token 125616|><|visual token 047235|><|visual token 030951|><|visual token 083450|><|visual token 129443|><|visual token 117859|><|visual token 095413|><|extra_200|><|visual token 004011|><|visual token 042168|><|visual token 072890|><|visual token 093986|><|visual token 013347|><|visual token 025116|><|visual token 075287|><|visual token 015158|><|visual token 074072|><|visual token 060657|><|visual token 070366|><|visual token 046393|><|visual token 025227|><|visual token 120198|><|visual token 024662|><|visual token 076071|><|visual token 107998|><|visual token 004241|><|visual token 085016|><|visual token 064282|><|visual token 124882|><|visual token 019034|><|visual token 018135|><|visual token 069905|><|visual token 064282|><|visual token 109589|><|visual token 040474|><|visual token 085319|><|visual token 001193|><|visual token 017872|><|visual token 070411|><|visual token 031764|><|visual token 085652|><|extra_200|><|visual token 054647|><|visual token 005598|><|visual token 096640|><|visual token 087608|><|visual token 006449|><|visual token 011953|><|visual token 088454|><|visual token 075467|><|visual token 104917|><|visual token 130076|><|visual token 000390|><|visual token 123362|><|visual token 039686|><|visual token 068525|><|visual token 103470|><|visual token 047178|><|visual token 061717|><|visual token 102564|><|visual token 103470|><|visual token 057449|><|visual token 022604|><|visual token 022604|><|visual token 101322|><|visual token 130346|><|visual token 001139|><|visual token 068525|><|visual token 002741|><|visual token 002741|><|visual token 057449|><|visual token 075731|><|visual token 021432|><|visual token 113872|><|visual token 125798|><|extra_200|><|visual token 076077|><|visual token 066903|><|visual token 119238|><|visual token 095967|><|visual token 124411|><|visual token 007053|><|visual token 097815|><|visual token 077405|><|visual token 056770|><|visual token 125788|><|visual token 034801|><|visual token 115975|><|visual token 126755|><|visual token 085025|><|visual token 065383|><|visual token 076308|><|visual token 097121|><|visual token 000659|><|visual token 067343|><|visual token 108933|><|visual token 125671|><|visual token 050136|><|visual token 095572|><|visual token 043093|><|visual token 005341|><|visual token 005341|><|visual token 115242|><|visual token 030825|><|visual token 041050|><|visual token 121886|><|visual token 117583|><|visual token 034369|><|visual token 049626|><|extra_200|><|visual token 091453|><|visual token 002085|><|visual token 117333|><|visual token 112197|><|visual token 015192|><|visual token 040148|><|visual token 077020|><|visual token 076349|><|visual token 001189|><|visual token 129954|><|visual token 067183|><|visual token 053902|><|visual token 079004|><|visual token 021816|><|visual token 035422|><|visual token 075683|><|visual token 032981|><|visual token 115126|><|visual token 032824|><|visual token 083914|><|visual token 049040|><|visual token 121674|><|visual token 018955|><|visual token 096881|><|visual token 032824|><|visual token 048676|><|visual token 058827|><|visual token 060288|><|visual token 049730|><|visual token 125960|><|visual token 124579|><|visual token 004053|><|visual token 077238|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 088941|><|visual token 000731|><|visual token 008027|><|visual token 063102|><|visual token 009579|><|visual token 034042|><|visual token 110873|><|visual token 025639|><|visual token 047049|><|visual token 088751|><|visual token 110221|><|visual token 015932|><|visual token 096157|><|visual token 012413|><|visual token 073411|><|visual token 086518|><|visual token 001001|><|visual token 030968|><|visual token 056293|><|visual token 022649|><|visual token 117515|><|visual token 051737|><|visual token 117361|><|visual token 041765|><|visual token 097612|><|visual token 073399|><|visual token 009932|><|visual token 000507|><|visual token 127560|><|visual token 070377|><|visual token 071237|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 073996|><|visual token 037558|><|visual token 051305|><|visual token 113784|><|visual token 059811|><|visual token 081993|><|visual token 017740|><|visual token 003789|><|visual token 086127|><|visual token 104734|><|visual token 035110|><|visual token 086334|><|visual token 032549|><|visual token 114290|><|visual token 112263|><|visual token 034591|><|visual token 091376|><|visual token 013946|><|visual token 095488|><|visual token 093217|><|visual token 042071|><|visual token 117283|><|visual token 069199|><|visual token 012834|><|visual token 042167|><|visual token 073591|><|visual token 022565|><|visual token 004734|><|visual token 068701|><|visual token 018507|><|visual token 076188|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 042624|><|visual token 084150|><|visual token 070587|><|visual token 068202|><|visual token 027992|><|visual token 046177|><|visual token 087286|><|visual token 116325|><|visual token 062386|><|visual token 102821|><|visual token 118845|><|visual token 113450|><|visual token 068431|><|visual token 126721|><|visual token 014315|><|visual token 117642|><|visual token 089711|><|visual token 123293|><|visual token 042624|><|visual token 041588|><|visual token 002306|><|visual token 024647|><|visual token 119033|><|visual token 018909|><|visual token 084827|><|visual token 062413|><|visual token 100417|><|visual token 009906|><|visual token 011189|><|visual token 102380|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 072580|><|visual token 107061|><|visual token 088837|><|visual token 095967|><|visual token 020662|><|visual token 060787|><|visual token 053261|><|visual token 012253|><|visual token 106176|><|visual token 086458|><|visual token 098593|><|visual token 015156|><|visual token 021889|><|visual token 040288|><|visual token 130558|><|visual token 092832|><|visual token 058244|><|visual token 102821|><|visual token 015642|><|visual token 104322|><|visual token 105849|><|visual token 069318|><|visual token 002751|><|visual token 096876|><|visual token 029539|><|visual token 097235|><|visual token 058670|><|visual token 007875|><|visual token 063119|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 072117|><|visual token 084150|><|visual token 086938|><|visual token 010120|><|visual token 095804|><|visual token 049920|><|visual token 082742|><|visual token 028846|><|visual token 075080|><|visual token 086127|><|visual token 103693|><|visual token 086270|><|visual token 041914|><|visual token 062724|><|visual token 001599|><|visual token 049816|><|visual token 072455|><|visual token 059929|><|visual token 013923|><|visual token 054026|><|visual token 120976|><|visual token 043772|><|visual token 070091|><|visual token 050756|><|visual token 128981|><|visual token 064269|><|visual token 053326|><|visual token 020148|><|visual token 109587|><|visual token 000815|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 042624|><|visual token 063812|><|visual token 046991|><|visual token 105400|><|visual token 031299|><|visual token 038034|><|visual token 129160|><|visual token 098361|><|visual token 097707|><|visual token 059681|><|visual token 118471|><|visual token 080816|><|visual token 076247|><|visual token 129895|><|visual token 017218|><|visual token 033366|><|visual token 085414|><|visual token 075683|><|visual token 106442|><|visual token 025025|><|visual token 074922|><|visual token 005533|><|visual token 058339|><|visual token 039377|><|visual token 104240|><|visual token 123816|><|visual token 068897|><|visual token 097587|><|visual token 125733|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 113177|><|visual token 072580|><|visual token 008816|><|visual token 095103|><|visual token 084575|><|visual token 062970|><|visual token 098059|><|visual token 051586|><|visual token 015074|><|visual token 043128|><|visual token 074488|><|visual token 090183|><|visual token 039752|><|visual token 020171|><|visual token 106262|><|visual token 097239|><|visual token 089202|><|visual token 114453|><|visual token 022516|><|visual token 054599|><|visual token 121533|><|visual token 124175|><|visual token 092682|><|visual token 109878|><|visual token 029749|><|visual token 037560|><|visual token 029571|><|visual token 113047|><|visual token 033310|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 072117|><|visual token 089507|><|visual token 065469|><|visual token 126247|><|visual token 038034|><|visual token 117138|><|visual token 029822|><|visual token 023009|><|visual token 051104|><|visual token 111986|><|visual token 118270|><|visual token 059710|><|visual token 040540|><|visual token 126799|><|visual token 110023|><|visual token 005407|><|visual token 117284|><|visual token 078839|><|visual token 097916|><|visual token 058816|><|visual token 083397|><|visual token 011189|><|visual token 115215|><|visual token 070424|><|visual token 083096|><|visual token 107824|><|visual token 067115|><|visual token 008687|><|visual token 103816|><|visual token 046565|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 033720|><|visual token 081044|><|visual token 055171|><|visual token 061530|><|visual token 080218|><|visual token 126612|><|visual token 051586|><|visual token 027020|><|visual token 119210|><|visual token 065426|><|visual token 093002|><|visual token 127023|><|visual token 115454|><|visual token 009746|><|visual token 078468|><|visual token 093343|><|visual token 029470|><|visual token 011246|><|visual token 116586|><|visual token 082184|><|visual token 011189|><|visual token 124154|><|visual token 062502|><|visual token 049492|><|visual token 085064|><|visual token 100487|><|visual token 090870|><|visual token 065565|><|visual token 032264|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 045819|><|visual token 029328|><|visual token 087793|><|visual token 012592|><|visual token 064031|><|visual token 076127|><|visual token 113521|><|visual token 052328|><|visual token 046349|><|visual token 117789|><|visual token 006010|><|visual token 126806|><|visual token 000406|><|visual token 051417|><|visual token 001606|><|visual token 013572|><|visual token 082069|><|visual token 027666|><|visual token 016784|><|visual token 060344|><|visual token 026527|><|visual token 006177|><|visual token 048652|><|visual token 046071|><|visual token 057208|><|visual token 100297|><|visual token 037084|><|visual token 070387|><|visual token 048549|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 088838|><|visual token 049412|><|visual token 116355|><|visual token 006009|><|visual token 108964|><|visual token 019519|><|visual token 083546|><|visual token 095166|><|visual token 124953|><|visual token 053338|><|visual token 080266|><|visual token 094900|><|visual token 102873|><|visual token 061736|><|visual token 128835|><|visual token 068845|><|visual token 102123|><|visual token 099532|><|visual token 072762|><|visual token 005538|><|visual token 126801|><|visual token 056624|><|visual token 020644|><|visual token 127637|><|visual token 083464|><|visual token 087766|><|visual token 045047|><|visual token 071428|><|visual token 110776|><|extra_200|><|visual token 093483|><|visual token 002270|><|visual token 117333|><|visual token 124246|><|visual token 004898|><|visual token 089192|><|visual token 071753|><|visual token 066320|><|visual token 031385|><|visual token 072176|><|visual token 045060|><|visual token 090775|><|visual token 123664|><|visual token 098888|><|visual token 091664|><|visual token 019169|><|visual token 085338|><|visual token 112727|><|visual token 024185|><|visual token 054120|><|visual token 085318|><|visual token 113846|><|visual token 051576|><|visual token 094946|><|visual token 013640|><|visual token 088276|><|visual token 123755|><|visual token 077664|><|visual token 120698|><|visual token 089375|><|visual token 081969|><|visual token 078624|><|extra_200|><|visual token 128739|><|visual token 073293|><|visual token 088941|><|visual token 102263|><|visual token 128631|><|visual token 092575|><|visual token 071963|><|visual token 059343|><|visual token 075549|><|visual token 102128|><|visual token 091880|><|visual token 056750|><|visual token 079104|><|visual token 118915|><|visual token 094468|><|visual token 027278|><|visual token 129205|><|visual token 044440|><|visual token 109746|><|visual token 049653|><|visual token 121781|><|visual token 068911|><|visual token 031970|><|visual token 124234|><|visual token 020614|><|visual token 048091|><|visual token 039602|><|visual token 023155|><|visual token 127125|><|visual token 017116|><|visual token 128224|><|visual token 001526|><|extra_200|><|visual token 099348|><|visual token 082451|><|visual token 091872|><|visual token 024031|><|visual token 126951|><|visual token 096801|><|visual token 092202|><|visual token 108740|><|visual token 047290|><|visual token 033052|><|visual token 087512|><|visual token 036981|><|visual token 126243|><|visual token 130134|><|visual token 071576|><|visual token 055888|><|visual token 035682|><|visual token 011022|><|visual token 130111|><|visual token 080526|><|visual token 009176|><|visual token 025783|><|visual token 023773|><|visual token 021277|><|visual token 009863|><|visual token 108113|><|visual token 070261|><|visual token 060251|><|visual token 017856|><|visual token 075646|><|visual token 041986|><|visual token 117403|><|visual token 118291|><|extra_200|><|visual token 078134|><|visual token 084303|><|visual token 049524|><|visual token 087866|><|visual token 079452|><|visual token 116261|><|visual token 075454|><|visual token 011202|><|visual token 047617|><|visual token 129851|><|visual token 055778|><|visual token 054180|><|visual token 074808|><|visual token 011770|><|visual token 003508|><|visual token 024031|><|visual token 101456|><|visual token 046778|><|visual token 067895|><|visual token 014818|><|visual token 028012|><|visual token 092211|><|visual token 129205|><|visual token 099176|><|visual token 131009|><|visual token 040417|><|visual token 102405|><|visual token 024430|><|visual token 079509|><|visual token 008979|><|visual token 079463|><|visual token 070282|><|extra_200|><|visual token 008607|><|visual token 089635|><|visual token 051176|><|visual token 034756|><|visual token 051235|><|visual token 118196|><|visual token 055554|><|visual token 050027|><|visual token 074736|><|visual token 045260|><|visual token 086252|><|visual token 104558|><|visual token 085651|><|visual token 100793|><|visual token 009967|><|visual token 085678|><|visual token 029419|><|visual token 002742|><|visual token 004165|><|visual token 115730|><|visual token 080772|><|visual token 019498|><|visual token 046564|><|visual token 130160|><|visual token 062632|><|visual token 011504|><|visual token 101141|><|visual token 105417|><|visual token 098572|><|visual token 015294|><|visual token 098699|><|visual token 036246|><|extra_200|><|visual token 005874|><|visual token 057124|><|visual token 008148|><|visual token 111606|><|visual token 037129|><|visual token 127451|><|visual token 022396|><|visual token 077824|><|visual token 034659|><|visual token 009013|><|visual token 013597|><|visual token 127198|><|visual token 085774|><|visual token 029024|><|visual token 031040|><|visual token 020157|><|visual token 126204|><|visual token 012563|><|visual token 086458|><|visual token 076431|><|visual token 120517|><|visual token 077207|><|visual token 036623|><|visual token 010288|><|visual token 065748|><|visual token 097553|><|visual token 101321|><|visual token 084821|><|visual token 074054|><|visual token 049726|><|visual token 053491|><|visual token 121237|><|extra_200|><|visual token 028244|><|visual token 071157|><|visual token 080068|><|visual token 017524|><|visual token 014494|><|visual token 110479|><|visual token 068091|><|visual token 112127|><|visual token 120126|><|visual token 118794|><|visual token 115162|><|visual token 053617|><|visual token 105076|><|visual token 029157|><|visual token 004153|><|visual token 030048|><|visual token 042007|><|visual token 115696|><|visual token 098744|><|visual token 072145|><|visual token 034812|><|visual token 113517|><|visual token 092977|><|visual token 010935|><|visual token 085878|><|visual token 028801|><|visual token 085001|><|visual token 016365|><|visual token 119176|><|visual token 099087|><|visual token 052390|><|visual token 124515|><|extra_200|><|visual token 101617|><|visual token 112302|><|visual token 086429|><|visual token 110642|><|visual token 054071|><|visual token 026568|><|visual token 013538|><|visual token 070594|><|visual token 099782|><|visual token 106482|><|visual token 114556|><|visual token 098572|><|visual token 063626|><|visual token 121317|><|visual token 059468|><|visual token 076036|><|visual token 007529|><|visual token 097566|><|visual token 025680|><|visual token 094529|><|visual token 033371|><|visual token 076784|><|visual token 081613|><|visual token 099423|><|visual token 049136|><|visual token 098670|><|visual token 082404|><|visual token 042195|><|visual token 064979|><|visual token 025813|><|visual token 103795|><|visual token 027898|><|extra_200|><|visual token 074243|><|visual token 078082|><|visual token 031353|><|visual token 094654|><|visual token 009266|><|visual token 050955|><|visual token 030295|><|visual token 089301|><|visual token 030295|><|visual token 065030|><|visual token 128850|><|visual token 104018|><|visual token 104018|><|visual token 101022|><|visual token 101278|><|visual token 057521|><|visual token 053253|><|visual token 061497|><|visual token 117670|><|visual token 071186|><|visual token 030089|><|visual token 058931|><|visual token 094204|><|visual token 033634|><|visual token 028622|><|visual token 103053|><|visual token 094204|><|visual token 015090|><|visual token 025822|><|visual token 044547|><|visual token 118537|><|visual token 098891|><|visual token 112953|><|extra_200|><|visual token 064945|><|visual token 105304|><|visual token 057130|><|visual token 019822|><|visual token 068945|><|visual token 018787|><|visual token 129558|><|visual token 087108|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 111561|><|visual token 045135|><|visual token 038694|><|visual token 096732|><|visual token 030966|><|visual token 011419|><|visual token 053253|><|visual token 120652|><|visual token 130798|><|visual token 119133|><|visual token 069428|><|visual token 013656|><|visual token 099950|><|visual token 004817|><|visual token 076079|><|visual token 068453|><|visual token 044923|><|visual token 041075|><|visual token 006420|><|visual token 078226|><|visual token 115288|><|visual token 112953|><|image end|><|image start|>32*32<|image token|><|visual token 027913|><|visual token 088473|><|visual token 054821|><|visual token 048384|><|visual token 007808|><|visual token 044303|><|visual token 099858|><|visual token 011332|><|visual token 029007|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 083909|><|visual token 125661|><|visual token 019128|><|visual token 125661|><|visual token 051964|><|visual token 004590|><|visual token 113201|><|visual token 016257|><|visual token 090820|><|extra_200|><|visual token 016747|><|visual token 044285|><|visual token 124640|><|visual token 000176|><|visual token 016365|><|visual token 001345|><|visual token 012484|><|visual token 089162|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 026492|><|visual token 034174|><|visual token 026492|><|visual token 039142|><|visual token 037873|><|visual token 077042|><|visual token 076157|><|visual token 116734|><|extra_200|><|visual token 021708|><|visual token 021115|><|visual token 112970|><|visual token 101487|><|visual token 022384|><|visual token 023893|><|visual token 010331|><|visual token 075986|><|visual token 124308|><|visual token 124308|><|visual token 033450|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 033450|><|visual token 109485|><|visual token 034087|><|visual token 086278|><|visual token 040333|><|visual token 048200|><|visual token 078908|><|visual token 094092|><|extra_200|><|visual token 059965|><|visual token 108993|><|visual token 085877|><|visual token 057360|><|visual token 122584|><|visual token 083965|><|visual token 097565|><|visual token 043515|><|visual token 126452|><|visual token 126749|><|visual token 126749|><|visual token 126452|><|visual token 126749|><|visual token 126749|><|visual token 126452|><|visual token 126452|><|visual token 126749|><|visual token 126452|><|visual token 126452|><|visual token 126749|><|visual token 069914|><|visual token 117050|><|visual token 117050|><|visual token 126452|><|visual token 117050|><|visual token 054014|><|visual token 084885|><|visual token 117050|><|visual token 054449|><|visual token 017612|><|visual token 072417|><|visual token 033009|><|extra_200|><|visual token 118115|><|visual token 121206|><|visual token 032772|><|visual token 017727|><|visual token 006419|><|visual token 063371|><|visual token 085133|><|visual token 024697|><|visual token 069821|><|visual token 121427|><|visual token 035244|><|visual token 058860|><|visual token 108832|><|visual token 013912|><|visual token 065293|><|visual token 043970|><|visual token 035244|><|visual token 065977|><|visual token 065111|><|visual token 061314|><|visual token 013912|><|visual token 055964|><|visual token 039699|><|visual token 091174|><|visual token 121427|><|visual token 030819|><|visual token 066271|><|visual token 097059|><|visual token 125644|><|visual token 066066|><|visual token 012336|><|visual token 092800|><|extra_200|><|visual token 053732|><|visual token 083161|><|visual token 091876|><|visual token 086281|><|visual token 090932|><|visual token 040386|><|visual token 018062|><|visual token 011139|><|visual token 013605|><|visual token 024344|><|visual token 111058|><|visual token 026219|><|visual token 094718|><|visual token 054129|><|visual token 004822|><|visual token 034788|><|visual token 109923|><|visual token 123870|><|visual token 009453|><|visual token 042952|><|visual token 123870|><|visual token 087818|><|visual token 129451|><|visual token 115650|><|visual token 048309|><|visual token 000135|><|visual token 009453|><|visual token 067487|><|visual token 037191|><|visual token 026387|><|visual token 075357|><|visual token 068189|><|extra_200|><|visual token 110776|><|visual token 108618|><|visual token 028407|><|visual token 012119|><|visual token 076479|><|visual token 057527|><|visual token 120858|><|visual token 034568|><|visual token 066429|><|visual token 009777|><|visual token 091080|><|visual token 118856|><|visual token 088805|><|visual token 034950|><|visual token 087154|><|visual token 016086|><|visual token 070216|><|visual token 053073|><|visual token 058254|><|visual token 056899|><|visual token 121830|><|visual token 103053|><|visual token 028536|><|visual token 081796|><|visual token 080758|><|visual token 125616|><|visual token 038377|><|visual token 096786|><|visual token 093693|><|visual token 095120|><|visual token 117859|><|visual token 095413|><|extra_200|><|visual token 004011|><|visual token 090520|><|visual token 103610|><|visual token 026387|><|visual token 018088|><|visual token 003832|><|visual token 050228|><|visual token 127540|><|visual token 059958|><|visual token 001005|><|visual token 070424|><|visual token 044988|><|visual token 000216|><|visual token 125236|><|visual token 025879|><|visual token 018786|><|visual token 052094|><|visual token 113860|><|visual token 119185|><|visual token 021082|><|visual token 111079|><|visual token 003217|><|visual token 088420|><|visual token 000650|><|visual token 069905|><|visual token 006422|><|visual token 074385|><|visual token 120571|><|visual token 094092|><|visual token 017872|><|visual token 070411|><|visual token 085652|><|extra_200|><|visual token 005086|><|visual token 108618|><|visual token 101764|><|visual token 018353|><|visual token 118361|><|visual token 024844|><|visual token 018002|><|visual token 055915|><|visual token 016809|><|visual token 048920|><|visual token 029998|><|visual token 052643|><|visual token 070481|><|visual token 077091|><|visual token 068064|><|visual token 128018|><|visual token 106071|><|visual token 061717|><|visual token 061717|><|visual token 068525|><|visual token 080722|><|visual token 022604|><|visual token 021824|><|visual token 046801|><|visual token 001139|><|visual token 068525|><|visual token 002741|><|visual token 002741|><|visual token 085641|><|visual token 075731|><|visual token 021432|><|visual token 113872|><|visual token 125798|><|extra_200|><|visual token 036248|><|visual token 118791|><|visual token 015156|><|visual token 047834|><|visual token 044353|><|visual token 053770|><|visual token 017985|><|visual token 017865|><|visual token 093072|><|visual token 029574|><|visual token 011696|><|visual token 085025|><|visual token 113811|><|visual token 007074|><|visual token 007181|><|visual token 052226|><|visual token 003924|><|visual token 010909|><|visual token 005320|><|visual token 034283|><|visual token 093280|><|visual token 023238|><|visual token 107223|><|visual token 043093|><|visual token 127621|><|visual token 016676|><|visual token 005341|><|visual token 030825|><|visual token 083751|><|visual token 129201|><|visual token 092151|><|visual token 086709|><|visual token 049626|><|extra_200|><|visual token 091453|><|visual token 122966|><|visual token 008789|><|visual token 011685|><|visual token 085390|><|visual token 003363|><|visual token 113709|><|visual token 089540|><|visual token 116655|><|visual token 126949|><|visual token 088383|><|visual token 047340|><|visual token 093513|><|visual token 089451|><|visual token 024954|><|visual token 056544|><|visual token 012192|><|visual token 117351|><|visual token 025497|><|visual token 076978|><|visual token 084895|><|visual token 028806|><|visual token 101388|><|visual token 115604|><|visual token 032824|><|visual token 061261|><|visual token 024566|><|visual token 023328|><|visual token 094121|><|visual token 125960|><|visual token 052050|><|visual token 079481|><|visual token 040903|><|extra_200|><|visual token 054647|><|visual token 002085|><|visual token 034346|><|visual token 072568|><|visual token 113298|><|visual token 020531|><|visual token 044509|><|visual token 020157|><|visual token 099071|><|visual token 027184|><|visual token 016671|><|visual token 005805|><|visual token 049967|><|visual token 084639|><|visual token 114958|><|visual token 088452|><|visual token 017474|><|visual token 053651|><|visual token 124628|><|visual token 122244|><|visual token 050255|><|visual token 025722|><|visual token 040582|><|visual token 118847|><|visual token 117361|><|visual token 117361|><|visual token 097612|><|visual token 129297|><|visual token 072301|><|visual token 000507|><|visual token 036227|><|visual token 071237|><|extra_200|><|visual token 091453|><|visual token 112987|><|visual token 117333|><|visual token 097557|><|visual token 127303|><|visual token 071333|><|visual token 013374|><|visual token 037531|><|visual token 116890|><|visual token 125161|><|visual token 085774|><|visual token 002782|><|visual token 094411|><|visual token 001148|><|visual token 091632|><|visual token 020705|><|visual token 040298|><|visual token 035422|><|visual token 013470|><|visual token 064495|><|visual token 034857|><|visual token 061225|><|visual token 020553|><|visual token 008909|><|visual token 086931|><|visual token 029332|><|visual token 051005|><|visual token 060673|><|visual token 003568|><|visual token 005607|><|visual token 128693|><|visual token 076188|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 053011|><|visual token 029287|><|visual token 094426|><|visual token 012700|><|visual token 046177|><|visual token 098818|><|visual token 045480|><|visual token 121433|><|visual token 033201|><|visual token 087755|><|visual token 024068|><|visual token 086200|><|visual token 042894|><|visual token 122835|><|visual token 127090|><|visual token 036873|><|visual token 031326|><|visual token 029480|><|visual token 059289|><|visual token 110754|><|visual token 120203|><|visual token 086961|><|visual token 090932|><|visual token 048143|><|visual token 071668|><|visual token 120781|><|visual token 034480|><|visual token 125622|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 028364|><|visual token 107061|><|visual token 088837|><|visual token 095967|><|visual token 020662|><|visual token 101680|><|visual token 053261|><|visual token 119823|><|visual token 075975|><|visual token 075451|><|visual token 049623|><|visual token 055170|><|visual token 083382|><|visual token 025744|><|visual token 109231|><|visual token 083032|><|visual token 083353|><|visual token 099171|><|visual token 009940|><|visual token 008475|><|visual token 123242|><|visual token 003331|><|visual token 001029|><|visual token 027173|><|visual token 038852|><|visual token 050795|><|visual token 006809|><|visual token 108400|><|visual token 057643|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 072117|><|visual token 084150|><|visual token 086938|><|visual token 088455|><|visual token 113298|><|visual token 049920|><|visual token 034685|><|visual token 028846|><|visual token 055144|><|visual token 086127|><|visual token 085810|><|visual token 091754|><|visual token 034199|><|visual token 046620|><|visual token 053057|><|visual token 113653|><|visual token 096329|><|visual token 114214|><|visual token 115397|><|visual token 057015|><|visual token 026460|><|visual token 001634|><|visual token 060326|><|visual token 080472|><|visual token 031899|><|visual token 070916|><|visual token 055388|><|visual token 113479|><|visual token 028206|><|visual token 043498|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 042624|><|visual token 063812|><|visual token 046991|><|visual token 093490|><|visual token 038890|><|visual token 038034|><|visual token 129160|><|visual token 020196|><|visual token 059151|><|visual token 068811|><|visual token 044054|><|visual token 090537|><|visual token 054861|><|visual token 009347|><|visual token 028761|><|visual token 052720|><|visual token 111678|><|visual token 008928|><|visual token 123598|><|visual token 079481|><|visual token 032512|><|visual token 021168|><|visual token 103928|><|visual token 100546|><|visual token 067189|><|visual token 016996|><|visual token 110754|><|visual token 057822|><|visual token 000788|><|visual token 070604|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 113177|><|visual token 072580|><|visual token 003704|><|visual token 095103|><|visual token 067305|><|visual token 062970|><|visual token 058828|><|visual token 037420|><|visual token 094946|><|visual token 001132|><|visual token 092838|><|visual token 127763|><|visual token 011762|><|visual token 102908|><|visual token 011858|><|visual token 035411|><|visual token 054599|><|visual token 000250|><|visual token 126961|><|visual token 089799|><|visual token 057283|><|visual token 108457|><|visual token 009932|><|visual token 060326|><|visual token 108979|><|visual token 092660|><|visual token 083781|><|visual token 119482|><|visual token 126758|><|visual token 010801|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 072117|><|visual token 089507|><|visual token 007425|><|visual token 126247|><|visual token 038034|><|visual token 063812|><|visual token 029822|><|visual token 046034|><|visual token 048667|><|visual token 082648|><|visual token 042931|><|visual token 009070|><|visual token 032119|><|visual token 043005|><|visual token 027828|><|visual token 112601|><|visual token 056467|><|visual token 023315|><|visual token 101852|><|visual token 117069|><|visual token 106714|><|visual token 062685|><|visual token 113426|><|visual token 091055|><|visual token 070377|><|visual token 091376|><|visual token 068090|><|visual token 007372|><|visual token 043961|><|visual token 019092|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 123094|><|visual token 081044|><|visual token 058339|><|visual token 061530|><|visual token 080218|><|visual token 126612|><|visual token 051586|><|visual token 037198|><|visual token 074054|><|visual token 077607|><|visual token 081148|><|visual token 079090|><|visual token 068895|><|visual token 061932|><|visual token 088971|><|visual token 111583|><|visual token 098415|><|visual token 054338|><|visual token 014672|><|visual token 034750|><|visual token 100637|><|visual token 030775|><|visual token 099599|><|visual token 102436|><|visual token 061249|><|visual token 026780|><|visual token 079457|><|visual token 022865|><|visual token 014297|><|visual token 072813|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 064045|><|visual token 100514|><|visual token 087793|><|visual token 012592|><|visual token 064031|><|visual token 066620|><|visual token 079520|><|visual token 104146|><|visual token 113854|><|visual token 105400|><|visual token 079104|><|visual token 019426|><|visual token 050224|><|visual token 012629|><|visual token 042634|><|visual token 049499|><|visual token 062104|><|visual token 089046|><|visual token 044884|><|visual token 004651|><|visual token 001573|><|visual token 115644|><|visual token 051391|><|visual token 059532|><|visual token 023249|><|visual token 037455|><|visual token 122307|><|visual token 068202|><|visual token 009072|><|visual token 117781|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 073996|><|visual token 118915|><|visual token 049412|><|visual token 116355|><|visual token 013379|><|visual token 119205|><|visual token 019519|><|visual token 039381|><|visual token 107467|><|visual token 020461|><|visual token 064838|><|visual token 020662|><|visual token 116625|><|visual token 039162|><|visual token 052674|><|visual token 030617|><|visual token 041330|><|visual token 097632|><|visual token 043050|><|visual token 000298|><|visual token 107359|><|visual token 029148|><|visual token 030365|><|visual token 066788|><|visual token 006807|><|visual token 128874|><|visual token 073204|><|visual token 112519|><|visual token 036130|><|visual token 041261|><|visual token 079786|><|extra_200|><|visual token 093483|><|visual token 002270|><|visual token 117333|><|visual token 096308|><|visual token 003731|><|visual token 089192|><|visual token 071753|><|visual token 116261|><|visual token 031385|><|visual token 072176|><|visual token 035682|><|visual token 086241|><|visual token 080813|><|visual token 099727|><|visual token 066559|><|visual token 120554|><|visual token 020097|><|visual token 053086|><|visual token 010764|><|visual token 080662|><|visual token 129490|><|visual token 082016|><|visual token 113226|><|visual token 086334|><|visual token 063383|><|visual token 102631|><|visual token 113716|><|visual token 015677|><|visual token 071389|><|visual token 019235|><|visual token 064563|><|visual token 023363|><|visual token 007879|><|extra_200|><|visual token 128739|><|visual token 073293|><|visual token 088941|><|visual token 102263|><|visual token 087833|><|visual token 026883|><|visual token 071963|><|visual token 050099|><|visual token 075549|><|visual token 102128|><|visual token 122250|><|visual token 096143|><|visual token 062266|><|visual token 060827|><|visual token 008140|><|visual token 049425|><|visual token 087326|><|visual token 127290|><|visual token 123450|><|visual token 012278|><|visual token 092660|><|visual token 049457|><|visual token 074448|><|visual token 107425|><|visual token 001386|><|visual token 063899|><|visual token 022136|><|visual token 102724|><|visual token 040187|><|visual token 120332|><|visual token 120781|><|visual token 097072|><|visual token 001475|><|extra_200|><|visual token 099348|><|visual token 082451|><|visual token 091872|><|visual token 024031|><|visual token 126951|><|visual token 096801|><|visual token 092202|><|visual token 108740|><|visual token 069404|><|visual token 036451|><|visual token 087512|><|visual token 095553|><|visual token 117649|><|visual token 092765|><|visual token 017436|><|visual token 045765|><|visual token 004297|><|visual token 072552|><|visual token 090496|><|visual token 061513|><|visual token 075873|><|visual token 039754|><|visual token 051531|><|visual token 053192|><|visual token 039061|><|visual token 096398|><|visual token 005089|><|visual token 015603|><|visual token 128898|><|visual token 044460|><|visual token 113959|><|visual token 125916|><|visual token 023341|><|extra_200|><|visual token 078134|><|visual token 084303|><|visual token 049524|><|visual token 087866|><|visual token 079452|><|visual token 116261|><|visual token 075454|><|visual token 011202|><|visual token 047617|><|visual token 129851|><|visual token 055778|><|visual token 054180|><|visual token 027816|><|visual token 011770|><|visual token 003508|><|visual token 049908|><|visual token 113177|><|visual token 096143|><|visual token 125751|><|visual token 099562|><|visual token 101485|><|visual token 006979|><|visual token 007395|><|visual token 037553|><|visual token 107482|><|visual token 030083|><|visual token 047088|><|visual token 129886|><|visual token 091581|><|visual token 072066|><|visual token 042123|><|visual token 063008|><|visual token 027740|><|extra_200|><|visual token 029537|><|visual token 089635|><|visual token 051176|><|visual token 034756|><|visual token 051235|><|visual token 118196|><|visual token 055554|><|visual token 050027|><|visual token 074736|><|visual token 045260|><|visual token 096215|><|visual token 066449|><|visual token 085651|><|visual token 043190|><|visual token 006406|><|visual token 085678|><|visual token 111102|><|visual token 002742|><|visual token 002581|><|visual token 035184|><|visual token 124852|><|visual token 057948|><|visual token 041665|><|visual token 055849|><|visual token 059182|><|visual token 112139|><|visual token 002377|><|visual token 116103|><|visual token 016559|><|visual token 004247|><|visual token 014050|><|visual token 108505|><|visual token 083295|><|extra_200|><|visual token 005874|><|visual token 057124|><|visual token 008148|><|visual token 111606|><|visual token 037129|><|visual token 127451|><|visual token 092160|><|visual token 077824|><|visual token 056098|><|visual token 009013|><|visual token 013597|><|visual token 114162|><|visual token 085774|><|visual token 112818|><|visual token 031040|><|visual token 020157|><|visual token 126204|><|visual token 119691|><|visual token 086458|><|visual token 008407|><|visual token 120517|><|visual token 129594|><|visual token 079168|><|visual token 058477|><|visual token 035269|><|visual token 064499|><|visual token 011690|><|visual token 029562|><|visual token 054197|><|visual token 023249|><|visual token 061577|><|visual token 128610|><|visual token 067020|><|extra_200|><|visual token 028244|><|visual token 071157|><|visual token 080068|><|visual token 017524|><|visual token 014494|><|visual token 110479|><|visual token 068091|><|visual token 112127|><|visual token 120126|><|visual token 118794|><|visual token 025076|><|visual token 112490|><|visual token 071436|><|visual token 073931|><|visual token 003000|><|visual token 025436|><|visual token 042007|><|visual token 113977|><|visual token 054847|><|visual token 023651|><|visual token 034812|><|visual token 113517|><|visual token 105598|><|visual token 045531|><|visual token 048309|><|visual token 054094|><|visual token 087578|><|visual token 015484|><|visual token 091003|><|visual token 013345|><|visual token 070649|><|visual token 018584|><|visual token 084688|><|extra_200|><|visual token 101617|><|visual token 112302|><|visual token 086429|><|visual token 110642|><|visual token 054071|><|visual token 026568|><|visual token 013538|><|visual token 070594|><|visual token 006393|><|visual token 041721|><|visual token 114556|><|visual token 078373|><|visual token 025441|><|visual token 121317|><|visual token 051104|><|visual token 024929|><|visual token 033085|><|visual token 019461|><|visual token 044387|><|visual token 005027|><|visual token 033371|><|visual token 009357|><|visual token 071370|><|visual token 012617|><|visual token 065495|><|visual token 113613|><|visual token 115192|><|visual token 041977|><|visual token 078360|><|visual token 051839|><|visual token 119853|><|visual token 012629|><|visual token 121432|><|extra_200|><|visual token 074243|><|visual token 078082|><|visual token 031353|><|visual token 094654|><|visual token 009266|><|visual token 050955|><|visual token 030295|><|visual token 089301|><|visual token 030295|><|visual token 022578|><|visual token 128850|><|visual token 104018|><|visual token 065111|><|visual token 101022|><|visual token 098692|><|visual token 068564|><|visual token 042624|><|visual token 033720|><|visual token 045333|><|visual token 011231|><|visual token 118542|><|visual token 100499|><|visual token 084524|><|visual token 095390|><|visual token 028970|><|visual token 106997|><|visual token 047458|><|visual token 020171|><|visual token 005378|><|visual token 094468|><|visual token 129290|><|visual token 123034|><|visual token 005839|><|extra_200|><|visual token 064945|><|visual token 105304|><|visual token 057130|><|visual token 019822|><|visual token 068945|><|visual token 018787|><|visual token 129558|><|visual token 087108|><|visual token 090285|><|visual token 090285|><|visual token 090285|><|visual token 013555|><|visual token 016044|><|visual token 005072|><|visual token 077607|><|visual token 026349|><|visual token 116061|><|visual token 005072|><|visual token 020882|><|visual token 022542|><|visual token 122307|><|visual token 070694|><|visual token 045765|><|visual token 082365|><|visual token 045765|><|visual token 115774|><|visual token 118827|><|visual token 111058|><|visual token 124957|><|visual token 027426|><|visual token 120581|><|visual token 061663|><|visual token 049251|><|image end|><|image start|>32*32<|image token|><|visual token 027913|><|visual token 088473|><|visual token 054821|><|visual token 048384|><|visual token 007808|><|visual token 044303|><|visual token 099858|><|visual token 011332|><|visual token 055887|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 032108|><|visual token 083909|><|visual token 125661|><|visual token 019128|><|visual token 125661|><|visual token 055887|><|visual token 094354|><|visual token 113201|><|visual token 016257|><|visual token 090820|><|extra_200|><|visual token 016747|><|visual token 044285|><|visual token 124640|><|visual token 000176|><|visual token 016365|><|visual token 001345|><|visual token 012484|><|visual token 089162|><|visual token 026492|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 074488|><|visual token 026492|><|visual token 034174|><|visual token 026492|><|visual token 039142|><|visual token 118492|><|visual token 077042|><|visual token 076157|><|visual token 013651|><|extra_200|><|visual token 021708|><|visual token 021115|><|visual token 112970|><|visual token 101487|><|visual token 022384|><|visual token 083324|><|visual token 010331|><|visual token 075986|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 124308|><|visual token 033450|><|visual token 109485|><|visual token 043016|><|visual token 086278|><|visual token 040333|><|visual token 048200|><|visual token 059364|><|visual token 041976|><|extra_200|><|visual token 059965|><|visual token 108993|><|visual token 085877|><|visual token 057360|><|visual token 122584|><|visual token 036690|><|visual token 020548|><|visual token 114447|><|visual token 105859|><|visual token 126749|><|visual token 015914|><|visual token 126749|><|visual token 126452|><|visual token 126452|><|visual token 126452|><|visual token 126749|><|visual token 126749|><|visual token 126452|><|visual token 126452|><|visual token 126749|><|visual token 117050|><|visual token 126452|><|visual token 126452|><|visual token 126452|><|visual token 117050|><|visual token 054014|><|visual token 084885|><|visual token 117050|><|visual token 073538|><|visual token 092708|><|visual token 072417|><|visual token 045597|><|extra_200|><|visual token 118115|><|visual token 121206|><|visual token 032772|><|visual token 017727|><|visual token 117263|><|visual token 073892|><|visual token 041721|><|visual token 097059|><|visual token 114843|><|visual token 048336|><|visual token 110627|><|visual token 031369|><|visual token 081188|><|visual token 013912|><|visual token 065293|><|visual token 033938|><|visual token 121427|><|visual token 091479|><|visual token 065111|><|visual token 090456|><|visual token 055964|><|visual token 055964|><|visual token 055964|><|visual token 010758|><|visual token 077347|><|visual token 030819|><|visual token 066271|><|visual token 097059|><|visual token 123156|><|visual token 081188|><|visual token 013470|><|visual token 092800|><|extra_200|><|visual token 053732|><|visual token 083161|><|visual token 091876|><|visual token 086281|><|visual token 020224|><|visual token 044945|><|visual token 121712|><|visual token 027621|><|visual token 022223|><|visual token 083610|><|visual token 090073|><|visual token 066199|><|visual token 092022|><|visual token 011149|><|visual token 011657|><|visual token 053259|><|visual token 125697|><|visual token 032387|><|visual token 032387|><|visual token 022316|><|visual token 013155|><|visual token 044905|><|visual token 009453|><|visual token 115650|><|visual token 048309|><|visual token 005072|><|visual token 009453|><|visual token 067487|><|visual token 037191|><|visual token 124141|><|visual token 075357|><|visual token 068189|><|extra_200|><|visual token 110776|><|visual token 108618|><|visual token 028407|><|visual token 042847|><|visual token 001649|><|visual token 048776|><|visual token 099873|><|visual token 012161|><|visual token 095566|><|visual token 009607|><|visual token 107963|><|visual token 007408|><|visual token 041787|><|visual token 070360|><|visual token 090183|><|visual token 106929|><|visual token 083202|><|visual token 053073|><|visual token 085430|><|visual token 057036|><|visual token 097382|><|visual token 051415|><|visual token 028536|><|visual token 076335|><|visual token 024501|><|visual token 123641|><|visual token 092362|><|visual token 059681|><|visual token 068278|><|visual token 114290|><|visual token 117859|><|visual token 095413|><|extra_200|><|visual token 004011|><|visual token 090520|><|visual token 086938|><|visual token 128935|><|visual token 070743|><|visual token 093744|><|visual token 017345|><|visual token 111940|><|visual token 008295|><|visual token 046564|><|visual token 039255|><|visual token 120688|><|visual token 106370|><|visual token 058786|><|visual token 007290|><|visual token 001340|><|visual token 019035|><|visual token 109589|><|visual token 029038|><|visual token 062751|><|visual token 042852|><|visual token 121959|><|visual token 041864|><|visual token 069905|><|visual token 069905|><|visual token 054214|><|visual token 016869|><|visual token 076141|><|visual token 032818|><|visual token 017872|><|visual token 074999|><|visual token 033227|><|extra_200|><|visual token 093483|><|visual token 108618|><|visual token 005110|><|visual token 036461|><|visual token 108041|><|visual token 117550|><|visual token 098530|><|visual token 043195|><|visual token 077254|><|visual token 064222|><|visual token 101812|><|visual token 090815|><|visual token 116139|><|visual token 040807|><|visual token 044732|><|visual token 014241|><|visual token 080128|><|visual token 014215|><|visual token 035857|><|visual token 114432|><|visual token 003661|><|visual token 066733|><|visual token 053744|><|visual token 012306|><|visual token 053389|><|visual token 090480|><|visual token 082558|><|visual token 124534|><|visual token 065946|><|visual token 075731|><|visual token 021432|><|visual token 084084|><|visual token 007921|><|extra_200|><|visual token 128739|><|visual token 005001|><|visual token 117401|><|visual token 084901|><|visual token 022547|><|visual token 120134|><|visual token 115125|><|visual token 074927|><|visual token 111101|><|visual token 033119|><|visual token 002676|><|visual token 123067|><|visual token 069337|><|visual token 005089|><|visual token 104326|><|visual token 110829|><|visual token 086379|><|visual token 022034|><|visual token 024511|><|visual token 045228|><|visual token 027515|><|visual token 114138|><|visual token 028334|><|visual token 108007|><|visual token 075756|><|visual token 093072|><|visual token 066136|><|visual token 012381|><|visual token 097748|><|visual token 122492|><|visual token 096703|><|visual token 054331|><|extra_200|><|visual token 128739|><|visual token 002270|><|visual token 073105|><|visual token 040122|><|visual token 087968|><|visual token 109401|><|visual token 085778|><|visual token 041113|><|visual token 115977|><|visual token 077020|><|visual token 031713|><|visual token 042504|><|visual token 017100|><|visual token 020070|><|visual token 103217|><|visual token 064514|><|visual token 074992|><|visual token 059995|><|visual token 110023|><|visual token 057796|><|visual token 110023|><|visual token 023581|><|visual token 063953|><|visual token 019524|><|visual token 024575|><|visual token 018483|><|visual token 101053|><|visual token 064371|><|visual token 033476|><|visual token 086529|><|visual token 058498|><|visual token 122390|><|visual token 104023|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 011775|><|visual token 106909|><|visual token 117966|><|visual token 021374|><|visual token 126912|><|visual token 102092|><|visual token 052721|><|visual token 118589|><|visual token 022913|><|visual token 101314|><|visual token 034689|><|visual token 003434|><|visual token 115248|><|visual token 076701|><|visual token 075826|><|visual token 072227|><|visual token 091981|><|visual token 044496|><|visual token 078904|><|visual token 069919|><|visual token 104627|><|visual token 042136|><|visual token 117178|><|visual token 118102|><|visual token 003356|><|visual token 105458|><|visual token 093498|><|visual token 120734|><|visual token 025870|><|extra_200|><|visual token 091453|><|visual token 112987|><|visual token 117333|><|visual token 048825|><|visual token 066924|><|visual token 125751|><|visual token 095264|><|visual token 036873|><|visual token 098593|><|visual token 073754|><|visual token 020157|><|visual token 009961|><|visual token 035555|><|visual token 076791|><|visual token 120386|><|visual token 113018|><|visual token 016135|><|visual token 024469|><|visual token 130537|><|visual token 012951|><|visual token 102908|><|visual token 111502|><|visual token 060919|><|visual token 040131|><|visual token 093966|><|visual token 058638|><|visual token 040870|><|visual token 028055|><|visual token 000507|><|visual token 082287|><|visual token 021693|><|visual token 064659|><|visual token 044159|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 072117|><|visual token 127303|><|visual token 063930|><|visual token 130827|><|visual token 082175|><|visual token 005059|><|visual token 045890|><|visual token 111266|><|visual token 116399|><|visual token 087752|><|visual token 024001|><|visual token 108054|><|visual token 069223|><|visual token 016869|><|visual token 064037|><|visual token 027910|><|visual token 048348|><|visual token 092289|><|visual token 049251|><|visual token 021231|><|visual token 046458|><|visual token 013067|><|visual token 115607|><|visual token 031771|><|visual token 062639|><|visual token 041019|><|visual token 104065|><|visual token 001599|><|visual token 094990|><|extra_200|><|visual token 091453|><|visual token 002270|><|visual token 117333|><|visual token 005027|><|visual token 027470|><|visual token 099611|><|visual token 108044|><|visual token 096041|><|visual token 020662|><|visual token 096556|><|visual token 068186|><|visual token 015409|><|visual token 081965|><|visual token 110393|><|visual token 087905|><|visual token 062886|><|visual token 073648|><|visual token 034689|><|visual token 075912|><|visual token 064726|><|visual token 069293|><|visual token 071210|><|visual token 107938|><|visual token 064984|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
    
    score, details = format_reward(test_response)
    print(f"Test Format Reward Score: {score}, Details: {details}")