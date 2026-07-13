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

import numpy as np
import torch

from vllm import SamplingParams

@torch.no_grad()
def generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
):

    if getattr(cfg, "streaming", False):
        raise ValueError("Streaming generation is not supported in VLLM yet.")
    else:
        yield non_streaming_generate(
            cfg, model, tokenizer, input_ids, unconditional_ids,
        )


@torch.no_grad()
def generate_batch(
    cfg,
    model,
    tokenizer,
    batch_input_ids,
    batch_unconditional_ids,
):
    """Batch generation - yields results for each input in batch."""
    results = non_streaming_generate_batch(
        cfg, model, tokenizer, batch_input_ids, batch_unconditional_ids,
    )
    return results


def non_streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
):
    # Check if CFG is enabled (guidance_scale > 0.0)
    guidance_scale = getattr(cfg, 'classifier_free_guidance', 3.0)
    cfg_enabled = guidance_scale > 0.0
    
    # Build inputs - only include uncond_prompt_token_ids when CFG is enabled
    inputs = {
        "prompt_token_ids": input_ids.tolist()[0],
    }
    
    if cfg_enabled:
        inputs["uncond_prompt_token_ids"] = unconditional_ids.tolist()[0]
    else:
        print(f"[INFO] CFG disabled (guidance_scale={guidance_scale}), using single-request mode for 2x speedup")

    extra_args = {
        "guidance_scale": cfg.classifier_free_guidance,
        "text_top_k": cfg.sampling_params["text_top_k"],
        "text_top_p": cfg.sampling_params["text_top_p"],
        "text_temperature": cfg.sampling_params["text_temperature"],
        "visual_top_k": cfg.sampling_params["image_top_k"],
        "visual_top_p": cfg.sampling_params["image_top_p"],
        "visual_temperature": cfg.sampling_params["image_temperature"],
        "width": getattr(cfg, "target_width", None),
        "height": getattr(cfg, "target_height", None),
        "area": cfg.image_area if getattr(cfg, "target_width", None) else None,
    }
    if cfg.task_type in ["t2i", "x2i"]:
        stop_token_ids = tokenizer.encode("<|image end|>")
    else:
        stop_token_ids = tokenizer.encode("<|extra_204|>")
    sampling_params = SamplingParams(
        top_k=cfg.sampling_params["top_k"],
        top_p=cfg.sampling_params["top_p"],
        temperature=cfg.sampling_params["temperature"],
        max_tokens=cfg.sampling_params["max_new_tokens"],
        detokenize=False,
        extra_args=extra_args,
        stop_token_ids=stop_token_ids,
    )

    results = model.generate(inputs, sampling_params=sampling_params)
    gen_token_ids = np.array(results[0].outputs[0].token_ids)

    return gen_token_ids


def non_streaming_generate_batch(
    cfg,
    model,
    tokenizer,
    batch_input_ids,
    batch_unconditional_ids,
):
    """
    Batch generation for multiple inputs.
    
    Args:
        batch_input_ids: list of input_ids tensors
        batch_unconditional_ids: list of unconditional_ids tensors
    
    Returns:
        list of generated token_ids arrays
    """
    # Check if CFG is enabled (guidance_scale > 0.0)
    guidance_scale = getattr(cfg, 'classifier_free_guidance', 3.0)
    cfg_enabled = guidance_scale > 0.0
    
    if cfg_enabled:
        print(f"[WARNING] Batch generation with CFG enabled is not fully supported. "
              f"Each request will be paired, so effective batch size is halved.")
    
    # Build inputs list with unique request IDs to avoid ID collision in vLLM scheduler
    # CRITICAL: We need to track request_id -> index mapping because vLLM does NOT
    # guarantee that results are returned in the same order as inputs!
    # 
    # NOTE: vLLM's CFG patch converts request_id internally:
    # - Input: "batch_{uuid}_{local_idx}" or just a simple string like "5"
    # - Output: "{cond_flag}{child_idx}_{parent_id}" where cond_flag='0' for conditional
    # - The output_processor uses regex to extract the original request_id
    # 
    # To properly track results, we use simple numeric IDs that we can parse back
    import uuid
    inputs_list = []
    request_id_to_index = {}  # Map request_id to original input index
    batch_uuid = uuid.uuid4().hex[:8]  # Unique batch identifier
    
    for i, (input_ids, unconditional_ids) in enumerate(zip(batch_input_ids, batch_unconditional_ids)):
        # Use a simple numeric request_id that we can parse back from the output
        # Format: "{batch_uuid}_{local_index}" - the local_index is what we need to recover
        unique_id = f"{batch_uuid}_{i}"
        request_id_to_index[unique_id] = i  # Track the mapping
        
        inputs = {
            "prompt_token_ids": input_ids.tolist()[0] if input_ids.dim() > 1 else input_ids.tolist(),
            "request_id": unique_id,
        }
        
        if cfg_enabled:
            uncond = unconditional_ids.tolist()[0] if unconditional_ids.dim() > 1 else unconditional_ids.tolist()
            inputs["uncond_prompt_token_ids"] = uncond
        
        inputs_list.append(inputs)
    
    print(f"[INFO] Batch generation: {len(inputs_list)} requests, CFG={'enabled' if cfg_enabled else 'disabled'}")

    extra_args = {
        "guidance_scale": cfg.classifier_free_guidance,
        "text_top_k": cfg.sampling_params["text_top_k"],
        "text_top_p": cfg.sampling_params["text_top_p"],
        "text_temperature": cfg.sampling_params["text_temperature"],
        "visual_top_k": cfg.sampling_params["image_top_k"],
        "visual_top_p": cfg.sampling_params["image_top_p"],
        "visual_temperature": cfg.sampling_params["image_temperature"],
        "width": getattr(cfg, "target_width", None),
        "height": getattr(cfg, "target_height", None),
        "area": cfg.image_area if getattr(cfg, "target_width", None) else None,
    }
    
    if cfg.task_type in ["t2i", "x2i"]:
        stop_token_ids = tokenizer.encode("<|image end|>")
    else:
        stop_token_ids = tokenizer.encode("<|extra_204|>")
    
    sampling_params = SamplingParams(
        top_k=cfg.sampling_params["top_k"],
        top_p=cfg.sampling_params["top_p"],
        temperature=cfg.sampling_params["temperature"],
        max_tokens=cfg.sampling_params["max_new_tokens"],
        detokenize=False,
        extra_args=extra_args,
        stop_token_ids=stop_token_ids,
    )

    # Batch generate
    results = model.generate(inputs_list, sampling_params=sampling_params)
    
    # DEBUG: Print request_id mapping to verify order
    print(f"[DEBUG] Input request_ids: {[inp.get('request_id', 'N/A') for inp in inputs_list]}")
    print(f"[DEBUG] Output request_ids: {[r.request_id for r in results]}")
    print(f"[DEBUG] Input count: {len(inputs_list)}, Output count: {len(results)}")
    
    # In CFG mode, vLLM's patch transforms request_ids:
    # - Original input: "{batch_uuid}_{local_idx}" (e.g., "abc12345_0")
    # - vLLM creates parent: "cfg_{batch_uuid}_{local_idx}"
    # - vLLM creates children with get_child_info(idx) which returns "{idx}_cfg_{batch_uuid}_{local_idx}"
    #   where idx=0 for conditional, idx=1 for unconditional
    # - output_processor.get_hybrid_outputs uses: re.sub(r'_cfg_', '', child_request_id)
    #   which transforms "0_cfg_abc12345_0" -> "0_abc12345_0"
    # 
    # So the final output request_id format is: "{cond_flag}_{batch_uuid}_{local_idx}"
    # We need to extract local_idx and cond_flag to match results back to inputs
    
    if cfg_enabled:
        # Parse and reorder results
        conditional_results = {}
        
        for result in results:
            req_id = result.request_id
            # Expected format: "{cond_flag}_{batch_uuid}_{local_idx}" (e.g., "0_abc12345_5")
            # or legacy format: "{cond_flag}{global_idx}" (e.g., "05", "064")
            
            parts = req_id.split('_')
            
            if len(parts) >= 3:
                # New format: "{cond_flag}_{batch_uuid}_{local_idx}"
                try:
                    cond_flag = parts[0]
                    local_idx = int(parts[-1])  # Last part is local index
                    is_conditional = (cond_flag == '0')
                    
                    if is_conditional:
                        if local_idx not in conditional_results:
                            conditional_results[local_idx] = np.array(result.outputs[0].token_ids)
                        else:
                            print(f"[WARNING] Duplicate conditional result for local index {local_idx}")
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Cannot parse request_id (new format): {req_id}, error: {e}")
            elif len(req_id) >= 2:
                # Legacy format: "{cond_flag}{index}" (e.g., "00", "064")
                # cond_flag: '0' for conditional, '1' for unconditional (FIRST character)
                # index: remaining chars are the original index
                cond_flag = req_id[0]
                index_str = req_id[1:]
                try:
                    original_idx = int(index_str)
                    is_conditional = (cond_flag == '0')
                    if is_conditional:
                        # For legacy format, we need to convert global index to local index
                        # This requires knowing the batch offset, which we don't have directly
                        # So we store using the global index and will handle mapping later
                        if original_idx not in conditional_results:
                            conditional_results[original_idx] = np.array(result.outputs[0].token_ids)
                        else:
                            print(f"[WARNING] Duplicate conditional result for index {original_idx}")
                except ValueError:
                    print(f"[WARNING] Cannot parse index from request_id (legacy format): {req_id}")
            else:
                print(f"[WARNING] Unexpected request_id format: {req_id}")
        
        # Build ordered result list
        gen_token_ids_list = []
        
        # Try to determine if we're using new format (keys are 0, 1, 2, ...) or legacy format
        keys = sorted(conditional_results.keys())
        if keys and keys[0] >= len(inputs_list):
            # Legacy format with global indices - need to remap
            print(f"[DEBUG] Detected legacy format with global indices: {keys}")
            min_key = min(keys)
            for i in range(len(inputs_list)):
                global_idx = min_key + i
                if global_idx in conditional_results:
                    gen_token_ids_list.append(conditional_results[global_idx])
                else:
                    print(f"[ERROR] Missing conditional result for global index {global_idx} (local {i})")
                    gen_token_ids_list.append(np.array([]))  # placeholder
        else:
            # New format with local indices (0, 1, 2, ...)
            for i in range(len(inputs_list)):
                if i in conditional_results:
                    gen_token_ids_list.append(conditional_results[i])
                else:
                    print(f"[ERROR] Missing conditional result for local index {i}")
                    gen_token_ids_list.append(np.array([]))  # placeholder
        
        print(f"[DEBUG] Extracted {len(conditional_results)} conditional results for {len(inputs_list)} inputs")
    else:
        # Non-CFG mode: results should be in order
        gen_token_ids_list = []
        for result in results:
            gen_token_ids = np.array(result.outputs[0].token_ids)
            gen_token_ids_list.append(gen_token_ids)
    
    if len(gen_token_ids_list) != len(inputs_list):
        print(f"[WARNING] Final results count ({len(gen_token_ids_list)}) != input count ({len(inputs_list)})")

    return gen_token_ids_list
