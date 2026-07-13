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
This script patches vllm to support:
1. CFG mode (batch_size=2, paired conditional + unconditional requests)
2. No-CFG mode (single request mode for 2x speedup)
3. LoRA support for Emu3.5 model (vocab_size fixes, embedding fixes)

The key changes:
1. Modify ClassifierFreeGuidanceLogitsForVisualTokenProcessor to handle single requests
2. When CFG is disabled (guidance_scale <= 1.0), the processor simply applies format token masking
3. Add SupportsLoRA interface to Emu3_5ForCausalLM
4. Raise vocab_size limit from 257024 to 512000 for Emu3.5 (vocab_size=282926)
5. Add lora_extra_vocab_size=0 support for models that don't need extra vocab
6. Add conditional checks for embeddings_tensors.shape[1] > 0

Usage:
    python src/patch/apply_no_cfg_support.py              # Apply all patches
    python src/patch/apply_no_cfg_support.py --lora-only  # Apply only LoRA patches
    python src/patch/apply_no_cfg_support.py --revert     # Revert all patches
"""

import argparse
import shutil
import sys
from pathlib import Path


def get_vllm_site():
    try:
        import vllm
        return Path(vllm.__file__).parent
    except ImportError:
        print("[ERROR] vllm is not installed")
        sys.exit(1)


def backup_file(file_path: Path, backup_dir: Path):
    """Backup a file before modifying it."""
    backup_path = backup_dir / file_path.name
    if file_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"[INFO] Backed up {file_path} to {backup_path}")
    return backup_path


def restore_file(file_path: Path, backup_dir: Path):
    """Restore a file from backup."""
    backup_path = backup_dir / file_path.name
    if backup_path.exists():
        shutil.copy2(backup_path, file_path)
        print(f"[INFO] Restored {file_path} from {backup_path}")
    else:
        print(f"[WARN] No backup found for {file_path}")


def patch_logits_processor(builtin_path: Path):
    """Patch the ClassifierFreeGuidanceLogitsForVisualTokenProcessor to support no-CFG mode and robust CFG pairing."""
    
    content = builtin_path.read_text()
    
    # Check if already fully patched
    if "# NO_CFG_SUPPORT_PATCHED" in content and "# CFG_PAIRING_FIX" in content:
        print("[INFO] builtin.py already fully patched (no-CFG support + CFG pairing fix)")
        return False
    
    patched = False
    
    # Patch 1: Add _cfg_pairs to __init__
    old_init = '''        self.metadata: dict[int, HybridSchedulerMetadata] = {}
        self.guidance_scale: dict[int, int] = {}
        self.activated = False'''
    
    new_init = '''        self.metadata: dict[int, HybridSchedulerMetadata] = {}
        self.guidance_scale: dict[int, int] = {}
        self.activated = False
        # CFG_PAIRING_FIX: Cache for quick pair lookup - pair_id -> {True: cond_idx, False: uncond_idx}
        self._cfg_pairs: dict[int, dict[bool, int]] = {}'''
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        patched = True
        print("[INFO] Patched __init__ with _cfg_pairs")
    
    # Patch 2: Update update_state to maintain _cfg_pairs cache
    # NOTE: The current installed code has:
    #   guidance_scale = params.extra_args.get("guidance_scale", None)
    #   if metadata is None or guidance_scale is None:
    # The reference code changes this to check metadata first, then get guidance_scale safely
    old_update_state = '''    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        for index, params, _, _, metadata in batch_update.added:
            guidance_scale = params.extra_args.get("guidance_scale", None)
            if metadata is None or guidance_scale is None:
                continue
            self.activated = True
            self.metadata[index] = metadata
            self.guidance_scale[index] = guidance_scale

        if self.activated is True:
            for index in batch_update.removed:
                self.metadata.pop(index, None)
                self.guidance_scale.pop(index, None)

            for i1, i2, direct in batch_update.moved:
                if direct == MoveDirectionality.SWAP:
                    self.metadata[i1], self.metadata[i2] = self.metadata[i2], self.metadata[i1]
                    self.guidance_scale[i1], self.guidance_scale[i2] = \\
                        self.guidance_scale[i2], self.guidance_scale[i1]
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    self.metadata[i2] = self.metadata.pop(i1, None)
                    self.guidance_scale[i2] = self.guidance_scale.pop(i1, 1.0)

            for index, params, metadata in batch_update.updated:
                self.metadata[index] = metadata'''
    
    new_update_state = '''    def update_state(self, batch_update: Optional[BatchUpdate]):
        # NO_CFG_SUPPORT_PATCHED: Simplified update_state with _cfg_pairs rebuild
        # CFG_PAIRING_FIX_V2: Rebuild _cfg_pairs from metadata to prevent stale entries
        # Root cause: When removed batch indices are reused by _register_add_request's
        # pop_removed(), those indices are absent from BatchUpdate.removed, so the old
        # pair's _cfg_pairs entry is never cleaned up. This causes CFG guidance to be
        # applied multiple times to the same logits position.
        if not batch_update:
            return

        # Step 1: Process removed requests (clean up metadata at freed positions)
        for index in batch_update.removed:
            self.metadata.pop(index, None)
            self.guidance_scale.pop(index, None)

        # Step 2: Process added requests (set metadata at new/reused positions)
        for index, params, _, _, metadata in batch_update.added:
            if metadata is None:
                continue
            guidance_scale = 0
            if params.extra_args is not None:
                guidance_scale = params.extra_args.get("guidance_scale", 0)
            self.activated = True
            self.metadata[index] = metadata
            self.guidance_scale[index] = guidance_scale

        # Step 3: Process moved requests (relocate metadata within the batch)
        if self.activated:
            for i1, i2, direct in batch_update.moved:
                if direct == MoveDirectionality.SWAP:
                    meta1 = self.metadata.pop(i1, None)
                    meta2 = self.metadata.pop(i2, None)
                    if meta2 is not None:
                        self.metadata[i1] = meta2
                    if meta1 is not None:
                        self.metadata[i2] = meta1
                    gs1 = self.guidance_scale.pop(i1, 1.0)
                    gs2 = self.guidance_scale.pop(i2, 1.0)
                    self.guidance_scale[i1] = gs2
                    self.guidance_scale[i2] = gs1
                elif direct == MoveDirectionality.UNIDIRECTIONAL:
                    self.metadata[i2] = self.metadata.pop(i1, None)
                    self.guidance_scale[i2] = self.guidance_scale.pop(i1, 1.0)

            # Step 4: Process updated requests (refresh metadata for continuing requests)
            for index, params, metadata in batch_update.updated:
                if metadata is not None:
                    self.metadata[index] = metadata
                    if params is not None and params.extra_args is not None:
                        self.guidance_scale[index] = params.extra_args.get("guidance_scale",
                            self.guidance_scale.get(index, 1.0))

        # Step 5: Rebuild _cfg_pairs from current metadata to eliminate stale entries
        if self.activated:
            self._cfg_pairs = {}
            for idx, meta in self.metadata.items():
                if meta is None:
                    continue
                cfg_pair_id = getattr(meta, 'cfg_pair_id', -1)
                if cfg_pair_id >= 0:
                    if cfg_pair_id not in self._cfg_pairs:
                        self._cfg_pairs[cfg_pair_id] = {}
                    is_conditional = getattr(meta, 'is_conditional', True)
                    self._cfg_pairs[cfg_pair_id][is_conditional] = idx'''
    
    if old_update_state in content:
        content = content.replace(old_update_state, new_update_state)
        patched = True
        print("[INFO] Patched update_state with _cfg_pairs maintenance")
    
    # Patch 3: Change is_argmax_invariant from True to False
    old_argmax = '''    def is_argmax_invariant(self) -> bool:
        return True'''
    
    new_argmax = '''    def is_argmax_invariant(self) -> bool:
        return False'''
    
    if old_argmax in content:
        content = content.replace(old_argmax, new_argmax)
        patched = True
        print("[INFO] Patched is_argmax_invariant to return False")
    
    # Patch 4: Find and replace the apply method
    old_apply = '''    def apply(self, logits):

        if not self.metadata:
            return logits

        indices = list(self.metadata.keys())
        if logits.shape[0] != len(indices) or logits.shape[0] % 2 != 0:
            return logits

        for i in range(0, len(indices), 2):
            format_token_ids = self.metadata[i].format_token_ids
            in_visual = self.metadata[i].in_visual
            in_image = self.metadata[i].in_image
            guidance_scale = self.guidance_scale[i]
            if len(format_token_ids) > 0:
                mask = torch.ones_like(logits[i], dtype=torch.bool)
                mask[format_token_ids] = False
                logits[i].masked_fill_(mask, float("-inf"))
            elif in_image and in_visual:
                cond_logits = torch.nn.functional.log_softmax(logits[i], dim=-1)
                uncond_logits = torch.nn.functional.log_softmax(logits[i+1], dim=-1)
                guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                logits[i] = guided_logits
                # logits[i+1] = guided_logits
                mask = torch.ones_like(logits[i], dtype=torch.bool)
                mask[self.visual_token_start_index:] = False
                logits[i].masked_fill_(mask, float("-inf"))
            elif in_image and not in_visual:
                top1_idx = torch.argmax(logits[i], dim=-1, keepdim=True)
                mask = torch.ones_like(logits[i], dtype=torch.bool)
                mask.scatter_(dim=-1, index=top1_idx, value=False)
                logits[i].masked_fill_(mask, float("-inf"))
            elif not in_image and not in_visual:
                mask = torch.ones_like(logits[i], dtype=torch.bool)
                mask[:self.visual_token_start_index] = False
                logits[i].masked_fill_(mask, float("-inf"))

        return logits'''
    
    new_apply = '''    def apply(self, logits):
        # EOL_SAFETY_FIX: Two-step architecture to decouple format_token_ids
        # forcing from CFG pairing. This ensures EOL/EOI tokens are ALWAYS
        # forced regardless of whether the CFG pair is complete.
        #
        # Step 1: Unconditionally apply format_token_ids (EOL/EOI) and
        #         resolution top-1 to ALL requests.
        # Step 2: For requests not handled in Step 1, apply CFG guidance
        #         (if paired) or mode-based masking.

        if not self.metadata:
            return logits

        # ── Step 1: Format forcing (unconditional, safety-critical) ──
        # This step runs for EVERY request that has format_token_ids set,
        # regardless of CFG pairing state. It guarantees that EOL/EOI
        # structural tokens are always produced at the correct positions.
        format_handled = set()

        for idx, meta in self.metadata.items():
            if idx >= logits.shape[0] or meta is None:
                continue

            format_token_ids = meta.format_token_ids

            if len(format_token_ids) > 0:
                # Force EOL / EOI / resolution tokens
                mask = torch.ones_like(logits[idx], dtype=torch.bool)
                mask[format_token_ids] = False
                logits[idx].masked_fill_(mask, float("-inf"))
                format_handled.add(idx)
            elif meta.in_image and not meta.in_visual:
                # Resolution prediction area — force top-1 sampling
                top1_idx = torch.argmax(logits[idx], dim=-1, keepdim=True)
                mask = torch.ones_like(logits[idx], dtype=torch.bool)
                mask.scatter_(dim=-1, index=top1_idx, value=False)
                logits[idx].masked_fill_(mask, float("-inf"))
                format_handled.add(idx)

        # ── Step 2: CFG guidance / mode-based masking ──
        # Only applies to requests NOT already handled in Step 1.
        if self._cfg_pairs:
            # CFG mode: apply classifier-free guidance for complete pairs
            processed_pairs = set()

            for pair_id, pair_dict in self._cfg_pairs.items():
                if pair_id in processed_pairs:
                    continue
                processed_pairs.add(pair_id)

                cond_idx = pair_dict.get(True)    # is_conditional=True
                uncond_idx = pair_dict.get(False)  # is_conditional=False

                # -- Handle conditional request --
                if cond_idx is not None and cond_idx not in format_handled:
                    if cond_idx < logits.shape[0]:
                        cond_meta = self.metadata.get(cond_idx)
                        if cond_meta is not None:
                            # Try CFG guidance if the pair is complete
                            if (uncond_idx is not None
                                    and uncond_idx < logits.shape[0]
                                    and cond_meta.in_image
                                    and cond_meta.in_visual):
                                guidance_scale = self.guidance_scale.get(
                                    cond_idx, 1.0)
                                cond_logits = torch.nn.functional.log_softmax(
                                    logits[cond_idx], dim=-1)
                                uncond_logits = torch.nn.functional.log_softmax(
                                    logits[uncond_idx], dim=-1)
                                guided_logits = (uncond_logits
                                    + guidance_scale
                                    * (cond_logits - uncond_logits))
                                logits[cond_idx] = guided_logits
                                # Mask non-visual tokens
                                mask = torch.ones_like(
                                    logits[cond_idx], dtype=torch.bool)
                                mask[self.visual_token_start_index:] = False
                                logits[cond_idx].masked_fill_(
                                    mask, float("-inf"))
                            else:
                                # Pair incomplete or not in visual area:
                                # fall back to mode-based masking
                                self._apply_mode_mask(logits, cond_idx,
                                                      cond_meta)

                # -- Handle unconditional request (mode mask only, no CFG) --
                if uncond_idx is not None and uncond_idx not in format_handled:
                    if uncond_idx < logits.shape[0]:
                        uncond_meta = self.metadata.get(uncond_idx)
                        if uncond_meta is not None:
                            self._apply_mode_mask(logits, uncond_idx,
                                                  uncond_meta)

            # Handle any requests that are not part of any pair
            paired_indices = set()
            for pair_dict in self._cfg_pairs.values():
                for v in pair_dict.values():
                    paired_indices.add(v)

            for idx, meta in self.metadata.items():
                if idx in paired_indices or idx in format_handled:
                    continue
                if idx >= logits.shape[0] or meta is None:
                    continue
                self._apply_mode_mask(logits, idx, meta)
        else:
            # No-CFG mode: apply mode-based masking only
            for idx, meta in self.metadata.items():
                if idx in format_handled:
                    continue
                if idx >= logits.shape[0] or meta is None:
                    continue
                self._apply_mode_mask(logits, idx, meta)

        return logits

    # ── Helper: mode-based masking (visual / text) ──
    def _apply_mode_mask(self, logits, idx, meta):
        """Apply mode-based logits masking for a single request.

        This is factored out to avoid duplicating the same if/elif chain
        in every branch of apply(). It does NOT touch format_token_ids —
        that is handled exclusively in Step 1 of apply().
        """
        if meta.in_image and meta.in_visual:
            # Visual area: mask everything below visual_token_start_index
            logits[idx] = torch.nn.functional.log_softmax(
                logits[idx], dim=-1)
            mask = torch.ones_like(logits[idx], dtype=torch.bool)
            mask[self.visual_token_start_index:] = False
            logits[idx].masked_fill_(mask, float("-inf"))
        elif not meta.in_image and not meta.in_visual:
            # Text area: mask everything above visual_token_start_index
            mask = torch.ones_like(logits[idx], dtype=torch.bool)
            mask[:self.visual_token_start_index] = False
            logits[idx].masked_fill_(mask, float("-inf"))'''
    
    if old_apply in content:
        content = content.replace(old_apply, new_apply)
        patched = True
        print("[INFO] Patched ClassifierFreeGuidanceLogitsForVisualTokenProcessor.apply()")
    else:
        print("[WARN] Could not find apply method to patch - checking for alternative pattern")
        # Try a more flexible match
        if "logits.shape[0] % 2 != 0:" in content and "for i in range(0, len(indices), 2):" in content:
            print("[ERROR] Found the pattern but couldn't replace. Manual inspection needed.")
            return False
        elif "# CFG_PAIRING_FIX" in content:
            print("[INFO] CFG pairing fix already applied")
            patched = True
        else:
            print("[ERROR] The apply method structure has changed. Manual patching required.")
            return False
    
    if patched:
        builtin_path.write_text(content)
        print(f"[SUCCESS] Patched {builtin_path}")
    return patched


def patch_preprocess(preprocess_path: Path):
    """Patch preprocess.py to handle missing uncond_prompt_token_ids."""
    
    content = preprocess_path.read_text()
    
    # Check if already patched
    if "# NO_CFG_FIX" in content:
        print("[INFO] preprocess.py already patched for no-CFG support")
        return False
    
    old_code = '''        prompt_token_ids = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs)
        uncond_prompt_token_ids = self._truncate_inputs(
            parsed_content["uncond_prompt_token_ids"], tokenization_kwargs)'''

    new_code = '''        prompt_token_ids = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs)
        # NO_CFG_FIX: Handle case where uncond_prompt_token_ids is not provided
        uncond_prompt_token_ids = None
        if parsed_content.get("uncond_prompt_token_ids") is not None:
            uncond_prompt_token_ids = self._truncate_inputs(
                parsed_content["uncond_prompt_token_ids"], tokenization_kwargs)'''

    if old_code in content:
        content = content.replace(old_code, new_code)
        preprocess_path.write_text(content)
        print(f"[SUCCESS] Patched {preprocess_path}")
        return True
    else:
        print("[WARN] Could not find preprocess.py pattern to patch")
        return False


def patch_scheduler(scheduler_path: Path):
    """Patch scheduler.py to add missing sampling_params, hybrid_metadata, and new_token_ids."""
    
    content = scheduler_path.read_text()
    patched = False
    
    # Fix 1: Always populate new_token_ids (CFG-patched gpu_model_runner needs it)
    old_token_ids = '''            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])'''

    new_token_ids = '''            # NO_CFG_FIX: Always populate new_token_ids for CFG-patched gpu_model_runner
            # The patched gpu_model_runner always needs token_ids from scheduler output
            token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                          num_computed_tokens + num_tokens]
            new_token_ids.append(token_ids)'''

    if old_token_ids in content:
        content = content.replace(old_token_ids, new_token_ids)
        patched = True
        print("[INFO] Fixed new_token_ids to always be populated")
    elif "# NO_CFG_FIX: Always populate new_token_ids" in content:
        print("[INFO] new_token_ids fix already applied")
    
    # Fix 2: Add sampling_params and hybrid_metadata to CachedRequestData
    old_code = '''        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )'''

    new_code = '''        # NO_CFG_FIX: Add empty sampling_params and hybrid_metadata for compatibility
        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            sampling_params=[None] * len(req_ids),
            hybrid_metadata=[None] * len(req_ids),
        )'''

    if old_code in content:
        content = content.replace(old_code, new_code)
        patched = True
        print("[INFO] Added sampling_params and hybrid_metadata to CachedRequestData")
    elif "# NO_CFG_FIX: Add empty sampling_params and hybrid_metadata" in content:
        print("[INFO] CachedRequestData fix already applied")
    
    if patched:
        scheduler_path.write_text(content)
        print(f"[SUCCESS] Patched {scheduler_path}")
        return True
    else:
        print("[WARN] Could not find scheduler.py pattern to patch")
        return False


def patch_gpu_model_runner(model_runner_path: Path):
    """Patch gpu_model_runner.py to handle None sampling_params and hybrid_metadata."""
    
    content = model_runner_path.read_text()
    
    # Check if already patched
    if "# NO_CFG_FIX: Only update if sampling_params" in content:
        print("[INFO] gpu_model_runner.py already patched for no-CFG support")
        return False
    
    patched = False
    
    # Fix 1: Add None check before accessing sampling_params attributes
    old_code1 = '''            # NOTE(zhaoyinglia): Update topk/topp/temp, cause different token type needs different value
            self.input_batch.top_k_cpu[req_index] = req_state.sampling_params.top_k
            self.input_batch.top_p_cpu[req_index] = req_state.sampling_params.top_p
            self.input_batch.temperature_cpu[req_index] = req_state.sampling_params.temperature
            self.input_batch.top_k[req_index] = req_state.sampling_params.top_k
            self.input_batch.top_p[req_index] = req_state.sampling_params.top_p
            self.input_batch.temperature[req_index] = req_state.sampling_params.temperature'''
    
    new_code1 = '''            # NOTE(zhaoyinglia): Update topk/topp/temp, cause different token type needs different value
            # NO_CFG_FIX: Only update sampling params if they are not None
            if req_state.sampling_params is not None:
                self.input_batch.top_k_cpu[req_index] = req_state.sampling_params.top_k
                self.input_batch.top_p_cpu[req_index] = req_state.sampling_params.top_p
                self.input_batch.temperature_cpu[req_index] = req_state.sampling_params.temperature
                self.input_batch.top_k[req_index] = req_state.sampling_params.top_k
                self.input_batch.top_p[req_index] = req_state.sampling_params.top_p
                self.input_batch.temperature[req_index] = req_state.sampling_params.temperature'''
    
    if old_code1 in content:
        content = content.replace(old_code1, new_code1)
        patched = True
        print("[INFO] Fixed sampling_params attribute access in gpu_model_runner.py")
    
    # Fix 2: Only update sampling_params/hybrid_metadata if not None
    old_code2 = '''            req_state.sampling_params = req_data.sampling_params[i]
            req_state.hybrid_metadata = req_data.hybrid_metadata[i]'''
    
    new_code2 = '''            # NO_CFG_FIX: Only update if sampling_params and hybrid_metadata are provided
            if req_data.sampling_params[i] is not None:
                req_state.sampling_params = req_data.sampling_params[i]
            if req_data.hybrid_metadata[i] is not None:
                req_state.hybrid_metadata = req_data.hybrid_metadata[i]'''
    
    if old_code2 in content:
        content = content.replace(old_code2, new_code2)
        patched = True
        print("[INFO] Fixed sampling_params/hybrid_metadata assignment in gpu_model_runner.py")
    
    if patched:
        model_runner_path.write_text(content)
        print(f"[SUCCESS] Patched {model_runner_path}")
        return True
    else:
        print("[WARN] Could not find gpu_model_runner.py patterns to patch")
        return False


def patch_batch_manager(batch_manager_path: Path):
    """Patch batch_manager.py to add robust error handling for parse_token_hw and CFG pairing support.
    
    This fixes:
    1. IndexError when boi or soi tokens are not found in the search window
    2. CFG_PAIRING_FIX: Add cfg_pair_id and is_conditional fields to HybridSchedulerMetadata
    """
    
    content = batch_manager_path.read_text()
    
    # Check if already patched
    if "# ROBUST_PARSE_TOKEN_HW_PATCHED" in content and "# CFG_PAIRING_FIX" in content:
        print("[INFO] batch_manager.py already patched for robust parse_token_hw and CFG pairing")
        return False
    
    patched = False
    
    # CFG_PAIRING_FIX Patch 1: Add new fields to HybridSchedulerMetadata
    old_metadata_class = '''class HybridSchedulerMetadata(
        msgspec.Struct,
        tag=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]

    format_token_ids: list[int] = msgspec.field(default_factory=list)
    in_image: bool = False
    in_visual: bool = False
    img_step: int = 0  # for cfg decay'''
    
    new_metadata_class = '''class HybridSchedulerMetadata(
        msgspec.Struct,
        tag=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]

    format_token_ids: list[int] = msgspec.field(default_factory=list)
    in_image: bool = False
    in_visual: bool = False
    img_step: int = 0  # for cfg decay
    # CFG_PAIRING_FIX: Fields for robust CFG pairing
    cfg_pair_id: int = -1  # Same pair_id for conditional and unconditional requests
    is_conditional: bool = True  # True for conditional, False for unconditional'''
    
    if old_metadata_class in content:
        content = content.replace(old_metadata_class, new_metadata_class)
        patched = True
        print("[INFO] Added cfg_pair_id and is_conditional to HybridSchedulerMetadata")
    elif "cfg_pair_id: int = -1" in content:
        print("[INFO] HybridSchedulerMetadata already has CFG pairing fields")
    
    # CFG_PAIRING_FIX Patch 2: Add _cfg_pair_counter and _parent_to_pair_id to __init__
    old_init_end = '''        # request_id -> ...
        self.request_metadata: dict[str, HybridSchedulerMetadata] = {}
        self.soi_token_idx: dict[str, int] = {}
        self.boi_token_idx: dict[str, int] = {}
        self.resolution_token_ids: dict[str, list[int]] = defaultdict(list)'''
    
    new_init_end = '''        # request_id -> ...
        self.request_metadata: dict[str, HybridSchedulerMetadata] = {}
        self.soi_token_idx: dict[str, int] = {}
        self.boi_token_idx: dict[str, int] = {}
        self.resolution_token_ids: dict[str, list[int]] = defaultdict(list)
        # CFG_PAIRING_FIX: Counter for generating unique cfg_pair_id
        self._cfg_pair_counter: int = 0
        # CFG_PAIRING_FIX: Map from parent_id to cfg_pair_id for consistent pairing
        self._parent_to_pair_id: dict[str, int] = {}'''
    
    if old_init_end in content:
        content = content.replace(old_init_end, new_init_end)
        patched = True
        print("[INFO] Added CFG pairing tracking to BatchSchedulerManager.__init__")
    elif "_cfg_pair_counter" in content:
        print("[INFO] BatchSchedulerManager already has CFG pairing tracking")
    
    # CFG_PAIRING_FIX Patch 3: Update set_request_metadata to parse request_id and set cfg_pair_id
    old_set_metadata_start = '''    def set_request_metadata(self, request: Request):
        req_id = request.request_id

        last_token_id = request.all_token_ids[-1]

        req_metadata = self.get_req_metadata(request)
        assert req_metadata is None, f"Request {req_id} already exists in BatchSchedulerManager"

        req_metadata = HybridSchedulerMetadata()
        self.request_metadata[req_id] = req_metadata

        req_metadata.format_token_ids = []
        self.soi_token_idx[req_id] = -1
        self.boi_token_idx[req_id] = -1

        extra_args = request.sampling_params.extra_args'''
    
    new_set_metadata_start = '''    def set_request_metadata(self, request: Request):
        req_id = request.request_id

        last_token_id = request.all_token_ids[-1]

        req_metadata = self.get_req_metadata(request)
        assert req_metadata is None, f"Request {req_id} already exists in BatchSchedulerManager"

        req_metadata = HybridSchedulerMetadata()
        self.request_metadata[req_id] = req_metadata

        req_metadata.format_token_ids = []
        self.soi_token_idx[req_id] = -1
        self.boi_token_idx[req_id] = -1

        # CFG_PAIRING_FIX: Parse request_id to extract cfg_pair_id and is_conditional
        # request_id format: "{index}_{parent_id}" where parent_id = "cfg_{original_id}"
        # index 0 = conditional, index 1 = unconditional
        if "_cfg_" in req_id:
            parts = req_id.split("_", 1)  # Split at first underscore
            if len(parts) >= 2:
                try:
                    child_index = int(parts[0])
                    parent_id = parts[1]  # "cfg_{original_id}"
                    
                    # Get or create cfg_pair_id for this parent
                    if parent_id not in self._parent_to_pair_id:
                        self._parent_to_pair_id[parent_id] = self._cfg_pair_counter
                        self._cfg_pair_counter += 1
                    
                    req_metadata.cfg_pair_id = self._parent_to_pair_id[parent_id]
                    req_metadata.is_conditional = (child_index == 0)
                except (ValueError, IndexError):
                    pass  # Keep default values if parsing fails

        extra_args = request.sampling_params.extra_args'''
    
    if old_set_metadata_start in content:
        content = content.replace(old_set_metadata_start, new_set_metadata_start)
        patched = True
        print("[INFO] Updated set_request_metadata with CFG pairing logic")
    elif "# CFG_PAIRING_FIX: Parse request_id" in content:
        print("[INFO] set_request_metadata already has CFG pairing logic")
    
    # CFG_PAIRING_FIX Patch 4: Update reset_request_metadata to clean up pairing data
    old_reset_metadata = '''    def reset_request_metadata(self, request: str):

        req_id = request.request_id

        self.request_metadata.pop(req_id, None)
        self.soi_token_idx.pop(req_id, None)
        self.boi_token_idx.pop(req_id, None)
        self.resolution_token_ids.pop(req_id, None)'''
    
    new_reset_metadata = '''    def reset_request_metadata(self, request: str):

        req_id = request.request_id

        # CFG_PAIRING_FIX: Clean up parent_to_pair_id when both requests of a pair are done
        metadata = self.request_metadata.get(req_id)
        if metadata:
            cfg_pair_id = getattr(metadata, 'cfg_pair_id', -1)
            if cfg_pair_id >= 0:
                # Find and remove the parent_id mapping if this was the last request in the pair
                parent_id_to_remove = None
                for parent_id, pair_id in self._parent_to_pair_id.items():
                    if pair_id == cfg_pair_id:
                        # Check if there are any other requests with this pair_id
                        has_other = False
                        for other_req_id, other_meta in self.request_metadata.items():
                            if other_req_id != req_id:
                                other_pair_id = getattr(other_meta, 'cfg_pair_id', -1)
                                if other_pair_id == cfg_pair_id:
                                    has_other = True
                                    break
                        if not has_other:
                            parent_id_to_remove = parent_id
                        break
                if parent_id_to_remove:
                    self._parent_to_pair_id.pop(parent_id_to_remove, None)

        self.request_metadata.pop(req_id, None)
        self.soi_token_idx.pop(req_id, None)
        self.boi_token_idx.pop(req_id, None)
        self.resolution_token_ids.pop(req_id, None)'''
    
    if old_reset_metadata in content:
        content = content.replace(old_reset_metadata, new_reset_metadata)
        patched = True
        print("[INFO] Updated reset_request_metadata with CFG pairing cleanup")
    elif "# CFG_PAIRING_FIX: Clean up parent_to_pair_id" in content:
        print("[INFO] reset_request_metadata already has CFG pairing cleanup")
    
    # Patch 5: parse_token_hw function - use larger window and add error handling
    old_parse_token_hw = '''    def parse_token_hw(self, request: Request):

        req_id = request.request_id

        if len(self.resolution_token_ids[req_id]) == 0:
            if len(request.all_token_ids) < 16:
                last_tokens = np.array(request.all_token_ids)
            else:
                last_tokens = np.array(request.all_token_ids[-16:])
            boi_idx = np.where(last_tokens == self.boi)[0][-1] # 151852
            img_idx = np.where(last_tokens == self.soi)[0][-1] # 151851
            self.resolution_token_ids[req_id] = last_tokens[boi_idx+1:img_idx].tolist()
            del last_tokens'''
    
    new_parse_token_hw = '''    def parse_token_hw(self, request: Request):
        # ROBUST_PARSE_TOKEN_HW_PATCHED: Add robust error handling for missing boi/soi tokens

        req_id = request.request_id

        if len(self.resolution_token_ids[req_id]) == 0:
            # Use a larger window to search for boi token, fallback to all tokens if needed
            search_window = 64
            if len(request.all_token_ids) < search_window:
                last_tokens = np.array(request.all_token_ids)
            else:
                last_tokens = np.array(request.all_token_ids[-search_window:])
            
            boi_indices = np.where(last_tokens == self.boi)[0]
            soi_indices = np.where(last_tokens == self.soi)[0]
            
            # Handle case where boi or soi is not found in the window
            if len(boi_indices) == 0 or len(soi_indices) == 0:
                logger.warning(f"Request {req_id}: boi or soi token not found in last {len(last_tokens)} tokens. "
                              f"boi_found={len(boi_indices)}, soi_found={len(soi_indices)}. "
                              f"Using default resolution.")
                # Use a default resolution (e.g., 32*32) when tokens are not found
                default_resolution = "32*32"
                resolution_tokens = []
                for c in default_resolution:
                    if c in self.resolution_map_rev:
                        resolution_tokens.append(self.resolution_map_rev[c])
                self.resolution_token_ids[req_id] = resolution_tokens
            else:
                boi_idx = boi_indices[-1]
                img_idx = soi_indices[-1]
                
                # Validate that boi comes before soi
                if boi_idx >= img_idx:
                    logger.warning(f"Request {req_id}: boi_idx ({boi_idx}) >= soi_idx ({img_idx}), "
                                  f"which is unexpected. Using default resolution.")
                    default_resolution = "32*32"
                    resolution_tokens = []
                    for c in default_resolution:
                        if c in self.resolution_map_rev:
                            resolution_tokens.append(self.resolution_map_rev[c])
                    self.resolution_token_ids[req_id] = resolution_tokens
                else:
                    self.resolution_token_ids[req_id] = last_tokens[boi_idx+1:img_idx].tolist()
            
            del last_tokens'''
    
    if old_parse_token_hw in content:
        content = content.replace(old_parse_token_hw, new_parse_token_hw)
        patched = True
        print("[INFO] Patched parse_token_hw with robust error handling")
    
    # Patch 2: set_request_metadata - replace assert with safeguard
    old_set_metadata_visual = '''        # in visual area of image area
        if req_metadata.in_visual is True:
            h, w = self.parse_token_hw(request)

            assert self.soi_token_idx[req_id] >= 0
            vis_idx = len(request.output_token_ids) - self.soi_token_idx[req_id]
            if vis_idx != 0:
                if (vis_idx + 1) == h * (w + 1): # the previous token of eoi
                    req_metadata.format_token_ids = [self.eoi] # 151853
                elif (vis_idx + 1) % (w + 1) == 0: # the pervious token of eol
                    req_metadata.format_token_ids = [self.eol] # 151846

        # in resolution area of image area
        elif req_metadata.in_image and req_metadata.in_visual is False:
            # 151852
            if len(self.resolution_token_ids[req_id]) > 0:
                assert self.boi_token_idx[req_id] >= 0
                hw_idx = len(request.output_token_ids) - self.boi_token_idx[req_id]
                if hw_idx < len(self.resolution_token_ids[req_id]):
                    req_metadata.format_token_ids = [self.resolution_token_ids[req_id][hw_idx]]
                else:
                    req_metadata.format_token_ids = [self.soi] # 151851'''
    
    new_set_metadata_visual = '''        # in visual area of image area
        if req_metadata.in_visual is True:
            h, w = self.parse_token_hw(request)

            # ROBUST_PARSE_TOKEN_HW_PATCHED: Safeguard - check if soi_token_idx is valid
            if self.soi_token_idx[req_id] < 0:
                logger.warning(f"Request {req_id}: in_visual=True but soi_token_idx is not set during set_request_metadata.")
            else:
                vis_idx = len(request.output_token_ids) - self.soi_token_idx[req_id]
                if vis_idx != 0:
                    if (vis_idx + 1) == h * (w + 1): # the previous token of eoi
                        req_metadata.format_token_ids = [self.eoi] # 151853
                    elif (vis_idx + 1) % (w + 1) == 0: # the pervious token of eol
                        req_metadata.format_token_ids = [self.eol] # 151846

        # in resolution area of image area
        elif req_metadata.in_image and req_metadata.in_visual is False:
            # 151852
            if len(self.resolution_token_ids[req_id]) > 0:
                # ROBUST_PARSE_TOKEN_HW_PATCHED: Safeguard - check if boi_token_idx is valid
                if self.boi_token_idx[req_id] < 0:
                    logger.warning(f"Request {req_id}: in_image=True but boi_token_idx is not set during set_request_metadata.")
                    req_metadata.format_token_ids = [self.soi]
                else:
                    hw_idx = len(request.output_token_ids) - self.boi_token_idx[req_id]
                    if hw_idx < len(self.resolution_token_ids[req_id]):
                        req_metadata.format_token_ids = [self.resolution_token_ids[req_id][hw_idx]]
                    else:
                        req_metadata.format_token_ids = [self.soi] # 151851'''
    
    if old_set_metadata_visual in content:
        content = content.replace(old_set_metadata_visual, new_set_metadata_visual)
        patched = True
        print("[INFO] Patched set_request_metadata with safeguards")
    
    # Patch 3: update_metadata_with_output - replace assert with safeguard
    old_update_visual = '''        if req_metadata.in_visual is True:
            h, w = self.parse_token_hw(request)

            assert self.soi_token_idx[req_id] >= 0
            vis_idx = len(request.output_token_ids) - self.soi_token_idx[req_id]
            if vis_idx != 0:
                if (vis_idx + 1) == h * (w + 1): # the previous token of eoi
                    req_metadata.format_token_ids = [self.eoi] # 151853
                    req_metadata.img_step += 1
                elif (vis_idx + 1) % (w + 1) == 0: # the pervious token of eol
                    req_metadata.format_token_ids = [self.eol] # 151846
                else:
                    req_metadata.format_token_ids = []

        # in resolution area of image area
        elif req_metadata.in_image and req_metadata.in_visual is False:
            # 151852
            if len(self.resolution_token_ids[req_id]) > 0:
                assert self.boi_token_idx[req_id] >= 0
                hw_idx = len(request.output_token_ids) - self.boi_token_idx[req_id]
                if hw_idx < len(self.resolution_token_ids[req_id]):
                    req_metadata.format_token_ids = [self.resolution_token_ids[req_id][hw_idx]]
                else:
                    req_metadata.format_token_ids = [self.soi] # 151851'''
    
    new_update_visual = '''        if req_metadata.in_visual is True:
            h, w = self.parse_token_hw(request)

            # ROBUST_PARSE_TOKEN_HW_PATCHED: Safeguard - check if soi_token_idx is valid
            if self.soi_token_idx[req_id] < 0:
                logger.warning(f"Request {req_id}: in_visual=True but soi_token_idx is not set. Skipping format token calculation.")
                req_metadata.format_token_ids = []
            else:
                vis_idx = len(request.output_token_ids) - self.soi_token_idx[req_id]
                if vis_idx != 0:
                    if (vis_idx + 1) == h * (w + 1): # the previous token of eoi
                        req_metadata.format_token_ids = [self.eoi] # 151853
                        req_metadata.img_step += 1
                    elif (vis_idx + 1) % (w + 1) == 0: # the pervious token of eol
                        req_metadata.format_token_ids = [self.eol] # 151846
                    else:
                        req_metadata.format_token_ids = []

        # in resolution area of image area
        elif req_metadata.in_image and req_metadata.in_visual is False:
            # 151852
            if len(self.resolution_token_ids[req_id]) > 0:
                # ROBUST_PARSE_TOKEN_HW_PATCHED: Safeguard - check if boi_token_idx is valid
                if self.boi_token_idx[req_id] < 0:
                    logger.warning(f"Request {req_id}: in_image=True but boi_token_idx is not set. Using soi as format token.")
                    req_metadata.format_token_ids = [self.soi]
                else:
                    hw_idx = len(request.output_token_ids) - self.boi_token_idx[req_id]
                    if hw_idx < len(self.resolution_token_ids[req_id]):
                        req_metadata.format_token_ids = [self.resolution_token_ids[req_id][hw_idx]]
                    else:
                        req_metadata.format_token_ids = [self.soi] # 151851'''
    
    if old_update_visual in content:
        content = content.replace(old_update_visual, new_update_visual)
        patched = True
        print("[INFO] Patched update_metadata_with_output with safeguards")
    
    if patched:
        batch_manager_path.write_text(content)
        print(f"[SUCCESS] Patched {batch_manager_path}")
        return True
    else:
        print("[WARN] Could not find batch_manager.py patterns to patch (may already be patched or structure changed)")
        return False


def patch_scheduler_metadata_manager(scheduler_path: Path):
    """Patch scheduler.py to add BatchSchedulerManager for dynamic sampling params.
    
    This enables dynamic sampling parameter updates (top_k, top_p, temperature)
    based on token type (visual vs text) even when CFG is disabled.
    """
    
    content = scheduler_path.read_text()
    
    # Check if already patched
    if "# NO_CFG_METADATA_MANAGER" in content:
        print("[INFO] scheduler.py already patched with metadata manager")
        return False
    
    patched = False
    
    # Add import for batch_manager at the top
    old_imports = '''from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)'''
    
    new_imports = '''from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)

# NO_CFG_METADATA_MANAGER: Import BatchSchedulerManager for dynamic sampling params
try:
    from vllm.v1.core.sched.batch_manager import BatchSchedulerManager
    HAS_BATCH_MANAGER = True
except ImportError:
    HAS_BATCH_MANAGER = False
    logger.warning("BatchSchedulerManager not available, dynamic sampling params disabled")'''

    if old_imports in content:
        content = content.replace(old_imports, new_imports)
        patched = True
        print("[INFO] Added batch_manager import")
    
    # Add batch_manager initialization in __init__
    old_init_end = '''        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1'''
    
    new_init_end = '''        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        
        # NO_CFG_METADATA_MANAGER: Initialize batch_manager for dynamic sampling params
        self.batch_manager = None
        if HAS_BATCH_MANAGER and vllm_config.additional_config:
            try:
                self.batch_manager = BatchSchedulerManager(vllm_config)
                logger.info("BatchSchedulerManager initialized for dynamic sampling params")
            except Exception as e:
                logger.warning(f"Failed to initialize BatchSchedulerManager: {e}")'''

    if old_init_end in content:
        content = content.replace(old_init_end, new_init_end)
        patched = True
        print("[INFO] Added batch_manager initialization")
    
    # Patch NewRequestData.from_request to pass hybrid_metadata
    old_new_req_data = '''        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]'''
    
    new_new_req_data = '''        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids(),
                hybrid_metadata=self.batch_manager.get_req_metadata(req) if self.batch_manager else None)
            for req in scheduled_new_reqs
        ]'''
    
    if old_new_req_data in content:
        content = content.replace(old_new_req_data, new_new_req_data)
        patched = True
        print("[INFO] Patched NewRequestData.from_request with hybrid_metadata")
    
    # Modify _make_cached_request_data to include sampling_params
    old_cached_data = '''        # NO_CFG_FIX: Add empty sampling_params and hybrid_metadata for compatibility
        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            sampling_params=[None] * len(req_ids),
            hybrid_metadata=[None] * len(req_ids),
        )'''
    
    new_cached_data = '''        # NO_CFG_METADATA_MANAGER: Get sampling_params and hybrid_metadata from batch_manager
        sampling_params_list = []
        hybrid_metadata_list = []
        for req in itertools.chain(running_reqs, resumed_reqs):
            if self.batch_manager is not None:
                # Update and get dynamic sampling params based on token type
                metadata = self.batch_manager.get_req_metadata(req)
                sampling_params_list.append(req.sampling_params)
                hybrid_metadata_list.append(metadata)
            else:
                sampling_params_list.append(None)
                hybrid_metadata_list.append(None)
        
        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            sampling_params=sampling_params_list,
            hybrid_metadata=hybrid_metadata_list,
        )'''

    if old_cached_data in content:
        content = content.replace(old_cached_data, new_cached_data)
        patched = True
        print("[INFO] Modified _make_cached_request_data")
    
    # Patch add_request
    old_add_request = '''    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)'''
    
    new_add_request = '''    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)
        # NO_CFG_METADATA_MANAGER: Initialize metadata for new request
        if self.batch_manager is not None:
            self.batch_manager.set_request_metadata(request)'''

    if old_add_request in content:
        content = content.replace(old_add_request, new_add_request)
        patched = True
        print("[INFO] Patched add_request")
    
    # Patch update_from_output
    old_update_output = '''            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)'''
    
    new_update_output = '''            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)
            
            # NO_CFG_METADATA_MANAGER: Update metadata after output
            if new_token_ids and self.batch_manager is not None:
                self.batch_manager.update_metadata_with_output(request)'''

    if old_update_output in content:
        content = content.replace(old_update_output, new_update_output)
        patched = True
        print("[INFO] Patched update_from_output")
    
    # Patch _free_request
    old_free = '''    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)'''
    
    new_free = '''    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        # NO_CFG_METADATA_MANAGER: Cleanup metadata
        if self.batch_manager is not None:
            self.batch_manager.reset_request_metadata(request)'''

    if old_free in content:
        content = content.replace(old_free, new_free)
        patched = True
        print("[INFO] Patched _free_request")
    
    if patched:
        scheduler_path.write_text(content)
        print(f"[SUCCESS] Patched {scheduler_path} with metadata manager")
        return True
    else:
        print("[WARN] Could not patch scheduler.py with metadata manager")
        return False


# ==============================================================================
# LoRA Support Patches for Emu3.5
# ==============================================================================

def patch_emu3_5_lora_support(site_dir: Path):
    """
    Patch vllm/model_executor/models/emu3_5.py to add LoRA support.
    
    This adds:
    - SupportsLoRA, SupportsPP interfaces
    - supports_lora class variable
    - embedding_modules and embedding_padding_modules class attributes
    
    This patch handles both:
    1. Original vllm format: class Emu3_5ForCausalLM(nn.Module):
    2. Multimodal format: class Emu3_5ForCausalLM(nn.Module, SupportsMultiModal):
    """
    import re
    
    emu3_path = site_dir / "model_executor" / "models" / "emu3_5.py"
    
    if not emu3_path.exists():
        print(f"[WARN] emu3_5.py not found at {emu3_path}")
        return False
    
    content = emu3_path.read_text()
    
    # Check if already patched
    if "# EMU3_LORA_SUPPORT_PATCHED" in content or "supports_lora: ClassVar[bool] = True" in content:
        print("[INFO] emu3_5.py already patched for LoRA support")
        return False
    
    patched = False
    
    # Patch 1: Add SupportsLoRA import - try multiple formats
    # Format 1: with relative imports
    old_import_relative = """from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)"""
    new_import_relative = """from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
from .interfaces import SupportsLoRA, SupportsPP"""
    
    # Format 2: with SupportsMultiModal already present
    old_import_mm = "from vllm.model_executor.models.interfaces import SupportsMultiModal"
    new_import_mm = "from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsLoRA, SupportsPP"
    
    if old_import_relative in content and "from .interfaces import" not in content:
        content = content.replace(old_import_relative, new_import_relative)
        patched = True
        print("[INFO] Added SupportsLoRA, SupportsPP imports (relative format)")
    elif old_import_mm in content:
        content = content.replace(old_import_mm, new_import_mm)
        patched = True
        print("[INFO] Added SupportsLoRA, SupportsPP imports (SupportsMultiModal format)")
    elif "from .interfaces import" not in content:
        # Fallback: add after logger line
        content = content.replace(
            'logger = init_logger(__name__)',
            'logger = init_logger(__name__)\n\n# EMU3_LORA_SUPPORT: Import LoRA interfaces\nfrom .interfaces import SupportsLoRA, SupportsPP'
        )
        patched = True
        print("[INFO] Added SupportsLoRA, SupportsPP imports (fallback location)")
    
    # Patch 3: Add ClassVar import if not present
    if "ClassVar" not in content:
        # Try to find and patch the typing import line
        # Handle format: from typing import Any, Optional, Union
        typing_import_match = re.search(r'from typing import ([^\n]+)', content)
        if typing_import_match:
            old_typing_import = typing_import_match.group(0)
            existing_imports = typing_import_match.group(1)
            new_typing_import = f"from typing import ClassVar, {existing_imports}"
            content = content.replace(old_typing_import, new_typing_import)
            patched = True
            print("[INFO] Added ClassVar to typing imports")
        else:
            # Fallback: add a separate import line after other imports
            content = content.replace(
                'from itertools import islice',
                'from itertools import islice\nfrom typing import ClassVar'
            )
            patched = True
            print("[INFO] Added ClassVar import (separate line)")

    # Patch 2: Modify class definition to add LoRA support
    # Handle both formats: with and without SupportsMultiModal
    
    # LoRA class definition to inject (without packed_modules_mapping as it already exists)
    lora_class_attrs = '''
    # EMU3_LORA_SUPPORT_PATCHED: Added LoRA support for Emu3.5
    
    supports_lora: ClassVar[bool] = True
    
    # Modules where LoRA can be applied
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    
    # Embedding modules for LoRA (required by SupportsLoRA interface)
    embedding_modules = {
        "model.embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    
    embedding_padding_modules = ["lm_head"]'''
    
    # Format 1: Original vllm format - class Emu3_5ForCausalLM(nn.Module):
    old_class_def_simple = "class Emu3_5ForCausalLM(nn.Module):"
    new_class_def_simple = f"class Emu3_5ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):{lora_class_attrs}"
    
    # Format 2: With SupportsMultiModal
    old_class_def_mm = "class Emu3_5ForCausalLM(nn.Module, SupportsMultiModal):"
    new_class_def_mm = f"class Emu3_5ForCausalLM(nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP):{lora_class_attrs}"
    
    if old_class_def_simple in content:
        content = content.replace(old_class_def_simple, new_class_def_simple)
        patched = True
        print("[INFO] Added LoRA class attributes to Emu3_5ForCausalLM (simple format)")
    elif old_class_def_mm in content:
        content = content.replace(old_class_def_mm, new_class_def_mm)
        patched = True
        print("[INFO] Added LoRA class attributes to Emu3_5ForCausalLM (multimodal format)")
    
    if patched:
        emu3_path.write_text(content)
        print(f"[SUCCESS] Patched {emu3_path} for LoRA support")
        return True
    else:
        print("[WARN] Could not patch emu3_5.py for LoRA support - already patched or format changed")
        return False


def patch_lora_vocab_size_limit(site_dir: Path):
    """
    Patch vllm/lora/layers/logits_processor.py to increase vocab_size limit.
    
    Emu3.5 has vocab_size=282926 which exceeds the default limit of 257024.
    This raises the limit to 512000.
    """
    logits_path = site_dir / "lora" / "layers" / "logits_processor.py"
    
    if not logits_path.exists():
        print(f"[WARN] logits_processor.py not found at {logits_path}")
        return False
    
    content = logits_path.read_text()
    
    # Check if already patched
    if "# LORA_VOCAB_SIZE_LIMIT_PATCHED" in content:
        print("[INFO] logits_processor.py already patched for vocab_size limit")
        return False
    
    patched = False
    
    # Patch 1: Increase vocab_size limit from 257024 to 512000
    # Look for the assertion that limits vocab_size
    old_limit_check = "257024"
    new_limit_check = "512000"
    
    if old_limit_check in content:
        content = content.replace(old_limit_check, new_limit_check)
        patched = True
        print("[INFO] Increased vocab_size limit from 257024 to 512000")
    
    # Add LORA_VOCAB_SIZE_LIMIT_PATCHED marker
    if "LORA_VOCAB_SIZE_LIMIT_PATCHED" not in content and patched:
        # Add marker comment at the top of the file after imports
        content = "# LORA_VOCAB_SIZE_LIMIT_PATCHED\n" + content
    
    if patched:
        logits_path.write_text(content)
        print(f"[SUCCESS] Patched {logits_path}")
        return True
    else:
        print("[WARN] Could not patch logits_processor.py - already patched or format changed")
        return False


def patch_vocab_parallel_embedding(site_dir: Path):
    """
    Patch vllm/lora/layers/vocal_parallel_embedding.py for lora_extra_vocab_size=0 support.
    
    Note: The file is named 'vocal_parallel_embedding.py' (not 'vocab_') in this version of vllm.
    When lora_extra_vocab_size=0, embeddings_tensors.shape[1] is 0, causing issues.
    This adds a conditional check to skip the operation.
    """
    # Try the actual filename first (vocal_ is correct for this vllm version)
    embed_path = site_dir / "lora" / "layers" / "vocal_parallel_embedding.py"
    if not embed_path.exists():
        # Fallback to the expected name
        embed_path = site_dir / "lora" / "layers" / "vocab_parallel_embedding.py"
    
    if not embed_path.exists():
        print(f"[WARN] vocal/vocab_parallel_embedding.py not found")
        return False
    
    content = embed_path.read_text()
    
    # Check if already patched
    if "# LORA_EMBED_ZERO_VOCAB_PATCHED" in content:
        print("[INFO] vocab_parallel_embedding.py already patched")
        return False
    
    patched = False
    
    # Look for the embeddings_tensors slicing operation
    old_slice = "embeddings_tensors = embeddings_tensors[:, :inputs_embeds.shape[0]]"
    new_slice = """# LORA_EMBED_ZERO_VOCAB_PATCHED: Skip if lora_extra_vocab_size=0
        if embeddings_tensors.shape[1] > 0:
            embeddings_tensors = embeddings_tensors[:, :inputs_embeds.shape[0]]"""
    
    if old_slice in content:
        content = content.replace(old_slice, new_slice)
        patched = True
        print("[INFO] Added conditional check for embeddings_tensors slicing")
    
    # Also check for the addition operation
    old_add = "inputs_embeds[added_tokens_mask] += embeddings_tensors"
    new_add = """if embeddings_tensors.shape[1] > 0:
            inputs_embeds[added_tokens_mask] += embeddings_tensors"""
    
    if old_add in content:
        if "if embeddings_tensors.shape[1] > 0:\n            inputs_embeds[added_tokens_mask]" not in content:
            content = content.replace(old_add, new_add)
            patched = True
            print("[INFO] Added conditional for embeddings addition")
    
    if patched:
        embed_path.write_text(content)
        print(f"[SUCCESS] Patched {embed_path}")
        return True
    else:
        print("[WARN] Could not patch vocab_parallel_embedding.py - already patched or format changed")
        return False


def patch_lora_config_zero_vocab(site_dir: Path):
    """
    Patch vllm LoRA config to allow lora_extra_vocab_size=0.
    
    Supports both old layout (config.py) and new layout (config/lora.py).
    The original code only allows values (256, 512) but the validation
    might fail for 0. This ensures 0 is explicitly allowed.
    """
    import re
    
    # Try new layout first (config/lora.py), then fall back to old layout (config.py)
    config_paths = [
        site_dir / "config" / "lora.py",  # New vLLM layout
        site_dir / "config.py",            # Old vLLM layout
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path is None:
        print(f"[WARN] LoRA config file not found in any of: {config_paths}")
        return False
    
    print(f"[INFO] Found LoRA config at: {config_path}")
    content = config_path.read_text()
    
    # Check if already patched
    if "LORA_ZERO_EXTRA_VOCAB_PATCHED" in content:
        print(f"[INFO] {config_path.name} already patched for lora_extra_vocab_size=0")
        return False
    
    patched = False
    
    # Pattern to find the possible_lora_extra_vocab_size assignment
    # Supports both tuple format (256, 512) and list format [256, 512]
    # Tuple pattern
    tuple_pattern = r'(possible_lora_extra_vocab_size\s*=\s*\()([^\)]+)(\))'
    tuple_match = re.search(tuple_pattern, content)
    
    # List pattern (fallback for older versions)
    list_pattern = r'(possible_lora_extra_vocab_size\s*=\s*\[)([^\]]+)(\])'
    list_match = re.search(list_pattern, content)
    
    if tuple_match:
        values_str = tuple_match.group(2)
        # Check if 0 is already in the tuple
        values = [v.strip() for v in values_str.split(',')]
        if '0' not in values:
            # Add 0 at the beginning
            new_values = '0, ' + values_str
            replacement = tuple_match.group(1) + new_values + tuple_match.group(3) + '  # LORA_ZERO_EXTRA_VOCAB_PATCHED'
            content = content[:tuple_match.start()] + replacement + content[tuple_match.end():]
            patched = True
            print("[INFO] Added 0 to possible_lora_extra_vocab_size tuple")
    elif list_match:
        values_str = list_match.group(2)
        # Check if 0 is already in the list
        values = [v.strip() for v in values_str.split(',')]
        if '0' not in values:
            # Add 0 at the beginning
            new_values = '0, ' + values_str
            replacement = list_match.group(1) + new_values + list_match.group(3) + '  # LORA_ZERO_EXTRA_VOCAB_PATCHED'
            content = content[:list_match.start()] + replacement + content[list_match.end():]
            patched = True
            print("[INFO] Added 0 to possible_lora_extra_vocab_size list")
    else:
        print("[WARN] Could not find possible_lora_extra_vocab_size definition")
    
    if patched:
        config_path.write_text(content)
        print(f"[SUCCESS] Patched {config_path}")
        return True
    else:
        print(f"[INFO] {config_path.name} may already support lora_extra_vocab_size=0 or format changed")
        return False


def apply_lora_patches(site_dir: Path, backup_dir: Path):
    """Apply all LoRA support patches."""
    print("\n" + "="*60)
    print("Applying LoRA Support Patches for Emu3.5")
    print("="*60 + "\n")
    
    # Paths for LoRA patches
    emu3_path = site_dir / "model_executor" / "models" / "emu3_5.py"
    logits_path = site_dir / "lora" / "layers" / "logits_processor.py"
    # Note: file is named vocal_ not vocab_ in this vllm version
    embed_path = site_dir / "lora" / "layers" / "vocal_parallel_embedding.py"
    if not embed_path.exists():
        embed_path = site_dir / "lora" / "layers" / "vocab_parallel_embedding.py"
    # Support both old and new config layouts
    config_path_new = site_dir / "config" / "lora.py"  # New vLLM layout
    config_path_old = site_dir / "config.py"            # Old vLLM layout
    
    # Backup files
    for path in [emu3_path, logits_path, embed_path]:
        if path.exists():
            backup_file(path, backup_dir)
    # Backup whichever config file exists
    if config_path_new.exists():
        backup_file(config_path_new, backup_dir)
    elif config_path_old.exists():
        backup_file(config_path_old, backup_dir)
    
    # Apply patches
    results = []
    results.append(("emu3_5.py (LoRA interface)", patch_emu3_5_lora_support(site_dir)))
    results.append(("logits_processor.py (vocab_size limit)", patch_lora_vocab_size_limit(site_dir)))
    results.append(("vocal_parallel_embedding.py (zero vocab)", patch_vocab_parallel_embedding(site_dir)))
    results.append(("lora config (zero extra vocab)", patch_lora_config_zero_vocab(site_dir)))
    
    # Summary
    print("\n" + "-"*60)
    print("LoRA Patch Summary:")
    for name, success in results:
        status = "✓ Applied" if success else "○ Skipped/Already applied"
        print(f"  {status}: {name}")
    print("-"*60 + "\n")
    
    return any(r[1] for r in results)



def patch_batch_scheduler(batch_scheduler_path: Path):
    """Patch batch_scheduler.py to fix multiple CFG-related bugs:

    1. batch_valid==1 branch variable name bug (request -> req)
    2. Preemption path missing reset_request_metadata, causing
       'already exists in BatchSchedulerManager' assertion error
    3. PREEMPTED requests not recognized as valid in WAITING scheduling
    4. can_schedule flag incorrectly reset to True when second request in
       batch succeeds after first fails (KV allocation)
    5. Preemption targeting wrong requests (running.pop() pops tail instead
       of the current batch that failed to allocate)
    """
    content = batch_scheduler_path.read_text()

    # Check if already patched
    if "# CFG_PAIRING_FIX" in content:
        print("[INFO] batch_scheduler.py already patched")
        return False

    patched = False

    # --- Patch 1: Fix batch_valid==1 variable name bug ---
    old_batch_valid_1 = """                if batch_valid == 1:
                    for req in batch_requests:
                        self.batch_manager.reset_request_metadata(req)
                        skipped_waiting_requests.prepend_request(request)
                        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue"""

    new_batch_valid_1 = """                if batch_valid == 1:
                    for req in batch_requests:
                        self.batch_manager.reset_request_metadata(req)
                        skipped_waiting_requests.prepend_request(req)
                        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue"""

    if old_batch_valid_1 in content:
        content = content.replace(old_batch_valid_1, new_batch_valid_1)
        patched = True
        print("[INFO] Patched batch_scheduler.py: batch_valid==1 variable name fix")
    else:
        print("[WARN] Could not find batch_valid==1 code to patch")

    # --- Patch 2+4+5: Fix KV allocation can_schedule bug, preemption target,
    #     and add reset_request_metadata ---
    # The original code has:
    # - else: can_schedule = True  (incorrectly resets after first failure)
    # - self.running.pop() (preempts tail instead of current batch)
    # - no reset_request_metadata call
    old_kv_alloc = """            can_schedule = True
            batch_new_blocks = []
            for request, num_new_tokens in zip(batch_requests, batch_num_new_tokens):
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    can_schedule = False
                    # # The request cannot be scheduled.
                    # # Preempt the lowest-priority request.
                    # if self.policy == SchedulingPolicy.PRIORITY:
                    #     preempted_req = max(
                    #         self.running,
                    #         key=lambda r: (r.priority, r.arrival_time),
                    #     )
                    #     self.running.remove(preempted_req)
                    #     if preempted_req in scheduled_running_reqs:
                    #         scheduled_running_reqs.remove(preempted_req)
                    # else:
                    #     preempted_req = self.running.pop()

                    # self.kv_cache_manager.free(preempted_req)
                    # self.encoder_cache_manager.free(preempted_req)
                    # preempted_req.status = RequestStatus.PREEMPTED
                    # preempted_req.num_computed_tokens = 0
                    # if self.log_stats:
                    #     preempted_req.record_event(
                    #         EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    # self.waiting.prepend_request(preempted_req)
                    # preempted_reqs.append(preempted_req)
                    # if preempted_req == request:
                    #     # No more request to preempt.
                    #     can_schedule = False
                    #     break
                else:
                    # The request can be scheduled.
                    can_schedule = True

                batch_new_blocks.append(new_blocks)

            if not can_schedule:
                for req in batch_requests:
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                break"""

    new_kv_alloc = """            can_schedule = True
            batch_new_blocks = []
            for request, num_new_tokens in zip(batch_requests, batch_num_new_tokens):
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # CFG_PAIRING_FIX: Do NOT reset can_schedule to True in the
                    # else branch below. If ANY request in the batch fails to
                    # allocate, the whole batch must be preempted.
                    can_schedule = False

                batch_new_blocks.append(new_blocks)

            if not can_schedule:
                # CFG_PAIRING_FIX: Preempt the CURRENT batch requests (not the
                # tail of self.running). The current batch failed to allocate KV
                # blocks, so they should be moved back to waiting.
                # We also need to free any blocks that were successfully
                # allocated for other requests in this batch.
                for preempted_req in batch_requests:
                    self.running.remove(preempted_req)
                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    # CFG_PAIRING_FIX: Clean up metadata before moving back to
                    # waiting, so set_request_metadata won't hit 'already exists'
                    self.batch_manager.reset_request_metadata(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                # Adjust req_index since we removed items from self.running
                # before current position. batch_requests were at positions
                # [req_index - len(batch_requests), req_index), so shift back.
                req_index -= len(batch_requests)
                break"""

    if old_kv_alloc in content:
        content = content.replace(old_kv_alloc, new_kv_alloc)
        patched = True
        print("[INFO] Patched batch_scheduler.py: can_schedule + preempt target + reset_metadata fix")
    else:
        print("[WARN] Could not find KV allocation code to patch")

    # --- Patch 3: Fix PREEMPTED requests not recognized in WAITING scheduling ---
    old_valid_check = """                    is_valid = False
                    if request.status == RequestStatus.WAITING:
                        is_valid = True

                    # KVTransfer: skip request if still waiting for remote kvs."""

    new_valid_check = """                    is_valid = False
                    if request.status == RequestStatus.WAITING:
                        is_valid = True

                    # CFG_PAIRING_FIX: Preempted requests moved back to waiting
                    # must be recognized as valid for re-scheduling.
                    if request.status == RequestStatus.PREEMPTED:
                        is_valid = True

                    # KVTransfer: skip request if still waiting for remote kvs."""

    if old_valid_check in content:
        content = content.replace(old_valid_check, new_valid_check)
        patched = True
        print("[INFO] Patched batch_scheduler.py: PREEMPTED status scheduling fix")
    else:
        print("[WARN] Could not find WAITING validation code to patch")

    # --- Patch 4: Safe access to req_to_new_blocks in _make_cached_request_data ---
    # When preempted requests are resumed, they may not have entries in
    # req_to_new_blocks, causing a KeyError. Use .get() with a None fallback.
    old_new_block_ids = """            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True))"""

    new_new_block_ids = """            req_new_blocks = req_to_new_blocks.get(req_id)
            new_block_ids.append(
                req_new_blocks.get_block_ids(allow_none=True)
                if req_new_blocks is not None else None)"""

    if old_new_block_ids in content:
        content = content.replace(old_new_block_ids, new_new_block_ids)
        patched = True
        print("[INFO] Patched batch_scheduler.py: safe req_to_new_blocks access fix")
    else:
        print("[WARN] Could not find req_to_new_blocks code to patch")

    if patched:
        batch_scheduler_path.write_text(content)
        print(f"[SUCCESS] batch_scheduler.py patched at {batch_scheduler_path}")
    return patched


def patch_llm_sort(llm_path: Path):
    """Patch llm.py to fix ValueError in _run_engine's sorted() call.

    In CFG mode, request_ids are transformed to format like "0_abc12345_0"
    by get_hybrid_outputs(). The original code does:
        sorted(outputs, key=lambda x: int(x.request_id))
    which raises ValueError on non-numeric request_ids.
    """
    content = llm_path.read_text()

    # Check if already patched
    if "# CFG_PAIRING_FIX" in content or "_safe_sort_key" in content:
        print("[INFO] llm.py already patched")
        return False

    old_sort = "        return sorted(outputs, key=lambda x: int(x.request_id))"

    new_sort = """        # CFG_PAIRING_FIX: Use a safe sort key that handles both numeric
        # request IDs (e.g. "5") and CFG-transformed IDs (e.g. "0_abc12345_0").
        def _safe_sort_key(x):
            try:
                return (0, int(x.request_id), "")
            except (ValueError, TypeError):
                return (1, 0, x.request_id)
        return sorted(outputs, key=_safe_sort_key)"""

    if old_sort in content:
        content = content.replace(old_sort, new_sort)
        llm_path.write_text(content)
        print(f"[SUCCESS] llm.py patched at {llm_path}")
        return True
    else:
        print("[WARN] Could not find sorted() call to patch in llm.py")
        return False


def main():
    parser = argparse.ArgumentParser(description="Patch vllm for no-CFG support and LoRA support")
    parser.add_argument("--revert", action="store_true", help="Revert patches")
    parser.add_argument("--lora-only", action="store_true", help="Apply only LoRA patches")
    args = parser.parse_args()
    
    site_dir = get_vllm_site()
    backup_dir = site_dir.parent / "vllm_no_cfg_backup"
    
    # Files to patch
    builtin_path = site_dir / "v1" / "sample" / "logits_processor" / "builtin.py"
    preprocess_path = site_dir / "inputs" / "preprocess.py"
    scheduler_path = site_dir / "v1" / "core" / "sched" / "scheduler.py"
    model_runner_path = site_dir / "v1" / "worker" / "gpu_model_runner.py"
    batch_manager_path = site_dir / "v1" / "core" / "sched" / "batch_manager.py"
    batch_scheduler_path = site_dir / "v1" / "core" / "sched" / "batch_scheduler.py"
    llm_path = site_dir / "entrypoints" / "llm.py"
    
    if args.revert:
        print("[INFO] Reverting patches...")
        restore_file(builtin_path, backup_dir)
        restore_file(preprocess_path, backup_dir)
        restore_file(scheduler_path, backup_dir)
        restore_file(model_runner_path, backup_dir)
        restore_file(batch_manager_path, backup_dir)
        restore_file(batch_scheduler_path, backup_dir)
        # Restore llm.py
        llm_path = site_dir / "entrypoints" / "llm.py"
        restore_file(llm_path, backup_dir)
        # Also restore LoRA patched files
        emu3_path = site_dir / "model_executor" / "models" / "emu3_5.py"
        logits_path = site_dir / "lora" / "layers" / "logits_processor.py"
        # Try both file names
        embed_path = site_dir / "lora" / "layers" / "vocal_parallel_embedding.py"
        if not embed_path.exists():
            embed_path = site_dir / "lora" / "layers" / "vocab_parallel_embedding.py"
        config_path_new = site_dir / "config" / "lora.py"
        config_path_old = site_dir / "config.py"
        restore_file(emu3_path, backup_dir)
        restore_file(logits_path, backup_dir)
        restore_file(embed_path, backup_dir)
        restore_file(config_path_new, backup_dir)
        restore_file(config_path_old, backup_dir)
        print("[SUCCESS] Reverted all patches")
        return
    
    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # If --lora-only, skip CFG patches
    if args.lora_only:
        print("[INFO] Applying LoRA patches only...")
        apply_lora_patches(site_dir, backup_dir)
        print("[SUCCESS] LoRA patches applied successfully!")
        print(f"[INFO] Backup stored at: {backup_dir}")
        print("\n[USAGE] To use LoRA with Emu3.5:")
        print("  1. Set lora_extra_vocab_size=0 in your LoRA config")
        print("  2. Use target_modules='all-linear' or specify specific modules")
        print("  3. Emu3.5 vocab_size (282926) is now supported")
        return
    
    # Backup original files
    backup_file(builtin_path, backup_dir)
    backup_file(preprocess_path, backup_dir)
    backup_file(scheduler_path, backup_dir)
    backup_file(model_runner_path, backup_dir)
    backup_file(batch_manager_path, backup_dir)
    backup_file(batch_scheduler_path, backup_dir)
    backup_file(llm_path, backup_dir)
    
    # Apply patches
    print("[INFO] Applying no-CFG support patches...")
    
    if builtin_path.exists():
        patch_logits_processor(builtin_path)
    else:
        print(f"[ERROR] builtin.py not found at {builtin_path}")
        print("[INFO] This might be because the CFG patches haven't been applied yet.")
        print("[INFO] Run 'python src/patch/apply.py' first to apply the base CFG patches.")
        sys.exit(1)
    
    if preprocess_path.exists():
        patch_preprocess(preprocess_path)
    else:
        print(f"[WARN] preprocess.py not found at {preprocess_path}")
    
    if scheduler_path.exists():
        patch_scheduler(scheduler_path)
        # Also patch for metadata manager (dynamic sampling params)
        patch_scheduler_metadata_manager(scheduler_path)
    else:
        print(f"[WARN] scheduler.py not found at {scheduler_path}")
    
    if model_runner_path.exists():
        patch_gpu_model_runner(model_runner_path)
    else:
        print(f"[WARN] gpu_model_runner.py not found at {model_runner_path}")
    
    if batch_manager_path.exists():
        patch_batch_manager(batch_manager_path)
    else:
        print(f"[WARN] batch_manager.py not found at {batch_manager_path}")
    
    if batch_scheduler_path.exists():
        patch_batch_scheduler(batch_scheduler_path)
    else:
        print(f"[WARN] batch_scheduler.py not found at {batch_scheduler_path}")
    
    if llm_path.exists():
        patch_llm_sort(llm_path)
    else:
        print(f"[WARN] llm.py not found at {llm_path}")
    
    # Also apply LoRA patches
    apply_lora_patches(site_dir, backup_dir)
    
    print("[SUCCESS] All patches applied successfully!")
    print(f"[INFO] Backup stored at: {backup_dir}")
    print("\n[USAGE] To use no-CFG mode:")
    print("  1. Set guidance_scale <= 1.0 in your config or CLI")
    print("  2. Do NOT pass uncond_prompt_token_ids to model.generate()")
    print("  3. Use the default scheduler (don't set scheduler_cls when CFG is disabled)")
    print("\n[USAGE] To use LoRA with Emu3.5:")
    print("  1. Set lora_extra_vocab_size=0 in your LoRA config")
    print("  2. Use target_modules='all-linear' or specify specific modules")
    print("  3. Emu3.5 vocab_size (282926) is now supported")


if __name__ == "__main__":
    main()
