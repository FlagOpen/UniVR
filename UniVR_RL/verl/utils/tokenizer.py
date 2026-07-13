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

"""Utils for tokenization."""

import os
from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin


def _is_emu3_tokenizer(model_path: str) -> bool:
    """Check if the tokenizer path contains Emu3 specific files."""
    emu3_vision_tokens = os.path.join(model_path, "emu3_vision_tokens.txt")
    emu3_tiktoken = os.path.join(model_path, "emu3.tiktoken")
    return os.path.exists(emu3_vision_tokens) or os.path.exists(emu3_tiktoken)


def _setup_emu3_special_tokens(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Setup Emu3 special tokens on the tokenizer."""
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
    print("[Emu3] Configured Emu3 special tokens on tokenizer")
    return tokenizer


def get_tokenizer(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    # Check for Emu3 tokenizer
    is_emu3 = _is_emu3_tokenizer(model_path)
    if is_emu3:
        special_tokens_file = os.path.join(model_path, "emu3_vision_tokens.txt")
        if os.path.exists(special_tokens_file):
            kwargs["special_tokens_file"] = special_tokens_file
        print(f"[Emu3] Loading Emu3 tokenizer from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    
    # Setup Emu3 special tokens
    if is_emu3:
        tokenizer = _setup_emu3_special_tokens(tokenizer)
    
    if override_chat_template is not None:
        with open(override_chat_template) as f:
            tokenizer.chat_template = f.read()

        print(f"New chat template: {tokenizer.chat_template}")

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    # Emu3 does not use processor - images are pre-tokenized with VQ
    if _is_emu3_tokenizer(model_path):
        print("[Emu3] Emu3 model detected, skipping processor (images are pre-tokenized)")
        return None
    
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except Exception as e:
        print(f"Warning: Failed to load processor from {model_path}: {e}")
        return None
    if override_chat_template is not None:
        with open(override_chat_template) as f:
            processor.chat_template = f.read()

        print(f"New chat template: {processor.chat_template}")

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/auto/processing_auto.py#L386
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
