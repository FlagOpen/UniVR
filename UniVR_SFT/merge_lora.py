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
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, AutoTokenizer
from peft import PeftModel
from src.emu3p5 import Emu3ForCausalLM, Emu3Config

# Add src to path to allow imports from src
sys.path.append(os.path.abspath("src"))

@dataclass
class MergeArguments:
    base_model_path: str = field(
        metadata={"help": "Path to the base model"}
    )
    adapter_path: str = field(
        metadata={"help": "Path to the LoRA adapter"}
    )
    output_path: str = field(
        metadata={"help": "Path to save the merged model"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer. If None, uses base_model_path or src/tokenizer_emu3_ibq"}
    )

def merge():
    parser = HfArgumentParser((MergeArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    print(f"Loading base model from {args.base_model_path}")
    # Load base model
    model_config = Emu3Config.from_pretrained(
        "./weights/Emu3.5",
        trust_remote_code=True,
    )

    model = Emu3ForCausalLM.from_pretrained(
        args.base_model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    print(f"Loading LoRA adapter from {args.adapter_path}")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging weights...")
    # Merge LoRA weights into base model
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}")
    model.save_pretrained(args.output_path)

    # Handle Tokenizer
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        # Try base model path first
        if os.path.exists(os.path.join(args.base_model_path, "tokenizer_config.json")):
            tokenizer_path = args.base_model_path
        else:
            # Fallback to default project structure
            tokenizer_path = "src/tokenizer_emu3_ibq"
    
    print(f"Loading tokenizer from {tokenizer_path}")
    try:
        # Try loading with special tokens file if it exists in the path
        special_tokens_file = os.path.join(tokenizer_path, "emu3_vision_tokens.txt")
        if os.path.exists(special_tokens_file):
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                special_tokens_file=special_tokens_file,
                trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )
            
        # Ensure special tokens are set as in train.py
        # This is important if loading from raw source where they might not be in config
        if tokenizer.bos_token != "<|extra_203|>":
            print("Setting special tokens...")
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

        print(f"Saving tokenizer to {args.output_path}")
        tokenizer.save_pretrained(args.output_path)
        
    except Exception as e:
        print(f"Warning: Failed to save tokenizer: {e}")
        print("You may need to manually copy tokenizer files.")

    print("Done!")

if __name__ == "__main__":
    merge()
