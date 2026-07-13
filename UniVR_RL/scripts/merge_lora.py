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



import argparse
import json
import os
import sys
import shutil

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)


def check_and_register_emu3(model_path: str) -> bool:
    """Check if the model at model_path is an Emu3 model and register it to transformers AutoClass."""
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        return False

    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)
        model_type = config_data.get("model_type", "").lower()
    except Exception:
        return False

    if model_type not in ("emu3", "emu3.5"):
        return False

    print(f"[Emu3] Detected Emu3 model at {model_path}")

    # Add Emu_VW/src to sys.path so that `emu3p5` is importable
    emu_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Emu_VW/src")
    emu_src_path = os.path.abspath(emu_src_path)
    # Fallback to known absolute path
    if not os.path.exists(emu_src_path):
        emu_src_path = "../UniVR_SFT/src"
    if os.path.exists(emu_src_path) and emu_src_path not in sys.path:
        sys.path.insert(0, emu_src_path)
        print(f"[Emu3] Added {emu_src_path} to sys.path")

    try:
        from emu3p5 import Emu3Config, Emu3ForCausalLM

        # Force-clear any existing Emu3 registrations
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

        for mapping in [CONFIG_MAPPING]:
            for attr in ["_mapping", "_extra_content"]:
                d = getattr(mapping, attr, None)
                if d is None:
                    continue
                for key in ["emu3", "Emu3"]:
                    if key in d:
                        del d[key]

        for mapping in [MODEL_FOR_CAUSAL_LM_MAPPING]:
            for attr in ["_mapping", "_extra_content"]:
                d = getattr(mapping, attr, None)
                if d is None:
                    continue
                keys_to_remove = [k for k in d if hasattr(k, "__name__") and "emu3" in k.__name__.lower()]
                for key in keys_to_remove:
                    del d[key]

        # Register custom Emu3
        try:
            AutoConfig.register("Emu3", Emu3Config)
        except ValueError:
            pass
        try:
            AutoModelForCausalLM.register(Emu3Config, Emu3ForCausalLM)
        except ValueError:
            pass

        print("[Emu3] Successfully registered Emu3Config and Emu3ForCausalLM")
        return True
    except Exception as e:
        print(f"[Emu3] Warning: Failed to register Emu3: {e}")
        return False


def get_auto_class(config: PretrainedConfig):
    """Determine the correct AutoModel class based on model architecture."""
    architectures: list[str] = getattr(config, "architectures", ["Unknown"])
    arch = architectures[0]

    if "ForTokenClassification" in arch:
        return AutoModelForTokenClassification
    elif "ForConditionalGeneration" in arch:
        return AutoModelForImageTextToText
    elif "ForCausalLM" in arch:
        return AutoModelForCausalLM
    else:
        raise NotImplementedError(f"Unknown architecture: {architectures}")


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    """Parse string dtype to torch.dtype."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def upload_model_to_huggingface(local_path: str, remote_path: str):
    """Push merged model to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")
    print(f"Uploaded to https://huggingface.co/{remote_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to the base model. If not specified, reads from adapter_config.json's base_model_name_or_path.",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Path to the lora_adapter directory (containing adapter_config.json and adapter_model.safetensors).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bf16",
        help="Data type for model loading (bf16, fp16, fp32). Default: bf16.",
    )
    parser.add_argument(
        "--copy_tokenizer_from",
        type=str,
        default=None,
        help="Path to copy tokenizer files from (if base model doesn't contain them). "
             "Typically the checkpoint's huggingface/ directory.",
    )
    parser.add_argument(
        "--hf_upload_path",
        type=str,
        default=None,
        help="HuggingFace repo id to upload (e.g., 'user/repo_name'). Optional.",
    )
    args = parser.parse_args()

    torch_dtype = parse_torch_dtype(args.torch_dtype)
    lora_adapter_path = args.lora_adapter_path
    output_path = args.output_path

    # -------------------------------------------------------------------------
    # Step 0: Validate lora_adapter directory
    # -------------------------------------------------------------------------
    adapter_config_file = os.path.join(lora_adapter_path, "adapter_config.json")
    adapter_weights_file = os.path.join(lora_adapter_path, "adapter_model.safetensors")

    if not os.path.exists(adapter_config_file):
        # Also check for .bin format
        adapter_weights_file_bin = os.path.join(lora_adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_config_file):
            raise FileNotFoundError(f"adapter_config.json not found in {lora_adapter_path}")

    if not os.path.exists(adapter_weights_file):
        adapter_weights_file = os.path.join(lora_adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_weights_file):
            raise FileNotFoundError(
                f"Neither adapter_model.safetensors nor adapter_model.bin found in {lora_adapter_path}"
            )

    with open(adapter_config_file, "r") as f:
        adapter_config = json.load(f)

    print(f"LoRA config: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")
    print(f"Target modules: {adapter_config.get('target_modules')}")

    # -------------------------------------------------------------------------
    # Step 1: Determine base model path
    # -------------------------------------------------------------------------
    base_model_path = args.base_model_path
    if base_model_path is None:
        base_model_path = adapter_config.get("base_model_name_or_path")
        if base_model_path is None:
            raise ValueError(
                "Cannot determine base model path. "
                "Please specify --base_model_path or ensure adapter_config.json has base_model_name_or_path."
            )
    print(f"\nBase model path: {base_model_path}")

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model not found at {base_model_path}. "
            "Please specify the correct path with --base_model_path."
        )

    # -------------------------------------------------------------------------
    # Step 2: Register custom model class if needed (e.g., Emu3)
    # -------------------------------------------------------------------------
    is_emu3 = check_and_register_emu3(base_model_path)

    # -------------------------------------------------------------------------
    # Step 3: Load base model
    # -------------------------------------------------------------------------
    print(f"\nLoading base model from {base_model_path} (dtype={torch_dtype})...")

    if is_emu3:
        from emu3p5 import Emu3ForCausalLM
        model = Emu3ForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    else:
        config = AutoConfig.from_pretrained(base_model_path)
        AutoClass = get_auto_class(config)
        model = AutoClass.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    print(f"Base model loaded: {type(model).__name__}, params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # -------------------------------------------------------------------------
    # Step 4: Load LoRA adapter
    # -------------------------------------------------------------------------
    print(f"\nLoading LoRA adapter from {lora_adapter_path}...")
    model = PeftModel.from_pretrained(
        model,
        lora_adapter_path,
        torch_dtype=torch_dtype,
    )
    print(f"LoRA adapter loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # -------------------------------------------------------------------------
    # Step 5: Merge LoRA into base model
    # -------------------------------------------------------------------------
    print("\nMerging LoRA weights into base model...")
    model = model.merge_and_unload()
    print("LoRA merge complete.")

    # -------------------------------------------------------------------------
    # Step 6: Save merged model
    # -------------------------------------------------------------------------
    os.makedirs(output_path, exist_ok=True)
    print(f"\nSaving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    print("Model weights saved.")

    # -------------------------------------------------------------------------
    # Step 7: Copy/save tokenizer
    # -------------------------------------------------------------------------
    tokenizer_source = args.copy_tokenizer_from or base_model_path

    # Try loading tokenizer from source
    tokenizer_saved = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        tokenizer_saved = True
        print(f"Tokenizer saved from {tokenizer_source}")
    except Exception as e:
        print(f"Warning: Failed to load tokenizer from {tokenizer_source}: {e}")

    # Fallback: copy tokenizer files manually
    if not tokenizer_saved and args.copy_tokenizer_from:
        print(f"Falling back to copying tokenizer files from {args.copy_tokenizer_from}...")
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "tokenizer.model", "vocab.json", "merges.txt",
            # Emu3 specific
            "emu3.tiktoken", "tokenization_emu3.py",
        ]
        for fname in tokenizer_files:
            src = os.path.join(args.copy_tokenizer_from, fname)
            if os.path.exists(src):
                dst = os.path.join(output_path, fname)
                shutil.copy2(src, dst)
                print(f"  Copied {fname}")

    # Also ensure config.json exists in output
    config_src = os.path.join(output_path, "config.json")
    if not os.path.exists(config_src):
        # Copy from base model
        base_config = os.path.join(base_model_path, "config.json")
        if os.path.exists(base_config):
            shutil.copy2(base_config, config_src)
            print("Copied config.json from base model")

    print(f"\n{'=' * 60}")
    print(f"Merge complete! Merged model saved to: {output_path}")
    print(f"{'=' * 60}")

    # -------------------------------------------------------------------------
    # Step 8: Optional HuggingFace upload
    # -------------------------------------------------------------------------
    if args.hf_upload_path:
        print(f"\nUploading to HuggingFace: {args.hf_upload_path}...")
        upload_model_to_huggingface(output_path, args.hf_upload_path)


if __name__ == "__main__":
    main()
