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
Emu3 model integration utilities for EasyR1 RL training framework.

This module provides:
1. Model registration to transformers AutoClass
2. Tokenizer setup with Emu3 special tokens
3. Vision tokenizer (VQ-IBQ) integration
"""

import os
import os.path as osp
from typing import Optional, Dict, Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizer


# Emu3 special tokens mapping
EMU3_SPECIAL_TOKENS = {
    "bos_token": "<|extra_203|>",
    "eos_token": "<|extra_204|>",
    "pad_token": "<|endoftext|>",
    "eol_token": "<|extra_200|>",
    "eof_token": "<|extra_201|>",
    "tms_token": "<|extra_202|>",
    "img_token": "<|image token|>",
    "boi_token": "<|image start|>",
    "eoi_token": "<|image end|>",
    "bss_token": "<|extra_100|>",
    "ess_token": "<|extra_101|>",
    "bog_token": "<|extra_60|>",
    "eog_token": "<|extra_61|>",
    "boc_token": "<|extra_50|>",
    "eoc_token": "<|extra_51|>",
}


def is_emu3_model(model_path: str) -> bool:
    """
    Check if the model at the given path is an Emu3 model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        True if it's an Emu3 model, False otherwise
    """
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower()
        return model_type in ("emu3", "emu3.5")
    except Exception:
        # Check if emu3 config files exist
        config_file = osp.join(model_path, "config.json")
        if osp.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return config_data.get("model_type", "").lower() in ("emu3", "emu3.5")
    return False


def register_emu3_model(emu_src_path: str = None, force: bool = True):
    """
    Register custom Emu3 model classes to transformers AutoClass.
    
    This allows using AutoModelForCausalLM.from_pretrained() to load Emu3 models
    with your custom implementation instead of transformers built-in.
    
    Args:
        emu_src_path: Path to Emu_VW/src directory. If None, will try to find it.
        force: If True, will unregister existing Emu3 and register custom one.
    """
    import sys
    
    # Default path
    if emu_src_path is None:
        # Try common locations
        possible_paths = [
            "../UniVR_SFT/src",
            osp.join(osp.dirname(__file__), "../../../../UniVR_SFT/src"),
        ]
        for path in possible_paths:
            if osp.exists(path):
                emu_src_path = osp.abspath(path)
                break
    
    if emu_src_path and emu_src_path not in sys.path:
        sys.path.insert(0, emu_src_path)
        print(f"[Emu3] Added {emu_src_path} to sys.path")
    
    try:
        from emu3p5 import Emu3ForCausalLM, Emu3Config
        # Force unregister existing Emu3 if needed
        if force:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
            
            # Remove from CONFIG_MAPPING._mapping (where built-in models are registered)
            # Key is lowercase 'emu3'
            if hasattr(CONFIG_MAPPING, "_mapping"):
                for key in ["emu3", "Emu3"]:
                    if key in CONFIG_MAPPING._mapping:
                        del CONFIG_MAPPING._mapping[key]
                        print(f"[Emu3] Removed '{key}' from CONFIG_MAPPING._mapping")
            
            # Also remove from _extra_content (where dynamically registered models go)
            if hasattr(CONFIG_MAPPING, "_extra_content"):
                for key in ["emu3", "Emu3"]:
                    if key in CONFIG_MAPPING._extra_content:
                        del CONFIG_MAPPING._extra_content[key]
                        print(f"[Emu3] Removed '{key}' from CONFIG_MAPPING._extra_content")
            
            # For MODEL_FOR_CAUSAL_LM_MAPPING, need to find the config class and remove it
            # Check both _mapping and _extra_content
            if hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, "_mapping"):
                keys_to_remove = [k for k in MODEL_FOR_CAUSAL_LM_MAPPING._mapping 
                                  if hasattr(k, "__name__") and "emu3" in k.__name__.lower()]
                for key in keys_to_remove:
                    del MODEL_FOR_CAUSAL_LM_MAPPING._mapping[key]
                    print(f"[Emu3] Removed {key.__name__} from MODEL_FOR_CAUSAL_LM_MAPPING._mapping")
            
            if hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, "_extra_content"):
                keys_to_remove = [k for k in MODEL_FOR_CAUSAL_LM_MAPPING._extra_content 
                                  if hasattr(k, "__name__") and "emu3" in k.__name__.lower()]
                for key in keys_to_remove:
                    del MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[key]
                    print(f"[Emu3] Removed {key.__name__} from MODEL_FOR_CAUSAL_LM_MAPPING._extra_content")
            
            print("[Emu3] Cleared existing Emu3 registrations from transformers")
        
        # Register custom Emu3
        try:
            AutoConfig.register("Emu3", Emu3Config)
        except ValueError:
            # Already registered with same class, that's fine
            pass
            
        try:
            AutoModelForCausalLM.register(Emu3Config, Emu3ForCausalLM)
        except ValueError:
            # Already registered with same class, that's fine
            pass
        
        print(f"[Emu3] Successfully registered custom Emu3Config and Emu3ForCausalLM")
        return True
        
    except ImportError as e:
        print(f"[Emu3] Warning: Could not import emu3p5: {e}")
        print(f"[Emu3] Make sure Emu_VW/src is in your Python path")
        return False
    except Exception as e:
        print(f"[Emu3] Warning: Registration failed: {e}")
        print(f"[Emu3] Will rely on trust_remote_code=True for model loading")
        return False


def setup_emu3_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    Setup Emu3 special tokens on a tokenizer.
    
    Args:
        tokenizer: A PreTrainedTokenizer instance
        
    Returns:
        The tokenizer with Emu3 special tokens configured
    """
    for attr_name, token_value in EMU3_SPECIAL_TOKENS.items():
        setattr(tokenizer, attr_name, token_value)
    
    print(f"[Emu3] Configured {len(EMU3_SPECIAL_TOKENS)} special tokens on tokenizer")
    return tokenizer


def get_emu3_tokenizer(
    tokenizer_path: str,
    trust_remote_code: bool = True,
    **kwargs
) -> PreTrainedTokenizer:
    """
    Load and configure an Emu3 tokenizer.
    
    Args:
        tokenizer_path: Path to the tokenizer directory
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for AutoTokenizer
        
    Returns:
        Configured Emu3 tokenizer
    """
    from transformers import AutoTokenizer
    
    # Check for special tokens file
    special_tokens_file = osp.join(tokenizer_path, "emu3_vision_tokens.txt")
    if osp.exists(special_tokens_file):
        kwargs["special_tokens_file"] = special_tokens_file
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    return setup_emu3_tokenizer(tokenizer)


class Emu3VisionTokenizer:
    """
    Wrapper for Emu3 Vision Tokenizer (VQ-IBQ).
    
    This class provides an interface to the IBQ vision tokenizer
    used by Emu3 to convert images to discrete tokens.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        tokenizer_type: str = "ibq",
    ):
        """
        Initialize the vision tokenizer.
        
        Args:
            model_path: Path to vision tokenizer weights (e.g., Emu3.5-VisionTokenizer)
            device: Device to load the model on
            tokenizer_type: Type of vision tokenizer ("ibq")
        """
        self.model_path = model_path
        self.device = device
        self.tokenizer_type = tokenizer_type
        self._model = None
        
    def _load_model(self):
        """Lazy load the vision tokenizer model."""
        if self._model is not None:
            return
            
        try:
            from omegaconf import OmegaConf
            
            # Try to import from Emu_VW
            import sys
            emu_src_path = "../UniVR_SFT/src"
            if emu_src_path not in sys.path:
                sys.path.insert(0, emu_src_path)
            
            from vision_tokenizer import build_vision_tokenizer
            
            self._model = build_vision_tokenizer(
                self.tokenizer_type,
                self.model_path,
                device=self.device
            )
            print(f"[Emu3] Loaded vision tokenizer from {self.model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Emu3 vision tokenizer: {e}")
    
    @property
    def model(self):
        """Get the underlying vision tokenizer model."""
        self._load_model()
        return self._model
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to discrete tokens.
        
        Args:
            images: Tensor of shape (B, C, H, W), normalized to [-1, 1]
            
        Returns:
            Token indices of shape (B, H', W')
        """
        self._load_model()
        with torch.no_grad():
            _, _, (_, _, indices) = self._model.encode(images.to(self.device))
        return indices
    
    def decode(self, tokens: torch.Tensor, shape: tuple = None) -> torch.Tensor:
        """
        Decode discrete tokens back to images.
        
        Args:
            tokens: Token indices
            shape: Optional shape (B, H', W', C) for reshaping
            
        Returns:
            Reconstructed images
        """
        self._load_model()
        with torch.no_grad():
            images = self._model.decode_code(tokens, shape=shape)
        return images
    
    def to(self, device: str):
        """Move the model to a device."""
        if self._model is not None:
            self._model = self._model.to(device)
        self.device = device
        return self


def load_emu3_model(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
    **kwargs
) -> torch.nn.Module:
    """
    Load an Emu3 model with proper configuration.
    
    Args:
        model_path: Path to the model
        torch_dtype: Data type for the model
        device_map: Device mapping strategy
        attn_implementation: Attention implementation
        **kwargs: Additional arguments
        
    Returns:
        Loaded Emu3 model
    """
    # Ensure model is registered
    register_emu3_model()
    
    try:
        from emu3p5 import Emu3ForCausalLM, Emu3Config
        
        config = Emu3Config.from_pretrained(model_path, trust_remote_code=True)
        model = Emu3ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            **kwargs
        )
        return model
        
    except ImportError:
        # Fallback to AutoModel if registered
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            **kwargs
        )
        return model
