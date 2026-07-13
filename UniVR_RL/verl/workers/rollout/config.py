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
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 6666
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256  # max sequences per iteration, Emu3 uses 2
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)
    
    # Emu3 CFG (Classifier-Free Guidance) settings
    emu3_cfg_enabled: bool = False  # Enable CFG for Emu3
    emu3_guidance_scale: float = 1.0  # CFG guidance scale (1.0 = disabled)
    # Text sampling params (must match inference_vllm_dist.py for consistent results)
    emu3_text_top_k: int = 1024       # inference_vllm_dist.py uses 1024
    emu3_text_top_p: float = 0.9      # inference_vllm_dist.py uses 0.9
    emu3_text_temperature: float = 1.0  # inference_vllm_dist.py uses 1.0
    # Visual/Image sampling params
    emu3_visual_top_k: int = 10240    # inference_vllm_dist.py uses 10240
    emu3_visual_top_p: float = 1.0    # inference_vllm_dist.py uses 1.0
    emu3_visual_temperature: float = 1.0
    emu3_target_width: Optional[int] = None  # Target image width for generation
    emu3_target_height: Optional[int] = None  # Target image height for generation
    image_area: int = 518400  # Total image area in pixels (e.g., 720*720)
    emu3_max_new_tokens: int = 32768
    emu3_temperature: float = 1.0
    emu3_top_p: float = 1.0
    emu3_top_k: int = 131072
    # Emu3 Vision Tokenizer (VQ) settings
    emu3_vq_path: str = ""  # Path to VQ tokenizer
    emu3_vq_type: str = "ibq"  # VQ tokenizer type
    
    # Memory optimization: disable image decoding during rollout (saves ~2GB GPU memory)
    enable_image_decode_for_reward: bool = False  # Set True only when using VLM reward
    
    # DINOv2 feature similarity computation in rollout (requires GPU)
    # When enabled, DINOv2 scores are pre-computed here and passed to reward function,
    # because the reward worker (Ray actor) does not have GPU access.
    enable_dinov2_in_rollout: bool = False
    dinov2_model_name: str = ""
    dinov2_pooling: str = "avg_patch"       # "cls" | "avg_patch" | "avg_all"
    dinov2_metric: str = "gaussian_rbf"     # "cosine" | "gaussian_rbf"
    dinov2_rbf_sigma: float = 3.0           # RBF bandwidth (higher = more peaked)
    
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
