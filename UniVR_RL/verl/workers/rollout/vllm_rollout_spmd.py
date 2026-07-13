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
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from PIL import Image

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin], tokenizer: Optional[PreTrainedTokenizer] = None) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    # For Emu3: block image tokens from generation
    elif tokenizer is not None and hasattr(tokenizer, "img_token"):
        try:
            img_token_id = tokenizer.encode(tokenizer.img_token)[0]
            return {img_token_id: -100}
        except Exception:
            pass
    return None


def _is_emu3_model(model_path: str) -> bool:
    """Check if the model is an Emu3 model."""
    import json
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            model_type = config_data.get("model_type", "").lower()
            return model_type in ("emu3", "emu3.5")
        except Exception:
            pass
    return False


def _decode_emu3_image(image_string: str, tokenizer, vision_tokenizer) -> Optional[Image.Image]:
    """Decode Emu3 visual tokens to PIL Image."""
    image_rows_data = []
    image_rows = re.split(re.escape(tokenizer.eol_token), image_string)
    for r in image_rows:
        token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
        if len(token_ids) > 0:
            row_token = [int(m) for m in token_ids]
            image_rows_data.append(row_token)
    
    if not image_rows_data:
        return None
    
    try:
        image_tensor = torch.tensor(
            image_rows_data, dtype=torch.long, device=next(iter(vision_tokenizer.parameters())).device
        )
        h, w = image_tensor.shape
        with torch.no_grad():
            decoded_tensor = vision_tokenizer.decode_code(image_tensor[None], shape=(1, h, w, 256)).float()
        decoded_tensor = decoded_tensor[0].permute(1, 2, 0)
        pil_image = Image.fromarray(
            ((decoded_tensor + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        )
        # Explicitly delete GPU tensors to free memory
        del image_tensor, decoded_tensor
        return pil_image
    except Exception as ex:
        print(f"[Emu3] decode image failed: {ex}")
        return None


def _multimodal_decode_emu3(outputs: str, tokenizer, vision_tokenizer=None) -> list:
    """Parse Emu3 multimodal output into text/image segments."""
    outputs = outputs.replace("<|extra_101|>", "").replace("<|extra_204|>", "")
    
    # Build pattern to match image/cot blocks
    pattern = re.compile(
        rf"({re.escape(tokenizer.bog_token)}.*?{re.escape(tokenizer.eog_token)}|"
        rf"{re.escape(tokenizer.boc_token)}.*?{re.escape(tokenizer.eoc_token)}|"
        rf"{re.escape(tokenizer.boi_token)}.*?{re.escape(tokenizer.eoi_token)})",
        re.DOTALL,
    )
    
    multimodal_output = []
    chunks = re.split(pattern, outputs)
    
    for c in chunks:
        if len(c) == 0:
            continue
        if tokenizer.boi_token in c and tokenizer.eoi_token in c:
            # Image block
            if vision_tokenizer is not None:
                image = _decode_emu3_image(c, tokenizer, vision_tokenizer)
                if image is not None:
                    multimodal_output.append(("image", image))
                else:
                    multimodal_output.append(("image_tokens", c[:200] + "..."))  # Save partial tokens for debug
            else:
                multimodal_output.append(("image_tokens", c[:200] + "..."))
        elif tokenizer.bog_token in c and tokenizer.eog_token in c:
            multimodal_output.append(
                ("global_cot", c.replace(tokenizer.bog_token, "").replace(tokenizer.eog_token, ""))
            )
        elif tokenizer.boc_token in c and tokenizer.eoc_token in c:
            multimodal_output.append(
                ("image_cot", c.replace(tokenizer.boc_token, "").replace(tokenizer.eoc_token, ""))
            )
        elif tokenizer.boi_token not in c and len(c.strip()) > 0:
            multimodal_output.append(("text", c))
    
    return multimodal_output


def _parse_image_token_grids(text: str, tokenizer) -> list[np.ndarray]:
    """
    Parse Emu3 response text to extract image VQ token grids (CPU only, no GPU).

    Scans for image blocks (<boi>...<eoi>) and extracts the visual token ID
    grids as numpy arrays. This is the CPU-only counterpart of _decode_emu3_image,
    separating parsing from GPU decode to enable batched processing.

    Args:
        text: Decoded response text containing Emu3 special tokens.
        tokenizer: Emu3 tokenizer (for special token strings).

    Returns:
        List of 2D numpy arrays of shape (h, w), each containing VQ codebook indices.
    """
    text = text.replace("<|extra_101|>", "").replace("<|extra_204|>", "")

    # Match image blocks between boi and eoi tokens
    pattern = re.compile(
        rf"{re.escape(tokenizer.boi_token)}(.*?){re.escape(tokenizer.eoi_token)}",
        re.DOTALL,
    )

    image_grids = []
    for match in pattern.finditer(text):
        image_content = match.group(1)
        image_rows_data = []
        image_rows = re.split(re.escape(tokenizer.eol_token), image_content)
        for r in image_rows:
            token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
            if token_ids:
                image_rows_data.append([int(m) for m in token_ids])

        if image_rows_data:
            try:
                grid = np.array(image_rows_data, dtype=np.int32)
                image_grids.append(grid)
            except ValueError:
                # Rows have inconsistent lengths - skip malformed image
                continue

    return image_grids


def _build_vision_tokenizer_lazy(vq_path: str, vq_type: str = "ibq", device: str = "cuda"):
    """Lazily build vision tokenizer for Emu3 image decoding.
    
    The vision tokenizer is always in eval mode with gradients disabled,
    as it's only used for decoding images, not for training.
    """
    try:
        import sys
        # Add Emu_VW to path if needed
        emu_vw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", "UniVR_SFT"))
        if emu_vw_path not in sys.path:
            sys.path.insert(0, emu_vw_path)
        
        from src.vision_tokenizer import build_vision_tokenizer
        print(f"[Emu3] Building vision tokenizer from {vq_path} (type={vq_type}, device={device})")
        vq_model = build_vision_tokenizer(vq_type, vq_path, device=device)
        
        # IMPORTANT: Ensure vision tokenizer is frozen and in eval mode
        # 1. Set to eval mode (disables dropout, batchnorm training, etc.)
        vq_model.eval()
        # 2. Disable gradient computation for all parameters
        for param in vq_model.parameters():
            param.requires_grad = False
        
        print(f"[Emu3] Vision tokenizer built successfully (eval mode, gradients disabled)")
        return vq_model
    except Exception as e:
        print(f"[Emu3] Warning: Failed to build vision tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None


def _save_emu3_debug_outputs(
    completions: list, 
    tokenizer, 
    vision_tokenizer=None,
    save_dir: str = "./debug_outputs",
    max_samples: int = 2,  # Reduced to save memory
    vq_path: str = None,
    vq_type: str = "ibq",
    vq_device: str = "cuda:0",
):
    """Save Emu3 generation outputs as JPG images and text files for visualization.
    
    NOTE: This is for debugging only. Set max_samples=0 to disable in production.
    """
    import datetime
    import traceback
    
    if max_samples <= 0:
        return None
    
    # Skip vision tokenizer loading to save memory - just save raw text
    # vision_tokenizer = None  # Uncomment to disable image decoding for memory savings
    
    # Lazily build vision tokenizer if not provided (uses memory!)
    if vision_tokenizer is None and vq_path is not None:
        vision_tokenizer = _build_vision_tokenizer_lazy(vq_path, vq_type, vq_device)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(save_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    print(f"[DEBUG] Saving Emu3 outputs to: {batch_dir}")
    
    # Only save up to max_samples
    num_samples = min(len(completions), max_samples)
    
    for i, completion in enumerate(completions[:num_samples]):
        sample_dir = os.path.join(batch_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        for j, output in enumerate(completion.outputs):
            try:
                decoded_text = tokenizer.decode(output.token_ids, skip_special_tokens=False)
                
                # Save raw decoded text (truncated for readability)
                text_preview = decoded_text[:2000] + "..." if len(decoded_text) > 2000 else decoded_text
                with open(os.path.join(sample_dir, f"output_{j}_raw.txt"), "w") as f:
                    f.write(f"Token IDs length: {len(output.token_ids)}\n")
                    f.write(f"First 20 tokens: {list(output.token_ids)[:20]}\n")
                    f.write(f"\n--- Decoded Text (preview) ---\n{text_preview}\n")
                
                # Parse multimodal content
                mm_outputs = _multimodal_decode_emu3(decoded_text, tokenizer, vision_tokenizer)
                
                img_count = 0
                text_parts = []
                
                for content_type, content in mm_outputs:
                    if content_type == "image" and isinstance(content, Image.Image):
                        img_path = os.path.join(sample_dir, f"output_{j}_img_{img_count:02d}.jpg")
                        content.save(img_path, "JPEG", quality=95)
                        print(f"  [DEBUG] Saved image: {img_path}")
                        img_count += 1
                    elif content_type == "text":
                        text_parts.append(content.strip())
                    elif content_type == "global_cot":
                        text_parts.append(f"[Global CoT]: {content.strip()}")
                    elif content_type == "image_cot":
                        text_parts.append(f"[Image CoT]: {content.strip()}")
                    elif content_type == "image_tokens":
                        text_parts.append(f"[Image Tokens (no decoder)]: {content}")
                
                # Save parsed text content
                if text_parts:
                    with open(os.path.join(sample_dir, f"output_{j}_text.txt"), "w") as f:
                        f.write("\n\n".join(text_parts))
                
                print(f"  [DEBUG] Sample {i}, Output {j}: {img_count} images, {len(text_parts)} text segments")
                
            except Exception as e:
                print(f"  [DEBUG] Error processing sample {i}, output {j}: {e}")
                traceback.print_exc()
    
    print(f"[DEBUG] Emu3 outputs saved to: {batch_dir}")
    return batch_dir


# ============== DINOv2 Feature Similarity (computed in rollout, has GPU) ==============

class _DINOv2FeatureExtractor:
    """
    Singleton DINOv2 feature extractor for rollout worker.

    This class is intentionally duplicated from the reward function because
    the reward worker (Ray actor) does NOT have GPU access.  By running
    DINOv2 here in the rollout worker we can use CUDA acceleration and
    pass the pre-computed scores to the reward function via
    ``non_tensor_batch["dinov2_scores"]``.
    """
    _instance = None
    _model = None
    _processor = None
    _device = None

    @classmethod
    def get_instance(cls, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        if cls._instance is None:
            from transformers import AutoImageProcessor, AutoModel
            cls._instance = cls()
            print(f"[DINOv2-Rollout] Loading model {model_name} on {device}...")
            cls._processor = AutoImageProcessor.from_pretrained(model_name)
            cls._model = AutoModel.from_pretrained(model_name).to(device)
            cls._model.eval()
            for p in cls._model.parameters():
                p.requires_grad = False
            cls._device = device
            print(f"[DINOv2-Rollout] Model loaded (eval mode, gradients disabled).")
        return cls._instance

    @classmethod
    def extract_features(cls, images: list, pooling: str = "avg_patch", batch_size: int = 32):
        """Extract L2-normalised DINOv2 features from a list of PIL Images.

        Args:
            images: List of PIL Images.
            pooling: Pooling strategy ('cls', 'avg_patch', 'avg_all').
            batch_size: Max images per forward pass to avoid OOM (default 32).

        Returns:
            Tensor of shape (N, D) with L2-normalised features, or None on failure.
        """
        import torch.nn.functional as F
        if not images:
            return None
        try:
            all_features = []
            for start in range(0, len(images), batch_size):
                chunk = images[start:start + batch_size]
                inputs = cls._processor(images=chunk, return_tensors="pt").to(cls._device)
                with torch.no_grad():
                    outputs = cls._model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                if pooling == "cls":
                    features = last_hidden_states[:, 0, :]
                elif pooling == "avg_patch":
                    features = last_hidden_states[:, 1:, :].mean(dim=1)
                elif pooling == "avg_all":
                    features = last_hidden_states.mean(dim=1)
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
                all_features.append(F.normalize(features, p=2, dim=-1))
                del inputs, outputs, last_hidden_states, features
            return torch.cat(all_features, dim=0)
        except Exception as e:
            print(f"[DINOv2-Rollout] Feature extraction failed: {e}")
            return None


def _dinov2_score(feat_a: torch.Tensor, feat_b: torch.Tensor,
                  metric: str = "gaussian_rbf", rbf_sigma: float = 3.0) -> float:
    """Compute [0, 1] similarity between two L2-normalised feature vectors."""
    if metric == "cosine":
        sim = (feat_a * feat_b).sum().item()
        return max(0.0, min(1.0, sim))
    elif metric == "gaussian_rbf":
        sq_dist = ((feat_a - feat_b) ** 2).sum().item()
        return float(np.exp(-rbf_sigma * sq_dist))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _compute_dinov2_scores_batch(
    gen_images_batch: list,
    gt_images_batch: list,
    model_name: str = "facebook/dinov2-base",
    device: str = "cuda",
    pooling: str = "avg_patch",
    metric: str = "gaussian_rbf",
    rbf_sigma: float = 3.0,
) -> list:
    """
    Compute DINOv2 similarity scores for a batch of (gen, gt) image pairs.

    For each sample, the *last* generated image is compared with the *last*
    GT image.  Samples without valid gen or GT images get a score of 0.0.

    Returns:
        List of float scores, one per sample, in [0, 1].
    """
    batch_size = len(gen_images_batch)
    scores = [0.0] * batch_size

    # Collect valid pairs
    valid_indices = []
    pair_gen = []
    pair_gt = []
    for i in range(batch_size):
        gen_imgs = gen_images_batch[i] if gen_images_batch[i] else []
        gt_imgs = gt_images_batch[i] if gt_images_batch[i] else []
        if len(gen_imgs) > 0 and len(gt_imgs) > 0:
            valid_indices.append(i)
            pair_gen.append(gen_imgs[-1])   # last frame
            pair_gt.append(gt_imgs[-1])     # last frame

    if not valid_indices:
        return scores

    _DINOv2FeatureExtractor.get_instance(model_name, device)

    # Interleave: [gen0, gt0, gen1, gt1, ...] for a single forward pass
    interleaved = []
    for g, t in zip(pair_gen, pair_gt):
        interleaved.append(g)
        interleaved.append(t)

    feats = _DINOv2FeatureExtractor.extract_features(interleaved, pooling=pooling)
    if feats is None:
        return scores

    for j in range(0, len(feats), 2):
        idx = valid_indices[j // 2]
        scores[idx] = _dinov2_score(feats[j], feats[j + 1], metric=metric, rbf_sigma=rbf_sigma)

    return scores


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        tokenizer_path: Optional[str] = None,
        **kwargs,
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            tokenizer_path: Optional separate tokenizer path (needed for Emu3 where tokenizer != model path)
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer  # Save tokenizer for Emu3 CFG
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

        engine_kwargs = {}
        self.is_emu3 = _is_emu3_model(model_path)  # Save as instance variable
        
        # Default chunked_prefill setting (can be overridden by Emu3)
        enable_chunked_prefill = config.enable_chunked_prefill
        
        if self.is_emu3:
            # Emu3-specific vLLM configuration
            print(f"[Emu3] Configuring vLLM for Emu3 model at {model_path}")
            print(f"[Emu3] Note: Make sure you have applied vLLM patches:")
            print(f"[Emu3]   1. python src/patch/apply.py")
            print(f"[Emu3]   2. python src/patch/apply_no_cfg_support.py")
            
            # Check if CFG is enabled (guidance_scale > 0)
            guidance_scale = getattr(config, 'emu3_guidance_scale', 1.0)
            cfg_enabled = getattr(config, 'emu3_cfg_enabled', False) and guidance_scale > 0
            self.emu3_cfg_enabled = cfg_enabled  # Save for generate_sequences
            
            # Determine max_num_seqs based on CFG mode
            base_max_num_seqs = getattr(config, 'max_num_seqs', 2)
            if cfg_enabled:
                max_num_seqs = max(2, base_max_num_seqs * 2)
            else:
                max_num_seqs = max(1, base_max_num_seqs)
            
            # Emu3 requires these settings
            enable_chunked_prefill = False
            engine_kwargs.update({
                "enable_prefix_caching": False,
                "max_num_seqs": max_num_seqs,
                "generation_config": 'vllm',
                "compilation_config": {
                    "full_cuda_graph": True,
                    "backend": "cudagraph",
                    "cudagraph_capture_sizes": list(range(1, max_num_seqs + 1)),
                },
            })
            print(f"[Emu3] CFG={'enabled' if cfg_enabled else 'disabled'} (guidance_scale={guidance_scale}), max_num_seqs={max_num_seqs}")
            
            # Only use batch_scheduler when CFG is actually enabled
            if cfg_enabled:
                engine_kwargs["scheduler_cls"] = "vllm.v1.core.sched.batch_scheduler.Scheduler"
                print(f"[Emu3] CFG enabled (guidance_scale > 0), using batch_scheduler")
            else:
                print(f"[Emu3] CFG disabled (guidance_scale <= 0), using default scheduler for 2x speedup")
            
            # Add Emu3 special token configuration
            if hasattr(tokenizer, "boi_token"):
                try:
                    resolution_map = {}
                    resolution_str = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]
                    for digit_str in resolution_str:
                        resolution_map[tokenizer.encode(digit_str)[0]] = digit_str
                    
                    engine_kwargs["additional_config"] = {
                        "boi_token_id": tokenizer.encode(tokenizer.boi_token)[0],
                        "soi_token_id": tokenizer.encode(tokenizer.img_token)[0],
                        "eol_token_id": tokenizer.encode(tokenizer.eol_token)[0],
                        "eoi_token_id": tokenizer.encode(tokenizer.eoi_token)[0],
                        "resolution_map": resolution_map,
                    }
                    print(f"[Emu3] Configured additional_config for Emu3 vision tokens")
                except Exception as e:
                    print(f"[Emu3] Warning: Failed to configure Emu3 additional_config: {e}")
            
            # Emu3 requires separate tokenizer path
            if tokenizer_path and tokenizer_path != model_path:
                engine_kwargs["tokenizer"] = tokenizer_path
                print(f"[Emu3] Using separate tokenizer path: {tokenizer_path}")
        elif processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        VLLMHijack.hijack()

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy" if not self.lora_kwargs else "safetensors",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_sleep_mode=True,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        # Emu3 sampling params (needed for both CFG and no-CFG modes)
        if self.is_emu3:
            guidance_scale = getattr(config, 'emu3_guidance_scale', 1.0)
            emu3_extra_args = {
                "guidance_scale": guidance_scale,
                # IMPORTANT: These defaults must match inference_vllm_dist.py for consistent results
                "text_top_k": getattr(config, 'emu3_text_top_k', 1024),  # Was 5, should be 1024
                "text_top_p": getattr(config, 'emu3_text_top_p', 0.9),   # Keep 0.9
                "text_temperature": getattr(config, 'emu3_text_temperature', 1.0),  # Was 0.7, should be 1.0
                "visual_top_k": getattr(config, 'emu3_visual_top_k', 10240),  # Was 2048, should be 10240
                "visual_top_p": getattr(config, 'emu3_visual_top_p', 1.0),  # Was 0.9, should be 1.0
                "visual_temperature": getattr(config, 'emu3_visual_temperature', 1.0),
                "width": getattr(config, "emu3_target_width", None),
                "height": getattr(config, "emu3_target_height", None),
                # FIXED: Always set area like inference_vllm_dist.py does
                # area is used by vLLM for image resolution calculation even when width/height are None
                "area": None,  # Default 518400 = 720*720
            }
            # Use eos token as stop token for text generation
            stop_token_ids = [tokenizer.encode(tokenizer.eos_token)[0]]
            
            # Use config.n for best-of-n sampling (same as other models)
            emu3_n = getattr(config, 'n', 1)
            print(f"[Emu3] SamplingParams: guidance_scale={guidance_scale}, n={emu3_n}")
            print(f"[Emu3] extra_args: {emu3_extra_args}")

            self.sampling_params = SamplingParams(
                n=emu3_n,  # Use config.n for best-of-n sampling
                # IMPORTANT: These defaults must match inference_vllm_dist.py for consistent results
                top_k=getattr(config, 'emu3_top_k', 131072),  # Was 2048, should be 131072 (like inference_vllm_dist.py)
                # top_p=getattr(config, 'emu3_top_p', 1.0),     # Was 0.9, should be 1.0
                # temperature=getattr(config, 'emu3_temperature', 1.0),
                temperature = 1.0,
                top_p=1.0,
                max_tokens=getattr(config, 'emu3_max_new_tokens', config.response_length),
                detokenize=False,
                extra_args=emu3_extra_args,
                stop_token_ids=stop_token_ids,
            )
            self.inference_engine.set_tokenizer(tokenizer)
        else:
            sampling_kwargs = {
                "max_tokens": config.response_length,
                "detokenize": False,
                "logit_bias": _get_logit_bias(processor, tokenizer),
            }
            default_sampling_params = SamplingParams()
            for key in config.to_dict().keys():
                if hasattr(default_sampling_params, key):
                    sampling_kwargs[key] = getattr(config, key)

            print(f"Sampling params: {sampling_kwargs}.")
            self.sampling_params = SamplingParams(**sampling_kwargs)
        
        # Vision tokenizer for Emu3 image decoding (optional, can be set later via set_vision_tokenizer)
        self.vision_tokenizer = None

    def set_vision_tokenizer(self, vision_tokenizer):
        """Set the vision tokenizer (VQ decoder) for Emu3 image decoding."""
        self.vision_tokenizer = vision_tokenizer
        print(f"[Emu3] Vision tokenizer set for image decoding")

    def offload_vision_tokenizer(self):
        """Offload VQ decoder from GPU to CPU to free ~1.7GB GPU memory during training phase."""
        if self.vision_tokenizer is not None:
            self.vision_tokenizer = self.vision_tokenizer.cpu()
            # NOTE: Do NOT call torch.cuda.empty_cache() here!
            # This runs while vLLM is still awake. The subsequent offload_vllm() will
            # call empty_cache() at the proper time (after vLLM sleep).
            # Calling empty_cache() mid-session disrupts PyTorch's caching allocator
            # and can cause memory fragmentation → CUBLAS_STATUS_ALLOC_FAILED later.
            print(f"[Emu3] Rank {self.rank}: Vision tokenizer offloaded to CPU")

    def load_vision_tokenizer(self):
        """Load VQ decoder back to GPU for image decoding during rollout phase."""
        if self.vision_tokenizer is not None:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            device = f"cuda:{local_rank}"
            self.vision_tokenizer = self.vision_tokenizer.to(device)
            print(f"[Emu3] Rank {self.rank}: Vision tokenizer loaded to {device}")

    def _init_vision_tokenizer(self):
        """Lazy initialization of vision tokenizer on each rank.
        Also handles reloading from CPU if it was previously offloaded."""
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        if self.vision_tokenizer is None:
            vq_path = getattr(self.config, 'emu3_vq_path',
                "../UniVR_SFT/weights/Emu3.5-VisionTokenizer")
            vq_type = getattr(self.config, 'emu3_vq_type', "ibq")
            print(f"[Emu3] Rank {self.rank}: Loading vision tokenizer on {device}...")
            self.vision_tokenizer = _build_vision_tokenizer_lazy(vq_path, vq_type, device=device)
        else:
            # Reload from CPU if previously offloaded
            vt_device = next(iter(self.vision_tokenizer.parameters())).device
            if str(vt_device) == "cpu":
                self.vision_tokenizer = self.vision_tokenizer.to(device)
                print(f"[Emu3] Rank {self.rank}: Vision tokenizer reloaded to {device} from CPU")
        return self.vision_tokenizer is not None

    def _batched_vq_decode_indexed(
        self,
        indexed_grids: list[tuple[int, int, np.ndarray]],
        device: torch.device,
        max_batch_size: int = 4,
    ) -> dict[tuple[int, int], Optional[Image.Image]]:
        """
        Batch VQ decode of indexed image token grids, grouped by resolution.

        Instead of decoding images one by one (each with a separate GPU kernel
        launch), this method:
        1. Groups images by (height, width)
        2. Stacks same-size images into a batch tensor
        3. Decodes the entire batch in a single VQ decoder call
        4. Converts all results to PIL in batch

        This can be 5-10x faster than sequential decoding for 100+ images,
        mainly by eliminating per-image torch.cuda.empty_cache() calls and
        maximizing GPU utilization through batched operations.

        Args:
            indexed_grids: List of (sample_idx, img_idx, token_grid_ndarray).
            device: GPU device for VQ decoder.
            max_batch_size: Max images per VQ decode call to avoid OOM.

        Returns:
            Dict mapping (sample_idx, img_idx) -> PIL Image (or None on failure).
        """
        results = {}
        if not indexed_grids or self.vision_tokenizer is None:
            return results

        # Group by (h, w) for batched decode
        size_groups = defaultdict(list)
        for sample_idx, img_idx, grid in indexed_grids:
            h, w = grid.shape
            size_groups[(h, w)].append((sample_idx, img_idx, grid))

        for (h, w), group in size_groups.items():
            # Process in sub-batches to avoid OOM
            for batch_start in range(0, len(group), max_batch_size):
                batch_items = group[batch_start:batch_start + max_batch_size]
                B = len(batch_items)

                try:
                    # Stack all same-size token grids into one tensor
                    batch_tensor = torch.tensor(
                        np.stack([item[2] for item in batch_items]),
                        dtype=torch.long, device=device
                    )  # (B, h, w)

                    with torch.no_grad():
                        decoded = self.vision_tokenizer.decode_code(
                            batch_tensor, shape=(B, h, w, 256)
                        ).float()  # (B, C, H_pix, W_pix)

                    # Batch convert: (B, C, H, W) -> (B, H, W, C) -> uint8 numpy
                    decoded_np = (
                        (decoded.permute(0, 2, 3, 1) + 1.0) * 127.5
                    ).clamp(0, 255).byte().cpu().numpy()

                    for i, (s_idx, i_idx, _) in enumerate(batch_items):
                        results[(s_idx, i_idx)] = Image.fromarray(decoded_np[i])

                    del batch_tensor, decoded, decoded_np

                except Exception as ex:
                    print(f"[Emu3] Batch VQ decode failed for ({h}x{w}), batch {B}: {ex}")
                    # Fallback: decode one by one without empty_cache between each
                    for s_idx, i_idx, grid in batch_items:
                        try:
                            t = torch.tensor(grid, dtype=torch.long, device=device)[None]
                            with torch.no_grad():
                                d = self.vision_tokenizer.decode_code(t, shape=(1, h, w, 256)).float()
                            d_np = ((d[0].permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
                            results[(s_idx, i_idx)] = Image.fromarray(d_np)
                            del t, d, d_np
                        except Exception:
                            results[(s_idx, i_idx)] = None

        return results

    def _decode_emu3_images_for_reward(
        self,
        response_ids: torch.Tensor,
        input_ids: torch.Tensor = None,
        eos_token_id: int = None,
    ) -> tuple[list[list[Image.Image]], list[list[Image.Image]]]:
        """
        Decode Emu3 visual tokens to PIL Images for reward function.

        Optimized with:
        1. Parallel CPU text parsing via ThreadPoolExecutor (Python regex releases GIL)
        2. Batched VQ decode grouped by image resolution (single GPU kernel per size)
        3. Single torch.cuda.empty_cache() at the end instead of per-image

        Args:
            response_ids: (batch_size, response_length) tensor of token IDs
            input_ids: (batch_size, prompt_length) tensor of token IDs (optional)
            eos_token_id: EOS token ID

        Returns:
            Tuple of (decoded_images_batch, decoded_reference_images_batch)
        """
        if not self._init_vision_tokenizer():
            print(f"[Emu3] Rank {self.rank}: Vision tokenizer not available, skipping image decoding")
            return [], []

        batch_size = response_ids.size(0)
        device = next(iter(self.vision_tokenizer.parameters())).device

        # ===== Phase 1: Remove padding from all sequences =====
        def _remove_padding(token_row):
            tokens = token_row.tolist()
            if self.pad_token_id is not None:
                return [t for t in tokens if t != self.pad_token_id]
            return tokens

        response_token_lists = [_remove_padding(response_ids[i]) for i in range(batch_size)]

        # ===== Phase 2: Text decode =====
        response_texts = [self.tokenizer.decode(t, skip_special_tokens=False) for t in response_token_lists]

        # ===== Phase 3: Parallel CPU parsing to extract VQ token grids =====
        # Python re module releases GIL, so ThreadPoolExecutor gives real speedup for regex
        num_workers = min(batch_size, 8)
        tokenizer_ref = self.tokenizer

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            response_grids_per_sample = list(executor.map(
                lambda text: _parse_image_token_grids(text, tokenizer_ref),
                response_texts
            ))

        # ===== Phase 4: Collect all grids and batch VQ decode =====
        all_grids = []
        for sample_idx, grids in enumerate(response_grids_per_sample):
            for img_idx, grid in enumerate(grids):
                all_grids.append((sample_idx, img_idx, grid))

        decoded_pil_map = self._batched_vq_decode_indexed(all_grids, device) if all_grids else {}

        # ===== Phase 5: Reconstruct per-sample image lists =====
        decoded_images_batch = []
        for sample_idx in range(batch_size):
            sample_images = []
            img_idx = 0
            while (sample_idx, img_idx) in decoded_pil_map:
                img = decoded_pil_map[(sample_idx, img_idx)]
                if img is not None:
                    sample_images.append(img)
                img_idx += 1
            decoded_images_batch.append(sample_images)

        # ===== Phase 6: Reference images (same pipeline) =====
        decoded_reference_images_batch = []
        if input_ids is not None:
            input_token_lists = [_remove_padding(input_ids[i]) for i in range(batch_size)]
            input_texts = [self.tokenizer.decode(t, skip_special_tokens=False) for t in input_token_lists]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                input_grids_per_sample = list(executor.map(
                    lambda text: _parse_image_token_grids(text, tokenizer_ref),
                    input_texts
                ))

            all_input_grids = []
            for sample_idx, grids in enumerate(input_grids_per_sample):
                for img_idx, grid in enumerate(grids):
                    all_input_grids.append((sample_idx, img_idx, grid))

            decoded_ref_map = self._batched_vq_decode_indexed(all_input_grids, device) if all_input_grids else {}

            for sample_idx in range(batch_size):
                ref_images = []
                img_idx = 0
                while (sample_idx, img_idx) in decoded_ref_map:
                    img = decoded_ref_map[(sample_idx, img_idx)]
                    if img is not None:
                        ref_images.append(img)
                    img_idx += 1
                decoded_reference_images_batch.append(ref_images)
        else:
            decoded_reference_images_batch = [[] for _ in range(batch_size)]

        total_gen_images = sum(len(imgs) for imgs in decoded_images_batch)
        total_ref_images = sum(len(imgs) for imgs in decoded_reference_images_batch)
        if total_gen_images > 0 or total_ref_images > 0:
            print(f"[Emu3] Rank {self.rank}: Decoded {total_gen_images} generated images, "
                  f"{total_ref_images} reference images from {batch_size} samples (batched VQ decode)")

        # Single cleanup at the end instead of per-image
        torch.cuda.empty_cache()

        return decoded_images_batch, decoded_reference_images_batch

    def _decode_gt_frames(self, batch_gt_frames_info: list) -> list[list[Image.Image]]:
        """
        Decode ground truth frames from raw VQ codebook indices to PIL Images.
        Optimized with batched VQ decode grouped by resolution.

        Args:
            batch_gt_frames_info: List of gt_frames_info per sample. Each element
                is a list of dicts with keys: 'image_tokens' (bytes), 'height' (int), 'width' (int).

        Returns:
            List of lists of PIL Images (one list per sample, one image per GT frame).
        """
        if not self._init_vision_tokenizer():
            print(f"[Emu3] Rank {self.rank}: Vision tokenizer not available, skipping GT frame decoding")
            return [[] for _ in batch_gt_frames_info]

        device = next(iter(self.vision_tokenizer.parameters())).device

        # Collect all GT frames with indices for batched decode
        all_grids = []
        for sample_idx, sample_frames_info in enumerate(batch_gt_frames_info):
            if not sample_frames_info:
                continue
            for frame_idx, frame_info in enumerate(sample_frames_info):
                try:
                    image_tokens_bytes = frame_info['image_tokens']
                    h = frame_info['height']
                    w = frame_info['width']
                    token_array = np.frombuffer(image_tokens_bytes, dtype=np.int64).astype(np.int32).reshape(h, w)
                    all_grids.append((sample_idx, frame_idx, token_array))
                except Exception as ex:
                    print(f"[Emu3] Rank {self.rank}: Failed to parse GT frame: {ex}")

        # Batched VQ decode (grouped by resolution internally)
        decoded_map = self._batched_vq_decode_indexed(all_grids, device) if all_grids else {}

        # Reconstruct per-sample lists
        decoded_gt_images_batch = []
        total_decoded = 0
        for sample_idx, sample_frames_info in enumerate(batch_gt_frames_info):
            sample_images = []
            if sample_frames_info:
                for frame_idx in range(len(sample_frames_info)):
                    img = decoded_map.get((sample_idx, frame_idx))
                    if img is not None:
                        sample_images.append(img)
                        total_decoded += 1
            decoded_gt_images_batch.append(sample_images)

        if total_decoded > 0:
            print(f"[Emu3] Rank {self.rank}: Decoded {total_decoded} GT frames from "
                  f"{len(batch_gt_frames_info)} samples (batched VQ decode)")

        torch.cuda.empty_cache()
        return decoded_gt_images_batch

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
    
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        # Emu3 CFG: get unconditional prompt ids if available
        batch_uncond_prompt_ids = non_tensor_batch.pop("uncond_prompt_ids", None)
        # Save gt_frames_info before non_tensor_batch is re-created later
        batch_gt_frames_info = non_tensor_batch.pop("gt_frames_info", None)
        # Pre-decoded JPEG bytes from dataset (skip VQ decode when available)
        batch_decoded_gt_bytes = non_tensor_batch.pop("decoded_gt_images_bytes", None)
        batch_decoded_ref_bytes = non_tensor_batch.pop("decoded_ref_images_bytes", None)
        
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # Build vllm inputs
        # Check if CFG should be enabled (determined at init time)
        cfg_enabled = getattr(self, 'emu3_cfg_enabled', False)
        
        if cfg_enabled:
            # Emu3 with CFG: need uncond_prompt_token_ids
            vllm_inputs = []
            for i, raw_prompt_ids in enumerate(batch_raw_prompt_ids):
                vllm_input = {"prompt_token_ids": list(raw_prompt_ids)}
                
                # Add unconditional prompt for CFG
                if batch_uncond_prompt_ids is not None:
                    vllm_input["uncond_prompt_token_ids"] = list(batch_uncond_prompt_ids[i])
                else:
                    # Generate default unconditional prompt (just BOS token)
                    # This is a simple fallback - ideally data should provide uncond_prompt_ids
                    uncond_ids = [self.tokenizer.encode(self.tokenizer.bos_token)[0]]
                    vllm_input["uncond_prompt_token_ids"] = uncond_ids
                
                vllm_inputs.append(vllm_input)
        elif batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            import time
            start = time.time()

            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=self.use_tqdm,
            )
            print(f"len(vllm_inputs), len(completions): {len(vllm_inputs)}, {len(completions)}")
            # print(f"Sample vLLM completions: {completions[0]}")
            # In CFG mode, vLLM returns 2x outputs but they may be UNORDERED!
            # request_id format: "{cond_flag}{original_index}" where cond_flag='0' for conditional, '1' for unconditional
            # We need to parse request_id to correctly filter conditional outputs and maintain order
            if cfg_enabled:
                # CFG mode: parse request_id to extract conditional outputs in correct order
                # vLLM does NOT guarantee output order, so we must parse request_ids
                num_inputs = len(vllm_inputs)
                conditional_results = {}  # original_index -> RequestOutput
                
                for completion in completions:
                    req_id = completion.request_id
                    if len(req_id) >= 2:
                        cond_flag = req_id[0]    # '0' for conditional, '1' for unconditional
                        index_str = req_id[1:]   # remaining chars are the original index
                        try:
                            original_idx = int(index_str)
                            is_conditional = (cond_flag == '0')
                            if is_conditional:
                                if original_idx not in conditional_results:
                                    conditional_results[original_idx] = completion
                                else:
                                    print(f"[WARNING] Duplicate conditional result for index {original_idx}")
                        except ValueError:
                            print(f"[WARNING] Cannot parse index from request_id: {req_id}")
                    else:
                        print(f"[WARNING] Unexpected request_id format: {req_id}")
                
                # Rebuild completions list in correct order
                ordered_completions = []
                for i in range(num_inputs):
                    if i in conditional_results:
                        ordered_completions.append(conditional_results[i])
                    else:
                        print(f"[ERROR] Missing conditional result for index {i}")
                        # Use first available completion as fallback (not ideal but prevents crash)
                        if completions:
                            ordered_completions.append(completions[0])
                
                completions = ordered_completions
                print(f"[Emu3 CFG] Extracted {len(conditional_results)} conditional outputs, reordered to {len(completions)} results")
        
            
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            
            # Sanity check
            expected_count = len(vllm_inputs) * self.sampling_params.n
            if len(response_ids) != expected_count:
                print(f"[WARNING] response_ids count ({len(response_ids)}) != expected ({expected_count})")
                # Truncate or pad as needed
                if len(response_ids) > expected_count:
                    response_ids = response_ids[:expected_count]
                else:
                    # This shouldn't happen, but handle gracefully
                    print(f"[ERROR] Not enough response_ids! Got {len(response_ids)}, expected {expected_count}")
            
            end = time.time()
            print(f"[vLLMRollout] vLLM generation time for batch_size {batch_size}: {end - start:.2f} seconds")
            
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)
                # Repeat gt_frames_info to match repeated outputs
                if batch_gt_frames_info is not None:
                    batch_gt_frames_info = _repeat_interleave(batch_gt_frames_info, self.sampling_params.n)
                if batch_decoded_gt_bytes is not None:
                    batch_decoded_gt_bytes = _repeat_interleave(batch_decoded_gt_bytes, self.sampling_params.n)
                if batch_decoded_ref_bytes is not None:
                    batch_decoded_ref_bytes = _repeat_interleave(batch_decoded_ref_bytes, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        # Log response length distribution for debugging
        if self.rank == 0:
            # Calculate actual lengths (excluding padding)
            response_lengths = []
            for i in range(response_ids.size(0)):
                tokens = response_ids[i].tolist()
                # Count non-padding tokens
                actual_len = sum(1 for t in tokens if t != self.pad_token_id)
                response_lengths.append(actual_len)
            
            response_lengths = np.array(response_lengths)
            print(f"\n[Rollout] Response length distribution (batch_size={len(response_lengths)}):")
            print(f"  Min: {response_lengths.min()}, Max: {response_lengths.max()}, Mean: {response_lengths.mean():.1f}, Std: {response_lengths.std():.1f}")
            print(f"  Percentiles - 25%: {np.percentile(response_lengths, 25):.0f}, 50%: {np.percentile(response_lengths, 50):.0f}, 75%: {np.percentile(response_lengths, 75):.0f}, 95%: {np.percentile(response_lengths, 95):.0f}")
            # Show histogram bins
            hist, bin_edges = np.histogram(response_lengths, bins=10)
            print(f"  Histogram: {list(zip([f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(len(hist))], hist.tolist()))}")

        # Emu3: Decode generated images in rollout worker (has GPU) for reward function
        # This avoids needing GPU in reward worker
        # WARNING: This is VERY memory intensive! Disable for initial training runs.
        # Set enable_image_decode_for_reward=True in config only when needed for VLM reward
        enable_image_decode = getattr(self.config, 'enable_image_decode_for_reward', False)

        # Check if pre-decoded images (JPEG bytes) are available from dataset
        _has_pre_decoded_ref = (
            batch_decoded_ref_bytes is not None
            and len(batch_decoded_ref_bytes) > 0
            and any(len(x) > 0 for x in batch_decoded_ref_bytes if x is not None)
        )
        _has_pre_decoded_gt = (
            batch_decoded_gt_bytes is not None
            and len(batch_decoded_gt_bytes) > 0
            and any(len(x) > 0 for x in batch_decoded_gt_bytes if x is not None)
        )

        if self.is_emu3 and enable_image_decode:
            import time as _time_mod
            from PIL import Image as _PILImage
            import io as _io_mod
            _decode_start = _time_mod.time()
            # Skip reference image VQ decode if pre-decoded bytes are available
            decoded_images_batch, decoded_reference_images_batch = self._decode_emu3_images_for_reward(
                response_ids, 
                input_ids=None if _has_pre_decoded_ref else input_ids,
                eos_token_id=prompts.meta_info.get("eos_token_id")
            )
            _decode_gen_time = _time_mod.time() - _decode_start
            if decoded_images_batch:
                non_tensor_batch["decoded_images"] = decoded_images_batch

            # Use pre-decoded reference images if available, otherwise use VQ-decoded ones
            if _has_pre_decoded_ref:
                _ref_start = _time_mod.time()
                decoded_reference_images_batch = []
                for sample_bytes in batch_decoded_ref_bytes:
                    sample_imgs = []
                    if sample_bytes is not None:
                        for b in sample_bytes:
                            if b is not None and len(b) > 0:
                                try:
                                    sample_imgs.append(_PILImage.open(_io_mod.BytesIO(b)).convert('RGB'))
                                except Exception:
                                    pass
                    decoded_reference_images_batch.append(sample_imgs)
                _ref_time = _time_mod.time() - _ref_start
                print(f"[Rollout Timing] Pre-decoded ref images loaded: {_ref_time:.2f}s (skipped VQ decode)")
            if decoded_reference_images_batch:
                non_tensor_batch["decoded_reference_images"] = decoded_reference_images_batch
            print(f"[Rollout Timing] Image decode (gen{'+ref' if not _has_pre_decoded_ref else ''}): {_decode_gen_time:.2f}s")

        # Emu3: Decode ground truth frames for Pref-GRPO pairwise reward
        if self.is_emu3 and enable_image_decode:
            if _has_pre_decoded_gt:
                # Use pre-decoded GT images (JPEG bytes → PIL), skip VQ decode
                _gt_start = _time_mod.time()
                decoded_gt_images = []
                for sample_bytes in batch_decoded_gt_bytes:
                    sample_imgs = []
                    if sample_bytes is not None:
                        for b in sample_bytes:
                            if b is not None and len(b) > 0:
                                try:
                                    sample_imgs.append(_PILImage.open(_io_mod.BytesIO(b)).convert('RGB'))
                                except Exception:
                                    pass
                    decoded_gt_images.append(sample_imgs)
                _gt_time = _time_mod.time() - _gt_start
                non_tensor_batch["decoded_gt_images"] = decoded_gt_images
                print(f"[Rollout Timing] Pre-decoded GT images loaded: {_gt_time:.2f}s (skipped VQ decode)")
            elif batch_gt_frames_info is not None:
                # Fallback: decode GT frames from raw VQ tokens using GPU
                _gt_start = _time_mod.time()
                decoded_gt_images = self._decode_gt_frames(batch_gt_frames_info)
                _gt_time = _time_mod.time() - _gt_start
                non_tensor_batch["decoded_gt_images"] = decoded_gt_images
                print(f"[Rollout Timing] GT frames VQ decode: {_gt_time:.2f}s")

        # Emu3: Compute DINOv2 feature similarity scores in rollout (has GPU)
        # These pre-computed scores are passed to the reward function via non_tensor_batch
        # so the reward worker (Ray actor without GPU) can directly use them.
        _dinov2_time = 0.0
        enable_dinov2_in_rollout = getattr(self.config, 'enable_dinov2_in_rollout', False)
        if self.is_emu3 and enable_image_decode and enable_dinov2_in_rollout:
            _dinov2_start = _time_mod.time()
            _gen_imgs = non_tensor_batch.get("decoded_images", None)
            _gt_imgs = non_tensor_batch.get("decoded_gt_images", None)
            if _gen_imgs is not None and _gt_imgs is not None:
                local_rank = int(os.getenv("LOCAL_RANK", "0"))
                _dino_device = f"cuda:{local_rank}"
                _dino_model = getattr(self.config, 'dinov2_model_name',
                    "")
                _dino_pooling = getattr(self.config, 'dinov2_pooling', "avg_patch")
                _dino_metric = getattr(self.config, 'dinov2_metric', "gaussian_rbf")
                _dino_sigma = getattr(self.config, 'dinov2_rbf_sigma', 3.0)
                dinov2_scores = _compute_dinov2_scores_batch(
                    _gen_imgs, _gt_imgs,
                    model_name=_dino_model,
                    device=_dino_device,
                    pooling=_dino_pooling,
                    metric=_dino_metric,
                    rbf_sigma=_dino_sigma,
                )
                non_tensor_batch["dinov2_scores"] = dinov2_scores
                _n_valid = sum(1 for s in dinov2_scores if s > 0)
                print(f"[DINOv2-Rollout] Computed scores for {batch_size} samples "
                      f"({_n_valid} valid pairs, pooling={_dino_pooling}, "
                      f"metric={_dino_metric}, sigma={_dino_sigma})")
            else:
                print(f"[DINOv2-Rollout] Skipped: decoded_images or decoded_gt_images not available")
            _dinov2_time = _time_mod.time() - _dinov2_start

        # Print rollout timing summary
        _rollout_total = time.time() - start  # 'start' and 'time' already imported earlier in this method
        print(f"\n{'='*80}")
        print(f"[Rollout Timing Summary] batch_size={batch_size} | vLLM generation: {end - start:.2f}s | Image decode: {_decode_gen_time if self.is_emu3 and enable_image_decode else 0:.2f}s | DINOv2: {_dinov2_time:.2f}s | Total generate_sequences: {_rollout_total:.2f}s")
        print(f"{'='*80}")

        # [Memory Fix] Offload VQ decoder to CPU after image decoding.
        # The VQ decoder (~1.7GB) is only needed during rollout for decoding images.
        # Keeping it on GPU during FSDP training wastes precious memory.
        if self.is_emu3 and self.vision_tokenizer is not None:
            self.offload_vision_tokenizer()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
