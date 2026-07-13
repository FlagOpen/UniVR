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

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        
        # Check if decoded_images are available (from rollout worker for Emu3)
        decoded_images_batch = data.non_tensor_batch.get("decoded_images", None)
        decoded_reference_images_batch = data.non_tensor_batch.get("decoded_reference_images", None)
        decoded_gt_images_batch = data.non_tensor_batch.get("decoded_gt_images", None)
        # Pre-computed DINOv2 scores from rollout worker (has GPU)
        dinov2_scores_batch = data.non_tensor_batch.get("dinov2_scores", None)
        
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
                "uid": data.non_tensor_batch["uid"][i] if "uid" in data.non_tensor_batch else f"sample_{i}",
                "dataset_source": data.non_tensor_batch["dataset_source"][i] if "dataset_source" in data.non_tensor_batch else "",
            }
            
            # Add decoded images if available (for Emu3 VLM reward)
            if decoded_images_batch is not None and i < len(decoded_images_batch):
                reward_input["decoded_images"] = decoded_images_batch[i]
            
            # Add decoded reference images if available (from input/prompt)
            if decoded_reference_images_batch is not None and i < len(decoded_reference_images_batch):
                reward_input["decoded_reference_images"] = decoded_reference_images_batch[i]
            
            # Add decoded ground truth images if available (for Pref-GRPO pairwise reward)
            if decoded_gt_images_batch is not None and i < len(decoded_gt_images_batch):
                reward_input["decoded_gt_images"] = decoded_gt_images_batch[i]
            
            # Add pre-computed DINOv2 score from rollout worker (avoids needing GPU in reward)
            if dinov2_scores_batch is not None and i < len(dinov2_scores_batch):
                reward_input["precomputed_dinov2_score"] = dinov2_scores_batch[i]
            
            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")
        
        # If tokenizer is not provided directly, load it from path
        # This avoids pickle issues with custom tokenizers (e.g., Emu3)
        if tokenizer is None and hasattr(config, 'tokenizer_path') and config.tokenizer_path is not None:
            from ...utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(
                config.tokenizer_path,
                trust_remote_code=getattr(config, 'tokenizer_trust_remote_code', True),
                use_fast=getattr(config, 'tokenizer_use_fast', True),
            )
            print(f"[AutoRewardManager] Loaded tokenizer from path: {config.tokenizer_path}")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
