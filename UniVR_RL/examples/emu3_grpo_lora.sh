#!/bin/bash
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

set -x

# ==================== Configuration ====================

# Base path for UniVR_SFT (used by config_emu3.yaml via ${oc.env:UNIVR_SFT_PATH})
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UNIVR_SFT_PATH="$(cd "${SCRIPT_DIR}/../../UniVR_SFT" && pwd)"

# Emu3 Tokenizer Path
TOKENIZER_PATH="${UNIVR_SFT_PATH}/src/tokenizer_emu3_ibq"

# Emu_VW source path (for importing Emu3 modules)
EMU_SRC_PATH="${UNIVR_SFT_PATH}/src"

# Training data path
TRAIN_DATA=""  # Set your training data path
VAL_DATA=""    # Set your validation data path

# ==================== Environment Setup ====================
# Add Emu_VW/src to Python path for Emu3 model imports
export PYTHONPATH="${EMU_SRC_PATH}:${PYTHONPATH}"

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# WANDB Configuration
EXPERIMENT_NAME="univr_vrgrpo_rl_lora"
export WANDB_PROJECT="UniVR_EasyR1_Project"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}"


LOCAL_MODEL="path to your local model checkpoint, emu3.5 or cold-start checkpoint"  # Set your local model checkpoint path
LOCAL_MODEL="/opt/tiger/Emu3.5"
# ==================== Multi-Node Configuration ====================
# Multi-node training parameters
NNODES=2              # number of nodes (default 1)
NODE_RANK=${ARNOLD_ID}              # rank of the current node (0, 1, 2, ...)
MASTER_ADDR=${ARNOLD_WORKER_0_HOST} # master node IP address
MASTER_PORT=${ARNOLD_WORKER_0_PORT}       # master node port
GPUS_PER_NODE=8       # number of GPUs per node
VLLM_PATH="http://<your-vllm-server>/v1"

# ==================== Logging Setup ====================
# Save logs to checkpoint directory and also output to stdout
LOG_DIR="checkpoints/${WANDB_PROJECT}/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Training log will be saved to: ${LOG_FILE}"


python3 -m verl.trainer.main \
    config=examples/config_emu3.yaml \
    worker.actor.model.model_path=${LOCAL_MODEL} \
    worker.actor.model.tokenizer_path=${TOKENIZER_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.padding_free=false \
    worker.rollout.tensor_parallel_size=8 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.max_num_seqs=128 \
    worker.rollout.enable_image_decode_for_reward=true \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    worker.actor.model.lora.rank=64 \
    worker.actor.model.lora.alpha=128 \
    trainer.n_gpus_per_node=8 \
    data.rollout_batch_size=8 \
    worker.actor.global_batch_size=8 \
    worker.reward.reward_function_kwargs.vlm_api_base=${VLLM_PATH} \
    worker.reward.val_reward_function_kwargs.vlm_api_base=${VLLM_PATH} \
    worker.reward.reward_function_kwargs.max_vlm_workers=128 \
    worker.rollout.n=6 \
    trainer.nnodes=1 \
    2>&1 | tee -a "${LOG_FILE}"
