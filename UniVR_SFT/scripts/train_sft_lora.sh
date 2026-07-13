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

MODEL_PATH="/opt/tiger/Emu3.5"

TOKENIZER_PATH="src/tokenizer_emu3_ibq"
DS_CONFIG="ds_config_zero3.json"
REPORT_TO="wandb" # Options: "wandb", "tensorboard", "none"

# WandB Configuration
WANDB_PROJECT="Emu3_VW_Project"   # customize your wandb project name
WANDB_RUN_NAME="emu3_lora_manual7" # customize your experiment run name
export WANDB_PROJECT=$WANDB_PROJECT

OUTPUT_DIR="outputs/${WANDB_RUN_NAME}"
# Multi-node configuration
# Set these environment variables on each node
# MASTER_ADDR="192.168.1.1"
# MASTER_PORT=9901
# NNODES=2
# NODE_RANK=0 # 0 for master, 1 for worker 1, etc.

# Default to single node if not set
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
# MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT:-40103}
NNODES=1
NODE_RANK=$ARNOLD_ID
GPUS_PER_NODE=$ARNOLD_WORKER_GPU

echo "Training on ${NNODES} nodes, rank ${NODE_RANK}, master ${MASTER_ADDR}:${MASTER_PORT}"
echo "GPUs per node: ${GPUS_PER_NODE}"

# Check if wandb is available if requested
if [ "$REPORT_TO" = "wandb" ]; then
    if ! python -c "import wandb" &> /dev/null; then
        echo "Warning: wandb not installed. Switching report_to to 'none'."
        REPORT_TO="none"
    fi
fi

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Log file: save all stdout/stderr to a timestamped log in output_dir
LOG_FILE="${OUTPUT_DIR}/train_$(date '+%Y%m%d_%H%M%S').log"
echo "Logging to: $LOG_FILE"

# Run training with torchrun for distributed support
# Use tee to write stdout+stderr to both terminal and log file
torchrun \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
    --debug_mode False \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 100 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
    --bf16 True \
    --deepspeed $DS_CONFIG \
    --report_to $REPORT_TO \
    --run_name "$WANDB_RUN_NAME" \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --max_grad_norm 5.0 \
    --weight_decay 0.1 \
    --logging_nan_inf_filter True \
    --loss_spike_threshold 3.0 \
    --loss_spike_delta_threshold 4.0 \
    --grad_norm_threshold 10.0 \
    --skip_anomaly_batches False \
    --seed 42 \
    2>&1 | tee -a "$LOG_FILE"

