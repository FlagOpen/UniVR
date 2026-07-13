# Emu3.5 Training Guide

This guide explains how to fine-tune Emu3.5 using LoRA (Low-Rank Adaptation) for a low-cost training solution.

## Prerequisites

Install the required packages:

```bash
pip install -r requirements_train.txt
```

## Training Script

We provide a `train.py` script that uses Hugging Face `Trainer` and `peft` library.

### Usage

You can run the training script directly or use the provided shell script.

```bash
bash scripts/train_lora.sh
```

### Multi-Node Distributed Training

The updated `scripts/train_lora.sh` supports multi-node training using `torchrun` and DeepSpeed ZeRO-3.

To run on multiple nodes:

1.  **Set Environment Variables**: On each node, set the following environment variables:
    *   `MASTER_ADDR`: IP address of the master node (rank 0).
    *   `MASTER_PORT`: Port for communication (default 9901).
    *   `NNODES`: Total number of nodes.
    *   `NODE_RANK`: Rank of the current node (0, 1, 2, ...).

2.  **Run the Script**: Execute `bash scripts/train_lora.sh` on each node.

Example for 2 nodes:

**Node 0 (Master):**
```bash
export MASTER_ADDR=192.168.1.100
export NNODES=2
export NODE_RANK=0
bash scripts/train_lora.sh
```

**Node 1 (Worker):**
```bash
export MASTER_ADDR=192.168.1.100
export NNODES=2
export NODE_RANK=1
bash scripts/train_lora.sh
```

### Custom Data

To use your own data, prepare a JSON or JSONL file where each line is a JSON object with a "text" field (or modify the script to match your data format).

Example `data.jsonl`:
```json
{"text": "User: Describe this image. <|image|> ... Assistant: This is a cat."}
{"text": "User: Write a poem. Assistant: ..."}
```

Then run:

```bash
python train.py \
    --model_name_or_path checkpoints/Emu3.5 \
    --data_path path/to/your/data.jsonl \
    --output_dir outputs/my_experiment \
    ...
```

### LoRA Configuration

You can adjust LoRA parameters via command line arguments:

- `--lora_r`: Rank of the LoRA matrices (default: 8)
- `--lora_alpha`: Scaling factor (default: 32)
- `--lora_target_modules`: Modules to apply LoRA to (default: "q_proj,v_proj")

## Note on Multimodal Training

The current `train.py` implements a basic text-based training loop. Emu3.5 is a multimodal model. To train on images:
1. You need to tokenize images using the VQ-VAE model.
2. Interleave image tokens with text tokens.
3. Ensure the `Emu3Tokenizer` handles the special tokens correctly.

The provided `train.py` assumes the input "text" already contains the necessary tokens or is pure text. For full multimodal training, you would need to expand the data loading logic to process images and convert them to tokens using the visual tokenizer.
