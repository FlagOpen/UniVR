# UniVR_SFT — Supervised Fine-Tuning

This module implements the **cold initialization** stage of UniVR: supervised fine-tuning (SFT) of Emu3.5 on the VR-X training set. The trained checkpoint serves as the starting point for the subsequent VR-GRPO reinforcement learning stage.

---

## Function Overview

UniVR_SFT fine-tunes Emu3.5 on a curated mixture of 310k visual reasoning samples from VR-X, spanning:

- **Long-horizon manipulation**: robotic tasks (tying knots, folding/hanging clothes, restocking shelves), cooking, and handcrafting.
- **General visual reasoning**: maze navigation, visual search, spatial perception, and image editing.

All training samples share a unified format: `[query image, textual instruction, visual reasoning trajectory]`. This stage trains the model to generate multi-step visual reasoning traces purely from visual demonstrations, without requiring dense textual annotations or image-text pairs.

Two training modes are supported:

- **LoRA fine-tuning** (`train_sft_lora.sh`) — parameter-efficient, suitable for limited GPU budgets.
- **Full parameter fine-tuning** (`train_sft_full.sh`) — used for the final UniVR model reported in the paper.

Both modes use **DeepSpeed ZeRO-3** and `torchrun` for distributed multi-node training.

---

## Data

### VR-X Dataset (UniVR Training Data)

Download the VR-X dataset from [ByteDance/VR-X-SFT-RL](https://huggingface.co/datasets/ByteDance/VR-X-SFT-RL) and place it under the repo root as `datasets/VR-X/`:

```bash
huggingface-cli download ByteDance/VR-X-SFT-RL --repo-type dataset --local-dir datasets/VR-X
```

```
UniVR/
└── datasets/
    └── VR-X/
        ├── Agibot/          # task_327.parquet, task_351.parquet, ...
        ├── Bridge/
        ├── Droid/
        ├── EgoDex/
        ├── Epic_Kitchen/
        ├── Spatial_Navigation/
        ├── VideoCraftBench/
        ├── Visual_Search/
        └── ...
```

Each subdirectory corresponds to one data source. Data is stored as **per-task parquet files**, where each row is one training sample.

### Parquet Schema

Each parquet row must contain the following fields:

| Column | Type | Description |
|---|---|---|
| `question` | `str` | Text instruction for the task (e.g. `"Tie the red rope around the white gift box. Finish this task in 3 steps."`) |
| `problem_image_bytes` | `list[bytes]` | JPEG-encoded query/input frames (usually 1 frame) |
| `answer_image_bytes` | `list[bytes]` | JPEG-encoded answer/output frames (the visual reasoning trajectory) |
| `problem_images` | `list[dict]` | Per-frame metadata with a `"caption"` field for each query frame |
| `answer_images` | `list[dict]` | Per-frame metadata with a `"caption"` field for each answer frame |
| `height` | `int` | VQ token grid height (e.g. `32` → 512 px at 16× downsampling) |
| `width` | `int` | VQ token grid width (e.g. `40` → 640 px) |
| `global_summary` | `str` | Optional text summary of the full trajectory |

### Configuring Data Paths

Data sources and sampling caps are configured directly in `train.py` via the `dataset_cfg` dict:

```python
dataset_cfg = {
    "Agibot": {"max_samples_per_task": 300, "enabled": True},
    "EgoDex": {"max_samples_per_task": 350, "enabled": True},
    "Epic_Kitchen": {"max_samples": 5000, "enabled": True},
    # Add your own source here...
}
```

The dataset class looks up each key in `DATASET_DIRS` (defined in `src/utils/dataset_vr_train.py`) to resolve the subdirectory path under a common data root. By default `DATA_ROOT` points to `datasets/VR-X/` relative to the repo root; update it in that file if your data lives elsewhere.

### Using Your Own Data

To train on your own data, you have two options:

**Option A — Add a new parquet source**: Create parquet files with the schema above, place them in a new subdirectory, register the subdirectory name in `DATASET_DIRS`, and add an entry to `dataset_cfg` in `train.py`.

**Option B — Implement a custom Dataset**: Replace the `VRTrainDataset` instantiation in `train.py` with your own `torch.utils.data.Dataset` subclass that returns tokenized samples in the same format. No other changes are needed.

---

## Training

### Prerequisites

Run the environment setup from the repository root before training:

```bash
cd UniVR
bash install.sh
```

Key dependencies: `torch==2.8.0`, `transformers==4.57.3`, `vllm==0.11.0`, `flash-attn==2.8.3`, `deepspeed`.

The install script also applies the required vLLM source patches (`src/patch/apply.py` and `src/patch/apply_no_cfg_support.py`) for Emu3.5 compatibility.

### Download Model Weights

Download the following weights into `UniVR_SFT/weights/` before training or inference:

```bash
mkdir -p UniVR_SFT/weights
# Emu3.5 base model
huggingface-cli download BAAI/Emu3.5 --local-dir UniVR_SFT/weights/Emu3.5
# Emu3.5 VisionTokenizer
huggingface-cli download BAAI/Emu3.5-VisionTokenizer --local-dir UniVR_SFT/weights/Emu3.5-VisionTokenizer
```

| Weight | HuggingFace | Local path |
|---|---|---|
| Emu3.5 (base model) | [BAAI/Emu3.5](https://huggingface.co/BAAI/Emu3.5) | `weights/Emu3.5` |
| Emu3.5-VisionTokenizer | [BAAI/Emu3.5-VisionTokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer) | `weights/Emu3.5-VisionTokenizer` |

Set `MODEL_PATH` (training scripts) and `vq_path` / `model_path` (inference config) to these local paths accordingly.

### Configuration

Edit the relevant launch script to configure paths and hyperparameters before running:

| Variable | Description |
|---|---|
| `MODEL_PATH` | Path to the base Emu3.5 checkpoint |
| `TOKENIZER_PATH` | Path to `src/tokenizer_emu3_ibq` |
| `DS_CONFIG` | DeepSpeed config file (default: `ds_config_zero3.json`) |
| `OUTPUT_DIR` | Directory for saving checkpoints and logs |
| `WANDB_PROJECT` / `WANDB_RUN_NAME` | WandB experiment tracking |
| `NNODES` / `GPUS_PER_NODE` | Cluster topology (auto-detected from Arnold env vars) |

### LoRA Fine-Tuning

Recommended for rapid experimentation. Trains a low-rank adapter (rank 128) on the attention projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`) while keeping the base model frozen.

```bash
bash scripts/train_sft_lora.sh
```

Key training arguments used:

```
--use_lora True
--lora_r 128  --lora_alpha 256  --lora_dropout 0.1
--lora_target_modules "q_proj,v_proj,k_proj,o_proj"
--learning_rate 1e-5
--num_train_epochs 10
--per_device_train_batch_size 1
--gradient_accumulation_steps 4
--lr_scheduler_type cosine  --warmup_ratio 0.05
--max_grad_norm 5.0  --weight_decay 0.1
--bf16 True  --gradient_checkpointing True
```

Default cluster: **2 nodes × 8 GPUs** (16 GPUs total).

### Full Parameter Fine-Tuning

Trains all model parameters end-to-end. This is the setting used to produce the final UniVR checkpoint.

```bash
bash scripts/train_sft_full.sh
```

Default cluster: **4 nodes × 8 GPUs** (32 GPUs total). All other hyperparameters are identical to LoRA mode.

### Training Logs

Checkpoints are saved to `outputs/<run_name>/` every `--save_steps` steps. Logs are written to `outputs/<run_name>/train_<timestamp>.log` and simultaneously printed to stdout via `tee`.

---

## Inference & Evaluation

### Pretrained Checkpoints

We provide two pretrained UniVR checkpoints ready for inference and RL fine-tuning:

| Model | HuggingFace | Best For |
|---|---|---|
| **UniVR-34B-Planning** | [ByteDance/UniVR-34B-Planning](https://huggingface.co/ByteDance/UniVR-34B-Planning) | Long-horizon planning tasks (robotic manipulation, tool use, multi-step control) |
| **UniVR-34B-General** | [ByteDance/UniVR-34B-General](https://huggingface.co/ByteDance/UniVR-34B-General)| General Tasks |

**UniVR-34B-Planning** is trained exclusively on manipulation and planning data for maximum performance on long-horizon robotic tasks.

**UniVR-34B-General** follows the full UniVR training recipe from the paper but additionally incorporates interleaved image-text data, preserving Emu3.5's original interleaved generation capability. This makes it more suitable for general visual reasoning tasks and Visual Guidance tasks compared to UniVR-34B-Planning.

To use either checkpoint, set `model_path` in `configs/config.py` to the downloaded checkpoint directory.

---

Inference is config-driven. A single config file specifies the model path, prompts, and generation settings.

### Quick Start

```bash
cd UniVR_SFT
bash scripts/inference.sh
```

This runs:

```bash
python3 inference.py --cfg ./configs/config.py
```

### Inference Configuration

Edit `configs/config.py` to customize inference:

| Field | Description |
|---|---|
| `model_path` | Path to the fine-tuned checkpoint to evaluate |
| `vq_path` | Path to `weights/Emu3.5-VisionTokenizer` |
| `tokenizer_path` | Path to `src/tokenizer_emu3_ibq` |
| `task_type` | Task category, e.g. `"vla"` (robotic manipulation) or `"visual reasoning"` |
| `input_mode` | `"manual"` for hand-crafted prompts, `"parquet"` for batch evaluation |
| `classifier_free_guidance` | CFG scale (e.g. `3.0`; set to `1.0` to disable) |
| `max_new_tokens` | Maximum tokens to generate per sample (default: 15000) |
| `save_path` | Output directory for generated results |

**Manual mode** — fill in the `_prompts_base` list in `config.py`, where each entry is a dict:

```python
{
    "prompt": "Tie the red rope around the white gift box. Finish this task in 3 steps.",
    "reference_image": "path/to/first_frame.jpg",
}
```

**Parquet mode** — set `input_mode = "parquet"` and populate `PARQUET_CONFIGS` with paths to pre-tokenized evaluation parquet files.

### Sampling Parameters

Adjust generation behavior in the `sampling_params` dict in `config.py`:

```python
sampling_params = dict(
    text_top_k=1024,    text_top_p=0.9,   text_temperature=1.0,
    image_top_k=10240,  image_top_p=1.0,  image_temperature=1.0,
    use_differential_sampling=True,
)
```
