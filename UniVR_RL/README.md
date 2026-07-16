# UniVR_RL — Reinforcement Learning (VR-GRPO)

This module implements the **VR-GRPO** reinforcement learning stage of UniVR. Starting from a cold-start SFT checkpoint produced by [UniVR_SFT](../UniVR_SFT/README.md), it applies GRPO-based RL with complementary global and step-focal rewards to improve logical coherence and physical consistency across diverse visual reasoning tasks.

---

## Function Overview

VR-GRPO trains the model to autonomously explore optimal visual reasoning policies without requiring dense image-text supervision. The reward signal consists of:

- **Format reward** ($R_{\text{format}}$): enforces structural constraints such as uniform frame resolution and adherence to the number of steps specified in the task instruction.
- **Global reward** ($R_g$): a VLM evaluator (Qwen3-VL-30B) assesses overall task completion and visual quality of each rollout trajectory against a ground-truth reference, using a pairwise comparison protocol.
- **Step-focal reward** ($R_s$): identifies the most uncertain/divergent sub-steps across rollout trajectories via CLIP-feature variance analysis, then focuses the VLM evaluator on those critical windows for fine-grained assessment.
- **Combined reward**: $R_{\text{reason}} = R_g - \lambda |R_g - R_s|$, which prevents reasoning shortcuts by requiring both terminal success and procedural coherence.

The RL pipeline is built on the **verl** framework (a fork of EasyR1/veRL with HybridEngine for efficient rollout and training).

### vLLM Source Patch for Emu3.5 No-CFG Parallel Inference

To support Emu3.5 during RL rollout, we apply a custom vLLM source patch that enables **no-CFG (classifier-free guidance) mode** for accelerated parallel inference:

```bash
python src/patch/apply_no_cfg_support.py
```

This patch modifies vLLM's `ClassifierFreeGuidanceLogitsForVisualTokenProcessor` to handle single (non-paired) requests, enabling ~2× throughput compared to standard CFG mode where each request requires a paired conditional + unconditional forward pass. It also adds LoRA support for Emu3.5 by extending the vocab size limit and fixing embedding tensor checks. The patch is applied automatically by `install.sh`; re-run it manually if vLLM is reinstalled.

---

## Data

### VR-X-RL Dataset (UniVR RL Training Data)

Download the VR-X-RL dataset from [ByteDance/VR-X-SFT-RL](https://huggingface.co/datasets/ByteDance/VR-X-SFT-RL) and place it under the repo root as `datasets/VR-X-RL/`:

```bash
huggingface-cli download ByteDance/VR-X-SFT-RL --repo-type dataset --local-dir datasets/VR-X-RL
```

```
UniVR/
└── datasets/
    └── VR-X-RL/
        ├── Agibot/          # task_327.parquet, task_351.parquet, ...
        ├── EgoDex/
        ├── Epic_Kitchen/
        ├── VideoCraftBench/
        ├── Visual_Search/
        ├── Zebra_Count/
        └── Zebra_Jigsaw/
```

Each subdirectory corresponds to one data source. Data is stored as **per-task parquet files**, one row per sample.

### Parquet Schema

Each parquet row must contain the following fields:

| Column | Type | Description |
|---|---|---|
| `vlm_task_instruction` | `str` | Text instruction for the task |
| `problem_image_bytes` | `list[bytes]` | JPEG-encoded query/input frames |
| `answer_image_bytes` | `list[bytes]` | JPEG-encoded ground-truth answer frames (used by the global reward) |
| `height` | `int` | VQ token grid height |
| `width` | `int` | VQ token grid width |
| `sub_tasks` | `list[dict]` | Sub-task annotations used by the step-focal reward (each entry has `duration`, `start`, `end`, `description`) |
| `frame_subtask_mapping` | `list[dict]` | Per-frame sub-task assignment (used to identify critical windows) |
| `vlm_frame_captions` | `list[str]` | Per-frame text captions (optional, used for context in reward evaluation) |
| `global_summary` | `str` | Reference summary for the global VLM reward evaluator |

> `sub_tasks` and `frame_subtask_mapping` are only required if you use the step-focal reward ($R_s$). If you replace the reward with a simpler function, these columns are not needed.

### Configuring Data Paths

Data paths are configured in `examples/config_emu3.yaml` under `data.train_files` as a dict mapping source type names to local directories:

```yaml
data:
  train_files:
    agibot:       datasets/VR-X-RL/Agibot
    egodex:       datasets/VR-X-RL/EgoDex
    epic_kitchen: datasets/VR-X-RL/Epic_Kitchen
    # Add your own source here...
```

Paths are resolved relative to the repo root (`UniVR/`). If your data is elsewhere, use the corresponding relative or absolute path.

### Using Your Own Data

To run RL on your own data:

1. Create parquet files with at least `vlm_task_instruction`, `problem_image_bytes`, `answer_image_bytes`, `height`, and `width`.
2. Place them in a directory and add a new key/path entry under `data.train_files` in `config_emu3.yaml`.
3. Replace the reward server (`VLLM_PATH` in `emu3_grpo.sh`) with your own reward endpoint. The rollout engine calls this server with the decoded generated images and the ground-truth reference — implement any scoring logic you need there.

The GRPO training loop, vLLM rollout engine, and Emu3.5 backend require no further changes.

---

## Training

### Prerequisites

Run the full environment setup from the repository root:

```bash
cd UniVR
bash install.sh
```

This installs `vllm==0.11.0`, `torch==2.8.0`, `transformers==4.57.3`, `flash-attn==2.8.3`, applies the vLLM patches for Emu3.5, and installs the verl RL framework.

### Configuration

Before launching, edit `examples/emu3_grpo_lora.sh` to set the following shell variables:

| Variable | Description |
|---|---|
| `LOCAL_MODEL` | Path to the SFT cold-start checkpoint (or base Emu3.5) |
| `TOKENIZER_PATH` | Auto-resolved from `UNIVR_SFT_PATH`; override if needed |
| `TRAIN_DATA` | Path to the RL training data directory (e.g. `datasets/VR-X-RL`) |
| `VAL_DATA` | Path to the validation data directory (optional) |
| `VLLM_PATH` | **HTTP endpoint of the reward server** (e.g. `http://<host>:9270/v1`). This is passed to `worker.reward.reward_function_kwargs.vlm_api_base` and `worker.reward.val_reward_function_kwargs.vlm_api_base` — changing it is the primary way to swap in a custom reward function. |
| `EXPERIMENT_NAME` | Experiment name (used for WandB run name and checkpoint directory) |
| `NNODES` / `GPUS_PER_NODE` | Cluster topology |

Key training arguments passed inline to `verl.trainer.main` (override `examples/config_emu3.yaml` defaults):

| Argument | Default in script | Description |
|---|---|---|
| `worker.actor.model.lora.rank` | `64` | LoRA rank; set to `0` for full-parameter RL |
| `worker.actor.model.lora.alpha` | `128` | LoRA alpha (typically `2 × rank`) |
| `worker.rollout.n` | `6` | Rollout samples per prompt — GRPO group size |
| `data.rollout_batch_size` | `8` | Prompts per rollout step |
| `worker.actor.global_batch_size` | `8` | Global batch size for actor update |
| `worker.rollout.tensor_parallel_size` | `8` | Tensor parallelism for vLLM rollout |
| `worker.rollout.gpu_memory_utilization` | `0.7` | vLLM GPU memory fraction |
| `worker.rollout.max_num_seqs` | `128` | Max concurrent sequences in vLLM |
| `worker.rollout.enable_image_decode_for_reward` | `true` | Decode image tokens to pixels before sending to reward server |
| `worker.reward.reward_function_kwargs.max_vlm_workers` | `128` | Parallel workers calling the VLM reward server |

### Launch Training

```bash
bash examples/emu3_grpo_lora.sh
```

Training is launched via `python3 -m verl.trainer.main` (single-node) or `torchrun` (multi-node). Checkpoints and logs are saved to `checkpoints/<WANDB_PROJECT>/<EXPERIMENT_NAME>/`.

Example log output path:

```
checkpoints/Emu3_VW_EasyR1_Project/emu3_5_pair_grpo_full_Mix1k_lr6_nodino_0325/train_20260518_120000.log
```

---

## Inference & Evaluation

For inference on the RL-trained checkpoint, reuse the SFT inference pipeline:

```bash
cd ../UniVR_SFT
bash scripts/inference.sh
```

Update `configs/config.py` to point `model_path` at the RL checkpoint directory. See [UniVR_SFT/README.md](../UniVR_SFT/README.md#inference--evaluation) for full configuration details.
