# UniVR: SFT & RL Training Framework for Emu3.5 Series Unified Models

<p align="center">
  <img src="UniVR_SFT/assets/Fig1_v1.png" alt="UniVR Overview" width="90%">
</p>


---

## What is This?

**UniVR is an end-to-end SFT and Reinforcement Learning (GRPO) training framework specifically built for [Emu3.5](https://github.com/baaivision/Emu3) series unified generative models.** It is designed to be easily adapted for your own tasks — you only need to swap in your own data loader and RL reward function.

Key framework features:

- **SFT** (`UniVR_SFT`): Multi-node distributed supervised fine-tuning of Emu3.5, supporting both LoRA and full-parameter training via DeepSpeed ZeRO-3.
- **RL** (`UniVR_RL`): GRPO-based reinforcement learning built on the [verl](https://github.com/volcengine/verl) framework with a custom HybridEngine for efficient rollout and training.
- **Emu3.5 vLLM support**: Custom vLLM source patches enabling fast no-CFG parallel inference for Emu3.5 during RL rollout, achieving ~2× throughput over standard CFG mode. LoRA support for Emu3.5 in vLLM is also included.
- **Bring your own task**: Replace the data loader (SFT) and the reward function (RL) to train on any visual task using Emu3.5 as the backbone.

---

## Customizing for Your Own Task

### Step 1 — Replace the Data Loader (SFT)

The SFT stage reads training samples in a unified format: `[query image, textual instruction, visual reasoning trajectory]`. To use your own data, implement a custom PyTorch `Dataset` that returns samples in this format and plug it into `UniVR_SFT/train.py`.

No changes to the training loop, DeepSpeed config, or model code are needed.

### Step 2 — Replace the Reward Function (RL)

The RL stage calls a reward server via HTTP to score rollout trajectories. To use your own reward:

1. Implement your reward logic as an HTTP server (see `UniVR_RL/verl/` for the existing VLM-based reward server reference).
2. Set `VLLM_PATH` in `UniVR_RL/examples/emu3_grpo_lora.sh` to point to your reward server endpoint.
3. Optionally adjust `worker.rollout.enable_image_decode_for_reward` in `examples/config_emu3.yaml` depending on whether your reward consumes raw image tokens or decoded pixel images.

The GRPO training loop, rollout engine, and Emu3.5 vLLM backend remain unchanged.

---

## Research Context (UniVR)

This framework was developed for **UniVR**, a system that learns complex visual reasoning, fine-grained physical dynamics, and long-term planning from pure visual demonstrations — without relying on dense image-text pairs or task-specific heuristics.

UniVR uses **VR-GRPO**, a reinforcement learning paradigm combining:
- **Format reward** ($R_{\text{format}}$): enforces structural constraints (uniform resolution, correct step count).
- **Global reward** ($R_g$): a VLM evaluator (Qwen3-VL-30B) assesses overall task completion via pairwise comparison.
- **Step-focal reward** ($R_s$): identifies the most uncertain sub-steps via CLIP-feature variance across rollout trajectories and applies fine-grained VLM evaluation on those critical windows.
- **Combined reward**: $R_{\text{reason}} = R_g - \lambda |R_g - R_s|$

Training and evaluation use **VR-X**, a benchmark of 1.5M raw samples from 16 diverse sources spanning manipulation, spatial puzzles, and physical reasoning.

---

## Architecture

UniVR adopts **Emu3.5** (34B) as its backbone — a unified generative model that tokenizes images and text into a shared discrete vocabulary via a VQ-VAE-style encoder. The model autoregressively generates visual reasoning traces (future frames) given an image sequence and instruction, with no intermediate text chain.

Training is a two-stage pipeline:

| Stage | Module | Data | Description |
|---|---|---|---|
| 1. Cold Initialization | `UniVR_SFT` | 310k VR-X samples | SFT to instill visual reasoning priors |
| 2. Reinforcement Learning | `UniVR_RL` | 3k curated samples | VR-GRPO with composite reward |

---

## Repository Structure

```
UniVR/
├── install.sh            # One-shot environment setup for both subprojects
├── UniVR_SFT/            # Supervised Fine-Tuning (cold initialization)
│   ├── train.py          # ← Replace dataset here for your task
│   ├── inference.py
│   ├── scripts/
│   │   ├── train_sft_lora.sh
│   │   ├── train_sft_full.sh
│   │   └── inference.sh
│   ├── configs/          # Inference configs
│   └── src/
│       ├── patch/        # vLLM source patches for Emu3.5 (no-CFG + LoRA)
│       └── tokenizer_emu3_ibq/
└── UniVR_RL/             # Reinforcement Learning (VR-GRPO)
    ├── examples/
    │   ├── emu3_grpo_lora.sh  # ← Set your reward server endpoint here
    │   └── config_emu3.yaml
    ├── verl/             # Modified verl framework with Emu3.5 HybridEngine
    └── src/
```

---

## Installation

```bash
bash install.sh
```

Installs all dependencies (`torch==2.8.0`, `transformers==4.57.3`, `vllm==0.11.0`, `flash-attn==2.8.3`, `deepspeed`), applies the vLLM source patches for Emu3.5, and installs the verl-based RL framework.

> See [UniVR_SFT/README.md](UniVR_SFT/README.md) and [UniVR_RL/README.md](UniVR_RL/README.md) for detailed usage of each subproject.

---

## Quick Start

### SFT (LoRA, 2 nodes × 8 GPUs)

```bash
# Edit scripts/train_sft_lora.sh to set MODEL_PATH, TOKENIZER_PATH, OUTPUT_DIR
cd UniVR_SFT
bash scripts/train_sft_lora.sh
```

### SFT (Full parameter, 4 nodes × 8 GPUs)

```bash
bash scripts/train_sft_full.sh
```

### RL (VR-GRPO)

```bash
# Edit examples/emu3_grpo_lora.sh to set MODEL_PATH, TOKENIZER_PATH, VQ_MODEL_PATH, VLLM_PATH
cd UniVR_RL
bash examples/emu3_grpo_lora.sh
```

---

<!-- ## Citation

```bibtex
@inproceedings{univr2026,
  title     = {UniVR: Thinking in Visual Space for Unified Visual Reasoning},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
``` -->
