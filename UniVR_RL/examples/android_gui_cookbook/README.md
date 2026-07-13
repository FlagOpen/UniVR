# Android GUI Number Game Reinforcement Learning Tutorial

This tutorial covers the three complete workflows: **Cloud Environment Deployment** → **Model Training** → **Model Testing**.

---

## 1. Cloud Android Deployment and Game Deployment

### 1.1 Game Deployment

#### Docker Deployment

```bash
# Pull and run the game container
docker run -d \
  --name number-game \
  -p 8000:8000 \
  ccr.ccs.tencentyun.com/yuehuazhang/number-game-rl:v1.4

# Access the game
# http://localhost:8000/number_game.html
```

#### Kubernetes Deployment

```bash
# Use the provided configuration file
kubectl apply -f examples/android_gui_cookbook/game_docker/game.yaml

# Get the external access address
kubectl get svc number-game -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Access: http://<EXTERNAL-IP>:8000/number_game.html
```

### 1.2 Android Device Connection

#### Create Cloud Android Device
Reference documentation: https://github.com/tkestack/tke-ai-playbook/pull/20

#### Connect Device and Open Game

```bash
# Connect device
adb connect <android_ip>:5555

# Open game in device browser
adb -s <android_ip>:5555 shell am start -a android.intent.action.VIEW \
  -d "http://<game_ip>:8000/number_game.html"

# Verify connection
adb devices
```

---

## 2. Model Training

### 2.1 Training Script Description

**Core files**:
- `examples/qwen2_5_vl_3b_android_gui_grpo.sh` - Training launch script
- `examples/format_prompt/android_gui.jinja` - Prompt template
- `examples/reward_function/android_gui.py` - Reward function

**Game rules** (defined by `android_gui.jinja`):
- 🟢 Green light: select the **largest** number → position index (0/1/2)
- 🔴 Red light: select the **smallest** number → position index (0/1/2)
- 🟡 Yellow light: select the **middle** number → position index (0/1/2)

**Scoring rules** (implemented by `android_gui.py`):
- Correct selection: `+1.0`
- Incorrect selection: `0.0`

### 2.2 Launch Training

```bash
# Switch to EasyR1 root directory
cd /path/to/EasyR1

# Run the training script
bash examples/qwen2_5_vl_3b_android_gui_grpo.sh
```

### 2.3 Key Training Parameters

The script uses the following configuration (based on `config.yaml`, overridden via command line):

| Parameter | Value | Description |
|------|-----|------|
| `data.train_files` | `yuehua-s/numbergame@train` | Training dataset |
| `data.val_files` | `yuehua-s/numbergame@test` | Validation dataset |
| `data.rollout_batch_size` | `32` | Rollout batch size |
| `algorithm.kl_coef` | `0.04` | KL divergence coefficient |
| `worker.actor.optim.lr` | `1e-5` | Learning rate |
| `worker.rollout.n` | `8` | Number of responses generated per step |
| `trainer.total_epochs` | `3` | Training epochs |
| `trainer.n_gpus_per_node` | `2` | GPUs per node |

### 2.4 Export Model

After training, checkpoints are saved in `checkpoints/<experiment_name>/global_step_<N>/actor`.

```bash
# Merge model (convert to HuggingFace format)
python3 scripts/model_merger.py \
  --local_dir /path/to/EasyR1/checkpoints/<experiment_name>/global_step_35/actor

# Export directory: checkpoints/<experiment_name>/global_step_35/actor/huggingface/
```

---

## 3. Testing Model Performance with an Agent

### 3.1 Launch Inference Service

Deploy the trained model using vLLM:

```bash
vllm serve /path/to/checkpoints/<experiment_name>/global_step_35/actor/huggingface/ \
  --host 0.0.0.0 \
  --port 8000
```

### 3.2 Run Agent Test

**Core files**:
- `examples/android_gui_cookbook/play_agent.py` - Agent main program
- `examples/android_gui_cookbook/adb_controller.py` - ADB controller
- `examples/android_gui_cookbook/vlm_client.py` - VLM inference client

#### Using vLLM Model

```bash
python examples/android_gui_cookbook/play_agent.py \
  --model-type vllm \
  --api-url http://<vllm_server_ip>:8000 \
  --model-name /path/to/checkpoints/xxx/global_step_35/actor/huggingface/ \
  --devices <android_ip>:5555 \
  --episodes 5 \
  --debug
```

#### Using Ollama Model

```bash
python examples/android_gui_cookbook/play_agent.py \
  --model-type ollama \
  --api-url http://localhost:11434 \
  --model-name qwen2.5vl:3b \
  --devices <android_ip1>:5555 <android_ip2>:5555 \
  --episodes 3 \
  --debug
```

### 3.3 Parameter Description

| Parameter | Default | Description |
|------|--------|------|
| `--model-type` | `ollama` | Model service type (`ollama` or `vllm`) |
| `--api-url` | `http://localhost:11434` | Model API address |
| `--model-name` | `qwen2.5vl:3b` | Model name or path |
| `--devices` | `101.43.137.83:5555` | Android device list (space-separated) |
| `--episodes` | `1` | Number of episodes per device |
| `--debug` | `False` | Enable debug mode (show VLM output) |
| `--screenshot-dir` | `game_screenshots` | Screenshot save directory |

### 3.4 Test Workflow

The agent automatically performs the following actions (10 rounds per episode):

1. **Screenshot** - Capture the current game screen
2. **VLM Inference** - Identify indicator light color and numbers, make a decision
3. **Click card** - Click the selected number (position 0/1/2)
4. **Verify click** - Check whether the card color has changed
5. **Click next round** - Proceed to the next round

### 3.5 View Results

After testing, results are saved in `game_screenshots/<device_id>/`:

```
game_screenshots/
└── <android_ip>_5555/
    ├── round_01_<timestamp>.png           # Screenshot before each round's decision
    ├── round_01_after_click_<timestamp>.png  # Screenshot after click
    ├── final_score_<timestamp>.png        # Final score screenshot
    └── result_<timestamp>.json            # Game result (JSON)
```

**Result file example**:
```json
{
  "device_id": "101.43.137.83:5555",
  "timestamp": "20251123_143025",
  "total_rounds": 10,
  "final_score": 80,
  "model_type": "vllm",
  "model_name": "/path/to/model"
}
```

---

## Appendix: File Structure

```
examples/
├── qwen2_5_vl_3b_android_gui_grpo.sh    # Training script
├── config.yaml                           # Base configuration
├── format_prompt/
│   └── android_gui.jinja                 # Prompt template
├── reward_function/
│   └── android_gui.py                    # Reward function
└── android_gui_cookbook/
    ├── README.md                         # This document
    ├── play_agent.py                     # Agent main program
    ├── adb_controller.py                 # ADB controller
    ├── vlm_client.py                     # VLM client
    └── game_docker/
        ├── game.yaml                     # K8s deployment configuration
        └── DOCKER_README.md              # Docker detailed instructions
```
