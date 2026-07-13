# Steps to Connect Android Device and Play the Game

For creating cloud Android devices, refer to: https://github.com/tkestack/tke-ai-playbook/pull/20

## 1. Open the Game in the Android Browser:
adb -s <android_ip>:5555 shell am start -a android.intent.action.VIEW -d "http://<game_ip>:8000/number_game.html"

## 2. Ensure the Device is Connected
adb connect <android_ip>:5555

## 3. Run the Game Script
- ollama
```shell
python examples/android_gui_cookbook/play_agent.py \
    --model-type ollama \
    --api-url http://localhost:11434 \
    --model-name qwen2.5vl:3b \
    --devices <android_ip>:5555 \
    --debug
```
- vllm
```shell
python examples/android_gui_cookbook/play_agent.py \
    --model-type vllm \
    --api-url <vllm_ip> \
    --model-name <model_id> \
    --devices <android_ip>:5555 \
    --debug
```

# Parameter Description

- --model-type: Model type (ollama or vllm), default ollama
- --api-url: API address, default http://localhost:11434
- --model-name: Model name, default qwen2.5vl:3b
- --devices: Device list
- --episodes: Number of episodes to run, default 1
- --debug: Enable debug mode, show VLM output
