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
import os.path as osp
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from src.emu3p5 import Emu3ForCausalLM, Emu3Config
from src.vision_tokenizer import build_vision_tokenizer
# Add src to path to allow imports from src
sys.path.append(os.path.abspath("src"))

try:
    from transformers import AutoTokenizer
    from emu3p5.modeling_emu3 import Emu3ForCausalLM
    from tokenizer_emu3_ibq.tokenization_emu3 import Emu3Tokenizer
except ImportError as e:
    print(f"Error importing Emu3 modules: {e}")
    print("Please ensure you are running this script from the project root.")
    sys.exit(1)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="checkpoints/Emu3.5",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_path: Optional[str] = field(
        default="src/tokenizer_emu3_ibq",
        metadata={"help": "Path to tokenizer"}
    )
    debug_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Debug mode for testing"}
    )

@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to the training data (json/jsonl)."}
    )

@dataclass
class LoraArguments:
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning. If False, full fine-tuning is used."}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA r"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,v_proj", 
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    # Anomaly detection parameters
    loss_spike_threshold: float = field(
        default=5.0,
        metadata={"help": "Loss spike threshold multiplier over EMA. E.g., 5.0 means loss > 5x EMA triggers alert. Set 0 to disable."}
    )
    loss_spike_delta_threshold: float = field(
        default=0,
        metadata={"help": "Max allowed loss increase over EMA. If loss > EMA + delta, triggers anomaly. Adapts to any loss scale. Set 0 to disable."}
    )
    grad_norm_threshold: float = field(
        default=10.0,
        metadata={"help": "Gradient norm threshold. Grad norms above this trigger alert and sample info logging."}
    )
    skip_anomaly_batches: bool = field(
        default=False,
        metadata={"help": "If True, zero out loss for anomalous batches to skip their gradient updates."}
    )

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    print(f"Loading model from {model_args.model_name_or_path}")
    model_path = model_args.model_name_or_path
    tokenizer_path = model_args.tokenizer_path
    
    model_config = Emu3Config.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # Load Model
    # For distributed training (DeepSpeed/FSDP), we should not use device_map='auto'
    # as the trainer/accelerator will handle device placement.
    device_map = None
    if not (training_args.deepspeed or training_args.fsdp):
        device_map = "auto"

    model = Emu3ForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        # attn_implementation="eager", # if you cann't install flash_attention
    )

    # text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=os.path.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<|extra_203|>"
    tokenizer.eos_token = "<|extra_204|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.eof_token = "<|extra_201|>"
    tokenizer.tms_token = "<|extra_202|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    tokenizer.bss_token = "<|extra_100|>"
    tokenizer.ess_token = "<|extra_101|>"
    tokenizer.bog_token = "<|extra_60|>"
    tokenizer.eog_token = "<|extra_61|>"
    tokenizer.boc_token = "<|extra_50|>"
    tokenizer.eoc_token = "<|extra_51|>"

    # vq tokenizer
    # Use the current device if available, otherwise let it be handled later or default to cpu/cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # In distributed setting, we might want to be careful with device placement
    # For now, we just load it. If it's not used in the loop, it might be fine.
    # However, build_vision_tokenizer likely puts it on a specific device.
    # Let's assume for now we don't strictly need it on a specific device for this dummy training script
    # or we put it on the current device.
    vq_model = build_vision_tokenizer('ibq', "./weights/Emu3.5-VisionTokenizer", device=device)

    # Gradient checkpointing (applies to both LoRA and full fine-tuning)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.config.use_cache = False 

    if lora_args.use_lora:
        # LoRA fine-tuning
        print("Using LoRA for parameter-efficient fine-tuning.")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules.split(",")
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        # Full fine-tuning: all parameters are trainable
        print("Using full fine-tuning (all parameters trainable).")
        # Ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
        # Print trainable parameters info
        # Note: With DeepSpeed ZeRO-3, parameters are partitioned across GPUs,
        # so p.numel() may return 0. Use ds_numel if available.
        total_params = 0
        trainable_params = 0
        for p in model.parameters():
            num_el = getattr(p, 'ds_numel', p.numel())
            total_params += num_el
            if p.requires_grad:
                trainable_params += num_el
        if total_params > 0:
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        else:
            print(f"  Total parameters: (deferred, ZeRO-3 partitioned)")

    # Prepare Dataset
    if not model_args.debug_mode:
        from src.utils.dataset_vr_train import VRTrainDataset
        # We need to pass the tokenizer and vq_model (if needed for online tokenization)
        # Note: vq_model is currently loaded on GPU, which might be an issue for DataLoader workers.
        # Ideally, pre-tokenize your data.


        dataset_cfg = {
            "Agibot":                       {"max_samples_per_task": 300, "enabled": True},  # 131 tasks × ≤50 = ~6500
            "Bridge":                       {"max_samples": 5000, "enabled": True},
            "Droid":                        {"max_samples": 5000, "enabled": True},
            "EgoDex":                       {"max_samples_per_task": 350, "enabled": True},  # 111 tasks × ≤5 = ~555
            "Epic_Kitchen":                 {"max_samples": 5000, "enabled": True},
            "Spatial_Navigation":           {"max_samples": 5000, "enabled": True},
            "Spatial_Navigation_maze":      {"max_samples": 5000, "enabled": True},
            "Spatial_Navigation_trapfield": {"max_samples": 5000, "enabled": True},
            "VideoCraftBench":              {"max_samples": 300, "repeat":35, "enabled": True},
            "Visual_CoT_GQA":               {"max_samples": 10000, "enabled": True},
            "Visual_Search":                {"max_samples": 5000, "enabled": True},
            "Zebra_Count":                  {"max_samples": 10000, "enabled": True},
            "Zebra_Jigsaw":                 {"max_samples": 5000, "enabled": True},
            "Zebra_Robot":                  {"max_samples": 5000, "enabled": True},
            "Action100M":                   {"max_samples": 5000, "enabled": True},
            # Or your own datasets...
        }

        dataset = VRTrainDataset(
            dataset_cfg=dataset_cfg,
            tokenizer=tokenizer,
            max_images=None,
            max_length=12000,
            use_vlm_caption=True,
            vq_model=vq_model,
            interleaved_text=True,
        )

        # No need for map/tokenize_function as the dataset class handles it
        tokenized_datasets = dataset
    else:
        print("Using dummy data for demonstration.")
        from datasets import Dataset
        data = [
            {"text": "Emu3.5 is a native multimodal model that can generate images and text."},
            {"text": "The weather is nice today for training a large language model."},
            {"text": "Deep learning requires significant computational resources."}
        ] * 10
        dataset = Dataset.from_list(data)

        def tokenize_function(examples):
            # Simple text tokenization
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=512 # You might want to make this an argument
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Trainer (with anomaly detection)
    from src.utils.custom_trainer import AnomalyAwareTrainer, AnomalyAwareDataCollator
    trainer = AnomalyAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        data_collator=AnomalyAwareDataCollator(),
        loss_spike_threshold=lora_args.loss_spike_threshold,
        loss_spike_delta_threshold=lora_args.loss_spike_delta_threshold,
        grad_norm_threshold=lora_args.grad_norm_threshold,
        skip_anomaly_batches=lora_args.skip_anomaly_batches,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
