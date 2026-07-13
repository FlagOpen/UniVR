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

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)


def check_and_register_emu3(hf_path: str) -> bool:
    """Check if the model at hf_path is an Emu3 model and register it to transformers AutoClass."""
    import json
    import sys

    config_file = os.path.join(hf_path, "config.json")
    if not os.path.exists(config_file):
        return False

    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        model_type = config_data.get("model_type", "").lower()
    except Exception:
        return False

    if model_type not in ("emu3", "emu3.5"):
        return False

    print(f"[Emu3] Detected Emu3 model at {hf_path}")

    # Add Emu_VW/src to sys.path so that `emu3p5` is importable
    emu_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Emu_VW/src")
    emu_src_path = os.path.abspath(emu_src_path)
    # Fallback to known absolute path
    if not os.path.exists(emu_src_path):
        emu_src_path = "../UniVR_SFT/src"
    if os.path.exists(emu_src_path) and emu_src_path not in sys.path:
        sys.path.insert(0, emu_src_path)
        print(f"[Emu3] Added {emu_src_path} to sys.path")

    try:
        from emu3p5 import Emu3Config, Emu3ForCausalLM

        # Force-clear any existing Emu3 registrations
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

        for mapping in [CONFIG_MAPPING]:
            for attr in ["_mapping", "_extra_content"]:
                d = getattr(mapping, attr, None)
                if d is None:
                    continue
                for key in ["emu3", "Emu3"]:
                    if key in d:
                        del d[key]

        for mapping in [MODEL_FOR_CAUSAL_LM_MAPPING]:
            for attr in ["_mapping", "_extra_content"]:
                d = getattr(mapping, attr, None)
                if d is None:
                    continue
                keys_to_remove = [k for k in d if hasattr(k, "__name__") and "emu3" in k.__name__.lower()]
                for key in keys_to_remove:
                    del d[key]

        # Register custom Emu3
        try:
            AutoConfig.register("Emu3", Emu3Config)
        except ValueError:
            pass
        try:
            AutoModelForCausalLM.register(Emu3Config, Emu3ForCausalLM)
        except ValueError:
            pass

        print("[Emu3] Successfully registered Emu3Config and Emu3ForCausalLM to transformers AutoClass")
        return True
    except Exception as e:
        print(f"[Emu3] Warning: Failed to register Emu3: {e}")
        return False


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(local_path: str, remote_path: str):
    # Push to hugging face
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)

    state_dict: dict[str, list[torch.Tensor]] = {}
    param_placements: dict[str, list[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print(f"Cannot find key {key} in rank {rank}.")

            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at ddp dimension can be discarded
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            # merge shards
            placements: tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")

    # Register Emu3 model if needed (custom model not built-in to transformers)
    is_emu3 = check_and_register_emu3(hf_path)

    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    architectures: list[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForImageTextToText
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    with torch.device("meta"):
        if is_emu3:
            # Emu3 uses custom model class, instantiate directly
            from emu3p5 import Emu3ForCausalLM
            model: PreTrainedModel = Emu3ForCausalLM(config)
        else:
            model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

    assert isinstance(model, PreTrainedModel)
    model.to_empty(device="cpu")

    print(f"Saving model to {hf_path}...")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict, model

    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
