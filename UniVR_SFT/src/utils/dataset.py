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

import json
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import random
from PIL import Image
from src.utils.input_utils import format_image_string
from src.vision_tokenizer import build_vision_tokenizer

def visualize_img(vq_model, token, h, w):
    indices = torch.from_numpy(token.flatten()).cuda()
    shape = (1, h, w, vq_model.quantize.e_dim)
    with torch.no_grad():
        decoded = vq_model.decode_code(indices, shape=shape)
    
    # Post-process
    decoded = decoded.clamp(-1, 1)
    decoded = (decoded + 1.0) / 2.0 * 255.0
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    
    # Save
    vis_file = os.path.join("./", "data_vis.jpg")
    Image.fromarray(decoded).save(vis_file)

class Emu3Dataset(Dataset):
    def __init__(self, data_path, tokenizer, task="story", max_images=None, max_length=6000, debug_mode=False):
        self.tokenizer = tokenizer
        self.data = pd.read_parquet(data_path)
        self.task = task
        self.max_images = max_images
        self.max_length = max_length

        self.bos_token = "<|extra_203|>"
        self.eos_token = "<|extra_204|>"
        self.bss_token = "<|extra_100|>"
        self.ess_token = "<|extra_101|>"
        self.bog_token = "<|extra_60|>"
        self.eog_token = "<|extra_61|>"
        self.boc_token = "<|extra_50|>"
        self.eoc_token = "<|extra_51|>"

        self.bos_id = self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.bss_id = self.tokenizer.convert_tokens_to_ids(self.bss_token)
        self.ess_id = self.tokenizer.convert_tokens_to_ids(self.ess_token)
        self.bog_id = self.tokenizer.convert_tokens_to_ids(self.bog_token)
        self.eog_id = self.tokenizer.convert_tokens_to_ids(self.eog_token)
        self.boc_id = self.tokenizer.convert_tokens_to_ids(self.boc_token)
        self.eoc_id = self.tokenizer.convert_tokens_to_ids(self.eoc_token)

        # For visualization
        if debug_mode:
            vq_path = "./weights/Emu3.5-VisionTokenizer"
            vq_type = "ibq"
            self.vq_model = build_vision_tokenizer(vq_type, vq_path, device="cuda")
            self.vq_model.eval()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx = 10
        sample = self.data.iloc[idx]
        frames_data = sample["frames"]
        global_summary = sample["global_summary"]

        
        # --- Frame Sampling ---
        if self.max_images is not None and len(frames_data) > self.max_images:
            first_frame = frames_data[0]
            other_frames = list(frames_data[1:])
            
            num_to_select = self.max_images - 1
            if num_to_select > 0 and len(other_frames) > num_to_select:
                sampled_frames = random.sample(other_frames, num_to_select)
                # Sort to maintain temporal order
                sampled_frames.sort(key=lambda x: x['frame_filename'])
                frames_data = [first_frame] + sampled_frames
            else: # if num_to_select is 0 or not enough frames to sample
                frames_data = [first_frame]

        # frames_data = [frames_data[0], frames_data[-1]]
        # --- Build Prompt ---
        # Template: <BOS>You are a helpful assistant. USER: <IMAGE> ASSISTANT: <BSS><BoG>{summary}<EoG>{text}<BoC>{caption}<EoC><IMAGE>...<ESS><EOS>
        
        # USER part
        # user_prompt = "You are a helpful assistant for howto task. USER: Fold the paper on the table into a paper airplane with just one step." if "airplane" in frames_data[0]['frame_filename'] else "You are a helpful assistant for howto task. USER: Fold the paper on the table into a paper boat with just one step.."
        user_prompt = "You are a helpful assistant for howto task. USER: Fold the paper on the table into a paper airplane with six step." if "airplane" in frames_data[0]['frame_filename'] else "You are a helpful assistant for howto task. USER: Fold the paper on the table into a paper boat with six step."
        user_prompt_ids = self.tokenizer.encode(user_prompt, add_special_tokens=False)

        # First frame for the user
        first_frame = frames_data[0]
        first_frame_tokens = np.frombuffer(first_frame['image_tokens'], dtype=np.int64).astype(np.int32).reshape(first_frame['height'], first_frame['width'])
        first_frame_image_string = format_image_string(self.tokenizer, first_frame_tokens)
        first_frame_image_ids = self.tokenizer.encode(first_frame_image_string, add_special_tokens=False)
        
        # print(sample["frames"][0]["frame_filename"], first_frame_image_string)
        # visualize_img(self.vq_model, first_frame_tokens, first_frame['height'], first_frame['width'])
        # ASSISTANT part
        assistant_prompt = " ASSISTANT: "
        assistant_prompt_ids = self.tokenizer.encode(assistant_prompt, add_special_tokens=False)

        summary_ids = [self.bog_id] + self.tokenizer.encode(global_summary, add_special_tokens=False) + [self.eog_id]

        # --- Interleave images and captions for assistant ---
        interleaved_ids = []
        for idx, frame in enumerate(frames_data[1:]): # Start from the second frame
            # Brief Text
            # text = frame.get('text', f"Now we begin step {idx+1}.")
            # text = frame.get('text', "Now, let's turn the paper on the table into the target object.")
            # text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            # interleaved_ids.extend(text_ids)

            # Detailed Caption
            # dense_caption = frame.get('caption', frame['caption'])
            # dense_caption_ids = [self.boc_id] + self.tokenizer.encode(dense_caption, add_special_tokens=False) + [self.eoc_id]
            # interleaved_ids.extend(dense_caption_ids)

            # Image
            image_tokens = np.frombuffer(frame['image_tokens'], dtype=np.int64).astype(np.int32).reshape(frame['height'], frame['width'])
            image_string = format_image_string(self.tokenizer, image_tokens)
            image_ids = self.tokenizer.encode(image_string, add_special_tokens=False)
            interleaved_ids.extend(image_ids)

            # visualize_img(self.vq_model, image_tokens, first_frame['height'], first_frame['width'])
        # --- Combine all parts for input_ids ---
        input_ids = (
            [self.bos_id] + user_prompt_ids + first_frame_image_ids + assistant_prompt_ids +
            [self.bss_id] + summary_ids + interleaved_ids + [self.ess_id, self.eos_id]
        )

        # input_ids = (
        #     [self.bos_id] + user_prompt_ids + first_frame_image_ids + assistant_prompt_ids +
        #     [self.bss_id] + summary_ids + [self.ess_id, self.eos_id]
        # )
    
        # --- Create Labels ---
        # Mask user part (including first image) and control tokens
        user_part_len = len(user_prompt_ids) + len(first_frame_image_ids) + len(assistant_prompt_ids)
        labels = [-100] * (1 + user_part_len + 1) # BOS + user_part + BSS

        # Add labels for summary
        labels.extend(summary_ids)

        # Add labels for interleaved sequence (images and captions from assistant)
        labels.extend(interleaved_ids)
        
        # labels.extend([-100, -100]) # ESS, EOS
        labels.extend([self.ess_id, self.eos_id]) # ESS, EOS

        # --- Padding and Attention Mask ---
        max_length = self.max_length # Or your desired max sequence length
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
        else:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]


        attention_mask = [1] * len(input_ids)
        if padding_length > 0:
            attention_mask[-padding_length:] = [0] * padding_length


        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


if __name__ == '__main__':
    workspace_path = Path("./")
    dataset_path = workspace_path / "datasets/Video-CraftBench/PaperFolding_Uniform_Keysteps_withText"
    tokenizer_path = workspace_path / "src/tokenizer_emu3_ibq"

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), 
        special_tokens_file=os.path.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True)
    
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

    dataset = Emu3Dataset(
        data_path=str(dataset_path / "encoded_data_grouped_512px.parquet"),
        tokenizer=tokenizer,
        debug_mode=True,
        max_images=10, # Example: Limit to a total of 4 images per sample
    )
    
    # --- Verification ---
    for i in range(10):
        sample_data = dataset[i]
        input_ids = sample_data['input_ids']
        labels = sample_data['labels']

        print("--- Decoded Input IDs ---")
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        # print(decoded_input)

        print("\n--- Decoded Labels ---")
        # Replace -100 with pad_token_id for decoding
        labels_for_decoding = [l if l != -100 else tokenizer.pad_token_id for l in labels.tolist()]
        decoded_labels = tokenizer.decode(labels_for_decoding, skip_special_tokens=False)
        # print(decoded_labels)

        print("\n--- Verifying lengths ---")
        print(f"Input IDs length: {len(input_ids)}")
        print(f"Labels length: {len(labels)}")
        print(f"Attention Mask length: {len(sample_data['attention_mask'])}")