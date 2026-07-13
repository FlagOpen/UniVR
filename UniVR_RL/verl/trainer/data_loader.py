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

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def _is_emu3_tokenizer(tokenizer: PreTrainedTokenizer) -> bool:
    """Check if the tokenizer is an Emu3 tokenizer."""
    # Check for Emu3-specific attributes
    return hasattr(tokenizer, 'boi_token') or hasattr(tokenizer, 'img_token')


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    # Check if this is Emu3 (pre-tokenized data, no processor)
    is_emu3 = _is_emu3_tokenizer(tokenizer)
    
    if is_emu3:
        # Use Emu3-specific dataset for pre-tokenized data
        from ..utils.emu3_dataset import Emu3RLHFDataset, emu3_collate_fn
        print("[Emu3] Using Emu3RLHFDataset for pre-tokenized data")
        
        train_dataset = Emu3RLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            prompt_key=config.prompt_key,
            answer_key=getattr(config, 'answer_key', 'global_summary'),
            max_prompt_length=config.max_prompt_length,
            max_images=getattr(config, 'max_images', None),
            truncation="right",
            format_prompt=config.format_prompt,
            enable_cfg=getattr(config, 'enable_cfg', False),
            max_samples_per_source=getattr(config, 'source_sample_counts', None),
        )
        dataset_collate_fn = emu3_collate_fn
    else:
        train_dataset = RLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        )
        dataset_collate_fn = collate_fn
        
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=dataset_collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # Create validation dataset (skipped when val_files is empty/null)
    val_dataloader = None
    if config.val_files:
        if is_emu3:
            from ..utils.emu3_dataset import Emu3RLHFDataset, emu3_collate_fn
            val_dataset = Emu3RLHFDataset(
                data_path=config.val_files,
                tokenizer=tokenizer,
                prompt_key=config.prompt_key,
                answer_key=getattr(config, 'answer_key', 'global_summary'),
                max_prompt_length=config.max_prompt_length,
                max_images=getattr(config, 'max_images', None),
                truncation="right",
                format_prompt=config.format_prompt,
                enable_cfg=getattr(config, 'enable_cfg', False),
            )
            val_collate_fn = emu3_collate_fn
        else:
            val_dataset = RLHFDataset(
                data_path=config.val_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
            )
            val_collate_fn = collate_fn

        if config.val_batch_size == -1:
            val_batch_size = len(val_dataset)
        else:
            val_batch_size = config.val_batch_size

        val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=val_collate_fn,
            pin_memory=False,
            drop_last=False,
        )
        print(f"Size of val dataset {config.val_files}: {len(val_dataset)}")
        assert len(val_dataloader) >= 1
    else:
        print("[data_loader] val_files is empty, skipping validation dataset creation.")

    assert len(train_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    return train_dataloader, val_dataloader
