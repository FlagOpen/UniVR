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

from PIL import Image
import torch
import numpy as np


def smart_resize(image: Image.Image, area: int = 512 * 512, ds_factor: int = 16):
    width, height = image.size
    aspect_ratio = width / height
    new_height = int((area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    # Round to nearest multiple of divisible_by
    new_height = ((new_height + ds_factor//2) // ds_factor) * ds_factor
    new_width = ((new_width + ds_factor//2) // ds_factor) * ds_factor
    return image.resize((new_width, new_height), Image.BICUBIC)

def smart_resize_short_side(image: Image.Image, short_side: int = 512, ds_factor: int = 16):
    width, height = image.size
    if width <= height:
        new_width = short_side
        new_height = int(round(height * short_side / width))
    else:
        new_height = short_side
        new_width = int(round(width * short_side / height))
    # Round to nearest multiple of ds_factor
    new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor
    new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor
    return image.resize((new_width, new_height), Image.BICUBIC)


def resize_and_center_crop(image: Image.Image, target_size: tuple):
    width, height = image.size
    target_height, target_width = target_size
    scale_w = target_width / width
    scale_h = target_height / height
    scale = max(scale_w, scale_h)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    image = image.crop((left, top, left + target_width, top + target_height))
    return image


def format_image_string(tokenizer, image_tokens):
    image_string = ""
    h, w = image_tokens.shape
    for _h in range(h):
        row_string = ""
        for _w in range(w):
            row_string += "<|visual token {token_id:0>6d}|>".format(token_id=image_tokens[_h, _w])

        if _h < h - 1:
            row_string += tokenizer.eol_token
        image_string += row_string

    return "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}".format(
        image_start=tokenizer.boi_token,
        token_height=h,
        token_width=w,
        image_token=tokenizer.img_token,
        token_str=image_string,
        image_end=tokenizer.eoi_token,
    )


@torch.no_grad()
def build_image(image, cfg, tokenizer, vq_model, manual_resize=False, target_size=(24, 32)):
    if manual_resize:
        image = resize_and_center_crop(image, target_size)
    else:
        image = smart_resize_short_side(image, getattr(cfg, "image_short_side", 512))

    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image = torch.tensor((np.array(image) / 127.5 - 1.0)).to(device, dtype).permute(2, 0, 1)
    _, _, token = vq_model.encode(image[None])
    token = token[-1].view(h // 16, w // 16)
    return format_image_string(tokenizer, token)
