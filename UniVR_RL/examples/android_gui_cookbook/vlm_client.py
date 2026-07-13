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

"""
VLM Model Client

Supports both Ollama and vLLM model services.
"""

import base64
from io import BytesIO

import requests
from PIL import Image


class VLMClient:
    """VLM model client (supports Ollama and vLLM)."""

    def __init__(self, model_type: str, api_url: str, model_name: str):
        """
        Args:
            model_type: "ollama" or "vllm"
            api_url: API URL, e.g. "http://localhost:11434" or "http://localhost:8000"
            model_name: Model name
        """
        self.model_type = model_type.lower()
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name

        if self.model_type not in ["ollama", "vllm"]:
            raise ValueError(f"Unsupported model type: {model_type}. Only 'ollama' or 'vllm' are supported.")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert a PIL Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64

    def query(self, image: Image.Image, prompt: str) -> str:
        """
        Query the VLM model.

        Args:
            image: PIL Image object
            prompt: Text prompt

        Returns:
            Model response text
        """
        img_base64 = self._image_to_base64(image)

        if self.model_type == "ollama":
            return self._query_ollama(img_base64, prompt)
        elif self.model_type == "vllm":
            return self._query_vllm(img_base64, prompt)

    def _query_ollama(self, img_base64: str, prompt: str) -> str:
        """Query the Ollama API."""
        try:
            payload = {"model": self.model_name, "prompt": prompt, "images": [img_base64], "stream": False}

            response = requests.post(f"{self.api_url}/api/generate", json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"⚠ Ollama API error: {response.status_code}")
                return ""
        except Exception as e:
            print(f"⚠ Ollama query failed: {e}")
            return ""

    def _query_vllm(self, img_base64: str, prompt: str) -> str:
        """Query the vLLM API (OpenAI compatible)."""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        ],
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.7,
            }

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"⚠ vLLM API error: {response.status_code}")
                return ""
        except Exception as e:
            print(f"⚠ vLLM query failed: {e}")
            return ""
