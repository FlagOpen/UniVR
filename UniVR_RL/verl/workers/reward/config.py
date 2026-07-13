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
Reward config
"""

from dataclasses import dataclass, field
from typing import Optional

from ...utils.py_functional import get_abs_path


@dataclass
class RewardConfig:
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    # Validation reward function override (if not set, uses same as training)
    val_reward_function: Optional[str] = None
    val_reward_function_kwargs: Optional[dict] = None
    skip_special_tokens: bool = True
    num_cpus: int = 1
    # Tokenizer settings (for loading tokenizer from path to avoid pickle issues with custom tokenizers like Emu3)
    tokenizer_path: Optional[str] = None
    tokenizer_trust_remote_code: bool = True
    tokenizer_use_fast: bool = True
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)
    val_reward_function_name: Optional[str] = field(default=None, init=False)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            self.reward_function = get_abs_path(self.reward_function, prompt="Reward function")

        # Process val_reward_function the same way
        if self.val_reward_function is not None:
            if ":" not in self.val_reward_function:
                self.val_reward_function_name = "main"
            else:
                self.val_reward_function, self.val_reward_function_name = self.val_reward_function.rsplit(":", maxsplit=1)

            self.val_reward_function = get_abs_path(self.val_reward_function, prompt="Validation reward function")
