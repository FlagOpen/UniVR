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

cd UniVR_SFT

pip install vllm==0.11.0 --user
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --user
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --user

pip install transformers==4.57.3 --user

python src/patch/apply.py
python src/patch/apply_no_cfg_support.py

pip install omegaconf
pip install pandas
pip install loguru

pip install -r ./requirements/transformers.txt --user

pip install pyarrow
pip install fastparquet

cd ../UniVR_RL

bash install.sh
pip install transformers==4.57.3 --user

