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

git add ds_config_zero3.json
git add inference.py
git add inference_vllm.py
git add requirements
git add offline_process_images.py
git add inference_vllm_parallel.py
git add run_emu_test.py

git add scripts
git add src
git add train.py
git add install.sh
git add tools
git add configs
git add git_push.sh
git add inference_vllm_from_parquet.py

git commit -m "Update"
git push origin main
