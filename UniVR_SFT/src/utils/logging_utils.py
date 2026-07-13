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
from datetime import datetime
import os.path as osp
import builtins
old_print = builtins.print


def setup_print_file(file):
    def print(*args, **kwargs):
        msg = " ".join(map(str, args))
        with open(file, "a") as f:
            f.write(msg + "\n")
        old_print(msg)

    builtins.print = print


def setup_logger(log_dir="./", log_name="log"):
    logfile = osp.join(
        log_dir,
        f'{log_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    )
    os.makedirs(osp.dirname(logfile), exist_ok=True)
    setup_print_file(logfile)
