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
import re
import argparse

def clean_checkpoints(folder_path):
    """
    Cleans up safetensor shard files matching the pattern model-xxxxx-of-xxxxx.safetensors
    in the specified directory.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        return

    # Regular expression to match files like model-00001-of-00005.safetensors
    # Matches 'model-', followed by digits, '-of-', digits, '.safetensors'
    pattern = re.compile(r"^model-\d+-of-\d+\.safetensors$")

    files_to_delete = []

    print(f"Scanning directory: {folder_path}")
    
    for filename in os.listdir(folder_path):
        if pattern.match(filename):
            files_to_delete.append(os.path.join(folder_path, filename))

    if not files_to_delete:
        print("No matching checkpoint shard files found.")
        return

    print(f"Found {len(files_to_delete)} files to delete:")
    for file_path in files_to_delete:
        print(f"  {os.path.basename(file_path)}")

    # Ask for confirmation
    response = input("\nDo you want to delete these files? (y/N): ").strip().lower()
    
    if response == 'y':
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        print(f"\nCleanup complete. {deleted_count} files deleted.")
    else:
        print("\nOperation cancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup model safetensors shards from a directory.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing the checkpoint files.")
    
    args = parser.parse_args()
    
    clean_checkpoints(args.folder_path)
