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
import subprocess
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Batch visualize proto files.")
    parser.add_argument("input_dir", type=str, help="Directory containing .pb files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist.")
        return

    # Find all .pb files in the input directory
    pb_files = list(input_dir.glob("*.pb"))
    
    if not pb_files:
        print(f"No .pb files found in {input_dir}")
        return

    print(f"Found {len(pb_files)} .pb files. Starting processing...")

    # Determine the path to vis_proto.py
    # Assuming the script is located in scripts/ and vis_proto.py is in src/utils/
    # relative to the project root.
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    vis_proto_script = project_root / "src" / "utils" / "vis_proto.py"

    if not vis_proto_script.exists():
        print(f"Warning: Could not find vis_proto.py at {vis_proto_script}")
        # Fallback: assume the user is running from project root
        vis_proto_script = Path("src/utils/vis_proto.py").resolve()
        if not vis_proto_script.exists():
             print(f"Error: Could not find vis_proto.py at {vis_proto_script}. Please run from project root.")
             return

    print(f"Using visualizer script: {vis_proto_script}")

    for pb_file in pb_files:
        file_stem = pb_file.stem # filename without extension (e.g., '000' from '000.pb')
        output_dir = input_dir / file_stem
        
        # Create the output directory if it doesn't exist
        try:
            output_dir.mkdir(exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {output_dir}: {e}")
            continue
        
        print(f"Processing {pb_file.name}...")
        print(f"  Output directory: {output_dir}")
        
        cmd = [
            sys.executable, # Use the current python interpreter
            str(vis_proto_script),
            "--input", str(pb_file),
            "--output", str(output_dir) # Pass the directory as output
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pb_file.name}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {pb_file.name}: {e}")

    print("Batch processing finished.")

if __name__ == "__main__":
    main()
