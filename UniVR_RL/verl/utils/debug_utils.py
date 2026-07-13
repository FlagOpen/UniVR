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
import time
import datetime
import traceback
from typing import Any


def remote_breakpoint(port: int = 5678, rank: int = None, wait: bool = True):
    """Start a debugpy remote debugging breakpoint inside a Ray worker.

    Only the specified rank (default rank 0) will pause and wait for a debugger
    to attach; all other ranks skip this call to avoid deadlocks.

    Args:
        port: debugpy listening port. Use different ports when debugging multiple ranks.
        rank: Only pause on this rank. None = pause only on rank 0.
        wait: Whether to block until a debugger connects. False = non-blocking.

    Usage:
        # Insert at the desired breakpoint location:
        from verl.utils.debug_utils import remote_breakpoint
        remote_breakpoint(port=5678)
    """
    current_rank = int(os.getenv("RANK", "0"))
    target_rank = rank if rank is not None else 0
    
    if current_rank != target_rank:
        return
    
    try:
        import debugpy
    except ImportError:
        print(f"[DEBUG] debugpy not installed! Run: pip install debugpy")
        print(f"[DEBUG] Falling back to print-based debugging.")
        # Print call stack as fallback
        traceback.print_stack()
        return
    
    try:
        debugpy.listen(("0.0.0.0", port))
        print(f"\n{'='*60}")
        print(f"[DEBUG] Rank {current_rank}: debugpy listening on port {port}")
        print(f"[DEBUG] Waiting for debugger attach on port {port}...")
        print(f"[DEBUG] In VS Code: Run > Attach to Process > port {port}")
        print(f"{'='*60}\n")
        
        if wait:
            debugpy.wait_for_client()
            print(f"[DEBUG] Debugger attached! Continuing execution...")
            debugpy.breakpoint()  # pause immediately at this line
    except Exception as e:
        print(f"[DEBUG] debugpy failed: {e}")
        print(f"[DEBUG] Port {port} may already be in use. Try a different port.")


def snapshot(tag: str, rank: int = None, save_dir: str = "/tmp/verl_debug", **kwargs):
    """Save a variable snapshot to file for offline inspection.

    Supports tensors, numpy arrays, dicts, lists, and primitive types.

    Args:
        tag: Snapshot label used in the filename.
        rank: Only save on this rank. None = save on all ranks.
        save_dir: Directory to save snapshots.
        **kwargs: Variable names and values to save.

    Usage:
        from verl.utils.debug_utils import snapshot
        snapshot("before_generate",
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 config_dict=vars(config))

        # Load later:
        # data = torch.load("/tmp/verl_debug/before_generate_rank0_20260218_143000.pt")
        # print(data['input_ids'].shape)
    """
    import torch
    
    current_rank = int(os.getenv("RANK", "0"))
    if rank is not None and current_rank != rank:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{tag}_rank{current_rank}_{timestamp}.pt"
    filepath = os.path.join(save_dir, filename)
    
    # Prepare serializable data
    save_data = {"_tag": tag, "_rank": current_rank, "_timestamp": timestamp}
    for key, value in kwargs.items():
        try:
            if isinstance(value, torch.Tensor):
                save_data[key] = value.detach().cpu()
            elif hasattr(value, '__dict__'):
                # Object → convert to repr string
                save_data[key] = repr(value)
            else:
                save_data[key] = value
        except Exception as e:
            save_data[key] = f"<failed to serialize: {e}>"
    
    torch.save(save_data, filepath)
    print(f"[DEBUG] Rank {current_rank}: Snapshot saved to {filepath}")
    print(f"[DEBUG]   Keys: {list(kwargs.keys())}")
    
    # Print tensor shape information
    import torch
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            print(f"[DEBUG]   {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")


def debug_log(tag: str, rank: int = None, **kwargs):
    """Conditional debug logging without stopping training.

    Args:
        tag: Log label.
        rank: Only print on this rank. None = all ranks.
        **kwargs: Variables to print.

    Usage:
        from verl.utils.debug_utils import debug_log
        debug_log("generate_sequences",
                  batch_size=batch_size,
                  input_shape=input_ids.shape,
                  eos_token_id=eos_token_id)
    """
    if not os.getenv("VERL_DEBUG", ""):
        return  # only output when VERL_DEBUG env var is set
    
    current_rank = int(os.getenv("RANK", "0"))
    if rank is not None and current_rank != rank:
        return
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    parts = [f"[DEBUG {timestamp} R{current_rank}] {tag}:"]
    
    import torch
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            parts.append(f"  {key}: Tensor(shape={value.shape}, dtype={value.dtype}, min={value.min().item():.4f}, max={value.max().item():.4f})")
        elif isinstance(value, (list, tuple)):
            parts.append(f"  {key}: {type(value).__name__}(len={len(value)})")
        else:
            parts.append(f"  {key}: {value}")
    
    print("\n".join(parts))


def debug_watch(tag: str, step: int = None, every_n: int = 1, rank: int = 0, **kwargs):
    """Print variables every N steps for monitoring training progress.

    Args:
        tag: Label name.
        step: Current step number. None = print every call.
        every_n: Print once every N steps.
        rank: Only print on this rank.
        **kwargs: Variables to monitor.
    """
    if step is not None and step % every_n != 0:
        return
    
    current_rank = int(os.getenv("RANK", "0"))
    if current_rank != rank:
        return
    
    import torch
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    step_str = f" step={step}" if step is not None else ""
    parts = [f"[WATCH {timestamp}] {tag}{step_str}:"]
    
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            if value.numel() <= 10:
                parts.append(f"  {key} = {value.tolist()}")
            else:
                parts.append(f"  {key}: shape={value.shape}, mean={value.float().mean():.4f}, std={value.float().std():.4f}")
        else:
            parts.append(f"  {key} = {value}")
    
    print("\n".join(parts))
