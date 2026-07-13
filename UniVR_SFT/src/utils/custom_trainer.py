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
Custom Trainer with anomaly detection for gradient norm spikes and loss anomalies.

Features:
- Tracks loss with exponential moving average (EMA)
- Detects loss spikes (loss >> EMA) and NaN/Inf losses
- Detects abnormal gradient norms
- Logs detailed sample information when anomalies are detected
- Optionally skips gradient updates for anomalous batches
"""

import json
import logging
import math
import os
import torch
from collections import deque
from datetime import datetime
from typing import Optional, Dict, List

from transformers import Trainer

logger = logging.getLogger(__name__)


class AnomalyAwareDataCollator:
    """
    Data collator that extracts non-tensor fields (e.g., sample_info)
    before stacking tensor fields into a batch.

    Robust to upstream bugs where a field comes through as a Python list
    instead of a torch.Tensor: it coerces with torch.as_tensor and warns
    once (with the offending key + types) so the dataset bug can be fixed.
    """

    _warned_keys: set = set()

    def __call__(self, features):
        # Extract sample_info before stacking tensors
        sample_infos = [f.pop("sample_info", "unknown") for f in features]

        batch = {}
        for key in features[0]:
            values = [f[key] for f in features]
            first = values[0]

            if isinstance(first, torch.Tensor):
                batch[key] = torch.stack(values)
                continue

            # Non-tensor field reached the collator. Warn once per key with
            # diagnostic info — this points at a dataset __getitem__ that
            # forgot to wrap a field with torch.tensor(...).
            if key not in AnomalyAwareDataCollator._warned_keys:
                AnomalyAwareDataCollator._warned_keys.add(key)
                types_seen = sorted({type(v).__name__ for v in values})
                sample_len = len(first) if hasattr(first, "__len__") else "?"
                logger.warning(
                    f"[Collator] field '{key}' is not a Tensor (types={types_seen}, "
                    f"first-element-len={sample_len}). Coercing with torch.as_tensor. "
                    f"Fix the dataset to return torch.tensor(...) for this field."
                )

            try:
                coerced = [torch.as_tensor(v) for v in values]
                batch[key] = torch.stack(coerced)
            except Exception as e:
                # Can't stack (e.g. variable-length lists or non-numeric).
                # Pass through as a Python list so training can continue;
                # the model will likely ignore unknown columns anyway.
                logger.warning(
                    f"[Collator] could not stack field '{key}': {e}. "
                    f"Passing through as list of length {len(values)}."
                )
                batch[key] = values

        # Add back sample_info as list of strings (not a tensor)
        batch["sample_info"] = sample_infos

        return batch


class AnomalyAwareTrainer(Trainer):
    """
    Extended HuggingFace Trainer that detects and logs anomalous training steps.
    
    When a loss spike or gradient norm spike is detected, it logs the full
    sample metadata (dataset source, video name, subtask, frame counts, etc.)
    to help identify problematic training data.
    
    Args:
        loss_spike_threshold (float): A batch loss exceeding EMA * this threshold
            triggers an anomaly warning. Default: 5.0
        grad_norm_threshold (float): A gradient norm exceeding this threshold
            triggers an anomaly warning. Default: 10.0
        skip_anomaly_batches (bool): If True, zero out the loss for anomalous
            batches to skip their gradient updates. Default: False
    """
    
    def __init__(self, *args,
                 loss_spike_threshold: float = 5.0,
                 loss_spike_delta_threshold: float = 0,
                 grad_norm_threshold: float = 10.0,
                 skip_anomaly_batches: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_spike_threshold = loss_spike_threshold
        self.loss_spike_delta_threshold = loss_spike_delta_threshold
        self.grad_norm_threshold = grad_norm_threshold
        self.skip_anomaly_batches = skip_anomaly_batches
        
        # Loss tracking with EMA
        self.loss_ema = None
        self.loss_ema_alpha = 0.99
        
        # Recent sample buffer for correlating grad_norm spikes with samples
        # Keep enough for gradient_accumulation_steps * 2
        buffer_size = max(getattr(self.args, 'gradient_accumulation_steps', 1) * 2, 10)
        self._recent_samples = deque(maxlen=buffer_size)
        self._current_sample_info = None
        self.anomaly_count = 0
        
        # Anomaly log file in output_dir
        self._anomaly_log_path = os.path.join(self.args.output_dir, "anomaly_log.jsonl")
        os.makedirs(self.args.output_dir, exist_ok=True)
    
    def _set_signature_columns_if_needed(self):
        """Override to preserve 'sample_info' from being stripped by RemoveColumnsCollator.
        
        HuggingFace Trainer wraps the data collator with RemoveColumnsCollator when
        remove_unused_columns=True (the default). This wrapper filters out any column
        not in the model's forward() signature BEFORE our collator sees it. We override
        this method to add 'sample_info' to the preserved columns list.
        """
        super()._set_signature_columns_if_needed()
        if "sample_info" not in self._signature_columns:
            self._signature_columns.append("sample_info")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to extract sample_info and detect loss anomalies."""
        # Extract sample_info (non-tensor field, won't be sent to model)
        sample_info = inputs.pop("sample_info", None)
        self._current_sample_info = sample_info
        
        # Compute loss normally
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        
        if return_outputs:
            loss_val = loss[0].detach().item()
        else:
            loss_val = loss.detach().item()
        
        # ---- Check for NaN / Inf loss ----
        if not math.isfinite(loss_val):
            self.anomaly_count += 1
            self._log_anomaly(
                anomaly_type="NAN_INF_LOSS",
                step=self.state.global_step,
                loss_value=loss_val,
                loss_ema=self.loss_ema,
                grad_norm=None,
                sample_info=sample_info,
            )
            if self.skip_anomaly_batches and self.state.global_step > 500:
                logger.warning("[ANOMALY] Skipping batch due to NaN/Inf loss")
                if return_outputs:
                    return (loss[0] * 0, loss[1])
                else:
                    return loss * 0
        
        # ---- Loss spike detection (EMA based) ----
        if self.loss_ema is None:
            self.loss_ema = loss_val
        else:
            is_spike = False
            spike_reason = ""
            if math.isfinite(loss_val):
                # Delta-based detection: loss > EMA + delta_threshold
                if (self.loss_spike_delta_threshold > 0
                        and loss_val > self.loss_ema + self.loss_spike_delta_threshold):
                    is_spike = True
                    spike_reason = (f"loss={loss_val:.4f} > EMA({self.loss_ema:.4f}) + "
                                    f"delta({self.loss_spike_delta_threshold})")
                # Multiplicative detection: loss > threshold * EMA (legacy, if configured)
                elif (self.loss_spike_threshold > 0 and self.loss_ema > 0.01
                        and loss_val > self.loss_spike_threshold * self.loss_ema):
                    is_spike = True
                    spike_reason = (f"loss={loss_val:.4f} > {self.loss_spike_threshold} * "
                                    f"EMA({self.loss_ema:.4f})")

            if is_spike:
                self.anomaly_count += 1
                self._log_anomaly(
                    anomaly_type="LOSS_SPIKE",
                    step=self.state.global_step,
                    loss_value=loss_val,
                    loss_ema=self.loss_ema,
                    grad_norm=None,
                    sample_info=sample_info,
                )
                if self.skip_anomaly_batches and self.state.global_step > 500:
                    logger.warning(
                        f"[ANOMALY] Skipping batch: {spike_reason}"
                    )
                    if return_outputs:
                        return (loss[0] * 0, loss[1])
                    else:
                        return loss * 0
            # Update EMA (only with finite values)
            if math.isfinite(loss_val):
                self.loss_ema = self.loss_ema_alpha * self.loss_ema + (1 - self.loss_ema_alpha) * loss_val
        
        # Store in recent buffer for grad_norm correlation
        self._recent_samples.append({
            "step": self.state.global_step,
            "loss": loss_val,
            "sample_info": sample_info,
        })
        
        return loss

    def log(self, logs: Dict[str, float], start_time=None, **kwargs) -> None:
        """Override to check grad_norm from logged metrics."""
        if "grad_norm" in logs:
            grad_norm = logs["grad_norm"]
            if isinstance(grad_norm, (int, float)) and math.isfinite(grad_norm):
                if grad_norm > self.grad_norm_threshold:
                    self.anomaly_count += 1
                    recent = list(self._recent_samples)
                    self._log_anomaly(
                        anomaly_type="GRAD_NORM_SPIKE",
                        step=self.state.global_step,
                        loss_value=logs.get("loss"),
                        loss_ema=self.loss_ema,
                        grad_norm=grad_norm,
                        sample_info=None,
                        recent_samples=recent,
                    )
            elif not math.isfinite(grad_norm):
                self.anomaly_count += 1
                recent = list(self._recent_samples)
                self._log_anomaly(
                    anomaly_type="NAN_INF_GRAD_NORM",
                    step=self.state.global_step,
                    loss_value=logs.get("loss"),
                    loss_ema=self.loss_ema,
                    grad_norm=grad_norm,
                    sample_info=None,
                    recent_samples=recent,
                )
        # Pass all arguments to parent (compatible with both old and new transformers)
        if start_time is not None:
            super().log(logs, start_time, **kwargs)
        else:
            super().log(logs, **kwargs)
    
    def _log_anomaly(self, anomaly_type, step, loss_value, loss_ema, grad_norm,
                     sample_info=None, recent_samples=None):
        """Log detailed anomaly information to help identify problematic samples."""
        msg_lines = [
            f"\n{'='*80}",
            f"⚠️  TRAINING ANOMALY DETECTED: {anomaly_type}",
            f"{'='*80}",
            f"  Global Step     : {step}",
            f"  Current Loss    : {loss_value}",
            f"  Loss EMA        : {loss_ema}",
        ]
        
        if grad_norm is not None:
            msg_lines.append(f"  Grad Norm       : {grad_norm}")
            msg_lines.append(f"  Grad Threshold  : {self.grad_norm_threshold}")
        
        if (loss_value is not None and loss_ema is not None
                and isinstance(loss_value, (int, float)) and loss_ema > 0):
            try:
                msg_lines.append(f"  Loss Spike Ratio: {loss_value / loss_ema:.2f}x")
            except Exception:
                pass
        
        msg_lines.append(f"  Total Anomalies : {self.anomaly_count}")
        msg_lines.append(f"  Skip Enabled    : {self.skip_anomaly_batches}")
        
        # --- Current batch sample info ---
        if sample_info is not None:
            msg_lines.append(f"  {'─'*60}")
            msg_lines.append(f"  Current Batch Samples:")
            if isinstance(sample_info, list):
                for i, info in enumerate(sample_info):
                    msg_lines.append(f"    [GPU Sample {i}] {info}")
            else:
                msg_lines.append(f"    {sample_info}")
        
        # --- Recent samples (for grad_norm spike correlation) ---
        if recent_samples:
            msg_lines.append(f"  {'─'*60}")
            msg_lines.append(f"  Recent Micro-Batches ({len(recent_samples)} entries):")
            for entry in recent_samples:
                s_info = entry.get("sample_info", "unknown")
                s_loss = entry.get("loss", float('nan'))
                s_step = entry.get("step", "?")
                loss_str = f"{s_loss:.4f}" if isinstance(s_loss, (int, float)) else str(s_loss)
                msg_lines.append(f"    [Step {s_step}, Loss={loss_str}]")
                if isinstance(s_info, list):
                    for i, info in enumerate(s_info):
                        msg_lines.append(f"      [Sample {i}] {info}")
                elif s_info is not None:
                    msg_lines.append(f"      {s_info}")
        
        msg_lines.append(f"{'='*80}\n")
        
        full_msg = "\n".join(msg_lines)
        logger.warning(full_msg)
        # Also print to stdout for visibility in training logs
        print(full_msg, flush=True)
        
        # ---- Persist to local anomaly log file (JSONL) ----
        try:
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "anomaly_type": anomaly_type,
                "global_step": step,
                "loss": loss_value if isinstance(loss_value, (int, float)) else str(loss_value),
                "loss_ema": loss_ema,
                "grad_norm": grad_norm,
                "total_anomalies": self.anomaly_count,
                "skip_enabled": self.skip_anomaly_batches,
            }
            if sample_info is not None:
                record["sample_info"] = sample_info if isinstance(sample_info, list) else [sample_info]
            if recent_samples:
                record["recent_samples"] = [
                    {"step": e.get("step"), "loss": e.get("loss"),
                     "sample_info": e.get("sample_info")}
                    for e in recent_samples
                ]
            with open(self._anomaly_log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"[ANOMALY] Failed to write anomaly log to {self._anomaly_log_path}: {e}")
