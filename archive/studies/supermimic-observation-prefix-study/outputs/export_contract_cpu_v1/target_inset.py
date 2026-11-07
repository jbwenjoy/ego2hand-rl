"""FP32 absolute-target projection for C2; no physical-state guarantee.

The caller validates its fixed domain once and only copies the returned value
to the absolute target cache. EMA residual, q/qd and object state are not inputs.
All margins, including zero, execute the same tensor operations.
"""
from __future__ import annotations

import math

import torch


def validate_inset_domain(lower: torch.Tensor, upper: torch.Tensor, margin_rad: float) -> None:
    """Initialization-only validation; the reductions may synchronize a device."""
    if not math.isfinite(margin_rad) or margin_rad < 0:
        raise ValueError("margin_rad must be finite and nonnegative")
    if lower.device != upper.device or not lower.is_floating_point() or not upper.is_floating_point():
        raise ValueError("Limits must be floating tensors on the same device")
    torch.broadcast_shapes(lower.shape, upper.shape)
    with torch.autocast(device_type=lower.device.type, enabled=False):
        lo, hi = lower.float() + margin_rad, upper.float() - margin_rad
        if not bool((torch.isfinite(lo) & torch.isfinite(hi) & (lo <= hi)).all()):
            raise ValueError("Nonfinite or empty inset command domain")


def target_inset(proposed_u: torch.Tensor, lower: torch.Tensor,
                 upper: torch.Tensor, margin_rad: float) -> torch.Tensor:
    """Project a command without mutating any input; output is always FP32.

    Use validate_inset_domain once before rollout. This function deliberately
    has no special zero-margin path or observation/physics side effects.
    """
    with torch.autocast(device_type=proposed_u.device.type, enabled=False):
        return proposed_u.float().clamp(min=lower.float() + margin_rad,
                                        max=upper.float() - margin_rad)
