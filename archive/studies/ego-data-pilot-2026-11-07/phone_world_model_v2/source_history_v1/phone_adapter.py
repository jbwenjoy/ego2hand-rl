"""Phone COM/world state adapter. No model, optimizer, simulator or normalizer.

The 52 outputs are *physical* increments. They are not the original pen WM's
right/body chart52, even though both arrays have 52 columns. Torch is lazy and
used only for tensor arithmetic; callers supply a separately qualified predictor.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import numpy as np

STATE_SCHEMA = "phone_COM_world_state58:1"
ACTION_SCHEMA = "phone_absolute_position_control20_rad_H20ms:1"
DELTA_SCHEMA = "phone_physical_world_delta52:2"
INPUT_SCHEMA = "phone_world_state58_absolute_control20:2"
STATE_DIM, ACTION_DIM, DELTA_DIM, INPUT_DIM = 58, 20, 52, 78
DELTA_FIELDS = {
    "dq_rad": (0, 20), "dqd_rad_s": (20, 40), "dp_COM_world_m": (40, 43),
    "drotation_world_rad": (43, 46), "dv_COM_world_m_s": (46, 49),
    "domega_world_rad_s": (49, 52),
}
CODEC_PATH = Path(__file__).resolve().parents[1] / "phone_transition_dataset_v1/transition_codec_v1.py"
CODEC_SHA256 = "1bf33a7fd27ae51d38a71c1f7e095dbe3047c4828ee6e65c199117bf536f4cac"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _schema(actual, expected):
    require(actual == expected, f"Explicit schema required: {expected}; a matching dimension is insufficient")


def _codec():
    require(hashlib.sha256(CODEC_PATH.read_bytes()).hexdigest() == CODEC_SHA256,
            "Frozen saved phone codec changed")
    name = "phone_world_model_v2_frozen_saved_codec"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, CODEC_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _numpy(a, width, name):
    a = np.asarray(a)
    require(a.ndim >= 1 and a.shape[-1] == width and a.dtype in (np.dtype("float32"), np.dtype("float64")),
            f"{name}: floating (...,{width}) required")
    require(bool(np.isfinite(a).all()), f"{name}: nonfinite")
    return a


def validate_state_numpy(x, *, state_schema):
    _schema(state_schema, STATE_SCHEMA)
    x = _numpy(x, 58, "state58")
    R = x[..., 43:52].reshape(*x.shape[:-1], 3, 3).astype(np.float64)
    tol = 1e-5 if x.dtype == np.float32 else 1e-10
    require(bool(np.all(np.max(np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)), axis=(-2, -1)) <= tol))
            and bool(np.all(np.abs(np.linalg.det(R) - 1) <= tol)),
            "Raw body-to-world rotation is not near SO(3); no projection/repair")
    return x


def state_action_numpy(x, absolute_u, *, state_schema, action_schema):
    """Return (...,78), state58 followed by actual absolute u20. No EMA/C2."""
    x = validate_state_numpy(x, state_schema=state_schema)
    _schema(action_schema, ACTION_SCHEMA)
    u = _numpy(absolute_u, 20, "absolute_u20")
    require(x.shape[:-1] == u.shape[:-1] and x.dtype == u.dtype, "Explicit equal batch shape/dtype required")
    return np.concatenate((x, u), axis=-1)


def physical_delta_numpy(x, target, *, state_schema, delta_schema):
    """Batch F64 saved-state encode, reusing the frozen world-Log scalar codec.

    Encoding requires SO(3) at the frozen F64 domain (1e-10); a rounded FP32
    rotation that exceeds it fails instead of being orthogonalized. Half-turn
    ambiguity stops the entire call. Decode can preserve rounded FP32 bases.
    """
    x = validate_state_numpy(x, state_schema=state_schema)
    target = validate_state_numpy(target, state_schema=state_schema)
    _schema(delta_schema, DELTA_SCHEMA)
    require(x.shape == target.shape, "Input/target shapes differ")
    codec = _codec()
    rows = [codec.delta52(a, b) for a, b in zip(x.reshape(-1, 58), target.reshape(-1, 58))]
    return np.asarray(rows, dtype=np.float64).reshape(*x.shape[:-1], 52)


def decode_numpy(x, physical_delta, *, state_schema, delta_schema):
    x = validate_state_numpy(x, state_schema=state_schema)
    _schema(delta_schema, DELTA_SCHEMA)
    d = _numpy(physical_delta, 52, "physical_delta52")
    require(x.shape[:-1] == d.shape[:-1], "Input/delta batch shapes differ")
    codec = _codec()
    rows = [codec.apply_delta52(a, b) for a, b in zip(x.reshape(-1, 58), d.reshape(-1, 52))]
    return np.asarray(rows, dtype=np.float64).reshape(*x.shape[:-1], 58)


def _torch_tensor(a, width, name, torch):
    require(isinstance(a, torch.Tensor) and a.ndim >= 1 and a.shape[-1] == width
            and a.dtype in (torch.float32, torch.float64), f"{name}: float tensor (...,{width}) required")
    require(bool(torch.isfinite(a).all()), f"{name}: nonfinite")


def validate_state_torch(x, *, state_schema):
    import torch
    _schema(state_schema, STATE_SCHEMA)
    _torch_tensor(x, 58, "state58", torch)
    R = x[..., 43:52].reshape(*x.shape[:-1], 3, 3)
    tol = 1e-5 if x.dtype == torch.float32 else 1e-10
    I = torch.eye(3, dtype=x.dtype, device=x.device)
    require(bool(((R.transpose(-1, -2) @ R - I).abs().amax(dim=(-2, -1)) <= tol).all())
            and bool(((torch.linalg.det(R) - 1).abs() <= tol).all()),
            "Raw body-to-world rotation is not near SO(3); no projection/repair")
    return x


def so3_exp_torch(world_rotvec):
    """Differentiable Rodrigues Exp, including a finite derivative at zero.

    sqrt sees a strictly positive clamped operand, even in the unselected branch.
    The small-angle series is in theta^2, so autograd never sees norm(0)/0.
    """
    import torch
    _torch_tensor(world_rotvec, 3, "world_rotvec", torch)
    a, b, c = world_rotvec.unbind(-1)
    z = torch.zeros_like(a)
    K = torch.stack((z, -c, b, c, z, -a, -b, a, z), dim=-1).reshape(*a.shape, 3, 3)
    t2 = world_rotvec.square().sum(-1)
    require(bool(torch.isfinite(t2).all()), "Rotation magnitude overflow")
    safe_theta = t2.clamp_min(1e-6).sqrt()
    small = t2 < 1e-6
    A = torch.where(small, 1 - t2 / 6 + t2.square() / 120 - t2.pow(3) / 5040,
                    torch.sinc(safe_theta / torch.pi))
    B = torch.where(small, .5 - t2 / 24 + t2.square() / 720 - t2.pow(3) / 40320,
                    .5 * torch.sinc(safe_theta / (2 * torch.pi)).square())
    I = torch.eye(3, dtype=world_rotvec.dtype, device=world_rotvec.device)
    return I + A[..., None, None] * K + B[..., None, None] * (K @ K)


def state_action_torch(x, absolute_u, *, state_schema, action_schema):
    import torch
    x = validate_state_torch(x, state_schema=state_schema)
    _schema(action_schema, ACTION_SCHEMA)
    _torch_tensor(absolute_u, 20, "absolute_u20", torch)
    require(x.shape[:-1] == absolute_u.shape[:-1] and x.dtype == absolute_u.dtype
            and x.device == absolute_u.device, "Explicit equal batch shape/dtype/device required")
    return torch.cat((x, absolute_u), dim=-1)


def decode_torch(x, physical_delta, *, state_schema, delta_schema):
    """Rnext=Exp(dR_world)@Rbody; all other blocks additive, no wrapping.

    Preserves Torch dtype/device/autograd. There is no conversion to NumPy,
    normalization, projection, detach, implicit FP32 cast, or old checkpoint path.
    """
    import torch
    x = validate_state_torch(x, state_schema=state_schema)
    _schema(delta_schema, DELTA_SCHEMA)
    _torch_tensor(physical_delta, 52, "physical_delta52", torch)
    require(x.shape[:-1] == physical_delta.shape[:-1] and x.dtype == physical_delta.dtype
            and x.device == physical_delta.device, "Input/delta batch shape/dtype/device differ")
    R = x[..., 43:52].reshape(*x.shape[:-1], 3, 3)
    Rnext = so3_exp_torch(physical_delta[..., 43:46]) @ R
    output = torch.cat((x[..., :43] + physical_delta[..., :43], Rnext.flatten(-2),
                        x[..., 52:58] + physical_delta[..., 46:52]), dim=-1)
    require(bool(torch.isfinite(output).all()), "Decoded state overflow")
    return output


def predict_next_torch(x, absolute_u, predict_physical_delta, *, state_schema,
                       action_schema, input_schema, delta_schema):
    """One external callable invocation on raw78; returns (next58,delta52).

    Caller owns any network and new phone-specific normalization/context. This
    is not a loader for native67, old xi46/dynamics278, or old pen weights.
    """
    _schema(input_schema, INPUT_SCHEMA)
    _schema(delta_schema, DELTA_SCHEMA)
    features = state_action_torch(x, absolute_u, state_schema=state_schema, action_schema=action_schema)
    d = predict_physical_delta(features)
    return decode_torch(x, d, state_schema=state_schema, delta_schema=delta_schema), d
