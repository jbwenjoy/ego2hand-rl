"""Pure NumPy phone COM/world state and integer-tick contract, version 1.

R_body_world maps BODY coordinates to world. It is not ximat/inertial R.
Raw free-joint translation velocity is world-frame; angular velocity is
body-frame. The new state uses world COM velocity AND world angular velocity.
No quaternion conversion, FK, dynamics, native model or learning code lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import numpy as np

STATE_SCHEMA = "phone_COM_world_state58:1"
TIME_SCHEMA = "phone_control_integer_ticks:1"
STATE_SIZE = 58
NATIVE_CHART_SIZE = 52  # Dimensional compatibility only; no LogR/chart is supplied.
ROTATION_ATOL = 1e-5  # Input arithmetic validation, permits saved FP32 rotations.
STATE_FIELDS = {
    "q20_rad": (0, 20), "qd20_rad_s": (20, 40),
    "p_COM_world_m": (40, 43), "R_body_world_rowmajor": (43, 52),
    "v_COM_world_m_s": (52, 55), "omega_world_rad_s": (55, 58),
}


def _float_array(value, name):
    a = np.asarray(value)
    if a.dtype.kind != "f" or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite floating-point array")
    return np.asarray(a, dtype=np.float64)


def _field(value, width, name):
    a = _float_array(value, name)
    if a.ndim < 1 or a.shape[-1] != width:
        raise ValueError(f"{name} must have shape (...,{width})")
    return a


def _rotation(value):
    a = _float_array(value, "R_body_world")
    if a.ndim >= 2 and a.shape[-2:] == (3, 3):
        r = a
    elif a.ndim >= 1 and a.shape[-1] == 9:
        r = a.reshape(a.shape[:-1] + (3, 3))  # C/row-major logical interpretation.
    else:
        raise ValueError("R_body_world must have shape (...,3,3) or (...,9)")
    err = np.max(np.abs(np.swapaxes(r, -1, -2) @ r - np.eye(3)))
    if err > ROTATION_ATOL or np.any(np.abs(np.linalg.det(r) - 1) > ROTATION_ATOL):
        raise ValueError("R_body_world must be a proper rotation; no repair is performed")
    return r


def _ipos(value, leading_shape):
    a = _field(value, 3, "body_ipos_body_m")
    if a.shape != (3,) and a.shape[:-1] != leading_shape:
        raise ValueError("body_ipos must be one fixed (3,) offset or match the state batch")
    return np.broadcast_to(a, leading_shape + (3,))


def _rotate(r, v):
    return np.einsum("...ij,...j->...i", r, v)


def to_com_world_state(q20, qd20, p_root_world, R_body_world,
                       v_root_world, omega_body, body_ipos_body):
    """Return float64 (...,58); only body_ipos may broadcast over the batch.

    pCOM = proot + R*ipos
    omega_world = R*omega_body
    vCOM = vroot + omega_world x (R*ipos)
    q/qd are copied numerically without wrapping, normalization or rescaling.
    FP32 inputs are exactly representable after their explicit float64 cast.
    """
    q = _field(q20, 20, "q20")
    qd = _field(qd20, 20, "qd20")
    p = _field(p_root_world, 3, "p_root_world")
    v = _field(v_root_world, 3, "v_root_world")
    w = _field(omega_body, 3, "omega_body")
    r = _rotation(R_body_world)
    lead = q.shape[:-1]
    if any(a.shape[:-1] != lead for a in (qd, p, v, w)) or r.shape[:-2] != lead:
        raise ValueError("all raw fields must share the same leading batch shape")
    offset = _rotate(r, _ipos(body_ipos_body, lead))
    ww = _rotate(r, w)
    return np.concatenate((q, qd, p + offset, r.reshape(lead + (9,)),
                           v + np.cross(ww, offset), ww), axis=-1)


def from_com_world_state(state58, body_ipos_body):
    """Inverse to_com_world_state, returning explicit raw-frame float64 fields.

    Solve against the supplied R for omega_body. This equals R.T for an exact
    rotation and preserves a numerical inverse for slightly rounded FP32 R.
    No orthogonalization, learned normalization or bit-exact promise is made.
    """
    s = _field(state58, STATE_SIZE, "state58")
    lead = s.shape[:-1]
    r = _rotation(s[..., 43:52])
    offset = _rotate(r, _ipos(body_ipos_body, lead))
    ww = s[..., 55:58]
    return {
        "q20": s[..., :20].copy(), "qd20": s[..., 20:40].copy(),
        "p_root_world": s[..., 40:43] - offset,
        "R_body_world": r.copy(),
        "v_root_world": s[..., 52:55] - np.cross(ww, offset),
        "omega_body": np.linalg.solve(r, ww[..., None])[..., 0],
    }


@dataclass(frozen=True)
class TickPlan:
    """One 20 ms control interval; ticks are supplied, never inferred from time."""
    integration_dt_s: float
    control_dt_s: float = .02

    def __post_init__(self):
        h = Decimal(str(self.integration_dt_s))
        control = Decimal(str(self.control_dt_s))
        if control != Decimal(".02") or h not in (Decimal(".0003125"), Decimal(".00015625")):
            raise ValueError("version 1 permits H=.02 and h=.0003125 or .00015625 only")
        n = control / h
        if n != n.to_integral_value():
            raise ValueError("control interval must contain an exact integer number of ticks")
        object.__setattr__(self, "substeps", int(n))

    def _ticks(self, ticks):
        t = np.asarray(ticks)
        if t.dtype.kind not in "iu" or np.any(t < 0) or np.any(t > self.substeps):
            raise ValueError("ticks must be supplied integers in [0,substeps]")
        return t.astype(np.int64, copy=False)

    def nominal_time_s(self, ticks):
        return self._ticks(ticks).astype(np.float64) * self.integration_dt_s

    def masks(self, ticks):
        t = self._ticks(ticks)
        return {"initial": t == 0, "terminal_POST": t == self.substeps,
                "solver_PRE": t < self.substeps, "integrated_POST": t > 0}

    def native_clock_descriptors(self, native_time_s, ticks):
        """Readback descriptor only: no time/h rounding or eligibility threshold."""
        t = self._ticks(ticks)
        native = _float_array(native_time_s, "native_time_s")
        if native.shape != t.shape:
            raise ValueError("native clocks and supplied ticks must have identical shape")
        nominal = self.nominal_time_s(t)
        return {"tick": t.copy(), "native_time_s": native.copy(),
                "nominal_time_s": nominal, "native_minus_nominal_s": native - nominal}

    def declaration(self):
        return {"schema": TIME_SCHEMA, "control_dt_s": self.control_dt_s,
                "requested_integration_dt_s": self.integration_dt_s,
                "integration_dt_FP32_s": float(np.float32(self.integration_dt_s)),
                "substeps": self.substeps, "state_ticks": self.substeps + 1,
                "solver_PRE_ticks": [0, self.substeps - 1],
                "integrated_POST_ticks": [1, self.substeps],
                "last_solver_PRE_nominal_s": (self.substeps - 1) * self.integration_dt_s,
                "terminal_POST_nominal_s": self.substeps * self.integration_dt_s,
                "native_clock_is_separate_readback": True,
                "ticks_inferred_from_accumulated_float_time": False}
