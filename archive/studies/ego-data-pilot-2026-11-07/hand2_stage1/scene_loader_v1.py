"""Exact Hand2/phone staging interfaces; no forward or integration on import.

load_model() returns the self-contained MuJoCo 3.11.0 binary model.
load_spec() returns a fresh source-identical MjSpec for later explicit editing.
Neither function creates MjData, performs forward, or moves the fixed wrist.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "completed_v2"


def _checked_report():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    for pin in report["source_pins"].values():
        path = Path(pin["path"])
        if path.stat().st_size != pin["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
            raise ValueError("Changed source: " + str(path))
    return report


def load_model():
    """Return exact compiled MjModel; no MjData/forward/physics step."""
    report = _checked_report()
    import mujoco
    if mujoco.__version__ != report["software"]["mujoco"]:
        raise ValueError("Binary model is version-specific; use load_spec with explicit new qualification")
    path = OUTPUT / "hand2_phone_scene.mjb"
    record = report["output_pins"][path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Changed binary model")
    return mujoco.MjModel.from_binary_path(str(path))


def load_spec():
    """Return fresh editable MjSpec via the reviewed existing hand factory.

    Do not serialize to XML and assume bit-identical gains/poses: MuJoCo's
    default XML writer rounds numeric values. Compile this MjSpec directly.
    """
    report = _checked_report()
    builder_path = Path(report["source_pins"]["builder"]["path"])
    module_spec = importlib.util.spec_from_file_location("hand2_phone_stage1_exact_builder", builder_path)
    builder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(builder)
    placement = json.loads((OUTPUT / "palm_keypoints_and_placement.json").read_text(encoding="utf-8"))
    spec, _cfg = builder.make_hand_spec()
    transform = placement["T_world_from_phone_model"]
    pos = [row[3] for row in transform[:3]]
    quat = placement["fixed_source_wrist_pos_quat_wxyz"][3:]
    return builder.attach_phone_and_table(spec, pos, quat)


def initial_state():
    """Return copied raw initial arrays, including qpos27/qvel26/ctrl20.

    Caller explicitly owns MjData construction and future forward/step budget.
    """
    report = _checked_report()
    import numpy as np
    path = OUTPUT / "zero_step_state.npz"
    expected = report["output_pins"][path.name]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Changed zero-step state")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}
