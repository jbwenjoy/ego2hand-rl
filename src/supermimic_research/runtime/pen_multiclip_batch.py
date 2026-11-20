"""Two distinct guarded W32 tasks sharing a solver binding and a policy interface."""
from contextlib import ExitStack, contextmanager, nullcontext
import json
from pathlib import Path
import tomllib

import numpy as np
import torch

from ..artifacts import file_pin, verify_file_pin
from .pen_clip_start import clip_start_constructor
from .pen_guarded_w32 import GuardedW32PenRuntime, guarded_w32


def _toml(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def multiclip_inputs(objective_001, objective_003):
    """Resolve and verify development references before constructing a CUDA graph."""
    inputs = {}
    for name, objective in (("clip_001", objective_001), ("clip_003", objective_003)):
        objective = Path(objective).resolve()
        adapter_file = (objective.parent / _toml(objective)["adapter_config"]).resolve()
        adapter = _toml(adapter_file)
        base_file = (adapter_file.parent / adapter["base_config"]).resolve()
        base = _toml(base_file)
        recipe = (base_file.parent / base["pen_recipe"]).resolve()
        expected_id = f"01_single_axis/{name}"
        if json.loads(recipe.read_text())["clips"] != [expected_id]:
            raise ValueError(f"Expected a single-clip recipe for {expected_id}")
        source = (base_file.parent / base["clips_root"] / (expected_id + ".npz")).resolve()
        index_file = source.parent / "index.json"
        catalog = json.loads(index_file.read_text())
        entries = [item for item in catalog["clips"] if item["clip_id"] == name]
        source_pin = file_pin(source)
        if (len(entries) != 1 or entries[0]["file"] != source.name
                or entries[0]["sha256"] != source_pin["sha256"] or catalog["frame_dt"] != .02):
            raise ValueError("Reference bytes or source clock differ from the catalog")
        with np.load(source) as f:
            frame_count = len(f["reference_qpos"])
        if name == "clip_001" and frame_count != 369:
            raise ValueError("Frozen primary reference must have 369 frames")
        entry = dict(objective=file_pin(objective), adapter=file_pin(adapter_file), base=file_pin(base_file),
                     recipe=file_pin(recipe), catalog=file_pin(index_file), clip=source_pin, frames=frame_count)
        if name == "clip_003":
            if adapter["clip_pin"] != source_pin or adapter["frame_count"] != frame_count:
                raise ValueError("Development adapter and source differ")
            for key in ("fixed_reference_pin", "projected_pin", "control_pin"):
                verify_file_pin(adapter[key])
                entry[key] = adapter[key]
        inputs[name] = entry
    return inputs


def _check_start(runtime, spec, name):
    sim = runtime.env.sim
    if (runtime.num_envs != 32 or runtime.owned_native_integrations != 0
            or sim.step_graph is None or sim.forward_graph is None
            or sim.wp_model.nv != 26 or sim.wp_model.nv_pad != 28):
        raise ValueError(f"Unexpected initial state or graph layout for {name}")
    with np.load(verify_file_pin(spec["clip"])) as f:
        reference = np.asarray(f["reference_qpos"], dtype=np.float32)
    np.testing.assert_array_equal(runtime.motion._frames["joint_pos"].cpu().numpy(), reference)
    phases = runtime.motion.time_steps.cpu().numpy()
    if phases.shape != (32,) or not np.all(phases == 1):
        raise ValueError("Fresh W32 worlds must start at phase one")
    pose = runtime.support_table.pose
    if pose.shape != (spec["frames"], 7):
        raise ValueError("Support sequence has the wrong clock")
    if name == "clip_003":
        with np.load(verify_file_pin(spec["fixed_reference_pin"])) as f:
            np.testing.assert_allclose(pose[:, :3], f["COM"], atol=1e-6, rtol=0)
            if np.min(np.abs(np.sum(pose[:, 3:] * f["quat_wxyz"], axis=1))) <= 1 - 1e-6:
                raise ValueError("Fixed-wrist orientation differs from the development reference")
        with np.load(verify_file_pin(spec["projected_pin"])) as f:
            hand = torch.tensor(f["initial_hand_q"], device=runtime.device, dtype=torch.float32)
        with np.load(verify_file_pin(spec["control_pin"])) as f:
            ctrl = torch.tensor(f["ctrl"], device=runtime.device, dtype=torch.float32)
        start = sim.data.qpos[:, runtime.features.qids]
        target = sim.data.ctrl[:, runtime.action._position_ctrl_ids]
        torch.testing.assert_close(start, hand.expand_as(start), atol=0, rtol=0)
        torch.testing.assert_close(target, ctrl.expand_as(target), atol=2e-7, rtol=0)
    runtime.validate_resources()


@contextmanager
def multiclip_w32(objective_001, objective_003, factory_manifest):
    """Enter inside backend_scope. Each task owns its graphs, reset state, data and references."""
    inputs = multiclip_inputs(objective_001, objective_003)
    with guarded_w32(factory_manifest) as binding, ExitStack() as cleanup:
        tasks = {}
        for name, objective in (("clip_001", objective_001), ("clip_003", objective_003)):
            calls = len(binding.audit["calls"])
            with clip_start_constructor() if name == "clip_003" else nullcontext():
                runtime = GuardedW32PenRuntime(Path(objective).resolve(), 32)
            cleanup.callback(runtime.close)
            if len(binding.audit["calls"]) <= calls:
                raise ValueError("New runtime did not invoke the guarded solver factory")
            _check_start(runtime, inputs[name], name)
            tasks[name] = runtime
        a, b = (task.env.sim for task in tasks.values())
        if (a.wp_data.qpos.ptr == b.wp_data.qpos.ptr or a.step_graph is b.step_graph
                or a.forward_graph is b.forward_graph):
            raise ValueError("Independent tasks unexpectedly share physics data or graphs")
        yield tasks
