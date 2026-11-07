"""Bounded, known-task Warp braking response diagnostic; no training.

The frozen evaluator owns policy/reset/20 ms stepping. Instrumentation copies
measurements only. Branches restore the complete same-world snapshot once, then
change absolute actuator targets only. The 0.1 rad pulse does not calibrate a
guaranteed stopping bound or enable C1. GPU execution requires the separate
reviewed launcher; --validate-only imports only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

STARTED = time.perf_counter()
STARTED_UTC = datetime.now(timezone.utc).isoformat()
SOFT_SECONDS = 180.0
ROOT = Path(__file__).resolve().parents[1]
STUDY = Path("D:/workspace/supermimic-chord-study")
PHYSICS_FIELDS = ("body_mass", "body_inertia", "body_iquat", "geom_size",
                  "geom_pos", "geom_friction", "actuator_gainprm", "actuator_biasprm")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def deadline():
    if time.perf_counter() - STARTED > SOFT_SECONDS:
        raise TimeoutError("180 s whole-worker soft cap; partial artifacts retained, no retry")


def pulse_sign(qd, vmin=.02):
    """Sign of initial velocity; near-zero chooses declared upper-inward."""
    return 1.0 if abs(qd) <= vmin or qd > 0 else -1.0


def cases_for(plan, method):
    for frame in plan["snapshot_frames"]:
        sid = f"{method}_seed1_frame{frame:03d}"
        for repeat in range(3):
            yield {"snapshot_id": sid, "frame": frame, "kind": "baseline", "repeat": repeat,
                   "pulse_joint": None, "branch_id": f"{sid}_baseline{repeat}"}
        for joint in plan["joint_cases"]:
            yield {"snapshot_id": sid, "frame": frame, "kind": "pulse", "repeat": None,
                   "pulse_joint": joint["joint"], "branch_id": f"{sid}_pulse_{joint['joint']}"}


def validate(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    checks = {}
    checks["fixed_cases"] = (plan["snapshot_frames"] == [40, 80, 120]
                             and plan["baseline_repeats"] == 3 and plan["total_branches"] == 54)
    checks["fixed_clocks"] = (plan["physics_dt_s"] == .005 and plan["control_dt_s"] == .02
                              and plan["steps_per_branch"] == 40 and plan["states_per_branch"] == 41)
    checks["no_parameter_enablement"] = (plan["m0"] is None and plan["a_brake"] is None
                                        and not plan["parameter_enablement"] and plan["training_updates"] == 0)
    checks["caps"] = plan["process_wall_cap_seconds"] == 240 and plan["worker_soft_cap_seconds"] == 180
    checks["case_count"] = sum(len(list(cases_for(plan, s["method"]))) for s in plan["sources"]) == 54
    checks["source_order"] = [s["method"] for s in plan["sources"]] == ["A1L", "A2", "P_ONLY"]
    checks["upper_side_separate_from_velocity"] = all(j["limit_side"] == "upper" for j in plan["joint_cases"])
    checks["fixed_pulse"] = plan["pulse_amplitude_rad"] == .1
    checks["runtime_self_pin"] = (Path(plan["runtime_path"]).resolve() == Path(__file__).resolve()
                                   and sha256(__file__) == plan["runtime_sha256"])
    checks["nearzero_rule"] = [pulse_sign(v) for v in [-.03, -.02, 0, .02, .03]] == [-1, 1, 1, 1, 1]
    for pathkey in ("reference", "snapshot_contract", "raw_schema_file"):
        checks[pathkey + "_hash"] = sha256(plan[pathkey]) == plan[pathkey + "_sha256"]
    for key, pin in plan["source_pins"].items():
        checks["pin_" + key] = sha256(pin["path"]) == pin["sha256"]
    for source in plan["sources"]:
        prefix = source["method"] + "_"
        for key in ("checkpoint", "model", "context"):
            checks[prefix + key] = sha256(source[key]) == source[key + "_sha256"]
        archive = Path(source["archived_evaluation"])
        checks[prefix + "archive_bound"] = archive.is_dir() and bool(source["archived_hashes"])
        for name, digest in source["archived_hashes"].items():
            checks[prefix + "archive_" + name] = sha256(archive / name) == digest
    if not all(checks.values()):
        raise ValueError(f"frozen-plan validation failed: {[k for k,v in checks.items() if not v]}")
    return plan, {"status": "passed", "checks": checks, "check_count": len(checks),
                  "plan_sha256": sha256(plan_path), "runtime_sha256": sha256(__file__),
                  "imports": "stdlib only; no GPU initialization or physical execution"}


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Recorder:
    """Read Warp output without an additional forward/step or physical write."""

    def __init__(self, env, evaluation, plan):
        import mujoco
        import numpy as np
        self.env, self.ev, self.plan, self.np, self.mj = env, evaluation, plan, np, mujoco
        self.model = model = env.sim.mj_model
        self.host = mujoco.MjData(model)
        self.names = list(env.scene["robot"].joint_names)
        if len(self.names) != 20:
            raise ValueError("expected twenty ordered finger joints")
        self.jids = np.array([model.joint("robot/" + n).id for n in self.names])
        self.qa, self.va = model.jnt_qposadr[self.jids], model.jnt_dofadr[self.jids]
        self.aids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j).item() for j in self.jids])
        self.oid = model.body("object/cube").id
        free = model.joint("object/cube_freejoint").id
        self.oq, self.ov = int(model.jnt_qposadr[free]), int(model.jnt_dofadr[free])
        self.ogeoms = set(np.flatnonzero(model.geom_bodyid == self.oid).tolist())
        import re
        self.rgeoms = {i for i in range(model.ngeom) if re.fullmatch(
            r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", model.geom(i).name)}
        lo, hi = model.jnt_range[self.jids].T
        self.lower = np.maximum(lo, model.actuator_ctrlrange[self.aids, 0])
        self.upper = np.minimum(hi, model.actuator_ctrlrange[self.aids, 1])
        if not np.all(model.actuator_ctrllimited[self.aids]) or not np.all(self.lower <= self.upper):
            raise ValueError("bounded joint/actuator intersection unavailable")
        self.static = {"joint_lower_rad": lo.copy(), "joint_upper_rad": hi.copy(),
                       "force_limit_Nm": np.max(np.abs(model.actuator_forcerange[self.aids]), -1)}
        if not (np.all(model.actuator_trntype[self.aids] == int(mujoco.mjtTrn.mjTRN_JOINT))
                and np.all(model.actuator_gear[self.aids, 0] == 1)
                and np.all(model.actuator_gear[self.aids, 1:] == 0)
                and np.all(model.actuator_gaintype[self.aids] == int(mujoco.mjtGain.mjGAIN_FIXED))
                and np.all(model.actuator_biastype[self.aids] == int(mujoco.mjtBias.mjBIAS_AFFINE))
                and np.all(model.actuator_gainprm[self.aids, 0] > 0)
                and np.allclose(model.actuator_gainprm[self.aids, 0], -model.actuator_biasprm[self.aids, 1])):
            raise ValueError("force-as-joint-torque requires verified unit-gear affine position PD")
        self.availability = {"qacc_rad_s2": True, "qfrc_constraint": True, "joint_limit_active": True}
        self.capacities = []

    def read(self, post_forward=False):
        import mujoco_warp as mjwarp
        import warp as wp
        deadline()
        wp.synchronize_device(self.env.device)
        mjwarp.get_data_into(self.host, self.model, self.env.sim.wp_data, world_id=0)
        h, np = self.host, self.np
        if not np.isfinite(h.qpos).all() or not np.isfinite(h.qvel).all():
            raise FloatingPointError("nonfinite actual Warp integration state")
        c = self.ev.capture_contacts(self.model, h, self.oid, self.ogeoms, self.rgeoms)
        active = np.zeros(20, dtype=np.int8)
        try:
            ids = np.asarray(h.efc_id)[:h.nefc]
            typ = np.asarray(h.efc_type)[:h.nefc]
            for j, jid in enumerate(self.jids):
                active[j] = np.any((typ == int(self.mj.mjtConstraint.mjCNSTR_LIMIT_JOINT)) & (ids == jid))
        except (AttributeError, ValueError):
            self.availability["joint_limit_active"] = False
            active.fill(-1)
        try:
            qacc = h.qacc[self.va].copy()
        except AttributeError:
            self.availability["qacc_rad_s2"] = False
            qacc = np.full(20, np.nan)
        try:
            constraint = h.qfrc_constraint.copy()
        except AttributeError:
            self.availability["qfrc_constraint"] = False
            constraint = np.full(self.model.nv, np.nan)
        # Counts at capacity are suspect, not evidence that truncation is absent.
        d = self.env.sim.wp_data
        nacon = int(d.nacon.numpy().reshape(-1)[0])
        ncon_capacity = int(d.contact.geom.shape[0])
        njmax = int(d.efc.force.shape[-1])
        try:
            overflow = int(np.bitwise_or.reduce(d.overflow.numpy().reshape(-1)))
        except (AttributeError, ValueError):
            overflow = None
        try:
            ncollision = int(d.ncollision.numpy().reshape(-1)[0])
        except (AttributeError, ValueError):
            ncollision = None
        counts_below_capacity = nacon < ncon_capacity and h.nefc < njmax
        cap_status = "UNKNOWN" if overflow is None else "OK" if overflow == 0 and counts_below_capacity else "OVERFLOW"
        cap = {"nacon": nacon, "contact_capacity": ncon_capacity,
               "nefc": int(h.nefc), "constraint_capacity": njmax,
               "overflow_bitmask": overflow, "ncollision": ncollision, "status": cap_status}
        self.capacities.append(cap)
        state = {"physics_time_s": float(h.time), "q_rad": h.qpos[self.qa].copy(),
                 "qd_rad_s": h.qvel[self.va].copy(), "actuator_torque_Nm": h.actuator_force[self.aids].copy(),
                 "object_pose_w": h.qpos[self.oq:self.oq + 7].copy(),
                 "object_twist_native": h.qvel[self.ov:self.ov + 6].copy(),
                 "qacc_rad_s2": qacc, "qfrc_constraint": constraint, "joint_limit_active": active,
                 "penetration_hand_object_m": c["penetration_by_category_m"]["hand_object"],
                 "penetration_hand_self_m": c["penetration_by_category_m"]["hand_self"],
                 "penetration_other_m": c["penetration_by_category_m"]["other"],
                 "hand_object_active_count": int(c["active"].sum()),
                 "external_force_max_abs": float(max(np.abs(h.xfrc_applied).max(), np.abs(h.qfrc_applied).max())),
                 "capacity_overflow_bitmask": overflow if overflow is not None else -1,
                 "derived_is_post_forward": bool(post_forward)}
        return state, h.ctrl[self.aids].copy()

    def arrays(self, states, controls):
        np = self.np
        out = {k: np.asarray([s[k] for s in states]) for k in states[0]}
        out.update(self.static)
        out["elapsed_s"] = np.arange(len(states)) * .005
        out["u_prev_rad"] = controls[0].copy()
        out["u_readback_for_step"] = np.asarray(controls[1:])
        out["u_proposed_for_step"] = out["u_readback_for_step"].copy()
        out["u_bounded_for_step"] = out["u_readback_for_step"].copy()
        out["control_tick_id_for_step"] = np.arange(len(states) - 1) // 4
        return out


def compare_trajectory(actual, expected, *, archived=False):
    """Same tolerances for donor drift and paired baseline replay; no fitting."""
    import numpy as np
    from chord_study.geometry import quat_matrix
    if archived:
        actual = {k: v[::4] if k not in ("u_readback_for_step",) else v[3::4]
                  for k, v in actual.items() if k in ("q_rad", "qd_rad_s", "object_pose_w", "object_twist_native", "u_readback_for_step")}
        pose = actual["object_pose_w"]
        twist = actual["object_twist_native"]
        expected = {"q_rad": expected["joint_pos_rad"], "qd_rad_s": expected["joint_vel_rad_s"],
                    "object_pose_w": expected["object_pose_w"],
                    "object_twist_native": np.concatenate((expected["object_lin_vel_w_m_s"],
                        np.einsum("nji,nj->ni", quat_matrix(expected["object_pose_w"][:, 3:]), expected["object_ang_vel_w_rad_s"])), -1),
                    "u_readback_for_step": expected["joint_target_rad"][1:]}
    fields = ("q_rad", "qd_rad_s", "object_pose_w", "object_twist_native", "u_readback_for_step")
    if any(actual[k].shape != expected[k].shape for k in fields):
        return {"passed": False, "reason": "shape mismatch", "shapes": {
            k: [list(actual[k].shape), list(expected[k].shape)] for k in fields}}
    errors = {k: float(np.abs(actual[k] - expected[k]).max()) for k in fields if k != "object_pose_w"}
    errors["object_position_m"] = float(np.abs(actual["object_pose_w"][:, :3] - expected["object_pose_w"][:, :3]).max())
    errors["object_rotation_matrix"] = float(np.abs(quat_matrix(actual["object_pose_w"][:, 3:])
                                                        - quat_matrix(expected["object_pose_w"][:, 3:])).max())
    position = max(errors[k] for k in ("q_rad", "object_position_m", "object_rotation_matrix"))
    velocity = max(errors[k] for k in ("qd_rad_s", "object_twist_native"))
    ctrl_exact = bool(np.array_equal(actual["u_readback_for_step"].astype(np.float32),
                                    expected["u_readback_for_step"].astype(np.float32)))
    return {"passed": position <= 2e-5 and velocity <= 2e-4 and ctrl_exact,
            "max_abs_errors": errors, "position_gate": 2e-5, "velocity_gate": 2e-4,
            "ctrl_float32_exact": ctrl_exact, "all_frames_compared": len(actual["q_rad"])}


def capture_donor(source, plan, output, evaluation):
    """Run the frozen evaluator unchanged, with read-only instance wrappers."""
    import numpy as np
    import torch
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp
    import mjlab.envs
    from dex_supermimic.rl.dex_supermimic.pen import PenSimReader, PEN_MEMORY_DIMS
    from dex_supermimic.rl.dex_supermimic.snapshot import WarpTaskSnapshot
    donor_dir = output / "donors" / source["method"]
    native_class = mjlab.envs.ManagerBasedRlEnv
    captured = SimpleNamespace(env=None, reader=None, rec=None, snapshots={}, states=[], controls=[],
                               active=False, frame=0, physics_steps=0)

    def factory(*args, **kwargs):
        env = native_class(*args, **kwargs)
        captured.env = env
        captured.original_step = env.step
        captured.original_sim_step = env.sim.step
        captured.original_sim_forward = env.sim.forward
        captured.original_close = env.close
        # The class and allocation remain native. A subclass would be skipped
        # by WarpTaskSnapshot's documented module whitelist.
        env.close = lambda: None

        def sim_step():
            captured.original_sim_step()
            if captured.active:
                state, ctrl = captured.rec.read(False)
                captured.states.append(state)
                captured.controls.append(ctrl)
                captured.physics_steps += 1

        def env_step(action):
            deadline()
            if captured.rec is None:
                captured.reader = PenSimReader(env)
                captured.rec = Recorder(env, evaluation, plan)
                state, ctrl = captured.rec.read(True)
                captured.states.append(state)
                captured.controls.append(ctrl)
            captured.active = True
            try:
                result = captured.original_step(action)
            finally:
                captured.active = False
            captured.frame += 1
            if captured.physics_steps != captured.frame * 4:
                raise RuntimeError("source did not execute exactly four native substeps")
            # Original env.step performed its normal final forward. Replace
            # the same postintegration state, do not add another forward.
            state, ctrl = captured.rec.read(True)
            captured.states[-1], captured.controls[-1] = state, ctrl
            if captured.frame in plan["snapshot_frames"]:
                reader = captured.reader
                reader.refresh_frame()
                signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
                size = mujoco.mj_stateSize(env.sim.mj_model, signature)
                integration = wp.zeros((1, size), dtype=wp.float32, device=env.device)
                mjwarp.get_state(env.sim.wp_model, env.sim.wp_data, integration, signature)
                memory = {k: v.detach().clone() for k, v in reader.memory().items()}
                if set(memory) != set(PEN_MEMORY_DIMS):
                    raise RuntimeError("incomplete pen controller memory sidecar")
                snapshot = WarpTaskSnapshot(SimpleNamespace(env=env, reader=reader))
                captured.snapshots[captured.frame] = {"full": snapshot, "integration": integration,
                    "integration_cpu": integration.numpy().copy(), "signature": signature,
                    "memory": memory, "observation": {k: v.detach().clone() for k,v in result[0].items()}}
                sidecar = {"memory_" + k: v.detach().cpu().numpy() for k,v in memory.items()}
                sidecar.update({"observation_" + k: v.detach().cpu().numpy() for k,v in result[0].items()})
                act = env.action_manager.get_term("joint_pos")
                action_metadata = {}
                for name in ("_raw_actions", "_processed_actions", "_targets", "_targets_valid",
                             "_mask_dofs", "_mask_remaining", "_mask_active"):
                    value = getattr(act, name, None)
                    if isinstance(value, torch.Tensor):
                        sidecar["action_term_" + name] = value.detach().cpu().numpy()
                        action_metadata[name] = {"kind": "tensor", "shape": list(value.shape)}
                    elif isinstance(value, (bool, int, float)):
                        action_metadata[name] = value
                    else:
                        action_metadata[name] = "not a tensor/scalar; full object snapshot retained inprocess"
                sidecar["integration"] = captured.snapshots[captured.frame]["integration_cpu"]
                sidecar_path = output / "snapshots" / f"{source['method']}_frame{captured.frame:03d}.npz"
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(sidecar_path, **sidecar)
                write_json(sidecar_path.with_suffix(".json"), {"frame": captured.frame,
                    "integration_signature": signature, "integration_size": size,
                    "pen_memory_dims": PEN_MEMORY_DIMS, "npz_sha256": sha256(sidecar_path),
                    "action_term_sidecar": action_metadata,
                    "same_world_allocation": True, "wp_data_python_identity": id(env.sim.wp_data),
                    "full_inprocess_snapshot": {"arrays": len(snapshot.arrays), "tensors": len(snapshot.tensors),
                                                "python_containers": len(snapshot.entries)},
                    "sampling_semantics": "after original 20ms forward/sense/command/observation update"})
            return result

        env.sim.step = sim_step
        env.step = env_step
        return env

    old_argv = sys.argv
    mjlab.envs.ManagerBasedRlEnv = factory
    try:
        sys.argv = [str(evaluation.__file__), "--reference", plan["reference"], "--checkpoint", source["checkpoint"],
                    "--seed", str(source["evaluation_seed"]), "--device", "cuda:0", "--output", str(donor_dir)]
        evaluation.main()
    except BaseException:
        if captured.env is not None:
            captured.original_close()
        raise
    finally:
        sys.argv = old_argv
        mjlab.envs.ManagerBasedRlEnv = native_class
        captured.active = False
    deadline()
    if captured.env is None or len(captured.states) != 601 or set(captured.snapshots) != {40,80,120}:
        if captured.env is not None:
            captured.original_close()
        raise RuntimeError("source capture incomplete; no branch qualification")
    arrays = captured.rec.arrays(captured.states, captured.controls)
    raw_path = donor_dir / "raw_5ms.npz"
    np.savez_compressed(raw_path, **arrays)
    with np.load(Path(source["archived_evaluation"]) / "rollout.npz", allow_pickle=False) as archive:
        drift = compare_trajectory(arrays, archive, archived=True)
    with np.load(source["context"], allow_pickle=False) as context:
        physical = {}
        for field in PHYSICS_FIELDS:
            actual = getattr(captured.env.sim.wp_model, field).numpy()
            expected = context[field][0]
            actual = actual[0] if actual.ndim == expected.ndim + 1 else actual
            physical[field] = {"exact": bool(np.array_equal(actual, expected)),
                               "max_abs": float(np.abs(actual - expected).max())}
        xi = captured.reader.context().detach().cpu().numpy()[0]
        physical["reader_context"] = {"exact": bool(np.array_equal(xi, context["reader_context"][0])),
                                      "max_abs": float(np.abs(xi-context["reader_context"][0]).max())}
    model_match = sha256(donor_dir / "model.mjb") == source["model_sha256"]
    capacity = captured.rec.capacities
    gates = {"archive_all_151frames": drift, "physical_context": physical, "nominal_model_exact": model_match,
             "capacity_all_ok": all(v["status"] == "OK" for v in capacity),
             "raw_5ms_sha256": sha256(raw_path), "sampling_is_read_only": True,
             "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "actor": "frozen evaluator eager FP32 mean; no training AMP/compile context imported"}
    gates["archive_reproduction_passed"] = drift["passed"]
    gates["physical_setup_passed"] = model_match and gates["capacity_all_ok"] and all(v["exact"] for v in physical.values())
    gates["passed"] = gates["physical_setup_passed"]
    gates["qualification_scope"] = "same-world newly captured donor; archive drift is separate and never relabeled as archive reproduction"
    write_json(donor_dir / "donor_gates.json", gates)
    write_json(donor_dir / "raw_5ms.json", {"joint_names": captured.rec.names,
        "availability": captured.rec.availability, "capacity_status": capacity_summary(capacity),
        "sampling_semantics": sampling_semantics(), "source": source, "gates": gates})
    print(json.dumps({"stage": "donor_complete", "method": source["method"], "passed": gates["passed"],
                      "worker_elapsed_s": time.perf_counter()-STARTED}), flush=True)
    return captured, arrays, gates


def sampling_semantics():
    return {"state": "actual Warp postintegration qpos/qvel, no native solve",
        "object_twist_native": "linear world xyz; angular local-body xyz from freejoint qvel",
        "force_contact_qacc_constraint": "preintegration solver at 5ms samples; derived_is_post_forward marks original 20ms final-forward boundaries",
        "t0_force": "snapshot cached derived values; not a newly executed branch response",
        "no_added_forward": "step graph x4 then original forward graph once per20ms",
        "targets": "step k drives state k->k+1; original target[frame+1] belongs next interval",
        "joint_limit_active": "actual efc joint-limit row present; causal limit braking contribution UNKNOWN",
        "actuator_torque_Nm": "verified joint transmission/unit gear/fixed gain/affine position PD; scalar actuator_force is joint Nm",
        "external_force": "max abs of actual xfrc_applied and qfrc_applied; no injected disturbance"}


def capacity_summary(capacities):
    statuses = {c["status"] for c in capacities}
    status = "OVERFLOW" if "OVERFLOW" in statuses else "UNKNOWN" if not capacities or "UNKNOWN" in statuses else "OK"
    masks = [c["overflow_bitmask"] for c in capacities if c.get("overflow_bitmask") is not None]
    return {"status": status,
            "samples": len(capacities), "max_nacon": max((c["nacon"] for c in capacities), default=None),
            "max_nefc": max((c["nefc"] for c in capacities), default=None),
            "overflow_bitmask_or": __import__("functools").reduce(lambda a,b: a|b, masks, 0) if len(masks)==len(capacities) and masks else None,
            "max_ncollision": max((c["ncollision"] for c in capacities if c.get("ncollision") is not None), default=None),
            "contact_capacity": min((c["contact_capacity"] for c in capacities), default=None),
            "constraint_capacity": min((c["constraint_capacity"] for c in capacities), default=None)}


def run_branches(captured, donor, source, plan, output, manifest):
    import numpy as np
    import mujoco_warp as mjwarp
    import warp as wp
    env, rec = captured.env, captured.rec
    baselines = {}
    for case in cases_for(plan, source["method"]):
        deadline()
        frame = case["frame"]
        saved = captured.snapshots[frame]
        # This is the sole prebranch restore transaction. No native qpos/qvel,
        # object, warmstart, manager, or integration writes occur in the loop.
        saved["full"].restore()
        restored = wp.zeros_like(saved["integration"])
        mjwarp.get_state(env.sim.wp_model, env.sim.wp_data, restored, saved["signature"])
        full_error = float(np.abs(restored.numpy() - saved["integration_cpu"]).max())
        if full_error != 0:
            raise RuntimeError(f"full snapshot restore mismatch: {case['branch_id']} {full_error}")
        mjwarp.set_state(env.sim.wp_model, env.sim.wp_data, saved["integration"], saved["signature"])
        mjwarp.get_state(env.sim.wp_model, env.sim.wp_data, restored, saved["signature"])
        restore_error = float(np.abs(restored.numpy() - saved["integration_cpu"]).max())
        if restore_error != 0:
            raise RuntimeError(f"integration ABI restore mismatch: {case['branch_id']} {restore_error}")
        memory_errors = {k: float((v-saved["memory"][k]).abs().max().item())
                         for k,v in captured.reader.memory().items()}
        if max(memory_errors.values()) != 0:
            raise RuntimeError("PEN_MEMORY_DIMS restore mismatch")
        start = frame * 4
        future_controls = donor["u_readback_for_step"][start:start + 40].copy()
        u_prev = donor["u_readback_for_step"][start - 1].copy()
        proposed = future_controls.copy()
        joint_index, sign = None, None
        if case["kind"] == "pulse":
            joint_index = rec.names.index(case["pulse_joint"])
            v0 = float(donor["qd_rad_s"][start, joint_index])
            sign = pulse_sign(v0)
            proposed[:, joint_index] = u_prev[joint_index] - sign * plan["pulse_amplitude_rad"]
        bounded = np.clip(proposed, rec.lower, rec.upper).astype(np.float32)
        cap_start = len(rec.capacities)
        state, ctrl = rec.read(True)
        states, controls = [state], [ctrl]
        if not np.array_equal(ctrl.astype(np.float32), u_prev.astype(np.float32)):
            raise RuntimeError("snapshot previous ctrl disagrees with donor target[k]")
        tick_targets = []
        for step in range(40):
            deadline()
            if step % 4 == 0:
                # Input is absolute u; no process_actions, EMA, raw-action or
                # state edits. Other joints retain source target[k+1].
                all_ctrl = rec.host.ctrl.astype(np.float32, copy=True)[None]
                all_ctrl[0, rec.aids] = bounded[step]
                wp.copy(env.sim.wp_data.ctrl, wp.array(all_ctrl, dtype=wp.float32, device=env.device))
                tick_targets.append(bounded[step].copy())
            captured.original_sim_step()
            boundary = (step + 1) % 4 == 0
            if boundary:
                captured.original_sim_forward()
            state, ctrl = rec.read(boundary)
            states.append(state)
            controls.append(ctrl)
        arrays = rec.arrays(states, controls)
        arrays["u_prev_rad"] = u_prev
        arrays["u_proposed_for_step"] = proposed
        arrays["u_bounded_for_step"] = bounded
        arrays["target_rate_rad_s"] = np.diff(np.vstack((u_prev, tick_targets)), axis=0) / .02
        expected = {k: donor[k][start:start + 41] for k in
                    ("q_rad", "qd_rad_s", "object_pose_w", "object_twist_native")}
        expected["u_readback_for_step"] = future_controls
        comparison = compare_trajectory(arrays, expected) if case["kind"] == "baseline" else None
        repeat_comparisons = []
        if case["kind"] == "baseline":
            for previous in baselines.get(frame, []):
                repeat_comparisons.append(compare_trajectory(arrays, previous))
            baselines.setdefault(frame, []).append(arrays)
        capacities = rec.capacities[cap_start:]
        capacity = capacity_summary(capacities)
        exact_controls = bool(np.array_equal(arrays["u_readback_for_step"].astype(np.float32), bounded))
        external_zero = bool(np.all(arrays["external_force_max_abs"] == 0))
        branch_ok = capacity["status"] == "OK" and exact_controls and external_zero
        if comparison is not None:
            branch_ok &= comparison["passed"] and all(c["passed"] for c in repeat_comparisons)
        elif len(baselines.get(frame, [])) != 3:
            branch_ok = False
        path = output / "branches" / (case["branch_id"] + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        metadata = {**case, "method": source["method"], "training_seed": source["training_seed"],
            "capturetime": frame * .02, "capture_frame": frame, "evaluation_seed": 0,
            "pulse_sign": -sign if sign is not None else None,
            "pulse_amplitude_rad": .1 if sign is not None else None,
            "declared_limit_side": "upper", "pulse_inward_wrt_declared_limit": bool(sign > 0) if sign is not None else None,
            "initial_velocity_rad_s": float(donor["qd_rad_s"][start, joint_index]) if joint_index is not None else None,
            "upper_braking_fit_eligible_initial_velocity": bool(donor["qd_rad_s"][start, joint_index] > .02) if joint_index is not None else None,
            "integration_restore_max_abs_error": restore_error, "full_snapshot_integration_restore_max_abs_error": full_error,
            "pen_memory_restore_max_abs_errors": memory_errors,
            "availability": rec.availability.copy(), "capacity_status": capacity,
            "complete": len(states) == 41, "engineering_gates_passed": bool(branch_ok),
            "actual_ctrl_matches_bounded_float32_exact": exact_controls,
            "external_forces_all_zero": external_zero, "joint_names": rec.names,
            "source_future_slice": {"npz": f"donors/{source['method']}/raw_5ms.npz",
                "start_state": start, "stop_state_exclusive": start+41,
                "start_step": start, "stop_step_exclusive": start+40},
            "baseline_vs_donor": comparison, "baseline_vs_previous_repeats": repeat_comparisons,
            "source_model_context_hashes": {k: source[k] for k in ("checkpoint_sha256", "model_sha256", "context_sha256")},
            "runtime_sha256": sha256(__file__), "sampling_semantics": sampling_semantics(),
            "inside_branch_physical_state_writes": 0, "inside_branch_state_writes": 0, "object_free": True,
            "target_clip_count": int(np.count_nonzero(proposed.astype(np.float32) != bounded)),
            "max_abs_target_rate_rad_s": float(np.abs(arrays["target_rate_rad_s"]).max()),
            "target_rate_limit_pass": bool(np.max(np.abs(arrays["target_rate_rad_s"])) <= 15.0),
            "npz_sha256": sha256(path), "m0": None, "a_brake": None,
            "qualification": "local strong-pulse response only; no C1 weaker-law calibration or safety guarantee"}
        write_json(path.with_suffix(".json"), metadata)
        manifest["branches"].append({"npz": path.relative_to(output).as_posix(),
                                      "metadata": path.with_suffix(".json").relative_to(output).as_posix()})
        manifest["completed_branches"] = len(manifest["branches"])
        write_json(output / "branches_manifest.json", manifest)
        print(json.dumps({"stage": "branch_complete", "branch_id": case["branch_id"], "passed": bool(branch_ok),
                          "completed_branches": len(manifest["branches"]), "worker_elapsed_s": time.perf_counter()-STARTED}), flush=True)
        if not branch_ok:
            raise RuntimeError(f"fixed restore/replay/capacity gate failed: {case['branch_id']}; no retries or extra cases")


def worker(plan_path, output):
    global SOFT_SECONDS
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runtime = {"status": "validating", "pid": os.getpid(), "parent_pid": os.getppid(),
               "started_utc": STARTED_UTC, "plan_path": str(plan_path), "plan_sha256": sha256(plan_path),
               "runtime_sha256": sha256(__file__), "soft_cap_seconds": 180,
               "training_updates": 0, "m0": None, "a_brake": None}
    write_json(output / "runtime.json", runtime)
    captured = None
    manifest = {"schema": "C1_braking_branches:1", "status": "running", "branches": [],
                "plan_sha256": runtime["plan_sha256"], "runtime_sha256": runtime["runtime_sha256"],
                "m0": None, "a_brake": None, "parameter_enablement": False}
    write_json(output / "branches_manifest.json", manifest)
    try:
        plan, check = validate(plan_path)
        SOFT_SECONDS = float(plan["worker_soft_cap_seconds"])
        write_json(output / "preflight.json", check)
        write_json(output / "exact_plan.json", plan)
        import shutil
        shutil.copyfile(__file__, output / "braking_microprobe_source.py")
        shutil.copyfile(plan["snapshot_contract"], output / "snapshot_contract.json")
        runtime["status"] = "running"
        write_json(output / "runtime.json", runtime)
        sys.path.insert(0, str(STUDY / "scripts"))
        sys.path.insert(0, str(STUDY / "src"))
        evaluation = import_file(plan["source_pins"]["evaluator"]["path"], "frozen_microprobe_evaluator")
        for source in plan["sources"]:
            deadline()
            captured, donor, gates = capture_donor(source, plan, output, evaluation)
            manifest.setdefault("donors", []).append({"method": source["method"],
                "gates": f"donors/{source['method']}/donor_gates.json",
                "archive_reproduction_passed": gates["archive_reproduction_passed"],
                "physical_setup_passed": gates["physical_setup_passed"]})
            if not gates["physical_setup_passed"]:
                raise RuntimeError(f"{source['method']} donor physical/context/capacity gate failed; no branches")
            try:
                run_branches(captured, donor, source, plan, output, manifest)
            finally:
                captured.original_close()
                captured.snapshots.clear()
                captured = None
        if len(manifest["branches"]) != 54:
            raise RuntimeError("incomplete fixed54 branch plan")
        manifest["status"] = "passed_engineering_diagnostic_only"
        runtime["status"] = "passed_engineering_diagnostic_only"
    except BaseException as error:
        runtime["status"] = "soft_cap_exceeded" if isinstance(error, TimeoutError) else "failed"
        runtime["error"] = repr(error)
        manifest["status"] = runtime["status"]
        manifest["error"] = repr(error)
        raise
    finally:
        if captured is not None and captured.env is not None:
            captured.original_close()
        runtime["finished_utc"] = datetime.now(timezone.utc).isoformat()
        runtime["wall_seconds_whole_worker"] = time.perf_counter() - STARTED
        runtime["completed_branches"] = len(manifest["branches"])
        write_json(output / "runtime.json", runtime)
        write_json(output / "branches_manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    if args.validate_only:
        _, result = validate(args.plan.resolve(strict=True))
        print(json.dumps(result, indent=2))
        return
    if args.output is None:
        p.error("--output is required for actual bounded diagnostic execution")
    worker(args.plan.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
