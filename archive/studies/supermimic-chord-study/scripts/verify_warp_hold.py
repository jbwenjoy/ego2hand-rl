"""Transfer a selected source-FD initial grasp to an unassisted Warp 0.3s hold.

Only command phase is frozen after t0; no physical state is overwritten. The
hand receives zero residuals around its first-frame q target. Actual Warp
contact snapshots cover each physics substep, with pre-integration contact
timing documented separately from integrated qpos timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))
from chord_study.geometry import quat_matrix
from chord_study.reference import Reference
from controlled_pen_rollout import sha256
from select_initial_grasp import contact_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    reference = Reference.load(a.reference)
    reference.validate()
    provenance = reference.meta["provenance"]
    if provenance.get("screening_velocity_group") != "source_fd":
        raise ValueError("this transfer probe requires the source-FD candidate group")
    protocol = json.loads(a.protocol.read_text(encoding="utf-8"))
    if protocol["schema"] != "supermimic_chord_initial_grasp_protocol:2":
        raise ValueError("freeze and supply the 5ms v2 screening protocol")
    source = Path(provenance["source_file"])
    if sha256(source) != provenance["source_sha256"]:
        raise ValueError("derived source clip integrity mismatch")
    if not a.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Warp CUDA evaluation required")
    out.mkdir(parents=True, exist_ok=True)
    recipe = out / "hold_recipe.json"
    recipe.write_text(json.dumps({"schema": "wuji_pen_spin_recipe:1", "clips": [provenance["source_trajectory_id"]]}, indent=2), encoding="utf-8")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    wp.set_device(a.device)
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.config.wuji_hand2.hand2_constants import RIGHT_FINGER_JOINT_NAMES
    import wuji_description
    cfg = make_env_cfg(num_envs=1, play=True, recipe_path=recipe, clips_root=source.parent.parent)
    cfg.seed = a.seed
    cfg.auto_reset = False
    cfg.commands["motion"].first_frame_hold_prob = 0.
    cfg.commands["motion"].trajectory_sampling_mode = "uniform"
    cfg.commands["motion"].debug_vis = False
    cfg.actions["joint_pos"].mask_prob = 0.
    cfg.terminations = {"unstable": cfg.terminations["unstable"]}
    if set(cfg.events) - {"reset_to_reference_state", "pen_dimensions"}:
        raise ValueError("unexpected DR or external-force event")
    for group in cfg.observations.values():
        if group.enable_corruption:
            raise ValueError("unexpected observation corruption")
    start_wall = time.perf_counter()
    env = ManagerBasedRlEnv(cfg, device=a.device)
    original_step = env.sim.step
    try:
        env.reset(seed=a.seed)
        motion = env.command_manager.get_term("motion")
        motion.time_steps.zero_()
        zero = torch.zeros((1, 20), device=a.device)
        env.action_manager.process_action(zero)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()
        model = env.sim.mj_model
        host = mujoco.MjData(model)
        joint_ids = np.asarray([model.joint("robot/" + name).id for name in RIGHT_FINGER_JOINT_NAMES])
        qadr = model.jnt_qposadr[joint_ids]
        lower, upper = model.jnt_range[joint_ids].T
        actuator_ids = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item() for jid in joint_ids])
        object_body = model.body("object/cube").id
        object_geoms = set(np.flatnonzero(model.geom_bodyid == object_body).tolist())
        import re
        hand_geoms = {g for g in range(model.ngeom) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", model.geom(g).name)}
        target = reference.arrays["object_pose_w"][0]
        q0 = reference.arrays["joint_pos_rad"][0]
        samples = []
        failures = set()
        physics_steps = 0
        censored = False
        censor_reason = None

        def snapshot(contact_time, joint_time, phase):
            wp.synchronize_device(a.device)
            mjwarp.get_data_into(host, model, env.sim.wp_data, world_id=0)
            if not np.isfinite(host.qpos).all() or not np.isfinite(host.qvel).all():
                raise FloatingPointError("nonfinite Warp hold state")
            c = contact_snapshot(model, host, object_geoms, hand_geoms)
            pose = np.r_[host.xpos[object_body], host.xquat[object_body]]
            position = float(np.linalg.norm(pose[:3] - target[:3]))
            cosine = np.clip(np.dot(quat_matrix(pose[3:])[:, 0], quat_matrix(target[3:])[:, 0]), -1, 1)
            angle = float(np.rad2deg(np.arccos(cosine)))
            joint_ok = bool(np.all((host.qpos[qadr] >= lower - protocol["joint_limit_margin_rad"]) &
                                   (host.qpos[qadr] <= upper + protocol["joint_limit_margin_rad"])))
            if not joint_ok:
                failures.add("hold_joint_limits")
            if position > protocol["position_error_limit_m"]:
                failures.add("hold_position_error")
            if angle > protocol["directed_axis_error_limit_deg"]:
                failures.add("hold_axis_error")
            for category, depth in c["penetration_m"].items():
                if depth > protocol["penetration_limits_m"][category]:
                    failures.add(category + "_penetration")
            if pose[2] < target[2] - .02:
                failures.add("drop_height")
            if c["nonhand_object_support_contacts"]:
                failures.add("object_nonhand_support")
            if np.any(np.abs(host.xfrc_applied) > 1e-9) or np.any(np.abs(host.qfrc_applied) > 1e-9):
                failures.add("unexpected_assistance")
            if not np.allclose(host.ctrl[actuator_ids], q0, atol=1e-6):
                failures.add("hand_target_not_fixed_at_initial_q")
            samples.append({"contact_state_time_s": float(contact_time), "joint_state_time_s": float(joint_time),
                            "raw_warp_clock_s": float(host.time), "phase": phase,
                            "object_pose_w": pose.tolist(), "position_error_m": position, "axis_error_deg": angle,
                            "joint_pos_rad": host.qpos[qadr].tolist(), "joint_target_rad": host.ctrl[actuator_ids].tolist(),
                            "actuator_force_Nm": host.actuator_force[actuator_ids].tolist(), "joint_limits_ok": joint_ok,
                            **c})

        snapshot(0., 0., "initial_forward")
        if not np.allclose(samples[0]["object_pose_w"][:3], target[:3], atol=1e-5):
            raise ValueError("Warp initial object differs from selected initial state")
        if samples[0]["active_hand_object_contacts"] < 1:
            failures.add("initial_no_active_hand_object_contact")

        def observed_step():
            nonlocal physics_steps
            original_step()
            physics_steps += 1
            snapshot((physics_steps - 1) * env.physics_dt, physics_steps * env.physics_dt,
                     "physics_step_contact_preintegration_joint_postintegration")

        env.sim.step = observed_step
        controls = int(round(protocol["hold_duration_s"] / env.step_dt))
        for k in range(controls):
            # Only the analytic reference clock is frozen. Object pose/velocity
            # and all robot physical states remain simulator-generated.
            motion.time_steps.zero_()
            try:
                with torch.no_grad():
                    _, _, terminated, truncated, _ = env.step(zero)
                snapshot((k + 1) * env.step_dt, (k + 1) * env.step_dt, "postcontrol_forward")
                if bool(terminated.item()) or bool(truncated.item()):
                    censored = True
                    censor_reason = {"control_interval": k + 1, "reason": "physical_instability"}
                    failures.add("physical_instability")
                    break
            except Exception as error:
                censored = True
                censor_reason = {"control_interval": k + 1, "reason": repr(error)}
                failures.add("hard_exception_or_nonfinite")
                break
        unique_times = np.unique(np.round([s["contact_state_time_s"] for s in samples], 9))
        coverage = len(unique_times) == int(round(protocol["hold_duration_s"] / env.physics_dt)) + 1
        if not coverage:
            failures.add("incomplete_physics_contact_time_coverage")
        result = {"schema": "supermimic_chord_warp_hold:1", "scope": "training-init transfer screening only, not following-motion or task success",
                  "protocol_sha256": sha256(a.protocol), "reference_sha256": sha256(a.reference), "script_sha256": sha256(__file__),
                  "source_candidate_id": provenance["selected_candidate_id"], "seed": a.seed,
                  "physics_engine": "mujoco_warp", "mujoco_version": mujoco.__version__, "mujoco_warp_version": mjwarp.__version__,
                  "wuji_resource_revision": getattr(wuji_description, "REVISION", None),
                  "physics_dt_s": float(env.physics_dt), "control_dt_s": float(env.step_dt),
                  "physics_steps": physics_steps, "control_intervals": physics_steps // cfg.decimation,
                  "unique_contact_time_count": len(unique_times), "contact_time_coverage_complete": coverage,
                  "contact_phase": "mj_step contacts describe preintegration state; after each control env.forward adds integrated-state endpoint; union covers every 5ms state",
                  "initialization": "one env reset to source_fd frame0; only reference phase frozen thereafter; zero residuals and no assistance/reset",
                  "joint_names": list(RIGHT_FINGER_JOINT_NAMES), "joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist(),
                  "actuator_force_limit_Nm": model.actuator_forcerange[actuator_ids].tolist(),
                  "max_penetration_m": {category: max(s["penetration_m"][category] for s in samples) for category in protocol["penetration_limits_m"]},
                  "max_position_error_m": max(s["position_error_m"] for s in samples),
                  "max_axis_error_deg": max(s["axis_error_deg"] for s in samples),
                  "failure_reasons": sorted(failures), "passed": not failures and not censored,
                  "censored": censored, "censor_reason": censor_reason, "wall_seconds": time.perf_counter() - start_wall}
        (out / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.sim.step = original_step
        env.close()


if __name__ == "__main__":
    main()
