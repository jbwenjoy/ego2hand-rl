"""Same-backend Warp evaluation of a fixed first-frame pen reference.

No checkpoint means A0 zero residual. A checkpoint evaluates its deterministic
actor (A1). Both cases use the same play=True scene, reference, initial state,
50 Hz controller, actual Warp physics, and strict no-reset/no-assist protocol.
Tracking failures are latched while physics continues to the fixed 3s endpoint,
so A0/A1 metrics use the same time interval. Only instability/hard errors censor
a rollout. It never resets a world or uses RSI.

GPU execution must be scheduled after other bring-up jobs finish. This script
uses mujoco_warp.get_data_into to copy measured Warp contacts/forces to CPU;
it never runs a native forward solve on a Warp measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from chord_study.evaluation import evaluate
from chord_study.reference import Reference
from chord_study.geometry import quat_matrix
from chord_study.contacts import sample_basis,wrench_support
from controlled_pen_rollout import capture_contacts, git_version, sha256, write_bundle


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clips-root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite prior artifacts: {output}")
    reference_path = args.reference.resolve(strict=True)
    reference = Reference.load(reference_path)
    inspection = reference.validate()
    if not inspection["fixed_wrist_admissible"]:
        raise ValueError("this evaluator requires an already declared fixed-wrist task reference")
    frames = reference.meta["provenance"].get("source_frames")
    if not frames or frames[0] != 0 or frames[1] != len(reference.arrays["time_s"]):
        raise ValueError("first implementation evaluates only source clips starting at frame zero")
    if not np.allclose(reference.arrays["time_s"], np.arange(frames[1]) * .02, rtol=0, atol=1e-8):
        raise ValueError("a zero-origin 50 Hz reference clock is required")
    source_id = reference.meta["provenance"].get("source_trajectory_id")
    if not isinstance(source_id, str) or len(source_id.split("/")) != 2:
        raise ValueError("reference needs an upstream group/clip source_trajectory_id")
    source_file = Path(reference.meta["provenance"]["source_file"])
    if not source_file.is_absolute():
        source_file = STUDY_ROOT / source_file
    clips_root = args.clips_root.resolve() if args.clips_root else source_file.parent.parent
    actual_source_file = clips_root / source_id.split("/")[0] / source_file.name
    if sha256(actual_source_file) != reference.meta["provenance"]["source_sha256"]:
        raise ValueError("source clip hash mismatch")
    with np.load(actual_source_file, allow_pickle=False) as source:
        if not np.allclose(source["reference_qpos"][:frames[1]], reference.arrays["joint_pos_rad"], atol=1e-7):
            raise ValueError("reference joint trajectory differs from the declared upstream source")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this comparison uses the trained Warp CUDA backend")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    wp.set_device(args.device)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import combine_frame_transforms
    from dex_supermimic.tasks.pen_spin import mdp
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.config.wuji_hand2.hand2_constants import RIGHT_FINGER_JOINT_NAMES
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    import dex_supermimic
    import wuji_description

    output.mkdir(parents=True, exist_ok=True)
    recipe = {"schema": "wuji_pen_spin_recipe:1", "clips": [source_id]}
    recipe_path = output / "eval_recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    cfg = make_env_cfg(num_envs=1, play=True, recipe_path=recipe_path, clips_root=clips_root)
    cfg.seed = args.seed
    cfg.auto_reset = False
    motion_cfg = cfg.commands["motion"]
    motion_cfg.first_frame_hold_prob = 0.
    motion_cfg.trajectory_sampling_mode = "uniform"
    motion_cfg.debug_vis = False
    cfg.actions["joint_pos"].mask_prob = 0.
    # Tracking failure is measured and latched independently below. Removing
    # these manager terminations is necessary because mjlab correctly refuses
    # to step a terminated manual-reset world. The evaluator still marks the
    # episode failed, while continuing all methods to the same physical time.
    cfg.terminations.pop("object_position_error", None)
    cfg.terminations.pop("object_orientation_error", None)
    if set(cfg.events) - {"reset_to_reference_state", "pen_dimensions"}:
        raise ValueError("play scene unexpectedly enables extra DR/disturbance events")
    for group in cfg.observations.values():
        if group.enable_corruption:
            raise ValueError("evaluation observation corruption must be disabled")
    if not np.isclose(cfg.decimation * cfg.sim.mujoco.timestep, .02):
        raise ValueError("invalid simulator/controller timing")
    checkpoint = None
    actor = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location=args.device, weights_only=False)
        if checkpoint["config"].get("task") != "hand2-pen-spin" or checkpoint.get("format_version") != 3:
            raise ValueError("expected a format-3 Hand2 pen SuperMimic checkpoint")
        if checkpoint.get("joint_names") != list(RIGHT_FINGER_JOINT_NAMES) or checkpoint["config"]["control_hz"] != 50:
            raise ValueError("checkpoint joint order/control clock does not match the evaluation contract")
        state = checkpoint["modules"]["actor"]
        mean_key = "norm._mean" if "norm._mean" in state else "norm.mean"
        actor = Actor(state[mean_key].numel()).to(args.device)
        actor.load_state_dict(state)
        actor.eval()

    started = time.perf_counter()
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        obs, _ = env.reset(seed=args.seed)
        motion = env.command_manager.get_term("motion")
        # The reset command callback can advance its integer phase even with
        # dt=0. Pin only command time at t0; never rewrite physical object state.
        motion.time_steps.zero_()
        zeros = torch.zeros((1, 20), device=args.device)
        env.action_manager.process_action(zeros)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()
        obs = env.observation_manager.compute(update_history=False)
        model = env.sim.mj_model
        host = mujoco.MjData(model)
        object_body = model.body("object/cube").id
        palm_body = model.body("robot/right_palm_link").id
        joint_ids = np.asarray([model.joint("robot/" + name).id for name in RIGHT_FINGER_JOINT_NAMES])
        qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
        actuator_ids = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item() for jid in joint_ids])
        object_geoms = set(np.flatnonzero(model.geom_bodyid == object_body).tolist())
        robot_geoms = {i for i in range(model.ngeom) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)",model.geom(i).name)}
        if tuple(env.scene["robot"].joint_names) != tuple(RIGHT_FINGER_JOINT_NAMES):
            raise ValueError("runtime joint ordering differs from the declared contract")
        rows = []
        contact_frames = []
        first_failure = None
        censored = False
        censor_reason = None
        term = trunc = False
        n = len(reference.arrays["time_s"])
        for k in range(n):
            if k:
                try:
                    with torch.no_grad():
                        action = actor(obs["policy"]) if actor is not None else zeros
                        obs, _, terminated, truncated, _ = env.step(action)
                except Exception as error:
                    censored = True
                    censor_reason = {"frame": k, "reason": "hard_step_exception", "detail": repr(error)}
                    if rows:
                        rows[-1]["terminated"] = True
                    break
                term, trunc = bool(terminated.item()), bool(truncated.item())
            # Guard target phase as well as physical timestamps. A dt=0 reset
            # callback or future upstream timing change must not silently shift
            # policy goals relative to the independent evaluator's reference.
            goal_position, goal_quat = combine_frame_transforms(
                motion.wrist_pose_w[:, :3], motion.wrist_pose_w[:, 3:],
                motion.object_pose[:, :3], motion.object_pose[:, 3:])
            if not np.allclose(cpu(goal_position)[0], reference.arrays["object_pose_w"][k, :3], atol=1e-5):
                raise ValueError(f"command object target/time misalignment at frame {k}")
            goal_dot = abs(np.dot(cpu(goal_quat)[0], reference.arrays["object_pose_w"][k, 3:]))
            if goal_dot < 1 - 1e-5:
                raise ValueError(f"command object orientation misalignment at frame {k}")
            wp.synchronize_device(args.device)
            # Copies actual Warp state, contacts and efc forces, no native solve.
            mjwarp.get_data_into(host, model, env.sim.wp_data, world_id=0)
            if not np.isfinite(host.qpos).all() or not np.isfinite(host.qvel).all():
                if rows:
                    rows[-1]["terminated"] = True
                censored = True
                censor_reason = {"frame": k, "reason": "non_finite_physics"}
                first_failure = first_failure or censor_reason
                break
            contact = capture_contacts(model, host, object_body, object_geoms, robot_geoms)
            velocity_w = np.zeros(6)
            # Read-only kinematic query: rot:lin, world orientation. This does
            # not run a second physics solve on copied Warp measurements.
            mujoco.mj_objectVelocity(model, host, mujoco.mjtObj.mjOBJ_BODY, object_body, velocity_w, 0)
            pose = np.r_[host.xpos[object_body], host.xquat[object_body]]
            wrist = np.r_[host.xpos[palm_body], host.xquat[palm_body]]
            if k == 0:
                if not np.allclose(pose[:3], reference.arrays["object_pose_w"][0, :3], atol=1e-5):
                    raise ValueError("Warp initial object position does not match the frozen reference")
                if not np.allclose(host.qpos[qadr], reference.arrays["joint_pos_rad"][0], atol=1e-5):
                    raise ValueError("Warp initial hand does not match the frozen reference")
            position_error = np.linalg.norm(pose[:3] - reference.arrays["object_pose_w"][k, :3])
            cosine = np.clip(np.dot(quat_matrix(pose[3:])[:, 0], quat_matrix(reference.arrays["object_pose_w"][k, 3:])[:, 0]), -1, 1)
            if first_failure is None and (position_error > .05 or np.arccos(cosine) >= np.pi / 3):
                first_failure = {"frame": k, "time_s": float(host.time),
                                 "reason": "position_5cm" if position_error > .05 else "shaft_angle_60deg"}
            natural_end = bool(env.termination_manager.get_term("reference_end").item())
            if term or (trunc and not natural_end):
                censored = True
                censor_reason = {"frame": k, "time_s": float(host.time), "reason": "physical_instability" if term else "unexpected_timeout",
                                 "terms": {name: bool(env.termination_manager.get_term(name).item()) for name in cfg.terminations}}
                first_failure = first_failure or censor_reason
            # Warp stores time in float32 and repeatedly adds physics_dt. Keep
            # that raw clock as evidence; canonical sample times are the fixed
            # integer control ticks, with a strict 50us drift check for this
            # short evaluation. No state or target trajectory is resampled.
            control_time = k * .02
            if abs(float(host.time) - control_time) > 5e-5:
                raise ValueError(f"physics clock exceeds the 50us numerical accumulation budget at frame {k}")
            rows.append({"time_s": control_time, "physics_time_s": float(host.time),
                         "object_pose_w": pose.copy(), "wrist_pose_w": wrist.copy(),
                         "joint_pos_rad": host.qpos[qadr].copy(), "joint_vel_rad_s": host.qvel[vadr].copy(),
                         "joint_target_rad": host.ctrl[actuator_ids].copy(), "actuator_torque_Nm": host.actuator_force[actuator_ids].copy(),
                         "object_ang_vel_w_rad_s": velocity_w[:3].copy(), "object_lin_vel_w_m_s": velocity_w[3:].copy(),
                         "object_assist_wrench_w": host.xfrc_applied[object_body].copy(), "object_pose_overwritten": False,
                         "terminated": first_failure is not None, "penetration_m": contact["penetration_m"],
                         "hand_object_active_contact_count": int(contact["active"].sum()),
                         "hand_object_penetration_m":contact["penetration_by_category_m"]["hand_object"],
                         "hand_self_penetration_m":contact["penetration_by_category_m"]["hand_self"],
                         "other_penetration_m":contact["penetration_by_category_m"]["other"],
                         "object_environment_contact_count": contact["object_environment_contacts"]})
            contact_frames.append(contact)
            if censored or natural_end:
                break
        rollout = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
        rollout["joint_lower_rad"] = model.jnt_range[joint_ids, 0].copy()
        rollout["joint_upper_rad"] = model.jnt_range[joint_ids, 1].copy()
        rollout["actuator_force_limit_Nm"] = np.max(np.abs(model.actuator_forcerange[actuator_ids]),axis=-1)
        max_contacts=max((len(c["active"]) for c in contact_frames),default=0)
        for field,tail,dtype in (("pos_o_m",(3,),float),("normal_o",(3,),float),("force_w_N",(3,),float),
                                 ("geom_pair",(2,),np.int32),("active",(),bool)):
            padded=np.zeros((len(rows),max_contacts)+tail,dtype=dtype)
            if field=="geom_pair":padded.fill(-1)
            for k,c in enumerate(contact_frames):padded[k,:len(c[field])]=c[field]
            rollout["live_contact_"+field]=padded
        com=model.body_ipos[object_body]
        radius=max(float(np.linalg.norm(model.geom_pos[i]-com)+model.geom_rbound[i]) for i in object_geoms)
        basis=sample_basis(512,0,dtype=torch.float64)
        rollout["live_contact_pos_com_o_m"]=rollout["live_contact_pos_o_m"]-com
        rollout["live_support512"]=wrench_support(torch.from_numpy(rollout["live_contact_pos_com_o_m"]),
                                                 torch.from_numpy(rollout["live_contact_normal_o"]),
                                                 torch.from_numpy(rollout["live_contact_active"]),basis,radius).numpy()
        rollout["support_basis"]=basis.numpy()
        mujoco.mj_saveModel(model, str(output / "model.mjb"), None)
        meta = {"schema": "supermimic_chord_rollout:1", "physics_engine": "mujoco_warp", "object_free_dynamics": True,
                "reference_source_id": reference.meta["source_id"], "world_frame": reference.meta["world_frame"],
                "joint_names": list(RIGHT_FINGER_JOINT_NAMES), "units": "SI", "quaternion_order": "wxyz", "seed": args.seed,
                "reference_bundle_sha256": sha256(reference_path), "source_clip_sha256": sha256(actual_source_file),
                "model_sha256": sha256(output / "model.mjb"), "script_sha256": sha256(__file__),
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
                "checkpoint_iteration": checkpoint.get("iteration") if checkpoint else None,
                "checkpoint_training_config": checkpoint.get("config") if checkpoint else None,
                "checkpoint_contact_head_semantics": "upstream 15-dimensional finger self-collision, not hand-object CHORD contact",
                "policy": "A1 deterministic actor" if actor is not None else "A0 zero residual",
                "resets_during_rollout": 0, "object_state_writes_after_initialization": 0,
                "initialization": "env initial reset only, first source frame; no hold, no RSI, zero command phase at t0",
                "domain_randomization": False, "observation_corruption": False, "external_disturbances": False,
                "object_pose_measurement": "actual Warp get_data_into output after env post-step forward",
                "contact_measurement": "actual Warp contacts and efc forces transferred with get_data_into; no native forward solve",
                "wrench_support_diagnostic":{"basis_count":512,"basis_seed":0,"mu":.1,"cone_edges":8,"radius_m":radius,
                                             "normal":"geometric force-on-object inward contact normal; solver-active mask", "moment_origin":"object COM"},
                "hand_contact_scope": "palm and finger collision geoms only; mount/fixture support counts as environment, never hand",
                "time_alignment": "initial physical/source frame0 at t0, subsequent samples at 50 Hz; action uses current command frame",
                "clock_semantics": "time_s is integer control tick times 0.02; physics_time_s preserves raw Warp FP32 accumulation without resampling",
                "physics_clock_max_abs_drift_s": float(abs(rollout["physics_time_s"] - rollout["time_s"]).max()),
                "physics_clock_drift_budget_s": 5e-5,
                "physics_clock_initial_s": float(rollout["physics_time_s"][0]),
                "recorded_control_intervals": len(rows) - 1,
                "recorded_frames": len(rows),
                "termination": "tracking failure latched, continue to common endpoint; instability/hard error censors; natural reference_end is not failure",
                "first_failure": first_failure, "rollout_censored": censored, "censor_reason": censor_reason,
                "control_hz": 50, "physics_dt_s": float(env.physics_dt), "decimation": cfg.decimation,
                "mujoco_version": mujoco.__version__, "mujoco_warp_version": mjwarp.__version__,
                "wuji_resource_revision": getattr(wuji_description, "REVISION", None),
                "dex_commit": git_version(Path(dex_supermimic.__file__).resolve().parents[2]),
                "wall_seconds": time.perf_counter() - started,
                "limitations": ["short checkpoint does not establish convergence", "axis metric ignores axial roll by task definition", "reference contact labels remain unknown"]}
        shutil.copyfile(reference_path, output / "reference.npz")
        shutil.copyfile(reference_path.with_suffix(".json"), output / "reference.json")
        write_bundle(output / "rollout.npz", rollout, meta)
        result = evaluate(reference, rollout, meta)
        result["first_failure"] = first_failure
        result["rollout_censored"] = censored
        result["censor_reason"] = censor_reason
        result["wall_seconds"] = time.perf_counter() - started
        (output / "evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
