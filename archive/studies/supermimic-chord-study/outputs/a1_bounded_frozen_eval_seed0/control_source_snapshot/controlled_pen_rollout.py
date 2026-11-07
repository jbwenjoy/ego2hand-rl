"""A0: one bounded native-MuJoCo playback, with no resets or object assistance.

Run with the Dex checkout's isolated Python, for example::

  D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe scripts/controlled_pen_rollout.py \
    --clips-root D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/data/rawdata/clips \
    --output outputs/a0_pen

Uses play=True's scene/assets/control settings, then steps its compiled model
with native MuJoCo on CPU. This is an independent baseline, not Warp parity or
a trained SuperMimic policy. Initial states are written only once at t=0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import mujoco
import numpy as np
import torch

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from chord_study.contacts import sample_basis, wrench_support
from chord_study.evaluation import evaluate
from chord_study.geometry import quat_matrix, rotation_error, world_to_object
from chord_study.reference import Reference


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_version(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_bundle(path, arrays, meta):
    np.savez_compressed(path, **arrays)
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def mul_quat(a, b):
    output = np.empty(4)
    mujoco.mju_mulQuat(output, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return output / np.linalg.norm(output)


def velocity_at_start(poses, dt):
    linear = (poses[1, :3] - poses[0, :3]) / dt
    conjugate = poses[0, 3:].copy()
    conjugate[1:] *= -1
    # Free-joint angular qvel is expressed in the moving object's local frame.
    delta_local = mul_quat(conjugate, poses[1, 3:])
    if delta_local[0] < 0:
        delta_local *= -1
    angular = np.empty(3)
    mujoco.mju_quat2Vel(angular, delta_local, dt)
    return np.r_[linear, angular]


def choose_clip(root, clip_id):
    if clip_id is None:
        index_path = root / "01_single_axis" / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8-sig"))
        entry = index["clips"][0]
    else:
        group, local = clip_id.split("/", 1)
        if group in {"", ".", ".."} or local in {"", ".", ".."} or "/" in local or "\\" in clip_id:
            raise ValueError("clip ID must be group/clip without suffix")
        index_path = root / group / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8-sig"))
        entry = next(item for item in index["clips"] if item["clip_id"] == local)
    clip = index_path.parent / entry["file"]
    if clip.resolve().parent != index_path.parent.resolve() or sha256(clip) != entry["sha256"]:
        raise ValueError("clip path/hash does not match its upstream catalog")
    return clip, index, entry


def capture_contacts(model, data, object_body, object_geoms, robot_geoms):
    """Normal is force-on-object: +frame[0] for object geom2, negated for geom1.

    MuJoCo convention: https://mujoco.readthedocs.io/en/stable/computation/index.html
    Contact force acts from geom1 toward geom2; contact frame rows are world axes.
    Capture geometric/solved-contact evidence, not reference contact targets.
    """
    pose = np.r_[data.xpos[object_body], data.xquat[object_body]]
    points, normals, forces, torques, pairs, active, friction = [], [], [], [], [], [], []
    penetration = 0.0
    category_depth = {"hand_object":0.,"hand_self":0.,"other":0.}
    object_environment_contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        penetration = max(penetration, -float(c.dist), 0.0)
        # MuJoCo 3.11's legacy geom1/geom2 fields are not aliases of geom.
        # get_data_into populates the current geom[2] field, so read that pair.
        g1, g2 = (int(g) for g in c.geom)
        has_object = g1 in object_geoms or g2 in object_geoms
        has_robot = g1 in robot_geoms or g2 in robot_geoms
        category = "hand_object" if has_object and has_robot else "hand_self" if g1 in robot_geoms and g2 in robot_geoms else "other"
        category_depth[category] = max(category_depth[category],max(0.,-float(c.dist)))
        if has_object and not has_robot:
            object_environment_contacts += 1
        if not (has_object and has_robot):
            continue
        sign = 1.0 if g2 in object_geoms else -1.0
        frame = np.asarray(c.frame).reshape(3, 3)
        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force6)
        points.append(np.asarray(c.pos).copy())
        normals.append(sign * frame[0])
        forces.append(sign * frame.T @ force6[:3])
        torques.append(sign * frame.T @ force6[3:])
        pairs.append([g1, g2])
        active.append(bool(c.efc_address >= 0 and force6[0] > 1e-6))
        friction.append(np.asarray(c.friction).copy())
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    points_o, normals_o = world_to_object(points, normals, pose)
    return {"pos_o_m": points_o, "normal_o": normals_o,
            "force_w_N": np.asarray(forces, dtype=float).reshape(-1, 3),
            "torque_at_contact_w_Nm": np.asarray(torques, dtype=float).reshape(-1, 3),
            "geom_pair": np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
            "active": np.asarray(active, dtype=bool),
            "friction": np.asarray(friction, dtype=float).reshape(-1, 5),
            "penetration_m": penetration, "penetration_by_category_m":category_depth,
            "object_environment_contacts": object_environment_contacts}


def verify_contact_sign():
    """Independent gravity/ground fixtures exercise object as either contact geom.

    The fixture is validation evidence, never a manipulation reference or result.
    A plane sorts before a sphere, while a sphere sorts before a box in MuJoCo's
    collision convention. Both cases must give an upward force on the object.
    """
    records = []
    object_sides = set()
    for floor in ("plane", "box"):
        floor_attrs = 'type="plane" size="1 1 .1"' if floor == "plane" else 'type="box" size="1 1 .1" pos="0 0 -.1"'
        xml = f'''<mujoco><option timestep=".002" iterations="100"/>
          <worldbody><geom name="robot/floor" {floor_attrs}/>
          <body name="object/cube" pos="0 0 .06"><freejoint/>
            <geom name="object/ball" type="sphere" size=".05" mass="1"/>
          </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        obj = model.body("object/cube").id
        og = {model.geom("object/ball").id}
        rg = {model.geom("robot/floor").id}
        c = capture_contacts(model, data, obj, og, rg)
        if not len(c["active"]) or not c["active"].any():
            raise AssertionError("contact-sign fixture did not settle onto its support")
        force = c["force_w_N"].sum(0)
        normal = c["normal_o"][c["active"]].mean(0)
        sides = [1 if int(pair[0]) in og else 2 for pair in c["geom_pair"]]
        object_sides.update(sides)
        if normal[2] < .99 or not np.isclose(force[2], 9.81, rtol=.05) or np.linalg.norm(force[:2]) > .01:
            raise AssertionError(f"wrong force-on-object sign in {floor} fixture: normal={normal}, force={force}")
        records.append({"floor_shape": floor, "object_contact_geom_side": sides, "normal_object_axes": normal.tolist(),
                        "force_on_object_world_N": force.tolist(), "expected_force_world_N": [0, 0, 9.81], "passed": True})
    if object_sides != {1, 2}:
        raise AssertionError(f"contact sign fixtures did not exercise both geom orderings: {object_sides}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--clip-id")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--basis-count", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.duration_s <= 10 or args.start_frame < 0:
        raise ValueError("duration must be in (0,10] seconds and start frame nonnegative")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite prior artifacts; choose a new output directory: {output}")
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    contact_sign_check = verify_contact_sign()
    clip, index, entry = choose_clip(args.clips_root.resolve(), args.clip_id)
    dt = float(index["frame_dt"])
    if not np.isclose(dt, .02, rtol=0, atol=1e-8):
        raise ValueError("this controlled baseline requires 50 Hz source data")
    frame_count = int(round(args.duration_s / dt)) + 1
    with np.load(clip, allow_pickle=False) as data:
        source = {key: np.asarray(data[key]).copy() for key in ("reference_qpos", "wrist_pose_w", "object_pose_w", "link_pos_w")}
        if "mediapipe_landmarks_w" in data:
            source["mediapipe_landmarks_w"] = np.asarray(data["mediapipe_landmarks_w"]).copy()
    stop = args.start_frame + frame_count
    if stop > len(source["reference_qpos"]):
        raise ValueError("selected source clip is shorter than the requested full segment")
    source = {key: value[args.start_frame:stop].astype(float) for key, value in source.items()}
    source["wrist_pose_w"][:, 3:] /= np.linalg.norm(source["wrist_pose_w"][:, 3:], axis=-1, keepdims=True)
    source["object_pose_w"][:, 3:] /= np.linalg.norm(source["object_pose_w"][:, 3:], axis=-1, keepdims=True)

    from mjlab.scene import Scene
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.config.wuji_hand2.hand2_constants import (
        RIGHT_FIXED_WRIST_POSE_W, RIGHT_FINGER_JOINT_NAMES,
    )
    import dex_supermimic

    cfg = make_env_cfg(num_envs=1, play=True)
    # Scene compilation does not execute managers, events, randomization, or resets.
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    if not np.isclose(cfg.decimation * model.opt.timestep, dt, rtol=0, atol=1e-8):
        raise ValueError("native physics/control dt does not equal the reference clock")
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    object_body = model.body("object/cube").id
    palm_body = model.body("robot/right_palm_link").id
    object_joint = model.joint("object/cube_freejoint").id
    object_qadr, object_vadr = int(model.jnt_qposadr[object_joint]), int(model.jnt_dofadr[object_joint])
    joint_ids = np.array([model.joint("robot/" + name).id for name in RIGHT_FINGER_JOINT_NAMES])
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    actuator_ids = []
    for jid in joint_ids:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == jid)
        if len(matches) != 1:
            raise ValueError(f"expected one actuator for joint {jid}")
        actuator_ids.append(matches[0])
    actuator_ids = np.asarray(actuator_ids)
    if not np.allclose(data.xpos[palm_body], RIGHT_FIXED_WRIST_POSE_W[:3], atol=1e-6) or rotation_error(data.xquat[palm_body], RIGHT_FIXED_WRIST_POSE_W[3:]) > 1e-6:
        raise ValueError("compiled model wrist does not match the declared fixed mount")

    fixed_wrist = np.tile(np.asarray(RIGHT_FIXED_WRIST_POSE_W), (frame_count, 1))
    source_wr = quat_matrix(source["wrist_pose_w"][:, 3:])
    object_local_pos = np.einsum("tji,tj->ti", source_wr, source["object_pose_w"][:, :3] - source["wrist_pose_w"][:, :3])
    fixed_rot = quat_matrix(fixed_wrist[0, 3:])
    target_object = np.empty((frame_count, 7))
    target_object[:, :3] = object_local_pos @ fixed_rot.T + fixed_wrist[0, :3]
    for i in range(frame_count):
        inverse_wrist = source["wrist_pose_w"][i, 3:].copy()
        inverse_wrist[1:] *= -1
        target_object[i, 3:] = mul_quat(fixed_wrist[i, 3:], mul_quat(inverse_wrist, source["object_pose_w"][i, 3:]))

    # Exactly one object-state initialization, before the recorded rollout.
    data.qpos[qadr] = source["reference_qpos"][0]
    data.qvel[vadr] = (source["reference_qpos"][1] - source["reference_qpos"][0]) / dt
    data.qpos[object_qadr:object_qadr + 7] = target_object[0]
    data.qvel[object_vadr:object_vadr + 6] = velocity_at_start(target_object, dt)
    data.ctrl[actuator_ids] = source["reference_qpos"][0]
    data.xfrc_applied[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    object_geoms = set(np.flatnonzero(model.geom_bodyid == object_body).tolist())
    robot_geoms = {i for i in range(model.ngeom) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)",model.geom(i).name)}
    com_offset = model.body_ipos[object_body].copy()
    radius = max(float(np.linalg.norm(model.geom_pos[i] - com_offset) + model.geom_rbound[i]) for i in object_geoms)
    inertia_rot = quat_matrix(model.body_iquat[object_body])
    inertia = inertia_rot @ np.diag(model.body_inertia[object_body]) @ inertia_rot.T
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact_sign_check.json").write_text(json.dumps(contact_sign_check, indent=2), encoding="utf-8")
    source_id = f"{index['group_id']}/{entry['clip_id']}:frames{args.start_frame}-{stop}"
    reference_meta = {"schema": "supermimic_chord_reference:1", "source_id": source_id,
                      "source_kind": "upstream_kinematic", "split_group": "wuji:v2026.10.22:unknown_original_video_parent",
                      "joint_names": list(RIGHT_FINGER_JOINT_NAMES), "units": "SI", "quaternion_order": "wxyz",
                      "world_frame": "fixed_hand2_mount_world", "contact_normal": "force_on_object_inward",
                      "object_radius_m": radius, "object_mass_kg": float(model.body_mass[object_body]),
                      "object_inertia_kg_m2": inertia.tolist(), "object_symmetry": "none", "orientation_metric": "directed_axis_x",
                      "orientation_task_equivalence": "ignore axial roll per upstream task; marker boxes break exact physical axial symmetry",
                      "object_pose_origin": "compiled object body origin", "object_com_offset_body_m": com_offset.tolist(),
                      "inertia_reference": "about center of mass, expressed in object body axes",
                      "phase_labels": {"0": "unspecified; source catalog supplies no contact-phase annotation"},
                      "provenance": {"source_file": str(clip), "source_sha256": sha256(clip),
                                     "source_catalog": str(clip.parent / "index.json"), "source_trajectory_id": f"{index['group_id']}/{entry['clip_id']}",
                                     "original_video_parent_id": None,
                                     "split_group_policy": "conservative whole-release grouping until original video parent identities are recovered; clip IDs are not independent video IDs",
                                     "source_frames": [args.start_frame, stop], "source_wrist_preserved_array": "source_wrist_pose_w",
                                     "transformation": "remove original wrist motion; preserve wrist-local object and finger reference, rebase onto fixed mount",
                                     "claim": "fixed-wrist task redesign; kinematic reference is not physically validated or a sim teacher",
                                     "contact_targets": "unknown; live rollout contacts are not labels for this reference"}}
    reference_arrays = {"time_s": np.arange(frame_count) * dt, "joint_pos_rad": source["reference_qpos"],
                        "object_pose_w": target_object, "wrist_pose_w": fixed_wrist, "phase": np.zeros(frame_count, dtype=np.int32),
                        "source_wrist_pose_w": source["wrist_pose_w"], "source_object_pose_w": source["object_pose_w"],
                        "source_link_pos_w": source["link_pos_w"]}
    if "mediapipe_landmarks_w" in source:
        reference_arrays["source_mediapipe_landmarks_w"] = source["mediapipe_landmarks_w"]
        reference_meta["provenance"]["human_keypoints_preserved"] = "source_mediapipe_landmarks_w; upstream field, camera/video calibration unavailable"
    reference = Reference(reference_arrays, reference_meta)
    inspection = reference.validate()
    write_bundle(output / "reference.npz", reference_arrays, reference_meta)
    mujoco.mj_saveModel(model, str(output / "model.mjb"), None)

    lists = {name: [] for name in ("time_s", "object_pose_w", "wrist_pose_w", "joint_pos_rad", "joint_vel_rad_s", "joint_target_rad",
                                  "actuator_torque_Nm", "object_assist_wrench_w", "object_pose_overwritten", "terminated", "penetration_m")}
    contact_frames = []
    residual = np.zeros(20)
    alpha = float(cfg.actions["joint_pos"].alpha)
    target = source["reference_qpos"][0].copy()
    first_failure = None
    for k in range(frame_count):
        if k:
            # A0 raw action is zero; keep exact residual EMA semantics explicit.
            residual = alpha * np.clip(.5 * np.zeros(20), -.6, .6) + (1 - alpha) * residual
            target = source["reference_qpos"][k - 1] + residual
            data.ctrl[actuator_ids] = target
            for _ in range(cfg.decimation):
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                if lists["terminated"]:
                    lists["terminated"][-1] = True
                first_failure = first_failure or {"frame": k, "reason": "non_finite_physics"}
                break
            # Refresh kinematics/contact at integrated state; no qpos/velocity writes.
            mujoco.mj_forward(model, data)
        pose = np.r_[data.xpos[object_body], data.xquat[object_body]]
        wrist = np.r_[data.xpos[palm_body], data.xquat[palm_body]]
        contact = capture_contacts(model, data, object_body, object_geoms, robot_geoms)
        contact_frames.append(contact)
        position_error = np.linalg.norm(pose[:3] - target_object[k, :3])
        cosine = np.clip(np.dot(quat_matrix(pose[3:])[:, 0], quat_matrix(target_object[k, 3:])[:, 0]), -1, 1)
        failed = bool(position_error > .05 or np.arccos(cosine) >= np.pi / 3)
        if failed and first_failure is None:
            first_failure = {"frame": k, "time_s": k * dt, "reason": "position_5cm" if position_error > .05 else "shaft_angle_60deg"}
        if not np.isclose(data.time, k * dt, rtol=0, atol=1e-8):
            raise ValueError("native simulator clock drifted from the prescribed trajectory")
        row = {"time_s": float(data.time), "object_pose_w": pose, "wrist_pose_w": wrist,
               "joint_pos_rad": data.qpos[qadr].copy(), "joint_vel_rad_s": data.qvel[vadr].copy(),
               "joint_target_rad": target.copy(), "actuator_torque_Nm": data.actuator_force[actuator_ids].copy(),
               "object_assist_wrench_w": data.xfrc_applied[object_body].copy(), "object_pose_overwritten": False,
               "terminated": first_failure is not None, "penetration_m": contact["penetration_m"]}
        for name, value in row.items():
            lists[name].append(value)
    rollout = {name: np.asarray(values) for name, values in lists.items()}
    rollout["joint_lower_rad"] = model.jnt_range[joint_ids, 0].copy()
    rollout["joint_upper_rad"] = model.jnt_range[joint_ids, 1].copy()
    max_contacts = max((len(c["active"]) for c in contact_frames), default=0)
    contact_count = len(contact_frames)
    for name, tail, dtype in (("pos_o_m", (3,), float), ("normal_o", (3,), float), ("force_w_N", (3,), float),
                              ("torque_at_contact_w_Nm", (3,), float), ("geom_pair", (2,), np.int32),
                              ("active", (), bool), ("friction", (5,), float)):
        padded = np.zeros((contact_count, max_contacts) + tail, dtype=dtype)
        if name == "geom_pair":
            padded.fill(-1)
        for k, c in enumerate(contact_frames):
            padded[k, :len(c[name])] = c[name]
        rollout["live_contact_" + name] = padded
    rollout["live_contact_count"] = np.asarray([len(c["active"]) for c in contact_frames])
    rollout["object_environment_contact_count"] = np.asarray([c["object_environment_contacts"] for c in contact_frames])
    # The logged pose/point coordinates retain the compiled body origin. Wrench
    # moments are separately shifted to COM, even when this symmetric pen has
    # a numerically tiny COM offset. Do not silently conflate these two origins.
    rollout["live_contact_pos_com_o_m"] = rollout["live_contact_pos_o_m"] - com_offset
    basis = sample_basis(args.basis_count, args.seed, dtype=torch.float64)
    rollout["live_support"] = wrench_support(torch.from_numpy(rollout["live_contact_pos_com_o_m"]),
                                              torch.from_numpy(rollout["live_contact_normal_o"]),
                                              torch.from_numpy(rollout["live_contact_active"]), basis, radius, mu=.1).numpy()
    rollout["support_basis"] = basis.numpy()
    dex_root = Path(dex_supermimic.__file__).resolve().parents[2]
    rollout_meta = {"schema": "supermimic_chord_rollout:1", "physics_engine": "mujoco", "object_free_dynamics": True,
                    "reference_source_id": source_id, "world_frame": reference_meta["world_frame"],
                    "joint_names": list(RIGHT_FINGER_JOINT_NAMES), "units": "SI", "quaternion_order": "wxyz",
                    "seed": args.seed, "native_mujoco_version": mujoco.__version__,
                    "dex_commit": git_version(dex_root), "model_file": "model.mjb", "model_sha256": sha256(output / "model.mjb"),
                    "control_hz": 1 / dt, "physics_dt_s": float(model.opt.timestep), "decimation": cfg.decimation,
                    "domain_randomization": False,
                    "controller": {"alpha": alpha, "action_scale_rad": .5, "residual_clip_rad": [-.6, .6],
                                   "encoder_bias_rad": [0.] * 20,
                                   "kp": model.actuator_gainprm[actuator_ids, 0].tolist(),
                                   "kd_bias": model.actuator_biasprm[actuator_ids, 2].tolist(),
                                   "actuator_force_limits_Nm": model.actuator_forcerange[actuator_ids].tolist(),
                                   "absolute_joint_target_limit": "not clamped, matching upstream pen residual controller; evaluator checks actual targets"},
                    "initialization": "only t0 writes hand/object pose and finite-difference source velocities; no settling or reference holding",
                    "object_initialization_count": 1, "resets_during_rollout": 0, "state_correction_during_rollout": False,
                    "policy": "A0 zero residual, EMA retained, encoder bias zero, no action mask or external disturbances",
                    "scene_configuration": "play=True scene, native CPU compile, no manager/event execution; asset pen dimensions already nominal 250mm",
                    "joint_target_timing": "frame k>0 stores actual target held during previous control interval, qref[k-1]",
                    "contact_sampling": "after mj_forward at every control sample, solver-active if normal force >1e-6 N",
                    "contact_normal_convention_source": "https://mujoco.readthedocs.io/en/stable/computation/index.html",
                    "contact_normal_physics_check": "contact_sign_check.json; independent resting sphere on plane/box verifies both geom orderings",
                    "support": {"purpose": "independent live geometry diagnostic, not reference labels or reward", "mu": .1,
                                "basis_seed": args.seed, "basis_count": args.basis_count, "radius_m": radius,
                                "moment_origin": "object_center_of_mass", "axes": "object_body_axes",
                                "object_com_offset_body_m": com_offset.tolist(),
                                "normalization": "unit force; torque divided by radius once"},
                    "first_failure": first_failure, "terminated_semantics": "latched standard training failure: position>5cm or shaft>=60deg; recorded without reset",
                    "wall_seconds": time.perf_counter() - started,
                    "limitations": ["native MuJoCo only, no claim of Warp parity", "axis metric ignores axial roll by task definition; marker-box geometry is not strictly axial-symmetric", "no reference contact targets", "no learned policy or ego input"]}
    write_bundle(output / "rollout_a0.npz", rollout, rollout_meta)
    result = evaluate(reference, rollout, rollout_meta)
    result["reference_inspection"] = inspection
    result["first_failure"] = first_failure
    result["live_contact_max_count"] = max_contacts
    result["live_contact_frames_fraction"] = float((rollout["live_contact_active"].sum(-1) > 0).mean())
    result["wall_seconds"] = time.perf_counter() - started
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
