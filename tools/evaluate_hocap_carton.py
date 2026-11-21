"""Evaluate a Wuji carton run against the original source, phase by phase."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


def errors(source, actual):
    distance = np.linalg.norm(actual[:, :3, 3] - source[:, :3, 3], axis=1) * 1000
    rotations = Rotation.from_matrix(actual[:, :3, :3]).inv() * Rotation.from_matrix(source[:, :3, :3])
    return distance, np.rad2deg(rotations.magnitude())


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def replay_wrist_and_fingers(model, trace):
    objjoint = model.joint("carton_free").id
    wrist_joint = model.joint("wrist_free").id
    if (model.nmocap or model.neq or model.nu != 20 or model.jnt_dofadr[objjoint] < 6
            or model.jnt_type[objjoint] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_dofadr[wrist_joint] != 0):
        raise ValueError("Evaluation requires a free carton, independent wrist and articulated fingers")
    if np.any(model.actuator_trnid[:, 0] == objjoint):
        raise ValueError("Carton joint has direct actuation")
    data = mujoco.MjData(model)
    data.qpos[:] = trace["sim_qpos"][0]
    data.qvel[:] = trace["sim_qvel"][0]
    data.ctrl[:] = trace["finger_ctrl"][0]
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, data)
    # Cache decompressed arrays once. Only the hand receives recorded commands.
    forces = trace["substep_wrist_generalized_wrench"]
    controls = trace["substep_finger_ctrl"]
    endpoints = trace["frame_end_substep"]
    poses, velocities = trace["sim_qpos"], trace["sim_qvel"]
    carton = model.body("carton").id
    body_poses = trace["world_from_carton_sim"]
    if len(endpoints) != len(poses) or endpoints[0] != 0 or np.any(np.diff(endpoints) <= 0):
        raise ValueError("Invalid command-frame boundaries")
    if endpoints[-1] != len(forces) or len(forces) != len(controls):
        raise ValueError("Incomplete command trace")
    err = 0.
    np.testing.assert_allclose(data.xpos[carton], body_poses[0, :3, 3], atol=1e-9, rtol=0)
    for i in range(1, len(endpoints)):
        for j in range(endpoints[i-1], endpoints[i]):
            data.qfrc_applied[:6] = forces[j]
            data.ctrl[:] = controls[j]
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        err = max(err, np.max(np.abs(data.qpos - poses[i])), np.max(np.abs(data.qvel - velocities[i])),
                  np.max(np.abs(data.xpos[carton] - body_poses[i, :3, 3])),
                  np.max(np.abs(data.xmat[carton].reshape(3, 3) - body_poses[i, :3, :3])))
    return float(err)


def evaluate(run, original, protocol, output):
    cfg = json.loads(protocol.read_bytes())
    report = json.loads(run.read_bytes())
    receipt = json.loads(original.read_bytes())
    trace_path = verify_file_pin(report["trace"])
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(report["mjb"])))
    with np.load(verify_file_pin(receipt["source_poses"]), allow_pickle=False) as human:
        source = human["world_from_carton"]
        tips = human["world_right_hand21"][:, [4, 8, 12, 16, 20]]
        frames = human["frames"]
    with np.load(trace_path, allow_pickle=False) as t:
        if len(frames) < 2 or not np.array_equal(np.diff(frames), np.ones(len(frames)-1, dtype=int)):
            raise ValueError("Expected contiguous absolute source frames")
        np.testing.assert_array_equal(t["frame_index"], np.arange(len(frames)))
        np.testing.assert_array_equal(frames, t["source_frame_ids"] if "source_frame_ids" in t else t["frame_index"])
        np.testing.assert_allclose(t["world_from_carton_source"], source, atol=1e-9, rtol=0)
        np.testing.assert_allclose(t["time"], (frames-frames[0]) / cfg["fps_nominal"], atol=1e-9, rtol=0)
        actual = t["world_from_carton_sim"]
        np.testing.assert_allclose(actual[0], source[0], atol=2e-7, rtol=0)
        position, angle = errors(source, actual)
        static_pos, static_rot = errors(source, np.repeat(source[:1], len(source), axis=0))
        # Gravity defines lift. Contact support is measured over the actual control interval.
        gravity = model.opt.gravity
        up = -gravity / np.linalg.norm(gravity)
        lift = (actual[:, :3, 3] - actual[0, :3, 3]) @ up * 1000
        ref_lift = (source[:, :3, 3] - source[0, :3, 3]) @ up * 1000
        weight = model.body_mass[model.body("carton").id] * np.linalg.norm(gravity)
        dt = np.diff(t["time"])
        table_load = t["interval_table_carton_normal_impulse_Ns"][1:] / dt / weight
        hand_load = t["interval_hand_carton_normal_impulse_Ns"][1:] / dt / weight
        airborne = ref_lift[1:] > cfg["source_lift_for_airborne_mm"]
        supported = ((table_load < cfg["maximum_table_mean_force_in_air_bodyweights"]) &
                     (hand_load >= cfg["minimum_hand_mean_force_in_air_bodyweights"]))
        release = slice(-cfg["release_last_n_intervals"], None)
        air_fraction = float(np.mean(supported[airborne])) if airborne.any() else 0.
        final_support = float(table_load[release].min())
        release_contacts = int(t["interval_max_hand_contacts"][release].max())
        metrics = {
            "rerun_max_state_error": replay_wrist_and_fingers(model, t),
            "position_rms_mm": rms(position), "position_peak_mm": float(position.max()),
            "rotation_rms_deg": rms(angle), "rotation_peak_deg": float(angle.max()),
            "worst_position_frame": int(frames[position.argmax()]), "worst_rotation_frame": int(frames[angle.argmax()]),
            "endpoint_error_mm": float(position[-1]), "peak_lift_mm": float(lift.max()),
            "peak_source_lift_mm": float(ref_lift.max()), "source_airborne_intervals": int(airborne.sum()),
            "supported_airborne_intervals": int((supported & airborne).sum()),
            "min_table_bodyweights_final": final_support, "max_hand_contacts_final": release_contacts,
            "max_carton_penetration_mm": float(t["interval_max_carton_penetration_m"].max() * 1000),
            "max_self_penetration_mm": float(t["interval_max_hand_self_penetration_m"].max() * 1000),
            "max_joint_excursion_deg": float(np.rad2deg(t["interval_joint_excursion_rad"].max())),
            "actual_fingertip_rms_mm": rms(np.linalg.norm(t["sim_tips"] - tips, axis=2) * 1000),
            "stationary_position_rms_mm": rms(static_pos), "stationary_rotation_rms_deg": rms(static_rot),
        }
        phases = {}
        first = 0
        if receipt["sequence"] == "subject_1/20231025_170650" and receipt["frames"] == [0,140]:
            bounds = cfg["phase_end_inclusive"]
            phase_policy = "Frozen development source intervals"
        else:
            # Phase display only. Scoring uses every frame and the same physical gates.
            moving = np.flatnonzero(ref_lift >= .10 * ref_lift.max())
            high = np.flatnonzero(ref_lift >= .85 * ref_lift.max())
            bounds = dict(approach=int(moving[0])-1, lift=int(high[0])-1, hold=int(high[-1]),
                          lower=int(moving[-1]), place_release=len(frames)-1)
            phase_policy = "First/last source crossings of 10% and 85% of peak world lift"
        for name, last in bounds.items():
            if last < first:
                phases[name] = None
                continue
            sl = slice(first, last + 1)
            phases[name] = dict(frames=[int(frames[first]), int(frames[last])], position_rms_mm=rms(position[sl]),
                                rotation_rms_deg=rms(angle[sl]), peak_lift_mm=float(lift[sl].max()))
            first = last + 1
    gates = {
        "position": rms(position) <= cfg["maximum_position_rms_mm"] and position.max() <= cfg["maximum_position_error_mm"],
        "rotation": rms(angle) <= cfg["maximum_rotation_rms_deg"] and angle.max() <= cfg["maximum_rotation_error_deg"],
        "beats_fixed_pose": rms(position) < rms(static_pos) and rms(angle) < rms(static_rot),
        "endpoint": position[-1] <= cfg["maximum_endpoint_error_mm"],
        "lift": abs(lift.max() - ref_lift.max()) <= cfg["maximum_peak_lift_error_mm"],
        "hand_support_in_air": air_fraction >= cfg["minimum_supported_airborne_fraction"],
        "release": final_support >= cfg["minimum_final_table_support_bodyweights"] and release_contacts == 0,
        "penetration": metrics["max_carton_penetration_mm"] <= cfg["maximum_carton_penetration_mm"] and
                       metrics["max_self_penetration_mm"] <= cfg["maximum_self_penetration_mm"],
        "joint_limits": metrics["max_joint_excursion_deg"] <= cfg["maximum_joint_excursion_deg"],
        "hand_tracking": metrics["actual_fingertip_rms_mm"] <= cfg["maximum_fingertip_rms_mm"],
        "recorded_command_replay": metrics["rerun_max_state_error"] <= 1e-9,
    }
    write_json(output, dict(protocol=file_pin(protocol), run=file_pin(run), source=file_pin(original),
                            evaluator=file_pin(__file__), metrics=metrics, phase_policy=phase_policy, phases=phases, gates=gates,
                            passed_numeric_development_gates=all(gates.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.run, args.original, args.protocol, args.output)
