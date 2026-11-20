"""Evaluate a Wuji carton run against the original source, phase by phase."""

import argparse
import json
from pathlib import Path

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


def evaluate(run, original, protocol, output):
    cfg = json.loads(protocol.read_bytes())
    report = json.loads(run.read_bytes())
    receipt = json.loads(original.read_bytes())
    trace_path = verify_file_pin(report["trace"])
    with np.load(verify_file_pin(receipt["source_poses"]), allow_pickle=False) as human:
        source = human["world_from_carton"]
        tips = human["world_right_hand21"][:, [4, 8, 12, 16, 20]]
        frames = human["frames"]
    with np.load(trace_path, allow_pickle=False) as t:
        np.testing.assert_array_equal(t["frame_index"], np.arange(141))
        np.testing.assert_array_equal(frames, t["frame_index"])
        np.testing.assert_allclose(t["world_from_carton_source"], source, atol=1e-9, rtol=0)
        np.testing.assert_allclose(t["time"], frames / cfg["fps_nominal"], atol=1e-9, rtol=0)
        actual = t["world_from_carton_sim"]
        position, angle = errors(source, actual)
        static_pos, static_rot = errors(source, np.repeat(source[:1], len(source), axis=0))
        # Gravity defines lift. Contact support is measured over the actual control interval.
        gravity = np.asarray(report["gravity_m_s2"])
        up = -gravity / np.linalg.norm(gravity)
        lift = (actual[:, :3, 3] - actual[0, :3, 3]) @ up * 1000
        ref_lift = (source[:, :3, 3] - source[0, :3, 3]) @ up * 1000
        weight = report["object_hypotheses"]["mass_kg"] * np.linalg.norm(gravity)
        dt = np.diff(t["time"])
        table_load = t["interval_table_carton_normal_impulse_Ns"][1:] / dt / weight
        hand_load = t["interval_hand_carton_normal_impulse_Ns"][1:] / dt / weight
        airborne = ref_lift[1:] > cfg["source_lift_for_airborne_mm"]
        supported = ((table_load < cfg["maximum_table_mean_force_in_air_bodyweights"]) &
                     (hand_load >= cfg["minimum_hand_mean_force_in_air_bodyweights"]))
        release = slice(-cfg["release_last_n_intervals"], None)
        air_fraction = float(np.mean(supported[airborne]))
        final_support = float(table_load[release].min())
        release_contacts = int(t["interval_max_hand_contacts"][release].max())
        metrics = {
            "position_rms_mm": rms(position), "position_peak_mm": float(position.max()),
            "rotation_rms_deg": rms(angle), "rotation_peak_deg": float(angle.max()),
            "worst_position_frame": int(position.argmax()), "worst_rotation_frame": int(angle.argmax()),
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
        for name, last in cfg["phase_end_inclusive"].items():
            sl = slice(first, last + 1)
            phases[name] = dict(frames=[first, last], position_rms_mm=rms(position[sl]),
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
        "recorded_command_replay": report["independent_substep_action_replay_max_state_error"] <= 1e-9 and
                                   report["external_object_wrench_or_reset_after_initialization"] is False,
    }
    write_json(output, dict(protocol=file_pin(protocol), run=file_pin(run), source=file_pin(original),
                            evaluator=file_pin(__file__), metrics=metrics, phases=phases, gates=gates,
                            passed_numeric_development_gates=all(gates.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.run, args.original, args.protocol, args.output)
