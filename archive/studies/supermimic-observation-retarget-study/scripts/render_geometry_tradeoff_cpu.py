"""Static geometric tradeoff figure, native collision queries only, no dynamics."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from native_filtered_contact_geometry import NativeFilteredContactGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    names = [("Palm-root IK", "cached_landmarks_palmroot_v2"),
             ("+ object overlap", "contact_refined_nominal_geometry_v1"),
             ("+ object and self overlap", "native_contact_refined_geometry_v1")]
    geometry = NativeFilteredContactGeometry(a.model, a.context)
    arrays, hashes = {}, {}
    for name, directory in names:
        p = a.root / "outputs" / directory / "geometric_retarget.npz"
        hashes[directory] = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p, allow_pickle=False) as z:
            q, pose = z["joint_pos_rad"], z["mapped_object_pose_w"]
            tip = np.linalg.norm(z["fk_hand_points_w"][:, [4, 8, 12, 16, 20]] - z["morphology_normalized_target_points_w"][:, [4, 8, 12, 16, 20]], axis=-1).mean(axis=1) * 1000
            bone = z["bone_angle_error_deg"].mean(axis=1)
            t = z["time_s"].copy()
        object_depth, self_depth = [], []
        for joints, object_pose in zip(q, pose):
            geometry.set_object_pose(object_pose)
            value = geometry.evaluate(joints)
            for kind, values in (("hand_object", object_depth), ("hand_self", self_depth)):
                selected = value.generated_mask & (geometry.pair_kind == kind)
                values.append(float(np.maximum(-value.signed_distance_m[selected], 0).max()) * 1000 if selected.any() else np.nan)
        arrays[directory + "_object_depth_mm"] = np.array(object_depth)
        arrays[directory + "_self_depth_mm"] = np.array(self_depth)
        arrays[directory + "_tip_error_mm"] = tip
        arrays[directory + "_bone_angle_deg"] = bone
    arrays["time_s"] = t
    np.savez_compressed(a.output / "plotted_geometry.npz", **arrays)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    columns = [("_object_depth_mm", "Generated hand-object depth (mm)"),
               ("_self_depth_mm", "Generated hand-self depth (mm)"),
               ("_tip_error_mm", "Normalized-target tip error (mm)"),
               ("_bone_angle_deg", "Mean bone-direction error (deg)")]
    for axis, (key, ylabel) in zip(ax.flat, columns):
        for color, (name, directory) in zip(("#566278", "#d08b28", "#177c79"), names):
            axis.plot(t, arrays[directory + key], label=name, color=color)
        if "depth" in key:
            axis.axhline(.5, color="black", ls="--", lw=.8)
            axis.axhline(3, color="gray", ls=":", lw=.8)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    for axis in ax[-1]:
        axis.set_xlabel("Candidate time (s)")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .005))
    fig.suptitle("Same landmark targets and fixed mapped object: geometry tradeoff only\nDashed0.5mm=soft allowance; dotted3mm=diagnostic screen; no force/dynamics/grasp qualification", fontsize=12)
    fig.text(.5, .065, "Object-only:148/151 optimizer successes; object+self:143/151. All limited frames retained. Unknown gaps remain unknown.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.87, bottom=.16, left=.08, right=.985, wspace=.25, hspace=.14)
    fig.savefig(a.output / "geometry_tradeoff.png", dpi=160)
    fig.savefig(a.output / "geometry_tradeoff.svg")
    plt.close(fig)
    report = {"scope": "Consistent native filtered static collision queries for allthree frozen candidates; no optimizer or physical rollout. No generated contact plotted as NaN, not zero clearance.", "inputs_sha256": hashes,
              "model_sha256": hashlib.sha256(a.model.read_bytes()).hexdigest(), "context_sha256": hashlib.sha256(a.context.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "module_sha256": hashlib.sha256(Path(__file__).with_name("native_filtered_contact_geometry.py").read_bytes()).hexdigest(),
              "dynamics_steps": 0, "source_robot_labels_read": False}
    (a.output / "render.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
