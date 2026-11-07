"""Export a label-restricted cached-observation fixture, never a real-ego claim.

This is an API/data-view boundary, not filesystem access control. The source
contains robot labels; its vision/camera lineage has not been established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = {
    "time_s": "time_s",
    "hand_landmarks_source_w": "source_mediapipe_landmarks_w",
    "object_pose_source_w": "source_object_pose_w",
    "wrist_pose_source_w": "source_wrist_pose_w",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    metadata_path = source.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("source_kind") != "upstream_kinematic":
        raise ValueError("This exporter is restricted to the cached controlled fixture")
    with np.load(source, allow_pickle=False) as raw:
        arrays = {dst: np.array(raw[src], copy=True) for dst, src in FIELDS.items()}
    t = arrays["time_s"]
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Time must be one-dimensional and strictly increasing")
    expected = {"hand_landmarks_source_w": (len(t), 21, 3),
                "object_pose_source_w": (len(t), 7),
                "wrist_pose_source_w": (len(t), 7)}
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite source observation: {name}")
        if name in expected and value.shape != expected[name]:
            raise ValueError(f"Unexpected shape: {name} {value.shape}")
    for name in ("object_pose_source_w", "wrist_pose_source_w"):
        if not np.allclose(np.linalg.norm(arrays[name][:, 3:], axis=-1), 1, atol=1e-4):
            raise ValueError(f"Nonunit wxyz source orientation: {name}")
    arrays["landmark_confidence_known"] = np.zeros((len(t), 21), dtype=bool)
    arrays["object_confidence_known"] = np.zeros(len(t), dtype=bool)
    arrays["wrist_confidence_known"] = np.zeros(len(t), dtype=bool)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "observations.npz"
    np.savez_compressed(target, **arrays)
    with np.load(target, allow_pickle=False) as view:
        if set(view.files) != set(arrays):
            raise AssertionError("Output is not the exact observation allowlist")
        for name in arrays:
            np.testing.assert_array_equal(view[name], arrays[name])
    report = {
        "schema": "upstream_cached_observation_fixture:1",
        "status": "export_and_exact_allowlist_check_pass; physical_adapter_not_qualified",
        "source_kind": "upstream_kinematic_cached_observation_view",
        "is_real_ego": False,
        "contains_robot_joint_labels": False,
        "contains_executed_controls": False,
        "contains_reference_contacts": False,
        "source_sha256": sha256(source),
        "source_metadata_sha256": sha256(metadata_path),
        "observation_sha256": sha256(target),
        "exporter_sha256": sha256(Path(__file__)),
        "source_id": metadata["source_id"],
        "split_group": metadata["split_group"],
        "original_source_frames": metadata["provenance"].get("original_source_frames"),
        "original_video_parent_id": metadata["provenance"].get("original_video_parent_id"),
        "frames": len(t),
        "units": "source cached SI; independent metric-scale calibration unavailable",
        "quaternion_order": "wxyz",
        "frame": "upstream source world; source-to-camera and source-to-mount unqualified",
        "fixed_wrist_rebase_applied": False,
        "camera_intrinsics": None,
        "camera_pose": None,
        "metric_scale_calibration": None,
        "confidence_values": None,
        "confidence_rule": "unknown masks are false; no confidence values fabricated",
        "allowlist": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                      for name, value in arrays.items()},
        "prohibited_candidate_inputs": ["robot joint angles/velocities", "robot link positions",
                                        "teacher controls", "reference contact targets",
                                        "fixed-mount robot reference or phase labels"],
        "boundary": "Only the observation NPZ is candidate input. This is not filesystem isolation; original files still exist.",
        "physical_adapter_blockers": ["unverified original video/camera lineage",
                                      "source frame and metric-scale calibration",
                                      "wrist/landmark observation semantics and uncertainty"],
        "claim": "Controlled label-restricted data interface only; no reconstruction, retargeting, dynamics or generalization result",
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = export(args.source, args.output)
    print(json.dumps({key: result[key] for key in ("status", "frames", "observation_sha256", "is_real_ego")}, ensure_ascii=False))
