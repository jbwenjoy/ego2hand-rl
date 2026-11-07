"""Finalize only this already collected source/saved-array interface review."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent


def pin(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
bodies = report["static_model"]["bodies"]
base, palm = bodies["robot/mocap_base"], bodies["robot/right_palm_link"]
checks = {
    "three_actual_saved_trajectories": set(report["saved_inventory"]) == {"A1L", "P_ONLY", "A2"},
    "asset_specific_palm_identity_child": (
        base["id"] == 3 and base["parent_id"] == 0 and base["mocap_id"] == 0
        and palm["id"] == 4 and palm["parent_id"] == 3 and palm["mocap_id"] == -1
        and palm["position_body"] == [0., 0., 0.]
        and palm["quaternion_wxyz_body"] == [1., 0., 0., 0.]
        and palm["joints"] == 0
    ),
    "object_COM_offset_zero": bodies["object/cube"]["COM_offset_body_m"] == [0., 0., 0.],
    "all_raw_coverage_61_60_15": all(
        v["state61_count"] == 61 and v["solver60_count"] == 60 and v["events15_count"] == 15
        for v in report["saved_inventory"].values()
    ),
    "all_palm_mapping_saved_assertions": all(
        v["mocap_constant_all61"] and v["mocap_equals_bridge_palm0_FP32"]
        for v in report["saved_inventory"].values()
    ),
    "all_actual_u_shapes_15x1x20": all(
        v["actualu_shapes"] == [[1, 20]] * 15 for v in report["saved_inventory"].values()
    ),
    "all_context_46_and_exogenous_zero": all(
        v["reader_xi_shape"] == [1, 46] and v["all_saved_exogenous_force0"]
        for v in report["saved_inventory"].values()
    ),
    "native_state_reconstruction_not_claimed_executed": report["executed_native_state_reconstruction"] is False,
    "all_self15_truth_explicit_unknown": all(
        v["self15_truth_status"] == "UNKNOWN_DIRECT_NATIVE_SENSOR_NOT_SAVED"
        and not v["native_self15_sensor_directly_saved"]
        for v in report["saved_inventory"].values()
    ),
    "three_finite_WM_headers_104_iteration2000": all(
        v["format_version"] == 104 and v["iteration"] == 2000 and v["all_WM_tensors_finite"]
        and v["WM_input_width"] == 298 and v["WM_delta_width"] == 52
        and v["WM_output_width"] == 67 and v["contact_dim"] == 15
        for v in report["three_final_WM_headers"].values()
    ),
    "no_new_physics_GPU_or_WM_inference": (
        report["WM_forward_calls"] == 0 and report["GPU_used"] is False
        and report["CUDA_initialized"] is False and not any(report["forbidden_calls"].values())
    ),
}
result = {
    "schema": "native104_WM_raw_interface_READY:1",
    "status": "READY_INTERFACE_REVIEW_ONLY" if all(checks.values()) else "REVIEW_ASSERTION_FAILED",
    "checks_kind": "Metadata/source/saved-inventory assertions; not physics or inference qualification",
    "checks": checks,
    "state_codec_execution": "NOT_RUN_IN_THIS_REVIEW",
    "WM_inference_execution": "NOT_RUN_IN_THIS_REVIEW",
    "native_self15_truth": "UNKNOWN",
    "causal_action_gradient_qualification": "NOT_GRANTED",
    "artifacts": {name: pin(OUT / name) for name in (
        "report.json", "README.md", "source_pins.json", "static_model_metadata.json",
        "collect_readonly_review.py", "finalize_review_cpu.py",
    )},
}
target = OUT / "READY.json"
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"status": result["status"], "checks": checks, "READY": pin(target)}, indent=2))
raise SystemExit(0 if all(checks.values()) else 2)
