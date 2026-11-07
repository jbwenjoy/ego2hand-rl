"""Record a source-bound static review; do not import codec, Torch, or MuJoCo."""
from pathlib import Path
import hashlib
import json
import ast

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


source = ROOT / "src/rough_prefix/saved_state_codec_v3.py"
proof_dir = ROOT / "outputs/saved_state_codec_cpu_v4"
proof = read(proof_dir / "report.json")
runplan_path = ROOT / "configs/WM-onpolicy-CPU-diagnostic-runplan-v1.json"
plan = read(runplan_path)
old_static_path = ROOT / "outputs/native104_WM_raw_interface_review_v1/static_model_metadata.json"
static = read(old_static_path)
source_sha = pin(source)["sha256"]
nodes = ast.parse(source.read_text(encoding="utf-8"))
calls = [ast.unparse(n.func) for n in ast.walk(nodes) if isinstance(n, ast.Call)]
checks = {
    "reviewed_exact_codec_8321538": source_sha == "8321538887c660456fc50ce8fc37d3b130760212fa63c905310bbd6a079d5156",
    "proof_archived_codec_matches_current": pin(proof_dir / source.name)["sha256"] == source_sha,
    "proof_declares_exact_current_codec": proof["source"]["sha256"] == source_sha,
    "proof32_metadata_complete_and_passed": proof["status"] == "PASS_RECONSTRUCTION_ONLY" and proof["passed"] == proof["total"] == len(proof["checks"]) == 32 and all(v is True for v in proof["checks"].values()),
    "runplan_pins_same_proof_and_codec": plan["proofs"]["codec32"]["sha256"] == pin(proof_dir / "report.json")["sha256"] and plan["source_pins"]["state_codec"]["sha256"] == source_sha,
    "original_state_method_called_not_constructor": "SimReader.state" in calls and "SimReader" not in calls and "PenSimReader" not in calls,
    "no_native_physics_query_calls_in_codec_AST": not any(n in calls for n in ("mujoco.MjData", "mujoco.mj_step", "mujoco.mj_forward", "mujoco.mj_collision", "mujoco.mj_geomDistance", "mujoco.mj_contactForce")),
    "proof_covers_exact_four_registered_paths": [v["method"] for v in proof["datasets"]] == ["A0", "A1L", "P_ONLY", "A2"] == list(plan["datasets"]),
    "proof_reconstruction_not_inference_or_new_physics": proof["Env_constructed"] is False and proof["physics_steps_added"] == 0 and proof["WM_quality_assessed"] is False and proof["native_features_recorded"] is False,
    "all_four_state_control_shape_metadata_correct": all(v["rows"] == 15 and v["state_dim"] == 58 and v["target_dim"] == 20 and v["absolute_executed_u"] is True for v in proof["datasets"]),
    "all_four_truths_explicitly_UNKNOWN": all(v["selfcollision_labels"] == "UNKNOWN_not_saved_sensor_found" for v in proof["datasets"]),
    "prior_model_identity_child_has_palm_not_mocap": static["bodies"]["robot/right_palm_link"]["mocap_id"] == -1 and static["bodies"]["robot/right_palm_link"]["parent_id"] == 3 and static["bodies"]["robot/mocap_base"]["mocap_id"] == 0,
}
datasets = []
for item in proof["datasets"]:
    expected = plan["datasets"][item["method"]]
    all_bound = set(item["pins"]) == set(expected["codec_input_pins"]) and all(
        item["pins"][k]["sha256"] == v["sha256"] == pin(v["path"])["sha256"]
        for k, v in expected["codec_input_pins"].items()
    )
    datasets.append({"method": item["method"], "registered_saved_bytes_match_existing_proof": all_bound,
                     "rows": item["rows"], "source_indices_k": item["source_indices_k"],
                     "source_indices_kplus1": item["source_indices_kplus1"],
                     "native_self15_truth": "UNKNOWN", "feature_kind": "CPU_native_method_reconstruction"})
checks["all_registered_saved_bytes_still_match_proof"] = all(v["registered_saved_bytes_match_existing_proof"] for v in datasets)
result = {
    "schema": "rough_saved_state_codec_independent_static_review:1",
    "status": "NO_BLOCKING_FINDING_FOR_REGISTERED_SAVED_STATE_TARGETS" if all(checks.values()) else "SOURCE_OR_PROOF_BINDING_FAILED",
    "checks_kind": "Static/source/proof/byte-identity checks; existing 32 reconstruction checks were not repeated",
    "checks": checks,
    "blocking_findings": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
    "reviewed_source": pin(source), "reviewed_CPU_proof": pin(proof_dir / "report.json"),
    "reviewed_checker": pin(proof_dir / "validate_saved_state_codec_cpu_v4.py"),
    "runplan": pin(runplan_path), "protocol": {"path": plan["protocol"]["path"], "sha256": plan["protocol"]["sha256"]},
    "prior_static_model_review": pin(old_static_path), "datasets": datasets,
    "state_target_validity": "Valid postintegration state targets for these registered complete noreset fixed-palm rollouts; reconstructed CPU features, not saved native-feature bit replay",
    "sensor_phase": "Last solver preintegration at t+15ms; separate from state target postintegration t+20ms",
    "self15_ground_truth": "UNKNOWN", "WM_accuracy_or_internal_gradient_reviewed": False,
    "reviewer_execution": {"codec_runs": 0, "native_state_reconstructions": 0, "WM_forward_calls": 0, "physics_or_geometry_queries": 0, "Env_or_Reader_constructors": 0, "GPU_calls": 0, "training_updates": 0},
    "documentation": pin(OUT / "README.md"), "reviewer_source": pin(__file__),
}
target = OUT / "report.json"
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"status": result["status"], "report": pin(target), "false_checks": result["blocking_findings"]}, indent=2))
raise SystemExit(0 if all(checks.values()) else 2)
