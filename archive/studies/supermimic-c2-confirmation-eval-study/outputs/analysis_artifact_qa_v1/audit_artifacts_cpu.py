"""Artifact-only report/ledger; no torch, simulation, or rewritten frozen output."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

analysis = read(ROOT / "outputs/analysis_v3/comparison.json")
render = read(OUT / "render.json")
assert analysis["status"] == "COMPLETE" and len(analysis["results"]) == 27
assert sha(ROOT / "scripts/analyze_confirmation.py") == "b2c77e9ef5b2162ca0130f7d83157b510e61ecc090272e0b233ff3781a7b0e75"
assert sha(ROOT / "outputs/analysis_v3/comparison.json") == render["analysis_sha256"]
for name, digest in render["figures_sha256"].items():
    assert sha(OUT / name) == digest

states = {
    "qualification": ROOT / "outputs/format103_qualification_v1/state.json",
    "V2_invalid": ROOT / "outputs/confirmation_evaluation_v2/state.json",
    "V3_confirmation": ROOT / "outputs/confirmation_evaluation_v3/state.json",
}
q, v2, v3 = (read(states[k]) for k in states)
assert q["status"] == "engineering_passed"
assert v2["status"] == "engineering_failed" and len(v2["jobs"]) == 1
assert v3["status"] == "completed_27_jobs" and len(v3["jobs"]) == 27
childsum = sum(j["process"]["elapsed_s"] for j in v3["jobs"])
evalsum = sum(j["cost"]["evaluator_report_wall_seconds"] for j in analysis["results"])
acost = analysis["evaluation_cost"]
assert abs(childsum - acost["sum_child_supervisor_seconds"]) < 1e-9
assert abs(evalsum - acost["sum_evaluator_report_seconds"]) < 1e-9
assert abs(v3["whole_queue_wall_seconds"] - acost["whole_queue_wall_seconds"]) < 1e-9
independent = read(ROOT / "outputs/control_cost_audit_v3/report.json")
assert independent["status"] == "passed"

ledger = {
    "status": "PASS",
    "scope": "Read existing attempt managers and frozen analysis only; no GPU/physics/training",
    "state_sha256": {k: sha(p) for k, p in states.items()},
    "qualification": {"status": q["status"], "driver_wall_s": q["whole_wall_seconds"], "child_wall_s": q["process"]["elapsed_s"], "main27_included": False},
    "V2_invalid": {"status": v2["status"], "driver_wall_s": v2["whole_queue_wall_seconds"], "started_children": 1, "not_run_jobs": 26, "child_wall_s": v2["jobs"][0]["process"]["elapsed_s"], "main27_included": False, "reason": "WinError5 metadata runtime rename; no qualified main sample"},
    "V3_confirmation": {"status": v3["status"], "driver_wall_s": v3["whole_queue_wall_seconds"], "sum_child_wall_s": childsum, "sum_evaluator_report_wall_s": evalsum, "main27_jobs": 27},
    "disjoint_attempt_driver_duration_sum_s": q["whole_wall_seconds"] + v2["whole_queue_wall_seconds"] + v3["whole_queue_wall_seconds"],
    "accounting_limits": ["Queue, child and evaluator scopes are nested: do not add those scopes", "The disjoint attempt-driver sum excludes preparation/debug/analysis/training and gaps; it is not end-to-end project time", "None of these wall times measure GPU kernel busy time", "Actual runtime metadata IO retry counts were not stored and remain unknown"],
    "independent_control_cost_audit": {"path": str(ROOT / "outputs/control_cost_audit_v3/report.json"), "sha256": sha(ROOT / "outputs/control_cost_audit_v3/report.json")},
}
save("cost_ledger.json", ledger)

qa = {
    "status": "PASS",
    "scope": "Actual image inspection plus artifact-only depth/interpretation checks; original frozen results unchanged",
    "source_sha256": sha(Path(__file__)),
    "frozen_analysis_sha256": sha(ROOT / "scripts/analyze_confirmation.py"),
    "comparison_sha256": render["analysis_sha256"],
    "new_figures_sha256": render["figures_sha256"],
    "viewed_figures": ["seed_summary_annotated.png", "penetration_diagnostic.png"],
    "visual_checks": {"all_labels_legible": True, "no_clipped_peaks": True, "seed_4_5_6_shape_legend": True, "support_zoom_explained": True, "N3_and_ranges_not_CI_explicit": True, "all27_depth_traces_and_3mm_gate": True},
    "original_figures_sha256": {n: sha(ROOT / "outputs/analysis_v3" / n) for n in ("tracking.png", "tracking.svg", "seed_summary.png", "seed_summary.svg")},
    "depth_components_match_aggregate_max_error_m": render["depth_component_aggregate_max_abs_error_m"],
    "P_ONLY_seed6_hand_object_peak_mm_by_repeat": render["P_ONLY_seed6_hand_object_peak_mm_by_repeat"],
    "complete_clip_explanation": render["complete_clip_explanation"],
    "P_ONLY_seed4_full_horizon_evidence": render["P_ONLY_seed4_latches"],
    "primary_all3_process_pass_training_seed_count": {"A1L": "1/3", "P_ONLY": "0/3", "A2": "1/3"},
    "interpretation": ["Training seed N=3 per method; three numerical process repeats are nested observations", "A2 does not demonstrate an overall primary task-rate improvement over A1L in this small cohort", "P_ONLY seed6 passes tracking but fails the unchanged 3mm depth gate in all three processes", "P_ONLY seed5 has the highest auxiliary support mean while failing tracking and actual joint limits; support is not a task-success surrogate", "A2 minus P_ONLY shaft maximum improves for seeds4/5 but increases by about0.06355deg for seed6", "A1L seed5 repeat2 has actual joint excess6.764464mrad; bounded target commands do not establish actual-state or hardware safety", "Qualification, invalid V2 and historical seed/dose cohorts are excluded from the main27"],
    "GPU_used": False,
    "new_physics_steps": 0,
}
save("visual_qa.json", qa)
print(json.dumps({"status": "PASS", "cost_ledger": str(OUT / "cost_ledger.json"), "visual_qa": str(OUT / "visual_qa.json")}, ensure_ascii=False))
