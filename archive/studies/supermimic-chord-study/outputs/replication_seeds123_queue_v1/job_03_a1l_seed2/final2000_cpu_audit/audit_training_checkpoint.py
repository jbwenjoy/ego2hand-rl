"""Read-only CPU checkpoint health audit; no simulator, GPU, or optimizer steps.

This checks serialization, finite values, frozen inputs and counted budgets.
Passing it does not establish physical task success or contact-model validity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def audit(run, iteration):
    torch.set_num_threads(2)
    attempt = run / "compiled"
    path = attempt / "train" / f"model_{iteration}.pt"
    sidecar = read(path.with_suffix(".meta.json"))
    inventory = read(run / "source/inventory.json")
    config = read(run / "input_config.json")
    effective = read(attempt / "effective_training_config.json")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)

    check("sha_matches_completed_sidecar", digest(path) == sidecar["checkpoint_sha256"])
    index = [json.loads(line) for line in (attempt / "checkpoint_index.jsonl").read_text(encoding="utf-8").splitlines()]
    check("completed_sidecar_is_in_index", sidecar in index)
    check("format_and_task", saved["format_version"] == 3 and saved["config"]["task"] == "hand2-pen-spin")
    check("iteration", saved["iteration"] == iteration == sidecar["iteration"])
    check("config_matches_frozen_effective_config", json.loads(json.dumps(saved["config"])) == effective)
    check("all_module_tensors_finite", finite(saved["modules"]))
    check("all_optimizer_tensors_finite", finite(saved["optimizers"]))
    check("torque_normalizer_finite", finite(saved["torque_norm"]))
    check("reference_signature_present", bool(saved.get("reference_signature")))
    expected_steps = {
        "wm": iteration * effective["wm_batches"],
        "fail_head": iteration * effective["wm_batches"],
        "actor": max(0, iteration - effective["wm_only_iters"]),
        "critic": max(0, iteration - effective["wm_only_iters"]) * effective["critic_epochs"] * effective["critic_minibatches"],
    }
    actual_steps = {name: sidecar["actual_optimizer_steps"].get(name, 0) for name in expected_steps}
    check("actual_optimizer_counts", actual_steps == expected_steps)
    expected_transitions = effective["num_envs"] * (max(effective["warmup_steps"], effective["wm_window"]) + int(iteration / effective["utd"]))
    check("simulator_transition_count", saved["sim_steps"] == sidecar["real_simulator_transitions"] == expected_transitions)
    optimizer_steps = {}
    for name, optimizer in saved["optimizers"].items():
        counts = sorted({float(v["step"]) for v in optimizer["state"].values() if "step" in v})
        optimizer_steps[name] = counts
        check(f"optimizer_{name}_stored_steps", (counts == [float(expected_steps[name])]) if expected_steps[name] else not counts)
    # Read only complete rows. The live writer can be appending its next row.
    metric_bytes = (attempt / "metrics.jsonl").read_bytes()
    complete_bytes = metric_bytes[:metric_bytes.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in complete_bytes.decode("utf-8").splitlines()]
    check("complete_metric_rows_present_and_finite", bool(rows) and all(finite(row) for row in rows))
    check("training_script_unchanged", digest(run / "source/train_controlled_pen.py") == inventory["script_sha256"] == digest(Path(__file__).with_name("train_controlled_pen.py")))
    dex = Path(inventory["dex_root"])
    mismatches = [name for name, sha in inventory["files"].items() if digest(dex / name) != sha]
    check("dex_sources_unchanged", not mismatches)
    runtime = read(run / "runtime.json")
    check("input_config_unchanged", digest(runtime["input_config"]) == digest(run / "input_config.json"))
    frozen_protocol = run / "source/paired_protocol.json"
    check("paired_protocol_unchanged", digest(frozen_protocol) == digest(run.parents[1] / config["paired_protocol_file"]))
    provenance = read(run / "reference/provenance.json")["frozen_evaluation_reference"]
    reference = run / "reference/frozen_evaluation_reference.npz"
    check("frozen_evaluation_reference_unchanged", digest(reference) == provenance["npz_sha256"] == digest(provenance["source"]))
    if config.get("contact_objective"):
        target_root = run / "source/contact_objective"
        spec = read(target_root / "frozen_inputs.json")
        expected = config["contact_objective"]["expected_hashes"]
        check("A2_expected_hashes_frozen", spec["expected_hashes"] == expected)
        files = {
            "manifest_sha256": target_root / "bundle/manifest.json",
            "target_npz_sha256": target_root / "bundle/contact_targets.npz",
            "rules_sha256": target_root / "bundle/rules.json",
            "raw_geometry_sha256": target_root / "bundle/raw_geometry.json",
            "time_windows_sha256": target_root / "bundle/time_windows.json",
            "reference_npz_sha256": target_root / "reference.npz",
            "reference_json_sha256": target_root / "reference.json",
            "model_sha256": target_root / "model.mjb",
        }
        for name, file in files.items():
            check(f"A2_{name}", digest(file) == expected[name])
        with zipfile.ZipFile(target_root / "study_contact_source.zip") as archive:
            for name, file in (("geometry_module_sha256", "contact_geometry.py"),
                               ("objective_module_sha256", "contact_objective.py")):
                frozen_hash = hashlib.sha256(archive.read(f"chord_study/{file}")).hexdigest()
                check(f"A2_{name}_frozen_and_current", frozen_hash == expected[name] == digest(run.parents[1] / "src/chord_study" / file))
        objective_meta = read(attempt / "contact_objective_metadata.json")
        check("A2_fixed_sampling", objective_meta["sampled_hand_surface_points"] == 1344 and objective_meta["basis"]["training_directions"] == 64)
        check("A2_qualification_and_dt_preserved", objective_meta["qualified_contact_predictor"] is False and objective_meta["dt_applied_inside_module"] is False and spec["per_control_step_scale"] == .02)
    context_difference = {}
    with np.load(attempt / "fixed_physics_context.npz", allow_pickle=False) as physical:
        for name in physical.files:
            value = physical[name]
            check(f"physics_{name}_finite", np.isfinite(value).all())
            context_difference[name] = float(np.max(np.abs(value - value[:1])))
    check("fixed_physics_equal_across_worlds", all(value == 0. for value in context_difference.values()))
    return {
        "status": "passed" if all(checks.values()) else "failed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": digest(__file__),
        "run": str(run), "checkpoint": str(path), "checkpoint_sha256": digest(path), "iteration": iteration,
        "checks": checks, "actual_optimizer_steps": actual_steps, "stored_optimizer_step_values": optimizer_steps,
        "real_simulator_transitions": saved["sim_steps"], "finite_complete_metric_rows": len(rows),
        "metrics_snapshot_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "physics_max_world_difference": context_difference, "source_mismatches": mismatches,
        "claim_scope": "serialization/finiteness/frozen-source-and-budget health only; physical task success requires independent evaluation",
        "phase": "world-model pretraining; actor/critic have not updated" if iteration <= effective["wm_only_iters"] else "policy training",
    }, complete_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace an audit: {args.output_dir}")
    result, metrics = audit(args.run.resolve(strict=True), args.iteration)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_snapshot.jsonl").write_bytes(metrics)
    shutil.copy2(__file__, args.output_dir / Path(__file__).name)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
