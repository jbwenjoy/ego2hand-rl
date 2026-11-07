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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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


def actual_u_coverage(run,saved):
    """Audit only the complete JSONL prefix named by this checkpoint marker."""
    controller=saved.get("c2_controller",{})
    marker=controller.get("latest_actualu_evidence")
    path=run/"compiled/c2_actualu_evidence.jsonl"
    if marker is None:
        return {"status":"PENDING_UNMEASURED","scope":"No checkpoint-linked all-batch marker; do not substitute sparse/static proof"}
    last=marker.get("collect_call")
    if type(last) is not int or last<1:raise ValueError("Invalid actualu checkpoint marker")
    raw=path.read_bytes();complete=raw[:raw.rfind(b"\n")+1];lines=complete.splitlines(keepends=True)
    if len(lines)<last:raise ValueError("Actualu JSONL checkpoint prefix incomplete")
    prefix=b"".join(lines[:last]);rows=[json.loads(x) for x in lines[:last]]
    errors={"cache_vs_actual","proposed_projection_vs_cache","residual_before_after_filter","replay_u_vs_actual","replay_c_vs_residual"}
    checks={"calls_contiguous_from1":[r.get("collect_call") for r in rows]==list(range(1,last+1)),
            "one_warmup_collect_plus_each_iteration":last==saved["iteration"]+1,
            "latest_collect_iteration_is_preincrement":marker.get("iteration")==max(0,saved["iteration"]-1),
            "last_record_equals_checkpoint_marker":rows[-1]==marker,
            "all_positive_steps_correct_worlds":all(type(r.get("steps")) is int and r["steps"]>0 and r.get("worlds")==saved["config"]["num_envs"] for r in rows),
            "all_batches_finite_inside_exact":all(r.get("all_batch_exact") is True and r.get("all_finite") is True and r.get("within_inset_domain") is True for r in rows),
            "every_declared_mismatch_zero":all(set(r.get("max_abs_errors",{}))==errors and all(type(v) in (int,float) and math.isfinite(v) and v==0 for v in r["max_abs_errors"].values()) for r in rows),
            "component_accounting":all(r.get("total_components")==r["steps"]*r["worlds"]*20 and type(r.get("clamped_components")) is int and 0<=r["clamped_components"]<=r["total_components"] for r in rows),
            "total_transitions_equal_checkpoint":sum(r["steps"]*r["worlds"] for r in rows)==saved["sim_steps"]}
    expected_calls=[n for n in range(1,last+1) if n in (1,2,3) or n%100==0]
    trace_hashes={};lower=np.asarray(controller["target_lower_rad"],dtype=np.float32)+np.float32(.020);upper=np.asarray(controller["target_upper_rad"],dtype=np.float32)-np.float32(.020)
    for call in expected_calls:
        trace=run/f"compiled/c2_actualu_trace_{call:05d}.npz";trace_hashes[trace.name]=digest(trace)
        with np.load(trace,allow_pickle=False) as a:
            shape=(min(4,rows[call-1]["steps"]),min(4,rows[call-1]["worlds"]),20)
            checks[f"trace{call}_shape_finite"]=all(k in a and a[k].shape==shape and np.isfinite(a[k]).all() for k in ("proposed","cached","actual","residual_before_filter","residual","replay_u","replay_c"))
            checks[f"trace{call}_projection_cache_actual_replay"]=np.array_equal(np.clip(a["proposed"].astype(np.float32),lower,upper),a["cached"]) and np.array_equal(a["cached"],a["actual"]) and np.array_equal(a["actual"],a["replay_u"]) and np.array_equal(a["residual"],a["replay_c"]) and np.array_equal(a["residual_before_filter"],a["residual"])
    return {"status":"PASSED_CHECKPOINT_PREFIX" if all(checks.values()) else "FAILED_CHECKPOINT_PREFIX","checks":{k:bool(v) for k,v in checks.items()},"source_path":str(path),"prefix_last_collect_call":last,"latest_collect_iteration":marker.get("iteration"),"prefix_row_count":len(rows),"prefix_byte_count":len(prefix),"prefix_sha256":hashlib.sha256(prefix).hexdigest(),"verified_real_transitions":sum(r["steps"]*r["worlds"] for r in rows),"sampled_trace_sha256":trace_hashes,"scope":"All recorded collection batches through this checkpoint: projected/cache/actual/replayu, residualbefore-afterfilter and replayc/residual comparisons. Fixed sparse traces independently recomputed. Does not establish physical-state bounds; later JSONL append is outside this checkpoint."}


def c2_contract_checks(run, saved, config, iteration, queue_manifest=None):
    """Metadata/source health, separately exposing unmeasured input coverage."""
    root=Path(__file__).resolve().parents[1]
    controller=saved.get("c2_controller",{})
    checks={"C2_required_format103_metadata":saved.get("format_version")==103 and controller.get("format_version")==103,
            "C2_fixed20_FP32_fromscratch":controller.get("kind")=="C2_TARGET_INSET" and controller.get("margin_rad")==.020 and controller.get("exact_resume_supported") is False and config.get("c2_controller")=={"kind":"C2_TARGET_INSET","margin_rad":.020,"checkpoint_format_version":103,"resume":"from_scratch_only"},
            "C2_real_arm_identity":controller.get("trained_method")==config["method"],
            "C2_no_warmstart_args":not ({"checkpoint","model_checkpoint"}&set(config.get("training",{}))),
            "C2_native_reward_WM_loss_claim":controller.get("no_reward_or_WM_loss_change") is True}
    for key,path in [("adapter_sha256",root/"src/c2_training_adapter.py"),("pure_projection_sha256",root/"src/target_inset.py"),("entry_sha256",root/"scripts/train_controlled_pen.py")]:
        checks["C2_current_"+key]=controller.get(key)==digest(path)
    lower,upper=controller.get("target_lower_rad"),controller.get("target_upper_rad")
    checks["C2_finite_nonempty20joint_domain"]=isinstance(lower,list) and isinstance(upper,list) and len(lower)==len(upper)==20 and finite(lower) and finite(upper) and all(a+.020<=b-.020 for a,b in zip(lower,upper))
    # Snapshots are an independent archival requirement. The adapter owner
    # supplies this map once freeze_source implements it; absence is explicit.
    inventory=read(run/"source/inventory.json")
    archived=inventory.get("c2_controller",{})
    extension=archived.get("files",{})
    current={"adapter":root/"src/c2_training_adapter.py","pure_projection":root/"src/target_inset.py","penalty_wrapper":root/"src/penalty_study/penalty_only_objective.py","entry":root/"scripts/train_controlled_pen.py","original_trainer":root/"source_baselines/original_train_controlled_pen.py"}
    source_root=run/"source/c2_controller"
    checks["C2_extension_sources_archived"]=set(extension)==set(current) and all((source_root/pin["filename"]).is_file() and digest(source_root/pin["filename"])==pin["sha256"]==digest(current[name]) for name,pin in extension.items())
    contract=read(source_root/"controller_contract.json")
    checks["C2_controller_contract_archived"]=digest(source_root/"controller_contract.json")==archived.get("contract_sha256") and contract.get("files")==extension and contract.get("controller")==config["c2_controller"] and contract.get("base_method")==config["method"] and contract.get("training_seed")==config["seed"]
    checks["C2_controller_input_and_inventory_bindings"]=controller.get("config_sha256")==digest(run/"input_config.json") and controller.get("native_source_inventory_sha256")==digest(run/"source/inventory.json") and controller.get("training_seed")==config["seed"] and controller.get("objective")==config.get("contact_objective")
    if config.get("c2_protocol_file"):
        protocol=Path(config["c2_protocol_file"]);protocol=protocol if protocol.is_absolute() else root/protocol
        checks["C2_study_protocol_archived"]=digest(source_root/"study_protocol.json")==digest(protocol)==config["c2_protocol_sha256"]==controller.get("study_protocol_sha256")
    if queue_manifest is not None:
        manifest=read(queue_manifest);jobs=[j for j in manifest["jobs"] if Path(j["output"]).resolve()==run.resolve()]
        checks["C2_formal_queue_identity"]=len(jobs)==1 and jobs[0]["method"]==config["method"] and jobs[0]["seed"]==config["seed"] and digest(jobs[0]["config"])==digest(run/"input_config.json")
        checks["C2_formal_protocol_frozen"]=digest(manifest["prospective_protocol_file"])==manifest["prospective_protocol_sha256"] and manifest.get("checkpoint_format_version")==103
    elif config.get("seed") in (4,5,6):
        checks["C2_formal_queue_identity"]=False
    try:
        coverage=actual_u_coverage(run,saved)
    except (OSError,KeyError,ValueError,TypeError) as error:
        coverage={"status":"FAILED_CHECKPOINT_PREFIX","error":repr(error)}
    checks["C2_actualu_recording_required_for103"]=controller.get("record_actual_u") is True
    checks["C2_checkpoint_linked_actualu_prefix_passed"]=coverage["status"]=="PASSED_CHECKPOINT_PREFIX"
    return checks,coverage


def prior_contract_checks(run,config,meta):
    root=Path(__file__).resolve().parents[1]
    checks={"prior_geometric_qualification_metadata_preserved":meta.get("qualified_contact_predictor") is False and meta.get("dynamics_gradient_qualified") is False and meta.get("physical_force_prediction") is False,
            "prior_fixed64_512_and_sharedP":meta["basis"]["training_directions"]==64 and meta["basis"]["evaluation_directions"]==512 and meta["penetration_prior"]["allowance_m"]==.003 and meta["penetration_prior"]["normalization_m"]==.001 and meta["penetration_prior"]["phase_masked"] is False,
            "prior_external_dt_once":meta.get("dt_applied_inside_module") is False and meta.get("caller_dt_s")==.02}
    if config["method"]=="P_ONLY":
        wrapper=root/"src/penalty_study/penalty_only_objective.py"
        checks.update(P_ONLY_wrapper_hash=meta.get("penalty_wrapper_sha256")==digest(wrapper)=="cfa53515fb7dc12ae49a2aec2d9ca37603061c1752fc8875f879cf5ea235c914",
            P_ONLY_direct_sharedP_formula=meta.get("formula")=="-.1*penetration_prior" and meta.get("support_coefficient")==0. and meta.get("penetration_coefficient")==.1,
            P_ONLY_support0_baseA2support1=meta["scoring"]["support_coefficient"]==0. and meta["base_A2_scoring"]["support_coefficient"]==1.)
    else:
        checks["A2_shared_formula"]=config["method"]=="A2" and meta.get("formula")=="phase_weight * support_match - .1*penetration_prior" and meta["scoring"]["support_coefficient"]==1.
    return checks


def audit(run, iteration, queue_manifest=None):
    torch.set_num_threads(2)
    attempt = run / "compiled"
    path = attempt / "train" / f"model_{iteration}.pt"
    sidecar = read(path.with_suffix(".meta.json"))
    inventory = read(run / "source/inventory.json")
    config = read(run / "input_config.json")
    effective = read(attempt / "effective_training_config.json")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    checks = {}
    c2_coverage = {"status":"PENDING_UNMEASURED","scope":"No full-batch actual-input claim without explicit measured evidence"}

    def check(name, passed):
        checks[name] = bool(passed)

    check("sha_matches_completed_sidecar", digest(path) == sidecar["checkpoint_sha256"])
    index = [json.loads(line) for line in (attempt / "checkpoint_index.jsonl").read_text(encoding="utf-8").splitlines()]
    check("completed_sidecar_is_in_index", sidecar in index)
    check("format_and_task", saved["format_version"] == 103 and saved["config"]["task"] == "hand2-pen-spin")
    check("iteration", saved["iteration"] == iteration == sidecar["iteration"])
    check("config_matches_frozen_effective_config", json.loads(json.dumps(saved["config"])) == effective)
    check("all_module_tensors_finite", finite(saved["modules"]))
    check("all_optimizer_tensors_finite", finite(saved["optimizers"]))
    check("torque_normalizer_finite", finite(saved["torque_norm"]))
    check("reference_signature_present", bool(saved.get("reference_signature")))
    check("CPU_audit_no_CUDA_initialization", os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not torch.cuda.is_initialized())
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
    if config.get("paired_protocol_file"):
        check("paired_protocol_unchanged", digest(frozen_protocol) == digest(run.parents[1] / config["paired_protocol_file"]))
    elif config.get("seed") in (4,5,6):
        check("paired_protocol_required_for_formal",False)
    provenance = read(run / "reference/provenance.json")["frozen_evaluation_reference"]
    reference = run / "reference/frozen_evaluation_reference.npz"
    check("frozen_evaluation_reference_unchanged", digest(reference) == provenance["npz_sha256"] == digest(provenance["source"]))
    check("nominal_collision_model_hash", digest(attempt/"nominal_model.mjb")=="24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd")
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
    c2_checks, c2_coverage = c2_contract_checks(run, saved, config, iteration, queue_manifest)
    checks.update(c2_checks)
    if config.get("contact_objective"):
        checks.update(prior_contract_checks(run, config, objective_meta))
    else:
        check("A1L_no_geometric_prior_constructed", config["method"]=="A1L" and not (attempt/"contact_objective_metadata.json").exists())
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
        "checkpoint_format":103,"C2_actual_u_evidence":c2_coverage,"exact_resume_supported":False,
        "paired_protocol_scope":"frozen" if config.get("paired_protocol_file") else "NOT_PROVIDED_ENGINEERING_ONLY",
        "phase": "world-model pretraining; actor/critic have not updated" if iteration <= effective["wm_only_iters"] else "policy training",
    }, complete_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queue-manifest", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace an audit: {args.output_dir}")
    result, metrics = audit(args.run.resolve(strict=True), args.iteration, args.queue_manifest)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_snapshot.jsonl").write_bytes(metrics)
    coverage=result["C2_actual_u_evidence"]
    if coverage.get("status")=="PASSED_CHECKPOINT_PREFIX":
        prefix=Path(coverage["source_path"]).read_bytes()[:coverage["prefix_byte_count"]]
        if hashlib.sha256(prefix).hexdigest()!=coverage["prefix_sha256"]:raise ValueError("Previously audited actualu prefix changed")
        (args.output_dir/"actualu_prefix_snapshot.jsonl").write_bytes(prefix)
    shutil.copy2(__file__, args.output_dir / Path(__file__).name)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
