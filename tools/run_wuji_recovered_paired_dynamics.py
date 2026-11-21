"""Execute the preregistered recovered actor + paired world-model transfer arm under WSL."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch

import run_wuji_heldout_dynamics_supplement as warm
from heldout_wuji_cost import summarize
from run_wuji_heldout import Jobs, artifact, check, digest, now, read, store, verify_release


ARM = "recovery_shared_wm_warm_dynamics"
CONTROL = "recovery_shared_wm"
ACTIVATION = "7a75b33b39fbb87fa82370daa463c102e443a5108b719cd68d70a1bb4a379a10"
DECISION = "5e11962a838089b65b089752994f5d1415fb69baaee1b76645f71095d2b583c1"


def require_equal(*values):
    if any(v != values[0] for v in values[1:]):
        raise ValueError(f"Registered identities differ: {values}")


def frozen_json(path, sha):
    require_equal(digest(path), sha)
    return read(path)


def registration(args):
    # Read development provenance and protocol only. The target masks file is hashed, never parsed here.
    act = frozen_json(args.out / "activation.json", ACTIVATION)
    decision = frozen_json(args.recovery / "paired_execution_decision.json", DECISION)
    ids = act["identities"]
    require_equal(act["status"], "activated")
    require_equal(act["arm"], ARM)
    require_equal(act["paired_ablation"], CONTROL)
    require_equal(decision["identical_zero_shot_reused"], True)
    require_equal(act["policy_jobs_present_at_activation"], [])
    reg = frozen_json(args.recovery / "recovery_selection.json", ids["recovery_selection"])
    prior = frozen_json(args.warm_final / "supplement.json", ids["warm_final"])
    require_equal(Path(reg["root"]), args.recovery)
    require_equal(Path(prior["root"]), args.warm_final)
    require_equal(decision["preregistration_sha256"], ids["recovery_selection"])
    require_equal(reg["rule"]["sha256"], ids["rule"])
    require_equal(prior["protocol"]["sha256"], ids["warm_rule"])
    require_equal(reg["parent_selection"]["sha256"], prior["base_selection"]["sha256"], ids["parent_final"])
    require_equal(reg["optional_dynamics"]["sha256"], decision["paired_source_sha256"], ids["paired_source"])
    for pin in [reg["rule"], reg["actor_anchors"], *reg["code"], prior["protocol"], *prior["code"]]:
        check(pin)
    primary_file = check(reg["parent_selection"])
    primary = read(primary_file)
    original = primary_file.parent
    require_equal(Path(primary["root"]), original, Path(reg["original"]), Path(prior["original"]))
    require_equal(digest(original / "registered_stages.json"), ids["source_stage_digest"])
    require_equal(primary["steps"], reg["steps"], prior["schedule_steps"], decision["target_per_reference_steps"])
    require_equal(primary["budget"], reg["budget"], prior["reference_budget"], primary["steps"][-1])
    require_equal(primary["worlds"], prior["worlds"])
    candidate = reg["decisions"]["shared_wm"]
    src = read(check(reg["optional_dynamics"]))
    manifest_file = check(src["manifest"])
    manifest = read(manifest_file)
    require_equal(manifest["development_fixture"], False)
    require_equal(candidate["status"], "selected_latest_finite_actor")
    require_equal(src["status"], "available_same_paired_weights_and_scales")
    require_equal(src["sidecar_file_sha_validated"], True)
    require_equal(check(src["actor"]), check(candidate["selected_pin"]))
    require_equal(act["source_actor_sha256"], src["actor"]["sha256"], manifest["actor_sha256"], candidate["selected_sha256"])
    require_equal(act["source_steps"], decision["source_step"], candidate["selected_step"], src["required_step"], manifest["source_training_steps"])
    require_equal(act["checkpoint_set_id"], decision["source_set"], candidate["selected_checkpoint_set_id"], src["required_set_id"], manifest["source_set_id"])
    require_equal(manifest["source_sidecar_sha256"], src["only_permitted_source"]["sha256"], src["expected_sidecar_sha256"])
    check(src["only_permitted_source"])
    require_equal(manifest["deployment_recovery_anchor_sha256"], reg["actor_anchors"]["sha256"])
    require_equal(manifest["source_selection_sha256"], ids["parent_final"])
    require_equal(act["transfer_manifest_sha256"], src["manifest"]["sha256"])
    require_equal(act["transfer_payload_sha256"], manifest["payload_sha256"], src["payload"]["sha256"])
    require_equal(check(src["payload"]), manifest_file.parent / manifest["payload_file"])
    # Derive isolated storage from the registered recovery layout, independent of invocation cwd.
    bulk = Path(reg["bulk"]).parent / args.out.name
    protected = [Path(p[k]) for p in (primary, reg, prior) for k in ("root", "bulk")]
    if args.out in protected or bulk in protected:
        raise ValueError("Paired deployment output collides with an existing study")
    return SimpleNamespace(act=act, reg=reg, prior=prior, primary=primary, source=src, original=original, bulk=bulk)


def recovered_plan(base, reg):
    decision = reg["decisions"]["shared_wm"]
    return dict(base, teachers=dict(base["teachers"], shared_wm=decision["selected_pin"]),
                teacher_health={"shared_wm": dict(usable=True, status=decision["status"],
                    inspection_cpu_wall_s=decision["selection_cpu_wall_s"])},
                protocol=dict(base["protocol"], zero_shot=["shared_wm"], adaptation=["shared_wm"]))


def prepare(args, verified):
    ids = verified.act["identities"]
    original = verified.original
    done = read(original / "complete.json")
    require_equal(done["selection_sha"], ids["parent_final"])
    require_equal(done["registered_stages_sha"], ids["source_stage_digest"])
    require_equal(read(args.warm_final / "complete.json")["supplement_sha256"], ids["warm_final"])
    require_equal(read(args.recovery / "complete.json")["recovery_sha256"], ids["recovery_selection"])
    base, original_jobs = verify_release(SimpleNamespace(out=original, released_plan_sha256=ids["parent_final"]))
    stage_file = frozen_json(original / "registered_stages.json", ids["source_stage_digest"])
    require_equal(stage_file["final_selection_sha256"], ids["parent_final"])
    stages = stage_file["subjects"]
    require_equal(set(stages), {s["subject"] for s in base["subjects"]})
    for stage in stages.values():
        for key in ("contract", "source", "parquet", "evidence"):
            check(stage[key])
    recovery_file = args.recovery / "deployment_recovery_cost_and_outcomes.json"
    recovery_report = read(recovery_file)
    require_equal(recovery_report["recovery_selection_sha256"], ids["recovery_selection"])
    require_equal(recovery_report["registered_stages_sha256"], ids["source_stage_digest"])
    require_equal(recovery_report["recovery_methods"][CONTROL]["selected_step"], verified.act["source_steps"])
    original_report = read(original / "cost_and_outcomes.json")
    require_equal(original_report["selection_sha256"], ids["parent_final"])
    require_equal(original_report["registered_stages_sha256"], ids["source_stage_digest"])
    # Pin all comparisons, source and runner before the first target step or resume.
    inputs = dict(activation=ACTIVATION, decision=DECISION, bulk=str(verified.bulk),
                  implementation=artifact(__file__), primary_report=artifact(original / "cost_and_outcomes.json"),
                  final_warm_report=artifact(args.warm_final / "supplement_outcomes.json"),
                  recovered_control_report=artifact(recovery_file))
    record = args.out / "execution_inputs.json"
    if record.exists():
        require_equal(inputs, read(record))
    else:
        store(record, inputs)
    plan = recovered_plan(base, verified.reg)
    extra = dict(source_teacher=verified.source["actor"], dynamics_manifest=verified.source["manifest"],
                 primary_entries=original_report["entries"],
                 recovered_report=recovery_report, final_warm_report=read(check(inputs["final_warm_report"])),
                 export_cpu_s=verified.reg["optional_model_preparation_s"], activation=verified.act)
    jobs = Jobs(dict(plan, root=str(args.out), bulk=str(verified.bulk)))
    return extra, plan, original_jobs, jobs, stages


def fold_outcomes(extra, base, original, jobs, stages):
    prior = extra["recovered_report"]
    control = prior["recovery_methods"][CONTROL]["standalone"]
    common = [e for e in extra["primary_entries"] if e["method"] is None]
    zero = [dict(e, method=ARM) for e in prior["unique_recovery_jobs"]
            if e["method"] == "shared_wm" and e["steps"] == 0 and e["kind"] in ("evaluate", "disturbance")]
    unique = []
    for entry in jobs.entries():
        if entry["method"] != "shared_wm" or entry["steps"] not in base["steps"] or entry["kind"] not in ("adapt", "evaluate", "disturbance"):
            raise ValueError("Unexpected job in the recovered paired-model ledger")
        unique.append(dict(entry, method=ARM))
    export = [dict(subject=s, method=None, steps=None, card=False, kind="paired_dynamics_cpu_share",
                   known_wall_s=extra["export_cpu_s"] / len(stages), unknown_cost=False, outcome=None) for s in stages]
    upfront = base["upfront_exclusive_card_s"]["shared_wm"]
    bounds = base.get("upfront_exclusive_card_s_range", {}).get("shared_wm", [upfront, upfront])
    plan = dict(base, protocol=dict(base["protocol"], adaptation=[ARM], zero_shot=[ARM]),
                teacher_health={ARM: base["teacher_health"]["shared_wm"]}, upfront_exclusive_card_s={ARM: upfront},
                upfront_exclusive_card_s_range={ARM: bounds})
    result = summarize(plan, [*common, *export, *zero, *unique], {s: v["source_status"] for s, v in stages.items()})
    paired = result["methods"][ARM]
    comparison = {}
    for subject, row in paired["subjects"].items():
        comparison[subject] = []
        for step in [0, *base["steps"]]:
            a, scored_a = warm.stage_status(control["subjects"][subject], step)
            b, scored_b = warm.stage_status(row, step)
            if step == 0:
                require_equal((a, scored_a), (b, scored_b))
            comparison[subject].append(dict(target_steps=step, actor_only=a, actor_and_dynamics=b,
                                           both_actually_scored=scored_a and scored_b,
                                           same_evaluation_reused=step == 0 and scored_a and scored_b))
    added_wall = sum(e["known_wall_s"] for e in unique)
    added_gpu = sum(e["known_wall_s"] for e in unique if e["card"])
    previous = prior["actual_costs"]
    final_warm = extra["final_warm_report"]["actual_costs"]
    previous_range = previous["original_plus_recovery_plus_distinct_teachers_s_range"]
    inc = added_wall + final_warm["supplement_new_s_known"]
    return dict(method=ARM, paired_ablation=CONTROL, activation_sha256=ACTIVATION,
                source_teacher=extra["source_teacher"], source_model_manifest=extra["dynamics_manifest"],
                standalone=paired, actor_only_standalone=control, paired_stage_status=comparison,
                reused_zero_shot_reports=sum(e["kind"] == "evaluate" and bool(e.get("outcome")) for e in zero),
                reused_zero_shot_entries=zero, unique_entries=unique,
                actual_costs=dict(new_target_job_s_known=added_wall, new_target_gpu_s_known=added_gpu,
                    new_zero_shot_jobs=0, new_source_preparation_jobs=0, new_teacher_search_jobs=0,
                    new_dynamics_extraction_jobs=0, new_pretraining_jobs=0,
                    paired_extraction_already_charged_to_recovery_s=extra["export_cpu_s"],
                    prior_final_warm_incremental_s=final_warm["supplement_new_s_known"],
                    original_and_all_supplements_plus_distinct_teachers_s_range=[
                        val + inc if val is not None else None for val in previous_range],
                    unknown_costs=previous["unknown_costs"] or final_warm["unknown_costs"] or any(e["unknown_cost"] for e in unique),
                    scope="Job wall time includes failed attempts and stress. Study coordination is logged separately."))


def execute(args, verified):
    prepared = prepare(args, verified)
    extra, base, original, jobs, stages = prepared
    recovered = SimpleNamespace(root=args.recovery / "recovery_engine")
    completed_job = warm.completed_job

    def reuse(journal, key):
        if key.endswith("_shared_wm_eval_0"):
            return completed_job(recovered, key)
        if key.endswith(("_cert_cpu", "_cert_cuda0")):
            return completed_job(journal, key)
        raise ValueError(f"Unexpected baseline reuse: {key}")

    loaded = set()
    for e in extra["recovered_report"]["unique_recovery_jobs"]:
        if e["method"] != "shared_wm" or e["steps"] != 0 or not e["successful_job"]:
            continue
        subject = e["subject"]
        expected_key = f"{subject}_shared_wm_" + {"evaluate": "eval_0", "disturbance": "disturb_0"}[e["kind"]]
        require_equal(e["key"], expected_key)
        item = stages[subject]
        require_equal(e["receipt"], completed_job(recovered, expected_key)["receipt"])
        scored = read(check(e["receipt"]))
        require_equal(scored["checkpoint_sha256"], extra["source_teacher"]["sha256"])
        require_equal(scored["parquet_sha256"], item["parquet"]["sha256"])
        if e["kind"] == "evaluate":
            require_equal(scored["contract_sha256"], item["contract"]["sha256"])
            expected_certs = [completed_job(original, f"{subject}_cert_{name}")["receipt"] for name in ("cpu", "cuda0")]
            require_equal([c["sha256"] for c in scored["force_validation"]["certificates"]],
                          [c["sha256"] for c in expected_certs])
            if subject in loaded:
                raise ValueError("Duplicate recovered zero-shot success")
            loaded.add(subject)
        else:
            binding = read(check(e["receipt"]).parent / "heldout_pairing.json")
            nominal = completed_job(recovered, f"{subject}_shared_wm_eval_0")["receipt"]
            require_equal(binding["heldout_report_sha256"], nominal["sha256"])
            require_equal(binding["stress_report_sha256"], e["receipt"]["sha256"])
    require_equal(loaded, {s for s, item in stages.items() if item["source_status"] == "scorable"})
    # Isolated CLI process: use the immutable warm runner's exact schedule, job builders and numerical stops.
    def write_report(path, data):
        if path != jobs.root / "complete.json":
            store(path, data)

    with ExitStack() as ctx:
        for name, value in (("ARM", ARM), ("ready", lambda _: prepared), ("outcomes", fold_outcomes),
                            ("completed_job", reuse), ("store", write_report)):
            ctx.enter_context(patch.object(warm, name, value))
        opts = SimpleNamespace(mode=args.mode, supplement_sha256=ACTIVATION)
        warm.run(opts)
    if args.mode == "run":
        report = read(jobs.root / "supplement_outcomes.json")
        if not report["standalone"]["target_branches_accounted"]:
            raise ValueError("Unaccounted assigned target after execution")
        store(jobs.root / "complete.json", dict(finished=now(), activation_sha256=ACTIVATION,
              deployment_decision_sha256=DECISION, report=artifact(jobs.root / "supplement_outcomes.json"),
              inputs=artifact(jobs.root / "execution_inputs.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify-registration", "run", "report"])
    root = Path(__file__).resolve().parents[1] / "runs"
    for arg, name in (("out", "wuji_heldout_recovered_warm_formal_v1"), ("recovery", "wuji_heldout_recovery_formal_v1"),
                      ("warm-final", "wuji_heldout_warm_formal_v1")):
        parser.add_argument("--" + arg, type=lambda p: Path(p).resolve(), default=root / name)
    args = parser.parse_args()
    verified = registration(args)
    if args.mode == "verify-registration":
        print(json.dumps(dict(activation_sha256=ACTIVATION, deployment_decision_sha256=DECISION, arm=ARM,
              checkpoint_set_id=verified.act["checkpoint_set_id"], reference_steps=verified.primary["steps"],
              worlds=verified.primary["worlds"], target_scores_opened=0, bulk=str(verified.bulk))))
        return
    with (args.out / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.perf_counter()
        def child_seconds():
            return sum(read(p).get("elapsed_s") or 0. for p in (args.out / "jobs").glob("*/attempt_*.json"))
        prior_s = child_seconds()
        try:
            execute(args, verified)
        finally:
            elapsed = time.perf_counter() - start
            child_s = child_seconds() - prior_s
            with (args.out / "coordination.jsonl").open("a") as fp:
                fp.write(json.dumps(dict(utc=now(), mode=args.mode, elapsed_s=elapsed, new_job_s=child_s,
                                         coordinator_s=max(0., elapsed - child_s))) + "\n")


if __name__ == "__main__":
    main()
