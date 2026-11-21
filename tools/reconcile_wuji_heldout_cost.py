"""Reconcile the four completed held-out ledgers. Reads JSON only. All charges are seconds."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


S5_ATTEMPT_SHA = "ca25dde1b4ee796b0273ccd1ef8bcdbb8c1db509659cfe11587ad289509eaadf"
S5_START = "2026-11-21T07:14:29.624792+00:00"
S5_MAX_S = 4.34


def require(condition, message):
    if not condition:
        raise ValueError(message)


def host_path(name):
    if os.name == "nt" and name.startswith("/mnt/") and name[6:7] == "/":
        return Path(name[5] + ":/" + name[7:])
    if os.name == "nt" and name.startswith("/"):
        return Path(r"\\wsl.localhost\Ubuntu-24.04") / name[1:]
    return Path(name)


@lru_cache(maxsize=512)
def read(path, sha=None):
    raw = path.read_bytes()
    if sha:
        require(hashlib.sha256(raw).hexdigest() == sha, f"Changed input: {path}")
    return json.loads(raw)


def pinned(pin):
    return read(host_path(pin["path"]), pin["sha256"])


def bounds(entries, fee=0., *, card=False):
    rows = [e for e in entries if not card or e["card"]]
    known = fee + math.fsum(e["known_wall_s"] for e in rows)
    hi = known
    for e in rows:
        if e["unknown_cost"]:
            require((e["key"], e["start"], e["card"], e["known_wall_s"]) ==
                    ("subject_5_cert_cpu", S5_START, False, 0),
                    f"Unbounded new interruption requires separate evidence: {e['key']}")
            hi += S5_MAX_S
    return [known, hi]


def plus(a, b):
    return [x+y for x,y in zip(a,b)]


def scaled(interval, denominator):
    return [x / denominator for x in interval] if denominator else None


def first_verified(common, work, setup, pre):
    for i, entry in enumerate(work):
        if (entry.get("outcome") or {}).get("status") == "passed":
            prefix = common + work[:i+1]
            wall = bounds(prefix, setup)
            return dict(subject=entry["subject"], steps=entry["steps"], target_wall_s=wall,
                        with_pretraining_wall_s=plus(pre,wall),
                        with_pretraining_gpu_s=plus(pre,bounds(prefix,card=True)), receipt=entry["receipt"])
    return None


def audit_jobs(root, entries):
    """Match every job attempt to its ledger, including unsuccessful attempts retained on resume."""
    rows = {(e["key"], e["start"]): e for e in entries if "start" in e}
    require(len(rows) == sum("start" in e for e in entries), "Duplicate job identifier")
    for path in sorted((root / "jobs").glob("*/attempt_*.json")):
        j = read(path)
        e = rows.pop((j["key"], j["start"]))
        require(all(j[k] == e[k] for k in ("subject","kind","card","steps"))
                and j.get("elapsed_s") == (None if e["unknown_cost"] else e["known_wall_s"])
                and j["success"] == e["successful_job"] and j.get("receipt") == e["receipt"],
                f"Ledger differs from attempted job: {path}")
        if e["kind"] in ("evaluate","disturbance") and e["successful_job"]:
            report = pinned(e["receipt"])
            if e["kind"] == "evaluate":
                require(report["status"] == e["outcome"]["status"], f"Changed evaluation: {path}")
                if report["status"] == "passed":
                    require(e["outcome"]["full_repeated"] and report["force_validation"]["validated"]
                            and len(report["realizations"]) == 3
                            and all(r["numerical_pass"] for r in report["realizations"]), f"Incomplete pass: {path}")
            else:
                require(report["assigned_summary"] == e["outcome"]["assigned_summary"], f"Changed stress: {path}")
    require(not rows, f"Ledger entries with missing attempt files: {list(rows)}")


def endpoint(row, step, budget, assigned):
    hits = [e["status"] for e in row["scored"] if e["steps"] == step]
    require(len(hits) <= 1, "Multiple scores for one subject/endpoint")
    if not assigned:
        require(not hits, "Unregistered zero-shot score")
        return dict(assigned=False, scored=False, status="not_registered")
    status = row["zero_shot_status" if step == 0 else "final_budget_status"]
    if hits:
        status = hits[0]
    if status is None:
        if row["source_status"] == "unscorable":
            status = "unscorable_source"
        else:
            status = row.get("adaptation_status") or (row.get("adaptation_stop") or {}).get("classification")
    require(status is not None, f"Unaccounted completed endpoint {step}/{budget}")
    return dict(assigned=True, scored=bool(hits), status=status)


def exposure(row, endpoint_state, events, root, subject, method, step, warm_layout):
    if not endpoint_state["assigned"] or not endpoint_state["scored"]:
        return dict(status=endpoint_state["status"], executed=False)
    jobs = [e for e in events if e["kind"] == "disturbance" and e["subject"] == subject
            and e["steps"] == step and e["successful_job"]]
    if jobs:
        require(len(jobs) == 1, "Repeated successful stress job")
        event = jobs[0]
        report = pinned(event["receipt"])
        nominal, = (e for e in events if e["kind"] == "evaluate" and e["subject"] == subject
                    and e["steps"] == step and e["successful_job"])
        binding = read(host_path(event["receipt"]["path"]).parent / "heldout_pairing.json")
        require(binding["heldout_report_sha256"] == nominal["receipt"]["sha256"] and
                binding["stress_report_sha256"] == event["receipt"]["sha256"], "Mismatched disturbance pairing")
        cases = []
        for r in report["runs"]:
            state = "unexposed" if not r["delivered_pulse_substeps"] else "partial"
            cases.append(dict(mode=r["mode"], direction=r["direction"],
                              exposure="full" if r["fully_exposed"] else state,
                              delivered_substeps=r["delivered_pulse_substeps"], recovery=r["recovery_state"]))
        require(len(cases) == 4 and {(r["mode"], r["direction"]) for r in cases} ==
                {(m,d) for m in ("open","closed") for d in (-1,1)}, "Incomplete signed disturbance pair")
        return dict(status="executed", executed=True, receipt=event["receipt"], cases=cases)
    name = f"disturb_{subject}_not_exposed.json" if warm_layout else f"disturb_{subject}_{method}_{step}.json"
    marker = root / name
    if marker.exists():
        skip = read(marker)
        require(skip["assigned"], f"Inconsistent stress eligibility: {marker}")
        return dict(status=skip.get("reason") or skip["exposure"], executed=False, receipt=str(marker))
    stop = row.get("adaptation_stop")
    require(stop and stop["step"] <= step and not stop["actor_usable"], "Missing scheduled stress disposition")
    return dict(status="actor_unavailable_after_" + stop["classification"], executed=False)


def method_result(name, model, inherited, own, common, budget, primary_subject, fee, export, locations):
    """inherited/own are disjoint in this hypothetical deployment, not across deployments."""
    events = inherited + own
    work = [e for e in events if e["kind"] != "disturbance"]
    batch = common + work
    wall = bounds(batch, fee+export)
    gpu = bounds(batch, card=True)
    pre = model["pretraining_exclusive_card_s_range"]
    require(all(x is not None for x in pre), "Unbounded development interval")
    require(model["target_branches_accounted"] and math.isclose(wall[0], model["target_end_to_end_known_s"])
            and math.isclose(gpu[0]/3600, model["target_gpu_hours_known"]), f"Standalone charges differ: {name}")
    subjects = {}
    for s, row in model["subjects"].items():
        cases = {}
        for label, step in (("zero_shot",0), ("final_budget",budget)):
            state = endpoint(row, step, budget, name != "cold_single" or step != 0)
            root, method, warm_layout, reused = locations[step]
            stress = exposure(row, state, events, root, s, method, step, warm_layout)
            cases[label] = dict(**state, score_reused=bool(reused and state["scored"]),
                                disturbance=dict(**stress, result_reused=bool(reused and stress["executed"])))
        subject_rows = [e for e in batch if e["subject"] == s]
        subjects[s] = dict(**cases, ever_verified=row["qualified"], adapted_steps=row["maximum_observed_trained_step"],
                           role="primary" if s == primary_subject else "supplementary",
                           source_status=row["source_status"], numerical_stop=(row.get("adaptation_stop") or {}).get("classification"),
                           first_verified_single_input=first_verified([e for e in common if e["subject"] == s],
                               [e for e in work if e["subject"] == s], fee+export, pre),
                           target_job_wall_s=bounds(subject_rows, export/len(model["subjects"])),
                           target_job_gpu_s=bounds(subject_rows, card=True))
    endpoints = {}
    for label in ("zero_shot","final_budget"):
        endpoints[label] = {}
        for group in ("primary","supplementary","all"):
            subset = [s[label] for s in subjects.values() if group == "all" or s["role"] == group]
            executed = [x["disturbance"] for x in subset if x["disturbance"]["executed"]]
            cases = [c for x in executed for c in x["cases"]]
            endpoints[label][group] = dict(assigned=sum(x["assigned"] for x in subset),
                                          scored=sum(x["scored"] for x in subset),
                                          verified=sum(x["status"] == "passed" for x in subset),
                                          stress_executed_endpoints=len(executed),
                                          stress_cases_by_exposure={k:sum(c["exposure"] == k for c in cases)
                                              for k in ("unexposed","partial","full")})
    passed = sum(s["ever_verified"] for s in subjects.values())
    require(passed == model["ever_verified"], "Mismatched verified denominator")
    return dict(subjects=subjects, endpoints=endpoints, ever_verified=passed,
                first_batch_verified=first_verified(common,work,fee+export,pre),
                teacher_check_or_search_cpu_s=fee, model_export_cpu_s=export, pretraining_card_s=pre,
                target_wall_s=wall,target_gpu_s=gpu, with_pretraining_wall_s=plus(wall,pre),
                with_pretraining_gpu_s=plus(gpu,pre),
                with_pretraining_wall_s_per_assignment=scaled(plus(wall,pre),len(subjects)),
                target_gpu_h_per_verified=scaled(gpu,3600*passed),
                with_pretraining_gpu_h_per_verified=scaled(plus(gpu,pre),3600*passed),
                separate_stress_card_s=bounds([e for e in events if e["kind"]=="disturbance"],card=True))


def reconcile(runs):
    primary = runs / "wuji_heldout_formal_v1"
    warm = runs / "wuji_heldout_warm_formal_v1"
    recovery = runs / "wuji_heldout_recovery_formal_v1"
    paired = runs / "wuji_heldout_recovered_warm_formal_v1"
    closed = [read(directory / "complete.json") for directory in (primary,warm,recovery,paired)]
    inputs = pinned(closed[3]["inputs"])
    p, w, r = [pinned(inputs[k]) for k in ("primary_report","final_warm_report","recovered_control_report")]
    z = pinned(closed[3]["report"])
    ids = read(paired / "activation.json", closed[3]["activation_sha256"])["identities"]
    require(closed[0]["selection_sha"] == p["selection_sha256"] == ids["parent_final"] and
            closed[0]["registered_stages_sha"] == r["registered_stages_sha256"] ==
            p["registered_stages_sha256"] == ids["source_stage_digest"] and
            closed[1]["supplement_sha256"] == ids["warm_final"] and
            closed[2]["recovery_sha256"] == r["recovery_selection_sha256"] == ids["recovery_selection"],
            "Completion records do not refer to the same registrations")
    plan = read(primary / "final_selection.json", ids["parent_final"])
    stages = read(primary / "registered_stages.json", ids["source_stage_digest"])["subjects"]
    register = read(recovery / "recovery_selection.json", ids["recovery_selection"])
    warm_register = read(warm / "supplement.json", ids["warm_final"])
    attempt = read(primary / "jobs/subject_5_cert_cpu/attempt_001.json", S5_ATTEMPT_SHA)
    require(attempt["start"] == S5_START and attempt["elapsed_s"] is None, "CPU interruption changed")
    engine = recovery / "recovery_engine"
    ledgers = dict(primary=p["entries"], final_warm=w["unique_entries"], recovery=r["unique_recovery_jobs"],
                   recovered_warm=z["unique_entries"])
    for directory, entries in zip((primary,warm,engine,paired),ledgers.values()):
        audit_jobs(directory,entries)
    common = [e for e in p["entries"] if e["method"] is None]
    for e in common:
        if e["kind"] == "source_visibility_review":
            require(stages[e["subject"]]["source_review_active_seconds"] == e["known_wall_s"],
                    f"Source review clock differs: {e['subject']}")
    budget = plan["budget"]
    arms = {}

    def add(name, model, source, model_events, directory, inherited=None, export=0., inherited_directory=None):
        inherited = inherited or []
        own = [e for e in model_events if e["method"] == source]
        zero_root = inherited_directory or directory
        locations = {0:(zero_root,"shared_wm" if inherited_directory else source,False,inherited_directory is not None),
                     budget:(directory,source,directory in (paired,warm),False)}
        arms[name] = method_result(name,model,inherited,own,common,budget,plan["protocol"]["primary"],
                                   model["teacher_inspection_cpu_s_once"],export,locations)

    for name,model in p["methods"].items():
        add(name,model,name,p["entries"],primary)
    old_zero = [e for e in p["entries"] if e["method"] == "shared_wm" and e["steps"] == 0]
    add(w["method"],w["standalone"],w["method"],w["unique_entries"],warm,old_zero,
        warm_register["export_cpu_s"],primary)
    for name,data in r["recovery_methods"].items():
        add(name,data["standalone"],data["primary_method"],r["unique_recovery_jobs"],engine)
    recovered_zero = [e for e in r["unique_recovery_jobs"] if e["method"] == "shared_wm" and e["steps"] == 0]
    require([dict(e,method=z["method"]) for e in recovered_zero] == z["reused_zero_shot_entries"],
            "Paired zero-shot entries differ from their actual source")
    add(z["method"],z["standalone"],z["method"],z["unique_entries"],paired,recovered_zero,
        register["optional_model_preparation_s"],engine)
    fees = dict(original_teacher_checks=sum(t["inspection_cpu_wall_s"] for t in plan["teacher_health"].values()),
                final_warm_export=warm_register["export_cpu_s"], recovery_search=register["selection_cpu_wall_s"],
                paired_export=register["optional_model_preparation_s"])
    pre = plus(*(plan["upfront_exclusive_card_s_range"][m] for m in ("shared_mf","shared_wm")))
    all_jobs = [e for entries in ledgers.values() for e in entries]
    study_wall = bounds(all_jobs, sum(fees.values()))
    study_gpu = bounds(all_jobs, card=True)
    inclusive_wall = plus(study_wall,pre)
    development_lower = warm_register["development_trial_s_known"]
    development_upper = None if warm_register["development_unknown_costs"] else development_lower
    registered_sum = z["actual_costs"]["original_and_all_supplements_plus_distinct_teachers_s_range"]
    require(math.isclose(inclusive_wall[0],registered_sum[0]) and
            math.isclose(inclusive_wall[1]-S5_MAX_S,registered_sum[1]), "Incremental costs disagree with frozen ledger")
    return dict(scope="Completed held-out jobs include failed attempts and stress. The deployment comparison charges two distinct shared teacher streams. "
                "The separate full-trial figure also charges cold-arm development. Both exclude supervisor overhead, shared development setup and source acquisition before intake. "
                "GPU time measures card reservation. Each standalone method excludes stress. Costs are active job wall seconds, not elapsed calendar time.",
                completion_and_report_pins=dict(paired=closed[3],primary=inputs["primary_report"],
                    final_warm=inputs["final_warm_report"],recovery=inputs["recovered_control_report"]),
                interruption=dict(key=attempt["key"],attempt_sha256=S5_ATTEMPT_SHA,wall_s=[0.,S5_MAX_S],gpu_s=[0.,0.],
                    host_receipt_creation_utc="2026-11-21T07:14:38.5943802Z",
                    host_confirmed_exit_utc="2026-11-21T07:14:42.931Z",
                    host_chat="01a19795-7981-7031-9c95-4d5df44ca4fd",
                    exit_event="exec-2b8ac9c1-906e-4294-a0bb-17f892106617",
                    basis="Host file creation precedes the job timer; the host event confirms the owned process tree has exited."),
                methods=arms, deduplicated_study=dict(target_wall_s=study_wall,target_gpu_s=study_gpu,
                    one_time_cpu_s=fees,distinct_teacher_card_s=pre,with_teachers_wall_s=inclusive_wall,
                    with_teachers_gpu_s=plus(study_gpu,pre),
                    all_development_card_s=[development_lower,development_upper],
                    all_development_plus_target_wall_s=[development_lower+study_wall[0],
                        None if development_upper is None else development_upper+study_wall[1]],
                    all_development_plus_target_gpu_s=[development_lower+study_gpu[0],
                        None if development_upper is None else development_upper+study_gpu[1]],
                    separate_shared_development_setup_wall_s=plan.get("common_plan_snapshot_s"),
                    unique_ledger_charges={
                        k:dict(wall_s=bounds(v),gpu_s=bounds(v,card=True),
                               stress_wall_s=bounds([e for e in v if e["kind"]=="disturbance"]))
                        for k,v in ledgers.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=Path(__file__).resolve().parents[1] / "runs")
    args = parser.parse_args()
    data = reconcile(args.runs)
    output = args.runs / "wuji_heldout_final_comparison_v1/cost_and_exposure.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump(data,f,ensure_ascii=False,allow_nan=False,indent=2)
            f.write("\n")
        tmp.replace(output)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
