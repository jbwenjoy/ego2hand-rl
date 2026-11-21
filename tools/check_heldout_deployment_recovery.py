"""Development-only CPU validation of the prospective deployment fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["JAX_PLATFORMS"]="cpu"

import numpy as np
from safetensors import safe_open


def digest(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def inspect_native_series(directory):
    """Independent oracle using native metadata and actual inference tensors. No rollout scores."""
    steps=sorted((int(p.stem.split("_")[-1]) for p in directory.glob("policy_*.safetensors")),reverse=True)
    if steps[:3]!=[639744,548352,456960]:
        raise ValueError("Expected the historical development single2 round 2 endpoints")
    source_fingerprint=None
    records=[]
    for step in steps:
        path=directory/f"policy_{step}.safetensors"
        pair=read(directory/f"paired_{step}.json")
        with safe_open(str(path),framework="numpy") as file:
            metadata=file.metadata()
            environment=json.loads(metadata["user.resolved_config_json"])
            assert environment["hybrid"] is False and len(environment["reference_shards"])==1
            assert environment["reference_shards"][0]["sequence"]=="subject4_G16_4"
            assert int(metadata["user.environment_steps"])==step and pair["environment_steps"]==step
            assert metadata["user.checkpoint_set_id"]==pair["checkpoint_set_id"]
            current=environment["source_fingerprint"]
            assert current==pair["source_fingerprint"] and (not source_fingerprint or current==source_fingerprint)
            source_fingerprint=current
            nonfinite=0
            for name in file.keys():
                data=file.get_tensor(name)
                if data.dtype.kind=="f":nonfinite+=int(data.size-np.count_nonzero(np.isfinite(data)))
            records.append(dict(step=step,actor_sha256=digest(path),checkpoint_set_id=pair["checkpoint_set_id"],
                                nonfinite=nonfinite,finite=nonfinite==0))
    assert next(x["step"] for x in records if x["finite"])==456960
    return records


def independent_prefix(directory,step):
    timeline=[json.loads(line) for line in (directory.parent/"timing.jsonl").read_text().splitlines()]
    target_state=directory/f"state_{step}.safetensors"
    native=[(i,r) for i,r in enumerate(timeline) if r.get("event")=="native_checkpoint_end" and r["state"]==str(target_state)]
    if len(native)!=1:raise ValueError("Missing unique source checkpoint emission")
    i=native[0][0]
    paired_end=next(r for r in timeline[i+1:] if r["event"]=="sidecar_checkpoint_end")
    calibrated=next(r for r in timeline[i+1:] if r["event"]=="checked_training_budget")
    assert calibrated["step"]==step and paired_end["wall_s"]<=calibrated["wall_s"]
    total=read(directory.parent.with_suffix(".process.json"))["elapsed_s"]
    assert 0<paired_end["wall_s"]<total
    return dict(step=step,within_segment_native_elapsed_at_saved_set_s=paired_end["wall_s"],
                within_segment_budget_check_s=calibrated["wall_s"],full_segment_process_s=total)


def check_selector(args,native):
    import heldout_recovery_selection as select
    from flash_chord.training import checkpoint as native_checkpoint
    p=args.development_checkpoints
    final=p/"policy_639744.safetensors";finite=p/"policy_456960.safetensors"
    initial=p.parent.parent/"initial_native_state"
    options=dict(method="shared_mf",expected_init=initial,allow_development_fixture=True)
    observed=[]
    read_source=select._read
    def no_scores(path):
        name=Path(path).name
        assert name.startswith("paired_") or name in ("initial.json","shared_initial_loader.json"),name
        observed.append(name)
        # Deliberately misleading synthetic fields are ignored.
        record=read_source(path)
        return {**record,"mock_quality_preference":1. if name.startswith("paired_182784") else 0.}
    with patch.object(select,"_read",no_scores):
        chosen=select.select_recovery(final,**options)
    assert chosen["status"]=="selected_latest_finite_actor"
    assert chosen["selected_step"]==456960 and Path(chosen["selected_actor"]).resolve()==finite.resolve()
    assert chosen["fixed_final"]["usable"] is False
    inspected=[r["step"] for r in chosen["inspected"]]
    assert inspected==sorted(inspected,reverse=True) and inspected[-1]==456960
    assert set(inspected)|{639744}=={639744,548352,456960}
    skipped=select.select_recovery(finite,**options)
    assert skipped["status"]=="not_triggered_finite_final" and skipped["selected_actor"] is None
    assert skipped["fixed_final"]["usable"] and [r["step"] for r in skipped["inspected"]] in ([],[456960])
    original_health=select.inspect_actor
    def all_invalid(file):
        # Retain the original step, checksum and set ID. Mock only inference availability.
        record=original_health(file)
        return {**record,"usable":False,"status":"nonfinite_frozen_teacher"}
    with patch.object(select,"inspect_actor",all_invalid):
        none=select.select_recovery(final,**options)
    assert none["status"]=="no_finite_actor_on_lineage" and none["selected_actor"] is None
    assert none["inspected"] and not any(row["usable"] for row in none["inspected"])
    assert all(row["step"]>0 for row in none["inspected"]), "Canonical step zero is not a trained rescue candidate"
    with tempfile.TemporaryDirectory(prefix="wrong_initial_bank_") as tmp:
        try:select.select_recovery(final,method="shared_mf",expected_init=Path(tmp),allow_development_fixture=True)
        except (OSError,ValueError,RuntimeError) as exc:wrong_init=str(exc)
        else:raise AssertionError("An unrelated canonical initial bank was accepted")
    try:select.select_recovery(final,method="shared_mf",expected_init=initial)
    except (OSError,ValueError,RuntimeError) as exc:wrong_stream=str(exc)
    else:raise AssertionError("Single-reference development actor passed production shared-stream validation")
    native_read=native_checkpoint.read_checkpoint_metadata
    def different_physics(path):
        meta=native_read(path)
        if Path(path)==finite:
            cfg=json.loads(meta["resolved_config_json"])
            cfg["physics_identity"]={**cfg["physics_identity"],"fixture_invalid_solver":"wrong_contact"}
            return {**meta,"resolved_config_json":json.dumps(cfg)}
        return meta
    with patch.object(native_checkpoint,"read_checkpoint_metadata",different_physics):
        try:select.select_recovery(final,**options)
        except (OSError,ValueError,RuntimeError) as exc:wrong_physics=str(exc)
        else:raise AssertionError("A finite actor with foreign physics metadata was selected")
    by_step={r["step"]:r for r in native}
    assert chosen["fixed_final"]["checkpoint_sha256"]==by_step[639744]["actor_sha256"]
    return dict(latest_finite_dev=chosen,already_finite_no_duplicate=skipped,
                all_invalid_simulated=none,noncanonical_initialization_rejected=wrong_init,
                single_reference_rejected_as_production_shared_method=wrong_stream,
                false_physics_rejected=wrong_physics,scored_evaluation_file_reads=0,
                fixture_provenance_records_inspected=list(dict.fromkeys(observed)))


def schedule_plan():
    people=["fixture_primary","fixture_A","fixture_B"]
    return dict(protocol=dict(primary=people[0],supplementary=people[1:],zero_shot=["shared_mf","shared_wm"],
                              adaptation=["cold_single","shared_mf","shared_wm"]),
                subjects=[{"subject":s} for s in people],steps=[24,48,72,96],budget=96,
                upfront_exclusive_card_s={"cold_single":0.,"shared_mf":100.,"shared_wm":200.},
                upfront_exclusive_card_s_range={"cold_single":[0.,0.],"shared_mf":[100.,100.],"shared_wm":[200.,200.]})


def check_schedule():
    from run_wuji_heldout_deployment_recovery import recovery_schedule
    plan=schedule_plan()
    people=[x["subject"] for x in plan["subjects"]]
    frozen={s:dict(source_status="scorable") for s in people}
    assert not list(recovery_schedule(plan,frozen,[]))
    counts={}
    for eligible in (["shared_mf"],["shared_wm"],["shared_mf","shared_wm"]):
        jobs=list(recovery_schedule(plan,frozen,eligible))
        assert all(job[2] in eligible for job in jobs)
        assert not any(job[0]=="certify" for job in jobs)
        # No duplicate policy scoring and target optimizer budgets independent of recovery depth.
        assert len(set(jobs))==len(jobs)
        adaptations=[j for j in jobs if j[0]=="adapt"]
        scores=[j for j in jobs if j[0]=="evaluate"]
        for method in eligible:
            for subject in people:
                assert [j[3] for j in adaptations if j[1:3]==(subject,method)]==plan["steps"]
                assert [j[3] for j in scores if j[1:3]==(subject,method)]==[0,*plan["steps"]]
        zero=[i for i,j in enumerate(jobs) if j[0]=="evaluate" and j[3]==0]
        first_train=next(i for i,j in enumerate(jobs) if j[0]=="adapt")
        assert len(zero)==len(eligible)*len(people) and max(zero)<first_train
        counts["+".join(eligible)]={"adapt":len(adaptations),"evaluate":len(scores),"zero_shot":len(zero)}
    missing=dict(frozen);missing[people[-1]]={"source_status":"unscorable"}
    assert all(j[1]!=people[-1] for j in recovery_schedule(plan,missing,["shared_mf","shared_wm"]))
    return counts


def check_costs():
    from heldout_wuji_cost import summarize
    from run_wuji_heldout_deployment_recovery import recovery_costs
    base=schedule_plan()
    methods=("shared_mf","shared_wm")
    base["teacher_health"]={m:dict(usable=False,status="nonfinite_frozen_teacher",inspection_cpu_wall_s=.5) for m in methods}
    people=[r["subject"] for r in base["subjects"]]
    frozen={s:dict(source_status="scorable") for s in people}
    def row(subject,method,kind,steps,s,passed=None):
        return dict(subject=subject,method=method,kind=kind,steps=steps,card=method is not None,
                    known_wall_s=s,unknown_cost=False,
                    outcome=(None if passed is None else dict(status="passed" if passed else "failed",full_repeated=passed)))
    original=[row(s,None,"source",None,1.) for s in people]
    original.append(row(people[0],"cold_single","adapt",24,9.))
    baseline=summarize(base,original,{s:"scorable" for s in people})
    baseline["entries"]=original
    old=json.dumps(baseline,sort_keys=True)
    decisions={"shared_mf":dict(status="selected_latest_finite_actor",selected_step=456960,selection_cpu_wall_s=.75),
               "shared_wm":dict(status="no_finite_actor_on_lineage",selected_step=None,selection_cpu_wall_s=1.25)}
    extra=[]
    for s in people:
        extra.append(row(s,"shared_mf","evaluate",0,1.,False))
        for n in base["steps"]:
            extra.append(row(s,"shared_mf","adapt",n,4.))
            extra.append(row(s,"shared_mf","evaluate",n,2.,n==base["budget"]))
    result=recovery_costs(base,baseline,base,{"decisions":decisions},extra,frozen)
    assert json.dumps(baseline,sort_keys=True)==old
    actual=result["actual_costs"]
    assert actual["new_pretraining_s"]==0
    assert actual["full_original_teacher_development_s_once"]==300.
    assert actual["teacher_selection_cpu_s"]==2.
    assert actual["extra_recovery_job_s_known"]==75.
    assert actual["original_plus_recovery_plus_distinct_teachers_s_known"]==12.+1.+2.+75.+300.
    for m in methods:
        rec=result["recovery_methods"]["recovery_"+m];stand=rec["standalone"]
        assert rec["original_primary_outcome"]==baseline["methods"][m]
        assert rec["original_primary_outcome"]["assigned_but_not_scored_invalid_teacher"]==3
        assert rec["original_primary_outcome"]["ever_verified"]==0
        assert rec["assigned"]==3 and stand["assigned"]==3 and stand["pretraining_exclusive_card_s"]==base["upfront_exclusive_card_s"][m]
        assert stand["ever_verified"]==(3 if m=="shared_mf" else 0)
    # A finite terminal is already counted by the baseline, not re-evaluated or re-trained.
    finite={**decisions,"shared_mf":dict(status="not_triggered_finite_final",selected_step=None,selection_cpu_wall_s=.1)}
    rep=recovery_costs(base,baseline,base,{"decisions":finite},[],frozen)
    assert rep["skipped_already_usable_final"]==["shared_mf"] and "recovery_shared_mf" not in rep["recovery_methods"]
    assert rep["actual_costs"]["extra_recovery_job_s_known"]==0
    assert rep["recovery_methods"]["recovery_shared_wm"]["standalone"]["assigned_but_not_scored_invalid_teacher"]==3
    return dict(scope="Synthetic seconds and pass labels exercise accounting only.",original_failed_assignments=6,
                fallback_ever_verified_fixture=3,uncounted_finite_terminal_duplicates=0,
                production_sunk_work_rule_verified=actual,no_rescue_target_jobs_when_no_candidate=True)


def check_supplement(args,rows,prefix):
    return dict(selector=check_selector(args,rows),schedule_counts=check_schedule(),cost_cases=check_costs())


def main(args):
    if args.development_checkpoints.parent.name!="r2_single2":
        raise ValueError("Development-only regression cannot open another experiment")
    clock=time.perf_counter()
    rows=inspect_native_series(args.development_checkpoints)
    prefix=independent_prefix(args.development_checkpoints,456960)
    result=dict(scope="Development S4 checkpoint tensors + synthetic method assignments/costs. No new deployment or held-out score is observed.",
                known_native_series=rows,prefix_oracle=prefix,used_gpu=False,opened_heldout_sources=False)
    result.update(check_supplement(args,rows,prefix))
    result["cpu_seconds"]=time.perf_counter()-clock
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with args.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(selected_step=456960,cpu_seconds=result["cpu_seconds"])))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=lambda s:Path(s).resolve(),required=True)
    p.add_argument("--development-checkpoints",type=lambda s:Path(s).resolve(),required=True)
    p.add_argument("--output",type=lambda s:Path(s).resolve(),required=True)
    a=p.parse_args();main(a)
