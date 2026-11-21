"""Freeze and run a separate deployment-time recovery study; never replace a primary teacher."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import fcntl
import json
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import run_wuji_heldout as primary
from benchmark_wuji_parallel import environment
from heldout_recovery_selection import select_recovery
from heldout_wuji_cost import summarize
from run_wuji_heldout import Jobs,artifact,check,comparison_schedule,digest,now,read,store,verify_release
from run_wuji_heldout_dynamics_supplement import completed_job


MODELS=("shared_mf","shared_wm")


def recovery_schedule(base,frozen,eligible_methods):
    models=[m for m in MODELS if m in eligible_methods]
    if not models:return []
    plan=dict(base,protocol=dict(base["protocol"],adaptation=models,zero_shot=models),teacher_health={})
    # Calibration is reused externally. There is no new cold arm in this supplement.
    return [job for job in comparison_schedule(plan,frozen) if job[0]!="certify"]


def register(args):
    original=Path(args.original);source_plan=original/"final_selection.json"
    if digest(source_plan)!=args.final_selection_sha256:raise ValueError("Wrong original final selection")
    if ((original/"registered_stages.json").exists() or (original/"source_jobs").exists()
            or any((original/"jobs").glob("*_prepare/attempt_*.json"))):
        raise ValueError("Freeze deployment recovery before any held-out evaluation source is opened")
    base,_=verify_release(SimpleNamespace(out=original,released_plan_sha256=args.final_selection_sha256))
    repo=Path(base["repo"]);cfg=repo/"configs/hocap_heldout_deployment_recovery_v1.json"
    if read(cfg)["base_rule_sha256"]!=base["protocol_file"]["sha256"]:raise ValueError("Wrong parent study protocol")
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    decisions={}
    for m in MODELS:
        d=select_recovery(check(base["teachers"][m]),method=m,expected_init=Path(base["canonical_init"]))
        observed=base["teacher_health"][m]
        if (observed["usable"]!=d["fixed_final"]["usable"] or observed["checkpoint_sha256"]!=d["fixed_final"]["checkpoint_sha256"]):
            raise ValueError("Canonical v1 final and recovery preflight disagree")
        if d["selected_actor"]:
            d["selected_pin"]=artifact(d["selected_actor"])
            if d["selected_pin"]["sha256"]!=d["selected_sha256"]:raise ValueError("Selected actor changed")
        decisions[m]=d
    record=dict(version="wuji_heldout_deployment_recovery_v1",registered_at=now(),root=str(args.out),bulk=str(args.bulk),
       original=str(original),parent_selection=artifact(source_plan),rule=artifact(cfg),decisions=decisions,
       selection_cpu_wall_s=sum(d["selection_cpu_wall_s"] for d in decisions.values()),budget=base["budget"],steps=base["steps"],
       code=[artifact(repo/p) for p in ("tools/heldout_recovery_selection.py","tools/run_wuji_heldout_deployment_recovery.py",
           "tools/heldout_teacher_health.py","tools/heldout_adaptation_health.py","tools/heldout_numeric_evaluation.py",
           "tools/export_wuji_heldout_dynamics.py")])
    # These are identities of optional future model-transfer inputs, not a second actor selection.
    anchors=args.out/"recovery_actor_anchors.json"
    store(anchors,dict(version="wuji_heldout_deployment_recovery_actor_anchors_v1",
                       parent_selection=record["parent_selection"],rule=record["rule"],decisions=decisions))
    record["actor_anchors"]=artifact(anchors)
    warm_info=decisions["shared_wm"].get("paired_dynamics")
    if warm_info:
        clock=time.perf_counter()
        source=warm_info.get("path")
        present=bool(source and Path(source).is_file() and Path(warm_info["state"]).is_file())
        observed=digest(Path(source)) if present else None
        valid=present and observed==warm_info.get("declared_sha256")
        companion=dict(actor=decisions["shared_wm"]["selected_pin"],
              required_set_id=decisions["shared_wm"]["selected_checkpoint_set_id"],required_step=decisions["shared_wm"]["selected_step"],
              sidecar_file_sha_validated=valid,only_permitted_source=artifact(source) if valid else None,
              observed_sidecar_sha256=observed,expected_sidecar_sha256=warm_info.get("declared_sha256"),
              status="unavailable_sidecar_digest_mismatch" if present and not valid else "unavailable_same_set_sidecar",
              scope="The actor is selected without reference to sidecar health; never search another step.")
        if valid:
            dst=args.out/"paired_transfer";env=environment(repo,Path(base["upstream"]))
            env.update(CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu")
            cmd=[sys.executable,str(repo/"tools/export_wuji_heldout_dynamics.py"),"--actor",
                 str(check(decisions["shared_wm"]["selected_pin"])),"--selection",str(source_plan),"--recovery-selection",
                 str(anchors),"--output",str(dst),"--record-numerical-unavailability"]
            with (args.out/"paired_transfer.log").open("x") as log:
                export_job=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            if export_job.returncode:
                companion.update(status="unavailable_optional_export_error",exit_code=export_job.returncode,
                                 technical_evidence=artifact(args.out/"paired_transfer.log"))
            elif (dst/"manifest.json").is_file():
                stamp=read(dst/"manifest.json")
                if stamp["source_set_id"]!=companion["required_set_id"] or stamp["actor_sha256"]!=companion["actor"]["sha256"]:
                    raise ValueError("Paired model is not from the selected recovery actor")
                companion.update(status="available_same_paired_weights_and_scales",manifest=artifact(dst/"manifest.json"),
                                 payload=artifact(dst/stamp["payload_file"]))
            else:
                data=read(dst/"unavailable.json")
                if data["source_set_id"]!=companion["required_set_id"]:raise ValueError("Wrong rejected auxiliary model")
                companion.update(status=data["status"],numerical_evidence=artifact(dst/"unavailable.json"))
        companion["verification_cpu_s"]=time.perf_counter()-clock
        store(args.out/"paired_dynamics_source.json",companion)
        record["optional_dynamics"]=artifact(args.out/"paired_dynamics_source.json")
        record["optional_model_preparation_s"]=companion["verification_cpu_s"]
    store(args.out/"recovery_selection.json",record)
    print(digest(args.out/"recovery_selection.json"))


class RecoveryJobs(Jobs):
    def __init__(self,plan,original):
        super().__init__(plan)
        self.original=original

    def one(self,key,subject,method,kind,card,build,result,**kwargs):
        if kind=="certify":
            job=completed_job(self.original,key)
            return Path(job["output"]),job
        return super().one(key,subject,method,kind,card,build,result,**kwargs)


def recovery_costs(base_plan,base_result,recovery_plan,registration,recovery_entries,frozen_stages):
    original=base_result["entries"]
    common=[e for e in original if e["method"] is None]
    decisions=registration["decisions"]; participants=base_plan["subjects"]
    statuses={s:v["source_status"] for s,v in frozen_stages.items()}
    # Each hypothetical deployment pays its own full shared-method training and candidate scan.
    rescued={}
    for m in MODELS:
        d=decisions[m]
        if d["status"]=="not_triggered_finite_final":continue
        alias="recovery_"+m
        costs={alias:base_plan["upfront_exclusive_card_s"][m]}
        r=base_plan.get("upfront_exclusive_card_s_range",{}).get(m,[costs[alias],costs[alias]])
        single_plan=dict(recovery_plan,protocol=dict(base_plan["protocol"],adaptation=[alias],zero_shot=[alias]),
                    upfront_exclusive_card_s=costs,upfront_exclusive_card_s_range={alias:r},
                    teacher_health={alias:dict(usable=d["status"]=="selected_latest_finite_actor",
                    status=d["status"],inspection_cpu_wall_s=d["selection_cpu_wall_s"])})
        unique=[dict(e,method=alias) for e in recovery_entries if e["method"]==m]
        standalone=summarize(single_plan,common+unique,statuses)["methods"][alias]
        rescued[alias]=dict(primary_method=m,selected_step=d.get("selected_step"),status=d["status"],
                            assigned=len(participants),standalone=standalone,
                            original_primary_outcome=base_result["methods"][m])
    search=sum(d["selection_cpu_wall_s"] for d in decisions.values())
    extra=sum(e["known_wall_s"] for e in recovery_entries)
    pre=base_plan["upfront_exclusive_card_s"];pt=None if pre["shared_wm"] is None or pre["shared_mf"] is None else pre["shared_wm"]+pre["shared_mf"]
    bounds=base_plan.get("upfront_exclusive_card_s_range",{})
    shared_bounds=[bounds.get(m,[pre[m],pre[m]]) for m in MODELS]
    pre_interval=[sum(v[k] for v in shared_bounds) if all(v[k] is not None for v in shared_bounds) else None for k in (0,1)]
    optional=registration.get("optional_model_preparation_s",0.)
    original_s=sum(e["known_wall_s"] for e in original)+sum(
        h.get("inspection_cpu_wall_s",0) for h in base_plan.get("teacher_health",{}).values())
    return dict(scope="Deployment recovery after a confirmed invalid fixed final actor. Original primary results remain separate.",
                recovery_methods=rescued,skipped_already_usable_final=[m for m in MODELS if decisions[m]["status"]=="not_triggered_finite_final"],
                actual_costs=dict(new_pretraining_s=0,teacher_selection_cpu_s=search,extra_recovery_job_s_known=extra,
                     extra_recovery_gpu_s_known=sum(e["known_wall_s"] for e in recovery_entries if e["card"]),
                     original_teacher_development_by_method_s=pre,full_original_teacher_development_s_once=pt,
                     full_original_teacher_development_s_range=pre_interval,optional_paired_dynamics_preparation_s=optional,
                     extra_recovery_study_s_known=extra+search+optional,
                     original_plus_recovery_plus_distinct_teachers_s_known=original_s+search+extra+optional+pt if pt is not None else None,
                     original_plus_recovery_plus_distinct_teachers_s_range=[
                        original_s+search+extra+optional+endpoint if endpoint is not None else None for endpoint in pre_interval],
                     unknown_costs=pt is None or any(e["unknown_cost"] for e in original+recovery_entries),
                     note="All accepted development segments and charged failures are retained, even after the selected recovery checkpoint. The separate WM-transfer supplement, if run, adds only its incremental charges, no second copy of shared pretraining."),
                unique_recovery_jobs=recovery_entries)


def run(args):
    path=args.out/"recovery_selection.json"
    if digest(path)!=args.recovery_sha256:raise ValueError("Wrong recovery rule/selection digest")
    reg=read(path)
    for pin in [reg["parent_selection"],reg["rule"],reg["actor_anchors"],*reg["code"]]:check(pin)
    if reg.get("optional_dynamics"):check(reg["optional_dynamics"])
    orig=Path(reg["original"]);base,original=verify_release(SimpleNamespace(out=orig,released_plan_sha256=reg["parent_selection"]["sha256"]))
    close=read(orig/"complete.json")
    src=orig/"registered_stages.json"
    if (close["selection_sha"]!=reg["parent_selection"]["sha256"] or close["registered_stages_sha"]!=args.registered_stages_sha256
            or digest(src)!=args.registered_stages_sha256):raise ValueError("Wait for completion of the original registered held-out comparison")
    frozen=read(src)["subjects"];eligible=[m for m,d in reg["decisions"].items() if d["status"]=="selected_latest_finite_actor"]
    for x in frozen.values():
        for k in ("contract","source","parquet","evidence"):check(x[k])
    for m in eligible:check(reg["decisions"][m]["selected_pin"])
    tmp=args.out/"recovery_engine";tmp.mkdir(parents=True,exist_ok=True)
    for name in ("final_selection.json","registered_stages.json"):
        target=tmp/name
        if not target.exists():shutil.copyfile(orig/name,target)
        if digest(target)!=digest(orig/name):raise ValueError("Recovery engine source-stage pointer changed")
    plan=dict(base,root=str(tmp),bulk=reg["bulk"],teachers=dict(base["teachers"]),teacher_health={},
              protocol=dict(base["protocol"],zero_shot=eligible,adaptation=eligible))
    for m in eligible:plan["teachers"][m]=reg["decisions"][m]["selected_pin"]
    jobs=RecoveryJobs(plan,original)
    # Primary.execute reuses identical four target budgets and numerical-stop logic. Its reports here
    # are internal; the separately labeled output below adds the actual common costs and selection.
    if eligible and args.mode=="run":
        source_args=SimpleNamespace(out=tmp,released_plan_sha256=reg["parent_selection"]["sha256"],
                                    registered_stages_sha256=args.registered_stages_sha256)
        primary.execute(source_args,plan,jobs)
    report=recovery_costs(base,read(orig/"cost_and_outcomes.json"),plan,reg,jobs.entries(),frozen)
    report.update(recovery_selection_sha256=args.recovery_sha256,registered_stages_sha256=args.registered_stages_sha256,
                  decisions=reg["decisions"])
    store(args.out/"deployment_recovery_cost_and_outcomes.json",report)
    if args.mode=="run":store(args.out/"complete.json",dict(utc=now(),recovery_sha256=args.recovery_sha256))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("register-final")
    for name in ("original","out","bulk"):p.add_argument("--"+name,required=True,type=lambda s:Path(s).resolve())
    p.add_argument("--final-selection-sha256",required=True)
    for op in ("run","report"):
        p=sub.add_parser(op);p.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
        p.add_argument("--recovery-sha256",required=True);p.add_argument("--registered-stages-sha256",required=True)
    a=parser.parse_args()
    if a.mode=="register-final":
        with (a.original/"supervisor.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);register(a)
    elif a.mode=="report":run(a)
    else:
        with (a.out/"supervisor.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)
