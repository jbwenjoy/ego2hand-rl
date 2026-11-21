"""Prospectively register and run the held-out dynamics transfer supplement to the v1 comparison."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from benchmark_wuji_parallel import environment, train_command
from heldout_wuji_cost import summarize
from run_wuji_heldout import Jobs, artifact, check, comparison_schedule, digest, now, read, store, verify_release
from summarize_wuji_tranche import fold, unscored_processes


ARM="shared_wm_warm_dynamics"


def preregister(args):
    registration_start=time.perf_counter()
    base=read(args.original/"final_selection.json")
    cfg_path=Path(base["repo"])/"configs/hocap_heldout_warm_dynamics_v2.json"
    cfg=read(cfg_path)
    if digest(args.original/"final_selection.json")!=args.final_selection_sha256:raise ValueError("Final selection digest mismatch")
    if cfg["parent_comparison_sha256"]!=base["protocol_file"]["sha256"]:raise ValueError("Wrong base comparison version")
    if ((args.original/"registered_stages.json").exists() or (args.original/"source_jobs").exists()
            or any((args.original/"jobs").glob("*_prepare/attempt_*.json"))):
        raise ValueError("Supplement must be registered BEFORE even the first held-out source intake")
    if args.out.exists():raise FileExistsError(args.out)
    # Revalidate the original code/assets, all development rounds, and both selected teachers.
    base,_=verify_release(SimpleNamespace(out=args.original,released_plan_sha256=args.final_selection_sha256))
    args.out.mkdir(parents=True)
    repo=Path(base["repo"]);teacher=check(base["teachers"]["shared_wm"])
    started=time.perf_counter()
    output=args.out/"transfer"
    cmd=[sys.executable,str(repo/"tools/export_wuji_heldout_dynamics.py"),"--actor",str(teacher),"--selection",
         str(args.original/"final_selection.json"),"--output",str(output)]
    env=environment(repo,Path(base["upstream"]))
    env.update(CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu")
    with (args.out/"extraction.log").open("x") as log:
        subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    export_s=time.perf_counter()-started
    manifest=read(output/"manifest.json")
    if manifest["development_fixture"] or manifest["actor_sha256"]!=base["teachers"]["shared_wm"]["sha256"]:
        raise ValueError("Only a matched final shared development actor and model may enter the study")
    supplemental_sources=[repo/"tools"/name for name in ("run_wuji_heldout_dynamics_supplement.py",
                          "export_wuji_heldout_dynamics.py","train_wuji_heldout_warm_dynamics.py")]
    supplemental_sources.append(repo/"src/supermimic_research/learning/heldout_dynamics_transfer.py")
    devplan=read(check(base["dev_plan"]));ledger=read(check(base["dev_ledger"]))
    totals=fold(devplan,ledger)["methods"];unscored,pending=unscored_processes(devplan,ledger)
    dev_known=sum(row["exclusive_card_s_known"] for row in totals.values())+sum(
        e["seconds"] for trials in unscored.values() for e in trials)
    plan=dict(created=now(),method=ARM,protocol=artifact(cfg_path),base_selection=artifact(args.original/"final_selection.json"),
              source_teacher=artifact(teacher),dynamics_manifest=artifact(output/"manifest.json"),
              dynamics_payload=artifact(output/manifest["payload_file"]),export_cpu_s=export_s,
              source_set_id=manifest["source_set_id"],code=[artifact(p) for p in supplemental_sources],
              development_trial_s_known=dev_known,development_unknown_costs=any(row["unknown_costs"] for row in totals.values()) or bool(pending),
              registration_coordinator_s=time.perf_counter()-registration_start-export_s,
              original=str(args.original),root=str(args.out),bulk=str(args.bulk),
              schedule_steps=base["steps"],reference_budget=base["budget"],worlds=base["worlds"])
    store(args.out/"supplement.json",plan)
    print(digest(args.out/"supplement.json"))


def ready(args):
    path=args.out/"supplement.json"
    if digest(path)!=args.supplement_sha256:raise ValueError("A registered supplemental plan digest is required")
    extra=read(path)
    if extra["root"]!=str(args.out):raise ValueError("Wrong supplementary study root")
    for pin in [extra["protocol"],extra["base_selection"],extra["source_teacher"],
                extra["dynamics_manifest"],extra["dynamics_payload"],*extra["code"]]:check(pin)
    orig=Path(extra["original"])
    base,jobs=verify_release(SimpleNamespace(out=orig,released_plan_sha256=extra["base_selection"]["sha256"]))
    if not (orig/"complete.json").is_file():raise ValueError("Run only after the fixed original comparison completes")
    complete=read(orig/"complete.json")
    src=orig/"registered_stages.json"
    if (digest(src)!=args.registered_stages_sha256 or complete["registered_stages_sha"]!=args.registered_stages_sha256
            or complete["selection_sha"]!=extra["base_selection"]["sha256"]):
        raise ValueError("V1 completion and frozen source masks do not agree")
    frozen=read(src)["subjects"]
    for item in frozen.values():
        for key in ("parquet","contract","source","evidence"):check(item[key])
    new_plan=dict(base,root=str(args.out),bulk=extra["bulk"])
    return extra,base,jobs,Jobs(new_plan),frozen


def completed_job(jobs,key):
    for path in sorted((jobs.root/"jobs"/key).glob("attempt_*.json")):
        value=read(path)
        if value.get("success"):
            check(value["receipt"])
            return value
    raise ValueError(f"Missing successful baseline job {key}")


def outcomes(extra,base,original,jobs,frozen):
    baseline=read(original.root/"cost_and_outcomes.json")
    shared=[]
    reused_zero_count=0
    for e in baseline["entries"]:
        if e["method"] is None:
            shared.append(e)
        elif e["method"]=="shared_wm" and e["steps"]==0 and e["kind"] in ("evaluate","disturbance"):
            shared.append(dict(e,method=ARM))
            if e.get("outcome") and e["kind"]=="evaluate":reused_zero_count+=1
    refs=[s for s in base["subjects"]]
    unique=[dict(e,method=ARM) for e in jobs.entries()]
    # Pro-rate the single CPU extraction cost across assignments only when reporting
    # stand-alone deployment economics. The real export ran once.
    extract=[dict(subject=s["subject"],method=None,steps=None,card=False,kind="source_dynamics_export_cpu_share",
                  known_wall_s=extra["export_cpu_s"]/len(refs),unknown_cost=False,outcome=None) for s in refs]
    alternate=dict(base,protocol=dict(base["protocol"],adaptation=[ARM],zero_shot=[ARM]),
                   upfront_exclusive_card_s={ARM:base["upfront_exclusive_card_s"]["shared_wm"]})
    new=summarize(alternate,[*extract,*shared,*unique],{s:v["source_status"] for s,v in frozen.items()})["methods"][ARM]
    baseline_method=baseline["methods"]["shared_wm"]
    paired={}
    for row in refs:
        s=row["subject"];former=baseline_method["subjects"][s]
        matched={r["steps"]:r for r in former["scored"]};pairs=[]
        for nowrow in new["subjects"][s]["scored"]:
            previous=matched.get(nowrow["steps"])
            if previous is None:raise ValueError("Warm transfer has no identical-budget actor-only comparator")
            pairs.append(dict(target_steps=nowrow["steps"],actor_only=previous["status"],
                              actor_and_dynamics=nowrow["status"],same_evaluation_reused=nowrow["steps"]==0))
        paired[s]=pairs
    original_s=sum(e["known_wall_s"] for e in baseline["entries"])
    incremental=extra["export_cpu_s"]+sum(e["known_wall_s"] for e in unique)
    teachers=base["upfront_exclusive_card_s"]
    pret=(teachers["shared_mf"]+teachers["shared_wm"]
          if teachers["shared_mf"] is not None and teachers["shared_wm"] is not None else None)
    return dict(method=ARM,standalone=new,paired_stage_status=paired,reused_zero_shot_reports=reused_zero_count,
                source_teacher=extra["source_teacher"],source_model_manifest=extra["dynamics_manifest"],
                actual_costs=dict(original_target_study_s_known=original_s,supplement_new_s_known=incremental,
                     supplement_new_gpu_s_known=sum(e["known_wall_s"] for e in unique if e["card"]),
                     new_zero_shot_jobs=0,new_pretraining_jobs=0,source_model_export_cpu_s=extra["export_cpu_s"],
                     distinct_teacher_pretraining_s_once=pret,
                     total_target_study_plus_distinct_teacher_s_known=original_s+incremental+pret if pret is not None else None,
                     whole_measured_development_trial_s_known=extra.get("development_trial_s_known"),
                     all_measured_development_and_heldout_jobs_s_known=(extra["development_trial_s_known"]+original_s+incremental
                        if "development_trial_s_known" in extra else None),
                     development_unknown_costs=extra.get("development_unknown_costs"),
                     unknown_costs=any(e["unknown_cost"] for e in [*baseline["entries"],*unique]) or pret is None,
                     scope="Actual measured target work includes charged failures and stress. Common prep, force certificates, teacher development and zero-shot are not rerun; study coordination logged separately."),
                supplemental_registration_coordinator_s=extra.get("registration_coordinator_s"),
                supplemental_run_coordination=[json.loads(l) for l in (jobs.root/"coordination.jsonl").read_text().splitlines()]
                    if hasattr(jobs,"root") and (jobs.root/"coordination.jsonl").exists() else [],
                unique_entries=unique)


def run(args):
    extra,base,original,jobs,frozen=ready(args)
    report_path=jobs.root/"supplement_outcomes.json"
    if args.mode=="report":
        store(report_path,outcomes(extra,base,original,jobs,frozen));return
    refs=[x["subject"] for x in base["subjects"]]
    teacher=check(extra["source_teacher"]); transfer=check(extra["dynamics_manifest"])
    repo=jobs.repo; upstream=Path(base["upstream"]);warp=upstream.parent/"cache/warp"
    native={};scores={}
    for s in refs:
        if frozen[s]["source_status"]!="scorable":continue
        old=read(check(completed_job(original,f"{s}_shared_wm_eval_0")["receipt"]))
        if (old["checkpoint_sha256"]!=extra["source_teacher"]["sha256"] or old["contract_sha256"]!=frozen[s]["contract"]["sha256"]
                or old["parquet_sha256"]!=frozen[s]["parquet"]["sha256"]):raise ValueError("Cannot reuse this zero-shot evaluation")
    for kind,s,m,steps,_ in comparison_schedule(base,frozen):
        if m!="shared_wm" or steps==0:continue
        item=frozen[s]; pq=check(item["parquet"]); contract=check(item["contract"])
        if kind=="adapt":
            prior=native.get(s)
            def cmd(out):
                argv=train_command(repo,upstream,out,[pq],base["worlds"],steps,steps,seed=42+refs.index(s),resume_dir=prior,hybrid=True)
                argv[argv.index(str(repo/"tools/time_wuji_flashsac.py"))]=str(repo/"tools/train_wuji_heldout_warm_dynamics.py")
                argv.extend(["--dynamics-transfer",str(transfer),"--evaluation-adaptation","--wm-fit-horizon","3",
                             "--curriculum-horizon-steps",str(base["budget"])])
                if not prior:argv.extend(["--adapt-from-actor",str(teacher),"--shared-initial-state",base["canonical_init"]])
                for opt in base["physics"]["overrides"]:argv.extend(["--override",opt])
                return argv
            dest,_=jobs.one(f"{s}_{ARM}_adapt_{steps}",s,"shared_wm",kind,True,cmd,"result.json",steps=steps)
            result=read(dest/"result.json");binding=read(dest/"warm_dynamics_binding.json")
            if result["learner_steps"]!=steps or result["role"]!="per_reference_adaptation" or binding["manifest_sha256"]!=extra["dynamics_manifest"]["sha256"]:
                raise ValueError("Unexpected target budget, role, or transferred model")
            native[s]=dest
        elif kind=="evaluate":
            actor=native[s]/f"checkpoints/policy_{steps}.safetensors"
            certs=[check(completed_job(original,f"{s}_cert_{name}")["receipt"]) for name in ("cpu","cuda0")]
            dest,_=jobs.one(f"{s}_{ARM}_eval_{steps}",s,"shared_wm",kind,True,lambda out:jobs.local("evaluate_wuji_heldout.py",
                "evaluate","--parquet",pq,"--contract",contract,"--contract-sha256",item["contract"]["sha256"],
                "--cpu-certificate",certs[0],"--gpu-certificate",certs[1],"--checkpoint",actor,
                "--physics",repo/"configs/flashchord_wuji_physics_v2.json","--warp-cache",warp,
                "--protocol",repo/"configs/hocap_carton_evaluation.json","--out",out),"report.json",steps=steps)
            score=read(dest/"report.json")
            if score["checkpoint_sha256"]!=digest(actor) or score["contract_sha256"]!=item["contract"]["sha256"]:
                raise ValueError("Supplementary score is bound to another policy or source")
            scores[s,steps]=dest,actor
            store(report_path,outcomes(extra,base,original,jobs,frozen))
        elif kind=="disturbance":
            import numpy as np
            from supermimic_research.runtime.flash_disturbance import onset_frame
            folder,actor=scores[s,steps]
            val=read(folder/"report.json");trace=folder/"raw/trace.npz";reason=None
            if not val["force_validation"]["validated"] or not trace.exists():reason="unavailable validated nominal run"
            else:
                with np.load(trace,allow_pickle=False) as a:
                    onset=onset_frame(a["raw_object_pose"],read(repo/"configs/hocap_wuji_disturbance.json"))
                    if onset is None:reason="not applicable"
                    elif len(a["actor_action"])<=onset:reason="unexposed"
            if reason:
                store(jobs.root/f"disturb_{s}_not_exposed.json",{"assigned":True,"reason":reason});continue
            jobs.one(f"{s}_{ARM}_disturb_{steps}",s,"shared_wm",kind,True,lambda out:jobs.local("evaluate_wuji_heldout.py",
                "disturb","--checkpoint",actor,"--parquet",pq,"--heldout-score",folder/"report.json","--warp-cache",warp,
                "--protocol",repo/"configs/hocap_wuji_disturbance.json","--out",out),"report.json",steps=steps)
    store(report_path,outcomes(extra,base,original,jobs,frozen))
    store(jobs.root/"complete.json",dict(finished=now(),supplement_sha256=args.supplement_sha256))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("register-final")
    for name in ("original","out","bulk"):p.add_argument("--"+name,required=True,type=lambda s:Path(s).resolve())
    p.add_argument("--final-selection-sha256",required=True)
    for name in ("run","report"):
        p=sub.add_parser(name);p.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
        p.add_argument("--supplement-sha256",required=True);p.add_argument("--registered-stages-sha256",required=True)
    args=parser.parse_args()
    if args.mode=="register-final":
        with (args.original/"supervisor.lock").open("a") as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);preregister(args)
    elif args.mode=="report":run(args)
    else:
        with (args.out/"supervisor.lock").open("a") as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def child_wall():
                return sum(read(f).get("elapsed_s") or 0. for f in (args.out/"jobs").glob("*/attempt_*.json"))
            prior=child_wall();tic=time.perf_counter()
            try:run(args)
            finally:
                elapsed=time.perf_counter()-tic;child=child_wall()-prior
                with (args.out/"coordination.jsonl").open("a") as log:
                    log.write(json.dumps(dict(utc=now(),invocation_s=elapsed,new_subprocess_and_artifact_s=child,
                                        coordinator_s=max(0.,elapsed-child)))+"\n")
            args.mode="report";run(args)
