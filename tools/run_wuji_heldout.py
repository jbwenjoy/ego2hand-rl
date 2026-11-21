"""Post-training, prospective three-subject evaluation. Run from the FlashCHORD WSL environment."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from benchmark_wuji_parallel import environment, train_command
from heldout_wuji_cost import summarize
from heldout_teacher_health import inspect_actor
from run_wuji_matched_tranche import digest, local_path, read, store, verified_phase
from summarize_wuji_tranche import apply_interruption_bounds, fold, unscored_processes


def now():
    return datetime.now(timezone.utc).isoformat()


def artifact(path):
    path=Path(path).resolve()
    return {"path":str(path),"sha256":digest(path),"bytes":path.stat().st_size}


def check(pin):
    p=local_path(pin["path"])
    if not p.is_file() or p.stat().st_size!=pin["bytes"] or digest(p)!=pin["sha256"]:
        raise ValueError(f"Frozen artifact changed: {p}")
    return p


def fractions(horizon,worlds):
    if horizon%worlds or horizon<4*worlds:raise ValueError("Four complete collection checkpoints required")
    return [((horizon//worlds*k+3)//4)*worlds for k in (1,2,3,4)]


def verify_split(path,protocol):
    if digest(path)!=protocol["split_manifest_sha256"]:raise ValueError("Wrong reserved split")
    sig=read(path.with_name("split_manifest.signature.json"))
    if sig["public_key_hex"]!=protocol["split_public_key_hex"]:raise ValueError("Wrong split public key")
    with tempfile.TemporaryDirectory() as tmp:
        pub,detached=Path(tmp)/"key.der",Path(tmp)/"sig"
        pub.write_bytes(bytes.fromhex("302a300506032b6570032100"+sig["public_key_hex"]))
        detached.write_bytes(bytes.fromhex(sig["signature_hex"]))
        subprocess.run(["openssl","pkeyutl","-verify","-pubin","-inkey",str(pub),"-keyform","DER","-rawin",
                        "-in",str(path),"-sigfile",str(detached)],check=True,capture_output=True)
    return read(path)


def terminal_streams(dplan,ledger,live_stage):
    rounds=len(dplan["nominal_seconds_per_single_cumulative"])
    expected={(s,r) for s in ("single0","single1","single2","shared_mf","shared_wm") for r in range(1,rounds+1)}
    if live_stage!="completed" or len(ledger)!=len(expected) or {(r["stream"],r["round"]) for r in ledger}!=expected:
        raise ValueError("The full predeclared development tranche must finish before final selection or unsealing")
    return {r["stream"]:r for r in ledger if r["round"]==rounds and r["stream"] in ("shared_mf","shared_wm")}


def freeze_final(args):
    repo=args.repo;protocol_path=repo/"configs/hocap_heldout_comparison_v1.json"
    protocol=read(protocol_path);dplan=read(args.tranche/"plan.json");ledger=read(args.tranche/"ledger.json")
    finals=terminal_streams(dplan,ledger,read(args.tranche/"live.json").get("stage"))
    if args.out.parts[:3]!=("/","mnt","d"):
        raise ValueError("Source-stage output must be under /mnt/d for the Windows-to-WSL pinned-source evaluator")
    verified_phase(dplan)
    split_path=repo/"runs/hocap_supplementary_reservation_v1/split_manifest.json"
    split=verify_split(split_path,protocol)
    subjects=[]
    for index,row in enumerate([split["original_primary"],*split["supplementary"]]):
        p=check(row["reservation"]);r=read(p)
        subjects.append(dict(subject=r["subject"],sequence=r["sequence"],object=r["object"],
                             role="primary" if index==0 else "supplementary",reservation=artifact(p),
                             package_sha256=r["source_bundle"]["sha256"],
                             dataset_root=str(local_path(r["pose_arrays"][0]["path"]).parents[2])))
    if [s["subject"] for s in subjects]!=[protocol["primary"],*protocol["supplementary"]]:raise ValueError("Wrong subject grouping")
    scored=fold(dplan,ledger);extra,pending=unscored_processes(dplan,ledger)
    bound_files=sorted(args.tranche.glob("recovery*/cost_bounds.json"))
    bounds={}
    for path in bound_files:
        for name,values in read(path)["attempts"].items():
            if name in bounds and bounds[name]!=values:raise ValueError(f"Conflicting interruption records for {name}")
            bounds[name]=values
    apply_interruption_bounds(scored,ledger,bounds)
    totals=scored["methods"]
    upfront={"cold_single":0.}
    upfront_range={"cold_single":[0.,0.]}
    teachers={}
    for method in protocol["zero_shot"]:
        final=finals[method]
        teachers[method]=artifact(final["checkpoint"])
        m=totals[method]
        unscored=sum(x["seconds"] for x in extra.get(method,[]))
        lower,upper=m["scored_card_s_interval"]
        upfront_range[method]=[lower+unscored,None if upper is None or pending.get(method) else upper+unscored]
        upfront[method]=None if m["unknown_costs"] or pending.get(method) else lower+unscored
    health={method:inspect_actor(check(pin)) for method,pin in teachers.items()}
    for method,pin in teachers.items():
        if health[method]["checkpoint_sha256"]!=pin["sha256"]:raise ValueError("Terminal actor changed during CPU precheck")
    # Passive calibration needs only the scene configuration of a checkpoint.
    passive=next((teachers[m] for m in protocol["zero_shot"] if health[m]["usable"]),None)
    if passive is None:
        last=max(r["round"] for r in ledger)
        passive=artifact(next(r["checkpoint"] for r in ledger if r["stream"]=="single0" and r["round"]==last))
    canonical=Path(dplan["canonical_initial_state"])
    init_files=[artifact(canonical/n) for n in ("initial.json","state_0.safetensors")]
    stage_cfg=repo/"configs/hocap_holdout_source_qualification_v1.json"
    if digest(stage_cfg)!=protocol["stage_config_sha256"]:raise ValueError("Prospective source-stage protocol changed")
    scripts=["tools/"+n for n in ("run_wuji_heldout.py","heldout_teacher_health.py","heldout_numeric_evaluation.py",
                 "heldout_wuji_cost.py","summarize_wuji_tranche.py","evaluate_wuji_heldout.py",
                 "stage_frozen_hocap_evaluation.py","fetch_heldout_depth_frames.py","freeze_hocap_source_stages.py",
                 "run_wuji_multiref_flashsac.py","evaluate_flash_wuji_quality.py","benchmark_wuji_parallel.py",
                 "validate_flash_wuji_contact_force.py","evaluate_wuji_disturbance.py","materialize_hocap_flashchord.py",
                 "prepare_hocap_flashchord.py","audit_hocap_carton_support.py","stage_hocap_development_window.py")]
    frozen=[artifact(repo/s) for s in [*scripts,"configs/hocap_carton_evaluation.json",
            "configs/hocap_carton_physical_repeat.json","configs/hocap_wuji_disturbance.json",
            "configs/flashchord_wuji_physics_v2.json","configs/hocap_holdout_source_qualification_v1.json",
            "src/supermimic_research/ego/source_stage_contract.py"]]
    plan=dict(created=now(),protocol=protocol,protocol_file=artifact(protocol_path),split=artifact(split_path),
              source_archive_urls=artifact(check(split["source_archives"])),
              dev_plan=artifact(args.tranche/"plan.json"),dev_ledger=artifact(args.tranche/"ledger.json"),
              dev_input_closure=artifact(Path(dplan["inputs"])),teachers=teachers,subjects=subjects,
              teacher_health=health,passive_config_checkpoint=passive,
              upfront_exclusive_card_s=upfront,upfront_exclusive_card_s_range=upfront_range,
              development_interruption_evidence=[artifact(p) for p in bound_files],
              canonical_init=str(canonical),initial_files=init_files,
              frozen_code=frozen,worlds=dplan["worlds"],budget=dplan["single_clip_curriculum_horizon"],
              steps=fractions(dplan["single_clip_curriculum_horizon"],dplan["worlds"]),
              physics=dplan["physics"],common_plan_snapshot_s=dplan.get("plan_and_snapshot_wall_s"),
              repo=str(repo),root=str(args.out),bulk=str(args.bulk),upstream=str(args.upstream),
              windows_python=str(args.hand_python),full_card_clock_scope="Closed child wall incl setup, verified saves, adaptation and scoring; passive source prep is separately timed.")
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/"final_selection.json"
    if path.exists():raise FileExistsError(path)
    store(path,plan)
    print(digest(path))


class Jobs:
    def __init__(self,plan):
        self.plan=plan
        self.root=Path(plan["root"])
        self.bulk=Path(plan["bulk"])
        self.repo=Path(plan["repo"])
        self.env=environment(self.repo,Path(plan["upstream"]))
        self.new_process_seconds=0.

    def one(self,key,subject,method,kind,card,build,result,*,steps=None,shared_disk=False):
        """Complete or reuse an immutable job. Record failed/unknown attempts in the same ledger."""
        logs=self.root/"jobs"/key;logs.mkdir(parents=True,exist_ok=True)
        prior=sorted(logs.glob("attempt_*.json"))
        for f in prior:
            j=read(f)
            if j.get("success"):
                check(j["receipt"])
                for pin in j.get("checkpoint_set",[]):check(pin)
                return Path(j["output"]),j
        index=len(prior)+1
        dest=(self.root/"source_jobs" if shared_disk else self.bulk)/key/f"attempt_{index:03d}"
        dest.parent.mkdir(parents=True,exist_ok=True)
        f=logs/f"attempt_{index:03d}.json";cmd=build(dest)
        j=dict(key=key,attempt=index,subject=subject,method=method,kind=kind,card=card,steps=steps,
               output=str(dest),start=now(),command=[str(x) for x in cmd],success=False,elapsed_s=None)
        store(f,j);tic=time.perf_counter()
        try:
            with (logs/f"attempt_{index:03d}.log").open("x") as stream:
                proc=subprocess.run(cmd,cwd=self.repo,env=self.env,stdout=stream,stderr=subprocess.STDOUT)
            j["returncode"]=proc.returncode
            if proc.returncode:raise RuntimeError(f"{key} failed, see {stream.name}")
            j["receipt"]=artifact(dest/result)
            if kind=="adapt":
                n=read(dest/result)["learner_steps"];ckpt=dest/"checkpoints"
                names=[f"{x}_{n}.safetensors" for x in ("policy","state","replay")]+[f"paired_{n}.json"]
                if method=="shared_wm":names.append(f"world_model_{n}.msgpack")
                j["checkpoint_set"]=[artifact(ckpt/name) for name in names]
            j["success"]=True
        finally:
            j["elapsed_s"]=time.perf_counter()-tic;j["end"]=now();store(f,j)
            self.new_process_seconds+=j["elapsed_s"]
        return dest,j

    def entries(self):
        rows=[]
        for folder in (self.root/"jobs").glob("*"):
            for file in sorted(folder.glob("attempt_*.json")):
                j=read(file);rec=None
                if j.get("success") and j["kind"]=="evaluate":
                    raw=read(check(j["receipt"]))
                    rec=dict(status=raw["status"],full_repeated=raw["status"]=="passed" and len(raw["realizations"])==3
                             and raw["force_validation"]["validated"] and all(v["numerical_pass"] for v in raw["realizations"]),
                             historical_strict=raw.get("legacy_exact_gates"),
                             historical_repeat=raw.get("legacy_repeat_status"))
                if j.get("success") and j["kind"]=="disturbance":
                    raw=read(check(j["receipt"]))
                    rec={k:raw.get(k) for k in ("onset_frame","assigned_summary","checkpoint_sha256","parquet_sha256")}
                rows.append(dict(**{k:j[k] for k in ("key","subject","method","kind","card","steps","start")},
                                 known_wall_s=j.get("elapsed_s") or 0,unknown_cost=j.get("elapsed_s") is None,
                                 receipt=j.get("receipt"),outcome=rec))
        return sorted(rows,key=lambda r:r["start"])

    def win(self,script,*args):
        def windows(path):
            return subprocess.check_output(["wslpath","-w",str(path)],text=True).strip()
        boot="import runpy,sys;from pathlib import Path;p=Path(sys.argv.pop(1));sys.path[:0]=[str(p.parent),str(p.parents[1]/'src')];sys.argv[0]=str(p);runpy.run_path(str(p),run_name='__main__')"
        return [self.plan["windows_python"],"-c",boot,windows(self.repo/"tools"/script),
                *[windows(a) if isinstance(a,Path) else str(a) for a in args]]

    def local(self,script,*args):
        return [sys.executable,"-u",str(self.repo/"tools"/script),*[str(a) for a in args]]


def verify_release(args):
    path=args.out/"final_selection.json"
    if digest(path)!=args.released_plan_sha256: raise ValueError("Explicit released final-selection digest is required")
    plan=read(path)
    for pin in [plan["dev_plan"],plan["dev_ledger"],plan["dev_input_closure"],plan["protocol_file"],plan["split"],plan["source_archive_urls"],
                *plan["teachers"].values(),*plan["initial_files"],*plan["frozen_code"]]:check(pin)
    verified_phase(read(check(plan["dev_plan"])))
    if plan.get("passive_config_checkpoint"):check(plan["passive_config_checkpoint"])
    return plan,Jobs(plan)


def prepare_sources(args,plan,jobs):
    for x in plan["subjects"]:
        s=x["subject"];r=read(check(x["reservation"]));package=local_path(r["source_bundle"]["path"])
        if digest(package)!=x["package_sha256"]:raise ValueError("Released ZIP differs from reservation")
        inp,_=jobs.one(f"{s}_prepare",s,None,"source_prepare",False,
            lambda dest:jobs.win("stage_frozen_hocap_evaluation.py","prepare",check(x["reservation"]),
                                 args.out/"final_selection.json",dest),"stage_timing.json",shared_disk=True)
        pq,_=jobs.one(f"{s}_pack",s,None,"reference_load",False,lambda dest:
                 [sys.executable,"-u",str(jobs.repo/"tools"/"run_wuji_heldout.py"),"_pack",
                  "--input",str(inp),"--output",str(dest)],"receipt.json",shared_disk=True)
        source=inp/"source/source.json"
        depth,_=jobs.one(f"{s}_depth",s,None,"depth_intake",False,lambda dest:jobs.win("fetch_heldout_depth_frames.py",
                 source,jobs.repo/"configs/hocap_holdout_source_qualification_v1.json",Path(x["dataset_root"]),
                 jobs.repo/"runs/ego_retarget_goal_review_v1/hocap",check(plan["source_archive_urls"]),
                 dest/"receipt.json"),"receipt.json",shared_disk=True)
        rec=read(source); review=dict(subject=s,original_source=artifact(source),
                  endpoint_image_pins=rec["endpoint_image_pins"],depth=artifact(depth/"receipt.json"),
                  materialization=artifact(pq/"receipt.json"),table_seen_at_both_ends=None,
                  source_review_active_seconds=None,source_review_timing_basis=None)
        view_file=args.out/f"{s}_source_review.json"
        if not view_file.exists():store(view_file,review)


def register_stages(args,plan,jobs):
    sealed={}
    for x in plan["subjects"]:
        s=x["subject"]; review=read(args.out/f"{s}_source_review.json")
        if review["subject"]!=s or type(review.get("table_seen_at_both_ends")) is not bool:
            raise ValueError(f"Record source-only endpoint support visibility before scoring {s}")
        active=review.get("source_review_active_seconds")
        if active is not None and (not isinstance(active,(float,int)) or active<0 or not review.get("source_review_timing_basis")):
            raise ValueError("Source-review time must be nonnegative and have an observed timing basis")
        source=check(review["original_source"]);ev=review["table_seen_at_both_ends"]
        born=args.out/"final_selection.json"
        if read(source)["freeze"]["sha256"]!=digest(born):raise ValueError("Source did not originate from this freeze")
        if read(source)["endpoint_image_pins"]!=review["endpoint_image_pins"]:raise ValueError("Changed endpoint evidence")
        cfg=jobs.repo/"configs/hocap_holdout_source_qualification_v1.json"
        d,_=jobs.one(f"{s}_source_evidence",s,None,"source_evidence",False,lambda dest:
                jobs.win("freeze_hocap_source_stages.py","evidence",source,cfg,Path(x["dataset_root"]),dest,
                         "--table-seen-at-both-ends","yes" if ev else "no"),"evidence.json",shared_disk=True)
        if read(d/"evidence.json")["table_seen_at_both_ends"] is not ev:raise ValueError("A completed source review was changed")
        c,_=jobs.one(f"{s}_source_masks",s,None,"source_freeze",False,lambda dest:
                jobs.win("freeze_hocap_source_stages.py","freeze",source,d/"evidence.json",cfg,dest),"contract.json",shared_disk=True)
        inp=source.parent.parent;parquet,=inp.glob("assets/human_motion_data/hocap/processed/*/*/data.parquet")
        sealed[s]=dict(parquet=artifact(parquet),contract=artifact(c/"contract.json"),
                       evidence=artifact(args.out/f"{s}_source_review.json"),source=artifact(source),
                       source_review_active_seconds=active,
                       source_status="scorable" if read(c/"contract.json")["scorable"] else "unscorable")
    file=args.out/"registered_stages.json"
    record=dict(final_selection_sha256=digest(args.out/"final_selection.json"),subjects=sealed)
    if file.exists() and read(file)!=record:raise ValueError("Cannot change registered source stages")
    if not file.exists():store(file,record)
    print(digest(file))


def comparison_schedule(plan, stages):
    """All zero-shot scores precede target adaptation. Endpoints use source transitions, not loss."""
    subjects=[r["subject"] for r in plan["subjects"] if stages[r["subject"]]["source_status"]=="scorable"]
    def usable(method):return plan.get("teacher_health",{}).get(method,{}).get("usable",True)
    for s in subjects:
        for device in ("cpu","cuda:0"):yield ("certify",s,None,0,device)
    for i,s in enumerate(subjects):
        methods=[m for m in plan["protocol"]["zero_shot"] if usable(m)]
        for m in methods[i%2:]+methods[:i%2]:yield ("evaluate",s,m,0,None)
    for j,steps in enumerate(plan["steps"]):
        for i,subject in enumerate(subjects):
            methods=[m for m in plan["protocol"]["adaptation"] if usable(m)]
            k=(i+j)%len(methods)
            for method in methods[k:]+methods[:k]:
                yield ("adapt",subject,method,steps,None)
                yield ("evaluate",subject,method,steps,None)
    for endpoint in (0,plan["budget"]):
        for s in subjects:
            for m in (plan["protocol"]["zero_shot"] if endpoint==0 else plan["protocol"]["adaptation"]):
                if not usable(m):continue
                yield ("disturbance",s,m,endpoint,None)


def execute(args,plan,jobs):
    path=args.out/"registered_stages.json"
    if digest(path)!=args.registered_stages_sha256:raise ValueError("Explicit registered source-stage digest required")
    registered=read(path)
    if registered["final_selection_sha256"]!=args.released_plan_sha256:raise ValueError("Source stages belong to another final selection")
    stages=registered["subjects"]
    if set(stages)!={r["subject"] for r in plan["subjects"]}:raise ValueError("Freeze all assigned subjects before first evaluation")
    for s,info in stages.items():
        for key in ("parquet","contract","source","evidence"):check(info[key])
        if ("scorable" if read(check(info["contract"]))["scorable"] else "unscorable")!=info["source_status"]:
            raise ValueError("Modified source eligibility")
    certificates={};learners={};scores={}
    teachers={m:check(ref) for m,ref in plan["teachers"].items()}
    passive=check(plan.get("passive_config_checkpoint",plan["teachers"]["shared_mf"]))
    warp_cache=Path(plan["upstream"]).parent/"cache/warp"
    physics=jobs.repo/"configs/flashchord_wuji_physics_v2.json"
    for kind,s,m,steps,device in comparison_schedule(plan,stages):
        info=stages[s];parquet=check(info["parquet"]);contract=check(info["contract"])
        base=["--parquet",parquet,"--contract",contract,"--contract-sha256",info["contract"]["sha256"],
              "--physics",physics,"--warp-cache",warp_cache]
        if kind=="certify":
            dest,_=jobs.one(s+"_cert_"+device.replace(":",""),s,None,kind,device!="cpu",
                     lambda out:jobs.local("evaluate_wuji_heldout.py","certify",*base,"--checkpoint",passive,
                                           "--device",device,"--out",out),"certificate.json")
            certificates[s,device]=dest/"certificate.json"
        elif kind=="adapt":
            previous=learners.get((s,m))
            def command(out):
                idx=[r["subject"] for r in plan["subjects"]].index(s)
                cmd=train_command(jobs.repo,Path(plan["upstream"]),out,[parquet],plan["worlds"],steps,steps,
                                  seed=42+idx,resume_dir=previous,hybrid=(m=="shared_wm"))
                # Fixed native sample endpoints, using the native runner, with no timed-budget early exit.
                cmd[cmd.index(str(jobs.repo/"tools/time_wuji_flashsac.py"))]=str(jobs.repo/"tools/run_wuji_multiref_flashsac.py")
                cmd.extend(["--evaluation-adaptation","--curriculum-horizon-steps",str(plan["budget"])])
                if previous is None:
                    cmd.extend(["--shared-initial-state",plan["canonical_init"]])
                    if m!="cold_single":cmd.extend(["--adapt-from-actor",str(teachers[m])])
                if m=="shared_wm":cmd.extend(["--wm-fit-horizon","3"])
                for override in plan["physics"]["overrides"]:cmd.extend(["--override",override])
                return cmd
            dest,_=jobs.one(f"{s}_{m}_adapt_{steps}",s,m,kind,True,command,"result.json",steps=steps)
            actual=read(dest/"result.json")
            if actual["learner_steps"]!=steps or actual["role"]!="per_reference_adaptation":raise ValueError("Wrong target budget/role")
            learners[s,m]=dest
        elif kind=="evaluate":
            ckpt=teachers[m] if steps==0 else learners[s,m]/f"checkpoints/policy_{steps}.safetensors"
            dest,_=jobs.one(f"{s}_{m}_eval_{steps}",s,m,kind,True,lambda out:jobs.local("heldout_numeric_evaluation.py",
                    "evaluate",*base,"--checkpoint",ckpt,"--cpu-certificate",certificates[s,"cpu"],
                    "--gpu-certificate",certificates[s,"cuda:0"],"--protocol",jobs.repo/"configs/hocap_carton_evaluation.json",
                    "--out",out),"report.json",steps=steps)
            raw=read(dest/"report.json")
            if raw["checkpoint_sha256"]!=digest(ckpt) or raw["parquet_sha256"]!=info["parquet"]["sha256"]:
                raise ValueError("Evaluator identity mismatch")
            scores[s,m,steps]=(dest,ckpt)
        else:
            dest,ckpt=scores[s,m,steps];rawdir=dest/"raw"
            raw=read(dest/"report.json")
            reason=None
            if not raw["force_validation"]["validated"]:reason="no validated contact-force calibration"
            elif not (rawdir/"trace.npz").exists():reason="no nominal source rollout"
            else:
                import numpy as np
                from supermimic_research.runtime.flash_disturbance import onset_frame
                with np.load(rawdir/"trace.npz",allow_pickle=False) as tr:
                    onset=onset_frame(tr["raw_object_pose"],read(jobs.repo/"configs/hocap_wuji_disturbance.json"))
                    if onset is None:reason="source intervention not applicable"
                    elif len(tr["actor_action"])<=onset:reason="nominal trajectory terminated before exposure"
            if reason:
                store(args.out/f"disturb_{s}_{m}_{steps}.json",dict(assigned=True,exposure=reason))
                continue
            jobs.one(f"{s}_{m}_disturb_{steps}",s,m,kind,True,lambda out:jobs.local("evaluate_wuji_heldout.py",
                "disturb","--heldout-score",dest/"report.json","--parquet",parquet,"--checkpoint",ckpt,
                "--protocol",jobs.repo/"configs/hocap_wuji_disturbance.json","--warp-cache",warp_cache,
                "--out",out),"report.json",steps=steps)
        store(args.out/"progress.json",dict(utc=now(),last_completed=[kind,s,m,steps,device]))
        # Read-only reports can also be requested between work units.
        if kind=="evaluate":report(args,plan,jobs)
    report(args,plan,jobs)
    store(args.out/"complete.json",dict(utc=now(),selection_sha=args.released_plan_sha256,
                                      registered_stages_sha=args.registered_stages_sha256))


def report(args,plan,jobs):
    registered=args.out/"registered_stages.json"
    frozen=read(registered)["subjects"] if registered.exists() else {}
    statuses={s:x["source_status"] for s,x in frozen.items()}
    entries=jobs.entries()
    for s,x in frozen.items():
        active=x.get("source_review_active_seconds")
        entries.insert(0,dict(key=f"{s}_source_visibility_review",subject=s,method=None,kind="source_visibility_review",
                              card=False,steps=None,known_wall_s=active or 0,unknown_cost=active is None,outcome=None))
    result=summarize(plan,entries,statuses)
    result.update(entries=entries,selection_sha256=digest(args.out/"final_selection.json"),
                  registered_stages_sha256=digest(registered) if registered.exists() else None,
                  study_common_development_setup_seconds=plan.get("common_plan_snapshot_s"),
                  coordinator_invocations=[json.loads(line) for line in (args.out/"coordinator.jsonl").read_text().splitlines()]
                        if (args.out/"coordinator.jsonl").exists() else [],
                  coordinator_scope="Measured separately from subject/method clocks; do not multiply whole-study supervisor overhead by the number of counterfactual methods.",
                  disturbance_unexposed=[read(f) | {"file":f.name} for f in sorted(args.out.glob("disturb_subject_*.json"))])
    store(args.out/"cost_and_outcomes.json",result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    freezer=sub.add_parser("freeze-final")
    for name in ("repo","tranche","out","bulk","upstream","hand-python"):
        freezer.add_argument("--"+name,type=lambda s:Path(s).resolve(),required=True)
    for name in ("stage","freeze-stages","run","report"):
        q=sub.add_parser(name);q.add_argument("--out",type=lambda s:Path(s).resolve(),required=True)
        q.add_argument("--released-plan-sha256",required=True)
        if name=="run":q.add_argument("--registered-stages-sha256",required=True)
    q=sub.add_parser("_pack")
    q.add_argument("--input",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.mode=="_pack":
        from stage_frozen_hocap_evaluation import verify_eval
        a.output.mkdir(parents=True,exist_ok=False)
        verify_eval(a.input)
        store(a.output/"receipt.json",dict(evaluation_reference=artifact(a.input/"evaluation_reference.json")))
        return
    if a.mode=="freeze-final":freeze_final(a);return
    clock=time.perf_counter();plan,jobs=verify_release(a)
    if a.mode=="report":report(a,plan,jobs);return
    with (a.out/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.mode=="stage":prepare_sources(a,plan,jobs)
            elif a.mode=="freeze-stages":register_stages(a,plan,jobs)
            else:execute(a,plan,jobs)
        finally:
            elapsed=time.perf_counter()-clock
            with (a.out/"coordinator.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(mode=a.mode,utc=now(),invocation_s=elapsed,
                    child_and_artifact_wall_s=jobs.new_process_seconds,
                    unallocated_coordinator_s=max(0.,elapsed-jobs.new_process_seconds)))+"\n")
        if a.mode=="run":report(a,plan,jobs)


if __name__=="__main__":main()
