"""CPU-only contract test of held-out scheduling and accounting using the existing development fixture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["JAX_PLATFORMS"]="cpu"

from run_wuji_heldout import Jobs, artifact, comparison_schedule, digest, execute, fractions, read, store, terminal_streams
from heldout_wuji_cost import summarize


class DryJobs(Jobs):
    """Build real command vectors, emitting explicitly synthetic child receipts instead of executing them."""
    def __init__(self,plan,truth):
        super().__init__(plan)
        self.truth=truth
        self.calls=[]
        self.rows=[]

    def one(self,key,subject,method,kind,card,build,result,*,steps=None,shared_disk=False):
        import numpy as np
        dest=self.bulk/key
        dest.mkdir(parents=True,exist_ok=False)
        command=list(map(str,build(dest)))
        self.calls.append(dict(key=key,subject=subject,method=method,steps=steps,kind=kind,argv=command))
        x=dict(dry_fixture=True);outcome=None
        if kind=="adapt":
            assert "--evaluation-adaptation" in command and command[command.index("--steps")+1]==str(steps)
            first=steps==self.plan["steps"][0]
            if first:
                assert "--shared-initial-state" in command
                assert ("--adapt-from-actor" in command)==(method in self.plan["teachers"])
                assert "--resume-state" not in command
            else:
                assert "--resume-state" in command and "--resume-replay" in command and "--adapt-from-actor" not in command
            assert ("--hybrid" in command)==(method=="shared_wm")
            x.update(learner_steps=steps,role="per_reference_adaptation")
            d=dest/"checkpoints";d.mkdir()
            (d/f"policy_{steps}.safetensors").write_text(f"fixture_{method}_{subject}_{steps}")
        elif kind=="evaluate":
            if steps==0:assert command[command.index("--checkpoint")+1]==self.plan["teachers"][method]["path"]
            ckpt=Path(command[command.index("--checkpoint")+1])
            q=Path(command[command.index("--parquet")+1])
            # Synthetic successes exercise bookkeeping only, never read a policy.
            status="passed" if method=="shared_mf" and (steps==0 or steps>=self.plan["steps"][1]) else "failed"
            x.update(status=status,realizations=[{"numerical_pass":status=="passed"}]*3,
                     force_validation={"validated":True},checkpoint_sha256=digest(ckpt),
                     parquet_sha256=digest(q),legacy_exact_gates=False,legacy_repeat_status="failed")
            outcome=dict(status=status,full_repeated=status=="passed",historical_strict=False,historical_repeat="failed")
            raw=dest/"raw";raw.mkdir()
            np.savez(raw/"trace.npz",raw_object_pose=self.truth,
                     actor_action=np.zeros((len(self.truth)-1 if method=="shared_mf" else 1,26),dtype=np.float32))
        elif kind=="disturbance":
            assert steps in (0,self.plan["budget"]) and method=="shared_mf" and "--heldout-score" in command
        store(dest/result,x)
        synthetic_s=4. if kind=="adapt" else 2.
        self.rows.append(dict(subject=subject,method=method,steps=steps,kind=kind,card=card,
                              known_wall_s=synthetic_s,unknown_cost=False,outcome=outcome))
        return dest,dict(x,checkpoint_health=dict(step=steps,actor_usable=True,resume_usable=True,
                       classification="healthy")) if kind=="adapt" else x

    def entries(self):return list(self.rows)


def main(args):
    import numpy as np
    repo,fixture=args.repo,args.fixture
    # The fixture must name the historical developer only.
    export_file,=fixture.glob("inputs/assets/human_motion_data/hocap/processed/*/*/export.json")
    exp=read(export_file)
    assert exp["original_sequence"]=="subject_1/20231025_170650" and exp["object_id"]=="G04_2"
    contract=fixture/"contract/contract.json"; c=read(contract)
    assert c["scorable"] and read(fixture/"freeze.json")["no_reserved_participant_access"] is True
    with np.load(fixture/"contract/source_intervals.npz",allow_pickle=False) as a:truth=a["source_poses"]
    with tempfile.TemporaryDirectory(prefix="dry_",dir=fixture) as tmp:
        base=Path(tmp)
        stages={"nominal_seconds_per_single_cumulative":[1,2,3]}
        scored=[dict(stream=s,round=r,arbitrary_score=10-r,checkpoint=f"{s}_{r}") for r in (1,2,3)
                for s in ("single0","single1","single2","shared_mf","shared_wm")]
        final=terminal_streams(stages,list(reversed(scored)),"completed")
        assert [final[s]["checkpoint"] for s in ("shared_mf","shared_wm")]==["shared_mf_3","shared_wm_3"]
        for records,status in ((scored[:-1],"completed"),(scored,"training")):
            try:terminal_streams(stages,records,status)
            except ValueError:pass
            else:raise AssertionError("Incomplete development ledger allowed release")
        protocol=dict(primary="fixture_primary",supplementary=["fixture_a","fixture_b"],
                      zero_shot=["shared_mf","shared_wm"],adaptation=["cold_single","shared_mf","shared_wm"])
        subjects=[{"subject":s,"role":"development_fixture_only"} for s in [protocol["primary"],*protocol["supplementary"]]]
        budget=384;worlds=96
        teacher={}
        for m in protocol["zero_shot"]:
            f=base/f"unused_{m}.bin";f.write_text(m);teacher[m]=artifact(f)
        plan=dict(root=str(base),bulk=str(base/"dry_outputs"),repo=str(repo),upstream=str(args.upstream),
                  protocol=protocol,teacher_fixture_only=True,teachers=teacher,subjects=subjects,worlds=worlds,budget=budget,
                  steps=fractions(budget,worlds),physics={"overrides":["sim.iterations=80"]},canonical_init=str(base/"unused_init"),
                  upfront_exclusive_card_s={"cold_single":0.,"shared_mf":100.,"shared_wm":200.})
        store(base/"final_selection.json",plan);sha=digest(base/"final_selection.json")
        refs={s["subject"]:dict(parquet=artifact(export_file.with_name("data.parquet")),contract=artifact(contract),
             source=artifact(fixture/"inputs/source/source.json"),evidence=artifact(fixture/"evidence/evidence.json"),
             source_status="scorable",source_review_active_seconds=3.0) for s in subjects}
        store(base/"registered_stages.json",dict(subjects=refs,final_selection_sha256=sha))
        opts=SimpleNamespace(out=base,released_plan_sha256=sha,registered_stages_sha256=digest(base/"registered_stages.json"))
        jobs=DryJobs(plan,truth)
        execute(opts,plan,jobs)
        events=jobs.calls
        first_adapt=next(i for i,e in enumerate(events) if e["kind"]=="adapt")
        assert sum(e["kind"]=="evaluate" and e["steps"]==0 for e in events[:first_adapt])==6
        assert sum(e["kind"]=="adapt" for e in events)==36
        assert sum(e["kind"]=="evaluate" for e in events)==42
        result=read(base/"cost_and_outcomes.json")
        mf=result["methods"]["shared_mf"]
        assert mf["zero_shot_verified"]==3 and mf["final_budget_verified"]==3 and mf["ever_verified"]==3
        assert mf["first_verified_including_pretraining_s"]-mf["first_verified_end_to_end_s"]==100.
        assert result["methods"]["cold_single"]["target_gpu_h_per_verified"] is None
        # Incomplete costs and source ambiguity suppress categorical inference.
        extra=dict(subject="fixture_a",method="shared_mf",kind="adapt",steps=0,card=True,
                   unknown_cost=True,known_wall_s=0.,outcome=None)
        gaps=summarize(plan,[extra,*jobs.entries()],{s:"scorable" for s in refs})["methods"]["shared_mf"]
        assert gaps["first_verified_including_pretraining_s"] is None
        missing={s:dict(v) for s,v in refs.items()};missing["fixture_b"]["source_status"]="unscorable"
        assert all(e[1]!="fixture_b" for e in comparison_schedule(plan,missing))
        # Exercise the actual executor branch, not just the post-hoc cost fold.
        class PoisonedTarget(DryJobs):
            def one(self,key,subject,method,kind,card,build,result,**kwargs):
                dest,returned=super().one(key,subject,method,kind,card,build,result,**kwargs)
                step=kwargs.get("steps")
                if kind=="adapt":
                    health=returned["checkpoint_health"]
                    if (subject,method,step)==("fixture_a","shared_mf",self.plan["steps"][1]):
                        health.update(actor_usable=False,resume_usable=False,classification="nonfinite_target_actor")
                    if (subject,method,step)==("fixture_b","shared_wm",self.plan["steps"][2]):
                        health.update(actor_usable=True,resume_usable=False,classification="nonfinite_target_resume")
                    self.rows[-1]["checkpoint_health"]=health
                return dest,returned
        second=base/"stopping";second.mkdir()
        plan2=dict(plan,root=str(second),bulk=str(second/"bulk"))
        store(second/"final_selection.json",plan2);sha2=digest(second/"final_selection.json")
        store(second/"registered_stages.json",dict(subjects=refs,final_selection_sha256=sha2))
        opts2=SimpleNamespace(out=second,released_plan_sha256=sha2,registered_stages_sha256=digest(second/"registered_stages.json"))
        stopped=PoisonedTarget(plan2,truth);execute(opts2,plan2,stopped)
        calls={(e["subject"],e["method"],e["kind"],e["steps"]) for e in stopped.calls}
        _,middle,later,final=plan2["steps"]
        assert ("fixture_a","shared_mf","evaluate",middle) not in calls
        assert ("fixture_a","shared_mf","adapt",later) not in calls
        assert ("fixture_a","shared_mf","disturbance",0) in calls
        assert ("fixture_b","shared_wm","evaluate",later) in calls
        assert ("fixture_b","shared_wm","adapt",final) not in calls
        assert ("fixture_primary","shared_wm","evaluate",final) in calls
        # Test the real Windows command routing without accessing a source archive.
        splan=dict(plan,windows_python=str(args.hand_python))
        smoke=Jobs(splan)
        tested=subprocess.run(smoke.win("stage_frozen_hocap_evaluation.py","--help"),cwd=repo,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if tested.returncode:raise RuntimeError(tested.stderr)
        # Test exact failed-attempt charging and reuse on ordinary CPU child processes.
        def failing(dest):return [sys.executable,"-c","raise SystemExit(4)"]
        try:smoke.one("cpu_retry","fixture_primary","shared_mf","adapt_test",False,failing,"ok.json")
        except RuntimeError:pass
        else:raise AssertionError("Expected a failed fixture process")
        def succeeds(dest):
            code="import sys;from pathlib import Path;p=Path(sys.argv[1]);p.mkdir();(p/'ok.json').write_text('{}')"
            return [sys.executable,"-c",code,str(dest)]
        _,recovered=smoke.one("cpu_retry","fixture_primary","shared_mf","adapt_test",False,succeeds,"ok.json")
        _,reused=smoke.one("cpu_retry","fixture_primary","shared_mf","adapt_test",False,failing,"ok.json")
        assert recovered==reused and len(smoke.entries())==2
        report=dict(scope="CPU software fixture. Three placeholder task IDs all reference the same development clip. Synthetic child outcomes and toy costs are not research results.",
                    opened_reserved_source=False,trained_policy=False,used_gpu=False,checks=dict(
                        final_budget_selection_independent_of_score_and_order=True,unfinished_study_forbids_release=True,
                        six_shared_zero_shot_before_any_target_training=True,four_equal_budgets_all_method_subjects=True,
                        fresh_canonical_then_actor_only_transfer=True,native_resume_after_first_stage=True,
                        exact_subject_and_source_identity=True,distinct_primary_and_supplementary_counts=True,
                        complete_repeat_required=True,zero_denominator_and_unknown_cost_propagation=True,
                        source_unscorable_preserved=True,disturbance_outside_qualification=True,
                        real_schedule_skips_only_poisoned_target_branches=True,
                        real_windows_source_cli_routing=True,exact_retry_and_reuse=True),
                    command_counts={k:sum(e["kind"]==k for e in events) for k in ("certify","evaluate","adapt","disturbance")})
    store(args.output,report)
    print(json.dumps(report["command_counts"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("repo","upstream","fixture","hand-python","output"):
        parser.add_argument("--"+k,required=True,type=lambda s:Path(s).resolve())
    main(parser.parse_args())
