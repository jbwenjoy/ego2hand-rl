"""Dev-only synthetic ledger checks for warm transfer scheduling and amortization."""
import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace

from heldout_wuji_cost import summarize
from run_wuji_heldout import comparison_schedule,store
from run_wuji_heldout_dynamics_supplement import outcomes


def test(output):
    people=["test_1","test_2","test_3"]
    base=dict(subjects=[dict(subject=s) for s in people],steps=[24,48,72,96],budget=96,
              protocol=dict(primary=people[0],supplementary=people[1:],zero_shot=["shared_mf","shared_wm"],
                            adaptation=["cold_single","shared_mf","shared_wm"]),
              upfront_exclusive_card_s={"shared_mf":60.,"shared_wm":100.,"cold_single":0.})
    statuses={s:"scorable" for s in people};frozen={s:dict(source_status=k) for s,k in statuses.items()}
    jobs=[];baseline=[];seq=comparison_schedule(base,frozen)
    def entry(k,s,m,n,amount):
        outcome=None
        if k=="evaluate":outcome=dict(status="passed" if n>=48 else "failed",full_repeated=n>=48)
        return dict(subject=s,method=m,kind=k,steps=n,card=k!="prepare",known_wall_s=amount,unknown_cost=False,outcome=outcome)
    for s in people:baseline.append(entry("prepare",s,None,None,3.))
    for kind,s,m,n,_ in seq:
        baseline.append(entry(kind,s,m,n,2.))
        if m=="shared_wm" and n>0:jobs.append(entry(kind,s,m,n,4.))
    with tempfile.TemporaryDirectory() as tmp:
        original=SimpleNamespace(root=Path(tmp))
        v1=summarize(base,baseline,statuses);v1["entries"]=baseline
        store(original.root/"cost_and_outcomes.json",v1)
        extras=dict(export_cpu_s=9.,source_teacher={},dynamics_manifest={})
        result=outcomes(extras,base,original,SimpleNamespace(entries=lambda:jobs),frozen)
    actual=result["actual_costs"]
    assert result["reused_zero_shot_reports"]==3
    assert actual["new_zero_shot_jobs"]==actual["new_pretraining_jobs"]==0
    assert actual["supplement_new_s_known"]==sum(e["known_wall_s"] for e in jobs)+9.
    assert actual["distinct_teacher_pretraining_s_once"]==160.
    assert actual["total_target_study_plus_distinct_teacher_s_known"]==(
        sum(e["known_wall_s"] for e in baseline)+sum(e["known_wall_s"] for e in jobs)+9+160)
    assert result["standalone"]["pretraining_exclusive_card_s"]==100.
    assert all([r["target_steps"] for r in result["paired_stage_status"][s]]==[0,24,48,72,96] for s in people)
    assert all(result["paired_stage_status"][s][0]["same_evaluation_reused"] for s in people)
    assert sum(e["kind"]=="adapt" for e in jobs)==12 and sum(e["kind"]=="evaluate" for e in jobs)==12
    store(output,dict(scope="Synthetic CPU scheduling/accounting contract; no participants or physics data read.",
                      native_target_budgets_identical=True,zero_shot_charged_per_method_and_computed_once=True,
                      teacher_charged_once_per_standalone_deployment=True,actual_experiment_teacher_compute_deduplicated=True,
                      failed_and_completed_reference_inputs_inherited_from_v1=True,
                      new_target_training_segments=12,new_target_evaluations=12,new_zero_shot_jobs=0,
                      actual_heldout_scores=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    test(p.parse_args().output)
