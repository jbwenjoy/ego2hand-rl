"""Verify outcome handling for two observed development terminal candidates without a GPU."""
import argparse
import json
from pathlib import Path

from heldout_teacher_health import inspect_actor
from heldout_wuji_cost import summarize
from run_wuji_heldout import comparison_schedule,store


def check_cases(good,bad,rejected,out):
    ok,broken=inspect_actor(good),inspect_actor(bad)
    if not ok["usable"] or broken["usable"] or not broken["nonfinite_leaves"]:
        raise ValueError("Expected known finite and nonfinite development policy examples")
    preserved=json.loads(rejected.read_bytes())
    record=preserved["numerical_failure"]
    assert preserved["checkpoint_sha256"]==broken["checkpoint_sha256"]
    assert preserved["status"]=="numerical_actor_output" and not record["rejected_action_applied"]
    assert record["applied_intervals"]==record["applied_substeps"]==0 and not preserved["realizations"] and record["scene_finite"]
    protocol=dict(primary="fixture_1",supplementary=["fixture_2","fixture_3"],
                  zero_shot=["shared_mf","shared_wm"],adaptation=["cold_single","shared_mf","shared_wm"])
    labels=[protocol["primary"],*protocol["supplementary"]]
    frozen={s:{"source_status":"scorable"} for s in labels}
    plan=dict(protocol=protocol,subjects=[{"subject":s} for s in labels],budget=96,steps=[24,48,72,96],
              upfront_exclusive_card_s={"cold_single":0.,"shared_mf":101.,"shared_wm":201.},
              teacher_health={"shared_mf":ok,"shared_wm":broken})
    planned=list(comparison_schedule(plan,frozen))
    assert not any(m=="shared_wm" for _,_,m,_,_ in planned)
    assert sum(k=="adapt" and m=="cold_single" for k,s,m,n,d in planned)==12
    assert sum(k=="evaluate" and m=="shared_mf" and n==0 for k,s,m,n,d in planned)==3
    ledger=[dict(subject=s,method=None,kind="source_evidence",known_wall_s=2.,unknown_cost=False,card=False,
                 steps=None,outcome=None) for s in labels]
    results=summarize(plan,ledger,{s:"scorable" for s in labels})["methods"]["shared_wm"]
    assert results["assigned"]==results["assigned_but_not_scored_invalid_teacher"]==3
    assert results["ever_verified"]==0 and not results["through_final_budget"]
    assert results["target_gpu_h_per_verified"] is None and results["pretraining_exclusive_card_s"]==201
    assert all(x["scored"]==[] and x["target_training_s"]==0 for x in results["subjects"].values())
    plan["teacher_health"]["shared_mf"]=broken
    schedule=list(comparison_schedule(plan,frozen))
    assert all(m in (None,"cold_single") for _,_,m,_,_ in schedule)
    store(out,dict(scope="CPU development checkpoints and a synthetic accounting ledger. No reserved source or GPU.",
                   finite=ok,nonfinite=broken,zero_target_commands_applied=True,confirmed_finite_development_state=True,
                   all_assignments_retained=3,one_bad_teacher_other_methods_continue=True,
                   two_bad_teachers_cold_method_continues=True,no_replacement_or_fabricated_score=True,
                   historical_pretraining_cost_retained=True))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("good","bad","rejected","output"):p.add_argument(arg,type=Path)
    a=p.parse_args();check_cases(a.good,a.bad,a.rejected,a.output)
