"""Fold a held-out ledger without pooling subjects, auxiliary stress or unverified outcomes."""
from collections import defaultdict
from heldout_adaptation_health import allow_after_stop


def summarize(plan, entries, source_status):
    participants=[r["subject"] for r in plan["subjects"]]
    roles=plan["protocol"]["adaptation"]
    common=defaultdict(lambda:{"total_s":0.,"card_s":0.,"unknown":False,"unknown_card":False,"by_kind_s":defaultdict(float)})
    per={(s,m):{"target_s":0.,"card_s":0.,"stress_s":0.,"unknown":False,"unknown_card":False,"scored":[],"disturbance":[],
                 "zero_shot_evaluation_s":0.,"target_training_s":0.,"adapted_actor_evaluation_s":0.,
                 "first_verified_target_s":None,"qualified":False,"adaptation_stop":None,
                 "maximum_observed_trained_step":0,"checkpoint_scan_cpu_s_included":0.} for s in participants for m in roles}
    for e in entries:
        subject,method=e["subject"],e.get("method")
        if subject not in participants: raise ValueError("Unknown subject in ledger")
        if method is None:
            x=common[subject];x["total_s"]+=e["known_wall_s"]
            x["by_kind_s"][e["kind"]]+=e["known_wall_s"]
            if e["card"]:x["card_s"]+=e["known_wall_s"]
            x["unknown"] |= e["unknown_cost"]
            x["unknown_card"] |= e["card"] and e["unknown_cost"]
            continue
        x=per[subject,method]
        if not allow_after_stop(x["adaptation_stop"],e["kind"],e["steps"]):
            raise ValueError("A target branch ran work after a confirmed nonresumable checkpoint")
        health=e.get("checkpoint_health")
        if health is not None:
            if e["kind"]!="adapt" or health["step"]!=e["steps"]:raise ValueError("Misbound adaptation health record")
            x["checkpoint_scan_cpu_s_included"]+=health.get("inspection_cpu_wall_s",0.)
            x["maximum_observed_trained_step"]=max(x["maximum_observed_trained_step"],e["steps"])
            if not health["resume_usable"]:
                if x["adaptation_stop"] is not None:raise ValueError("Duplicate terminal target checkpoint")
                x["adaptation_stop"]=dict(health,stopped_step=e["steps"],
                             unreached_budget_steps=[i for i in plan["steps"] if i>e["steps"]])
        if e["kind"]=="disturbance":
            x["stress_s"]+=e["known_wall_s"]
            if e.get("outcome"):x["disturbance"].append(dict(steps=e["steps"],result=e["outcome"]))
            continue
        x["target_s"]+=e["known_wall_s"]
        if e["kind"]=="adapt":x["target_training_s"]+=e["known_wall_s"]
        if e["kind"]=="evaluate":
            key="zero_shot_evaluation_s" if e["steps"]==0 else "adapted_actor_evaluation_s"
            x[key]+=e["known_wall_s"]
        if e["card"]:x["card_s"]+=e["known_wall_s"]
        x["unknown"]|=e["unknown_cost"]
        x["unknown_card"] |= e["card"] and e["unknown_cost"]
        if e["kind"]!="evaluate" or not e.get("outcome"):continue
        outcome=e["outcome"]
        observed=outcome.get("numerical_stop")
        if observed:
            if observed["step"]!=e["steps"] or observed["classification"]!="nonfinite_target_actor_output":
                raise ValueError("Misbound target policy-output stop")
            prior=x["adaptation_stop"]
            x["adaptation_stop"]=dict(observed,stopped_step=e["steps"],prior_resume_state_evidence=prior,
                                      unreached_budget_steps=[i for i in plan["steps"] if i>e["steps"]])
        if outcome["status"]=="passed":
            if outcome.get("full_repeated") is not True:raise ValueError("Unverified physical success")
            x["qualified"]=True
        record=dict(steps=e["steps"],stage="zero_shot" if e["steps"]==0 else "adapted",
                    historical_strict=outcome.get("historical_strict"),
                    historical_repeat=outcome.get("historical_repeat"),status=outcome["status"],
                    target_exclusive_s=x["target_s"],end_to_end_s=x["target_s"]+common[subject]["total_s"])
        if x["qualified"] and "first_verified_check" not in x:
            x["first_verified_target_s"]=None if x["unknown"] else record["target_exclusive_s"]
            x["first_verified_check"]=record
        x["scored"].append(record)
    output={}
    for m in roles:
        rows={}
        teacher=plan.get("teacher_health",{}).get(m)
        unavailable=bool(teacher and not teacher["usable"])
        precheck_s=teacher.get("inspection_cpu_wall_s",0.) if teacher else 0.
        for s in participants:
            x=per[s,m];sf=source_status.get(s)
            rows[s]=dict(**x,source_status=sf,common_source_s=common[s]["total_s"],
                         common_cost_components_s=dict(common[s]["by_kind_s"]),
                         end_to_end_known_s=x["target_s"]+common[s]["total_s"],
                         target_gpu_hours_known=(x["card_s"]+common[s]["card_s"])/3600,
                         zero_shot_status=next((v["status"] for v in x["scored"] if v["steps"]==0),None),
                         final_budget_status=next((v["status"] for v in x["scored"] if v["steps"]==plan["budget"]),None))
            if unavailable:
                if x["scored"] or x["target_training_s"] or x["zero_shot_evaluation_s"]:
                    raise ValueError("Invalid frozen teacher was nevertheless scored or adapted")
                rows[s].update(method_availability=teacher["status"],scored_on_target=False,
                               zero_shot_status="not_run_invalid_frozen_teacher",
                               final_budget_status="not_run_invalid_frozen_teacher")
            stop=x["adaptation_stop"]
            if stop:
                result_at_stop=next((v["status"] for v in x["scored"] if v["steps"]==stop["step"]),None)
                status=result_at_stop or (None if stop["actor_usable"] else "not_run_invalid_target_actor")
                rows[s]["adaptation_stop"]=dict(stop,actor_evaluation_at_stop=status)
                if stop["step"]<plan["budget"]:
                    rows[s]["final_budget_status"]="unreachable_after_"+stop["classification"]
                elif not stop["actor_usable"]:
                    rows[s]["final_budget_status"]=stop["classification"]
            first=rows[s].get("first_verified_check")
            if first:
                rows[s]["first_verified_end_to_end_s"]=None if x["unknown"] or common[s]["unknown"] else first["end_to_end_s"]
                prior=[r for r in x["scored"] if r["steps"]<first["steps"]]
                rows[s]["previous_nonpassing_check_target_s"]=prior[-1]["target_exclusive_s"] if prior else None
        t=sum(x["end_to_end_known_s"] for x in rows.values())+precheck_s
        gpu=sum(x["target_gpu_hours_known"] for x in rows.values())
        count=sum(x["qualified"] for x in rows.values())
        pre=plan["upfront_exclusive_card_s"][m]
        pre_low,pre_high=plan.get("upfront_exclusive_card_s_range",{}).get(m,[pre,pre])
        unknown_target=any(common[s]["unknown"] or rows[s]["unknown"] for s in participants)
        unknown_card=any(common[s]["unknown_card"] or rows[s]["unknown_card"] for s in participants)
        unknown=unknown_target or pre is None
        n=len(participants)
        def settled(x):
            if x["source_status"]=="unscorable" or unavailable:return True
            stop=x["adaptation_stop"]
            if stop:return not stop["actor_usable"] or stop["actor_evaluation_at_stop"] is not None
            return x["final_budget_status"] is not None
        closed=all(settled(x) for x in rows.values())
        finished=closed and not unavailable and all(not x["adaptation_stop"] or
                    x["adaptation_stop"]["step"]==plan["budget"] for x in rows.values())
        clock=sum(common[s]["total_s"] for s in participants)+precheck_s
        seen=set();curve=[]
        for e in entries:
            if e.get("method")!=m or e["kind"]=="disturbance":continue
            clock+=e["known_wall_s"]
            if e["kind"]=="evaluate" and e.get("outcome"):
                if e["outcome"]["status"]=="passed":seen.add(e["subject"])
                curve.append(dict(end_to_end_known_s=clock,subject=e["subject"],target_step=e["steps"],
                                  distinct_verified=len(seen)))
        first=next((k["end_to_end_known_s"] for k in curve if k["distinct_verified"]),None)
        output[m]=dict(subjects=rows,assigned=n,ever_verified=count,through_final_budget=finished,
                      target_branches_accounted=closed,
                      numerically_stopped_targets=sum(x["adaptation_stop"] is not None for x in rows.values()),
                      method_availability=teacher["status"] if teacher else "available",
                      frozen_teacher_precheck=teacher,teacher_inspection_cpu_s_once=precheck_s,
                      assigned_but_not_scored_invalid_teacher=n if unavailable else 0,
                      zero_shot_verified=sum(x["zero_shot_status"]=="passed" for x in rows.values()),
                      final_budget_verified=sum(x["final_budget_status"]=="passed" for x in rows.values()),
                      unscorable=sum(x["source_status"]=="unscorable" for x in rows.values()),
                      event_curve=curve,first_verified_end_to_end_s=first if not unknown_target else None,
                      first_verified_including_pretraining_s=pre+first if first is not None and not unknown else None,
                      primary_ever_verified=int(rows[plan["protocol"]["primary"]]["qualified"]),
                      supplementary_ever_verified=sum(rows[s]["qualified"] for s in plan["protocol"]["supplementary"]),
                      target_end_to_end_known_s=t,target_gpu_hours_known=gpu,pretraining_exclusive_card_s=pre,
                      with_pretraining_end_to_end_known_s=None if pre is None else t+pre,
                      amortized_end_to_end_s_per_assignment=(t+pre)/n if closed and not unknown else None,
                      target_gpu_h_per_verified=gpu/count if count and not unknown_card and closed else None,
                      with_pretraining_gpu_h_per_verified=(gpu+pre/3600)/count if count and pre is not None and not unknown_card and closed else None,
                      pretraining_exclusive_card_s_range=[pre_low,pre_high],
                      with_pretraining_gpu_h_per_verified_range=(
                          [None if pre_low is None else (gpu+pre_low/3600)/count,
                           None if pre_high is None or unknown_card else (gpu+pre_high/3600)/count]
                          if count and closed else None),
                      with_pretraining_s_per_assignment_range=(
                          [None if pre_low is None else (t+pre_low)/n,
                           None if pre_high is None or unknown_target else (t+pre_high)/n]
                          if closed else None),
                      incomplete_costs=unknown)
    return dict(methods=output,reporting="Assigned primary and supplements are retained separately. First pass is first verified at a scheduled evaluation. Common reference/force calibration once per target per counterfactual method. Stress is excluded; source RGB capture/annotation before this experiment has no measured cost.")
