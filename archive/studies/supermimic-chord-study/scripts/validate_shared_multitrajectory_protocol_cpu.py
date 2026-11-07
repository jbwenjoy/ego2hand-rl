"""Independently validate the new draft bundle, source data and aggregate math."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    p = Path(p)
    if p.exists(): raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def validate(p, inventory):
    if p["GPU_authorized"] or p["training_executed"] or p["teacher_dataset_generated"]: raise ValueError("CPU draft cannot authorize/claim execution")
    if inventory["source_count"] != 10 or len(inventory["rows"]) != 10 or inventory["verified_independent_ego_demonstrations"] != 0: raise ValueError("Source/ego count")
    if len({r["split_group"] for r in inventory["rows"]}) != 1 or len({r["lineage"] for r in inventory["rows"]}) != 1: raise ValueError("Conservative shared lineage")
    trajectory = p["data_tracks"]["controlled_within_lineage_holdout"]
    if set(trajectory["training_pool"]) & (set(trajectory["validation"]) | set(trajectory["test"])): raise ValueError("External trajectory split overlap")
    if not trajectory["no_independent_video_generalization"] or not p["split_and_statistics"]["native_env_holdout_not_trajectory_holdout"]: raise ValueError("Overclaimedholdout")
    if p["data_tracks"]["N20"]["status"] != "NOT_AVAILABLE": raise ValueError("Invented20trajectories")
    cursor = 0
    for r in inventory["rows"]:
        if sha(r["source_file"]) != r["source_sha256"]: raise ValueError("Sourcehash")
        if r["source_frame_range_half_open"] != [0,r["frames"]] or r["global_source_frame_range_half_open"] != [cursor,cursor+r["frames"]]: raise ValueError("Framewindow/prefix")
        cursor += r["frames"]
        exp = r["observation_export"]
        if sha(exp["path"]) != exp["sha256"]: raise ValueError("Observationhash")
        with np.load(exp["path"], allow_pickle=False) as z:
            if set(z.files) != {"time_s","hand_landmarks_source_w","object_pose_source_w"}: raise ValueError("Observationlabel leakage")
            if z["hand_landmarks_source_w"].shape != (r["frames"],21,3) or z["object_pose_source_w"].shape != (r["frames"],7): raise ValueError("Observationshape")
            if not np.array_equal(z["time_s"], np.arange(r["frames"])*.02): raise ValueError("Observationtime")
            with np.load(r["source_file"],allow_pickle=False) as original:
                if not np.array_equal(z["hand_landmarks_source_w"],original["mediapipe_landmarks_w"]) or not np.array_equal(z["object_pose_source_w"],original["object_pose_w"]): raise ValueError("Observationvalues")
    if cursor != inventory["total_frames"] or cursor >= 2**24: raise ValueError("Frameoffset precision")
    for binding in p["source_bindings"].values():
        if sha(binding["path"]) != binding["sha256"]: raise ValueError("Implementation source drift")
    # Independently reproduce actual runner's fractional collection accumulator.
    budget_checks = {}
    for ns, q in p["budget_DRAFT"]["per_N_aggregate_equal_across_objective_and_sharing_conditions"].items():
        n = int(ns); accumulator = 0.; collected = q["shared_warmup_steps"]
        for _ in range(q["shared_max_iterations"]):
            accumulator += 1/.4; step = math.floor(accumulator); accumulator -= step; collected += step
        actual = dict(real_transitions=128*collected, WM_optimizer_steps=2*q["shared_max_iterations"],
                      failure_head_optimizer_steps=2*q["shared_max_iterations"],
                      actor_optimizer_steps=q["shared_max_iterations"]-q["shared_wm_only_iters"])
        actual["critic_optimizer_steps"] = 4*actual["actor_optimizer_steps"]
        for key,value in actual.items():
            if value != q[key]: raise ValueError("Sharedaggregate budget mismatch")
        unit = {"real_transitions":648192,"WM_optimizer_steps":4000,"failure_head_optimizer_steps":4000,"actor_optimizer_steps":1800,"critic_optimizer_steps":7200}
        if any(actual[k]!=v*n for k,v in unit.items()): raise ValueError("Independent/shared aggregate mismatch")
        if q["independent_runs"] != n or q["independent_iterations_each"] != 2000 or q["independent_warmup_each"] != 64: raise ValueError("Independentrecipe")
        budget_checks[ns] = actual
    return {"source_count":10,"frames":cursor,"all_source_hashes":True,"exports_only_observations":True,"export_values_exact_to_source":True,"source_bindings_current":True,"independent_vs_shared_DRAFT_aggregate_budget":budget_checks}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol",type=Path,default=ROOT/"protocols/shared-multitrajectory-controlled-v1.json")
    ap.add_argument("--output",type=Path,default=ROOT/"outputs/shared_multitrajectory_inventory_v1/validation_v1")
    a=ap.parse_args(); p=read(a.protocol); inventory=read(p["inventory"]["path"])
    if sha(p["inventory"]["path"])!=p["inventory"]["sha256"]: raise ValueError("Inventorybinding")
    result=validate(p,inventory)
    negatives={}
    for name in ("claim_GPU_authorized","claim_independent_ego_N","relax_conservative_lineage","leak_test_into_train","invent_N20","forget_shared_extra_warmup","changed_source_SHA"):
        pp,ii=copy.deepcopy(p),copy.deepcopy(inventory)
        if name=="claim_GPU_authorized":pp["GPU_authorized"]=True
        elif name=="claim_independent_ego_N":ii["verified_independent_ego_demonstrations"]=10
        elif name=="relax_conservative_lineage":ii["rows"][-1]["split_group"]="falsely_independent"
        elif name=="leak_test_into_train":pp["data_tracks"]["controlled_within_lineage_holdout"]["training_pool"].append("clip_010")
        elif name=="invent_N20":pp["data_tracks"]["N20"]["status"]="READY"
        elif name=="forget_shared_extra_warmup":pp["budget_DRAFT"]["per_N_aggregate_equal_across_objective_and_sharing_conditions"]["5"]["shared_warmup_steps"]=64
        else:ii["rows"][0]["source_sha256"]="0"*64
        try:validate(pp,ii)
        except ValueError as error:negatives[name]={"rejected":True,"reason":str(error)}
        else:raise AssertionError("Invaliddraft accepted: "+name)
    pilot=ROOT/"outputs/pilot_rotating_reference_v2/reference.json"; acc=read(pilot)["task_acceptance"]
    common_keys=("position_max_m","shaft_angle_max_deg","penetration_max_m","progress_abs_error_max_deg","progress_endpoint_error_max_deg","angular_step_max_deg","drop_center_below_wrist_m","allow_object_environment_contacts","actuator_load_ratio_max","load_charge_max","charge_decay_time_s","target_speed_abs_max_rad_s","joint_limit_margin_rad")
    template={"schema":"shared_controlled_task_acceptance_DRAFT:1","status":"DRAFT_NOT_APPLIED_TO_NEW_REFERENCES","source_pilot_reference":{"path":str(pilot),"sha256":sha(pilot)},
              "candidate_common_thresholds":{k:acc[k] for k in common_keys},
              "trajectory_specific_required":{"reference_npz_sha256":None,"task_classification":None,"progress_frame_fit":None,"orientation_metric":"directed_axis_x","new_init_physics_registry_SHA256":None,"control_context_SHA256":None},
              "sampling":{"policy_hz":50,"physics_hz":200,"evaluation_duration_s":3.,"states":151,"intervals":150,"continuous_safety_qualified":False},
              "engineering_qualification":["source/103ornewformat/reference/controller identity and actualu/EMA/t0 join","no auto_reset/RSI/noise/DR/exogenous support","stickyoverflow0 terminalsingle read/no resets","fullfinite151states; failures latch and fullhorizon continues","all actualtarget/actualjoint/torque/load/penetration fields independently recomputed"],
              "initialization_gate":{"status":"DRAFT_REQUIRES_PHYSICAL_MEASUREMENT","short_hold_s":.3,"samples_every_physics_step":True,"zero_vs_candidateFD_velocity":"Separate explicitlydesigned conditions; never call finite_difference velocity a video measurement","failure_does_not_disprove_entire_motion":True},
              "task_scope":"Do not copy pilot007 progressaxis/targetprogress into otherclips. Fit/freeze beforemodelcomparison; insufficientrotation gets explicitineligible rotatingtask status, not replacement by an easier static task.",
              "primary_hierarchy":{"trainingseed_N":3,"process_repeats_per_checkpoint_trajectory":3,"checkpoint_trajectory_pass":"all3pass","no_best_repeat":True,"initially_unqualified_trajectory":"reportcoverage/PENDING, do not turn into taskFAIL/drop silently"},
              "GPU_authorized":False}
    template_path=ROOT/"protocols/shared-multitrajectory-task-template-v1.json";write(template_path,template)
    a.output.mkdir(parents=True,exist_ok=False)
    check={"status":"PASS_CPU_BUNDLE_VALIDATION_ONLY","protocol_sha256":sha(a.protocol),"validator_source_sha256":sha(__file__),"validated":result,"negative_cases":negatives,"task_template_sha256":sha(template_path),"GPU_used":False,"new_physics_steps":0,"training_executed":False}
    write(a.output/"report.json",check)
    bundle={"schema":"shared_multitrajectory_controlled_bundle:1","status":"DRAFT_NOT_EXECUTION_AUTHORIZATION",
            "artifacts":{name:{"path":str(path),"sha256":sha(path)} for name,path in (("main_protocol",a.protocol.resolve()),("task_acceptance_template",template_path),("inventory",Path(p["inventory"]["path"])),("documentation",ROOT/"docs/shared-multitrajectory-protocol-v1.md"),("producer",ROOT/"scripts/prepare_shared_multitrajectory_protocol_cpu.py"),("validator",Path(__file__)),("CPU_validation",a.output/"report.json"))},
            "physical_initializations_qualified_for_observation_route":0,"teacher_dataset_generated":False,"GPU_authorized":False}
    write(ROOT/"protocols/shared-multitrajectory-controlled-v1-bundle.json",bundle)
    print(json.dumps({"status":check["status"],"negative_cases":len(negatives),"frames":result["frames"],"bundle":str(ROOT/"protocols/shared-multitrajectory-controlled-v1-bundle.json")},ensure_ascii=False))


if __name__=="__main__":main()
