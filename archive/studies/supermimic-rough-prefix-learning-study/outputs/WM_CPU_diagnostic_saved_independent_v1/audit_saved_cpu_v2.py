"""Independent NumPy/stdlib review of existing WM diagnostic outputs only.

No Torch/MuJoCo imports, checkpoint deserialization, WM calls, AD or FD reruns.
Numerical comparison budgets are declared here before execution: 32*eps32
relative + 8*eps32 absolute; rotation adds a separately reported 64*eps32
radian arithmetic allowance for FP32 native so3_log vs FP64 atan2 geometry.
These are saved-arithmetic review budgets, never physical acceptance gates.
V2 preserves v1 and corrects two harness comparisons: an archived JSON copy is
checked semantically plus original-source raw pin, and a cancellation-prone
reward difference is checked by the exact saved FP32 term arithmetic, with the
independent FP64 difference retained as a diagnostic rather than changing tol.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "outputs/WM_onpolicy_CPU_diagnostic_v1"
MODELS = ["A1L", "P_ONLY", "A2"]
PATHS = ["A0", "A1L", "P_ONLY", "A2"]
EPS32 = float(np.finfo(np.float32).eps)
REL = 32 * EPS32
ABS = 8 * EPS32
ROT_ABS = 64 * EPS32
START = time.perf_counter()
checks = {}
details = {}
numeric = {}
HASHES = {}
RESULT = {"schema": "rough_WM_CPU_saved_independent_audit:1"}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    p = Path(p).resolve(strict=True)
    if p not in HASHES:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1024 * 1024), b""):
                h.update(b)
        HASHES[p] = h.hexdigest()
    return HASHES[p]


def pin(p):
    p = Path(p).resolve(strict=True)
    return {"path": p.as_posix(), "sha256": sha(p)}


def loadnp(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def check(name, ok, detail=None):
    checks[name] = bool(ok)
    if detail is not None:
        details[name] = detail


def close(category, actual, expected, rotation=False):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        numeric.setdefault(category, {"count": 0, "max_abs_difference": 0., "max_budget_fraction": 0., "failed": []})["failed"].append("shape_or_nonfinite")
        return False
    diff = np.abs(a-e)
    budget = (ROT_ABS if rotation else ABS) + REL * np.abs(e)
    n = numeric.setdefault(category, {"count": 0, "max_abs_difference": 0., "max_budget_fraction": 0., "failed": []})
    n["count"] += a.size
    n["max_abs_difference"] = max(n["max_abs_difference"], float(np.max(diff, initial=0)))
    n["max_budget_fraction"] = max(n["max_budget_fraction"], float(np.max(diff/budget, initial=0)))
    ok = bool(np.all(diff <= budget))
    if not ok:
        n["failed"].append({"max_abs_difference": float(np.max(diff)), "expected_max_abs": float(np.max(np.abs(e)))})
    return ok


def all_pins(v):
    if isinstance(v, dict):
        if set(v) == {"path", "sha256"}:
            yield v
        else:
            for x in v.values():
                yield from all_pins(x)
    elif isinstance(v, list):
        for x in v:
            yield from all_pins(x)


def quatmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q/np.linalg.norm(q, axis=-1, keepdims=True)
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                     2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                     2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3,3))


def expmat(v):
    v = np.asarray(v, dtype=np.float64)
    t2 = np.sum(v*v, axis=-1)
    t = np.sqrt(t2)
    a = np.divide(np.sin(t), t, out=np.ones_like(t), where=t>1e-12)
    b = np.divide(1-np.cos(t), t2, out=np.full_like(t, .5), where=t2>1e-16)
    x,y,z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    k = np.stack((zero,-z,y,z,zero,-x,-y,x,zero), axis=-1).reshape(v.shape[:-1]+(3,3))
    return np.eye(3)+a[...,None,None]*k+b[...,None,None]*(k@k)


def rotation_angle(pred, truth):
    relative = np.swapaxes(pred, -1, -2)@truth
    skew = np.stack((relative[...,2,1]-relative[...,1,2], relative[...,0,2]-relative[...,2,0],
                     relative[...,1,0]-relative[...,0,1]), axis=-1)*.5
    sine = np.linalg.norm(skew, axis=-1)
    cosine = (np.trace(relative, axis1=-2, axis2=-1)-1)*.5
    return np.arctan2(sine, cosine)


def metrics(pred, truth, lower, upper):
    pred = pred.astype(np.float64); truth = truth.astype(np.float64)
    e = truth-pred; qe=e[:,:20]; qde=e[:,20:40]
    out = {"rows": 15, "q_RMSE_rad": np.sqrt(np.mean(qe*qe)), "q_max_abs_rad": np.max(np.abs(qe)),
           "qd_RMSE_rad_s": np.sqrt(np.mean(qde*qde)), "qd_max_abs_rad_s": np.max(np.abs(qde)),
           "q_per_joint_RMSE_rad": np.sqrt(np.mean(qe*qe, axis=0)), "q_per_joint_max_abs_rad": np.max(np.abs(qe),axis=0),
           "qd_per_joint_RMSE_rad_s": np.sqrt(np.mean(qde*qde,axis=0)), "qd_per_joint_max_abs_rad_s": np.max(np.abs(qde),axis=0),
           "predicted_per_joint_min_rad": pred[:,:20].min(0), "predicted_per_joint_max_rad":pred[:,:20].max(0),
           "true_per_joint_min_rad":truth[:,:20].min(0), "true_per_joint_max_rad":truth[:,:20].max(0)}
    groups = {"position_m":np.linalg.norm(e[:,40:43],axis=-1),
              "rotation_rad":rotation_angle(pred[:,43:52].reshape(-1,3,3),truth[:,43:52].reshape(-1,3,3)),
              "linear_velocity_m_s":np.linalg.norm(e[:,52:55],axis=-1),
              "angular_velocity_rad_s":np.linalg.norm(e[:,55:58],axis=-1)}
    for key,v in groups.items():
        out.update({key+"_mean":v.mean(),key+"_RMS":np.sqrt(np.mean(v*v)),key+"_max":v.max()})
    excess=lambda z: np.maximum(np.maximum(lower-z[:,:20],z[:,:20]-upper),0)
    px,tx = excess(pred), excess(truth)
    out.update(predicted_joint_excess_max_rad=px.max(),true_joint_excess_max_rad=tx.max(),
               predicted_joint_over1mrad_count=int((px>.001).sum()),true_joint_over1mrad_count=int((tx>.001).sum()))
    phase=[{"q_RMSE_rad":float(np.sqrt(np.mean(qe[k]**2))), "qd_RMSE_rad_s":float(np.sqrt(np.mean(qde[k]**2))),
            **{key:float(v[k]) for key,v in groups.items()}} for k in range(15)]
    return out,phase,px>.001,tx>.001


def object_terms(state,gp,gR):
    state=state.astype(np.float64); p=state[:,40:43]; R=state[:,43:52].reshape(-1,3,3)
    off=.1*R[:,:,0]; goff=.1*gR[:,:,0]
    endpoint=(np.linalg.norm((p+off)-(gp+goff),axis=-1)+np.linalg.norm((p-off)-(gp-goff),axis=-1))*.5
    return np.stack((2*np.exp(-(np.linalg.norm(p-gp,axis=-1)/.05)**2),8*np.exp(-(endpoint/.04)**2)),axis=-1)


try:
    source_report=read(DATA/"report.json"); plan=read(DATA/"runplan.json"); protocol=read(DATA/"protocol.json")
    result_names=("report.json","runplan.json","protocol.json","metrics.json","phase_errors.csv","reward_rows.json","FD_rows.json",
                  "predictions_truth_baselines_gradients.npz","codec_receipts.json","loader_A1L.json","loader_P_ONLY.json","loader_A2.json")
    original_pins={name:pin(DATA/name) for name in result_names}
    check("complete_registered_diagnostic",source_report["status"]=="COMPLETED_CPU_DIAGNOSTIC_NOT_PHYSICAL_CERTIFICATE"
          and source_report["one_step_prediction_rows"]==180 and source_report["FD_pairs"]==288)
    check("runplan_protocol_entry_exact_pins",source_report["runplan"]["sha256"]==sha(source_report["runplan"]["path"])
          and read(source_report["runplan"]["path"])==plan and source_report["protocol"]["sha256"]==sha(source_report["protocol"]["path"])
          and read(source_report["protocol"]["path"])==protocol
          and source_report["entry"]["sha256"]==sha(DATA/"entry__diagnose_saved_world_models_cpu.py"))
    bad_pins=[]
    for p in list(all_pins(plan))+list(all_pins(protocol)):
        if sha(p["path"])!=p["sha256"]:bad_pins.append(p)
    check("all_registered_input_source_pins_match_after_execution",not bad_pins,{"unique_hashed_files":len(HASHES),"mismatches":bad_pins})
    for key,copyname in (("entry","entry__diagnose_saved_world_models_cpu.py"),("state_codec","state_codec__saved_state_codec_v3.py"),("WM_loader","WM_loader__world_model_checkpoint_v2.py")):
        check("executed_source_copy_"+key,sha(DATA/copyname)==plan["source_pins"][key]["sha256"])
    z=loadnp(DATA/"predictions_truth_baselines_gradients.npz")
    check("all92_arrays_finite_and_expected_count",len(z)==92 and all(np.isfinite(a).all() for a in z.values()))
    proof_arrays=loadnp(ROOT/"outputs/saved_state_codec_cpu_v4/reconstructed_native_transitions.npz")
    codecs=read(DATA/"codec_receipts.json")
    check("all_four_paths_no_quality_selection",[x["method"] for x in codecs]==PATHS and protocol["scope"]["trajectory_order"]==PATHS)
    check("all_saved_codec_outputs_match_previous_actual32_proof",all(np.array_equal(z[f"{path}__{field}"],proof_arrays[f"{path}__{field}"])
          for path in PATHS for field in ("x","x_next","u")))
    raw={}; bounds={}; goals={}
    ref=loadnp(protocol["inputs"]["reference_arrays"]["path"])
    for path,receipt in zip(PATHS,codecs):
        raw[path]=loadnp(Path(receipt["run"])/"raw_states.npz")
        desc=read(Path(receipt["run"])/"controller_descriptor.json")
        outerlo=np.asarray(desc["target_lower_rad"],np.float32); outerhi=np.asarray(desc["target_upper_rad"],np.float32)
        lower=raw[path]["joint_lower_rad"].astype(np.float32);upper=raw[path]["joint_upper_rad"].astype(np.float32)
        check(path+"_hard_and_outer_control_bounds_equal_this_asset",np.array_equal(lower,outerlo) and np.array_equal(upper,outerhi))
        bounds[path]=(lower,upper,outerlo+np.float32(.02),outerhi-np.float32(.02))
        events=read(Path(receipt["run"])/"control_events.json")
        check(path+"_15_actualu_four_substeps_and_knots",np.array_equal(z[path+"__u"],np.asarray([e["actualu"][0] for e in events],np.float32))
              and all(np.array_equal(raw[path]["ctrl_rad"][4*k+1:4*k+5].astype(np.float32),np.broadcast_to(z[path+"__u"][k],(4,20))) for k in range(15))
              and receipt["source_indices_k"]==list(range(0,60,4)) and receipt["source_indices_kplus1"]==list(range(4,61,4)))
        check(path+"_state_shape_dtype_and_chain",all(z[f"{path}__{key}"].shape==shape and z[f"{path}__{key}"].dtype==np.float32
              for key,shape in (("x",(15,58)),("x_next",(15,58)),("u",(15,20))))
              and np.array_equal(z[path+"__x"][1:],z[path+"__x_next"][:-1]))
        palm=np.asarray(receipt["palm"]["FP32_pose"],np.float64); inv=quatmat(palm[3:]).T
        poses=ref["mapped_object_pose_w"].astype(np.float32).astype(np.float64)
        goals[path]=((poses[:,:3]-palm[:3])@inv.T, inv[None]@quatmat(poses[:,3:]))
        x=z[path+"__x"];cv=x.astype(np.float64).copy()
        cv[:,:20]+=.02*x[:,20:40].astype(np.float64);cv[:,40:43]+=.02*x[:,52:55].astype(np.float64)
        cv[:,43:52]=(expmat(.02*x[:,55:58].astype(np.float64))@x[:,43:52].reshape(-1,3,3).astype(np.float64)).reshape(-1,9)
        check(path+"_identity_exact",np.array_equal(z[path+"__identity"],x))
        check(path+"_CV_left_Exp_fixed_palm_worldomega",close("CV_left_Exp",z[path+"__constant_velocity"],cv))
    metrics_rows=read(DATA/"metrics.json"); reward_rows=read(DATA/"reward_rows.json");fd=read(DATA/"FD_rows.json")
    phase_csv=list(csv.DictReader((DATA/"phase_errors.csv").open(newline="",encoding="utf-8")))
    metric_index={(r["model"],r["trajectory"]):r for r in metrics_rows}
    expected_groups={(m,p) for m in MODELS+["identity","constant_velocity"] for p in PATHS}
    check("all20_WM_and_baseline_groups_unique",len(metric_index)==len(metrics_rows)==20 and set(metric_index)==expected_groups)
    check("summary_exact12_WM_groups_and_all180",len(source_report["groups"])==12 and
          {(r["model"],r["trajectory"]) for r in source_report["groups"]}=={(m,p) for m in MODELS for p in PATHS}
          and all(r==metric_index[(r["model"],r["trajectory"])] for r in source_report["groups"]) and sum(r["rows"] for r in source_report["groups"])==180)
    phase_index={(r["model"],r["trajectory"],int(r["phase"])):r for r in phase_csv}
    check("all300_phase_rows_unique",len(phase_index)==len(phase_csv)==300)
    metric_ok=True; phase_ok=True;mask_ok=True;logits_ok=True
    for (method,path),row in metric_index.items():
        pred=z[path+"__"+method] if method in ("identity","constant_velocity") else z[method+"__"+path+"__prediction"]
        calculated,perphase,pmask,tmask=metrics(pred,z[path+"__x_next"],*bounds[path][:2])
        for key,value in calculated.items():metric_ok=close("metrics_rotation" if key.startswith("rotation_rad") else "metrics_nonrotation",row[key],value,key.startswith("rotation_rad")) and metric_ok
        for k,r in enumerate(perphase):
            saved=phase_index[(method,path,k)]
            for key,value in r.items():phase_ok=close("phase_rotation" if key=="rotation_rad" else "phase_nonrotation",float(saved[key]),value,key=="rotation_rad") and phase_ok
        if method in MODELS:
            prefix=method+"__"+path
            check(prefix+"_prediction_logits_gradient_shapes",pred.shape==(15,58) and z[prefix+"__selfcollision_logits"].shape==(15,15) and z[prefix+"__object_gradient"].shape==(15,20))
            mask_ok=(np.array_equal(z[prefix+"__predicted_over1mrad"],pmask) and np.array_equal(z[prefix+"__true_over1mrad"],tmask)) and mask_ok
            logits=z[prefix+"__selfcollision_logits"].astype(np.float64)
            logits_ok=close("sigmoid_probabilities",z[prefix+"__selfcollision_probability"],1/(1+np.exp(-logits))) and logits_ok
            metric_ok=close("gradient_norm",row["object_gradient_norm_mean"],np.linalg.norm(z[prefix+"__object_gradient"].astype(float),axis=-1).mean()) and metric_ok
    check("all_metrics_recomputed_from_saved_predictions",metric_ok)
    check("all_phase_errors_recomputed_from_saved_predictions",phase_ok)
    check("saved_joint_excess_masks_recomputed_exact",mask_ok)
    check("self15_probability_is_sigmoid_only",logits_ok)
    check("self15_truth_unknown_no_accuracy_BCE_F1",source_report["selfcollision_truth"].startswith("UNKNOWN") and all(
        r["selfcollision_truth"].startswith("UNKNOWN") and all(r[k] is None for k in ("selfcollision_BCE","selfcollision_F1","selfcollision_accuracy")) for r in source_report["groups"]))
    expected_reward={(m,p,k,phase) for m,p in expected_groups for k in range(15) for phase in ("k_native","kplus1_tracking")}
    reward_index={(r["model"],r["trajectory"],r["phase"],r["reference_phase"]):r for r in reward_rows}
    check("all600_reward_rows_unique_phases_k_and_kplus1",len(reward_index)==len(reward_rows)==600 and set(reward_index)==expected_reward)
    reward_ok=True;reward_fp32_ok=True;reward_cancellation_differences=[]
    for method,path in expected_groups:
        pred=z[path+"__"+method] if method in ("identity","constant_velocity") else z[method+"__"+path+"__prediction"]
        gp,gR=goals[path]
        for shift,label in ((0,"k_native"),(1,"kplus1_tracking")):
            ep=object_terms(pred,gp[shift:shift+15],gR[shift:shift+15]); ea=object_terms(z[path+"__x_next"],gp[shift:shift+15],gR[shift:shift+15])
            for k in range(15):
                row=reward_index[(method,path,k,label)]; error=ep[k].sum()-ea[k].sum()
                expected={"pred_position_term":ep[k,0],"actual_position_term":ea[k,0],"pred_endpoint_term":ep[k,1],"actual_endpoint_term":ea[k,1]}
                for key,value in expected.items():reward_ok=close("object_reward",row[key],value) and reward_ok
                ep32=np.float32(row["pred_position_term"])+np.float32(row["pred_endpoint_term"])
                ea32=np.float32(row["actual_position_term"])+np.float32(row["actual_endpoint_term"])
                saved_error=np.float32(ep32-ea32)
                scaled=np.float32(np.float32(.02)*saved_error)
                reward_fp32_ok=(float(saved_error)==row["unscaled_sum_error"] and float(scaled)==row["dt_scaled_sum_error"]) and reward_fp32_ok
                reward_cancellation_differences.append({"model":method,"trajectory":path,"phase":k,"reference_phase":label,
                    "saved_FP32_difference":row["unscaled_sum_error"],"independent_FP64_difference":float(error),
                    "absolute_difference":abs(row["unscaled_sum_error"]-float(error))})
    check("object_rewards_geometry_k_kplus1_and_dt_once",reward_ok)
    check("reward_saved_FP32_terms_sum_subtraction_dt_exact",reward_fp32_ok)
    fd_expected={(m,p,k,d,e) for m in MODELS for p in PATHS for k in (0,7,14) for d in range(4) for e in (.002,.001)}
    fd_index={(r["model"],r["trajectory"],r["phase"],r["direction"],r["epsilon"]):r for r in fd}
    check("all288_FD_cases_unique_predeclared_grid",len(fd_index)==len(fd)==288 and set(fd_index)==fd_expected)
    rng=np.random.default_rng(20261104); vector_ok=True;target_ok=True;domain_ok=True;null_ok=True;ad_ok=True;disc_ok=True;eps_ok=True
    fd_discrepancies=[]; valid_by_group={}
    for method in MODELS:
        for path in PATHS:
            for k in (0,7,14):
                directions=rng.standard_normal((4,20));directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
                for d,vector in enumerate(directions):
                    pair=[]
                    for eps in (.002,.001):
                        row=fd_index[(method,path,k,d,eps)];pair.append(row)
                        vector_ok=np.array_equal(np.asarray(row["vector"]),vector) and vector_ok
                        center=z[path+"__u"][k];plus=(center+eps*vector).astype(np.float32);minus=(center-eps*vector).astype(np.float32)
                        target_ok=(np.array_equal(plus,np.asarray(row["plus_target"],np.float32)) and np.array_equal(minus,np.asarray(row["minus_target"],np.float32))) and target_ok
                        lo,hi=bounds[path][2:];mask=(plus<lo)|(plus>hi)|(minus<lo)|(minus>hi);valid=not bool(mask.any())
                        domain_ok=(np.array_equal(mask,np.asarray(row["crossing_mask"],bool)) and type(row["valid_domain"]) is bool and row["valid_domain"]==valid) and domain_ok
                        grad=z[method+"__"+path+"__object_gradient"][k]
                        ad=float(np.dot(grad,vector));ad_ok=abs(ad-row["AD"])<=1e-12 and ad_ok
                        key=method+"__"+path;cnt=valid_by_group.setdefault(key,{"valid":0,"skipped":0});cnt["valid" if valid else "skipped"]+=1
                        if valid:
                            vals=[row[n] for n in ("FD","absolute_discrepancy","relative_discrepancy","AD_effective_FP32_direction")]
                            null_ok=all(type(v) in (int,float) and math.isfinite(v) for v in vals) and null_ok
                            discrepancy=abs(ad-row["FD"]);relative=discrepancy/max(abs(ad),abs(row["FD"]),1e-12)
                            effective=float(np.dot(grad.astype(float),(plus.astype(float)-minus.astype(float))/(2*eps)))
                            disc_ok=abs(discrepancy-row["absolute_discrepancy"])<=1e-12 and abs(relative-row["relative_discrepancy"])<=1e-12 and abs(effective-row["AD_effective_FP32_direction"])<=1e-12 and disc_ok
                            fd_discrepancies.append([row["absolute_discrepancy"],row["relative_discrepancy"]])
                        else:
                            null_ok=all(row[n] is None for n in ("FD","absolute_discrepancy","relative_discrepancy","AD_effective_FP32_direction")) and null_ok
                    disagreement=None if any(r["FD"] is None for r in pair) else abs(pair[0]["FD"]-pair[1]["FD"])/max(abs(pair[0]["FD"]),abs(pair[1]["FD"]),1e-12)
                    eps_ok=all((r["two_epsilon_relative_disagreement"] is None if disagreement is None else abs(r["two_epsilon_relative_disagreement"]-disagreement)<=1e-12) for r in pair) and eps_ok
    check("FD_direction_PRNG_order_and_vectors_exact",vector_ok)
    check("FD_plus_minus_FP32_targets_exact_no_clamp",target_ok)
    check("FD_domain_masks_recomputed_from_registered_bounds",domain_ok)
    check("FD_skips_keep_null_and_valid_values_finite",null_ok)
    check("FD_saved_AD_is_saved_gradient_dot_direction",ad_ok)
    check("FD_error_and_effective_direction_formula_recomputed",disc_ok)
    check("FD_two_epsilon_disagreement_and_null_recomputed",eps_ok)
    skipped=sum(not r["valid_domain"] for r in fd)
    check("FD_175_skipped_113_valid_exact",skipped==source_report["FD_domain_skipped"]==175 and len(fd)-skipped==113)
    check("source_scope_no_physics_or_contact_truth_upgrade",source_report["physics_steps_added"]==0 and source_report["GPU_used"] is False
          and source_report["physical_gradient_qualified"] is False and source_report["source_lineage_N"]==1 and source_report["independent_ego_N"]==0)
    model_receipts=[]
    for model_meta,spec in zip(source_report["models"],protocol["inputs"]["models"]):
        method=spec["method"];loader=read(DATA/("loader_"+method+".json"))
        model_receipts.append({"method":method,"loader":pin(DATA/("loader_"+method+".json")),"wrapper_before_after_equal":model_meta["before_state_sha256"]==model_meta["after_state_sha256"],
                               "wrapper_digest_namespace":"FrozenCPUWorldModel.state_dict includes physical_xi; distinct from loader WM-only digest",
                               "evidence_kind":"Pinned source-backed runtime assertion; checkpoint file bytes separately hashed, no model tensors reloaded by reviewer"})
        check(method+"_strict_model_receipt_and_checkpoint_identity",model_meta["method"]==method and model_meta["loader_receipt"]["sha256"]==sha(DATA/("loader_"+method+".json"))
              and loader["checkpoint"]==spec["checkpoint"] and sha(spec["checkpoint"]["path"])==spec["checkpoint"]["sha256"] and loader["method"]==method
              and loader["iteration"]==2000 and loader["actor104_validation"]["format_version"]==104 and loader["status"]=="PASS_FROZEN_CPU_WORLD_MODEL_LOAD"
              and loader["physical_xi_dim"]==46 and loader["disturbance_dim"]==174 and loader["WM_total_input_dim"]==298 and loader["self_collision_logits_dim"]==15
              and loader["parameter_requires_grad"] is False and loader["amp_enabled_for_CPU_diagnostic"] is False
              and model_meta["unchanged"] is True and model_meta["before_state_sha256"]==model_meta["after_state_sha256"])
    check("RNG_and_CPU_boundary_source_backed_assertions",source_report["RNG_unchanged"] is True and source_report["no_Warp_or_CUDA"] is True,
          "No RNG before/after tensors are serialized; verified executed source checks and saved assertion, not an independent RNG-state recomputation.")
    # Fresh rehash the saved result files to certify this reviewer did not change them.
    results_unchanged=True
    for name,p in original_pins.items():
        h=hashlib.sha256((DATA/name).read_bytes()).hexdigest();results_unchanged=results_unchanged and h==p["sha256"]
    check("reviewer_left_original_results_unchanged",results_unchanged)
    check("reviewer_frameworks_not_imported",not any(name in sys.modules for name in ("torch","warp","mujoco","mujoco_warp")))
    RESULT.update(status="PASS_SAVED_NUMERIC_AND_IDENTITY_REVIEW" if all(checks.values()) else "FAILED_SAVED_REVIEW_RETAINED",
                  checks=checks,check_details=details,numeric_comparisons=numeric,
                  previous_harness_failure_preserved=pin(OUT/"report.json"),
                  previous_harness_corrections=["Original protocol raw file pin plus semantic archived JSON copy; no false raw-byte equality after reserialization",
                      "Saved FP32 reward error/dt checked exactly from saved four terms; independent FP64 cancellation discrepancy remains descriptive, unchanged numerical budgets"],
                  reward_FP64_cancellation_diagnostic={"rows":len(reward_cancellation_differences),
                      "max_absolute_difference":max(v["absolute_difference"] for v in reward_cancellation_differences),
                      "largest_difference_row":max(reward_cancellation_differences,key=lambda v:v["absolute_difference"]),
                      "interpretation":"FP32 term sum/subtraction vs independent FP64 geometry; not a change to saved result or physical gate"},
                  original_artifacts=original_pins,model_receipts=model_receipts,
                  FD={"pairs":len(fd),"valid":len(fd)-skipped,"skipped":skipped,"by_group":valid_by_group,
                      "valid_absolute_discrepancy_max":max(r[0] for r in fd_discrepancies),"valid_relative_discrepancy_max":max(r[1] for r in fd_discrepancies),
                      "FD_slope_independently_recomputed":False,"slope_limit":"Saved FD rows omit the two forward reward values; reviewer recomputed masks/AD projections/discrepancy formulas, without rerunning either forward."},
                  evidence_limits={"RNG_raw_states_saved":False,"model_memory_state_raw_before_after_saved":False,
                                   "source_backed_runtime_unchanged_flags_checked":True,"checkpoint_files_rehashed":True,
                                   "self15_truth":"UNKNOWN","physical_Jacobian_qualification":False,"no_method_benefit_inference":True},
                  arithmetic_budgets={"float32_epsilon":EPS32,"general_relative":REL,"general_absolute":ABS,"rotation_absolute_radian":ROT_ABS,
                                      "rotation_method":"Independent FP64 atan2(norm(vee(skew(Rpred.T Rtrue))), (trace-1)/2); native uses FP32 quaternion-based so3_log. Original results unchanged."})
except BaseException as exc:
    RESULT.update(status="AUDIT_ERROR_RETAINED",error=repr(exc),traceback=traceback.format_exc(),checks=checks,check_details=details,numeric_comparisons=numeric)
finally:
    RESULT.update(passed=sum(checks.values()),total=len(checks),failed=[k for k,v in checks.items() if not v],reviewer_source=pin(__file__),
                  execution={"NumPy_stdlib_only":True,"WM_forward_calls":0,"WM_or_checkpoint_loads":0,"Env_or_Reader_construction":0,"new_physics_queries":0,"GPU_calls":0},
                  wall_seconds=time.perf_counter()-START)
    target=OUT/"report_v2.json"
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(RESULT,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":RESULT["status"],"passed":RESULT["passed"],"total":RESULT["total"],"failed":RESULT["failed"],"report":pin(target)},indent=2))
raise SystemExit(0 if RESULT["status"]=="PASS_SAVED_NUMERIC_AND_IDENTITY_REVIEW" else 2)
