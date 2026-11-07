"""Independent CPU analysis of all predeclared braking microprobe branches.

No simulator, GPU, controller update, positive-only fit or parameter enablement.
Baselines estimate numerical repeatability, not independent tasks. A pulse
opposing inward initial motion is not an upper-limit braking observation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SETTINGS={"vmin_rad_s":.02,"flat_velocity_floor_rad_s":2e-4,
          "position_noise_floor_rad":2e-5,"sustained_reversal_seconds":.02,
          "static_float_copy_atol":2e-7,"force_saturation_relative_tolerance":1e-6,
          "force_saturation_absolute_tolerance_Nm":1e-7,
          "m0":None,"a_brake":None,"parameter_enablement":False,
          "scope":"sampled local response diagnostics only; strong fixed0.1rad pulse is not weak C1 law or a hard deceleration bound"}


def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item())+"\n",encoding="utf-8")
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def resolve(base,p):
    p=Path(p);return p if p.is_absolute() else base/p
def status(values):
    values=[None if v is None else bool(v) for v in values]
    return "FAIL" if any(v is False for v in values) else "UNKNOWN" if any(v is None for v in values) else "PASS"
def maximum_abs(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def matrices(quat):
    q=np.asarray(quat,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))


def first_run(mask,time,duration):
    for k in np.flatnonzero(mask):
        end=np.searchsorted(time,time[k]+duration-1e-12)
        if end<len(time) and np.all(mask[k:end+1]):return int(k)
    return None


def response(time,q,qd,side,velocity_noise,position_noise,replay_gate):
    """Retain signed/flat/adverse cases; stop events are sampled observations."""
    t=np.asarray(time,float);q=np.asarray(q,float);v=(1 if side=="upper" else -1)*np.asarray(qd,float)
    displacement=(1 if side=="upper" else -1)*(q-q[0]);v0=float(v[0])
    if abs(v0)<=velocity_noise:kind="FLAT_OR_NOISE_LEVEL"
    elif v0<0:kind="INITIAL_INWARD" if v0< -SETTINGS["vmin_rad_s"] else "LOW_SPEED_INWARD"
    elif v0<=SETTINGS["vmin_rad_s"]:kind="LOW_SPEED_OUTWARD"
    else:kind="OUTWARD_ABOVE_VMIN"
    initial_sign=1 if qd[0]>velocity_noise else -1 if qd[0]<-velocity_noise else 0
    initial_reverse=np.flatnonzero(initial_sign*np.asarray(qd)<-velocity_noise) if initial_sign else np.array([],dtype=int)
    initial_sustained=first_run(initial_sign*np.asarray(qd)<=-velocity_noise,t,SETTINGS["sustained_reversal_seconds"]) if initial_sign else None
    nonpositive=np.flatnonzero(v[1:]<=0)+1 if v0>0 else np.array([],dtype=int)
    k=int(nonpositive[0]) if len(nonpositive) else None
    crossing=None;stop_displacement=None;stop_bracket=None
    if k is not None:
        fraction=float(v[k-1]/(v[k-1]-v[k])) if v[k-1]!=v[k] else 1.
        crossing=float(t[k-1]+fraction*(t[k]-t[k-1]))
        crossing_q=float(displacement[k-1]+fraction*(displacement[k]-displacement[k-1]))
        stop_displacement=float(max(0,float(displacement[:k].max()),crossing_q))
        stop_bracket=[float(t[k-1]),float(t[k])]
    resolved=np.flatnonzero(v<=-velocity_noise) if v0>velocity_noise else np.array([],dtype=int)
    sustained=first_run(v<=-velocity_noise,t,SETTINGS["sustained_reversal_seconds"]) if v0>velocity_noise else None
    reaccelerates=bool(k is not None and np.any(v[k+1:]>velocity_noise))
    accel=-np.diff(v)/np.diff(t) if len(t)>1 else np.array([])
    tick_end=min(len(t)-1,int(np.searchsorted(t,.02-1e-12)))
    mean20=float((v[0]-v[tick_end])/(t[tick_end]-t[0])) if tick_end>0 else None
    reasons=[]
    if replay_gate!="PASS":reasons.append("RESTORE_REPLAY_CONTEXT_CAPACITY_NOT_PASS")
    if v0<=SETTINGS["vmin_rad_s"]:reasons.append("INITIAL_UPPER_OUTWARD_VELOCITY_NOT_ABOVE_VMIN")
    if sustained is None:reasons.append("NO_20MS_NOISE_RESOLVED_SUSTAINED_REVERSAL")
    if stop_displacement is None or stop_displacement<=position_noise:reasons.append("STOP_DISPLACEMENT_UNRESOLVED_OR_CENSORED")
    if reaccelerates:reasons.append("OUTWARD_REACCELERATION_AFTER_INITIAL_STOP")
    return {"velocity_class":kind,"initial_outward_velocity_rad_s":v0,
            "initial_motion_reversal_time_s":float(t[initial_reverse[0]]) if len(initial_reverse) else None,
            "initial_motion_sustained_reversal_start_s":float(t[initial_sustained]) if initial_sustained is not None else None,
            "first_outward_nonpositive_time_linear_s":crossing,"zero_crossing_sample_bracket_s":stop_bracket,
            "first_noise_resolved_reversal_time_s":float(t[resolved[0]]) if len(resolved) else None,
            "sustained_reversal_start_s":float(t[sustained]) if sustained is not None else None,
            "sustained_definition":"velocity<=negative repeat-noise floor for at least20ms, using every recorded5ms sample",
            "remains_nonoutward_after_first_stop":bool(not reaccelerates) if k is not None else None,
            "outward_reacceleration":reaccelerates,"no_stop_within_observed_horizon":bool(k is None) if v0>velocity_noise else None,
            "horizon_seconds":float(t[-1]-t[0]),"max_outward_excursion_rad":float(max(0,displacement.max())),
            "stopping_displacement_observed_rad":stop_displacement,
            "right_censored_no_stop":bool(k is None) if v0>velocity_noise else None,
            "initial_net_deceleration_rad_s2":float(accel[0]) if len(accel) else None,
            "first20ms_net_deceleration_rad_s2":mean20,
            "net_deceleration_min_rad_s2":float(accel.min()) if len(accel) else None,
            "net_deceleration_max_rad_s2":float(accel.max()) if len(accel) else None,
            "negative_net_deceleration_intervals":int((accel<0).sum()),
            "upper_stopping_observation_eligible":not reasons,"qualification_reasons":reasons,
            "velocity_noise_floor_rad_s":float(velocity_noise),"position_noise_floor_rad":float(position_noise),
            "actuator_only_braking_contribution":"UNKNOWN","joint_limit_braking_contribution":"UNKNOWN",
            "parameter_fit_authorized":False,"a_brake":None,
            "interpretation":"FD deceleration is signed net dynamics including contact, passive/coupled and constraint contributions; no hard bound or positive-only fit"}


def target_rate(a,control_dt):
    if not len(a["u_readback_for_step"]):
        return {"target_speed_abs_max_rad_s":None,"hold_exact":None,"tick_count":0,"rate_timebase_s":control_dt,"rate_includes_snapshot_previous_target":True}
    tick=np.asarray(a["control_tick_id_for_step"]);indices=np.r_[0,np.flatnonzero(np.diff(tick)!=0)+1]
    u=a["u_readback_for_step"][indices]
    rates=np.diff(np.vstack([a["u_prev_rad"],u]),axis=0)/control_dt
    hold=True
    for k in np.unique(tick):hold=hold and np.array_equal(a["u_readback_for_step"][tick==k],np.repeat(a["u_readback_for_step"][tick==k][:1],int((tick==k).sum()),axis=0))
    return {"target_speed_abs_max_rad_s":float(np.abs(rates).max()),"hold_exact":hold,
            "tick_count":len(indices),"rate_timebase_s":control_dt,"rate_includes_snapshot_previous_target":True}


def validate_arrays(a,meta,plan,names):
    n=len(a["elapsed_s"]);steps=n-1
    shapes={"q_rad":(n,20),"qd_rad_s":(n,20),"actuator_torque_Nm":(n,20),"object_pose_w":(n,7),"object_twist_native":(n,6),
            "u_prev_rad":(20,),"u_proposed_for_step":(steps,20),"u_bounded_for_step":(steps,20),"u_readback_for_step":(steps,20)}
    checks={"required_numeric_shapes":all(k in a and a[k].shape==shape for k,shape in shapes.items()),
            "required_state_and_control_finite":all(k in a and np.isfinite(a[k]).all() for k in shapes),
            "canonical_clock":bool(np.allclose(a["elapsed_s"],np.arange(n)*plan["physics_dt_s"],atol=1e-12,rtol=0)),
            "complete41_states":bool(meta.get("complete") is True and n==41 and steps==40),
            "integration_restore_exact":meta.get("integration_restore_max_abs_error")==0.,
            "capacity_explicit_OK":True if meta.get("capacity_status",{}).get("status")=="OK" else False if meta.get("capacity_status",{}).get("status")=="OVERFLOW" else None,
            "joint_order_explicit":list(meta.get("joint_names",[]))==names,
            "no_external_forces":bool(np.max(np.abs(a["external_force_max_abs"]))<1e-9),
            "native_float32_control_readback_exact":bool(np.array_equal(a["u_bounded_for_step"].astype(np.float32),a["u_readback_for_step"].astype(np.float32)))}
    rate=target_rate(a,plan["control_dt_s"])
    checks["cached_sample_hold_exact"]=rate["hold_exact"]
    tick=np.asarray(a["control_tick_id_for_step"])
    checks["four_substeps_per_control_tick"]=bool(np.array_equal(tick-tick[0],np.repeat(np.arange(10),4)[:len(tick)])) if len(tick) else None
    return checks


def source_slice(meta,manifest_dir):
    s=meta["source_future_slice"];p=resolve(manifest_dir,s["npz"]);a=load(p)
    states=slice(s["start_state"],s["stop_state_exclusive"]);steps=slice(s["start_step"],s["stop_step_exclusive"])
    fields=["q_rad","qd_rad_s","object_pose_w","object_twist_native","u_readback_for_step"]
    if "hand_object_active_count" in a:fields.append("hand_object_active_count")
    return {k:a[k][steps if k=="u_readback_for_step" else states] for k in fields},p


def replay_errors(a,b):
    return {"position_q_rotation_max_abs":max(maximum_abs(a["q_rad"],b["q_rad"]),maximum_abs(a["object_pose_w"][:,:3],b["object_pose_w"][:,:3]),maximum_abs(matrices(a["object_pose_w"][:,3:]),matrices(b["object_pose_w"][:,3:]))),
            "velocity_max_abs":max(maximum_abs(a["qd_rad_s"],b["qd_rad_s"]),maximum_abs(a["object_twist_native"],b["object_twist_native"])),
            "ctrl_float32_exact":bool(np.array_equal(a["u_readback_for_step"].astype(np.float32),b["u_readback_for_step"].astype(np.float32)))}


def archival_compare(donor,source,plan):
    path=source.get("archived_evaluation")
    if not path:return {"status":"UNKNOWN","reason":"no bound archived evaluation"}
    try:
        archived=load(Path(path)/"rollout.npz");a=load(donor);index=np.arange(0,len(a["q_rad"]),4);index=index[:len(archived["time_s"])]
        old={"q_rad":archived["joint_pos_rad"][:len(index)],"qd_rad_s":archived["joint_vel_rad_s"][:len(index)],"object_pose_w":archived["object_pose_w"][:len(index)],
             "object_twist_native":np.c_[archived["object_lin_vel_w_m_s"][:len(index)],np.einsum("tji,tj->ti",matrices(archived["object_pose_w"][:len(index),3:]),archived["object_ang_vel_w_rad_s"][:len(index)])],
             "u_readback_for_step":archived["joint_target_rad"][1:len(index)]}
        current={k:a[k][index] for k in ["q_rad","qd_rad_s","object_pose_w","object_twist_native"]}
        current["u_readback_for_step"]=a["u_readback_for_step"][:4*(len(index)-1):4]
        e=replay_errors(current,old);p=plan["gates"]
        return {"status":"PASS" if e["position_q_rotation_max_abs"]<=p["baseline_position_matrix_q_max_abs"] and e["velocity_max_abs"]<=p["baseline_velocity_max_abs"] and e["ctrl_float32_exact"] else "FAIL",**e,
                "scope":"fresh donor vs historical rollout, separate from within-snapshot direct-u replay; mismatch is never relabeled historical equivalence"}
    except (OSError,KeyError,ValueError) as error:return {"status":"UNKNOWN","reason":str(error)}


def task_diagnostics(a,reference,capturetime,plan):
    # Geometry/orientation are evaluated at observed5ms samples; no fictitious
    # full-clip progress or cumulative-load history is initialized at zero.
    t=np.asarray(reference["time_s"]);time=capturetime+np.asarray(a["elapsed_s"])
    targets=np.stack([np.interp(time,t,reference["object_pose_w"][:,i]) for i in range(3)],-1)
    position=np.linalg.norm(a["object_pose_w"][:,:3]-targets,axis=-1)
    # Reference quaternion interpolation was not authorized. Only coincident
    # original50Hz samples have an exact frozen reference orientation.
    indices=np.arange(0,len(time),4);frames=np.rint(time[indices]/plan["control_dt_s"]).astype(int)
    axis=matrices(a["object_pose_w"][indices,3:])[:,:,0];refaxis=matrices(reference["object_pose_w"][frames,3:])[:,:,0]
    angle=np.rad2deg(np.arccos(np.clip((axis*refaxis).sum(-1),-1,1)))
    lo,hi=a["joint_lower_rad"],a["joint_upper_rad"]
    state=np.maximum(np.maximum(lo-a["q_rad"],a["q_rad"]-hi),0)
    target=np.maximum(np.maximum(lo-a["u_readback_for_step"],a["u_readback_for_step"]-hi),0)
    target_max=float(target.max()) if target.size else None
    depth=max(float(np.max(a[k])) for k in ["penetration_hand_object_m","penetration_hand_self_m","penetration_other_m"])
    force=a["actuator_torque_Nm"][1:];limits=a["force_limit_Nm"]
    rates=target_rate(a,plan["control_dt_s"])
    return {"actual_state_violation_max_rad":float(state.max()),"target_violation_max_rad":target_max,
            "starts_outside_hard_limits_any_joint":bool((state[0]>0).any()),"starts_outside_frozen_margin_any_joint":bool((state[0]>plan["limit_margin_rad"]).any()),
            "state_limit_gate_1mrad":bool(state.max()<=plan["limit_margin_rad"]),"target_limit_gate_1mrad":bool(target_max<=plan["limit_margin_rad"]) if target_max is not None else None,
            "penetration_max_m":depth,"penetration_3mm_gate":bool(depth<=.003),
            "position_max_at_original50Hz_m":float(position[indices].max()),"position20mm_at50Hz":bool(position[indices].max()<=.02),
            "shaft_max_original50Hz_deg":float(angle.max()),"shaft20deg_original50Hz":bool(angle.max()<=20),
            "load_ratio_max_logged_solver_samples":float((np.abs(force)/limits).max()) if force.size else None,
            "load_ratio1p01_observed":bool((np.abs(force)/limits).max()<=1.01) if force.size else None,
            "target_rate15rad_s_gate":bool(rates["target_speed_abs_max_rad_s"]<=plan["target_rate_limit_rad_s"]) if rates["target_speed_abs_max_rad_s"] is not None else None,**rates,
            "load_charge_gate":"UNKNOWN_REQUIRES_FULL_PREFIX_HISTORY","full_clip_progress_gate":"NOT_A_FULL_CLIP",
            "object_drop_environment_gate":"UNKNOWN_UNLESS_COMPLETE_ORIGINAL_EVIDENCE_PROVIDED",
            "task_success":"NOT_DEFINED_FOR_MICROBRANCH","reference_timing_note":"orientation/position gate restricted to coincident original50Hz samples; no quaternion interpolation"}


def forces(a,meta,j):
    availability=meta.get("availability",{});force=a["actuator_torque_Nm"][1:,j];limit=float(a["force_limit_Nm"][j])
    tolerance=max(SETTINGS["force_saturation_absolute_tolerance_Nm"],limit*SETTINGS["force_saturation_relative_tolerance"])
    active="UNKNOWN"
    if availability.get("joint_limit_active") is True:
        active="OBSERVED" if np.any(a["joint_limit_active"][1:,j]>0) else "NOT_OBSERVED_AT_LOGGED_SOLVER_SAMPLES"
    return {"actuator_torque_min_Nm":float(force.min()) if force.size else None,"actuator_torque_max_Nm":float(force.max()) if force.size else None,
            "force_saturation_sample_fraction":float((np.abs(force)>=limit-tolerance).mean()) if force.size else None,
            "joint_limit_constraint_activity":active,"joint_limit_braking_contribution":"UNKNOWN",
            "actuator_only_braking":"UNKNOWN","force_decomposition_reason":"net constraint force and activity bits do not identify per-limit generalized braking work or coupled/contact contributions",
            "force_timing_scope":"samples1..40 only; t0 derived caches not treated as restored branch forces; preintegration/post-forward timing retained in metadata"}


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def plots(snapshots,responses,branches,out,names):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"svg.fonttype":"none"})
    pulses=[r for r in responses if r["kind"]=="pulse"]
    colors=["#1764ab","#d55e00","#15845c"]
    methods=["A1L","A2","P_ONLY"];times=[.8,1.6,2.4]
    fig,axes=plt.subplots(3,3,figsize=(13,9),constrained_layout=True)
    for s in snapshots:
        row=methods.index(s["method"]);col=times.index(round(s["capturetime"],1));ax=axes[row,col]
        for k,joint in enumerate(names):
            runs=[b for b in branches if b["meta"].get("snapshot_id")==s["snapshot_id"] and b["meta"].get("kind")=="pulse" and b["meta"].get("pulse_joint")==joint]
            if runs:
                b=runs[0];j=b["joint_names"].index(joint);ax.plot(b["a"]["elapsed_s"],b["a"]["qd_rad_s"][:,j],color=colors[k],label=joint.replace("right_","").replace("_joint"," j"))
            baseline=[b for b in branches if b["meta"].get("snapshot_id")==s["snapshot_id"] and b["meta"].get("kind")=="baseline" and len(b["a"]["elapsed_s"])==41]
            if len(baseline)==3:
                j=baseline[0]["joint_names"].index(joint);v=np.stack([b["a"]["qd_rad_s"][:,j] for b in baseline]);ax.plot(baseline[0]["a"]["elapsed_s"],v.mean(0),color=colors[k],linestyle=":",alpha=.65)
        ax.axhline(0,color="0.3",linewidth=.8);ax.grid(alpha=.2);ax.set_title(f"{s['method']} donor t={s['capturetime']:.1f}s | replay {s['replay_gate']}")
        ax.set(xlabel="Branch elapsed (s)",ylabel="Upper-outward velocity (rad/s)")
    axes[0,0].legend(fontsize=7)
    fig.suptitle("All9 snapshots: pulse solid, three-baseline mean dotted; invalid/low/inward cases retained\nSampled net response only. Fixed0.1rad pulse is not weak C1; no braking parameter enabled.")
    for ext in ["png","svg"]:fig.savefig(out/f"velocity.{ext}",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,constrained_layout=True)
    for i,r in enumerate(pulses):
        color="#1764ab" if r["upper_stopping_observation_eligible"] else "#b65e20" if r["replay_gate"]=="PASS" else "0.5"
        marker="o" if r["upper_stopping_observation_eligible"] else "x"
        for ax,key,scale in zip(axes,["initial_outward_velocity_rad_s","max_outward_excursion_rad","first20ms_net_deceleration_rad_s2"],[1,1000,1]):
            if r[key] is not None:ax.scatter(i,r[key]*scale,color=color,marker=marker,s=30)
    for ax,label in zip(axes,["Initial upper-outward velocity (rad/s)","Observed maximum outward excursion (mrad)","First20ms signed NET deceleration (rad/s²)"]):ax.set_ylabel(label);ax.axhline(0,color="0.3",linewidth=.8);ax.grid(alpha=.2)
    axes[0].axhline(SETTINGS["vmin_rad_s"],color="0.3",linestyle=":")
    axes[-1].set_xticks(range(len(pulses)),[f"{r['method']} {r['capturetime']:.1f}\n{r['joint'].replace('right_','')}" for r in pulses],rotation=70,ha="right",fontsize=7)
    fig.suptitle("Every predeclared pulse: blue eligible sampled stopping observation; orange other valid local cases; gray replay invalid/unknown\nNegative/zero deceleration, low/inward/no-stop and incomplete cases are not removed; no positive-only fit.")
    for ext in ["png","svg"]:fig.savefig(out/f"response_summary.{ext}",dpi=170)
    plt.close(fig)


def analyze(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True);plan=read(args.plan);schema=read(ROOT/"raw_branch_schema_v1.json")
    header={"schema":"C1_braking_response_analysis:1","plan_sha256":sha(args.plan),"raw_schema_sha256":sha(ROOT/"raw_branch_schema_v1.json"),"analysis_source_sha256":sha(__file__),"settings":SETTINGS,"gpu_used":False,"m0":None,"a_brake":None,"parameter_enablement":False}
    if not args.manifest.exists():
        write(out/"analysis.json",{**header,"status":"PENDING_RAW","branches":[],"responses":[],"missing":str(args.manifest)});print("PENDING_RAW");return
    manifest=read(args.manifest);base=args.manifest.parent;reference=load(Path(plan["reference"]));names=read(Path(plan["reference"]).with_suffix(".json"))["joint_names"]
    manifest_plan_match=manifest.get("plan_sha256")==sha(args.plan) if "plan_sha256" in manifest else None
    branches=[];errors=[]
    for entry in manifest.get("branches",[]):
        try:
            mp=resolve(base,entry["metadata"]);ap=resolve(base,entry["npz"]);m=read(mp);a=load(ap)
            for k in schema["metadata_required"]:
                if k not in m:raise ValueError(f"missing metadata {k}")
            checks=validate_arrays(a,m,plan,names)
            src=next(s for s in plan["sources"] if s["method"]==m["method"])
            checks["training_seed1_matches_source"]=m["training_seed"]==src["training_seed"]==1
            evaluation_seed=m.get("evaluation_seed",manifest.get("evaluation_seed"))
            checks["evaluation_seed0_matches_source"]=evaluation_seed==src["evaluation_seed"]==0 if evaluation_seed is not None else None
            if manifest_plan_match is not None:checks["manifest_plan_sha_binding"]=manifest_plan_match
            writes=m.get("inside_branch_physical_state_writes",m.get("inside_branch_state_writes"))
            checks["no_inside_branch_state_writes"]=writes==0 if writes is not None else None
            bindings=m.get("source_model_context_hashes",{})
            checks["source_pins_match_plan"]=all(bindings.get(k+"_sha256")==src[k+"_sha256"] for k in ["checkpoint","model","context"])
            future,donor=source_slice(m,base)
            branches.append({"meta":m,"a":a,"checks":checks,"npz":str(ap),"metadata":str(mp),"npz_sha256":sha(ap),"metadata_sha256":sha(mp),"future":future,"donor":donor,"joint_names":names})
        except (OSError,KeyError,ValueError,StopIteration) as error:errors.append({"entry":entry,"status":"INVALID_OR_MISSING_RAW","reason":str(error)})
    snapshots=[];responses=[];branch_rows=[]
    for sid in dict.fromkeys(b["meta"]["snapshot_id"] for b in branches):
        members=[b for b in branches if b["meta"]["snapshot_id"]==sid];m=members[0]["meta"];baselines=[b for b in members if b["meta"]["kind"]=="baseline"]
        pulse_members=[b for b in members if b["meta"]["kind"]=="pulse"]
        gating={"three_baselines":len(baselines)==3 and len({b["meta"]["repeat"] for b in baselines})==3,
                "three_declared_pulses":len(pulse_members)==3 and {b["meta"]["pulse_joint"] for b in pulse_members}=={c["joint"] for c in plan["joint_cases"]},
                "six_unique_branch_ids":len(members)==len({b["meta"]["branch_id"] for b in members})==6}
        gating.update({b["meta"]["branch_id"]+":"+k:v for b in members for k,v in b["checks"].items()})
        comparisons=[];floor_v=SETTINGS["flat_velocity_floor_rad_s"];floor_q=SETTINGS["position_noise_floor_rad"]
        for b in baselines:
            try:
                e=replay_errors(b["a"],b["future"]);comparisons.append({"branch":b["meta"]["branch_id"],"source_future":e})
                gating[b["meta"]["branch_id"]+":source_future_replay"]=e["position_q_rotation_max_abs"]<=plan["gates"]["baseline_position_matrix_q_max_abs"] and e["velocity_max_abs"]<=plan["gates"]["baseline_velocity_max_abs"] and e["ctrl_float32_exact"]
            except ValueError:gating[b["meta"]["branch_id"]+":source_future_replay"]=False
        if len(baselines)==3 and all(len(b["a"]["elapsed_s"])==41 for b in baselines):
            for i in range(3):
                for j in range(i):
                    e=replay_errors(baselines[i]["a"],baselines[j]["a"]);comparisons.append({"repeat_pair":[i,j],**e})
                    gating[f"repeat_pair_{i}_{j}"]=e["position_q_rotation_max_abs"]<=plan["gates"]["baseline_position_matrix_q_max_abs"] and e["velocity_max_abs"]<=plan["gates"]["baseline_velocity_max_abs"] and e["ctrl_float32_exact"]
            qstack=np.stack([b["a"]["q_rad"] for b in baselines]);vstack=np.stack([b["a"]["qd_rad_s"] for b in baselines]);noise_q=np.maximum(floor_q,np.max(np.abs(qstack-qstack.mean(0)),axis=(0,1)));noise_v=np.maximum(floor_v,np.max(np.abs(vstack-vstack.mean(0)),axis=(0,1)))
        else:noise_q=np.full(20,floor_q);noise_v=np.full(20,floor_v)
        gate=status(gating.values());src=next(s for s in plan["sources"] if s["method"]==m["method"])
        snap={"snapshot_id":sid,"method":m["method"],"capturetime":m["capturetime"],"replay_gate":gate,"checks":gating,"comparisons":comparisons,"archived_policy_reproduction":archival_compare(members[0]["donor"],src,plan),"scope":"response qualification refers to fresh donor and its within-snapshot replay; archival mismatch remains separately visible, never historical equivalence"}
        snapshots.append(snap)
        for b in members:
            bm,a=b["meta"],b["a"];diagnostic=task_diagnostics(a,reference,float(bm["capturetime"]),plan)
            clipping=np.abs(a["u_proposed_for_step"]-a["u_bounded_for_step"])
            branch_rows.append({"branch_id":bm["branch_id"],"snapshot_id":sid,"method":bm["method"],"capturetime":bm["capturetime"],"kind":bm["kind"],"repeat":bm["repeat"],"pulse_joint":bm["pulse_joint"],"pulse_sign":bm["pulse_sign"],"pulse_amplitude_rad":bm["pulse_amplitude_rad"],"replay_gate":gate,"complete":bm["complete"],"target_clipping_max_rad":float(clipping.max()) if clipping.size else None,"target_clipping_components":int((clipping>0).sum()),**diagnostic})
            joints=[bm["pulse_joint"]] if bm["kind"]=="pulse" else [c["joint"] for c in plan["joint_cases"]]
            for joint in joints:
                j=names.index(joint);side=next(c["limit_side"] for c in plan["joint_cases"] if c["joint"]==joint)
                r={"branch_id":bm["branch_id"],"snapshot_id":sid,"method":bm["method"],"capturetime":bm["capturetime"],"kind":bm["kind"],"repeat":bm["repeat"],"joint":joint,"declared_limit_side":side,"replay_gate":gate,
                   **response(a["elapsed_s"],a["q_rad"][:,j],a["qd_rad_s"][:,j],side,float(noise_v[j]),float(noise_q[j]),gate),**forces(a,bm,j)}
                r["pulse_inward_wrt_upper"]=bool(bm["pulse_sign"]<0) if bm["kind"]=="pulse" else None
                initial_excess=max(float(a["joint_lower_rad"][j]-a["q_rad"][0,j]),float(a["q_rad"][0,j]-a["joint_upper_rad"][j]),0.)
                r.update(starts_outside_hard_limits=initial_excess>0,starts_outside_frozen_margin=initial_excess>plan["limit_margin_rad"],initial_upper_reserve_rad=float(a["joint_upper_rad"][j]-a["q_rad"][0,j]),horizon_complete=bm["complete"])
                if initial_excess>0:
                    r["upper_stopping_observation_eligible"]=False;r["qualification_reasons"].append("STARTS_OUTSIDE_HARD_LIMIT_CONFOUNDED")
                if bm["kind"]=="pulse":
                    r["effective_initial_offset_from_u_prev_rad"]=float(a["u_readback_for_step"][0,j]-a["u_prev_rad"][j]) if len(a["u_readback_for_step"]) else None
                    r["initial_pulse_noop_vs_previous_target"]=bool(a["u_readback_for_step"][0,j]==a["u_prev_rad"][j]) if len(a["u_readback_for_step"]) else None
                    r["pulse_noop_vs_baseline_future"]=bool(np.array_equal(a["u_readback_for_step"][:,j].astype(np.float32),b["future"]["u_readback_for_step"][:len(a["u_readback_for_step"]),j].astype(np.float32)))
                    r["nominal_pulse_amplitude_rad"]=bm["pulse_amplitude_rad"]
                    r["pulse_target_clipped"]=bool(np.any(clipping[:,j]>0))
                    r["active_contact_count_changed_vs_source"]=bool(np.any(a["hand_object_active_count"]!=b["future"]["hand_object_active_count"][:len(a["hand_object_active_count"])])) if "hand_object_active_count" in b["future"] else "UNKNOWN_NO_SOURCE_COUNT_FIELD"
                responses.append(r)
    unique_ids=len({b["branch_id"] for b in branch_rows})
    complete=len(branch_rows)==unique_ids==plan["total_branches"] and len(snapshots)==plan["snapshot_count"] and not errors
    data={**header,"status":"COMPLETE_WITH_QUALIFICATION_MASKS" if complete else "PENDING_OR_INVALID_RAW","branches_manifest_sha256":sha(args.manifest),"branches":branch_rows,"responses":responses,"snapshots":snapshots,"raw_errors":errors,
          "counts":{"recorded_branch_entries":len(branch_rows),"unique_branches":unique_ids,"snapshots":len(snapshots),"joint_response_rows":len(responses),"expected_branches":54,"expected_joint_response_rows":108},
          "no_positive_only_fit":True,"parameter_fit_authorized":False,"force_attribution":"UNKNOWN: no per-source generalized force/work decomposition supplied; near-bound q or net constraint force is not joint-limit impact evidence"}
    write(out/"analysis.json",data);csv_write(out/"branches.csv",branch_rows);csv_write(out/"responses.csv",responses);csv_write(out/"snapshots.csv",snapshots)
    (out/"REPORT.txt").write_text(f"Status: {data['status']}\nAll recorded branches retained: {len(branch_rows)}/54; snapshots {len(snapshots)}/9; descriptive joint rows {len(responses)}/108.\nBaseline repeats are numerical-repeatability evidence, not independent tasks.\nm0=null; a_brake=null; parameter enablement=false.\nStopping events are sampled observations; no-stop means right-censored within0.2s.\nSigned net deceleration retains negative/zero values. Force attribution and actuator-only/joint-limit braking contribution remain UNKNOWN without decomposition.\nThe fixed0.1rad counter-initial-velocity pulse is not weak C1 and does not establish a hard braking bound.\nFresh-donor vs historical-rollout drift is reported separately from within-snapshot replay.\n",encoding="utf-8")
    if snapshots:plots(snapshots,responses,branches,out,[x["joint"] for x in plan["joint_cases"]])
    print(json.dumps({"status":data["status"],"counts":data["counts"],"replay_gates":{x["snapshot_id"]:x["replay_gate"] for x in snapshots},"m0":None,"a_brake":None},indent=2))


def self_test():
    class Checks(unittest.TestCase):
        def test_no_stop_right_censored(self):
            t=np.arange(41)*.005;r=response(t,t,np.ones(41),"upper",.0002,.00002,"PASS");self.assertTrue(r["right_censored_no_stop"]);self.assertIsNone(r["stopping_displacement_observed_rad"])
        def test_initial_inward_is_not_upper_fit(self):
            t=np.arange(41)*.005;r=response(t,-t,-np.ones(41),"upper",.0002,.00002,"PASS");self.assertEqual(r["velocity_class"],"INITIAL_INWARD");self.assertFalse(r["upper_stopping_observation_eligible"])
        def test_flat_not_fictitious_stop(self):
            t=np.arange(41)*.005;r=response(t,t*0,t*0,"upper",.0002,.00002,"PASS");self.assertIsNone(r["no_stop_within_observed_horizon"]);self.assertFalse(r["upper_stopping_observation_eligible"])
        def test_known_stop_and_sustained_reversal(self):
            t=np.arange(41)*.005;r=response(t,t-5*t*t,1-10*t,"upper",.0002,.00002,"PASS");self.assertAlmostEqual(r["first_outward_nonpositive_time_linear_s"],.1);self.assertAlmostEqual(r["stopping_displacement_observed_rad"],.05);self.assertTrue(r["upper_stopping_observation_eligible"])
        def test_transient_reversal_not_sustained(self):
            t=np.arange(41)*.005;v=np.ones(41);v[10:12]=-.1;r=response(t,np.cumsum(v)*.005,v,"upper",.0002,.00002,"PASS");self.assertTrue(r["outward_reacceleration"]);self.assertIsNone(r["sustained_reversal_start_s"])
        def test_replay_failure_disqualifies_but_keeps_signed_deceleration(self):
            t=np.arange(41)*.005;r=response(t,t+5*t*t,1+10*t,"upper",.0002,.00002,"FAIL");self.assertFalse(r["upper_stopping_observation_eligible"]);self.assertLess(r["net_deceleration_min_rad_s2"],0)
        def test_one_baseline_mismatch_disqualifies_whole_snapshot(self):
            pose=np.zeros((41,7));pose[:,3]=1
            a={"q_rad":np.zeros((41,20)),"qd_rad_s":np.zeros((41,20)),"object_pose_w":pose,"object_twist_native":np.zeros((41,6)),"u_readback_for_step":np.zeros((40,20))}
            b={k:v.copy() for k,v in a.items()};b["q_rad"][20,3]=2.1e-5;e=replay_errors(a,b)
            gate=status([True,True,np.bool_(e["position_q_rotation_max_abs"]<=2e-5),True])
            self.assertEqual(gate,"FAIL")
            t=np.arange(41)*.005
            for _ in range(3):self.assertFalse(response(t,t-5*t*t,1-10*t,"upper",.0002,.00002,gate)["upper_stopping_observation_eligible"])
        def test_inward_initial_motion_can_reverse_but_never_upper_fit(self):
            t=np.arange(41)*.005;r=response(t,-t+5*t*t,-1+10*t,"upper",.0002,.00002,"PASS");self.assertIsNotNone(r["initial_motion_sustained_reversal_start_s"]);self.assertFalse(r["upper_stopping_observation_eligible"])
        def test_rate_uses20ms_not5ms(self):
            a={"control_tick_id_for_step":np.repeat(np.arange(10),4),"u_prev_rad":np.zeros(20),"u_readback_for_step":np.full((40,20),.1)};r=target_rate(a,.02);self.assertEqual(r["target_speed_abs_max_rad_s"],5.)
        def test_force_activity_unknown_is_not_false(self):
            a={"actuator_torque_Nm":np.zeros((41,20)),"force_limit_Nm":np.ones(20),"joint_limit_active":np.zeros((41,20))};self.assertEqual(forces(a,{"availability":{"joint_limit_active":False}},0)["joint_limit_constraint_activity"],"UNKNOWN")
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=ROOT/"outputs/braking_analysis_cpu_checks_v1";out.mkdir(parents=True,exist_ok=True)
    record={"scope":"synthetic mathematical/negative parser-unit checks only; no measured physical response, no GPU/simulator", "tests_run":result.testsRun,"passed":result.wasSuccessful(),"failures":len(result.failures),"errors":len(result.errors),"source_sha256":sha(__file__),"settings":SETTINGS,"log":stream.getvalue()}
    write(out/"checks.json",record);(out/"tests.log").write_text(stream.getvalue(),encoding="utf-8");print(json.dumps(record,indent=2));return result.wasSuccessful()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",type=Path);p.add_argument("--plan",type=Path,default=ROOT/"calibration_plan_v1.json");p.add_argument("--output",type=Path,default=ROOT/"outputs/braking_response_analysis_v1");p.add_argument("--self-test",action="store_true");a=p.parse_args()
    if a.self_test:raise SystemExit(0 if self_test() else 1)
    if a.manifest is None:p.error("--manifest is required for raw analysis; no synthetic physical numbers are generated")
    analyze(a)


if __name__=="__main__":main()
