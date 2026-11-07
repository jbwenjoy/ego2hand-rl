"""Author saved-only description of every interval of the one MPC pilot.

No model reload, framework import, optimizer, forward or physics operation.
Thresholds 0 and .1 N are descriptive/no-positive and the original loaded-pad
threshold; all ticks, intervals and line-search candidates are retained.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent
OUT = ROOT / "failure_mechanism_saved_v1"
DT = .00015625


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return {"path":str(p),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_record(rec):
    assert pin(rec["path"]) == rec
    with np.load(rec["path"],allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def quat_R(q):
    w,x,y,z = q
    assert abs(float(q@q)-1) < 1e-12
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def angle_R(r,r0):
    return float(np.arccos(np.clip((np.trace(r0.T@r)-1)/2,-1,1)))


def response_metrics(e, initial_COM, initial_R):
    rows=[]
    for snap in e["snapshots"]:
        s=snap["state"]
        rows.append({"horizon_POST_s":snap["elapsed_time_nominal_s"],
            "COM_drift_mm":1000*float(np.linalg.norm(np.asarray(s["COM_world_m"])-initial_COM)),
            "orientation_deg":math.degrees(angle_R(np.asarray(s["R_body_world"]),initial_R)),
            "COM_speed_m_s":float(np.linalg.norm(s["v_COM_world_m_s"])),
            "angular_speed_rad_s":float(np.linalg.norm(s["omega_world_rad_s"])),
            "last_solver_PRE_pad_normal_N":snap["pad_normal_N"]})
    return rows


def runs(mask):
    ii=np.flatnonzero(mask)
    if not len(ii):return []
    groups=np.split(ii,np.flatnonzero(np.diff(ii)>1)+1)
    return [{"first_PRE_tick":int(g[0]),"last_PRE_tick":int(g[-1]),"samples":len(g),
        "first_PRE_s":float(g[0]*DT),"last_PRE_s":float(g[-1]*DT),
        "sampled_intervals_total_s":float(len(g)*DT)} for g in groups]


def force_profiles(tr):
    fn=tr["pre_pad_normal_N"]
    result={}
    for threshold in (0.,.1):
        masks={"thumb":fn[:,0]>threshold,"index":fn[:,1]>threshold,
               "both":np.all(fn>threshold,axis=1),"neither":np.all(fn<=threshold,axis=1)}
        result[str(threshold)]={name:{"sample_count":int(mask.sum()),"fraction":float(mask.mean()),
            "runs":runs(mask)} for name,mask in masks.items()}
    return result


def endpoint(tr,tick,initial_COM):
    i=tick-1
    return {"POST_tick":tick,"nominal_s":tick*DT,
        "COM_drift_mm":1000*float(np.linalg.norm(tr["post_COM"][i]-initial_COM)),
        "orientation_deg":math.degrees(float(tr["post_angle"][i])),
        "last_solver_PRE_pad_normal_N":tr["pre_pad_normal_N"][i].tolist()}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    run=ROOT/"feasibility_fine_v1"
    producer=read(run/"report.json");protocol=read(producer["protocol"]["path"])
    assert pin(producer["protocol"]["path"])==producer["protocol"]
    tr=load_record(producer["trace"])
    baseline_path=protocol["inputs"]["matched_constant_u_report"]["path"]
    assert pin(baseline_path)==protocol["inputs"]["matched_constant_u_report"]
    baseline=read(baseline_path)
    bcase=next(row for row in baseline["cases"] if row["dt_s"]==DT)
    btr=load_record(bcase["trace"])
    assert bcase["source_candidate_state"]==protocol["inputs"]["candidate_state"]
    assert np.array_equal(tr["initial_qpos27"],btr["initial_qpos27"])
    assert np.array_equal(tr["initial_ctrl20"],btr["ctrl20"])
    assert np.array_equal(tr["initial_COM_world3"],btr["initial_COM_world3"])
    assert np.array_equal(tr["pre_tick"],np.arange(len(tr["pre_tick"])))
    assert np.array_equal(tr["post_tick"],tr["pre_tick"]+1)
    assert len(tr["post_tick"])==producer["actual_integration_steps"]
    ipath=PILOT/"hand2_stage1/completed_v2/joint_actuator_interface.json"
    interface=read(ipath)
    controls=np.array([a["ctrl_range"] for a in interface["actuators"]])
    assert interface["joint_order"]==producer["actuator_joint_names"]
    initial_COM=tr["initial_COM_world3"];initial_R=quat_R(tr["initial_qpos27"][23:27]);u0=tr["initial_ctrl20"]
    intervals=[];all_lines=[];all_control_rows=[];pins=[]
    previous=u0.copy()
    for path in sorted(run.glob("interval_*.json")):
        raw=read(path);pins.append(pin(path))
        assert raw["interval"]==len(intervals) and raw["start_tick"]==128*len(intervals)
        evs=raw["evaluations"];lookup={e["branch_index"]:e for e in evs}
        start=evs[0];selected=lookup[raw["selected_branch"]]
        assert start["purpose"]=="previous_control"
        assert selected["cost"]==raw["selected_cost"]
        u=np.array(raw["selected_control20"])
        assert np.array_equal(np.array(selected["controls2x20"]),np.tile(u,(2,1)))
        assert np.array_equal(np.array(start["controls2x20"]),np.tile(previous,(2,1)))
        low=np.maximum(controls[:8,0],previous[:8]-.05)
        high=np.minimum(controls[:8,1],previous[:8]+.05)
        assert np.all(u>=controls[:,0]) and np.all(u<=controls[:,1])
        assert np.all(u[:8]>=low) and np.all(u[:8]<=high)
        assert np.array_equal(u[8:],u0[8:])
        sl=slice(raw["start_tick"],min(raw["start_tick"]+128,len(tr["post_tick"])))
        assert np.array_equal(tr["chosen_ctrl20"][sl],np.tile(u,(len(tr["post_tick"][sl]),1)))
        base_m=response_metrics(start,initial_COM,initial_R);sel_m=response_metrics(selected,initial_COM,initial_R)
        for record in raw["bounded_GN"]:
            for line in record["line_exact_costs"]:
                e=lookup[line["branch_index"]]
                m=response_metrics(e,initial_COM,initial_R)
                all_lines.append({"interval":raw["interval"],"GN_iteration":record["iteration"],
                    "scale":line["scale"],"branch_index":line["branch_index"],"exact_cost":line["cost"],
                    "GN_base_exact_cost":record["base_exact_cost"],
                    "cost_lower_than_GN_base":bool(line["cost"]<record["base_exact_cost"]),
                    "horizon_end_orientation_deg":m[-1]["orientation_deg"],
                    "original_horizon_safety_limits_pass":e.get("original_horizon_safety_limits_pass"),
                    "selected_final":line["branch_index"]==raw["selected_branch"]})
        fn=tr["pre_pad_normal_N"][sl]
        row={"interval":raw["interval"],"source_interval":pin(path),"actual_steps":len(fn),
            "start_nominal_s":raw["start_tick"]*DT,"actual_end_nominal_s":(raw["start_tick"]+len(fn))*DT,
            "branch_evaluations":len(evs),"previous_control_cost":start["cost"],"selected_cost":selected["cost"],
            "cost_decrease":start["cost"]-selected["cost"],
            "predicted_horizon_previous_control":base_m,"predicted_horizon_selected":sel_m,
            "selected_cost_lower_and_horizon_rotation_worse":bool(selected["cost"]<start["cost"] and sel_m[-1]["orientation_deg"]>base_m[-1]["orientation_deg"]),
            "actual_both_pad_positive_fraction":float(np.mean(np.all(fn>0,axis=1))),
            "actual_both_pad_gt_0p1N_fraction":float(np.mean(np.all(fn>.1,axis=1))),
            "actual_mean_pad_N":fn.mean(axis=0).tolist(),"actual_max_pad_N":fn.max(axis=0).tolist(),
            "actual_end":endpoint(tr,raw["start_tick"]+len(fn),initial_COM),
            "actual_vs_selected_full_integration_source_receipt":raw["actual_vs_selected_first_state_raw_bit_equal"],
            "selected_control20_rad":u.tolist(),"delta_from_u0_rad":(u-u0).tolist(),
            "delta_from_previous_interval_rad":(u-previous).tolist(),
            "minimum_control_range_margin_rad":float(np.min(np.minimum(u-controls[:,0],controls[:,1]-u))),
            "minimum_trust_margin_rad":float(np.min(np.minimum(u[:8]-low,high-u[:8]))),
            "FD_records": [{"GN_iteration":r["iteration"],"actual_FD_steps_rad":r["FD_steps_rad"],
                "status":r["status"],"success":r["success"]} for r in raw["bounded_GN"]]}
        intervals.append(row)
        for j in range(20):
            all_control_rows.append({"interval":raw["interval"],"actuator_index":j,"name":interface["actuators"][j]["name"],
                "optimized":j<8,"original_u_rad":float(u0[j]),"selected_u_rad":float(u[j]),
                "delta_original_rad":float(u[j]-u0[j]),"delta_previous_rad":float(u[j]-previous[j]),
                "saved_compiled_ctrl_range_rad":controls[j].tolist()})
        previous=u
    common=min(len(tr["post_tick"]),len(btr["post_time"]))
    ticks=sorted(set([min((i+1)*128,common) for i in range((common+127)//128)]))
    comparisons=[{"MPC":endpoint(tr,t,initial_COM),"constant_u":endpoint(btr,t,initial_COM)} for t in ticks]
    profiles={"MPC":force_profiles(tr),"constant_u_fine":force_profiles(btr)}
    contact_counts=[]
    for pair in np.unique(tr["contact_rows24"][:,1:3].astype(np.int64),axis=0):
        mask=np.all(tr["contact_rows24"][:,1:3]==pair,axis=1)
        rows=tr["contact_rows24"][mask]
        contact_counts.append({"ordered_native_geom_ids":pair.tolist(),"all_generated_rows":len(rows),
            "first_PRE_tick":int(rows[:,0].min()),"last_PRE_tick":int(rows[:,0].max()),
            "first_PRE_s":float(rows[:,0].min()*DT),"last_PRE_s":float(rows[:,0].max()*DT),
            "native_fn_positive_rows":int(np.sum(rows[:,18]>0))})
    result={"schema":"native_MPC_saved_failure_mechanism_author_description:1", "author_not_independent":True,
        "producer_report":pin(run/"report.json"),"producer_protocol":pin(producer["protocol"]["path"]),
        "MPC_trace":producer["trace"],"baseline_report":pin(baseline_path),"baseline_fine_trace":bcase["trace"],
        "saved_compiled_actuator_interface":pin(ipath),"source":pin(__file__),"interval_pins":pins,
        "definitions":{"all7_intervals_all_lines_retained":True,"pad_force_thresholds_N":[0.,.1],
            "force_phase":"solver PRE tick i, state POST tick i+1","native_geom_generated_not_contact_truth":True,
            "bounds_scope":"Previous saved compiled interface plus current runtime source assertion; no MJB readback in this analysis.",
            "previous_control_cost":"Exact H2 hold-last-selected-u from the same current MPC source, not constant-original-u after interval0.",
            "constant_u_comparison":"Same initial qpos/u/COM and fine dt; at later times states diverge, not same-current-state counterfactuals."},
        "counts":{"intervals":len(intervals),"line_records":len(all_lines),"actual_steps":len(tr["post_tick"]),
            "planning_branches":sum(r["branch_evaluations"] for r in intervals),"saved_NPZ_reads_in_this_script":2,
            "new_model_forward_step_NN_GPU_calls":0},
        "control_summary":{"all_controls_inside_saved_bounds":True,"all_steps_inside_previous_u_trust":True,
            "rest12_bit_equal_original":True,"max_abs_delta_from_original_rad":float(np.max(np.abs(tr["chosen_ctrl20"]-u0))),
            "max_abs_interval_delta_rad":max(max(abs(v) for v in r["delta_from_previous_interval_rad"]) for r in intervals)},
        "intervals":intervals,"all_interval_actuator_rows":all_control_rows,"all_line_search_records":all_lines,
        "all_tick_contact_force_profiles":profiles,"generated_pair_profiles_native_IDs":contact_counts,
        "matched_time_trajectory_comparison":comparisons,
        "terminal_comparison":{"MPC":endpoint(tr,len(tr["post_tick"]),initial_COM),
            "constant_u":endpoint(btr,len(btr["post_time"]),initial_COM)},
        "all_complete_first_intervals_source_bit_receipt":producer["all_complete_selected_intervals_raw_bit_equal"],
        "first_interval_full_raw_state_limit":"Actual complete integration vectors are not saved; full 300-value equality is source assertion and SHA receipt, not independently reconstructed from qpos/qvel trace.",
        "holding_qualified":False,"causal_uniqueness_or_global_infeasibility_claim":False}
    assert not any(n in sys.modules for n in ("torch","mujoco","mujoco_warp","warp"))
    OUT.mkdir()
    (OUT/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"report":pin(OUT/"report.json"),"controls":result["control_summary"],
        "contact_profiles":profiles,"intervals":[{k:r[k] for k in ("interval","previous_control_cost","selected_cost","cost_decrease","actual_both_pad_gt_0p1N_fraction","actual_end","selected_cost_lower_and_horizon_rotation_worse")} for r in intervals],
        "terminal":result["terminal_comparison"]}))


if __name__=="__main__":main()
