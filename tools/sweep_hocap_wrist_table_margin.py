"""CPU sensitivity study of the development S1 wrist normal offset on the original inclined table."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import mujoco as mj
import numpy as np
from fit_hocap_wrist_table_clearance import PARENT, ROOT, contacts, desired_gaps
from evaluate_hocap_carton import evaluate
from package_hocap_replay import package
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.ego.source_stage_contract import score_frozen_source_stages
from supermimic_research.strict_json import write_json


def measure(model, trace):
    """Query the native signed distances at all actual states, including the last frame."""
    table,hand=contacts(model)
    hand=sorted(hand)
    d,observer=mj.MjData(model),mj.MjData(model)
    d.qpos[:],d.qvel[:],d.ctrl[:]=trace["sim_qpos"][0],trace["sim_qvel"][0],trace["finger_ctrl"][0]
    mj.mj_forward(model,d);mj.mj_forward(model,d)
    def gap():
        observer.qpos[:]=d.qpos
        mj.mj_kinematics(model,observer)
        return min(mj.mj_geomDistance(model,observer,table,g,1.,None) for g in hand)
    gaps=[]
    error=0.
    for frame,(lo,hi) in enumerate(zip(trace["frame_end_substep"][:-1],trace["frame_end_substep"][1:]),1):
        for tick in range(lo,hi):
            gaps.append(gap())
            d.ctrl[:]=trace["substep_finger_ctrl"][tick]
            d.qfrc_applied[:6]=trace["substep_wrist_generalized_wrench"][tick]
            mj.mj_step(model,d)
        mj.mj_forward(model,d)
        error=max(error,float(np.max(abs(d.qpos-trace["sim_qpos"][frame]))),
                  float(np.max(abs(d.qvel-trace["sim_qvel"][frame]))))
    gaps.append(gap())
    if error != 0.:raise ValueError(f"Command replay diverged: {error:g}")
    return 1000*np.asarray(gaps)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"runs/hocap_carton_table_margin_sensitivity_v1")
    out=parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False,parents=True)
    release=json.loads((PARENT/"report.json").read_bytes())
    parent=PARENT / f"shift_{release['chosen']:+d}" / "report.json"
    desc=json.loads(parent.read_bytes());source=verify_file_pin(desc["source_report"])
    stage=verify_file_pin(release["source_stage_definition"]); contract=json.loads(stage.read_bytes())
    masks=stage.with_name("source_intervals.npz");protocol=ROOT/"configs/hocap_carton_evaluation.json"
    if file_pin(masks)["sha256"]!=contract["intervals_sha256"]:raise ValueError("Changed stage masks")
    with np.load(masks) as f:stages=dict(f)
    with np.load(verify_file_pin(desc["trace"])) as f:base=dict(f)
    initial_model=verify_file_pin(desc["mjb"])
    baseline_model=mj.MjModel.from_binary_path(str(initial_model))
    gaps,normal=desired_gaps(baseline_model,base["target_wrist"],base["target_finger_q"])
    times=base["source_nominal_time_s"]
    src_force=baseline_model.body_mass[baseline_model.body("carton").id]*np.linalg.norm(baseline_model.opt.gravity)*np.diff(times)
    result=dict(parent=file_pin(parent),source=file_pin(source),original_model=file_pin(initial_model),
        source_stage=file_pin(stage),original_protocol=file_pin(protocol),script=file_pin(__file__),
        target_actual_clearance_mm=3.,command_margins_mm=[4,6,8,10],
        rule="Test all four margins at 0.4 m/s. If none qualify, also test all at 0.2 and 0.6 m/s. "
             "Require all original development gates, frozen source-stage gates including stable release, and actual minimum hand-table distance >=3 mm. "
             "Among eligible runs minimize RMS wrist-target translation, then peak translation. "
             "3 mm is a development engineering target, not measured source uncertainty.",
        prior_2mm_comparison=file_pin(ROOT/"runs/hocap_s1_table_clear_independent_audit_v1/report.json"),tested=[],chosen=None)
    write_json(out/"comparison.json",result)
    start=perf_counter()
    for rate in (.4,.2,.6):
        for margin in result["command_margins_mm"]:
            folder=out/f"margin_{margin}mm_rate_{int(rate*100)}cm_s";folder.mkdir()
            model=mj.MjModel.from_binary_path(str(initial_model))
            needed=np.maximum(0.,margin/1000-gaps)
            shift=np.maximum(0.,(needed[:,None]-rate*np.abs(times[:,None]-times[None,:])).max(axis=0))
            wrists=base["target_wrist"].copy();wrists[:,:3,3]+=shift[:,None]*normal
            model.qpos0[:3]+=shift[0]*normal
            mj.mj_saveModel(model,str(folder/"scene.mjb"),None)
            model=mj.MjModel.from_binary_path(str(folder/"scene.mjb"))
            tr=control.replay(model,wrists,base["target_finger_q"],times,desc["wrist"])
            measured=measure(model,tr);idx=int(measured.argmin())
            for key in ("world_from_carton_source","source_right_hand21","source_nominal_time_s","target_finger_q"):
                tr[key]=base[key]
            tr.update(target_wrist=wrists,table_clearance_offset_m=shift)
            np.savez_compressed(folder/"trace.npz",**tr)
            np.save(folder/"actual_gaps_mm.npy",measured)
            run={k:desc[k] for k in ("source_report","wrist","physics_dt_s","nominal_fps","table")}
            run.update(trace=file_pin(folder/"trace.npz"),mjb=file_pin(folder/"scene.mjb"),parent=file_pin(parent),
                       independent_substep_action_replay_max_state_error=0.)
            write_json(folder/"report.json",run)
            evaluate(folder/"report.json",source,protocol,folder/"phase_evaluation.json")
            legacy=json.loads((folder/"phase_evaluation.json").read_bytes())
            gates,details=score_frozen_source_stages(stages,contract,contract["config"],tr["world_from_carton_sim"],
                tr["interval_hand_carton_normal_impulse_Ns"][1:]/src_force,tr["interval_table_carton_normal_impulse_Ns"][1:]/src_force,
                tr["interval_max_hand_contacts"][1:])
            passed=bool(legacy["passed_numeric_development_gates"] and all(gates.values()) and measured[idx]>=3.)
            trial=dict(folder=folder.name,margin_mm=margin,rate_m_s=rate,minimum_actual_gap_mm=float(measured[idx]),
                       minimum_tick=idx,measured_states=len(measured),rms_wrist_offset_mm=float(1000*np.sqrt(np.mean(shift**2))),
                       peak_wrist_offset_mm=float(1000*shift.max()),legacy_gates=legacy["gates"],source_gates=gates,
                       source_stage_metrics=details,qualified=passed)
            result["tested"].append(trial);result["search_wall_s"]=perf_counter()-start
            write_json(out/"comparison.json",result)
            print(folder.name,trial["minimum_actual_gap_mm"],passed,flush=True)
        if rate==.4 and any(row["qualified"] for row in result["tested"]):break
    qualified=[r for r in result["tested"] if r["qualified"]]
    if qualified:
        win=min(qualified,key=lambda row:(row["rms_wrist_offset_mm"],row["peak_wrist_offset_mm"]))
        folder=out/win["folder"]
        package(folder/"report.json",folder/"phase_evaluation.json",out/"export")
        archive=out/"export/Wuji-HOCap-carton-replay.zip"
        subprocess.run([sys.executable,str(ROOT/"tools/audit_hocap_stable_release_hand_table.py"),
                        "--source",str(archive),"--out",str(out/"independent_audit"),"--compare-with",
                        str(PARENT/"export/Wuji-HOCap-carton-replay.zip")],check=True)
        audit=json.loads((out/"independent_audit/report.json").read_bytes())
        if (abs(audit["worst"]["signed_gap_mm"]-win["minimum_actual_gap_mm"])>1e-6 or
                audit["worst"]["signed_gap_mm"]<3. or audit["all_substeps"]["solver_contact_steps"]!=0):
            raise ValueError("Independent geometric audit differs from candidate measurement")
        result["chosen"]=dict(folder=win["folder"],run=file_pin(folder/"report.json"),
                              package=file_pin(archive),audit=file_pin(out/"independent_audit/report.json"))
    result["total_wall_s"]=perf_counter()-start;write_json(out/"comparison.json",result)


if __name__=="__main__":
    main()
