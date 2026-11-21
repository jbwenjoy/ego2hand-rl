"""Check a bounded release-timing correction against the source placement and original whole-clip gates."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_hocap_carton import evaluate
from limit_hocap_carton_finger_rate import rate_limit
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.ego.source_stage_contract import score_frozen_source_stages
from supermimic_research.strict_json import write_json


def run(parent, stage, original_protocol, out):
    out.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    report=json.loads(parent.read_bytes())
    contract=json.loads(stage.read_bytes())
    intervals=stage.with_name("source_intervals.npz")
    if file_pin(intervals)["sha256"]!=contract["intervals_sha256"]:
        raise ValueError("Source interval artifact changed")
    with np.load(intervals,allow_pickle=False) as a: masks=dict(a)
    with np.load(verify_file_pin(report["trace"]),allow_pickle=False) as a: base=dict(a)
    source_path=verify_file_pin(report["source_report"])
    np.testing.assert_array_equal(masks["source_poses"],base["world_from_carton_source"])
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(report["mjb"])))
    weight=model.body_mass[model.body("carton").id]*np.linalg.norm(model.opt.gravity)
    q,clock=base["target_finger_q"],base["source_nominal_time_s"]
    frames=base["frame_index"]
    np.testing.assert_array_equal(frames,np.arange(141))
    rows=[]
    chosen=None
    # Start each variant with identical applied commands up through source frame 105.
    # A positive shift advances the finger commands during table handover.
    for lead in (0,1,-1,2,-2,3,-3):
        folder=out/f"shift_{lead:+d}"
        folder.mkdir()
        targets=q.copy()
        if lead:
            shifted=frames+lead*np.interp(frames,[0,105,116,132,140],[0,0,1,1,0])
            candidate=np.column_stack([np.interp(shifted,frames,j) for j in q.T])
            targets[105:],_=rate_limit(candidate[105:],clock[105:],30.)
        np.testing.assert_array_equal(targets[:106],q[:106])
        tr=control.replay(model,base["target_wrist"],targets,clock,report["wrist"])
        err=control.audit_commands(mujoco.MjModel.from_binary_path(str(verify_file_pin(report["mjb"]))),tr)
        if lead==0:
            np.testing.assert_array_equal(tr["sim_qpos"],base["sim_qpos"])
        for k in ("world_from_carton_source","source_right_hand21","source_nominal_time_s","target_wrist"):
            tr[k]=base[k]
        tr["target_finger_q"]=targets
        np.savez_compressed(folder/"trace.npz",**tr)
        common={k:report[k] for k in ("mjb","source_report","wrist","physics_dt_s","nominal_fps","table")}
        write_json(folder/"report.json",dict(**common,parent=file_pin(parent),trace=file_pin(folder/"trace.npz"),
                    release_shift_frames=lead,independent_substep_action_replay_max_state_error=err))
        evaluate(folder/"report.json",source_path,original_protocol,folder/"phase_evaluation.json")
        legacy=json.loads((folder/"phase_evaluation.json").read_bytes())
        poses=tr["world_from_carton_sim"]
        den=weight*np.diff(clock)
        gates,metrics=score_frozen_source_stages(masks,contract,contract["config"],poses,
                    tr["interval_hand_carton_normal_impulse_Ns"][1:]/den,tr["interval_table_carton_normal_impulse_Ns"][1:]/den,
                    tr["interval_max_hand_contacts"][1:])
        suffix=poses[-(contract["config"]["release_suffix_intervals"]+1):]
        terminal_translation=1000*np.max(np.linalg.norm(suffix[:,:3,3]-suffix[0,:3,3],axis=1))
        terminal_angle=np.rad2deg(Rotation.from_matrix(suffix[0,:3,:3].T@suffix[:,:3,:3]).magnitude()).max()
        passed=bool(legacy["passed_numeric_development_gates"] and all(gates.values()))
        if passed and chosen is None:chosen=lead
        rows.append(dict(shift_frames=lead,passed_both=passed,legacy=legacy["gates"],stages=gates,
                         release_movement_mm=terminal_translation,release_rotation_deg=terminal_angle,
                         source_stage_metrics=metrics,physical_metrics=legacy["metrics"],trajectory=file_pin(folder/"report.json")))
    write_json(out/"report.json",dict(original=file_pin(parent),source_stage_definition=file_pin(stage),
               code=file_pin(__file__),selection="First passing in declared increasing absolute release shift; positive tested before negative.",
               chosen=chosen,elapsed_s=perf_counter()-start,results=rows,scope="Development G04_2 only. Nominal free-object physics."))
    print([(r["shift_frames"],r["passed_both"],round(r["release_movement_mm"],2)) for r in rows])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("parent","stage","protocol","out"):p.add_argument(k,type=Path)
    a=p.parse_args()
    run(a.parent,a.stage,a.protocol,a.out)
