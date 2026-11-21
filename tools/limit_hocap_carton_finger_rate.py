"""Measure how explicit joint-position target rate limits affect the nominal pick-and-place."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from ablate_hocap_carton_thumb import scores
from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def rate_limit(q,times,bound):
    # Midpoint of Lipschitz minorant/majorant envelopes avoids a one-sided command delay.
    allowance=bound*np.abs(times[:,None]-times[None,:])
    lower=np.min(q[None,:,:]+allowance[:,:,None],axis=1)
    upper=np.max(q[None,:,:]-allowance[:,:,None],axis=1)
    limited=(lower+upper)*.5
    initial_allowance=bound*(times-times[0])[:,None]
    limited=np.clip(limited,q[0]-initial_allowance,q[0]+initial_allowance)
    vmax=np.max(np.abs(np.diff(limited,axis=0))/np.diff(times)[:,None])
    if vmax>bound+1e-8: raise ValueError('Command rate bound not achieved')
    return limited,vmax


def run(parent_path,out):
    out.mkdir(parents=True,exist_ok=False)
    parent=json.loads(parent_path.read_bytes())
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(parent['mjb'])))
    keys=('target_finger_q','target_wrist','source_nominal_time_s','world_from_carton_source','source_right_hand21')
    with np.load(verify_file_pin(parent['trace']),allow_pickle=False) as archive:
        ref={k:archive[k] for k in keys}
        for key in ('source_pad_gate','pad_ik_cost'):
            if key in archive:
                ref[key]=archive[key]
    rows=[];best_score=None
    for bound in (4,6,8,10,12,16,20,30):
        q,peak=rate_limit(ref['target_finger_q'],ref['source_nominal_time_s'],bound)
        trace=control.replay(model,ref['target_wrist'],q,ref['source_nominal_time_s'],parent['wrist'])
        row=dict(target_slew_cap_rad_s=bound,measured_target_slew_rad_s=float(peak),
                 frame_endpoint_joint_speed_peak_rad_s=float(np.abs(trace['sim_qvel'][:,6:26]).max()),
                 **scores(trace,ref['world_from_carton_source'],parent['table']['unit_normal']))
        rows.append(row)
        score=(row['joint_excursion_rad']>np.deg2rad(2) or row['lift_mm']<200 or row['carton_pen_mm']>2,
               max(row['rms_mm']/20,row['rms_deg']/15),bound)
        if best_score is None or score<best_score:
            best_score=score;best=row;selected={**trace,**ref,'target_finger_q':q}
    actual=control.audit_commands(mujoco.MjModel.from_binary_path(str(verify_file_pin(parent['mjb']))),selected)
    mjb=out/'scene.mjb';mujoco.mj_saveModel(model,str(mjb),None)
    trace=out/'trace.npz';np.savez_compressed(trace,**selected)
    write_json(out/'report.json',dict(parent=file_pin(parent_path),source_report=parent['source_report'],source=file_pin(__file__),
               mjb=file_pin(mjb),trace=file_pin(trace),results=rows,best=best,wrist=parent['wrist'],table=parent['table'],
               mass_kg=float(model.body_mass[model.body('carton').id]),nominal_fps=parent['nominal_fps'],physics_dt_s=parent['physics_dt_s'],
               independent_substep_action_replay_max_state_error=actual,cap_is_hardware_spec=False,
               source_fidelity_qualified=False))
    print(json.dumps(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.parent,a.out)
