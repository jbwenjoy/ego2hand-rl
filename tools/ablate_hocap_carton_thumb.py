"""Local acquisition command ablation at a measured thumb limit overshoot."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def scores(a,obj,n):
    pred=a['world_from_carton_sim']
    pos=np.linalg.norm(pred[:,:3,3]-obj[:,:3,3],axis=1)*1000
    ang=np.rad2deg((Rotation.from_matrix(pred[:,:3,:3]).inv()*Rotation.from_matrix(obj[:,:3,:3])).magnitude())
    lift=1000*((pred[:,:3,3]-pred[0,:3,3])@n)
    return dict(rms_mm=float(np.sqrt(np.mean(pos**2))),max_mm=float(pos.max()),
                rms_deg=float(np.sqrt(np.mean(ang**2))),max_deg=float(ang.max()),
                lift_mm=float(lift.max()),end_mm=float(pos[-1]),
                joint_excursion_rad=float(a['interval_joint_excursion_rad'].max()),
                carton_pen_mm=float(1000*a['interval_max_carton_penetration_m'].max()))


def run(parent_path,out):
    out.mkdir(parents=True,exist_ok=False)
    parent=json.loads(parent_path.read_bytes())
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(parent['mjb'])))
    with np.load(verify_file_pin(parent['trace']),allow_pickle=False) as a:
        original={k:a[k] for k in ('target_finger_q','target_wrist','source_nominal_time_s','world_from_carton_source',
                                    'source_right_hand21','source_pad_gate','pad_ik_cost')}
    q0=original['target_finger_q'];steps=np.arange(len(q0))
    # Smooth raised-cosine onset/offset, fully active across f26-31.
    gate=np.interp(steps,[0,21,26,31,36,140],[0,0,1,1,0,0])
    gate=.5-.5*np.cos(np.pi*gate)
    jids=model.actuator_trnid[:,0]
    thumb=int(np.flatnonzero(jids==model.joint('right_finger1_joint4').id)[0])
    lo,hi=model.jnt_range[jids].T
    rows=[];best_score=None
    for all_digits in (False,True):
        for margin,sigma in ((.05,0.),(.09,0.),(.14,0.),(.09,.65),(.09,1.),(.17,1.),(.18,1.5)):
            affected=np.arange(20) if all_digits else np.array([thumb])
            subset=q0[:,affected]
            filtered=gaussian_filter1d(subset,sigma,axis=0,mode='nearest') if sigma else subset
            guarded=np.clip(filtered,lo[affected]+margin,hi[affected]-margin)
            q=q0.copy()
            q[:,affected]+=gate[:,None]*(guarded-subset)
            a=control.replay(model,original['target_wrist'],q,original['source_nominal_time_s'],parent['wrist'])
            row=dict(all_digits=all_digits,margin_rad=margin,gaussian_sigma_frames=sigma,
                     **scores(a,original['world_from_carton_source'],parent['table']['unit_normal']))
            rows.append(row)
            # Preserve the lift and obey the original 2-degree dynamic limit criterion.
            score=(row['joint_excursion_rad']>np.deg2rad(2) or row['lift_mm']<200,
                   max(row['rms_mm']/20,row['rms_deg']/15),row['joint_excursion_rad'])
            if best_score is None or score<best_score:
                best_score=score
                best=row
                selected={**a,**original,'target_finger_q':q}
            if not all_digits and row['lift_mm']>200 and row['joint_excursion_rad']<=np.deg2rad(2):
                print(json.dumps(row),flush=True)
    err=control.audit_commands(mujoco.MjModel.from_binary_path(str(verify_file_pin(parent['mjb']))),selected)
    mjb=out/'scene.mjb';mujoco.mj_saveModel(model,str(mjb),None)
    trace=out/'trace.npz';np.savez_compressed(trace,**selected)
    write_json(out/'report.json',dict(parent=file_pin(parent_path),source_report=parent['source_report'],
               source=file_pin(__file__),mjb=file_pin(mjb),trace=file_pin(trace),results=rows,best=best,
               independent_substep_action_replay_max_state_error=err,wrist=parent['wrist'],table=parent['table'],
               mass_kg=parent['mass_kg'],nominal_fps=parent['nominal_fps'],physics_dt_s=parent['physics_dt_s'],
               source_fidelity_qualified=False))
    print('BEST',json.dumps(best))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();run(args.parent,args.out)
