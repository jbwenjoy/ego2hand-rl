"""Explore digit-specific virtual preload on two explicitly hypothesized carton masses."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.ego import hocap_carton_grasp as grasp
from supermimic_research.artifacts import verify_file_pin,file_pin
from supermimic_research.strict_json import write_json


def run(root, light, heavy, out):
    out.mkdir(parents=True,exist_ok=False)
    report=json.loads(root.read_bytes())
    source=json.loads(verify_file_pin(report['source_report']).read_bytes())
    mesh=trimesh.load_mesh(verify_file_pin(source['geometry']['hull']),process=False)
    with np.load(verify_file_pin(report['trace']),allow_pickle=False) as a:
        hand=a['source_right_hand21']; obj=a['world_from_carton_source'];wrist=a['target_wrist']
        times=a['source_nominal_time_s'];finger=a['target_finger_q']
    ls,hs=[json.loads(p.read_bytes()) for p in (light,heavy)]
    models=[mujoco.MjModel.from_binary_path(str(verify_file_pin(rep['mjb']))) for rep in (ls,hs)]
    m=models[0]; joints=m.actuator_trnid[:,0];lo=m.jnt_range[joints,0];hi=m.jnt_range[joints,1]
    values=[10,14,18,22]
    runs={v:grasp.pad_corrected_ik(m,hand,obj,wrist,finger,mesh,.001*v) for v in values}
    nm=np.array(ls['table']['unit_normal'])
    rows=[];best=None
    for thumb in values:
        for fingers in values:
            for guard,smooth in [(0.,0.),(.06,.75)]:
                q=runs[fingers][0].copy()
                q[:,:4]=runs[thumb][0][:,:4]
                if smooth:
                    q=gaussian_filter1d(q,smooth,axis=0,mode='nearest')
                q=np.clip(q,lo+guard,hi-guard)
                outcomes=[]
                for model in models:
                    trace=control.replay(model,wrist,q,times,report['wrist'])
                    pred=trace['world_from_carton_sim']
                    err=np.linalg.norm(pred[:,:3,3]-obj[:,:3,3],axis=1)*1000
                    ang=np.rad2deg((Rotation.from_matrix(pred[:,:3,:3]).inv()*Rotation.from_matrix(obj[:,:3,:3])).magnitude())
                    lift=1000*((pred[:,:3,3]-pred[0,:3,3])@nm)
                    outcomes.append(dict(rms_mm=float(np.sqrt(np.mean(err**2))),rms_deg=float(np.sqrt(np.mean(ang**2))),
                                         peak_lift_mm=float(lift.max()),joint_excursion_rad=float(trace['interval_joint_excursion_rad'].max()),
                                         max_carton_pen_mm=float(1000*trace['interval_max_carton_penetration_m'].max())))
                row=dict(thumb_mm=thumb,opposing_mm=fingers,guard_rad=guard,smoothing_frames=smooth,results=outcomes)
                # Both masses contribute the worst normalized tracking and soft violation cost.
                score=max(max(o['rms_mm']/20,o['rms_deg']/15,o['joint_excursion_rad']/.035,
                              o['max_carton_pen_mm']/2) for o in outcomes)
                row['score']=score;rows.append(row)
                if best is None or best[0]>score:
                    best=(score,row)
                    np.savez_compressed(out/'selected_commands.npz',target_finger_q=q,source_pad_gate=runs[fingers][1],
                                        pad_ik_cost=runs[fingers][2])
    write_json(out/'report.json',dict(baseline=file_pin(root),light=file_pin(light),heavy=file_pin(heavy),
               source=file_pin(__file__),controls=file_pin(out/'selected_commands.npz'),
               rows=rows,best=best[1],mass_pair_kg=[ls['mass_kg'],hs['mass_kg']],
               exploration_only=True,independent_clips=0))
    print(json.dumps(best[1]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','light','heavy','out'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.root,a.light,a.heavy,a.out)
