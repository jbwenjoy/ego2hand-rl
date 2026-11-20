"""Attribute observed shaft-angle change to actual and reference motion, with contact torque."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from analyse_pen_axis_contacts import ROOT,arc
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.strict_json import write_json


def inspect_case(pin):
    case=json.loads(verify_file_pin(pin).read_bytes())
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])))
    body=case['model_layout']['object_body']
    qa=case['model_layout']['object_qposadr']
    ids=case['qids']
    gids=[normal_group(model.joint(int(model.dof_jntid[k])).name) for k in case['dofs']]
    assert sorted(gids)==sorted(list(range(5))*4)
    with np.load(verify_file_pin(case['trace'])) as data:
        ref=Slerp(data['time_s'],Rotation.from_quat(data['quaternion_wxyz'][:,[1,2,3,0]]))
    phase_rows=[]
    for p in case['controls'][205:225]:
        c=json.loads(verify_file_pin(p).read_bytes())
        phase=c['phase']
        t0=(phase-1)*.02+np.arange(8)*.0025
        b0=ref(t0).as_matrix()[:,:,0]; b1=ref(t0+.0025).as_matrix()[:,:,0]
        with np.load(verify_file_pin(c['trace'])) as z:
            pre=z['pre_qpos'];post=z['post_qpos']
            a0=Rotation.from_quat(pre[:,qa+np.array([4,5,6,3])]).as_matrix()[:,:,0]
            a1=Rotation.from_quat(post[:,qa+np.array([4,5,6,3])]).as_matrix()[:,:,0]
            ang0=arc(a0,b0);ang1=arc(a1,b1)
            # Average the two finite-difference orders, symmetric and exactly additive.
            moved=.5*((arc(a1,b0)-ang0)+(ang1-arc(a0,b1)))
            target=.5*((arc(a0,b1)-ang0)+(ang1-arc(a1,b0)))
            np.testing.assert_allclose(moved+target,ang1-ang0,atol=1e-9)
            direction=np.cross(a0,b0)
            norm=np.linalg.norm(direction,axis=1)
            if norm.min()<1e-6:raise ValueError('Undefined axis correction')
            direction/=norm[:,None]
            torque=np.zeros((8,6,3));force=np.zeros_like(torque)
            for row in z['raw_contact_tick_plus23']:
                tick,g,h=map(int,row[:3]);on=model.geom_bodyid[[g,h]]==body
                if on.sum()!=1:continue
                digit=normal_group(model.geom(h if on[0] else g).name or '')
                if digit<0:
                    if np.linalg.norm(row[18:24])>1e-9:raise ValueError('Non-hand contact load')
                    continue
                frame=row[9:18].reshape(3,3)
                sign=1 if on[1] else -1
                f=sign*frame.T@row[18:21]
                force[tick,digit]+=f
                torque[tick,digit]+=sign*frame.T@row[21:24]+np.cross(row[6:9]-z['pre_object_COM'][tick],f)
            with np.load(verify_file_pin(c['window'])) as w:
                np.testing.assert_allclose(np.r_[force.sum((0,1)),torque.sum((0,1))],w['wrench_sum'][1],atol=1e-8)
            src=torch.load(verify_file_pin(c['control_source']),map_location='cpu',weights_only=False)
            refq=np.asarray(src['policy_observation'])[0,:20]
            pd=np.array(c['actual_PD20'])
            np.testing.assert_array_equal(z['post_ctrl'][:,c['aids']],np.broadcast_to(pd,(8,20)))
            phase_rows.append(dict(phase=phase,angle0=ang0[0],angle1=ang1[-1],actual_contribution=moved.sum(),
                reference_contribution=target.sum(),corrective_mNm=(1000*np.einsum('tij,tj->ti',torque,direction)).mean(0).tolist(),
                target_minus_ref=(pd-refq).tolist(),actual_minus_target=(post[:,ids]-pd).tolist(),actual_minus_ref=(post[:,ids]-refq).tolist()))
    rows=phase_rows[:17]  # 4.10 to peak at 4.44, same window for both arms
    rms=lambda x:np.sqrt(np.mean(np.square(x)))
    means={}
    for key in ['target_minus_ref','actual_minus_target','actual_minus_ref']:
        v=np.asarray([r[key] for r in rows])
        means[key+'_rad_rms_by_finger']=[float(rms(v[...,np.asarray(gids)==digit])) for digit in range(5)]
    return dict(case=pin,phases=phase_rows,to_4p44=dict(start_deg=rows[0]['angle0'],end_deg=rows[-1]['angle1'],
                actual_deg=sum(r['actual_contribution'] for r in rows),reference_deg=sum(r['reference_contribution'] for r in rows),
                mean_corrective_mNm=np.mean([r['corrective_mNm'] for r in rows],axis=0).tolist(),**means))


def main(out):
    out.mkdir(exist_ok=False)
    results={}
    for arm,run in [('residual','pen_scaled_train_v1'),('support','pen_scaled_support_matched_v1')]:
        source=ROOT/'runs'/run/'report.json'
        e=json.loads(verify_file_pin(json.loads(source.read_bytes())['evaluations'][-1]).read_bytes())
        results[arm]=[inspect_case(row['report']) for row in e['cases']]
        print(arm,[row['to_4p44'] for row in results[arm]])
    write_json(out/'report.json',dict(cases=results,units='Angular change in degrees; torque projected on actual-axis cross reference-axis in mNm; joint errors in radians. Descriptive motion attribution, not intervention.'))
if __name__=='__main__':main(Path(sys.argv[1]))
