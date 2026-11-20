"""Test bounded joint-servo feedforward from reference contact wrenches."""
from pathlib import Path
import json,sys
import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from track_mug_tip_sequence import ROOT
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench
from supermimic_research.mechanics.multiclip_wrench_screen import contact_rays
from supermimic_research.strict_json import write_json


def wrench_target(m,d,b,hand,w,qnext,original):
    K=m.actuator_gainprm[:,0]
    col=[];react=[]
    for c in d.contact:
        g,h=map(int,c.geom);on=m.geom_bodyid[[g,h]]==b
        if on.sum()!=1 or c.dist>0:continue
        other=h if on[0] else g
        if other not in hand:continue
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[other]))
        arm=c.pos-d.xipos[b]
        for ray in contact_rays(c,1 if on[1] else -1,d.xipos[b]):
            col.append(ray);react.append((-jp.T@ray[:3]-jr.T@(ray[3:]-np.cross(arm,ray[:3])))[:20])
    if not col:return None,'empty'
    A=np.array(col).T;B=-np.array(react).T
    bias=(d.qfrc_bias-d.qfrc_passive)[:20]
    lo=np.maximum(m.actuator_forcerange[:,0],K*(m.actuator_ctrlrange[:,0]-qnext));hi=np.minimum(m.actuator_forcerange[:,1],K*(m.actuator_ctrlrange[:,1]-qnext))
    # tau = bias + B*alpha; minimize normalized distance from the archived preload torque.
    scale=np.array([1,1,1,100,100,100]);zero=np.zeros((20,20));eye=np.eye(20)
    ub=np.block([[B,zero],[-B,zero],[B,-eye],[-B,-eye]])
    rhs=np.r_[hi-bias,bias-lo,K*original-bias,bias-K*original]
    lp=linprog(np.r_[np.full(len(col),.001),1/m.actuator_forcerange[:,1]],A_eq=np.c_[scale[:,None]*A,np.zeros((6,20))],b_eq=scale*w,A_ub=ub,b_ub=rhs,bounds=(0,None),method='highs')
    if not lp.success:return None,str(lp.status)
    alpha=lp.x[:len(col)];tau=bias+B@alpha
    np.testing.assert_allclose(A@alpha,w,rtol=0,atol=2e-7)
    if np.any(tau<lo-1e-7) or np.any(tau>hi+1e-7):raise ValueError('Infeasible joint torque')
    return qnext+tau/K,'ok'


def main(parent,out):
    out.mkdir(exist_ok=False)
    trial=json.loads((parent/'contact_ik/report.json').read_bytes());intent=json.loads(verify_file_pin(trial['intent']).read_bytes())
    binding=json.loads(verify_file_pin(intent['binding']).read_bytes());ref=json.loads(verify_file_pin(intent['reference']).read_bytes())
    write_json(out/'intent.json',dict(parent=file_pin(parent/'report.json'),design='Instantaneous quasi-static hand torque bounds under nominal interval-average cup wrench. Minimize deviation from archived preload. Default original preload for infeasible/undefined intervals. No external force.'))
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(ref['model'])))
    with np.load(verify_file_pin(ref['output'])) as f:z=dict(f)
    with np.load(verify_file_pin(trial['trajectory'])) as f:q=f['q20']
    body=binding['mug_body_id'];qa=binding['mug_free_qpos'];hand={m.geom(n).id for n in binding['hand_collision_names']};K=m.actuator_gainprm[:,0]
    np.testing.assert_array_equal(m.jnt_qposadr[m.actuator_trnid[:,0]],np.arange(20));np.testing.assert_allclose(m.actuator_gear[:,0],1.)
    np.testing.assert_allclose(m.actuator_biasprm[:,1],-K);assert m.neq==0 and not np.any(m.body_gravcomp)
    R=Rotation.from_quat(m.body_iquat[body,[1,2,3,0]]).as_matrix()
    wrench=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],m.body_mass[body],R@np.diag(m.body_inertia[body])@R.T,m.opt.gravity)
    offset=z['feedforward_PD_target20'][0]-q[0]
    raw=np.clip(q+offset,*m.actuator_ctrlrange.T);cmd=raw.copy();statuses=[]
    d=mujoco.MjData(m)
    for i in range(100):
        if not wrench['valid_interval'][i]:statuses.append('undefined');continue
        d.qpos[:20]=q[i];d.qpos[qa:qa+7]=np.r_[z['T_world_from_BODY'][i,:3,3],z['quaternion_wxyz'][i]]
        mujoco.mj_forward(m,d)
        u,status=wrench_target(m,d,body,hand,wrench['wrench_world'][i],q[i+1],offset)
        statuses.append(status)
        if u is not None:cmd[i+1]=u
    with np.load(verify_file_pin(trial['arms']['retarget']['trace'])) as f:expected=f['qpos']
    results={}
    for label,controls in [('fixed',raw),('interval_lp',cmd)]:
        d=mujoco.MjData(m);d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=controls[0];mujoco.mj_forward(m,d)
        qs=[];samples=round(.02/m.opt.timestep)
        for k in range(1,101):
            d.ctrl[:]=controls[k]
            for _ in range(samples):
                assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
                mujoco.mj_step(m,d)
            mujoco.mj_kinematics(m,d);qs.append(d.qpos.copy())
        qs=np.array(qs);assert np.isfinite(qs).all()
        if label=='fixed':np.testing.assert_array_equal(qs,expected)
        rots=Rotation.from_quat(qs[:,qa+np.array([4,5,6,3])])
        pos=qs[:,qa:qa+3]+rots.apply(z['COM_BODY_m'])
        err=np.linalg.norm(pos-z['COM_world_m'][1:],axis=1);angle=np.rad2deg((rots*Rotation.from_quat(z['quaternion_wxyz'][1:,[1,2,3,0]]).inv()).magnitude())
        def first(v):
            ids=np.flatnonzero(v);return .02*(int(ids[0])+1) if len(ids) else None
        path=out/(label+'.npz');np.savez_compressed(path,qpos=qs,ctrl=controls,com_error_m=err,angle_deg=angle)
        results[label]=dict(trace=file_pin(path),first_gt_1cm_s=first(err>.01),first_gt_10deg_s=first(angle>10),com_rms_mm=1000*np.sqrt(np.mean(err**2)),angle_rms_deg=np.sqrt(np.mean(angle**2)))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),statuses=statuses,feasible=statuses.count('ok'),results=results,steps=2*100*samples))
    print(statuses.count('ok'),results)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
