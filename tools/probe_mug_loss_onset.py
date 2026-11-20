"""Substep chronology and same-state unilateral contact-normal action probes."""
from pathlib import Path
import sys,json,re
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_mug_live_wrench import ROOT,verify_file_pin,file_pin,write_json


def read(pin):return json.loads(verify_file_pin(pin).read_bytes())


def contact_rows(m,d,velocity,body,table):
    loaded=np.zeros(5);slip=np.zeros(5);table_load=0.;directions=np.zeros((5,20))
    for ic,c in enumerate(d.contact):
        g,h=map(int,c.geom);on=m.geom_bodyid[[g,h]]==body
        if on.sum()!=1:continue
        other=h if on[0] else g
        f=np.zeros(6);mujoco.mj_contactForce(m,d,ic,f)
        if other==table:table_load+=max(0,f[0]);continue
        match=re.match(r'right_finger([1-5])_',m.geom(other).name or '')
        if not match or f[0]<=0:continue
        ix=int(match[1])-1;normal=c.frame[:3]*(1 if on[1] else -1)
        # Positive direction is hand toward mug, evaluated at the point on the hand.
        jh=np.zeros((3,m.nv));jo=jh.copy()
        mujoco.mj_jac(m,d,jh,None,c.pos,int(m.geom_bodyid[other]))
        mujoco.mj_jac(m,d,jo,None,c.pos,body)
        rel=(jh-jo)@velocity;tangent=rel-normal*(normal@rel)
        loaded[ix]+=f[0];slip[ix]+=f[0]*np.linalg.norm(tangent)
        directions[ix]+=f[0]*(normal@jh[:,:20])
    return loaded,slip,table_load,directions


def main(out):
    out.mkdir(exist_ok=False);p=ROOT/'runs';src=file_pin(p/'mug_live_wrench_v1/report.json');data=read(src)
    ff=read(read(data['intent'])['parent']);tr=read(read(ff['intent'])['parent']);orig=read(read(tr['dynamic'])['intent'])
    ref=read(orig['reference']);binding=read(orig['binding'])
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(ref['model'])))
    with np.load(verify_file_pin(ref['output'])) as f:z=dict(f)
    refR=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    body=binding['mug_body_id'];qa=binding['mug_free_qpos'];table=m.geom('mug_task_table').id
    n=round(.02/m.opt.timestep)
    assert n==64 and m.nu==20
    write_json(out/'intent.json',dict(source=src,arms=['replay','live_pose_PD'],probes_at_controls=[10,13,15],probe_length_controls=4,
        probe='first 20ms only: change ONE of D1-D3 target vectors by signed 0.025 rad L2 along current normal-load-weighted hand point Jacobian; remaining 60ms archived commands. Zero loaded direction gives no perturbation. Future controls frozen for diagnostic, not a realizable feedback evaluation.'))
    results={}
    def errors(d,k):
        R=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])]);c=d.qpos[qa:qa+3]+R.apply(z['COM_BODY_m'])
        return np.array([1000*np.linalg.norm(c-z['COM_world_m'][k]),np.rad2deg((R*refR[k].inv()).magnitude())])
    for label in ('replay','live_pose_PD'):
        with np.load(verify_file_pin(data['arms'][label]['trace'])) as f:cmd=np.vstack([z['feedforward_PD_target20'][0],f['cmd']]);gold=f['qpos'];assert len(cmd)==101
        d=mujoco.MjData(m);d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=cmd[0];mujoco.mj_forward(m,d)
        rows=[];branches=[]
        for i in range(24):
            if i in (10,13,15):
                obs=mujoco.MjData(m);mujoco.mj_copyData(obs,m,d);mujoco.mj_forward(m,obs)
                forces,_,_,J=contact_rows(m,obs,obs.qvel,body,table)
                for digit,sign in [(-1,0)]+[(j,s) for j in range(3) for s in (-1,1)]:
                    branch=mujoco.MjData(m);mujoco.mj_copyData(branch,m,d);delta=np.zeros(20)
                    if digit>=0 and forces[digit]>.1 and np.linalg.norm(J[digit])>1e-9:
                        delta=sign*.025*J[digit]/np.linalg.norm(J[digit])
                    hist=[]
                    for k in range(i+1,i+5):
                        branch.ctrl[:]=np.clip(cmd[k]+(delta if k==i+1 else 0),*m.actuator_ctrlrange.T)
                        if k==i+1:applied=float(np.linalg.norm(branch.ctrl-cmd[k]))
                        mujoco.mj_step(m,branch,nstep=n);mujoco.mj_kinematics(m,branch)
                        if sign==0:np.testing.assert_array_equal(branch.qpos,gold[k-1])
                        hist.append(errors(branch,k).tolist())
                    branches.append(dict(start=i*.02,digit=digit+1,sign=sign,delta_norm=applied,error_mm_deg=hist))
            totals=[];vels=[];tables=[]
            d.ctrl[:]=cmd[i+1]
            for _ in range(n):
                v=d.qvel.copy()
                assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                mujoco.mj_step(m,d)
                load,weighted,t,_=contact_rows(m,d,v,body,table)
                totals.append(load);vels.append(weighted);tables.append(t)
            mujoco.mj_kinematics(m,d);np.testing.assert_array_equal(d.qpos,gold[i])
            totals=np.array(totals);vels=np.array(vels)
            rows.append(dict(end_s=.02*(i+1),error_mm_deg=errors(d,i+1).tolist(),mean_normal_N=totals.mean(0).tolist(),
                active_substeps=(totals>.1).sum(0).tolist(),weighted_slip_mm_s=[float(1000*vels[:,k].sum()/totals[:,k].sum()) if totals[:,k].sum()>1e-8 else None for k in range(5)],
                table_substeps=int(np.count_nonzero(np.asarray(tables)>.1))))
        results[label]=dict(rows=rows,probes=branches)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),results=results))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())

