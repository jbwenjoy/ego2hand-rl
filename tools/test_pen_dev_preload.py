"""Static gravity-support preload via equal/opposite contact wrenches on hand bodies."""
import json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from screen_pen_reference_catalog import ROOT,contact_rays
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(proj,out):
    out.mkdir(exist_ok=False)
    source=json.loads((proj/'report.json').read_bytes());intent=json.loads(verify_file_pin(source['intent']).read_bytes())
    sr=json.loads(verify_file_pin(intent['screen']).read_bytes());sp=json.loads(verify_file_pin(sr['intent']).read_bytes())
    c=json.loads(verify_file_pin(sp['case']).read_bytes());m=mujoco.MjModel.from_binary_path(str(verify_file_pin(sp.get('cpu_model',c['model']))))
    with np.load(verify_file_pin(c['trace'])) as f:mount=(f['initial_mocap_pos'][0],f['initial_mocap_quat'][0])
    qids=np.array(c['qids']);dofs=np.array(c['dofs']);aids=np.array(c['aids']);body=c['model_layout']['object_body'];qa=c['model_layout']['object_qposadr']
    np.testing.assert_allclose(m.actuator_gear[aids,0],1.)
    np.testing.assert_allclose(m.actuator_biasprm[aids,1],-m.actuator_gainprm[aids,0])
    assert not m.neq and not np.any(m.body_gravcomp)
    write_json(out/'intent.json',dict(source=file_pin(proj/'report.json'),scope='Static gravity wrench at projected development starts only. Solve minimum total normal contact load, then Jacobian-transpose inverse statics at zero velocity. Apply joint servos only with a 0.10 rad offset cap. Fixed-target 200ms CPU comparison.'))
    rows=[]
    for r in source['rows']:
        if not r.get('variants',{}).get('projection',{}).get('geometry_pass'):continue
        with np.load(verify_file_pin(r['variants']['projection']['trace'])) as z:
            q=z['initial_hand_q'];pos=z['object_pos'];quat=z['object_quat'];control_trace=z['qpos']
        def initial():
            d=mujoco.MjData(m);d.mocap_pos[:],d.mocap_quat[:]=mount;d.qpos[qa:qa+7]=np.r_[pos,quat];d.qpos[qids]=q;d.ctrl[aids]=q;mujoco.mj_forward(m,d);return d
        d=initial();columns=[];jforces=[]
        for con in d.contact:
            geoms=np.asarray(con.geom);on=m.geom_bodyid[geoms]==body
            if on.sum()!=1 or con.dist>0:continue
            hand=int(geoms[1 if on[0] else 0]);handbody=int(m.geom_bodyid[hand])
            if not (m.geom(hand).name or '').startswith('robot/'):continue
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,con.pos,handbody)
            for w in contact_rays(con,1 if on[1] else -1,pos):
                direct=w[3:]-np.cross(con.pos-pos,w[:3])
                columns.append(w);jforces.append((-jp.T@w[:3]-jr.T@direct)[dofs])
        A=np.array(columns).T;target=np.r_[-m.body_mass[body]*m.opt.gravity,np.zeros(3)]
        lp=linprog(np.ones(len(columns)),A_eq=A*np.array([1,1,1,100,100,100])[:,None],b_eq=target,bounds=(0,None),method='highs') if columns else None
        if lp is None or not lp.success:
            rows.append(dict(clip=r['clip'],static_feasible=False,lp_status=lp.message if lp is not None else 'empty'));continue
        np.testing.assert_allclose(A@lp.x,target,atol=1e-7)
        tau=(d.qfrc_bias-d.qfrc_passive)[dofs]-np.asarray(jforces).T@lp.x
        requested=tau/m.actuator_gainprm[aids,0]
        ctrl=np.clip(q+np.clip(requested,-.10,.10),*m.actuator_ctrlrange[aids].T)
        res={}
        for label,cmd in [('baseline',q),('preloaded',ctrl)]:
            d=initial();d.ctrl[aids]=cmd;traj=[]
            for _ in range(80):
                assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
                mujoco.mj_step(m,d);traj.append(d.qpos.copy())
            traj=np.array(traj)
            if label=='baseline':np.testing.assert_allclose(traj,control_trace,atol=1e-10,rtol=0)
            rot=Rotation.from_quat(traj[:,qa+np.array([4,5,6,3])])*Rotation.from_quat(quat[[1,2,3,0]]).inv()
            res[label]=dict(max_mm=1000*np.linalg.norm(traj[:,qa:qa+3]-pos,axis=1).max(),max_deg=np.rad2deg(rot.magnitude()).max())
            path=out/f"{r['clip']}_{label}.npz";np.savez_compressed(path,qpos=traj,ctrl=cmd);res[label]['trace']=file_pin(path)
        rows.append(dict(clip=r['clip'],static_feasible=True,normal_N=lp.fun,max_requested_rad=np.abs(requested).max(),max_applied_rad=np.abs(ctrl-q).max(),holds=res))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))
    print([(r['clip'],r['static_feasible'],[(k,round(v['max_mm'],2),round(v['max_deg'],2)) for k,v in r.get('holds',{}).items()]) for r in rows])
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
