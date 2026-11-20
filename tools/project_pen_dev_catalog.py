"""Joint-only proposals on the five declared development clips and 200ms CPU holds."""
from pathlib import Path
import sys,json,tomllib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from screen_pen_reference_catalog import ROOT,contact_rays,min_normal
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.bounded_contact_projection import BoundedContactProjection
from supermimic_research.strict_json import write_json


def main(screen,out):
    out.mkdir(exist_ok=False)
    report=json.loads((screen/'report.json').read_bytes());intent=json.loads(verify_file_pin(report['intent']).read_bytes())
    case=json.loads(verify_file_pin(intent['case']).read_bytes())
    cp=ROOT/'configs/controlled_pen_projection_v1.toml';cfg=tomllib.loads(cp.read_text())
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])))
    with np.load(verify_file_pin(case['trace'])) as z:mount=(z['initial_mocap_pos'][0],z['initial_mocap_quat'][0])
    body=case['model_layout']['object_body'];qa=case['model_layout']['object_qposadr'];qids=np.array(case['qids']);aids=np.array(case['aids']);j=m.actuator_trnid[aids,0]
    np.testing.assert_array_equal(m.jnt_qposadr[j],qids)
    assert m.neq==0 and not np.any(m.body_gravcomp) and np.isclose(m.opt.timestep,.0025)
    parts=[i for i in range(m.ngeom) if m.geom_bodyid[i]==body and (m.geom_contype[i] or m.geom_conaffinity[i])]
    hand=[i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('robot/') and (m.geom_contype[i] or m.geom_conaffinity[i])]
    groups={i:[m.geom(f'robot/right_finger{i}_{suffix}').id for suffix in cfg['contact_suffixes']] for i in cfg['contact_fingers']}
    write_json(out/'intent.json',dict(screen=file_pin(screen/'report.json'),settings=file_pin(cp),scope='Fixed contact fingers 1/2/3 as original projection; clips 002-006 only. Fixed PD targets, zero initial velocity and no applied forces in 200ms CPU hold. Diagnostic displacement from start, no reference advance.'))
    rows=[]
    for k in range(2,7):
        name=f'clip_{k:03}';entry=report['results'][name]
        with np.load(verify_file_pin(entry['source'])) as f:q=f['reference_qpos'][0].astype(float)
        with np.load(verify_file_pin(entry['series'])) as f:pos=f['COM'][0];quat=f['quat_wxyz'][0];target=f['nominal_wrench'][0]
        d=mujoco.MjData(m);d.mocap_pos[:],d.mocap_quat[:]=mount;d.qpos[qa:qa+7]=np.r_[pos,quat];d.qpos[qids]=q;mujoco.mj_forward(m,d);base=d.qpos.copy()
        lo=np.maximum(q-.1,np.maximum(m.jnt_range[j,0],m.actuator_ctrlrange[aids,0])+1e-6)
        hi=np.minimum(q+.1,np.minimum(m.jnt_range[j,1],m.actuator_ctrlrange[aids,1])-1e-6)
        if np.any(q<lo) or np.any(q>hi):
            rows.append(dict(clip=name,reason='initial point outside bounded optimizer domain'));continue
        engine=BoundedContactProjection(m,d,qids,lo,hi,parts,hand,groups,cfg);result=engine.solve()
        variants={}
        for label,v in [('source',q),('projection',result['q'])]:
            s=mujoco.MjData(m);s.mocap_pos[:],s.mocap_quat[:]=mount;s.qpos[:]=base;s.qpos[qids]=v;s.ctrl[aids]=v;mujoco.mj_forward(m,s)
            gaps=np.array([mujoco.mj_geomDistance(m,s,g,h,cfg['distance_limit_m'],np.zeros(6)) for g,h in engine.pairs]);n=len(engine.object_pairs)
            checks=dict(object=bool(np.all(gaps[:n]>=-cfg['qualification_penetration_m'])),self_collision=bool(np.all(gaps[n:]>=-cfg['qualification_self_penetration_m'])),
                groups=all(-cfg['qualification_penetration_m']<=gaps[z].min()<=cfg['qualification_contact_gap_m'] for z in engine.group_indices.values()),bounds=bool(np.all(v>=lo)&np.all(v<=hi)))
            rays=[]
            for con in s.contact:
                g,h=map(int,con.geom)
                if con.dist<=0 and ((g in parts and h in hand) or (h in parts and g in hand)):
                    rays.extend(contact_rays(con,1 if h in parts else -1,pos))
            load=min_normal(rays,target)
            traj=[]
            for _ in range(80):
                if np.any(s.qfrc_applied) or np.any(s.xfrc_applied):raise ValueError('Applied force')
                mujoco.mj_step(m,s);traj.append(s.qpos.copy())
            traj=np.array(traj)
            if not np.isfinite(traj).all():raise ValueError('Nonfinite CPU hold')
            angle=(Rotation.from_quat(traj[:,qa+np.array([4,5,6,3])])*Rotation.from_quat(quat[[1,2,3,0]]).inv()).magnitude()
            dst=out/f'{name}_{label}.npz';np.savez_compressed(dst,qpos=traj,initial_hand_q=v,raw_q=q,object_pos=pos,object_quat=quat)
            variants[label]=dict(trace=file_pin(dst),checks=checks,geometry_pass=all(checks.values()),nominal_wrench_min_normal_N=load,max_pos_drift_mm=1000*np.linalg.norm(traj[:,qa:qa+3]-pos,axis=1).max(),max_angle_drift_deg=np.rad2deg(angle.max()))
        rows.append(dict(clip=name,solver_message=result['message'],max_joint_delta_rad=float(np.max(np.abs(result['delta']))),variants=variants))
        write_json(out/'progress.json',dict(rows=rows))
    write_json(out/'report.json',dict(rows=rows,intent=file_pin(out/'intent.json')))
    print([(r['clip'],r.get('reason') or [(v['geometry_pass'],round(v['max_pos_drift_mm'],2),round(v['max_angle_drift_deg'],1)) for v in r['variants'].values()]) for r in rows])
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
