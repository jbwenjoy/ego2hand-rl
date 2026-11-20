"""Offline geometry and nominal interval-wrench screen for the local pen clip catalog."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dex_supermimic.tasks.pen_spin.config.wuji_hand2.hand2_constants import RIGHT_FIXED_WRIST_POSE_W,RIGHT_FINGER_JOINT_NAMES,RIGHT_HAND_LINK_NAMES
from supermimic_research.mechanics.multiclip_wrench_screen import contact_rays,fixed_reference,min_normal
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(out,explicit_model=None):
    out.mkdir(exist_ok=False)
    config=ROOT/'configs/controlled_evaluation_v4.toml'
    cfg=tomllib.loads(config.read_text())
    group=config.parent/cfg['clips_root']/'01_single_axis'
    index=json.loads((group/'index.json').read_bytes());dt=index['frame_dt']
    assert dt==.02 and len(index['clips'])==10
    casepath=ROOT/'runs/pen_scaled_train_v1/eval_02781/repeat_0/report.json'
    case=json.loads(casepath.read_bytes())
    model_pin=case['model'] if explicit_model is None else file_pin(explicit_model)
    write_json(out/'intent.json',dict(cpu_model=model_pin,index=file_pin(group/'index.json'),case=file_pin(casepath),config=file_pin(config),
        split='001 existing controlled training clip. For new development 002-006; validation 007-008; reserve 009-010 for subsequent policy evaluation. Geometry screened on all, no policy score used for assignment.',
        screen='At reference frame i use hand-only nonpositive-gap contact cones to support the nominal interval i wrench about moving COM. No actuator load cap. Geometry and instantaneous cone feasibility only.'))
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(model_pin)))
    assert int(model.opt.cone)==int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    d=mujoco.MjData(model);body=case['model_layout']['object_body'];qa=case['model_layout']['object_qposadr']
    qids=np.array(case['qids']);joints=np.array([model.joint('robot/'+n).id for n in RIGHT_FINGER_JOINT_NAMES]);np.testing.assert_array_equal(model.jnt_qposadr[joints],qids)
    links=[model.body('robot/'+n).id for n in RIGHT_HAND_LINK_NAMES];bounds=model.jnt_range[joints]
    with np.load(verify_file_pin(case['trace'])) as f:
        d.mocap_pos[:]=f['initial_mocap_pos'][0];d.mocap_quat[:]=f['initial_mocap_quat'][0];d.qpos[:]=f['initial_qpos'];known_p=f['COM_world_m'];known_q=f['quaternion_wxyz']
    fixed=np.asarray(RIGHT_FIXED_WRIST_POSE_W)
    mujoco.mj_forward(model,d)
    np.testing.assert_allclose(d.xpos[links[0]],fixed[:3],atol=1e-6)
    assert abs(d.xquat[links[0]]@fixed[3:])>1-1e-6
    physics=case['physics'];result={}
    for entry in index['clips']:
        name=entry['clip_id'];path=group/entry['file'];pin=file_pin(path)
        if pin['sha256']!=entry['sha256']:raise ValueError(name)
        with np.load(path) as f:
            wrist=f['wrist_pose_w'].astype(float);obj=f['object_pose_w'].astype(float);q=f['reference_qpos'];link=f['link_pos_w'].astype(float)
        n=len(q)
        if wrist.shape!=(n,7) or obj.shape!=(n,7) or link.shape!=(n,21,3) or q.shape!=(n,20) or not all(np.isfinite(v).all() for v in (wrist,obj,q,link)):raise ValueError('Bad source frames')
        pos,quat,transform=fixed_reference(wrist,obj,fixed)
        if name=='clip_001':
            np.testing.assert_allclose(pos,known_p,atol=1e-6)
            assert np.max((Rotation.from_quat(quat[:,[1,2,3,0]])*Rotation.from_quat(known_q[:,[1,2,3,0]]).inv()).magnitude())<1e-5
        nominal=mean_reference_wrench(pos,quat,physics['mass'],np.array(physics['inertia_BODY']),np.array(physics['gravity']),dt)
        metrics=[];normal=[];unresolved=[]
        for i in range(n):
            d.qpos[qids]=q[i];d.qpos[qa:qa+7]=np.r_[pos[i],quat[i]]
            mujoco.mj_forward(model,d)
            expected=fixed[:3]+transform[i].apply(link[i]-wrist[i,:3])
            fk=float(np.max(np.linalg.norm(d.xpos[links]-expected,axis=1)))
            pen=0.;self_pen=0.;other_pen=0.;rays=[];ncontact=0
            for c in d.contact:
                g,h=map(int,c.geom);og,oh=model.geom_bodyid[[g,h]]==body
                gn=model.geom(g).name or '';hn=model.geom(h).name or ''
                if og!=oh:
                    other=hn if og else gn
                    pen=max(pen,-c.dist)
                    if other.startswith('robot/') and c.dist<=0:
                        ncontact+=1;rays.extend(contact_rays(c,1 if oh else -1,d.xipos[body]))
                    elif not other.startswith('robot/'):other_pen=max(other_pen,-c.dist)
                elif gn.startswith('robot/') and hn.startswith('robot/'):self_pen=max(self_pen,-c.dist)
            excursion=max(0.,float(np.max(bounds[:,0]-q[i])),float(np.max(q[i]-bounds[:,1])))
            legal=pen<=cfg['max_penetration_m'] and self_pen<=cfg['max_self_penetration_m'] and excursion<=cfg['max_joint_limit_excursion_rad'] and other_pen==0
            valid=i<n-1 and nominal['valid_interval'][i]
            try:
                load=min_normal(rays,nominal['wrench_world'][i]) if valid else None
            except (RuntimeError,ValueError) as exc:
                unresolved.append(dict(frame=i,error=str(exc)));load=None
            metrics.append([pen,self_pen,excursion,ncontact,fk,legal,valid,load is not None])
            normal.append(load if load is not None else np.nan)
        v=np.array(metrics);eligible=v[:,6].astype(bool);both=eligible&(v[:,5]>0)&(v[:,7]>0)
        dst=out/(name+'.npz');np.savez_compressed(dst,metrics=v,normal_N=normal,COM=pos,quat_wxyz=quat,nominal_wrench=nominal['wrench_world'])
        result[name]=dict(source=pin,series=file_pin(dst),frames=n,duration_s=(n-1)*dt,raw_wrist_max_translation_m=float(np.linalg.norm(wrist[:,:3]-wrist[0,:3],axis=1).max()),
            max_FK_disagreement_m=v[:,4].max(),first_penetration_mm=1000*v[0,0],first_self_pen_mm=1000*v[0,1],initial_geometry_pass=bool(v[0,5]),
            geometry_pass_frames=int(v[:,5].sum()),nominal_intervals=int(eligible.sum()),cone_feasible=int((eligible&(v[:,7]>0)).sum()),geometry_and_cone=int(both.sum()),
            unresolved=unresolved,first_contact_count=int(v[0,3]),first_min_normal_N=normal[0] if np.isfinite(normal[0]) else None)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),columns=['object_pen_m','self_pen_m','joint_excursion_rad','object_hand_contacts','max_link_FK_error_m','geometry_pass','nominal_valid','wrench_feasible'],results=result))
    for k,v in result.items():print(k,v['frames'],round(v['first_penetration_mm'],3),v['initial_geometry_pass'],v['geometry_and_cone'],v['nominal_intervals'],round(v['max_FK_disagreement_m']*1000,3))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--cpu-model',type=Path);args=parser.parse_args();main(args.out,args.cpu_model)

