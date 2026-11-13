"""Independent saved pose/contact/controller review of fixed cube/pen runs."""
from pathlib import Path
import argparse
import json
import sys
import tomllib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    if args.output.exists(): raise ValueError('Fresh output required')
    report=json.loads(args.report.read_text(encoding='utf8'))
    config=tomllib.loads(verify_file_pin(report['config']).read_text(encoding='utf8'))
    for p in report['sources']:verify_file_pin(p)
    args.output.mkdir()
    items=[]; raw_count=0; datasets={}; max_errors=dict(forces=0.,penetration=0.,COM=0.,pre_COM=0.,metrics=0.,actor_action=0.)
    import torch
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    torch.set_num_threads(4)
    for case in report['cases']:
        task,mode=case['task'],case['controller'];z=dict(np.load(verify_file_pin(case['trace'])))
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['host_topology'])))
        physical=dict(np.load(verify_file_pin(case['GPU_physical_context'])))
        dt=case['dt'];N=len(z['post_qpos']);n=round(.02/dt);K=len(z['actions20']);body=case['model_layout']['object_body'];qa=case['model_layout']['object_qposadr'];da=case['model_layout']['object_dofadr'];com=np.array(case['model_layout']['COM_BODY'])
        datasets[task,mode]=z
        checks=dict(integer_clock=N==K*n==case['physics_steps'],one_reset=case['initial_reset_count']==1 and case['resets_after_start']==0,
            pose_carry=np.array_equal(z['pre_qpos'][1:],z['post_qpos'][:-1]),velocity_carry=np.array_equal(z['pre_qvel'][1:],z['post_qvel'][:-1]),
            initial_pose=np.array_equal(z['pre_qpos'][0],z['initial_qpos']),initial_velocity=np.array_equal(z['pre_qvel'][0],z['initial_qvel']),
            phase=np.array_equal(z['pre_action_phase'],np.arange(1,K+1)),
            first_terminal_only=not np.any(z['terminated_truncated'][:-1]) and not np.any(z['termination_flags'][:-1]),
            final_flags=all(bool(z['termination_flags'][-1,i])==case['termination_flags'][name] for i,name in enumerate(case['termination_names'])),
            no_applied_force=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
            no_overflow=not np.any(z['overflow']),finite=all(np.isfinite(v).all() for v in z.values()),
            bounded_actions=bool(np.all(abs(z['actions20'])<=1)),
            legal_control=bool(np.all(z['post_ctrl']>=physical['actuator_ctrlrange'][0,:,0]-1e-7) and np.all(z['post_ctrl']<=physical['actuator_ctrlrange'][0,:,1]+1e-7)),
            no_eq_gravity_aid=m.neq==0 and not np.any(m.body_gravcomp),
            single_finite_reference=(len(z['time_s'])==369 and z['time_s'][-1]==7.36) if task=='hand2-pen-spin' else (len(z['time_s'])==2 and z['time_s'][-1]==2.))
        raw=z['raw_contact_tick_plus23'];raw_count+=len(raw)
        tick=raw[:,0].astype(int);g1,g2=raw[:,1].astype(int),raw[:,2].astype(int)
        checks['raw_indices']=bool(np.all(raw[:,0]==tick) and np.all((tick>=0)&(tick<N)) and np.all((g1>=0)&(g1<m.ngeom)) and np.all((g2>=0)&(g2<m.ngeom)))
        o1,o2=m.geom_bodyid[g1]==body,m.geom_bodyid[g2]==body;boundary=o1^o2
        hand=np.array([(m.geom(g).name or '').startswith('robot/') for g in range(m.ngeom)])
        pen=np.zeros(N);selfpen=np.zeros(N);handN=np.zeros(N);otherN=np.zeros(N);world=np.zeros((N,6))
        np.maximum.at(pen,tick,np.maximum(-raw[:,5],0))
        mask=hand[g1]&hand[g2];np.maximum.at(selfpen,tick[mask],np.maximum(-raw[mask,5],0))
        rows=raw[boundary];ticks=tick[boundary];other=np.where(o2[boundary],g1[boundary],g2[boundary]);normal=np.maximum(rows[:,18],0.)
        for mask,dst in [(hand[other],handN),(~hand[other],otherN)]:np.add.at(dst,ticks[mask],normal[mask])
        frame=rows[:,9:18].reshape(-1,3,3);sign=np.where(o2[boundary],1.,-1.)
        F=sign[:,None]*np.einsum('nji,nj->ni',frame,rows[:,18:21]);T=sign[:,None]*np.einsum('nji,nj->ni',frame,rows[:,21:24])+np.cross(rows[:,6:9]-z['pre_object_COM'][ticks],F)
        np.add.at(world,ticks,np.c_[F,T])
        max_errors['forces']=max(max_errors['forces'],float(abs(handN-z['pre_hand_normal_N']).max()),float(abs(otherN-z['pre_object_nonhand_normal_N']).max()))
        max_errors['penetration']=max(max_errors['penetration'],float(abs(pen-z['pre_penetration_m']).max()),float(abs(selfpen-z['pre_self_penetration_m']).max()))
        q=z['post_qpos'];pre=z['pre_qpos'];v=z['post_qvel'];pv=z['pre_qvel']
        R=Rotation.from_quat(q[:,qa+3:qa+7][:,[1,2,3,0]]).as_matrix();pR=Rotation.from_quat(pre[:,qa+3:qa+7][:,[1,2,3,0]]).as_matrix()
        COM=q[:,qa:qa+3]+R@com;preCOM=pre[:,qa:qa+3]+pR@com
        max_errors['COM']=max(max_errors['COM'],float(abs(COM-z['post_COM']).max()))
        max_errors['pre_COM']=max(max_errors['pre_COM'],float(abs(preCOM-z['pre_object_COM']).max()))
        times=(np.arange(N)+1)*dt;clock=z['time_s'];target=np.stack([np.interp(times,clock,z['COM_world_m'][:,j]) for j in range(3)],axis=1)
        goal=Slerp(clock,Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]]))(np.minimum(times,clock[-1])).as_matrix()
        delta=R@np.swapaxes(goal,-1,-2);angle=np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)*.5,-1,1));distance=np.linalg.norm(COM-target,axis=1)
        joints=np.flatnonzero(m.jnt_type==int(mujoco.mjtJoint.mjJNT_HINGE));qi=m.jnt_qposadr[joints];ranges=m.jnt_range[joints]
        excursion=np.maximum(0.,np.maximum((ranges[:,0]-q[:,qi]).max(axis=1),(q[:,qi]-ranges[:,1]).max(axis=1)))
        metrics=dict(actual_duration_s=N*dt,COM_RMS_m=float(np.sqrt(np.mean(distance**2))),COM_max_m=float(distance.max()),angle_RMS_rad=float(np.sqrt(np.mean(angle**2))),angle_max_rad=float(angle.max()),
            max_penetration_m=float(pen.max()),max_self_penetration_m=float(selfpen.max()),max_joint_limit_excursion_rad=float(excursion.max()),max_object_nonhand_normal_force_N=float(otherN.max()))
        max_errors['metrics']=max(max_errors['metrics'],max(abs(value-case['summary']['metrics'][key]) for key,value in metrics.items()))
        complete=K==case['planned_intervals'] and not bool(z['terminated_truncated'][-1,0])
        gates={key:value<=config[key] for key,value in metrics.items() if key!='actual_duration_s'}
        gates.update(duration_complete=complete,no_applied_force=checks['no_applied_force'],finite_dynamics=checks['finite'],no_contact_constraint_overflow=checks['no_overflow'])
        checks['gates_match']=gates==case['summary']['gates'] and all(gates.values())==case['summary']['qualified']
        if mode=='ZERO':checks['actor_or_zero_output']=not np.any(z['actions20'])
        else:
            saved=torch.load(verify_file_pin(report['checkpoints'][task]),map_location='cpu',weights_only=False)
            actor=Actor(z['pre_action_observations'].shape[1]).eval();actor.load_state_dict(saved['modules']['actor'])
            with torch.no_grad():pred=actor(torch.from_numpy(z['pre_action_observations'])).numpy()
            error=float(abs(pred-z['actions20']).max());max_errors['actor_action']=max(max_errors['actor_action'],error);checks['actor_or_zero_output']=error<2e-6
        wbody=np.c_[np.einsum('nji,nj->ni',pR,world[:,:3]),np.einsum('nji,nj->ni',pR,world[:,3:])]
        mass=float(physical['body_mass'][0,body]);arm=R@com;parm=pR@com;omega=R@v[:,da+3:da+6,None];pomega=pR@pv[:,da+3:da+6,None]
        vel=v[:,da:da+3]+np.cross(omega[:,:,0],arm);pvel=pv[:,da:da+3]+np.cross(pomega[:,:,0],parm)
        closure=mass*(vel-pvel)-dt*(world[:,:3]+mass*m.opt.gravity)
        artifact=args.output/f'{task}_{mode}_wrenches.npz'
        np.savez_compressed(artifact,world_COM_wrench=world,object_BODY_COM_wrench=wbody,
            mean20ms_object_BODY_COM_wrench=wbody.reshape(K,n,6).mean(axis=1),linear_momentum_residual_N_s=closure)
        items.append(dict(task=task,controller=mode,checks={k:bool(v) for k,v in checks.items()},qualified=case['summary']['qualified'],
            wrenches=file_pin(artifact),linear_momentum_residual_RMS_N_s=float(np.sqrt(np.mean(np.sum(closure**2,axis=1)))),
            linear_momentum_residual_max_N_s=float(np.linalg.norm(closure,axis=1).max())))
    pairs=[]
    for task in config['tasks']:
        a,b=datasets[task,'ZERO'],datasets[task,'SAVED_ACTOR']
        equal=all(np.array_equal(a[key],b[key]) for key in ['initial_native_state','initial_qpos','initial_qvel','time_s','COM_world_m','quaternion_wxyz'])
        pairs.append(dict(task=task,exact=equal))
    checks=dict(four_fixed_cases={(c['task'],c['controller']) for c in items}=={(t,c) for t in config['tasks'] for c in config['controllers']} and len(items)==4,
        cases=all(all(c['checks'].values()) for c in items),exact_initial_and_reference_pairs=all(p['exact'] for p in pairs),
        raw_contacts=max(max_errors['forces'],max_errors['penetration'])<1e-9,pose_metric_readback=max(max_errors['COM'],max_errors['metrics'])<1e-6,
        GPU_PRE_COM_float32_tolerance=max_errors['pre_COM']<1e-6)
    result=dict(schema='controlled_saved_review:1',source=file_pin(__file__),run=file_pin(args.report),checks=checks,passed=sum(checks.values()),total=len(checks),max_errors=max_errors,
        cases=items,pairs=pairs,raw_contacts_reviewed=raw_count,neural_updates=0,native_integration_steps=0,
        scope='Independent saved chronology, raw GPU contact/pose/metric/flag, control-bound and CPU saved-Actor readback. Native warning counts unavailable, no fresh convex audit or task success beyond explicit failed qualifications. Wrenches are contact labels, not fitted support targets or WM gradient validation.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
    if not all(checks.values()):raise ValueError('Keep failed independent review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
