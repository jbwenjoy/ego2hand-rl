"""Saved-data control, raw-contact, impulse and first-terminal readback."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_snapshot import native_controlled_snapshot


def main(args):
    if args.output.exists():raise ValueError('Fresh readback output')
    r=json.loads(args.report.read_text());p=json.loads(verify_file_pin(r['projection']).read_text());m,d,source,index=native_controlled_snapshot(p['case'])
    for pin in r['sources']+p['sources']:verify_file_pin(pin)
    projection=np.load(verify_file_pin(p['proposal']));qref=np.load(verify_file_pin(r['clip']))['reference_qpos'];cfg=tomllib.loads(verify_file_pin(json.loads(verify_file_pin(p['source']).read_text())['config']).read_text())
    body=p['case']['model_layout']['object_body'];qa=p['case']['model_layout']['object_qposadr'];da=p['case']['model_layout']['object_dofadr'];mass=m.body_mass[body]
    a=index['actuators'];j=index['joints'];lo=np.maximum(m.jnt_range[j,0],m.actuator_ctrlrange[a,0]);hi=np.minimum(m.jnt_range[j,1],m.actuator_ctrlrange[a,1]);items=[]
    for c in r['cases']:
        z=np.load(verify_file_pin(c['trace']));dt=c['dt'];N=len(z['post_qpos']);n=round(.02/dt);K=len(z['pre_action_phase']);projected=c['name']=='PROJECTED_CONSTANT_RESIDUAL';delta=projection['delta20'] if projected else np.zeros(20)
        checks=dict(initial_state=bool(np.array_equal(z['initial_qpos'],projection['qpos'] if projected else source['initial_qpos']) and np.array_equal(z['initial_qvel'],source['initial_qvel'])),
            pose_carry=np.array_equal(z['pre_qpos'][1:],z['post_qpos'][:-1]),velocity_carry=np.array_equal(z['pre_qvel'][1:],z['post_qvel'][:-1]),
            initial_first_tick=np.array_equal(z['pre_qpos'][0],z['initial_qpos']) and np.array_equal(z['pre_qvel'][0],z['initial_qvel']),
            source_clock=np.array_equal(z['pre_action_phase'],np.arange(1,K+1)) and N==K*n,
            native_clock=np.max(abs(z['post_native_time']-(np.arange(N)+1)*dt))<1e-12,
            raw_action=np.array_equal(z['raw_action20'],2*delta),no_force_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']))
        memory=delta.copy();targets=[];ems=[]
        for phase in z['pre_action_phase']:
            before=memory.copy();memory=.5*np.clip(.5*(2*delta),-.6,.6)+.5*memory
            targets.append(np.clip(qref[int(phase)]+memory,lo,hi));ems.append(np.r_[before,memory])
        checks['EMA_targets']=np.array_equal(z['actual_targets20'],np.array(targets)) and np.array_equal(z['EMA_before_after40'],np.array(ems)) and np.array_equal(z['post_ctrl'][:,a],np.repeat(targets,n,axis=0))
        raw=z['raw_contact_tick_plus23'];ticks=raw[:,0].astype(int);g1,g2=raw[:,1].astype(int),raw[:,2].astype(int);o1,o2=m.geom_bodyid[g1]==body,m.geom_bodyid[g2]==body;mask=o1^o2
        rows=raw[mask];ti=ticks[mask];other=np.where(o2[mask],g1[mask],g2[mask]);hand=np.array([(m.geom(g).name or '').startswith('robot/') for g in range(m.ngeom)])
        handN=np.zeros(N);otherN=np.zeros(N);penetration=np.zeros(N);selfpen=np.zeros(N)
        for select,dst in [(hand[other],handN),(~hand[other],otherN)]:np.add.at(dst,ti[select],np.maximum(rows[select,18],0))
        np.maximum.at(penetration,ticks,np.maximum(-raw[:,5],0));select=hand[g1]&hand[g2];np.maximum.at(selfpen,ticks[select],np.maximum(-raw[select,5],0))
        sign=np.where(o2[mask],1.,-1.);frame=rows[:,9:18].reshape(-1,3,3);F=sign[:,None]*np.einsum('nji,nj->ni',frame,rows[:,18:21]);T=sign[:,None]*np.einsum('nji,nj->ni',frame,rows[:,21:24])+np.cross(rows[:,6:9]-z['pre_object_COM'][ti],F)
        world=np.zeros((N,6));np.add.at(world,ti,np.c_[F,T]);force_error=float(abs(world-z['pre_world_COM_wrench']).max());force_error=max(force_error,float(abs(handN-z['pre_hand_normal_N']).max()),float(abs(otherN-z['pre_object_nonhand_normal_N']).max()))
        impulse=mass*(z['post_qvel'][:,da:da+3]-z['pre_qvel'][:,da:da+3])-dt*(world[:,:3]+mass*m.opt.gravity)
        generalized=float(abs(world[:,:3]-z['post_qfrc_constraint'][:,da:da+3]).max())
        checks['raw_contact_reconstruction']=force_error<1e-10 and np.array_equal(penetration,z['pre_penetration_m']) and np.array_equal(selfpen,z['pre_self_penetration_m'])
        checks['force_and_impulse']=generalized<1e-9 and float(abs(impulse).max())<1e-10
        times=(np.arange(N)+1)*dt;clock=z['time_s'];target=np.stack([np.interp(times,clock,z['COM_world_m'][:,i]) for i in range(3)],axis=1);R=Rotation.from_quat(z['post_qpos'][:,qa+3:qa+7][:,[1,2,3,0]]).as_matrix();goal=Slerp(clock,Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]]))(np.minimum(times,clock[-1])).as_matrix()
        dist=np.linalg.norm(z['post_qpos'][:,qa:qa+3]-target,axis=1);angle=Rotation.from_matrix(R@np.swapaxes(goal,-1,-2)).magnitude();metrics=c['summary']['metrics']
        error=max(abs(np.sqrt(np.mean(dist**2))-metrics['COM_RMS_m']),abs(dist.max()-metrics['COM_max_m']),abs(np.sqrt(np.mean(angle**2))-metrics['angle_RMS_rad']),abs(angle.max()-metrics['angle_max_rad']))
        checks['pose_metrics']=error<1e-10
        endpoints=np.arange(n-1,N,n);shaft=np.arccos(np.clip(np.sum(R[endpoints,:,0]*goal[endpoints,:,0],axis=1),-1,1));stop=(dist[endpoints]>.05)|(shaft>=np.pi/3)
        checks['first_terminal']=not np.any(stop[:-1]) and bool(stop[-1]) and c['termination']=='POSITION_ERROR_OVER_50MM'
        gates=c['summary']['gates'];expected={key:metrics[key]<=cfg[key] for key in ['COM_RMS_m','COM_max_m','angle_RMS_rad','angle_max_rad','max_penetration_m','max_self_penetration_m','max_joint_limit_excursion_rad','max_object_nonhand_normal_force_N']}
        expected.update(duration_complete=K==368 and not stop[-1],no_applied_force=checks['no_force_aid'],finite_dynamics=all(np.isfinite(z[k]).all() for k in z.files),no_contact_constraint_overflow=not np.any(z['overflow']))
        checks['gates']=expected==gates and bool(all(expected.values()))==c['summary']['qualified']
        items.append(dict(name=c['name'],dt=dt,checks={k:bool(v) for k,v in checks.items()},force_error_N_or_Nm=force_error,generalized_force_error_N=generalized,max_impulse_residual_N_s=float(abs(impulse).max()),pose_metric_error=error,qualified=c['summary']['qualified']))
    checks=dict(four_cases=len(items)==4,all_saved_checks=all(all(c['checks'].values()) for c in items))
    result=dict(schema='projected_pen_native_saved_review:1',source=file_pin(__file__),run=file_pin(args.report),checks=checks,cases=items,passed=sum(sum(c['checks'].values()) for c in items),total=sum(len(c['checks']) for c in items),neural_updates=0,native_integration_steps=0)
    args.output.mkdir();(args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
    if not all(checks.values()):raise ValueError('Retain failed native readback')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
