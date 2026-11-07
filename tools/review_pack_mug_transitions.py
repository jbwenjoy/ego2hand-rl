"""Read actual raw H20 labels, verify causal roles and pack explicit WM arrays."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.residual_control import residual_ema


def rotation55(x):
    R=np.stack([x[...,43:46],x[...,46:49],np.cross(x[...,43:46],x[...,46:49])],axis=-1)
    if np.max(abs(np.swapaxes(R,-1,-2)@R-np.eye(3)))>1e-8:raise ValueError('Invalid raw state55 rotation columns')
    return R


def main(args):
    import mujoco
    run,out=args.run.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((run/'report.json').read_text());binding=json.loads(verify_file_pin(report['binding']).read_text());reference=json.loads(verify_file_pin(report['reference']).read_text());ref=np.load(verify_file_pin(reference['output']))
    for pin in report['sources']:verify_file_pin(pin)
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(binding['model'])));parts=set(binding['mug_part_geom_ids']);body=binding['mug_body_id'];groups={g:i-1 for i in [1,2,3] for g in [model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']]}
    base=ref['feedforward_PD_target20'][0];negative=np.minimum(.05,base-model.actuator_ctrlrange[:,0]);positive=np.minimum(.05,model.actuator_ctrlrange[:,1]-base)
    packs={role:[] for role in ['TRAIN','CHECK','RESPONSE']};labels=[];episodes=report['episodes']+report['branches'];native_velocity_error=0.;raw_wrench_error=0.;raw_Fn_error=0.;controller_error=0.;total_partial=0
    for episode_id,episode in enumerate(episodes):
        a=np.load(verify_file_pin(episode['transitions']));trace=np.load(verify_file_pin(episode['trace']));role='RESPONSE' if episode['role']=='CHECK_RESPONSE_ONLY_NEVER_FIT' else episode['role'];dt=episode['dt_s'];n=round(.02/dt)
        for key in a.files:
            if a[key].dtype.kind in 'fc' and not np.isfinite(a[key]).all():raise ValueError('Nonfinite saved transition')
        expected=residual_ema(a['action20'],a['EMA_before20'],negative,positive)
        controller_error=max(controller_error,float(abs(expected-a['EMA_after20']).max()),float(abs(base+expected-a['actual_control20']).max()))
        if np.any(abs(a['action20'])>1) or np.any(a['actual_control20']<model.actuator_ctrlrange[:,0]) or np.any(a['actual_control20']>model.actuator_ctrlrange[:,1]):raise ValueError('Illegal native action/control')
        if len(a['source55'])>1 and (not np.array_equal(a['EMA_after20'][:-1],a['EMA_before20'][1:]) or not np.array_equal(a['next55'][:-1],a['source55'][1:])):raise ValueError('Causal controller/state continuity')
        raw=trace['raw_contact_tick_plus23'];tick0=int(a['k'][0])*n;world=np.zeros((len(trace['post_qpos']),6));Fn=np.zeros((len(world),3))
        for row in raw:
            local_tick=int(row[0])-tick0;g1,g2=map(int,row[1:3]);o1,o2=g1 in parts,g2 in parts
            if o1==o2:continue
            sign=1 if o2 else -1;frame=row[9:18].reshape(3,3);F=sign*(frame.T@row[18:21]);T=sign*(frame.T@row[21:24])+np.cross(row[6:9]-trace['pre_object_COM'][local_tick],F)
            world[local_tick]+=np.r_[F,T];other=g1 if o2 else g2
            if other in groups:Fn[local_tick,groups[other]]+=max(row[18],0.)
        for i,(start,end) in enumerate(a['tick_ranges']):
            if end-start!=round(a['actual_duration_s'][i]/dt) or bool(a['complete_H'][i])!=(end-start==n):raise ValueError('Partial duration silently stretched')
            w=world[start:end].mean(0);normal=Fn[start:end].mean(0);R=rotation55(a['source55'][i]);body_wrench=np.r_[R.T@w[:3],R.T@w[3:]]
            raw_wrench_error=max(raw_wrench_error,float(abs(w-a['mean_native_wrench_world_COM'][i]).max()),float(abs(body_wrench-a['mean_native_wrench_source_BODY_COM'][i]).max()));raw_Fn_error=max(raw_Fn_error,float(abs(normal-a['mean_native_group_Fn_N'][i]).max()))
            # Fresh native BODY velocity for both endpoints; no integration/aid.
            for prefix,tick,qkey,vkey in [('source',start,'pre_qpos','pre_qvel'),('next',end-1,'post_qpos','post_qvel')]:
                d=mujoco.MjData(model);d.qpos[:]=trace[qkey][tick];d.qvel[:]=trace[vkey][tick];mujoco.mj_forward(model,d);v=np.empty(6);mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
                state=a[prefix+'55'][i];native_velocity_error=max(native_velocity_error,float(abs(np.r_[d.xipos[body],v[3:],v[:3]]-np.r_[state[40:43],state[49:55]]).max()))
            if not a['complete_H'][i]:total_partial+=1;continue
            source,next_state=a['source55'][i],a['next55'][i];R0,R1=rotation55(source),rotation55(next_state)
            delta=np.r_[next_state[:43]-source[:43],Rotation.from_matrix(R1@R0.T).as_rotvec(),next_state[49:]-source[49:]]
            x=np.r_[source,a['warmstart26_before'][i],a['actual_control20'][i],dt]
            y=np.r_[delta,a['warmstart26_after'][i],a['mean_native_wrench_source_BODY_COM'][i],a['mean_native_group_Fn_N'][i]]
            if x.shape!=(102,) or y.shape!=(87,):raise ValueError('Declared new WM schema')
            k=int(a['k'][i]);goal=np.r_[ref['COM_world_m'][k],ref['T_world_from_BODY'][k,:3,0],ref['T_world_from_BODY'][k,:3,1],ref['COM_velocity_world_m_s'][k],ref['angular_velocity_world_rad_s'][k]]
            tracking_COM=float(np.linalg.norm(next_state[40:43]-ref['COM_world_m'][k+1]));tracking_angle=float(Rotation.from_matrix(R1@ref['T_world_from_BODY'][k+1,:3,:3].T).magnitude())
            packs[role].append(dict(x=x,y=y,source55=source,next55=next_state,EMA_before20=a['EMA_before20'][i],EMA_after20=a['EMA_after20'][i],action20=a['action20'][i],goal15=goal,episode=episode_id,k=k,dt=dt,tracking_COM=tracking_COM,tracking_angle=tracking_angle,joint=episode.get('joint',-1),sign=episode.get('sign',0)))
        labels.append(dict(role=role,seed=episode['seed'],dt_s=dt,rows=int(a['complete_H'].sum()),episode=episode_id))
    if (len(packs['TRAIN']),len(packs['CHECK']),len(packs['RESPONSE']))!=(378,193,48):raise ValueError('Unexpected fitting/check roles')
    train_keys={row['x'].tobytes() for row in packs['TRAIN']};check_keys={row['x'].tobytes() for role in ['CHECK','RESPONSE'] for row in packs[role]}
    overlap=len(train_keys&check_keys)
    checks=dict(physical_baseline_barriers=len(report['barriers'])==2 and all(b['passed'] for b in report['barriers']),full_restore_barriers=len(report['resume_checks'])==6 and all(b['passed'] for b in report['resume_checks']),
        causal_controller_and_raw_control=controller_error<1e-14,raw_simulated_contact_labels=raw_wrench_error<1e-8 and raw_Fn_error<1e-10,
        actual_COM_observation=native_velocity_error<1e-10,role_disjoint_inputs=overlap==0,explicit_role_counts=True)
    checks={k:bool(v) for k,v in checks.items()};out.mkdir(parents=True);files={}
    for role,rows in packs.items():
        np.savez_compressed(out/(role+'.npz'),**{k:np.array([r[k] for r in rows]) for k in rows[0]});files[role]=file_pin(out/(role+'.npz'))
    result=dict(schema='mug_H20_WM_packed_review:1',source=file_pin(__file__),collection=file_pin(run/'report.json'),outputs=files,checks=checks,passed=sum(checks.values()),total=len(checks),
        controller_max_error=controller_error,raw_moving_COM_wrench_error=raw_wrench_error,raw_group_Fn_error=raw_Fn_error,native_COM_velocity_error=native_velocity_error,excluded_partial_intervals=total_partial,
        roles=labels,exact_input_overlap=overlap,input_schema='compound_state55_warm26_actualU20_nativeDt1:1',target_schema='world_left_delta52_nextwarm26_meanwrench_source_BODY6_Fn3:1',
        source_human_contact_truth_known=False,neural_updates=0,integration_steps=0,
        scope='Actual raw simulation/control/time/endpoint readback and explicit new fitting roles. Same reset/task; independent action seeds and numerical dt checks are not heldout trajectories, independent RGB or generalization. Source object reference errors evaluated at full-H endpoints; partials never relabeled at future H.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(dict(report=file_pin(out/'report.json'),checks=checks,rows={k:len(v) for k,v in packs.items()},partial=total_partial)))
    if not all(checks.values()):raise ValueError('Retain failed transition review; do not train')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
