"""Shared saved-native readback, derived from the frozen H3 reviewer.

Cohort and horizon are explicit. No controller execution or model fitting.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation,Slerp
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.residual_control import residual_ema
from supermimic_research.runtime.native_evaluation import GATES


def equal_bytes(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def review_saved_cases(run,out,*,root,expected_cases,horizon,historical_zero_manifest=None):
    ROOT=Path(root).resolve();run,out=Path(run).resolve(),Path(out).resolve()
    expected_cases=set(expected_cases)
    if horizon<1 or not expected_cases:raise ValueError("Explicit cohort/horizon required")
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified review')
    report=json.loads((run/'report.json').read_text());binding=json.loads(verify_file_pin(report['inputs']['binding']).read_text());ref_report=json.loads(verify_file_pin(report['inputs']['reference']).read_text());ref=np.load(verify_file_pin(ref_report['output']))
    for pin in report['sources']:verify_file_pin(pin)
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(binding['model'])));parts=np.array(binding['mug_part_geom_ids']);body=binding['mug_body_id'];table=model.geom('mug_task_table').id;base=ref['feedforward_PD_target20'][0]
    neg=np.minimum(.05,base-model.actuator_ctrlrange[:,0]);pos=np.minimum(.05,model.actuator_ctrlrange[:,1]-base);groups=np.full(model.ngeom,-1,dtype=int)
    for i in [1,2,3]:
        for suffix in ['link3_col','link4_col','tip_sensor_col']:groups[model.geom(f'right_finger{i}_{suffix}').id]=i-1
    old=json.loads(Path(historical_zero_manifest).read_text()) if historical_zero_manifest is not None else {};baselines={b['dt_s']:b['episode']['trace'] for b in old.get('barriers',[])};maxima=dict(control=0.,normal_forces=0.,penetration=0.,wrench_mean=0.,COM_observer=0.,COM_velocity=0.,metrics=0.);items=[];raw_count=0;data=mujoco.MjData(model)
    for case in report['cases']:
        a=np.load(verify_file_pin(case['execution']['transitions']));z=np.load(verify_file_pin(case['execution']['trace']));raw=z['raw_contact_tick_plus23'];raw_count+=len(raw);N=len(z['post_qpos']);dt=case['dt'];n=round(.02/dt)
        expected=base+residual_ema(a['action20'],a['EMA_before20'],neg,pos);maxima['control']=max(maxima['control'],float(abs(expected-a['actual_control20']).max()))
        legal=bool(np.all(abs(a['action20'])<=1) and np.all(expected>=model.actuator_ctrlrange[:,0]) and np.all(expected<=model.actuator_ctrlrange[:,1]))
        no_aid=True
        for state in np.concatenate([a['source_native_state'],a['next_native_state']]):
            mujoco.mj_resetData(model,data);mujoco.mj_setState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
            no_aid=no_aid and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
        tick=raw[:,0].astype(int);g1,g2=raw[:,1].astype(int),raw[:,2].astype(int);o1,o2=np.isin(g1,parts),np.isin(g2,parts);boundary=o1^o2;world=np.zeros((N,6));Fn=np.zeros((N,3));other_force=np.zeros(N);table_force=np.zeros(N);penetration=np.zeros(N);self_pen=np.zeros(N)
        np.maximum.at(penetration,tick,np.maximum(-raw[:,5],0));self_mask=(~o1)&(~o2)&(g1!=table)&(g2!=table);np.maximum.at(self_pen,tick[self_mask],np.maximum(-raw[self_mask,5],0))
        rows=raw[boundary];ticks=tick[boundary];object_second=o2[boundary];other=np.where(object_second,g1[boundary],g2[boundary]);sgn=np.where(object_second,1.,-1.);frame=rows[:,9:18].reshape(-1,3,3)
        F=sgn[:,None]*np.einsum('nji,nj->ni',frame,rows[:,18:21]);T=sgn[:,None]*np.einsum('nji,nj->ni',frame,rows[:,21:24])+np.cross(rows[:,6:9]-z['pre_object_COM'][ticks],F);np.add.at(world,ticks,np.c_[F,T])
        normal=np.maximum(rows[:,18],0);group=groups[other]
        for i in range(3):mask=group==i;np.add.at(Fn[:,i],ticks[mask],normal[mask])
        mask=other==table;np.add.at(table_force,ticks[mask],normal[mask]);mask=(group<0)&(other!=table);np.add.at(other_force,ticks[mask],normal[mask])
        maxima['normal_forces']=max(maxima['normal_forces'],float(abs(Fn-z['pre_group_normal_N']).max()),float(abs(other_force-z['pre_other_normal_N']).max()),float(abs(table_force-z['pre_table_normal_N']).max()));maxima['penetration']=max(maxima['penetration'],float(abs(penetration-z['pre_penetration_m']).max()),float(abs(self_pen-z['pre_self_penetration_m']).max()))
        for i,(ta,tb) in enumerate(z['tick_ranges']):
            R=np.stack([a['source55'][i,43:46],a['source55'][i,46:49],np.cross(a['source55'][i,43:46],a['source55'][i,46:49])],axis=-1);w=world[ta:tb].mean(0);b=np.r_[R.T@w[:3],R.T@w[3:]];maxima['wrench_mean']=max(maxima['wrench_mean'],float(abs(b-a['mean_native_wrench_source_BODY_COM'][i]).max()))
        q,v=z['post_qpos'],z['post_qvel'];R=Rotation.from_quat(np.c_[q[:,24:27],q[:,23:24]]).as_matrix();arm=np.einsum('nij,j->ni',R,model.body_ipos[body]);COM=q[:,20:23]+arm;omega=np.einsum('nij,nj->ni',R,v[:,23:26]);velocity=v[:,20:23]+np.cross(omega,arm)
        maxima['COM_observer']=max(maxima['COM_observer'],float(abs(COM-z['post_COM']).max()));maxima['COM_velocity']=max(maxima['COM_velocity'],float(abs(velocity-z['post_COM_velocity']).max()),float(abs(omega-z['post_omega_world']).max()))
        times=(np.arange(N)+1)*dt;clock=np.arange(101)*.02
        if case['task']=='HOLD':target=np.tile(ref['COM_world_m'][0],(N,1));targetR=np.tile(ref['T_world_from_BODY'][0,:3,:3],(N,1,1));targetv=np.zeros((N,3));targetw=targetv
        else:
            target=np.array([np.interp(times,clock,ref['COM_world_m'][:,i]) for i in range(3)]).T;targetR=Slerp(clock,Rotation.from_matrix(ref['T_world_from_BODY'][:,:3,:3]))(np.minimum(times,2.)).as_matrix()
            targetv=np.array([np.interp(times,clock,ref['COM_velocity_world_m_s'][:,i]) for i in range(3)]).T;targetw=np.array([np.interp(times,clock,ref['angular_velocity_world_rad_s'][:,i]) for i in range(3)]).T
        dist=np.linalg.norm(COM-target,axis=1);angle=Rotation.from_matrix(R@np.swapaxes(targetR,-1,-2)).magnitude();tail=times>=1.5;pretail=np.arange(N)*dt>=1.5
        metrics=dict(actual_duration_s=N*dt,tracking_COM_RMS_m=float(np.sqrt(np.mean(dist**2))),tracking_COM_max_m=float(dist.max()),tracking_angle_RMS_rad=float(np.sqrt(np.mean(angle**2))),tracking_angle_max_rad=float(angle.max()),
            final_tracking_COM_m=float(dist[-1]),final_tracking_angle_rad=float(angle[-1]),tracking_velocity_RMS_m_s=float(np.sqrt(np.mean(np.sum((velocity-targetv)**2,axis=1)))),tracking_omega_RMS_rad_s=float(np.sqrt(np.mean(np.sum((omega-targetw)**2,axis=1)))),
            final_COM_drift_m=float(np.linalg.norm(COM[-1]-ref['COM_world_m'][0])),final_angle_rad=float(Rotation.from_matrix(R[-1]@ref['T_world_from_BODY'][0,:3,:3].T).magnitude()),
            tail_linear_velocity_RMS_m_s=float(np.sqrt(np.mean(np.sum(velocity[tail]**2,axis=1)))) if tail.any() else None,tail_angular_velocity_RMS_rad_s=float(np.sqrt(np.mean(np.sum(omega[tail]**2,axis=1)))) if tail.any() else None,
            tail_triple_pad_force_coverage=float(np.mean(np.all(Fn[pretail]>.1,axis=1))) if pretail.any() else None,max_native_penetration_m=float(penetration.max()),max_self_penetration_m=float(self_pen.max()),max_phone_nonpad_normal_force_N=float(other_force.max()),max_table_normal_force_N=float(table_force.max()),
            max_joint_limit_excursion_rad=float(max(0.,(model.jnt_range[:20,0]-q[:,:20]).max(),(q[:,:20]-model.jnt_range[:20,1]).max())))
        for key,value in metrics.items():
            original=case['summary']['metrics'][key]
            if (value is None)!=(original is None):raise ValueError('Unavailable tail metric changed')
            if value is not None:maxima['metrics']=max(maxima['metrics'],abs(value-original))
        complete=len(a['k'])==100 and not a['terminated'][-1] and a['complete_H'][-1];physical=dict(case['summary']['physical_gates'])
        for key in physical:
            if key=='warning_count':continue
            value=metrics[key];expected=value is not None and (value>=GATES[key] if key=='tail_triple_pad_force_coverage' else value<=GATES[key])
            if physical[key]!=bool(expected):raise ValueError('Native physical gate changed')
        tracking=bool(complete and metrics['tracking_COM_RMS_m']<=.005 and metrics['tracking_COM_max_m']<=.01 and metrics['tracking_angle_RMS_rad']<=np.pi/36 and metrics['tracking_angle_max_rad']<=np.pi/18 and metrics['tracking_velocity_RMS_m_s']<=.05 and metrics['tracking_omega_RMS_rad_s']<=.5)
        hold=bool(complete and all(physical.values()) and all(metrics[k] is not None and metrics[k]<=GATES[k] for k in ['final_COM_drift_m','final_angle_rad','tail_linear_velocity_RMS_m_s','tail_angular_velocity_RMS_rad_s']));qualified=hold if case['task']=='HOLD' else tracking and all(physical.values())
        if qualified!=case['summary']['task_qualified']:raise ValueError('Qualification changed on independent readback')
        baseline=True;optimizer=True;forecast=True
        if case['controller']=='ZERO':
            history=np.load(verify_file_pin(baselines[dt]));baseline=all(equal_bytes(z[k],history[k]) for k in history.files);optimizer=bool(np.all(a['action20']==0))
        else:
            dec=np.load(verify_file_pin(case['decisions']));Ndec=len(dec['actions']);optimizer=equal_bytes(dec['actions'],a['action20']) and bool(np.all(dec['actions'][:,12:]==0))
            for i in range(Ndec):
                lo,hi=dec['proposal_ranges'][i:i+2];ra,rb=dec['proposal_residual_offsets'][i:i+2];shape=dec['proposal_residual_shapes'][i];residual=dec['proposal_residuals_flat'][ra:rb].reshape(shape);cost=dec['proposal_costs'][lo:hi];proposals=dec['proposal_actions'][lo:hi];eligible=np.r_[0,1,np.arange(26,30)]
                optimizer=optimizer and len(cost)==30 and np.allclose(np.sum(residual**2,axis=1),cost,rtol=1e-12,atol=1e-9) and dec['selected_cost'][i]<=dec['incumbent_cost'][i]+1e-12 and abs(dec['selected_cost'][i]-cost[eligible].min())<=1e-9 and bool(np.any(np.all(proposals[eligible]==dec['actions'][i],axis=1)))
            mask=a['complete_H'];forecast=equal_bytes(dec['selected_first_native_state'][mask],a['next_native_state'][mask]) and all(p['passed'] for p in case['prediction_checks'])
            expected_calls=30*(Ndec+len(case['decision_replays']));expected_steps=30*n*(sum(min(horizon,100-int(k)) for k in a['k'])+sum(min(horizon,100-r['k']) for r in case['decision_replays']))
            optimizer=optimizer and case['oracle_calls']==expected_calls and case['oracle_steps']==expected_steps and all(r['passed'] for r in case['decision_replays'])
        items.append(dict(task=case['task'],controller=case['controller'],dt=dt,legal=legal,no_aid=no_aid,historical_zero_parity=baseline if case['controller']=='ZERO' else None,optimizer_log_valid=bool(optimizer),forecast_actual_full_state_exact=forecast if case['controller']=='NATIVE_MPC' else None,qualified=qualified))
    checks=dict(fixed_expected_cases=len(items)==len(expected_cases) and {(i['task'],i['controller'],i['dt']) for i in items}==expected_cases,raw_control_and_pose=max(maxima[k] for k in ['control','COM_observer','COM_velocity'])<1e-10,raw_contact_readback=max(maxima[k] for k in ['normal_forces','penetration','wrench_mean'])<1e-8,
        independent_metrics=maxima['metrics']<1e-10,all_legal_no_aid=all(i['legal'] and i['no_aid'] for i in items),zero_baselines_preserved=all(i['historical_zero_parity'] for i in items if i['controller']=='ZERO') if any(i['controller']=='ZERO' for i in items) else None,optimizer_and_budget=all(i['optimizer_log_valid'] for i in items),selected_forecasts_equal_actual=all(i['forecast_actual_full_state_exact'] for i in items if i['controller']=='NATIVE_MPC'))
    out.mkdir();result=dict(schema='mug_native_feedback_saved_review:2',source=file_pin(__file__),run=file_pin(run/'report.json'),model=binding['model'],horizon=horizon,expected_cases=sorted(expected_cases),checks={k:bool(v) if v is not None else None for k,v in checks.items()},passed=sum(v is not None and bool(v) for v in checks.values()),total=sum(v is not None for v in checks.values()),max_errors=maxima,cases=items,raw_contact_records_reviewed=raw_count,
        neural_updates=0,native_integration_steps=0,scope='Independent raw contact/control/pose/metric, legacy-zero, optimizer-cost/budget and selected-forecast readback. Warning count comes from recorded online counter; no fresh per-state convex audit, policy training or holding/retargeting success beyond explicit qualification.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result))
    if any(v is not None and not v for v in checks.values()):raise ValueError('Retain failed native review; no success claim')
    return result
