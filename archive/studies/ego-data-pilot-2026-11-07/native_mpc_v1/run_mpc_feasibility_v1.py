"""Bounded true-model MPC feasibility pilot; no learning, object aid, or resets.

Planning clones full native state. Only the selected first20ms control advances
the actual state. A frozen protocol binds source, model, seed and branch fixture.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

from rollout_probe_v1 import RolloutProbe, pin, DT, STEPS_PER_CONTROL

ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent
GATES={
 'duration_s':2.,'final_COM_drift_m':.005,'final_angle_rad':math.radians(5),
 'tail_duration_s':.5,'tail_linear_velocity_RMS_m_s':.01,
 'tail_angular_velocity_RMS_rad_s':.2,'max_native_penetration_m':.002,
 'tail_each_pad_normal_force_N':.1,'tail_dual_pad_force_coverage':.95,
 'max_phone_nonpad_normal_force_N':.1,'max_self_penetration_m':.0002,
 'max_joint_limit_excursion_rad':.001,'max_table_normal_force_N':1e-8,
 'warning_count':0,'abort_COM_drift_m':.05,'abort_angle_rad':math.radians(45)}
CONFIG={
 'actual_control_intervals_max':100,'control_interval_s':.02,
 'native_dt_s':DT,'substeps_per_control':STEPS_PER_CONTROL,'planning_horizon_controls':2,
 'optimized_actuator_indices':list(range(8)),'remaining_ctrl':'original seed control',
 'FD_absolute_control_step_rad':.01,'per_interval_trust_rad':.05,
 'Gauss_Newton_iterations_max':2,'line_search_scales':[1.,.5,.25],
 'damping_residual_weight':.01,'cost_improvement_absolute_min':1e-9,
 'branches_per_interval_max':26,'planning_steps_max':665600,
 'selection_requires_all_original_horizon_safety_limits':True,
 'planning_wall_budget_s':600.,'model_digest_period_branches':128,
 'cost_scales':{'position_m':.005,'rotation_rad':math.radians(5),
   'COM_velocity_m_s':.01,'omega_world_rad_s':.2,
   'posture_rad':.15,'posture_weight':.03,'joint_velocity_weight':.01,
   'minimum_pad_N':.1,'soft_mean_pad_budget_N':3.},
 'scope':'Privileged simulator feasibility oracle, no policy, training, real-time or SuperMimic/CHORD benefit claim.'}


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(x),f,indent=2,allow_nan=False)


def checked(x):
    if pin(x['path'])!=x:raise ValueError('Frozen input changed: '+x['path'])
    return Path(x['path'])


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def residual(branch,initial,q_reference,np,Rotation):
    if branch['status']!='COMPLETE_H2_PLANNING_ONLY' or len(branch['snapshots'])!=2:
        raise ValueError('Incomplete planning is unknown, never an attractive cost')
    cs=CONFIG['cost_scales'];r=[]
    for k,snap in enumerate(branch['snapshots']):
        s=snap['state'];weight=math.sqrt((k+1)/2)
        r.extend(weight*(s['COM_world_m']-initial['COM_world_m'])/cs['position_m'])
        r.extend(weight*Rotation.from_matrix(initial['R_body_world'].T@s['R_body_world']).as_rotvec()/cs['rotation_rad'])
        r.extend(weight*s['v_COM_world_m_s']/cs['COM_velocity_m_s'])
        r.extend(weight*s['omega_world_rad_s']/cs['omega_world_rad_s'])
        r.extend(weight*cs['posture_weight']*(s['q20']-q_reference)/cs['posture_rad'])
        r.extend(weight*cs['joint_velocity_weight']*s['qd20'])
        sl=slice(k*128,(k+1)*128);tr=branch['trace'];fn=tr['pre_pad_normal_N'][sl]
        # Continuous hinge averages, rather than a discontinuous binary coverage objective.
        r.extend(2*np.mean(np.maximum(cs['minimum_pad_N']-fn,0),axis=0)/cs['minimum_pad_N'])
        r.extend(np.maximum(np.mean(fn,axis=0)-cs['soft_mean_pad_budget_N'],0)/cs['soft_mean_pad_budget_N'])
        for key,limit,scale in (
          ('pre_phone_nonpad_normal_N',.1,.1),('pre_table_normal_N',1e-8,.1),
          ('pre_self_penetration_m',.0002,.0002),('pre_penetration_m',.002,.002)):
            r.append(10*float(np.maximum(tr[key][sl]-limit,0).max())/scale)
    r.append(10*max(branch['aggregates']['max_joint_limit_excursion_rad']-.001,0)/.001)
    a=np.asarray(r,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('Nonfinite physical residual')
    return a


def branch_summary(branch,cost):
    return {'branch_index':branch['branch_index'],'cost':cost,'status':branch['status'],
      'controls2x20':branch['controls2x20'],'source_state_before':branch['source_state_before'],
      'source_state_after':branch['source_state_after'],'complete_clone_equal':branch['complete_clone_equal'],
      'aggregates':branch['aggregates'],'actual_planning_steps':branch['actual_planning_steps'],
      'wall_seconds':branch['wall_seconds'],'snapshots':[{k:v for k,v in s.items() if k!='integration_state'} for s in branch['snapshots']],
      'final_state_digest':branch['final_state_digest']}


def horizon_safe(b):
    a=b['aggregates']
    return all(a[k]<=v for k,v in {
      'max_native_penetration_m':.002,'max_self_penetration_m':.0002,
      'max_phone_nonpad_normal_N':.1,'max_table_normal_N':1e-8,
      'max_joint_limit_excursion_rad':.001,'warning_count':0}.items())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--run',action='store_true');a=ap.parse_args();protocol=read(a.protocol)
    if protocol['schema']!='native_true_model_MPC_feasibility_protocol:1':raise ValueError('Protocol schema')
    if protocol['config']!=CONFIG or protocol['gates']!=GATES:raise ValueError('Exact frozen rules')
    for p in protocol['inputs'].values():checked(p)
    if protocol['inputs']['controller_source']!=pin(__file__):raise ValueError('Exact controller source')
    fixture=read(checked(protocol['inputs']['branch_fixture_report']))
    checks=fixture.get('checks',{})
    if (fixture.get('schema')!='native_mpc_two_branch_H2_contract_result:1'
      or fixture.get('status')!='PASS_REPEATABLE_PLANNING_BRANCH_NOT_CHOSEN_ROLLOUT'
      or fixture.get('passed')!=42 or fixture.get('total')!=42 or len(checks)!=42
      or any(v is not True for v in checks.values())
      or fixture.get('source_state_before')!=fixture.get('source_state_after')):
        raise ValueError('First-run native42 clone/snapshot fixture must pass')
    fp=read(checked(fixture['protocol']))
    if (fp['source_pins']['evaluator']!=protocol['inputs']['branch_source']
      or fp['inputs']['initial_state']!=protocol['inputs']['candidate_state']
      or fp['inputs']['model_binding']!=protocol['inputs']['model_binding']):
        raise ValueError('Exact tested branch/seed/model binding')
    for rec in [*fp['source_pins'].values(),*fp['inputs'].values()]:checked(rec)
    out=Path(protocol['output_path'])
    if out.exists():raise ValueError('No overwrite or automatic retries')
    if not a.run:
        print(json.dumps({'status':'METADATA_ONLY_NO_MODEL_STEPS','max_actual_steps':12800,
          'max_planning_steps':CONFIG['planning_steps_max']}));return
    if protocol['CPU_run_authorized'] is not True:raise ValueError('Explicit run binding')
    out.mkdir();write(out/'protocol.json',protocol)
    np=None;probe=None;phase='backend';started=time.perf_counter();actual_steps=0
    rows=[];contacts=[];plans=[];first_interval_all_equal=True;initial_forward=0;actual_contact_reads=0
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation
        from scipy.optimize import lsq_linear
        probe=RolloutProbe.from_existing_pilot(model_digest_period=CONFIG['model_digest_period_branches'])
        m,mj=probe.m,probe.mj
        if int(m.opt.noslip_iterations)!=0:raise ValueError('Original soft friction, no NoSlip')
        with np.load(checked(protocol['inputs']['candidate_state']),allow_pickle=False) as z:
            q0=z['qpos27'].copy();v0=z['qvel26'].copy();u0=z['ctrl20'].copy()
        if np.any(v0):raise ValueError('Registered zero-speed seed')
        d=mj.MjData(m);d.qpos[:]=q0;d.qvel[:]=v0;d.ctrl[:]=u0
        mj.mj_forward(m,d);initial_forward+=1
        initial=probe.observe(d);initial_digest=probe.source_digest(d)
        if initial_digest!=fixture['source_state_before']:raise ValueError('Matched fixture initial state')
        actuator_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j)) for j in m.actuator_trnid[:,0]]
        if (not np.array_equal(m.actuator_trnid[:8,0],np.arange(8))
            or any(not name.startswith('right_finger'+str(1+j//4)+'_') for j,name in enumerate(actuator_names[:8]))):
            raise ValueError('First8 actuators must be original thumb/index in order')
        if np.any(d.warning.number) or np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise ValueError('Clean no-aid start')
        palm_pos=d.xpos[probe.palm].copy();palm_R=d.xmat[probe.palm].copy()
        control=u0.copy();lower=m.actuator_ctrlrange[:8,0];upper=m.actuator_ctrlrange[:8,1]
        if not np.all(m.actuator_ctrllimited[:8]):raise ValueError('Explicit original bounded controls')
        termination='COMPLETE';planning_started=time.perf_counter()
        for interval in range(CONFIG['actual_control_intervals_max']):
            phase='planning_'+str(interval);cache={};evaluations=[]
            lo=np.maximum(lower,control[:8]-.05);hi=np.minimum(upper,control[:8]+.05)
            def evaluate(u,label):
                if time.perf_counter()-planning_started>CONFIG['planning_wall_budget_s']:
                    raise TimeoutError('Registered wall budget, no retry')
                key=np.ascontiguousarray(u).tobytes()
                if key in cache:return cache[key]
                if len(evaluations)>=CONFIG['branches_per_interval_max']:raise ValueError('Registered branch bound')
                if probe.counts['planning_steps_attempted']+256>CONFIG['planning_steps_max']:raise ValueError('Planning step budget')
                b=probe.evaluate(d,np.tile(u,(2,1)))
                if b['status']!='COMPLETE_H2_PLANNING_ONLY':
                    write(out/('unknown_branch_'+str(b['branch_index'])+'.json'),b)
                    raise ValueError('Unknown planning branch preserved, no retry')
                r=residual(b,initial,q0[:20],np,Rotation)
                cost=float(r@r);summary=branch_summary(b,cost);summary['purpose']=label
                summary['original_horizon_safety_limits_pass']=horizon_safe(b)
                evaluations.append(summary);cache[key]=(b,r,cost);return cache[key]
            best_u=control.copy();best_b,best_r,best_cost=evaluate(best_u,'previous_control')
            lsq_records=[]
            for iteration in range(CONFIG['Gauss_Newton_iterations_max']):
                J=np.empty((len(best_r),8));base_u=best_u.copy();base_r=best_r.copy();base_cost=best_cost
                fd_steps=[]
                for j in range(8):
                    du=min(.01,hi[j]-base_u[j])
                    if du<1e-8:du=-min(.01,base_u[j]-lo[j])
                    if abs(du)<1e-8:J[:,j]=0;fd_steps.append(0.);continue
                    fd=base_u.copy();fd[j]+=du
                    _,rr,_=evaluate(fd,'FD_iter'+str(iteration)+'_act'+str(j))
                    actual_du=float(fd[j]-base_u[j])
                    J[:,j]=(rr-base_r)/actual_du;fd_steps.append(actual_du)
                # z=delta_u/.05 keeps ridge meaningful in actuator coordinates.
                A=np.vstack((.05*J,.01*np.eye(8)));rhs=np.r_[-base_r,np.zeros(8)]
                lb=(lo-base_u[:8])/.05;ub=(hi-base_u[:8])/.05
                sol=lsq_linear(A,rhs,bounds=(lb,ub),tol=1e-10,max_iter=100)
                if not np.isfinite(sol.x).all():raise ValueError('Unknown bounded least-squares direction')
                record={'iteration':iteration,'success':bool(sol.success),'status':int(sol.status),
                   'optimality':float(sol.optimality),'FD_steps_rad':fd_steps,'delta_u_rad':.05*sol.x,
                   'base_exact_cost':base_cost,'line_exact_costs':[]}
                # Solver status is descriptive; acceptance always requires a finite exact rollout.
                for scale in CONFIG['line_search_scales']:
                    proposal=base_u.copy();proposal[:8]+=scale*.05*sol.x
                    if np.any(proposal[:8]<lo) or np.any(proposal[:8]>hi):raise ValueError('No bound repair')
                    b,r,cost=evaluate(proposal,'line_iter'+str(iteration)+'_scale'+str(scale))
                    record['line_exact_costs'].append({'scale':scale,'cost':cost,'branch_index':b['branch_index']})
                    if horizon_safe(b) and (not horizon_safe(best_b) or cost<best_cost-CONFIG['cost_improvement_absolute_min']):
                        best_u,best_b,best_r,best_cost=proposal,b,r,cost
                lsq_records.append(record)
                if best_cost>=base_cost-CONFIG['cost_improvement_absolute_min']:break
            if not horizon_safe(best_b):
                write(out/('no_safe_branch_interval_'+str(interval)+'.json'),
                    {'interval':interval,'evaluations':evaluations,'bounded_GN':lsq_records})
                termination='ABORT_NO_SAFE_PREDICTED_CONTROL';break
            selected=best_b['snapshots'][0]
            interval_info={'interval':interval,'start_tick':actual_steps,'evaluations':evaluations,
              'bounded_GN':lsq_records,'selected_branch':best_b['branch_index'],'selected_control20':best_u,
              'selected_cost':best_cost,'source_digest_before_actual':probe.source_digest(d),
              'selected_first_snapshot':selected,'selected_horizon_aggregates':best_b['aggregates']}
            write(out/('selection_before_actual_'+str(interval).zfill(3)+'.json'),interval_info)
            control=best_u.copy();d.ctrl[:]=control
            phase='actual_interval_'+str(interval)
            completed_interval=True
            for local in range(128):
                tick=actual_steps;mj.mj_step(m,d);actual_steps+=1
                if not all(np.isfinite(v).all() for v in (d.qpos,d.qvel,d.qacc,d.energy,d.qfrc_actuator,d.actuator_force)):
                    termination='ABORT_NONFINITE';completed_interval=False;break
                if (np.any(d.qfrc_applied) or np.any(d.xfrc_applied) or not np.array_equal(d.ctrl,control)
                    or not np.array_equal(d.xpos[probe.palm],palm_pos) or not np.array_equal(d.xmat[probe.palm],palm_R)):
                    raise ValueError('No-aid, chosen control or fixed-wrist mutation')
                pad=np.zeros(2);other=table=pen=selfpen=0.
                for ci in range(d.ncon):
                    con=d.contact[ci];g1,g2=map(int,con.geom);force=np.empty(6)
                    mj.mj_contactForce(m,d,ci,force)
                    actual_contact_reads+=1
                    if not np.isfinite(np.r_[con.dist,con.pos,con.frame,force]).all():raise ValueError('Nonfinite actual contact')
                    contacts.append(np.r_[tick,g1,g2,int(con.dim),int(con.efc_address),con.dist,con.pos,con.frame,force])
                    pen=max(pen,-float(con.dist))
                    if probe.phone in (g1,g2):
                        othergeom=g2 if g1==probe.phone else g1
                        if othergeom in probe.pads:pad[probe.pads.index(othergeom)]+=max(0.,float(force[0]))
                        elif othergeom==probe.table:table+=max(0.,float(force[0]))
                        else:other+=max(0.,float(force[0]))
                    elif probe.table not in (g1,g2):selfpen=max(selfpen,-float(con.dist))
                s=probe.observe(d)
                angle=float(2*np.arccos(np.clip(abs(d.qpos[23:27]@q0[23:27]),0,1)))
                rows.append({'pre_tick':tick,'post_tick':tick+1,'post_time_native_s':float(d.time),
                  'post_COM':s['COM_world_m'],'post_angle':angle,'post_qpos':d.qpos.copy(),'post_qvel':d.qvel.copy(),
                  'post_v_COM':s['v_COM_world_m_s'],'post_omega_world':s['omega_world_rad_s'],
                  'pre_native_energy':d.energy.copy(),'pre_pad_normal_N':pad,'pre_phone_nonpad_normal_N':other,
                  'pre_table_normal_N':table,'pre_penetration_m':pen,'pre_self_penetration_m':selfpen,
                  'pre_qfrc_actuator':d.qfrc_actuator.copy(),'pre_actuator_force':d.actuator_force.copy(),
                  'pre_solver_niter':np.asarray(d.solver_niter).copy(),'chosen_ctrl20':control.copy()})
                if np.any(d.warning.number):termination='ABORT_NATIVE_WARNING';completed_interval=False;break
                if abs(d.time-(tick+1)*DT)>1e-10:raise ValueError('Actual native clock reset/drift')
                if np.linalg.norm(s['COM_world_m']-initial['COM_world_m'])>.05 or angle>math.radians(45):
                    termination='ABORT_LOSS_OF_HOLD';completed_interval=False;break
            if completed_interval:
                actual_state=probe.integration_state(d);predicted=np.asarray(selected['integration_state'],dtype=np.float64)
                same=actual_state.tobytes()==predicted.tobytes();first_interval_all_equal &= same
                interval_info['actual_vs_selected_first_state_raw_bit_equal']=same
                interval_info['actual_boundary_integration_sha256']=hashlib.sha256(actual_state.tobytes()).hexdigest()
                if not same:raise ValueError('Oracle selected first interval does not match actual integration state')
            else:interval_info['actual_vs_selected_first_state_raw_bit_equal']='NOT_FULL_INTERVAL'
            plans.append(interval_info)
            write(out/('interval_'+str(interval).zfill(3)+'.json'),interval_info)
            print(json.dumps({'interval':interval,'actual_time_s':float(d.time),'COM_drift_mm':1000*float(np.linalg.norm(probe.observe(d)['COM_world_m']-initial['COM_world_m'])),
              'selected_cost':best_cost,'branches':len(evaluations),'planning_steps':probe.counts['planning_steps_returned'],'termination':termination}),flush=True)
            if not completed_interval:break
        phase='save_and_final_audit';audit=probe.audit_model()
        if not rows:raise ValueError('No finite actual records')
        tr={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(out/'trace.npz',**tr,contact_rows24=np.asarray(contacts).reshape(-1,24),
          initial_qpos27=q0,initial_qvel26=v0,initial_ctrl20=u0,initial_COM_world3=initial['COM_world_m'])
        tail=tr['post_tick']>=round(1.5/DT);pretail=tr['pre_tick']>=round(1.5/DT)
        rms=lambda a:float(np.sqrt(np.mean(np.sum(a*a,axis=1)))) if len(a) else None
        qvio=np.maximum(m.jnt_range[:20,0]-tr['post_qpos'][:,:20],0)+np.maximum(tr['post_qpos'][:,:20]-m.jnt_range[:20,1],0)
        metrics={'final_COM_drift_m':float(np.linalg.norm(tr['post_COM'][-1]-initial['COM_world_m'])),
          'final_angle_rad':float(tr['post_angle'][-1]),
          'tail_linear_velocity_RMS_m_s':rms(tr['post_qvel'][tail,20:23]),
          'tail_angular_velocity_RMS_rad_s':rms(tr['post_qvel'][tail,23:26]),
          'tail_dual_pad_force_coverage':float(np.mean(np.all(tr['pre_pad_normal_N'][pretail]>.1,axis=1))) if np.any(pretail) else None,
          'max_native_penetration_m':float(tr['pre_penetration_m'].max()),
          'max_self_penetration_m':float(tr['pre_self_penetration_m'].max()),
          'max_phone_nonpad_normal_force_N':float(tr['pre_phone_nonpad_normal_N'].max()),
          'max_table_normal_force_N':float(tr['pre_table_normal_N'].max()),
          'max_joint_limit_excursion_rad':float(qvio.max()),'warning_count':int(np.sum(d.warning.number))}
        gates={'duration_complete':termination=='COMPLETE' and actual_steps==12800}
        for key,value in metrics.items():gates[key]=value is not None and (value>=GATES[key] if key=='tail_dual_pad_force_coverage' else value<=GATES[key])
        write(out/'report.json',{'schema':'native_true_model_MPC_feasibility_result:1',
          'status':'COMPLETED_BOUNDED_FINE_DT_PILOT','termination':termination,'protocol':pin(a.protocol),
          'initial_boundary_digest':initial_digest,'initial_forward_calls':initial_forward,
          'actuator_joint_names':actuator_names,
          'actual_integration_steps':actual_steps,'actual_finite_rows':len(rows),'actual_final_time_s':float(d.time),
          'actual_contact_force_reads':actual_contact_reads,'planning_actual_count_scope':'probe counters describe clone planning only; actual steps and contact reads are separate top-level fields',
          'planning_counts':probe.counts,'model_audit':audit,'metrics':metrics,'gates':gates,
          'original_2s_single_dt_holding_pass':all(gates.values()),'two_dt_qualified':False,
          'all_complete_selected_intervals_raw_bit_equal':first_interval_all_equal,
          'extra_COM_tail_RMS_m_s':rms(tr['post_v_COM'][tail]),
          'mean_pad_normal_N':np.mean(tr['pre_pad_normal_N'],axis=0),'peak_pad_normal_N':np.max(tr['pre_pad_normal_N'],axis=0),
          'mean_pad_3N_design_budget_pass':bool(np.all(np.mean(tr['pre_pad_normal_N'],axis=0)<=3)),
          'contact_force_phase':'PRE native solver; qpos/qvel/COM POST, integer tick explicit',
          'velocity_gate_compatibility':'Original root-linear/body-angular metrics retained; COM RMS separately reported.',
          'mass_friction_calibrated':False,'training_or_policy_benefit_qualified':False,
          'trace':pin(out/'trace.npz'),'selected_intervals':len(plans),'wall_seconds':time.perf_counter()-started,
          'scope':CONFIG['scope']})
    except BaseException as e:
        # Preserve unknown/aborted diagnostics; no automatic rerun or candidate selection.
        if np is not None and rows:
            tr={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
            np.savez_compressed(out/'failure_actual_trace.npz',**tr,contact_rows24=np.asarray(contacts).reshape(-1,24))
        write(out/'failure.json',{'phase':phase,'error':repr(e),'actual_steps':actual_steps,
          'actual_finite_rows':len(rows),'completed_intervals':len(plans),'initial_forward_calls':initial_forward,
          'planning_counts':None if probe is None else probe.counts,'wall_seconds':time.perf_counter()-started,
          'automatic_retry':False})
        raise


if __name__=='__main__':main()
