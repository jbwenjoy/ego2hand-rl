"""Saved trace/control audit, fixed-reset Jacobian check and LP sensitivity.

One MJB is read for actual dry-friction/limit metadata; no MjData/forward/step.
Pure Torch FK input AD checks the six saved contact point Jacobians without
constructing any NN. LP force witnesses remain mathematical, never applied.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
from scipy.optimize import linprog
from evaluate_online import preflight
from control_authority import compare_trace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning import online_contract as C,runner as R
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime import native_evaluation as E


def support_interval(wrench,loads,offset,lower,upper,friction,radius):
    """Allow dry-friction load anywhere in its actual symmetric limit box."""
    n=wrench.shape[1];eq=np.c_[wrench,-np.array([0.,0.,1.,0.,0.,0.])];eq[3:]/=radius
    A=np.r_[np.c_[loads,np.zeros(20)],np.c_[-loads,np.zeros(20)]]
    b=np.r_[upper+friction-offset,offset-lower+friction];answers={}
    for name,sign in [('minimum',1),('maximum',-1)]:
        objective=np.zeros(n+1);objective[-1]=sign
        result=linprog(objective,A_eq=eq,b_eq=np.zeros(6),A_ub=A,b_ub=b,bounds=(0,None),method='highs')
        answers[name]=dict(success=bool(result.success),status=int(result.status),support_weight_N=float(result.x[-1]) if result.success else None,message=result.message)
    return answers


def review(args):
    path,c,registry,root,final,health,zero,binding,out=preflight(args)
    authority=C.read(args.authority_report);torque=C.read(args.torque_report)
    C.need(C.positive(authority) and C.positive(torque),'Completed qualified diagnostic records')
    C.need(authority['data_role']=='NO_FITTING_CONTROL_AUTHORITY_DIAGNOSTIC','No-fitting role')
    with np.load(registry.resolve_pin(C.read(registry.resolve_pin(authority['intent']))['plan']),allow_pickle=False) as z:plan=z['normalized_action65x100x20_F64'].copy()
    bundle=C.read(registry.resolve(c['artifacts']['task']));base=np.array(bundle['initial']['ctrl20_F64']);scale=np.array(bundle['action']['symmetric_residual_scale20_rad']);checks={};details={};counts=dict(MJB_reads=0,MjData=0,initial_forward=0,integration_steps=0,geometry_input_AD=0,LP_calls=0,NN_constructors=0,NN_FW=0,learning_updates=0,GPU_calls=0)
    tick=time.perf_counter();total_steps=0;total_calls=0;all_controls=all_gates=True
    for case in authority['cases']:
        summary=case['native_summary'];records=C.read(registry.resolve_pin(case['controller_records']));dt=case['dt_s'];n=round(.02/dt);residual=np.zeros(20);last=base.copy();a1=np.zeros(20);a2=np.zeros(20)
        issued=[]
        for k,record in enumerate(records):
            memory=np.r_[k,a1,a2,residual,last,dt,n];action=plan[case['schedule_index'],k]
            all_controls=all_controls and np.array(record['memory83_before']).tobytes()==memory.tobytes() and np.array(record['action20_F64']).tobytes()==action.tobytes()
            residual=.5*scale*action+.5*residual;last=base+residual
            all_controls=all_controls and np.array(record['residual20_F64']).tobytes()==residual.tobytes() and np.array(record['executed_u20_F64']).tobytes()==last.tobytes()
            issued.append(last.copy());a2=a1.copy();a1=action.copy()
        with np.load(registry.resolve_pin(summary['trace']),allow_pickle=False) as z:
            N=len(z['post_time']);total_steps+=N;total_calls+=len(records)
            all_controls=all_controls and len(records)==(N+n-1)//n and np.array(issued)[np.arange(N)//n].tobytes()==z['pre_executed_ctrl20'].tobytes()
            tail=np.arange(1,N+1)>=round(1.5/dt);pre_tail=np.arange(N)>=round(1.5/dt)
            q=z['post_qpos'][:,:20];qr=np.array(bundle['joint_range20x2']);violation=np.maximum(qr[:,0]-q,0)+np.maximum(q-qr[:,1],0)
            metrics=dict(final_COM_drift_m=float(np.linalg.norm(z['post_COM'][-1]-z['initial_COM_world3'])),final_angle_rad=float(z['post_angle'][-1]),
                tail_linear_velocity_RMS_m_s=float(np.sqrt(np.mean(np.sum(z['post_qvel'][tail,20:23]**2,axis=1)))) if tail.any() else None,
                tail_angular_velocity_RMS_rad_s=float(np.sqrt(np.mean(np.sum(z['post_qvel'][tail,23:26]**2,axis=1)))) if tail.any() else None,
                tail_triple_pad_force_coverage=float(np.mean(np.all(z['pre_pad_normal_N'][pre_tail]>.1,axis=1))) if pre_tail.any() else None,
                max_native_penetration_m=float(z['pre_penetration_m'].max()),max_self_penetration_m=float(z['pre_self_penetration_m'].max()),
                max_phone_nonpad_normal_force_N=float(z['pre_phone_nonpad_normal_N'].max()),max_table_normal_force_N=float(z['pre_table_normal_N'].max()),max_joint_limit_excursion_rad=float(violation.max()),warning_count=int(np.sum(summary['warnings'])))
            gates=dict(duration_complete=summary['termination']=='COMPLETE' and N==round(2/dt))
            gates.update({key:value is not None and (value>=E.GATES[key] if key=='tail_triple_pad_force_coverage' else value<=E.GATES[key]) for key,value in metrics.items()})
            all_gates=all_gates and metrics==summary['metrics'] and gates==summary['gates'] and summary['nominal_holding_dynamics_pass']==all(gates.values())
    checks['all130_actual_controls_memory_and_no_hidden_clamp']=all_controls
    checks['all130_metrics_original_gates_and_missing_tails']=all_gates
    checks['all130_cases_and_actual_native_budget']=len(authority['cases'])==130 and total_steps==authority['counts']['integration_steps'] and total_calls==authority['counts']['controller_calls']
    checks['both_zero_baselines_all_trace_bytes']=all(all(compare_trace(next(z['trace'] for z in zero['cases'] if z['dt_s']==case['dt_s']),registry.resolve_pin(case['native_summary']['trace']),registry).values()) for case in authority['cases'] if case['schedule']=='zero')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import mujoco,torch
    torch.set_num_threads(4)
    model=mujoco.MjModel.from_binary_path(str(registry.resolve(c['artifacts']['model'])));counts['MJB_reads']=1;E.assert_model(model,mujoco,np,C.read(binding));digest=E.model_digest_except_dt(model,np)
    friction=model.dof_frictionloss[:20].copy();geometry=R._geometry(registry,c);jacobian_checks=[];sensitivities=[]
    for rec in torque['records']:
        with np.load(registry.resolve_pin(rec['raw']),allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
        q=raw['qpos27'];distance=np.minimum(q[:20]-model.jnt_range[:20,0],model.jnt_range[:20,1]-q[:20])-model.jnt_margin[:20]
        C.need((distance>0).all(),'No active joint-limit contact at the frozen reset')
        checks[str(rec['dt_s'])+'_other_constraint_within_dry_friction']=bool(np.all(abs(raw['observed_other_constraint_torque20'])<=friction+2e-12))
        source=np.array(bundle['initial']['native_PRE_state58_F64']);x=torch.tensor(source,dtype=torch.float64);baseposes=geometry.provider.poses(x)
        rows=raw['contact_rows24'];phone_rows=rows[(rows[:,1]==71)|(rows[:,2]==71)]
        for i,row in enumerate(phone_rows):
            gid=int(row[1] if row[2]==71 else row[2]);p=baseposes['geom_position_world_m'][gid];rot=baseposes['geom_rotation_world'][gid]
            local=rot.T@(torch.tensor(row[6:9])-p)
            def point(q20):
                state=torch.cat([q20,x[20:]]);v=geometry.provider.poses(state)
                return v['geom_position_world_m'][gid]+v['geom_rotation_world'][gid]@local
            jac=torch.autograd.functional.jacobian(point,torch.tensor(q[:20],dtype=torch.float64,requires_grad=True)).detach().numpy();counts['geometry_input_AD']+=1
            error=float(abs(jac-raw['contact_jacobians3x3x20'][i]).max());jacobian_checks.append(dict(dt_s=rec['dt_s'],geom_id=gid,max_abs_m_per_rad=error,passed=bool(np.allclose(jac,raw['contact_jacobians3x3x20'][i],rtol=2e-12,atol=2e-12))))
        COM=raw['body_COM_world_m'];offset=raw['qfrc_bias20']-raw['qfrc_passive20']
        for rays in (4,16):
            for approximation,amplitude in [('inscribed',.7),('circumscribed_ideal_circle',.7/np.cos(np.pi/rays))]:
                columns=[];loads=[]
                for row,jac in zip(phone_rows,raw['contact_jacobians3x3x20']):
                    sign=1 if int(row[2])==71 else -1;frame=row[9:18].reshape(3,3)
                    for theta in np.arange(rays)*2*np.pi/rays:
                        force=sign*(frame[0]+amplitude*(np.cos(theta)*frame[1]+np.sin(theta)*frame[2]));columns.append(np.r_[force,np.cross(row[6:9]-COM,force)]);loads.append(jac.T@force)
                G=np.array(columns).T;B=np.array(loads).T
                for domain in ('first_H_EMA_half_scale','steady_residual_scale','full_actuator_target_range_analytical_only'):
                    answer=support_interval(G,B,offset,raw[domain+'__torque_lower'],raw[domain+'__torque_upper'],friction,bundle['support']['radius_m']);counts['LP_calls']+=2
                    sensitivities.append(dict(dt_s=rec['dt_s'],rays=rays,cone=approximation,domain=domain,support_weight_interval=answer))
    checks['all_six_kinematic_point_Jacobians_match_native']=all(row['passed'] for row in jacobian_checks)
    checks['metadata_model_unchanged']=digest==E.model_digest_except_dt(model,np)
    checks['no_CUDA_or_neural_execution']=not torch.cuda.is_initialized()
    details=dict(jacobian_checks=jacobian_checks,dof_dry_friction_limits_Nm=friction.tolist(),support_weight_sensitivity=sensitivities,total_cases=130,actual_native_steps_from_saved=total_steps)
    report=dict(schema='supermimic_research.control_authority_saved_review:1',checks=checks,passed=sum(checks.values()),total=len(checks),counts=counts,details=details,
        authority_report=file_pin(args.authority_report),torque_report=file_pin(args.torque_report),auditor=file_pin(__file__),elapsed_s=time.perf_counter()-tick,
        scope='Saved control/gate audit plus isolated reset-metadata read, six FK input ADs and ideal-cone LP bounds with dry-friction load freely varied. No integration, new neural inference, fitting or force application. Numerical LP results are not formal native/dynamic impossibility certificates.')
    C.need(all(checks.values()),'Authority saved review failure: '+','.join(k for k,v in checks.items() if not v))
    out.mkdir(parents=True);C.write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),passed=report['passed'],total=report['total'],counts=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--zero-report',type=Path,required=True);p.add_argument('--authority-report',type=Path,required=True);p.add_argument('--torque-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);review(p.parse_args())
