"""Read-only saved-case review and fresh static candidate replay, no integration."""
from pathlib import Path
import argparse,json,os,sys
import numpy as np
from evaluate_online import preflight
from control_authority import compare_trace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.learning import online_contract as C
from supermimic_research.runtime import native_evaluation as E


def review(args):
    path,c,registry,root,final,health,zero,binding,out=preflight(args)
    report=C.read(args.report);intent=C.read(registry.resolve_pin(report['intent']))
    # Validate the declared dependency closure. Unused new mechanics modules
    # do not change the bytes executed by the completed experiment.
    C.need(C.positive(report) and all(C.content(pin)==C.content(file_pin(ROOT/name.replace('\\','/'))) for name,pin in intent['sources'].items()),'Unchanged declared mechanical dependencies')
    C.need(report['data_role']=='NO_FITTING_INITIALIZATION_DEVELOPMENT' and report['NN_training_updates']==0 and report['new_task_bundles']==0,'No fitting/task promotion')
    checks={};steps=calls=0;all_controls=all_gates=True
    b=C.read(registry.resolve(c['artifacts']['task']));q0=np.array(b['initial']['qpos27_F64']);u0=np.array(b['initial']['ctrl20_F64']);qr=np.array(b['joint_range20x2'])
    best=[]
    for candidate in report['candidates']:
        log=[json.loads(line) for line in registry.resolve_pin(candidate['proposal_log']).read_text(encoding='utf8').splitlines()]
        minimum=min(log,key=lambda row:row['cost']);selected=candidate['search']['best']
        C.need([row['call'] for row in log]==list(range(1,len(log)+1)) and len(log)==candidate['search']['calls'],'All declared proposal rows retained')
        C.need(minimum['call']==selected['call'] and minimum['cost']==selected['cost'] and np.array(minimum['z38']).tobytes()==np.array(selected['z']).tobytes(),'Predeclared best-score selection, no physical-result selection')
        best.append(selected)
    for case in report['cases']:
        s=case['native_summary'];dt=s['dt_s'];n=round(.02/dt)
        with np.load(registry.resolve_pin(s['trace']),allow_pickle=False) as z:
            N=len(z['post_time']);steps+=N;calls+=(N+n-1)//n
            expected=u0
            if case['role']!='original_zero':
                with np.load(registry.resolve_pin(next(x for x in report['candidates'] if x['name']==case['candidate'])['reset_state']),allow_pickle=False) as reset:expected=reset['ctrl20'].copy()
            all_controls=all_controls and z['pre_executed_ctrl20'].tobytes()==np.broadcast_to(expected,(N,20)).copy().tobytes()
            tail=np.arange(1,N+1)>=round(1.5/dt);pre_tail=np.arange(N)>=round(1.5/dt);q=z['post_qpos'][:,:20]
            metrics=dict(final_COM_drift_m=float(np.linalg.norm(z['post_COM'][-1]-z['initial_COM_world3'])),final_angle_rad=float(z['post_angle'][-1]),
                tail_linear_velocity_RMS_m_s=float(np.sqrt(np.mean(np.sum(z['post_qvel'][tail,20:23]**2,axis=1)))) if tail.any() else None,
                tail_angular_velocity_RMS_rad_s=float(np.sqrt(np.mean(np.sum(z['post_qvel'][tail,23:26]**2,axis=1)))) if tail.any() else None,
                tail_triple_pad_force_coverage=float(np.mean(np.all(z['pre_pad_normal_N'][pre_tail]>.1,axis=1))) if pre_tail.any() else None,
                max_native_penetration_m=float(z['pre_penetration_m'].max()),max_self_penetration_m=float(z['pre_self_penetration_m'].max()),
                max_phone_nonpad_normal_force_N=float(z['pre_phone_nonpad_normal_N'].max()),max_table_normal_force_N=float(z['pre_table_normal_N'].max()),
                max_joint_limit_excursion_rad=float((np.maximum(qr[:,0]-q,0)+np.maximum(q-qr[:,1],0)).max()),warning_count=int(sum(s['warnings'])))
            gates=dict(duration_complete=s['termination']=='COMPLETE' and N==round(2/dt))
            gates.update({key:value is not None and (value>=E.GATES[key] if key=='tail_triple_pad_force_coverage' else value<=E.GATES[key]) for key,value in metrics.items()})
            all_gates=all_gates and metrics==s['metrics'] and gates==s['gates'] and s['nominal_holding_dynamics_pass']==all(gates.values())
    checks['all_search_rows_and_score_selection']=len(best)==2
    checks['six_constant_legal_native_controls']=all_controls
    checks['six_original_metrics_gates_missing_tails']=all_gates
    checks['saved_native_step_and_command_counts']=steps==report['counts']['integration_steps'] and calls==report['counts']['controller_calls']
    checks['both_original_zero_traces_raw']=all(all(compare_trace(next(x['trace'] for x in zero['cases'] if x['dt_s']==case['native_summary']['dt_s']),registry.resolve_pin(case['native_summary']['trace']),registry).values()) for case in report['cases'] if case['role']=='original_zero')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import mujoco
    from supermimic_research.mechanics.initialization import NativeInitialization
    from supermimic_research.mechanics.collision import CollisionAudit
    model=mujoco.MjModel.from_binary_path(str(registry.resolve(c['artifacts']['model'])));E.assert_model(model,mujoco,np,C.read(binding));digest=E.model_digest_except_dt(model,np);model.opt.timestep=E.DTS[0]
    engine=NativeInitialization(model,q0,u0);audit=CollisionAudit(model);replays=[]
    for candidate in report['candidates']:
        replay=engine.state(np.array(candidate['search']['best']['z']));replayed_score=float(np.sum(engine.residual(np.array(candidate['search']['best']['z']),replay)**2))
        with np.load(registry.resolve_pin(candidate['reset_state']),allow_pickle=False) as z:exact=all(np.asarray(replay[k]).dtype==z[k].dtype and np.asarray(replay[k]).shape==z[k].shape and np.asarray(replay[k]).tobytes()==z[k].tobytes() for k in replay)
        geometry=audit.evaluate(engine.data);expected=C.read(registry.resolve_pin(candidate['geometry']))
        raw=np.empty(26);mujoco.mj_mulM(model,engine.data,raw,engine.data.qacc);inertia_error=float(abs(raw-replay['net_generalized26']).max())
        replays.append(dict(name=candidate['name'],all_static_arrays_raw_equal=bool(exact),score_matches=abs(replayed_score-candidate['search']['fresh_final_cost'])<1e-10,
            full_geometry_matches=geometry==expected,inertia_times_acceleration_error=inertia_error,reference_deviation_matches=engine.deviation(replay['qpos27'])==C.read(registry.resolve_pin(candidate['reference_deviation']))))
    checks['fresh_selected_static_all_arrays_and_scores']=all(x['all_static_arrays_raw_equal'] and x['score_matches'] for x in replays)
    checks['fresh_full_geometry_and_reference_deviation']=all(x['full_geometry_matches'] and x['reference_deviation_matches'] for x in replays)
    checks['actual_inertia_times_acceleration_matches_net']=all(x['inertia_times_acceleration_error']<1e-10 for x in replays)
    checks['unchanged_nominal_model']=digest==E.model_digest_except_dt(model,np)
    checks['no_dynamic_or_neural_execution']=engine.counts['integration_steps']==0 and 'torch' not in sys.modules
    C.need(all(checks.values()),'Initialization review failed: '+','.join(k for k,v in checks.items() if not v))
    out.mkdir(parents=True);answer=dict(schema='supermimic_research.initialization_saved_review:1',report=file_pin(args.report),auditor=file_pin(__file__),checks=checks,passed=sum(checks.values()),total=len(checks),
        fresh_static_counts=engine.counts,model_reads=1,matrix_vector_reads=2,replays=replays,NN_FW=0,learning_updates=0,GPU_calls=0,
        scope='Saved full proposal-selection/control/gate audit plus three reset forwards, two independent inertia relations and fresh full convex geometry; no integration or fitting. Shared native reset code is replayed, not an independent contact solver.')
    C.write(out/'report.json',answer);print(json.dumps(dict(report=file_pin(out/'report.json'),passed=answer['passed'],total=answer['total'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('config','result','terminal','zero-report','report','output'):p.add_argument('--'+flag,type=Path,required=True)
    review(p.parse_args())
