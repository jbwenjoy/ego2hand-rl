"""Saved source/selection/native-case checks and fresh selected-state geometry.

No optimization, native integration or archived Python execution. Replaying
geometry uses the current shared numerical checker; old source bytes are
verified separately through their archived pins.
"""
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
    path,c,registry,root,training,health,zero,binding,out=preflight(args)
    mapping=C.read(ROOT/'provenance/patch_migration/v1_source_map.json');snapshots={x['original']['sha256']:ROOT/x['snapshot'] for x in mapping}
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import mujoco
    from supermimic_research.mechanics.collision import CollisionAudit
    from supermimic_research.mechanics.initialization import NativeInitialization
    from supermimic_research.mechanics.sliding import GeometryRestoration,collider_axes,compose_deviation
    model=mujoco.MjModel.from_binary_path(str(registry.resolve(c['artifacts']['model'])));E.assert_model(model,mujoco,np,C.read(binding));digest=E.model_digest_except_dt(model,np);model.opt.timestep=E.DTS[0]
    b=C.read(registry.resolve(c['artifacts']['task']));q0=np.array(b['initial']['qpos27_F64']);u0=np.array(b['initial']['ctrl20_F64']);audit=CollisionAudit(model);checks={};details=[]
    totals=dict(model_reads=1,restoration_kinematics=0,selected_force_forwards=0,integration_steps=0,NN_FW=0,GPU_calls=0)
    axis_data=mujoco.MjData(model);axis_data.qpos[:]=q0;mujoco.mj_kinematics(model,axis_data)
    actual_axes=collider_axes(audit.locals[71],axis_data.geom_xmat[71].reshape(3,3))
    for filename in args.reports:
        report=C.read(filename);intent=C.read(registry.resolve_pin(report['intent']));label=filename.parent.name
        C.need(C.positive(report) and report['role']=='NO_FITTING_PATCH_MIGRATION_DEVELOPMENT','Completed no-fitting migration record')
        for pin in intent['sources'].values():
            candidate=Path(pin['path'])
            if file_pin(candidate)['sha256']!=pin['sha256']:candidate=snapshots[pin['sha256']]
            actual=file_pin(candidate);C.need((actual['bytes'],actual['sha256'])==(pin['bytes'],pin['sha256']),'Declared executed source bytes')
        C.need(C.fingerprint(intent['runtime_sources'])==C.fingerprint(C.source_pins()),'Unchanged actual native runtime')
        saved_axes=C.read(filename.parent/'compiled_axes.json');checks[label+'_compiled_axes']=actual_axes==saved_axes
        seed_details=[]
        for record in report['records']:
            log=[json.loads(line) for line in registry.resolve_pin(record['restoration_proposals']).read_text().splitlines()]
            best=record['restoration']['best'];minimum=min(log,key=lambda row:row['score'])
            C.need(minimum==best and [x['call'] for x in log]==list(range(1,len(log)+1)),'All proposal rows and predeclared minimum')
            joints=intent['restoration'].get('variable_joints',12);z=np.array(best.get('z_joint',best.get('z12')))
            qseed=q0.copy();qseed[20:23]+=np.array(record['shift_world_m']);engine=GeometryRestoration(model,audit,qseed,variable_joints=joints)
            q,geometry=engine.state(z);totals['restoration_kinematics']+=1
            with np.load(registry.resolve_pin(record['restored_state']),allow_pickle=False) as raw:exact=q.tobytes()==raw['qpos27'].tobytes()
            C.need(exact and geometry==C.read(registry.resolve_pin(record['restored_geometry'])),'Fresh full selected restoration geometry matches')
            C.need(record['status']=='FORCE_CANDIDATE_EVALUATED' or not geometry['passed'],'Skipped seed has no qualified restored geometry')
            if record['status']=='FORCE_CANDIDATE_EVALUATED':
                screen=C.read(registry.resolve_pin(record['native_screen']));native=NativeInitialization(model,q,np.array(screen['legal_seed_u20']))
                force_log=[json.loads(line) for line in registry.resolve_pin(record['force_proposals']).read_text().splitlines()]
                C.need(min(force_log,key=lambda row:row['cost'])['call']==record['force_search']['best']['call'],'Force choice precedes dynamics')
                final=native.state(np.array(record['force_search']['best']['z']));totals['selected_force_forwards']+=native.counts['static_forward']
                with np.load(registry.resolve_pin(record['final_state']),allow_pickle=False) as raw:C.need(all(np.asarray(v).tobytes()==raw[k].tobytes() for k,v in final.items()),'Fresh selected native reset arrays')
                C.need(audit.evaluate(native.data)==C.read(registry.resolve_pin(record['final_geometry'])),'Fresh final convex geometry')
                dev=compose_deviation(native.deviation(final['qpos27']),q,q0,record['shift_world_m'])
                C.need(dev==C.read(registry.resolve_pin(record['reference_deviation'])),'Full original-reference deviation')
            qualification=C.read(registry.resolve_pin(record['qualification']));C.need(qualification['qualified']==record['qualified'],'Final pair qualification retained')
            seed_details.append(dict(index=record['index'],proposal_rows=len(log),geometry_pass=geometry['passed'],status=record['status']))
        steps=0;controls=True;metrics_match=True;zero_match=True
        for case in report['cases']:
            s=case['native_summary'];dt=s['dt_s']
            if case['role']=='original_zero':expected=u0;zero_match=zero_match and all(compare_trace(next(x['trace'] for x in zero['cases'] if x['dt_s']==dt),registry.resolve_pin(s['trace']),registry).values())
            else:
                rec=next(x for x in report['records'] if x['index']==case['seed'])
                with np.load(registry.resolve_pin(rec['final_state']),allow_pickle=False) as raw:expected=raw['ctrl20'].copy()
            with np.load(registry.resolve_pin(s['trace']),allow_pickle=False) as z:
                N=len(z['post_time']);steps+=N;controls=controls and z['pre_executed_ctrl20'].tobytes()==np.broadcast_to(expected,(N,20)).copy().tobytes()
                metrics=s['metrics'];angle=float(2*np.arccos(np.clip(abs(z['post_qpos'][-1,23:27]@z['initial_qpos27'][23:27]),0,1)))
                metrics_match=metrics_match and metrics['final_COM_drift_m']==float(np.linalg.norm(z['post_COM'][-1]-z['initial_COM_world3'])) and metrics['final_angle_rad']==angle
                gates=dict(duration_complete=s['termination']=='COMPLETE' and N==round(2/dt))
                gates.update({k:v is not None and (v>=E.GATES[k] if k=='tail_triple_pad_force_coverage' else v<=E.GATES[k]) for k,v in metrics.items()})
                metrics_match=metrics_match and gates==s['gates'] and s['nominal_holding_dynamics_pass']==all(gates.values())
                if N<round(1.5/dt):metrics_match=metrics_match and all(metrics[k] is None for k in ('tail_linear_velocity_RMS_m_s','tail_angular_velocity_RMS_rad_s','tail_triple_pad_force_coverage'))
        checks[label+'_all_eight_selections_and_geometry']=len(seed_details)==8
        checks[label+'_original_zero_raw']=zero_match
        checks[label+'_actual_controls_counts_and_gates']=controls and metrics_match and steps==report['counts']['integration_steps']
        if report.get('geometry_query_cache'):
            cache=report['geometry_query_cache'];expected=(sum(x['geometry_audits'] for x in report['restoration_counts'])+len(report['force_counts']))*211
            checks[label+'_pair_cache_accounting']=cache['calls']==cache['hits']+cache['misses']==expected
        details.append(dict(report=file_pin(filename),selected_seeds=seed_details,native_steps=steps,native_cases=len(report['cases'])))
    checks['fixed_model']=digest==E.model_digest_except_dt(model,np);checks['no_neural_execution']='torch' not in sys.modules
    C.need(all(checks.values()),'Migration saved replay failed')
    out.mkdir(parents=True);answer=dict(schema='supermimic_research.patch_migration_saved_review:1',auditor=file_pin(__file__),checks=checks,passed=sum(checks.values()),total=len(checks),counts=totals,details=details,
        scope='Archived/current executed source bytes verified without importing archives; selected restored/full final geometry recomputed with current checker, one candidate native reset replayed, saved controls/final COM-angle/physical gates and missing tails checked. No optimization or integration; not an independent native solver.')
    C.write(out/'report.json',answer);print(json.dumps(dict(report=file_pin(out/'report.json'),passed=answer['passed'],total=answer['total'],counts=totals)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('config','result','terminal','zero-report','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--reports',type=Path,nargs='+',required=True);review(p.parse_args())
