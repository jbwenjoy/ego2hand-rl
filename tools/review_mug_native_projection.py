"""Verify actual native projections, source snapshots and twelve failed traces."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.native_projection import InversePoseProjection,ForwardForceProjection
from supermimic_research.mechanics.acceleration_projection import AccelerationProjection,zero_velocity_COM_acceleration
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.contact_modes import differences
from review_mug_support_trials import check_case


def main(args):
    import mujoco
    reference,projection,acceleration,out=map(lambda p:p.resolve(),[args.reference,args.projection,args.acceleration,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((projection/'report.json').read_text());response=json.loads((acceleration/'report.json').read_text());intent=json.loads((projection/'intent.json').read_text());bind=json.loads((reference/'binding.json').read_text())
    for source_report in [report,response]:
        for pin in source_report['sources']:verify_file_pin(pin)
    legacy=Path(intent['reuse_inverse']['path']).parent;original=json.loads((legacy/'intent.json').read_text());snapshots=[]
    for pin in original['sources']:
        if Path(pin['path']).name in ['native_projection.py','project_mug_native_forces.py']:
            snap=legacy/'source_snapshot'/Path(pin['path']).name
            if file_pin(snap)['sha256']!=pin['sha256']:raise ValueError('Lost inverse executed source')
            snapshots.append(file_pin(snap))
        else:verify_file_pin(pin)
    inputs=np.load(legacy/'inputs.npz');q0,u0=inputs['qpos27'],inputs['ctrl20']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));model.opt.timestep=.0003125
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]};gids={g:i-1 for i,gs in groups.items() for g in gs}
    hand=[model.geom(n).id for n in bind['hand_collision_names']];audit=CompoundCollisionAudit(model,bind['mug_part_geom_ids'],hand,model.geom('mug_task_table').id,groups)
    nominal=ForwardForceProjection(model,q0,u0,bind,groups).state(np.zeros(40));tight=mujoco.MjModel.from_binary_path(str(bind['model']['path']));tight.opt.timestep=.0003125;tight.opt.tolerance=1e-10
    strict=ForwardForceProjection(tight,q0,u0,bind,groups).state(np.zeros(40));solver_probe=dict(allowed_model_differences=differences(model,tight),nominal_gap=float(abs(nominal['mass_balance_gap26']).max()),tight_gap=float(abs(strict['mass_balance_gap26']).max()),
        max_qacc_difference=float(abs(nominal['qacc26']-strict['qacc26']).max()),nominal_iterations=nominal['solver_niter'].tolist(),tight_iterations=strict['solver_niter'].tolist(),no_integration=True)
    rows=[];traces=[];total_steps=0;proposal_calls=0;fingerprint_changes=0
    by_name={r['name']:r for r in report['results']}
    all_rows=[('force',r) for r in report['results']]+[('response',r) for r in response['results']]
    for kind,row in all_rows:
        model.opt.timestep=.0003125
        if kind=='force':
            actual=np.load(verify_file_pin(row['actual_forward']));projected=np.load(verify_file_pin(row['projected_state']));name=row['name']
            if name.startswith('inverse'):
                base=inputs['opened_qpos27'] if name=='inverse_opened' else q0;engine=InversePoseProjection(model,base,bind,groups)
                rebuilt=engine.state(projected['scaled_z'])
                if np.max(abs(rebuilt['inverse_required26']-projected['inverse_required26']))>1e-8 or not np.array_equal(rebuilt['qpos27'],projected['qpos27']):raise ValueError('Inverse result not reproducible')
            else:
                if name=='forward_parent':base,control=q0,u0
                else:
                    warm=np.load(verify_file_pin(by_name[row['warm_reference']]['actual_forward']));base,control=warm['qpos27'],warm['ctrl20']
                engine=ForwardForceProjection(model,base,control,bind,groups)
        else:
            actual=np.load(verify_file_pin(row['state']));parent=np.load(verify_file_pin(row['parent_state']));engine=AccelerationProjection(model,parent['qpos27'],parent['ctrl20'],bind,groups);name=f'response{row["seed"]}'
        log=verify_file_pin(row['proposals']);zrows=[json.loads(line) for line in log.read_text().splitlines()];proposal_calls+=len(zrows)
        for item in zrows:
            zz=np.array(item['z'])
            if zz.shape!=engine.lo.shape or np.any(zz<engine.lo-1e-12) or np.any(zz>engine.hi+1e-12):raise ValueError('Unbounded proposal probe')
        solver=json.loads(verify_file_pin(row['solver']).read_text());fingerprint_changes+=solver['changed_contact_signature_columns']
        fresh=ForwardForceProjection(model,actual['qpos27'],actual['ctrl20'],bind,groups).state(np.zeros(40));error=max(float(np.max(abs(fresh[key]-actual[key]))) for key in ['qpos27','ctrl20','qacc26','net_generalized26','normal_by_finger_N','net_object_COM_wrench_world'])
        if error>1e-6 or not np.array_equal(actual['qpos27'][20:],q0[20:]):raise ValueError('Actual nominal readback/object pose mismatch')
        readback=ForwardForceProjection(model,actual['qpos27'],actual['ctrl20'],bind,groups);readback.state(np.zeros(40));geometry=audit.evaluate(readback.data);saved=json.loads(verify_file_pin(row['geometry']).read_text())
        same=geometry['passed']==saved['passed'] and len(geometry['rows'])==len(saved['rows'])==3025
        for a,b in zip(geometry['rows'],saved['rows']):
            same &= a['geom_ids']==b['geom_ids'] and a['passed']==b['passed'] and a['status']==b['status']
            if a.get('gap_m') is not None:same &= abs(a['gap_m']-b['gap_m'])<1e-10
        if not same:raise ValueError('Projection geometry mismatch')
        linear,angular=zero_velocity_COM_acceleration(model,readback.data,bind['mug_body_id'],20)
        if kind=='response' and (np.max(abs(linear-row['COM_acceleration_world']))>1e-6 or np.max(abs(angular-row['angular_acceleration_world']))>1e-6):raise ValueError('Response coordinates mismatch')
        rows.append(dict(name=name,actual_readback_error=error,geometry_pass=geometry['passed'],joint_acceleration_max=float(abs(fresh['qacc26'][:20]).max()),COM_acceleration=linear.tolist(),angular_acceleration=angular.tolist(),
            actual_Ma_net_gap=float(abs(fresh['mass_balance_gap26']).max()),max_q20_change_from_original_rad=float(abs(actual['qpos27'][:20]-q0[:20]).max()),max_ctrl_change_from_original_rad=float(abs(actual['ctrl20']-u0).max())))
        for case in row['native_cases']:
            model.opt.timestep=case['dt_s'];traces.append(check_case(case,actual['qpos27'],actual['ctrl20'],model,bind,gids));total_steps+=case['summary']['completed_steps']
        print(json.dumps(dict(name=name,geometry=True,actual_error=error,trace_cases=len(row['native_cases']))),flush=True)
    checks=dict(executed_inverse_sources_preserved=len(snapshots)==2,nominal_finite_solver_gap=solver_probe['allowed_model_differences']==['option.tolerance'] and solver_probe['nominal_gap']>1e-9 and solver_probe['tight_gap']<1e-7,
        six_native_state_and_geometry_reruns=len(rows)==6,all_probes_bounded=proposal_calls==7105,
        twelve_failed_case_readback=len(traces)==12 and all(all(t['checks'].values()) for t in traces),no_reused_steps_double_counted=report['integration_steps']==6923 and report['reused_integration_steps']==3459 and total_steps==report['total_unique_integration_steps']+response['integration_steps']==17950)
    checks={k:bool(v) for k,v in checks.items()};out.mkdir(parents=True);result=dict(schema='mug_native_projection_saved_review:1',source=file_pin(__file__),inputs=[file_pin(projection/'report.json'),file_pin(acceleration/'report.json')],snapshots=snapshots,
        checks=checks,passed=sum(checks.values()),total=len(checks),rows=rows,baseline_solver_tolerance_probe=solver_probe,traces=traces,
        trace_checks=[sum(v for t in traces for v in t['checks'].values()),sum(len(t['checks']) for t in traces)],proposal_probes_checked=proposal_calls,changed_signature_columns=fingerprint_changes,total_unique_native_integrations=total_steps,
        native_static_readbacks=6,native_geometry_reruns=6,integration_steps=0,neural_updates=0,
        scope='Source snapshots, all bounded evaluated parameters, nominal forward/zero-acc inverse/COM-response and saved failed trace readback. Shared geometry reuse; no exact predicates, global infeasibility, calibrated material, source truth, smooth derivative certificate or dynamic success.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(dict(report=file_pin(out/'report.json'),checks=checks,trace_checks=result['trace_checks'])))
    if not all(checks.values()):raise ValueError('Retain failed native projection review')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','projection','acceleration','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
