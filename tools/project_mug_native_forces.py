"""Fixed-object inverse/forward native-force projection and physical readback."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.native_projection import InversePoseProjection,ForwardForceProjection,search_projection
from supermimic_research.mechanics.component_preload import engage_pairs
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_holding import evaluate_hold
from supermimic_research.mechanics.patch_intent import match_patches
from supermimic_research.runtime.native_evaluation import DTS,GATES


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,parent,patches,out=map(lambda p:p.resolve(),[args.reference,args.parent,args.patches,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());prior=json.loads((parent/'report.json').read_text());row=prior['results'][0]
    if not row['geometry_pass'] or not row['native_eligible']:raise ValueError('Qualified native6D source-conditioned parent required')
    q0=np.load(verify_file_pin(row['candidate']))['qpos27'];case=next(c for c in row['native_cases'] if c['control']=='inverse_PD_strict_LP' and c['dt_s']==DTS[0]);u0=np.load(verify_file_pin(case['trace']))['ctrl20']
    pr=json.loads((patches/'report.json').read_text());raw=np.load(verify_file_pin(pr['output']));p=raw['source_points_BODY_m'];inward=-raw['outward_source_normals_BODY'];pg=raw['groups']
    known=np.zeros(len(p),bool);known[row['selection']['source_candidate_ids']]=True
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));model.opt.timestep=DTS[0]
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]};gid_to_group={g:i for i,gs in groups.items() for g in gs}
    parts=set(bind['mug_part_geom_ids']);hand=[model.geom(n).id for n in bind['hand_collision_names']]
    audit=CompoundCollisionAudit(model,sorted(parts),hand,model.geom('mug_task_table').id,groups)
    before=json.loads(verify_file_pin(row['geometry']).read_text());pairs=[min([r for r in before['rows'] if r['kind']=='object' and r['geom_ids'][1] in gs and r.get('gap_m') is not None],key=lambda r:r['gap_m'])['geom_ids'] for gs in groups.values()]
    q_open,opening=engage_pairs(model,q0,audit,pairs,-.00002)
    sources=[file_pin(ROOT/path) for path in ['tools/project_mug_native_forces.py','src/supermimic_research/mechanics/native_projection.py',
        'src/supermimic_research/mechanics/force_initialization.py','src/supermimic_research/mechanics/component_preload.py',
        'src/supermimic_research/mechanics/compound_contacts.py','src/supermimic_research/mechanics/compound_collision.py',
        'src/supermimic_research/mechanics/collision.py','src/supermimic_research/mechanics/convex.py',
        'src/supermimic_research/mechanics/compound_holding.py','src/supermimic_research/mechanics/patch_intent.py']]
    reuse=args.reuse_inverse.resolve() if args.reuse_inverse else None;reuse_pins=[]
    if reuse is not None:
        previous=json.loads((reuse/'intent.json').read_text())
        for pin in previous['sources']:
            if Path(pin['path']).name in ['native_projection.py','project_mug_native_forces.py']:
                snapshot=reuse/'source_snapshot'/Path(pin['path']).name
                if file_pin(snapshot)['sha256']!=pin['sha256']:raise ValueError('Lost executed inverse source')
                reuse_pins.append(file_pin(snapshot))
            else:verify_file_pin(pin)
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_native_force_projection_intent:2',sources=sources,
        reuse_inverse=None if reuse is None else file_pin(reuse/'intent.json'),reuse_executed_sources=reuse_pins,
        binding=file_pin(reference/'binding.json'),model=bind['model'],parent=file_pin(parent/'report.json'),initial=row['candidate'],PD_seed_trace=case['trace'],patches=file_pin(patches/'report.json'),
        source_candidate_ids=row['selection']['source_candidate_ids'],object_and_robot_root_fixed=True,gravity_and_material_unchanged=True,
        inverse_seeds=['parent50um','opened20um'],inverse_scaled_secants=[.0001,.0003],forward_seeds=['parent_with_LP_PD','best_legal_inverse_by_actual_generalized_residual'],
        maximum_projections=4,q_axis_radius_rad=.02,u_axis_radius_rad=.3,fit_dt_s=DTS[0],
        maximum_calls_per_projection=2400,maximum_seconds_per_projection=35.,maximum_optimizer_nfev=100,
        joint_and_control_hard_bounds='Actual ranges; no artificial interior margin or target projection',
        residual_scales=dict(joint_Nm=.01,root_force_N=.05,root_BODY_torque_Nm=.003,inverse_control_violation_rad=.02,inverse_torque_violation_Nm=.01),
        minimum_finger_force_N=.2,penetration_soft_target_m=.000075,penetration_scale_m=.00002,
        physical_eligibility='Fresh actual forward, raw legal command, full3025 geometry,chosen candidate-region/normal match and3Fn>0.1N; inverse force itself never applied',
        static_balance_diagnostic=dict(net_COM_force_max_N=.01,net_COM_torque_max_Nm=.0001,max_joint_acceleration_rad_s2=.01),
        native_gates=GATES,dts=DTS,maximum_native_cases=8,maximum_integrations=76800,neural_updates=0))
    write(out/'opening.json',opening);np.savez_compressed(out/'inputs.npz',qpos27=q0,ctrl20=u0,opened_qpos27=q_open)
    results=[];warm=[];steps=0;tick=time.perf_counter()
    recipes=[('inverse_parent',q0,None,.0001),('inverse_opened',q_open,None,.0003),('forward_parent',q0,u0,.0001),('forward_warm',None,None,.0001)]
    for index,(name,q,u,secant) in enumerate(recipes):
        if reuse is not None and name.startswith('inverse'):
            old=json.loads((reuse/name/'result.json').read_text())
            for key in ['projected_state','actual_forward','geometry','solver','proposals']:
                if old[key] is not None:verify_file_pin(old[key])
            old['reused_from']=file_pin(reuse/name/'result.json');results.append(old)
            if old['raw_control_legal']:
                state=np.load(verify_file_pin(old['actual_forward']));warm.append((old['actual_scaled_generalized_cost'],state['qpos27'],state['ctrl20'],name))
            print(json.dumps(dict(name=name,reused=True,new_integrations=0)),flush=True);continue
        if name=='forward_warm':
            if not warm:
                results.append(dict(name=name,skipped=True,reason='No raw legal inverse command'));continue
            _,q,u,warm_source=min(warm,key=lambda r:r[0]);warm_reference=warm_source
        else:warm_reference=None
        model.opt.timestep=DTS[0];engine=InversePoseProjection(model,q,bind,groups) if name.startswith('inverse') else ForwardForceProjection(model,q,u,bind,groups)
        folder=out/name;folder.mkdir()
        with (folder/'proposals.jsonl').open('w',encoding='utf8') as log:solution=search_projection(engine,log,maximum_calls=2400,maximum_seconds=35.,secant_step=secant)
        final=solution.pop('final');candidate=final['qpos27'];command=final['ctrl20'];legal=bool(np.all(command>=model.actuator_ctrlrange[:,0]) and np.all(command<=model.actuator_ctrlrange[:,1]))
        if not np.array_equal(candidate[20:],q0[20:]):raise ValueError('Fixed object moved')
        data=mujoco.MjData(model);data.qpos[:]=candidate;mujoco.mj_kinematics(model,data);geometry=audit.evaluate(data);write(folder/'geometry.json',geometry)
        np.savez_compressed(folder/'projected_state.npz',**{k:v for k,v in final.items() if isinstance(v,np.ndarray)},scaled_z=np.array(solution['best']['z']))
        write(folder/'solver.json',solution);actual=None;matched=None;native_cases=[];score=None;balanced=False
        if legal:
            readback=ForwardForceProjection(model,candidate,command,bind,groups);actual=readback.state(np.zeros(40));data=readback.data
            np.savez_compressed(folder/'actual_forward.npz',**{k:v for k,v in actual.items() if isinstance(v,np.ndarray)})
            rp=[];rn=[];rg=[];R=data.xmat[bind['mug_body_id']].reshape(3,3);origin=data.xpos[bind['mug_body_id']]
            for ci in range(data.ncon):
                con=data.contact[ci];a,b=map(int,con.geom);oa,ob=a in parts,b in parts
                if oa==ob or con.efc_address<0:continue
                other=a if ob else b
                if other not in gid_to_group:continue
                force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
                if force[0]<=1e-8:continue
                rp.append(R.T@(con.pos-origin));rn.append(R.T@((1 if ob else -1)*con.frame.reshape(3,3)[0]));rg.append(gid_to_group[other])
            matched=match_patches(np.array(rp).reshape(-1,3),np.array(rn).reshape(-1,3),rg,p,inward,pg,known)
            net=actual['net_generalized26'];score=float(np.sum((net[:20]/.01)**2)+np.sum((net[20:23]/.05)**2)+np.sum((net[23:]/.003)**2))
            F=actual['net_object_COM_wrench_world'];balanced=bool(abs(F[:3]).max()<=.01 and abs(F[3:]).max()<=.0001 and abs(actual['qacc26'][:20]).max()<=.01 and abs(actual['mass_balance_gap26']).max()<=1e-7)
            tight=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));tight.opt.timestep=DTS[0];tight.opt.tolerance=1e-10
            tight_state=ForwardForceProjection(tight,candidate,command,bind,groups).state(np.zeros(40))
            write(folder/'solver_tolerance_probe.json',dict(nominal_tolerance=float(model.opt.tolerance),probe_tolerance=float(tight.opt.tolerance),probe_only=True,no_integration=True,
                nominal_max_Ma_net_gap=float(abs(actual['mass_balance_gap26']).max()),tight_max_Ma_net_gap=float(abs(tight_state['mass_balance_gap26']).max()),
                max_qacc_difference=float(abs(actual['qacc26']-tight_state['qacc26']).max()),max_net_generalized_difference=float(abs(actual['net_generalized26']-tight_state['net_generalized26']).max())))
            if name.startswith('inverse'):warm.append((score,candidate.copy(),command.copy(),name))
            if geometry['passed'] and matched['all_groups_matched'] and np.all(actual['normal_by_finger_N'][:3]>.1):
                for dt in DTS:
                    case_folder=folder/str(dt);case_result=evaluate_hold(model,candidate,command,bind,groups,dt,case_folder);steps+=case_result['completed_steps']
                    native_cases.append(dict(dt_s=dt,summary=case_result,case=file_pin(case_folder/'report.json'),trace=file_pin(case_folder/'trace.npz')))
        record=dict(name=name,warm_reference=warm_reference,skipped=False,mode='INVERSE_REQUIRED_NOT_APPLIED' if name.startswith('inverse') else 'ACTUAL_NATIVE_FORWARD',
            projected_state=file_pin(folder/'projected_state.npz'),geometry=file_pin(folder/'geometry.json'),geometry_pass=geometry['passed'],
            geometry_failed_pairs=sum(not r['passed'] for r in geometry['rows']),unknown_geometry=geometry['unknown'],raw_control_legal=legal,
            actual_forward=file_pin(folder/'actual_forward.npz') if actual is not None else None,actual_scaled_generalized_cost=score,static_balance_diagnostic_pass=balanced,
            tolerance_probe=file_pin(folder/'solver_tolerance_probe.json') if actual is not None else None,
            actual_COM_net_wrench=None if actual is None else actual['net_object_COM_wrench_world'],actual_joint_accel_max=None if actual is None else float(abs(actual['qacc26'][:20]).max()),
            actual_finger_normal_N=None if actual is None else actual['normal_by_finger_N'][:3],actual_source_candidate_matches=matched,
            source_object_qpos_unchanged=True,solver=file_pin(folder/'solver.json'),proposals=file_pin(folder/'proposals.jsonl'),calls=solution['calls'],
            secant_columns=solution['secant_columns'],changed_contact_signature_columns=solution['changed_contact_signature_columns'],native_cases=native_cases)
        write(folder/'result.json',record);results.append(record)
        print(json.dumps(dict(name=name,geometry=geometry['passed'],legal=legal,balanced=balanced,actual_cost=score,wrench=record['actual_COM_net_wrench'],calls=solution['calls'],native_cases=[(c['dt_s'],c['summary']['actual_duration_s'],c['summary']['native_holding_gates_pass']) for c in native_cases],elapsed_s=time.perf_counter()-tick),default=lambda x:x.tolist()),flush=True)
    reused_steps=sum(c['summary']['completed_steps'] for r in results if 'reused_from' in r for c in r['native_cases'])
    if steps+reused_steps>76800:raise ValueError('Declared physical integration budget exceeded')
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_fixed_object_native_force_projection:1',sources=sources,inputs=[file_pin(reference/'binding.json'),file_pin(parent/'report.json'),file_pin(patches/'report.json')],
        results=results,projection_count=sum(not r['skipped'] for r in results),geometry_pass_count=sum(r.get('geometry_pass',False) for r in results),
        static_balance_count=sum(r.get('static_balance_diagnostic_pass',False) for r in results),native_cases=sum(len(r.get('native_cases',[])) for r in results),integration_steps=steps,reused_integration_steps=reused_steps,total_unique_integration_steps=steps+reused_steps,neural_updates=0,wall_s=time.perf_counter()-tick,
        scope='Bounded fixed-object native inverse/forward pose/control searches, fresh actual legal-command readback and qualified no-aid holding. Inverse-required root forces are never applied. Neither solver termination nor local state balance proves robust holding, calibrated source dynamics, contact truth, physical WM gradients or full ego trajectory success.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass=report['geometry_pass_count'],balanced=report['static_balance_count'],native_cases=report['native_cases'],integrations=steps)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','parent','patches','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reuse-inverse',type=Path)
    main(p.parse_args())
