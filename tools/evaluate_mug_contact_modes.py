"""Matched material-mode ablation: same pose/command, separately pinned models."""
from pathlib import Path
import argparse,json,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.contact_modes import differences,enable_pad_spin,contact_wrench_basis
from supermimic_research.mechanics.compound_contacts import CompoundContacts
from supermimic_research.mechanics.compound_holding import evaluate_hold
from supermimic_research.mechanics.support import torque_box,static_feasibility,maximum_support
from supermimic_research.runtime.native_evaluation import GATES,DTS,PAIR_GATES


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,out=args.reference.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());model_path=verify_file_pin(bind['model'])
    parentA=ROOT/'runs/mug_equilibrium_v1';A=json.loads((parentA/'report.json').read_text())['results'][0]
    stateA=np.load(verify_file_pin(A['state']));geoA=json.loads(verify_file_pin(A['geometry']).read_text())
    parentB=ROOT/'runs/mug_gravity_engagement_v2';B=json.loads((parentB/'report.json').read_text())['results'][1]
    stateB=np.load(verify_file_pin(B['raw']));geoB=json.loads(verify_file_pin(B['geometry']).read_text())
    if not geoA['passed'] or not geoB['passed']:raise ValueError('Both frozen poses must have full geometry qualification')
    seeds=[dict(name='local_force_seed0',q=stateA['qpos27'].copy(),u=stateA['ctrl20'].copy(),
                pose=A['state'],geometry=A['geometry'],original_baselines=A['native_cases'],conditional_Z_up=False),
           dict(name='conditional_Z_up_preload50um',q=stateB['qpos27'].copy(),u=stateB['ctrl20'].copy(),
                pose=B['raw'],geometry=B['geometry'],original_baselines=None,conditional_Z_up=True)]
    specs=[dict(name='point3',dim=3,spin_m=None),dict(name='spin4_1mm',dim=4,spin_m=.001),dict(name='spin4_3mm',dim=4,spin_m=.003)]
    sources=[file_pin(p) for p in [Path(__file__),ROOT/'src/supermimic_research/mechanics/contact_modes.py',
        ROOT/'src/supermimic_research/mechanics/compound_contacts.py',ROOT/'src/supermimic_research/mechanics/compound_holding.py',
        ROOT/'src/supermimic_research/mechanics/support.py',ROOT/'src/supermimic_research/runtime/native_evaluation.py']]
    model=mujoco.MjModel.from_binary_path(str(model_path))
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in range(1,4)}
    pads={g for values in groups.values() for g in values};allowed_differences=['model.pair_dim','model.pair_friction']
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_matched_contact_modes_intent:1',sources=sources,
        baseline_model=bind['model'],seeds=[{k:v for k,v in s.items() if k not in ['q','u','original_baselines']} for s in seeds],
        model_specs=specs,spin_units='metres, length-valued torsional friction coefficient; exploratory geometry-scale values, not calibrated patch sizes',
        measured=False,spin_value_selection='predeclared1mm and3mm; no automatic enabling of unused baseline .005 coefficient',
        changed_pairs='128 mug components x9 designated distal hand geoms; table/other/self contacts unchanged',
        allowed_model_differences=allowed_differences,matched_pose_and_constant_controls=True,
        no_control_or_pose_optimization=True,numerical_gates=GATES,dt_pair_gates=PAIR_GATES,dts=DTS,
        maximum_cases=12,maximum_integrations=115200,expected_static_forward=12,maximum_LP_attempts=48,
        baseline_A_raw_trace_byte_parity_before_variants=True,role='MATERIAL_PRIOR_SENSITIVITY_NOT_POLICY_BENEFIT',neural_updates=0))
    # Persist all variants independently; source MJB remains untouched.
    models=[]
    for spec in specs:
        folder=out/spec['name'];folder.mkdir();variant=mujoco.MjModel.from_binary_path(str(model_path))
        changed=[]
        if spec['dim']==4:changed=enable_pad_spin(variant,bind['mug_part_geom_ids'],pads,spec['spin_m'])
        diff=differences(model,variant)
        if diff!=(allowed_differences if spec['dim']==4 else []):raise ValueError('Unexpected physical model differences')
        if spec['dim']==3:shutil.copyfile(model_path,folder/'model.mjb')
        else:mujoco.mj_saveModel(variant,str(folder/'model.mjb'))
        loaded=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
        if differences(variant,loaded):raise ValueError('Model roundtrip changed physical arrays/options')
        record=dict(**spec,model=file_pin(folder/'model.mjb'),changed_pair_count=len(changed),differences=diff,
            robot_geometry_inertia_limits_gains_wrist_unchanged=True,table_and_nonintent_contact_pairs_unchanged=True)
        write(folder/'variant.json',record);models.append(record)
    cases=[];barrier=[];tick=time.perf_counter();static_count=0;lp_attempts=0
    for spec in models:
        for seed_index,seed in enumerate(seeds):
            for dt in DTS:
                folder=out/spec['name']/seed['name']/str(dt);folder.parent.mkdir(parents=True,exist_ok=True)
                active=mujoco.MjModel.from_binary_path(str(verify_file_pin(spec['model'])));active.opt.timestep=dt
                data=mujoco.MjData(active);data.qpos[:]=seed['q'];data.qvel[:]=0;data.ctrl[:]=seed['u'];mujoco.mj_forward(active,data);static_count+=1
                if np.any(data.warning.number) or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):raise ValueError('Static material mode warning or aid')
                basis=contact_wrench_basis(active,data,bind['mug_part_geom_ids'],bind['mug_body_id'],pads)
                reader=CompoundContacts(active,bind['mug_part_geom_ids'],bind['mug_body_id'],groups,active.geom('mug_task_table').id)
                snapshot=reader.read(data,raw=True);mujoco.mj_rnePostConstraint(active,data)
                external_error=float(np.abs(np.r_[snapshot['torque_about_COM_world_Nm'],snapshot['force_world_N']]-data.cfrc_ext[bind['mug_body_id']]).max())
                if external_error>1e-8:raise ValueError('Actual couple/COM aggregation failed')
                W,L=basis['wrench'],basis['loads'];target=np.r_[-active.body_mass[bind['mug_body_id']]*active.opt.gravity,np.zeros(3)]
                offset=data.qfrc_bias[:20]-data.qfrc_passive[:20];friction=active.dof_frictionloss[:20]
                low,high=torque_box(active,data,active.actuator_ctrlrange[:,0],active.actuator_ctrlrange[:,1])
                fixed_low,fixed_high=torque_box(active,data,seed['u'],seed['u'])
                cone=static_feasibility(W,target)
                fixed=static_feasibility(W,target,loads=L,offset=offset,lower=fixed_low,upper=fixed_high,friction=friction)
                full=static_feasibility(W,target,loads=L,offset=offset,lower=low,upper=high,friction=friction)
                maximum=maximum_support(W,L,offset,low,high,friction);lp_attempts+=4
                static=dict(contacts=basis['contacts'],native_EFC_unit_queries=basis['native_EFC_unit_queries'],
                    native_wrench_column_error=basis['native_wrench_error'],native_joint_load_column_error=basis['native_load_error'],
                    actual_external_wrench_error=external_error,cone6D=cone,fixed_command=fixed,full_command=full,maximum_support=maximum,
                    current_finger_normal_N=snapshot['normal_by_finger_N'][:3],readonly=basis['readonly'])
                result=evaluate_hold(active,seed['q'],seed['u'],bind,groups,dt,folder)
                write(folder/'static.json',static)
                np.savez_compressed(folder/'basis.npz',wrench6xN=W,joint_load20xN=L,native_wrench=basis['native_wrench'],native_loads=basis['native_loads'],
                    target6=target,offset20=offset,dry_friction20=friction,torque_lower20=low,torque_upper20=high,
                    fixed_torque_lower20=fixed_low,fixed_torque_upper20=fixed_high)
                if spec['name']=='point3' and seed_index==0:
                    old=next(c for c in seed['original_baselines'] if c['summary']['dt_s']==dt)
                    original=np.load(verify_file_pin(old['trace']));new=np.load(folder/'trace.npz')
                    parity={k:original[k].dtype==new[k].dtype and original[k].shape==new[k].shape and original[k].tobytes()==new[k].tobytes() for k in original.files}
                    passed=set(original.files)==set(new.files) and all(parity.values())
                    barrier.append(dict(dt_s=dt,passed=passed,arrays=parity,reference=old['trace'],current=file_pin(folder/'trace.npz')))
                    write(out/'baseline_barrier.json',barrier)
                    if not passed:raise ValueError('Baseline raw trace parity barrier failed; no variant integration allowed')
                if spec['name']!='point3' and (len(barrier)!=2 or not all(x['passed'] for x in barrier)):
                    raise ValueError('Both baseline dt barriers required before material variant')
                cases.append(dict(mode=spec['name'],seed=seed['name'],dt_s=dt,model=spec['model'],initial=seed['pose'],geometry=seed['geometry'],
                    case=file_pin(folder/'report.json'),trace=file_pin(folder/'trace.npz'),static=file_pin(folder/'static.json'),basis=file_pin(folder/'basis.npz'),summary=result))
                print(json.dumps(dict(mode=spec['name'],seed=seed['name'],dt=dt,duration_s=result['actual_duration_s'],termination=result['termination'],
                    native_pass=result['native_holding_gates_pass'],full_static=full['success'],maximum_support_N=maximum['support_weight_N'],elapsed_s=time.perf_counter()-tick)),flush=True)
    pairs=[]
    for spec in models:
        for seed in seeds:
            selected=[c for c in cases if c['mode']==spec['name'] and c['seed']==seed['name']];ends=[np.load(c['trace']['path']) for c in selected]
            distance=float(np.linalg.norm(ends[0]['post_COM'][-1]-ends[1]['post_COM'][-1]));angle=float(2*np.arccos(np.clip(abs(ends[0]['post_quat'][-1]@ends[1]['post_quat'][-1]),0,1)))
            complete=all(c['summary']['gates']['duration_complete'] for c in selected)
            consistent=complete and distance<=.002 and angle<=PAIR_GATES['final_orientation_difference_rad']
            pairs.append(dict(mode=spec['name'],seed=seed['name'],both_complete=complete,COM_endpoint_difference_m=distance,
                orientation_endpoint_difference_rad=angle,dt_pair_consistency_pass=consistent,
                qualified=bool(consistent and all(c['summary']['native_holding_gates_pass'] for c in selected))))
    steps=sum(c['summary']['completed_steps'] for c in cases)
    if len(cases)!=12 or static_count!=12 or lp_attempts!=48 or steps>115200:raise ValueError('Declared study budget mismatch')
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_matched_contact_modes_evaluation:1',sources=sources,models=models,cases=cases,dt_pairs=pairs,
        baseline_barrier=barrier,qualified_pair_count=sum(p['qualified'] for p in pairs),integration_steps=steps,
        static_forward=static_count,LP_attempts=lp_attempts,linprog_invocations=sum(4 for c in cases if json.loads(Path(c['static']['path']).read_text())['contacts']),
        neural_updates=0,wall_s=time.perf_counter()-tick,
        scope='Same fixed poses and constant legal PD targets, unchanged geometry/robot/inertia and separately pinned explicit contact-mode priors. Not calibrated materials, policy improvement, original source trajectory replay or verified sequence gravity. Positive holding under a changed prior is not algorithm benefit.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),qualified_pairs=report['qualified_pair_count'],steps=steps)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
