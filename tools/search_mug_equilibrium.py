"""Bounded native pose/PD search, full geometry and no-aid dual-dt tests."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.force_initialization import NativeEquilibrium
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_holding import evaluate_hold
from supermimic_research.runtime.native_evaluation import DTS,GATES


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,parent,out=args.reference.resolve(),args.parent.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());prior=json.loads((parent/'report.json').read_text());intent=json.loads((parent/'intent.json').read_text())
    q=np.load(verify_file_pin(prior['initial_state']))['qpos27'].copy()
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])))
    hand=[model.geom(n).id for n in bind['hand_collision_names']]
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in range(1,4)}
    audit=CompoundCollisionAudit(model,bind['mug_part_geom_ids'],hand,model.geom('mug_task_table').id,groups)
    sources=[file_pin(p) for p in [Path(__file__),ROOT/'src/supermimic_research/mechanics/force_initialization.py',
        ROOT/'src/supermimic_research/mechanics/compound_contacts.py',ROOT/'src/supermimic_research/mechanics/compound_holding.py',
        ROOT/'src/supermimic_research/mechanics/compound_collision.py',ROOT/'src/supermimic_research/mechanics/collision.py',ROOT/'src/supermimic_research/mechanics/convex.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_native_equilibrium_search_intent:1',sources=sources,parent=file_pin(parent/'report.json'),
        initial_state=prior['initial_state'],seed_targets=intent['targets'][1:],fit_dt_s=DTS[0],maximum_calls_per_seed=2400,maximum_seconds_per_seed=35.,
        q20_axis_radius_rad=.03,object_translation_axis_radius_m=.002,object_rotvec_axis_radius_rad=.04,control_axis_radius_rad=.3,
        secant_scaled_step=.001,net_residual_scales=dict(joint_Nm=.01,object_force_N=.05,object_root_BODY_torque_Nm=.003),
        minimum_finger_force_N=.2,force_scale_N=.1,native_penetration_soft_target_m=.000075,penetration_scale_m=.00002,
        patch_gap_band_m=[-.0001,.0002],patch_gap_scale_m=.0001,normal_force_regularizer=.002,scaled_parameter_regularizer=.003,
        independent_geometry_band_m=[-.0001,.0002],nonintent_and_self_minimum_m=0,native_gates=GATES,
        maximum_native_integrations=38400,neural_updates=0,source_gravity_preserved=False,
        scope='Native zero-velocity initial pose/control optimization only. Not neural training, a certified derivative, an applied LP force, a policy resume or corrected ego trajectory.'))
    results=[];tick=time.perf_counter();total_steps=0
    for index,target in enumerate(intent['targets'][1:]):
        folder=out/f'seed{index:02d}';folder.mkdir();model.opt.timestep=DTS[0]
        engine=NativeEquilibrium(model,q,np.array(target['u']),bind,groups)
        with (folder/'proposals.jsonl').open('x',encoding='utf8') as log:result=engine.search(log,maximum_calls=2400,maximum_seconds=35.)
        final=result.pop('final');geometry=audit.evaluate(engine.data);write(folder/'geometry.json',geometry)
        arrays={k:v for k,v in final.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(folder/'final_state.npz',**arrays,z46=np.array(result['best']['z46']))
        result.update(state=file_pin(folder/'final_state.npz'),geometry=file_pin(folder/'geometry.json'),geometry_pass=geometry['passed'],
            geometry_unknown=geometry['unknown'],failed_pair_counts={kind:sum(r['kind']==kind and not r['passed'] for r in geometry['rows']) for kind in ['object','self','hand_table']},
            net_object_COM_wrench_world=final['net_object_COM_wrench_world'],finger_normal_N=final['normal_by_finger_N'][:3],
            proposals=file_pin(folder/'proposals.jsonl'),native_cases=[])
        if geometry['passed']:
            for dt in DTS:
                case_folder=folder/f'dt{dt}';case=evaluate_hold(model,final['qpos27'],final['ctrl20'],bind,groups,dt,case_folder)
                result['native_cases'].append(dict(summary=case,report=file_pin(case_folder/'report.json'),trace=file_pin(case_folder/'trace.npz')));total_steps+=case['completed_steps']
                print(json.dumps(dict(seed=index,dt=dt,duration_s=case['actual_duration_s'],native_pass=case['native_holding_gates_pass'],termination=case['termination'])),flush=True)
        write(folder/'result.json',result);results.append(result)
        print(json.dumps(dict(seed=index,geometry=geometry['passed'],failed=result['failed_pair_counts'],cost=result['fresh_final_squared_cost'],
            force_wrench_world=final['net_object_COM_wrench_world'].tolist(),finger_N=result['finger_normal_N'].tolist(),calls=result['calls'],elapsed_s=time.perf_counter()-tick)),flush=True)
    if total_steps>38400:raise ValueError('Declared native budget exceeded')
    report=dict(schema='mug_native_equilibrium_search:1',sources=sources,parent=file_pin(parent/'report.json'),results=results,
        geometry_pass_count=sum(r['geometry_pass'] for r in results),integration_steps=total_steps,neural_updates=0,
        wall_s=time.perf_counter()-tick,source_gravity_preserved=False,
        scope='Two bounded native equilibrium proposals and only geometry-qualified dual-dt physical tests. Native static balance is not continuous clearance, robust dynamic stability or source-trajectory/policy benefit.')
    for pin in sources:verify_file_pin(pin)
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass_count=report['geometry_pass_count'],steps=total_steps)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
