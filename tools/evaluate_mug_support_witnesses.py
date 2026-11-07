"""Test legal PD targets derived from static LPs; never apply their force rays."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_holding import evaluate_hold
from supermimic_research.runtime.native_evaluation import GATES,DTS,PAIR_GATES


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,diagnostic,out=args.reference.resolve(),args.diagnostic.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    diagnosis=json.loads((diagnostic/'report.json').read_text());bind=json.loads((reference/'binding.json').read_text())
    candidates=[r for r in diagnosis['records'] if r['cone6D']['success'] and all(d['feasibility']['success'] for d in r['domains'][1:])]
    if not candidates:raise ValueError('No six-dimensional support proposal')
    chosen=candidates[0];verify_file_pin(chosen['raw']);z=np.load(chosen['raw']['path']);q=z['qpos27'].copy()
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    hand=[model.geom(n).id for n in bind['hand_collision_names']]
    permitted={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in range(1,4)}
    audit=CompoundCollisionAudit(model,bind['mug_part_geom_ids'],hand,model.geom('mug_task_table').id,permitted)
    data.qpos[:]=q;mujoco.mj_kinematics(model,data);geometry=audit.evaluate(data)
    if not geometry['passed']:raise ValueError('LP-possible source pose did not pass full geometric qualification')
    targets=[dict(name='original_command_new_pose',u=z['ctrl20'].copy(),role='BASELINE_NEW_STATIC_INIT')]
    for domain in chosen['domains'][1:]:
        tau=np.array(domain['feasibility']['witness']['required_torque20_Nm'])
        u=(tau-z['affine20'])/z['gain20']
        if np.any(u<model.actuator_ctrlrange[:,0]) or np.any(u>model.actuator_ctrlrange[:,1]):
            raise ValueError('Direct inverse PD target outside actual command interval, never clamp')
        targets.append(dict(name=domain['name'],u=u,role='LEGAL_PD_TARGET_FROM_HYPOTHETICAL_LP_TORQUE',
            hypothetical_torque20=tau,maximum_target_change_rad=float(np.max(np.abs(u-z['ctrl20'])))))
    source_q=np.load(reference/'reference61.npz')['initial_qpos']
    deltaR=Rotation.from_quat(q[23:27][[1,2,3,0]])*Rotation.from_quat(source_q[23:27][[1,2,3,0]]).inv()
    deviation=dict(object_root_delta_world_m=q[20:23]-source_q[20:23],object_rotation_delta_world_rotvec_rad=deltaR.as_rotvec(),
        object_rotation_delta_angle_rad=float(deltaR.magnitude()),hand_q20_max_change_rad=float(np.abs(q[:20]-source_q[:20]).max()),
        reference_rebuilt=False,new_initial_pose_is_distinct_task_variant=True,source_gravity_preserved=False)
    sources=[file_pin(p) for p in [Path(__file__),ROOT/'src/supermimic_research/mechanics/support.py',
        ROOT/'src/supermimic_research/mechanics/compound_holding.py',ROOT/'src/supermimic_research/mechanics/compound_contacts.py',
        ROOT/'src/supermimic_research/mechanics/compound_collision.py',ROOT/'src/supermimic_research/mechanics/collision.py',ROOT/'src/supermimic_research/mechanics/convex.py']]
    out.mkdir(parents=True);write(out/'geometry.json',geometry)
    np.savez_compressed(out/'initial_state.npz',qpos27=q,ctrl20=targets[0]['u'])
    write(out/'intent.json',dict(schema='mug_support_witness_native_intent:1',sources=sources,parent=file_pin(diagnostic/'report.json'),
        selection='first declared snapshot with cone6D and both full-command static feasibility',selected_operand=chosen['raw'],
        selected_source_dt_s=chosen['source_dt_s'],selected_source_time_s=chosen['source_time_s'],
        zero_velocity_reinitialization=True,initial=file_pin(out/'initial_state.npz'),geometry=file_pin(out/'geometry.json'),
        targets=targets,deviation=deviation,numerical_gates=GATES,dt_pair_gates=PAIR_GATES,dts=DTS,
        maximum_integrations=57600,duration_s=2.,no_LP_force_or_torque_witness_applied=True,neural_updates=0))
    cases=[];tick=time.perf_counter()
    for index,target in enumerate(targets):
        for dt in DTS:
            folder=out/f'target{index:02d}_dt{dt}';result=evaluate_hold(model,q,target['u'],bind,permitted,dt,folder)
            cases.append(dict(target_index=index,target_name=target['name'],case=file_pin(folder/'report.json'),trace=file_pin(folder/'trace.npz'),summary=result))
            print(json.dumps(dict(target=target['name'],dt=dt,duration_s=result['actual_duration_s'],termination=result['termination'],
                pass_native=result['native_holding_gates_pass'],COM_drift_m=result['metrics']['final_COM_drift_m'],elapsed_s=time.perf_counter()-tick)),flush=True)
    pairs=[]
    for index,target in enumerate(targets):
        selected=[c for c in cases if c['target_index']==index];ends=[np.load(c['trace']['path']) for c in selected]
        delta=float(np.linalg.norm(ends[0]['post_COM'][-1]-ends[1]['post_COM'][-1]));angle=float(2*np.arccos(np.clip(abs(ends[0]['post_quat'][-1]@ends[1]['post_quat'][-1]),0,1)))
        complete=all(c['summary']['gates']['duration_complete'] for c in selected)
        consistent=complete and delta<=.002 and angle<=PAIR_GATES['final_orientation_difference_rad']
        pairs.append(dict(target_name=target['name'],both_complete=complete,dt_pair_consistency_pass=consistent,
            COM_endpoint_difference_m=delta,orientation_endpoint_difference_rad=angle,
            qualified=bool(consistent and all(c['summary']['native_holding_gates_pass'] for c in selected))))
    steps=sum(c['summary']['completed_steps'] for c in cases)
    if steps>57600:raise ValueError('Predeclared integration budget')
    report=dict(schema='mug_support_witness_native_evaluation:1',sources=sources,cases=cases,dt_pairs=pairs,
        qualified_pair_count=sum(p['qualified'] for p in pairs),integration_steps=steps,neural_updates=0,
        initial_state=file_pin(out/'initial_state.npz'),initial_geometry=file_pin(out/'geometry.json'),deviation=deviation,
        wall_s=time.perf_counter()-tick,source_trajectory_replayed=False,source_gravity_preserved=False,
        scope='New zero-velocity initialization from one previously falling PRE pose. Complete component geometry qualified, fixed robot/physics retained. Only legal PD targets are applied, never the hypothetical LP contact force/torque coefficients. New static-hold diagnostic, not original task resume, trajectory correction, policy benefit or independent ego generalization.')
    for pin in sources:verify_file_pin(pin)
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),qualified_pairs=report['qualified_pair_count'],steps=steps)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
