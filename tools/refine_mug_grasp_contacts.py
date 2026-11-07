"""Contact-aware static proposals, followed by independent all-piece audit."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.contact_ik import ContactAwareIK
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,parent,out=args.reference.resolve(),args.parent.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((reference/'report.json').read_text());binding=json.loads((reference/'binding.json').read_text())
    previous=json.loads((parent/'report.json').read_text())
    for pin in report['outputs'].values():verify_file_pin(pin)
    for result in previous['results']:verify_file_pin(result['candidate']);verify_file_pin(result['geometry'])
    z=np.load(reference/'reference61.npz');model=mujoco.MjModel.from_binary_path(str(reference/'hand2_mug.mjb'))
    pose=z['T_robot_world_from_object_BODY'][0];targets=z['hand_landmarks21_object_BODY_F64'][0,:5]@pose[:3,:3].T+pose[:3,3]
    hand=[model.geom(n).id for n in binding['hand_collision_names']]
    permitted={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in range(1,4)}
    audit=CompoundCollisionAudit(model,binding['mug_part_geom_ids'],hand,model.geom('mug_task_table').id,permitted)
    best=previous['best_kinematic_cost_seed'];fitted=np.load(parent/f'seed{best:02d}'/'candidate.npz')['qpos27'][:20]
    starts=[fitted,np.zeros(20)]
    sources=[file_pin(p) for p in [Path(__file__),ROOT/'src/supermimic_research/mechanics/contact_ik.py',
        ROOT/'src/supermimic_research/mechanics/retargeting.py',ROOT/'src/supermimic_research/mechanics/compound_collision.py',
        ROOT/'src/supermimic_research/mechanics/collision.py',ROOT/'src/supermimic_research/mechanics/convex.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_contact_ik_intent:1',sources=sources,
        parent=file_pin(parent/'report.json'),reference=file_pin(reference/'report.json'),starts=starts,
        max_nfev_each=160,maximum_seconds_each=45,tip_weights=[1,1,1,.25,.25],tip_scale_m=.02,regularization=.003,
        penalty_scale_m=.0005,object_intent_floor_m=.00005,nonintent_self_table_floor_m=.0002,
        canonical_contact_attraction_upper_target_m=.0001,attraction_scale_m=.002,
        independent_qualification_band_m=[-.0001,.0002],nonintent_self_table_qualification_min_m=0,
        native_distance_limit_object_m=.03,native_distance_limit_self_m=.001,
        source_gravity_preserved=False,neural_updates=0,integration_steps=0,
        scope='Native pair distances/witness Jacobians used as proposal heuristics only; final independent convex component audit, unknown never passes. No learned model or force labels.'))
    results=[];tick=time.perf_counter()
    for i,start in enumerate(starts):
        folder=out/f'seed{i:02d}';folder.mkdir();engine=ContactAwareIK(model,z['initial_qpos'],targets,audit)
        with (folder/'proposals.jsonl').open('x',encoding='utf8') as log:
            result=engine.solve_contact(start,log,max_nfev=160,maximum_seconds=45.)
        data=engine.data;data.qpos[:]=engine.base;data.qpos[:20]=result['q20'];mujoco.mj_kinematics(model,data)
        geometry=audit.evaluate(data);write(folder/'geometry.json',geometry)
        failed=[r for r in geometry['rows'] if not r['passed']]
        result.update(geometry_pass=geometry['passed'],geometry_unknown=geometry['unknown'],
            canonical_contact_candidates=geometry['canonical_contact_candidates'],
            failed_pair_counts={kind:sum(r['kind']==kind for r in failed) for kind in ['object','self','hand_table']},
            minimum_failed_convex_gap_m=min((r.get('gap_m',0) for r in failed),default=None))
        np.savez_compressed(folder/'candidate.npz',qpos27=data.qpos.copy(),ctrl20=result['q20'],targets_world=targets,
                            tip_errors_m=result['tip_errors_m'])
        result.update(candidate=file_pin(folder/'candidate.npz'),geometry=file_pin(folder/'geometry.json'),proposals=file_pin(folder/'proposals.jsonl'))
        write(folder/'result.json',result);results.append(result)
        print(json.dumps(dict(seed=i,geometry_pass=geometry['passed'],failed=result['failed_pair_counts'],
            minimum_failed_gap_m=result['minimum_failed_convex_gap_m'],tip_errors_m=result['tip_errors_m'].tolist(),
            native=result['best'],elapsed_s=time.perf_counter()-tick)),flush=True)
    final=dict(schema='mug_contact_ik_proposals:1',sources=sources,parent=file_pin(parent/'report.json'),results=results,
        geometry_pass_count=sum(r['geometry_pass'] for r in results),neural_updates=0,integration_steps=0,
        physical_holding_qualified=False,wall_s=time.perf_counter()-tick,
        scope='Two bounded first-frame collision-aware proposals. Native witness gradients are optimization aids, not verified physical gradients, union SDFs or global clearance. All-piece independent qualification and future no-aid dynamics remain necessary.')
    for pin in sources:verify_file_pin(pin)
    write(out/'report.json',final);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass_count=final['geometry_pass_count'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
