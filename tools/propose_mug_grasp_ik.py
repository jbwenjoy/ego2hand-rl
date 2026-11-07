"""Fixed-budget kinematic seeds and exhaustive compound collision qualification."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.retargeting import HandTipIK
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    tick=time.perf_counter();reference=args.reference.resolve();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((reference/'report.json').read_text());binding=json.loads((reference/'binding.json').read_text())
    for pin in report['outputs'].values():verify_file_pin(pin)
    z=np.load(reference/'reference61.npz');model=mujoco.MjModel.from_binary_path(str(reference/'hand2_mug.mjb'))
    pose=z['T_robot_world_from_object_BODY'][0];targets=z['hand_landmarks21_object_BODY_F64'][0,:5]@pose[:3,:3].T+pose[:3,3]
    engine=HandTipIK(model,z['initial_qpos'],targets,weights=[1,1,1,.25,.25],regularization=.003)
    starts=[np.zeros(20),(engine.lo+engine.hi)/2,engine.lo+.65*(engine.hi-engine.lo)]
    sources=[file_pin(p) for p in [Path(__file__),ROOT/'src/supermimic_research/mechanics/retargeting.py',
        ROOT/'src/supermimic_research/mechanics/compound_collision.py',ROOT/'src/supermimic_research/mechanics/collision.py',
        ROOT/'src/supermimic_research/mechanics/convex.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_kinematic_grasp_proposal_intent:1',sources=sources,
        reference=file_pin(reference/'report.json'),model=file_pin(reference/'hand2_mug.mjb'),frame=47,starts=starts,
        max_nfev_each=300,weights=[1,1,1,.25,.25],regularization=.003,positions_scale_m=.01,
        permitted_intent='thumb/index/middle link3/link4/tip_sensor; group must have a near component pair',
        band_m=[-.0001,.0002],nonintent_self_and_hand_table_minimum_m=0,
        role='KINEMATIC_PROPOSALS_NO_TRAINING_NO_HOLDING',integration_steps=0,neural_updates=0))
    hand=[model.geom(name).id for name in binding['hand_collision_names']]
    permitted={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in range(1,4)}
    audit=CompoundCollisionAudit(model,binding['mug_part_geom_ids'],hand,model.geom('mug_task_table').id,permitted)
    results=[]
    for i,start in enumerate(starts):
        result=engine.solve(start,max_nfev=300);folder=out/f'seed{i:02d}';folder.mkdir()
        d=engine.data;d.qpos[:]=engine.base;d.qpos[:20]=result['q20'];mujoco.mj_kinematics(model,d)
        geometry=audit.evaluate(d);write(folder/'geometry.json',geometry)
        result['geometry_pass']=geometry['passed'];result['geometry_unknown']=geometry['unknown']
        result['canonical_contact_candidates']=geometry['canonical_contact_candidates']
        result['failed_pair_counts']={kind:sum(not r['passed'] and r['kind']==kind for r in geometry['rows']) for kind in ['object','self','hand_table']}
        np.savez_compressed(folder/'candidate.npz',qpos27=d.qpos.copy(),ctrl20=result['q20'],targets_world=targets,
            tip_positions_world=result['tip_positions_world'],tip_errors_m=result['tip_errors_m'])
        result['candidate']=file_pin(folder/'candidate.npz');result['geometry']=file_pin(folder/'geometry.json')
        write(folder/'result.json',result);results.append(result)
        print(json.dumps(dict(seed=i,tip_errors_m=result['tip_errors_m'].tolist(),geometry_pass=geometry['passed'],
            failed=result['failed_pair_counts'],unknown=geometry['unknown'],elapsed_s=time.perf_counter()-tick)),flush=True)
    best=int(np.argmin([x['cost'] for x in results]))
    result=dict(schema='mug_kinematic_grasp_proposals:1',sources=sources,results=results,best_kinematic_cost_seed=best,
        geometry_pass_count=sum(x['geometry_pass'] for x in results),counts=engine.counts,
        parts=len(audit.parts),hand_geoms=len(audit.hand),self_pairs=len(audit.self),hand_table_pairs=len(audit.hand_table),
        native_integration_steps=0,neural_updates=0,physical_holding_qualified=False,source_gravity_preserved=False,
        wall_s=time.perf_counter()-tick,scope='Five source fingertip landmark alignment proposals only. All compound pairs checked at the fixed first BODY pose; later frames not fitted. Skin contact patches may be10-24mm from these tip markers. Solver success does not imply contact, collision clearance, native holding or source dynamics preservation.')
    for pin in sources:verify_file_pin(pin)
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass_count=result['geometry_pass_count'],counts=engine.counts)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
