"""Conditional world-up-preserving mug placement with unchanged robot root."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.retargeting import HandTipIK
from supermimic_research.mechanics.contact_ik import ContactAwareIK
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,diagnostic,out=args.reference.resolve(),args.diagnostic.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());diag=json.loads((diagnostic/'report.json').read_text())
    source=np.load(reference/'reference61.npz');firstR=source['T_source_world_from_object_BODY_F64'][0,:3,:3]
    feasible=next(r for r in diag['records'] if r['cone6D']['success']);z=np.load(verify_file_pin(feasible['raw']))
    q0=z['qpos27'].copy();oldR=Rotation.from_quat(q0[23:27][[1,2,3,0]]).as_matrix()
    contacts=np.array([c['position_world_m'] for c in feasible['contacts']]);anchor=contacts.mean(0)
    anchor_BODY=oldR.T@(anchor-q0[20:23]);yaws=np.arange(6)*np.pi/3
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])))
    hand=[model.geom(n).id for n in bind['hand_collision_names']]
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in range(1,4)}
    audit=CompoundCollisionAudit(model,bind['mug_part_geom_ids'],hand,model.geom('mug_task_table').id,groups)
    sources=[file_pin(p) for p in [Path(__file__),ROOT/'src/supermimic_research/mechanics/retargeting.py',
        ROOT/'src/supermimic_research/mechanics/contact_ik.py',ROOT/'src/supermimic_research/mechanics/compound_collision.py',
        ROOT/'src/supermimic_research/mechanics/collision.py',ROOT/'src/supermimic_research/mechanics/convex.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_conditional_gravity_placement_intent:1',sources=sources,
        diagnostic=file_pin(diagnostic/'report.json'),reference=file_pin(reference/'report.json'),feasible_operand=feasible['raw'],
        yaw_rad=yaws,anchor_world_m=anchor,anchor_BODY_m=anchor_BODY,
        placement='R_robotWorld_object=Rz(yaw)*R_sourceWorld_object(first); translate to hold declared handle-anchor centroid',
        source_Z_up_is_conditional=True,per_sequence_gravity_measured=False,robot_root_unchanged=True,
        fingertip_targets='original source first-frame human annotations in object BODY, mapped through proposed object pose',
        max_tip_nfev_each=300,max_contact_nfev_each=160,max_contact_seconds_each=45,
        no_scale_or_collision_asset_change=True,neural_updates=0,integration_steps=0))
    results=[];tick=time.perf_counter()
    for index,yaw in enumerate(yaws):
        folder=out/f'seed{index:02d}';folder.mkdir();R=Rotation.from_rotvec([0,0,yaw]).as_matrix()@firstR
        p=anchor-R@anchor_BODY;q=q0.copy();q[20:23]=p;q[23:27]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        targets=source['hand_landmarks21_object_BODY_F64'][0,:5]@R.T+p
        tip=HandTipIK(model,q,targets,weights=[1,1,1,.25,.25]);tip_result=tip.solve(q0[:20],max_nfev=300)
        engine=ContactAwareIK(model,q,targets,audit)
        with (folder/'proposals.jsonl').open('x',encoding='utf8') as log:
            result=engine.solve_contact(tip_result['q20'],log,max_nfev=160,maximum_seconds=45.)
        data=engine.data;data.qpos[:]=q;data.qpos[:20]=result['q20'];mujoco.mj_kinematics(model,data)
        geometry=audit.evaluate(data);write(folder/'geometry.json',geometry)
        np.savez_compressed(folder/'candidate.npz',qpos27=data.qpos.copy(),ctrl20=result['q20'],targets_world=targets,
            tip_errors_m=result['tip_errors_m'],conditional_world_to_robot_rotation=Rotation.from_rotvec([0,0,yaw]).as_matrix())
        failed=[r for r in geometry['rows'] if not r['passed']]
        deltaR=R@oldR.T
        result.update(yaw_rad=float(yaw),geometry_pass=geometry['passed'],all_pairs_clear=geometry['all_pairs_clear_with_declared_pad_band'],
            geometry_unknown=geometry['unknown'],canonical_contact_candidates=geometry['canonical_contact_candidates'],
            failed_pair_counts={kind:sum(r['kind']==kind for r in failed) for kind in ['object','self','hand_table']},
            candidate=file_pin(folder/'candidate.npz'),geometry=file_pin(folder/'geometry.json'),proposals=file_pin(folder/'proposals.jsonl'),
            source_world_up_preserved_conditionally=True,original_rest_palm_alignment_preserved=False,
            object_rotation_delta_from_previous_rad=float(Rotation.from_matrix(deltaR).magnitude()),object_root_delta_from_previous_m=p-q0[20:23])
        write(folder/'result.json',result);results.append(result)
        print(json.dumps(dict(yaw_deg=float(np.rad2deg(yaw)),geometry_pass=geometry['passed'],all_pairs_clear=result['all_pairs_clear'],
            failed=result['failed_pair_counts'],canonical=geometry['canonical_contact_candidates'],tip_errors_m=result['tip_errors_m'].tolist(),elapsed_s=time.perf_counter()-tick)),flush=True)
    report=dict(schema='mug_conditional_gravity_placement_proposals:1',sources=sources,results=results,
        geometry_pass_count=sum(r['geometry_pass'] for r in results),all_pairs_clear_count=sum(r['all_pairs_clear'] for r in results),
        neural_updates=0,integration_steps=0,wall_s=time.perf_counter()-tick,
        scope='Six first-frame placements preserving conditional source Z-up orientation modulo yaw. Handle anchor fixed in robot world; robot root and assets unchanged. Original rest-palm alignment intentionally lost, target clock/full trajectory/force labels not rebuilt. Native-distance proposal followed by all-piece static audit; not physical holding or verified per-sequence gravity.')
    for pin in sources:verify_file_pin(pin)
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass_count=report['geometry_pass_count'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
