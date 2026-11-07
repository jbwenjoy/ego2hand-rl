"""Declared response-scale sensitivity at two existing legal projected states."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.acceleration_projection import AccelerationProjection,zero_velocity_COM_acceleration
from supermimic_research.mechanics.native_projection import search_projection
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
    previous=json.loads((parent/'report.json').read_text());bind=json.loads((reference/'binding.json').read_text());pr=json.loads((patches/'report.json').read_text());raw=np.load(verify_file_pin(pr['output']))
    targets=raw['source_points_BODY_m'];inward=-raw['outward_source_normals_BODY'];pg=raw['groups'];known=np.zeros(len(targets),bool);known[[0,13,29]]=True
    selected=[next(r for r in previous['results'] if r['name']==name) for name in ['inverse_parent','forward_parent']]
    sources=[file_pin(ROOT/p) for p in ['tools/project_mug_acceleration.py','src/supermimic_research/mechanics/acceleration_projection.py',
        'src/supermimic_research/mechanics/native_projection.py','src/supermimic_research/mechanics/compound_collision.py',
        'src/supermimic_research/mechanics/compound_holding.py','src/supermimic_research/mechanics/patch_intent.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_acceleration_projection_intent:1',sources=sources,parent=file_pin(parent/'report.json'),binding=file_pin(reference/'binding.json'),model=bind['model'],patches=file_pin(patches/'report.json'),
        seeds=[r['actual_forward'] for r in selected],q_axis_radius_rad=.02,u_axis_radius_rad=.3,object_and_robot_root_fixed=True,
        residual_scales=dict(hand_rad_s2=.5,COM_m_s2=.005,angular_world_rad_s2=.1),
        shared_geometry_force_penalties_unchanged=True,maximum_calls_per_seed=2400,maximum_seconds_per_seed=35.,scaled_secant=.0001,
        exact_source_candidate_ids=[0,13,29],maximum_new_native_cases=4,maximum_new_integration_steps=38400,gates=GATES,dts=DTS,neural_updates=0))
    results=[];steps=0
    for i,parent_row in enumerate(selected):
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));model.opt.timestep=DTS[0]
        state=np.load(verify_file_pin(parent_row['actual_forward']));q,u=state['qpos27'],state['ctrl20']
        groups={g:[model.geom(f'right_finger{g}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for g in [1,2,3]};gids={gid:g for g,gs in groups.items() for gid in gs};parts=set(bind['mug_part_geom_ids'])
        hand=[model.geom(n).id for n in bind['hand_collision_names']];audit=CompoundCollisionAudit(model,sorted(parts),hand,model.geom('mug_task_table').id,groups)
        engine=AccelerationProjection(model,q,u,bind,groups);folder=out/f'seed{i:02d}';folder.mkdir()
        with (folder/'proposals.jsonl').open('w',encoding='utf8') as log:solution=search_projection(engine,log,maximum_calls=2400,maximum_seconds=35.,secant_step=.0001)
        final=solution.pop('final');data=engine.data;geometry=audit.evaluate(data);write(folder/'geometry.json',geometry);write(folder/'solver.json',solution)
        linear,angular=zero_velocity_COM_acceleration(model,data,bind['mug_body_id'],20);np.savez_compressed(folder/'actual_state.npz',**{k:v for k,v in final.items() if isinstance(v,np.ndarray)},scaled_z=np.array(solution['best']['z']),COM_acceleration_world=linear,angular_acceleration_world=angular)
        rp=[];rn=[];rg=[];R=data.xmat[bind['mug_body_id']].reshape(3,3);origin=data.xpos[bind['mug_body_id']]
        for ci in range(data.ncon):
            con=data.contact[ci];a,b=map(int,con.geom);oa,ob=a in parts,b in parts
            if oa==ob or con.efc_address<0:continue
            other=a if ob else b
            if other not in gids:continue
            force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
            if force[0]<=1e-8:continue
            rp.append(R.T@(con.pos-origin));rn.append(R.T@((1 if ob else -1)*con.frame.reshape(3,3)[0]));rg.append(gids[other])
        matched=match_patches(np.array(rp).reshape(-1,3),np.array(rn).reshape(-1,3),rg,targets,inward,pg,known)
        cases=[]
        if geometry['passed'] and matched['all_groups_matched'] and np.all(final['normal_by_finger_N'][:3]>.1):
            for dt in DTS:
                case_folder=folder/str(dt);case=evaluate_hold(model,final['qpos27'],final['ctrl20'],bind,groups,dt,case_folder);steps+=case['completed_steps'];cases.append(dict(dt_s=dt,summary=case,case=file_pin(case_folder/'report.json'),trace=file_pin(case_folder/'trace.npz')))
        row=dict(seed=i,parent_name=parent_row['name'],parent_state=parent_row['actual_forward'],state=file_pin(folder/'actual_state.npz'),geometry=file_pin(folder/'geometry.json'),solver=file_pin(folder/'solver.json'),proposals=file_pin(folder/'proposals.jsonl'),
            geometry_pass=geometry['passed'],matched_source_candidate_regions=matched,COM_acceleration_world=linear,angular_acceleration_world=angular,
            joint_acceleration_max=float(abs(final['qacc26'][:20]).max()),net_COM_wrench=final['net_object_COM_wrench_world'],native_cases=cases)
        write(folder/'result.json',row);results.append(row);print(json.dumps(dict(seed=i,geometry=geometry['passed'],matched=matched['all_groups_matched'],COM_acceleration=linear.tolist(),angular_acceleration=angular.tolist(),joint_accel=row['joint_acceleration_max'],calls=solution['calls'],native_cases=[(c['dt_s'],c['summary']['actual_duration_s'],c['summary']['native_holding_gates_pass']) for c in cases])),flush=True)
    if steps>38400:raise ValueError('Declared new integration budget exceeded')
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_acceleration_projection:1',sources=sources,results=results,parent=file_pin(parent/'report.json'),native_cases=sum(len(r['native_cases']) for r in results),integration_steps=steps,neural_updates=0,
        scope='Declared acceleration-scaled local projection sensitivity. Centers differ from original support pose, so total stage displacement must be reported. Same original model/gravity/controls, no external object force, no optimizer-return or state-response success claim.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),cases=report['native_cases'],steps=steps)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','parent','patches','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
