"""Counterfactual gravity targets on identical native/source candidate cones."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.contact_envelopes import unit_normal_envelope,rotate_wrenches
from supermimic_research.mechanics.support import static_feasibility
from supermimic_research.mechanics.wrench_separation import separating_witness
from supermimic_research.mechanics.contact_modes import contact_wrench_basis


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,patches,engagement,out=map(lambda p:p.resolve(),[args.reference,args.patches,args.engagement,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());patch_report=json.loads((patches/'report.json').read_text());parent=json.loads((engagement/'report.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    reference_pin=file_pin(reference/'reference61.npz');z=np.load(verify_file_pin(reference_pin));p=np.load(verify_file_pin(patch_report['output']))
    seed=np.load(verify_file_pin(parent['results'][1]['candidate']));data.qpos[:]=seed['qpos27'];data.ctrl[:]=seed['ctrl20'];mujoco.mj_forward(model,data)
    R=data.xmat[bind['mug_body_id']].reshape(3,3).copy();sourceR=z['T_source_world_from_object_BODY_F64'][0,:3,:3];COM=z['object_COM_in_BODY_m']
    gravity_robot=model.opt.gravity.copy();gravity_conditional=R@sourceR.T@np.array([0,0,-9.81]);mass=float(model.body_mass[bind['mug_body_id']])
    targets={'robot_gravity':np.r_[-mass*gravity_robot,np.zeros(3)],'conditional_source_Z_up':np.r_[-mass*gravity_conditional,np.zeros(3)]}
    sources=[file_pin(ROOT/path) for path in ['tools/compare_mug_gravity_envelopes.py','src/supermimic_research/mechanics/contact_envelopes.py',
        'src/supermimic_research/mechanics/support.py','src/supermimic_research/mechanics/wrench_separation.py','src/supermimic_research/mechanics/contact_modes.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_gravity_envelope_intent:1',sources=sources,
        binding=file_pin(reference/'binding.json'),model=bind['model'],reference=reference_pin,patches=file_pin(patches/'report.json'),engagement=file_pin(engagement/'report.json'),
        gravity_robot_world=gravity_robot,gravity_conditional_source_Z_up_in_robot_world=gravity_conditional,
        source_gravity_is_assumed_not_measured=True,simulator_gravity_unchanged=True,
        source_envelopes=['all30','nonnegative21'],source_edges=[0,4,8],source_mu_prior=.7,phase='BODY Frisvad, rotate complete wrench columns to world',
        native_envelopes='Three fixed engaged native condim3 manifolds; no geometry or phase change between gravity targets',
        maximum_feasibility_LP_attempts=36,maximum_separation_LP_attempts=18,integration_steps=0,neural_updates=0))
    envelopes=[]
    for name,mask in [('all30',p['orientation_confirmed_by_approx_union']),('nonnegative21',p['orientation_confirmed_by_approx_union']&(p['local_face_normal_offset_m']>=0))]:
        ids=np.flatnonzero(mask)
        for edges in [0,4,8]:
            BODY,local_ids=unit_normal_envelope(p['source_points_BODY_m'][ids]-COM,-p['outward_source_normals_BODY'][ids],edges=edges)
            W=rotate_wrenches(BODY,R);np.savez_compressed(out/f'{name}_edges{edges}.npz',wrench_BODY=BODY,wrench_world=W,source_candidate_index=ids[local_ids])
            envelopes.append(dict(name=f'{name}_edges{edges}',kind='UNCERTAIN_CANDIDATE_ENVELOPE',W=W,basis=file_pin(out/f'{name}_edges{edges}.npz'),edges=edges,candidate_count=len(ids)))
    allowed=[model.geom(f'right_finger{i}_{s}').id for i in [1,2,3] for s in ['link3_col','link4_col','tip_sensor_col']]
    for i,row in enumerate(parent['results']):
        state=np.load(verify_file_pin(row['candidate']));data.qpos[:]=state['qpos27'];data.qvel[:]=0;data.ctrl[:]=state['ctrl20'];mujoco.mj_forward(model,data)
        basis=contact_wrench_basis(model,data,bind['mug_part_geom_ids'],bind['mug_body_id'],allowed);saved=np.load(verify_file_pin(row['basis']))
        if np.max(abs(saved['wrench6xN']-basis['wrench']))>1e-10:raise ValueError('Native envelope changed')
        envelopes.append(dict(name=f'native_depth{i}',kind='NATIVE_FIXED_MANIFOLD',W=basis['wrench'],basis=row['basis'],edges=4,candidate_count=len(basis['contacts'])))
    results=[];attempts=0;separations=0
    for envelope in envelopes:
        answers={}
        for name,target in targets.items():
            force=static_feasibility(envelope['W'],target,force_only=True);full=static_feasibility(envelope['W'],target);attempts+=2
            separator=None
            if not full['success']:separator=separating_witness(envelope['W'],target);separations+=1
            answers[name]=dict(force_only=force,full6D=full,separation=separator)
        result={k:v for k,v in envelope.items() if k!='W'};result['targets']=answers;results.append(result)
        print(json.dumps(dict(envelope=envelope['name'],targets={k:dict(force=v['force_only']['success'],full6D=v['full6D']['success']) for k,v in answers.items()})),flush=True)
    if attempts!=36 or separations>18 or not np.array_equal(model.opt.gravity,gravity_robot):raise ValueError('Declared counterfactual scope changed')
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_counterfactual_gravity_envelopes:1',sources=sources,inputs=[file_pin(reference/'binding.json'),reference_pin,file_pin(patches/'report.json'),file_pin(engagement/'report.json')],
        gravity_robot_world=gravity_robot,gravity_conditional_source_Z_up_in_robot_world=gravity_conditional,
        gravity_angle_deg=float(np.degrees(np.arccos(np.clip(gravity_robot@gravity_conditional/(np.linalg.norm(gravity_robot)*np.linalg.norm(gravity_conditional)),-1,1)))),
        object_rotation_robot_from_BODY=R,source_world_from_BODY=sourceR,targets=targets,results=results,
        feasibility_LP_attempts=attempts,separation_LP_attempts=separations,source_gravity_measured=False,simulator_gravity_changed=False,integration_steps=0,neural_updates=0,
        scope='Same frozen ray cones under two targets only. Source candidates are uncertain geometry with unbounded force and no hand/actuator or simultaneous-contact constraint. Conditional Z-up is not sequence calibration, an executed easier task or human-force truth. Native snapshot feasibility is not dynamic attainability.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),gravity_angle_deg=report['gravity_angle_deg'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['reference','patches','engagement','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
