"""Activate source-conditioned candidate contacts and test eligible native holds."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.component_preload import engage_pairs
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_contacts import CompoundContacts
from supermimic_research.mechanics.contact_modes import contact_wrench_basis
from supermimic_research.mechanics.patch_intent import match_patches
from supermimic_research.mechanics.support import static_feasibility,torque_box,maximum_support
from supermimic_research.mechanics.compound_holding import evaluate_hold
from supermimic_research.runtime.native_evaluation import DTS,GATES


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,parent,patches,out=map(lambda p:p.resolve(),[args.reference,args.parent,args.patches,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    prior=json.loads((parent/'report.json').read_text());selected=[r for r in prior['results'] if r['geometry_pass']]
    if len(selected)!=1:raise ValueError('This bounded stage requires exactly one qualified parent')
    seed=selected[0];bind=json.loads((reference/'binding.json').read_text());q0=np.load(verify_file_pin(seed['candidate']))['qpos27']
    before=json.loads(verify_file_pin(seed['geometry']).read_text());pr=json.loads((patches/'report.json').read_text());p=np.load(verify_file_pin(pr['output']))
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model);data.qpos[:]=q0;mujoco.mj_kinematics(model,data)
    R=data.xmat[bind['mug_body_id']].reshape(3,3).copy();origin=data.xpos[bind['mug_body_id']].copy()
    points=p['source_points_BODY_m'];inward=-p['outward_source_normals_BODY'];pg=p['groups'];mask=p['orientation_confirmed_by_approx_union']
    if seed['hypothesis']!='all_oriented_candidates':mask=mask&(p['local_face_normal_offset_m']>=0)
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]};gid_to_group={g:i for i,gs in groups.items() for g in gs}
    parts=set(bind['mug_part_geom_ids']);hand=[model.geom(n).id for n in bind['hand_collision_names']]
    audit=CompoundCollisionAudit(model,sorted(parts),hand,model.geom('mug_task_table').id,groups)
    reader=CompoundContacts(model,sorted(parts),bind['mug_body_id'],groups,model.geom('mug_task_table').id)
    pairs=[min([r for r in before['rows'] if r['kind']=='object' and r['geom_ids'][1] in geoms and r.get('gap_m') is not None],key=lambda r:r['gap_m'])['geom_ids'] for geoms in groups.values()]
    sources=[file_pin(ROOT/path) for path in ['tools/engage_mug_source_patches.py','src/supermimic_research/mechanics/component_preload.py',
        'src/supermimic_research/mechanics/compound_collision.py','src/supermimic_research/mechanics/compound_contacts.py',
        'src/supermimic_research/mechanics/contact_modes.py','src/supermimic_research/mechanics/patch_intent.py',
        'src/supermimic_research/mechanics/support.py','src/supermimic_research/mechanics/compound_holding.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_source_patch_engagement_intent:1',sources=sources,
        parent=file_pin(parent/'report.json'),seed=seed['candidate'],binding=file_pin(reference/'binding.json'),model=bind['model'],patches=file_pin(patches/'report.json'),
        target_depths_m=[-.00002,-.00005,-.000075],maximum_iterations_per_group=8,maximum_step_Linf_rad=.02,
        exact_pair_central_step_rad=1e-5,final_pair_tolerance_m=5e-6,pairs=pairs,material_and_gravity_unchanged=True,
        native_eligibility='full geometry and actual source-region/normal match, all groupFn>0.1N and cone6D success',
        maximum_LP_attempts=12,maximum_native_cases=6,maximum_integrations=57600,gates=GATES,dts=DTS,neural_updates=0))
    results=[];integrations=0;tick=time.perf_counter()
    for index,depth in enumerate([-.00002,-.00005,-.000075]):
        folder=out/f'preload{index:02d}';folder.mkdir();q,log=engage_pairs(model,q0,audit,pairs,depth)
        data.qpos[:]=q;data.qvel[:]=0;data.ctrl[:]=q[:20];mujoco.mj_forward(model,data)
        geometry=audit.evaluate(data);write(folder/'geometry.json',geometry);write(folder/'proposals.json',log)
        snapshot=reader.read(data,raw=True);rp=[];rn=[];rg=[]
        for ci in range(data.ncon):
            con=data.contact[ci];a,b=map(int,con.geom);oa,ob=a in parts,b in parts
            if oa==ob or con.efc_address<0:continue
            other=a if ob else b
            if other not in gid_to_group:continue
            force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
            if force[0]<=1e-8:continue
            rp.append(R.T@(con.pos-origin));rn.append(R.T@((1 if ob else -1)*con.frame.reshape(3,3)[0]));rg.append(gid_to_group[other])
        actual=match_patches(np.array(rp).reshape(-1,3),np.array(rn).reshape(-1,3),rg,points,inward,pg,mask)
        static=None
        if geometry['passed']:
            basis=contact_wrench_basis(model,data,sorted(parts),bind['mug_body_id'],gid_to_group);W,L=basis['wrench'],basis['loads']
            target=np.r_[-model.body_mass[bind['mug_body_id']]*model.opt.gravity,np.zeros(3)];offset=data.qfrc_bias[:20]-data.qfrc_passive[:20]
            lower,upper=torque_box(model,data,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1]);friction=model.dof_frictionloss[:20]
            static=dict(cone6D=static_feasibility(W,target),force_only=static_feasibility(W,target,force_only=True),
                full_command=static_feasibility(W,target,loads=L,offset=offset,lower=lower,upper=upper,friction=friction),
                maximum_support=maximum_support(W,L,offset,lower,upper,friction),EFC_columns=basis['native_EFC_unit_queries'],
                EFC_wrench_error=basis['native_wrench_error'],EFC_hand_load_error=basis['native_load_error'])
            np.savez_compressed(folder/'basis.npz',wrench6xN=W,joint_load20xN=L,target6=target,offset20=offset,lower20=lower,upper20=upper,friction20=friction)
        np.savez_compressed(folder/'candidate.npz',qpos27=q,ctrl20=q[:20]);cases=[]
        eligible=bool(geometry['passed'] and actual['all_groups_matched'] and np.all(snapshot['normal_by_finger_N'][:3]>.1) and static['cone6D']['success'])
        if eligible:
            for dt in DTS:
                case=evaluate_hold(model,q,q[:20],bind,groups,dt,folder/str(dt));integrations+=case['completed_steps']
                cases.append(dict(dt_s=dt,summary=case,case=file_pin(folder/str(dt)/'report.json'),trace=file_pin(folder/str(dt)/'trace.npz')))
        row=dict(target_depth_m=depth,candidate=file_pin(folder/'candidate.npz'),geometry=file_pin(folder/'geometry.json'),proposals=file_pin(folder/'proposals.json'),
            geometry_pass=geometry['passed'],unknown=geometry['unknown'],native_finger_normal_N=snapshot['normal_by_finger_N'][:3],
            native_source_patch_matches=actual,static=static,native_eligible=eligible,native_cases=cases,max_q_change_rad=float(abs(q[:20]-q0[:20]).max()))
        if static is not None:row['basis']=file_pin(folder/'basis.npz')
        write(folder/'report.json',row);results.append(row)
        print(json.dumps(dict(depth=depth,geometry=geometry['passed'],Fn=row['native_finger_normal_N'].tolist(),patch_match=actual['all_groups_matched'],cone6D=None if static is None else static['cone6D']['success'],native_cases=len(cases),elapsed_s=time.perf_counter()-tick)),flush=True)
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_source_patch_engagement:1',sources=sources,parent=file_pin(parent/'report.json'),results=results,
        geometry_pass_count=sum(r['geometry_pass'] for r in results),eligible_count=sum(r['native_eligible'] for r in results),
        native_cases=sum(len(r['native_cases']) for r in results),integration_steps=integrations,neural_updates=0,wall_s=time.perf_counter()-tick,
        scope='Small selected-pair pose engagement of a source-conditioned static grasp under the original point-force material prior. Source patches are uncertain annotation geometry; static LP is not native attainable force, global feasibility, measured source force or trajectory success.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),eligible=report['eligible_count'],cases=report['native_cases'],integrations=integrations)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['reference','parent','patches','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
