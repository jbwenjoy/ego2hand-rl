"""Fixed-object material-anchor reach trials and full compound qualification.

Source regions are hypotheses from annotations, never observed contact labels.
Only independently qualified candidates may enter no-aid native integration.
"""
from pathlib import Path
import argparse,json,itertools,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.surface_patch_ik import material_anchor,SurfacePatchIK
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_contacts import CompoundContacts
from supermimic_research.mechanics.contact_modes import contact_wrench_basis
from supermimic_research.mechanics.patch_intent import match_patches
from supermimic_research.mechanics.support import static_feasibility
from supermimic_research.mechanics.compound_holding import evaluate_hold
from supermimic_research.runtime.native_evaluation import DTS,GATES


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,patches,out=map(lambda p:p.resolve(),[args.reference,args.patches,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])))
    parent=ROOT/'runs/mug_equilibrium_v1/seed00/final_state.npz';q=np.load(parent)['qpos27'];data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_forward(model,data)
    patch_report=json.loads((patches/'report.json').read_text());p=np.load(verify_file_pin(patch_report['output']))
    for pin in patch_report['sources']:verify_file_pin(pin)
    points=p['source_points_BODY_m'];inward=-p['outward_source_normals_BODY'];known=p['orientation_confirmed_by_approx_union'];pg=p['groups']
    masks={'all_oriented_candidates':known,'nonnegative_local_face_offset':known&(p['local_face_normal_offset_m']>=0)}
    R=data.xmat[bind['mug_body_id']].reshape(3,3).copy();origin=data.xpos[bind['mug_body_id']].copy()
    world_points=points@R.T+origin;world_inward=inward@R.T
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]}
    hand=[model.geom(n).id for n in bind['hand_collision_names']];parts=set(bind['mug_part_geom_ids'])
    lower=np.maximum(model.jnt_range[:20,0],model.actuator_ctrlrange[:,0])+1e-6
    upper=np.minimum(model.jnt_range[:20,1],model.actuator_ctrlrange[:,1])-1e-6
    if np.any(q[:20]<lower) or np.any(q[:20]>upper):raise ValueError('Frozen seed not interior-legal')
    sources=[file_pin(ROOT/path) for path in ['tools/propose_mug_source_patches.py','src/supermimic_research/mechanics/surface_patch_ik.py',
        'src/supermimic_research/mechanics/compound_collision.py','src/supermimic_research/mechanics/compound_contacts.py',
        'src/supermimic_research/mechanics/contact_modes.py','src/supermimic_research/mechanics/patch_intent.py',
        'src/supermimic_research/mechanics/support.py','src/supermimic_research/mechanics/compound_holding.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_source_material_anchor_intent:1',sources=sources,
        binding=file_pin(reference/'binding.json'),model=bind['model'],parent=file_pin(parent),patches=file_pin(patches/'report.json'),
        source_masks=list(masks),source_labels_are_unknown=True,object_pose_robot_root_material_and_mass_unchanged=True,
        anchor_rule='Nearest native component-distance endpoint on each of9 declared distal geoms; recover full convex hand surface normal. Fix that material point/normal during each local solve.',
        fixed_standoff_m=.00005,position_scale_m=.003,normal_scale=.5,regularization=.005,
        maximum_reach_solves=36,starts=['original4q','15percent_toward_legal_interval_midpoint'],max_nfev_per_solve=120,
        composition_rule='Best-cost thumb and middle anchor; each of3 index-anchor alternatives, separately for each source mask',
        maximum_full_geometry_audits=6,qualification='all3026 pair band plus material and actual native patch matches,3 actual groupFn>0.1N and cone6D before integration',
        maximum_native_cases=12,maximum_integrations=115200,native_gates=GATES,dts=DTS,neural_updates=0))
    anchors={};distance_queries=0
    for group,geoms in groups.items():
        for geom in geoms:
            best=None
            for part in sorted(parts):
                endpoint=np.zeros(6);gap=float(mujoco.mj_geomDistance(model,data,part,geom,.1,endpoint));distance_queries+=1
                if best is None or gap<best[0]:best=(gap,endpoint.copy(),part)
            gap,endpoint,part=best;vector=endpoint[3:]-endpoint[:3]
            if abs(gap)<1e-10 or abs(np.linalg.norm(vector)/abs(gap)-1)>.01:raise ValueError('No reliable native endpoint normal for anchor')
            anchor=material_anchor(model,data,geom,endpoint[3:],-vector/gap)
            anchors[geom]=dict(**anchor,group=group,initial_native_gap_m=gap,initial_object_part=part,geom_name=model.geom(geom).name)
    write(out/'anchors.json',list(anchors.values()));solutions=[];tick=time.perf_counter()
    for name,mask in masks.items():
        for group,geoms in groups.items():
            indices=np.arange((group-1)*4,group*4);target=np.flatnonzero(mask&(pg==group))
            if not len(target):raise ValueError('Unknown source patch cannot become a zero target')
            for geom in geoms:
                for seed,start in enumerate([q[indices],.85*q[indices]+.15*(lower[indices]+upper[indices])/2]):
                    problem=SurfacePatchIK(model,q,indices,indices,anchors[geom],world_points[target],world_inward[target],target)
                    result=problem.solve(start,lower[indices],upper[indices],max_nfev=120)
                    solutions.append(dict(hypothesis=name,group=group,geom=geom,seed=seed,**result))
        selected=[r for r in solutions if r['hypothesis']==name]
        print(json.dumps(dict(hypothesis=name,reach_solutions=len(selected),material_gate_pass=sum(r['material_point_patch_gate'] for r in selected),elapsed_s=time.perf_counter()-tick)),flush=True)
    write(out/'reach_solutions.json',solutions)
    audit=CompoundCollisionAudit(model,sorted(parts),hand,model.geom('mug_task_table').id,groups)
    gid_to_group={g:i for i,geoms in groups.items() for g in geoms};reader=CompoundContacts(model,sorted(parts),bind['mug_body_id'],groups,model.geom('mug_task_table').id)
    candidates=[];integrations=0
    for name,mask in masks.items():
        best={g:sorted([r for r in solutions if r['hypothesis']==name and r['group']==g],key=lambda r:r['proposal_cost']) for g in [1,2,3]}
        choices=[min([r for r in best[2] if r['geom']==geom],key=lambda r:r['proposal_cost']) for geom in groups[2]]
        for alternative,index in enumerate(choices):
            selected=[best[1][0],index,best[3][0]];candidate=q.copy()
            for group,row in enumerate(selected,1):candidate[(group-1)*4:group*4]=row['qpos'][(group-1)*4:group*4]
            folder=out/f'candidate{len(candidates):02d}';folder.mkdir();data.qpos[:]=candidate;data.qvel[:]=0;data.ctrl[:]=candidate[:20];mujoco.mj_forward(model,data)
            if not np.array_equal(candidate[12:],q[12:]):raise ValueError('Nonproposal joints/object changed')
            geometry=audit.evaluate(data);write(folder/'geometry.json',geometry)
            snapshot=reader.read(data,raw=True);rp=[];rn=[];rg=[]
            for ci in range(data.ncon):
                con=data.contact[ci];a,b=map(int,con.geom);oa,ob=a in parts,b in parts
                if oa==ob or con.efc_address<0:continue
                other=a if ob else b
                if other not in gid_to_group:continue
                force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
                if force[0]<=1e-8:continue
                rp.append(R.T@(con.pos-origin));rn.append(R.T@((1 if ob else -1)*con.frame.reshape(3,3)[0]));rg.append(gid_to_group[other])
            actual_match=match_patches(np.array(rp).reshape(-1,3),np.array(rn).reshape(-1,3),rg,points,inward,pg,mask)
            static=None
            if geometry['passed']:
                basis=contact_wrench_basis(model,data,sorted(parts),bind['mug_body_id'],gid_to_group)
                static=static_feasibility(basis['wrench'],np.r_[-model.body_mass[bind['mug_body_id']]*model.opt.gravity,np.zeros(3)])
            material_gate=all(r['material_point_patch_gate'] for r in selected)
            native_fn=snapshot['normal_by_finger_N'][:3]
            eligible=bool(geometry['passed'] and material_gate and actual_match['all_groups_matched'] and np.all(native_fn>.1) and static['success'])
            np.savez_compressed(folder/'candidate.npz',qpos27=candidate,ctrl20=candidate[:20],material_points_world=np.array([r['point_world'] for r in selected]),material_normals_world=np.array([r['normal_world'] for r in selected]))
            cases=[]
            if eligible:
                for dt in DTS:
                    case=evaluate_hold(model,candidate,candidate[:20],bind,groups,dt,folder/str(dt));integrations+=case['completed_steps']
                    cases.append(dict(dt_s=dt,summary=case,case=file_pin(folder/str(dt)/'report.json'),trace=file_pin(folder/str(dt)/'trace.npz')))
            row=dict(hypothesis=name,index_alternative=alternative,selected=[{k:r[k] for k in ['group','geom','seed','source_id','proposal_cost','source_distance_m','normal_dot']} for r in selected],
                material_point_patch_gate=material_gate,geometry_pass=geometry['passed'],geometry=file_pin(folder/'geometry.json'),candidate=file_pin(folder/'candidate.npz'),
                all_pairs_clear=geometry['all_pairs_clear_with_declared_pad_band'],unknown_geometry=geometry['unknown'],native_normal_by_group_N=native_fn,
                native_source_patch_matches=actual_match,static_cone6D=static,native_integration_eligible=eligible,native_cases=cases,
                geometry_failure_counts={kind:sum(not r['passed'] for r in geometry['rows'] if r['kind']==kind) for kind in ['object','self','hand_table']})
            write(folder/'result.json',row);candidates.append(row)
            print(json.dumps(dict(candidate=len(candidates)-1,hypothesis=name,material_gate=material_gate,geometry=geometry['passed'],failures=row['geometry_failure_counts'],native_match=actual_match['all_groups_matched'],elapsed_s=time.perf_counter()-tick)),flush=True)
    if len(solutions)!=36 or len(candidates)!=6 or integrations>115200:raise ValueError('Declared proposal budget mismatch')
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_source_material_anchor_proposals:1',sources=sources,inputs=[file_pin(reference/'binding.json'),file_pin(parent),file_pin(patches/'report.json')],
        anchors=file_pin(out/'anchors.json'),reach_solutions=file_pin(out/'reach_solutions.json'),native_anchor_distance_queries=distance_queries,reach_solve_count=len(solutions),
        candidates=candidates,geometry_pass_count=sum(c['geometry_pass'] for c in candidates),native_eligible_count=sum(c['native_integration_eligible'] for c in candidates),
        integration_steps=integrations,neural_updates=0,wall_s=time.perf_counter()-tick,
        scope='Fixed-object local material-point/normal reach proposals and full component geometry checks. Source regions are annotation-derived hypotheses, not contact truth. Selected fixed anchors and starts do not exhaust hand surfaces, pose space, contact modes or global reachability. No policy, calibrated dynamics, full source trajectory or independent RGB reconstruction claim.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass=report['geometry_pass_count'],native_eligible=report['native_eligible_count'],integration_steps=integrations)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['reference','patches','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
