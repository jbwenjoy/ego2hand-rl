"""Enumerate fixed candidate triplets before new robot pose optimization."""
from pathlib import Path
import argparse,itertools,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.contact_envelopes import unit_normal_envelope,rotate_wrenches
from supermimic_research.mechanics.contact_selection import triplet_balance


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,patches,envelopes,out=map(lambda p:p.resolve(),[args.reference,args.patches,args.envelopes,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());prior=json.loads((envelopes/'report.json').read_text());pr=json.loads((patches/'report.json').read_text())
    for pin in prior['sources']:verify_file_pin(pin)
    p=np.load(verify_file_pin(pr['output']));z=np.load(reference/'reference61.npz');COM=z['object_COM_in_BODY_m'];points=p['source_points_BODY_m'];normals=-p['outward_source_normals_BODY'];groups=p['groups']
    R=np.array(prior['object_rotation_robot_from_BODY']);targets={k:np.array(v) for k,v in prior['targets'].items()}
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    seed_path=ROOT/'runs/mug_patch_engagement_v1/preload01/candidate.npz';seed=np.load(seed_path);data.qpos[:]=seed['qpos27'];data.ctrl[:]=seed['ctrl20'];mujoco.mj_forward(model,data)
    origin=data.xpos[bind['mug_body_id']];points_world=points@R.T+origin;normals_world=normals@R.T
    anchors=json.loads((ROOT/'runs/mug_patch_proposals_v1/anchors.json').read_text());anchor_by_geom={a['geom']:a for a in anchors}
    coupled=json.loads((ROOT/'runs/mug_patch_coupled_v1/report.json').read_text());row=coupled['results'][0]
    composition=json.loads(Path(row['parent']['path']).with_name('result.json').read_text())
    current_points=[];current_normals=[]
    for chosen in composition['selected']:
        a=anchor_by_geom[chosen['geom']];geomR=data.geom_xmat[a['geom']].reshape(3,3)
        current_points.append(data.geom_xpos[a['geom']]+geomR@a['point_local']);current_normals.append(geomR@a['normal_local'])
    sources=[file_pin(ROOT/path) for path in ['tools/select_mug_support_triplets.py','src/supermimic_research/mechanics/contact_envelopes.py','src/supermimic_research/mechanics/contact_selection.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_source_triplet_selection_intent:1',sources=sources,
        prior=file_pin(envelopes/'report.json'),patches=file_pin(patches/'report.json'),reference=file_pin(reference/'reference61.npz'),seed=file_pin(seed_path),
        primary_edges=4,sensitivity_edges=8,mu=.7,minimum_normal_coefficient_per_finger_N=.1,
        enumerated_group_counts={'all30':[8,8,14],'nonnegative21':[3,6,12]},maximum_triplet_feasibility_calls=4448,
        targets=list(targets),maximum_selected=4,selection='Per source mask: minimum normal coefficient and least geometric anchor change among robot-gravity4-edge feasible triplets; distinct within mask, deterministic tie by enumeration.',
        geometry_score='sum((material_point-source_point)^2/(3mm)^2 + (material_normal-source_inward)^2/.5^2)',
        source_contact_truth_known=False,neural_updates=0,integration_steps=0))
    summaries=[];selected=[];calls=0;tick=time.perf_counter()
    for name,mask in [('all30',p['orientation_confirmed_by_approx_union']),('nonnegative21',p['orientation_confirmed_by_approx_union']&(p['local_face_normal_offset_m']>=0))]:
        triples=np.array(list(itertools.product(*[np.flatnonzero(mask&(groups==i)) for i in [1,2,3]])));N=len(triples)
        scores=np.array([sum(np.sum((current_points[j]-points_world[k])**2)/.003**2+np.sum((current_normals[j]-normals_world[k])**2)/.5**2 for j,k in enumerate(ids)) for ids in triples])
        raw=dict(triples=triples,geometric_anchor_change_score=scores)
        for edges in [4,8]:
            for gravity,target in targets.items():
                feasible=np.zeros(N,bool);normal=np.full(N,np.nan);coeff=np.zeros((N,3*(edges+1)));group_N=np.zeros((N,3))
                for i,ids in enumerate(triples):
                    W,local=unit_normal_envelope(points[ids]-COM,normals[ids],edges=edges);W=rotate_wrenches(W,R)
                    answer=triplet_balance(W,target,local+1);calls+=1
                    if answer['success']:
                        feasible[i]=True;normal[i]=answer['normal_coefficients_N'];coeff[i]=answer['coefficients'];group_N[i]=answer['group_normal_coefficients_N']
                prefix=f'edges{edges}_{gravity}';raw[prefix+'_feasible']=feasible;raw[prefix+'_normal_N']=normal;raw[prefix+'_coefficients']=coeff;raw[prefix+'_group_N']=group_N
                summaries.append(dict(hypothesis=name,edges=edges,target=gravity,triples=N,feasible_count=int(feasible.sum()),
                    minimum_normal_coefficients_N=float(normal[feasible].min()) if feasible.any() else None))
                print(json.dumps(summaries[-1]),flush=True)
        np.savez_compressed(out/(name+'.npz'),**raw)
        eligible=np.flatnonzero(raw['edges4_robot_gravity_feasible']);chosen=[]
        for criterion in ['minimum_normal','least_anchor_change']:
            if not len(eligible):continue
            ranking=sorted(eligible,key=lambda i:(raw['edges4_robot_gravity_normal_N'][i] if criterion=='minimum_normal' else scores[i],int(i)))
            index=next((int(i) for i in ranking if int(i) not in chosen),None)
            if index is None:continue
            chosen.append(index);selected.append(dict(hypothesis=name,criterion=criterion,index=index,source_candidate_ids=triples[index],
                ideal_normal_coefficients_N=float(raw['edges4_robot_gravity_normal_N'][index]),ideal_group_normal_N=raw['edges4_robot_gravity_group_N'][index],
                ideal_ray_coefficients_N=raw['edges4_robot_gravity_coefficients'][index],geometric_anchor_change_score=float(scores[index]),
                eight_edge_also_feasible=bool(raw['edges8_robot_gravity_feasible'][index]),conditional_source_also_feasible=bool(raw['edges4_conditional_source_Z_up_feasible'][index]),
                archive=file_pin(out/(name+'.npz'))))
    if calls!=4448 or len(selected)>4:raise ValueError('Enumeration budget changed')
    for pin in sources:verify_file_pin(pin)
    report=dict(schema='mug_source_support_triplet_selection:1',sources=sources,inputs=[file_pin(envelopes/'report.json'),file_pin(patches/'report.json'),file_pin(seed_path)],
        summaries=summaries,selected=selected,LP_calls=calls,wall_s=time.perf_counter()-tick,integration_steps=0,neural_updates=0,
        scope='Enumerated only1112 fixed estimated-source point triples, explicit3-group minimum and4/8-edge cone priors. No hand reachability, actual native manifold, actuator capacity, measured source force/gravity or trajectory success. Lower geometric-change scores are proposal heuristics, not policy reward.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),selected=[r['source_candidate_ids'].tolist() for r in selected],calls=calls)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','patches','envelopes','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
