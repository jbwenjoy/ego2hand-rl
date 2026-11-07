"""Read back candidate geometry and unit-force descriptors without Torch.

This is an implementation/convention check, not contact-truth verification.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def fingerprint(points,normals,mask,basis,COM):
    # Explicit per-contact/per-ray construction, distinct from the batched kernel.
    best=np.zeros(len(basis))
    for p,n in zip(points[mask],normals[mask]):
        n=n/np.linalg.norm(n);s=1. if n[2]>=0 else -1.;a=-1/(s+n[2]);b=n[0]*n[1]*a
        t1=np.array([1+s*n[0]**2*a,s*b,-s*n[0]]);t1/=np.linalg.norm(t1)
        t2=np.array([b,s+n[1]**2*a,-n[1]]);t2/=np.linalg.norm(t2)
        rays=[n]
        for angle in np.arange(8)*(2*np.pi/8):
            f=n+.7*(np.cos(angle)*t1+np.sin(angle)*t2);rays.append(f/np.linalg.norm(f))
        for f in rays:
            w=np.r_[f,np.cross(p-COM,f)/.1];best=np.maximum(best,basis@w)
    return best


def score(sim,target):
    active=target>1e-3
    if not active.any():return 0.
    loss=np.maximum(.9*target-sim,0)**2+np.maximum(sim-1.1*target,0)**2
    return float(np.sum(np.where(active&(sim>1e-3),np.exp(-loss/.1),0))/active.sum())


def main(args):
    import mujoco
    run,modes,reference,out=map(lambda p:p.resolve(),[args.run,args.modes,args.reference,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((run/'report.json').read_text());mode=json.loads((modes/'report.json').read_text())
    binding=json.loads((reference/'binding.json').read_text())
    for pin in report['sources']:verify_file_pin(pin)
    raw=np.load(verify_file_pin(report['output']));inputs={k:np.load(verify_file_pin(v)) for k,v in report['inputs'].items()}
    ref,mesh,proximity=inputs['reference'],inputs['source_mesh'],inputs['proximity'];COM=ref['object_COM_in_BODY_m']
    p=raw['source_points_BODY_m'];inward=-raw['outward_source_normals_BODY'];groups=raw['groups'];known=raw['orientation_confirmed_by_approx_union']
    ids=raw['source_vertex_ids'];triids=raw['source_triangle_ids'];tri=mesh['vertices_model_m'][mesh['faces'][triids]]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    distances=[];barycentric=[]
    for point,t in zip(p,tri):
        uv=np.linalg.lstsq((t[1:]-t[0]).T,point-t[0],rcond=None)[0]
        distances.append(np.linalg.norm(point-(t[0]+uv@(t[1:]-t[0]))));barycentric.append(min(*uv,1-uv.sum()))
    offset=np.sum((ref['hand_vertices_object_BODY_F64'][0,ids]-p)*normals,axis=1)
    checks=dict(source_triangles_reconstructed=max(distances)<1e-12 and min(barycentric)>-1e-8,
        source_normals_and_offsets=np.max(np.abs(normals+inward))<1e-12 and np.max(np.abs(offset-raw['local_face_normal_offset_m']))<1e-12,
        source_point_triangle_group_identity=np.array_equal(triids,proximity['source_triangle_ids'][0,ids]) and np.array_equal(groups,proximity['dominant_skinning_group'][ids]) and np.array_equal(p,proximity['closest_source_points_BODY_m'][0,ids]),
        COM_subtracted_once=np.array_equal(raw['source_points_COM_relative_m'],p-COM),
        no_contact_truth_fabricated=not raw['contact_ground_truth_known'].any() and not report['contact_ground_truth_or_real_force_available'],
        candidate_counts=len(p)==30 and np.sum(offset<0)==9 and np.sum(known&(offset>=0))==21)
    basis=raw['basis512'];masks=[known,known&(offset>=0)];names=['all_oriented_candidates','nonnegative_local_face_offset']
    source_support=[fingerprint(p,inward,m,basis,COM) for m in masks]
    source_group=[np.stack([fingerprint(p,inward,m&(groups==g),basis,COM) for g in [1,2,3]]) for m in masks]
    errors=[float(np.max(np.abs(source_support[i]-raw[key]))) for i,key in enumerate(['support_all_oriented','support_nonnegative_local'])]
    errors.extend(float(np.max(np.abs(source_group[i]-raw[key]))) for i,key in enumerate(['source_group_support_all','source_group_support_nonnegative']))
    records=[];contacts_checked=0
    for index,robot in enumerate(report['robot']):
        case=next(c for c in mode['cases'] if c['mode']=='point3' and c['seed']==robot['seed'] and c['dt_s']==.0003125)
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])));trace=np.load(verify_file_pin(case['trace']))
        data=mujoco.MjData(model);data.qpos[:]=trace['initial_qpos'];data.ctrl[:]=trace['ctrl20'];mujoco.mj_forward(model,data)
        static=json.loads(verify_file_pin(case['static']).read_text());parts=set(binding['mug_part_geom_ids'])
        geoms={model.geom(f'right_finger{i}_{suffix}').id:i for i in [1,2,3] for suffix in ['link3_col','link4_col','tip_sensor_col']}
        R=data.xmat[binding['mug_body_id']].reshape(3,3);origin=data.xpos[binding['mug_body_id']]
        rp=[];rn=[];rg=[]
        for saved in static['contacts']:
            if saved['local_force6'][0]<=1e-8:continue
            if saved['dim']!=3:raise ValueError('Point-force descriptor cannot silently drop a spin couple')
            a,b=saved['geom_ids'];sign=1 if b in parts else -1;other=a if sign==1 else b
            rp.append(R.T@(np.array(saved['position_world_m'])-origin));rn.append(R.T@(sign*np.array(saved['frame_world_rows'])[0]));rg.append(geoms[other])
        rp,rn,rg=np.array(rp),np.array(rn),np.array(rg);contacts_checked+=len(rp)
        sim=fingerprint(rp,rn,np.ones(len(rp),bool),basis,COM)
        per=np.stack([fingerprint(rp,rn,rg==g,basis,COM) for g in [1,2,3]])
        errors.extend([float(np.max(np.abs(sim-raw['robot_point_supports'][index]))),float(np.max(np.abs(per-raw['robot_per_finger_supports'][index])))])
        match_ok=True;score_error=0.
        for i,(name,mask) in enumerate(zip(names,masks)):
            score_error=max(score_error,abs(score(sim,source_support[i])-robot['geometric_similarity_only'][name]))
            for j,g in enumerate([1,2,3]):
                score_error=max(score_error,abs(score(per[j],source_group[i][j])-robot['per_finger_geometric_similarity'][name][j]))
                pairs=[]
                for ri in np.flatnonzero(rg==g):
                    for ti in np.flatnonzero(mask&(groups==g)):
                        if np.linalg.norm(rp[ri]-p[ti])<=.003 and rn[ri]@inward[ti]>=.5:pairs.append([int(ri),int(ti)])
                saved=robot['candidate_patch_matches'][name]['groups'][j]
                match_ok &= saved['pairs']==pairs and saved['matched']==bool(pairs) and not robot['candidate_patch_matches'][name]['source_contact_truth_known']
        records.append(dict(seed=robot['seed'],native_contact_count=len(rp),score_error=score_error,patch_pairs_reconstructed=bool(match_ok)))
    checks.update(numpy_descriptor_readback=max(errors)<2e-12,numpy_scores_and_patch_matches=all(r['score_error']<2e-12 and r['patch_pairs_reconstructed'] for r in records),
        source_remains_first_frame_only=report['neural_updates']==0 and report['integration_steps']==0 and not report['CUDA_initialized'])
    out.mkdir(parents=True);result=dict(schema='mug_source_patch_saved_review:1',source=file_pin(__file__),run=file_pin(run/'report.json'),
        modes=file_pin(modes/'report.json'),binding=file_pin(reference/'binding.json'),checks={k:bool(v) for k,v in checks.items()},
        max_unit_force_descriptor_error=max(errors),native_contacts_checked=contacts_checked,records=records,
        passed=sum(bool(v) for v in checks.values()),total=len(checks),native_static_forwards=2,integration_steps=0,neural_updates=0,
        scope='NumPy unit-force ray/COM/score and saved candidate triangle/identity/normal/patch readback. Reuses saved approximate-union orientation decisions; does not verify contact truth, gravity, source normal global orientation, source physical forces or a valid learning reward.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(result))
    if not all(checks.values()):raise ValueError('Retain failed patch readback')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['run','modes','reference','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
