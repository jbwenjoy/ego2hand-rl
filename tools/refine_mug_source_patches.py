"""Coupled whole-hand collision/patch repair of two explicit hypotheses."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.collision_patch_ik import CollisionPatchIK
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.patch_intent import match_patches
from supermimic_research.mechanics.contact_modes import contact_wrench_basis
from supermimic_research.mechanics.support import static_feasibility


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,parent,patches,out=map(lambda p:p.resolve(),[args.reference,args.parent,args.patches,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());prior=json.loads((parent/'report.json').read_text())
    for pin in prior['sources']:verify_file_pin(pin)
    anchor_rows=json.loads(verify_file_pin(prior['anchors']).read_text());anchors={a['geom']:a for a in anchor_rows}
    for a in anchors.values():a['point_local']=np.array(a['point_local']);a['normal_local']=np.array(a['normal_local'])
    pr=json.loads((patches/'report.json').read_text());raw=np.load(verify_file_pin(pr['output']))
    points=raw['source_points_BODY_m'];inward=-raw['outward_source_normals_BODY'];pg=raw['groups'];known=raw['orientation_confirmed_by_approx_union']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    hand=[model.geom(n).id for n in bind['hand_collision_names']];parts=set(bind['mug_part_geom_ids'])
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]};gids={g:i for i,geoms in groups.items() for g in geoms}
    audit=CompoundCollisionAudit(model,sorted(parts),hand,model.geom('mug_task_table').id,groups)
    selected=[]
    for name in ['all_oriented_candidates','nonnegative_local_face_offset']:
        candidates=[r for r in prior['candidates'] if r['hypothesis']==name]
        selected.append(min(candidates,key=lambda r:(sum(r['geometry_failure_counts'].values()),r['index_alternative'])))
    sources=[file_pin(ROOT/path) for path in ['tools/refine_mug_source_patches.py','src/supermimic_research/mechanics/collision_patch_ik.py',
        'src/supermimic_research/mechanics/surface_patch_ik.py','src/supermimic_research/mechanics/contact_ik.py',
        'src/supermimic_research/mechanics/retargeting.py','src/supermimic_research/mechanics/compound_collision.py',
        'src/supermimic_research/mechanics/patch_intent.py','src/supermimic_research/mechanics/contact_modes.py',
        'src/supermimic_research/mechanics/support.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_coupled_patch_repair_intent:1',sources=sources,
        binding=file_pin(reference/'binding.json'),model=bind['model'],parent=file_pin(parent/'report.json'),patches=file_pin(patches/'report.json'),
        selection='Lowest total failed exact pairs per source mask, tie by index alternative',selected=[r['candidate'] for r in selected],
        maximum_seeds=2,maximum_seconds_per_seed=30.,maximum_nfev_per_seed=160,full20q_legal_bounds=True,
        collision_penalties='Unchanged shared ContactAwareIK object/self/table/intent rows; no tip-fit rows',
        material_anchor_position_scale_m=.003,normal_scale=.5,stand_off_m=.00005,maximum_full_audits=2,
        integration_steps=0,neural_updates=0,source_contact_truth_available=False))
    results=[];tick=time.perf_counter()
    for seed,item in enumerate(selected):
        q=np.load(verify_file_pin(item['candidate']))['qpos27'];data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        R=data.xmat[bind['mug_body_id']].reshape(3,3).copy();origin=data.xpos[bind['mug_body_id']].copy()
        mask=known if item['hypothesis']=='all_oriented_candidates' else known&(raw['local_face_normal_offset_m']>=0)
        chosen=[anchors[r['geom']] for r in item['selected']]
        folder=out/f'seed{seed:02d}';folder.mkdir();problem=CollisionPatchIK(model,q,audit,chosen,points@R.T+origin,inward@R.T,pg,mask)
        with (folder/'proposals.jsonl').open('w',encoding='utf8') as log:solution=problem.solve(log,maximum_seconds=30.,max_nfev=160)
        final=q.copy();final[:20]=solution['best']['q20'];data.qpos[:]=final;data.qvel[:]=0;data.ctrl[:]=final[:20];mujoco.mj_forward(model,data)
        if not np.array_equal(final[20:],q[20:]):raise ValueError('Object changed')
        geometry=audit.evaluate(data);write(folder/'geometry.json',geometry)
        rp=[];rn=[];rg=[];fn=np.zeros(3)
        for ci in range(data.ncon):
            con=data.contact[ci];a,b=map(int,con.geom);oa,ob=a in parts,b in parts
            if oa==ob or con.efc_address<0:continue
            other=a if ob else b
            if other not in gids:continue
            force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
            if force[0]<=1e-8:continue
            group=gids[other];fn[group-1]+=force[0]
            rp.append(R.T@(con.pos-origin));rn.append(R.T@((1 if ob else -1)*con.frame.reshape(3,3)[0]));rg.append(group)
        actual=match_patches(np.array(rp).reshape(-1,3),np.array(rn).reshape(-1,3),rg,points,inward,pg,mask)
        static=None
        if geometry['passed']:
            basis=contact_wrench_basis(model,data,sorted(parts),bind['mug_body_id'],gids)
            static=static_feasibility(basis['wrench'],np.r_[-model.body_mass[bind['mug_body_id']]*model.opt.gravity,np.zeros(3)])
        np.savez_compressed(folder/'candidate.npz',qpos27=final,ctrl20=final[:20]);write(folder/'solver.json',solution)
        row=dict(seed=seed,hypothesis=item['hypothesis'],parent=item['candidate'],candidate=file_pin(folder/'candidate.npz'),geometry=file_pin(folder/'geometry.json'),
            proposals=file_pin(folder/'proposals.jsonl'),solver=file_pin(folder/'solver.json'),geometry_pass=geometry['passed'],unknown=geometry['unknown'],
            all_pairs_clear=geometry['all_pairs_clear_with_declared_pad_band'],geometry_failure_counts={kind:sum(not r['passed'] for r in geometry['rows'] if r['kind']==kind) for kind in ['object','self','hand_table']},
            maximum_penetration_m=max(0.,-min(r['gap_m'] for r in geometry['rows'] if r.get('gap_m') is not None)),
            material_patch_metrics=[{k:p[k] for k in ['source_id','source_distance_m','normal_dot']} for p in solution['best']['patches']],
            native_source_patch_matches=actual,native_finger_normal_N=fn,static_cone6D=static,
            solver_success=solution['solver_success'],coupled_calls=solution['coupled_calls'],no_physical_success_claim=True)
        write(folder/'result.json',row);results.append(row);print(json.dumps(dict(seed=seed,geometry=row['geometry_pass'],failures=row['geometry_failure_counts'],material=row['material_patch_metrics'],actual_matches=actual['all_groups_matched'],elapsed_s=time.perf_counter()-tick)),flush=True)
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_coupled_patch_repair:1',sources=sources,results=results,parent=file_pin(parent/'report.json'),
        geometry_pass_count=sum(r['geometry_pass'] for r in results),integration_steps=0,neural_updates=0,wall_s=time.perf_counter()-tick,
        scope='Two local coupled20q source-patch and native collision-penalty searches; full independent ideal-convex final audits. No global infeasibility, physical support, force labels, policy, complete source trajectory or RGB reconstruction claim.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass=report['geometry_pass_count'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['reference','parent','patches','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
