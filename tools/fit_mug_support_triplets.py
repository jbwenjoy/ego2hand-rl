"""Fit fixed support-selected source triplets, then qualify actual native support."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.surface_patch_ik import SurfacePatchIK
from supermimic_research.mechanics.collision_patch_ik import CollisionPatchIK
from supermimic_research.mechanics.component_preload import engage_pairs
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_contacts import CompoundContacts
from supermimic_research.mechanics.contact_modes import contact_wrench_basis
from supermimic_research.mechanics.patch_intent import match_patches
from supermimic_research.mechanics.support import static_feasibility,torque_box
from supermimic_research.mechanics.compound_holding import evaluate_hold
from supermimic_research.runtime.native_evaluation import DTS,GATES


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,patches,selection,out=map(lambda p:p.resolve(),[args.reference,args.patches,args.selection,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());prior=json.loads((selection/'report.json').read_text());pr=json.loads((patches/'report.json').read_text())
    for pin in prior['sources']:verify_file_pin(pin)
    raw=np.load(verify_file_pin(pr['output']));p=raw['source_points_BODY_m'];n=-raw['outward_source_normals_BODY'];pg=raw['groups']
    seed_path=ROOT/'runs/mug_patch_engagement_v1/preload01/candidate.npz';q0=np.load(seed_path)['qpos27'];anchor_path=ROOT/'runs/mug_patch_proposals_v1/anchors.json'
    anchors=json.loads(anchor_path.read_text());anchors_by_group={i:[] for i in [1,2,3]}
    for a in anchors:a['point_local']=np.array(a['point_local']);a['normal_local']=np.array(a['normal_local']);anchors_by_group[a['group']].append(a)
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));base_dt=float(model.opt.timestep)
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]};gids={g:i for i,gs in groups.items() for g in gs}
    parts=set(bind['mug_part_geom_ids']);hand=[model.geom(name).id for name in bind['hand_collision_names']]
    audit=CompoundCollisionAudit(model,sorted(parts),hand,model.geom('mug_task_table').id,groups)
    reader=CompoundContacts(model,sorted(parts),bind['mug_body_id'],groups,model.geom('mug_task_table').id)
    lower=np.maximum(model.jnt_range[:20,0],model.actuator_ctrlrange[:,0])+1e-6;upper=np.minimum(model.jnt_range[:20,1],model.actuator_ctrlrange[:,1])-1e-6
    if np.any(q0[:20]<lower) or np.any(q0[:20]>upper):raise ValueError('Legal seed, no implicit projection')
    sources=[file_pin(ROOT/path) for path in ['tools/fit_mug_support_triplets.py','src/supermimic_research/mechanics/surface_patch_ik.py',
        'src/supermimic_research/mechanics/collision_patch_ik.py','src/supermimic_research/mechanics/contact_ik.py',
        'src/supermimic_research/mechanics/component_preload.py','src/supermimic_research/mechanics/compound_collision.py',
        'src/supermimic_research/mechanics/compound_contacts.py','src/supermimic_research/mechanics/contact_modes.py',
        'src/supermimic_research/mechanics/patch_intent.py','src/supermimic_research/mechanics/support.py','src/supermimic_research/mechanics/compound_holding.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_support_selected_fit_intent:1',sources=sources,
        binding=file_pin(reference/'binding.json'),model=bind['model'],selection=file_pin(selection/'report.json'),patches=file_pin(patches/'report.json'),seed=file_pin(seed_path),anchors=file_pin(anchor_path),
        selected_count=len(prior['selected']),maximum_material_reach_solves=72,maximum_reach_nfev=120,
        material_anchor_rule='Three existing anchors per finger, original start or15percent toward joint interval midpoint; choose minimum cost for the exact selected source point/normal',
        maximum_coupled_seeds=4,maximum_seconds_per_coupled_seed=30.,maximum_coupled_nfev=160,
        joint_or_object_projection=False,robot_root_object_pose_and_material_unchanged=True,exact_selected_component_preload_m=-.00005,
        native_qualification='full geometry plus actual chosen-source patch/normal match,all3Fn>0.1N and native cone6D',
        control_variants='constant q; inverse-affine PD only if strict full-command LP with no optimistic dry-friction allowance is feasible and raw command is legal',
        maximum_native_cases=16,maximum_integrations=153600,gates=GATES,dts=DTS,neural_updates=0))
    results=[];reach_count=0;integration_steps=0;tick=time.perf_counter()
    for index,item in enumerate(prior['selected']):
        model.opt.timestep=base_dt;data=mujoco.MjData(model);data.qpos[:]=q0;data.ctrl[:]=q0[:20];mujoco.mj_forward(model,data)
        R=data.xmat[bind['mug_body_id']].reshape(3,3).copy();origin=data.xpos[bind['mug_body_id']].copy();wp=p@R.T+origin;wn=n@R.T
        ids=np.array(item['source_candidate_ids'],int);mask=np.zeros(len(p),bool);mask[ids]=True
        if set(pg[ids])!={1,2,3}:raise ValueError('Explicit one source point per finger')
        folder=out/f'candidate{index:02d}';folder.mkdir();reach=[];chosen=[];initial=q0.copy()
        for digit,target in enumerate(ids,1):
            joints=np.arange((digit-1)*4,digit*4);rows=[]
            for a in anchors_by_group[digit]:
                for start in [q0[joints],.85*q0[joints]+.15*(lower[joints]+upper[joints])/2]:
                    problem=SurfacePatchIK(model,q0,joints,joints,a,wp[[target]],wn[[target]],[target]);answer=problem.solve(start,lower[joints],upper[joints],max_nfev=120);reach_count+=1
                    rows.append(dict(geom=a['geom'],**answer))
            best=min(rows,key=lambda r:r['proposal_cost']);chosen.append(next(a for a in anchors_by_group[digit] if a['geom']==best['geom']));initial[joints]=best['qpos'][joints];reach.extend(rows)
        write(folder/'reach.json',reach)
        problem=CollisionPatchIK(model,initial,audit,chosen,wp,wn,pg,mask)
        with (folder/'coupled_proposals.jsonl').open('w',encoding='utf8') as log:solution=problem.solve(log,maximum_seconds=30.,max_nfev=160)
        fitted=q0.copy();fitted[:20]=solution['best']['q20'];data.qpos[:]=fitted;mujoco.mj_forward(model,data)
        geometry=audit.evaluate(data);write(folder/'fitted_geometry.json',geometry);write(folder/'solver.json',solution)
        material=[{k:r[k] for k in ['source_id','source_distance_m','normal_dot']} for r in solution['best']['patches']]
        material_pass=all(r['source_distance_m']<=.003 and r['normal_dot']>=.5 for r in material)
        q=fitted.copy();preload=None;final_geometry=geometry
        if geometry['passed'] and material_pass:
            pairs=[min([r for r in geometry['rows'] if r['kind']=='object' and r['geom_ids'][1] in geoms and r.get('gap_m') is not None],key=lambda r:r['gap_m'])['geom_ids'] for geoms in groups.values()]
            q,preload=engage_pairs(model,q,audit,pairs,-.00005);write(folder/'preload.json',preload)
            data.qpos[:]=q;data.qvel[:]=0;data.ctrl[:]=q[:20];mujoco.mj_forward(model,data);final_geometry=audit.evaluate(data)
        write(folder/'final_geometry.json',final_geometry)
        if not np.array_equal(q[20:],q0[20:]):raise ValueError('Source-conditioned proposal changed object')
        data.qpos[:]=q;data.qvel[:]=0;data.ctrl[:]=q[:20];mujoco.mj_forward(model,data)
        snapshot=reader.read(data);rp=[];rn=[];rg=[]
        for ci in range(data.ncon):
            con=data.contact[ci];a,b=map(int,con.geom);oa,ob=a in parts,b in parts
            if oa==ob or con.efc_address<0:continue
            other=a if ob else b
            if other not in gids:continue
            force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
            if force[0]<=1e-8:continue
            rp.append(R.T@(con.pos-origin));rn.append(R.T@((1 if ob else -1)*con.frame.reshape(3,3)[0]));rg.append(gids[other])
        match=match_patches(np.array(rp).reshape(-1,3),np.array(rn).reshape(-1,3),rg,p,n,pg,mask)
        static=None;commands=[('constant_q',q[:20].copy())];command_unavailable=None
        if final_geometry['passed']:
            basis=contact_wrench_basis(model,data,sorted(parts),bind['mug_body_id'],gids);W,L=basis['wrench'],basis['loads'];target=np.r_[-model.body_mass[bind['mug_body_id']]*model.opt.gravity,np.zeros(3)]
            offset=data.qfrc_bias[:20]-data.qfrc_passive[:20];lo,hi=torque_box(model,data,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
            cone=static_feasibility(W,target);full=static_feasibility(W,target,loads=L,offset=offset,lower=lo,upper=hi,friction=np.zeros(20))
            static=dict(cone6D=cone,strict_full_command=full,EFC_wrench_error=basis['native_wrench_error'],EFC_load_error=basis['native_load_error'],EFC_columns=basis['native_EFC_unit_queries'])
            if full['success']:
                tau=np.array(full['witness']['required_torque20_Nm']);bias=model.actuator_biasprm[:,0]+model.actuator_biasprm[:,1]*data.actuator_length
                u=(tau-bias)/model.actuator_gainprm[:,0]
                if np.all(u>=model.actuator_ctrlrange[:,0]) and np.all(u<=model.actuator_ctrlrange[:,1]):commands.append(('inverse_PD_strict_LP',u))
                else:command_unavailable='Raw inverse PD command outside control interval; never projected'
            np.savez_compressed(folder/'basis.npz',wrench6xN=W,joint_load20xN=L,target6=target,offset20=offset,lower20=lo,upper20=hi)
        eligible=bool(final_geometry['passed'] and match['all_groups_matched'] and np.all(snapshot['normal_by_finger_N'][:3]>.1) and static['cone6D']['success'])
        np.savez_compressed(folder/'candidate.npz',qpos27=q,ctrl20=q[:20]);np.savez_compressed(folder/'fitted.npz',qpos27=fitted,ctrl20=fitted[:20]);cases=[]
        if eligible:
            for name,u in commands:
                (folder/name).mkdir()
                for dt in DTS:
                    result=evaluate_hold(model,q,u,bind,groups,dt,folder/name/str(dt));integration_steps+=result['completed_steps']
                    cases.append(dict(control=name,dt_s=dt,summary=result,case=file_pin(folder/name/str(dt)/'report.json'),trace=file_pin(folder/name/str(dt)/'trace.npz')))
        row=dict(index=index,selection=item,material_metrics=material,material_gate=material_pass,coupled_geometry_pass=geometry['passed'],geometry_pass=final_geometry['passed'],
            geometry=file_pin(folder/'final_geometry.json'),fitted_geometry=file_pin(folder/'fitted_geometry.json'),candidate=file_pin(folder/'candidate.npz'),fitted=file_pin(folder/'fitted.npz'),
            solver=file_pin(folder/'solver.json'),reach=file_pin(folder/'reach.json'),coupled_proposals=file_pin(folder/'coupled_proposals.jsonl'),
            actual_source_patch_matches=match,native_finger_normal_N=snapshot['normal_by_finger_N'][:3],static=static,native_eligible=eligible,native_cases=cases,command_unavailable=command_unavailable,
            geometry_failure_counts={kind:sum(not r['passed'] for r in final_geometry['rows'] if r['kind']==kind) for kind in ['object','self','hand_table']})
        if preload is not None:row['preload']=file_pin(folder/'preload.json')
        if static is not None:row['basis']=file_pin(folder/'basis.npz')
        write(folder/'result.json',row);results.append(row)
        print(json.dumps(dict(candidate=index,ids=ids.tolist(),material=material,geometry=row['geometry_pass'],actual_match=match['all_groups_matched'],native6D=None if static is None else static['cone6D']['success'],native_cases=len(cases),elapsed_s=time.perf_counter()-tick)),flush=True)
    if reach_count>72 or sum(len(r['native_cases']) for r in results)>16 or integration_steps>153600:raise ValueError('Declared fit budget exceeded')
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_support_selected_native_fit:1',sources=sources,inputs=[file_pin(reference/'binding.json'),file_pin(patches/'report.json'),file_pin(selection/'report.json'),file_pin(seed_path),file_pin(anchor_path)],
        results=results,reach_solves=reach_count,coupled_seeds=len(results),geometry_pass_count=sum(r['geometry_pass'] for r in results),
        native_eligible_count=sum(r['native_eligible'] for r in results),native_cases=sum(len(r['native_cases']) for r in results),integration_steps=integration_steps,neural_updates=0,wall_s=time.perf_counter()-tick,
        scope='Support-selected uncertain source triplets followed by bounded material-point, collision and engagement proposals; qualify actual native manifolds before hold integration. No label truth, calibrated dynamics/source gravity, global search, neural/gradient benefit or full trajectory claim.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass=report['geometry_pass_count'],eligible=report['native_eligible_count'],native_cases=report['native_cases'],integrations=integration_steps)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','patches','selection','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
