"""Two predeclared analytical object poses; static FK/query only, no optimizer."""
from pathlib import Path
import hashlib,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent;PILOT=ROOT.parent


def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)


def main():
    import numpy as np
    import mujoco
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(PILOT/'geometry_query_v2'))
    from convex_query import ConvexPolytope
    from query_interface_v2 import support_features,jsonable
    started=time.perf_counter();p=json.loads((ROOT/'protocol_v1.json').read_text(encoding='utf8'))
    if (ROOT/'report.json').exists():raise ValueError('Fresh probe only')
    for record in [*p['inputs'].values(),*p['source_pins'].values()]:assert pin(record['path'])==record
    support=json.loads(Path(p['inputs']['support_report']['path']).read_text(encoding='utf8'))
    assert support['geometry_recheck_passed'] is True and support['counts']['integration_steps']==0
    with np.load(p['inputs']['operands']['path'],allow_pickle=False) as a:operands={k:a[k].copy() for k in a.files}
    with np.load(p['inputs']['state']['path'],allow_pickle=False) as a:saved={k:a[k].copy() for k in a.files}
    assert np.array_equal(operands['qpos27'],saved['qpos27'])
    model=mujoco.MjModel.from_binary_path(p['inputs']['model']['path']);data=mujoco.MjData(model)
    oid=model.body('hot3d_phone24').id;pid=model.body('right_palm_link').id;phone=model.geom('phone24_collision_convex').id;table=model.geom('stage1_single_table').id
    q0=operands['qpos27'];points=operands['object_contact_points_world'];COM0=operands['object_COM_world'];gravity=model.opt.gravity.copy();mass=float(model.body_mass[oid]);origin=points[0];axis=points[1]-points[0];axis=axis/np.linalg.norm(axis)
    v=COM0-origin;vperp=v-axis*(v@axis);gperp=gravity-axis*(gravity@axis)
    if np.linalg.norm(vperp)<=1e-12 or np.linalg.norm(gperp)<=1e-12:theta=0.;rule_status='DEGENERATE_NO_UNIQUE_ALIGNMENT'
    else:theta=float(np.arctan2(axis@np.cross(vperp,gperp),vperp@gperp));rule_status='UNIQUE_SHORTEST_PROPER_AXIS_ROTATION'
    R0=Rotation.from_quat(q0[23:27][[1,2,3,0]]).as_matrix();t0=q0[20:23];material=(points-t0)@R0
    Q=Rotation.from_rotvec(theta*axis).as_matrix();R1=Q@R0;t1=origin+Q@(t0-origin)
    tips=[model.site(f'right_finger{i}_tip').id for i in range(1,6)];pads=[model.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)]
    hand=[g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g]) and model.geom_bodyid[g]!=oid and g!=table]
    targets=json.loads(Path(p['inputs']['human_targets']['path']).read_text(encoding='utf8'));htips=np.array([m['reference_tip_object_m'] for m in targets['mapping']]);anchors=json.loads(Path(p['inputs']['anchors']['path']).read_text(encoding='utf8'))['preparation']
    def mesh(g):
        mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);return model.mesh_vert[a:a+n].astype(np.float64)
    phone_local=mesh(phone);local={g:mesh(g) for g in hand if int(model.geom_type[g])==7};cases=[];counts={'MJB_reads':1,'MjData':1,'mj_kinematics':0,'independent_heavy_pair_queries_attempted':0,'independent_heavy_pair_queries_returned':0,'optimization_integration_training_GPU':0}
    base_hand_pose=None
    for case_name,R,t,angle in [('original',R0,t0,0.),('gravity_line_aligned',R1,t1,theta)]:
        data.qpos[:]=q0;data.qpos[20:23]=t;quat=Rotation.from_matrix(R).as_quat();data.qpos[23:27]=quat[[3,0,1,2]];data.qvel[:]=0;data.ctrl[:]=0;mujoco.mj_kinematics(model,data);counts['mj_kinematics']+=1
        actualR=data.xmat[oid].reshape(3,3).copy();actualt=data.xpos[oid].copy();COM=actualt+actualR@model.body_ipos[oid];fixed_material=material@actualR.T+actualt
        hp=np.array([data.geom_xpos[g] for g in hand]);hr=np.array([data.geom_xmat[g] for g in hand])
        if base_hand_pose is None:base_hand_pose=(hp.copy(),hr.copy(),data.xpos[pid].copy(),data.xquat[pid].copy())
        same_hand=np.array_equal(hp,base_hand_pose[0]) and np.array_equal(hr,base_hand_pose[1]) and np.array_equal(data.xpos[pid],base_hand_pose[2]) and np.array_equal(data.xquat[pid],base_hand_pose[3])
        assert same_hand and np.array_equal(data.qpos[:20],q0[:20])
        pv=phone_local@data.geom_xmat[phone].reshape(3,3).T+data.geom_xpos[phone];poly=ConvexPolytope(pv);queryrows=[];unknown=[]
        for g in hand+[table]:
            gr=data.geom_xmat[g].reshape(3,3);gp=data.geom_xpos[g];kind=int(model.geom_type[g]);counts['independent_heavy_pair_queries_attempted']+=1
            try:
                if kind==3:
                    a=gp-model.geom_size[g,1]*gr[:,2];b=gp+model.geom_size[g,1]*gr[:,2];r=float(model.geom_size[g,0]);q=poly.capsule(a,b,r);q.update(support_features(q,pv,a=a,b=b,radius=r))
                else:
                    if kind==6:
                        from itertools import product
                        hv=np.array(list(product(*[(-s,s) for s in model.geom_size[g]])))@gr.T+gp
                    elif kind==7:hv=local[g]@gr.T+gp
                    else:raise ValueError('Unsupported collider')
                    q=poly.convex(hv,max_pair_points=4_000_000);q.update(support_features(q,pv,hand_vertices=hv))
                counts['independent_heavy_pair_queries_returned']+=1;queryrows.append({'geom_id':g,'name':model.geom(g).name,'is_candidate_pad':g in pads,'gap_m':q['signed_translation_margin_m'],'query':jsonable(q)})
            except Exception as error:unknown.append({'geom_id':g,'name':model.geom(g).name,'error':f'{type(error).__name__}: {error}'})
        actualanchors=np.array([np.asarray(a['anchor_geom_local_m'])@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g] for a,g in zip(anchors,pads)]);desiredanchors=np.array([a['target_projected_phone_convex_object_m'] for a in anchors])@actualR.T+actualt;desiredtips=htips@actualR.T+actualt
        torque=np.cross(COM-origin,mass*gravity);axial=float(axis@torque);perp=COM-origin-axis*((COM-origin)@axis)
        state=ROOT/f'{case_name}_state.npz';np.savez_compressed(state,qpos27=data.qpos.copy(),qvel26=data.qvel.copy(),ctrl20=data.ctrl.copy(),body_xpos=data.xpos.copy(),body_xquat=data.xquat.copy(),geom_xpos=data.geom_xpos.copy(),geom_xmat=data.geom_xmat.copy(),site_xpos=data.site_xpos.copy(),COM_world=COM,object_material_points_world=fixed_material)
        complete=len(queryrows)==22 and not unknown;padok=complete and all(q['query']['margin_roundoff_guarded_interval_m'][0]>=0 and q['query']['margin_roundoff_guarded_interval_m'][1]<=.0002 for q in queryrows if q['is_candidate_pad']);nonpadok=complete and all(q['query']['margin_roundoff_guarded_interval_m'][0]>=0 for q in queryrows if not q['is_candidate_pad'])
        cases.append({'name':case_name,'rotation_about_world_line_rad':angle,'rotation_about_world_line_deg':float(np.rad2deg(angle)),'proper_rotation_determinant':float(np.linalg.det(actualR)),'COM_world_m':COM.tolist(),'gravity_torque_about_line_Nm_signed':axial,'gravity_torque_about_line_Nm_abs':abs(axial),'gravity_potential_J':float(-mass*gravity@COM),'COM_perpendicular_line_m':perp.tolist(),'material_point_residuals_m':(fixed_material-points).tolist(),'max_material_point_residual_m':float(np.linalg.norm(fixed_material-points,axis=1).max()),'robot_q20_and_all_hand_geom_poses_unchanged':same_hand,'self189_evidence_reused_not_rerun':True,'all22_queries_complete':complete,'pad_gap_window_pass':padok,'all_nonpad_nonpenetration_pass':nonpadok,'deepest_penetration_m':max([max(0.,-q['gap_m']) for q in queryrows],default=0.),'queries':queryrows,'unknown':unknown,'all5_anatomical_tip_error_m':np.linalg.norm(data.site_xpos[tips]-desiredtips,axis=1).tolist(),'thumb_index_anchor_error_m':np.linalg.norm(actualanchors-desiredanchors,axis=1).tolist(),'state':pin(state)})
    result={'schema':'single_analytic_gravity_line_alignment_probe:1','status':'COMPLETED_TWO_STATIC_CANDIDATES_NOT_EGO_REFERENCE_OR_HOLDING_SUCCESS','protocol':pin(ROOT/'protocol_v1.json'),'source':pin(__file__),'rotation_rule_status':rule_status,'world_line_origin_m':origin.tolist(),'world_line_unit_axis':axis.tolist(),'material_contact_points_world_m':points.tolist(),'gravity_world_m_s2':gravity.tolist(),'mass_kg_assumed':mass,'cases':cases,'counts':counts,'self_evidence_reuse':{'source':p['inputs']['self_recheck'],'eligible_pairs':189,'reason':'All hand joint and all compiledhand geomworldposes bitidentical inbothcandidates; only objectSE3changed. Source189 self gate remains applicable.'},'interpretation':'New gravity-aware object pose deviates from originalhuman reference; only axialgravity torque minimized. Normal directions/contact topology may change and collisions may beintroduced. Zero line torque does not prove fullfrictionwrench feasibility, holding, or ego methodsuccess.','elapsed_seconds':time.perf_counter()-started}
    result['outputs']={x.name:pin(x) for x in ROOT.iterdir() if x.is_file() and x.name!='report.json'};write(ROOT/'report.json',result);print(json.dumps({'report':pin(ROOT/'report.json'),'rotation_deg':cases[1]['rotation_about_world_line_deg'],'old_line_torque':cases[0]['gravity_torque_about_line_Nm_signed'],'new_line_torque':cases[1]['gravity_torque_about_line_Nm_signed'],'deepest_new_penetration_mm':cases[1]['deepest_penetration_m']*1000,'counts':counts}))


if __name__=='__main__':main()
