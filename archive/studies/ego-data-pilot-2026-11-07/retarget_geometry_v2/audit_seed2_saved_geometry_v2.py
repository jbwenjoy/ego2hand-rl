"""Separate saved-data author recheck, not an independent investigator audit.

One replay kinematics call checks captured poses. Saved separating planes verify
all phone pairs; only the17 near self pairs need fresh static geometry arithmetic.
No optimization, force calculation, integration, training or native distances.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent;PILOT=ROOT.parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    import numpy as np
    import mujoco
    sys.path.insert(0,str(PILOT/'geometry_query_v2'))
    from convex_query import ConvexPolytope
    protocol=json.loads((ROOT/'protocol_v2.json').read_text(encoding='utf8'));report=json.loads((ROOT/'report.json').read_text(encoding='utf8'));result=report['seed_results'][2]
    if (ROOT/'seed2_saved_gate_recheck_v2.json').exists():raise ValueError('Fresh audit only')
    for p in protocol['inputs'].values():assert pin(p['path'])==p
    for p in protocol['source_pins'].values():assert pin(p['path'])==p
    ls=importlib.util.spec_from_file_location('audit_exact_loader',protocol['inputs']['loader']['path']);loader=importlib.util.module_from_spec(ls);ls.loader.exec_module(loader);model=loader.load_model();initial=loader.initial_state()
    assert pin(result['selected_state']['path'])==result['selected_state']
    with np.load(result['selected_state']['path'],allow_pickle=False) as a:saved={k:a[k].copy() for k in a.files}
    data=mujoco.MjData(model);data.qpos[:]=saved['qpos27'];data.qvel[:]=0;data.ctrl[:]=0;mujoco.mj_kinematics(model,data)
    replay={'geom_xpos':bool(np.array_equal(data.geom_xpos,saved['geom_xpos'])),'geom_xmat':bool(np.array_equal(data.geom_xmat,saved['geom_xmat'])),'site_xpos':bool(np.array_equal(data.site_xpos,saved['site_xpos'])),'body_xpos':bool(np.array_equal(data.xpos,saved['body_xpos'])),'body_xquat':bool(np.array_equal(data.xquat,saved['body_xquat']))}
    pid=model.body('right_palm_link').id;oid=model.body('hot3d_phone24').id;pg=model.geom('phone24_collision_convex').id;table=model.geom('stage1_single_table').id
    # Derive hand membership from the compiled body tree, not producer name masks.
    descendants={pid}
    for body in range(model.nbody):
        current=body
        while current and current not in descendants:current=int(model.body_parentid[current])
        if current in descendants:descendants.add(body)
    all_hand=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants]
    active=[g for g in all_hand if model.geom_contype[g] or model.geom_conaffinity[g]]
    inactive=[g for g in all_hand if not(model.geom_contype[g] or model.geom_conaffinity[g])]
    pads=[model.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)]
    phone_rows={r['geom_id']:r for r in result['verification']['all_phone_pairs']}
    R=saved['phone_R_world'];p=saved['phone_p_world'];raw_geometry={}
    def shape(g,phone_frame=False):
        gr=saved['geom_xmat'][g].reshape(3,3);gp_=saved['geom_xpos'][g];kind=int(model.geom_type[g])
        if kind==3:
            a=gp_-model.geom_size[g,1]*gr[:,2];b=gp_+model.geom_size[g,1]*gr[:,2]
            if phone_frame:a=(a-p)@R;b=(b-p)@R
            return {'type':'capsule','a':a,'b':b,'r':float(model.geom_size[g,0])}
        if kind==7:
            mid=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);v=model.mesh_vert[start:start+n].astype(np.float64)@gr.T+gp_
        elif kind==6:
            from itertools import product
            v=np.array(list(product(*[(-x,x) for x in model.geom_size[g]])))@gr.T+gp_
        else:raise ValueError('Unsupported collider')
        if phone_frame:v=(v-p)@R
        return {'type':'convex','v':v}
    def bounds(s):
        return np.array([np.minimum(s['a'],s['b'])-s['r'],np.maximum(s['a'],s['b'])+s['r']]) if s['type']=='capsule' else np.array([s['v'].min(0),s['v'].max(0)])
    def lower(a,b):return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0)))
    phonev=shape(pg)['v'];phone_checks=[]
    for g in active+[table]:
        row=phone_rows[g];s=shape(g);n=np.asarray(row['query']['normal_world']);o_max=float((phonev@n).max())
        h_min=float(min(s['a']@n,s['b']@n)-s['r']) if s['type']=='capsule' else float((s['v']@n).min())
        reconstructed=h_min-o_max;guard=float(row['query']['roundoff_guard_m'])
        phone_checks.append({'geom_id':g,'name':model.geom(g).name,'saved_gap_m':row['gap_m'],'direct_full_geometry_support_gap_m':reconstructed,'support_gap_error_m':abs(reconstructed-row['gap_m']),'guard_m':guard,'interval_exactly_margin_plusminus_guard':row['interval_m']==[row['gap_m']-guard,row['gap_m']+guard],'guarded_lower_ge_zero':row['interval_m'][0]>=0,'pad_upper_le_200um':g not in pads or row['interval_m'][1]<=.0002})
        if s['type']=='capsule':raw_geometry[f'geom{g}_capsule_world_a_b_r']=np.r_[s['a'],s['b'],s['r']]
        else:raw_geometry[f'geom{g}_vertices_world_m']=s['v']
    raw_geometry['phone_vertices_world_m']=phonev
    # Independent eligibility expression: masks first, then compiled bodies/welds.
    explicit={tuple(sorted([int(a),int(b)])) for a,b in zip(model.pair_geom1,model.pair_geom2)};ex=set(int(x) for x in model.exclude_signature);parent_filter=not bool(int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    expected=set();explanations=[]
    for i,g in enumerate(active):
        for h in active[i+1:]:
            b1,b2=int(model.geom_bodyid[g]),int(model.geom_bodyid[h]);w1,w2=int(model.body_weldid[b1]),int(model.body_weldid[b2]);pw1=int(model.body_weldid[model.body_parentid[w1]]);pw2=int(model.body_weldid[model.body_parentid[w2]])
            pair=tuple(sorted([g,h]));mask=bool((int(model.geom_contype[g])&int(model.geom_conaffinity[h])) or (int(model.geom_contype[h])&int(model.geom_conaffinity[g])))
            is_ex=((min(b1,b2)<<16)|max(b1,b2)) in ex
            is_parent=parent_filter and w1!=0 and w2!=0 and (w1==pw2 or w2==pw1)
            eligible=pair in explicit or (mask and w1!=w2 and b1!=b2 and not is_ex and not is_parent)
            if eligible:expected.add(pair)
            explanations.append({'pair':list(pair),'bodies':[b1,b2],'welds':[w1,w2],'weld_parents':[pw1,pw2],'explicit':pair in explicit,'mask':mask,'excluded_signature':is_ex,'parent_filtered':is_parent,'eligible':eligible})
    producer_self={tuple(row['geom_ids']):row for row in result['verification']['self_pairs']}
    near=[];far=[];heavy=0;analytic=0
    def segment_distance_by_candidates(a,b,c,d):
        u=b-a;v=d-c;r=a-c;A=float(u@u);E=float(v@v);B=float(u@v);C=float(u@r);F=float(v@r);st=[(0.,0.),(0.,1.),(1.,0.),(1.,1.)]
        if E>0:st.extend([(0.,float(np.clip(F/E,0,1))),(1.,float(np.clip((F+B)/E,0,1)))])
        if A>0:st.extend([(float(np.clip(-C/A,0,1)),0.),(float(np.clip((B-C)/A,0,1)),1.)])
        H=np.array([[A,-B],[-B,E]])
        if abs(np.linalg.det(H))>1e-30:
            point=np.linalg.solve(H,np.array([-C,F]))
            if ((point>=0)&(point<=1)).all():st.append(tuple(point))
        return min(float(np.linalg.norm(r+s*u-t*v)) for s,t in st)
    for pair in sorted(expected):
        old=producer_self[pair];g,h=pair;a=shape(g,True);b=shape(h,True);ab=lower(bounds(a),bounds(b))
        if old['method']=='AABB_DISJOINT_CERTIFICATE':
            far.append({'pair':list(pair),'recomputed_lower_m':ab,'stored_lower_m':old['lower_m'],'agreement_error_m':abs(ab-old['lower_m']),'proper_disjoint_certificate':ab>.002,'not_an_exact_gap':True});continue
        # The independent coordinate reconstruction is world-frame here.
        a=shape(g);b=shape(h)
        if a['type']==b['type']=='capsule':
            distance=segment_distance_by_candidates(a['a'],a['b'],b['a'],b['b'])-a['r']-b['r'];guard=256*np.finfo(float).eps;analytic+=1
        else:
            heavy+=1
            if a['type']=='convex':q=ConvexPolytope(a['v']).capsule(b['a'],b['b'],b['r']) if b['type']=='capsule' else ConvexPolytope(a['v']).convex(b['v'],max_pair_points=4_000_000)
            else:q=ConvexPolytope(b['v']).capsule(a['a'],a['b'],a['r'])
            distance=float(q['signed_translation_margin_m']);guard=float(q['roundoff_guard_m'])
        near.append({'pair':list(pair),'names':old['names'],'AABB_lower_phone_frame_m':ab,'recomputed_world_frame_signed_m':distance,'saved_phone_frame_signed_m':old['signed_gap_m'],'absolute_frame_difference_m':abs(distance-old['signed_gap_m']),'recomputed_guarded_lower_m':distance-guard,'strict_nonpenetration':distance-guard>=0})
    initialpalm=initial['T_world_from_palm'];checks={'saved_qpos_replays_all_pose_arrays_bitexact':all(replay.values()),'all20_q_within_true_compiled_hard_limits':bool(((saved['qpos27'][:20]>=model.jnt_range[:20,0])&(saved['qpos27'][:20]<=model.jnt_range[:20,1])).all()),'fixed_palm_has_no_joint_or_mocap':int(model.body_jntnum[pid])==0 and int(model.body_mocapid[pid])==-1,'saved_fixed_palm_matches_source':bool(np.array_equal(saved['body_xpos'][pid],initialpalm[:3,3])),'proper_phone_rotation':abs(float(np.linalg.det(R))-1)<1e-12 and float(np.max(abs(R.T@R-np.eye(3))))<1e-12,'all_active_hand_plus_table_phone_pairs_exact_covered':set(phone_rows)==set(active+[table]),'all_phone_margin_intervals_exact':all(x['interval_exactly_margin_plusminus_guard'] for x in phone_checks),'all_phone_lower_ge_zero_and_pad_upper_le_200um':all(x['guarded_lower_ge_zero'] and x['pad_upper_le_200um'] for x in phone_checks),'world_support_planes_match_full_actual_colliders':all(x['support_gap_error_m']<=4*x['guard_m'] for x in phone_checks),'all_eligible_self_pairs_exactly_accounted':set(producer_self)==expected,'far_AABB_certificates_direction_and_threshold_valid':all(x['proper_disjoint_certificate'] and x['agreement_error_m']<1e-12 for x in far),'every_near_self_pair_fresh_static_arithmetic_nonpenetrating':all(x['strict_nonpenetration'] and x['AABB_lower_phone_frame_m']<=.002 for x in near),'producer_unknown_list_empty':not result['verification']['unknown'],'inactive_geoms_only_excluded_when_both_masks_zero':all(not(model.geom_contype[g] or model.geom_conaffinity[g]) for g in inactive)}
    np.savez_compressed(ROOT/'seed2_audit_geometry_inputs_v2.npz',**raw_geometry)
    audit={'schema':'retarget_v2_saved_gate_author_recheck:1','status':'PASS_SAVED_GATES_RECHECK_NOT_HOLDING' if all(checks.values()) else 'FAIL_PRESERVED','reviewer_scope':'Same agent authored producer; this is a separate saved-data/coordinate reconstruction recheck, NOT an independent investigator audit. Coregeometry fixtures and any peer sourcecheck remain separately attributed.','source':pin(__file__),'producer_report':pin(ROOT/'report.json'),'producer_source':pin(ROOT/'solve_contact_first_v2.py'),'protocol':pin(ROOT/'protocol_v2.json'),'selected_state':result['selected_state'],'checks':checks,'passed':sum(checks.values()),'total':len(checks),'pose_replay_checks':replay,'counts':{'passive_MJB_reads':1,'MjData_instances':1,'mj_kinematics':1,'fresh_heavy_near_self_queries':heavy,'fresh_analytic_capsule_pairs':analytic,'optimization_mj_step_training_GPU':0},'coverage':{'all_hand_geoms':len(all_hand),'active_hand_geoms':len(active),'disabled_hand_geoms':len(inactive),'exact_phone_pairs':len(phone_checks),'possible_active_self_pairs':len(explanations),'eligible_self_pairs':len(expected),'far_disjoint_AABB_certificates':len(far),'near_recomputed_pairs':len(near),'explicit_self_pairs':len(explicit)},'phone_pair_support_checks':phone_checks,'self_eligibility':explanations,'far_self_certificates':far,'near_self_rechecks':near,'mesh_semantics':'Each mesh is the ideal convex envelope of ALL compiled mesh_vert source points. This is not its possiblyconcave visual triangle surface, original HOT3D scan distance, formal interval proof, or nativeCCD convergence certificate. Capsule overlay is the unchanged source approximation.','filter_semantics':'Compiled weld-parent rule only if both weld IDs nonzero; same-weld always filtered. Explicitpairs separate; compiled exclude_signature and geom mask respected. Inactive visual/disabledcolliders do not become new collision targets. Sleep does not suppress geometric pairs.','not_checked':'No exhaustive hand/table check beyond producer phone/table scope; no continuous swept motion, force support, compliant pad semantics, or holding dynamics.','controls_warning':'Saved ctrl20 is ZERO from puregeometry. It is not a holding command. A future explicitholding protocol must choose and document absolute actuator targets, gains/forceclamps/contactpairs; do not accidentally reusezeroctrl as optimizedqhold.','thresholds_unchanged':True,'plan_files':'Root separately confirmedv20/v21 normal sizes; no docs read or modified for that canceledinvestigation.','geometry_input_arrays':pin(ROOT/'seed2_audit_geometry_inputs_v2.npz')}
    write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__))),encoding='utf8')
    write(ROOT/'seed2_saved_gate_recheck_v2.json',audit)
    print(json.dumps({'report':pin(ROOT/'seed2_saved_gate_recheck_v2.json'),'status':audit['status'],'passed':audit['passed'],'total':audit['total'],'counts':audit['counts']}))
    if not all(checks.values()):raise ValueError('Gate recheck failure preserved')


if __name__=='__main__':main()
