"""Fixed-wrist, contact-first static candidates using independent convex queries.

No mj_geomDistance, integration, controller, training or GPU. All inputs/settings
are frozen in protocol_v2.json before the one bounded execution.
"""
from pathlib import Path
import hashlib, importlib.util, json, os, sys, time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent


def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)


class BudgetStop(RuntimeError):pass
class UnknownGeometry(RuntimeError):pass


def segment_distance(a,b,c,d,np):
    """Exact analytic closest finite segments, including parallel endpoints."""
    u=b-a;v=d-c;r=a-c;A=float(u@u);E=float(v@v);F=float(v@r)
    if A<=1e-30 and E<=1e-30:return float(np.linalg.norm(a-c)),a,c
    if A<=1e-30:s=0.;t=float(np.clip(F/E,0,1))
    else:
        C=float(u@r)
        if E<=1e-30:t=0.;s=float(np.clip(-C/A,0,1))
        else:
            B=float(u@v);den=A*E-B*B
            s=float(np.clip((B*F-C*E)/den,0,1)) if den>1e-30 else 0.
            t=(B*s+F)/E
            if t<0:t=0.;s=float(np.clip(-C/A,0,1))
            elif t>1:t=1.;s=float(np.clip((B-C)/A,0,1))
    x=a+s*u;y=c+t*v
    return float(np.linalg.norm(x-y)),x,y


def main():
    execution_started=time.perf_counter()
    import numpy as np
    import mujoco
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(PILOT/'geometry_query_v2'))
    from convex_query import ConvexPolytope,signed_origin_polytope,jsonable
    from query_interface_v2 import support_features
    protocol=json.loads((ROOT/'protocol_v2.json').read_text(encoding='utf8'))
    if (ROOT/'report.json').exists():raise ValueError('Fresh v2 only')
    for p in protocol['source_pins'].values():
        if pin(p['path'])!=p:raise ValueError('Changed source')
    for p in protocol['inputs'].values():
        if pin(p['path'])!=p:raise ValueError('Changed input')
    # Three analytic self-capsule contracts; no model or dynamics.
    z=np.zeros(3);one=np.array([1.,0,0]);up=np.array([0.,1.,0])
    assert abs(segment_distance(z,one,up,one+up,np)[0]-1)<1e-14
    assert segment_distance(z,one,np.array([.5,-1,0]),np.array([.5,1,0]),np)[0]==0
    assert abs(segment_distance(z,one,2*one,3*one,np)[0]-1)<1e-14
    loader_spec=importlib.util.spec_from_file_location('contact_first_exact_loader',protocol['inputs']['loader']['path']);loader=importlib.util.module_from_spec(loader_spec);loader_spec.loader.exec_module(loader)
    model=loader.load_model();data=mujoco.MjData(model);initial=loader.initial_state()
    target=json.loads(Path(protocol['inputs']['targets']['path']).read_text(encoding='utf8'))
    parent=json.loads(Path(protocol['inputs']['old_report']['path']).read_text(encoding='utf8'))
    qlo=model.jnt_range[:20,0].copy();qhi=model.jnt_range[:20,1].copy();q0=initial['qpos'][:20].copy()
    pid=model.body('right_palm_link').id;oid=model.body('hot3d_phone24').id;phone=model.geom('phone24_collision_convex').id;table=model.geom('stage1_single_table').id
    Rp=initial['T_world_from_palm'][:3,:3];pp=initial['T_world_from_palm'][:3,3]
    tip_sites=[model.site(f'right_finger{i}_tip').id for i in range(1,6)]
    pad_ids=[model.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)]
    hand=[g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g]) and model.geom_bodyid[g]!=oid and g!=table]
    all_phone_pairs=hand+[table]
    def Rquat(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
    def local_mesh(g):
        mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid])
        return model.mesh_vert[a:a+n].astype(np.float64).copy()
    local={};metadata={}
    for g in all_phone_pairs+[phone]:
        kind=int(model.geom_type[g]);info={'id':g,'name':model.geom(g).name,'body':model.body(int(model.geom_bodyid[g])).name,'type':kind}
        if kind==7:
            local[g]=local_mesh(g);cp=ConvexPolytope(local[g]);info.update(source_compiled_vertices=len(local[g]),exact_hull_vertices=len(cp.vertices),vertex_decimation=False)
        elif kind==6:
            from itertools import product
            local[g]=np.array(list(product(*[(-x,x) for x in model.geom_size[g]])))
        elif kind!=3:raise ValueError('Unsupported active collider; no fallback')
        metadata[g]=info
    phone_body_vertices=local[phone]@Rquat(model.geom_quat[phone]).T+model.geom_pos[phone]
    poly=ConvexPolytope(phone_body_vertices);phone_box=np.array([phone_body_vertices.min(0),phone_body_vertices.max(0)])
    anchors=[];contact_targets=[];prep=[]
    for finger,g in enumerate(pad_ids):
        body=int(model.geom_bodyid[g]);br=Rquat(initial['body_xquat'][body]);gp=initial['body_xpos'][body]+br@model.geom_pos[g];gr=br@Rquat(model.geom_quat[g])
        site=initial['site_xpos'][tip_sites[finger]];site_local=(site-gp)@gr
        projection=signed_origin_polytope(local[g]-site_local);anchor=site_local+projection['closest_unrounded_boundary_world']
        human=np.asarray(target['mapping'][finger]['nearest_scan_surface_point_object_m']);hp=signed_origin_polytope(phone_body_vertices-human);surface=human+hp['closest_unrounded_boundary_world']
        anchors.append(anchor);contact_targets.append(surface)
        prep.append({'finger':target['mapping'][finger]['finger'],'geom':metadata[g],'anchor_geom_local_m':anchor.tolist(),'site_geom_local_m':site_local.tolist(),'site_to_anchor_m':float(np.linalg.norm(anchor-site_local)),'anchor_definition':'Source q0 existing tip site projected to actual compiled sensor convex boundary. Fixed geometric anchor, no compliant/soft-pad semantics.','human_scan_target_object_m':human.tolist(),'target_projected_phone_convex_object_m':surface.tolist(),'projection_displacement_m':float(np.linalg.norm(surface-human))})
    anchors=np.asarray(anchors);contact_targets=np.asarray(contact_targets);alltips=np.array([r['reference_tip_object_m'] for r in target['mapping']])
    write(ROOT/'fixed_anchors_and_targets.json',{'schema':'fixed_contact_first_geometry_anchors:1','preparation':prep,'phone_body_vertices':phone_body_vertices.tolist(),'no_reflection_or_scale':True})
    bounds=np.array(protocol['bounds']['phone_translation_palm_m']);scale=np.r_[np.ones(20),np.full(3,.1),np.ones(3)]
    lo=np.r_[qlo,bounds[0],[-np.pi]*3]/scale;hi=np.r_[qhi,bounds[1],[np.pi]*3]/scale
    step=np.r_[np.full(20,1e-4),np.full(3,1e-5),np.full(3,1e-4)]/scale
    counters={'mj_kinematics':0,'heavy_Minkowski_attempted':0,'heavy_Minkowski_returned':0,'heavy_unknown':0,'exact_query_cache_hits':0,'AABB_separation_certificates':0,'analytic_capsule_capsule':0}
    model_before=hashlib.sha256(b''.join(np.ascontiguousarray(getattr(model,n)).tobytes() for n in ('jnt_range','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','dof_armature','dof_damping','dof_frictionloss','body_pos','body_quat'))).hexdigest()
    job_start=execution_started;job_deadline=job_start+180.;seed_deadline=job_deadline;optimization_deadline=job_deadline;query_cache={};query_log=[];evaluations=[];callbacks=[];seed_results=[];active_seed=-1;current_phase='OPTIMIZATION'
    def budget():
        now=time.perf_counter()
        deadline=optimization_deadline if current_phase=='OPTIMIZATION' else seed_deadline
        if now>=min(deadline,job_deadline):raise BudgetStop('Time deadline')
        if counters['heavy_Minkowski_attempted']>=6000:raise BudgetStop('6000 heavy-query cap')
    def box(shape):
        if shape['kind']=='capsule':return np.array([np.minimum(shape['a'],shape['b'])-shape['radius'],np.maximum(shape['a'],shape['b'])+shape['radius']])
        return np.array([shape['vertices'].min(0),shape['vertices'].max(0)])
    def bbox_lower(a,b):return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0)))
    def shape_phone_frame(g,R,p):
        gr=data.geom_xmat[g].reshape(3,3);gp=data.geom_xpos[g]
        if int(model.geom_type[g])==3:
            a=gp-model.geom_size[g,1]*gr[:,2];b=gp+model.geom_size[g,1]*gr[:,2]
            return {'kind':'capsule','a':(a-p)@R,'b':(b-p)@R,'radius':float(model.geom_size[g,0])}
        return {'kind':'convex','vertices':(local[g]@gr.T+gp-p)@R}
    def exact(shape,tag):
        budget();key=(shape['kind'],(np.r_[shape['a'],shape['b'],shape['radius']] if shape['kind']=='capsule' else shape['vertices']).tobytes())
        if key in query_cache:
            counters['exact_query_cache_hits']+=1
            q=query_cache[key]
            if 'error' in q:raise UnknownGeometry(q['error'])
            return q
        counters['heavy_Minkowski_attempted']+=1;t=time.perf_counter()
        try:
            if shape['kind']=='capsule':
                q=poly.capsule(shape['a'],shape['b'],shape['radius']);feat=support_features(q,phone_body_vertices,a=shape['a'],b=shape['b'],radius=shape['radius'])
            else:
                q=poly.convex(shape['vertices'],max_pair_points=4_000_000);feat=support_features(q,phone_body_vertices,hand_vertices=shape['vertices'])
            q={**q,**feat};query_cache[key]=q;counters['heavy_Minkowski_returned']+=1
            query_log.append({'seed':active_seed,'phase':current_phase,'tag':tag,'seconds':time.perf_counter()-t,'query_kind':q['query_kind'],'signed_gap_m':q['signed_translation_margin_m'],'input_points':q['input_points'],'hull_vertices':q['hull_vertices'],'max_pair_points':4_000_000})
            return q
        except Exception as e:
            error=f'{type(e).__name__}: {e}';query_cache[key]={'error':error};counters['heavy_unknown']+=1;query_log.append({'seed':active_seed,'phase':current_phase,'tag':tag,'error':error,'seconds':time.perf_counter()-t});raise UnknownGeometry(error)
    def query_pair(g,R,p,force=False):
        shape=shape_phone_frame(g,R,p);lower=bbox_lower(phone_box,box(shape));guard=256*np.finfo(float).eps*max(1.,float(np.abs(box(shape)).max()))
        if not force and g not in pad_ids and lower>protocol['acceleration']['AABB_exact_switch_m']:
            counters['AABB_separation_certificates']+=1
            return {'kind':'AABB_SEPARATION_CERTIFICATE_NOT_EXACT_GAP','lower_m':lower-guard,'upper_m':None,'signed_m':None,'aabb_lower_m':lower,'guard_m':guard}
        q=exact(shape,model.geom(g).name)
        return {'kind':'INDEPENDENT_EXACT_CONVEX_QUERY_NUMERICAL','lower_m':float(q['margin_roundoff_guarded_interval_m'][0]),'upper_m':float(q['margin_roundoff_guarded_interval_m'][1]),'signed_m':float(q['signed_translation_margin_m']),'guard_m':float(q['roundoff_guard_m']),'query':q}
    def set_state(x):
        budget();R=Rp@Rotation.from_rotvec(x[23:]).as_matrix();p=pp+Rp@x[20:23]
        data.qpos[:]=initial['qpos'];data.qpos[:20]=x[:20];data.qpos[20:23]=p;quat=Rotation.from_matrix(R).as_quat();data.qpos[23:27]=quat[[3,0,1,2]];data.qvel[:]=0;data.ctrl[:]=0;data.time=0
        mujoco.mj_kinematics(model,data);counters['mj_kinematics']+=1
        if not np.array_equal(data.xpos[pid],pp) or not np.array_equal(data.xmat[pid].reshape(3,3),Rp):raise ValueError('Fixed palm changed')
        return R,p
    def world_query(q,R,p):
        out=jsonable(q);out['raw_query_phone_body_frame']=jsonable(q);out['query_input_frame']='phone_model_body';out['reported_support_frame']='world'
        for key in ('translation_normal_world','normal_world','closest_unrounded_boundary_world'):
            if key in q:out[key]=(R@np.asarray(q[key])).tolist()
        for key in ('object_support_points_world_m','hand_support_points_world_m'):
            if key in q:out[key]=(np.asarray(q[key])@R.T+p).tolist()
        projection=float((R@np.asarray(q['normal_world']))@p)
        for key in ('object_plane_max_projection_m','hand_plane_min_projection_m'):
            if key in q:out[key]=float(q[key])+projection
        return out
    # source-backed self-pair eligibility, plus explicitly declared pairs.
    exclusions=set(int(x) for x in model.exclude_signature)
    explicit={tuple(sorted((int(a),int(b)))) for a,b in zip(model.pair_geom1,model.pair_geom2)}
    eligible=[];filtered=[];dsbl=bool(int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    for ii,g in enumerate(hand):
        for h in hand[ii+1:]:
            b1,b2=int(model.geom_bodyid[g]),int(model.geom_bodyid[h]);w1,w2=int(model.body_weldid[b1]),int(model.body_weldid[b2]);pw1=int(model.body_weldid[model.body_parentid[w1]]);pw2=int(model.body_weldid[model.body_parentid[w2]])
            reason=None
            if (g,h) not in explicit:
                if b1==b2 or w1==w2:reason='same_body_or_weld'
                elif not dsbl and w1 and w2 and (w1==pw2 or w2==pw1):reason='compiled_weld_parent_filter'
                elif ((min(b1,b2)<<16)+max(b1,b2)) in exclusions:reason='compiled_exclude_signature'
                elif not ((int(model.geom_contype[g])&int(model.geom_conaffinity[h])) or (int(model.geom_contype[h])&int(model.geom_conaffinity[g]))):reason='geom_masks'
            record={'geom_ids':[g,h],'names':[model.geom(g).name,model.geom(h).name],'body_ids':[b1,b2],'weld_ids':[w1,w2],'explicit_pair':(g,h) in explicit,'filter_reason':reason}
            (filtered if reason else eligible).append(record)
    write(ROOT/'self_pair_eligibility.json',{'eligible':eligible,'filtered':filtered,'source':'MuJoCo3.11 collision_driver.c287–313,664–674,1689–1697; static sleep is not used as geometric clearance filter.'})
    def snapshot(context):
        return {**context,'qpos':data.qpos.copy(),'geom_xpos':data.geom_xpos.copy(),'geom_xmat':data.geom_xmat.copy(),'site_xpos':data.site_xpos.copy(),'body_xpos':data.xpos.copy(),'body_xquat':data.xquat.copy()}
    seed_start_heavy=0
    for active_seed in range(3):
        seed_start=time.perf_counter();seed_deadline=min(job_deadline,seed_start+60.);optimization_deadline=min(seed_deadline,seed_start+45.);current_phase='OPTIMIZATION';seed_start_heavy=counters['heavy_Minkowski_attempted']
        cache={};jac_cache={};iteration=0;best_feasible=None;best_violation=None;best_objective=None;last=None;seed_eval_count=0
        xseed=np.asarray(parent['all_seed_results'][active_seed]['last_complete']['x26']);zseed=xseed/scale
        def evaluate(z,purpose='base'):
            nonlocal best_feasible,best_violation,best_objective,last,seed_eval_count
            budget();key=np.asarray(z,np.float64).tobytes()
            if key in cache:
                context=cache[key]
                # Reconstruct actual FK only when a cached point is selected for capture.
                return context
            if counters['heavy_Minkowski_attempted']-seed_start_heavy>=1900:raise BudgetStop('Per-seed1900heavy optimization reserve')
            x=np.asarray(z)*scale;R,p=set_state(x);tips=data.site_xpos[tip_sites].copy();desired=alltips@R.T+p
            anchor_world=np.stack([anchors[i]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g] for i,g in enumerate(pad_ids)]);target_world=contact_targets@R.T+p
            blocks={'contact_anchor_position':((anchor_world-target_world)/.005).ravel(),'candidate_tip_style':(.15*(tips[:2]-desired[:2])/.01).ravel(),'noncontact_tip_style':(.05*(tips[2:]-desired[2:])/.02).ravel(),'joint_regularizer':.01*(x[:20]-q0)/(qhi-qlo)}
            objective=float(sum(v@v for v in blocks.values()));pairs={g:query_pair(g,R,p) for g in all_phone_pairs};c=[]
            for g in pad_ids:c.extend([pairs[g]['lower_m']-.00001,.00019-pairs[g]['upper_m']])
            for g in all_phone_pairs:
                if g not in pad_ids:c.append(pairs[g]['lower_m']-.00002)
            c=np.asarray(c)/.001;violation=float(np.maximum(-c,0).max());feasible=bool((c>=0).all())
            seed_eval_count+=1;row={'seed':active_seed,'evaluation':seed_eval_count,'purpose':purpose,'x26':x.tolist(),'objective':objective,'block_squared_norms':{k:float(v@v) for k,v in blocks.items()},'max_constraint_violation_mm':violation,'optimization_constraints_feasible':feasible,'tip_error_m':np.linalg.norm(tips-desired,axis=1).tolist(),'anchor_error_m':np.linalg.norm(anchor_world-target_world,axis=1).tolist(),'pad_signed_gap_m':[pairs[g]['signed_m'] for g in pad_ids],'elapsed_seed_seconds':time.perf_counter()-seed_start,'counters':dict(counters)}
            context={'row':row,'z':np.asarray(z).copy(),'x':x.copy(),'R':R.copy(),'p':p.copy(),'constraints':c,'pairs':pairs};cache[key]=context;evaluations.append(row);last=snapshot(context)
            if purpose=='base':
                candidate=last
                if feasible and (best_feasible is None or objective<best_feasible['row']['objective']):best_feasible=candidate
                if best_violation is None or (violation,objective)<(best_violation['row']['max_constraint_violation_mm'],best_violation['row']['objective']):best_violation=candidate
                if best_objective is None or objective<best_objective['row']['objective']:best_objective=candidate
            return context
        def objective(z):return evaluate(z)['row']['objective']
        def constraints(z):return evaluate(z)['constraints']
        def derivatives(z):
            key=np.asarray(z,np.float64).tobytes()
            if key in jac_cache:return jac_cache[key]
            base=evaluate(z);gradient=np.empty(26);jac=np.empty((len(base['constraints']),26))
            for k,h in enumerate(step):
                signed=h if z[k]+h<=hi[k] else -h
                zz=np.array(z).copy();zz[k]+=signed;probe=evaluate(zz,'finite_difference')
                gradient[k]=(probe['row']['objective']-base['row']['objective'])/signed;jac[:,k]=(probe['constraints']-base['constraints'])/signed
            jac_cache[key]=(gradient,jac);return gradient,jac
        def callback(z):
            nonlocal iteration
            iteration+=1;c=evaluate(z);callbacks.append({'seed':active_seed,'iteration':iteration,'objective':c['row']['objective'],'max_constraint_violation_mm':c['row']['max_constraint_violation_mm'],'x26':c['x'].tolist(),'elapsed_seed_seconds':time.perf_counter()-seed_start,'counters':dict(counters)})
        status='NOT_STARTED';message='';solver=None
        try:
            fit=minimize(objective,zseed,method='SLSQP',jac=lambda z:derivatives(z)[0],bounds=list(zip(lo,hi)),constraints=[{'type':'ineq','fun':constraints,'jac':lambda z:derivatives(z)[1]}],callback=callback,options={'maxiter':200,'ftol':1e-9,'disp':False})
            status='SOLVER_RETURNED';message=str(fit.message);solver={'success':bool(fit.success),'status':int(fit.status),'nit':int(fit.nit),'nfev':int(fit.nfev),'njev':int(fit.njev),'fun':float(fit.fun),'returned_x26':(fit.x*scale).tolist()};evaluate(fit.x)
        except (BudgetStop,UnknownGeometry,ValueError,ArithmeticError) as e:status=type(e).__name__.upper()+'_PRESERVED';message=str(e)
        current_phase='FINAL_VERIFICATION';selected=best_feasible if best_feasible is not None else best_violation
        result={'seed':active_seed,'status':status,'message':message,'solver':solver,'actual_callback_iterations':iteration,'complete_evaluations':seed_eval_count,'source_start_snapshot':protocol['inputs'][f'seed{active_seed}'],'selection':'BEST_EVALUATED_FEASIBLE_BASE' if best_feasible else 'BEST_EVALUATED_VIOLATION_BASE','best_objective_observed':None if best_objective is None else best_objective['row'],'last_complete':None if last is None else last['row']}
        if selected is not None:
            x=selected['x'];verification={'all_phone_pairs':[],'self_pairs':[],'unknown':[]}
            try:
                R,p=set_state(x)
                for g in all_phone_pairs:
                    q=query_pair(g,R,p,True);detail={'geom_id':g,'name':model.geom(g).name,'is_candidate_pad':g in pad_ids,'gap_m':q['signed_m'],'interval_m':[q['lower_m'],q['upper_m']],'query':world_query(q['query'],R,p)};verification['all_phone_pairs'].append(detail)
                for pair in eligible:
                    g,h=pair['geom_ids'];sg=shape_phone_frame(g,R,p);sh=shape_phone_frame(h,R,p);lower=bbox_lower(box(sg),box(sh))
                    if lower>.002:
                        verification['self_pairs'].append({**pair,'method':'AABB_DISJOINT_CERTIFICATE','lower_m':lower,'nonpenetration':True});continue
                    budget()
                    if sg['kind']==sh['kind']=='capsule':
                        dist,_,_=segment_distance(sg['a'],sg['b'],sh['a'],sh['b'],np);gap=dist-sg['radius']-sh['radius'];guard=256*np.finfo(float).eps;counters['analytic_capsule_capsule']+=1;q={'signed_translation_margin_m':gap,'margin_roundoff_guarded_interval_m':[gap-guard,gap+guard],'query_kind':'analytic_capsule_capsule'}
                    else:
                        counters['heavy_Minkowski_attempted']+=1
                        try:
                            if sg['kind']=='convex':q=ConvexPolytope(sg['vertices']).capsule(sh['a'],sh['b'],sh['radius']) if sh['kind']=='capsule' else ConvexPolytope(sg['vertices']).convex(sh['vertices'],max_pair_points=4_000_000)
                            else:q=ConvexPolytope(sh['vertices']).capsule(sg['a'],sg['b'],sg['radius'])
                            counters['heavy_Minkowski_returned']+=1
                        except Exception as e:counters['heavy_unknown']+=1;raise UnknownGeometry(str(e))
                    verification['self_pairs'].append({**pair,'method':q['query_kind'],'signed_gap_m':float(q['signed_translation_margin_m']),'interval_m':q['margin_roundoff_guarded_interval_m'],'nonpenetration':bool(q['margin_roundoff_guarded_interval_m'][0]>=0)})
            except (BudgetStop,UnknownGeometry,ValueError,ArithmeticError) as e:verification['unknown'].append(f'{type(e).__name__}: {e}')
            allpair=len(verification['all_phone_pairs'])==len(all_phone_pairs);selfcomplete=len(verification['self_pairs'])==len(eligible)
            pairpass=allpair and all(v['interval_m'][0]>=0 and (not v['is_candidate_pad'] or v['interval_m'][1]<=.0002) for v in verification['all_phone_pairs']);selfpass=selfcomplete and all(v['nonpenetration'] for v in verification['self_pairs'])
            result.update(selected_evaluation=selected['row'],verification=verification,all_pad_and_nonpad_hard_constraints_pass=bool(pairpass),self_collision_check_complete=bool(selfcomplete),self_nonpenetration_pass=bool(selfpass),geometry_candidate_qualified=bool(pairpass and selfpass and not verification['unknown']),selected_q20_rad=x[:20].tolist(),selected_phone_translation_palm_m=x[20:23].tolist(),selected_phone_rotation_vector_rad=x[23:].tolist())
            # Preserve the selected state's FK capture even if verification stopped.
            np.savez_compressed(ROOT/f'seed{active_seed}_selected_state.npz',qpos27=selected['qpos'],qvel26=np.zeros(26),ctrl20=np.zeros(20),geom_xpos=selected['geom_xpos'],geom_xmat=selected['geom_xmat'],site_xpos=selected['site_xpos'],body_xpos=selected['body_xpos'],body_xquat=selected['body_xquat'],x26=x,phone_R_world=selected['R'],phone_p_world=selected['p'])
            result['selected_state']=pin(ROOT/f'seed{active_seed}_selected_state.npz')
        result['elapsed_seed_seconds']=time.perf_counter()-seed_start;result['heavy_queries_this_seed']=counters['heavy_Minkowski_attempted']-seed_start_heavy;seed_results.append(result)
        write(ROOT/f'seed{active_seed}_result.json',result)
    model_after=hashlib.sha256(b''.join(np.ascontiguousarray(getattr(model,n)).tobytes() for n in ('jnt_range','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','dof_armature','dof_damping','dof_frictionloss','body_pos','body_quat'))).hexdigest();assert model_after==model_before
    for name,rows in [('all_evaluations.jsonl',evaluations),('actual_iterations.jsonl',callbacks),('heavy_query_log.jsonl',query_log)]:
        with (ROOT/name).open('x',encoding='utf8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    rank=sorted([r for r in seed_results if 'selected_evaluation'in r],key=lambda r:(not r['geometry_candidate_qualified'],not r['all_pad_and_nonpad_hard_constraints_pass'],r['selected_evaluation']['max_constraint_violation_mm'],r['selected_evaluation']['objective'],r['seed']))
    report={'schema':'contact_first_fixed_wrist_hand2_geometry_v2:1','status':'COMPLETED_BOUNDED_STATIC_CANDIDATES_NOT_HOLDING','protocol':pin(ROOT/'protocol_v2.json'),'source':pin(__file__),'seed_results':seed_results,'selected_seed_by_predeclared_rule':rank[0]['seed'] if rank else None,'qualified_seed_count':sum(r.get('geometry_candidate_qualified',False) for r in seed_results),'counters':counters,'all_model_parameters_unchanged':True,'model_digest':model_before,'geometry_metadata':metadata,'fixed_anchor_targets':pin(ROOT/'fixed_anchors_and_targets.json'),'elapsed_seconds':time.perf_counter()-job_start,'native_distance_calls':0,'integration_optimization_training_GPU_scope':'Static SciPy geometry optimization only;0integration/training/GPU','holding_or_contact_force_qualified':False,'limits':'Independent numerical ideal-convex geometry, not interval proof/scan contact/realhandpad/force/stability. Unknown self or phone query is not PASS.'}
    report['outputs']={p.name:pin(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='report.json'};write(ROOT/'report.json',report)
    print(json.dumps({'report':pin(ROOT/'report.json'),'qualified_seed_count':report['qualified_seed_count'],'selected':report['selected_seed_by_predeclared_rule'],'seconds':report['elapsed_seconds'],'counters':counters,'seed_statuses':[r['status'] for r in seed_results]}),flush=True)


if __name__=='__main__':main()
