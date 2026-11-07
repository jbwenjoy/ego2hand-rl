"""SOURCE-only until explicitly called: fixed-phone q0..11 gap plane Jacobians.

No full geometry, optimizer, native forward/step, LP force or NN is invoked here.
Each exact PHONE query's misleading *_world names are still PHONE BODY in the
frozen StaticRetargetEvaluator phone_geometry_rows/query_phone_frame container.
This helper transforms them explicitly. The normal translates HAND away from
phone, so dgap = n_world dot (dH_world - dP_world).

Independent support-set means need not be a mutually nearest pair; their
rotational Jacobian is only a named numerical search heuristic, not a certified
distance subgradient, true contact/force point, or finite pressure patch.
Every proposed pose requires fresh complete22phone/189self/table verification.
"""
import numpy as np


def _array(value,shape,label):
    x=np.asarray(value,dtype=np.float64)
    if x.shape!=shape or not np.isfinite(x).all():raise ValueError(label+' finite shape')
    return x


def phone_gap_jacobian(tripod,full_result):
    """Return {geom_id: record}; no gradient is invented for missing/AABB data.

    Requires the very same CURRENT tripod.E.d that produced full_result['state'].
    The only native calls are2 read-only mj_jac per usable pair; the returned
    counts and tripod.counts['separating_plane_point_jac_calls'] record them.
    q12 derivatives keep phone pose and q12..19 fixed. No model options change.
    """
    E=tripod.E;m,d=E.m,E.d;k=full_result['k'];state=full_result['state']
    qpos=_array(k['qpos27'],(27,),'current qpos')
    if d.qpos.dtype!=qpos.dtype or d.qpos.tobytes()!=qpos.tobytes():
        raise ValueError('Stale E.d: obtain a fresh full geometry state before Jacobians')
    if np.any(d.qvel):raise ValueError('Static geometry Jacobian expects saved zero velocity')
    R=_array(k['R'],(3,3),'phone body R');p=_array(k['p'],(3,),'phone body p')
    if np.max(abs(R.T@R-np.eye(3)))>1e-9:raise ValueError('Phone rigid frame invalid; no repair')
    contacts={int(c['geom_id']):c for c in state.get('contacts',[])}
    rows=state['phone_geometry_rows']
    if len(rows)!=22 or len({int(row['geom_id']) for row in rows})!=22:
        raise ValueError('Expected all22 current phone rows including table')
    out={};counter_key='separating_plane_point_jac_calls'
    tripod.counts.setdefault(counter_key,0)
    for row in rows:
        gid=int(row['geom_id']);base=dict(geom_id=gid,name=row.get('name'),
            lower_bound_m=row.get('lower_bound_m'),upper_bound_m=row.get('upper_bound_m'),
            gap_m=row.get('exact_gap_m'),grad_q12=None,native_point_jacobian_calls=0,
            fresh_full_geometry_required=True,force_or_patch_qualified=False)
        if row['status']!='EXACT_NUMERICAL_IDEAL_CONVEX':
            out[gid]=dict(base,status='NOT_LINEARIZED_'+row['status'],support_scope='Keep original bound/UNKNOWN; no zero-gradient claim.')
            continue
        try:
            query=row['query_phone_frame']
            nbody=_array(query.get('normal_world',query.get('translation_normal_world')),(3,),'hand-exit BODY normal')
            if abs(np.linalg.norm(nbody)-1)>1e-9:raise ValueError('Non-unit query normal')
            Pbody=np.asarray(query['object_support_points_world_m'],dtype=np.float64)
            Hbody=np.asarray(query['hand_support_points_world_m'],dtype=np.float64)
            if (Pbody.ndim!=2 or Hbody.ndim!=2 or Pbody.shape[1:]!=(3,) or Hbody.shape[1:]!=(3,)
                    or min(len(Pbody),len(Hbody))<1 or not np.isfinite(Pbody).all() or not np.isfinite(Hbody).all()):
                raise ValueError('Missing finite support sets; no centroid fallback from arbitrary mesh')
            n=R@nbody;Pworld=Pbody@R.T+p;Hworld=Hbody@R.T+p
            paired=contacts.get(gid,{}).get('pairing',{})
            pair=paired.get('closest_pair') if paired.get('status')=='UNIQUE_CLOSEST_PAIR' else None
            if pair is not None:
                pointP=_array(pair['object_point_world_m'],(3,),'saved unique object point')
                pointH=_array(pair['hand_point_world_m'],(3,),'saved unique hand point')
                point_rule='SAVED_CURRENT_UNIQUE_CLOSEST_PAIR'
            else:
                pointP=Pworld.mean(0);pointH=Hworld.mean(0)
                point_rule='INDEPENDENT_SUPPORT_SET_MEANS_NUMERICAL_PLANE_HEURISTIC'
            gap=float(query['signed_gap_m']);plane_gap=float(n@(pointH-pointP))
            guard=float(query['roundoff_guard_m'])
            if abs(plane_gap-gap)>max(1e-10,4*guard):
                raise ValueError('Support plane/sign/frame fails saved gap reconstruction')
            gp=np.asarray(d.geom_xpos[gid]);gR=np.asarray(d.geom_xmat[gid]).reshape(3,3)
            op=np.asarray(d.geom_xpos[E.phone]);oR=np.asarray(d.geom_xmat[E.phone]).reshape(3,3)
            localH=gR.T@(pointH-gp);localP=oR.T@(pointP-op)
            JH=np.zeros((3,m.nv));JP=np.zeros((3,m.nv))
            E.mj.mj_jac(m,d,JH,None,pointH,int(m.geom_bodyid[gid]));tripod.counts[counter_key]+=1
            E.mj.mj_jac(m,d,JP,None,pointP,int(m.geom_bodyid[E.phone]));tripod.counts[counter_key]+=1
            if not np.isfinite(JH).all() or not np.isfinite(JP).all():raise ValueError('Nonfinite point Jacobian')
            if np.max(abs(JP[:,:12]))>1e-12:raise ValueError('Phone is not fixed with respect to robot q0..11')
            gradient=n@(JH[:,:12]-JP[:,:12])
            out[gid]=dict(base,status='OK_LOCAL_PLANE_SEARCH_MODEL',gap_m=gap,grad_q12=gradient,
                hand_exit_normal_world=n,hand_exit_normal_phone_body=nbody,
                object_point_world_m=pointP,hand_point_world_m=pointH,
                object_point_geom_local_m=localP,hand_point_geom_local_m=localH,
                object_support_count=len(Pbody),hand_support_count=len(Hbody),
                point_rule=point_rule,support_scope='Optimization plane points only; support means may be non-paired. Never use as LP load points, native contacts or force truth.',
                reconstructed_support_plane_gap_m=plane_gap,plane_reconstruction_error_m=abs(plane_gap-gap),
                native_point_jacobian_calls=2,frame='PHONE BODY query -> WORLD points/normal -> native geom-local anchors',
                gradient_units='metres_per_robot_joint_radian')
        except (KeyError,ValueError,ArithmeticError) as err:
            out[gid]=dict(base,status='UNKNOWN_LOCAL_PLANE',error=repr(err),support_scope='No gradient/zero repair; caller may obtain new true geometry or an explicitly counted fallback.')
    return out


def phone_margin_linearization(by_geom,*,designated_geom_ids=(15,28,41),pad_window_m=(-.0001,.0002),nonpad_buffer_m=0.):
    """Only modeled phone inequalities; unmodeled rows remain explicitly listed.

    Uses guarded base lower/upper bounds from true full geometry and derivatives
    +/-dgap. This is not the full214 constraint vector and grants no PASS.
    """
    margins=[];jac=[];labels=[];omitted=[];lo,hi=map(float,pad_window_m)
    for gid,row in by_geom.items():
        if row['grad_q12'] is None:
            omitted.append(dict(geom_id=int(gid),status=row['status']));continue
        g=_array(row['grad_q12'],(12,),'gap q12 derivative')
        if int(gid) in designated_geom_ids:
            margins.extend([float(row['lower_bound_m'])-lo,hi-float(row['upper_bound_m'])]);jac.extend([g,-g])
            labels.extend([row['name']+'_pad_lower',row['name']+'_pad_upper'])
        else:
            margins.append(float(row['lower_bound_m'])-float(nonpad_buffer_m));jac.append(g);labels.append(row['name']+'_nonpad')
    return dict(margins_m=np.asarray(margins),jacobian_m_per_rad=np.asarray(jac).reshape(-1,12),labels=labels,
        omitted_phone_rows=omitted,self_rows_linearized=False,all214_fresh_check_required=True,geometry_qualified=False)


def linear_feasibility_step(model,q12,joint_range12x2,*,trust_rad=.02,length_scale_m=.001):
    """Pure LP numerical proposal, not force LP/true geometry. No execution on import.

    Minimize maximum negative predicted margin plus1e-6 normalized step L1.
    Bounds intersect the original true joint box; no clipping of proposed q.
    Root driver must fresh-check all22phone/189self/table and accept/reject.
    """
    from scipy.optimize import linprog
    q=_array(q12,(12,),'q12');limits=_array(joint_range12x2,(12,2),'true joint range')
    M=np.asarray(model['margins_m'],dtype=float);J=np.asarray(model['jacobian_m_per_rad'],dtype=float)
    if J.shape!=(len(M),12) or not len(M) or not np.isfinite(J).all() or not np.isfinite(M).all():raise ValueError('Finite nonempty plane model')
    if trust_rad<=0 or length_scale_m<=0 or np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError('True bounds/trust')
    lower=np.maximum(-1.,(limits[:,0]-q)/trust_rad);upper=np.minimum(1.,(limits[:,1]-q)/trust_rad)
    # Variables are normalized jointstep z12, slack t>=0, abs(z)12.
    A=np.zeros((len(M)+24,25));b=np.r_[M/length_scale_m,np.zeros(24)]
    A[:len(M),:12]=-J*trust_rad/length_scale_m;A[:len(M),12]=-1
    for j in range(12):A[len(M)+2*j,j]=1;A[len(M)+2*j,13+j]=-1;A[len(M)+2*j+1,j]=-1;A[len(M)+2*j+1,13+j]=-1
    objective=np.r_[np.zeros(12),1.,np.full(12,1e-6)]
    result=linprog(objective,A_ub=A,b_ub=b,bounds=list(zip(lower,upper))+[(0,None)]+[(0,None)]*12,
        method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:return dict(status='NO_NUMERICAL_PLANE_PROPOSAL',solver_status=int(result.status),message=str(result.message),proposal_q12=None,geometry_qualified=False)
    step=result.x[:12]*trust_rad;proposal=q+step
    return dict(status='UNVALIDATED_SMALL_STEP_PROPOSAL',solver_status=int(result.status),proposal_q12=proposal,step_q12=step,
        predicted_margins_m=M+J@step,linear_slack_m=float(result.x[12]*length_scale_m),trust_rad=trust_rad,
        geometry_qualified=False,all214_fresh_check_required=True,no_force_or_holding_claim=True)


def declaration():
    return dict(api='phone_gap_jacobian(tripod,full_result) -> {geom_id: record}; phone_margin_linearization; linear_feasibility_step',
        scope='PHONE rows only, q0..11, fixed phone/rest8. Self/AABB/new support mode changes require fresh full true geometry.',
        normal_sign='hand exit +n increases gap; gradient=n@(J_hand-J_phone)',
        no_import_native_model_or_execution=True,no_actual_query_FK_Jac_LP_or_physics_executed=True)
