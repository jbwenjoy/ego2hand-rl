"""Nonintegrating seed2 support/preload preparation. No holding rollout."""
from pathlib import Path
import hashlib, json, math, sys
import numpy as np
from scipy.optimize import linprog

HERE=Path(__file__).resolve().parent
P=HERE.parents[1]
sys.path.insert(0,str(P/'geometry_query_v2'))
sys.path.insert(0,str(P/'retarget_contact_design_v2'))
from convex_query import ConvexPolytope
from query_interface_v2 import support_features
from static_support_lp import static_support_lp


def pin(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes()
    return {'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def save(path,obj):
    with Path(path).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x


def closest_support_pair(Ps,Hs,n,gap,tol=1e-10):
    """Solve barycentric tangential coincidence, then independently check all3 axes."""
    Ps,Hs,n=[np.asarray(x,dtype=np.float64) for x in (Ps,Hs,n)]
    if gap<=0 or abs(np.linalg.norm(n)-1)>1e-10:raise ValueError('Positive gap/unit normal required')
    axis=np.eye(3)[np.argmin(np.abs(n))];t1=np.cross(n,axis);t1/=np.linalg.norm(t1);t2=np.cross(n,t1)
    T=np.stack((t1,t2));center=Ps[0];scale=max(.01,float(np.max(np.abs(np.r_[Ps-center,Hs-center]))))
    np_,nh=len(Ps),len(Hs)
    A=np.zeros((4,np_+nh));A[0,:np_]=1;A[1,np_:]=1
    A[2:,:np_]=-T@(Ps-center).T/scale;A[2:,np_:]=T@(Hs-center).T/scale
    b=np.array([1.,1.,0.,0.]);rank=int(np.linalg.matrix_rank(A))
    if rank==np_+nh:
        w=np.linalg.lstsq(A,b,rcond=None)[0];method='unique_full_column_rank_barycentric_solution'
    else:
        sol=linprog(np.zeros(np_+nh),A_eq=A,b_eq=b,bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9})
        if not sol.success:raise ArithmeticError('No exact support-feature tangential pairing')
        w=sol.x;method='one_feasible_closest_pair_nonunique_not_all_patch_support'
    pp=w[:np_]@Ps;hh=w[np_:]@Hs;res=hh-pp-gap*n
    checks={'weights_nonnegative':bool(w.min()>=-1e-10),
      'weights_each_sum_one':bool(abs(w[:np_].sum()-1)<=1e-10 and abs(w[np_:].sum()-1)<=1e-10),
      'full_three_axis_difference_is_gap_normal':bool(np.max(np.abs(res))<=tol),
      'euclidean_gap_matches':bool(abs(np.linalg.norm(hh-pp)-gap)<=tol),
      'support_planes_match':bool(np.ptp(Ps@n)<=tol and np.ptp(Hs@n)<=tol and abs((hh-pp)@n-gap)<=tol)}
    if not all(checks.values()):raise ArithmeticError('Closest-pair reconstruction failed: '+str(checks))
    return {'object_point_world_m':pp,'hand_point_world_m':hh,'object_weights':w[:np_],
       'hand_weights':w[np_:],'rank':rank,'unknown_count':len(w),'method':method,
       'difference_minus_gap_normal_m':res,'numerical_witness_tolerance_m':tol,'checks':checks}


def frame(n):
    a=np.eye(3)[np.argmin(np.abs(n))];t=np.cross(n,a);t/=np.linalg.norm(t)
    return t,np.cross(n,t)


def controller(name,tau,q,m,offset,gain,lower,upper,definition):
    if tau is None:return {'name':name,'reachable':False,'ctrl20':None,'tau_at_initial_q_Nm':None,'definition':definition,'reason':'Conditional two-point LP infeasible'}
    tau=np.asarray(tau);ctrl=(tau-offset)/gain
    checks={'finite':bool(np.isfinite(ctrl).all()),'within_reachable_torque_intersection':bool(np.all(tau>=lower-1e-10) and np.all(tau<=upper+1e-10)),
       'unclamped_ctrl_within_actual_range':bool(np.all(ctrl>=m.actuator_ctrlrange[:,0]-1e-10) and np.all(ctrl<=m.actuator_ctrlrange[:,1]+1e-10)),
       'inverse_PD_reconstructs_tau':bool(np.max(np.abs(gain*ctrl+offset-tau))<=1e-12)}
    return {'name':name,'reachable':all(checks.values()),'ctrl20':ctrl.tolist(),'tau_at_initial_q_Nm':tau.tolist(),
       'definition':definition,'checks':checks,'clamped':False,'q20_rad':q.tolist()}


def main():
    import mujoco
    c=json.loads((HERE/'preregistration.json').read_text())
    out=HERE/'actual_v1';out.mkdir(exist_ok=False)
    for x in c['pins'].values():
        if pin(x['path'])!=x:raise ValueError('Changed pinned input/source: '+x['path'])
    if pin(__file__)!=c['source']:raise ValueError('Changed preparation source')
    binding=json.loads(Path(c['pins']['binding']['path']).read_text())
    gr=json.loads(Path(c['pins']['geometry_report']['path']).read_text());sr=json.loads(Path(c['pins']['seed_result']['path']).read_text())
    if gr['selected_seed_by_predeclared_rule']!=2 or not sr['geometry_candidate_qualified'] or sr['selected_state']!=c['pins']['state']:raise ValueError('Different/unqualified selected candidate')
    if binding['source']!=c['pins']['scene_source']:raise ValueError('Compiled scene source mismatch')
    if pin(binding['model']['path'])!=binding['model']:raise ValueError('Changed compiled model')
    m=mujoco.MjModel.from_binary_path(binding['model']['path']);d=mujoco.MjData(m)
    with np.load(c['pins']['state']['path'],allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
    d.qpos[:]=s['qpos27'];d.qvel[:]=0;d.ctrl[:]=0
    mujoco.mj_forward(m,d)  # one nonintegrating FK/bias evaluation; zero mj_step calls
    if (m.nq,m.nv,m.nu,m.nmocap,m.neq)!=(27,26,20,0,0):raise ValueError('Model layout changed')
    for k,a in [('geom_xpos',d.geom_xpos),('geom_xmat',d.geom_xmat),('body_xpos',d.xpos),('body_xquat',d.xquat)]:
        if not np.array_equal(a,s[k]):raise ValueError('Candidate kinematics changed: '+k)
    if not np.array_equal(d.qvel,np.zeros(26)):raise ValueError('Nonzero velocity')
    oid=m.body('hot3d_phone24').id;phone=m.geom('phone24_collision_convex').id;table=m.geom('stage1_single_table').id
    hand=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g]!=oid and g!=table]
    pads=[m.geom('right_finger1_tip_sensor_col').id,m.geom('right_finger2_tip_sensor_col').id]
    if m.body_mass[oid]!=.2 or not np.array_equal(m.opt.gravity,np.array([0.,0.,-9.81])) or int(m.opt.cone)!=int(mujoco.mjtCone.mjCONE_PYRAMIDAL):raise ValueError('Nominal object/cone assumptions changed')
    pair_evidence=[]
    for g in pads:
        ids=[i for i in range(m.npair) if set((int(m.pair_geom1[i]),int(m.pair_geom2[i])))=={g,phone}]
        if len(ids)!=1:raise ValueError('No unique actual explicit pad pair')
        i=ids[0]
        if int(m.pair_dim[i])!=3 or not np.array_equal(m.pair_friction[i,:2],np.array([.7,.7])):raise ValueError('Actual pair does not match native condim3/mu assumptions')
        pair_evidence.append({'pair_id':i,'pair_name':m.pair(i).name,'dim':int(m.pair_dim[i]),'friction5':m.pair_friction[i].copy(),'solref':m.pair_solref[i].copy(),'solimp':m.pair_solimp[i].copy()})
    def shape(g):
        R=d.geom_xmat[g].reshape(3,3);p=d.geom_xpos[g];kind=int(m.geom_type[g])
        if kind==3:return {'kind':'capsule','a':p-m.geom_size[g,1]*R[:,2],'b':p+m.geom_size[g,1]*R[:,2],'r':float(m.geom_size[g,0])}
        if kind==7:
            mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];v=m.mesh_vert[a:a+m.mesh_vertnum[mid]].astype(float)
            v=ConvexPolytope(v).vertices
        elif kind==6:
            from itertools import product
            v=np.array(list(product(*[(-x,x) for x in m.geom_size[g]])))
        else:raise ValueError('Unsupported physical geometry')
        return {'kind':'convex','vertices':v@R.T+p}
    shapes={g:shape(g) for g in hand+[table,phone]};poly=ConvexPolytope(shapes[phone]['vertices']);pv=poly.vertices
    def bounds(a):
        if a['kind']=='capsule':return np.array([np.minimum(a['a'],a['b'])-a['r'],np.maximum(a['a'],a['b'])+a['r']])
        return np.array([a['vertices'].min(0),a['vertices'].max(0)])
    def bsep(a,b):
        aa,bb=bounds(a),bounds(b);return float(np.linalg.norm(np.maximum(np.maximum(aa[0]-bb[1],bb[0]-aa[1]),0)))
    def exact(pa,a):
        if a['kind']=='capsule':return pa.capsule(a['a'],a['b'],a['r'])
        return pa.convex(a['vertices'],max_pair_points=4000000)
    geometry=[];contacts=[]
    for g in hand+[table]:
        a=shapes[g];sep=bsep(shapes[phone],a)
        if g not in pads and sep>1e-10:geometry.append({'geom':m.geom(g).name,'method':'AABB_lower_bound','lower_m':sep-1e-10,'passed':True});continue
        result=exact(poly,a);gap=result['signed_translation_margin_m']
        good=(gap>=-result['roundoff_guard_m']) if g not in pads else (0<=gap<=.0002)
        geometry.append({'geom':m.geom(g).name,'method':'independent_convex_query','gap_m':gap,'passed':good})
        if g in pads:
            f=support_features(result,pv,hand_vertices=a['vertices'])
            w=closest_support_pair(f['object_support_points_world_m'],f['hand_support_points_world_m'],f['normal_world'],gap)
            contacts.append({'geom':m.geom(g).name,'geom_id':g,'body_id':int(m.geom_bodyid[g]),'gap_m':gap,
                'hand_exit_normal_world':f['normal_world'],'force_on_object_normal_world':-f['normal_world'],
                'support_features':f,'closest_pair':w})
    # Recheck the original eligibility table; new explicit pairs only involve phone.
    pairs=json.loads(Path(c['pins']['self_eligibility']['path']).read_text())['eligible'];selfchecks=[]
    oldmod=__import__('importlib.util',fromlist=['spec_from_file_location'])
    spec=oldmod.spec_from_file_location('frozen_segment_math',c['pins']['geometry_source']['path']);old=oldmod.module_from_spec(spec);spec.loader.exec_module(old)
    for row in pairs:
        g,h=row['geom_ids'];a,b=shapes[g],shapes[h];sep=bsep(a,b)
        if sep>1e-10:method='AABB_lower_bound';gap=sep-1e-10
        elif a['kind']==b['kind']=='capsule':method='analytic_capsule_capsule';gap=old.segment_distance(a['a'],a['b'],b['a'],b['b'],np)[0]-a['r']-b['r']
        else:
            method='independent_convex_query'
            if a['kind']=='convex':gap=exact(ConvexPolytope(a['vertices']),b)['signed_translation_margin_m']
            else:gap=exact(ConvexPolytope(b['vertices']),a)['signed_translation_margin_m']
        selfchecks.append({'geom_ids':[g,h],'names':row['names'],'method':method,'gap_or_lower_m':gap,'passed':gap>=-1e-10})
    save(out/'geometry_recheck.json',plain({'phone_pairs':geometry,'eligible_self_pairs':selfchecks,'contacts':contacts,
       'excluded_geometry_scope':'Original frozen active proxy/eligibility set only; disabled or unrepresented geometry not newly certified'}))
    if not all(x['passed'] for x in geometry+selfchecks) or len(contacts)!=2:raise ValueError('Geometry recheck failed')
    if np.any(m.actuator_trntype!=int(mujoco.mjtTrn.mjTRN_JOINT)) or not np.array_equal(m.actuator_trnid[:,0],np.arange(20)) or not np.array_equal(m.actuator_gear[:,0],np.ones(20)) or np.any(m.actuator_gear[:,1:]!=0):raise ValueError('Non-unit direct joint transmission')
    if np.any(m.actuator_gaintype!=int(mujoco.mjtGain.mjGAIN_FIXED)) or np.any(m.actuator_biastype!=int(mujoco.mjtBias.mjBIAS_AFFINE)) or np.any(m.actuator_dyntype!=int(mujoco.mjtDyn.mjDYN_NONE)):raise ValueError('Not direct affine PD')
    if not np.all(m.actuator_ctrllimited) or not np.all(m.actuator_forcelimited) or not np.all(m.jnt_actfrclimited[:20]):raise ValueError('Expected actual finite clamps')
    q=d.qpos[:20].copy();bias=d.qfrc_bias[:20].copy();gain=m.actuator_gainprm[:,0].copy()
    if np.any(gain<=0):raise ValueError('Positive gains required')
    if not np.allclose(d.actuator_length,q,rtol=0,atol=1e-14) or np.max(np.abs(d.actuator_velocity))!=0:raise ValueError('Actual actuator length/velocity does not match unit direct joint PD')
    offset=m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*q+m.actuator_biasprm[:,2]*d.actuator_velocity
    pdlo=gain*m.actuator_ctrlrange[:,0]+offset;pdhi=gain*m.actuator_ctrlrange[:,1]+offset
    lower=np.maximum.reduce([pdlo,m.actuator_forcerange[:,0],m.jnt_actfrcrange[:20,0]])
    upper=np.minimum.reduce([pdhi,m.actuator_forcerange[:,1],m.jnt_actfrcrange[:20,1]])
    if np.any(lower>upper):raise ValueError('No unclamped PD/actuator/joint intersection')
    J=[];points=[];normals=[]
    for x in contacts:
        jp=np.zeros((3,26));jr=np.zeros((3,26));mujoco.mj_jac(m,d,jp,jr,np.asarray(x['closest_pair']['hand_point_world_m']),x['body_id'])
        if np.any(jp[:,20:]!=0):raise ValueError('Hand point depends on phone free joint')
        J.append(jp[:,:20]);points.append(x['closest_pair']['object_point_world_m']);normals.append(x['force_on_object_normal_world'])
    J=np.array(J);points=np.array(points);normals=np.array(normals);mu=.7/math.sqrt(2)
    lp=static_support_lp(points,normals,d.xipos[oid].copy(),.2,mu,J,bias,lower,upper,mode='ideal_inscribed8')
    desired_B=bias+np.einsum('nij,ni->j',J,3*normals)
    variants=[controller('gravity_compensation',bias,q,m,offset,gain,lower,upper,'A: actual qfrc_bias at qvel0; ignores helpful frictionloss'),
      controller('two_pad_3N_normal_preload',desired_B,q,m,offset,gain,lower,upper,'B: bias + sum(J_hand^T * (3N * n_force_on_object)); diagnostic only, not support-feasible claim'),
      controller('conditional_two_point_LP',lp['tau_act_required_Nm'] if lp['feasible'] else None,q,m,offset,gain,lower,upper,'LP: potential closed-point statics using object closest witnesses, hand-point Jacobians, mu_design=.7/sqrt(2) inscribed8')]
    np.savez_compressed(out/'support_operands_and_controls.npz',qpos27=d.qpos.copy(),qvel26=d.qvel.copy(),object_COM_world=d.xipos[oid].copy(),object_contact_points_world=points,
       hand_contact_points_world=np.array([x['closest_pair']['hand_point_world_m'] for x in contacts]),normals_force_on_object=normals,J_hand3x20=J,qfrc_bias20=bias,
       gain20=gain,biasprm=m.actuator_biasprm.copy(),actuator_length=d.actuator_length.copy(),actuator_velocity=d.actuator_velocity.copy(),PD_offset20=offset,
       ctrlrange=m.actuator_ctrlrange.copy(),actuator_forcerange=m.actuator_forcerange.copy(),joint_actfrcrange=m.jnt_actfrcrange[:20].copy(),torque_lower=lower,torque_upper=upper,
       qfrc_passive26=d.qfrc_passive.copy(),qfrc_applied26=d.qfrc_applied.copy(),xfrc_applied=d.xfrc_applied.copy())
    save(out/'LP_result.json',plain(lp))
    report={'schema':'hand2_seed2_static_support_and_preload_preparation:1','status':'COMPLETED_NONINTEGRATING_CONDITIONAL_SUPPORT_PREPARATION',
       'preregistration':pin(HERE/'preregistration.json'),'source':pin(__file__),'input_pins':c['pins'],'compiled_binding':binding,
       'geometry_recheck_passed':True,'pad_gaps_m':[x['gap_m'] for x in contacts],'normal_source':'Independent compiled-mesh convex closest-gap query; object force normal is negative HAND exit normal',
       'contact_points_scope':'Barycentric actual closest feature pair; positive gap is potential closure, not current contact force',
       'mass_kg_assumed':.2,'actual_pair_sliding_mu_assumed':.7,'design_mu':mu,'cone_inclusion':'inscribed8 subset of circular mu/sqrt(2) cone subset of any orientation native mu L1 pyramid; no torsion',
       'actual_compiled_pad_pairs':pair_evidence,
       'LP_feasible':lp['feasible'],'controller_variants':variants,
       'limits_evidence':{'unclamped_PD_torque_lower':pdlo.tolist(),'unclamped_PD_torque_upper':pdhi.tolist(),'intersection_lower':lower.tolist(),'intersection_upper':upper.tolist(),'no_command_or_force_clipping_in_preparation':True},
       'counts':{'compiled_MjModel_loads':1,'MjData':1,'mj_forward_nonintegrating':1,'mj_jac':2,'LP_support_calls':1,'integration_steps':0,'optimization_pose_calls':0,'training_or_GPU':0},
       'scope':'Nominal assumed parameters; candidate patch represented by ideal convex sensor colliders. Not calibrated material, actual force/holding, finite-patch impossibility or dynamic stability evidence.',
       'holding_qualified':False,'outputs':{}}
    for f in out.iterdir():report['outputs'][f.name]=pin(f)
    save(out/'report.json',plain(report));print(json.dumps({'status':report['status'],'LP_feasible':lp['feasible'],'variants':[{k:v[k] for k in ['name','reachable']} for v in variants],'report':pin(out/'report.json')}))


if __name__=='__main__':main()
