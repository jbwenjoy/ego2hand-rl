"""H123 q32 native-contact wholehand PD/stiction LPs. CPU static surrogate only.

Both geometric candidates and both LPs are retained. Baseline is canonical;
zero-stiction mathematical feasibility takes precedence, else bounded-stiction.
Selected u is cast once; no target/force clipping or alternate-witness retry.
No physical stepping, forward, inverse dynamics, solved forces or GPU.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import mujoco
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('D:/workspace/supermimic-observation-retarget-study')
WARM=Path('D:/workspace/supermimic-observation-warmseed-grounding-study')
CONE=Path('D:/workspace/supermimic-observation-grounding-study/scripts/diagnose_rest_wrench_feasibility_cpu_v2.py')
CONE_SHA='ba3f0f934ff7cc2ce6d674c92f3a755487bcc0994f04736664c5ceb87f31fdca'
sys.path.insert(0,str(OLD/'scripts'))
from native_filtered_contact_geometry import NativeFilteredContactGeometry
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pin=lambda p:{'path':str(p),'sha256':sha(p)}
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')
POLICY={'canonical':'baseline','retained_candidates':['baseline','midpoint'],'LPs':['zero_stiction','bounded_stiction'],'selection':'Zero-stiction mathematical feasible witness first; otherwise bounded-stiction. A cast failure blocks export without retry or switching LP.','objective':'min sum native normal-edge weights','numerical_torque_row_length_scale_m':.01,'LP_primal_dual_tolerance':1e-9,'LP_time_limit_seconds':10,'CPU_soft_wall_seconds':60,'postcast_policy':'FP32 u must be bitexact under pinned C2 and both affine/difference CPU FP32 PD evaluations remain inside actuator and jointactforce caps. Numerical equilibrium residual compared with explicit quantization+roundoff bound; no physical tolerance is changed.','LP_numeric_inset_buffer_rad':0.,'physical_reference_qualified':False,'GPU_authorized':False}
assert sha(CONE)==CONE_SHA
spec=importlib.util.spec_from_file_location('h123_preload_pinned_cone',CONE);cone=importlib.util.module_from_spec(spec);spec.loader.exec_module(cone)

def build(plan,q64,pose64):
    q32,pose32=q64.astype(np.float32),pose64.astype(np.float32)
    geom=NativeFilteredContactGeometry(plan['model']['path'],plan['context']['path']);m,d=geom.model,geom.data
    physical={}
    with np.load(plan['context']['path'],allow_pickle=False) as z:
        for key in ('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm'):
            physical[key]=z[key][0].copy();assert np.array_equal(z[key],np.broadcast_to(z[key][:1],z[key].shape));getattr(m,key)[:]=physical[key]
    geom.set_object_pose(pose32.astype(float));geom.robot.last_q=None;native=geom.evaluate(q32.astype(float))
    robot=geom.robot;obj=m.body('object/cube').id;object_geoms=set(geom.object_ids.tolist());hand=set(geom.hand_ids.tolist())
    allcontacts=[];contacts=[];unsupported=[]
    for i,c in enumerate(d.contact):
        g1,g2=map(int,c.geom);hasobj=g1 in object_geoms or g2 in object_geoms;hashand=g1 in hand or g2 in hand
        category='hand_object' if hasobj and hashand else 'hand_self' if g1 in hand and g2 in hand else 'other'
        row={'contact_id':i,'geom':[g1,g2],'names':[m.geom(g1).name,m.geom(g2).name],'category':category,
          'distance_m':float(c.dist),'position_w':c.pos.tolist(),'frame_rows_geom0_to_geom1_w':c.frame.reshape(3,3).tolist(),
          'effective_condim':int(c.dim),'friction5':c.friction.tolist()};allcontacts.append(row)
        if category!='hand_object':unsupported.append('Non-HO contact would require additional generalized-force terms: '+str(row['geom']));continue
        sign=1. if g2 in object_geoms else -1.;frame=sign*c.frame.reshape(3,3).copy();hg=g1 if g1 in hand else g2
        row['frame_rows_force_on_object_w']=frame.tolist();row['hand_body_id']=int(m.geom_bodyid[hg])
        assert int(c.dim) in (1,3,4,6) and np.isfinite(frame).all()
        priorities=(int(m.geom_priority[g1]),int(m.geom_priority[g2]))
        if priorities[0]!=priorities[1]:selected=g1 if priorities[0]>priorities[1] else g2;mu=np.asarray(m.geom_friction[selected]);condim=int(m.geom_condim[selected])
        else:mu=np.maximum(m.geom_friction[g1],m.geom_friction[g2]);condim=int(max(m.geom_condim[g1],m.geom_condim[g2]))
        expected=np.array([mu[0],mu[0],mu[1],mu[2],mu[2]])
        assert condim==int(c.dim) and np.array_equal(expected,c.friction),'Actual material mix mismatch'
        contacts.append({'id':str(i),'frame':frame,'point':c.pos.copy(),'friction':c.friction.copy(),'condim':condim,'hand_body':int(m.geom_bodyid[hg])})
    assert contacts,'No canonical native HO contacts'
    assert int(m.opt.cone)==int(mujoco.mjtCone.mjCONE_PYRAMIDAL) and m.npair==0 and not m.flg_adhesion
    gravity=np.asarray(m.opt.gravity,np.float32).astype(float);mass=float(physical['body_mass'][obj]);com=d.xipos[obj].copy();target=np.r_[-mass*gravity,np.zeros(3)]
    object_result,A=cone.solve(contacts,com,target)
    # Same q32 state, all bodies included. Object/free-body contributions
    # have zero columns on the robot20 DOFs and cannot be double counted.
    body_jac=[];body_com=[];body_g=[]
    for b in range(m.nbody):
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBodyCom(m,d,jp,jr,b)
        J=jp[:,robot.va];body_jac.append(J);body_com.append(d.xipos[b].copy());body_g.append(J.T@(float(physical['body_mass'][b])*gravity))
    tau_g=np.sum(body_g,axis=0);Gblocks=[];direct_blocks=[]
    for c in contacts:
        world,local=cone.native_pyramid_rays(c['frame'],c['point'],com,c['friction'],c['condim'])
        force=c['frame'].T@local[:3];direct=c['frame'].T@local[3:]
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,c['point'],c['hand_body'])
        Gblocks.append(-jp[:,robot.va].T@force-jr[:,robot.va].T@direct);direct_blocks.append(direct)
    G=np.concatenate(Gblocks,axis=1)
    jids=np.array([m.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
    aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j).item() for j in jids]);kp=physical['actuator_gainprm'][aids,0].astype(float);bias=physical['actuator_biasprm'][aids].astype(float)
    lower=np.maximum(m.jnt_range[jids,0],m.actuator_ctrlrange[aids,0]);upper=np.minimum(m.jnt_range[jids,1],m.actuator_ctrlrange[aids,1])
    lower32,upper32=lower.astype(np.float32),upper.astype(np.float32);innerlo=lower32+np.float32(.020);innerhi=upper32-np.float32(.020)
    actual_force_bounds=m.actuator_forcerange[aids].astype(np.float32).astype(float)
    joint_force_bounds=m.jnt_actfrcrange[jids].astype(np.float32).astype(float)
    tau_lo=np.maximum(kp*(innerlo.astype(float)-q32.astype(float)),np.maximum(actual_force_bounds[:,0],joint_force_bounds[:,0]))
    tau_hi=np.minimum(kp*(innerhi.astype(float)-q32.astype(float)),np.minimum(actual_force_bounds[:,1],joint_force_bounds[:,1]))
    mujoco.mj_transmission(m,d)
    passive={'body_gravcomp':m.body_gravcomp.tolist(),'joint_stiffness':m.jnt_stiffness.tolist(),
      'dof_damping':m.dof_damping[robot.va].astype(np.float32).tolist(),'damping_effect_at_designed_qd0':'exactly0',
      'robot_dof_frictionloss':m.dof_frictionloss[robot.va].astype(np.float32).tolist(),'frictionloss_nonzero_count':int(np.count_nonzero(m.dof_frictionloss[robot.va])),
      'static_friction_policy':'Explicit zero-stiction restricted witness versus independent bounded zeta in[-loss,+loss] relaxed existence; neither predicts actual solved friction.',
      'tendons':int(m.ntendon),'equality_constraints':int(m.neq),'flex':int(m.nflex),'fluid_density':float(m.opt.density),'fluid_viscosity':float(m.opt.viscosity),
      'joint_min_hard_margin_rad':float(np.minimum(q32-m.jnt_range[jids,0],m.jnt_range[jids,1]-q32).min()),'joint_margin':m.jnt_margin[jids].tolist(),
      'joint_limit_reaction':'Excluded only because candidate is strictly outside all limit/margin activation bands; not modeled for a limit-active candidate.'}
    model_checks={'no_gravcomp':not np.any(m.body_gravcomp),'no_springs':not np.any(m.jnt_stiffness),'no_tendon_equality_flex':m.ntendon==m.neq==m.nflex==0,
      'fluid0':m.opt.density==m.opt.viscosity==0,'gravity_enabled':not (int(m.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)),
      'gear_unit_joint':np.array_equal(m.actuator_gear[aids],np.tile([1.,0,0,0,0,0],(20,1))) and np.all(m.actuator_trntype[aids]==int(mujoco.mjtTrn.mjTRN_JOINT)),
      'affine_position_PD':bool(np.all(m.actuator_gaintype[aids]==int(mujoco.mjtGain.mjGAIN_FIXED)) and np.all(m.actuator_biastype[aids]==int(mujoco.mjtBias.mjBIAS_AFFINE)) and np.all(m.actuator_dyntype[aids]==int(mujoco.mjtDyn.mjDYN_NONE)) and np.all(kp>0) and not bias[:,0].any() and np.array_equal(bias[:,1],-kp) and not bias[:,3:].any()),
      'transmission_length_q':np.array_equal(d.actuator_length[aids],q32.astype(float)),'no_actuator_plugins':bool(np.all(m.actuator_plugin[aids]<0)),
      'q_strictly_outside_limit_bands':bool(np.all(q32>m.jnt_range[jids,0]+m.jnt_margin[jids]) and np.all(q32<m.jnt_range[jids,1]-m.jnt_margin[jids])),
      'force_limits_enabled':bool(np.all(m.actuator_forcelimited[aids]) and np.all(m.jnt_actfrclimited[jids])),
      'no_other_generated_contacts':not unsupported,'q32_hard_legal':bool(np.all(q32>=m.jnt_range[jids,0]) and np.all(q32<=m.jnt_range[jids,1]))}
    # Preserve exact source64 versus actual32 domain distinction at a hardlimit.
    model_checks['q32_in_original_float64_hard_range']=model_checks.pop('q32_hard_legal')
    model_checks['q32_actual_FP32_hard_legal']=bool(np.all(q32>=m.jnt_range[jids,0].astype(np.float32)) and np.all(q32<=m.jnt_range[jids,1].astype(np.float32)))
    active=np.flatnonzero(np.minimum(q32-m.jnt_range[jids,0].astype(np.float32),m.jnt_range[jids,1].astype(np.float32)-q32)<=m.jnt_margin[jids].astype(np.float32))
    passive['potential_joint_limit_activation_joint_indices']=active.tolist()
    passive['joint_limit_reaction']='Excluded from this restricted surrogate even when q is on a limit/margin boundary. No infeasibility claim for full physical dynamics.'
    passive['static_friction_policy']='Separate zero-zeta and independently bounded zeta in +/-actual FP32 loss LPs. Selected zeta is a mathematical witness, never a solved-friction prediction.'
    ignore={'q32_in_original_float64_hard_range','q_strictly_outside_limit_bands'}
    unsupported.extend('Unsupported model contract: '+k for k,v in model_checks.items() if not v and k not in ignore)
    return dict(q64=q64,pose64=pose64,q32=q32,pose32=pose32,contacts=allcontacts,object_LP=object_result,A=A,G=G,tau_g=tau_g,target=target,com=com,body_jac=np.asarray(body_jac),body_com=np.asarray(body_com),body_g=np.asarray(body_g),body_mass=physical['body_mass'],kp=kp,bias=bias,lower=lower,upper=upper,lower32=lower32,upper32=upper32,innerlo=innerlo,innerhi=innerhi,act_force_bounds=actual_force_bounds,joint_force_bounds=joint_force_bounds,tau_lo=tau_lo,tau_hi=tau_hi,loss=m.dof_frictionloss[robot.va].astype(np.float32).astype(float),passive=passive,checks=model_checks,unsupported=unsupported,actuator_ids=aids,joint_ids=jids,robot_dof_ids=robot.va.copy())

def solve_hand(d,bounded,projection):
    ne=d['A'].shape[1];Aeq=np.block([[d['A'],np.zeros((6,40))],[d['G'],np.eye(20),np.eye(20)]])
    b=np.r_[d['target'],-d['tau_g']];scale=np.r_[np.ones(3),np.full(23,100.)]
    bounds=[(0,None)]*ne+list(zip(d['tau_lo'],d['tau_hi']))+list(zip(-d['loss'],d['loss'])) if bounded else [(0,None)]*ne+list(zip(d['tau_lo'],d['tau_hi']))+[(0,0)]*20
    report={'kind':'bounded_stiction' if bounded else 'zero_stiction','mathematical_feasible':False,'normal_force_cap_N':None,'zeta_bounds_Nm':list(zip(-d['loss'],d['loss'])) if bounded else [[0.,0.]]*20,'force_or_target_clipping_used':False,'buffer_added_to_inner_domain_rad':0.}
    if np.any(d['tau_lo']>d['tau_hi']):report.update(status='EMPTY_PD_TORQUE_INTERVAL',empty_joint_indices=np.flatnonzero(d['tau_lo']>d['tau_hi']).tolist());return report,None,Aeq,b
    lp=linprog(np.r_[np.ones(ne),np.zeros(40)],A_eq=scale[:,None]*Aeq,b_eq=scale*b,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':10.})
    report.update(solver_status=int(lp.status),solver_message=str(lp.message))
    if not lp.success:return report,None,Aeq,b
    lam,tau,zeta=lp.x[:ne],lp.x[ne:ne+20],lp.x[ne+20:];eq=Aeq@lp.x-b
    nonnegative=bool(np.all(lam>=0));zbound=bool(np.all(np.abs(zeta)<=d['loss'])) if bounded else bool(not np.any(zeta))
    mathematical=nonnegative and zbound and bool(np.max(np.abs(eq))<1e-8)
    u64=d['q32'].astype(float)+tau/d['kp'];u32=u64.astype(np.float32);projected=projection(u32,d['lower32'],d['upper32'])
    q32=d['q32'];kp32=d['kp'].astype(np.float32);bias32=d['bias'].astype(np.float32)
    pd_difference32=kp32*(u32-q32)
    pd_affine32=kp32*u32+(bias32[:,0]+bias32[:,1]*q32)
    pd_affine_u32_f64=d['kp']*u32.astype(float)+d['bias'][:,0]+d['bias'][:,1]*q32.astype(float)
    act=d['act_force_bounds'];joint=d['joint_force_bounds']
    def inside(t,bound):return bool(np.all(t>=bound[:,0]) and np.all(t<=bound[:,1]))
    force_ok=all(inside(t,bound) for t in [pd_difference32.astype(float),pd_affine32.astype(float),pd_affine_u32_f64] for bound in [act,joint])
    c2identity=bool(np.array_equal(u32.view(np.uint32),projected.view(np.uint32)))
    balance_difference=pd_difference32.astype(float)+d['tau_g']+d['G']@lam+zeta
    balance_affine=pd_affine32.astype(float)+d['tau_g']+d['G']@lam+zeta
    eps=np.finfo(np.float32).eps
    # This is an arithmetic-error check, not a new physical acceptance margin.
    numerical_bound=d['kp']*np.abs(u32.astype(float)-u64)+8*eps*(np.abs(d['kp']*u32.astype(float))+np.abs(d['bias'][:,1]*q32.astype(float))+np.abs(d['bias'][:,0])+1e-30)+np.abs(eq[6:])+1e-14
    numerical_ok=bool(np.all(np.abs(balance_difference)<=numerical_bound) and np.all(np.abs(balance_affine)<=numerical_bound))
    finite=bool(np.isfinite(lp.x).all() and np.isfinite(u32).all())
    eligible=mathematical and c2identity and force_ok and numerical_ok and finite
    report.update(mathematical_feasible=mathematical,status='FEASIBLE_STATIC_SURROGATE' if mathematical else 'SOLVER_SOLUTION_FAILED_NUMERIC_CONTRACT',sum_normal_ray_weights_N=float(lam.sum()),lambda_edge_weights_N=lam.tolist(),tau_act_math_Nm=tau.tolist(),zeta_stiction_witness_Nm=zeta.tolist(),u64_mathematical_rad=u64.tolist(),u32_candidate_rad=u32.tolist(),object_contact_wrench=(d['A']@lam).tolist(),object_equilibrium_residual=eq[:6].tolist(),hand_equilibrium_residual_math_Nm=eq[6:].tolist(),nonnegative_lambda_exact=nonnegative,zeta_within_declared_domain=zbound,post_cast_C2_bitexact_identity=c2identity,post_cast_force_bounds_exact=force_ok,PD_unclipped_difference_FP32_Nm=pd_difference32.tolist(),PD_unclipped_affine_FP32_Nm=pd_affine32.tolist(),PD_affine_with_FP32_inputs_computed_FP64_Nm=pd_affine_u32_f64.tolist(),PD_after_cast_vs_math_tau_Nm=(pd_affine32.astype(float)-tau).tolist(),hand_balance_residual_after_FP32_affine_PD_Nm=balance_affine.tolist(),hand_balance_residual_after_FP32_difference_PD_Nm=balance_difference.tolist(),explicit_FP32_arithmetic_error_bound_Nm=numerical_bound.tolist(),FP32_residual_within_arithmetic_bound=numerical_ok,finite=finite,post_cast_export_admissible=eligible,PD_force_semantics='CPU surrogate evaluation of actual affine gain/bias at qd=0; not a Warp measured actuator force or FMA equivalence claim.',stiction_semantics='Mathematical witness, not predicted actual static friction.')
    allocations=[];offset=0
    for row in d['contacts']:
        world,local=cone.native_pyramid_rays(np.array(row['frame_rows_force_on_object_w']),np.array(row['position_w']),d['com'],np.array(row['friction5']),row['effective_condim'])
        count=world.shape[1];weights=lam[offset:offset+count];wrench=world@weights
        allocations.append({'contact_id':row['contact_id'],'geom':row['geom'],'edge_start':offset,'edge_stop_exclusive':offset+count,'normal_force_sum_N':float(weights.sum()),'force_on_object_world_N':wrench[:3].tolist(),'torque_on_object_about_COM_world_Nm':wrench[3:].tolist(),'direct_contact_torque_on_object_world_Nm':(np.array(row['frame_rows_force_on_object_w']).T@local[3:]@weights).tolist(),'generalized_force_on_hand_Nm':(d['G'][:,offset:offset+count]@weights).tolist()});offset+=count
    assert offset==ne
    report['per_contact_witness_wrench']=allocations
    return report,lp.x,Aeq,b


def main():
    started=time.perf_counter();out=ROOT/'outputs/h123_preload_cpu_v1';out.mkdir(parents=True,exist_ok=False)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());write(out/'policy_before_LP.json',POLICY)
    plans={s:read(WARM/f'configs/h123-rest-{s}-review-v1.json') for s in ['baseline','midpoint']}
    sources={}
    for seed,p in plans.items():
        directory=ROOT/f'outputs/frame0_h123_seeded_{seed}_v3'
        assert p['candidate']['path']==str(directory/'initialization_candidate.npz')
        sources[seed]={'canonical_candidate':p['candidate'],'model':p['model'],'context':p['context'],'pure_projection':p['source_pins']['pure_projection'],'original_pure_projection':p['source_pins']['original_pure_projection'],'native_geometry':pin(OLD/'scripts/native_filtered_contact_geometry.py'),'kinematics':pin(OLD/'scripts/retarget_palmroot_cpu.py'),'native_pyramid_helper':pin(CONE),'saved_q64_native_contacts':pin(directory/'native_contacts.json'),'frozen_material_bindings':pin(directory/'frozen_anchor_bindings.json'),'geometric_report':pin(directory/'report.json'),'review_plan':pin(WARM/f'configs/h123-rest-{seed}-review-v1.json'),'design_source':pin(__file__)}
        for v in sources[seed].values():assert sha(v['path'])==v['sha256']
    write(out/'source_pins_before_LP.json',sources)
    import torch
    specp=importlib.util.spec_from_file_location('pinned_preload_projection',plans['baseline']['source_pins']['pure_projection']['path']);pure=importlib.util.module_from_spec(specp);specp.loader.exec_module(pure)
    def projection(u,lower,upper):return pure.target_inset(torch.from_numpy(u.copy()),torch.from_numpy(lower.copy()),torch.from_numpy(upper.copy()),.020).numpy()
    forbidden={k:0 for k in ['mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_contactForce']};saved={k:getattr(mujoco,k) for k in forbidden}
    def deny(k):
        def fail(*a,**kw):forbidden[k]+=1;raise AssertionError('Forbidden physical/force API '+k)
        return fail
    index={}
    try:
        for k in saved:setattr(mujoco,k,deny(k))
        for seed,p in plans.items():
            if time.perf_counter()-started>60:raise TimeoutError('Fixed60sCPU; no retry')
            caseout=out/seed;caseout.mkdir(exist_ok=False);begin=time.perf_counter()
            with np.load(p['candidate']['path'],allow_pickle=False) as z:q64=z['joint_pos_rad'][0].copy();pose64=z['mapped_object_pose_w'][0].copy()
            d=build(p,q64,pose64)
            blocked=bool(d['unsupported']) or not d['object_LP']['feasible_in_native_object_wrench_screen']
            lp_results={};xs={};eqs={}
            for bounded,name in [(False,'zero_stiction'),(True,'bounded_stiction')]:
                if blocked:result={'status':'NOT_SOLVED_UNSUPPORTED_GEOMETRY_OR_OBJECT_PREREQUISITE','mathematical_feasible':False};x,Aeq,b=None,None,None
                else:result,x,Aeq,b=solve_hand(d,bounded,projection)
                lp_results[name]=result;xs[name]=x;eqs[name]=(Aeq,b)
            selected='zero_stiction' if lp_results['zero_stiction']['mathematical_feasible'] else 'bounded_stiction' if lp_results['bounded_stiction']['mathematical_feasible'] else None
            chosen=lp_results[selected] if selected else None
            feasible=bool(chosen and chosen['post_cast_export_admissible'] and not blocked)
            status='FEASIBLE_RESTRICTED_STATIC_PD_WITNESS' if feasible else 'BLOCKED_UNSUPPORTED_OR_OBJECT_PREREQUISITE' if blocked else 'BLOCKED_POST_CAST_COMMAND_CONTRACT' if selected else 'NO_FEASIBLE_WHOLEHAND_STATIC_SURROGATE'
            arrays=dict(A_object_wrench=d['A'],G_hand_generalized_contact=d['G'],gravity_robot_tau=d['tau_g'],target_object_wrench=d['target'],object_COM_world=d['com'],all_body_COM_Jacobian=d['body_jac'],all_body_COM_world=d['body_com'],all_body_gravity_tau=d['body_g'],body_mass=d['body_mass'],q32=d['q32'],pose32=d['pose32'],kp=d['kp'],actual_bias=d['bias'],outer_lower64=d['lower'],outer_upper64=d['upper'],inset_lower32=d['innerlo'],inset_upper32=d['innerhi'],tau_lower=d['tau_lo'],tau_upper=d['tau_hi'],actuator_force_bounds=d['act_force_bounds'],joint_force_bounds=d['joint_force_bounds'],stiction_loss=d['loss'],actuator_ids=d['actuator_ids'],joint_ids=d['joint_ids'],robot_dof_ids=d['robot_dof_ids'])
            for name,x in xs.items():
                if eqs[name][0] is not None:arrays[name+'_Aeq']=eqs[name][0];arrays[name+'_beq']=eqs[name][1]
                if x is not None:arrays[name+'_full_solution']=x
            np.savez_compressed(caseout/'static_matrices.npz',**arrays)
            write(caseout/'q32_native_contacts.json',d['contacts'])
            limits=['Restricted static existence surrogate only; no Warp force/PD/friction prediction or dynamic qualification.','Joint-limit reactions excluded even for potential boundary activation; infeasibility is not full physical impossibility.','Soft-contact force versus depth/solref/solimp constitutive law is not enforced.','No arbitrary contact normal-force cap; actuator and jointactforce limits and actual FP32 C2 domain are enforced.','Selected zeta is a mathematical witness, not a prediction of realized static friction.','Both predeclared candidates retained; canonical baseline selection precedes these results and is not changed by quality.','CPU FP32 affine/difference arithmetic is recorded, not claimed identical to Warp FMA or solved actuator force.']
            report={'status':status,'seed':seed,'canonical':seed=='baseline','policy':POLICY,'q32_object_static_LP':d['object_LP'],'LPs':lp_results,'selected_LP':selected,'selected_post_cast_admissible':feasible,'q32_contacts':len(d['contacts']),'self_or_other_unsupported':d['unsupported'],'model_contract_checks':d['checks'],'passive_terms':d['passive'],'equations':{'object':'A*lambda= -m*g with zero COM torque','hand':'tau_act+tau_g+G*lambda+zeta=0; G is force ON HAND','actuation':'actual gear1: gain*u + bias0 + bias1*q + bias2*qd; qd=0, bias1=-kp. tau_act=-tau_g-G*lambda-zeta','G':'Point Jacobian includes r_cross_F. Rotation Jacobian maps DIRECT torque only; object-side contact wrench is negated onto hand.'},'source_pins':sources[seed],'matrix_SHA256':sha(caseout/'static_matrices.npz'),'q32_native_contacts_SHA256':sha(caseout/'q32_native_contacts.json'),'limits':limits,'physical_reference_qualified':False,'GPU_used':False,'physics_steps':0,'CPU_case_wall_s_excluding_imports':time.perf_counter()-begin}
            write(caseout/'report.json',report)
            design={'schema':'seeded_static_PD_preload:1','status':status,'feasible':feasible,'canonical_candidate':p['candidate'],'canonical':seed=='baseline','seed':seed,'q0_rad':q64.tolist(),'q0_FP32_rad':d['q32'].tolist(),'object_pose_w':pose64.tolist(),'object_pose_w_FP32':d['pose32'].tolist(),'u0_rad':chosen['u32_candidate_rad'] if feasible else None,'C2_margin_rad':.020,'target_lower_rad':d['lower'].tolist(),'target_upper_rad':d['upper'].tolist(),'inset_lower_rad':d['innerlo'].tolist(),'inset_upper_rad':d['innerhi'].tolist(),'source_pins':sources[seed],'selected_LP':selected,'selected_mathematical_witness':chosen,'all_LPs':lp_results,'stiction_witness_is_prediction':False,'joint_limit_reactions_excluded':True,'softcontact_constitutive_enforced':False,'no_force_or_target_clipping':True,'physical_reference_qualified':False,'no_dynamic_qualification':True,'report_sha256':sha(caseout/'report.json'),'static_matrices_sha256':sha(caseout/'static_matrices.npz'),'limits':limits}
            write(caseout/'designed_target_u0.json',design)
            index[seed]={'status':status,'feasible':feasible,'selected_LP':selected,'source_contacts_q32':len(d['contacts']),'report':pin(caseout/'report.json'),'design':pin(caseout/'designed_target_u0.json'),'LP_math_feasibility':{k:v['mathematical_feasible'] for k,v in lp_results.items()},'postcast_admissibility':{k:v.get('post_cast_export_admissible') for k,v in lp_results.items()}}
            print(json.dumps({seed:index[seed]},indent=2))
    except BaseException as exc:
        write(out/'engineering_failure.json',{'error':repr(exc),'completed':index,'no_retry':True});raise
    finally:
        for k,v in saved.items():setattr(mujoco,k,v)
    assert not any(forbidden.values()) and not torch.cuda.is_initialized()
    summary={'status':'COMPLETE_BOTH_FIXED_H123_WHOLEHAND_STATIC_PD_DESIGNS','cases':index,'policy':POLICY,'source_SHA256':sha(__file__),'forbidden_calls':forbidden,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'physical_reference_qualified':False,'GPU_used':False,'physics_steps':0,'CPU_wall_seconds_excluding_imports':time.perf_counter()-started}
    write(out/'report.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
