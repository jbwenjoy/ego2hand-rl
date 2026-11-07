"""Independent static-witness audit; never calls the design build/solve code.

Native CPU kinematics/Jacobians and one collision-list check only. No dynamics,
inverse dynamics, solved forces, LP re-selection, new target or GPU.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('D:/workspace/supermimic-observation-contact-intent-study/outputs/h123_preload_cpu_v1')
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/h123_preload_independent_review_v1');args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    final=read(SOURCE/'FINAL_PINS.json');design=read(SOURCE/'baseline/designed_target_u0.json');report=read(SOURCE/'baseline/report.json');records=read(SOURCE/'baseline/q32_native_contacts.json');z=np.load(SOURCE/'baseline/static_matrices.npz',allow_pickle=False)
    checks={};errors={};details={};proof_arrays={}
    checks['FINAL_PINS_frozen_SHA']=sha(SOURCE/'FINAL_PINS.json')=='404dff8c62cbf84916e72df28332bea5b199dd1fc6fea22f7e8deb43e0efcadd'
    checks['all_FINAL_PINS_files_match']=all(sha(v['path'])==v['sha256'] for v in final.values() if isinstance(v,dict) and 'path'in v and 'sha256'in v)
    checks['all_design_source_pins_match']=all(sha(p['path'])==p['sha256'] for p in design['source_pins'].values())
    checks['design_report_matrix_bindings']=design['report_sha256']==sha(SOURCE/'baseline/report.json') and design['static_matrices_sha256']==sha(SOURCE/'baseline/static_matrices.npz') and report['q32_native_contacts_SHA256']==sha(SOURCE/'baseline/q32_native_contacts.json')
    checks['canonical_selected_baseline_no_midpoint_substitution']=design['canonical'] is True and design['seed']=='baseline' and design['feasible'] is True
    with np.load(design['canonical_candidate']['path'],allow_pickle=False) as candidate:
        q64=candidate['joint_pos_rad'][0].copy();pose64=candidate['mapped_object_pose_w'][0].copy()
    q32=q64.astype(np.float32);pose32=pose64.astype(np.float32)
    checks['q64_pose64_exact_canonical_candidate']=np.array_equal(q64,design['q0_rad']) and np.array_equal(pose64,design['object_pose_w'])
    checks['q32_pose32_exact_cast_and_saved_matrices']=np.array_equal(q32,design['q0_FP32_rad']) and np.array_equal(pose32,design['object_pose_w_FP32']) and np.array_equal(q32,z['q32']) and np.array_equal(pose32,z['pose32'])
    model=mujoco.MjModel.from_binary_path(design['source_pins']['model']['path']);data=mujoco.MjData(model)
    context={}
    with np.load(design['source_pins']['context']['path'],allow_pickle=False) as c:
        for name in FIELDS:
            v=c[name];assert np.isfinite(v).all() and np.array_equal(v,np.broadcast_to(v[:1],v.shape));context[name]=v[0].copy();getattr(model,name)[:]=v[0]
    ids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)]);qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids];aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in ids]);obj=model.body('object/cube').id;oj=model.joint('object/cube_freejoint').id;oq=int(model.jnt_qposadr[oj]);object_ids=set(np.flatnonzero(model.geom_bodyid==obj).tolist())
    data.qpos[qa]=q32.astype(float);data.qpos[oq:oq+7]=pose32.astype(float);data.qvel[:]=0
    forbidden={n:0 for n in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_contactForce','mj_geomDistance')}
    def deny(name):
        def f(*a,**kw):forbidden[name]+=1;raise AssertionError('Forbidden dynamics/force '+name)
        return f
    saved={n:getattr(mujoco,n) for n in forbidden}
    for n in forbidden:setattr(mujoco,n,deny(n))
    def kinematics(q):
        data.qpos[qa]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
    try:
        kinematics(q32.astype(float));mujoco.mj_transmission(model,data);mujoco.mj_collision(model,data)
        checks['native_CPU_contactlist_count_and_clock']=len(data.contact)==len(records)==4 and data.time==0 and not np.any(data.qvel)
        contact_error=[]
        for c,r in zip(data.contact,records):
            contact_error.extend([float(np.max(np.abs(c.pos-r['position_w']))),float(np.max(np.abs(c.frame.reshape(3,3)-r['frame_rows_geom0_to_geom1_w']))),abs(float(c.dist)-r['distance_m'])])
            assert list(map(int,c.geom))==r['geom'] and c.dim==r['effective_condim'] and np.array_equal(c.friction,r['friction5'])
        errors['requeried_native_contact_geometry_max_abs']=max(contact_error);checks['native_CPU_contactlist_reproduces_saved']=max(contact_error)<1e-12
        gravity=model.opt.gravity.astype(np.float32).astype(float);mass=context['body_mass'].astype(float);com=data.xipos[obj].copy();target=np.r_[-mass[obj]*gravity,np.zeros(3)]
        Acols=[];Gcols=[];directcols=[];material_locals=[];hand_bodies=[];edge_contact=[]
        for contact_index,r in enumerate(records):
            a,b=r['geom'];sign=1. if b in object_ids else -1.;frame=sign*np.array(r['frame_rows_geom0_to_geom1_w']);n,t1,t2=frame;point=np.array(r['position_w']);hand=a if b in object_ids else b;body=int(model.geom_bodyid[hand]);hand_bodies.append(body)
            checks[f'contact{contact_index}_force_on_object_normal_sign']=np.array_equal(frame,r['frame_rows_force_on_object_w']) and body==r['hand_body_id']
            pa,pb=int(model.geom_priority[a]),int(model.geom_priority[b]);selected=a if pa>pb else b;mu=model.geom_friction[selected] if pa!=pb else np.maximum(model.geom_friction[a],model.geom_friction[b]);dim=int(model.geom_condim[selected]) if pa!=pb else max(int(model.geom_condim[a]),int(model.geom_condim[b]));expected=np.array([mu[0],mu[0],mu[1],mu[2],mu[2]])
            checks[f'contact{contact_index}_actual_material_priority_condim4']=dim==4 and np.array_equal(expected,r['friction5'])
            pJ,rJ=np.zeros((3,model.nv)),np.zeros((3,model.nv));mujoco.mj_jac(model,data,pJ,rJ,point,body)
            Rbody=data.xmat[body].reshape(3,3).copy();material_locals.append(Rbody.T@(point-data.xpos[body]))
            force_direct=[]
            for tangent,friction in ((t1,mu[0]),(t2,mu[0])):
                for s in (1.,-1.):force_direct.append((n+s*friction*tangent,np.zeros(3)))
            for s in (1.,-1.):force_direct.append((n,s*mu[1]*n))
            for force,direct in force_direct:
                Acols.append(np.r_[force,np.cross(point-com,force)+direct]);Gcols.append(-pJ[:,va].T@force-rJ[:,va].T@direct);directcols.append(direct);edge_contact.append(contact_index)
        A=np.array(Acols).T;G=np.array(Gcols).T;direct=np.array(directcols).T
        bodyJ=[];bodyG=[]
        for body in range(model.nbody):
            jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv));mujoco.mj_jacBodyCom(model,data,jp,jr,body);J=jp[:,va];bodyJ.append(J);bodyG.append(J.T@(mass[body]*gravity))
        bodyJ=np.array(bodyJ);bodyG=np.array(bodyG);tau_g=bodyG.sum(axis=0)
        for name,rebuilt,stored in (('A',A,z['A_object_wrench']),('G',G,z['G_hand_generalized_contact']),('body_Jcom',bodyJ,z['all_body_COM_Jacobian']),('body_gravity',bodyG,z['all_body_gravity_tau']),('gravity_tau',tau_g,z['gravity_robot_tau']),('object_target',target,z['target_object_wrench'])):
            error=float(np.max(np.abs(rebuilt-stored)));errors[name+'_max_abs']=error;checks[name+'_independently_rebuilt']=error<1e-12
        # Independent virtual-work FD: force acts at a material point on hand;
        # only direct torque uses body angular displacement. No r×F duplication.
        FDG=np.zeros_like(G);FDgrav=np.zeros(20);eps=1e-6
        for j in range(20):
            point_pm=[];rotation_pm=[];work_pm=[]
            for s in (1.,-1.):
                qq=q32.astype(float).copy();qq[j]+=s*eps;kinematics(qq)
                rotations=[data.xmat[b].reshape(3,3).copy() for b in hand_bodies]
                point_pm.append([data.xpos[b]+R@local for b,R,local in zip(hand_bodies,rotations,material_locals)]);rotation_pm.append(rotations)
                work_pm.append(float(np.sum(mass[:,None]*data.xipos*gravity)))
            dp=(np.array(point_pm[0])-np.array(point_pm[1]))/(2*eps)
            dw=np.array([Rotation.from_matrix(rp@rm.T).as_rotvec()/(2*eps) for rp,rm in zip(rotation_pm[0],rotation_pm[1])])
            for e,c in enumerate(edge_contact):FDG[j,e]=-dp[c]@A[:3,e]-dw[c]@direct[:,e]
            FDgrav[j]=(work_pm[0]-work_pm[1])/(2*eps)
        errors['G_virtual_work_FD_max_abs']=float(np.max(np.abs(FDG-G)));errors['gravity_potential_work_FD_max_abs_Nm']=float(np.max(np.abs(FDgrav-tau_g)))
        checks['G_sign_and_direct_only_virtual_work_FD']=errors['G_virtual_work_FD_max_abs']<1e-7
        checks['all_body_gravity_potential_FD']=errors['gravity_potential_work_FD_max_abs_Nm']<1e-7
        kinematics(q32.astype(float))
        kp=context['actuator_gainprm'][aids,0].astype(float);bias=context['actuator_biasprm'][aids].astype(float)
        outerlo=np.maximum(model.jnt_range[ids,0],model.actuator_ctrlrange[aids,0]);outerhi=np.minimum(model.jnt_range[ids,1],model.actuator_ctrlrange[aids,1]);lo32=outerlo.astype(np.float32);hi32=outerhi.astype(np.float32);innerlo=lo32+np.float32(.020);innerhi=hi32-np.float32(.020)
        actbounds=model.actuator_forcerange[aids].astype(np.float32).astype(float);jointbounds=model.jnt_actfrcrange[ids].astype(np.float32).astype(float);loss=model.dof_frictionloss[va].astype(np.float32).astype(float)
        taulo=np.maximum(kp*(innerlo.astype(float)-q32.astype(float)),np.maximum(actbounds[:,0],jointbounds[:,0]));tauhi=np.minimum(kp*(innerhi.astype(float)-q32.astype(float)),np.minimum(actbounds[:,1],jointbounds[:,1]))
        checks['actual_KP_bias_affine_position_PD']=np.array_equal(kp,z['kp']) and np.array_equal(bias,z['actual_bias']) and not bias[:,0].any() and np.array_equal(bias[:,1],-kp) and np.all(kp>0) and np.all(model.actuator_gaintype[aids]==int(mujoco.mjtGain.mjGAIN_FIXED)) and np.all(model.actuator_biastype[aids]==int(mujoco.mjtBias.mjBIAS_AFFINE))
        checks['actual_unit_gears_and_no_activation']=np.array_equal(model.actuator_gear[aids],np.tile([1.,0,0,0,0,0],(20,1))) and np.all(model.actuator_dyntype[aids]==int(mujoco.mjtDyn.mjDYN_NONE))
        checks['all_original_intersection_and_FP32_inset_bounds_exact']=np.array_equal(outerlo,design['target_lower_rad']) and np.array_equal(outerhi,design['target_upper_rad']) and np.array_equal(innerlo,design['inset_lower_rad']) and np.array_equal(innerhi,design['inset_upper_rad']) and np.array_equal(taulo,z['tau_lower']) and np.array_equal(tauhi,z['tau_upper'])
        checks['actual_actuator_joint_forcecaps_enabled_exact']=np.all(model.actuator_forcelimited[aids]) and np.all(model.jnt_actfrclimited[ids]) and np.array_equal(actbounds,z['actuator_force_bounds']) and np.array_equal(jointbounds,z['joint_force_bounds'])
        checks['stiction_actual20_loss_retained']=np.array_equal(loss,z['stiction_loss']) and np.count_nonzero(loss)==20
        checks['no_omitted_springs_gravcomp_fluid_tendon_equality_flex']=not np.any(model.body_gravcomp) and not np.any(model.jnt_stiffness) and model.ntendon==model.neq==model.nflex==0 and model.opt.density==model.opt.viscosity==0
        original_excess=np.maximum(np.maximum(model.jnt_range[ids,0]-q32.astype(float),q32.astype(float)-model.jnt_range[ids,1]),0)
        boundary=np.flatnonzero(np.minimum(q32-model.jnt_range[ids,0].astype(np.float32),model.jnt_range[ids,1].astype(np.float32)-q32)<=model.jnt_margin[ids].astype(np.float32))
        details['initial_q_range']={'source_q64_legal':bool(np.all(q64>=model.jnt_range[ids,0]) and np.all(q64<=model.jnt_range[ids,1])),'q32_legal_in_FP32_bounds':bool(np.all(q32>=model.jnt_range[ids,0].astype(np.float32)) and np.all(q32<=model.jnt_range[ids,1].astype(np.float32))),'q32_excess_vs_original_double_max_rad':float(original_excess.max()),'potential_limit_boundary_joint_indices':boundary.tolist(),'limit_reaction_not_modeled':True}
        checks['boundary_joint_not_misrepresented_as_interior']=boundary.tolist()==report['passive_terms']['potential_joint_limit_activation_joint_indices']==[3] and design['joint_limit_reactions_excluded'] is True
        checks['no_claim_constitutive_or_dynamic_qualification']=design['softcontact_constitutive_enforced'] is False and design['no_dynamic_qualification'] is True and design['physical_reference_qualified'] is False
    finally:
        for n,v in saved.items():setattr(mujoco,n,v)
    # Use the real pinned pure projection, not the design solver. CPU only.
    import torch
    spec=importlib.util.spec_from_file_location('independent_preload_actual_C2',design['source_pins']['pure_projection']['path']);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    lp_checks={};ne=A.shape[1];Aeq=np.block([[A,np.zeros((6,40))],[G,np.eye(20),np.eye(20)]]);beq=np.r_[target,-tau_g]
    for name in ('zero_stiction','bounded_stiction'):
        item=design['all_LPs'][name];x=z[name+'_full_solution'];lam=x[:ne];tau=x[ne:ne+20];zeta=x[ne+20:];eq=Aeq@x-beq
        checks[name+'_rebuilt_matrix_equations_and_saved_solution']=np.max(np.abs(Aeq-z[name+'_Aeq']))<1e-12 and np.array_equal(beq,z[name+'_beq']) and np.max(np.abs(eq))<1e-8 and np.array_equal(lam,item['lambda_edge_weights_N']) and np.array_equal(tau,item['tau_act_math_Nm']) and np.array_equal(zeta,item['zeta_stiction_witness_Nm'])
        checks[name+'_lambda_and_zeta_domain']=np.all(lam>=0) and (not zeta.any() if name=='zero_stiction' else np.all(np.abs(zeta)<=loss))
        checks[name+'_tau_in_actual_command_force_intersection']=np.all(tau>=taulo) and np.all(tau<=tauhi)
        u64=q32.astype(float)+tau/kp;u32=u64.astype(np.float32);projected=pure.target_inset(torch.tensor(u32),torch.tensor(lo32),torch.tensor(hi32),.020).numpy()
        checks[name+'_u64_u32_C2_bitexact_no_clipping']=np.array_equal(u64,item['u64_mathematical_rad']) and np.array_equal(u32,item['u32_candidate_rad']) and np.array_equal(u32.view(np.uint32),projected.view(np.uint32)) and np.array_equal(projected,np.clip(u32,innerlo,innerhi))
        pd_diff=kp.astype(np.float32)*(u32-q32);pd_affine=kp.astype(np.float32)*u32+(bias.astype(np.float32)[:,0]+bias.astype(np.float32)[:,1]*q32);pd64=kp*u32.astype(float)+bias[:,0]+bias[:,1]*q32.astype(float)
        checks[name+'_both_FP32_PD_groupings_and_FP64_exact']=np.array_equal(pd_diff,item['PD_unclipped_difference_FP32_Nm']) and np.array_equal(pd_affine,item['PD_unclipped_affine_FP32_Nm']) and np.array_equal(pd64,item['PD_affine_with_FP32_inputs_computed_FP64_Nm'])
        checks[name+'_postcast_all_PD_groups_inside_caps']=all(np.all(v>=b[:,0]) and np.all(v<=b[:,1]) for v in (pd_diff.astype(float),pd_affine.astype(float),pd64) for b in (actbounds,jointbounds))
        balance_affine=pd_affine.astype(float)+tau_g+G@lam+zeta;balance_diff=pd_diff.astype(float)+tau_g+G@lam+zeta
        # The declared arithmetic envelope uses the original saved LP residual;
        # independently reconstructed A/G differ at ordinary FP64 roundoff.
        # Check declaration bitexact, then test independently rebuilt balances.
        saved_eq=z[name+'_Aeq']@x-z[name+'_beq']
        eps=np.finfo(np.float32).eps;bound=kp*np.abs(u32.astype(float)-u64)+8*eps*(np.abs(kp*u32.astype(float))+np.abs(bias[:,1]*q32.astype(float))+np.abs(bias[:,0])+1e-30)+np.abs(saved_eq[6:])+1e-14
        checks[name+'_explicit_quantization_roundoff_bound_reproduces']=np.array_equal(bound,item['explicit_FP32_arithmetic_error_bound_Nm']) and np.all(np.abs(balance_affine)<=bound) and np.all(np.abs(balance_diff)<=bound)
        checks[name+'_report_feasibility_and_postcast_supported']=item['mathematical_feasible'] is True and item['post_cast_export_admissible'] is True and item['solver_status']==0
        lp_checks[name]={'lambda_sum_N':float(lam.sum()),'zeta_max_abs_Nm':float(np.max(np.abs(zeta))),'object_balance_max_abs_mixedunits':float(np.max(np.abs(eq[:6]))),'hand_math_balance_max_abs_Nm':float(np.max(np.abs(eq[6:]))),'hand_FP32_affine_balance_max_abs_Nm':float(np.max(np.abs(balance_affine))),'hand_FP32_difference_balance_max_abs_Nm':float(np.max(np.abs(balance_diff))),'roundoff_bound_min_max_Nm':[float(bound.min()),float(bound.max())]}
    checks['zero_stiction_selected_even_when_bounded_lower_objective']=design['selected_LP']=='zero_stiction' and design['selected_mathematical_witness']==design['all_LPs']['zero_stiction'] and np.array_equal(design['u0_rad'],design['all_LPs']['zero_stiction']['u32_candidate_rad'])
    source=Path(design['source_pins']['design_source']['path']);tree=ast.parse(source.read_text());policy=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    checks['policy_before_LP_file_and_source_literal_match']=policy==report['policy']==read(SOURCE/'policy_before_LP.json')
    selected=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='selected' for t in n.targets) and isinstance(n.value,ast.IfExp) and 'lp_results' in ast.unparse(n.value))
    details['selection_expression_AST']=ast.unparse(selected.value)
    checks['selection_uses_math_not_physics_quality']=details['selection_expression_AST']=="'zero_stiction' if lp_results['zero_stiction']['mathematical_feasible'] else 'bounded_stiction' if lp_results['bounded_stiction']['mathematical_feasible'] else None" and 'evaluation.json' not in source.read_text() and 'raw_states.npz' not in source.read_text()
    checks['no_forbidden_physics_or_force_calls']=not any(forbidden.values());checks['Torch_CPU_no_CUDA_initialized']=not torch.cuda.is_initialized() and 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules
    checks['frozen_worker_supervisor_unchanged']=sha(ROOT/'scripts/rest_hold_warp.py')=='03770389af4f47a9659da015557df04707d574b6c76827d36ded2fa45c47b6a0' and sha(ROOT/'scripts/run_rest_hold.py')=='2f29bcf1cbe0b96282fb825c18c4ac257de17ddf2e760bce8d200577ddad681f'
    np.savez_compressed(out/'independently_rebuilt_static_arrays.npz',A=A,G=G,tau_g=tau_g,body_Jcom=bodyJ,body_gravity=bodyG,FD_G=FDG,FD_gravity=FDgrav,Aeq=Aeq,beq=beq)
    result={'status':'CANONICAL_STATIC_SURROGATE_AUDIT_PASS' if all(checks.values()) else 'STATIC_SURROGATE_AUDIT_FAIL','checks':{k:bool(v) for k,v in checks.items()},'errors':errors,'LP_recomputations':lp_checks,'details':details,'source_final_pins':final,'source_final_pins_sha256':sha(SOURCE/'FINAL_PINS.json'),'design_sha256':sha(SOURCE/'baseline/designed_target_u0.json'),'reviewer_sha256':sha(__file__),'rebuilt_arrays_sha256':sha(out/'independently_rebuilt_static_arrays.npz'),'CPU_collision_list_queries':1,'physics_steps':0,'GPU_used':False,'forbidden_calls':forbidden,'new_target_created':False,'physical_reference_qualified':False,'limits':['Existing fixedcontact static surrogate only; one CPU geometry query and kinematic FD checks, no physical stepping or solver force evaluation.','Zero-stiction/bounded-stiction are independently checked mathematical allocations, not predicted actual friction.','Potential thumbjoint4 limit activation and omitted limit reaction are explicit; q32 is on FP32 lowerbound and differs from originaldouble limit by5.05e-8rad.','Softcontact depth/solref/solimp force law, actualWarp force/FMA, contact persistence and dynamic stability are not validated.','No hypothesis/LP choice was selected using physical outcomes; the planned comparison is still sequential exploratory engineering, not randomized/statistical replication.']}
    write(out/'report.json',result);print(json.dumps({'status':result['status'],'checks_passed':int(sum(checks.values())),'checks_total':len(checks),'failed':[k for k,v in checks.items() if not v],'errors':errors,'LPs':lp_checks,'design_sha256':result['design_sha256']},indent=2));raise SystemExit(not all(checks.values()))


if __name__=='__main__':main()
