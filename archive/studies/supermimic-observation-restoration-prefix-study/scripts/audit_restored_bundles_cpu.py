"""Independent CPU bundle/controller audit; no exporter or geometry imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,math
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from scipy.linalg import expm,expm_frechet
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
RECEIPT_SHA='5bcd45d8e15e13579fb8d4debe9efcbd1e765bd3d24725acbf6ee38bf09f3afc'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':Path(p).resolve().as_posix(),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def bitwise(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def hat(x):return np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])
def vee(x):return np.array([x[2,1]-x[1,2],x[0,2]-x[2,0],x[1,0]-x[0,1]])/2


def exp_with_derivatives(phi,v,a):
    A,V=hat(phi),hat(v);E=expm(A);Ed=expm_frechet(A,V,compute_expm=False)
    block=np.zeros((9,9))
    for i in range(3):block[i*3:(i+1)*3,i*3:(i+1)*3]=A
    block[:3,3:6]=V;block[3:6,6:9]=V
    Edd=expm_frechet(A,hat(a),compute_expm=False)+2*expm(block)[:3,6:9]
    return E,Ed,Edd


def hermite_coefficients(y0,y1,v0,v1,a0,a1,h):
    # Independent endpoint linear solve; does not import frozen curve helper.
    c=np.zeros((6,3));c[0]=y0;c[1]=h*v0;c[2]=.5*h*h*a0
    rhs=np.stack((y1-c[:3].sum(0),h*v1-c[1]-2*c[2],h*h*a1-2*c[2]))
    c[3:]=np.linalg.solve(np.array([[1.,1,1],[3,4,5],[6,12,20]]),rhs);return c


def polynomial(c,s,h,order):
    v=np.zeros(3)
    for k in range(order,6):v+=c[k]*math.factorial(k)/math.factorial(k-order)*s**(k-order)
    return v/h**order


def object_curve_independent(raw,times):
    pose=raw['object_pose_mapped_w'][:16];rot=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix();clock=raw['time_s'][:16]
    pc=[];rc=[]
    for i,h in enumerate(np.diff(clock)):
        pc.append(hermite_coefficients(pose[i,:3],pose[i+1,:3],raw['raw_v_world_m_s'][i],raw['raw_v_world_m_s'][i+1],raw['raw_a_world_m_s2'][i],raw['raw_a_world_m_s2'][i+1],h))
        phi=Rotation.from_matrix(rot[i].T@rot[i+1]).as_rotvec()
        if np.linalg.norm(phi)>=np.pi-1e-5:raise ValueError('Unqualified nearpi object branch')
        M=np.column_stack([vee((rot[i]@expm_frechet(hat(phi),hat(np.eye(3)[j]),compute_expm=False))@rot[i+1].T) for j in range(3)])
        v1=np.linalg.solve(M,raw['raw_omega_world_rad_s'][i+1]);E,Ed,Edd=exp_with_derivatives(phi,v1,np.zeros(3));Rd,Rdd=rot[i]@Ed,rot[i]@Edd
        bias=vee(Rdd@rot[i+1].T+Rd@Rd.T);a1=np.linalg.solve(M,raw['raw_alpha_world_rad_s2'][i+1]-bias)
        rc.append(hermite_coefficients(np.zeros(3),phi,rot[i].T@raw['raw_omega_world_rad_s'][i],v1,rot[i].T@raw['raw_alpha_world_rad_s2'][i],a1,h))
    out={k:[] for k in ('object_COM_world_m','object_v_world_m_s','object_a_world_m_s2','object_rotation_w','object_omega_world_rad_s','object_alpha_world_rad_s2')}
    for t in times:
        i=min(np.searchsorted(clock,t,side='right')-1,14);h=clock[i+1]-clock[i];s=(t-clock[i])/h
        for order,key in enumerate(('object_COM_world_m','object_v_world_m_s','object_a_world_m_s2')):out[key].append(polynomial(pc[i],s,h,order))
        E,Ed,Edd=exp_with_derivatives(*(polynomial(rc[i],s,h,k) for k in range(3)));R,Rd,Rdd=rot[i]@E,rot[i]@Ed,rot[i]@Edd
        out['object_rotation_w'].append(R);out['object_omega_world_rad_s'].append(vee(Rd@R.T));out['object_alpha_world_rad_s2'].append(vee(Rdd@R.T+Rd@Rd.T))
    return {k:np.asarray(v) for k,v in out.items()}


def audit(inp,out):
    if sha(inp/'receipt.json')!=RECEIPT_SHA:raise ValueError('Only frozen5bcd export receipt')
    out.mkdir(parents=True,exist_ok=False)
    receipt=read(inp/'receipt.json');man={k:read(inp/(k+'_manifest.json')) for k in ('reference','init','control')}
    ref,init,control=(load(inp/name) for name in ('reference_bspline.npz','init_requested.npz','planned_control.npz'))
    pins=receipt['source_pins'];pipeline=load(pins['pipeline_return']['path']);candidate=load(pins['candidate']['path']);manifest=read(pins['manifest']['path']);proto=read(pins['protocol']['path']);parent=read(pins['input_parent_protocol']['path'])
    q0=load(proto['inputs']['canonical_H123_q0']['path'])['joint_pos_rad'][0];raw=load(pins['rawFD_design_arrays']['path']);design=read(pins['static_preload_design']['path'])
    checks={};errors={};hash_cache={}
    def hashok(p):
        key=str(Path(p['path']).resolve()).lower()
        if key not in hash_cache:hash_cache[key]=sha(p['path'])
        return hash_cache[key]==p['sha256']
    checks['all_receipt_source_pins_hash_bound']=all(hashok(v) for v in pins.values())
    shared_keys=('schema','retarget_variant_id','protocol','constrained_protocol','input_parent_protocol','candidate','source_plan','source_identity','source_pins','physical_reference_qualified','init_measured','u_readback','GPU_authorized','training_authorized')
    checks['three_manifests_same_full_identity_and_sources']=all(all(man[k][key]==man['reference'][key] for key in shared_keys) and man[k]['bundle_kind']==k and hashok(man[k]['artifact']) for k in man)
    checks['bundle_schema_variant_and_ancestors']=man['reference']['schema']=='restored_prefix_separate_bundles:1' and man['reference']['retarget_variant_id']==proto['retarget_variant_id']=='H123_cubic_minimax_restoration_filtered_prefix16_v1' and man['reference']['protocol']==manifest['protocol'] and man['reference']['constrained_protocol']==proto['parent_protocol'] and man['reference']['input_parent_protocol']==proto['input_parent_protocol'] and proto['inputs']==parent['inputs']
    checks['source_window_and_newtrajectory_distinct']=man['reference']['source_identity']==proto['source_identity'] and proto['source_identity']['primary_pose_source_frame_range_half_open']==[200,216] and man['reference']['candidate']==pins['candidate'] and man['reference']['source_plan']==pins['source_plan']
    checks['normal_pipeline_canonical_not_QP']=bool(pipeline['normal_termination']) and np.array_equal(candidate['optimizer_x'],pipeline['final_x']) and np.array_equal(pipeline['final_x'],pipeline['last_accepted_x']) and np.array_equal(pipeline['final_x'],pipeline['R_terminal_x']) and receipt['pipeline_canonical_record_only']==manifest['canonical_candidate'] and manifest['canonical_candidate']['selected_phase']=='R_start_F_NO_ACCEPTED_STEP'
    checks['physical_actualreadback_and_GPU_still_unknown']=all(m['physical_reference_qualified'] is False and m['init_measured'] is None and m['u_readback'] is None and m['GPU_authorized'] is False for m in man.values())
    checks['canonical_three_fields_source_exact']=np.array_equal(ref['q_goal_rad'],candidate['q_goal_rad']) and np.array_equal(ref['mapped_object_pose_w'],candidate['mapped_object_pose_w']) and np.array_equal(ref['q_goal_rad'][0],q0) and np.array_equal(ref['time_s'],np.arange(16)*.02) and np.array_equal(ref['mapped_object_pose_w'],raw['object_pose_mapped_w'][:16])
    knots=np.r_[np.zeros(4),np.arange(1,15)*.02,np.repeat(.3,4)];spline=BSpline(knots,np.eye(18),3,extrapolate=False);P=ref['control_points_rad'];delta=P[1:]-q0;t16=ref['time_s'];t61=np.arange(61)*.005
    def sample(t,order=0):
        x=spline(t,nu=order)[:,1:]@delta
        return q0+x if order==0 else x
    q16,qd16,qdd16=(sample(t16,k) for k in range(3));q61,qd61,qdd61=(sample(t61,k) for k in range(3))
    checks['basis_knots_coefficient_source_exact']=np.array_equal(ref['Bspline_knots_s'],knots) and np.array_equal(P,candidate['control_points_rad']) and np.array_equal(P[0],q0)
    checks['independent_Bspline16_and61_exact']=all(np.array_equal(a,ref[k]) for a,k in [(q16,'q_goal_rad'),(qd16,'analytic_qd16_rad_s'),(qdd16,'analytic_qdd16_rad_s2'),(q61,'q61_rad'),(qd61,'qd61_rad_s'),(qdd61,'qdd61_rad_s2')])
    fd=np.diff(q16,axis=0)/.02;fd=np.r_[fd,fd[-1:]];f32=np.float32;fd32=np.diff(q16.astype(f32),axis=0)/f32(.02);fd32=np.r_[fd32,fd32[-1:]]
    checks['native_FD64_32_separate_exact']=np.array_equal(fd,ref['native_FD64_rad_s']) and bitwise(fd32,ref['native_FD32_rad_s'])
    # Only load a static model. All native query/force/step functions are denied.
    import mujoco
    import scipy.optimize as optimize
    forbidden={k:0 for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce','mj_collision','mj_kinematics')};orig={k:getattr(mujoco,k) for k in forbidden}
    solvers={k:0 for k in ('linprog','minimize','least_squares')};optorig={k:getattr(optimize,k) for k in solvers}
    def deny(c,k):
        def fail(*a,**kw):c[k]+=1;raise AssertionError('Independent pure audit forbids '+k)
        return fail
    for k in orig:setattr(mujoco,k,deny(forbidden,k))
    for k in optorig:setattr(optimize,k,deny(solvers,k))
    try:
        model=mujoco.MjModel.from_binary_path(pins['model']['path']);ids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)]);aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in ids]);body=model.body('object/cube').id
        context=load(pins['actual_context']['path']);gain=context['actuator_gainprm'];biasfull=context['actuator_biasprm'];kp=gain[0,aids,0].astype(f32);bias=biasfull[0,aids,:3].astype(f32)
        checks['actual_affine_PD_type_and_unique20mapping']=np.array_equal(ids,np.asarray(receipt['PD_model']['joint_ids'])) and np.array_equal(aids,np.asarray(receipt['PD_model']['actuator_ids'])) and np.all(model.actuator_gaintype[aids]==mujoco.mjtGain.mjGAIN_FIXED) and np.all(model.actuator_biastype[aids]==mujoco.mjtBias.mjBIAS_AFFINE) and np.all(model.actuator_dyntype[aids]==mujoco.mjtDyn.mjDYN_NONE) and np.array_equal(model.actuator_gear[aids],np.tile([1.,0,0,0,0,0],(20,1))) and np.array_equal(bias[:,1],-kp)
        checks['actual_PD_context_shared_worlds']=np.array_equal(gain,np.broadcast_to(gain[:1],gain.shape)) and np.array_equal(biasfull,np.broadcast_to(biasfull[:1],biasfull.shape))
        lower64=model.jnt_range[ids,0];upper64=model.jnt_range[ids,1];outerlo=np.maximum(lower64,model.actuator_ctrlrange[aids,0]).astype(f32);outerhi=np.minimum(upper64,model.actuator_ctrlrange[aids,1]).astype(f32);innerlo=outerlo+f32(.020);innerhi=outerhi-f32(.020)
        act=model.actuator_forcerange[aids].astype(f32);joint=model.jnt_actfrcrange[ids].astype(f32);flo=np.maximum(act[:,0],joint[:,0]);fhi=np.minimum(act[:,1],joint[:,1])
        checks['control_hull_original_joint_range']=bool(np.max(np.maximum(np.maximum(lower64-P,P-upper64),0))<=1e-12)
        uP=np.asarray(design['u0_rad'],dtype=f32);tau=kp*uP+(bias[:,0]+bias[:,1]*q0.astype(f32));pre=(((tau-bias[:,0])-bias[:,1]*q16.astype(f32))-bias[:,2]*qd16.astype(f32))/kp;ex=np.minimum(np.maximum(pre,innerlo),innerhi);track=ex[:15].copy();hold=np.repeat(track[:1],15,axis=0);rates=np.diff(track.astype(float),axis=0)/.02
        prefix='control__fp32_old_export__';get=lambda key:control[prefix+key]
        checks['independent_FP32_static_prior_proposal_C2_bitwise']=bitwise(tau,get('tau_ff_static0_FP32_Nm')) and bitwise(pre,get('u_pre_16_FP32_rad')) and bitwise(ex,get('u_exec_16_diagnostic_FP32_rad')) and bitwise(track,get('tracking_u_exec_15_FP32_rad')) and bitwise(hold,get('hold_u_exec_15_FP32_rad'))
        checks['actual15_proposal_C2_identity_zero_clamp']=bitwise(pre[:15],track)
        checks['final15_C2_domain_strict']=bool(np.all(track>=innerlo) and np.all(track<=innerhi))
        checks['final14_actualplanned_rate_exact_le15']=np.array_equal(rates,get('actual_target_rate_14_rad_s')) and bool(np.max(np.abs(rates))<=15.)
        checks['same_new_u0_and_nonconstant_tracking']=bitwise(track[0],hold[0]) and bitwise(init['u_planned0_FP32_rad'],track[0]) and bool(np.any(track[1:]!=track[0]))
        checks['exact60_solver_prestate_command_indices']=np.array_equal(get('solver_prestate_control_interval'),np.arange(60)//4)
        ql,qr=sample(t16[:-1]),sample(t16[1:]);vl,vr=sample(t16[:-1],1),sample(t16[1:],1);h=np.diff(t16)[:,None];qbez=np.stack((ql,ql+h*vl/3,qr-h*vr/3,qr),axis=1);v2=3*np.diff(qbez,axis=1)/h[:,None,:];vbez=np.stack((v2[:,0],v2[:,0]/3+2*v2[:,1]/3,2*v2[:,1]/3+v2[:,2]/3,v2[:,2]),axis=1)
        caps={};force_errors={};all_sample_gates=[]
        def force(q,v,u):
            q,v,u=(np.asarray(z,dtype=f32) for z in (q,v,u));return kp*u+((bias[:,0]+bias[:,1]*q)+bias[:,2]*v),(kp*(u-q)+bias[:,0])+bias[:,2]*v
        for name,commands in [('tracking',track),('hold',hold)]:
            coef=kp.astype(float)*commands.astype(float)[:,None,:]+bias[:,0].astype(float)+bias[:,1].astype(float)*qbez+bias[:,2].astype(float)*vbez
            savedcoef=control['control__postC2_real_affine_Bernstein_coefficients__'+name];err=float(np.max(np.abs(coef-savedcoef)));force_errors[name]=err;caps[name]={'raw_coefficient_caps_pass':bool(np.all(coef>=flo) and np.all(coef<=fhi)),'max_raw_coefficient_cap_excess_Nm':float(np.maximum(np.maximum(flo-coef,coef-fhi),0).max())}
            checks[name+'_postC2_real_Bernstein_recomputed']=err<=1e-12 and caps[name]['raw_coefficient_caps_pass']
            a,d=force(q61,qd61,commands[np.minimum(np.arange(61)//4,14)])
            for grouping,values in [('affine',a),('difference',d)]:
                saved=control[f'control__FP32_reference61_samples__{name}_rawPD_{grouping}61_FP32_Nm'];checks[f'{name}_{grouping}61_FP32_saved_bitwise_and_caps']=bitwise(values,saved) and bool(np.all(values>=act[:,0]) and np.all(values<=act[:,1]) and np.all(values>=joint[:,0]) and np.all(values<=joint[:,1]))
                if name=='tracking':key='reference_PD_'+grouping+'_60_FP32_Nm'
                else:key='hold_reference_PD_'+grouping+'_60_FP32_Nm'
                checks[f'{name}_{grouping}60_FP32_phase_alignment']=bitwise(values[:60],get(key))
        errors['real_Bernstein_independent_grouping_max_Nm']=force_errors
        import torch
        tk,tb=(torch.tensor(z) for z in (kp,bias));tt=tk*torch.tensor(uP)+(tb[:,0]+tb[:,1]*torch.tensor(q0.astype(f32)));tp=(((tt-tb[:,0])-tb[:,1]*torch.tensor(q16.astype(f32)))-tb[:,2]*torch.tensor(qd16.astype(f32)))/tk;te=torch.minimum(torch.maximum(tp,torch.tensor(innerlo)),torch.tensor(innerhi))
        spec=importlib.util.spec_from_file_location('independent_bundle_original_C2',pins['pure_C2']['path']);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure);oracle=pure.target_inset(tp,torch.tensor(outerlo),torch.tensor(outerhi),.020)
        checks['independent_Torch_and_original_C2_FP32_parity']=bitwise(tt.numpy(),tau) and bitwise(tp.numpy(),pre) and bitwise(te.numpy(),ex) and bitwise(oracle.numpy(),ex)
        checks['Torch_CPU_only_CUDA_not_initialized']=not torch.cuda.is_initialized()
        obj=object_curve_independent(raw,t61)
        object_errors={k:float(np.max(np.abs(v-ref[k]))) for k,v in obj.items()};limits={'object_COM_world_m':1e-12,'object_v_world_m_s':1e-8,'object_a_world_m_s2':1e-6,'object_rotation_w':1e-12,'object_omega_world_rad_s':1e-8,'object_alpha_world_rad_s2':1e-6}
        checks['independent_quintic_SO3_Frechet61_matches']=all(object_errors[k]<=limits[k] for k in object_errors)
        errors['independent_object_curve_max_abs']=object_errors
        pose32=ref['mapped_object_pose_w'][0].astype(f32);R32=Rotation.from_quat(pose32[[4,5,6,3]].astype(float)).as_matrix();v0=raw['raw_v_world_m_s'][0];omega0=raw['raw_omega_world_rad_s'][0];vbodyorigin=(v0-np.cross(omega0,R32@model.body_ipos[body])).astype(f32);wbody=(R32.T@omega0).astype(f32)
        checks['init_q_qd_pose_castonce_FP32_exact']=bitwise(init['q_init_requested_FP32_rad'],q0.astype(f32)) and bitwise(init['qd_init_requested_FP32_rad_s'],qd61[0].astype(f32)) and bitwise(init['object_pose_init_requested_FP32_w'],pose32)
        checks['init_world_COM_to_bodyorigin_and_freeangular_FP32']=bitwise(init['object_bodyorigin_v_requested_FP32_world_m_s'],vbodyorigin) and bitwise(init['object_angular_requested_FP32_body_rad_s'],wbody) and bitwise(init['object_angular_requested_FP32_world_rad_s'],omega0.astype(f32))
        checks['init_nonzero_design_not_zero_rest']=bool(np.any(init['qd_init_requested_FP32_rad_s']!=0) and np.any(init['object_angular_requested_FP32_body_rad_s']!=0) and np.any(init['object_bodyorigin_v_requested_FP32_world_m_s']!=0))
        errors['FP32_angular_roundtrip_world_max_rad_s']=float(np.max(np.abs(R32@wbody.astype(float)-omega0)))
        lin=load(proto['linear_contract']['path']);A=sparse.csr_matrix((lin['A_CSR_data'],lin['A_CSR_indices'],lin['A_CSR_indptr']),shape=lin['A_shape']);linear_res=A@(candidate['optimizer_x']-lin['x0'])-lin['b_centered']
        checks['raw_exactlinear_false_and_predeclared_1e12_retained']=not np.all(linear_res<=0) and bool(np.all(linear_res<=1e-12)) and receipt['raw_exact_linear_planning_pass'] is False and receipt['arithmetic_tolerances']['linear_each_native_unit']==1e-12 and abs(float(linear_res.max())-receipt['max_linear_Ax_minus_b'])<1e-20
        checks['saved_geometry61_plan_and_qualification_strict']=np.array_equal(ref['saved_generated_depth61_m'],candidate['operational_generated_maxdepth_m']) and bool(np.all(ref['saved_generated_depth61_m']<=.002999)) and bool(np.all(ref['saved_generated_depth61_m']<=.003))
        checks['no_native_query_or_optimizer_calls']=not any(forbidden.values()) and not any(solvers.values())
        independent18={
            'terminal_and_source_P0_identity':checks['normal_pipeline_canonical_not_QP'] and checks['canonical_three_fields_source_exact'],
            'candidate_q16_q61_objectcurve_match':checks['independent_Bspline16_and61_exact'] and checks['independent_quintic_SO3_Frechet61_matches'],
            'controlpoint_convexhull':checks['control_hull_original_joint_range'],
            'linear_planning_declared_arithmetic_1e12':bool(np.all(linear_res<=1e-12)),
            'pipeline_final_original_primal_declared_arithmetic':bool(np.all(candidate['constraint_scaled_mm']<=2.999+1e-9) and np.all(linear_res<=1e-12) and np.all(P>=lower64-1e-12) and np.all(P<=upper64+1e-12)),
            'NumPy_Torch_originalC2_FP32_parity':checks['independent_Torch_and_original_C2_FP32_parity'],
            'final15_C2_domain':checks['final15_C2_domain_strict'],'final15_proposal_C2_identity':checks['actual15_proposal_C2_identity_zero_clamp'],
            'planned14_target_rate_le15':checks['final14_actualplanned_rate_exact_le15'],
            'postC2_tracking_real_affine_Bernstein':checks['tracking_postC2_real_Bernstein_recomputed'],'postC2_hold_real_affine_Bernstein':checks['hold_postC2_real_Bernstein_recomputed'],
            'tracking61_FP32_reference_caps':checks['tracking_affine61_FP32_saved_bitwise_and_caps'] and checks['tracking_difference61_FP32_saved_bitwise_and_caps'],
            'hold61_FP32_reference_caps':checks['hold_affine61_FP32_saved_bitwise_and_caps'] and checks['hold_difference61_FP32_saved_bitwise_and_caps'],
            'same_new_u0':checks['same_new_u0_and_nonconstant_tracking'],'nonconstant_tracking_planned_targets':checks['same_new_u0_and_nonconstant_tracking'],
            'saved_generated_depth61_planning_le2_999mm':bool(np.all(ref['saved_generated_depth61_m']<=.002999)),'saved_generated_depth61_qualification_le3mm':bool(np.all(ref['saved_generated_depth61_m']<=.003)),
            'source_input_hashes_unchanged':checks['all_receipt_source_pins_hash_bound']}
        checks['all18_receipt_gates_independently_reproduced']=set(independent18)==set(receipt['gates']) and all(independent18[k]==receipt['gates'][k] for k in independent18)
        np.savez_compressed(out/'independent_math.npz',tau32=tau,proposal16_32=pre,tracking15_32=track,hold15_32=hold,rate14=rates,q16=q16,qd16=qd16,q61=q61,qd61=qd61,**{'object_independent__'+k:v for k,v in obj.items()})
        summary={'status':'PASS_INDEPENDENT_BUNDLE_CPU_MATH_PHYSICS_PENDING' if all(checks.values()) else 'FAIL_INDEPENDENT_BUNDLE_AUDIT','checks':{k:bool(v) for k,v in checks.items()},'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'independent18_gates':{k:bool(v) for k,v in independent18.items()},'inputs':{'receipt':pin(inp/'receipt.json'),**{k+'_manifest':pin(inp/(k+'_manifest.json')) for k in man},**{k:man[k]['artifact'] for k in man}},'analyzer':pin(__file__),'errors':errors,
                 'values':{'planned_rate_max_rad_s':float(np.abs(rates).max()),'C2_clamped_components_actual15':int(np.count_nonzero(pre[:15]!=track)),'static_prior_is_not_dynamic_inverse_solution':True,'actual_body_ipos_m':model.body_ipos[body].tolist(),'linear_raw_max_native':float(linear_res.max()),'raw_exactlinear_pass':bool(np.all(linear_res<=0)),'linear_1e12_arithmetic_pass':bool(np.all(linear_res<=1e-12)),'q_analytic_initial_rad_s':qd61[0].tolist(),'object_world_v_initial_m_s':v0.tolist(),'object_world_omega_initial_rad_s':omega0.tolist(),'object_free_angular_FP32_body_rad_s':wbody.tolist(),'postC2_real_Bernstein_certificates':caps},
                 'scope':'Independent direct SciPy cubic basis, explicitFP32 PD math, pinned originalC2 plus TorchCPU parity, direct cubicBezier force coefficients, independent SO3/quintic reconstruction using matrix-exponential Frechet derivatives. No exporter/basis-control/curve-helper import. MjModel read only; no MjData or native query. No Warp/FMA/WM/physical guarantee.',
                 'forbidden_native_calls':forbidden,'optimizer_calls':solvers,'geometry_queries':0,'physics_steps':0,'GPU_used':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'init_measured':None,'u_applied_measured':None,'physical_reference_qualified':False,'full_motion_qualified':False}
        write(out/'report.json',summary);print(json.dumps({'status':summary['status'],'passed':summary['passed'],'total':summary['total'],'failed':[k for k,v in checks.items() if not v],'errors':errors,'values':summary['values']},indent=2));return 0 if all(checks.values()) else 1
    finally:
        for k,v in orig.items():setattr(mujoco,k,v)
        for k,v in optorig.items():setattr(optimize,k,v)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--input',type=Path,default=ROOT/'outputs/restored_prefix_control_export_v1');a.add_argument('--output',type=Path,default=ROOT/'outputs/restored_bundles_independent_review_v1');args=a.parse_args();raise SystemExit(audit(args.input,args.output))
