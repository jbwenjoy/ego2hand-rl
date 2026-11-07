"""Saved failed full151 LP candidate audit; no production imports or solve."""
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

ROOT=Path('D:/workspace/supermimic-full151-observation-study');RUN=ROOT/'outputs/full151_cpu_input_v1';OUT=Path(__file__).parent
J=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in (a.files if keys is None else keys)}
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def maxabs(a):return float(np.max(np.abs(a)))
def pins(x):
    if isinstance(x,dict):
        if set(x)=={'path','sha256'}:yield x
        else:
            for v in x.values():yield from pins(v)
    elif isinstance(x,list):
        for v in x:yield from pins(v)
checks={};details={};exceptions={}
def C(k,v):
    if k in checks:raise ValueError('Duplicate key '+k)
    checks[k]=bool(v)
def section(name):
    def deco(fn):
        try:fn()
        except Exception:checks[name+'_exception_free']=False;exceptions[name]=traceback.format_exc()
    return deco
receipt=J(RUN/'receipt.json');protocol=J(receipt['protocol']['path'])
ref=load(RUN/'reference_bspline151.npz');init=load(RUN/'init_requested.npz');ctrl=load(RUN/'planned_control150.npz');pd=load(RUN/'actual_PD_parameters.npz');fit=load(RUN/'fit_solutions.npz')
P=ref['control_points_rad'];kp=pd['kp'];bias=pd['bias'];tau=ctrl['tau_ff_static0_FP32_Nm'];U=ctrl['u_postC2_150_FP32_rad'];h=.02
B=BSpline(ref['Bspline_knots_s'],np.eye(153),3,extrapolate=False)
times=np.arange(151)*h;dense_t=np.arange(601)*.005
joint_rows=[]

@section('identity')
def identity():
    pp=list(pins(protocol))
    C('protocol_current_exact_3f43',sha(receipt['protocol']['path'])==receipt['protocol']['sha256']=='3f436240b04d6e428e396dd7a0fbf297469d9e5d59b43519f08e8eb988ec6a37')
    C('23_protocol_pins_current_exact',len(pp)==23 and all(sha(p['path'])==p['sha256'] for p in pp))
    C('all_recorded_artifact_pins_exact',all(sha(p['path'])==p['sha256'] for p in receipt['artifacts']))
    C('source_before_copies_exact',all(sha(RUN/'source_before'/(name+Path(p['path']).suffix))==p['sha256'] for name,p in protocol['sources'].items()))
    C('original_FAILED_and_READY_absent_preserved',receipt['status']=='CPU_CANDIDATE_GATES_FAIL_NO_QUALIFICATION' and not (RUN/'READY.json').exists())
    C('only_original_raw_arithmetic_gate_failed',len(receipt['checks'])==34 and [k for k,v in receipt['checks'].items() if not v]==['all20_raw_arithmetic_1e12'])
    C('actual20LP_no_physics_WM_training_GPU_record',receipt['cost']['LP_calls']==20 and all(receipt['cost'][k]==0 for k in ['physics_steps','geometry_queries','WM_forwards','training_updates']) and receipt['cost']['GPU_used'] is False and receipt['runtime']['CUDA_VISIBLE_DEVICES']=='-1')
    C('summed_solver_cost_matches_all20_records',sum(r['wall_s'] for r in receipt['joint_results'])==receipt['cost']['solver_wall_s_sum'] and receipt['wall_s']>=receipt['cost']['solver_wall_s_sum'])
    details['cost_scopes']={'CPU_total_wall_s':receipt['wall_s'],'nested_solver_wall_sum_s':receipt['cost']['solver_wall_s_sum'],'not_additive':True}

@section('LP_raw')
def LP_raw():
    allraw=True;allbounds=True;allobject=True;matrixlinks=True;allshape=True
    coeff_bad=[];primal_bad=[];functional_bad=[];objective_maxerr=0.;residual_maxerr=0.
    for j in range(20):
        base=RUN/'joint_LPs'/f'joint{j:02d}'
        a=load(str(base)+'_input.npz');r=load(str(base)+'_result.npz');c=J(str(base)+'_contract.json');saved=J(str(base)+'_receipt.json')
        A=sparse.csr_matrix((a['A_data'],a['A_indices'],a['A_indptr']),shape=tuple(a['A_shape']));x=r['x'];res=A@x-a['b']
        bounds=np.maximum(np.maximum(a['bound_lower']-x,x-a['bound_upper']),0)
        maxraw=float(res.max());maxbound=float(bounds.max());valid=maxraw<=1e-12 and maxbound<=1e-12
        obj=float(a['c']@x);objerr=abs(obj-saved['objective']);objective_maxerr=max(objective_maxerr,objerr)
        res_err=maxabs(res-r['raw_Ax_minus_b']);residual_maxerr=max(residual_maxerr,res_err)
        groups={g['name']:float(res[g['start']:g['stop']].max()) for g in c['row_groups']}
        allraw &= exact(res,r['raw_Ax_minus_b']) and maxraw==saved['maximum_raw_Ax_minus_b'] and groups==saved['row_group_max_raw_Ax_minus_b'] and valid==saved['arithmetic_1e12_gate']
        allbounds &= maxbound==saved['max_bound_excess']==0.
        allobject &= objerr<=1e-12 and np.all(x[151:]>=-1e-12)
        allshape &= A.shape==(2402,453) and x.shape==(453,) and c['joint_index']==j and c['nvariables']==453
        matrixlinks &= exact(x[:151],fit['free_control_points_rad'][:,j]) and exact(x[:151],P[2:,j])
        q=a['q_matrix']@x[:151]+a['q_base'];qd=a['qd_matrix']@x[:151]+a['qd_base'];u=a['command_matrix']@x[:151]+a['command_base']
        f=np.einsum('irc,c->ir',a['force_Bernstein_matrix'],x[:151])+a['force_Bernstein_base'];second=a['second_difference_matrix']@x[:151]+a['second_difference_base']
        l1=float(np.mean(np.abs(q-a['target151']))+4*np.mean(np.abs(second)))
        functional_groups=['target_upper','target_lower','target_rate_upper','target_rate_lower','tracking_rawforce_upper','tracking_rawforce_lower']
        functional=max(groups[k] for k in functional_groups)
        if not valid:primal_bad.append(j)
        if functional>1e-12:functional_bad.append(j)
        violated=[]
        for g in c['row_groups']:
            rows=np.flatnonzero(res[g['start']:g['stop']]>1e-12)
            for ix in rows:
                violated.append({'group':g['name'],'group_index':int(ix),'row_index':g['start']+int(ix),'raw_Ax_minus_b':float(res[g['start']+ix]),'units':g['units']})
        joint_rows.append({'joint_index':j,'solver_success':saved['success'],'solver_status':saved['status'],'raw_arithmetic_gate':bool(valid),'max_raw_Ax_minus_b':maxraw,'max_bound_excess':maxbound,
                           'objective_saved':saved['objective'],'objective_dot_cx':obj,'objective_cx_error':objerr,'exact_abs_fit_plus_regularizer':l1,'epigraph_objective_minus_exact_abs':obj-l1,
                           'row_group_max':groups,'violated_rows':violated,'functional_planning_row_max':functional,
                           'LP_nominal_real_u_vs_final_U32_maxabs':maxabs(u-U[:,j]),'LP_nominal_reference_force_bounds_max_excess':float(np.maximum(np.maximum(a['planning_force_bounds'][0]-f,f-a['planning_force_bounds'][1]),0).max())})
    C('20_saved453x2402_finite_dimensions',allshape and all(np.isfinite(load(str(RUN/'joint_LPs'/f'joint{j:02d}')+'_result.npz')['x']).all() for j in range(20)))
    C('all20_raw_Axb_exact_recomputed_including_failures',allraw)
    C('all20_bounds_zero_excess_recomputed',allbounds)
    C('all20_c_dot_x_objective_matches',allobject)
    C('all20_terminal_free_coefficients_match_saved_export',matrixlinks)
    C('independent_candidate_FAIL_same5joints',primal_bad==[4,12,13,14,15])
    C('only_joint4_has_functional_tightened_row_failure',functional_bad==[4])
    C('all20_HiGHS_success_distinct_from_qualification',all(r['success'] is True and r['status']==0 for r in receipt['joint_results']))
    details['LP']={'joint_results':joint_rows,'failed_arithmetic_joint_indices':primal_bad,'failed_functional_planning_joint_indices':functional_bad,'max_objective_dot_error':objective_maxerr,'max_residual_recomputation_error':residual_maxerr,
                   'max_raw_Ax_minus_b':max(r['max_raw_Ax_minus_b'] for r in joint_rows),'original_qualification_unchanged':'FAIL; five joints exceed the preregistered1e-12 native-row tolerance. Most are auxiliary fit/regularizer epigraphs; joint4 also exceeds a tightened target row. Original post-C2 bounds passing does not repair the failed planning qualification.'}

@section('curves_controls')
def curves_controls():
    # Direct basis evaluation is independent of the producer's source-centered
    # evaluation grouping; retain its floating difference instead of bit claiming.
    q=B(times)@P;qd=B(times,nu=1)@P;qdd=B(times,nu=2)@P
    q601=B(dense_t)@P;v601=B(dense_t,nu=1)@P;acc601=B(dense_t,nu=2)@P
    err={name:maxabs(actual-ref[key]) for name,actual,key in [('q151',q,'q_goal_rad'),('qd151',qd,'analytic_qd151_rad_s'),('qdd151',qdd,'analytic_qdd151_rad_s2'),('q601',q601,'planned_q601_rad'),('qd601',v601,'planned_qd601_rad_s'),('qdd601',acc601,'planned_qdd601_rad_s2')]}
    C('all_saved_bspline_samples_reconstruct_direct_basis',all(v<=1e-10 for v in err.values()))
    q32=ref['q_goal_rad'].astype(np.float32);v32=ref['analytic_qd151_rad_s'].astype(np.float32)
    proposed=(((tau-bias[:,0])-bias[:,1]*q32[:150])-bias[:,2]*v32[:150])/kp
    lo=pd['target_lower']+np.float32(.020);hi=pd['target_upper']-np.float32(.020)
    executed=np.minimum(np.maximum(proposed,lo),hi)
    rates=np.diff(executed.astype(float),axis=0)/.02
    C('FP32_original_grouped_proposal150_bitexact',exact(proposed,ctrl['u_pre150_FP32_rad']))
    C('FP32_C2_150_bitexact_and_identity',exact(executed,U) and exact(executed,proposed))
    C('actual_u0_regenerated_exact_not_replaced',exact(U[0],init['u_planned0_FP32_rad']))
    C('rate149_saved_exact_original_gate_pass',exact(rates,ctrl['planned_rate149_rad_s']) and maxabs(rates)<=15.)
    C('original_C2_domain_pass',np.all(U>=lo)&np.all(U<=hi))
    C('coefficient_hull_original_q_limits_pass',np.all(P>=pd['q_lower_rad']-1e-12)&np.all(P<=pd['q_upper_rad']+1e-12))
    left=times[:-1];right=times[1:];seg=np.diff(times)[:,None,None]
    qb=np.stack([B(left)@P,B(left)@P+seg[:,0,:]*(B(left,nu=1)@P)/3,B(right)@P-seg[:,0,:]*(B(right,nu=1)@P)/3,B(right)@P],axis=1)
    vb2=3/seg*np.diff(qb,axis=1)
    vb=np.stack([vb2[:,0],vb2[:,0]/3+2*vb2[:,1]/3,2*vb2[:,1]/3+vb2[:,2]/3,vb2[:,2]],axis=1)
    fb=kp*U.astype(float)[:,None,:]+bias[:,0]+bias[:,1]*qb+bias[:,2]*vb
    f_err=maxabs(fb-ctrl['postC2_real_affine_rawforce_Bernstein_Nm'])
    flo=np.maximum(pd['actuator_force_bounds'][:,0],pd['joint_force_bounds'][:,0]);fhi=np.minimum(pd['actuator_force_bounds'][:,1],pd['joint_force_bounds'][:,1])
    excess=np.maximum(np.maximum(flo-fb,fb-fhi),0)
    C('reference_force_Bernstein_reconstruction_roundoff',f_err<=1e-10)
    C('original_reference_force_caps_pass_independent',not np.any(excess>0))
    ix=np.minimum(np.arange(601)//4,149);uu=U[ix]
    qs=ref['planned_q601_rad'].astype(np.float32);vs=ref['planned_qd601_rad_s'].astype(np.float32)
    affine=kp*uu+((bias[:,0]+bias[:,1]*qs)+bias[:,2]*vs)
    difference=(kp*(uu-qs)+bias[:,0])+bias[:,2]*vs
    C('all601_affine_FP32_saved_exact',exact(affine,ctrl['reference_affine601_FP32_Nm']))
    C('all601_difference_FP32_saved_exact',exact(difference,ctrl['reference_difference601_FP32_Nm']))
    C('all601_both_original_reference_force_caps_pass',np.all(affine>=flo)&np.all(affine<=fhi)&np.all(difference>=flo)&np.all(difference<=fhi))
    C('600_solver_and601_terminal_command_indexing_exact',np.array_equal(ctrl['solver_prestate_interval600'],np.arange(600)//4) and np.array_equal(ctrl['reference_diagnostic_interval601'],ix))
    C('zero_startup_jump_exact',exact(ctrl['initial_ctrl_to_first_target_jump_rad'],np.zeros(20)) and exact(U[0],init['u_planned0_FP32_rad']))
    fd=np.diff(ref['q_goal_rad'],axis=0)/.02;fd32=np.diff(q32,axis=0)/np.float32(.02)
    C('native_FD64_andFD32_saved_distinct_recomputed',exact(ref['native_forward_FD64_rad_s'],np.r_[fd,fd[-1:]]) and exact(ref['native_forward_FD32_rad_s'],np.r_[fd32,fd32[-1:]]))
    details['postC2']={'direct_basis_sample_max_abs_errors':err,'Bernstein_force_recompute_max_abs_difference_Nm':f_err,'max_actual_target_rate_rad_s':maxabs(rates),'original_rate_gate_rad_s':15.,'planning_real_LP_rate_margin_gate_rad_s':14.9999,'postFP32_rate_minus_planning_value_rad_s':maxabs(rates)-14.9999,
                      'max_original_reference_force_excess_Nm':float(excess.max()),'minimum_original_force_bound_slack_Nm':float(np.minimum(fb-flo,fhi-fb).min()),'maximum_analytic_qd_rad_s':maxabs(ref['analytic_qd151_rad_s']),
                      'scope':'601sample/reference force only; no actual physical safety, global collision or wholeinterval FP32/FMA guarantee. Target rate15 is not a joint-velocity cap.'}

def Jr_terms(phi,v):
    angle=float(np.linalg.norm(phi));dot=float(phi@v)
    if angle<.01:
        a2=angle*angle
        A=.5-a2/24+a2*a2/720-a2**3/40320
        BB=1/6-a2/120+a2*a2/5040-a2**3/362880
        dA=2*dot*(-1/24+a2/360-a2*a2/13440)
        dB=2*dot*(-1/120+a2/2520-a2*a2/120960)
    else:
        A=(1-np.cos(angle))/angle**2;BB=(angle-np.sin(angle))/angle**3
        dtheta=dot/angle
        dA=(angle*np.sin(angle)-2*(1-np.cos(angle)))/angle**3*dtheta
        dB=(angle*(1-np.cos(angle))-3*(angle-np.sin(angle)))/angle**4*dtheta
    K=np.array([[0,-phi[2],phi[1]],[phi[2],0,-phi[0]],[-phi[1],phi[0],0.]])
    jr=np.eye(3)-A*K+BB*(K@K)
    derivative_v=-dA*np.cross(phi,v)+dB*np.cross(phi,np.cross(phi,v))+BB*np.cross(v,np.cross(phi,v))
    return jr,derivative_v

@section('init_object_law_fit')
def init_object_law_fit():
    inputs=protocol['inputs'];anchor=load(inputs['anchor_reference']['path']);oldinit=load(inputs['anchor_init']['path']);oldcontrol=load(inputs['anchor_control']['path']);pdu=J(inputs['static_preload']['path'])
    C('all7_initial_fields_byte_identical_to_original',set(init)==set(oldinit) and all(exact(init[k],oldinit[k]) for k in init))
    C('P0_P1_source64_exact',exact(P[:2],anchor['control_points_rad'][:2]) and exact(P[0].astype(np.float32),init['q_init_requested_FP32_rad']))
    C('analytic_qd0_FP32_original_exact',exact(ref['analytic_qd151_rad_s'][0].astype(np.float32),init['qd_init_requested_FP32_rad_s']))
    C('saved_tau_exact_old_target',exact(tau,oldcontrol['control__fp32_old_export__tau_ff_static0_FP32_Nm']))
    tau_rebuilt=kp*np.asarray(pdu['u0_rad'],np.float32)+(bias[:,0]+bias[:,1]*P[0].astype(np.float32))
    C('static_tau_original_FP32_formula_exact',exact(tau,tau_rebuilt))
    actualctx=load(inputs['actual_context']['path'],['actuator_gainprm','actuator_biasprm'])
    ids=np.asarray(receipt['PD_metadata']['actuator_ids'])
    C('saved_actual_gain_bias_match_context_row0',exact(kp,actualctx['actuator_gainprm'][0,ids,0]) and exact(bias,actualctx['actuator_biasprm'][0,ids,:3]))
    C('source_backed_actual_PD_semantics_all_true',all(v is True for v in receipt['PD_metadata']['model_checks'].values()))
    src=load(inputs['derived_geometry']['path'],['time_s','joint_pos_rad','mapped_object_pose_w','joint_lower_rad','joint_upper_rad']);raw=load(inputs['designed_FD']['path'],['time_s','object_pose_mapped_w','raw_v_world_m_s','raw_a_world_m_s2','raw_omega_world_rad_s','raw_alpha_world_rad_s2'])
    C('all151_pose_clock_mapping_and_hardbounds_exact',exact(src['time_s'],ref['time_s']) and exact(src['mapped_object_pose_w'],ref['mapped_object_pose_w']) and exact(src['mapped_object_pose_w'],raw['object_pose_mapped_w']) and exact(src['joint_lower_rad'],pd['q_lower_rad']) and exact(src['joint_upper_rad'],pd['q_upper_rad']))
    pose=ref['mapped_object_pose_w'];Rknots=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
    output={k:[] for k in ['object_COM_world_m','object_v_world_m_s','object_a_world_m_s2','object_rotation_w','object_omega_world_rad_s','object_alpha_world_rad_s2']}
    for t in dense_t:
        i=min(int(np.searchsorted(times,t,side='right')-1),149);duration=times[i+1]-times[i];s=(t-times[i])/duration
        pc=ref['object_p_coefficients'][i];rc=ref['object_phi_coefficients'][i]
        def poly(coef,order):
            result=np.zeros(3)
            for n in range(order,6):
                factor=1
                for m in range(order):factor*=n-m
                result+=factor*coef[n]*s**(n-order)/duration**order
            return result
        for k,order in [('object_COM_world_m',0),('object_v_world_m_s',1),('object_a_world_m_s2',2)]:output[k].append(poly(pc,order))
        phi,v,a=poly(rc,0),poly(rc,1),poly(rc,2)
        R=Rknots[i]@Rotation.from_rotvec(phi).as_matrix();jr,djv=Jr_terms(phi,v)
        output['object_rotation_w'].append(R);output['object_omega_world_rad_s'].append(R@(jr@v));output['object_alpha_world_rad_s2'].append(R@(jr@a+djv))
    errs={k:maxabs(np.asarray(v)-ref[k]) for k,v in output.items()}
    tol=protocol['object_numeric_tolerances']
    C('all601_saved_object_polynomial_SO3_law_independent',all(errs[k]<=tol[k] for k in errs))
    knot_expected={'object_COM_world_m':pose[:,:3],'object_rotation_w':Rknots,'object_v_world_m_s':raw['raw_v_world_m_s'],'object_a_world_m_s2':raw['raw_a_world_m_s2'],'object_omega_world_rad_s':raw['raw_omega_world_rad_s'],'object_alpha_world_rad_s2':raw['raw_alpha_world_rad_s2']}
    knoterr={k:maxabs(ref[k][::4]-v) for k,v in knot_expected.items()};prefixerr={k:maxabs(ref[k][:61]-anchor[k]) for k in knot_expected}
    C('all151_object_knots_original_tolerances_pass',all(knoterr[k]<=tol[k] for k in knoterr))
    C('object_old61prefix_unchanged_original_tolerances_pass',all(prefixerr[k]<=tol[k] for k in prefixerr))
    R32=Rotation.from_quat(init['object_pose_init_requested_FP32_w'][[4,5,6,3]]).as_matrix()
    body0=(R32.T@raw['raw_omega_world_rad_s'][0]).astype(np.float32)
    C('object_world_linear_and_body_omega_initial_values',exact(pose[0].astype(np.float32),init['object_pose_init_requested_FP32_w']) and exact(raw['raw_v_world_m_s'][0].astype(np.float32),init['object_bodyorigin_v_requested_FP32_world_m_s']) and np.allclose(body0,init['object_angular_requested_FP32_body_rad_s'],rtol=0,atol=4*np.finfo(np.float32).eps))
    oldq=anchor['q_goal_rad'];newq=ref['q_goal_rad'];target=src['joint_pos_rad']
    delta=newq-target
    C('saved_global_fit_metrics_recomputed',float(np.mean(np.abs(delta)))==receipt['fit']['mean_abs_q_error_rad'] and maxabs(delta)==receipt['fit']['max_abs_q_error_rad'] and maxabs(delta[1:])==receipt['fit']['max_abs_error_after_initial_rad'])
    ranges=[]
    for label,a,b in [('frame0',0,1),('first16_including0',0,16),('first16_after0',1,16),('tail135',16,151),('all151',0,151)]:
        ranges.append({'range':label,'local_frames_half_open':[a,b],'time_first_last_s':[float(times[a]),float(times[b-1])],'abs_fit_mean_rad':float(np.mean(np.abs(delta[a:b]))),'abs_fit_max_rad':maxabs(delta[a:b])})
    prefix_delta=newq[:16]-oldq
    details['object_law']={'independent_saved601_reconstruction_maxabs_errors':errs,'original151_knot_maxabs_errors':knoterr,'old_prefix61_maxabs_errors':prefixerr,'world_vs_body_omega_semantics':'Reference angular velocity is world. Reset angular velocity is in body, computed using original FP32 reset quaternion; COM offset0. These are designed reference derivatives, not video measurements.'}
    details['fit_time_distribution']={'vs539d_target':ranges,'vs_old_RF16':{'mean_abs_q_difference_rad':float(np.mean(np.abs(prefix_delta))),'max_abs_q_difference_rad':maxabs(prefix_delta),'q0_difference_rad':maxabs(prefix_delta[0]),'per_frame_maxabs_rad':np.abs(prefix_delta).max(1).tolist()},'all151_perframe_fit_absmean_rad':np.mean(np.abs(delta),axis=1).tolist(),'all151_perframe_fit_absmax_rad':np.abs(delta).max(axis=1).tolist(),
                                      'scope':'The full151 fit changes the old first16 q beyond fixed P0/P1. Old prefix geometry or physical task qualification is not inherited; no new geometry was queried.'}

passed=sum(checks.values());total=len(checks)
result={'schema':'full151_CPU_input_saved_independent_audit:1','status':'PASS_REVIEW_CANDIDATE_REMAINS_FAILED' if passed==total and not exceptions else 'REVIEW_FAILURE_PRESERVED',
        'engineering_review_passed':passed==total and not exceptions,'candidate_qualified':False,'original_candidate_status':receipt['status'],'passed':passed,'total':total,'checks':checks,'exceptions':exceptions,
        'review_source':pin(__file__),'original_receipt':pin(RUN/'receipt.json'),'details':details,'solver_FK_collision_Env_WM_physics_GPU_calls':0,
        'independent_scope':'Saved arrays plus NumPy/SciPy interpolation/rotation/CSR algebra; no production imports, optimization, Torch replay, MjModel/Env or physics. CPU Torch parity/actual MJB semantics are original source-bound report evidence, not independently re-executed here.',
        'qualification_boundary':'Raw1e-12 gate remains FAIL/noREADY. Original target/rate/reference-force bounds passing is reported separately and never overrides the failed planning gate. No fullmotion physical, contact or training qualification.'}
(OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':passed,'total':total,'failed':[k for k,v in checks.items() if not v],'exceptions':exceptions,'report':pin(OUT/'report.json')},indent=2))
sys.exit(0 if result['engineering_review_passed'] else 1)
