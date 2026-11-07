"""Saved8-major SCP algebra/native-record audit; no producer import or solve."""
from pathlib import Path
import hashlib,json,sys,traceback
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
ROOT=Path('D:/workspace/supermimic-full151-observation-study');RUN=ROOT/'outputs/full151_contact_coupled_scp_v1';OUT=Path(__file__).parent
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in (a.files if keys is None else keys)}
def eq(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def maxabs(a):return float(np.max(np.abs(a))) if np.size(a) else 0.
def csr(a,prefix):return sparse.csr_matrix((a[prefix+'_data'],a[prefix+'_indices'],a[prefix+'_indptr']),shape=tuple(a[prefix+'_shape']))
def matrix_error(a,b):
    if a.shape!=b.shape:return float('inf')
    v=(a-b).tocsr();return float(np.max(np.abs(v.data))) if v.nnz else 0.
checks={};details={};exceptions={}
def C(k,v):
    if k in checks:raise ValueError('Duplicate check '+k)
    checks[k]=bool(v)
def section(name):
    def deco(fn):
        try:fn()
        except Exception:checks[name+'_exception_free']=False;exceptions[name]=traceback.format_exc()
    return deco
receipt=J(RUN/'receipt.json');protocol=J(receipt['protocol']['path']);inputs=protocol['inputs'];parent=Path(inputs['parent_reference']['path']).parent
ref0=load(inputs['parent_reference']['path']);finalref=load(RUN/'reference_candidate151.npz');finalcontrol=load(RUN/'control_candidate150.npz')
pd=load(parent/'actual_PD_parameters.npz');control0=load(inputs['parent_control']['path'],['tau_ff_static0_FP32_Nm']);tau=control0['tau_ff_static0_FP32_Nm'];init=load(inputs['initialization']['path'])
target=load(inputs['derived_geometry']['path'],['joint_pos_rad'])['joint_pos_rad'];catalog=load(RUN/'catalog.npz');pairs=catalog['pairs'];kind=catalog['pair_kind'];names=catalog['geom_names']
B=BSpline(ref0['Bspline_knots_s'],np.eye(153),3,extrapolate=False);t151=np.arange(151)*.02;t601=np.arange(601)*.005
B151=B(t151);B601=B(t601);Bfree=B601[:,2:]
def curve(P,t,nu=0):
    v=B(t,nu=nu)[:,1:]@(P[1:]-P[0]);return v+P[0] if nu==0 else v
def fit_cost(P):return float(np.mean(np.abs(curve(P,t151)-target),axis=0).sum()+4*np.mean(np.abs(np.diff(P,n=2,axis=0)),axis=0).sum())
oldparts=[load(parent/'joint_LPs'/f'joint{j:02d}_input.npz') for j in range(20)]
oldA=sparse.block_diag([sparse.csr_matrix((a['A_data'],a['A_indices'],a['A_indptr']),shape=tuple(a['A_shape'])) for a in oldparts],format='csr')
oldb=np.concatenate([a['b'] for a in oldparts]);oldc=np.concatenate([a['c'] for a in oldparts]);P0=ref0['control_points_rad'][0]
shift=np.concatenate([np.r_[np.full(151,P0[j]),np.zeros(302)] for j in range(20)])
grid_cache={};grid_stats=[];major_stats=[]

def read_grid(folder,label,P=None):
    g=load(folder/'native_grid.npz');stored=J(folder/'summary.json');ids=J(folder/'witness_ids.json');M=len(g['contactdist_m'])
    si=g['sample_index'].astype(int);pi=g['pair_index'].astype(int);key=si*273+pi
    count=np.bincount(key,minlength=601*273).reshape(601,273);mask=count>0
    d=np.full(601*273,np.inf);np.minimum.at(d,key,g['contactdist_m']);d=d.reshape(601,273);d[~mask]=np.nan
    tie=np.bincount(key[np.abs(g['contactdist_m']-d.ravel()[key])<=1e-9],minlength=601*273).reshape(601,273)
    valid=np.all((si>=0)&(si<601)) and np.all((pi>=0)&(pi<273)) and np.isfinite(g['contactdist_m']).all() and np.isfinite(g['position_w_m']).all() and np.isfinite(g['normal_a_to_b_w']).all() and np.isfinite(g['distance_jacobian_q_m_per_rad']).all()
    valid &= np.all(np.abs(np.linalg.norm(g['normal_a_to_b_w'],axis=1)-1)<=1e-6)
    valid &= all(tuple(sorted(order))==tuple(pairs[p]) for order,p in zip(g['geom_order'],pi))
    expectedids=[{'sample':int(s),'pair_index':int(p),'native_contact_index':int(n),'native_geom_order':list(map(int,o))} for s,p,n,o in zip(si,pi,g['native_contact_index'],g['geom_order'])]
    C(label+'_raw_witness_fields_and_ids_consistent',valid and ids==expectedids and len(ids)==M)
    C(label+'_selected_min_mask_counts_ties_rebuilt',np.array_equal(mask,g['generated_mask']) and np.array_equal(count,g['contact_count']) and np.array_equal(tie,g['tie_count']) and np.array_equal(d[mask],g['selected_distance_m'][mask]) and np.isnan(g['selected_distance_m'][~mask]).all())
    if P is not None:C(label+'_q601_matches_prescribed_bspline_noFK',maxabs(curve(P,t601)-g['q601_rad'])<=1e-12)
    C(label+'_object_pose_same_prescribed_all601',eq(g['object_pose601_w'],grid_cache['base']['grid']['object_pose601_w']) if label!='base' else g['object_pose601_w'].shape==(601,7))
    depth=np.where(mask,np.maximum(-d,0),0.);maxdepth=float(depth.max())
    reconstructed={'maximum_generated_depth_m':maxdepth,'planning_max_violation_m':max(0.,maxdepth-.002999),
                   'hand_object_max_m':float(depth[:,kind=='hand_object'].max()),'hand_self_max_m':float(depth[:,kind=='hand_self'].max()),
                   'hand_object_samples_over3mm':int(np.count_nonzero(depth[:,kind=='hand_object'].max(1)>.003)),
                   'hand_self_samples_over3mm':int(np.count_nonzero(depth[:,kind=='hand_self'].max(1)>.003)),
                   'hand_object_absent_unknown_samples':int(np.count_nonzero(~mask[:,kind=='hand_object'].any(1))),
                   'tied_samples':int(np.count_nonzero(np.any(tie>1,axis=1))),'sampled_geometry_gate':bool(maxdepth<=.003),'witness_count':M}
    C(label+'_all_summary_numbers_exact',stored==reconstructed)
    row={'label':label,**reconstructed,'generated_pair_sample_count':int(mask.sum()),'raw_witness_count':M,'geometry_branch_signature':sha(folder/'witness_ids.json')}
    grid_stats.append(row);out={'grid':g,'summary':reconstructed,'folder':folder,'witness_ids':ids};grid_cache[label]=out;return out

def geometry_A(g):
    rr=[];cc=[];vv=[]
    for r,(s,Jq) in enumerate(zip(g['sample_index'],g['distance_jacobian_q_m_per_rad'])):
        ci=np.flatnonzero(Bfree[int(s)]!=0)
        for j in np.flatnonzero(Jq!=0):
            rr.extend([r]*len(ci));cc.extend((j*453+ci).tolist());vv.extend((-Jq[j]*Bfree[int(s),ci]).tolist())
    return sparse.csr_matrix((vv,(rr,cc)),shape=(len(g['contactdist_m']),9060))

def post_control(P):
    kp=pd['kp'];b=pd['bias'];q=curve(P,t151);v=curve(P,t151,1);q32=q.astype(np.float32);v32=v.astype(np.float32)
    pre=(((tau-b[:,0])-b[:,1]*q32[:150])-b[:,2]*v32[:150])/kp
    U=np.minimum(np.maximum(pre,pd['target_lower']+np.float32(.020)),pd['target_upper']-np.float32(.020))
    rate=np.diff(U.astype(float),axis=0)/.02
    bsL=B(t151[:-1]);bsR=B(t151[1:]);dL=B(t151[:-1],nu=1);dR=B(t151[1:],nu=1);hh=np.diff(t151)[:,None,None]
    # Evaluate endpoint geometry directly, independently of the stored Bernstein
    # matrix chain. FP64 grouping differences are recorded, never called bitwise.
    qL=curve(P,t151[:-1]);qR=curve(P,t151[1:]);vL=curve(P,t151[:-1],1);vR=curve(P,t151[1:],1)
    qb=np.stack([qL,qL+hh[:,0,:]*vL/3,qR-hh[:,0,:]*vR/3,qR],axis=1)
    v2=3/hh*np.diff(qb,axis=1);vb=np.stack([v2[:,0],v2[:,0]/3+2*v2[:,1]/3,2*v2[:,1]/3+v2[:,2]/3,v2[:,2]],axis=1)
    force=kp*U.astype(float)[:,None,:]+b[:,0]+b[:,1]*qb+b[:,2]*vb
    flo=np.maximum(pd['actuator_force_bounds'][:,0],pd['joint_force_bounds'][:,0]);fhi=np.minimum(pd['actuator_force_bounds'][:,1],pd['joint_force_bounds'][:,1])
    qq=curve(P,t601).astype(np.float32);vv=curve(P,t601,1).astype(np.float32);index=np.minimum(np.arange(601)//4,149);uu=U[index]
    affine=kp*uu+((b[:,0]+b[:,1]*qq)+b[:,2]*vv);difference=(kp*(uu-qq)+b[:,0])+b[:,2]*vv
    gates={'q_controlpoint_hull':bool(np.maximum(np.maximum(pd['q_lower_rad']-P,P-pd['q_upper_rad']),0).max()<=1e-12),
           'C2_domain':bool(np.all(U>=pd['target_lower']+np.float32(.020)) and np.all(U<=pd['target_upper']-np.float32(.020))),
           'C2_projection_identity':eq(pre,U),'same_fixed_initial_u0':eq(U[0],init['u_planned0_FP32_rad']),
           'target149_updates_le15':bool(maxabs(rate)<=15.),'postC2_wholeinterval_real_affine_Bernstein_caps':bool(np.all(force>=flo) and np.all(force<=fhi)),
           'FP32_affine601_reference_caps':bool(np.all(affine>=flo) and np.all(affine<=fhi)),'FP32_difference601_reference_caps':bool(np.all(difference>=flo) and np.all(difference<=fhi))}
    return {'gates':gates,'pre':pre,'U':U,'rate':rate,'force':force,'affine':affine,'difference':difference,'q':q,'qd':v}

@section('major_math_and_grids')
def major_math_and_grids():
    P=ref0['control_points_rad'].copy();base=read_grid(RUN/'base_geometry','base',P);grid=base;phi=fit_cost(P)+1e4*base['summary']['planning_max_violation_m']
    C('base_direct_fit_phi_recompute',fit_cost(P)==receipt['base_fit_cost'] and base['summary']==receipt['base_summary'])
    radius=.15;lastaccepted=None;lasteta=None
    for n,record in enumerate(receipt['majors'],1):
        folder=RUN/f'major{n:02d}';a=load(folder/'subproblem.npz');sol=load(folder/'solver_result.npz');la=load(folder/'linear_audit_arrays.npz');trialfile=load(folder/'trial_control_points.npz');merit=load(folder/'merit_raw.npz');contract=J(folder/'contract.json');local=J(folder/'receipt.json')
        z=sol['x'];A=csr(a,'A_raw');Ao=csr(a,'A_original');res=A@z-a['b_raw'];ores=Ao@(z[:9060]+a['absolute_shift'])-a['b_original'];excess=np.maximum(np.maximum(a['lower']-z,z-a['upper']),0)
        rawpass=bool(res.max()<=1e-12 and ores.max()<=1e-12 and excess.max()<=1e-12)
        C(f'major{n}_record_and_center_acceptance_identity',local==record and record['major']==n and record['center_accepted_major']==lastaccepted and record['radius_rad']==radius and eq(a['center_P'],P))
        C(f'major{n}_global_original20_matrix_and_shift_exact',matrix_error(Ao,oldA)==0 and eq(a['b_original'],oldb) and eq(a['absolute_shift'],shift) and eq(a['c'],np.r_[oldc,1e4]))
        C(f'major{n}_original_and_centered_raw_residual_arrays_exact',eq(res,la['raw_residuals']) and eq(ores,la['original_absolute_raw_residuals']) and eq(excess,la['bound_excesses']))
        fields={'max_raw_Ax_minus_b':float(res.max()),'max_centered_control_raw_Ax_minus_b':float(res[:48040].max()),'max_original_absolute_raw_Ax_minus_b':float(ores.max()),'max_geometry_linear_raw_residual_m':float(res[48040:].max()),'max_bound_excess':float(excess.max()),'raw_primal_pass':rawpass,'eta_m':float(z[-1]),'eta_gate_pass':bool(0<=z[-1]<=1e-9)}
        C(f'major{n}_all_reported_raw_primal_numbers_recomputed',all(record['linear_audit'][k]==v for k,v in fields.items()) and rawpass)
        center=[];lower=[];upper=[]
        for j,p in enumerate(oldparts):
            free=P[2:,j];ff=np.abs(p['q_matrix']@free+p['q_base']-p['target151']);ss=np.abs(p['second_difference_matrix']@free+p['second_difference_base']);center.extend(np.r_[free-P0[j],ff,ss])
            lower.extend(p['bound_lower']-np.r_[np.full(151,P0[j]),np.zeros(302)]);upper.extend(p['bound_upper']-np.r_[np.full(151,P0[j]),np.zeros(302)])
        center=np.asarray(center);eta_center=max(0.,float(np.max(-(grid['grid']['contactdist_m']+.002999))))
        C(f'major{n}_center_vector_direct_epigraph_eta_rebuilt',eq(a['center_vector'],np.r_[center,eta_center]))
        lower=np.r_[lower,0.];upper=np.r_[upper,np.inf]
        for j in range(20):
            ids=j*453+np.arange(151);lower[ids]=np.maximum(lower[ids],center[ids]-radius);upper[ids]=np.minimum(upper[ids],center[ids]+radius)
        C(f'major{n}_oldhard_plus_trust_bounds_exact',eq(lower,a['lower']) and eq(upper,a['upper']))
        G=geometry_A(grid['grid']);rhs=grid['grid']['contactdist_m']+.002999
        geometry_rhs_error=maxabs(a['b_raw'][48040:]-(rhs+G@center))
        C(f'major{n}_all_witness_geometry_CSR_and_center_rhs',A.shape==(48040+len(rhs),9061) and matrix_error(A[:48040,:9060],oldA)==0 and A[:48040,9060:].nnz==0 and matrix_error(A[48040:,:9060],G)==0 and np.array_equal(A[48040:,9060].toarray().ravel(),-np.ones(len(rhs))) and eq(a['b_raw'][:48040],oldb-oldA@shift) and geometry_rhs_error<=1e-12)
        C(f'major{n}_contract_retains_complete_witnessIDs_units',contract['geometry_witness_ids']==grid['witness_ids'] and record['witness_rows']==len(rhs) and contract['row_scale']==1e4 and contract['geometry_planning_depth_m']==.002999)
        trial=P.copy()
        for j in range(20):trial[2:,j]=z[j*453:j*453+151]+P0[j]
        eta=float(z[-1]);C(f'major{n}_trial_eta_exact_unpacked',eq(trial,trialfile['P']) and eta==float(trialfile['eta_m']) and eta==record['eta_m'])
        # c@z includes epigraph variables; acceptance intentionally recomputes
        # the original direct absolute-fit cost from P, not solver.fun alone.
        objerr=abs(float(a['c']@z)-float(sol['fun']))
        C(f'major{n}_solver_objective_cdotz_match',objerr<=1e-10)
        post=post_control(trial);C(f'major{n}_postC2_reference_gates_independently_match',post['gates']==record['postC2_gates'] and all(post['gates'].values()))
        tg=read_grid(folder/'geometry',f'major{n}',trial)
        delta=(trial[2:]-P[2:]).T.ravel()
        # G uses global block columns. Map the joint-major3020 coefficient step
        # explicitly, without importing geometry_to_global_map.
        delta_global=np.zeros(9060)
        for j in range(20):delta_global[j*453:j*453+151]=delta[j*151:(j+1)*151]
        pred_violation=max(0.,float(np.max(G@delta_global-rhs))) if len(rhs) else 0.
        direct=fit_cost(trial);pred_phi=direct+1e4*pred_violation;true_phi=direct+1e4*tg['summary']['planning_max_violation_m'];pred_gain=phi-pred_phi;gain=phi-true_phi;ratio=None if pred_gain<=0 else gain/pred_gain
        accept=bool(np.isfinite([phi,pred_phi,true_phi,pred_gain,gain]).all() and pred_gain>0 and gain>0 and ratio is not None and np.isfinite(ratio) and ratio>=.1)
        vals={'old_phi':phi,'predicted_phi':pred_phi,'new_phi':true_phi,'predicted_gain':pred_gain,'actual_gain':gain,'ratio':np.nan if ratio is None else ratio}
        numeric_error=max(abs(float(merit[k])-v) for k,v in vals.items())
        C(f'major{n}_direct_fit_true_prediction_Phi_rho_exact',numeric_error<=1e-10 and direct==record['direct_fit_cost'] and accept==record['accepted'] and abs(phi-record['old_actual_phi'])<=1e-10 and abs(pred_phi-record['predicted_phi'])<=1e-10 and abs(true_phi-record['new_actual_phi'])<=1e-10)
        C(f'major{n}_actual_geometry_summary_matches_raw',tg['summary']==record['geometry_summary'])
        major_stats.append({'major':n,'radius_rad':radius,'center_accepted_major':lastaccepted,'accepted':accept,'eta_m':eta,'max_raw_primal':float(res.max()),'max_old_absolute_primal':float(ores.max()),'max_bound_excess':float(excess.max()),'solver_objective_cdot_error':objerr,'independent_geometry_center_rhs_roundoff_m':geometry_rhs_error,
                            'direct_fit':direct,'old_phi':phi,'predicted_violation_m':pred_violation,'predicted_phi':pred_phi,'actual_phi':true_phi,'predicted_gain':pred_gain,'actual_gain':gain,'rho':ratio,'LP_eta_minus_predicted_max_violation_m':eta-pred_violation,'actual_max_violation_minus_eta_m':tg['summary']['planning_max_violation_m']-eta})
        if accept:P=trial;grid=tg;phi=true_phi;lastaccepted=n;lasteta=eta
        else:radius*=.5
        C(f'major{n}_radius_after_decision_exact',record['next_radius_rad']==radius)
    final=read_grid(RUN/'final_geometry','final',P)
    C('last_accepted_major8_eta_source_and_finalP',lastaccepted==receipt['last_accepted_major']==8 and lasteta==receipt['last_accepted_eta_m'] and eq(P,finalref['control_points_rad']))
    C('only_major6_rejected_same_center5_next_radius075',[m['major'] for m in major_stats if not m['accepted']]==[6] and major_stats[6]['center_accepted_major']==5 and major_stats[6]['radius_rad']==.075)
    C('final_fresh_geometry_and_direct_Phi_recomputed',final['summary']==receipt['final_geometry'] and fit_cost(P)==receipt['final_fit_cost'] and fit_cost(P)+1e4*final['summary']['planning_max_violation_m']==receipt['final_actual_phi'])
    C('final_lastaccepted_positive_eta_and_geometry_FAIL',lasteta>1e-9 and not final['summary']['sampled_geometry_gate'] and receipt['status']=='CPU_COUPLED_CANDIDATE_FAIL_RETAINED')
    details['majors']=major_stats;details['grid_summaries']=grid_stats
    details['global_merit_scope']={'base_Phi':receipt['base_fit_cost']+1e4*base['summary']['planning_max_violation_m'],'final_Phi':receipt['final_actual_phi'],'base_fit_cost':receipt['base_fit_cost'],'final_fit_cost':receipt['final_fit_cost'],
        'base_HO_depth_m':base['summary']['hand_object_max_m'],'final_HO_depth_m':final['summary']['hand_object_max_m'],'base_self_depth_m':base['summary']['hand_self_max_m'],'final_self_depth_m':final['summary']['hand_self_max_m'],
        'interpretation':'The specified maximum-depth merit decreased, but self penetration and fit cost increased. This is not uniform contact/geometry improvement; eta is local LP elastic slack and does not equal allowed physical penetration.'}
    details['scope']='All raw supplied witness/minimum/tie inventories are independently reconstructed. Native_ncon, query clock/qvel and uncatalogued-contact assertions were not serialized pergrid and remain pinned in-process source evidence; no claim of offline native-library/state replay.'

@section('final_control_and_counters')
def final_control_and_counters():
    P=finalref['control_points_rad'];post=post_control(P)
    C('final_all_original_postC2_gates_recomputed',all(post['gates'].values()) and all(receipt['checks']['final_'+k]==v for k,v in post['gates'].items()))
    C('final_FP32_proposal_C2_149rate_saved_exact',eq(post['pre'],finalcontrol['u_pre150_FP32_rad']) and eq(post['U'],finalcontrol['u_postC2_150_FP32_rad']) and eq(post['rate'],finalcontrol['planned_rate149_rad_s']))
    C('final_reference_force_Bernstein_saved_roundoff_and_FP32_exact',maxabs(post['force']-finalcontrol['postC2_real_affine_rawforce_Bernstein_Nm'])<=1e-10 and eq(post['affine'],finalcontrol['reference_affine601_FP32_Nm']) and eq(post['difference'],finalcontrol['reference_difference601_FP32_Nm']))
    totalpairs=sum(g['generated_pair_sample_count'] for g in grid_stats);totalwitness=sum(g['raw_witness_count'] for g in grid_stats)
    C('ten_grid_raw_count_and124122_pointJac_accounting',len(grid_stats)==receipt['cost']['grid_audits']==10 and 2*(totalpairs+totalwitness)==receipt['actual_native_calls']['allowed']['mj_jac']==124122)
    C('6010_collision6011_FKCOM_source_count_consistency',receipt['actual_native_calls']['allowed']['mj_collision']==601*len(grid_stats)==6010 and receipt['actual_native_calls']['allowed']['mj_kinematics']==receipt['actual_native_calls']['allowed']['mj_comPos']==6011 and all(v==0 for v in receipt['actual_native_calls']['forbidden'].values()))
    C('8LP_cost_nested_sum_and_status_noREADY',receipt['cost']['LP_attempted']==receipt['cost']['LP_returned']==len(major_stats)==8 and sum(m['solver_wall_s'] for m in receipt['majors'])==receipt['cost']['solver_wall_s'] and not (RUN/'READY_CPU_SAMPLED_ONLY.json').exists())
    details['counters']={'generated_pair_sample_count_all10grids':totalpairs,'supplied_native_witness_count_all10grids':totalwitness,'inferred_mj_jac_by_pinned_loop':2*(totalpairs+totalwitness),'geometry_calls_recorded':6010,'FKCOM_each_recorded':6011,'LP_attempted_returned':8,'solver_wall_sum_s':receipt['cost']['solver_wall_s'],'CPU_total_wall_s':receipt['wall_s'],'cost_nested_not_additive':True}

passed=sum(checks.values());total=len(checks)
result={'schema':'full151_contact_coupled_SCP_saved_independent_audit:1','status':'PASS_SAVED_MATH_REVIEW_CANDIDATE_REMAINS_FAILED' if passed==total and not exceptions else 'REVIEW_FAILURE_PRESERVED',
        'engineering_math_review_passed':passed==total and not exceptions,'passed':passed,'total':total,'checks':checks,'exceptions':exceptions,'source':pin(__file__),'original_receipt':pin(RUN/'receipt.json'),'details':details,
        'candidate_qualified':False,'physical_reference_qualified':False,'contact_intent_target_ready':False,'reviewer_new_LP_FK_collision_Env_WM_physics_GPU_calls':0,
        'metadata_scope':'Lifecycle/source/lastaccepted file identity independently reviewed by Super. Original receipt preterminal live_state pin is known mutable/stale and is not silently asserted exact by this math audit.'}
(OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':passed,'total':total,'failed':[k for k,v in checks.items() if not v],'exceptions':exceptions,'report':pin(OUT/'report.json')},indent=2))
sys.exit(0 if result['engineering_math_review_passed'] else 1)
