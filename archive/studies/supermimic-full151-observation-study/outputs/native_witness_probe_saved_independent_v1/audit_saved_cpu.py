"""Saved164-query local probe audit. No geometry imports, calls or solver."""
from pathlib import Path
import collections,hashlib,json,sys,traceback
import numpy as np
ROOT=Path('D:/workspace/supermimic-full151-observation-study');RUN=ROOT/'outputs/full151_native_witness_probe_v1';OUT=Path(__file__).parent
J=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in (a.files if keys is None else keys)}
def eq(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def pins(x):
    if isinstance(x,dict):
        if set(x)=={'path','sha256'}:yield x
        else:
            for v in x.values():yield from pins(v)
    elif isinstance(x,list):
        for v in x:yield from pins(v)
checks={};details={};exceptions={}
def C(k,v):
    if k in checks:raise ValueError('Duplicate check '+k)
    checks[k]=bool(v)
def section(name):
    def deco(fn):
        try:fn()
        except Exception:checks[name+'_exception_free']=False;exceptions[name]=traceback.format_exc()
    return deco
receipt=J(RUN/'receipt.json');protocol=J(receipt['protocol']['path']);catalog=load(RUN/'catalog.npz');pairs=catalog['pairs'];kind=catalog['pair_kind'];names=catalog['geom_names'];h=1e-5
source_ref=load(protocol['inputs']['reference']['path'],['planned_q601_rad'])
oldgeom=load(protocol['inputs']['saved_geometry']['path'],['selected_distance_m','generated_mask','tie_count'])
oldraw=[json.loads(s) for s in Path(protocol['inputs']['saved_raw_contacts']['path']).read_text().splitlines()]
pd=load(Path(protocol['inputs']['reference']['path']).parent/'actual_PD_parameters.npz',['q_lower_rad','q_upper_rad'])
inventories=[];case_results=[]

@section('identity')
def identity():
    pp=list(pins(protocol))
    C('exact_v2_protocol_and_all13_current_pins',sha(receipt['protocol']['path'])==receipt['protocol']['sha256']=='c23a34badec739bd057cb3be1e8741e59c5c71039ed59bf5cbbd4fd5e91b13b6' and len(pp)==13 and all(sha(p['path'])==p['sha256'] for p in pp))
    C('all177_recorded_artifacts_SHA_match',len(receipt['artifacts'])==177 and all(sha(p['path'])==p['sha256'] for p in receipt['artifacts']))
    C('all7_source_before_copies_match',len(protocol['sources'])==7 and all(sha(RUN/'source_before'/(name+Path(p['path']).suffix))==p['sha256'] for name,p in protocol['sources'].items()))
    C('same_fixed4states_pairs_h_gate',protocol['states']==[17,335,489,330] and protocol['pair_indices']==[228,77,243,193] and protocol['epsilon_rad']==h and protocol['unique_FD_gate_m_per_rad']==1e-3 and protocol['queries']==164)
    C('original9_source_engineering_checks_true',len(receipt['checks'])==9 and all(v is True for v in receipt['checks'].values()) and receipt['status']=='PASS_LOCAL_WORST_BRANCHES_TIES_DIAGNOSTIC_ONLY')
    local=J(RUN/'READY_LOCAL_ONLY.json')
    C('local_READY_does_not_qualify_global_or_physics',local['receipt']['sha256']==sha(RUN/'receipt.json') and local['global_or_physical_qualified'] is False and receipt['physical_qualified'] is False and receipt['global_contact_Jacobian_qualified'] is False and receipt['all_future_SCP_branches_qualified'] is False)
    C('parent_full151_candidate_still_FAILED_noREADY',J(ROOT/'outputs/full151_cpu_input_v1/receipt.json')['status']=='CPU_CANDIDATE_GATES_FAIL_NO_QUALIFICATION' and not (ROOT/'outputs/full151_cpu_input_v1/READY.json').exists())

def inventory(v,label,base):
    m=len(v['contactdist_m']);mask=np.zeros(273,bool);count=np.zeros(273,int);tie=np.zeros(273,int);distance=np.full(273,np.nan);selected=np.full(273,-1,int)
    normal=np.full((273,3),np.nan);groups=collections.defaultdict(list);valid=True;jsame=0.
    for k in range(m):
        p=int(v['pair_index'][k]);idx=int(v['native_contact_index'][k]);order=tuple(v['geom_order'][k]);n=v['normal_a_to_b_w'][k]
        valid &= 0<=p<273 and tuple(sorted(order))==tuple(pairs[p]) and 0<=idx<int(v['native_ncon']) and np.isfinite(v['contactdist_m'][k]) and np.isfinite(v['position_w_m'][k]).all() and abs(np.linalg.norm(n)-1)<=1e-6
        groups[p].append(k)
    for p,ids in groups.items():
        d=v['contactdist_m'][ids];k=ids[int(np.argmin(d))];mask[p]=True;count[p]=len(ids);distance[p]=v['contactdist_m'][k];selected[p]=v['native_contact_index'][k]
        tie[p]=int(np.count_nonzero(np.abs(d-distance[p])<=1e-9));normal[p]=v['normal_a_to_b_w'][k]*(1 if tuple(v['geom_order'][k])==tuple(pairs[p]) else -1)
        if base:jsame=max(jsame,float(np.max(np.abs(v['distance_jacobian_q_m_per_rad'][k]-v['selected_distance_jacobian_q'][p]))))
    C(label+'_raw_inventory_distance_masks_count_ties_selected_exact',valid and np.array_equal(mask,v['generated_mask']) and np.array_equal(count,v['contact_count']) and np.array_equal(tie,v['tie_count']) and np.array_equal(selected,v['selected_native_index']) and np.array_equal(distance[mask],v['selected_distance_m'][mask]) and np.isnan(v['selected_distance_m'][~mask]).all() and np.array_equal(normal[mask],v['selected_normal_canonical_w'][mask]))
    C(label+'_Jacobian_presence_scope',np.isfinite(v['distance_jacobian_q_m_per_rad']).all() if base else np.isnan(v['distance_jacobian_q_m_per_rad']).all())
    C(label+'_selected_J_generated_finite_absent_NaN',np.isfinite(v['selected_distance_jacobian_q'][mask]).all() and np.isnan(v['selected_distance_jacobian_q'][~mask]).all())
    if base:C(label+'_selected_J_matches_independent_allwitness_array',jsame<=1e-12)
    valid &= len(set(v['native_contact_index'].tolist()))==m and m==int(v['native_ncon'])
    C(label+'_native_contact_count_complete',valid)
    inventories.append({'label':label,'base':base,'pairs':int(mask.sum()),'witnesses':m,'selected_allwitness_maxdiff':jsame if base else None})

@section('raw_FD')
def raw_FD():
    for ci,(case,sample,pair) in enumerate(zip(protocol['case_names'],protocol['states'],protocol['pair_indices'])):
        base=load(RUN/'samples'/(case+'_base.npz'));fdfile=load(RUN/(case+'_FD.npz'));inventory(base,case+'_base',True)
        C(case+'_source_q0_and_old_selected_geometry_match',eq(base['q'],source_ref['planned_q601_rad'][sample]) and np.array_equal(base['generated_mask'],oldgeom['generated_mask'][sample]) and np.max(np.abs(base['selected_distance_m'][base['generated_mask']]-oldgeom['selected_distance_m'][sample,base['generated_mask']]))<=1e-12)
        base_records=oldraw[sample]['records'];same_old_raw=len(base_records)==len(base['contactdist_m'])
        old_raw_max={'distance':0.,'position':0.,'normal':0.}
        if same_old_raw:
            for k,v in enumerate(base_records):
                same_old_raw &= v[:4]==[int(base['pair_index'][k]),int(base['native_contact_index'][k]),int(base['geom_order'][k,0]),int(base['geom_order'][k,1])]
                old_raw_max['distance']=max(old_raw_max['distance'],abs(v[4]-base['contactdist_m'][k]))
                old_raw_max['position']=max(old_raw_max['position'],float(np.max(np.abs(np.asarray(v[5])-base['position_w_m'][k]))))
                old_raw_max['normal']=max(old_raw_max['normal'],float(np.max(np.abs(np.asarray(v[6])-base['normal_a_to_b_w'][k]))))
        dp=[];dm=[];tp=[];tm=[];cp=[];cm=[];np_=[];nm=[];op=[];om=[];cross=[];qexcess=0.
        def order(v):
            k=np.flatnonzero((v['pair_index']==pair)&(v['native_contact_index']==v['selected_native_index'][pair]));return np.array([-1,-1]) if len(k)!=1 else v['geom_order'][k[0]]
        bo=order(base)
        for j in range(20):
            for sign,label,dv,tv,cv,nv,ov in [(1,'plus',dp,tp,cp,np_,op),(-1,'minus',dm,tm,cm,nm,om)]:
                v=load(RUN/'samples'/(case+f'_{label}_j{j:02d}.npz'));inventory(v,case+f'_{label}{j:02d}',False)
                q=base['q'].copy();q[j]+=sign*h
                C(case+f'_{label}{j:02d}_unclamped_coordinate_pose_exact',eq(v['q'],q) and eq(v['object_pose_w'],base['object_pose_w']))
                dv.append(v['selected_distance_m']);tv.append(v['tie_count']);cv.append(v['contact_count']);nv.append(v['selected_normal_canonical_w']);ov.append(order(v))
                excess=float(np.maximum(np.maximum(pd['q_lower_rad']-v['q'],v['q']-pd['q_upper_rad']),0).max());qexcess=max(qexcess,excess)
                if excess:cross.append({'joint':j,'sign':sign,'max_excess_rad':excess})
        dp,dm,tp,tm,cp,cm,np_,nm,op,om=map(np.asarray,(dp,dm,tp,tm,cp,cm,np_,nm,op,om))
        # The probe accumulates selected geom orders in default-int output arrays.
        op=op.astype(np.int64);om=om.astype(np.int64)
        finite=np.isfinite(dp[:,pair])&np.isfinite(dm[:,pair]);fd=np.full(20,np.nan);fd[finite]=(dp[finite,pair]-dm[finite,pair])/(2*h)
        fp=(dp[:,pair]-base['selected_distance_m'][pair])/h;fm=(dm[:,pair]-base['selected_distance_m'][pair])/h;jac=base['selected_distance_jacobian_q'][pair]
        n0=base['selected_normal_canonical_w'][pair]
        ns=bool(np.all(np.einsum('ij,j->i',np_[:,pair],n0)>1-1e-6) and np.all(np.einsum('ij,j->i',nm[:,pair],n0)>1-1e-6))
        unique=bool(base['tie_count'][pair]==1 and np.all(tp[:,pair]==1) and np.all(tm[:,pair]==1))
        cs=bool(np.all(cp[:,pair]==base['contact_count'][pair]) and np.all(cm[:,pair]==base['contact_count'][pair]));os=bool(np.all(op==bo) and np.all(om==bo))
        err=None if not finite.all() or not np.isfinite(fd-jac).all() else float(np.abs(fd-jac).max())
        gate=bool(finite.all() and unique and ns and cs and os and err is not None and err<=1e-3)
        active=(base['pair_index']==pair)&(np.abs(base['contactdist_m']-base['selected_distance_m'][pair])<=1e-9);grads=base['distance_jacobian_q_m_per_rad'][active]
        predp=grads.min(0) if len(grads) else np.full(20,np.nan);predm=-grads.max(0) if len(grads) else np.full(20,np.nan)
        C(case+'_central_plus_minus_and_selected_J_saved_exact',eq(fd,fdfile['FD_central']) and eq(fp,fdfile['FD_plus']) and eq(fm,fdfile['FD_minus']) and eq(jac,fdfile['selected_J']))
        C(case+'_all_branch_distance_tie_count_order_arrays_saved_exact',all(eq(a,fdfile[k]) for a,k in [(dp,'selected_dist_plus_allpairs'),(dm,'selected_dist_minus_allpairs'),(tp,'ties_plus'),(tm,'ties_minus'),(cp,'contact_counts_plus'),(cm,'contact_counts_minus'),(op,'selected_geom_order_plus'),(om,'selected_geom_order_minus'),(bo,'base_selected_geom_order')]))
        C(case+'_tied_gradient_inventory_and_directional_min_predictions_exact',eq(grads,fdfile['base_tied_witness_J']) and eq(predp,fdfile['tied_branch_pred_plus']) and eq(predm,fdfile['tied_branch_pred_minus']))
        original=receipt['cases'][ci]
        recompute={'base_tie_count':int(base['tie_count'][pair]),'all20_generated_both_sides':bool(finite.all()),'all20_unique':unique,'all20_normal_stable':ns,'all20_contact_count_stable':cs,'all20_selected_geom_order_stable':os,'max_FD_error_m_per_rad':err,'FD_max_qhard_excess_rad':qexcess,'local_unique_gate_applicable':case.startswith('worst'),'local_unique_gate_pass':gate}
        C(case+'_all_reported_gate_and_error_fields_recomputed',all(original[k]==v for k,v in recompute.items()))
        case_results.append({'case':case,'sample':sample,'pair_index':pair,'geom_names':[str(names[k]) for k in pairs[pair]],**recompute,
                             'source_old_raw_inventory_order_equal':bool(same_old_raw),'source_old_raw_numeric_maxdiff':old_raw_max,'hardbound_crossing_directions':cross,
                             'contact_count_changed_plus_joint_indices':np.flatnonzero(cp[:,pair]!=base['contact_count'][pair]).tolist(),'contact_count_changed_minus_joint_indices':np.flatnonzero(cm[:,pair]!=base['contact_count'][pair]).tolist(),
                             'one_sided_plus_minus_pred_max_abs_errors_m_per_rad':[float(np.max(np.abs(fp-predp))),float(np.max(np.abs(fm-predm)))],
                             'qualification_scope':'Necessary local detected-stability gate only' if case.startswith('worst') else 'Tie directional diagnostic only, never qualified even when central FD error is small'})

@section('counts_scope')
def counts_scope():
    selected_pair_calls=2*sum(v['pairs'] for v in inventories);extra_witness_calls=2*sum(v['witnesses'] for v in inventories if v['base'])
    C('exact164_inventory41_per_case',len(inventories)==164 and sum(v['base'] for v in inventories)==4)
    C('independent_pointJac_call_count1782',selected_pair_calls+extra_witness_calls==receipt['actual_calls']['allowed']['mj_jac']==1782)
    C('recorded164_collision165_FK_COM_counts',receipt['actual_calls']['allowed']['mj_collision']==164 and receipt['actual_calls']['allowed']['mj_kinematics']==receipt['actual_calls']['allowed']['mj_comPos']==165)
    C('all_recorded_forbidden_calls_zero',all(v==0 for v in receipt['actual_calls']['forbidden'].values()))
    C('two_necessary_unique_cases_pass_no_reselection',len(case_results)==4 and all(r['local_unique_gate_pass'] and r['local_unique_gate_applicable'] for r in case_results[:2]))
    C('two_tie_cases_not_qualified_by_small_error',all(not r['local_unique_gate_applicable'] and not r['local_unique_gate_pass'] and r['base_tie_count']>1 for r in case_results[2:]))
    details['call_counts']={'base_queries':4,'coordinate_perturbation_queries':160,'all_queries':164,'native_selected_pair_pointJac_calls_all164':selected_pair_calls,'additional_allwitness_pointJac_calls_4bases':extra_witness_calls,'total_mj_jac':selected_pair_calls+extra_witness_calls,'constructor_plus_query_FK_COM':165,'call_receipt_is_source_bound_not_rerun':True}
    details['cases']=case_results;details['inventory_summary']=inventories

passed=sum(checks.values());total=len(checks)
result={'schema':'native_allwitness_probe_saved_independent_audit:1','status':'PASS_SAVED_AUDIT_TWO_LOCAL_UNIQUE_GATES_ONLY' if passed==total and not exceptions else 'REVIEW_FAILURE_PRESERVED',
        'passed':passed,'total':total,'engineering_review_passed':passed==total and not exceptions,'checks':checks,'exceptions':exceptions,'source':pin(__file__),'original_receipt':pin(RUN/'receipt.json'),'details':details,
        'global_contact_Jacobian_qualified':False,'all_future_SCP_branches_qualified':False,'physical_reference_qualified':False,'parent_full151_candidate_qualified':False,
        'reviewer_FK_collision_pointJac_solver_Env_WM_physics_GPU_calls':0,
        'scope':'All164saved native inventories, actual perturbations, FD arrays and original local gates independently recomputed. Full body point-Jacobian matrices were not serialized; the all-witness/selected Jacobian agreement compares two source-pinned in-process computations, not an independent bodyJac replay. Native indices are inventory IDs, not persistent features. No new geometry queries or optimizer calls.'}
(OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':passed,'total':total,'failed':[k for k,v in checks.items() if not v],'exceptions':exceptions,'report':pin(OUT/'report.json')},indent=2))
sys.exit(0 if result['engineering_review_passed'] else 1)
