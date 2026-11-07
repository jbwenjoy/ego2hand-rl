"""Independent saved fresh design/score/projection arithmetic, NumPy only.

No model, Torch, native Log, AD or physical derivative is replayed/certified.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'src/full151_gradient/public_self15_fresh_gradient105_v1.py'
CORE_SHA='7700904f0fcb1bc296dc9b860bf4ea784098ff915c7b7b318f30feb16586b037'
HELPERS={'task':('scripts/audit_component_task_gradient_saved_cpu_v1.py','472c72117dde51f7edd7c28d34301cef88939ebf6f5a64cb9bc6849dab42d0c5'),
 'scalar':('src/full151_gradient/boundary_final_eval_independent_math_v1.py','e6cf8bdf3273f13e78c3b81e0b79e262780296f9697e9ea4e0dcfb0fdb0d5457'),
 'response':('src/full151_gradient/calibration_confirmation_independent_math_v2.py','cac8f8fa098aa2ccfae24d0d710613c5b272b442ca591c55a9531a8a877a23d2')}
def require(v,s):
    if not v:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def module(p,h):
    require(sha(p)==h,'Frozen independent saved helper');s=importlib.util.spec_from_file_location('fresh_independent_'+Path(p).stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=module(ROOT/HELPERS['task'][0],HELPERS['task'][1]);M=module(ROOT/HELPERS['scalar'][0],HELPERS['scalar'][1]);R=module(ROOT/HELPERS['response'][0],HELPERS['response'][1])
checked=T.checked;same=T.same;bits=T.bits;load=T.load
ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
def close(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(close(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if type(a) is float:return type(b) is float and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-14+2e-12*abs(b)
    return type(a) is type(b) and a==b
def subset_close(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and all(k in b and subset_close(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(subset_close(x,y) for x,y in zip(a,b))
    return close(a,b)
def masks(ids):
    r={'ALL':np.ones(152,bool),'A0':np.asarray([x['path']=='A0' for x in ids]),'A2':np.asarray([x['path']=='A2' for x in ids])}
    r.update({'snapshot_'+str(s):np.asarray([x['snapshot_id']==s for x in ids]) for s in range(8)})
    r.update({m:np.asarray([x['type']==m for x in ids]) for m in ('baseline','direction')});return r
def sensor_statistics(logits,target,ids):
    probability=M.probability(logits);bce=M.bce_elements(logits,target.astype('?'));decision=logits>0;truth=target>0
    def one(mask,feature=None):
        l=logits[mask] if feature is None else logits[mask,feature];p=probability[mask] if feature is None else probability[mask,feature];y=truth[mask] if feature is None else truth[mask,feature];d=decision[mask] if feature is None else decision[mask,feature];bc=bce[mask] if feature is None else bce[mask,feature]
        tp=int((d&y).sum());fp=int((d&~y).sum());fn=int((~d&y).sum());tn=int((~d&~y).sum())
        return {'rows':int(mask.sum()),'BCE_mean_elements_F64' if feature is None else 'BCE_mean_F64':M.fmean(bc),'Brier_F64':M.fmean((p-y)**2),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'tie_zero_negative':int((l==0).sum()),'positive_recall':tp/(tp+fn) if tp+fn else None,'positive_precision':tp/(tp+fp) if tp+fp else None}
    result={}
    for name,mask in masks(ids).items():
        x=one(mask);pcount=[math.fsum(float(v) for v in row) for row in probability[mask]];ycount=[math.fsum(float(v) for v in row) for row in target[mask]]
        x.update(cached_sensor_labels=int(mask.sum()*15),perfeature15=[{'feature':j,**one(mask,j)} for j in range(15)],predicted_soft_count_mean_F64=M.fmean(pcount),cached_positive_count_mean_F64=M.fmean(ycount),predicted_penalty_component_mean_F64_reward_dt_0_02=M.fmean([-.02*v for v in pcount]),cached_penalty_component_mean_F64_reward_dt_0_02=M.fmean([-.02*v for v in ycount]),self15_reward_proxy_error_mean_F64_dt_0_02=M.fmean([.02*(y-p) for y,p in zip(ycount,pcount)]));result[name]=x
    return result,probability,bce
def native52_stats(D):
    a=D.astype('f8');blocks={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),'native_LogR':(43,46,'rad'),'v':(46,49,'m/s'),'omega':(49,52,'rad/s')}
    return {'perfeature_RMS':[math.sqrt(M.fmean(a[:,j]**2)) for j in range(52)],'perfeature_MAE':[M.fmean(abs(a[:,j])) for j in range(52)],'perfeature_maxabs':abs(a).max(0).tolist(),'blocks':{name:{'unit':unit,'component_RMS':math.sqrt(M.fmean(a[:,lo:hi]**2)),'MAE':M.fmean(abs(a[:,lo:hi])),'maxabs':float(abs(a[:,lo:hi]).max())} for name,(lo,hi,unit) in blocks.items()}}
def no_models():require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'No Torch/native/AD imports')
def metadata(protocol):
    no_models();require(sha(CORE)==CORE_SHA,'Executed frozen public15 source');p=read(protocol)
    require(p['schema']=='full151105_public_self15_fresh_gradient_CPU_protocol:1' and p['CPU_run_authorized'] is True and p['training_authorized'] is p['GPU_run_authorized'] is p['Env_or_physics_authorized'] is False,'Authorized fixed CPU evaluation only')
    require(len(p['inputs'])==13 and p['model_roles']==list(ROLES) and p['endpoint_input_scope']=='SNAPSHOT8_REPEAT19_COMMON_VISIBLE_DESIGN_NOT_ACTUAL_PRE152' and p['fresh_impulse_or_consistency_loss'] is False,'Fixed13 inputs; designed PRE scope')
    for q in [*p['inputs'].values(),*p['source_pins'].values()]:checked(q)
    require(any(same(q,pin(CORE)) for q in p['source_pins'].values()),'Protocol core membership');return p
def synthetic():
    no_models();a,b=module(ROOT/T.MATH_PATH.relative_to(ROOT),T.MATH_SHA),np.zeros(20,np.float32);b[0]=3;u=np.zeros((2,20),np.float32);v=u.copy();u[:,0]=[-.002,-.001];v[:,0]=[0,.001]
    r=a.projection_conventions(b,np.zeros(20,np.float32),u,v,np.zeros(2,np.float32),np.ones(2,np.float32),0)
    require(np.all(r['model_actual_coordinate_projection']==3.) and abs(r['model_legacy_effective_projection'][0]-1.5)<1e-6 and close({'none':None,'f':1.},{'none':None,'f':1.+1e-15}),'Synthetic clipped-coordinate and scalar postprocessing');return {'passed':1,'total':1,'actual_arrays_read':0}
def audit(protocol,run,out):
    p=metadata(protocol);out.mkdir(exist_ok=False);write(out/'equations_before_actual_arrays.json',{'auditor':pin(__file__),'independent_helpers':{k:pin(ROOT/v[0]) for k,v in HELPERS.items()},'F64_atol':2e-14,'F64_rtol':2e-12,'captured_native52_Log_and_uVJP_recomputed':False,'actual_arrays_read':False})
    (out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes());r=read(run/'report.json');i=p['inputs'];checks={};details={};gate=lambda k,v:checks.__setitem__(k,bool(v))
    require(r['schema']=='full151105_public_self15_fresh_gradient_CPU_analysis:1' and r['status']=='COMPLETED_COMMON_VISIBLE_FRESH_ENDPOINTS_AND_OBJECT_uVJP_NOT_PHYSICAL_JACOBIAN' and same(r['protocol'],pin(protocol)),'Completed actual fresh evaluation')
    sources=read(run/'source_inventory.json');outputs=read(run/'output_pins.json')
    for q in [*sources.values(),*outputs.values()]:checked(q)
    expectedzip={k+'/'+checked(v).name:v for k,v in sources.items()}
    with zipfile.ZipFile(run/'execution_sources.zip') as z:zipok=set(z.namelist())==set(expectedzip) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in expectedzip.items())
    gate('actual_protocol13_sources_unique_archive_outputs_exact',zipok and read(run/'protocol.json')==p and r['input_pins']==i and len({(checked(v).stat().st_dev,checked(v).stat().st_ino) for v in sources.values()})==len(sources))
    a=load(checked(i['fresh_arrays']));d=load(run/'fresh_design_and_physical_inputs.npz');expected,ids,pairs=T.BASE.expected_input(a,read(checked(i['fresh_branch_index'])),read(checked(i['fresh_snapshot_records'])))
    gate('fresh152_common_visible_design64_chords_source_controls_exact',all(bits(v,d[k]) for k,v in expected.items()) and T.subset(ids,read(run/'all152_design_identities.json')) and bits(d['target_cached_self15_FP32'],a['branch_self15']) and all(bits(d[k],a[v]) for k,v in [('snapshot_x8_FP32','snapshot_x'),('snapshot_u8_FP32','snapshot_u'),('goal_p8_FP32','goal_p'),('goal_R8_FP32','goal_R')]))
    T.typed(d['target_cached_self15_FP32'],(152,15),'f4','cached15');require(np.isin(d['target_cached_self15_FP32'],[0,1]).all(),'All cached binary15')
    physicalterms=T.score_terms64(a['branch_x_next'],a['goal_p'][np.arange(152)//19],a['goal_R'][np.arange(152)//19]);bound=8*np.finfo('f4').eps*np.maximum(1.,abs(physicalterms))
    gate('physical_POST20_goalk_2plus8_primary_GPU_original_bound',bool((abs(a['branch_object_terms_k'].astype('f8')-physicalterms)<=bound).all()) and bits(a['branch_object_terms_k'].sum(-1,dtype='f4'),a['branch_object_score_k']))
    bases=T.baseline_rows(a);mathref=module(T.MATH_PATH,T.MATH_SHA);groups,physical=T.physical_rows(a,pairs,mathref)
    gate('all8_baseline_gates32_shared_floor_groups_and64_rows_CSV',all(q['passed'] for q in bases) and T.raw_tree_equal(bases,read(run/'baseline8.json')) and T.subset(groups,read(run/'physical_groups32.json')) and T.csv_matches(run/'physical_groups32.csv',groups) and T.subset(physical,read(run/'physical_rows64.json')) and T.csv_matches(run/'physical_rows64.csv',physical))
    parent_context=load(checked(i['physical_context_buffers']));gate('original_frozen_effective52_and_xi_bytes_not_fresh_impulse_targets',bits(d['effective_s52_FP32'],parent_context['effective_s52_FP32']) and bits(d['physical_xi46_FP32'],parent_context['physical_xi46_FP32']) and (d['effective_s52_FP32']>0).all())
    allsummary={};native_ok=True;sensor_ok=True;projection_ok=True;score_ok=True
    for role in ROLES:
        f=load(run/(role+'_endpoint152.npz'));e=load(run/(role+'_center8_object_uVJP.npz'));s=r['summary'][role]
        for key,shape in [('prediction58_FP32',(152,58)),('self15_logits_FP32',(152,15)),('native_point_D52_FP32',(152,52)),('pred_common52_FP32',(152,52)),('point_rows_FP32',(152,))]:T.typed(f[key],shape,'f4',key)
        for key,shape in [('prediction58_FP32',(8,58)),('self15_logits_FP32',(8,15)),('object_terms_FP32',(8,2)),('object_score_FP32',(8,)),('gradient_u8x20_FP32',(8,20))]:T.typed(e[key],shape,'f4',key)
        D=f['native_point_D52_FP32'];C=f['pred_common52_FP32'];Ct=d['target_common52_FP32'];x=d['x152_FP32'];target=d['target152_FP32'];pred=f['prediction58_FP32']
        for lo,hi,a0,b0 in ((0,40,0,40),(40,43,40,43),(46,49,52,55),(49,52,55,58)):
            native_ok &= bits(D[:,lo:hi],target[:,a0:b0]-pred[:,a0:b0]) and bits(C[:,lo:hi],pred[:,a0:b0]-x[:,a0:b0]) and bits(Ct[:,lo:hi],target[:,a0:b0]-x[:,a0:b0])
        native_ok &= all(bool((np.linalg.norm(q[:,43:46].astype('f8'),axis=1)<math.pi-.01).all()) for q in (D,C,Ct))
        calc=R.all_confirmation_arithmetic(D,C,Ct,d['effective_s52_FP32'],d['minus_indices'],d['plus_indices'],d['actual_du_rad_F64'],pred,target)
        native_ok &= bits(f['raw58_signed_error_F64'],calc['arrays']['raw_state_error152x58_F64']) and close(T.BASE.raw58_schema_stats(calc['raw58_state_error_original_units']),s['raw58_state_statistics_all152']) and close(native52_stats(D),s['native52_point_statistics_all152']) and close(calc['all64_normalized_pair_MSE_F64'],s['common_pre_response64_MSE_F64'])
        squared=(D/d['effective_s52_FP32'])**2;row_checks=[M.native_fp32_reduction_evidence(f['point_rows_FP32'][j],squared[j]) for j in range(152)]
        native_ok &= all(v['within_envelope'] for v in row_checks) and M.native_fp32_reduction_evidence(s['point52_old_confirmation_flatmean_FP32'],squared)['within_envelope'] and M.native_fp32_reduction_evidence(s['point52_newfamily_rowmean_FP32'],f['point_rows_FP32'])['within_envelope']
        sensors,probability,bce=sensor_statistics(f['self15_logits_FP32'],d['target_cached_self15_FP32'],ids)
        sensor_ok &= close(sensors,s['cached_sensor15_all13_cohorts']) and np.allclose(probability,f['probability15_F64'],atol=2e-14,rtol=2e-12) and np.allclose(bce,f['BCE_elements_F64'],atol=2e-14,rtol=2e-12)
        refscore=T.score_terms64(e['prediction58_FP32'],a['goal_p'],a['goal_R']);score_ok &= bool((abs(e['object_terms_FP32'].astype('f8')-refscore)<=8*np.finfo('f4').eps*np.maximum(1.,abs(refscore))).all()) and bits(e['object_terms_FP32'].sum(-1,dtype='f4'),e['object_score_FP32'])
        rows=[]
        for ss in range(8):
            for j,q in enumerate(T.JOINTS):
                ix=np.asarray([ss*19+3+8*ei+2*j for ei in range(2)]);v=mathref.projection_conventions(e['gradient_u8x20_FP32'][ss],a['snapshot_u'][ss],a['branch_u'][ix],a['branch_u'][ix+1],a['branch_object_score_k'][ix],a['branch_object_score_k'][ix+1],q)
                for ei in range(2):
                    rr=dict(physical[(ss*4+j)*2+ei]);aa=float(v['model_actual_coordinate_projection'][ei]);ll=float(v['model_legacy_effective_projection'][ei]);rr.update(WM_AD_coordinate_at_baseline_u0=float(v['model_nominal_coordinate_gradient']),WM_AD_actual_delta_projected_per_coordinate=aa,WM_AD_legacy_effective_direction=ll,actual_secant_minus_WM_AD=rr['physical_actual_coordinate_secant']-aa,legacy_slope_minus_WM_AD_effective=rr['physical_legacy_nominal_epsilon_slope']-ll);rows.append(rr)
        comparison=T.comparison_summary(rows);projection_ok &= T.subset(rows,read(run/(role+'_object_gradient_comparison64.json'))) and T.csv_matches(run/(role+'_object_gradient_comparison64.csv'),rows) and subset_close(comparison,s['task_object_gradient'])
        allsummary[role]={'point_scaled_MSE_F64':M.fmean((D.astype('f8')/d['effective_s52_FP32'].astype('f8'))**2),'common_pre_response64_MSE_F64':calc['all64_normalized_pair_MSE_F64'],'object_gradient':comparison,'cached15_ALL':sensors['ALL'],'raw58':s['raw58_state_statistics_all152']}
    gate('all3_native52_nonrot_chart_F64_common_response_and_conditional_FP32_reductions',native_ok);gate('all39_cached15_cohort_feature_metrics_null_denominators',sensor_ok);gate('all3_center8_object2plus8_scores_original_FP32_bound',score_ok);gate('all3x64_actual_vs_legacy_projections_and_CSV_summaries',projection_ok)
    rng=T.raw_rng(run);gate('raw_RNG_values_vectors_unchanged',all(rng.values()) and all(r['raw_RNG_value_and_byte_equality'].values()))
    labels={key:sum(q['label']==key for q in groups) for key in ('STABLE_NONFLAT','FLAT_OR_BELOW_NOISE','NONFLAT_UNSTABLE')}
    gate('fixed_budget_newfamily_states_and_scientific_limits',r['counts']==p['expected_counts'] and all(r['model_state_and_norm_checks'].values()) and labels==r['physical_group_counts']=={'STABLE_NONFLAT':6,'FLAT_OR_BELOW_NOISE':25,'NONFLAT_UNSTABLE':1} and r['primary_score_contract']==p['score_contract'] and all(r[k] is False for k in ('fresh_J_K_r_saved_targets_or_consistency_loss','old_and_fresh_cohorts_merged','whole_hidden_state_qualified','physical_gradient_qualified','Actor_or_CHORD_benefit_qualified','new_blind_confirmation')))
    no_models();passed=sum(checks.values());result={'schema':'full151105_public_self15_fresh_gradient_saved_independent:1','status':'PASS_SAVED_ALL3_FRESH152_AND64_OBJECT_GRADIENT_ARITHMETIC' if passed==len(checks) else 'FAIL_SAVED_FRESH_ARITHMETIC_RETAINED','engineering_qualified':passed==len(checks),'checks':checks,'passed':passed,'total':len(checks),'auditor':pin(__file__),'protocol':pin(protocol),'producer_report':pin(run/'report.json'),'run_dir':str(run),'physical_group_counts':labels,'independent_summary':allsummary,'raw_RNG_evidence':rng,'new_Torch_model_Log_AD_physics_GPU':0,'limitations':['Native Log52 and input-u AD are captured/source-backed, not independently replayed or physical derivatives.','Point FP32 reductions use conditional roundoff envelopes; no kernel-bit replay claim.','Common-visible repeat19 input is not measured branch PRE152. Cached15 mapping delegated to fresh540.','All rows retained; one known lineage, dependent eps/windows, no Actor/CHORD/ego efficacy.']};write(out/'report.json',result);require(result['engineering_qualified'],'Independent saved arithmetic check failed');return {'status':result['status'],'passed':passed,'total':len(checks),'report':pin(out/'report.json')}
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol');p.add_argument('--run-dir');p.add_argument('--output');p.add_argument('--run',action='store_true');p.add_argument('--synthetic',action='store_true');a=p.parse_args()
    if a.synthetic:print(json.dumps(synthetic()));return
    require(a.protocol,'Protocol required')
    if a.run:require(a.run_dir and a.output,'Actual saved paths required');v=audit(Path(a.protocol).resolve(strict=True),Path(a.run_dir).resolve(strict=True),Path(a.output).resolve())
    else:metadata(Path(a.protocol).resolve(strict=True));v={'status':'METADATA_ONLY_NO_SAVED_ARRAYS'}
    print(json.dumps(v))
if __name__=='__main__':main()
