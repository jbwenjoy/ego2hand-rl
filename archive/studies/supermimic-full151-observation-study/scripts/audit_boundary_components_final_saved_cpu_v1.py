"""Independent final-eval saved arithmetic. No Torch, payload or native Log."""
import argparse,csv,hashlib,importlib.util,json,math,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'scripts/audit_boundary_initial_parity_saved_cpu_v3.py';BASE_SHA='5f05a82e114a08fadd4bca17d6e661f6c8c04590af46e16a58c60ae8eb2fb35b'
MATH=ROOT/'src/full151_gradient/boundary_final_eval_independent_math_v1.py';MATH_SHA='e6cf8bdf3273f13e78c3b81e0b79e262780296f9697e9ea4e0dcfb0fdb0d5457'
CORE=ROOT/'src/full151_gradient/boundary_component_final_evaluation105_v2.py';CORE_SHA='5eea0973150225a3b7cc659f8fbc11562000d98e8db9d98d9c9eafd6105a75e8'
STATS=ROOT/'scripts/describe_boundary_component_predictions_saved_cpu_v1.py';STATS_SHA='7487e16b819271fd3b6d3d6562d4802fd97216afa4e3f666f705948f88cce036'
ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
BLOCKS={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),'native_LogR':(43,46,'rad'),'v':(46,49,'m/s'),'omega':(49,52,'rad/s')}

def module(p,h):
    if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Frozen helper '+str(p))
    s=importlib.util.spec_from_file_location('saved_final_'+p.stem,p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a);return a
A=module(BASE,BASE_SHA);M=module(MATH,MATH_SHA)
pin=A.pin;sha=A.sha;checked=A.checked;read=A.read;dump=A.dump;load=A.load;bits=A.bits;same=A.same;require=A.require;typed=A.U.typed

def equivalent(a,b):
    if isinstance(b,dict):return isinstance(a,dict) and set(a)==set(b) and all(equivalent(a[k],b[k]) for k in b)
    if isinstance(b,list):return isinstance(a,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(b,str) or b is None or isinstance(b,bool):return type(a) is type(b) and a==b
    if isinstance(b,(int,np.integer)):return type(a) is int and a==int(b)
    return bool(np.isfinite(float(a)) and np.isfinite(float(b)) and abs(float(a)-float(b))<=2e-14+2e-12*abs(float(b)))

def metadata(path):
    require(sha(CORE)==CORE_SHA and sha(STATS)==STATS_SHA,'Final v2 source/statistics')
    p=read(path);require(p['schema']=='full151105_boundary_component_final_evaluation_CPU_protocol:1' and p['role']=='fixed_initial_and_two_final1000_component_evaluation','Fixed final namespace')
    require(p['CPU_run_authorized'] is True and p['GPU_run_authorized'] is p['training_authorized'] is False,'Root CPU-only evaluation')
    i=p['inputs'];require(set(i)=={'target_report','target_arrays','target_identities','target_saved8','target_source_inventory','original_initial_component','initial_validation','initial_saved_parity10','initial_saved_baseline','initial_parity_report','training_protocol','training_result','final_pair_manifest','final_saved_health5'},'Exact14 semantic input fields')
    for v in [*i.values(),*p['source_pins'].values()]:checked(v)
    require(any(same(pin(CORE),v) for v in p['source_pins'].values()),'Current v2 core enrolled')
    health=read(checked(i['final_saved_health5']));A.U.allchecks(health);require(health['iteration']==1000 and health['final_iteration_scope'] is True and same(health['producer_pair_manifest'],i['final_pair_manifest']),'Fixed actual final1000 saved health')
    pair=read(checked(i['final_pair_manifest']));require(pair['iteration']==1000 and set(pair['checkpoints'])==set(ROLES[1:]),'Both fixed finals')
    for k,v in pair['checkpoints'].items():require(same(v,health['checkpoints'][k]),'Same fixed checkpoint '+k)
    training=read(checked(i['training_protocol']));require('inputs' in training and 'input_pins' not in training and same(training['inputs']['initial_component'],i['original_initial_component']),'Real protocol input namespace')
    for key in ('initial_saved_parity10','target_saved8'):A.U.allchecks(read(checked(i[key])))
    for q in (ROOT/'outputs/full151105_boundary_final_eval_independent_math_preparation_v1/READY.json',ROOT/'outputs/full151105_boundary_component_statistics_contract_preparation_v1/READY.json'):
        z=read(q);A.U.allchecks(read(checked(z['cpu_report'])))
        for v in z['source_pins'].values():checked(v)
    return p

def schedule():
    ids=[]
    for pi,path in enumerate(('A0','A2')):
        for ki,k in enumerate((0,49,99,149)):
            for off in range(19):ids.append({'row_id':len(ids),'group_id':226*pi+150+19*ki+off,'path':path,'k':k,'mode':'baseline' if off<3 else 'direction'})
    return ids

def cohort_masks(ids):
    return [('ALL',np.ones(152,dtype='?'))]+[(p,np.asarray([x['path']==p for x in ids])) for p in ('A0','A2')]+[(f'{p}/k{k}',np.asarray([x['path']==p and x['k']==k for x in ids])) for p in ('A0','A2') for k in (0,49,99,149)]+[(m,np.asarray([x['mode']==m for x in ids])) for m in ('baseline','direction')]

def summary(f,e,d,initial,mask):
    D=e['native_point_delta52_FP32'].astype('f8');scale=d['effective_s52_FP32'].astype('f8');z=f['z12_FP32'].astype('f8');l=f['mode_logits2_FP32'].astype('f8');y=d['boundary_constraint_witness'];guess=l>0
    p=M.probability(l);bce=M.bce_elements(l,y);v=e['r_v_hat_F64']-d['r_v_saved_m_s_F64'];w=e['r_w_hat_F64']-d['r_w_saved_rad_s_F64'];cons=np.c_[v/scale[46:49],w/scale[49:52]]
    out={'groups':int(mask.sum()),'native52_scaled_MSE_F64_of_saved_FP32_delta':M.fmean((D[mask]/scale)**2),'mode_BCE_F64_of_saved_FP32_logits':M.fmean(bce[mask]),'mode_Brier_F64':M.fmean((p[mask]-y[mask])**2),'mode_Brier_from_captured_FP32_probability_F64_reduction':M.fmean((e['mode_probability2_FP32'][mask].astype('f8')-y[mask])**2),'impulse_z_MSE_F64':M.fmean((z[mask]-e['z_target_F64'][mask])**2),'saved_defect_consistency_MSE_F64':M.fmean(cons[mask]**2),'state_native_delta':{},'raw_R9_difference':M.vector_statistics((f['prediction58_FP32'].astype('f8')-d['target58_FP32'])[:,43:52][mask],'raw matrix entry; vector_RMS is Frobenius_RMS'),'mode':{},'boundary_impulse_error':{},'dimensionless_impulse_error':{},'observed_defect_match_error':{},'raw58_difference':{}}
    for name,(lo,hi,unit) in BLOCKS.items():out['state_native_delta'][name]=M.vector_statistics(D[mask,lo:hi],unit)
    vstats=M.vector_statistics(D[mask],'Mixed units: per-feature only; do not interpret this aggregate component_RMSE/vector_RMS physically');out['native52_per_feature']={k:v for k,v in vstats.items() if k.startswith('per_feature_') or k=='unit'}
    for name,lo,hi,unit in [('q',0,20,'rad'),('qd',20,40,'rad/s'),('p',40,43,'m'),('R_raw9',43,52,'raw matrix entry'),('v',52,55,'m/s'),('omega',55,58,'rad/s')]:out['raw58_difference'][name]=M.vector_statistics((f['prediction58_FP32'].astype('f8')-d['target58_FP32'])[mask,lo:hi],unit)
    out['mode']['both_bits_correct_groups']=int(np.all(guess[mask]==y[mask],axis=1).sum())
    for c,name in enumerate(('HO','nonhand')):
        yy,gg=y[mask,c],guess[mask,c];out['mode'][name]={'positive_witness':int(yy.sum()),'negative_witness':int((~yy).sum()),'TP':int((gg&yy).sum()),'FP':int((gg&~yy).sum()),'TN':int((~gg&~yy).sum()),'FN':int((~gg&yy).sum()),'exact_zero_logit_ties':int((l[mask,c]==0).sum()),'BCE_F64':M.fmean(bce[mask,c]),'Brier_F64':M.fmean((p[mask,c]-yy)**2)}
        for kind,sl in [('zJ',slice(0,3)),('zK',slice(3,6))]:out['dimensionless_impulse_error'][kind+'_'+name]=M.vector_statistics((z-e['z_target_F64'])[mask,c,sl],'dimensionless')
        for kind,key,truth,unit in [('J','J_hat_palm_F64','J_palm_F64','N s'),('K','K_hat_palm_F64','K_palm_F64','N m s')]:out['boundary_impulse_error'][kind+'_'+name]=M.vector_statistics((e[key]-d[truth])[mask,c],unit)
    for kind,key,truth,unit in [('J','J_hat_palm_F64','J_palm_F64','N s'),('K','K_hat_palm_F64','K_palm_F64','N m s')]:out['boundary_impulse_error'][kind+'_two_source_total']=M.vector_statistics((e[key].sum(1)-d[truth].sum(1))[mask],unit)
    for name,value,unit in [('linear',v,'m/s'),('angular',w,'rad/s')]:out['observed_defect_match_error'][name]=M.vector_statistics(value[mask],unit)
    current=f['self15_logits_FP32'].astype('f8');pr=M.probability(current);old=M.probability(initial);drift=current-initial
    out['self15_initial_drift']={'logit':M.vector_statistics(drift[mask],'logit; no true self15 labels'),'probability':M.vector_statistics((pr-old)[mask],'probability; no accuracy claim'),'mean_sum_sigmoid':M.fmean(pr[mask].sum(1)),'initial_mean_sum_sigmoid':M.fmean(old[mask].sum(1)),'mean_change_sum_sigmoid':M.fmean((pr-old)[mask].sum(1)),'strict_positive_logit_bit_flips':int(((current>0)!=(initial>0))[mask].sum()),'per_feature_strict_positive_logit_bit_flips':((current>0)!=(initial>0))[mask].sum(0).tolist(),'current_exact_zero_logit_ties':int((current[mask]==0).sum()),'initial_exact_zero_logit_ties':int((initial[mask]==0).sum()),'captured_FP32_mean_penalty_rate':M.fmean(e['self15_penalty_rate_FP32'][mask]),'captured_FP32_mean_penalty_component_dt002':M.fmean(e['self15_penalty_component_FP32'][mask]),'F64_rescore_mean_penalty_rate':-M.fmean(pr[mask].sum(1)),'F64_rescore_mean_penalty_component_dt002':-M.fmean(pr[mask].sum(1))*.02,'probability_precision_scope':'Unprefixed sigmoid/drift/Brier statistics use stable F64 rescore from saved FP32 logits; captured FP32 arrays/penalty summaries are separately named, not claimed bit-identical.','scope':'Initial function drift only. No self15 label, task reward, physical clearance or penalty accuracy qualification.'}
    return out

def audit(protocol_path,run,description,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes());dump(out/'equations_before_numeric_read.json',{'auditor':auditor,'math':pin(MATH),'base':pin(BASE),'actual_prediction_arrays_read':False,'F64_compare_rtol':2e-12,'F64_compare_atol':2e-14,'source_scope':'Captured native Log only; no checkpoint/Torch/native re-execution.'})
    p=metadata(protocol_path);i=p['inputs'];r=read(run/'report.json');desc=read(description/'report.json');checks={};details={}
    def gate(k,v):checks[k]=bool(v)
    require(r['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_AFTER1000_EVAL_NOT_CONFIRMATION' and same(r['protocol'],pin(protocol_path)),'Completed v2 actual predictions, never v1 failure')
    sources=read(run/'source_inventory.json');outputs=read(run/'output_pins.json')
    for q in [*sources.values(),*outputs.values(),*r['outputs'].values()]:checked(q)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:zipok=set(z.namelist())==set(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in sources.items())
    gate('authorized_v2_completed_source_ZIP_outputs_protocol_identity',zipok and read(run/'protocol.json')==p and r['input_pins']==i and (not 'output' in p or run.samefile(p['output'])) and same(desc['producer'],pin(run/'report.json')) and same(desc['source'],pin(STATS)))
    d=load(run/'prepared_inputs_and_context.npz');original=load(checked(i['target_arrays']));ids=read(run/'group152_identities.json');expected=schedule();require(len(ids)==152 and all(all(v[k]==q[k] for k in q) for v,q in zip(ids,expected)),'All152 source identity schedule')
    require((run/'group152_identities.json').read_bytes()==checked(i['target_identities']).read_bytes(),'Unmodified source identity JSON')
    fields={'x58_FP32':(152,58),'u20_FP32':(152,20),'target58_FP32':(152,58),'physical_xi46_FP32':(46,),'palm_rot_raw_FP32':(152,3,3),'gravity_palm_F64':(152,3),'DeltaT_F64':(152,),'mass_actual_F64':(),'I_body_F64':(3,3),'J_palm_F64':(152,2,3),'K_palm_F64':(152,2,3),'r_v_saved_m_s_F64':(152,3),'r_w_saved_rad_s_F64':(152,3),'boundary_constraint_witness':(152,2),'source_group_id152':(152,),'source_substep_rows152x4':(152,4),'effective_s52_FP32':(52,)}
    require(set(d)==set(fields),'Exact data ABI')
    for k,shape in fields.items():typed(d[k],shape,'?' if k=='boundary_constraint_witness' else 'i8' if k.startswith('source_') else 'f4' if k.endswith('FP32') else 'f8',k)
    gate('all152_source_inputs_targets_impulses_witness_group_substeps_bytes',all(bits(v,original[k]) for k,v in d.items() if k!='effective_s52_FP32') and bits(d['source_group_id152'],np.asarray([x['group_id'] for x in expected],dtype='i8')) and bits(d['source_substep_rows152x4'],4*d['source_group_id152'][:,None]+np.arange(4,dtype='i8')))
    ctx=load(run/'physical_context_buffers.npz');before=load(run/'normalizer_buffers_before.npz');after=load(run/'normalizer_buffers_after.npz');receipt=read(checked(i['initial_validation']));hosts=r['normalizer_host_configuration_before'];normok=set(before)==set(after) and all(bits(v,after[k]) for k,v in before.items())
    for role in ROLES:
        one={k.split('__',1)[1]:v for k,v in before.items() if k.startswith(role+'__')};normok &= A.tensor_digest_arrays(one)==receipt['frozen_norm_sha256'] and hosts[role]==receipt['normalizer_host_configuration']
    oldrun=checked(i['original_initial_component']).parent;prior=[load(checked({'path':str(oldrun/(arm+'_iteration_0001_preupdate_loss_evidence.npz')),'sha256':'b51f83d56db7b5358eaccef062fb0f5b1972fb98bcc58ab58ee43b34c79e2f2b'}))['scale52_FP32'] for arm in ('pointwise','pointwise_plus_common_pre_response')]
    require(bits(d['effective_s52_FP32'],prior[0]) and bits(prior[0],prior[1]) and (d['effective_s52_FP32']>0).all(),'Actual captured initial scale exact to prior initial-bound bytes, not NumPy sqrt')
    expected_ctx={'mass_actual_FP32':d['mass_actual_F64'].astype('f4'),'I_body_F64':d['I_body_F64'],'gravity_palm_F64':d['gravity_palm_F64'][0],'DeltaT_F64':d['DeltaT_F64'][0],'palm_rot_FP32':d['palm_rot_raw_FP32'][0],'effective_s52_FP32':d['effective_s52_FP32'],'physical_xi46_FP32':d['physical_xi46_FP32'],'I_body_antisymmetry_maxabs_F64':np.asarray(abs(d['I_body_F64']-d['I_body_F64'].T).max())}
    gate('all_roles_raw_norm_before_after_host_context_scale_same_initial',normok and hosts==r['normalizer_host_configuration_after'] and set(ctx)==set(expected_ctx) and all(bits(v,ctx[k]) for k,v in expected_ctx.items()) and all(bits(d[k],np.broadcast_to(d[k][:1],d[k].shape).copy()) for k in ('gravity_palm_F64','DeltaT_F64','palm_rot_raw_FP32')))
    fs={role:load(run/(role+'_forward152.npz')) for role in ROLES};ls={role:load(run/(role+'_descriptive_losses.npz')) for role in ROLES};oldbase=load(checked(i['initial_saved_baseline']));math_ok=True;cap_ok=True;loss_details={}
    for index,role in enumerate(ROLES):
        f,e=fs[role],ls[role]
        require(set(f)=={'prediction58_FP32','self15_logits_FP32','z12_FP32','mode_logits2_FP32'},'Exact FW fields')
        for k,shape in [('prediction58_FP32',(152,58)),('self15_logits_FP32',(152,15)),('z12_FP32',(152,2,6)),('mode_logits2_FP32',(152,2))]:typed(f[k],shape,'f4',k)
        D=e['native_point_delta52_FP32'];typed(D,(152,52),'f4','native52 capture')
        for lo,hi,slo,shi in ((0,40,0,40),(40,43,40,43),(46,49,52,55),(49,52,55,58)):math_ok &= bits(D[:,lo:hi],d['target58_FP32'][:,slo:shi]-f['prediction58_FP32'][:,slo:shi])
        chart=np.sqrt(np.sum(D[:,43:46].astype('f8')**2,axis=1));math_ok &= (chart<math.pi-.01).all() and A.close(chart.max(),r['summary_common_scalars'][role]['native_chart']['native_log_rotvec_norm_max_rad'])
        ref=M.auxiliary(d['x58_FP32'],f['prediction58_FP32'],f['z12_FP32'],d['effective_s52_FP32'],float(ctx['mass_actual_FP32']),d['I_body_F64'],d['gravity_palm_F64'],d['DeltaT_F64'],d['J_palm_F64'],d['K_palm_F64'],d['r_v_saved_m_s_F64'],d['r_w_saved_rad_s_F64'])
        errs={k:float(np.max(abs(np.asarray(e[k])-v))) for k,v in ref.items()};math_ok &= all(A.close(e[k],v) for k,v in ref.items())
        point=e['native_point_squared_scaled52_FP32'];typed(point,(152,52),'f4','native point elements');normalized=(D.astype('f8')/d['effective_s52_FP32'].astype('f8'))**2;u=2**-24
        cap_ok &= np.all(abs(point.astype('f8')-normalized)<=8*u*np.maximum(normalized,np.finfo('f4').tiny))
        point_mean=M.native_fp32_reduction_evidence(e['L_state_FP32'],point);bce=M.bce_elements(f['mode_logits2_FP32'],d['boundary_constraint_witness']);be=e['mode_BCE_elements_FP32'];typed(be,(152,2),'f4','BCE elements')
        cap_ok &= np.all(abs(be.astype('f8')-bce)<=32*u*(1+abs(f['mode_logits2_FP32'].astype('f8'))));mode_mean=M.native_fp32_reduction_evidence(e['L_mode_FP32'],be)
        for key,logits,shape in [('mode_probability2_FP32',f['mode_logits2_FP32'],(152,2)),('self15_probability15_FP32',f['self15_logits_FP32'],(152,15))]:
            typed(e[key],shape,'f4',key);cap_ok &= ((e[key]>=0)&(e[key]<=1)).all() and np.all(abs(e[key].astype('f8')-M.probability(logits))<=8*u)
        rate=e['self15_penalty_rate_FP32'];component=e['self15_penalty_component_FP32'];typed(rate,(152,),'f4','self penalty rate');typed(component,(152,),'f4','self penalty component')
        cap_ok &= np.all(abs(rate.astype('f8')+e['self15_probability15_FP32'].astype('f8').sum(1))<=18*u*15) and bits(component,rate*np.float32(.02)) and point_mean['within_envelope'] and mode_mean['within_envelope']
        public,total=A.public_order(e['L_state_FP32'],e['L_mode_FP32'],float(ref['L_imp_F64']),float(ref['L_cons_F64']),float(role==ROLES[2]));math_ok &= A.close(e['L_public'],public) and A.close(e['L_total'],total)
        q=r['summary_common_scalars'][role];math_ok &= q['state_point_FP32']==float(e['L_state_FP32']) and q['mode_BCE_FP32']==float(e['L_mode_FP32']) and A.close(q['impulse_z_MSE_F64'],ref['L_imp_F64']) and A.close(q['saved_defect_consistency_MSE_F64'],ref['L_cons_F64']) and A.close(q['shared_public_loss'],public)
        loss_details[role]={'maximum_absolute_auxiliary_differences':errs,'native_point_reduction':point_mean,'native_mode_reduction':mode_mean,'impulse_z_MSE_F64':float(ref['L_imp_F64']),'consistency_MSE_F64':float(ref['L_cons_F64']),'shared_public_loss':public,'native_Log_scope':'Captured native output; only nonrotational bits and recorded chart-vector norm rechecked, no native Log replay/SO3 truth.'}
    gate('all3x152_nonrot_native_sign_and_independent_JK_RHS_pivot_defect_losses',math_ok);gate('captured_FP32_elements_probabilities_BCE_proxy_and_reduction_envelopes',cap_ok)
    gate('initial_native_outputs_old_baseline_exactbits_aux_positivezero_algebra',bits(fs[ROLES[0]]['prediction58_FP32'],oldbase['prediction58_FP32']) and bits(fs[ROLES[0]]['self15_logits_FP32'],oldbase['self15_logits_FP32']) and A.positive_zero(fs[ROLES[0]]['z12_FP32']) and A.positive_zero(fs[ROLES[0]]['mode_logits2_FP32']))
    rng=A.U.raw_rng(run);gate('raw_three_RNG_values_headers_vectors_equal',all(rng.values()) and all(v is True for v in r['raw_rng_comparison'].values()))
    storage=read(run/'named_storage_inventory.json');sets=[];storage_ok=set(storage)==set(ROLES)
    for role in ROLES:
        addresses=set()
        for v in storage[role]:
            size={'torch.float32':4,'torch.float64':8}[v['dtype']];storage_ok &= v['device']=='cpu' and v['data_ptr']==v['storage_base_ptr']+v['storage_offset']*size
            if v['storage_nbytes']>0:addresses.add(v['storage_base_ptr'])
        sets.append(addresses)
    gate('named_storage_disjoint_complete_source_budget_and_digest_assertions',storage_ok and all(not(sets[a]&sets[b]) for a in range(3) for b in range(a+1,3)) and r['counts']=={'own_trusted_torch_loads':3,'native_WM_constructors':1,'raw_deepcopies':2,'native_FW_attempted':3,'native_FW_returned':3,'aux_head_FW':2,'predicted_rows':456,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0} and r['state_digests_before']==r['state_digests_after'] and all(v is True for v in r['checks'].values()))
    masks=cohort_masks(expected);require([int(m.sum()) for _,m in masks]==[152,76,76]+[19]*8+[24,128],'Exact13 cohorts');all_stats={role:{name:summary(fs[role],ls[role],d,fs[ROLES[0]]['self15_logits_FP32'].astype('f8'),mask) for name,mask in masks} for role in ROLES}
    gate('all39_cohort_full_units_mode_z_JK_defect_self_drift_statistics',equivalent(desc['statistics'],all_stats) and desc['cohorts']==[x for x,_ in masks] and desc['roles']==list(ROLES))
    comparison={}
    for a,b in ((ROLES[1],ROLES[0]),(ROLES[2],ROLES[0]),(ROLES[2],ROLES[1])):
        comparison[a+'_minus_'+b]={}
        for name in all_stats[a]:
            x,y=all_stats[a][name],all_stats[b][name];comparison[a+'_minus_'+b][name]={k:x[k]-y[k] for k in ('native52_scaled_MSE_F64_of_saved_FP32_delta','mode_BCE_F64_of_saved_FP32_logits','impulse_z_MSE_F64','saved_defect_consistency_MSE_F64')};comparison[a+'_minus_'+b][name]['captured_FP32_mean_self15_penalty_component_difference']=x['self15_initial_drift']['captured_FP32_mean_penalty_component_dt002']-y['self15_initial_drift']['captured_FP32_mean_penalty_component_dt002']
    csv_ok=True;cohort_rows=list(csv.DictReader((description/'all39_model_cohort_summary.csv').open(encoding='utf-8-sig')));require(len(cohort_rows)==39,'All39 CSV rows')
    for row,(role,(name,_)) in zip(cohort_rows,[(r,g) for r in ROLES for g in masks]):
        q=all_stats[role][name];csv_ok &= row['role']==role and row['cohort']==name and int(row['groups'])==q['groups']
        for k in ('native52_scaled_MSE_F64_of_saved_FP32_delta','mode_BCE_F64_of_saved_FP32_logits','impulse_z_MSE_F64','saved_defect_consistency_MSE_F64'):csv_ok &= equivalent(float(row[k]),q[k])
        for block in BLOCKS:csv_ok &= equivalent(float(row[block+'_vector_RMS']),q['state_native_delta'][block]['vector_RMS'])
        csv_ok &= equivalent(float(row['self15_logit_component_RMSE_vs_initial']),q['self15_initial_drift']['logit']['component_RMSE']) and equivalent(float(row['self15_mean_sum_sigmoid']),q['self15_initial_drift']['mean_sum_sigmoid'])
    rows=list(csv.DictReader((description/'all456_model_row_metrics.csv').open(encoding='utf-8-sig')));require(len(rows)==456,'All456 CSV rows')
    for n,row in enumerate(rows):
        role=ROLES[n//152];j=n%152;f,e=fs[role],ls[role];s=d['effective_s52_FP32'].astype('f8');D=e['native_point_delta52_FP32'][j].astype('f8');v=(e['r_v_hat_F64'][j]-d['r_v_saved_m_s_F64'][j])/s[46:49];w=(e['r_w_hat_F64'][j]-d['r_w_saved_rad_s_F64'][j])/s[49:52]
        csv_ok &= row['role']==role and int(row['row_id'])==j and int(row['group_id'])==expected[j]['group_id'] and row['path']==expected[j]['path'] and int(row['k'])==expected[j]['k'] and row['mode']==expected[j]['mode']
        target={'native52_scaled_MSE_F64':M.fmean((D/s)**2),'mode_BCE_F64':M.fmean(M.bce_elements(f['mode_logits2_FP32'][j],d['boundary_constraint_witness'][j])),'impulse_z_MSE_F64':M.fmean((f['z12_FP32'][j].astype('f8')-e['z_target_F64'][j])**2),'consistency_MSE_F64':M.fmean(np.r_[v,w]**2),'captured_FP32_self15_penalty_component':float(e['self15_penalty_component_FP32'][j]),'self15_logit_vector_drift':math.sqrt(M.fmean((f['self15_logits_FP32'][j].astype('f8')-fs[ROLES[0]]['self15_logits_FP32'][j])**2)*15)}
        for block,(lo,hi,_) in BLOCKS.items():target[block+'_native_delta_vector_norm']=math.sqrt(math.fsum(float(v*v) for v in D[lo:hi]))
        for c,name in enumerate(('HO','nonhand')):
            target[name+'_mode_logit']=float(f['mode_logits2_FP32'][j,c]);target[name+'_mode_witness']=int(d['boundary_constraint_witness'][j,c]);target[name+'_predicted_bit_strict_positive']=int(f['mode_logits2_FP32'][j,c]>0)
            for kind,key,truth in [('J','J_hat_palm_F64','J_palm_F64'),('K','K_hat_palm_F64','K_palm_F64')]:target[name+'_'+kind+'_error_vector_norm']=math.sqrt(math.fsum(float(x*x) for x in e[key][j,c]-d[truth][j,c]))
        csv_ok &= all(equivalent(float(row[k]),float(v)) for k,v in target.items())
    gate('all39_and456_CSV_identities_values_three_pair_differences_captured_scalars',csv_ok and equivalent(desc['signed_metric_differences_not_a_composite_winner'],comparison) and all(all(float(ls[role][k])==v for k,v in vals.items()) for role,vals in desc['captured_global_losses'].items()))
    for v in [*sources.values(),*outputs.values(),*desc['outputs'].values()]:checked(v)
    gate('no_Torch_checkpoint_nativeLog_FW_or_physical_qualification_upgrade',not any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules) and pin(__file__)==auditor and all(r[k] is False for k in ('physical_force_or_gradient_qualified','whole_hidden_state_qualified','Actor_or_task_benefit_qualified')))
    dump(out/'independent_statistics39.json',all_stats);dump(out/'loss_arithmetic_details.json',loss_details)
    result={'schema':'full151105_boundary_component_final_prediction_saved_independent:1','status':'PASS_SAVED_ALL3_ALL152_FIXED_FINAL_PREDICTIONS' if all(checks.values()) else 'FAILED_SAVED_FINAL_PREDICTION_REVIEW','engineering_qualified':all(checks.values()),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'auditor':auditor,'protocol':pin(protocol_path),'producer_report':pin(run/'report.json'),'description_report':pin(description/'report.json'),'run_dir':str(run),'source_files':len(sources),'counts':r['counts'],'raw_RNG':rng,'independent_statistics':pin(out/'independent_statistics39.json'),'loss_details':pin(out/'loss_arithmetic_details.json'),'scope':['Captured native Log is source evidence, not independently replayed or SO3 truth.','Weights/caches/call counts and actual forward argument identity remain producer source evidence; no checkpoint/model reload. Raw norm/context/RNG bytes are directly checked.','FP32 element/reduction bounds are arithmetic consistency envelopes under FP32 elementary-rounding assumptions, not a formal library-kernel or scientific accuracy certificate.','Known dependent152 training rows; no new confirmation, causality, physical Jacobian, force, hidden-state, Actor or ego qualification.'],'reviewer_Torch_pt_models_FW_AD_MJB_Env_physics_GPU':0,'accuracy_physical_gradient_wholehidden_Actor_or_generalization_qualified':False}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total']},indent=2));return 0 if all(checks.values()) else 2

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run-dir',type=Path);ap.add_argument('--description-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();metadata(a.protocol)
    if not a.run:print(json.dumps({'status':'PASS_SOURCE_METADATA_ONLY_NO_ACTUAL_ARRAYS','auditor':pin(__file__),'math':pin(MATH)}));return 0
    require(all(v is not None for v in (a.run_dir,a.description_dir,a.output)),'Actual saved paths required')
    try:return audit(a.protocol,a.run_dir.resolve(strict=True),a.description_dir.resolve(strict=True),a.output.resolve())
    except Exception as e:
        if a.output.exists() and not(a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'no_automatic_retry':True,'source_and_producer_unchanged':True})
        raise
if __name__=='__main__':raise SystemExit(main())
