"""Independent public15 final saved arithmetic; no Torch, pt, Log or NN replay."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def module(path,expected):
    require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Frozen independent helper')
    s=importlib.util.spec_from_file_location('public15_independent_'+path.stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def require(value,message):
    if not value:raise ValueError(message)

C=module(ROOT/'scripts/audit_boundary_cached_self15_saved_cpu_v1.py','b103b8191d7fe5fad5ebedb9971d67a5abd0600babd0fe25a3ac5b925793ae9b')
V=C.V;M=V.M;A=V.A
pin=V.pin;checked=V.checked;same=V.same;read=V.read;dump=V.dump;load=V.load;bits=V.bits;typed=V.typed
ROLES=V.ROLES;BLOCKS=V.BLOCKS
CORE=ROOT/'src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py'
CORE_SHA='81ad78161c75b1720a77aead414ccb6c5a447b0f614e5e597a940aaa1ea77419'
STATS=ROOT/'scripts/describe_public_self15_predictions_saved_cpu_v1.py'
STATS_SHA='fb4465bcf12ae98db966420ce24b25961673f523353119ffa4ec9a91843cc2c8'
FIELDS={'state':'L_state_FP32','self15':'L_self15_sum_FP32','native_pair':'L_native_pair_FP32','mode':'L_mode_FP32','impulse':'L_imp_F64','consistency':'L_cons_F64','public':'L_public','total':'L_total'}
INPUTS={'target_report','target_arrays','target_identities','target_saved8','target_source_inventory','original_initial_component','initial_validation','initial_saved_parity10','initial_saved_baseline','initial_parity_report','training_protocol','training_result','final_pair_manifest','final_saved_health','cached_sensor_labels','cached_sensor_report','cached_sensor_saved7'}

def reduction_rows(captured,elements):
    require(captured.dtype==elements.dtype==np.float32 and captured.shape==(len(elements),),'Saved FP32 row reductions')
    rows=[M.native_fp32_reduction_evidence(captured[j],elements[j]) for j in range(len(elements))]
    return all(q['within_envelope'] for q in rows),max(abs(q['difference']) for q in rows)

def loss_evidence(f,e,d,context):
    """All quantities derive from saved operands and scalar/pivot references."""
    n=len(f['prediction58_FP32']);scale=d['effective_s52_FP32'];D=e['native_D52_FP32'];u=2**-24;eps=2**-23
    typed(D,(n,52),'f4','D52');ok=bits(D,e['native_point_delta52_FP32']) and bits(e['effective_s52_FP32'],scale)
    for lo,hi,slo,shi in ((0,40,0,40),(40,43,40,43),(46,49,52,55),(49,52,55,58)):
        ok &= bits(D[:,lo:hi],d['target58_FP32'][:,slo:shi]-f['prediction58_FP32'][:,slo:shi])
    chart=np.asarray([math.sqrt(math.fsum(float(v)*float(v) for v in row)) for row in D[:,43:46]])
    ok &= bool((chart<math.pi-.01).all())
    ref=M.auxiliary(d['x58_FP32'],f['prediction58_FP32'],f['z12_FP32'],scale,float(context['mass_actual_FP32']),d['I_body_F64'],d['gravity_palm_F64'],d['DeltaT_F64'],d['J_palm_F64'],d['K_palm_F64'],d['r_v_saved_m_s_F64'],d['r_w_saved_rad_s_F64'])
    auxiliary_max={k:float(np.max(abs(np.asarray(e[k])-v))) for k,v in ref.items()}
    ok &= all(A.close(e[k],v) for k,v in ref.items())
    squared=e['point_squared52_FP32'];point=e['point_per_row_FP32'];sb=e['self15_BCE_elements_FP32'];ss=e['self15_sum_per_row_FP32'];pair=e['native_pair_per_row_FP32'];mb=e['mode_BCE_elements_FP32']
    for name,value,shape in [('squared52',squared,(n,52)),('pointrow',point,(n,)),('self15BCE',sb,(n,15)),('selfsum',ss,(n,)),('pairrow',pair,(n,)),('modeBCE',mb,(n,2))]:typed(value,shape,'f4',name)
    require(bits(squared,e['native_point_squared_scaled52_FP32']) and bits(e['self15_logits_FP32'],f['self15_logits_FP32']) and bits(e['self15_target_FP32'],d['cached_self15_target_FP32']),'Actual capture aliases and targets')
    normalized=(D.astype('f8')/scale.astype('f8'))**2
    cap=bool(np.all(abs(squared.astype('f8')-normalized)<=8*u*np.maximum(normalized,np.finfo('f4').tiny)))
    pr,maxpr=reduction_rows(point,squared)
    # sum15 is compared to a mean of the same nonnegative operands times 15;
    # use the unscaled conditional gamma envelope, never cross-stride BCE bits.
    selfsum=[]
    for j in range(n):
        total=math.fsum(float(v) for v in sb[j]);bound=(18*u/(1-18*u))*abs(total)+18*float(np.nextafter(np.float32(0),np.float32(1)))
        selfsum.append(abs(float(ss[j])-total)<=bound)
    cap &= pr and all(selfsum) and bits(pair,point+np.float32(.1)*ss)
    bce_evidence={}
    for name,logits,truth,actual in [('self15',f['self15_logits_FP32'],d['cached_self15_target_FP32'].astype('?'),sb),('mode',f['mode_logits2_FP32'],d['boundary_constraint_witness'],mb)]:
        analytical=M.bce_elements(logits,truth);bound=16*eps*(1+abs(logits.astype('f8'))+abs(analytical))
        cap &= bool(np.all(abs(actual.astype('f8')-analytical)<=bound))
        bce_evidence[name]={'maxabs':float(abs(actual.astype('f8')-analytical).max()),'maximum_fraction_of_16eps_input_envelope':float((abs(actual.astype('f8')-analytical)/bound).max()),'kernel_bit_replayed':False}
    reductions={k:M.native_fp32_reduction_evidence(e[k],v) for k,v in [('L_state_FP32',point),('L_self15_sum_FP32',ss),('L_native_pair_FP32',pair),('L_mode_FP32',mb)]}
    cap &= all(q['within_envelope'] for q in reductions.values())
    for key,logits,shape in [('mode_probability2_FP32',f['mode_logits2_FP32'],(n,2)),('self15_probability15_FP32',f['self15_logits_FP32'],(n,15))]:
        typed(e[key],shape,'f4',key);cap &= bool(((e[key]>=0)&(e[key]<=1)).all() and np.all(abs(e[key].astype('f8')-M.probability(logits))<=8*u))
    rate=e['self15_penalty_rate_FP32'];proxy=e['self15_penalty_component_FP32'];typed(rate,(n,),'f4','rate');typed(proxy,(n,),'f4','proxy')
    cap &= bool(np.all(abs(rate.astype('f8')+e['self15_probability15_FP32'].astype('f8').sum(1))<=18*u*15)) and bits(proxy,rate*np.float32(.02))
    return bool(ok),bool(cap),ref,{'auxiliary_maximum_absolute_differences':auxiliary_max,'point_row_max_difference':maxpr,'reductions':reductions,'BCE':bce_evidence,'captured_chart_max_rad':float(chart.max())}

def add_sensor(common,score,mask,loss):
    s=C.summary(score,np.flatnonzero(mask).tolist());q=s['confusion_total'];f=s['confusion_per_feature']
    ratio=lambda a,b:a/b if b else None
    s.update(positive_recall=ratio(q['TP'],q['TP']+q['FN']),positive_precision=ratio(q['TP'],q['TP']+q['FP']),negative_specificity=ratio(q['TN'],q['TN']+q['FP']),
        per_feature_positive_recall=[ratio(t,t+n) for t,n in zip(f['TP'],f['FN'])],per_feature_positive_precision=[ratio(t,t+n) for t,n in zip(f['TP'],f['FP'])])
    common['cached_self15']=s;common['native_pair_F64_rescore']=common['native52_scaled_MSE_F64_of_saved_FP32_delta']+.1*s['BCE_sum15_mean_rows_F64']
    common['full_public_F64_rescore']=common['native_pair_F64_rescore']+common['mode_BCE_F64_of_saved_FP32_logits']+common['impulse_z_MSE_F64']
    common['captured_native_pair_row_FP32_mean_F64']=M.fmean(loss['native_pair_per_row_FP32'][mask])
    return common

def metadata(protocol):
    require(V.sha(CORE)==CORE_SHA and V.sha(STATS)==STATS_SHA,'Final reviewed evaluator and stats sources')
    p=read(protocol);require(p['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_protocol:1' and p['role']=='fixed_initial_and_two_public_self15_final1000_component_evaluation','New public15 family')
    require(p['CPU_run_authorized'] is True and p['GPU_run_authorized'] is p['training_authorized'] is False and set(p['inputs'])==INPUTS,'Authorized CPU-only exact17 aliases')
    for v in [*p['inputs'].values(),*p['source_pins'].values()]:checked(v)
    i=p['inputs'];h=read(checked(i['final_saved_health']));A.U.allchecks(h)
    require(h['schema']=='full151105_boundary_public_self15_checkpoint_saved_review:1' and h['status']=='PASS_SAVED_PAIRED_PUBLIC_SELF15_COMPONENT_HEALTH_NOT_ACCURACY' and h['iteration']==1000 and h['final_iteration_scope'] is True,'Actual new fixed final1000 health')
    pair=read(checked(i['final_pair_manifest']));require(pair['schema']=='full151105_boundary_public_self15_matched_pair:1' and set(pair['checkpoints'])==set(ROLES[1:]) and same(h['producer_pair_manifest'],i['final_pair_manifest']),'Both new fixed finals')
    for k,v in pair['checkpoints'].items():checked(v);require(same(v,h['checkpoints'][k]),'Actual health binds same fixed file')
    train=read(checked(i['training_protocol']));require(set(train['inputs'])=={'target_report','target_arrays','target_saved8','target_source_inventory','initial_component','initial_validation','initial_saved_health29','initial_parity_report','initial_parity_saved_review','initial_parity_baseline','component_ready','component_source_review','cached_sensor_labels','cached_sensor_report','cached_sensor_saved7','target_identities'},'Exact real16 training input map')
    require(same(train['inputs']['initial_component'],i['original_initial_component']) and same(h['protocol'],i['training_protocol']) and 'input_pins' not in train,'Original own initial; true protocol.inputs map')
    ready=read(ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_preparation_CPU_v1/READY.json');proof=read(checked(ready['cpu_report']));A.U.allchecks(proof);require(proof['passed']==3,'Bound evaluator Toy3')
    source=read(ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_source_independent_v1/report.json');A.U.allchecks(source);require(source['source_review_qualified'] is True,'Current source review')
    for value in [*ready['source_pins'].values(),*source['source_pins'].values()]:checked(value)
    require(same(source['source_pins']['consumer'],pin(CORE)) and same(source['source_pins']['statistics'],pin(STATS)),'Exact source review identity')
    return p

def audit(protocol,run,description,out):
    out.mkdir(exist_ok=False);selfpin=pin(__file__);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_arrays.json',{'auditor':selfpin,'scalar_auxiliary_reference':pin(V.MATH),'old_independent_summary':pin(C.BASE),'cached_scalar_reference':pin(ROOT/'scripts/audit_boundary_cached_self15_saved_cpu_v1.py'),'actual_arrays_read':False,'F64_atol':2e-14,'F64_rtol':2e-12,'BCE_element_bound':'16eps32*(1+abs(logit)+abs(F64stableBCE)); no cross-stride bitkernel replay','pair_row':'FP32(point_row+FP32(.1)*selfsum_row); meanrows independently bounded'})
    p=metadata(protocol);i=p['inputs'];r=read(run/'report.json');desc=read(description/'report.json');checks={};detail={}
    gate=lambda key,value:checks.__setitem__(key,bool(value))
    require(r['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1' and r['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION' and same(r['protocol'],pin(protocol)),'Completed new fixed actual run')
    sources=read(run/'source_inventory.json');outputs=read(run/'output_pins.json')
    for q in [*sources.values(),*outputs.values(),*r['outputs'].values(),*desc['outputs'].values()]:checked(q)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:zipok=set(z.namelist())==set(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in sources.items())
    gate('new_family_protocol_input_maps_source_ZIP_outputs_and_description_identity',zipok and read(run/'protocol.json')==p and r['input_pins']==i and same(desc['producer'],pin(run/'report.json')) and same(desc['source'],pin(STATS)))
    d=load(run/'prepared_inputs_and_context.npz');original=load(checked(i['target_arrays']));ids=read(run/'group152_identities.json');expected=V.schedule()
    require(len(ids)==152 and all(all(v[k]==q[k] for k in q) for v,q in zip(ids,expected)) and (run/'group152_identities.json').read_bytes()==checked(i['target_identities']).read_bytes(),'Exact original152 identities, no snapshot-repeat replacement')
    fields={'x58_FP32':(152,58),'u20_FP32':(152,20),'target58_FP32':(152,58),'physical_xi46_FP32':(46,),'palm_rot_raw_FP32':(152,3,3),'gravity_palm_F64':(152,3),'DeltaT_F64':(152,),'mass_actual_F64':(),'I_body_F64':(3,3),'J_palm_F64':(152,2,3),'K_palm_F64':(152,2,3),'r_v_saved_m_s_F64':(152,3),'r_w_saved_rad_s_F64':(152,3),'boundary_constraint_witness':(152,2),'source_group_id152':(152,),'source_substep_rows152x4':(152,4),'effective_s52_FP32':(52,),'cached_self15_target_FP32':(152,15)}
    require(set(d)==set(fields),'Exact prepared original152 plus scale/cached15 ABI')
    for key,shape in fields.items():typed(d[key],shape,'?' if key=='boundary_constraint_witness' else 'i8' if key.startswith('source_') else 'f4' if key.endswith('FP32') else 'f8',key)
    gate('all152_source_state_control_xi_impulse_defect_witness_and_group_bytes',all(bits(v,original[k]) for k,v in d.items() if k not in ('effective_s52_FP32','cached_self15_target_FP32')) and bits(d['source_group_id152'],np.asarray([q['group_id'] for q in expected],dtype='i8')) and bits(d['source_substep_rows152x4'],4*d['source_group_id152'][:,None]+np.arange(4,dtype='i8')))
    labels=read(checked(i['cached_sensor_labels']));target=[]
    for g,row in zip(ids,labels):
        require(all(row[k]==g[k] for k in ('row_id','snapshot_id','path','mode','k','branch_id','group_id','physical_rows','terminal_observation')),'All cached label identities')
        require(row['label_time']=='lastsolver_preintegration~kdt+.015' and row['state_time']=='postintegration_kdt+.020' and row['source_frame']==200+g['k'] and row['valid_next_policy_decision'] is (g['k']<149) and row['no_u150'] is True,'Cached15 versus POST20 timing')
        y=C.mapped15([row['raw_found25']]);require(bits(y,np.asarray(row['contact15'],dtype='f4')),'Independent stride5 rawfound mapping');target.append(y)
    typed(d['cached_self15_target_FP32'],(152,15),'f4','All cached15');gate('all152_cached_rawfound_label_only_JOIN_and_phase',len(labels)==152 and bits(d['cached_self15_target_FP32'],np.asarray(target,dtype='f4')))
    context=load(run/'physical_context_buffers.npz');before=load(run/'normalizer_buffers_before.npz');after=load(run/'normalizer_buffers_after.npz');receipt=read(checked(i['initial_validation']));normok=set(before)==set(after) and all(bits(v,after[k]) for k,v in before.items())
    for role in ROLES:
        one={k.split('__',1)[1]:v for k,v in before.items() if k.startswith(role+'__')};normok &= A.tensor_digest_arrays(one)==receipt['frozen_norm_sha256'] and r['normalizer_host_configuration_before'][role]==receipt['normalizer_host_configuration']
    oldrun=checked(i['original_initial_component']).parent;scales=[load(checked({'path':str(oldrun/(arm+'_iteration_0001_preupdate_loss_evidence.npz')),'sha256':'b51f83d56db7b5358eaccef062fb0f5b1972fb98bcc58ab58ee43b34c79e2f2b'}))['scale52_FP32'] for arm in ('pointwise','pointwise_plus_common_pre_response')]
    require(bits(d['effective_s52_FP32'],scales[0]) and bits(scales[0],scales[1]),'Old actual captured scale52 bytes, no NumPy sqrt substitution')
    ctxref={'mass_actual_FP32':d['mass_actual_F64'].astype('f4'),'I_body_F64':d['I_body_F64'],'gravity_palm_F64':d['gravity_palm_F64'][0],'DeltaT_F64':d['DeltaT_F64'][0],'palm_rot_FP32':d['palm_rot_raw_FP32'][0],'effective_s52_FP32':d['effective_s52_FP32'],'physical_xi46_FP32':d['physical_xi46_FP32'],'I_body_antisymmetry_maxabs_F64':np.asarray(abs(d['I_body_F64']-d['I_body_F64'].T).max())}
    gate('all_roles_raw_norm_context_host_and_captured_scale_same_initial',normok and r['normalizer_host_configuration_before']==r['normalizer_host_configuration_after'] and set(context)==set(ctxref) and all(bits(v,context[k]) for k,v in ctxref.items()))
    fs={role:load(run/(role+'_forward152.npz')) for role in ROLES};ls={role:load(run/(role+'_descriptive_losses.npz')) for role in ROLES};arith=True;capture=True
    for role in ROLES:
        f,e=fs[role],ls[role]
        require(set(f)=={'prediction58_FP32','self15_logits_FP32','z12_FP32','mode_logits2_FP32'},'Exact four forward captures')
        for k,shape in [('prediction58_FP32',(152,58)),('self15_logits_FP32',(152,15)),('z12_FP32',(152,2,6)),('mode_logits2_FP32',(152,2))]:typed(f[k],shape,'f4',k)
        ok,cap,ref,diag=loss_evidence(f,e,d,context);arith &= ok;capture &= cap
        public,total=A.public_order(e['L_native_pair_FP32'],e['L_mode_FP32'],float(ref['L_imp_F64']),float(ref['L_cons_F64']),float(role==ROLES[2]))
        arith &= A.close(e['L_public'],public) and A.close(e['L_total'],total)
        arith &= all(A.close(r['summary_common_scalars'][role][name],e[key]) for name,key in FIELDS.items()) and A.close(r['summary_common_scalars'][role]['native_chart']['native_log_rotvec_norm_max_rad'],diag['captured_chart_max_rad'])
        detail[role]={**diag,'public_FP32_pair_plus_mode_then_F64_impulse':public,'total':total}
    gate('all3x152_nonrot_native_D52_JK_pivot_solve_saved_defect_and_public_order',arith)
    gate('saved52_self15_mode_FP32_elements_rows_reductions_probabilities_proxy',capture)
    baseline=load(checked(i['initial_saved_baseline']));gate('raw_initial_58_self15_bits_and_positivezero_aux_algebra',bits(fs[ROLES[0]]['prediction58_FP32'],baseline['prediction58_FP32']) and bits(fs[ROLES[0]]['self15_logits_FP32'],baseline['self15_logits_FP32']) and A.positive_zero(fs[ROLES[0]]['z12_FP32']) and A.positive_zero(fs[ROLES[0]]['mode_logits2_FP32']))
    rng=A.U.raw_rng(run);gate('raw_RNG_headers_vectors_values_unchanged',all(rng.values()) and all(v is True for v in r['raw_rng_comparison'].values()))
    storage=read(run/'named_storage_inventory.json');sets=[];storageok=set(storage)==set(ROLES)
    for role in ROLES:
        addresses=set()
        for v in storage[role]:
            size={'torch.float32':4,'torch.float64':8}[v['dtype']];storageok &= v['device']=='cpu' and v['data_ptr']==v['storage_base_ptr']+v['storage_offset']*size
            if v['storage_nbytes']>0:addresses.add(v['storage_base_ptr'])
        sets.append(addresses)
    gate('named_storage_disjoint_counts_and_source_digest_assertions',storageok and all(not(sets[a]&sets[b]) for a in range(3) for b in range(a+1,3)) and r['counts']=={'own_trusted_torch_loads':3,'native_WM_constructors':1,'raw_deepcopies':2,'native_FW_attempted':3,'native_FW_returned':3,'aux_head_FW':2,'predicted_rows':456,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0} and r['state_digests_before']==r['state_digests_after'] and all(v is True for v in r['checks'].values()))
    masks=V.cohort_masks(expected);require([int(m.sum()) for _,m in masks]==[152,76,76]+[19]*8+[24,128],'All13 dependent cohorts');scores={role:C.score(fs[role]['self15_logits_FP32'],d['cached_self15_target_FP32']) for role in ROLES}
    statistics={role:{name:add_sensor(V.summary(fs[role],ls[role],d,fs[ROLES[0]]['self15_logits_FP32'].astype('f8'),mask),scores[role],mask,ls[role]) for name,mask in masks} for role in ROLES}
    gate('all39_stats_native52_raw58_mode_z_JK_defects_self15_perfeature_and_rescores',V.equivalent(desc['statistics'],statistics) and desc['roles']==list(ROLES) and desc['cohorts']==[k for k,_ in masks])
    csvok=True;cohorts=list(csv.DictReader((description/'all39_public15_model_cohort_summary.csv').open(encoding='utf-8-sig')));rows=list(csv.DictReader((description/'all456_public15_model_row_metrics.csv').open(encoding='utf-8-sig')));require(len(cohorts)==39 and len(rows)==456,'Complete CSV cardinalities')
    keys=('native52_scaled_MSE_F64_of_saved_FP32_delta','mode_BCE_F64_of_saved_FP32_logits','impulse_z_MSE_F64','saved_defect_consistency_MSE_F64','native_pair_F64_rescore','full_public_F64_rescore')
    for row,(role,(name,_)) in zip(cohorts,[(r,g) for r in ROLES for g in masks]):
        q=statistics[role][name];csvok &= row['role']==role and row['cohort']==name and int(row['groups'])==q['groups'] and all(V.equivalent(float(row[k]),q[k]) for k in keys)
        for block in BLOCKS:csvok &= V.equivalent(float(row[block+'_vector_RMS']),q['state_native_delta'][block]['vector_RMS'])
        for col,key in [('cached15_BCE_mean_F64','BCE_mean_entry_F64'),('cached15_Brier_F64','Brier_mean_entry_F64'),('cached15_positive_recall','positive_recall'),('cached15_pred_intensity','pred_intensity_mean'),('cached15_true_intensity','target_intensity_mean')]:csvok &= V.equivalent(None if row[col]=='' else float(row[col]),q['cached_self15'][key])
        csvok &= V.equivalent(float(row['Pen_proxy_error_mean_F64']),q['cached_self15']['Pen_proxy_penalty_error_F64']['mean'])
    for n,row in enumerate(rows):
        role=ROLES[n//152];j=n%152;f,e=fs[role],ls[role];s=d['effective_s52_FP32'].astype('f8');D=e['native_point_delta52_FP32'][j].astype('f8');ss=C.summary(scores[role],[j]);sv=(e['r_v_hat_F64'][j]-d['r_v_saved_m_s_F64'][j])/s[46:49];sw=(e['r_w_hat_F64'][j]-d['r_w_saved_rad_s_F64'][j])/s[49:52]
        csvok &= row['role']==role and int(row['row_id'])==j and int(row['group_id'])==expected[j]['group_id'] and row['path']==expected[j]['path'] and int(row['k'])==expected[j]['k'] and row['mode']==expected[j]['mode']
        values={'native52_scaled_MSE_F64':M.fmean((D/s)**2),'mode_BCE_F64':M.fmean(M.bce_elements(f['mode_logits2_FP32'][j],d['boundary_constraint_witness'][j])),'impulse_z_MSE_F64':M.fmean((f['z12_FP32'][j].astype('f8')-e['z_target_F64'][j])**2),'consistency_MSE_F64':M.fmean(np.r_[sv,sw]**2),'captured_FP32_self15_penalty_component':float(e['self15_penalty_component_FP32'][j]),'self15_logit_vector_drift':math.sqrt(math.fsum(float(x*x) for x in (f['self15_logits_FP32'][j].astype('f8')-fs[ROLES[0]]['self15_logits_FP32'][j]))),'cached15_BCE_mean_F64':ss['BCE_mean_entry_F64'],'cached15_Brier_F64':ss['Brier_mean_entry_F64'],'cached15_pred_intensity':ss['pred_intensity_mean'],'cached15_true_intensity':ss['target_intensity_mean'],**{'cached15_'+k:ss['confusion_total'][k] for k in ('TP','FP','FN','TN')},'captured_native_pair_row_FP32':float(e['native_pair_per_row_FP32'][j])}
        values['native_pair_F64_rescore']=values['native52_scaled_MSE_F64']+.1*ss['BCE_sum15_mean_rows_F64'];values['full_public_F64_rescore']=values['native_pair_F64_rescore']+values['mode_BCE_F64']+values['impulse_z_MSE_F64']
        for block,(lo,hi,_) in BLOCKS.items():values[block+'_native_delta_vector_norm']=math.sqrt(math.fsum(float(v*v) for v in D[lo:hi]))
        for c,name in enumerate(('HO','nonhand')):
            values[name+'_mode_logit']=float(f['mode_logits2_FP32'][j,c]);values[name+'_mode_witness']=int(d['boundary_constraint_witness'][j,c]);values[name+'_predicted_bit_strict_positive']=int(f['mode_logits2_FP32'][j,c]>0)
            for kind,key,truth in [('J','J_hat_palm_F64','J_palm_F64'),('K','K_hat_palm_F64','K_palm_F64')]:values[name+'_'+kind+'_error_vector_norm']=math.sqrt(math.fsum(float(v*v) for v in e[key][j,c]-d[truth][j,c]))
        csvok &= all(V.equivalent(float(row[k]),float(v)) for k,v in values.items())
    gate('all39_and456_CSV_complete_identities_metrics_and_captured_global8',csvok and all(set(q)==set(FIELDS.values()) and all(float(ls[role][k])==v for k,v in q.items()) for role,q in desc['captured_global_losses'].items()))
    gate('no_model_Log_replay_or_scientific_qualification_upgrade',not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules) and pin(__file__)==selfpin and all(r[k] is False for k in ('physical_force_or_gradient_qualified','whole_hidden_state_qualified','Actor_or_task_benefit_qualified')))
    dump(out/'independent_statistics39.json',statistics);dump(out/'independent_loss_arithmetic.json',detail)
    result={'schema':'full151105_boundary_public_self15_final_prediction_saved_independent:1','status':'PASS_SAVED_ALL3_ALL152_PUBLIC_SELF15_FIXED_FINAL_PREDICTIONS' if all(checks.values()) else 'FAILED_SAVED_PUBLIC_SELF15_FINAL_PREDICTION_REVIEW','engineering_qualified':all(checks.values()),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'auditor':selfpin,'protocol':pin(protocol),'producer_report':pin(run/'report.json'),'description_report':pin(description/'report.json'),'run_dir':str(run),'source_files':len(sources),'counts':r['counts'],'raw_RNG':rng,'statistics':pin(out/'independent_statistics39.json'),'arithmetic':pin(out/'independent_loss_arithmetic.json'),'scope':['Native Log52 source/captured evidence only. Nonrot sign is target-pred; no Log replay or true SO3 claim.','Weights/caches/forward calls and arguments are producer/source-backed assertions. Raw saved norm/context/RNG and predictions directly rechecked; no pt reload.','FP32 elementary/reduction envelopes are arithmetic consistency diagnostics, not library bitkernel or scientific accuracy certificates.','Cached15 lastsolver~15ms differs from POST20 state. No POST20 geometry, HO/human truth, global clearance or physical reward qualification.','All152 known dependent training rows/13 overlapping cohorts; no generalization, winner, causality, task/gradient/hidden-state/Actor qualification.'],'reviewer_Torch_pt_models_Log_FW_AD_MJB_Env_physics_GPU':0,'accuracy_physical_gradient_wholehidden_Actor_or_generalization_qualified':False}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total']}));return 0 if all(checks.values()) else 2

def selfcheck(out):
    out.mkdir(exist_ok=False);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes());dump(out/'source_before_fixture.json',{'auditor':pin(__file__),'math':pin(V.MATH),'actual_arrays_weights_read':False})
    point=np.asarray([1/16,1/4],dtype='f4');bce=np.full((2,15),np.float32(math.log(2)),dtype='f4');ss=bce.sum(1,dtype='f4');pair=point+np.float32(.1)*ss
    scalars=[M.native_fp32_reduction_evidence(np.asarray(v.mean(dtype='f4'),dtype='f4'),v) for v in (point,ss,pair)]
    logits=np.zeros((2,15),dtype='f4');target=np.zeros_like(logits);target[:,:3]=1;scores=C.score(logits,target);q=C.summary(scores,[0,1]);s=add_sensor({'native52_scaled_MSE_F64_of_saved_FP32_delta':M.fmean(point),'mode_BCE_F64_of_saved_FP32_logits':math.log(2),'impulse_z_MSE_F64':.125},scores,np.ones(2,dtype='?'),{'native_pair_per_row_FP32':pair})
    public,total=A.public_order(np.asarray(pair.mean(dtype='f4')),np.asarray(np.float32(math.log(2))),.125,.25,1.)
    yes=all(v['within_envelope'] for v in scalars) and q['confusion_total']['FN']==6 and q['confusion_total']['TN']==24 and s['cached_self15']['positive_recall']==0 and s['cached_self15']['positive_precision'] is None
    yes &= abs(s['native_pair_F64_rescore']-(5/32+.1*15*math.log(2)))<1e-14 and public==float(np.float32(pair.mean(dtype='f4')+np.float32(math.log(2))))+.125 and total==public+.25
    yes &= not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules)
    result={'schema':'full151105_public_self15_saved_adapter_pure_contract:1','checks':{'nonconstant_point_rows_sum15_pair_precision_public_F32_then_F64_and_tie_denominator':bool(yes)},'passed':int(yes),'total':1,'source_pins':{'auditor':pin(__file__),'scalar_auxiliary_reference':pin(V.MATH),'prior_independent_summary':pin(C.BASE),'cached_scalar_reference':pin(ROOT/'scripts/audit_boundary_cached_self15_saved_cpu_v1.py')},'actual_arrays_pt_models_FW_AD_physics':0}
    dump(out/'report.json',result);dump(out/'READY.json',{'schema':'full151105_public_self15_final_saved_adapter_READY:1','source_pins':result['source_pins'],'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':int(yes),'total':1},'actual_run_authorized':False});print(json.dumps({'READY':pin(out/'READY.json'),'passed':int(yes),'total':1}));return 0 if yes else 2

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path);ap.add_argument('--run-dir',type=Path);ap.add_argument('--description-dir',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--self-check',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    require(not(a.self_check and a.run),'Distinct synthetic and saved modes')
    if a.self_check:return selfcheck(a.output.resolve())
    require(a.protocol is not None,'Root protocol required');metadata(a.protocol)
    if not a.run:print(json.dumps({'status':'PASS_SOURCE_METADATA_ONLY_NO_ACTUAL_ARRAYS','auditor':pin(__file__)}));return 0
    require(a.run_dir is not None and a.description_dir is not None,'Actual saved paths required')
    try:return audit(a.protocol,a.run_dir.resolve(strict=True),a.description_dir.resolve(strict=True),a.output.resolve())
    except Exception as error:
        if a.output.exists() and not(a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(error),'auditor':pin(__file__),'no_automatic_retry':True})
        raise
if __name__=='__main__':raise SystemExit(main())
