"""SOURCE DRAFT: independent new-family H8 saved audit, NumPy/stdlib only.

No producer import, checkpoint/Torch/native Log/model/FW/AD execution. Actual
--run waits for Root terminal authorization and final frozen producer ABI.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,struct,sys,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'src/full151_gradient/public_self15_H8_donor105_v1.py'
READY=ROOT/'outputs/full151105_public_self15_H8_donor_preparation_CPU_v1/READY.json'
CONTRACT=ROOT/'docs/public_self15_H8_donor_contract_v1.md'
CORE_SHA='f124aca35d8478c7b421e0721351c66a23007567ca940f5263db6a07eb7f27b0'
READY_SHA='f01b0932e5101fa597f4eab4a17e5923a3e7aa4e7f61102ec832c01e6a70d435'
CONTRACT_SHA='096e3f38b67c711905429a4a7d4480e87a2b1f478ff45369724c31ce7126cdbc'
UTIL_PATH=ROOT/'scripts/audit_component_task_gradient_saved_cpu_v1.py'
UTIL_SHA='472c72117dde51f7edd7c28d34301cef88939ebf6f5a64cb9bc6849dab42d0c5'
MATH_PATH=ROOT/'src/full151_gradient/component_H8_donor_independent_math_v1.py'
MATH_SHA='4158040c650986e8f65979debbe79648e0f2c5806f1006c289cdb99d6464ea55'
ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
ARMS=ROLES[1:]
INPUTS={'initial_component','initial_validation','training_protocol','training_result','final_pair_manifest','final_saved_health',
 'public15_eval_protocol','public15_eval_report','public15_eval_saved_review','public15_eval_prepared',
 'H1_config','H1_report','H1_matched_inputs','H1_prediction_arrays','H1_identities','H1_saved_independent25','prior_H8_contiguity',
 'donor_self15_producer_report','donor_self15_labels','donor_self15_saved_independent'}
FIT_KEYS=('fit_input_digest','fit_label_digest','fit_self15_label_digest','fit_combined_label_digest')
EXPECTED_COUNTS={'own_component_loads':3,'native_WM_constructors':1,'raw_deepcopies':2,'native_forward_attempted':24,'native_forward_returned':24,
 'aux_head_forward_attempted':16,'aux_head_forward_returned':16,'predicted_rows':6864,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0}
FIXED_WEIGHTS={ROLES[0]:'ea051225b7279fc69fbd97725eb15462faaf4b3a6706881f8fe4a05b658a14bf',
 ARMS[0]:'6cac9913bc943113b862c0b5cd92e884e904821269a5f4193a37e829324fc547',ARMS[1]:'78cd4584adc456dd63cd1bb7013e2f931802b85c1efb29462320f67bf38620bd'}
CACHED_TABLE_FILENAME='all72_cached_self15_statistics.json'
CACHED_WINDOW_FILENAME='matched_donor_windows286.npz'
CACHED_WINDOW_KEY='target_self15_286x8x15_FP32'
F64_ATOL=2e-14;F64_RTOL=2e-12
AUDITOR_PREP=ROOT/'outputs/full151105_public_self15_H8_saved_auditor_preparation_v1'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve(strict=True)),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def module(path,expected):
    require(sha(path)==expected,'Frozen independent helper source')
    spec=importlib.util.spec_from_file_location('new_public15_H8_independent_'+Path(path).stem,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

U=module(UTIL_PATH,UTIL_SHA)
checked=U.checked;same=U.same;load=U.load;bits=U.bits

def no_models():
    require(not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules),'No model/native framework imports')

def close(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(close(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if type(a) is float:return type(b) is float and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=F64_ATOL+F64_RTOL*abs(b)
    return type(a) is type(b) and a==b

def metadata(protocol_path):
    no_models();require(all(type(x) is str for x in (CORE_SHA,READY_SHA,CONTRACT_SHA,CACHED_TABLE_FILENAME,CACHED_WINDOW_FILENAME,CACHED_WINDOW_KEY)),'UNFROZEN_SOURCE_DRAFT_PENDING_FINAL_OWNER_ABI')
    require(sha(CORE)==CORE_SHA and sha(READY)==READY_SHA and sha(CONTRACT)==CONTRACT_SHA,'Fixed producer, proof and scientific contract')
    p=read(protocol_path);require(p['schema']=='full151105_public_self15_H8_donor_CPU_protocol:1' and p['role']=='three_public_self15_components_fixed_control_H8_donor_state_error','New H8 namespace')
    require(set(p['inputs'])==INPUTS and len(p['inputs'])==20 and p['model_roles']==list(ROLES),'True20 aliases/current three model roles')
    require(p['cpu_threads']==4 and p['horizon']==8 and p['batch_windows']==286 and p['starts_per_path']==143 and p['accuracy_gate'] is None
        and p['evaluate_donor_self15'] is True and p['GPU_run_authorized'] is p['training_authorized'] is p['automatic_retry'] is False,'Bounded CPU H8 with required cached labels/no threshold')
    for value in [*p['inputs'].values(),*p['source_pins'].values()]:checked(value)
    require(same(p['source_contract'],pin(CONTRACT)) and any(same(value,pin(CORE)) for value in p['source_pins'].values()),'Actual contract/core closure')
    i=p['inputs'];health=read(checked(i['final_saved_health']));pair=read(checked(i['final_pair_manifest']));U.allchecks(health)
    require(health['schema']=='full151105_boundary_public_self15_checkpoint_saved_review:1' and health['status']=='PASS_SAVED_PAIRED_PUBLIC_SELF15_COMPONENT_HEALTH_NOT_ACCURACY'
        and health['engineering_qualified'] is True and health['iteration']==1000 and health['final_iteration_scope'] is True,'Only actual new public15 final health')
    require(pair['schema']=='full151105_boundary_public_self15_matched_pair:1' and pair['iteration']==1000 and set(pair['checkpoints'])==set(ARMS)
        and same(health['producer_pair_manifest'],i['final_pair_manifest']) and same(pair['protocol'],i['training_protocol']) and same(health['protocol'],i['training_protocol']),'One fixed public15 pair/training source')
    require(i['initial_component']['sha256']==FIXED_WEIGHTS[ROLES[0]],'Original ea051 role, not an initial_pair or fake final')
    for arm in ARMS:require(same(pair['checkpoints'][arm],health['checkpoints'][arm]) and pair['checkpoints'][arm]['sha256']==FIXED_WEIGHTS[arm],'Exact new-family fixed final')
    ev=read(checked(i['public15_eval_report']));evp=read(checked(i['public15_eval_protocol']));evs=read(checked(i['public15_eval_saved_review']));U.allchecks(evs)
    require(ev['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1' and ev['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION'
        and evs['schema']=='full151105_boundary_public_self15_final_prediction_saved_independent:1' and evs['status']=='PASS_SAVED_ALL3_ALL152_PUBLIC_SELF15_FIXED_FINAL_PREDICTIONS'
        and evs['engineering_qualified'] is True and same(evs['producer_report'],i['public15_eval_report']) and same(evs['protocol'],i['public15_eval_protocol']) and same(ev['protocol'],i['public15_eval_protocol']),'Actual current native-eval saved12')
    for alias,parent in {'initial_component':'original_initial_component','initial_validation':'initial_validation','training_protocol':'training_protocol','training_result':'training_result','final_pair_manifest':'final_pair_manifest','final_saved_health':'final_saved_health'}.items():require(same(i[alias],evp['inputs'][parent]),'Current family semantic input route '+alias)
    require(same(ev['outputs']['prepared_inputs_and_context.npz'],i['public15_eval_prepared']) and checked(i['public15_eval_prepared']).samefile(checked(i['public15_eval_report']).parent/'prepared_inputs_and_context.npz'),'Original enrolled context-row packet only')
    training=read(checked(i['training_protocol']));result=read(checked(i['training_result']))
    require(training['schema']=='full151105_boundary_public_self15_matched_training_CPU_protocol:1' and result['schema']=='full151105_boundary_public_self15_matched_training_completion:1'
        and result['iteration']==1000 and result['self15_supervised'] is True and same(result['protocol'],i['training_protocol']),'Completed public15 training family')
    require(all(ev[k]==result[k] for k in FIT_KEYS),'Old152 four fit digests, independent of donor input digest')
    h1=read(checked(i['H1_report']));h1s=read(checked(i['H1_saved_independent25']));U.allchecks(h1s)
    require(h1s['schema']=='full151105_state_prediction452_saved_independent:1' and h1s['engineering_qualified'] is True and same(h1s['config'],i['H1_config'])
        and same(h1s['producer_report'],i['H1_report']) and same(h1['config'],i['H1_config']) and h1['prediction_rows']==452,'Qualified original native H1 source25')
    manifest=read(checked(i['H1_report']).parent/'output_pins.json')
    for alias,name in [('H1_matched_inputs','matched_input_arrays.npz'),('H1_prediction_arrays','prediction_arrays.npz'),('H1_identities','group_identities452.json')]:require(same(i[alias],manifest[name]) and checked(i[alias]).samefile(checked(i['H1_report']).parent/name),'H1 actual immutable artifact '+alias)
    prior=read(checked(i['prior_H8_contiguity']))
    require(prior['schema']=='full151105_H8_donor_saved_contiguity_source_feasibility:1' and prior['all_links_all58_bitexact'] is True and prior['unique_adjacent_links']==298
        and prior['all_donor_four_controls_bitexact'] is True and same(prior['input_pins']['matched_input_arrays.npz'],i['H1_matched_inputs']) and same(prior['input_pins']['group_identities452.json'],i['H1_identities']),'Same source contiguity evidence, runtime still rechecked')
    donor=read(checked(i['donor_self15_producer_report']));ds=read(checked(i['donor_self15_saved_independent']));U.allchecks(donor);U.allchecks(ds)
    require(donor['schema']=='full151105_donor300_cached_self15_saved_analysis:1' and donor['status']=='COMPLETED_ALL300_DONOR_CACHED_SENSOR_LABEL_JOIN_ONLY'
        and ds['schema']=='full151105_donor300_cached_self15_saved_independent:1' and ds['status']=='PASS_SAVED_ALL300_DONOR_RAW25_TO_NATIVE15_AND_H1_JOIN' and ds['engineering_qualified'] is True,'Separate donor300 label qualification, never old152')
    require(same(ds['producer_report'],i['donor_self15_producer_report']) and same(ds['protocol'],donor['protocol']) and same(ds['label_JSON'],i['donor_self15_labels']) and same(ds['H1_identities'],i['H1_identities'])
        and same(donor['outputs']['joined300_cached_sensor_labels.json'],i['donor_self15_labels']),'One exact donor300 source/label package')
    ready=read(READY);proof=read(checked(ready['cpu_report']));U.allchecks(proof)
    require(ready['source_pins']==proof['source_pins'] and any(same(value,pin(CORE)) for value in ready['source_pins'].values()),'Current executed focused proof/source')
    for value in ready['source_pins'].values():checked(value)
    return p,{'pair':pair,'training_result':result,'evaluation':ev}

def window_identities(mapping):
    return [{'window_id':w,'path':('A0','A2')[int(pi)],'start_k':int(k),'action_phases':list(range(int(k),int(k)+8)),
        'target_observations':list(range(int(k)+1,int(k)+9)),'end_observation':int(k)+8,'no_u150':True}
        for w,(pi,k) in enumerate(zip(mapping['window_path_index286'],mapping['window_start_k286']))]

def state_rows(stats,role):
    rows=[]
    for h in range(1,9):
        for path in ('ALL','A0','A2'):
            view=stats['per_horizon'][str(h)] if path=='ALL' else stats['by_path_per_horizon'][path][str(h)]
            for name,b in view['blocks'].items():rows.append({'horizon':h,'path':path,'block':name,'unit':b['units'],'window_count':view['overlapping_windows'],
                'component_RMSE':b['component_RMSE'],'MAE':b['mean_absolute'],'maximum_absolute':b['maximum_absolute'],
                'vector_or_Frobenius_RMS':b.get('vector_RMS',b.get('Frobenius_RMS','')),'model_role':role})
    return rows

def expanded_labels(labels,identities,mapping):
    """Independent raw25→15 and noncontiguous global-ID lookup, all donor300."""
    require(type(labels) is list and len(labels)==300,'All donor labels, not old152 broadcast')
    donors={g['group_id']:g for g in identities if g['mode']=='donor'};lookup={};ordered=[]
    middle=(1,2,6,7,11,12,16,17,21,22);distal=(3,8,13,18,23);tips=(4,9,14,19,24)
    for row in labels:
        gid=row['group_id'];require(type(gid) is int and gid in donors and gid not in lookup,'Unique native global donor ID')
        source=donors[gid]
        for key in ('group_id','path','mode','k','branch_id','source_frame','terminal_observation','first_physical_pre_index','last_physical_index','post20_GPU_row_index','valid_next_policy_decision','no_u150'):
            require(type(row.get(key)) is type(source[key]) and row[key]==source[key],'Donor label/H1 exact identity '+key)
        require(row['mode']=='donor' and row['branch_id'] is None and row['label_time']=='lastsolver_preintegration~kdt+.015' and row['state_time']=='postintegration_kdt+.020','Cached15/state20 distinct phases')
        raw=row['raw_found25'];target=row['contact15'];require(type(raw) is list and len(raw)==25 and all(type(v) in (int,float,bool) and math.isfinite(v) and v>=0 and int(v)==v and v<=np.iinfo(np.int64).max for v in raw),'Raw25 finite count ABI')
        require(type(target) is list and len(target)==15 and all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in target),'Binary native15')
        found=[v>0 for v in raw];expected=[float(found[j]) for j in middle]+[float(found[d] or found[t]) for d,t in zip(distal,tips)]
        require(all(struct.pack('<f',float(a))==struct.pack('<f',b) for a,b in zip(target,expected)),'All raw25→15 float bytes')
        lookup[gid]=np.asarray(target,np.float32);ordered.append(gid)
    wanted=mapping['donor_row_indices2x150'].reshape(-1)
    require(ordered==wanted.tolist() and set(lookup)==set(donors),'Exact all300 path/k order and identity coverage')
    window=mapping['window_transition_row_indices286x8'];index_lookup={int(gid):j for j,gid in enumerate(wanted)}
    index=np.asarray([[index_lookup[int(gid)] for gid in row] for row in window],np.int64)
    unique=np.stack([lookup[int(gid)] for gid in wanted]);y=unique[index].copy();repetition=np.bincount(index.reshape(-1),minlength=300)
    require((repetition>0).all() and int(repetition.sum())==2288,'All300 unique labels and all2288 overlapping occurrences')
    return {'donor_source_group_ids300':wanted,'donor_target300x15_FP32':unique,'donor_raw_found300x25_int64':np.asarray([r['raw_found25'] for r in labels],np.int64),
        'target_self15_286x8x15_FP32':y,'donor_label_index286x8':index,'donor_label_repetition_count300':repetition}

def fmean(values):
    a=np.asarray(values,np.float64);require(a.size>0 and np.isfinite(a).all(),'Finite scalar reduction operands');return math.fsum(float(v) for v in a.flat)/a.size

def score_cached(logits,target):
    require(logits.dtype==target.dtype==np.float32 and logits.shape==target.shape and logits.ndim==2 and logits.shape[1]==15 and np.isfinite(logits).all() and np.isin(target,[0,1]).all(),'Finite matched cached logits15/target15')
    z=logits.astype(np.float64);y=target.astype(np.float64);prob=np.empty(z.shape,np.float64);bce=np.empty(z.shape,np.float64)
    for ix in np.ndindex(z.shape):
        value=float(z[ix]);e=math.exp(-abs(value));prob[ix]=1/(1+e) if value>=0 else e/(1+e)
        bce[ix]=math.log1p(e)+(max(value,0.) if not y[ix] else max(-value,0.))
    intensity=np.asarray([math.fsum(float(v) for v in row) for row in prob]);truth=np.asarray([math.fsum(float(v) for v in row) for row in y]);error=intensity-truth
    return {'logits15_FP32':logits.copy(),'probability15_F64':prob,'BCE15_F64':bce,'Brier15_F64':(prob-y)**2,'decision15_bool':z>0,'truth15_bool':y>0,'zero_logit_tie15_bool':z==0,
        'pred_intensity_F64':intensity,'target_intensity_F64':truth,'intensity_error_F64':error,'pred_proxy_penalty_F64':-.02*intensity,'target_proxy_penalty_F64':-.02*truth,'proxy_penalty_error_F64':-.02*error}

def cached_summary(scores,indices):
    p=scores['decision15_bool'][indices];y=scores['truth15_bool'][indices];tie=scores['zero_logit_tie15_bool'][indices]
    masks={'TP':p&y,'FP':p&~y,'FN':~p&y,'TN':~p&~y,'zero_logit_tie':tie,'tie_positive_target':tie&y,'tie_negative_target':tie&~y}
    total={key:int(value.sum()) for key,value in masks.items()};features={key:value.sum(0).astype(int).tolist() for key,value in masks.items()}
    def describe(a):return {'mean':fmean(a),'MAE':fmean(abs(a)),'RMS':math.sqrt(fmean(a*a)),'maxabs':float(abs(a).max())}
    bc=scores['BCE15_F64'][indices];br=scores['Brier15_F64'][indices]
    return {'rows':len(indices),'decisions':15*len(indices),'BCE_mean_entry_F64':fmean(bc),'BCE_sum15_mean_rows_F64':math.fsum(math.fsum(float(v) for v in row) for row in bc)/len(indices),
        'Brier_mean_entry_F64':fmean(br),'per_feature_BCE_F64':[fmean(bc[:,j]) for j in range(15)],'per_feature_Brier_F64':[fmean(br[:,j]) for j in range(15)],
        'confusion_total':total,'confusion_per_feature':features,'pred_intensity_mean':fmean(scores['pred_intensity_F64'][indices]),'target_intensity_mean':fmean(scores['target_intensity_F64'][indices]),
        'intensity_error':describe(scores['intensity_error_F64'][indices]),'Pen_proxy_penalty_error_F64':describe(scores['proxy_penalty_error_F64'][indices]),
        'positive_recall':total['TP']/(total['TP']+total['FN']) if total['TP']+total['FN'] else None,'positive_precision':total['TP']/(total['TP']+total['FP']) if total['TP']+total['FP'] else None,
        'per_feature_positive_recall':[tp/(tp+fn) if tp+fn else None for tp,fn in zip(features['TP'],features['FN'])],
        'per_feature_positive_precision':[tp/(tp+fp) if tp+fp else None for tp,fp in zip(features['TP'],features['FP'])]}

def expected_cached72(rollouts,target,path_index):
    rows=[]
    for role in ROLES:
        logits=rollouts[role]['self15_logits_FP32'];U.typed(logits,(286,8,15),'f4','all H8 cached logits')
        for h in range(8):
            scores=score_cached(logits[:,h],target[:,h])
            for path,indices in [('ALL',np.arange(286)),('A0',np.flatnonzero(path_index==0)),('A2',np.flatnonzero(path_index==1))]:
                rows.append({**cached_summary(scores,indices),'model_role':role,'horizon':h+1,'cohort':path,
                    'reference_phase':'lastsolver_preintegration~kdt+.015','state_phase':'postintegration_kdt+.020'})
    require(len(rows)==72,'All three roles/all horizons/path cached views');return rows

def csv_close(path,expected):
    with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(expected):return False
    for row,wanted in zip(rows,expected):
        if set(row)!=set(wanted):return False
        for key,value in wanted.items():
            actual=row[key]
            if isinstance(value,(dict,list)):ok=close(value,json.loads(actual))
            elif value is None:ok=actual==''
            elif type(value) is float:ok=close(value,float(actual))
            elif type(value) is int:ok=int(actual)==value
            elif type(value) is bool:ok=actual==str(value)
            else:ok=actual==value
            if not ok:return False
    return True

def own_proof():
    ready=read(AUDITOR_PREP/'READY.json');proof=read(checked(ready['cpu_report']));U.allchecks(proof)
    require(ready['schema']=='full151105_public_self15_H8_saved_auditor_READY:1' and proof['schema']=='full151105_public_self15_H8_saved_auditor_contract:1'
        and proof['actual_saved_arrays_model_FW_AD_physics']==0 and same(ready['source_pins']['auditor'],pin(__file__))
        and same(proof['source_pins']['auditor'],pin(__file__)),'Executed own single closed-fixture/source qualification')
    for value in ready['source_pins'].values():checked(value)
    return pin(AUDITOR_PREP/'READY.json'),ready['cpu_report']

def audit_cached_outputs(run,rollouts,label_packet,mapping):
    """Explicit formed flat72/97b-stack adapter, with scalar independent math."""
    target=label_packet[CACHED_WINDOW_KEY];expected=expected_cached72(rollouts,target,mapping['window_path_index286']);saved_scores={}
    for role in ROLES:
        stacked={}
        for h in range(8):
            values=score_cached(rollouts[role]['self15_logits_FP32'][:,h],target[:,h])
            for key,value in values.items():
                if key not in stacked:stacked[key]=np.empty((286,8,*value.shape[1:]),dtype=value.dtype)
                stacked[key][:,h]=value
        saved=load(run/(role+'_cached_self15_scores.npz'));saved_scores[role]=saved
        require(set(saved)==set(stacked)|{'relative_initial_logits15_F64','relative_initial_probability15_F64'},'Full 97b stack keys plus explicit initial drift')
        for key,value in stacked.items():
            actual=saved[key];require(actual.dtype==value.dtype and actual.shape==value.shape,'Cached element raw dtype/shape '+key)
            if value.dtype==np.float64:require(np.isfinite(actual).all() and np.allclose(value,actual,atol=F64_ATOL,rtol=F64_RTOL),'Scalar/F64 cached arithmetic '+key)
            else:require(bits(value,actual),'Raw logits/target decision/tie bytes '+key)
    for role,saved in saved_scores.items():
        require(bits(saved['relative_initial_logits15_F64'],saved['logits15_FP32'].astype(np.float64)-saved_scores[ROLES[0]]['logits15_FP32'].astype(np.float64))
            and bits(saved['relative_initial_probability15_F64'],saved['probability15_F64']-saved_scores[ROLES[0]]['probability15_F64']),'Captured probability/logit rollout drift, not same-input head drift')
    require(close(expected,read(run/CACHED_TABLE_FILENAME)) and csv_close(run/'all72_cached_self15_statistics.csv',expected),'All72 flat cached scalar/perfeature/null JSON and CSV arithmetic')
    return expected

def audit(protocol_path,run,out):
    p,parents=metadata(protocol_path);require(p['CPU_run_authorized'] is p['execution_authorized'] is True,'Root actual saved authorization');proof_ready,proof_report=own_proof()
    out.mkdir(exist_ok=False);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes())
    write(out/'equations_before_actual_arrays.json',{'auditor':pin(__file__),'math':pin(MATH_PATH),'helpers':pin(UTIL_PATH),'F64_atol':F64_ATOL,'F64_rtol':F64_RTOL,
        'actual_arrays_read':False,'model_weights_Torch_Log_FW_AD_physics':0})
    checks={};gate=lambda key,value:checks.__setitem__(key,bool(value));i=p['inputs'];r=read(run/'report.json');mathref=module(MATH_PATH,MATH_SHA)
    require(r['schema']=='full151105_public_self15_H8_donor_CPU_analysis:1' and r['status']=='COMPLETED_ALL3_PUBLIC_SELF15_FIXED_CONTROL_H8_NOT_POLICY_IMAGINATION','Complete new public15 H8 actual run')
    gate('new_protocol20_same_actual_run_and_inputs',same(r['protocol'],pin(protocol_path)) and p==read(run/'protocol.json') and r['input_pins']==i)
    gate('reported_fixed_newfamily_pair_checkpoints',set(r['checkpoint_pins'])==set(ARMS) and all(same(value,parents['pair']['checkpoints'][arm]) for arm,value in r['checkpoint_pins'].items()))
    sources=read(run/'source_inventory.json');outputs=read(run/'output_pins.json')
    for value in [*sources.values(),*outputs.values()]:checked(value)
    with zipfile.ZipFile(run/'execution_sources.zip') as archive:archive_ok=set(archive.namelist())==set(sources) and all(hashlib.sha256(archive.read(k)).hexdigest()==v['sha256'] for k,v in sources.items())
    gate('all_current_unique_source_and_execution_ZIP_output_pins',archive_ok and len({(checked(v).stat().st_dev,checked(v).stat().st_ino) for v in sources.values()})==len(sources))
    a=load(checked(i['H1_matched_inputs']));identities=read(checked(i['H1_identities']));mapping=mathref.donor_window_mapping(identities)
    original=mathref.source_byte_guards(a['pre_x452_FP32'],a['target_x452_FP32'],a['actual_u452_FP32'],a['all_substep_u452x4x20_FP32'],mapping);rows=mapping['window_transition_row_indices286x8']
    xi=load(checked(i['H1_prediction_arrays']))['physical_xi46_FP32'];U.typed(xi,(46,),'f4','old fixed xi46')
    label_packet=expanded_labels(read(checked(i['donor_self15_labels'])),identities,mapping)
    wanted={'initial_x286_FP32':a['pre_x452_FP32'][mapping['window_initial_row_indices286']], 'target286x8_FP32':a['target_x452_FP32'][rows],
        'actual_u286x8_FP32':a['actual_u452_FP32'][rows],'source_group_ids286x8':rows,'donor_indices2x150':mapping['donor_row_indices2x150'],
        'all298_links58_bitmask':np.ones((2,149,58),bool),'all300_control4_bitmask':np.ones((2,150,4,20),bool),'physical_xi46_FP32':xi,**label_packet}
    data=load(run/'matched_donor_windows286.npz')
    gate('all298_links300_held_controls286_windows_and_no_u150',all(original[k] is True for k in ('full58_links_bitexact','four_substep_controls_bitexact')) and set(data)==set(wanted)
        and all(bits(v,data[k]) for k,v in wanted.items()) and window_identities(mapping)==read(run/'all286_window_identities.json')
        and (run/'joined300_cached_sensor_labels.json').read_bytes()==checked(i['donor_self15_labels']).read_bytes())
    receipt=read(checked(i['initial_validation']));memory=U.tensor_digest_arrays({'initial_x':wanted['initial_x286_FP32'],'controls':wanted['actual_u286x8_FP32'],'xi':xi})
    gate('donor_memory_digest_separate_from_bound_old152_fit_digests',memory==r['donor_window_input_digest'] and U.tensor_digest_arrays({'xi':xi})==receipt['xi46_sha256']
        and all(r['old152_fit_digests'][k]==parents['training_result'][k]==parents['evaluation'][k] for k in FIT_KEYS))
    rollouts={};allstate=[];fullstate={};trace={};completed=0
    for role in ROLES:
        final=load(run/(role+'_rollout_final.npz'));partial=load(run/(role+'_rollout_partial.npz'));rollouts[role]=final
        gate(role+'_final_and_last_partial_masks_all2288_finite',set(partial).issubset(final) and all(bits(v,final[k]) for k,v in partial.items())
            and all(final[k].dtype==bool and final[k].shape==(286,8) and final[k].all() for k in ('attempted_mask','completed_mask')))
        U.typed(final['prediction58_FP32'],(286,8,58),'f4','native nextstate');U.typed(final['self15_logits_FP32'],(286,8,15),'f4','cached logits')
        U.typed(final['z12_FP32'],(286,8,2,6),'f4','raw aux12');U.typed(final['mode_logits2_FP32'],(286,8,2),'f4','raw aux2')
        trace[role]=mathref.trace_byte_guards(data['initial_x286_FP32'],final['prediction58_FP32'],final['state_input58_FP32'],final['dynamics_state58_FP32'],xi,final['dynamics278_FP32'])
        gate(role+'_every_x_and_full278_recursion_xi_zero174_actual_controls',all(trace[role][k] is True for k in ('initial_native_PRE_only','previous_predicted_state_each_h','dynamics_state_refreshed_each_h','captured_full_dynamics_bytes_checked')) and bits(final['actual_call_u20_FP32'],data['actual_u286x8_FP32']))
        if role==ROLES[0]:gate('initial_aux_is_algebraic_positive_zero_not_head_FW',np.array_equal(final['z12_FP32'].view(np.uint32),np.zeros((286,8,2,6),np.uint32)) and np.array_equal(final['mode_logits2_FP32'].view(np.uint32),np.zeros((286,8,2),np.uint32)))
        numeric=mathref.raw58_error_statistics(final['prediction58_FP32'],data['target286x8_FP32'],mapping['window_path_index286'])
        gate(role+'_all_RAW58_signed_errors_F64_original_six_units',bits(final['error58_F64'],numeric['signed_error286x8x58_F64']))
        fullstate[role]=numeric['statistics']
        allstate.extend(state_rows(numeric['statistics'],role));completed+=int(final['completed_mask'].sum())
    gate('all432_state_statistics_JSON_CSV_exact',U.raw_tree_equal(allstate,read(run/'all432_state_statistics.json')) and U.csv_matches(run/'all432_state_statistics.csv',allstate))
    write(out/'independent_raw58_perfeature_statistics.json',fullstate)
    cached=audit_cached_outputs(run,rollouts,label_packet,mapping)
    gate('all300_raw25_to15_and2288_expansion_all72_metrics',len(cached)==72);np.savez_compressed(out/'independent_cached_targets.npz',**label_packet);write(out/'independent72_cached_statistics.json',cached)
    before=load(run/'normalizer_buffers_before.npz');after=load(run/'normalizer_buffers_after.npz')
    norm_ok=set(before)==set(after) and all(bits(v,after[k]) for k,v in before.items())
    norm={role:{k.split('__',1)[1]:v for k,v in before.items() if k.startswith(role+'__')} for role in ROLES}
    norm_ok &= all(U.tensor_digest_arrays(values)==receipt['frozen_norm_sha256'] for values in norm.values())
    gate('raw_normalizer_arrays_three_roles_same_initial_unchanged',norm_ok)
    context_keys=('mass_actual_F64','I_body_F64','gravity_palm_F64','DeltaT_F64','palm_rot_raw_FP32','effective_s52_FP32','physical_xi46_FP32')
    with np.load(checked(i['public15_eval_prepared']),allow_pickle=False) as packet:prior_context={key:packet[key].copy() for key in context_keys}
    ctx=load(run/'physical_context_buffers.npz')
    for key in ('gravity_palm_F64','DeltaT_F64','palm_rot_raw_FP32'):
        value=prior_context[key];require(bits(value,np.broadcast_to(value[:1],value.shape).copy()),'All152 source context rows share exact bytes')
    Ib=prior_context['I_body_F64'];expected_ctx={'mass_actual_FP32':np.asarray(prior_context['mass_actual_F64'],np.float32),'I_body_F64':Ib,
        'gravity_palm_F64':prior_context['gravity_palm_F64'][0],'DeltaT_F64':np.asarray(prior_context['DeltaT_F64'][0]),'palm_rot_FP32':prior_context['palm_rot_raw_FP32'][0],
        'effective_s52_FP32':prior_context['effective_s52_FP32'],'physical_xi46_FP32':prior_context['physical_xi46_FP32'],'I_body_antisymmetry_maxabs_F64':np.asarray(abs(Ib-Ib.T).max())}
    gate('original152_context_allrows_and_registered_buffers_rawbytes',set(ctx)==set(expected_ctx) and all(bits(v,ctx[k]) for k,v in expected_ctx.items()) and bits(ctx['physical_xi46_FP32'],xi))
    storage=read(run/'named_storage_inventory.json');require(set(storage)==set(ROLES),'All three actual storage inventories')
    storage_sets={role:{(v['device'],v['storage_base_ptr']) for v in values if v['storage_nbytes']>0} for role,values in storage.items()}
    gate('captured_named14_16_parameter_inventory_and_disjoint_storage',all(sum(v['kind']=='parameter' for v in storage[role])==(14 if role==ROLES[0] else 16) for role in ROLES)
        and all(not (storage_sets[ROLES[a]]&storage_sets[ROLES[b]]) for a in range(3) for b in range(a+1,3)))
    rng=U.raw_rng(run);gate('raw_RNG_vectors_headers_values_unchanged',all(rng.values()) and all(r['raw_rng_comparison'].values()))
    gate('new_budget_3reads_1ctor_2copies24FW16head6864_zeroAD',completed==6864 and r['counts']==EXPECTED_COUNTS)
    gate('source_backed_model_host_invariants_and_limits',all(r['checks'].values()) and set(r['model_invariants'])==set(ROLES)
        and all(v['state_sha256_before']==v['state_sha256_after'] for v in r['model_invariants'].values()) and r['normalizer_host_before']==r['normalizer_host_after']
        and r['donor_self15_truth_evaluated'] is True and all(r[k] is False for k in ('teacher_forcing','output_clamping_or_R_projection','accuracy_qualified','physical_gradient_qualified','whole_hidden_state_qualified','Actor_or_task_benefit_qualified')))
    no_models();gate('reviewer_no_Torch_checkpoint_Log_model_FW_AD_or_physics',True)
    result={'schema':'full151105_public_self15_H8_donor_saved_independent:1','status':'PASS_SAVED_ALL286_H8_PUBLIC15_STATE_AND_CACHED_SENSOR_ARITHMETIC' if all(checks.values()) else 'FAIL_SAVED_PUBLIC15_H8_REVIEW_RETAINED',
        'engineering_qualified':all(checks.values()),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'auditor':pin(__file__),'math':pin(MATH_PATH),'protocol':pin(protocol_path),
        'producer_report':pin(run/'report.json'),'run_dir':str(run),'input_pins':i,'auditor_ready':proof_ready,'auditor_cpu_report':proof_report,
        'independent_statistics':{'raw58_perfeature':pin(out/'independent_raw58_perfeature_statistics.json'),'cached72':pin(out/'independent72_cached_statistics.json'),'cached_targets':pin(out/'independent_cached_targets.npz')},
        'window_count':286,'horizon':8,'dependent_prediction_rows':completed,'cached_cohort_tables':72,
        'details':{'source_unique_files':len(sources),'runtime_original_guards':original,'recurrence_traces':trace,'raw_RNG':rng,
            'model_scope':'Weight/head/host/fit/norm/context source-backed captures and saved relations; reviewer never loads component files. Raw norm vectors and call traces independently checked.',
            'label_scope':'All300 packet raw25->15 is recomputed; original reward-cache join is delegated to separately qualified donor source review. H>1 cached-error includes different predicted states, not pure head forgetting.'},
        'accuracy_qualified':False,'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'Actor_or_task_benefit_qualified':False,'reviewer_weights_Torch_models_Log_FW_AD_MJB_physics_GPU':0}
    write(out/'report.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol');ap.add_argument('--run-dir');ap.add_argument('--output');ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if not args.protocol:print(json.dumps({'status':'SOURCE_DRAFT_PENDING_FINAL_OWNER_BINDINGS','inputs':sorted(INPUTS),'default_arrays_models':0}));return 0
    path=Path(args.protocol).resolve(strict=True)
    if not args.run:metadata(path);print(json.dumps({'status':'PASS_NEW_NAMESPACE_JSON_HASH_ONLY','actual_arrays_models':0}));return 0
    require(args.run_dir and args.output,'Actual saved run/fresh output required');out=Path(args.output).resolve()
    try:
        result=audit(path,Path(args.run_dir).resolve(strict=True),out);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total']}));return 0 if result['engineering_qualified'] else 2
    except BaseException as error:
        if out.exists() and not (out/'failure.json').exists():write(out/'failure.json',{'error':repr(error),'actual_saved_only':True,'automatic_retry':False})
        raise

if __name__=='__main__':raise SystemExit(main())
