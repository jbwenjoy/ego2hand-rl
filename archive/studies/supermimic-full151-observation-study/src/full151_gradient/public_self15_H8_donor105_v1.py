"""UNFROZEN SOURCE: new-family three-model fixed teacher-control donor H8.

Metadata defaults never decode arrays or construct a model. Separately qualified
donor300 cached15 reference is mandatory; old152 labels are never broadcast.
"""
from __future__ import annotations
import csv,math,os,struct,sys,time,zipfile
from pathlib import Path
from . import component_H8_donor105_v1 as old_h8
from . import boundary_public_self15_final_evaluation105_v1 as final_eval

c=old_h8.c;ROOT=c.ROOT;read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require
SCHEMA='full151105_public_self15_H8_donor_CPU_protocol:1'
ROLE='three_public_self15_components_fixed_control_H8_donor_state_error'
MODEL_ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
ARMS=MODEL_ROLES[1:]
PREFIX='full151105_public_self15_H8_donor_CPU_'
CONTRACT=ROOT/'docs/public_self15_H8_donor_contract_v1.md'
CONTRACT_SHA='096e3f38b67c711905429a4a7d4480e87a2b1f478ff45369724c31ce7126cdbc'
INPUTS={'initial_component','initial_validation','training_protocol','training_result','final_pair_manifest','final_saved_health',
        'public15_eval_protocol','public15_eval_report','public15_eval_saved_review','public15_eval_prepared',
        'H1_config','H1_report','H1_matched_inputs','H1_prediction_arrays','H1_identities','H1_saved_independent25','prior_H8_contiguity',
        'donor_self15_producer_report','donor_self15_labels','donor_self15_saved_independent'}
OWN={'consumer':ROOT/'src/full151_gradient/public_self15_H8_donor105_v1.py','CLI':ROOT/'scripts/diagnose_full151_public_self15_H8_donor_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_public_self15_H8_donor_cpu_v1.py'}
DEPENDENCIES={'old_H8_source':ROOT/'src/full151_gradient/component_H8_donor105_v1.py',
              'new_family_consumer':ROOT/'src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py',
              'new_family_trainer':ROOT/'src/full151_gradient/boundary_public_self15_training105_v1.py',
              'component_wrapper':ROOT/'src/full151_gradient/boundary_impulse_component105_v1.py',
              'cached_metrics_source':ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py'}
EXPECTED_SOURCES={'old_H8_source':'0c77dc5255ffd2d7294dabdc7bbaf8ed09e80f3f2bc0eb89fee3954d268a6b13',
                  'new_family_consumer':'81ad78161c75b1720a77aead414ccb6c5a447b0f614e5e597a940aaa1ea77419',
                  'new_family_trainer':'796c360539961143fe1f588fd18eb08e4b168d141b34daee23da802a72194e7e',
                  'component_wrapper':'003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317',
                  'cached_metrics_source':'97b74e0173bb6d95b55c59d924db7c5502c83cf610907c6c3aa357d54808da5e'}
CONTEXT_KEYS=('mass_actual_F64','I_body_F64','gravity_palm_F64','DeltaT_F64','palm_rot_raw_FP32','effective_s52_FP32','physical_xi46_FP32')
DONOR_IDENTITY={'group_id','path','mode','k','branch_id','source_frame','terminal_observation','valid_next_policy_decision','no_u150','first_physical_pre_index','last_physical_index','post20_GPU_row_index'}


def validate_protocol(p,authorized=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE and p.get('output_prefix')==PREFIX,'New public15 H8 namespace')
    require(type(p.get('execution_authorized')) is bool and type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is False and p.get('training_authorized') is False and p.get('automatic_retry') is False,'Explicit bounded CPU-only/no-training/no-retry')
    require(p.get('model_roles')==list(MODEL_ROLES) and p.get('cpu_threads')==4 and p.get('horizon')==8 and p.get('batch_windows')==286 and p.get('starts_per_path')==143,'Fixed three roles/H8/286')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and p.get('source_pins') and p.get('accuracy_gate') is None,'Exact20 inputs; no scientific accuracy gate')
    require(p.get('evaluate_donor_self15') is True,'Required separately qualified donor300 truth; no silent drift-only fallback')
    if authorized:require(p['CPU_run_authorized'] and p['execution_authorized'],'Root explicit actual H8 authorization')
    return p


def declaration():
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Metadata-only default')
    return {'schema':'full151105_public_self15_H8_declaration:1','status':'UNFROZEN_SOURCE_NOT_ACTUAL_RUN','protocol_schema':SCHEMA,'role':ROLE,'model_roles':list(MODEL_ROLES),
        'input_aliases':sorted(INPUTS),'evaluate_donor_self15':True,'actual_H8_authorized':False,
        'budget':{'own_component_loads':3,'native_WM_constructors':1,'raw_deepcopies':2,'native_B286_FW':24,'aux_head_FW':16,'predicted_rows':6864,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0},
        'source_pins':{k:pin(v) for k,v in OWN.items()}}


def metadata_preflight(protocol_path,output=None):
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Default JSON/hash only')
    p=validate_protocol(read(protocol_path));i=p['inputs']
    require(type(CONTRACT_SHA) is str and same(p['source_contract'],{'path':str(CONTRACT),'sha256':CONTRACT_SHA}),'Frozen new H8 scientific contract required')
    for value in i.values():checked(value)
    for name,value in EXPECTED_SOURCES.items():require(pin(DEPENDENCIES[name])['sha256']==value,'Frozen source dependency '+name)
    # Reuse the source-proved new-family metadata gates; no tensor/model path.
    fm=final_eval.metadata_preflight(checked(i['public15_eval_protocol']))
    eval_report=read(checked(i['public15_eval_report']));eval_review=read(checked(i['public15_eval_saved_review']));c.allchecks(eval_report);c.allchecks(eval_review)
    require(eval_report['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1' and eval_report['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION','Completed current-family evaluation')
    require(eval_review['schema']=='full151105_boundary_public_self15_final_prediction_saved_independent:1' and eval_review['status']=='PASS_SAVED_ALL3_ALL152_PUBLIC_SELF15_FIXED_FINAL_PREDICTIONS' and eval_review['engineering_qualified'] is True and same(eval_review['producer_report'],i['public15_eval_report']) and same(eval_review['protocol'],i['public15_eval_protocol']),'Independent current-family saved prediction review')
    require(same(eval_report['protocol'],i['public15_eval_protocol']) and same(eval_report['outputs']['prepared_inputs_and_context.npz'],i['public15_eval_prepared']) and checked(i['public15_eval_prepared']).samefile(checked(i['public15_eval_report']).parent/'prepared_inputs_and_context.npz'),'Actual immutable context-row packet')
    mapping={'initial_component':'original_initial_component','initial_validation':'initial_validation','training_protocol':'training_protocol','training_result':'training_result','final_pair_manifest':'final_pair_manifest','final_saved_health':'final_saved_health'}
    for alias,parent_alias in mapping.items():require(same(i[alias],fm['protocol']['inputs'][parent_alias]),'Same new-family initial/final source '+alias)
    fit={key:fm['training_result'][key] for key in final_eval.FIT_DIGEST_KEYS}
    require(all(type(v) is str and len(v)==64 and eval_report[k]==v for k,v in fit.items()),'Prior old152 fit digests, not donor window digest')
    h1=read(checked(i['H1_report']));review=read(checked(i['H1_saved_independent25']));c.allchecks(review)
    require(review['schema']=='full151105_state_prediction452_saved_independent:1' and review['status']=='PASS_SAVED_ALL452_STATE_ERROR_REVIEW_NOT_MODEL_QUALIFICATION' and review['engineering_qualified'] is True and same(review['config'],i['H1_config']) and same(review['producer_report'],i['H1_report']),'Qualified old H1 saved25 donor input source')
    require(h1['schema']=='full151105_state_prediction452_CPU_analysis:1' and h1['prediction_rows']==452 and same(h1['config'],i['H1_config']),'Original452 source, no new state converter')
    run=checked(i['H1_report']).parent;manifest=read(run/'output_pins.json')
    for alias,name in (('H1_matched_inputs','matched_input_arrays.npz'),('H1_prediction_arrays','prediction_arrays.npz'),('H1_identities','group_identities452.json')):
        require(same(i[alias],manifest[name]) and checked(i[alias]).samefile(run/name),'H1 immutable artifact route '+alias)
    prior=read(checked(i['prior_H8_contiguity']))
    require(prior['schema']=='full151105_H8_donor_saved_contiguity_source_feasibility:1' and prior['all_links_all58_bitexact'] is True and prior['unique_adjacent_links']==298 and prior['all_donor_four_controls_bitexact'] is True,'Prior contiguity evidence, runtime recheck still mandatory')
    require(same(prior['input_pins']['matched_input_arrays.npz'],i['H1_matched_inputs']) and same(prior['input_pins']['group_identities452.json'],i['H1_identities']),'Same old donor continuity authority')
    donor=read(checked(i['donor_self15_producer_report']));donor_review=read(checked(i['donor_self15_saved_independent']));c.allchecks(donor);c.allchecks(donor_review)
    require(donor['schema']=='full151105_donor300_cached_self15_saved_analysis:1' and donor['status']=='COMPLETED_ALL300_DONOR_CACHED_SENSOR_LABEL_JOIN_ONLY','New complete donor300 label-only producer')
    require(donor_review['schema']=='full151105_donor300_cached_self15_saved_independent:1' and donor_review['status']=='PASS_SAVED_ALL300_DONOR_RAW25_TO_NATIVE15_AND_H1_JOIN' and donor_review['engineering_qualified'] is True,'Separate positive donor300 label qualification, not oldcached7')
    require(same(donor_review['producer_report'],i['donor_self15_producer_report']) and same(donor_review['protocol'],donor['protocol']) and same(donor_review['label_JSON'],i['donor_self15_labels']) and same(donor_review['H1_identities'],i['H1_identities']),'One exact donor label/source identity package')
    require(same(donor['outputs']['joined300_cached_sensor_labels.json'],i['donor_self15_labels']) and checked(i['donor_self15_labels']).samefile(checked(i['donor_self15_producer_report']).parent/'joined300_cached_sensor_labels.json'),'Immutable donor300 label-only JSON route')
    for alias in ('H1_config','H1_report','H1_identities','H1_saved_independent25'):require(same(donor['input_pins'][alias],i[alias]),'Same donor label and state authority '+alias)
    receipt=read(checked(i['initial_validation']))
    require(same(receipt['strict_loader']['checkpoint'],h1['input_pins']['fixed_A2_checkpoint']),'Initial baseline parent105 agrees with old H1 physical-xi source')
    sources=c.canonical_sources(list(p['source_pins'].values())+list(fm['training']['source_pins'].values())+list(read(checked(i['H1_config']))['source_pins'].values())+list(i.values())+list(fm['pair']['checkpoints'].values())+[pin(path) for path in list(OWN.values())+list(DEPENDENCIES.values())]+[p['source_contract'],donor['protocol']]+list(donor['source_pins'].values())+list(donor_review['source_pins'].values()))
    for path in list(OWN.values())+list(DEPENDENCIES.values())+[CONTRACT]:require(any(same(pin(path),v) for v in p['source_pins'].values()),'Current H8/helper source membership')
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists() and ('output' not in p or out==Path(p['output']).resolve()),'Fresh exact H8 output')
    return {'status':'PASS_NEW_FAMILY_H8_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'sources':sources,'output':out,'family':fm,'fit_digests':fit,'actual_windows_models_FW':0}


def join_donor_labels(identities,joined,np):
    """Only label-only JSON; H1 global gids are not dense donor indices."""
    from . import boundary_self15_cached_saved105_v1 as metrics
    require(type(identities) is list and len(identities)==452 and [g['group_id'] for g in identities]==list(range(452)),'Original H1 all452 row order')
    donors=[g for g in identities if g['mode']=='donor']
    require(len(donors)==300 and type(joined) is list and len(joined)==300 and [(g['path'],g['k']) for g in donors]==[(path,k) for path in ('A0','A2') for k in range(150)],'All300 donors in H1 order, not first300')
    labels=[];raw_rows=[];gids=[]
    for g,r in zip(donors,joined):
        require(type(g) is dict and set(g)==DONOR_IDENTITY and type(r) is dict and set(r)==DONOR_IDENTITY|{'contact15','raw_found25','label_time','state_time'},'Exact H1 identity plus label-only fields')
        require(all(type(r[key]) is type(g[key]) and r[key]==g[key] for key in DONOR_IDENTITY),'Every original donor identity field exact, including boolean types')
        k=g['k'];gid=k+(226 if g['path']=='A2' else 0)
        require(type(k) is int and g['branch_id'] is None and all(type(g[key]) is int for key in ('group_id','source_frame','terminal_observation','first_physical_pre_index','last_physical_index','post20_GPU_row_index')) and
                g['group_id']==gid and g['source_frame']==200+k and g['terminal_observation']==k+1 and g['first_physical_pre_index']==4*gid and g['last_physical_index']==4*gid+3 and g['post20_GPU_row_index']==1808+gid and
                g['valid_next_policy_decision'] is (k<149) and g['no_u150'] is True,'Native donor PRE/POST/source/terminal mapping; no u150')
        require(r['label_time']=='lastsolver_preintegration~kdt+.015' and r['state_time']=='postintegration_kdt+.020','Cached lastsolver15 reference versus POST20 state')
        target=r['contact15'];raw=r['raw_found25']
        require(type(target) is list and len(target)==15 and all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in target),'Binary15 label-only target')
        require(type(raw) is list and len(raw)==25 and all(type(v) in (int,float,bool) and math.isfinite(v) and v>=0 and int(v)==v and v<=np.iinfo(np.int64).max for v in raw),'Nonnegative integer found25 counts')
        found=[v>0 for v in raw];reconstructed=[float(found[j]) for j in metrics.MIDDLE]+[float(found[d] or found[t]) for d,t in zip(metrics.DISTAL,metrics.TIPS)]
        require(all(struct.pack('<f',float(a))==struct.pack('<f',b) for a,b in zip(target,reconstructed)),'All300 raw25 reconstruct cached15 FP32 bytes')
        labels.append(target);raw_rows.append(raw);gids.append(gid)
    require(len(set(gids))==300,'Unique donor gids; A2 begins226')
    return {'donor_source_group_ids300':np.asarray(gids,np.int64),'donor_target300x15_FP32':np.asarray(labels,np.float32),'donor_raw_found300x25_int64':np.asarray(raw_rows,np.int64)}


def expand_donor_labels(source_group_ids,donor,np):
    require(source_group_ids.dtype==np.int64 and source_group_ids.shape==(286,8),'Exact286x8 source-gid mapping')
    lookup={int(gid):row for row,gid in enumerate(donor['donor_source_group_ids300'])}
    require(len(lookup)==300 and all(int(gid) in lookup for gid in source_group_ids.flat),'Every window target references separately qualified donor300')
    index=np.asarray([[lookup[int(gid)] for gid in row] for row in source_group_ids],np.int64)
    repetitions=np.bincount(index.ravel(),minlength=300)
    require(index.shape==(286,8) and (repetitions>0).all() and int(repetitions.sum())==2288,'All300 unique references retained with2288 overlapping occurrences')
    return {'target_self15_286x8x15_FP32':donor['donor_target300x15_FP32'][index].copy(),'donor_label_index286x8':index,'donor_label_repetition_count300':repetitions}


def cached_self15_statistics(logits,target,starts,role,np):
    """Frozen97b arithmetic; own H8 cohorts, never its old152 cohorts()."""
    from . import boundary_self15_cached_saved105_v1 as metrics
    require(logits.dtype==target.dtype==np.float32 and logits.shape==target.shape==(286,8,15) and role in MODEL_ROLES and len(starts)==286,'AllH8 logits/reference rows')
    groups={'ALL':list(range(286)),'A0':[j for j,s in enumerate(starts) if s['path']=='A0'],'A2':[j for j,s in enumerate(starts) if s['path']=='A2']}
    require([len(v) for v in groups.values()]==[286,143,143],'H8 exact overlapping cohorts')
    elements={};tables=[]
    def ratio(a,b):return None if b==0 else a/b
    for h in range(8):
        scores=metrics.score_matrix(logits[:,h],target[:,h],np)
        for key,value in scores.items():
            if key not in elements:elements[key]=np.empty((286,8,*value.shape[1:]),dtype=value.dtype)
            elements[key][:,h]=value
        for cohort,ids in groups.items():
            row=metrics.summary(scores,ids,np);total=row['confusion_total'];features=row['confusion_per_feature']
            row.update(model_role=role,horizon=h+1,cohort=cohort,reference_phase='lastsolver_preintegration~kdt+.015',state_phase='postintegration_kdt+.020',
                positive_recall=ratio(total['TP'],total['TP']+total['FN']),positive_precision=ratio(total['TP'],total['TP']+total['FP']),
                per_feature_positive_recall=[ratio(tp,tp+fn) for tp,fn in zip(features['TP'],features['FN'])],
                per_feature_positive_precision=[ratio(tp,tp+fp) for tp,fp in zip(features['TP'],features['FP'])])
            tables.append(row)
    require(len(tables)==24,'Every role8step3cohort retained; empty positive denominators null')
    return elements,tables


def context_from_prior_rows(rows,initial,payloads,torch,np):
    """All152 prior enrolled context rows, never silently take one checkpoint row."""
    from . import boundary_impulse_component105_v1 as component
    scale=initial['frozen_s52'];xi=initial['physical_xi46']
    require(component.bit_equal(torch.from_numpy(rows['effective_s52_FP32']),scale) and component.bit_equal(torch.from_numpy(rows['physical_xi46_FP32']),xi),'Own initial/context-row xi and effective52 bytes')
    ctx=component.fixed_context_from_rows(torch.from_numpy(rows['mass_actual_F64'].astype(np.float32)),torch.from_numpy(rows['I_body_F64']),torch.from_numpy(rows['gravity_palm_F64']),torch.from_numpy(rows['DeltaT_F64']),torch.from_numpy(rows['palm_rot_raw_FP32']),scale,xi)
    for arm,payload in payloads.items():
        require(payload['schema']==final_eval.trainer.ARTIFACT_SCHEMA and payload['pair_schema']==final_eval.trainer.PAIR_SCHEMA and payload['arm']==arm and payload['iteration']==1000 and payload['checkpoint_tag']=='AFTER_OPTIMIZER_I','New own AFTER1000 namespace before any windows')
        state=payload['physical_context_state'];require(set(state)==set(ctx.state_dict()) and all(component.bit_equal(value,state[name]) for name,value in ctx.state_dict().items()),'Both final contexts equal prior full-row enrolled bytes')
    return ctx


def autoregress(role,model,initial_x,controls,xi,torch,np,counters,save_partial):
    """Actual x/dynamics278/u captured for every call; no teacher state forcing."""
    n=initial_x.shape[0];require(n==286 and initial_x.shape==(n,58) and controls.shape==(n,8,20),'Fixed donor window shapes')
    result={key:np.full((n,8,*tail),np.nan,np.float32) for key,tail in {'prediction58_FP32':(58,),'self15_logits_FP32':(15,),'z12_FP32':(2,6),'mode_logits2_FP32':(2,),
        'state_input58_FP32':(58,),'dynamics_state58_FP32':(58,),'dynamics278_FP32':(278,),'actual_call_u20_FP32':(20,)}.items()}
    result.update(attempted_mask=np.zeros((n,8),bool),completed_mask=np.zeros((n,8),bool));current=initial_x
    wrapped=role in ARMS
    with torch.inference_mode():
        for h in range(8):
            u=controls[:,h];dynamics=torch.cat((current,xi[None].expand(n,-1),current.new_zeros((n,174))),-1)
            for key,value in (('state_input58_FP32',current),('dynamics_state58_FP32',dynamics[:,:58]),('dynamics278_FP32',dynamics),('actual_call_u20_FP32',u)):result[key][:,h]=value.numpy()
            before=c._tensor_digest({'x':current,'u':u,'dynamics':dynamics},torch);result['attempted_mask'][:,h]=True
            counters['native_forward_attempted']+=1
            if wrapped:
                counters['aux_head_forward_attempted']+=1;prediction,self15,z,mode=model(current,u,dynamics);counters['aux_head_forward_returned']+=1
            else:prediction,self15=model(current,u,dynamics);z=torch.zeros((n,2,6),dtype=torch.float32);mode=torch.zeros((n,2),dtype=torch.float32)
            counters['native_forward_returned']+=1
            require(prediction.shape==(n,58) and self15.shape==(n,15) and z.shape==(n,2,6) and mode.shape==(n,2) and all(value.dtype==torch.float32 for value in (prediction,self15,z,mode)),'Native/raw-head H8 output ABI')
            for key,value in (('prediction58_FP32',prediction),('self15_logits_FP32',self15),('z12_FP32',z),('mode_logits2_FP32',mode)):result[key][:,h]=value.numpy()
            finite=torch.isfinite(prediction).all(-1)&torch.isfinite(self15).all(-1)&torch.isfinite(z).all(-1).all(-1)&torch.isfinite(mode).all(-1)
            result['completed_mask'][:,h]=finite.numpy();save_partial(result)
            require(c._tensor_digest({'x':current,'u':u,'dynamics':dynamics},torch)==before,'Model changed actual call inputs')
            require(bool(finite.all()),'Any nonfinite row stops ALL windows; no clamp/subset/teacher-state repair')
            counters['predicted_rows']+=n;current=prediction
    return result


def run_analysis(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,value in m['sources'].items():z.write(checked(value),name)
    counts={'own_component_loads':0,'native_WM_constructors':0,'raw_deepcopies':0,'native_forward_attempted':0,'native_forward_returned':0,'aux_head_forward_attempted':0,'aux_head_forward_returned':0,'predicted_rows':0,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0}
    phase='source_inputs';modules={}
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        from . import boundary_impulse_component105_v1 as component
        torch.set_num_threads(4);rng_before=c._rng(torch,np);old_h8.cf.save_raw_rng(out,'before',rng_before,np)
        with np.load(checked(i['H1_matched_inputs']),allow_pickle=False) as f:old={key:f[key].copy() for key in ('pre_x452_FP32','target_x452_FP32','actual_u452_FP32','all_substep_u452x4x20_FP32')}
        with np.load(checked(i['H1_prediction_arrays']),allow_pickle=False) as f:old_xi=f['physical_xi46_FP32'].copy()
        with np.load(checked(i['public15_eval_prepared']),allow_pickle=False) as f:ctx_rows={key:f[key].copy() for key in CONTEXT_KEYS}
        initial=torch.load(checked(i['initial_component']),map_location='cpu',weights_only=False);counts['own_component_loads']+=1
        payloads={}
        for arm in ARMS:payloads[arm]=torch.load(checked(m['family']['pair']['checkpoints'][arm]),map_location='cpu',weights_only=False);counts['own_component_loads']+=1
        old_h8.verify_xi(initial['physical_xi46'],old_xi,np,torch)
        ctx=context_from_prior_rows(ctx_rows,initial,payloads,torch,np)
        for arm,payload in payloads.items():
            require(same(payload['protocol'],i['training_protocol']) and same(payload['recipe'],m['family']['training']['recipe']) and all(payload[key]==value for key,value in m['fit_digests'].items()),'Actual new-family old152 fit provenance, not donor digest')
        phase='all_guards_before_windows'
        identities=read(checked(i['H1_identities']));joined=read(checked(i['donor_self15_labels']));donor=join_donor_labels(identities,joined,np)
        data,starts=old_h8.assemble_donor_windows(old,identities,np)
        expanded=expand_donor_labels(data['source_group_ids286x8'],donor,np);data.update(donor);data.update(expanded)
        (out/'joined300_cached_sensor_labels.json').write_bytes(checked(i['donor_self15_labels']).read_bytes())
        data['physical_xi46_FP32']=old_xi.copy();np.savez_compressed(out/'matched_donor_windows286.npz',**data);c._write_json(out/'all286_window_identities.json',starts)
        x0=torch.from_numpy(data['initial_x286_FP32']);u=torch.from_numpy(data['actual_u286x8_FP32']);xi=initial['physical_xi46']
        data_before=c._tensor_digest({'initial_x':x0,'controls':u,'xi':xi},torch)
        raw=old_h8.make_initial_eval_component(initial,read(checked(i['initial_validation'])),torch);counts['native_WM_constructors']+=1
        modules,construction=component.make_matched_components(raw,ctx);counts['raw_deepcopies']=2
        for arm in ARMS:final_eval.load_final_into(modules[arm],payloads[arm],initial,i['training_protocol'],m['family']['training'],m['fit_digests'],torch)
        models={'unwrapped_baseline':raw,**modules};before={role:component.state_digest(model) for role,model in models.items()}
        require(len(list(raw.named_parameters()))==14 and all(len(list(module.named_parameters()))==16 for module in modules.values()),'Raw14 and each frozen wrapper16 named parameters')
        storage,disjoint=final_eval.parity.named_storage_inventory(models);require(disjoint,'Named storage bases disjoint for raw and both wrapped models');c._write_json(out/'named_storage_inventory.json',storage)
        host_before={role:c._norm_config(raw if role==MODEL_ROLES[0] else model.wm) for role,model in models.items()}
        np.savez_compressed(out/'normalizer_buffers_before.npz',**{role+'__'+name:value.numpy().copy() for role,model in models.items() for name,value in c._norm_state(raw if role==MODEL_ROLES[0] else model.wm).items()})
        np.savez_compressed(out/'physical_context_buffers.npz',**{name:value.numpy().copy() for name,value in ctx.state_dict().items()})
        all_stats=[];all_cached=[];invariants={};initial_cached=None
        for role in MODEL_ROLES:
            phase=role
            def save_partial(value):np.savez_compressed(out/(role+'_rollout_partial.npz'),**value)
            result=autoregress(role,models[role],x0,u,xi,torch,np,counts,save_partial)
            error,stats=old_h8.error_statistics(result['prediction58_FP32'],data['target286x8_FP32'],starts,np)
            np.savez_compressed(out/(role+'_rollout_final.npz'),**result,error58_F64=error)
            elements,tables=cached_self15_statistics(result['self15_logits_FP32'],data['target_self15_286x8x15_FP32'],starts,role,np)
            if initial_cached is None:initial_cached={key:elements[key].copy() for key in ('logits15_FP32','probability15_F64')}
            elements['relative_initial_logits15_F64']=elements['logits15_FP32'].astype(np.float64)-initial_cached['logits15_FP32'].astype(np.float64)
            elements['relative_initial_probability15_F64']=elements['probability15_F64']-initial_cached['probability15_F64']
            np.savez_compressed(out/(role+'_cached_self15_scores.npz'),**elements);all_cached.extend(tables)
            for row in stats:row['model_role']=role
            all_stats.extend(stats);invariants[role]={'source_artifact_role':'initial_unmodified_A2' if role==MODEL_ROLES[0] else 'own_public_self15_AFTER1000','state_sha256_before':before[role],'state_sha256_after':component.state_digest(models[role])}
        after={role:component.state_digest(model) for role,model in models.items()}
        host_after={role:c._norm_config(raw if role==MODEL_ROLES[0] else model.wm) for role,model in models.items()}
        np.savez_compressed(out/'normalizer_buffers_after.npz',**{role+'__'+name:value.numpy().copy() for role,model in models.items() for name,value in c._norm_state(raw if role==MODEL_ROLES[0] else model.wm).items()})
        for module in modules.values():module.assert_frozen();require(module.counts['hook_calls']==module.counts['forward_returned']==module.counts['head_forward_returned']==8 and module._cache is None and not module._active,'Exactly eight wrapper/head calls, empty caches')
        rng_after=c._rng(torch,np);old_h8.cf.save_raw_rng(out,'after',rng_after,np);rng=final_eval.parity.raw_rng_equality(rng_before,rng_after,np,torch)
        checks={'all298_links58_and_all300_actual_u4_bits':bool(data['all298_links58_bitmask'].all() and data['all300_control4_bitmask'].all()),'weights_heads_norm_context_unchanged':before==after,
            'normalizer_host_configuration_unchanged':host_before==host_after,'raw_rng_values_bytes_unchanged':all(rng.values()),'fixed_window_inputs_xi_unchanged':data_before==c._tensor_digest({'initial_x':x0,'controls':u,'xi':xi},torch),
            'all_parameters_frozen_no_grad':all(not q.requires_grad and q.grad is None for model in models.values() for q in model.parameters()),'no_CUDA_Warp_MuJoCo':not torch.cuda.is_initialized() and 'warp' not in sys.modules and 'mujoco' not in sys.modules}
        checks.update(named_model_storage_disjoint=disjoint,all300_cached_reference_2288_expansion_and_all72_tables=len(all_cached)==72 and int(data['donor_label_repetition_count300'].sum())==2288)
        require(all(checks.values()) and counts=={'own_component_loads':3,'native_WM_constructors':1,'raw_deepcopies':2,'native_forward_attempted':24,'native_forward_returned':24,'aux_head_forward_attempted':16,'aux_head_forward_returned':16,'predicted_rows':6864,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0},'Exact fixed H8 budget and invariants')
        c._write_json(out/'all432_state_statistics.json',all_stats)
        c._write_json(out/'all72_cached_self15_statistics.json',all_cached)
        with (out/'all72_cached_self15_statistics.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_cached[0]));writer.writeheader()
            import json
            for row in all_cached:writer.writerow({key:json.dumps(value,allow_nan=False) if isinstance(value,(dict,list)) else value for key,value in row.items()})
        with (out/'all432_state_statistics.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_stats[0]));writer.writeheader();writer.writerows(all_stats)
        for value in m['sources'].values():checked(value)
        require(same(pin(protocol_path),m['protocol_pin']),'Protocol bytes unchanged')
        report={'schema':'full151105_public_self15_H8_donor_CPU_analysis:1','status':'COMPLETED_ALL3_PUBLIC_SELF15_FIXED_CONTROL_H8_NOT_POLICY_IMAGINATION','protocol':m['protocol_pin'],'input_pins':i,'checkpoint_pins':m['family']['pair']['checkpoints'],
            'counts':counts,'checks':checks,'passed':len(checks),'total':len(checks),'raw_rng_comparison':rng,'model_roles':list(MODEL_ROLES),'normalizer_host_before':host_before,'normalizer_host_after':host_after,
            'construction_receipt':construction,'model_invariants':invariants,'old152_fit_digests':m['fit_digests'],'donor_window_input_digest':data_before,'donor_self15_truth_evaluated':True,
            'donor_self15_reference':{'source_producer_report':i['donor_self15_producer_report'],'source_label_JSON':i['donor_self15_labels'],'source_saved_independent':i['donor_self15_saved_independent'],'unique_labels':300,'expanded_reference_rows':2288,'per_role_tables':24,'total_tables':72,'F64_score_matrix_calls':24,'positive_decision':'logit>0; ties negative','empty_positive_denominator':None},
            'teacher_forcing':False,'output_clamping_or_R_projection':False,'accuracy_qualified':False,'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'Actor_or_task_benefit_qualified':False,
            'scope':'Known A0/A2 donor lineages:286 overlapping windows, starts0..142, actual u through149, targets throughobs150. Per-h both x and dynamics state use own preceding prediction; xi46/zero174 fixed. Not Actor/memory/prior/survival or independent-trajectory replay. Raw R9 is dimensionless, not angle. H>1 cross-role self15 drift is matched-control rollout drift at different predicted states, not same-input head forgetting. No J/K truth or consistency loss, thresholds, model selection or CI.',
            'elapsed_seconds':time.perf_counter()-started,'outputs':{f.name:pin(f) for f in out.iterdir() if f.is_file()}}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        c._write_json(out/'failure.json',{'schema':'full151105_public_self15_H8_failure:1','status':'FAILED_PARTIAL_NO_ROW_FILTER_OR_TEACHER_REPAIR','phase':phase,'error':repr(error),'counts':counts,'accuracy_evaluated_from_partial':False,'automatic_retry':False});raise
    finally:
        for module in modules.values():module.close()
