"""SOURCE PREPARATION: three-component fixed-actual-control donor H8.

No sealed/new window values or models are touched by metadata defaults. Explicit
Root execution rechecks the OLD native GPU donor continuity/control authority,
then autoregresses all286 windows. This is not Actor imagination or training.
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from . import component_confirmation105_v1 as cf

c=cf.c;ROOT=c.ROOT
SCHEMA='full151105_three_component_H8_donor_CPU_protocol:1'
ROLE='three_component_fixed_control_H8_donor_state_error'
MODEL_ROLES=('initial_unmodified_A2','pointwise_final1000','pointwise_plus_response_final1000')
OUTPUT_PREFIX='full151105_three_component_H8_donor_CPU_'
PROPOSAL=ROOT/'docs/full151_component_H8_three_model_rollout_proposal_v1.md'
PROPOSAL_SHA='daf460f6614dda946300d6b57df13f0c60ca277c71ea0bd5c776a826d3b4d193'
INPUTS={'calibration_recipe','training_protocol','initial_component','initial_validation','pointwise_component','response_component',
        'final_checkpoint_saved_review','H1_config','H1_report','H1_matched_inputs','H1_prediction_arrays','H1_identities','H1_saved_independent25','prior_H8_contiguity'}
OWN={'consumer':ROOT/'src/full151_gradient/component_H8_donor105_v1.py',
     'CLI':ROOT/'scripts/diagnose_full151_component_H8_donor_cpu_v1.py'}
read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require


def validate_protocol(p,require_authorized=False):
    require(type(p) is dict and p.get('schema')==SCHEMA and p.get('role')==ROLE,'New H8 protocol/schema')
    for key in ('execution_authorized','CPU_run_authorized','GPU_run_authorized'):
        require(type(p.get(key)) is bool,'Explicit CPU authorization bool')
    require(p['GPU_run_authorized'] is False and p.get('model_roles')==list(MODEL_ROLES),'Three explicit component roles, CPU only')
    for key,value in (('cpu_threads',4),('horizon',8),('batch_windows',286),('starts_per_path',143)):
        require(type(p.get(key)) is int and p[key]==value,'Fixed '+key)
    require(p.get('output_prefix')==OUTPUT_PREFIX and p.get('accuracy_gate') is None,'Descriptive output/no accuracy gate')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS,'Exact14 semantic input aliases')
    require(type(p.get('source_pins')) is dict and p['source_pins'],'Source closure')
    require(type(p.get('proposal')) is dict,'Source proposal pin')
    if require_authorized:require(p['execution_authorized'] and p['CPU_run_authorized'],'Root actual CPU run authorization required')
    return p


def declaration():
    cf.no_backend()
    return {'schema':'full151105_three_component_H8_declaration:1','status':'SOURCE_PREPARATION_NOT_RUN',
            'protocol_schema':SCHEMA,'role':ROLE,'model_roles':list(MODEL_ROLES),'input_aliases':sorted(INPUTS),
            'budget':{'component_file_reads':3,'native_WM_constructors':3,'WM_forwards':24,'batch':286,'predictions':6864,'AD_optimizer_Actor_MJB_Env_physics_GPU':0},
            'source_pins':{k:pin(v) for k,v in OWN.items()},'actual_window_construction_model_FW':0}


def metadata_preflight(protocol_path,output=None):
    cf.no_backend();p=validate_protocol(read(protocol_path));i=p['inputs']
    require(same(p['proposal'],{'path':str(PROPOSAL),'sha256':PROPOSAL_SHA}),'Fixed exploratory source proposal')
    for v in i.values():checked(v)
    final=read(checked(i['final_checkpoint_saved_review']));c.allchecks(final)
    require(final['schema']=='full151105_WM_component_saved_checkpoint_CPU_review:1' and final['status']=='PASS_SERIALIZED_COMPONENT_HEALTH_NOT_ACCURACY'
            and final['iteration']==1000 and final['final_iteration_scope'] is True and final['engineering_qualified'] is True,'Actual final1000 component review')
    require(same(final['initial_component'],i['initial_component']) and same(final['protocol'],i['training_protocol']),'Actual component initial/protocol')
    for arm,alias in zip(c.ARMS,('pointwise_component','response_component')):require(same(final['checkpoints'][arm],i[alias]),'Fixed final artifact '+arm)
    training=read(checked(i['training_protocol']));recipe=read(checked(i['calibration_recipe']))
    require(training['recipe']==recipe['recipe']==c.RECIPE,'Original frozen calibration recipe')
    h1=read(checked(i['H1_report']));review=read(checked(i['H1_saved_independent25']));c.allchecks(review)
    require(review['schema']=='full151105_state_prediction452_saved_independent:1' and review['status']=='PASS_SAVED_ALL452_STATE_ERROR_REVIEW_NOT_MODEL_QUALIFICATION'
            and review['engineering_qualified'] is True and same(review['config'],i['H1_config']) and same(review['producer_report'],i['H1_report']),'Actual H1 saved25 source')
    require(h1['schema']=='full151105_state_prediction452_CPU_analysis:1' and h1['prediction_rows']==452 and same(h1['config'],i['H1_config']),'Completed old native452 input authority')
    run=checked(i['H1_report']).parent;manifest=read(run/'output_pins.json')
    for alias,name in (('H1_matched_inputs','matched_input_arrays.npz'),('H1_prediction_arrays','prediction_arrays.npz'),('H1_identities','group_identities452.json')):
        require(same(i[alias],manifest[name]) and checked(i[alias]).samefile(run/name),'Original H1 immutable artifact')
    prior=read(checked(i['prior_H8_contiguity']))
    require(prior['schema']=='full151105_H8_donor_saved_contiguity_source_feasibility:1' and prior['all_links_all58_bitexact'] is True
            and prior['unique_adjacent_links']==298 and prior['all_donor_four_controls_bitexact'] is True,'Existing source feasibility, not new H8 qualification')
    require(same(prior['input_pins']['matched_input_arrays.npz'],i['H1_matched_inputs']) and same(prior['input_pins']['group_identities452.json'],i['H1_identities']),'Same contiguity source')
    initial_receipt=read(checked(i['initial_validation']))
    require(same(initial_receipt['strict_loader']['checkpoint'],h1['input_pins']['fixed_A2_checkpoint']),'Initial baseline is same old H1 A2 WM source')
    sources=c.canonical_sources(list(p['source_pins'].values())+list(training['source_pins'].values())+
        list(read(checked(i['H1_config']))['source_pins'].values())+[pin(v) for v in OWN.values()])
    for v in OWN.values():require(any(same(pin(v),x) for x in p['source_pins'].values()),'Current own code not frozen in protocol')
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(OUTPUT_PREFIX) and not out.exists(),'Fresh scoped output')
    cf.no_backend()
    return {'status':'PASS_H8_SOURCE_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'sources':sources,'output':out,
            'actual_NPZ_decodes_window_construction_model_FW':0}


def assemble_donor_windows(arrays,identities,np):
    """Runtime recheck before building ANY window; no first300-row shortcut."""
    for key,shape in (('pre_x452_FP32',(452,58)),('target_x452_FP32',(452,58)),('actual_u452_FP32',(452,20)),('all_substep_u452x4x20_FP32',(452,4,20))):
        a=arrays[key];require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Original finite FP32 '+key)
    require(type(identities) is list and len(identities)==452,'All original group identities')
    mapping={}
    for row_id,row in enumerate(identities):
        require(type(row['group_id']) is int and row['group_id']==row_id,'Original group order')
        if row['mode']=='donor':
            key=(row['path'],row['k'])
            require(key not in mapping and row['path'] in ('A0','A2') and type(row['k']) is int and 0<=row['k']<150 and row['branch_id'] is None,'Unique native donor identity')
            require(row['terminal_observation']==row['k']+1 and row['valid_next_policy_decision']==(row['k']+1<150) and row['no_u150'] is True,'Native terminal boundary')
            mapping[key]=row_id
    require(set(mapping)=={(path,k) for path in ('A0','A2') for k in range(150)},'All300 donor rows, no availability repair')
    ids=np.array([[mapping[path,k] for k in range(150)] for path in ('A0','A2')],np.int64)
    pre=arrays['pre_x452_FP32'][ids];target=arrays['target_x452_FP32'][ids];u=arrays['actual_u452_FP32'][ids]
    links=target[:,:149].view(np.uint32)==pre[:,1:].view(np.uint32)
    controls=arrays['all_substep_u452x4x20_FP32'][ids].view(np.uint32)==np.broadcast_to(u[:,:,None,:],(2,150,4,20)).view(np.uint32)
    require(links.shape==(2,149,58) and links.all(),'Any donor target->nextPRE bit mismatch blocks all windows; no repair')
    require(controls.shape==(2,150,4,20) and controls.all(),'Any actual four-substep control mismatch blocks all windows')
    # Construction occurs only after every source continuity/control gate passed.
    starts=[];initial=[];truth=[];control=[];source_ids=[]
    for pi,path in enumerate(('A0','A2')):
        for k in range(143):
            wid=len(starts);starts.append({'window_id':wid,'path':path,'start_k':k,'action_phases':list(range(k,k+8)),
                'target_observations':list(range(k+1,k+9)),'end_observation':k+8,'no_u150':True})
            initial.append(pre[pi,k]);truth.append(target[pi,k:k+8]);control.append(u[pi,k:k+8]);source_ids.append(ids[pi,k:k+8])
    return {'initial_x286_FP32':np.asarray(initial,np.float32),'target286x8_FP32':np.asarray(truth,np.float32),
            'actual_u286x8_FP32':np.asarray(control,np.float32),'source_group_ids286x8':np.asarray(source_ids,np.int64),
            'donor_indices2x150':ids,'all298_links58_bitmask':links,'all300_control4_bitmask':controls},starts


def verify_xi(initial_xi,old_h1_xi,np,torch):
    require(isinstance(initial_xi,torch.Tensor) and initial_xi.device.type=='cpu' and initial_xi.dtype==torch.float32 and initial_xi.shape==(46,),'Component physical xi46')
    source=np.asarray(old_h1_xi)
    require(source.shape==(46,) and source.dtype==np.float32 and np.isfinite(source).all(),'Old H1 physical xi46')
    require(initial_xi.detach().contiguous().numpy().tobytes()==source.tobytes(),'Old native donor/three-component xi bit mismatch')


def make_initial_eval_component(initial,receipt,torch):
    """Initial namespace/WM key, never a fabricated final1000 or old105 payload."""
    require(initial.get('schema')==c.ARTIFACT_SCHEMA and 'WM' in initial and 'WM_state' not in initial
            and 'arm' not in initial and 'iteration' not in initial and 'format_version' not in initial,'Independent initial artifact role')
    require(c._tensor_digest(initial['WM'],torch)==receipt['initial_WM_sha256'] and c._finite_tree(initial['WM'],torch),'Unmodified initial native WM hash/finite state')
    require(initial['parent_checkpoint105']['sha256']==c.FROZEN_CHECKPOINT_SHA
            and same(initial['parent_checkpoint105'],receipt['strict_loader']['checkpoint']),'Original strict A2 parent source')
    require(c._tensor_digest({'xi':initial['physical_xi46']},torch)==receipt['xi46_sha256'],'Original physical xi digest')
    from dex_supermimic.rl.dex_supermimic.networks import WorldModel
    with torch.random.fork_rng(devices=[]):
        model=WorldModel(278,15,hidden=(512,512,512),amp=False);expected=model.state_dict();weights=initial['WM']
        require(set(weights)==set(expected) and all(isinstance(weights[k],torch.Tensor) and weights[k].device.type=='cpu' and weights[k].shape==v.shape and weights[k].dtype==v.dtype and bool(torch.isfinite(weights[k]).all()) for k,v in expected.items()),'Initial full native architecture/dtype')
        model.load_state_dict(weights,strict=True);model.eval().requires_grad_(False)
    require(c._tensor_digest(model.state_dict(),torch)==receipt['initial_WM_sha256'],'Initial strict ctor load exact')
    require(c._tensor_digest(c._norm_state(model),torch)==receipt['frozen_norm_sha256']
            and c._norm_config(model)==initial['normalizer_host_configuration']==receipt['normalizer_host_configuration'],'Initial frozen norm host/buffer identity')
    require(cf._bits(torch.maximum(model.delta_norm.std(),model.error_floor),initial['frozen_s52'],torch),'Initial native s52 exact')
    return model


def autoregress(model,initial_x,controls,xi,torch,np,counters,save_partial):
    """All286 simultaneous; actual call arguments saved at every horizon."""
    n=initial_x.shape[0];require(n==286 and initial_x.shape==(n,58) and controls.shape==(n,8,20),'Fixed H8 batch')
    result={'prediction58_FP32':np.full((n,8,58),np.nan,np.float32),'self15_logits_FP32':np.full((n,8,15),np.nan,np.float32),
            'state_input58_FP32':np.full((n,8,58),np.nan,np.float32),'dynamics_state58_FP32':np.full((n,8,58),np.nan,np.float32),
            'actual_call_u20_FP32':np.full((n,8,20),np.nan,np.float32),'attempted_mask':np.zeros((n,8),bool),'completed_mask':np.zeros((n,8),bool)}
    current=initial_x
    with torch.inference_mode():
        for horizon in range(8):
            u=controls[:,horizon];dynamics=torch.cat((current,xi[None].expand(n,-1),current.new_zeros((n,174))),-1)
            result['state_input58_FP32'][:,horizon]=current.numpy();result['dynamics_state58_FP32'][:,horizon]=dynamics[:,:58].numpy()
            result['actual_call_u20_FP32'][:,horizon]=u.numpy();result['attempted_mask'][:,horizon]=True
            before=c._tensor_digest({'x':current,'u':u,'dynamics':dynamics},torch)
            counters['WM_forward_attempted']+=1
            prediction,self15=model(current,u,dynamics);counters['WM_forward_returned']+=1
            require(prediction.shape==(n,58) and self15.shape==(n,15) and prediction.dtype==self15.dtype==torch.float32,'Raw native H8 output ABI')
            result['prediction58_FP32'][:,horizon]=prediction.numpy();result['self15_logits_FP32'][:,horizon]=self15.numpy()
            finite=torch.isfinite(prediction).all(-1)&torch.isfinite(self15).all(-1)
            result['completed_mask'][:,horizon]=finite.numpy();save_partial(result)
            require(c._tensor_digest({'x':current,'u':u,'dynamics':dynamics},torch)==before,'WM mutated call inputs')
            require(bool(finite.all()),'Nonfinite H8 row stops whole experiment; no teacher-forcing/subset repair')
            counters['predicted_rows']+=n
            current=prediction  # Both x and next dynamics[:58] use this tensor.
    return result


def error_statistics(prediction,truth,starts,np):
    error=prediction.astype(np.float64)-truth.astype(np.float64);rows=[]
    for horizon in range(8):
        for path in ('ALL','A0','A2'):
            ids=np.arange(286) if path=='ALL' else np.array([r['window_id'] for r in starts if r['path']==path])
            for block,(lo,hi,unit) in cf.RAW58.items():
                z=error[ids,horizon,lo:hi];item={'horizon':horizon+1,'path':path,'block':block,'unit':unit,'window_count':len(ids),
                    'component_RMSE':float(np.sqrt(np.mean(z*z))),'MAE':float(np.mean(np.abs(z))),'maximum_absolute':float(np.max(np.abs(z)))}
                if block in ('p','v','omega','R_raw9'):item['vector_or_Frobenius_RMS']=float(np.sqrt(np.mean(np.sum(z*z,axis=-1))))
                else:item['vector_or_Frobenius_RMS']=''
                rows.append(item)
    return error,rows


def run_analysis(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for key,value in m['sources'].items():z.write(checked(value),key)
    counters={'component_file_reads':0,'native_WM_constructors':0,'WM_forward_attempted':0,'WM_forward_returned':0,'predicted_rows':0,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0}
    phase='old_native_saved_donor_inputs'
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        torch.set_num_threads(4)
        rng=c._rng(torch,np);rng_sha=c._rng_digest(rng);cf.save_raw_rng(out,'before',rng,np)
        with np.load(checked(i['H1_matched_inputs']),allow_pickle=False) as f:
            old={k:f[k].copy() for k in ('pre_x452_FP32','target_x452_FP32','actual_u452_FP32','all_substep_u452x4x20_FP32')}
        data,starts=assemble_donor_windows(old,read(checked(i['H1_identities'])),np)
        np.savez_compressed(out/'matched_donor_windows286.npz',**data);c._write_json(out/'all286_window_identities.json',starts)
        with np.load(checked(i['H1_prediction_arrays']),allow_pickle=False) as f:old_xi=f['physical_xi46_FP32'].copy()
        phase='trusted_component_files'
        initial=torch.load(checked(i['initial_component']),map_location='cpu',weights_only=False);counters['component_file_reads']+=1
        payloads={}
        for arm,alias in zip(c.ARMS,('pointwise_component','response_component')):
            payloads[arm]=torch.load(checked(i[alias]),map_location='cpu',weights_only=False);counters['component_file_reads']+=1
            require(same(payloads[arm]['protocol'],i['training_protocol']),'Fixed final run protocol')
        receipt=read(checked(i['initial_validation']));verify_xi(initial['physical_xi46'],old_xi,np,torch)
        x0=torch.from_numpy(data['initial_x286_FP32']);u=torch.from_numpy(data['actual_u286x8_FP32']);xi=initial['physical_xi46']
        memory_before=c._tensor_digest({'initial_x':x0,'controls':u,'xi':xi},torch);outputs={};invariants={};all_stats=[]
        for role in MODEL_ROLES:
            phase=role
            if role==MODEL_ROLES[0]:model=make_initial_eval_component(initial,receipt,torch)
            else:
                arm=c.ARMS[MODEL_ROLES.index(role)-1];model=cf.make_eval_component(payloads[arm],initial,arm,torch)
            counters['native_WM_constructors']+=1
            state_before=c._tensor_digest(model.state_dict(),torch);norm_before=c._tensor_digest(c._norm_state(model),torch);host_before=c._norm_config(model)
            def save_partial(value):np.savez_compressed(out/(role+'_rollout_partial.npz'),**value)
            result=autoregress(model,x0,u,xi,torch,np,counters,save_partial)
            error,stats=error_statistics(result['prediction58_FP32'],data['target286x8_FP32'],starts,np)
            np.savez_compressed(out/(role+'_rollout_final.npz'),**result,error58_F64=error)
            for row in stats:row['model_role']=role
            all_stats.extend(stats)
            invariants[role]={'WM_sha256_before':state_before,'WM_sha256_after':c._tensor_digest(model.state_dict(),torch),
                'norm_sha256_before':norm_before,'norm_sha256_after':c._tensor_digest(c._norm_state(model),torch),
                'normalizer_host_configuration_unchanged':c._norm_config(model)==host_before,
                'artifact_role':role,'not_native105_resume_qualification':True}
            require(invariants[role]['WM_sha256_before']==invariants[role]['WM_sha256_after']
                    and invariants[role]['norm_sha256_before']==invariants[role]['norm_sha256_after']
                    and invariants[role]['normalizer_host_configuration_unchanged'],'Frozen H8 model mutated')
            outputs[role]=pin(out/(role+'_rollout_final.npz'));del model
        after=c._rng(torch,np);cf.save_raw_rng(out,'after',after,np)
        require(c._rng_digest(after)==rng_sha,'RNG changed, no manual repair')
        memory_after=c._tensor_digest({'initial_x':x0,'controls':u,'xi':xi},torch);require(memory_after==memory_before,'Input/xi bytes changed')
        for value in i.values():checked(value)
        for value in m['sources'].values():checked(value)
        require(same(pin(protocol_path),m['protocol_pin']),'H8 protocol changed')
        require(counters['component_file_reads']==3 and counters['native_WM_constructors']==3 and counters['WM_forward_attempted']==counters['WM_forward_returned']==24 and counters['predicted_rows']==6864,'Complete fixed H8 budget')
        with (out/'all_models_horizons_units.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_stats[0]));writer.writeheader();writer.writerows(all_stats)
        report={'schema':'full151105_three_component_H8_donor_CPU_analysis:1','status':'COMPLETED_EXPLORATORY_FIXED_CONTROL_H8_NOT_POLICY_IMAGINATION',
            'protocol':m['protocol_pin'],'input_pins':i,'counts':counters,'model_invariants':invariants,'outputs':outputs,'statistics':all_stats,
            'raw_RNG_saved':True,'RNG_unchanged_without_repair':True,'input_memory_digests':{'before':memory_before,'after':memory_after},
            'runtime_source_gates':{'all298_links_full58_bitexact':True,'all300_actual_u_four_substeps_bitexact':True,'component_oldH1_xi46_bitexact':True},
            'all286_windows_all8_horizons_used':True,'teacher_forcing':False,'output_clamping_or_R_projection':False,
            'accuracy_qualified':False,'physical_gradient_qualified':False,'physical_reference_qualified':False,'whole_hidden_state_qualified':False,
            'source_lineage_N':1,'independent_ego_N':0,'scope':'Exploratory known donor paths/overlapping windows. Fixed executed controls, no Actor/memory/prior/survival replay; zero174 inherited approximation; CPU FP32 shadow, not native GPUAMP. Initial baseline is its own initial.WM artifact, not fabricated final1000. R9 is raw element/Frobenius error, not angle. No scientific pass threshold/CI/model winner.',
            'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        try:c._write_json(out/'failure.json',{'schema':'full151105_three_component_H8_failure:1','status':'FAILED_PARTIAL_NO_ROW_FILTER_OR_TEACHER_REPAIR',
            'phase':phase,'error':repr(error),'counts':counters,'protocol':m['protocol_pin'],'accuracy_evaluated_from_partial':False})
        except BaseException as save_error:
            try:error.add_note('Failure evidence save failed: '+repr(save_error))
            except BaseException:pass
        raise
