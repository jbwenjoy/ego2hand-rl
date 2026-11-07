"""Independent saved full151 format105 health; never construct a native object.

Separate schema2 learning allows A1L/P_ONLY/A2 with the same full151 inputs.
A2/P_ONLY require a new pinned exploratory declared-H123 spec and evidence. Health covers serialization, from-scratch and run evidence,
never learned dynamics/contact gradients or physical task success. Indexed
checkpoint prefixes and a completed model_final run have distinct scopes.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import argparse,hashlib,json,math,sys,zipfile
import numpy as np
import torch

FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
VARIANT='legacy_H123_full151_SCP8_knownrough_v1'
MODE='rough_observation_reference_full151_v1'
ROOT=Path(__file__).resolve().parents[1]
NFRAME=151
NINTERVAL=150
MODEL_SHA='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd'
CONTEXT_SHA='5edd125973015c034d45e47092786b7fac105a06af52864582afa08d8660b89a'
ERROR_KEYS={'cache_actual','cache_expected','residual_before_after','replay_u_actual','replay_c_residual'}
CONFIG_SCHEMA='full151_knownrough_native_training_run:2'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'sha256':digest(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def loadnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def checked(value):
    if set(value)!={'path','sha256'} or not Path(value['path']).is_absolute() or digest(value['path'])!=value['sha256']:raise ValueError('Pinned source/input missing or changed')
    return Path(value['path'])
def finite(v):
    if torch.is_tensor(v):return bool(torch.isfinite(v).all())
    if isinstance(v,np.ndarray):return bool(np.isfinite(v).all()) if v.dtype.kind in 'fiucb' else True
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,(tuple,list)):return all(finite(x) for x in v)
    return math.isfinite(v) if isinstance(v,float) else True
def pure(v):
    if isinstance(v,dict):return {str(k):pure(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [pure(x) for x in v]
    if isinstance(v,np.ndarray):return pure(v.tolist())
    if isinstance(v,np.generic):return pure(v.item())
    if isinstance(v,Path):return str(v)
    if torch.is_tensor(v):return pure(v.detach().cpu().tolist())
    if isinstance(v,float) and not math.isfinite(v):return repr(v)
    return v
def json_write(path,value):Path(path).write_text(json.dumps(pure(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def complete_rows(path):
    data=Path(path).read_bytes();end=data.rfind(b'\n')+1
    return [json.loads(x) for x in data[:end].splitlines()],data[:end]

def effective_config_equal(saved_config,effective):
    # Tuple/list share the JSON array ABI. Canonical bytes preserve JSON scalar
    # types, unlike Python dict equality where True==1 and 1==1.0.
    return canonical(saved_config)==canonical(effective)

def optimizer_counts_match(observed, expected):
    """Only absent counters whose declared expected count is zero may mean zero.

    Native hooks populate defaultdict keys lazily during the WM-only prefix.
    Unknown keys, booleans, nonintegers, negatives and absent positive counts
    remain invalid. This does not normalize Adam state or update budgets.
    """
    names={'wm','fail_head','actor','critic'}
    if type(expected) is not dict or set(expected)!=names:
        return False
    if not all(type(v) is int and v>=0 for v in expected.values()):
        return False
    if type(observed) is not dict or not set(observed)<=names:
        return False
    if not all(type(v) is int and v>=0 for v in observed.values()):
        return False
    if any(expected[k]>0 and k not in observed for k in names):
        return False
    return all(observed.get(k,0)==expected[k] for k in names)


def expected_collect_steps(cfg,iteration):
    out=[max(cfg['warmup_steps'],cfg['wm_window'])];acc=0.
    for _ in range(iteration):acc+=1./cfg['utd'];n=int(acc);acc-=n;out.append(n)
    return out

def budget_counts(cfg,iteration):
    if type(iteration) is not int or not 1<=iteration<=cfg['max_iterations']:raise ValueError('Checkpoint integer iteration outside declared budget')
    active=max(0,iteration-cfg['wm_only_iters'])
    return {'wm':iteration*cfg['wm_batches'],'fail_head':iteration*cfg['wm_batches'],'actor':active,'critic':active*cfg['critic_epochs']*cfg['critic_minibatches']},cfg['num_envs']*sum(expected_collect_steps(cfg,iteration))

def new105_header_ok(saved,cfg):
    return (type(saved.get('format_version')) is int and saved['format_version']==105
        and cfg.get('schema')==CONFIG_SCHEMA and cfg.get('role') in ('engineering_smoke','primary_matched')
        and cfg.get('method') in ('A1L','P_ONLY','A2')
        and ((cfg['method']=='A1L' and cfg.get('contact_binding') is None) or (cfg['method']!='A1L' and type(cfg.get('contact_binding')) is dict and set(cfg['contact_binding'])=={'path','sha256'}))
        and not any(k in saved for k in ('rough_controller','rough_runtime_evidence','rough_capacity_evidence','rough_initial_parameter_fingerprints','c2_controller'))
        and all(type(saved.get(k)) is dict and bool(saved[k]) for k in ('full151_controller','full151_runtime_evidence','full151_capacity_evidence','full151_initial_parameter_fingerprints','full151_prior_evidence')))

def is_final_artifact(checkpoint,run,iteration,max_iterations):
    return (type(iteration) is int and type(max_iterations) is int and iteration==max_iterations
        and Path(checkpoint).resolve()==(Path(run)/'train/model_final.pt').resolve())

def qualified_final_scope(checkpoint,run,iteration,max_iterations,checks):
    names=('actual_final_completion_not_task_quality','worker_training_completion_marker_schema_config',
           'training_wrapper_return_final_capacity','worker_terminal_completed',
           'capacity_final_scope_run_final_once','capacity_final_raw_OK_no_overflow_or_unknown')
    return is_final_artifact(checkpoint,run,iteration,max_iterations) and all(checks.get(k) is True for k in names)

def expected_shapes(hidden=(512,512,512)):
    def mlp(dims,offset=2):
        d={}
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):d[f'net.{offset*i}.weight']=[b,a];d[f'net.{offset*i}.bias']=[b]
        return d
    shapes={'actor':mlp([155,512,256,128,20]),'fail_head':mlp([426,256,128,1]),'critic':mlp([301,512,512,256,128,1]),'wm':{}}
    for n,d in [('actor',155),('fail_head',426),('critic',301)]:
        norm=['_mean','_var','_std'] if n=='actor' else ['mean','var']
        shapes[n].update({f'norm.{k}':([1,d] if n=='actor' else [d]) for k in norm});shapes[n]['norm.count']=[]
    prev=298
    for i,d in enumerate(hidden):
        shapes['wm'].update({f'net.{3*i}.weight':[d,prev],f'net.{3*i}.bias':[d],f'net.{3*i+1}.weight':[d],f'net.{3*i+1}.bias':[d]});prev=d
    shapes['wm'].update({f'net.{3*len(hidden)}.weight':[67,prev],f'net.{3*len(hidden)}.bias':[67],'error_floor':[52],'norm.mean':[298],'norm.var':[298],'norm.count':[],'delta_norm.mean':[52],'delta_norm.var':[52],'delta_norm.count':[]})
    return shapes

def capacity_checks(value,kind):
    """Recompute OR/unknown from raw records, never infer worlds from flagshape."""
    checks={};records=value.get('records',[]);mask=0;unknown=0
    checks['schema_and_backend']=value.get('schema')=='rough_prefix_reset_aware_capacity_observation:1' and value.get('backend_field')=='env.sim.wp_data.overflow'
    checks['records_nonempty_contiguous']=bool(records) and [r.get('sequence') for r in records]==list(range(1,len(records)+1))
    every=True
    for r in records:
        valid=True;rawmask=0
        try:
            dtype=np.dtype(r['raw_flags_dtype']);flags=np.asarray(r['raw_flags_values'],dtype=dtype)
            valid=dtype.kind in 'iu' and flags.size>0 and list(flags.shape)==r['raw_flags_shape']
            if valid:
                for flag in flags.reshape(-1):rawmask|=int(flag)
            valid=valid and r.get('read_status')==('OK' if rawmask==0 else 'OVERFLOW') and r.get('raw_bitmask')==rawmask
        except (ValueError,TypeError,KeyError):valid=False
        if r.get('read_status')=='UNKNOWN':unknown+=1
        elif valid:mask|=rawmask
        else:every=False
        expected_status='OVERFLOW' if mask else 'UNKNOWN' if unknown else 'OK'
        every=every and valid and r.get('read_attempts_for_record')==1 and r.get('cumulative_OR_bitmask')==mask and r.get('cumulative_unknown_reads')==unknown and r.get('overall_status')==expected_status and r.get('backend_field')=='env.sim.wp_data.overflow' and r.get('flag_shape_is_world_count') is False and r.get('flags_cleared_by_observer') is False
    checks['raw_record_OR_status_and_no_clear']=every
    checks['raw_OK_no_overflow_or_unknown']=mask==0 and unknown==0 and value.get('cumulative_OR_bitmask')==0 and value.get('unknown_read_count')==0 and value.get('status')=='OK'
    checks['record_counts']=value.get('record_count')==len(records) and value.get('before_reset_read_count')==sum(r.get('event')=='before_reset' for r in records) and value.get('final_read_count')==sum(r.get('event')=='final' for r in records)
    checks['fixed_capacity']=value.get('configured_fixed_capacities')=={'nconmax':192,'njmax':4096} and value.get('flag_shape_is_world_count') is False
    if kind=='checkpoint_prefix':
        checks['scope_prefix_not_final']=value.get('evidence_kind')=='checkpoint_prefix' and value.get('prefix_at_checkpoint_capacity_qualified') is True and value.get('entire_run_finalized') is False and value.get('finalized') is False and value.get('engineering_capacity_qualified') is False and value.get('final_read_count')==0 and records[-1].get('event')=='checkpoint'
    else:
        checks['scope_run_final_once']=value.get('entire_run_finalized') is True and value.get('finalized') is True and value.get('engineering_capacity_qualified') is True and value.get('final_read_count')==1 and records[-1].get('event')=='final'
    return checks,{'raw_OR_bitmask':mask,'raw_unknown_read_count':unknown,'record_count':len(records),'before_reset_read_count':sum(r.get('event')=='before_reset' for r in records)}

def policy_checks(evidence,cfg):
    p=evidence.get('policy',{});checks={}
    start=np.asarray(p.get('start_phases',[]));mask=np.asarray(p.get('active_mask_H_B',[]));h=cfg['horizon'];b=cfg['pi_batch']
    checks['H8_shapes_and_integer_phases']=start.shape in ((b,1),(b,)) and start.dtype.kind in 'fiu' and mask.shape==(h,b) and mask.dtype.kind=='b' and finite(start) and bool(((start>=0)&(start<150)&(start==np.floor(start))).all())
    if checks['H8_shapes_and_integer_phases']:
        phase=start.reshape(-1);expected=phase[None,:]+np.arange(h)[:,None]<150;last=phase+h<150
        checks['terminal_active_mask_independent']=np.array_equal(mask,expected)
        checks['critic_norm_eligible_independent']=type(p.get('critic_norm_eligible')) is int and p['critic_norm_eligible']==int(expected.sum()+last.sum())
        counts=p.get('critic_loss_minibatch_active_counts',[])
        checks['critic_minibatch_mask_counts']=len(counts)==cfg['critic_epochs']*cfg['critic_minibatches'] and all(type(c) is int and 0<=c<=math.ceil(h*b/cfg['critic_minibatches']) for c in counts) and sum(counts)==cfg['critic_epochs']*int(expected.sum())
    else:checks.update(terminal_active_mask_independent=False,critic_norm_eligible_independent=False,critic_minibatch_mask_counts=False)
    checks['actual_policy_call_contract']=p.get('optimizer_actor_calls')==1 and p.get('optimizer_critic_calls')==cfg['critic_epochs']*cfg['critic_minibatches'] and p.get('critic_total_candidate_rows')==h*b and p.get('hard_mask_rule')=='phase<150 only; physical soft termination retains native survival semantics'
    w=evidence.get('latest_window',{})
    checks['validated_native_window_summary']=all(type(w.get(k)) is int and w[k]>=0 for k in ('valid_window_entries','candidate_window_entries','terminal_valid_entries','reset_suffix_entries_masked')) and w.get('candidate_window_entries',-1)==w.get('valid_window_entries',0)+w.get('reset_suffix_entries_masked',0) and w.get('terminal_valid_entries',10**9)<=w.get('valid_window_entries',-1)
    return checks

def actualu_checks(run,saved,baseline):
    c=saved['full151_controller'];marker=c['latest_actualu_evidence'];last=marker['collect_call'];rows,raw=complete_rows(run/'full151_actualu_evidence.jsonl');prefix=rows[:last];steps=expected_collect_steps(saved['config'],saved['iteration']);checks={}
    checks['checkpoint_prefix_complete_and_contiguous']=type(last) is int and last==saved['iteration']+1 and len(prefix)==last and [r.get('collect_call') for r in prefix]==list(range(1,last+1))
    checks['marker_equals_last_prefix_row']=bool(prefix) and prefix[-1]==marker and marker['iteration']==max(0,saved['iteration']-1)
    checks['per_call_steps_worlds_and_total']=len(prefix)==len(steps) and all(r['steps']==n and r['worlds']==saved['config']['num_envs'] and r['real_transitions']==n*r['worlds'] for r,n in zip(prefix,steps)) and sum(r['real_transitions'] for r in prefix)==saved['sim_steps']
    checks['per_call_native_integer_fields_and_iteration']=all(all(type(r.get(k)) is int for k in ('collect_call','iteration','steps','worlds','real_transitions','phase_min','phase_max','inactive_real_actions')) and r['iteration']==max(0,r['collect_call']-2) for r in prefix)
    checks['allbatch_exact_finite_phase0to149']=all(r.get('all_batch_exact') is True and r.get('all_finite') is True and r.get('inactive_real_actions')==0 and 0<=r.get('phase_min',-1)<=r.get('phase_max',99)<150 and len(r.get('phase_histogram',[]))==150 and sum(r['phase_histogram'])==r['real_transitions'] and all(type(x) is int and x>=0 for x in r['phase_histogram']) for r in prefix)
    checks['all5_mismatches_zero']=all(set(r.get('max_abs_errors',{}))==ERROR_KEYS and all(type(x) in (float,int) and math.isfinite(x) and x==0 for x in r['max_abs_errors'].values()) for r in prefix)
    checks['clamped_count_accounting']=all(type(r.get('clamped_components')) is int and 0<=r['clamped_components']<=r['real_transitions']*20 for r in prefix)
    lower=np.array(c['target_lower_rad'],np.float32);upper=np.array(c['target_upper_rad'],np.float32);low=lower+np.float32(.02);high=upper-np.float32(.02);traces={}
    for call in [i for i in range(1,last+1) if i in (1,2,3) or i%100==0]:
        p=run/f'full151_actualu_trace_{call:05d}.npz';a=loadnp(p);traces[p.name]=digest(p);shape=(min(4,prefix[call-1]['steps']),min(4,prefix[call-1]['worlds']),20)
        checks[f'trace{call}_shapes_finite']=set(a)=={'phase','baseline','proposed','residual_before_filter','cached','actual','residual','replay_u','replay_c'} and all(a[k].shape==shape and a[k].dtype==np.float32 and np.isfinite(a[k]).all() for k in a if k!='phase') and a['phase'].shape==shape[:2] and a['phase'].dtype.kind in 'iu'
        phase=a['phase'];expected=np.clip(np.clip(a['proposed'],lower,upper),low,high)
        checks[f'trace{call}_projection_actual_replay']=np.array_equal(expected,a['cached']) and np.array_equal(a['cached'],a['actual']) and np.array_equal(a['actual'],a['replay_u']) and np.array_equal(a['residual_before_filter'],a['residual']) and np.array_equal(a['residual'],a['replay_c']) and bool(((a['actual']>=low)&(a['actual']<=high)).all())
        checks[f'trace{call}_baseline_phase_residual']=bool(((phase>=0)&(phase<150)).all()) and np.array_equal(a['baseline'],baseline[phase]) and np.array_equal(a['proposed'],a['baseline']+a['residual_before_filter'])
    # Hash onlycomplete prefix throughcheckpoint; later append isnotdrift.
    lines=raw.splitlines(keepends=True);prefixraw=b''.join(lines[:last])
    histogram=np.sum([r['phase_histogram'] for r in prefix],axis=0,dtype=np.int64).tolist()
    return checks,{'last_collect_call':last,'prefix_rows':len(prefix),'prefix_sha256':hashlib.sha256(prefixraw).hexdigest(),'prefix_byte_count':len(prefixraw),'real_transitions':sum(r['real_transitions'] for r in prefix),'sample_trace_sha256':traces,'later_complete_rows_ignored':len(rows)-last,'phase_hist150':histogram,'latest_collect_phase_hist150':prefix[-1]['phase_histogram'],'observed_phases':[i for i,n in enumerate(histogram) if n],'unobserved_phases':[i for i,n in enumerate(histogram) if not n],'phase149_rows':histogram[149]},prefixraw

def initial_checks(saved,run,result,*,require_result_delta=True):
    raw=torch.load(run/'initial_parameters.pt',map_location='cpu',weights_only=False);finger=read(run/'initial_parameters.json');checks={};recomputed={};delta={}
    checks['initial_raw_schema_CPU_finite']=raw.get('schema')=='full151_native_initial_parameters:1' and set(raw.get('modules',{}))==set(saved['modules']) and finite(raw) and all(v.device.type=='cpu' for m in raw['modules'].values() for v in m['parameters'])
    checks['initial_capture_source_pins']=bool(raw.get('source_pins')) and all(checked(p) for p in raw['source_pins'].values()) and raw.get('source_pins')==saved.get('full151_capacity_evidence',{}).get('source_pins') and raw.get('capture_point')=='after_native_DexRunner_init_before_first_collect_or_optimizer'
    for name,m in raw['modules'].items():
        values=m['parameters'];names=m['parameter_names'];h=hashlib.sha256();squared=0.;valid=len(values)==len(names)==len(set(names)) and set(names)=={k for k in saved['modules'][name] if k.startswith('net.') and k.endswith(('.weight','.bias'))}
        valid=valid and m['shapes']==[list(v.shape) for v in values] and m['dtypes']==[str(v.dtype) for v in values]
        for key,v in zip(names,values):
            h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes());final=saved['modules'][name][key]
            valid=valid and final.shape==v.shape and final.dtype==v.dtype
            squared+=float((final-v).double().square().sum())
        recomputed[name]={'sha256':h.hexdigest(),'tensors':len(values),'parameters':sum(v.numel() for v in values)};delta[name]=squared**.5
        checks[f'initial_{name}_names_shapes_and_hash']=valid and recomputed[name]==finger.get(name)==saved.get('full151_initial_parameter_fingerprints',{}).get(name)
        if require_result_delta:checks[f'initial_{name}_delta_independent']=name in result.get('parameter_delta_l2',{}) and math.isclose(delta[name],result['parameter_delta_l2'][name],rel_tol=1e-10,abs_tol=1e-12)
    active=max(0,saved['iteration']-saved['config']['wm_only_iters'])
    checks['actual_actor_WM_critic_change_matches_training_phase']=delta.get('wm',0)>0 and all((delta.get(k,0)>0) if active else (delta.get(k,-1)==0) for k in ('actor','critic'))
    return checks,{'independently_recomputed_fingerprints':recomputed,'independent_final_parameter_delta_l2':delta,'capture_point':raw.get('capture_point'),'raw_initial_tensor_artifact':pin(run/'initial_parameters.pt'),'scope':'Raw CPU constructor-return parameters hashed against later native-before-learning fingerprint; source rejects allresume/modelwarmstart. Crossarm same-seed comparison remains separate.'}

PRIOR_COUNTERS=('policy_update_calls','objective_forward_calls','active_rows','inactive_rows','dt_applications')


def prior_counter_checks(evidence,cfg,effective,iteration,log_rows,latest_policy=None):
    """Checkpoint-prefix counters; later live log appends are outside scope.

    Whole-prefix masks are source-reported summaries. Only the latest H8 mask
    is saved in runtime evidence, permitting independent last-call counts.
    """
    checks={};method=cfg.get('method');policy_calls=max(0,iteration-effective['wm_only_iters'])
    counts={k:evidence.get(k) for k in PRIOR_COUNTERS};none=method=='A1L'
    checks['schema_method_native_integer_counters']=(evidence.get('schema')=='full151_imagination_prior_checkpoint_evidence:1'
        and method in ('A1L','P_ONLY','A2') and evidence.get('method')==method and all(type(v) is int and v>=0 for v in counts.values()))
    checks['imagination_only_dt_once_and_no_logging_error']=(evidence.get('dt_s')==.02 and evidence.get('imagined_soft_required') is True
        and evidence.get('real_collect_shaped') is False and evidence.get('WMfit_shaped') is False and evidence.get('logging_errors')==[]
        and evidence.get('dead_end_rule')=='unchanged native full151 phase<150 hardmask after reward')
    checks['method_spec_metadata_attachment']=((evidence.get('status')=='NONE' and evidence.get('spec_pin') is None and evidence.get('objective_metadata') is None and cfg.get('contact_binding') is None)
        if none else (type(evidence.get('objective_metadata')) is dict and bool(evidence['objective_metadata']) and evidence.get('spec_pin')==cfg.get('contact_binding') and type(cfg.get('contact_binding')) is dict))
    expected_calls=0 if none else policy_calls
    checks['counter_H8_policy_and_external_dt_once']=(counts.get('policy_update_calls')==expected_calls
        and counts.get('objective_forward_calls')==expected_calls*effective['horizon']
        and counts.get('dt_applications')==expected_calls*effective['horizon']
        and type(counts.get('active_rows')) is int and type(counts.get('inactive_rows')) is int
        and counts['active_rows']>=0 and counts['inactive_rows']>=0
        and counts['active_rows']+counts['inactive_rows']==expected_calls*effective['horizon']*effective['pi_batch'])
    prefix=log_rows[:expected_calls]
    if none:
        checks['A1L_no_prior_policy_log']=not log_rows
        checks['A1L_all_prior_counters_zero']=all(v==0 for v in counts.values())
        return checks,{'scope':'A1L no objective/reward hook; all prior counters0','prefix_policy_calls':0,'later_complete_rows_ignored':0}
    checks['policy_log_checkpoint_prefix_complete']=len(prefix)==expected_calls
    cumulative={k:0 for k in PRIOR_COUNTERS};valid=True
    for index,row in enumerate(prefix):
        scalar_keys=('policy_update_call','iteration','objective_forward_calls','active_rows','inactive_rows','dt_applications')
        valid=valid and row.get('schema')=='full151_imagination_prior_policy_log:1' and all(type(row.get(k)) is int and row[k]>=0 for k in scalar_keys)
        valid=valid and row.get('complete') is True and row.get('policy_update_call')==index+1 and row.get('iteration')==effective['wm_only_iters']+index
        valid=valid and row.get('method')==method and row.get('dt_s')==.02
        valid=valid and row.get('objective_forward_calls')==effective['horizon'] and row.get('dt_applications')==effective['horizon']
        if not all(type(row.get(k)) is int for k in ('active_rows','inactive_rows')):
            valid=False;continue
        valid=valid and row['active_rows']>=0 and row['inactive_rows']>=0 and row['active_rows']+row['inactive_rows']==effective['horizon']*effective['pi_batch']
        cumulative['policy_update_calls']+=1
        for k in PRIOR_COUNTERS[1:]:cumulative[k]+=row.get(k,0)
        valid=valid and all(type(row.get('cumulative',{}).get(k)) is int and row['cumulative'][k]==cumulative[k] for k in PRIOR_COUNTERS)
        valid=valid and all(canonical(row.get('cumulative',{}).get(k))==canonical(evidence[k]) for k in set(evidence)-set(PRIOR_COUNTERS))
        means=row.get('active_means',{})
        valid=valid and type(means) is dict and 'prior/bonus_unscaled' in means and finite(means) and all(type(v) in (int,float) for v in means.values())
        if row['active_rows']==0:valid=valid and all(v==0 for v in means.values())
    checks['policy_log_H8_counts_complete_cumulative_and_finite']=valid
    checks['policy_log_matches_checkpoint_cumulative']=cumulative==counts
    if expected_calls:
        mask=np.asarray((latest_policy or {}).get('active_mask_H_B',[]))
        checks['latest_prior_active_rows_matches_saved_H8_mask']=(mask.dtype.kind=='b' and mask.shape==(effective['horizon'],effective['pi_batch'])
            and prefix[-1]['active_rows']==int(mask.sum()) and prefix[-1]['inactive_rows']==int(mask.size-mask.sum()))
    else:checks['WM_only_prior_NOT_RUN_not_actor_quality']=counts=={k:0 for k in PRIOR_COUNTERS}
    return checks,{'prefix_policy_calls':expected_calls,'later_complete_rows_ignored':len(log_rows)-len(prefix),'checkpoint_counters':counts,
        'scope':'Prefix logs are source-reported masks/counts. Latest saved H8 mask independently reconstructs last-call active/inactive rows. No raw whole-prefix imagined state tensor exists.'}


def objective_metadata_checks(metadata,method,cfg,spec,manifest):
    """Strict data/scientific bindings without constructing an objective."""
    checks={}
    checks['reference_and_target_pins']=(metadata.get('input_bundle')==cfg['input_bundle'] and metadata.get('binding_manifest')==spec['binding_manifest']
        and metadata.get('target')==manifest['target'] and metadata.get('target_receipt')==spec['target_receipt'] and metadata.get('independent_target_review')==spec['independent_target_review'])
    checks['spec_canonical_checksum_and_new_enrollment']=(metadata.get('objective_spec_schema')==spec.get('schema') and metadata.get('objective_spec_sha256')==hashlib.sha256(canonical(spec).encode()).hexdigest()
        and metadata.get('execution_scope')=='EXPLORATORY_MATCHED_TRAINING' and metadata.get('CPU_authorized') is True and metadata.get('training_authorized') is True
        and metadata.get('GPU_authorized')==spec.get('GPU_authorized'))
    checks['new_full151_window_identity']=(metadata.get('retarget_variant_id')==VARIANT and metadata.get('contact_intent_variant_id')=='source_rigid_H123_surface_projection_all151_v1'
        and metadata.get('source_window_half_open')==[200,351] and metadata.get('frame_count')==151 and metadata.get('active_real_imagination_intervals')==[0,150])
    checks['nonqualification_retained']=(all(metadata.get(k) is False for k in ('human_contact_observation_known','source_confidence_known','target_defined_from_policy_contacts','robotq_used_to_define_targets',
        'qualified_contact_predictor','forceclosure_qualified','dynamics_gradient_qualified','physical_reference_qualified'))
        and metadata.get('local_proxy_qualification')=='FAILED_RETAINED_EXPLORATORY_ONLY' and metadata.get('compiled_precision_qualification')=='FAILED_RETAINED_NOT_REVALIDATED')
    checks['same_frozen_penalty_external_dt']=(metadata.get('dt_applied_inside_module') is False and metadata.get('caller_dt_s')==.02
        and metadata.get('penetration_allowance_m')==.003 and metadata.get('penetration_normalization_m')==.001)
    checks['method_formula_support_separation']=((metadata.get('formula')=='-.1*penetration_prior' and metadata.get('support_coefficient')==0. and metadata.get('penetration_coefficient')==.1
        and metadata.get('scoring',{}).get('support_coefficient')==0. and metadata.get('support_score_is_diagnostic_only') is True)
        if method=='P_ONLY' else (metadata.get('formula')=='phase_weight*official_support_match-.1*penetration_prior' and metadata.get('scoring',{}).get('support_coefficient')==1.))
    checks['fixed_64_training_512_target_basis']=(metadata.get('basis',{}).get('training_directions')==64 and metadata.get('basis',{}).get('evaluation_directions')==512
        and metadata.get('basis',{}).get('training_indices')==list(range(0,512,8)))
    checks['unknown_attraction_only_no_finger_penalty']=(metadata.get('unknown_masks_attraction_only') is True and metadata.get('scoring',{}).get('no_finger_identity_or_extra_contact_penalty') is True
        and metadata.get('known_frame_count')==151 and metadata.get('active_slot_count')==453)
    representation=metadata.get('representation_vs_native',{});legacy=metadata.get('legacy_proxy_cone_status',{});matched=metadata.get('matched_native_parameter_cone',{})
    checks['conditional_nativecone_distinct_from_mu_point1_support_and_legacy_failure']=(representation.get('target_support_mu')==.1 and representation.get('target_unitforce_edges')==8
        and representation.get('target_normal_ray') is True and representation.get('target_direct_torsion') is False and representation.get('native_condim')==4
        and representation.get('same_representation') is False and representation.get('forward_and_targets_changed') is False
        and matched.get('condition')=='H123' and matched.get('feasible_frames')==matched.get('frame_count')==151 and matched.get('normalforce_uppercap','MISSING') is None
        and matched.get('columns_per_contact')==6 and matched.get('condim')==4 and legacy.get('feasible')==23 and legacy.get('solver_infeasible')==128
        and legacy.get('frame0_feasible') is False and legacy.get('retained') is True)
    return checks


def objective_build_rng_checks(value,method,spec_pin):
    checks={'schema':value.get('schema')=='full151_objective_build_rng:1'}
    if method=='A1L':
        checks['A1L_not_constructed']=value.get('status')=='NOT_CONSTRUCTED' and value.get('method')=='A1L' and value.get('objective')=='NONE' and value.get('RNG_capture_required') is False
    else:
        names={'Python_exact','NumPy_exact','TorchCPU_exact','CUDA_initialization_unchanged','already_initialized_CUDA_exact'}
        checks['source_report_checks_and_identity']=(value.get('status')=='PRESERVED' and value.get('method')==method and value.get('spec_pin')==spec_pin and value.get('factory_error') is None
            and set(value.get('checks',{}))==names and all(v is True for v in value['checks'].values()))
        before=value.get('before',{});after=value.get('after',{})
        checks['before_after_raw_state_digest_comparison']=(set(before)==set(after)=={'python','numpy','torch_cpu','torch_cuda','cuda_initialized'} and before==after
            and all(type(before[k]) is str and len(before[k])==64 for k in ('python','numpy','torch_cpu'))
            and type(before['cuda_initialized']) is bool and type(before['torch_cuda']) is list
            and all(type(v) is str and len(v)==64 for v in before['torch_cuda']) and (before['cuda_initialized'] or not before['torch_cuda']))
    return checks


def prior_section(config_path,run,saved,cfg,effective,result,isfinal,checks,details):
    evidence=saved['full151_prior_evidence'];method=cfg['method'];iteration=saved['iteration'];path=run/'full151_prior_evidence.jsonl'
    rows,raw=complete_rows(path) if path.exists() else ([],b'')
    pc,pd=prior_counter_checks(evidence,cfg,effective,iteration,rows,saved['full151_runtime_evidence'].get('policy'))
    checks.update({'prior_'+k:v for k,v in pc.items()})
    prefixraw=b''.join(raw.splitlines(keepends=True)[:pd['prefix_policy_calls']]);pd.update(prefix_SHA256=hashlib.sha256(prefixraw).hexdigest(),prefix_byte_count=len(prefixraw))
    details['prior_checkpoint_prefix']=pd
    def check(name,value):checks['prior_'+name]=bool(value)
    metadata_file=read(run/'prior_metadata.json')
    check('initial_metadata_zero_counters_static_matches_checkpoint',all(metadata_file.get(k)==0 for k in PRIOR_COUNTERS)
        and {k:v for k,v in metadata_file.items() if k not in PRIOR_COUNTERS}=={k:v for k,v in evidence.items() if k not in PRIOR_COUNTERS})
    build_rng=read(run/'objective_build_rng.json')
    check('build_rng_file_bound_checkpoint',build_rng==evidence.get('build_rng'))
    checks.update({'prior_build_rng_'+k:v for k,v in objective_build_rng_checks(build_rng,method,cfg['contact_binding']).items()})
    check('hook_source_pin_current_and_execution_bound',set(evidence.get('source_pins',{}))=={'prior_hook'} and all(checked(p) and any(Path(p['path']).resolve()==Path(v['path']).resolve() and p['sha256']==v['sha256'] for v in cfg['source_pins'].values()) for p in evidence['source_pins'].values()))
    if method=='A1L':check('A1L_metadata_NONE_no_contact_spec',metadata_file.get('status')=='NONE' and cfg.get('contact_binding') is None and not (run/'contact_objective_metadata.json').exists())
    else:
        spec=read(checked(cfg['contact_binding']));metadata=evidence['objective_metadata'];manifest=read(checked(spec['binding_manifest']))
        specfields={'schema','binding_manifest','target_receipt','independent_target_review','input_bundle','expected_code_pins','hyperparameters','execution_scope',
            'CPU_authorized','GPU_authorized','training_authorized','prior_CPU_binding_review','actual_material_contract','matched_material_screen','matched_material_saved_review','legacy_proxy_dynamic_cone_review','matched_material_lifecycle_review'}
        check('new_spec_schema_and_exact_authorization',set(spec)==specfields and spec.get('schema')=='full151_declared_H123_objective_spec:2'
            and spec.get('execution_scope')=='EXPLORATORY_MATCHED_TRAINING' and spec.get('CPU_authorized') is True and spec.get('training_authorized') is True
            and type(spec.get('GPU_authorized')) is bool and (cfg.get('GPU_run_authorized') is not True or spec['GPU_authorized'] is True)
            and canonical(spec.get('hyperparameters'))==canonical({'bandwidth_m':.001,'tolerance':.1,'var':.1}) and spec.get('input_bundle')==cfg['input_bundle'])
        check('spec_and_metadata_file_exact_checkpoint',metadata_file.get('objective_metadata')==metadata and cfg['contact_binding']==evidence['spec_pin'])
        prereq=('prior_CPU_binding_review','actual_material_contract','matched_material_screen','matched_material_saved_review','legacy_proxy_dynamic_cone_review','matched_material_lifecycle_review')
        needed=[cfg['contact_binding'],spec['binding_manifest'],spec['target_receipt'],spec['independent_target_review'],manifest['target']]+[spec[k] for k in prereq]
        check('target_and_spec_dependency_pins_in_execution_inventory',all(checked(p) and any(Path(p['path']).resolve()==Path(v['path']).resolve() and p['sha256']==v['sha256'] for v in cfg['source_pins'].values()) for p in needed))
        check('binding_code_metadata_current_and_config_bound',bool(metadata.get('source_pins')) and metadata['source_pins']==spec['expected_code_pins'] and all(checked(p) and any(Path(p['path']).resolve()==Path(v['path']).resolve() and p['sha256']==v['sha256'] for v in cfg['source_pins'].values()) for p in metadata['source_pins'].values()))
        check('prerequisite_report_pins_bound_metadata',metadata.get('prerequisites')=={k:spec[k] for k in prereq})
        checks.update({'prior_metadata_'+k:v for k,v in objective_metadata_checks(metadata,method,cfg,spec,manifest).items()})
        target=loadnp(checked(manifest['target']))
        check('target_actual151_unknownhuman_declared_only',target['support512'].shape==(151,1,1,512) and target['geometry_valid_mask'].shape==(151,3)
            and target['declared_intent_active'].all() and target['human_contact_unknown_mask'].all() and not target['human_contact_known'].any()
            and not target['human_contact_present'].any() and not target['known_no_contact'].any()
            and np.array_equal(target['phase_weight'],np.ones(151)) and target['point_com_body_m'].shape==(151,3,3))
        details['contact_binding']=cfg['contact_binding'];details['objective_metadata']=metadata
    metricrows,_=complete_rows(run/'metrics.jsonl');prefix=metricrows[:iteration]
    check('no_prior_terms_in_WM_only_metrics',not any(any(k.startswith('prior/') for k in row) for row in prefix[:min(iteration,effective['wm_only_iters'])]))
    if method=='A1L':check('A1L_metrics_no_prior',not any(any(k.startswith('prior/') for k in row) for row in prefix))
    else:
        check('each_completed_policy_iteration_has_prior_metrics',all(any(k.startswith('prior/') for k in row) for row in prefix[effective['wm_only_iters']:]))
        check('policy_metrics_equal_per_call_active_means',all(all(k in prefix[row['iteration']] and canonical(prefix[row['iteration']][k])==canonical(v) for k,v in row['active_means'].items()) for row in rows[:pd['prefix_policy_calls']]))
    if isfinal:check('final_result_prior_snapshot',result.get('full151_prior_evidence')==evidence)


def audit(config_path,run,checkpoint,expected_role=None,expected_method=None):
    torch.set_num_threads(1);checks={};details={};prefixraw=None;actualu_detail=None
    cfg=read(config_path);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    effective=read(run/'effective_training_config.json');meta=read(checkpoint.with_suffix('.meta.json'))
    iteration=saved['iteration'];isfinal=is_final_artifact(checkpoint,run,iteration,effective['max_iterations'])
    result=read(run/'result.json') if (run/'result.json').exists() else {}
    def check(name,value):checks[name]=bool(value)
    def section(name,fn):
        try:return fn()
        except Exception as exc:check(name+'_readable_and_valid',False);details.setdefault('errors',{})[name]=repr(exc);return None
    check('new105_schema2_three_arm_fromscratch_namespace',new105_header_ok(saved,cfg))
    check('declared_expected_method',expected_method is None or expected_method==cfg.get('method'))
    check('declared_expected_role',expected_role is None or expected_role==cfg.get('role'))
    check('native_integer_iteration_and_seed',type(iteration) is int and 1<=iteration<=effective['max_iterations'] and type(cfg.get('seed')) is int and type(saved['config'].get('seed')) is int and cfg['seed']==saved['config']['seed'])
    check('no_resume_or_warmstart',not any(effective.get(k) is not None for k in ('checkpoint','model_checkpoint')) and all(cfg.get(k) in (None,False,'') for k in ('resume','checkpoint','load_checkpoint','model_only','warm_start')))
    check('config_copy_exact',digest(config_path)==digest(run/'config.json'))
    index,_=complete_rows(run/'checkpoint_index.jsonl')
    check('checkpoint_index_sidecar_SHA',digest(checkpoint)==meta.get('sha256') and sum(row==meta for row in index)==1 and Path(meta['checkpoint']).resolve()==checkpoint.resolve() and meta.get('iteration')==iteration and (checkpoint.name=='model_final.pt' or checkpoint.name==f'model_{iteration}.pt') and checkpoint.parent.resolve()==(run/'train').resolve())
    check('checkpoint_effective_config_exact',effective_config_equal(saved['config'],effective))
    check('all_saved_state_and_optimizer_tensors_finite',finite(saved))
    check('module_and_optimizer_names',set(saved.get('modules',{}))==set(saved.get('optimizers',{}))=={'actor','wm','fail_head','critic'})
    check('native_state_architecture_dimensions',type(saved.get('context_dim')) is int and saved['context_dim']==368 and type(saved.get('contact_dim')) is int and saved['contact_dim']==15)
    arch={n:{k:list(v.shape) for k,v in m.items()} for n,m in saved['modules'].items()}
    check('module_state_architecture_exact',arch==expected_shapes((512,512,512)) and canonical(effective['wm_hidden'])==canonical([512,512,512]));details['actual_architecture']=arch
    for name,weights in saved['modules'].items():
        good=all(v.device.type=='cpu' and v.dtype==(torch.int64 if name=='actor' and k=='norm.count' else torch.float64 if k in ('norm.count','delta_norm.count') else torch.float32) for k,v in weights.items())
        good=good and all(bool((v>=0).all()) for k,v in weights.items() if k.endswith(('.var','._var','._std'))) and all(v.numel()==1 and float(v)>=0 for k,v in weights.items() if k in ('norm.count','delta_norm.count'))
        check('module_'+name+'_native_dtype_nonnegative_normalizers',good)
    common={'num_envs':128,'buffer_steps':256,'warmup_steps':64,'utd':.4,'wm_batches':2,'wm_batch':256,'wm_window':8,'horizon':8,'max_horizon':8,'pi_batch':128,'critic_epochs':2,'critic_minibatches':2,'holdout_fraction':.1,'log_interval':1,'amp':True,'compile':False,'compile_policy':False,'compile_policy_dynamic':False,'action_noise':.1}
    role_budget={'max_iterations':40,'wm_only_iters':20,'save_interval':40} if cfg['role']=='engineering_smoke' else {'max_iterations':2000,'wm_only_iters':200,'save_interval':100}
    check('fixed_declared_role_budget_and_hyperparameters',all(canonical(effective.get(k))==canonical(v) for k,v in (common|role_budget).items()))
    check('run_input_training_subset_exact',all(canonical(effective.get(k))==canonical(v) for k,v in cfg['training'].items()))
    active=max(0,iteration-effective['wm_only_iters'])
    counts,transitions=budget_counts(effective,iteration)
    check('actual_optimizer_hook_counts',optimizer_counts_match(meta.get('actual_optimizer_steps'),counts) and (not isfinal or optimizer_counts_match(result.get('actual_optimizer_steps'),counts)))
    check('actual_transition_count',type(saved.get('sim_steps')) is int and transitions==saved['sim_steps']==meta.get('sim_steps') and (not isfinal or transitions==result.get('sim_steps')))
    optsteps={}
    for name,opt in saved['optimizers'].items():
        states=opt['state'];parameter_ids=[v for group in opt['param_groups'] for v in group['params']]
        valid=all(type(v) is int for v in parameter_ids) and len(parameter_ids)==len(set(parameter_ids))
        valid=valid and ((not states) if counts[name]==0 else set(states)==set(parameter_ids))
        steps=[]
        for state in states.values():
            valid=valid and {'step','exp_avg','exp_avg_sq'}<=set(state) and torch.is_tensor(state['step']) and state['step'].numel()==1 and float(state['step'])==counts[name]
            steps.append(float(state['step']))
        optsteps[name]=sorted(set(steps));check('optimizer_'+name+'_stored_steps_and_state_coverage',valid and optsteps[name]==([float(counts[name])] if counts[name] else []))
    details.update(actual_optimizer_hook_counts=meta.get('actual_optimizer_steps'),optimizer_stored_steps=optsteps,real_transitions=transitions,actual_checkpoint_role_scope='normal_final_run' if isfinal else 'indexed_checkpoint_prefix')
    def inputs():
        for value in cfg['source_pins'].values():checked(value)
        b=read(checked(cfg['input_bundle']));protocol=read(checked(cfg['protocol']))
        check('source_config_protocol_alias',cfg['source_pins']['protocol']==cfg['protocol'])
        budget_key='engineering_smoke_training' if cfg['role']=='engineering_smoke' else 'primary_matched_training'
        check('protocol_input_budget_role',protocol['input_bundle']==cfg['input_bundle'] and canonical(protocol[budget_key])==canonical(cfg['training']) and cfg['role'] in protocol['allowed_roles'] and cfg['seed'] in protocol['allowed_seeds'])
        for value in b['source_pins'].values():checked(value)
        check('same_full151_knownrough_no_label_scope',b['schema']=='full151_knownrough_input_bundle:1' and b['mode']==MODE and b['retarget_variant_id']==VARIANT and b['raw_source_identity']['primary_pose_source_frame_range_half_open']==[200,351] and b['raw_source_identity']['local_reference_range_half_open']==[0,151] and b['physical_reference_qualified'] is False and b['geometric_candidate_qualified'] is False and b['known_rough_reference'] is True and b['training_authorized'] is False and b['contact_intent_bundle'] is None and b['human_contact_observation_known'] is False)
        parent=read(checked(b['source_pins']['parent_scp_receipt']))
        check('parent_SCP_FAIL_not_promoted',parent['status']=='CPU_COUPLED_CANDIDATE_FAIL_RETAINED' and parent['last_accepted_major']==8 and {k for k,v in parent['checks'].items() if not v}=={'last_accepted_eta_final_gate','final_fresh_sampled_geometry'})
        inventory=read(run/'source/inventory.json');check('source_archive_inventory_exact',inventory==cfg['source_pins'])
        with zipfile.ZipFile(run/'source/execution_sources.zip') as z:
            names=[f'{i:04d}_{Path(v["path"]).name}' for i,v in enumerate(inventory.values())]
            check('source_archive_all_bytes_and_members_exact',set(z.namelist())==set(names) and all(hashlib.sha256(z.read(name)).hexdigest()==value['sha256'] for name,value in zip(names,inventory.values())))
        check('nominal_model_and_actual_context_pins',digest(run/'nominal_model.mjb')==b['model']['sha256']==MODEL_SHA and b['context']['sha256']==CONTEXT_SHA)
        expected=loadnp(checked(b['context']));readbacks=loadnp(run/'actual_context_readbacks.npz');applied=loadnp(run/'expected_context_fields.npz');fixed=loadnp(run/'fixed_physics_context.npz')
        same=lambda a,b:a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==np.ascontiguousarray(b).tobytes()
        check('actual8_raw_readbacks_and_final_context_exact',all(same(applied[k],expected[k][0]) and np.isfinite(fixed[k]).all() and same(fixed[k],np.broadcast_to(expected[k][0],fixed[k].shape)) and all(same(readbacks[stage+'__'+k],np.broadcast_to(expected[k][0],readbacks[stage+'__'+k].shape)) for stage in ('after_copy','after_set_const_0')) for k in FIELDS))
        check('reader_physical_xi46_all128_raw_exact',fixed['reader_context'].shape==(128,46) and fixed['reader_context'].dtype==np.float32 and same(fixed['reader_context'],np.broadcast_to(expected['reader_context'][0],fixed['reader_context'].shape)))
        ref=loadnp(checked(b['source_pins']['reference_arrays']));init=loadnp(checked(b['source_pins']['init_arrays']));control=loadnp(checked(b['source_pins']['control_arrays']));native=loadnp(checked(b['source_pins']['native_data']));reader=loadnp(checked(b['source_pins']['native_reader_arrays']));baseline=control['u_postC2_150_FP32_rad']
        check('151states150baseline_source_identity',ref['q_goal_rad'].shape==(151,20) and ref['mapped_object_pose_w'].shape==(151,7) and baseline.shape==(150,20) and baseline.dtype==np.float32 and same(ref['q_goal_rad'],native['reference_qpos']) and same(ref['mapped_object_pose_w'],native['object_pose_w']) and reader['trajectory_lengths'].tolist()==[151])
        check('analytic_init_not_native_FD_substitution',same(init['q_init_requested_FP32_rad'],ref['q_goal_rad'][0].astype(np.float32)) and same(init['qd_init_requested_FP32_rad_s'],ref['analytic_qd151_rad_s'][0].astype(np.float32)) and same(init['u_planned0_FP32_rad'],baseline[0]) and same(reader['joint_vel'],ref['native_forward_FD32_rad_s']))
        c=saved['full151_controller'];basec={k:v for k,v in c.items() if k not in ('latest_actualu_evidence','reference_controller_signature')};root=c['root_descriptor'];extra={'trained_method','training_seed','run_role','config_sha256','execution_source_pins'};desc={k:v for k,v in root.items() if k not in extra}
        check('controller_signatures',c['reference_controller_signature']==hashlib.sha256(canonical(basec).encode()).hexdigest()==saved['reference_signature'] and c['root_descriptor_sha256']==hashlib.sha256(canonical(root).encode()).hexdigest())
        check('controller_authoritative_input_signature',desc.get('input_signature_sha256')==hashlib.sha256(canonical({k:v for k,v in desc.items() if k!='input_signature_sha256'}).encode()).hexdigest()=='3599bee61d92152b0c4f7c3bf8b71e99d2c00af460c81d23cecbd76d7e3fdd04')
        check('controller_input_source_arm_role_exact',set(extra)<=set(root) and root['schema']=='full151_knownrough_controller_input_descriptor:1' and root['trained_method']==cfg['method'] and type(root['training_seed']) is int and root['training_seed']==cfg['seed'] and root['run_role']==cfg['role'] and root['config_sha256']==digest(config_path) and root['execution_source_pins']==cfg['source_pins'] and root['source_pins']==b['source_pins'] and root['input_bundle']==cfg['input_bundle'] and root['retarget_variant_id']==VARIANT and root['mode']==MODE and root['physical_reference_qualified'] is False)
        check('controller105_fixed_baseline_EMA_C2_150intervals',type(c['format_version']) is int and c['format_version']==105 and c['schema']=='full151_knownrough_baseline_EMA_C2:1' and c['kind']=='FULL151_KNOWNROUGH_BASELINE_EMA_C2' and c['mode']==MODE and c['retarget_variant_id']==VARIANT and c['margin_rad']==.02 and c['exact_resume_supported'] is False and c['baseline_shape']==[150,20] and c['baseline_u150_FP32_sha256']==hashlib.sha256(baseline.tobytes()).hexdigest() and c['active_intervals']==[0,150] and type(c['terminal_observation']) is int and c['terminal_observation']==150 and c['inactive_imagination']=='finite u149 placeholder; runtime must keep survival0 and mask dead losses')
        check('controller_descriptor_snapshot',read(run/'controller_descriptor.json')==basec|{'reference_controller_signature':c['reference_controller_signature']})
        controllerpin=next(v for v in cfg['source_pins'].values() if Path(v['path']).resolve()==(ROOT/'src/full151_native/controller.py').resolve());purepin=next(v for v in cfg['source_pins'].values() if Path(v['path']).name=='target_inset.py')
        check('controller_source_projection_bound',c['controller_source_sha256']==controllerpin['sha256']=='b81c5376caaed93941078600983afa6356df60f01be50dcc2b9c259b19a257e0' and c['pure_projection_sha256']==purepin['sha256']=='2d88859c62d2833633f8b09a3be42d11c943e0c3422095e1d941100cb1bf86f4')
        details.update(source_pin_count=len(cfg['source_pins']),input_bundle=cfg['input_bundle'],reference_signature=saved['reference_signature'],init_source=b['source_pins']['init_arrays'],input_signature_sha256=desc['input_signature_sha256'])
        return baseline
    baseline=section('inputs_sources_context',inputs)
    if baseline is not None:
        au=section('actual_u',lambda:actualu_checks(run,saved,baseline))
        if au is not None:checks.update({'actualu_'+k:v for k,v in au[0].items()});details['actualu_checkpoint_prefix']=actualu_detail=au[1];prefixraw=au[2]
    ini=section('initial_parameters',lambda:initial_checks(saved,run,result,require_result_delta=isfinal))
    if ini is not None:checks.update(ini[0]);details['initial_parameters']=ini[1]
    def capacity():
        prefix=saved['full151_capacity_evidence'];pc,pd=capacity_checks(prefix,'checkpoint_prefix');checks.update({'capacity_prefix_'+k:v for k,v in pc.items()});details['capacity_checkpoint_prefix']=pd
        final=read(run/'capacity_observation.json') if (run/'capacity_observation.json').exists() else None
        if isfinal:
            fc,fd=capacity_checks(final,'finalrun');checks.update({'capacity_final_'+k:v for k,v in fc.items()});details['capacity_final_run']=fd
        elif final is not None:details['later_final_capacity_outside_checkpoint_scope']=final['status']
        if final is not None:check('capacity_checkpoint_records_are_finalrun_prefix',final['records'][:len(prefix['records'])]==prefix['records'])
        check('capacity_checkpoint_budget_binding',type(prefix.get('checkpoint_iteration')) is int and prefix['checkpoint_iteration']==iteration and type(prefix.get('checkpoint_sim_steps')) is int and prefix['checkpoint_sim_steps']==saved['sim_steps'])
        expectedkeys={'learning','training_capacity','controller','runtime','native'}
        check('capacity_evidence_sources_bound',set(prefix.get('source_pins',{}))==expectedkeys and all(checked(value) and any(Path(value['path']).resolve()==Path(v['path']).resolve() and value['sha256']==v['sha256'] for v in cfg['source_pins'].values()) for value in prefix['source_pins'].values()) and (final is None or final.get('source_pins')==prefix['source_pins']))
    section('capacity',capacity)
    def runtime():
        e=saved['full151_runtime_evidence']
        if active:checks.update({'runtime_'+k:v for k,v in policy_checks(e,effective).items()})
        else:check('policy_NOT_RUN_in_WM_only_prefix_not_policy_qualification',e.get('policy')=={'status':'NOT_RUN'} and e.get('policy_not_run_is_not_a_pass') is True)
        r=e['replay_counts'];check('runtime_replay_transition_count',type(r['transitions']) is int and r['transitions']==saved['sim_steps'] and all(type(r[k]) is int and 0<=r[k]<=r['transitions'] for k in ('natural_end_rows','physical_terminated_rows')))
        hist=e['phase_hist150'];coverage=e['phase_coverage']
        check('runtime_phase_histogram150_counts',type(hist) is list and len(hist)==150 and all(type(n) is int and n>=0 for n in hist) and sum(hist)==saved['sim_steps'] and coverage['phase_hist150']==hist and coverage['observed_transition_count']==sum(hist) and coverage['all150_phases_observed']==all(n>0 for n in hist))
        check('policy_NOT_RUN_marker_matches_phase',e.get('policy_not_run_is_not_a_pass') is (active==0))
        if actualu_detail is not None:check('independent_controller_replay_phasehist_and_natural_end_equal',hist==actualu_detail['phase_hist150'] and r['natural_end_rows']==hist[149] and coverage['latest_collect_phase_hist150']==actualu_detail['latest_collect_phase_hist150'])
        if isfinal:check('runtime_final_result_snapshot',result.get('runtime_evidence')==e)
        details['runtime_saved_evidence']=e;details['replay_generation_scope']='Latest native replay window is a source-validated summary. No raw per-window generation tensor is saved; this audit does not claim independent reconstruction of all window generations.'
    section('runtime',runtime)
    def metrics():
        rows,bytes_=complete_rows(run/'metrics.jsonl');metricprefix=rows[:iteration]
        check('checkpoint_metric_prefix_contiguous_and_finite',len(metricprefix)==iteration and [r['iteration'] for r in metricprefix]==list(range(iteration)) and all(finite(r) for r in metricprefix))
        details['metrics_prefix_sha256']=hashlib.sha256(b''.join(bytes_.splitlines(keepends=True)[:iteration])).hexdigest();details['metrics_later_complete_rows_ignored']=len(rows)-iteration
        details['metrics_prior_phase_only']={'active_policy_iterations':sum(any(k.startswith('prior/') for k in row) for row in metricprefix),'WM_only_rows_with_prior_fields':sum(any(k.startswith('prior/') for k in row) for row in metricprefix[:min(iteration,effective['wm_only_iters'])])}
    section('metrics',metrics)
    section('prior',lambda:prior_section(config_path,run,saved,cfg,effective,result,isfinal,checks,details))
    if isfinal:
        check('actual_final_completion_not_task_quality',result.get('schema')=='full151_native_training_completion:2' and result.get('status')=='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY' and result.get('physical_reference_qualified') is False and result.get('cleanup_errors')==[] and result.get('from_scratch') is True and result.get('method')==cfg['method'] and result.get('contact_objective')==cfg['contact_binding'] and result.get('checkpoint')==pin(checkpoint) and result.get('final_iteration')==iteration)
        def completion():
            marker=read(run/'training_completion.json');details['training_completion_marker']=pin(run/'training_completion.json');summary=marker['result']
            check('worker_training_completion_marker_schema_config',set(marker)=={'schema','config_sha256','result'} and marker['schema']=='full151_native_worker_training_completion:2' and marker['config_sha256']==digest(config_path))
            check('training_wrapper_return_final_capacity',summary.get('full151_capacity_qualified_final') is True and summary.get('full151_capacity_observation')==pin(run/'capacity_observation.json') and summary.get('status')=='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY')
        section('completion',completion)
    runtime=read(run/'runtime.json');check('worker_config_source_identity_no_retry',runtime.get('automatic_retry') is False and runtime.get('plan_sha256')==digest(config_path) and type(runtime.get('pid')) is int and type(runtime.get('parent_pid')) is int and any(v['sha256']==runtime.get('source_sha256') for v in cfg['source_pins'].values()))
    if isfinal:check('worker_terminal_completed',runtime.get('status')=='completed_full151_training_v2')
    check('CPU_no_CUDA_or_native_environment_import',os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and not torch.cuda.is_initialized() and not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules))
    finalscope=qualified_final_scope(checkpoint,run,iteration,effective['max_iterations'],checks)
    status='PASS_SAVED_ENGINEERING_HEALTH' if checks and all(checks.values()) else 'FAIL_OR_MISSING_EVIDENCE'
    return {'schema':'full151105_checkpoint_CPU_health:1','status':status,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'method':cfg['method'],'seed':cfg['seed'],'role':cfg['role'],'mode':MODE,'retarget_variant_id':VARIANT,'iteration':iteration,'final_iteration_scope':finalscope,'config':pin(config_path),'run_dir':str(run),'checkpoint':pin(checkpoint),'auditor':pin(__file__),'input_bundle':cfg['input_bundle'],'source_pins':cfg['source_pins'],'budget_counts':{'real_transitions':transitions,'optimizer_steps':counts,'collect_calls':iteration+1,'metric_rows':iteration},'details':details,'physical_task_success':'NOT_ASSESSED','physical_reference_qualified':False,'GPU_used':False,'created_environment':False,'claim_scope':'Saved105 source/input/finiteness/architecture/raw fromscratch initialization/native optimizer counts/actualu prefix/H8 masks/phase histogram and capacity evidence only. Latest generation replaywindow is source-validated summary, not raw-window reconstruction. WM-only prefix health is not Actor/critic quality. Knownrough parentSCP/geometry FAIL retained.'},prefixraw

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-role',choices=['engineering_smoke','primary_matched']);p.add_argument('--expected-method',choices=['A1L','P_ONLY','A2']);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Refusing to replace an audit')
    a.output.mkdir(parents=True)
    try:report,prefix=audit(a.config.resolve(strict=True),a.run_dir.resolve(strict=True),a.checkpoint.resolve(strict=True),a.expected_role,a.expected_method)
    except Exception as exc:report={'schema':'full151105_checkpoint_CPU_health:1','status':'INVALID_OR_MISSING_ARTIFACT','checks':{},'passed':0,'total':0,'final_iteration_scope':False,'error':repr(exc),'auditor':pin(__file__),'GPU_used':False,'created_environment':False};prefix=None
    json_write(a.output/'report.json',report)
    if prefix is not None:(a.output/'actualu_checkpoint_prefix_snapshot.jsonl').write_bytes(prefix)
    print(json.dumps({'status':report['status'],'passed':report.get('passed'),'total':report.get('total'),'failed':[k for k,v in report.get('checks',{}).items() if not v],'error':report.get('error'),'report':pin(a.output/'report.json')},indent=2))
    return 0 if report['status']=='PASS_SAVED_ENGINEERING_HEALTH' else 1

if __name__=='__main__':raise SystemExit(main())
