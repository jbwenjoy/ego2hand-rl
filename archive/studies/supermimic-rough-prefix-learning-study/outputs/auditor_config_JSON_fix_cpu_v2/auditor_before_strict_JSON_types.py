"""Read saved rough104 tensors and evidence on CPU; never construct an Env.

Health is serialization/input/budget/capacity evidence, not task quality.
V2 preserves V1 and only compares native tuple config values in their JSON wire
representation, matching the already-written effective_training_config.json.
Checkpoint collection prefixes and the later finalized run are separate scopes.
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
VARIANT='H123_cubic_minimax_restoration_filtered_prefix16_v1'
MODEL_SHA='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd'
CONTEXT_SHA='5edd125973015c034d45e47092786b7fac105a06af52864582afa08d8660b89a'
ERROR_KEYS={'cache_actual','cache_expected','residual_before_after','replay_u_actual','replay_c_residual'}

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
    return json.loads(json.dumps(saved_config,allow_nan=False))==effective

def expected_collect_steps(cfg,iteration):
    out=[max(cfg['warmup_steps'],cfg['wm_window'])];acc=0.
    for _ in range(iteration):acc+=1./cfg['utd'];n=int(acc);acc-=n;out.append(n)
    return out

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
    checks['H8_shapes_and_integer_phases']=start.shape in ((b,1),(b,)) and mask.shape==(h,b) and mask.dtype.kind=='b' and finite(start) and bool(((start>=0)&(start<15)&(start==np.floor(start))).all())
    if checks['H8_shapes_and_integer_phases']:
        phase=start.reshape(-1);expected=phase[None,:]+np.arange(h)[:,None]<15;last=phase+h<15
        checks['terminal_active_mask_independent']=np.array_equal(mask,expected)
        checks['critic_norm_eligible_independent']=p.get('critic_norm_eligible')==int(expected.sum()+last.sum())
        counts=p.get('critic_loss_minibatch_active_counts',[])
        checks['critic_minibatch_mask_counts']=len(counts)==cfg['critic_epochs']*cfg['critic_minibatches'] and all(type(c) is int and 0<=c<=h*b for c in counts) and sum(counts)==cfg['critic_epochs']*int(expected.sum())
    else:checks.update(terminal_active_mask_independent=False,critic_norm_eligible_independent=False,critic_minibatch_mask_counts=False)
    checks['actual_policy_call_contract']=p.get('optimizer_actor_calls')==1 and p.get('optimizer_critic_calls')==cfg['critic_epochs']*cfg['critic_minibatches'] and p.get('critic_total_candidate_rows')==h*b and p.get('hard_mask_rule')=='phase<15 only; physical soft termination retains native survival semantics'
    w=evidence.get('latest_window',{})
    checks['validated_native_window_summary']=all(type(w.get(k)) is int and w[k]>=0 for k in ('valid_window_entries','candidate_window_entries','terminal_valid_entries','reset_suffix_entries_masked')) and w.get('candidate_window_entries',-1)==w.get('valid_window_entries',0)+w.get('reset_suffix_entries_masked',0) and w.get('terminal_valid_entries',10**9)<=w.get('valid_window_entries',-1)
    return checks

def actualu_checks(run,saved,baseline):
    c=saved['rough_controller'];marker=c['latest_actualu_evidence'];last=marker['collect_call'];rows,raw=complete_rows(run/'rough_actualu_evidence.jsonl');prefix=rows[:last];steps=expected_collect_steps(saved['config'],saved['iteration']);checks={}
    checks['checkpoint_prefix_complete_and_contiguous']=type(last) is int and last==saved['iteration']+1 and len(prefix)==last and [r.get('collect_call') for r in prefix]==list(range(1,last+1))
    checks['marker_equals_last_prefix_row']=bool(prefix) and prefix[-1]==marker and marker['iteration']==max(0,saved['iteration']-1)
    checks['per_call_steps_worlds_and_total']=len(prefix)==len(steps) and all(r['steps']==n and r['worlds']==saved['config']['num_envs'] and r['real_transitions']==n*r['worlds'] for r,n in zip(prefix,steps)) and sum(r['real_transitions'] for r in prefix)==saved['sim_steps']
    checks['allbatch_exact_finite_phase0to14']=all(r.get('all_batch_exact') is True and r.get('all_finite') is True and r.get('inactive_real_actions')==0 and 0<=r.get('phase_min',-1)<=r.get('phase_max',99)<15 and len(r.get('phase_histogram',[]))==15 and sum(r['phase_histogram'])==r['real_transitions'] and all(type(x) is int and x>=0 for x in r['phase_histogram']) for r in prefix)
    checks['all5_mismatches_zero']=all(set(r.get('max_abs_errors',{}))==ERROR_KEYS and all(type(x) in (float,int) and math.isfinite(x) and x==0 for x in r['max_abs_errors'].values()) for r in prefix)
    checks['clamped_count_accounting']=all(type(r.get('clamped_components')) is int and 0<=r['clamped_components']<=r['real_transitions']*20 for r in prefix)
    lower=np.array(c['target_lower_rad'],np.float32);upper=np.array(c['target_upper_rad'],np.float32);low=lower+np.float32(.02);high=upper-np.float32(.02);traces={}
    for call in [i for i in range(1,last+1) if i in (1,2,3) or i%100==0]:
        p=run/f'rough_actualu_trace_{call:05d}.npz';a=loadnp(p);traces[p.name]=digest(p);shape=(min(4,prefix[call-1]['steps']),min(4,prefix[call-1]['worlds']),20)
        checks[f'trace{call}_shapes_finite']=set(a)=={'phase','baseline','proposed','residual_before_filter','cached','actual','residual','replay_u','replay_c'} and all(a[k].shape==shape and a[k].dtype==np.float32 and np.isfinite(a[k]).all() for k in a if k!='phase') and a['phase'].shape==shape[:2] and a['phase'].dtype.kind in 'iu'
        phase=a['phase'];expected=np.clip(np.clip(a['proposed'],lower,upper),low,high)
        checks[f'trace{call}_projection_actual_replay']=np.array_equal(expected,a['cached']) and np.array_equal(a['cached'],a['actual']) and np.array_equal(a['actual'],a['replay_u']) and np.array_equal(a['residual_before_filter'],a['residual']) and np.array_equal(a['residual'],a['replay_c']) and bool(((a['actual']>=low)&(a['actual']<=high)).all())
        checks[f'trace{call}_baseline_phase_residual']=bool(((phase>=0)&(phase<15)).all()) and np.array_equal(a['baseline'],baseline[phase]) and np.array_equal(a['proposed'],a['baseline']+a['residual_before_filter'])
    # Hash onlycomplete prefix throughcheckpoint; later append isnotdrift.
    lines=raw.splitlines(keepends=True);prefixraw=b''.join(lines[:last])
    return checks,{'last_collect_call':last,'prefix_rows':len(prefix),'prefix_sha256':hashlib.sha256(prefixraw).hexdigest(),'prefix_byte_count':len(prefixraw),'real_transitions':sum(r['real_transitions'] for r in prefix),'sample_trace_sha256':traces,'later_complete_rows_ignored':len(rows)-last},prefixraw

def initial_checks(saved,run,result,*,require_result_delta=True):
    raw=torch.load(run/'initial_parameter_tensors.pt',map_location='cpu',weights_only=False);finger=read(run/'initial_parameters.json');checks={};recomputed={};delta={}
    checks['initial_raw_schema_CPU_finite']=raw.get('schema')=='rough_prefix_initial_parameters:1' and set(raw.get('modules',{}))==set(saved['modules']) and finite(raw) and all(v.device.type=='cpu' for m in raw['modules'].values() for v in m['parameters'])
    checks['initial_capture_source_pins']=bool(raw.get('source_pins')) and all(checked(p) for p in raw['source_pins'].values()) and raw.get('source_pins')==saved.get('rough_capacity_evidence',{}).get('source_pins') and raw.get('capture_point')=='after_native_DexRunner_init_before_first_collect_or_optimizer'
    for name,m in raw['modules'].items():
        values=m['parameters'];names=m['parameter_names'];h=hashlib.sha256();squared=0.;valid=len(values)==len(names)==len(set(names)) and set(names)=={k for k in saved['modules'][name] if k.startswith('net.') and k.endswith(('.weight','.bias'))}
        valid=valid and m['shapes']==[list(v.shape) for v in values] and m['dtypes']==[str(v.dtype) for v in values]
        for key,v in zip(names,values):
            h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes());final=saved['modules'][name][key]
            valid=valid and final.shape==v.shape and final.dtype==v.dtype
            squared+=float((final-v).double().square().sum())
        recomputed[name]={'sha256':h.hexdigest(),'tensors':len(values),'parameters':sum(v.numel() for v in values)};delta[name]=squared**.5
        checks[f'initial_{name}_names_shapes_and_hash']=valid and recomputed[name]==finger.get(name)==saved.get('rough_initial_parameter_fingerprints',{}).get(name)
        if require_result_delta:checks[f'initial_{name}_delta_independent']=name in result.get('parameter_delta_l2',{}) and math.isclose(delta[name],result['parameter_delta_l2'][name],rel_tol=1e-10,abs_tol=1e-12)
    active=max(0,saved['iteration']-saved['config']['wm_only_iters'])
    checks['actual_actor_WM_critic_change_matches_training_phase']=delta.get('wm',0)>0 and all((delta.get(k,0)>0) if active else (delta.get(k,-1)==0) for k in ('actor','critic'))
    return checks,{'independently_recomputed_fingerprints':recomputed,'independent_final_parameter_delta_l2':delta,'capture_point':raw.get('capture_point'),'raw_initial_tensor_artifact':pin(run/'initial_parameter_tensors.pt'),'scope':'Raw CPU constructor-return parameters hashed against later native-before-learning fingerprint; source rejects allresume/modelwarmstart. Crossarm same-seed comparison remains separate.'}

def audit(config_path,run,checkpoint):
    torch.set_num_threads(1);checks={};details={};prefixraw=None
    cfg=read(config_path);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);effective=read(run/'effective_training_config.json');meta=read(checkpoint.with_suffix('.meta.json'));iteration=saved['iteration'];isfinal=iteration==effective['max_iterations'];result=read(run/'result.json') if (run/'result.json').exists() else {}
    def check(name,value):checks[name]=bool(value)
    def section(name,fn):
        nonlocal prefixraw
        try:return fn()
        except Exception as exc:check(name+'_readable_and_valid',False);details.setdefault('errors',{})[name]=repr(exc);return None
    check('format104_fromscratch_newvariant',saved.get('format_version')==104 and cfg.get('schema')=='rough_prefix_native_run:1' and cfg.get('role') in ('engineering_smoke','primary_matched') and cfg.get('method') in ('A1L','P_ONLY','A2') and cfg.get('controller')=={'kind':'ROUGH_PREFIX_BASELINE_EMA_C2','format_version':104,'margin_rad':.02,'resume':'from_scratch_only'} and not any(effective.get(k) is not None for k in ('checkpoint','model_checkpoint')))
    check('config_copy_exact',digest(config_path)==digest(run/'config.json'))
    check('checkpoint_index_sidecar_SHA',digest(checkpoint)==meta.get('sha256') and meta in complete_rows(run/'checkpoint_index.jsonl')[0] and Path(meta['checkpoint']).resolve()==checkpoint.resolve())
    check('checkpoint_effective_config_exact',effective_config_equal(saved['config'],effective))
    check('all_saved_tensors_finite',finite(saved))
    check('module_and_optimizer_names',set(saved.get('modules',{}))==set(saved.get('optimizers',{}))=={'actor','wm','fail_head','critic'})
    check('native_state_architecture_dimensions',saved.get('context_dim')==368 and saved.get('contact_dim')==15)
    arch={n:{k:list(v.shape) for k,v in m.items()} for n,m in saved['modules'].items()};check('module_state_architecture_exact',arch==expected_shapes(tuple(effective['wm_hidden'])));details['actual_architecture']=arch
    if cfg['role']=='engineering_smoke':check('engineering10_budget_configuration',all(effective.get(k)==v for k,v in {'num_envs':128,'max_iterations':10,'warmup_steps':16,'utd':.4,'wm_batches':2,'wm_only_iters':2,'horizon':8,'max_horizon':8,'pi_batch':128,'critic_epochs':2,'critic_minibatches':2,'wm_window':8}.items()) and 1<=iteration<=10)
    check('run_input_training_subset_exact',all(effective.get(k)==v for k,v in cfg['training'].items()))
    active=max(0,iteration-effective['wm_only_iters']);counts={'wm':iteration*effective['wm_batches'],'fail_head':iteration*effective['wm_batches'],'actor':active,'critic':active*effective['critic_epochs']*effective['critic_minibatches']};transitions=effective['num_envs']*sum(expected_collect_steps(effective,iteration))
    check('actual_optimizer_hook_counts',meta.get('actual_optimizer_steps')==counts and (not isfinal or counts==result.get('actual_optimizer_steps')))
    check('actual_transition_count',transitions==saved.get('sim_steps')==meta.get('sim_steps') and (not isfinal or transitions==result.get('sim_steps')))
    optsteps={}
    for n,opt in saved['optimizers'].items():
        steps=sorted({float(v['step']) for v in opt['state'].values() if 'step' in v});optsteps[n]=steps;check('optimizer_stored_steps_'+n,steps==([float(counts[n])] if counts[n] else []))
    details['actual_optimizer_hook_counts']=meta.get('actual_optimizer_steps');details['optimizer_stored_steps']=optsteps;details['real_transitions']=transitions
    def inputs():
        for p in cfg['source_pins'].values():checked(p)
        b=read(checked(cfg['input_bundle']));check('source_config_protocol_alias',cfg['source_pins']['protocol']==cfg['protocol'])
        protocol=read(checked(cfg['protocol']));budget_key='engineering_smoke_training' if cfg['role']=='engineering_smoke' else 'primary_matched_training';check('protocol_input_budget_role',protocol['input_bundle']==cfg['input_bundle'] and protocol[budget_key]==cfg['training'] and cfg['role'] in protocol['allowed_roles'] and cfg['seed'] in protocol['allowed_seeds'])
        for p in b['source_pins'].values():checked(p)
        check('input_bundle_variant_source_window',b['retarget_variant_id']==VARIANT and b['raw_source_identity']['primary_pose_source_frame_range_half_open']==[200,216] and b['physical_reference_qualified'] is False and b['training_authorized'] is False)
        inventory=read(run/'source/inventory.json');check('source_archive_inventory_exact',inventory==cfg['source_pins'])
        with zipfile.ZipFile(run/'source/execution_sources.zip') as z:
            check('source_archive_bytes_exact',all(hashlib.sha256(z.read(f'{i:04d}_{Path(v["path"]).name}')).hexdigest()==v['sha256'] for i,v in enumerate(inventory.values())))
        check('nominal_model_and_actual_context_pins',digest(run/'nominal_model.mjb')==b['model']['sha256']==MODEL_SHA and b['context']['sha256']==CONTEXT_SHA)
        expected=loadnp(checked(b['context']));readbacks=loadnp(run/'actual_context_readbacks.npz');applied=loadnp(run/'expected_context_fields.npz');fixed=loadnp(run/'fixed_physics_context.npz')
        check('actual8_raw_readbacks_exact',all(np.array_equal(applied[k],expected[k][0]) and np.isfinite(fixed[k]).all() and np.array_equal(fixed[k],np.broadcast_to(expected[k][0].astype(fixed[k].dtype),fixed[k].shape)) and all(np.array_equal(readbacks[stage+'__'+k],np.broadcast_to(expected[k][0].astype(readbacks[stage+'__'+k].dtype),readbacks[stage+'__'+k].shape)) for stage in ('after_copy','after_set_const_0')) for k in FIELDS))
        check('reader_context_raw_exact',np.array_equal(fixed['reader_context'],np.broadcast_to(expected['reader_context'][0],fixed['reader_context'].shape)))
        ref=loadnp(checked(b['source_pins']['reference_arrays']));init=loadnp(checked(b['source_pins']['init_arrays']));control=loadnp(checked(b['source_pins']['control_arrays']));baseline=control['control__fp32_old_export__tracking_u_exec_15_FP32_rad'];bridge=loadnp(checked(b['source_pins']['contact_reference']))
        check('new16_reference_and_analytic_init',np.array_equal(ref['q_goal_rad'],bridge['joint_pos_rad']) and np.array_equal(ref['mapped_object_pose_w'],bridge['object_pose_w']) and np.array_equal(init['q_init_requested_FP32_rad'],ref['q_goal_rad'][0].astype(np.float32)) and np.array_equal(init['qd_init_requested_FP32_rad_s'],ref['analytic_qd16_rad_s'][0].astype(np.float32)) and np.array_equal(init['u_planned0_FP32_rad'],baseline[0]))
        c=saved['rough_controller'];basec={k:v for k,v in c.items() if k not in ('latest_actualu_evidence','reference_controller_signature')};root=c['root_descriptor']
        check('controller_signatures',c['reference_controller_signature']==hashlib.sha256(canonical(basec).encode()).hexdigest()==saved['reference_signature'] and c['root_descriptor_sha256']==hashlib.sha256(canonical(root).encode()).hexdigest())
        check('controller_input_source_arm_identity',root['trained_method']==cfg['method'] and root['training_seed']==cfg['seed'] and root['run_role']==cfg['role'] and root['config_sha256']==digest(config_path) and root['execution_source_pins']==cfg['source_pins'] and root['source_pins']==b['source_pins'] and root['retarget_variant_id']==VARIANT)
        check('controller104_fixed_baseline_EMA_C2',c['format_version']==104 and c['kind']=='ROUGH_PREFIX_BASELINE_EMA_C2' and c['margin_rad']==.02 and c['exact_resume_supported'] is False and c['baseline_shape']==[15,20] and c['baseline_u15_FP32_sha256']==hashlib.sha256(baseline.tobytes()).hexdigest() and c['active_intervals']==[0,15] and c['terminal_observation']==15)
        check('controller_descriptor_snapshot',read(run/'controller_descriptor.json')==basec|{'reference_controller_signature':c['reference_controller_signature']})
        controllerpin=next(p for p in cfg['source_pins'].values() if Path(p['path']).name=='controller.py');purepin=next(p for p in cfg['source_pins'].values() if Path(p['path']).name=='target_inset.py')
        check('controller_source_projection_bound',c['controller_source_sha256']==controllerpin['sha256'] and c['pure_projection_sha256']==purepin['sha256'])
        details['source_pin_count']=len(cfg['source_pins']);details['input_bundle']=cfg['input_bundle'];details['reference_signature']=saved['reference_signature'];details['init_source']=b['source_pins']['init_arrays']
        return baseline
    baseline=section('inputs_sources_context',inputs)
    if baseline is not None:
        au=section('actual_u',lambda:actualu_checks(run,saved,baseline))
        if au is not None:checks.update({'actualu_'+k:v for k,v in au[0].items()});details['actualu_checkpoint_prefix']=au[1];prefixraw=au[2]
    ini=section('initial_parameters',lambda:initial_checks(saved,run,result,require_result_delta=isfinal))
    if ini is not None:checks.update(ini[0]);details['initial_parameters']=ini[1]
    def capacity():
        prefix=saved['rough_capacity_evidence'];pc,pd=capacity_checks(prefix,'checkpoint_prefix');checks.update({'capacity_prefix_'+k:v for k,v in pc.items()});details['capacity_checkpoint_prefix']=pd
        final=read(run/'capacity_observation.json') if (run/'capacity_observation.json').exists() else None
        if isfinal:
            fc,fd=capacity_checks(final,'finalrun');checks.update({'capacity_final_'+k:v for k,v in fc.items()});details['capacity_final_run']=fd
        elif final is not None:details['later_final_capacity_outside_checkpoint_scope']=final['status']
        if final is not None:check('capacity_checkpoint_records_are_finalrun_prefix',final['records'][:len(prefix['records'])]==prefix['records'])
        check('capacity_checkpoint_budget_binding',prefix.get('checkpoint_iteration')==saved['iteration'] and prefix.get('checkpoint_sim_steps')==saved['sim_steps'])
        check('capacity_evidence_sources_bound',set(prefix.get('source_pins',{}))=={'training_adapter','capacity_observer','native_v2'} and all(checked(p) and p in cfg['source_pins'].values() for p in prefix['source_pins'].values()) and (final is None or final.get('source_pins')==prefix['source_pins']))
    section('capacity',capacity)
    def runtime():
        e=saved['rough_runtime_evidence']
        if active:checks.update({'runtime_'+k:v for k,v in policy_checks(e,effective).items()})
        else:check('policy_NOT_RUN_pretraining_not_qualification',e.get('policy',{}).get('status')=='NOT_RUN')
        r=e['replay_counts'];check('runtime_replay_transition_count',r['transitions']==saved['sim_steps'] and 0<=r['natural_end_rows']<=r['transitions'] and 0<=r['physical_terminated_rows']<=r['transitions'])
        if isfinal:check('runtime_final_result_snapshot',result.get('runtime_evidence')==e)
        details['runtime_saved_evidence']=e
    section('runtime',runtime)
    def prior():
        rows,bytes_=complete_rows(run/'metrics.jsonl');metrics=rows[:iteration];check('checkpoint_metric_prefix_finite',len(metrics)==iteration and [r['iteration'] for r in metrics]==list(range(iteration)) and all(finite(r) for r in metrics));details['metrics_prefix_sha256']=hashlib.sha256(b''.join(bytes_.splitlines(keepends=True)[:iteration])).hexdigest()
        pri=[r for r in metrics if any(k.startswith('prior/') for k in r)]
        if cfg['method']=='A1L':check('A1L_no_prior_file_or_metrics',cfg['contact_binding'] is None and not (run/'contact_objective_metadata.json').exists() and not pri);return
        spec=read(checked(cfg['contact_binding']));m=read(run/'contact_objective_metadata.json')
        for k,v in spec['expected_hashes'].items():check('prior_bound_'+k,m.get(k)==v)
        check('prior_intent_semantics_failed_qualification',m.get('known_semantics')=='declared_H123_robot_kinematic_geometry_hypothesis_only_not_human_observation' and m.get('human_contact_observation_known') is False and m.get('target_defined_from_policy_contacts') is False and m.get('local_proxy_qualification')=='FAILED_RETAINED_EXPLORATORY_ONLY' and m.get('compiled_precision_qualification')=='FAILED_RETAINED_NOT_REVALIDATED' and m.get('qualified_contact_predictor') is False and m.get('physical_reference_qualified') is False)
        check('prior_same64_512_P_and_external_dt',m['basis']['training_directions']==64 and m['basis']['evaluation_directions']==512 and m['penetration_prior']=={'allowance_m':.003,'normalization_m':.001,'coefficient':.1,'formula':'square(max(relu(-sample_sdf-.003))/.001)','phase_masked':False,'meaning':'always-active sampled geometric prior, not full penetration measurement'} and m['dt_applied_inside_module'] is False and m['caller_dt_s']==.02)
        check('prior_policy_rows_actually_logged',len(pri)==active and [r['iteration'] for r in pri]==list(range(effective['wm_only_iters'],iteration)) and all('prior/geometry_intent_bonus_unscaled' in r and finite(r) for r in pri))
        if cfg['method']=='P_ONLY':check('P_ONLY_direct_sharedP_formula',m['formula']=='-.1*penetration_prior' and m['scoring']['support_coefficient']==0. and m['base_A2_scoring']['support_coefficient']==1.)
        else:check('A2_union_support_formula',m['formula']=='phase_weight * support_match - .1*penetration_prior' and m['scoring']['support_coefficient']==1. and m['known_frame_count']==16 and m['support_experiment_degenerate_to_penalty_only'] is False)
        details['prior_metadata']=pin(run/'contact_objective_metadata.json')
    section('prior_and_metrics',prior)
    if isfinal:
        check('completed_engineering_not_task_result',result.get('status')=='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY' and result.get('physical_reference_qualified') is False and result.get('cleanup_errors')==[])
        def completion():
            c=read(run/'training_completion.json');details['training_completion_marker']=pin(run/'training_completion.json')
            # CLI owns this newer return marker; preserve native result.json.
            summary=c['result']
            check('training_completion_schema_and_config',set(c)=={'schema','config_sha256','result'} and c['schema']=='rough_prefix_training_completion:1' and c['config_sha256']==digest(config_path))
            check('training_wrapper_return_final_capacity',summary.get('rough_capacity_qualified_final') is True and summary.get('rough_capacity_observation')==pin(run/'capacity_observation.json') and summary.get('status')=='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY')
        section('completion',completion)
    runtime=read(run/'runtime.json');check('worker_config_identity_no_retry',runtime.get('automatic_retry') is False and runtime.get('plan_sha256')==digest(config_path))
    if isfinal:check('worker_terminal_completed',runtime.get('status')=='completed_rough_training')
    check('CPU_no_CUDA_or_environment',os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and not torch.cuda.is_initialized() and not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules))
    return {'schema':'rough104_checkpoint_CPU_health:1','status':'PASS_SAVED_ENGINEERING_HEALTH' if all(checks.values()) else 'FAIL_OR_MISSING_EVIDENCE','checks':checks,'passed':sum(checks.values()),'total':len(checks),'method':cfg['method'],'seed':cfg['seed'],'iteration':saved['iteration'],'final_iteration_scope':isfinal,'config':pin(config_path),'run_dir':str(run),'checkpoint':pin(checkpoint),'auditor':pin(__file__),'details':details,'physical_task_success':'NOT_ASSESSED','physical_reference_qualified':False,'GPU_used':False,'created_environment':False,'claim_scope':'Saved format104 source/input/finiteness/architecture/initialization/budget/control/replay-mask/capacity evidence only. Latest native replaywindow is source-validated summary, not a rawwindow independently rerun here. No contact/dynamics-gradient or learning-benefit qualification.'},prefixraw

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Refusing to replace an audit')
    a.output.mkdir(parents=True)
    try:report,prefix=audit(a.config.resolve(strict=True),a.run_dir.resolve(strict=True),a.checkpoint.resolve(strict=True))
    except Exception as exc:report={'status':'INVALID_OR_MISSING_ARTIFACT','error':repr(exc),'auditor':pin(__file__),'GPU_used':False,'created_environment':False};prefix=None
    json_write(a.output/'report.json',report)
    if prefix is not None:(a.output/'actualu_checkpoint_prefix_snapshot.jsonl').write_bytes(prefix)
    print(json.dumps({'status':report['status'],'passed':report.get('passed'),'total':report.get('total'),'failed':[k for k,v in report.get('checks',{}).items() if not v],'error':report.get('error'),'report':pin(a.output/'report.json')},indent=2))
    return 0 if report['status']=='PASS_SAVED_ENGINEERING_HEALTH' else 1

if __name__=='__main__':raise SystemExit(main())
