"""UNFROZEN matched public native-self15 CPU component trainer. Root execution is separate.

No resume, old119f load, Actor, MJB, Env or physics. Original152 only. The first
training forward is also the fixed saved-initial parity check, not an extra FW.
"""
from __future__ import annotations
import copy,datetime,hashlib,json,math,os,struct,sys,time,zipfile
from pathlib import Path
from . import boundary_component_initial_parity105_v1 as parity
c=parity.c;ROOT=c.ROOT;read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require
SCHEMA='full151105_boundary_public_self15_matched_training_CPU_protocol:1'
ROLE='matched_native_self15_public_aux_vs_cross_target_consistency'
PAIR_SCHEMA='full151105_boundary_public_self15_matched_pair:1'
INITIAL_SCHEMA='full151105_boundary_public_self15_matched_initial:1'
ARTIFACT_SCHEMA='full151105_boundary_public_self15_WM_component:1'
PREFIX='full151105_boundary_public_self15_matched_training_CPU_'
RECIPE=ROOT/'protocols/full151105_boundary_public_self15_matched_training_recipe_v1.json'
RECIPE_SHA='9e0765fd562c3e24bdb09d52f78219204af4228bb6e8b1d8ffd67104ce094bce'
HEALTH_READY=ROOT/'outputs/full151105_boundary_public_self15_checkpoint_auditor_preparation_CPU_v1/READY.json'
HEALTH_READY_SHA='b0709fe11e39aca835abf6cd26ceaf59b62669ebfe70b77924c99143f5040fe0'
SOURCE_CONTRACT=ROOT/'docs/full151_boundary_public_self15_matched_training_contract_v1.md'
SOURCE_CONTRACT_SHA='5a12bb913b21a45f6cfc3bd6df39a8bbe4074e0588ac47f1791f9561ec3f4ed7'
INPUTS={'target_report','target_arrays','target_saved8','target_source_inventory','initial_component','initial_validation','initial_saved_health29',
        'initial_parity_report','initial_parity_saved_review','initial_parity_baseline','component_ready','component_source_review',
        'cached_sensor_labels','cached_sensor_report','cached_sensor_saved7','target_identities'}
OWN={'consumer':ROOT/'src/full151_gradient/boundary_public_self15_training105_v1.py','CLI':ROOT/'scripts/train_full151_boundary_public_self15_components_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_boundary_public_self15_training_cpu_v1.py'}
DEPENDENCIES={**parity.DEPENDENCIES,'parity_helper':ROOT/'src/full151_gradient/boundary_component_initial_parity105_v1.py',
              'old_trainer_helpers':ROOT/'src/full151_gradient/boundary_impulse_training105_v2.py',
              'cached_label_source':ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py'}
COUNTS=('native_forward_attempted','native_forward_returned','feature_hook_calls','aux_head_forward_returned','backward_calls','optimizer_steps')
LOSS_FIELDS={'state':'L_state_FP32','self15':'L_self15_sum_FP32','native_pair':'L_native_pair_FP32','mode':'L_mode_FP32','impulse':'L_imp_F64','consistency':'L_cons_F64','public':'L_public','total':'L_total'}
PLAN_FIELDS=('row_id','snapshot_id','path','mode','k','branch_id','group_id','physical_rows','terminal_observation')


def fixed_recipe():
    require(type(RECIPE_SHA) is str and pin(RECIPE)['sha256']==RECIPE_SHA,'New Root recipe must be frozen before execution')
    value=read(RECIPE);require(value['schema']=='full151105_boundary_public_self15_matched_training_recipe:1','Numerical recipe namespace')
    r=value['recipe']
    require(r['lambda_state']==r['lambda_mode']==r['lambda_imp']==1. and r['lambda_self15']==.1 and r['window_length']==1 and r['all_rows_valid'] is True and r['native15_label_only_JSON'] is True,'Fixed W1/native15 public coefficients')
    require(r['self15_reduction']=='sum_features_then_mean_rows' and r['native_pair_reduction']=='mean_rows_of_FP32_state_row_plus_weighted_self15_sum_row','Native per-row FP32 reduction, not two reduced scalar addition')
    return r


def validate_protocol(p,execute=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE,'New matched trainer namespace')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is False and p.get('automatic_retry') is False and type(p.get('background_continue_to_final')) is bool,'Explicit CPU/noGPU/noRetry/background policy')
    require(p.get('output_prefix')==PREFIX and type(p.get('output')) is str and Path(p['output']).is_absolute() and type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and p.get('source_pins'),'Exact16 inputs/fresh owned output namespace')
    require(same(p['recipe'],{'path':str(RECIPE),'sha256':RECIPE_SHA}),'Same frozen Root recipe')
    require(same(p['source_contract'],{'path':str(SOURCE_CONTRACT),'sha256':SOURCE_CONTRACT_SHA}),'New fixed public15 source/loss contract')
    if execute:require(p['CPU_run_authorized'],'Root explicit training authorization required')
    return p


def required_sources(p):
    inventory=read(checked(p['inputs']['target_source_inventory']))
    health=read(HEALTH_READY)
    cached=read(checked(p['inputs']['cached_sensor_report']))
    return c.canonical_sources(list(inventory.values())+list(p['source_pins'].values())+list(p['inputs'].values())+[p['recipe'],p['source_contract'],p['source_review'],pin(HEALTH_READY),health['cpu_report'],cached['protocol']]+list(cached['source_pins'].values())+list(health['source_pins'].values())+[pin(v) for v in list(OWN.values())+list(DEPENDENCIES.values())])


def metadata_preflight(protocol_path,output=None):
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Default hash/JSON only')
    p=validate_protocol(read(protocol_path));recipe=fixed_recipe();i=p['inputs']
    for value in i.values():checked(value)
    target=read(checked(i['target_report']));review=read(checked(i['target_saved8']));c.allchecks(review)
    require(target['enrollment_qualified'] is True and target['counts']['enrolled_groups']==152 and target['counts']['selected_substeps']==608 and same(review['producer_report'],i['target_report']),'Actual old152/608 enrolled target/positive8')
    require(checked(i['target_identities']).samefile(checked(i['target_report']).parent/'group152_identities.json'),'Original target152 identity route')
    cached=read(checked(i['cached_sensor_report']));cached_review=read(checked(i['cached_sensor_saved7']));c.allchecks(cached_review)
    require(cached['schema']=='full151105_cached_self15_saved_evaluation:1' and cached['status']=='COMPLETED_ALL152_CACHED_SELF15_LABEL_METRICS_NOT_GEOMETRY_ACCURACY' and cached['counts']['selected_labels']==152,'Completed all152 cached-label producer, no metric archive as supervision')
    require(cached_review['schema']=='full151105_cached_self15_saved_independent:1' and cached_review['status']=='PASS_ALL152_ALL3ROLES_CACHED15_METRICS_NOT_GEOMETRY' and cached_review['engineering_qualified'] is True and cached_review['passed']==cached_review['total']==7 and same(cached_review['producer_report'],i['cached_sensor_report']),'Positive cached15 independent7 and exact source producer')
    require(same(cached['input_pins']['target_report'],i['target_report']) and same(cached['input_pins']['target_group152'],i['target_identities']),'Cached15 source and original target row identity cohort')
    require(checked(i['cached_sensor_labels']).samefile(checked(i['cached_sensor_report']).parent/'joined152_cached_sensor_labels.json'),'Only immutable label-only JSON route')
    outputs=read(checked(i['cached_sensor_report']).parent/'output_pins.json')
    require(same(outputs['report.json'],i['cached_sensor_report']) and same(outputs['joined152_cached_sensor_labels.json'],i['cached_sensor_labels']),'Label-only output exact producer pin')
    frozen=read(RECIPE)
    require(same(frozen['cached15_label_only_JSON'],i['cached_sensor_labels']) and same(frozen['cached15_saved_review7'],i['cached_sensor_saved7']) and same(frozen['source_targets'],i['target_arrays']),'Frozen recipe labels/review/target file identities')
    actual=read(checked(i['initial_parity_report']));c.allchecks(actual)
    saved=read(checked(i['initial_parity_saved_review']));c.allchecks(saved)
    require(actual['function_identity_qualified'] is True and saved['schema']=='full151105_boundary_initial_parity_saved_independent:1' and saved['status']=='PASS_SAVED_REAL_INITIAL_FUNCTION_PARITY' and saved['engineering_qualified'] is True and saved['function_identity_qualified'] is True and same(saved['producer_report'],i['initial_parity_report']),'Actual independent initial function parity, not training qualification')
    for key in ('target_report','target_arrays','target_saved8','initial_component','initial_validation','initial_saved_health29','component_ready','component_source_review'):
        require(same(actual['input_pins'][key],i[key]),'Same real-initial cohort/source: '+key)
    require(checked(i['target_arrays']).samefile(checked(i['target_report']).parent/'all152_boundary_impulse_targets.npz') and checked(i['target_source_inventory']).samefile(checked(i['target_report']).parent/'source_inventory.json'),'Actual target routes')
    require(checked(i['initial_parity_baseline']).samefile(checked(i['initial_parity_report']).parent/'unwrapped_baseline_forward152.npz'),'Exact frozen initial baseline152')
    health=read(checked(i['initial_saved_health29']));c.allchecks(health);require(same(health['initial_component'],i['initial_component']),'One trusted own initial namespace')
    for value in list(OWN.values())+list(DEPENDENCIES.values()):require(any(same(pin(value),v) for v in p['source_pins'].values()),'Current trainer/helper source membership')
    require(pin(HEALTH_READY)['sha256']==HEALTH_READY_SHA,'Frozen new namespace health READY')
    health_ready=read(HEALTH_READY);health_proof=read(checked(health_ready['cpu_report']));c.allchecks(health_proof)
    for value in health_ready['source_pins'].values():require(any(same(value,v) for v in p['source_pins'].values()),'New health reader source/proof direct enrollment')
    source_review=read(checked(p['source_review']));c.allchecks(source_review)
    require(source_review['schema']=='full151105_boundary_public_self15_training_source_independent:1' and source_review['source_review_qualified'] is True,'Positive new namespace SOURCE review, not old trainer health')
    for name,path in OWN.items():require(any(same(pin(path),value) for value in source_review['source_pins'].values()),'Source review binds exact tested trainer/CLI/checker bytes: '+name)
    sources=required_sources(p);out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists() and out==Path(p['output']).resolve(),'Fresh exact protocol output; no resume/overwrite')
    return {'status':'PASS_MATCHED_TRAINER_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'recipe':recipe,'sources':sources,'output':out,'actual_data_load_model_FW_AD_optimizer':0}


def declaration():
    return {'schema':'full151105_boundary_public_self15_training_declaration:1','status':'UNFROZEN_SOURCE_SYNTHETIC_ONLY','protocol_schema':SCHEMA,'role':ROLE,'input_aliases':sorted(INPUTS),
        'recipe':pin(RECIPE),'own':{k:pin(v) for k,v in OWN.items()},'budget':{'own_initial_load':1,'native_WM_constructor':1,'training_native_FW':2000,'training_rows':304000,'training_backward_step':2000,'extra_validation_FW':0,'Actor_MJB_Env_GPU_physics':0}}


def cached_label_rows(plan,joined):
    """Pure JSON identity/binary check. Actual caller alone converts152x15 to FP32.

    No logits/metric archive is accepted. Timing remains native cached15 at
    lastsolver~15ms while the native transition target is postintegration20ms.
    """
    from . import boundary_self15_cached_saved105_v1 as source
    require(type(plan) is list and type(joined) is list and len(plan)==len(joined)==152,'Complete original152 plan and label-only rows')
    result=[]
    for row_id,(g,row) in enumerate(zip(plan,joined)):
        source.validate_plan_row(g)
        require(g['row_id']==row_id and all(k in row and type(row[k]) is type(g[k]) and row[k]==g[k] for k in PLAN_FIELDS),'Exact all152 label→original target plan row/branch/group/path/phase order')
        require(row['source_frame']==200+g['k'] and type(row['source_frame']) is int and row['label_time']=='lastsolver_preintegration~kdt+.015' and row['state_time']=='postintegration_kdt+.020' and row['valid_next_policy_decision'] is (g['k']<149) and row['no_u150'] is True,'Cached15/POST20 timing and terminal rule')
        values=row['contact15'];found=row['raw_found25']
        require(type(values) is list and len(values)==15 and all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in values),'Binary15 JSON values, no smoothing or probability labels')
        require(type(found) is list and len(found)==25 and all(type(v) in (bool,int,float) and math.isfinite(v) and v>=0 and int(v)==v for v in found),'Saved nonnegative integer raw_found25 label evidence')
        bits=[v>0 for v in found];logical=[float(bits[j]) for j in source.MIDDLE]+[float(bits[d] or bits[t]) for d,t in zip(source.DISTAL,source.TIPS)]
        require(all(struct.pack('<f',float(v))==struct.pack('<f',q) for v,q in zip(values,logical)),'Original middle10+distalORtip5 mapping matches all15 FP32 bytes')
        result.append([float(v) for v in values])
    return result


def validate_target_identity_arrays(plan,group_ids,substep_rows,np):
    require(group_ids.shape==(152,) and group_ids.dtype==np.int64 and substep_rows.shape==(152,4) and substep_rows.dtype==np.int64,'Original target NPZ source identity arrays')
    require(np.array_equal(group_ids,np.asarray([g['group_id'] for g in plan],dtype=np.int64)) and np.array_equal(substep_rows,np.asarray([g['physical_rows'] for g in plan],dtype=np.int64)),'Forward target NPZ rows mechanically join all152 label/plan identities')


def public_self15_losses(module,x,pred,self15_logits,z,mode,delta52,scale52,target15,boundary_labels,r,torch):
    """Same forward: original native point52 plus public native15 W1 supervision.

    The five-key003 boundary label dictionary stays unchanged. The scalar state
    loss remains mean(point_row); native_pair is its separately named per-row
    FP32 weighted reduction, never a relabelled L_state or detached oracle.
    """
    from . import boundary_impulse_component105_v1 as component
    require(delta52.ndim==2 and delta52.shape[1]==52 and delta52.dtype==scale52.dtype==torch.float32 and scale52.shape==(52,) and bool(torch.isfinite(delta52).all()) and bool(torch.isfinite(scale52).all()) and bool((scale52>0).all()),'Finite native FP32 point52/effective52')
    require(self15_logits.shape==target15.shape==(delta52.shape[0],15) and self15_logits.dtype==target15.dtype==torch.float32 and not target15.requires_grad and bool(torch.isfinite(self15_logits).all()) and bool(torch.isfinite(target15).all()) and bool(((target15==0)|(target15==1)).all()),'Same-batch original binary15 target only outside forward')
    squared=(delta52/scale52).square();point_row=squared.mean(-1);point=point_row.mean()
    bce=torch.nn.functional.binary_cross_entropy_with_logits(self15_logits,target15,reduction='none');self_row=bce.sum(-1)
    native_row=point_row+torch.tensor(r['lambda_self15'],dtype=torch.float32,device=point_row.device)*self_row
    native_pair=native_row.mean();self_scalar=self_row.mean()
    loss=component.component_losses(module,x,pred,z,mode,point,boundary_labels,lambda_mode=r['lambda_mode'],lambda_imp=r['lambda_imp'],lambda_cons=r['lambda_cons_by_arm'][module.arm])
    mode_elements=torch.nn.functional.binary_cross_entropy_with_logits(mode,boundary_labels['boundary_constraint_witness'].to(mode.dtype),reduction='none')
    loss['L_mode_FP32']=mode_elements.mean()
    public=(native_pair+r['lambda_mode']*loss['L_mode_FP32'])+r['lambda_imp']*loss['L_imp_F64']
    total=public+r['lambda_cons_by_arm'][module.arm]*loss['L_cons_F64']
    loss.update(L_self15_sum_FP32=self_scalar,L_native_pair_FP32=native_pair,L_public=public,L_total=total)
    require(all(v.numel()==1 and bool(torch.isfinite(v)) for k,v in loss.items() if k in LOSS_FIELDS.values()),'Finite all8 actual scalar losses')
    operands={'native_D52_FP32':delta52,'effective_s52_FP32':scale52,'point_squared52_FP32':squared,'point_per_row_FP32':point_row,
        'self15_logits_FP32':self15_logits,'self15_target_FP32':target15,'self15_BCE_elements_FP32':bce,'self15_sum_per_row_FP32':self_row,'native_pair_per_row_FP32':native_row,
        'mode_BCE_elements_FP32':mode_elements}
    return loss,operands


def loss_snapshot_before_update(loss,operands,iteration,torch):
    return {'iteration':iteration,'tag':'PRE_OPTIMIZER_I','weights_after_updates':iteration-1,**snapshot_state(operands,torch),
            'scalars':{name:snapshot_state(loss[key],torch) for name,key in LOSS_FIELDS.items()}}


def snapshot_state(value,torch):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:snapshot_state(v,torch) for k,v in value.items()}
    if isinstance(value,list):return [snapshot_state(v,torch) for v in value]
    if isinstance(value,tuple):return tuple(snapshot_state(v,torch) for v in value)
    return copy.deepcopy(value)


def ordered_parameters(module,expected=None):
    pairs=list(module.named_parameters());names=[name for name,p in pairs]
    require(len(set(names))==len(names) and all(p.requires_grad for name,p in pairs),'Unique trainable named parameters only')
    if expected is not None:require(tuple(names)==tuple(expected),'Source-native actual16 order')
    return names,[p for name,p in pairs]


def observed_counts(module,extra,offset):
    raw=module.counts
    return {'native_forward_attempted':raw['forward_attempted']-offset['forward_attempted'],'native_forward_returned':raw['native_forward_returned']-offset['native_forward_returned'],
        'feature_hook_calls':raw['hook_calls']-offset['hook_calls'],'aux_head_forward_returned':raw['head_forward_returned']-offset['head_forward_returned'],
        'backward_calls':extra['backward_calls'],'optimizer_steps':extra['optimizer_steps']}


def clip_gradients(module,iteration,torch,save_raw):
    names,params=ordered_parameters(module)
    require(all(p.grad is not None and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) for p in params),'Every named raw gradient must exist/finite')
    pre={name:p.grad.detach().clone() for name,p in zip(names,params)} if save_raw else None
    returned=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True,foreach=False)
    postnorm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad,2) for p in params]),2)
    require(bool(torch.isfinite(returned)) and bool(torch.isfinite(postnorm)),'Finite actual clip norms')
    if not save_raw:return None
    return {'iteration':iteration,'tag':'PRE_OPTIMIZER_I','weights_after_updates':iteration-1,'parameter_names':names,'pre_clip':pre,
        'post_clip':{name:p.grad.detach().clone() for name,p in zip(names,params)},'max_norm':1.,'pre_clip_returned_norm_FP32':returned.detach().clone(),'post_clip_total_norm_FP32':postnorm.detach().clone()}


def optimizer_steps_exact(optimizer,params,iteration,torch):
    return len(optimizer.state)==len(params) and all(p in optimizer.state and isinstance(optimizer.state[p].get('step'),torch.Tensor) and bool(torch.isfinite(optimizer.state[p]['step'])) and float(optimizer.state[p]['step'])==iteration for p in params)


def checkpoint_payload(module,optimizer,counts,gradients,iteration,common,rng,log_prefix,loss_snapshot,torch):
    return {'schema':ARTIFACT_SCHEMA,'pair_schema':PAIR_SCHEMA,'arm':module.arm,'iteration':iteration,'checkpoint_tag':'AFTER_OPTIMIZER_I',**common,
        'WM_state':snapshot_state(module.wm.state_dict(),torch),'aux_head_state':snapshot_state(module.aux_head.state_dict(),torch),
        'physical_context_state':snapshot_state(module.context.state_dict(),torch),'normalizer_host_configuration':c._norm_config(module.wm),
        'ordered_parameter_names':[name for name,p in module.named_parameters()],'optimizer_state':snapshot_state(optimizer.state_dict(),torch),'counts':dict(counts),
        'RNG':snapshot_state(rng,torch),'gradient_snapshot':gradients,'log_prefix':dict(log_prefix),'loss_snapshot':snapshot_state(loss_snapshot,torch)}


def estimate_eta_100(pair_seconds,save_seconds,elapsed_seconds,now_utc):
    require(len(pair_seconds)==100 and all(type(v) in (int,float) and v>=0 for v in pair_seconds),'All100 complete pair costs')
    mean=sum(pair_seconds[3:])/97;remaining_pairs=900;final_overhead=20.;remaining=remaining_pairs*mean+save_seconds+final_overhead
    return {'scope':'Planning heuristic, not statistical CI; remaining900 PAIRS, not1800pairs','completed_pairs':100,'remaining_pairs':900,'trim_first_pair_count':3,
        'observed_mean_pair_seconds_after_trim':mean,'observed_checkpoint100_pair_save_seconds':save_seconds,'actual_elapsed_including_setup_archive_save_seconds':elapsed_seconds,
        'explicit_unmeasured_finalization_overhead_seconds':final_overhead,'estimated_remaining_seconds':remaining,'estimated_total_elapsed_seconds':elapsed_seconds+remaining,
        'planning_remaining_range_seconds':[.85*remaining_pairs*mean+save_seconds+final_overhead,1.3*remaining_pairs*mean+save_seconds+final_overhead],
        'estimated_finish_UTC':(now_utc+datetime.timedelta(seconds=remaining)).isoformat(),'independent_health_reader_time_not_measured_or_included':True,'background_continuation_is_protocol_field':True}


def run_training(protocol_path,output,emit=None):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);r=m['recipe'];i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,value in m['sources'].items():z.write(checked(value),name)
    phase='setup';modules={};iteration=0;extra={};offsets={};training_counts={};pair_seconds=[];saved_pairs=[]
    process_counts={'own_initial_torch_loads':0,'native_WM_constructors':0,'raw_deepcopies':0,'validation_FW':0,'Actor_old119f_MJB_Env_GPU_physics':0}
    def progress(status,**fields):
        value={'schema':'full151105_boundary_public_self15_training_runtime:1','status':status,'phase':phase,'completed_pair_iteration':iteration,'training_counts':training_counts,'process_counts':process_counts,'elapsed_seconds':time.perf_counter()-started,**fields}
        c._write_json(out/'runtime.json',value,replace=True)
        if emit is not None:emit(value)
        return value
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        from . import boundary_impulse_component105_v1 as component
        from . import component_H8_donor105_v1 as h8
        from dex_supermimic.rl.dex_supermimic.state import difference
        torch.set_num_threads(4);rng_start=c._rng(torch,np);h8.cf.save_raw_rng(out,'startup',rng_start,np)
        with np.load(checked(i['target_arrays']),allow_pickle=False) as f:raw_data={k:f[k].copy() for k in f.files}
        data=parity.prepare_arrays(raw_data,np)
        plan=read(checked(i['target_identities']));joined=read(checked(i['cached_sensor_labels']))
        target15=torch.tensor(cached_label_rows(plan,joined),dtype=torch.float32)
        validate_target_identity_arrays(plan,raw_data['source_group_id152'],raw_data['source_substep_rows152x4'],np)
        with np.load(checked(i['initial_parity_baseline']),allow_pickle=False) as f:baseline={k:torch.from_numpy(f[k].copy()) for k in ('prediction58_FP32','self15_logits_FP32')}
        initial=torch.load(checked(i['initial_component']),map_location='cpu',weights_only=False);process_counts['own_initial_torch_loads']+=1
        xi,scale=initial['physical_xi46'],initial['frozen_s52'];require(component.bit_equal(xi,torch.from_numpy(data['physical_xi46_FP32'])),'Target/initial actualxi46')
        x,u,target=(torch.from_numpy(data[k]) for k in ('x58_FP32','u20_FP32','target58_FP32'));dynamics=torch.cat((x,xi[None].expand(152,-1),torch.zeros((152,174),dtype=torch.float32)),dim=-1)
        labels={k:torch.from_numpy(data[k]) for k in ('J_palm_F64','K_palm_F64','boundary_constraint_witness','r_v_saved_m_s_F64','r_w_saved_rad_s_F64')}
        fit_input_digest=c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch);fit_label_digest=c._tensor_digest(labels,torch)
        fit_self15_label_digest=c._tensor_digest({'cached_self15_target_FP32':target15},torch)
        combined_labels={**labels,'cached_self15_target_FP32':target15};fit_combined_label_digest=c._tensor_digest(combined_labels,torch)
        require(fit_input_digest==read(checked(i['initial_parity_report']))['input_digest_before'],'Exact actual parity training input identity')
        ctx=component.fixed_context_from_rows(torch.from_numpy(data['mass_actual_F64'].astype(np.float32)),torch.from_numpy(data['I_body_F64']),torch.from_numpy(data['gravity_palm_F64']),torch.from_numpy(data['DeltaT_F64']),torch.from_numpy(data['palm_rot_raw_FP32']),scale,xi)
        raw=h8.make_initial_eval_component(initial,read(checked(i['initial_validation'])),torch);process_counts['native_WM_constructors']+=1
        modules,construction=component.make_matched_components(raw,ctx);process_counts['raw_deepcopies']=2
        expected_names=[f'wm.net.{j}.{field}' for j in (0,1,3,4,6,7,9) for field in ('weight','bias')]+['aux_head.weight','aux_head.bias']
        optimizers={};names={};params={}
        for arm,module in modules.items():
            names[arm],params[arm]=ordered_parameters(module,expected_names)
            optimizers[arm]=torch.optim.AdamW(params[arm],lr=r['lr'],weight_decay=r['weight_decay'],betas=tuple(r['betas']),eps=r['optimizer_eps'],foreach=False)
            offsets[arm]=dict(module.counts);extra[arm]={'backward_calls':0,'optimizer_steps':0};training_counts[arm]=observed_counts(module,extra[arm],offsets[arm])
        common={'protocol':m['protocol_pin'],'recipe':p['recipe'],'input_pins':i,'source_pins':p['source_pins'],'fit_input_digest':fit_input_digest,'fit_label_digest':fit_label_digest,
                'fit_self15_label_digest':fit_self15_label_digest,'fit_combined_label_digest':fit_combined_label_digest}
        rng_this_run=c._rng(torch,np);require(all(parity.raw_rng_equality(rng_start,rng_this_run,np,torch).values()),'Initialization/import/constructor changed RNG, never restore caller state')
        shared=modules[component.ARMS[0]]
        shared_initial={'schema':INITIAL_SCHEMA,'pair_schema':PAIR_SCHEMA,**common,'source_parent':i['initial_component'],
            'WM_state':snapshot_state(shared.wm.state_dict(),torch),'aux_head_state':snapshot_state(shared.aux_head.state_dict(),torch),'physical_context_state':snapshot_state(shared.context.state_dict(),torch),
            'normalizer_host_configuration':c._norm_config(shared.wm),'ordered_parameter_names':expected_names,'RNG_this_run':snapshot_state(rng_this_run,torch),'construction_receipt':construction,
            'wrapper_start_count_offsets':offsets,'training_counts':copy.deepcopy(training_counts),'cached_self15_target_FP32':target15.detach().clone(),'source_identity_fields':copy.deepcopy(plan)}
        c._atomic_torch_save(out/'initial_pair_state.pt',shared_initial,torch)
        phase='lockstep_training';progress('running');fit_started=time.perf_counter()
        with (out/'updates.jsonl').open('xb') as log:
            for current in range(1,r['iterations']+1):
                if current%100==0:c._write_json(out/f'progress_before_{current:04d}.json',progress('running',entering_iteration=current))
                pair_start=time.perf_counter();rows={};gradient_snapshots={};loss_snapshots={}
                for arm in component.ARMS:
                    phase=arm+'/update_'+str(current);module=modules[arm];opt=optimizers[arm];arm_started=time.perf_counter();opt.zero_grad(set_to_none=True)
                    prediction,self15,z,mode=module(x,u,dynamics)
                    if current==1:
                        require(component.bit_equal(prediction.detach(),baseline['prediction58_FP32']) and component.bit_equal(self15.detach(),baseline['self15_logits_FP32']) and component.bit_equal(z.detach(),torch.zeros_like(z)) and component.bit_equal(mode.detach(),torch.zeros_like(mode)),'First TRAINING forward actual-saved initial parity, zerohead; no extra validationFW')
                    delta=difference(prediction,target);chart=c.chart_guard(prediction,target,delta,r['native_log_chart_margin_rad'],torch)
                    losses,operands=public_self15_losses(module,x,prediction,self15,z,mode,delta,scale,target15,labels,r,torch)
                    if current in r['checkpoint_iterations']:loss_snapshots[arm]=loss_snapshot_before_update(losses,operands,current,torch)
                    losses['L_total'].backward();extra[arm]['backward_calls']+=1
                    snapshot=clip_gradients(module,current,torch,current in r['checkpoint_iterations'])
                    opt.step();extra[arm]['optimizer_steps']+=1
                    require(all(bool(torch.isfinite(q).all()) for q in params[arm]) and c._finite_tree(opt.state_dict(),torch) and optimizer_steps_exact(opt,params[arm],current,torch),'Finite parameters/rawAdam actual step count')
                    module.assert_frozen();training_counts[arm]=observed_counts(module,extra[arm],offsets[arm]);require(all(v==current for v in training_counts[arm].values()),'Actual single FW/head/backward/step per arm')
                    rows[arm]={'counts':dict(training_counts[arm]),'losses':{k:float(losses[v].detach()) for k,v in LOSS_FIELDS.items()},'update_seconds':time.perf_counter()-arm_started,'native_chart':chart}
                    if snapshot is not None:gradient_snapshots[arm]=snapshot
                iteration=current;pair_seconds.append(time.perf_counter()-pair_start)
                record={'iteration':iteration,'arms':rows};log.write((json.dumps(record,allow_nan=False)+'\n').encode('utf-8'));log.flush()
                if current%100==0:os.fsync(log.fileno())
                if current in r['checkpoint_iterations']:
                    phase='save_pair_'+str(current);progress('saving_checkpoint_pair');save_started=time.perf_counter()
                    prefix_bytes=log.tell();prefix={'path':str(out/'updates.jsonl'),'records':current,'bytes':prefix_bytes,'sha256':hashlib.sha256((out/'updates.jsonl').read_bytes()[:prefix_bytes]).hexdigest()}
                    checkpoints={};checks={}
                    for arm,module in modules.items():
                        rng=c._rng(torch,np);require(all(parity.raw_rng_equality(rng_this_run,rng,np,torch).values()),'This-run raw RNG changed; no repair')
                        require(c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch)==fit_input_digest and c._tensor_digest(labels,torch)==fit_label_digest and c._tensor_digest({'cached_self15_target_FP32':target15},torch)==fit_self15_label_digest and c._tensor_digest(combined_labels,torch)==fit_combined_label_digest,'Frozen all152 fit inputs/old boundary/new cached15 labels unchanged')
                        payload=checkpoint_payload(module,optimizers[arm],training_counts[arm],gradient_snapshots[arm],current,common,rng,prefix,loss_snapshots[arm],torch)
                        path=out/f'{arm}_update_{current:04d}.pt';c._atomic_torch_save(path,payload,torch);checkpoints[arm]=pin(path)
                        checks[arm+'_rawAdam_steps_inmemory']=optimizer_steps_exact(optimizers[arm],params[arm],current,torch);module.assert_frozen()
                    save_seconds=time.perf_counter()-save_started
                    health={'schema':'full151105_boundary_public_self15_inmemory_health:1','iteration':current,'checks':checks,'passed':len(checks),'total':len(checks),'engineering_inmemory_pass':all(checks.values()),'serialized_readback_performed':False,'independent_saved_health_pending':True,'gradient_phase':'PRE_OPTIMIZER_I; checkpoint weights AFTER_OPTIMIZER_I'}
                    c._write_json(out/f'paired_checkpoint_{current:04d}_inmemory_health.json',health)
                    manifest={'schema':PAIR_SCHEMA,'iteration':current,'initial_pair_state':pin(out/'initial_pair_state.pt'),'checkpoints':checkpoints,'log_prefix':prefix,'producer_health':pin(out/f'paired_checkpoint_{current:04d}_inmemory_health.json'),'save_seconds':save_seconds,'protocol':m['protocol_pin'],'independent_health_required_before_Goal_pause':True}
                    c._write_json(out/f'paired_checkpoint_{current:04d}.json',manifest);saved_pairs.append(manifest)
                    if current==100:c._write_json(out/'checkpoint100_ETA.json',estimate_eta_100(pair_seconds,save_seconds,time.perf_counter()-started,datetime.datetime.now(datetime.timezone.utc)))
                if current%100==0:c._write_json(out/f'progress_after_{current:04d}.json',progress('running'))
                if current==100 and not p['background_continue_to_final']:break
        phase='finalization';require(all(parity.raw_rng_equality(rng_this_run,c._rng(torch,np),np,torch).values()),'Final this-run raw RNG unchanged')
        require(c._tensor_digest(labels,torch)==fit_label_digest and c._tensor_digest({'cached_self15_target_FP32':target15},torch)==fit_self15_label_digest and c._tensor_digest(combined_labels,torch)==fit_combined_label_digest and c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch)==fit_input_digest,'Final inputs and both label contracts unchanged')
        for value in m['sources'].values():checked(value)
        for value in i.values():checked(value)
        require(same(pin(protocol_path),m['protocol_pin']),'Protocol unchanged')
        status='COMPLETED_FIXED1000_MATCHED_PUBLIC_SELF15_COMPONENT_TRAINING' if iteration==1000 else 'STOPPED_AFTER_PAIRED100_EXPLICIT_PROTOCOL_NO_RESUME'
        report={'schema':'full151105_boundary_public_self15_matched_training_completion:1','status':status,'protocol':m['protocol_pin'],'recipe':p['recipe'],'input_pins':i,'source_pins':p['source_pins'],'iteration':iteration,'training_counts':training_counts,'process_counts':process_counts,'fit_input_digest':fit_input_digest,'fit_label_digest':fit_label_digest,
            'fit_self15_label_digest':fit_self15_label_digest,'fit_combined_label_digest':fit_combined_label_digest,'paired_checkpoints':saved_pairs,'training_elapsed_seconds':time.perf_counter()-fit_started,'total_elapsed_seconds':time.perf_counter()-started,'independent_saved_checkpoint_health_pending':True,'background_continue_to_final':p['background_continue_to_final'],'native105_or_policy_resume_qualified':False,'physical_force_or_gradient_qualified':False,'whole_hidden_state_qualified':False,'self15_supervised':True,'self15_label_phase':'lastsolver_preintegration~kdt+.015; native target58 POST20','self15_not_HO_geometry_or_human':True}
        c._write_json(out/'result.json',report);progress(status);c._write_json(out/'output_pins_final.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        try:
            for arm,module in modules.items():training_counts[arm]=observed_counts(module,extra.get(arm,{'backward_calls':0,'optimizer_steps':0}),offsets.get(arm,{k:0 for k in module.counts}))
            failure=progress('FAILED_PARTIAL_NO_RETRY',error=repr(error));c._write_json(out/'failure.json',failure)
        except BaseException as save_error:
            print(json.dumps({'original_error':repr(error),'failure_artifact_error':repr(save_error),'phase':phase,'completed_pair_iteration':iteration}),file=sys.stderr)
        raise
    finally:
        for module in modules.values():module.close()
