"""UNFROZEN SOURCE DRAFT: native state52 + cached self15 only, no aux losses.

Root alone may later authorize actual data/model/training. One own initial load,
one native constructor and the frozen003 two-RAW-copy factory. Its control
wrapper alone forwards; the unused second copy never forwards. All16 parameters
remain trainable, but loss has no auxiliary-head path: 14 gradients/Adam states.
"""
from __future__ import annotations
import copy,datetime,hashlib,json,os,sys,time,zipfile
from pathlib import Path
from . import boundary_public_self15_final_evaluation105_v1 as parent_eval
from . import boundary_public_self15_training105_v1 as parent_trainer
c=parent_trainer.c;parity=parent_trainer.parity
ROOT=c.ROOT;read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require
SCHEMA='full151105_native_public_only_training_CPU_protocol:1'
ROLE='native52_self15_only_component_training'
ARM='native52_self15_only'
INTERNAL_ARM='shared_public_losses'
ARTIFACT_SCHEMA='full151105_native_public_only_WM_component:1'
INITIAL_SCHEMA='full151105_native_public_only_initial:1'
MANIFEST_SCHEMA='full151105_native_public_only_checkpoint_manifest:1'
RECIPE_SCHEMA='full151105_native_public_only_training_recipe:1'
PREFIX='full151105_native_public_only_training_CPU_'
INPUTS={'public15_eval_protocol','public15_eval_report','public15_eval_saved12'}
OWN={'consumer':ROOT/'src/full151_gradient/native_public_only_training105_v1.py','CLI':ROOT/'scripts/train_full151_native_public_only_cpu_v1.py','checker':ROOT/'scripts/check_full151_native_public_only_training_cpu_v1.py'}
DEPENDENCIES={**parent_trainer.DEPENDENCIES,'parent_trainer':ROOT/'src/full151_gradient/boundary_public_self15_training105_v1.py','parent_eval':ROOT/'src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py'}
RECIPE=ROOT/'protocols/full151105_native_public_only_training_recipe_v1.json'
RECIPE_SHA='ffb89e90aaa03099d2cb444d5c660a61fd8fa44b73b9673bd9651f3ce641b4ee'
SOURCE_CONTRACT=ROOT/'docs/public_native_only_training_contract_v1.md'
SOURCE_CONTRACT_SHA='1766b3fe88ee74015df2907762dd70bd0cd241aefa94b8765c008701324196e1'
HEALTH_READY=ROOT/'outputs/full151105_native_public_only_checkpoint_auditor_preparation_CPU_v1/READY.json'
HEALTH_READY_SHA='748acb519823ec44934ff6ada0e89ebfcc6fb566b2b843904247d2b1713d56be'
COUNTS=parent_trainer.COUNTS
LOSS_FIELDS={'state':'L_state_FP32','self15':'L_self15_sum_FP32','native_pair':'L_native_pair_FP32'}
PARAMETER_NAMES=tuple([f'wm.net.{j}.{field}' for j in (0,1,3,4,6,7,9) for field in ('weight','bias')]+['aux_head.weight','aux_head.bias'])
FIT_DIGEST_KEYS=parent_eval.FIT_DIGEST_KEYS
snapshot_state=parent_trainer.snapshot_state
observed_counts=parent_trainer.observed_counts


def fixed_recipe():
    require(isinstance(RECIPE_SHA,str) and pin(RECIPE)['sha256']==RECIPE_SHA,'Root frozen native-only recipe pending')
    doc=read(RECIPE);require(doc['schema']==RECIPE_SCHEMA,'Native-only numerical recipe namespace');r=doc['recipe']
    old=read(parent_trainer.RECIPE)['recipe'];expected={key:value for key,value in old.items() if key not in ('lambda_mode','lambda_imp','lambda_cons_by_arm','compute_consistency_both_arms')}
    expected.update(compute_auxiliary_losses=False,optimized_loss='L_native_pair_FP32',named_parameter_count=16,active_gradient_parameter_count=14,inactive_head_gradient_None=True,optimizer_state_count=14)
    require(r==expected,'Exact unchanged native numerical recipe plus explicit no-aux/14-active contract')
    return r


def validate_protocol(p,execute=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE and p.get('arm')==ARM,'New native-only protocol/external arm')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is False and p.get('automatic_retry') is False and p.get('background_continue_to_final') is True,'Explicit Root CPU/background-through-final/noRetry contract')
    require(p.get('output_prefix')==PREFIX and type(p.get('output')) is str and Path(p['output']).is_absolute() and type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and p.get('source_pins'),'Exact3 own inputs; never confuse protocol.inputs with payload.input_pins')
    require(type(RECIPE_SHA) is str and same(p['recipe'],{'path':str(RECIPE),'sha256':RECIPE_SHA}),'Frozen new numerical recipe required')
    require(type(SOURCE_CONTRACT_SHA) is str and same(p['source_contract'],{'path':str(SOURCE_CONTRACT),'sha256':SOURCE_CONTRACT_SHA}),'Frozen native-only source contract required')
    if execute:require(p['CPU_run_authorized'],'Root actual training authorization')
    return p


def required_sources(p,family):
    parent_inputs=family['training']['inputs'];inventory=read(checked(parent_inputs['target_source_inventory']));health=read(HEALTH_READY)
    cached=read(checked(parent_inputs['cached_sensor_report']))
    values=list(inventory.values())+list(p['source_pins'].values())+list(p['inputs'].values())+list(parent_inputs.values())+list(family['training']['source_pins'].values())
    values += [p['recipe'],p['source_contract'],p['source_review'],pin(HEALTH_READY),health['cpu_report'],cached['protocol']]+list(cached['source_pins'].values())+list(health['source_pins'].values())+[pin(path) for path in list(OWN.values())+list(DEPENDENCIES.values())]
    return c.canonical_sources(values)


def metadata_preflight(protocol_path,output=None):
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Default JSON/hash only, no data tensors/native backend')
    p=validate_protocol(read(protocol_path));r=fixed_recipe();i=p['inputs']
    for value in i.values():checked(value)
    require(pin(DEPENDENCIES['parent_eval'])['sha256']=='81ad78161c75b1720a77aead414ccb6c5a447b0f614e5e597a940aaa1ea77419' and pin(DEPENDENCIES['parent_trainer'])['sha256']=='796c360539961143fe1f588fd18eb08e4b168d141b34daee23da802a72194e7e','Unmodified parent metadata/data helper sources')
    family=parent_eval.metadata_preflight(checked(i['public15_eval_protocol']))
    report=read(checked(i['public15_eval_report']));review=read(checked(i['public15_eval_saved12']));c.allchecks(report);c.allchecks(review)
    require(report['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1' and report['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION' and same(report['protocol'],i['public15_eval_protocol']),'Actual current-family saved eval, metadata provenance only')
    require(review['schema']=='full151105_boundary_public_self15_final_prediction_saved_independent:1' and review['status']=='PASS_SAVED_ALL3_ALL152_PUBLIC_SELF15_FIXED_FINAL_PREDICTIONS' and review['engineering_qualified'] is True and same(review['producer_report'],i['public15_eval_report']) and same(review['protocol'],i['public15_eval_protocol']),'Positive current-family saved12')
    parent_inputs=family['training']['inputs'];require(set(parent_inputs)==parent_trainer.INPUTS,'Source-derived original16 data aliases, not a renamed own3 cohort')
    expected_fit={key:family['training_result'][key] for key in FIT_DIGEST_KEYS}
    require(all(type(value) is str and len(value)==64 and report[key]==value for key,value in expected_fit.items()),'Allfour fixed parent fit digests')
    require(type(HEALTH_READY_SHA) is str and pin(HEALTH_READY)['sha256']==HEALTH_READY_SHA,'New native-only reader READY required')
    ready=read(HEALTH_READY);proof=read(checked(ready['cpu_report']));c.allchecks(proof)
    source_review=read(checked(p['source_review']));c.allchecks(source_review)
    require(source_review['schema']=='full151105_native_public_only_training_source_independent:1' and source_review['source_review_qualified'] is True,'New source review, not prior matched trainer qualification')
    for path in list(OWN.values())+list(DEPENDENCIES.values()):require(any(same(pin(path),v) for v in p['source_pins'].values()),'Current native-only source membership')
    for path in OWN.values():require(any(same(pin(path),v) for v in source_review['source_pins'].values()),'Independent source review pins current producer bytes')
    for value in ready['source_pins'].values():require(any(same(value,v) for v in p['source_pins'].values()),'New reader direct sources enrolled')
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and out==Path(p['output']).resolve() and not out.exists(),'Fresh exact protocol output; no resume/overwrite')
    return {'status':'PASS_NATIVE_ONLY_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'recipe':r,'family':family,'parent_training_inputs':parent_inputs,'expected_fit_digests':expected_fit,'sources':required_sources(p,family),'output':out,'actual_data_load_model_FW_AD_optimizer':0}


def declaration():
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Metadata-only declaration')
    return {'schema':'full151105_native_public_only_training_declaration:1','status':'UNFROZEN_SOURCE_DRAFT_NOT_EXECUTABLE_BEFORE_NEW_RECIPE_READER_AND_REVIEW','protocol_schema':SCHEMA,'role':ROLE,'arm':ARM,'internal_wrapper_arm':INTERNAL_ARM,'input_aliases':sorted(INPUTS),
        'budget':{'own_initial_load':1,'native_WM_constructor':1,'raw_copies':2,'training_native_FW':1000,'training_rows':152000,'training_head_FW':1000,'unused_factory_component_FW':0,'backward_optimizer_steps':1000,'extra_validation_FW':0,'Actor_119f_MJB_Env_GPU_physics':0},
        'active_gradient_parameters':14,'head_gradient_None_parameters':2,'optimizer_group_parameters':16,'optimizer_state_parameters':14,'pending_freeze':{'recipe':RECIPE_SHA,'source_contract':SOURCE_CONTRACT_SHA,'reader_READY':HEALTH_READY_SHA}}


def native_only_losses(delta52,scale52,logits15,target15,r,torch):
    """Only native point/state + cached15; no auxiliary tensor/loss arguments."""
    require(delta52.ndim==2 and delta52.shape[1]==52 and scale52.shape==(52,) and delta52.dtype==scale52.dtype==torch.float32 and bool(torch.isfinite(delta52).all()) and bool(torch.isfinite(scale52).all()) and bool((scale52>0).all()),'Native point52 and effective frozen52')
    require(logits15.shape==target15.shape==(delta52.shape[0],15) and logits15.dtype==target15.dtype==torch.float32 and not target15.requires_grad and bool(torch.isfinite(logits15).all()) and bool(torch.isfinite(target15).all()) and bool(((target15==0)|(target15==1)).all()),'Same-row FP32 cached binary15 only')
    squared=(delta52/scale52).square();point_row=squared.mean(-1)
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits15,target15,reduction='none');self_row=bce.sum(-1)
    native_row=point_row+torch.tensor(r['lambda_self15'],dtype=torch.float32,device=point_row.device)*self_row
    losses={'L_state_FP32':point_row.mean(),'L_self15_sum_FP32':self_row.mean(),'L_native_pair_FP32':native_row.mean()}
    require(all(value.numel()==1 and bool(torch.isfinite(value)) for value in losses.values()),'Finite native-only three scalars')
    operands={'native_D52_FP32':delta52,'effective_s52_FP32':scale52,'point_squared52_FP32':squared,'point_per_row_FP32':point_row,'self15_logits_FP32':logits15,'self15_target_FP32':target15,'self15_BCE_elements_FP32':bce,'self15_sum_per_row_FP32':self_row,'native_pair_per_row_FP32':native_row}
    return losses,operands


def clip_native_gradients(module,iteration,torch,save_raw):
    names,params=parent_trainer.ordered_parameters(module,PARAMETER_NAMES)
    require(len(params)==16 and all(p.grad is not None and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) for p in params[:14]) and all(p.grad is None for p in params[14:]),'14 finite WM gradients +2 auxiliary None, all16 remain requires_grad')
    pre={name:None if p.grad is None else p.grad.detach().clone() for name,p in zip(names,params)} if save_raw else None
    returned=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True,foreach=False)
    postnorm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad,2) for p in params if p.grad is not None]),2)
    require(bool(torch.isfinite(returned)) and bool(torch.isfinite(postnorm)) and all(p.grad is None for p in params[14:]),'Finite clip16 naturally skips None')
    if not save_raw:return None
    return {'iteration':iteration,'tag':'PRE_OPTIMIZER_I','weights_after_updates':iteration-1,'parameter_names':names,'pre_clip':pre,'post_clip':{name:None if p.grad is None else p.grad.detach().clone() for name,p in zip(names,params)},'max_norm':1.,'pre_clip_returned_norm_FP32':returned.detach().clone(),'post_clip_total_norm_FP32':postnorm.detach().clone()}


def optimizer_steps_exact(optimizer,params,iteration,torch):
    return len(params)==16 and len(optimizer.param_groups)==1 and len(optimizer.param_groups[0]['params'])==16 and all(a is b for a,b in zip(optimizer.param_groups[0]['params'],params)) and len(optimizer.state)==14 and all(p in optimizer.state and isinstance(optimizer.state[p].get('step'),torch.Tensor) and bool(torch.isfinite(optimizer.state[p]['step'])) and float(optimizer.state[p]['step'])==iteration for p in params[:14]) and all(p not in optimizer.state and p.grad is None for p in params[14:])


def loss_snapshot_before_update(losses,operands,iteration,torch):
    return {'iteration':iteration,'tag':'PRE_OPTIMIZER_I','weights_after_updates':iteration-1,**snapshot_state(operands,torch),'scalars':{name:snapshot_state(losses[key],torch) for name,key in LOSS_FIELDS.items()}}


def checkpoint_payload(module,optimizer,counts,gradients,iteration,common,rng,log_prefix,loss_snapshot,torch):
    return {'schema':ARTIFACT_SCHEMA,'manifest_schema':MANIFEST_SCHEMA,'arm':ARM,'internal_wrapper_arm':INTERNAL_ARM,'iteration':iteration,'checkpoint_tag':'AFTER_OPTIMIZER_I',**common,
        'WM_state':snapshot_state(module.wm.state_dict(),torch),'aux_head_state':snapshot_state(module.aux_head.state_dict(),torch),'physical_context_state':snapshot_state(module.context.state_dict(),torch),'normalizer_host_configuration':c._norm_config(module.wm),'ordered_parameter_names':[name for name,p in module.named_parameters()],
        'optimizer_state':snapshot_state(optimizer.state_dict(),torch),'counts':dict(counts),'RNG':snapshot_state(rng,torch),'gradient_snapshot':gradients,'log_prefix':dict(log_prefix),'loss_snapshot':snapshot_state(loss_snapshot,torch)}


def estimate_eta_100(update_seconds,save_seconds,elapsed_seconds,now_utc):
    require(len(update_seconds)==100 and all(type(value) in (int,float) and value>=0 for value in update_seconds),'100 actual single-arm update costs')
    mean=sum(update_seconds[3:])/97;remaining=900*mean+save_seconds+20.
    return {'scope':'Planning heuristic, not CI;900 single-arm updates remaining','completed_updates':100,'remaining_updates':900,'trim_first_update_count':3,'observed_mean_update_seconds_after_trim':mean,'observed_checkpoint100_save_seconds':save_seconds,'actual_elapsed_including_setup_archive_save_seconds':elapsed_seconds,'explicit_unmeasured_finalization_overhead_seconds':20.,'estimated_remaining_seconds':remaining,'estimated_total_elapsed_seconds':elapsed_seconds+remaining,'planning_remaining_range_seconds':[.85*900*mean+save_seconds+20.,1.3*900*mean+save_seconds+20.],'estimated_finish_UTC':(now_utc+datetime.timedelta(seconds=remaining)).isoformat(),'independent_health_reader_time_not_included':True,'background_continuation_is_protocol_field':True}


def run_training(protocol_path,output,emit=None):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);r=m['recipe'];i=p['inputs'];data_inputs=m['parent_training_inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name,value in m['sources'].items():archive.write(checked(value),name)
    phase='setup';modules={};iteration=0;extra={'backward_calls':0,'optimizer_steps':0};offset={};counts={key:0 for key in COUNTS};update_seconds=[];saved=[]
    process_counts={'own_initial_torch_loads':0,'native_WM_constructors':0,'raw_deepcopies':0,'validation_FW':0,'native_difference_calls':0,'unused_factory_component_FW':0,'unused_factory_component_optimizer_steps':0,'Actor_old119f_MJB_Env_GPU_physics':0}
    def progress(status,**fields):
        result={'schema':'full151105_native_public_only_training_runtime:1','status':status,'arm':ARM,'phase':phase,'completed_iteration':iteration,'training_counts':dict(counts),'process_counts':dict(process_counts),'elapsed_seconds':time.perf_counter()-started,**fields}
        c._write_json(out/'runtime.json',result,replace=True)
        if emit is not None:emit(result)
        return result
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        from . import boundary_impulse_component105_v1 as component
        from . import component_H8_donor105_v1 as h8
        from dex_supermimic.rl.dex_supermimic.state import difference
        torch.set_num_threads(4);rng_start=c._rng(torch,np);h8.cf.save_raw_rng(out,'startup',rng_start,np)
        with np.load(checked(data_inputs['target_arrays']),allow_pickle=False) as f:raw_data={key:f[key].copy() for key in f.files}
        data=parity.prepare_arrays(raw_data,np);plan=read(checked(data_inputs['target_identities']));joined=read(checked(data_inputs['cached_sensor_labels']))
        target15=torch.tensor(parent_trainer.cached_label_rows(plan,joined),dtype=torch.float32)
        parent_trainer.validate_target_identity_arrays(plan,raw_data['source_group_id152'],raw_data['source_substep_rows152x4'],np)
        with np.load(checked(data_inputs['initial_parity_baseline']),allow_pickle=False) as f:baseline={key:torch.from_numpy(f[key].copy()) for key in ('prediction58_FP32','self15_logits_FP32')}
        initial=torch.load(checked(data_inputs['initial_component']),map_location='cpu',weights_only=False);process_counts['own_initial_torch_loads']+=1
        xi=initial['physical_xi46'];scale=initial['frozen_s52'];require(component.bit_equal(xi,torch.from_numpy(data['physical_xi46_FP32'])),'Same original target/initial xi46')
        x,u,target=(torch.from_numpy(data[key]) for key in ('x58_FP32','u20_FP32','target58_FP32'))
        dynamics=torch.cat((x,xi[None].expand(152,-1),torch.zeros((152,174),dtype=torch.float32)),-1)
        # Only preserve old label bytes for cohort provenance. Never feed these
        # tensors into forward/loss or compute mode/impulse/consistency losses.
        old_labels={key:torch.from_numpy(data[key]) for key in ('J_palm_F64','K_palm_F64','boundary_constraint_witness','r_v_saved_m_s_F64','r_w_saved_rad_s_F64')}
        combined={**old_labels,'cached_self15_target_FP32':target15}
        def fit_digests():
            return {'fit_input_digest':c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch),'fit_label_digest':c._tensor_digest(old_labels,torch),'fit_self15_label_digest':c._tensor_digest({'cached_self15_target_FP32':target15},torch),'fit_combined_label_digest':c._tensor_digest(combined,torch)}
        fit=fit_digests();require(fit==m['expected_fit_digests'] and fit['fit_input_digest']==read(checked(data_inputs['initial_parity_report']))['input_digest_before'],'Allfour actual original fit digests, not new aliases silently swapped')
        ctx=component.fixed_context_from_rows(torch.from_numpy(data['mass_actual_F64'].astype(np.float32)),torch.from_numpy(data['I_body_F64']),torch.from_numpy(data['gravity_palm_F64']),torch.from_numpy(data['DeltaT_F64']),torch.from_numpy(data['palm_rot_raw_FP32']),scale,xi)
        raw=h8.make_initial_eval_component(initial,read(checked(data_inputs['initial_validation'])),torch);process_counts['native_WM_constructors']+=1
        modules,construction=component.make_matched_components(raw,ctx);process_counts['raw_deepcopies']=2
        module=modules[INTERNAL_ARM];unused=modules[component.ARMS[1]];unused.eval().requires_grad_(False)
        unused_before=component.state_digest(unused);zero_head_before=component.state_digest(module.aux_head)
        require(all(component.bit_equal(value,torch.zeros_like(value)) for value in module.aux_head.state_dict().values()),'Original direct-zero auxiliary parameter bytes')
        names,params=parent_trainer.ordered_parameters(module,PARAMETER_NAMES)
        require(len(params)==16 and all(p.requires_grad for p in params),'Same16 optimizer-group parameters, not head requires_grad=False shortcut')
        optimizer=torch.optim.AdamW(params,lr=r['lr'],weight_decay=r['weight_decay'],betas=tuple(r['betas']),eps=r['optimizer_eps'],foreach=False)
        offset=dict(module.counts);counts=observed_counts(module,extra,offset)
        common={'protocol':m['protocol_pin'],'recipe':p['recipe'],'input_pins':i,'parent_training_inputs':data_inputs,'parent_training_protocol':m['family']['protocol']['inputs']['training_protocol'],'source_pins':p['source_pins'],**fit,
            'active_gradient_parameter_names':names[:14],'inactive_gradient_parameter_names':names[14:],'zero_head_state_sha256':zero_head_before,
            'first_training_forward_parity':{'status':'PENDING_ITERATION1_NO_EXTRA_FW','source_baseline':data_inputs['initial_parity_baseline'],'source_parity_report':data_inputs['initial_parity_report']}}
        rng_this_run=c._rng(torch,np);require(all(parity.raw_rng_equality(rng_start,rng_this_run,np,torch).values()),'This-run initialization RNG unchanged without repair')
        shared_initial={'schema':INITIAL_SCHEMA,'arm':ARM,'internal_wrapper_arm':INTERNAL_ARM,**common,'source_parent':data_inputs['initial_component'],
            'WM_state':snapshot_state(module.wm.state_dict(),torch),'aux_head_state':snapshot_state(module.aux_head.state_dict(),torch),'physical_context_state':snapshot_state(module.context.state_dict(),torch),
            'normalizer_host_configuration':c._norm_config(module.wm),'ordered_parameter_names':names,'RNG_this_run':snapshot_state(rng_this_run,torch),'construction_receipt':construction,
            'wrapper_start_count_offsets':offset,'training_counts':dict(counts),'cached_self15_target_FP32':target15.detach().clone(),'source_identity_fields':copy.deepcopy(plan)}
        c._atomic_torch_save(out/'initial_component_state.pt',shared_initial,torch)
        phase='native_only_training';progress('running');fit_started=time.perf_counter()
        with (out/'updates.jsonl').open('xb') as log:
            for current in range(1,r['iterations']+1):
                if current%100==0:c._write_json(out/f'progress_before_{current:04d}.json',progress('running',entering_iteration=current))
                phase='update_'+str(current);update_started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                prediction,self15,z,mode=module(x,u,dynamics)
                if current==1:
                    require(component.bit_equal(prediction.detach(),baseline['prediction58_FP32']) and component.bit_equal(self15.detach(),baseline['self15_logits_FP32']) and component.bit_equal(z.detach(),torch.zeros_like(z)) and component.bit_equal(mode.detach(),torch.zeros_like(mode)),'First training forward saved initial/native58+self15/+0head parity, no extra FW')
                    common['first_training_forward_parity']={**common['first_training_forward_parity'],'status':'PASS_ITERATION1_NATIVE58_SELF15_AND_POSITIVE_ZERO14_BITS','training_iteration':1,'extra_validation_FW':0}
                delta=difference(prediction,target);process_counts['native_difference_calls']+=1;chart=c.chart_guard(prediction,target,delta,r['native_log_chart_margin_rad'],torch)
                losses,operands=native_only_losses(delta,scale,self15,target15,r,torch)
                loss_snapshot=loss_snapshot_before_update(losses,operands,current,torch) if current in r['checkpoint_iterations'] else None
                losses['L_native_pair_FP32'].backward();extra['backward_calls']+=1
                gradients=clip_native_gradients(module,current,torch,current in r['checkpoint_iterations'])
                optimizer.step();extra['optimizer_steps']+=1
                require(all(bool(torch.isfinite(q).all()) for q in params) and c._finite_tree(optimizer.state_dict(),torch) and optimizer_steps_exact(optimizer,params,current,torch),'Finite parameters/14 active rawAdam exact step values, no head state')
                require(component.state_digest(module.aux_head)==zero_head_before and all(component.bit_equal(value,torch.zeros_like(value)) for value in module.aux_head.state_dict().values()),'Aux head exact positive-zero state unchanged by native-only objective')
                module.assert_frozen();unused.assert_frozen()
                require(all(value==0 for value in unused.counts.values()),'Second RAW-copy wrapper unused:0FW/optimizer')
                counts=observed_counts(module,extra,offset);require(all(value==current for value in counts.values()),'One native/head FW/backward/update each iteration')
                iteration=current;seconds=time.perf_counter()-update_started;update_seconds.append(seconds)
                row={'iteration':current,'arms':{ARM:{'counts':dict(counts),'losses':{key:float(losses[value].detach()) for key,value in LOSS_FIELDS.items()},'update_seconds':seconds,'native_chart':chart}}}
                log.write((json.dumps(row,allow_nan=False)+'\n').encode('utf-8'));log.flush()
                if current%100==0:os.fsync(log.fileno())
                if current in r['checkpoint_iterations']:
                    phase='save_checkpoint_'+str(current);progress('saving_checkpoint');save_started=time.perf_counter();prefix_bytes=log.tell()
                    prefix={'path':str(out/'updates.jsonl'),'records':current,'bytes':prefix_bytes,'sha256':hashlib.sha256((out/'updates.jsonl').read_bytes()[:prefix_bytes]).hexdigest()}
                    rng=c._rng(torch,np);require(all(parity.raw_rng_equality(rng_this_run,rng,np,torch).values()),'This-run raw RNG unchanged at checkpoint, no restore')
                    require(fit_digests()==fit,'Frozen actual152 inputs/target/cached15 and old provenance labels unchanged')
                    require(component.state_digest(unused)==unused_before,'Unused wrapper complete parameter/buffer state unchanged')
                    payload=checkpoint_payload(module,optimizer,counts,gradients,current,{**common,'native_difference_calls':process_counts['native_difference_calls']},rng,prefix,loss_snapshot,torch)
                    checkpoint=out/f'{ARM}_update_{current:04d}.pt';c._atomic_torch_save(checkpoint,payload,torch)
                    save_seconds=time.perf_counter()-save_started
                    checks={'raw_Adam14_steps_inmemory':optimizer_steps_exact(optimizer,params,current,torch),'head2_None_no_Adam_positive_zero_bytes':all(q.grad is None and q not in optimizer.state for q in params[14:]) and component.state_digest(module.aux_head)==zero_head_before,'unused_second_wrapper_0FW_unchanged':all(value==0 for value in unused.counts.values()) and component.state_digest(unused)==unused_before}
                    health={'schema':'full151105_native_public_only_inmemory_health:1','arm':ARM,'iteration':current,'checks':checks,'passed':len(checks),'total':len(checks),'engineering_inmemory_pass':all(checks.values()),'serialized_readback_performed':False,'independent_saved_health_pending':True,'gradient_phase':'PRE_OPTIMIZER_I;checkpoint weights AFTER_OPTIMIZER_I'}
                    health_path=out/f'checkpoint_{current:04d}_inmemory_health.json';c._write_json(health_path,health)
                    manifest={'schema':MANIFEST_SCHEMA,'arm':ARM,'iteration':current,'protocol':m['protocol_pin'],'recipe':p['recipe'],'initial_state':pin(out/'initial_component_state.pt'),'single_checkpoint':pin(checkpoint),'producer_health':pin(health_path),'log_prefix':prefix,'save_seconds':save_seconds,'independent_health_required_before_Goal_pause':True}
                    c._write_json(out/f'checkpoint_{current:04d}.json',manifest);saved.append(manifest)
                    if current==100:c._write_json(out/'checkpoint100_ETA.json',estimate_eta_100(update_seconds,save_seconds,time.perf_counter()-started,datetime.datetime.now(datetime.timezone.utc)))
                if current%100==0:c._write_json(out/f'progress_after_{current:04d}.json',progress('running'))
                if current==100 and not p['background_continue_to_final']:break
        phase='finalization';require(fit_digests()==fit and all(parity.raw_rng_equality(rng_this_run,c._rng(torch,np),np,torch).values()) and component.state_digest(unused)==unused_before and process_counts['native_difference_calls']==iteration,'Final frozen fit tensors, unused component, difference budget and this-run RNG unchanged')
        for value in m['sources'].values():checked(value)
        for value in [*i.values(),*data_inputs.values()]:checked(value)
        require(same(pin(protocol_path),m['protocol_pin']),'Run protocol unchanged')
        status='COMPLETED_FIXED1000_NATIVE_PUBLIC_ONLY_TRAINING' if iteration==1000 else 'STOPPED_AFTER_SINGLE100_EXPLICIT_PROTOCOL_NO_RESUME'
        report={'schema':'full151105_native_public_only_training_completion:1','status':status,'arm':ARM,'protocol':m['protocol_pin'],'recipe':p['recipe'],'input_pins':i,'parent_training_inputs':data_inputs,'source_pins':p['source_pins'],'iteration':iteration,'training_counts':counts,'process_counts':process_counts,**fit,
            'checkpoints':saved,'first_training_forward_parity':common['first_training_forward_parity'],'training_elapsed_seconds':time.perf_counter()-fit_started,'total_elapsed_seconds':time.perf_counter()-started,'independent_saved_checkpoint_health_pending':True,'background_continue_to_final':p['background_continue_to_final'],
            'native105_or_policy_resume_qualified':False,'whole_hidden_state_qualified':False,'physical_force_or_gradient_qualified':False,'self15_supervised':True,'self15_label_phase':'lastsolver_preintegration~kdt+.015;native target58 POST20','auxiliary_losses_computed':False,'auxiliary_head_optimized':False}
        c._write_json(out/'result.json',report);progress(status);c._write_json(out/'output_pins_final.json',{path.name:pin(path) for path in out.iterdir() if path.is_file()});return report
    except BaseException as error:
        try:
            if INTERNAL_ARM in modules:counts=observed_counts(modules[INTERNAL_ARM],extra,offset or {key:0 for key in modules[INTERNAL_ARM].counts})
            failure=progress('FAILED_PARTIAL_NO_RETRY',error=repr(error));c._write_json(out/'failure.json',failure)
        except BaseException as save_error:print(json.dumps({'original_error':repr(error),'failure_artifact_error':repr(save_error),'phase':phase,'completed_iteration':iteration}),file=sys.stderr)
        raise
    finally:
        for module in modules.values():module.close()
