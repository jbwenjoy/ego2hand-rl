"""UNFROZEN: fixed initial + two AFTER1000 components, three saved-data FWs.

Default JSON/hash only. Three own loads, one H8 native constructor, two raw
copies before hooks; no wrapped deepcopy/pickle, optimizer, AD or physics.
"""
from __future__ import annotations
import os,sys,time,zipfile
from pathlib import Path
from . import boundary_component_initial_parity105_v1 as parity
from . import boundary_public_self15_training105_v1 as trainer
c=parity.c;ROOT=c.ROOT;read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require
SCHEMA='full151105_boundary_public_self15_final_evaluation_CPU_protocol:1'
ROLE='fixed_initial_and_two_public_self15_final1000_component_evaluation'
PREFIX='full151105_boundary_public_self15_final_evaluation_CPU_'
ARMS=('shared_public_losses','shared_public_plus_cross_target_consistency')
ROLES=('unwrapped_baseline',)+ARMS
CONTRACT=ROOT/'docs/full151_boundary_public_self15_final_evaluation_contract_v1.md'
CONTRACT_SHA='94268782d3f1cb282e123145cd85245d8f51ee8eabddc9e872c279e27e4d9046'
INPUTS={'target_report','target_arrays','target_identities','target_saved8','target_source_inventory','original_initial_component','initial_validation',
        'initial_saved_parity10','initial_saved_baseline','initial_parity_report','training_protocol','training_result','final_pair_manifest','final_saved_health','cached_sensor_labels','cached_sensor_report','cached_sensor_saved7'}
OWN={'consumer':ROOT/'src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py','CLI':ROOT/'scripts/evaluate_full151_boundary_public_self15_components_final_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_boundary_public_self15_components_final_cpu_v1.py'}
DEPENDENCIES={**parity.DEPENDENCIES,'parity_helper':ROOT/'src/full151_gradient/boundary_component_initial_parity105_v1.py',
              'public_self15_trainer':ROOT/'src/full151_gradient/boundary_public_self15_training105_v1.py',
              'cached_label_source':ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py'}
TRAINER_SHA='796c360539961143fe1f588fd18eb08e4b168d141b34daee23da802a72194e7e'
FIT_DIGEST_KEYS=('fit_input_digest','fit_label_digest','fit_self15_label_digest','fit_combined_label_digest')


def validate_protocol(p,execute=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE and p.get('output_prefix')==PREFIX,'New fixed-final evaluation namespace')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is False and p.get('training_authorized') is False,'Explicit CPU evaluation, no training/GPU')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and p.get('source_pins'),'Exact17 semantic inputs')
    if execute:require(p['CPU_run_authorized'],'Root explicit final evaluation authorization')
    return p


def metadata_preflight(protocol_path,output=None):
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Default metadata only')
    p=validate_protocol(read(protocol_path));i=p['inputs']
    for value in i.values():checked(value)
    require(pin(CONTRACT)['sha256']==CONTRACT_SHA,'Frozen scientific source contract')
    pair=read(checked(i['final_pair_manifest']));health=read(checked(i['final_saved_health']));c.allchecks(health)
    require(pair['schema']=='full151105_boundary_public_self15_matched_pair:1' and pair['iteration']==1000 and set(pair['checkpoints'])==set(ARMS),'Fixed AFTER1000 pair only')
    require(health['schema']=='full151105_boundary_public_self15_checkpoint_saved_review:1' and health['status']=='PASS_SAVED_PAIRED_PUBLIC_SELF15_COMPONENT_HEALTH_NOT_ACCURACY' and health['engineering_qualified'] is True and health['iteration']==1000 and health['final_iteration_scope'] is True,'Actual final1000 saved health, not accuracy')
    require(same(health['producer_pair_manifest'],i['final_pair_manifest']) and same(health['protocol'],i['training_protocol']) and same(pair['protocol'],i['training_protocol']),'One actual trained pair/protocol')
    for arm in ARMS:require(same(pair['checkpoints'][arm],health['checkpoints'][arm]),'Exact own final component file')
    train=read(checked(i['training_protocol']));result=read(checked(i['training_result']))
    require(train['schema']==trainer.SCHEMA and train['role']==trainer.ROLE and set(train['inputs'])==trainer.INPUTS and same(train['recipe'],pin(trainer.RECIPE)),'Actual new public15 protocol namespace/16inputs/frozen recipe')
    require(result['schema']=='full151105_boundary_public_self15_matched_training_completion:1' and result['iteration']==1000 and result['status']=='COMPLETED_FIXED1000_MATCHED_PUBLIC_SELF15_COMPONENT_TRAINING' and result['self15_supervised'] is True and same(result['protocol'],i['training_protocol']),'Complete fixed1000 public15 source run')
    require(set(result['input_pins'])==set(train['inputs']) and all(same(v,train['inputs'][k]) for k,v in result['input_pins'].items()),'Result.input_pins maps actual protocol.inputs, never a fabricated old field')
    for k in ('target_report','target_arrays','target_identities','target_saved8','target_source_inventory','initial_validation','initial_parity_report','initial_parity_baseline','cached_sensor_labels','cached_sensor_report','cached_sensor_saved7'):
        alias='initial_saved_baseline' if k=='initial_parity_baseline' else k
        require(same(train['inputs'][k],i[alias]),'Same fitted original152 cohort')
    require(same(train['inputs']['initial_component'],i['original_initial_component']),'Unmodified original own initial, no fake initialpair conversion')
    require(pin(DEPENDENCIES['public_self15_trainer'])['sha256']==TRAINER_SHA and any(same(pin(DEPENDENCIES['public_self15_trainer']),v) for v in train['source_pins'].values()),'Executed frozen796c loss/label helper only')
    cached=read(checked(i['cached_sensor_report']));cached_review=read(checked(i['cached_sensor_saved7']));c.allchecks(cached_review)
    require(cached['schema']=='full151105_cached_self15_saved_evaluation:1' and cached['status']=='COMPLETED_ALL152_CACHED_SELF15_LABEL_METRICS_NOT_GEOMETRY_ACCURACY' and cached['counts']['selected_labels']==152,'Completed original152 cached native15 source, no metric NPZ target')
    require(cached_review['schema']=='full151105_cached_self15_saved_independent:1' and cached_review['status']=='PASS_ALL152_ALL3ROLES_CACHED15_METRICS_NOT_GEOMETRY' and cached_review['engineering_qualified'] is True and same(cached_review['producer_report'],i['cached_sensor_report']),'Positive cached15 source review, not a geometry qualification')
    require(same(cached['input_pins']['target_report'],i['target_report']) and same(cached['input_pins']['target_group152'],i['target_identities']),'Labels/target same original row authority')
    cached_pins=read(checked(i['cached_sensor_report']).parent/'output_pins.json')
    require(same(cached_pins['joined152_cached_sensor_labels.json'],i['cached_sensor_labels']) and checked(i['cached_sensor_labels']).samefile(checked(i['cached_sensor_report']).parent/'joined152_cached_sensor_labels.json'),'Only exact label-only JSON')
    recipe_doc=read(trainer.RECIPE)
    require(same(recipe_doc['cached15_label_only_JSON'],i['cached_sensor_labels']) and same(recipe_doc['cached15_saved_review7'],i['cached_sensor_saved7']) and same(recipe_doc['source_targets'],i['target_arrays']),'Same original targets and label-only fixed-recipe source')
    initial_parity=read(checked(i['initial_parity_report']));saved_parity=read(checked(i['initial_saved_parity10']));c.allchecks(initial_parity);c.allchecks(saved_parity)
    require(saved_parity['function_identity_qualified'] is True and same(saved_parity['producer_report'],i['initial_parity_report']),'Independent actual initial-function proof')
    require(same(train['inputs']['initial_parity_saved_review'],i['initial_saved_parity10']),'Same independent initial proof used by training')
    require(same(initial_parity['input_pins']['initial_component'],i['original_initial_component']) and same(initial_parity['input_pins']['target_arrays'],i['target_arrays']),'Same cached initial baseline authority')
    for alias,parent,name in (('target_arrays','target_report','all152_boundary_impulse_targets.npz'),('target_identities','target_report','group152_identities.json'),('target_source_inventory','target_report','source_inventory.json'),('initial_saved_baseline','initial_parity_report','unwrapped_baseline_forward152.npz')):
        require(checked(i[alias]).samefile(checked(i[parent]).parent/name),'Actual artifact route')
    target=read(checked(i['target_report']));target_review=read(checked(i['target_saved8']));c.allchecks(target_review)
    require(target['enrollment_qualified'] is True and same(target_review['producer_report'],i['target_report']),'Conditional original target152 only')
    for value in list(OWN.values())+list(DEPENDENCIES.values())+[CONTRACT]:require(any(same(pin(value),v) for v in p['source_pins'].values()),'Current evaluator/helper/contract source membership')
    # Exactly one prior map, plus direct own final files; no recursive aliases.
    sources=c.canonical_sources(list(read(checked(i['target_source_inventory'])).values())+list(p['source_pins'].values())+list(train['source_pins'].values())+list(health['source_pins'].values())+list(i.values())+list(pair['checkpoints'].values())+[train['recipe'],health['initial_pair'],cached['protocol']])
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists() and ('output' not in p or out==Path(p['output']).resolve()),'Fresh exact owned evaluation output')
    return {'status':'PASS_PUBLIC_SELF15_FINAL_COMPONENT_EVAL_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'sources':sources,'output':out,'pair':pair,'training':train,'training_result':result,'recipe':trainer.fixed_recipe(),'actual_tensor_loads_FW':0}


def declaration():
    return {'schema':'full151105_boundary_public_self15_final_eval_declaration:1','status':'UNFROZEN_SOURCE_ONLY','protocol_schema':SCHEMA,'role':ROLE,'input_aliases':sorted(INPUTS),
        'contract':pin(CONTRACT),'source_pins':{k:pin(v) for k,v in OWN.items()},'budget':{'own_initial_and_final_loads':3,'native_WM_ctor':1,'raw_deepcopies':2,'native_B152_FW':3,'aux_head_FW':2,'predicted_rows':456,'AD_optimizer_Actor_MJB_physics_GPU':0}}


def load_final_into(module,payload,initial,training_pin,training,expected_fit_digests,torch):
    """Strict own namespace, raw initial norm/context; load WM/head ONLY."""
    from . import boundary_impulse_component105_v1 as component
    require(payload['schema']==trainer.ARTIFACT_SCHEMA and payload['pair_schema']=='full151105_boundary_public_self15_matched_pair:1' and payload['arm']==module.arm and type(payload['iteration']) is int and payload['iteration']==1000 and payload['checkpoint_tag']=='AFTER_OPTIMIZER_I','True own AFTER1000 artifact')
    require(same(payload['protocol'],training_pin) and same(payload['recipe'],training['recipe']),'Frozen training provenance')
    for key in ('input_pins','source_pins'):
        protocol_key='inputs' if key=='input_pins' else key
        require(set(payload[key])==set(training[protocol_key]) and all(same(v,training[protocol_key][k]) for k,v in payload[key].items()),'Exact source/fit cohort pin map')
    require(set(expected_fit_digests)==set(FIT_DIGEST_KEYS) and all(payload[k]==expected_fit_digests[k] for k in FIT_DIGEST_KEYS),'Frozen original152 x/u/target plus old5/new15/combined label byte digests')
    reference=module.wm.state_dict();weights=payload['WM_state']
    require(set(reference)==set(weights) and all(isinstance(weights[k],torch.Tensor) and weights[k].device.type=='cpu' and weights[k].dtype==v.dtype and weights[k].shape==v.shape and bool(torch.isfinite(weights[k]).all()) for k,v in reference.items()),'Strict final native architecture/dtype/finite')
    norm=c._norm_state(module.wm)
    require(all(component.bit_equal(weights[k],v) for k,v in norm.items()) and payload['normalizer_host_configuration']==c._norm_config(module.wm)==initial['normalizer_host_configuration'],'Frozen initial norm buffers/host')
    context=module.context.state_dict();saved=payload['physical_context_state']
    require(set(saved)==set(context) and all(component.bit_equal(saved[k],v) for k,v in context.items()),'Never replace/repair fixed physical context')
    head=module.aux_head.state_dict();savedhead=payload['aux_head_state']
    require(set(head)==set(savedhead) and all(isinstance(savedhead[k],torch.Tensor) and savedhead[k].shape==v.shape and savedhead[k].dtype==v.dtype and savedhead[k].device.type=='cpu' and bool(torch.isfinite(savedhead[k]).all()) for k,v in head.items()),'Strict trained auxiliary14')
    module.wm.load_state_dict(weights,strict=True);module.aux_head.load_state_dict(savedhead,strict=True)
    module.eval().requires_grad_(False);module.assert_frozen()


def forward_three(raw,modules,x,u,dynamics,torch,counts):
    from . import boundary_impulse_component105_v1 as component
    require(x.shape==(152,58) and u.shape==(152,20) and dynamics.shape==(152,278),'One fixed original fullbatch')
    outputs={}
    with torch.inference_mode():
        counts['native_FW_attempted']+=1;pred,self15=raw(x,u,dynamics);counts['native_FW_returned']+=1;counts['predicted_rows']+=152
        outputs[ROLES[0]]={'prediction58_FP32':pred,'self15_logits_FP32':self15,'z12_FP32':torch.zeros((152,2,6),dtype=torch.float32),'mode_logits2_FP32':torch.zeros((152,2),dtype=torch.float32)}
        for arm in ARMS:
            counts['native_FW_attempted']+=1;pred,self15,z,mode=modules[arm](x,u,dynamics);counts['native_FW_returned']+=1;counts['aux_head_FW']+=1;counts['predicted_rows']+=152
            outputs[arm]={'prediction58_FP32':pred,'self15_logits_FP32':self15,'z12_FP32':z,'mode_logits2_FP32':mode}
    for role,values in outputs.items():
        for key,shape in (('prediction58_FP32',(152,58)),('self15_logits_FP32',(152,15)),('z12_FP32',(152,2,6)),('mode_logits2_FP32',(152,2))):component.tensor_check(values[key],shape,torch.float32,role+'/'+key)
    return outputs


def capture_point_probability_proxy(operands,mode_logits,self15_logits,torch):
    """Native FP32 arithmetic captures, separate from any F64 saved rescore."""
    mode_probability=torch.sigmoid(mode_logits);self_probability=torch.sigmoid(self15_logits)
    rate=-self_probability.sum(dim=-1)
    return {'native_point_delta52_FP32':operands['native_D52_FP32'],'native_point_squared_scaled52_FP32':operands['point_squared52_FP32'],
        'mode_probability2_FP32':mode_probability,
        'self15_probability15_FP32':self_probability,'self15_penalty_rate_FP32':rate,'self15_penalty_component_FP32':rate*torch.tensor(.02,dtype=torch.float32)}


def run_evaluation(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,value in m['sources'].items():z.write(checked(value),name)
    counts={'own_trusted_torch_loads':0,'native_WM_constructors':0,'raw_deepcopies':0,'native_FW_attempted':0,'native_FW_returned':0,'aux_head_FW':0,'predicted_rows':0,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0};phase='setup';modules={}
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        torch.set_num_threads(4);rng_before=c._rng(torch,np)
        from . import component_H8_donor105_v1 as h8
        from . import boundary_impulse_component105_v1 as component
        from dex_supermimic.rl.dex_supermimic.state import difference
        h8.cf.save_raw_rng(out,'before',rng_before,np)
        with np.load(checked(i['target_arrays']),allow_pickle=False) as f:
            original_arrays={k:f[k].copy() for k in f.files};data=parity.prepare_arrays(original_arrays,np)
        identities=read(checked(i['target_identities']));gids=original_arrays['source_group_id152'];substeps=original_arrays['source_substep_rows152x4']
        require(type(identities) is list and len(identities)==152 and all(r['row_id']==j for j,r in enumerate(identities)),'All original target identities/order')
        require(gids.dtype==np.int64 and gids.shape==(152,) and np.array_equal(gids,np.asarray([r['group_id'] for r in identities],np.int64)) and substeps.dtype==np.int64 and substeps.shape==(152,4) and np.array_equal(substeps,4*gids[:,None]+np.arange(4)),'Original group/substep array identity join')
        trainer.validate_target_identity_arrays(identities,gids,substeps,np)
        data.update(source_group_id152=gids.copy(),source_substep_rows152x4=substeps.copy())
        data['cached_self15_target_FP32']=np.asarray(trainer.cached_label_rows(identities,read(checked(i['cached_sensor_labels']))),dtype=np.float32)
        initial=torch.load(checked(i['original_initial_component']),map_location='cpu',weights_only=False);counts['own_trusted_torch_loads']+=1
        payloads={}
        for arm in ARMS:payloads[arm]=torch.load(checked(m['pair']['checkpoints'][arm]),map_location='cpu',weights_only=False);counts['own_trusted_torch_loads']+=1
        scale=initial['frozen_s52'];xi=initial['physical_xi46'];require(component.bit_equal(xi,torch.from_numpy(data['physical_xi46_FP32'])),'Same target/initial physical xi46')
        data['effective_s52_FP32']=scale.numpy().copy();np.savez_compressed(out/'prepared_inputs_and_context.npz',**data)
        (out/'group152_identities.json').write_bytes(checked(i['target_identities']).read_bytes())
        x,u,target=(torch.from_numpy(data[k]) for k in ('x58_FP32','u20_FP32','target58_FP32'));dynamics=torch.cat((x,xi[None].expand(152,-1),torch.zeros((152,174),dtype=torch.float32)),dim=-1)
        input_before=c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch)
        require(input_before==read(checked(i['initial_parity_report']))['input_digest_before'],'Exact original training/parity input digest')
        target15=torch.from_numpy(data['cached_self15_target_FP32'])
        labels={k:torch.from_numpy(data[k]) for k in ('J_palm_F64','K_palm_F64','boundary_constraint_witness','r_v_saved_m_s_F64','r_w_saved_rad_s_F64')}
        combined_labels={**labels,'cached_self15_target_FP32':target15}
        fit_digests={'fit_input_digest':input_before,'fit_label_digest':c._tensor_digest(labels,torch),'fit_self15_label_digest':c._tensor_digest({'cached_self15_target_FP32':target15},torch),
                     'fit_combined_label_digest':c._tensor_digest(combined_labels,torch)}
        require(all(m['training_result'][key]==value for key,value in fit_digests.items()),'Exact original152 inputs/old5 labels/cached15 labels match actual completed training')
        ctx=component.fixed_context_from_rows(torch.from_numpy(data['mass_actual_F64'].astype(np.float32)),torch.from_numpy(data['I_body_F64']),torch.from_numpy(data['gravity_palm_F64']),torch.from_numpy(data['DeltaT_F64']),torch.from_numpy(data['palm_rot_raw_FP32']),scale,xi)
        phase='one_native_constructor_then_unwrapped_copies';raw=h8.make_initial_eval_component(initial,read(checked(i['initial_validation'])),torch);counts['native_WM_constructors']+=1
        modules,construction=component.make_matched_components(raw,ctx);counts['raw_deepcopies']=2
        for arm in ARMS:load_final_into(modules[arm],payloads[arm],initial,m['protocol']['inputs']['training_protocol'],m['training'],fit_digests,torch)
        models={'unwrapped_baseline':raw,**modules};before={role:component.state_digest(model) for role,model in models.items()};host_before={role:c._norm_config(raw if role==ROLES[0] else model.wm) for role,model in models.items()}
        np.savez_compressed(out/'normalizer_buffers_before.npz',**{role+'__'+name:value.numpy().copy() for role,model in models.items() for name,value in c._norm_state(raw if role==ROLES[0] else model.wm).items()})
        np.savez_compressed(out/'physical_context_buffers.npz',**{name:value.numpy().copy() for name,value in ctx.state_dict().items()})
        storage,disjoint=parity.named_storage_inventory(models);require(disjoint,'Independent raw/loaded final named storage');c._write_json(out/'named_storage_inventory.json',storage)
        phase='exact_three_native_forward';outputs=forward_three(raw,modules,x,u,dynamics,torch,counts)
        for role,values in outputs.items():np.savez_compressed(out/(role+'_forward152.npz'),**{k:v.numpy().copy() for k,v in values.items()})
        with np.load(checked(i['initial_saved_baseline']),allow_pickle=False) as f:
            require(component.bit_equal(outputs[ROLES[0]]['prediction58_FP32'],torch.from_numpy(f['prediction58_FP32'])) and component.bit_equal(outputs[ROLES[0]]['self15_logits_FP32'],torch.from_numpy(f['self15_logits_FP32'])),'Raw initial same saved function/precision; no extra FW')
        summary={}
        phase='captured_native_point_and_aux_operands'
        with torch.no_grad():
            for role,values in outputs.items():
                delta=difference(values['prediction58_FP32'],target);chart=c.chart_guard(values['prediction58_FP32'],target,delta,.01,torch)
                module=modules[ARMS[0] if role==ROLES[0] else role]
                loss,operands=trainer.public_self15_losses(module,x,values['prediction58_FP32'],values['self15_logits_FP32'],values['z12_FP32'],values['mode_logits2_FP32'],delta,scale,target15,labels,m['recipe'],torch)
                captures=capture_point_probability_proxy(operands,values['mode_logits2_FP32'],values['self15_logits_FP32'],torch)
                tensors={**captures,**operands,**{k:v for k,v in loss.items() if isinstance(v,torch.Tensor)}}
                np.savez_compressed(out/(role+'_descriptive_losses.npz'),**{k:v.numpy().copy() for k,v in tensors.items()})
                summary[role]={**{name:float(loss[key]) for name,key in trainer.LOSS_FIELDS.items()},'native_chart':chart}
        after={role:component.state_digest(model) for role,model in models.items()};host_after={role:c._norm_config(raw if role==ROLES[0] else model.wm) for role,model in models.items()}
        np.savez_compressed(out/'normalizer_buffers_after.npz',**{role+'__'+name:value.numpy().copy() for role,model in models.items() for name,value in c._norm_state(raw if role==ROLES[0] else model.wm).items()})
        for module in modules.values():
            module.assert_frozen();require(all(component.bit_equal(value,ctx.state_dict()[name]) for name,value in module.context.state_dict().items()),'Common saved physical context actual raw bytes remain equal')
        rng_after=c._rng(torch,np);h8.cf.save_raw_rng(out,'after',rng_after,np);rng=parity.raw_rng_equality(rng_before,rng_after,np,torch)
        invariants={'weights_heads_norm_context_unchanged':before==after,'normalizer_host_unchanged':host_before==host_after,'all_inputs_unchanged':input_before==c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch),
            'all_old5_and_cached15_labels_unchanged':fit_digests['fit_label_digest']==c._tensor_digest(labels,torch) and fit_digests['fit_self15_label_digest']==c._tensor_digest({'cached_self15_target_FP32':target15},torch) and fit_digests['fit_combined_label_digest']==c._tensor_digest(combined_labels,torch),
            'raw_rng_values_bytes_unchanged':all(rng.values()),'all_parameters_frozen_no_grad':all(not q.requires_grad and q.grad is None for model in models.values() for q in model.parameters()),
            'all_hook_caches_empty':all(module._cache is None and module.counts['hook_calls']==module.counts['forward_returned']==1 for module in modules.values()),'CUDA_uninitialized_no_Warp_MuJoCo':not torch.cuda.is_initialized() and 'warp' not in sys.modules and 'mujoco' not in sys.modules}
        require(all(invariants.values()) and counts=={'own_trusted_torch_loads':3,'native_WM_constructors':1,'raw_deepcopies':2,'native_FW_attempted':3,'native_FW_returned':3,'aux_head_FW':2,'predicted_rows':456,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0},'Exact fixed evaluation budget/invariants')
        for value in m['sources'].values():checked(value)
        for value in i.values():checked(value)
        for value in m['pair']['checkpoints'].values():checked(value)
        require(same(pin(protocol_path),m['protocol_pin']),'Protocol unchanged')
        report={'schema':'full151105_boundary_public_self15_final_evaluation_CPU_analysis:1','status':'COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION','protocol':m['protocol_pin'],'input_pins':i,'checkpoint_pins':m['pair']['checkpoints'],'counts':counts,'checks':invariants,'passed':len(invariants),'total':len(invariants),
            'outputs':{f.name:pin(f) for f in out.iterdir() if f.is_file()},
            **fit_digests,
            'summary_common_scalars':summary,'raw_rng_comparison':rng,'state_digests_before':before,'state_digests_after':after,'construction_receipt':construction,
            'normalizer_host_configuration_before':host_before,'normalizer_host_configuration_after':host_after,
            'baseline_head_scope':'Positive-zero algebraic initial template, explicitly no baseline head forward; raw native/self15 baseline actually evaluated',
            'loss_scope':'Frozen796c point_row mean52 then state meanrows; self15 BCE sum15 then meanrows; native_pair meanrows(point_row+FP32(.1)*selfsum). FP32 native_pair+mode then F64 impulse/cons. Compare all common8 separately; different consistency-weighted total is not a winner criterion. Old pure-state aliases equal actual new D52/squared52 tensors, not old scalar/public semantics. Raw58 physical-unit statistics are saved-only.',
            'scope':'Known old152 training cohort, not unseen or independent confirmation. Mode>0 is strict with tie0 negative. Self15 targets are original cached lastsolver~15ms sensors, not post20 geometry, HO/human contact or task benefit.',
            'physical_force_or_gradient_qualified':False,'whole_hidden_state_qualified':False,'Actor_or_task_benefit_qualified':False,'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        c._write_json(out/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'no_retry_or_frozen_helper_patch':True});raise
    finally:
        for module in modules.values():module.close()
