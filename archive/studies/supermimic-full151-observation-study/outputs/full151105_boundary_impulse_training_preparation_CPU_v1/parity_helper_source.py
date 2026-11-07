"""REVIEWED SOURCE: real-initial execution remains separately authorized.

Only Root --execute may read real tensors/weights. One own initial payload,
one frozen H8 initial constructor helper, three B152 forwards, no training.
"""
from __future__ import annotations
import csv,json,os,sys,time,zipfile
from pathlib import Path
from . import response_calibration105_v1 as c
ROOT=c.ROOT;read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require
SCHEMA='full151105_boundary_component_initial_parity_CPU_protocol:1'
ROLE='real_initial_boundary_component_function_parity'
PREFIX='full151105_boundary_component_initial_parity_CPU_'
PREREG=ROOT/'protocols/full151105_boundary_component_initial_parity_CPU_prereg_v1.json'
PREREG_SHA='499ec25b5a9ad97c730a3b10ccbc94ecbbc624c2d324290bc14cf7b2632f5bdc'
INPUTS={'target_report','target_arrays','target_saved8','target_source_inventory','initial_component','initial_validation','initial_saved_health29','component_ready','component_source_review','recipe_draft'}
OWN={'consumer':ROOT/'src/full151_gradient/boundary_component_initial_parity105_v1.py','CLI':ROOT/'scripts/diagnose_full151_boundary_component_initial_parity_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_boundary_component_initial_parity_cpu_v1.py'}
DEPENDENCIES={'component_wrapper':ROOT/'src/full151_gradient/boundary_impulse_component105_v1.py','initial_helper':ROOT/'src/full151_gradient/component_H8_donor105_v1.py',
    'confirmation_helper':ROOT/'src/full151_gradient/component_confirmation105_v1.py','native_point_IO':ROOT/'src/full151_gradient/response_calibration105_v1.py'}
COMPONENT_SHA='003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317'
INITIAL_HELPER_SHA='0c77dc5255ffd2d7294dabdc7bbaf8ed09e80f3f2bc0eb89fee3954d268a6b13'


def validate_protocol(p,execute=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE,'Initial parity namespace')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is False and type(p.get('evaluate_auxiliary_losses')) is bool,'Explicit CPU/noGPU/diagnostic option')
    require(p.get('accuracy_gate') is None and p.get('output_prefix')==PREFIX,'Parity engineering, no scientific accuracy gate')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and type(p.get('source_pins')) is dict and p['source_pins'],'Exact10 inputs/current sources')
    if execute:require(p['CPU_run_authorized'],'Root explicit realinitial parity authorization')
    return p


def metadata_preflight(protocol_path,output=None):
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Metadata-only fresh process')
    p=validate_protocol(read(protocol_path));i=p['inputs'];pre=read(PREREG)
    require(PREREG_SHA is not None and same(p['preregistration'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Frozen parity budget/inputs')
    require(set(pre['inputs'])==set(i) and all(same(v,pre['inputs'][k]) for k,v in i.items()),'Exact scientific input identities')
    require(p['evaluate_auxiliary_losses'] is pre['evaluate_auxiliary_losses'],'Frozen optional diagnostic scope')
    for v in i.values():checked(v)
    dataset=read(checked(i['target_report']));review=read(checked(i['target_saved8']));c.allchecks(review)
    require(dataset['schema']=='full151105_saved_boundary_impulse_target_dataset_CPU_analysis:1' and dataset['enrollment_qualified'] is True and all(dataset['guards'].values()),'All152 conditional label dataset enrollment')
    require(dataset['counts']['enrolled_groups']==152 and dataset['counts']['selected_substeps']==608 and dataset['cohort']=={'baseline_groups':24,'direction_groups':128,'donor_groups':0,'fresh_groups':0},'Old152/608 cohort only')
    require(review['engineering_qualified'] is True and same(review['producer_report'],i['target_report']),'Actual independent target8 same run')
    for alias,name in (('target_arrays','all152_boundary_impulse_targets.npz'),('target_source_inventory','source_inventory.json')):
        require(checked(i[alias]).samefile(checked(i['target_report']).parent/name),'Exact target artifact route')
    health=read(checked(i['initial_saved_health29']));c.allchecks(health)
    require(health['schema']=='full151105_WM_component_saved_checkpoint_CPU_review:1' and health['engineering_qualified'] is True and same(health['initial_component'],i['initial_component']),'Trusted own initial, not fabricated final/native105')
    require(checked(i['initial_validation']).samefile(checked(i['initial_component']).parent/'initial_validation.json'),'Actual initial validation route')
    receipt=read(checked(i['initial_validation']));require(receipt['data_rows']==152 and receipt['donor_rows_for_fit']==0,'Original initial source namespace')
    ready=read(checked(i['component_ready']));proof=read(checked(ready['cpu_report']));c.allchecks(proof)
    require(ready['schema']=='full151105_boundary_impulse_component_CPU_READY:1' and ready['status']=='PASS_THREE_TOY_WRAPPER_AND_DIFFERENTIABLE_LOSS_CONTRACTS','Current component toy READY')
    require(ready['source_pins']['component']['sha256']==COMPONENT_SHA and same(ready['recipe_draft'],i['recipe_draft']),'Frozen component/draft pair')
    source=read(checked(i['component_source_review']))
    require(source['schema']=='full151105_boundary_impulse_component_source_review:1' and source['status']=='NO_KNOWN_SOURCE_BLOCKER_FOR_ISOLATED_COMPONENT_TOY_CONTRACT_NOT_TRAINING_QUALIFICATION' and same(source['source_pins']['component'],ready['source_pins']['component']),'Actual source review scope/current component')
    require(pin(DEPENDENCIES['component_wrapper'])['sha256']==COMPONENT_SHA and pin(DEPENDENCIES['initial_helper'])['sha256']==INITIAL_HELPER_SHA,'No changes to frozen helper/component')
    for v in list(OWN.values())+list(DEPENDENCIES.values()):require(any(same(pin(v),q) for q in p['source_pins'].values()),'Current source closure membership')
    inventory=read(checked(i['target_source_inventory']))
    for value in pre['source_pins'].values():require(any(same(value,v) for v in p['source_pins'].values()),'Frozen helper/native source membership')
    sources=c.canonical_sources(list(inventory.values())+list(p['source_pins'].values())+list(i.values())+list(ready['source_pins'].values())+[ready['cpu_report'],pin(PREREG),health['protocol']])
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists(),'Fresh output')
    return {'status':'PASS_REALINITIAL_PARITY_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'sources':sources,'output':out,'actual_arrays_weights_models_FW':0}


def declaration():
    return {'schema':'full151105_boundary_component_initial_parity_declaration:1','status':'REVIEWED_SOURCE_REAL_INITIAL_EXECUTION_PENDING','protocol_schema':SCHEMA,'role':ROLE,
        'input_aliases':sorted(INPUTS),'source_pins':{k:pin(v) for k,v in {**OWN,**DEPENDENCIES}.items()},
        'budget':{'own_trusted_loads':1,'native_WM_constructors':1,'raw_deepcopies':2,'B152_forwards':3,'predicted_rows':456,'AD_optimizer_train_old119f_Actor_MJB_Env_physics_GPU':0},'actual_arrays_weights_models_FW':0}


def prepare_arrays(data,np):
    shapes={'x58_FP32':(152,58),'u20_FP32':(152,20),'target58_FP32':(152,58),'physical_xi46_FP32':(46,),
        'palm_rot_raw_FP32':(152,3,3),'gravity_palm_F64':(152,3),'DeltaT_F64':(152,),'mass_actual_F64':(),'I_body_F64':(3,3),
        'J_palm_F64':(152,2,3),'K_palm_F64':(152,2,3),'r_v_saved_m_s_F64':(152,3),'r_w_saved_rad_s_F64':(152,3),'boundary_constraint_witness':(152,2)}
    for k,shape in shapes.items():
        dtype=np.bool_ if k=='boundary_constraint_witness' else np.float32 if k.endswith('FP32') else np.float64
        a=data[k];require(a.shape==shape and a.dtype==dtype and np.isfinite(a).all(),'Exact conditional target operand '+k)
    require(data['mass_actual_F64']==np.float64(np.float32(data['mass_actual_F64'])),'Actual mass must retain original FP32 source')
    return {k:data[k].copy() for k in shapes}


def triplet_forward(raw,components,x,u,dynamics,torch,counts):
    """No target argument: exactly three same-context inference-only calls."""
    from . import boundary_impulse_component105_v1 as component
    require(x.shape==(152,58) and u.shape==(152,20) and dynamics.shape==(152,278),'Fixed B152 parity input')
    outputs={}
    with torch.inference_mode():
        counts['WM_forward_attempted']+=1;prediction,self15=raw(x,u,dynamics);counts['WM_forward_returned']+=1
        outputs['unwrapped_baseline']={'prediction58_FP32':prediction,'self15_logits_FP32':self15};counts['predicted_rows']+=152
        for arm in component.ARMS:
            counts['WM_forward_attempted']+=1;prediction,self15,z,mode=components[arm](x,u,dynamics);counts['WM_forward_returned']+=1;counts['predicted_rows']+=152
            outputs[arm]={'prediction58_FP32':prediction,'self15_logits_FP32':self15,'z12_FP32':z,'mode_logits2_FP32':mode}
    return outputs


def parity_masks(outputs,torch):
    from . import boundary_impulse_component105_v1 as component
    base=outputs['unwrapped_baseline'];masks={}
    for k,shape in (('prediction58_FP32',(152,58)),('self15_logits_FP32',(152,15))):
        component.tensor_check(base[k],shape,torch.float32,'baseline '+k)
    for arm in component.ARMS:
        out=outputs[arm];row={}
        for k in ('prediction58_FP32','self15_logits_FP32'):
            component.tensor_check(out[k],base[k].shape,torch.float32,arm+' '+k)
            row[k+'_bit_equal']=(out[k].contiguous().view(torch.int32)==base[k].contiguous().view(torch.int32)).all(dim=1)
        for k,shape in (('z12_FP32',(152,2,6)),('mode_logits2_FP32',(152,2))):
            component.tensor_check(out[k],shape,torch.float32,arm+' '+k)
            row[k+'_exact_positive_zero']=(out[k].contiguous().view(torch.int32).reshape(152,-1)==0).all(dim=1)
        masks[arm]=row
    return masks


def raw_rng_equality(before,after,np,torch):
    """Compare saved state values/bytes directly, never pickle storage hashes."""
    a,b=before['numpy'],after['numpy']
    return {'Python_state_values_equal':json.dumps(before['python'],allow_nan=False)==json.dumps(after['python'],allow_nan=False),
        'NumPy_MT_header_equal':a[0]==b[0] and np.asarray(a[2:4],np.int64).tobytes()==np.asarray(b[2:4],np.int64).tobytes() and np.asarray(a[4],np.float64).tobytes()==np.asarray(b[4],np.float64).tobytes(),
        'NumPy_MT_raw_bytes_equal':a[1].dtype==b[1].dtype and a[1].shape==b[1].shape and a[1].tobytes()==b[1].tobytes(),
        'TorchCPU_raw_bytes_equal':before['torch_CPU'].dtype==after['torch_CPU'].dtype and before['torch_CPU'].shape==after['torch_CPU'].shape and torch.equal(before['torch_CPU'],after['torch_CPU']),
        'CUDA_never_initialized':before['CUDA_initialized'] is False and after['CUDA_initialized'] is False}


def named_storage_inventory(models):
    """Named model/physical buffers only; not a general hidden-state claim."""
    inventory={}
    for role,model in models.items():
        rows=[]
        for kind,values in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
            for name,t in values:
                storage=t.untyped_storage()
                rows.append({'kind':kind,'name':name,'dtype':str(t.dtype),'shape':list(t.shape),'stride':list(t.stride()),'storage_offset':t.storage_offset(),
                    'data_ptr':t.data_ptr(),'storage_base_ptr':storage.data_ptr(),'storage_nbytes':storage.nbytes(),'device':str(t.device)})
        inventory[role]=rows
    sets=[{(r['device'],r['storage_base_ptr']) for r in rows if r['storage_nbytes']>0} for rows in inventory.values()]
    return inventory,all(not (sets[a]&sets[b]) for a in range(len(sets)) for b in range(a+1,len(sets)))


def auxiliary_initial_losses(components,outputs,x,target,scale,labels,torch):
    from . import boundary_impulse_component105_v1 as component
    from dex_supermimic.rl.dex_supermimic.state import difference
    diagnostics={};arrays={}
    with torch.no_grad():
        for arm in component.ARMS:
            o=outputs[arm];delta=difference(o['prediction58_FP32'],target)
            chart=c.chart_guard(o['prediction58_FP32'],target,delta,c.RECIPE['chart_margin_rad'],torch)
            point=(delta/scale).square().mean()
            loss=component.component_losses(components[arm],x,o['prediction58_FP32'],o['z12_FP32'],o['mode_logits2_FP32'],point,labels,
                lambda_mode=1.,lambda_imp=1.,lambda_cons=0. if arm==component.ARMS[0] else 1.)
            diagnostics[arm]={'native_point_FP32':float(point),'mode_BCE_FP32':float(loss['L_mode_FP32']),'impulse_z_MSE_F64':float(loss['L_imp_F64']),
                'saved_defect_consistency_MSE_F64':float(loss['L_cons_F64']),'public_loss':float(loss['L_public']),'diagnostic_total':float(loss['L_total']),
                'chart':chart,'coefficients_descriptive_not_training_recipe':{'mode':1.,'impulse':1.,'consistency':0. if arm==component.ARMS[0] else 1.}}
            arrays[arm]={'native_point_delta52_FP32':delta,**{k:v for k,v in loss.items() if isinstance(v,torch.Tensor)}}
    return diagnostics,arrays


def run_analysis(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,value in m['sources'].items():z.write(checked(value),name)
    counts={'trusted_initial_torch_load_attempted':0,'trusted_initial_torch_load_returned':0,'native_WM_constructors':0,'raw_deepcopies':0,'WM_forward_attempted':0,'WM_forward_returned':0,'predicted_rows':0,'AD_optimizer_train_checkpoint_old119f_Actor_MJB_Env_physics_GPU':0}
    phase='authorized_saved_arrays';components={}
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        torch.set_num_threads(4);before_rng=c._rng(torch,np)
        from . import component_H8_donor105_v1 as h8
        from . import boundary_impulse_component105_v1 as component
        h8.cf.save_raw_rng(out,'before',before_rng,np)
        with np.load(checked(i['target_arrays']),allow_pickle=False) as f:data=prepare_arrays({k:f[k].copy() for k in f.files},np)
        np.savez_compressed(out/'prepared_inputs_and_context.npz',**data)
        phase='one_trusted_initial_payload'
        counts['trusted_initial_torch_load_attempted']+=1;initial=torch.load(checked(i['initial_component']),map_location='cpu',weights_only=False);counts['trusted_initial_torch_load_returned']+=1
        require(set(initial)=={'schema','parent_checkpoint105','WM','frozen_s52','physical_xi46','normalizer_host_configuration','RNG'},'Exact own unmodified initial namespace, never fake final1000/native105')
        scale=initial['frozen_s52'];xi=initial['physical_xi46'];component.tensor_check(scale,(52,),torch.float32,'initial frozen s52');component.tensor_check(xi,(46,),torch.float32,'initial physical xi46')
        require(bool((scale>0).all()) and component.bit_equal(xi,torch.from_numpy(data['physical_xi46_FP32'])),'Actual target/initial xi46 bytes')
        initial_before=c._tensor_digest(initial['WM'],torch);extra_before=c._tensor_digest({'xi':xi,'s52':scale},torch)
        x=torch.from_numpy(data['x58_FP32']);u=torch.from_numpy(data['u20_FP32']);target=torch.from_numpy(data['target58_FP32'])
        dynamics=torch.cat((x,xi[None].expand(152,-1),torch.zeros((152,174),dtype=torch.float32)),dim=-1)
        np.savez_compressed(out/'actual_forward_inputs152.npz',x58_FP32=x.numpy().copy(),actual_u20_FP32=u.numpy().copy(),actual_dynamics278_FP32=dynamics.numpy().copy())
        ctx=component.fixed_context_from_rows(torch.from_numpy(data['mass_actual_F64'].astype(np.float32)),torch.from_numpy(data['I_body_F64']),torch.from_numpy(data['gravity_palm_F64']),
            torch.from_numpy(data['DeltaT_F64']),torch.from_numpy(data['palm_rot_raw_FP32']),scale,xi)
        np.savez_compressed(out/'initial_frozen_context_buffers.npz',**{k:v.numpy().copy() for k,v in ctx.named_buffers()})
        phase='one_native_initial_constructor'
        raw=h8.make_initial_eval_component(initial,read(checked(i['initial_validation'])),torch);counts['native_WM_constructors']+=1
        np.savez_compressed(out/'initial_normalizer_buffers.npz',**{k:v.numpy().copy() for k,v in c._norm_state(raw).items()})
        components,construction=component.make_matched_components(raw,ctx);counts['raw_deepcopies']=2
        models={'unwrapped_baseline':raw,**components}
        storage_inventory,storage_disjoint=named_storage_inventory(models);c._write_json(out/'named_model_context_storage_inventory.json',storage_inventory)
        require(storage_disjoint,'Fresh raw copies must have disjoint underlying named storage bases')
        state_before={name:component.state_digest(model) for name,model in models.items()}
        host_before={name:c._norm_config(raw if name=='unwrapped_baseline' else model.wm) for name,model in models.items()}
        input_before=c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch)
        phase='exact_three_B152_inference_calls';outputs=triplet_forward(raw,components,x,u,dynamics,torch,counts);masks=parity_masks(outputs,torch)
        for name,values in outputs.items():np.savez_compressed(out/(name+'_forward152.npz'),**{k:v.numpy().copy() for k,v in values.items()})
        np.savez_compressed(out/'all152_parity_masks.npz',**{arm+'__'+k:v.numpy().copy() for arm,rows in masks.items() for k,v in rows.items()})
        require(all(bool(mask.all()) for rows in masks.values() for mask in rows.values()),'Real same-function/head-zero bit parity failed; no source bypass or retry')
        require(all(module.counts=={'forward_attempted':1,'native_forward_returned':1,'hook_calls':1,'head_forward_returned':1,'forward_returned':1} and module._cache is None and not module._active for module in components.values()),'Exactly one hook/native/head per wrapper, cache empty')
        loss_summary={}
        if p['evaluate_auxiliary_losses']:
            phase='optional_supervised_arithmetic_not_training'
            labels={k:torch.from_numpy(data[k]) for k in ('J_palm_F64','K_palm_F64','boundary_constraint_witness','r_v_saved_m_s_F64','r_w_saved_rad_s_F64')}
            loss_summary,loss_arrays=auxiliary_initial_losses(components,outputs,x,target,scale,labels,torch)
            for name,values in loss_arrays.items():np.savez_compressed(out/(name+'_descriptive_losses.npz'),**{k:v.numpy().copy() for k,v in values.items()})
        state_after={name:component.state_digest(model) for name,model in models.items()}
        host_after={name:c._norm_config(raw if name=='unwrapped_baseline' else model.wm) for name,model in models.items()}
        for model in components.values():model.assert_frozen()
        after_rng=c._rng(torch,np);h8.cf.save_raw_rng(out,'after',after_rng,np);rng_equality=raw_rng_equality(before_rng,after_rng,np,torch)
        input_after=c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch)
        invariants={'all152_native58_self15_bit_parity':True,'all14_aux_exact_zero':True,'all_inputs_unchanged':input_before==input_after,
            'all_model_head_norm_context_buffers_unchanged':state_before==state_after,'all_norm_host_configuration_unchanged':host_before==host_after,
            'loaded_initial_WM_s52_xi_unchanged':initial_before==c._tensor_digest(initial['WM'],torch) and extra_before==c._tensor_digest({'xi':xi,'s52':scale},torch),
            'raw_Python_NumPy_TorchCPU_RNG_unchanged':all(rng_equality.values()),'no_parameter_gradients':all(parameter.grad is None for model in models.values() for parameter in model.parameters()),
            'CUDA_uninitialized_and_no_Warp_MuJoCo':'warp' not in sys.modules and 'mujoco' not in sys.modules and not torch.cuda.is_initialized(),
            'disjoint_storage_and_caches':storage_disjoint and construction['disjoint_model_head_buffer_storage'] and construction['feature_hooks_independent']}
        require(all(invariants.values()),'Initial parity engineering invariant failed')
        for value in i.values():checked(value)
        for value in m['sources'].values():checked(value)
        expected_counts={'trusted_initial_torch_load_attempted':1,'trusted_initial_torch_load_returned':1,'native_WM_constructors':1,'raw_deepcopies':2,'WM_forward_attempted':3,'WM_forward_returned':3,'predicted_rows':456,'AD_optimizer_train_checkpoint_old119f_Actor_MJB_Env_physics_GPU':0}
        require(same(pin(protocol_path),m['protocol_pin']) and counts==expected_counts,'Unchanged protocol/exact full call budget')
        report={'schema':'full151105_boundary_component_initial_parity_CPU_analysis:1','status':'COMPLETED_REAL_INITIAL_FUNCTION_PARITY_NOT_TRAINING_QUALIFICATION',
            'protocol':m['protocol_pin'],'input_pins':i,'counts':counts,'invariants':invariants,'construction_receipt':construction,'state_digests_before':state_before,'state_digests_after':state_after,
            'checks':invariants,'passed':len(invariants),'total':len(invariants),'engineering_qualified':True,'source_inventory':pin(out/'source_inventory.json'),
            'initial_component':i['initial_component'],'initial_WM_sha256':initial_before,'native105_or_training_health_qualified':False,
            'normalizer_host_configuration_before':host_before,'normalizer_host_configuration_after':host_after,'wrapper_counts':{name:module.counts for name,module in components.items()},
            'input_digest_before':input_before,'input_digest_after':input_after,'named_storage_scope':'named parameters/buffers including physical context; no general opaque or hidden-state certificate',
            'descriptive_initial_losses':loss_summary,'raw_RNG_value_and_byte_equality':rng_equality,'function_identity_qualified':True,'actual_training_authorized':False,'new_loss_recipe_frozen':False,
            'scope':'One own unmodified A2 native WM and two independent raw copies; original152 inputs, same FP32/inference-mode. Zero heads are ungated. Optional losses use targets only outside forward and preserve saved nonzero defects; no optimizer or training benefit claim.',
            'physical_force_or_gradient_qualified':False,'whole_hidden_state_qualified':False,'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        c._write_json(out/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'no_retry_or_module_patch':True});raise
    finally:
        for model in components.values():model.close()
