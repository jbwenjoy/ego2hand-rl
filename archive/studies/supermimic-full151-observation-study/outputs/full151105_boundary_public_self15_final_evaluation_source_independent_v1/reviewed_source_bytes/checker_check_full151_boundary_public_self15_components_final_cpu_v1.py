"""Exactly3 Toy public15 final-loader/forward/capture cases; no real tensors."""
import copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
if 'torch' in sys.modules:raise RuntimeError('Fresh masked synthetic process required')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import boundary_public_self15_final_evaluation105_v1 as core


def main():
    torch.set_num_threads(4);out=ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_preparation_CPU_v1'
    core.require(not out.exists(),'Fresh once-only Toy preparation');out.mkdir()
    fixture=ROOT/'scripts/check_full151_boundary_impulse_component_cpu_v1.py'
    actual_protocol_path=ROOT/'protocols/full151105_boundary_public_self15_matched_training_CPU_run_v1.json'
    trainer_ready=ROOT/'outputs/full151105_boundary_public_self15_training_preparation_CPU_v1/READY.json'
    core.require(core.pin(fixture)['sha256']=='365d624d8a4fcb8a191459cc5fabe31539027f08b0c4d0919735fc95ac3256a0','Frozen Toy definitions only')
    core.require(core.pin(trainer_ready)['sha256']=='c2f852a7530e93db0a312e6fbd0cb241d21c4da94fbfbf3f4bf89cdd9e434e05','Frozen796c meaningful Toy proof')
    tr=core.read(trainer_ready);core.c.allchecks(core.read(core.checked(tr['cpu_report'])))
    pins={k:core.pin(v) for k,v in {**core.OWN,**core.DEPENDENCIES,'toy_fixture':fixture,'scientific_contract':core.CONTRACT,
        'actual_training_protocol_metadata':actual_protocol_path,'frozen_numerical_recipe':core.trainer.RECIPE,'trainer_READY':trainer_ready}.items()}
    pins['trainer_cpu_report']=tr['cpu_report']
    core.c._write_json(out/'source_freeze_before_tests.json',pins)
    archive=out/'executed_sources';archive.mkdir()
    for name,value in pins.items():(archive/(name+Path(value['path']).suffix)).write_bytes(Path(value['path']).read_bytes())
    checks={};modules={};cuda=[]
    def eq(value,message='Synthetic public15 final evaluation'):core.require(bool(value),message)
    def reject(fn):
        try:fn()
        except (ValueError,KeyError,TypeError):return
        raise AssertionError('Expected strict new-namespace rejection')
    def cuda_point(stage):
        value=torch.cuda.is_initialized();cuda.append({'stage':stage,'CUDA_initialized':value});eq(not value)
    try:
        cuda_point('after_import_before_Toy')
        spec=importlib.util.spec_from_file_location('_frozen_public15_final_toy',fixture);toy=importlib.util.module_from_spec(spec);spec.loader.exec_module(toy)
        component=toy.core;raw=toy.ToyWM().eval().requires_grad_(False);xi=torch.arange(46,dtype=torch.float32)/100
        scale=torch.maximum(raw.delta_norm.std(),raw.error_floor)
        ctx=component.fixed_context_from_rows(torch.tensor(2.,dtype=torch.float32),torch.eye(3,dtype=torch.float64),torch.zeros((152,3),dtype=torch.float64),
            torch.full((152,),.02,dtype=torch.float64),torch.eye(3)[None].expand(152,-1,-1).clone(),scale,xi)
        modules,construction=component.make_matched_components(raw,ctx)
        actual=core.read(actual_protocol_path)
        eq(actual['schema']==core.trainer.SCHEMA and actual['role']==core.trainer.ROLE and set(actual['inputs'])==core.trainer.INPUTS and 'input_pins' not in actual)
        dummy=out/'synthetic_training_protocol.json';core.c._write_json(dummy,{'synthetic_only':True});trainpin=core.pin(dummy)
        training={key:None for key in actual}
        training.update(schema=actual['schema'],role=actual['role'],recipe=trainpin,inputs={k:trainpin for k in actual['inputs']},source_pins={'toy':trainpin})
        eq(set(training)==set(actual) and 'input_pins' not in training)
        initial={'normalizer_host_configuration':core.c._norm_config(raw)}
        fit={name:hashlib.sha256(('synthetic-'+name).encode()).hexdigest() for name in core.FIT_DIGEST_KEYS};payloads={}
        for index,arm in enumerate(core.ARMS):
            module=modules[arm];weights={k:v.detach().clone() for k,v in module.wm.state_dict().items()};weights['net.2.bias'][index]+=.1
            payloads[arm]={'schema':core.trainer.ARTIFACT_SCHEMA,'pair_schema':core.trainer.PAIR_SCHEMA,'arm':arm,'iteration':1000,'checkpoint_tag':'AFTER_OPTIMIZER_I',
                'protocol':trainpin,'recipe':training['recipe'],'input_pins':training['inputs'],'source_pins':training['source_pins'],**fit,
                'WM_state':weights,'aux_head_state':{'weight':torch.full((14,512),.001),'bias':torch.full((14,),.05)},
                'physical_context_state':{k:v.clone() for k,v in module.context.state_dict().items()},'normalizer_host_configuration':core.c._norm_config(module.wm)}
        control=modules[core.ARMS[0]];before=component.state_digest(control)
        bad=copy.deepcopy(payloads[core.ARMS[0]]);bad['schema']=component.ARTIFACT_SCHEMA;bad['pair_schema']='full151105_boundary_impulse_matched_pair:1'
        reject(lambda:core.load_final_into(control,bad,initial,trainpin,training,fit,torch))
        bad=copy.deepcopy(payloads[core.ARMS[0]]);bad['physical_context_state']['DeltaT_F64']+=.001
        reject(lambda:core.load_final_into(control,bad,initial,trainpin,training,fit,torch));eq(component.state_digest(control)==before)
        bad=copy.deepcopy(payloads[core.ARMS[0]]);bad['WM_state']['norm.mean'][0]+=1.
        reject(lambda:core.load_final_into(control,bad,initial,trainpin,training,fit,torch))
        bad=copy.deepcopy(payloads[core.ARMS[0]]);bad['fit_self15_label_digest']='0'*64
        reject(lambda:core.load_final_into(control,bad,initial,trainpin,training,fit,torch))
        bad_train=dict(training);bad_train['input_pins']=bad_train.pop('inputs')
        reject(lambda:core.load_final_into(control,payloads[core.ARMS[0]],initial,trainpin,bad_train,fit,torch))
        for arm in core.ARMS:core.load_final_into(modules[arm],payloads[arm],initial,trainpin,training,fit,torch)
        checks['new_family_only_real_protocol_inputs_map_strict_norm_context_and_label_digests']=True

        x,u,dynamics=toy.make_inputs(152,xi)
        counts={'native_FW_attempted':0,'native_FW_returned':0,'aux_head_FW':0,'predicted_rows':0}
        models={'unwrapped_baseline':raw,**modules};before={k:component.state_digest(v) for k,v in models.items()}
        outputs=core.forward_three(raw,modules,x,u,dynamics,torch,counts)
        eq(counts=={'native_FW_attempted':3,'native_FW_returned':3,'aux_head_FW':2,'predicted_rows':456})
        eq(set(outputs)==set(core.ROLES) and all(not v.requires_grad for row in outputs.values() for v in row.values()))
        eq(torch.equal(outputs['unwrapped_baseline']['z12_FP32'].view(torch.int32),torch.zeros((152,2,6),dtype=torch.int32)) and
            torch.equal(outputs['unwrapped_baseline']['mode_logits2_FP32'].view(torch.int32),torch.zeros((152,2),dtype=torch.int32)))
        eq(all(m.counts['hook_calls']==m.counts['forward_returned']==1 and m._cache is None and not m._active for m in modules.values()))
        eq(before=={k:component.state_digest(v) for k,v in models.items()});cuda_point('after_three_Toy_FW')
        checks['three_B152_Toy_forward_two_heads_positive_zero_baseline_no_AD_or_state_change']=True

        target15=((torch.arange(152)[:,None]+torch.arange(15)[None])%3==0).float()
        labels={'J_palm_F64':torch.zeros((152,2,3),dtype=torch.float64),'K_palm_F64':torch.zeros((152,2,3),dtype=torch.float64),
            'boundary_constraint_witness':torch.zeros((152,2),dtype=torch.bool),'r_v_saved_m_s_F64':torch.full((152,3),.1,dtype=torch.float64),
            'r_w_saved_rad_s_F64':torch.full((152,3),.2,dtype=torch.float64)}
        recipe=core.trainer.fixed_recipe();delta=2*scale[None].expand(152,-1)
        with torch.no_grad():
            for role,row in outputs.items():
                module=modules[core.ARMS[0] if role==core.ROLES[0] else role]
                loss,ops=core.trainer.public_self15_losses(module,x,row['prediction58_FP32'],row['self15_logits_FP32'],row['z12_FP32'],row['mode_logits2_FP32'],delta,scale,target15,labels,recipe,torch)
                captures=core.capture_point_probability_proxy(ops,row['mode_logits2_FP32'],row['self15_logits_FP32'],torch)
                eq(set(core.trainer.LOSS_FIELDS)=={'state','self15','native_pair','mode','impulse','consistency','public','total'})
                eq(component.bit_equal(loss['L_state_FP32'],torch.tensor(4.)) and component.bit_equal(ops['point_per_row_FP32'],torch.full((152,),4.)))
                eq(component.bit_equal(ops['self15_BCE_elements_FP32'].sum(-1),ops['self15_sum_per_row_FP32']) and component.bit_equal(ops['mode_BCE_elements_FP32'].mean(),loss['L_mode_FP32']))
                eq(component.bit_equal((ops['point_per_row_FP32']+torch.tensor(.1)*ops['self15_sum_per_row_FP32']).mean(),loss['L_native_pair_FP32']))
                eq(component.bit_equal((loss['L_native_pair_FP32']+loss['L_mode_FP32'])+loss['L_imp_F64'],loss['L_public']) and bool(loss['L_native_pair_FP32']>loss['L_state_FP32']))
                eq(component.bit_equal(captures['native_point_delta52_FP32'],ops['native_D52_FP32']) and component.bit_equal(captures['native_point_squared_scaled52_FP32'],ops['point_squared52_FP32']))
                ell=row['self15_logits_FP32'].numpy().astype(np.float64);y=target15.numpy().astype(np.float64)
                analytic=np.maximum(ell,0)-ell*y+np.log1p(np.exp(-np.abs(ell)))
                eq(np.max(np.abs(analytic-ops['self15_BCE_elements_FP32'].numpy()))<2e-6)
                eq(all(v.dtype==torch.float32 for v in captures.values()) and all(not v.requires_grad for v in loss.values() if isinstance(v,torch.Tensor)))
        zero_capture=core.capture_point_probability_proxy({'native_D52_FP32':delta,'point_squared52_FP32':torch.full((152,52),4.)},torch.zeros((152,2)),torch.zeros((152,15)),torch)
        eq(torch.equal(zero_capture['mode_probability2_FP32'],torch.full((152,2),.5)) and torch.equal(zero_capture['self15_probability15_FP32'],torch.full((152,15),.5)))
        eq(torch.equal(zero_capture['self15_penalty_rate_FP32'],torch.full((152,),-7.5)) and torch.equal(zero_capture['self15_penalty_component_FP32'],torch.full((152,),-.14999999105930328)))
        eq(counts['native_FW_returned']==3 and before=={k:component.state_digest(v) for k,v in models.items()})
        cuda_point('after_pure_loss_and_capture')
        eq('mujoco' not in sys.modules and 'warp' not in sys.modules and 'dex_supermimic.rl.dex_supermimic.networks' not in sys.modules)
        checks['frozen796c_eight_scalars_row_native_pair_aliases_and_probability_proxy_no_extra_FW']=True
        for value in pins.values():core.checked(value)
        report={'schema':'full151105_boundary_public_self15_final_eval_CPU_contract:1','status':'PASS_THREE_TOY_PUBLIC15_FINAL_LOAD_FORWARD_AND_CAPTURE_CONTRACTS',
            'checks':checks,'passed':len(checks),'total':3,'source_pins':pins,'CUDA_state_observations':cuda,
            'actual_training_protocol_metadata_ABI':{'pin':core.pin(actual_protocol_path),'schema':actual['schema'],'top_level_keys':sorted(actual),'checkpoint_input_field':'input_pins','protocol_input_field':'inputs','input_aliases':sorted(actual['inputs'])},
            'synthetic_scope':{'Toy_WM_constructors':1,'Toy_B152_native_FW':3,'Toy_aux_head_FW':2,'AD_optimizer':0,'actual16_model_qualification':False},
            'actual_pt_NPZ_label_values_reads_native_WM_ctor_FW_MJB_physics_GPU':0}
        core.c._write_json(out/'report.json',report)
        ready={'schema':'full151105_boundary_public_self15_final_eval_CPU_READY:1','status':report['status'],'source_pins':pins,'cpu_report':core.pin(out/'report.json'),
            'cpu_checks':{'passed':len(checks),'total':3},'api':{'entry':str(core.OWN['CLI']),'module':str(core.OWN['consumer']),'metadata':'metadata_preflight(protocol_path,output=None)','run':'run_evaluation(protocol_path,output)'},'actual_evaluation_authorized':False}
        core.c._write_json(out/'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':core.pin(out/'READY.json')}))
    except BaseException as error:
        core.c._write_json(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_data_weights_native_models_FW_MJB_GPU':0,'CUDA_state_observations':cuda});raise
    finally:
        for module in modules.values():module.close()


if __name__=='__main__':main()
