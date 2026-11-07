"""Three toy final-loader/forward/capture checks; no real models or data."""
import copy,importlib.util,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import boundary_component_final_evaluation105_v2 as core


def main():
    torch.set_num_threads(4);out=ROOT/'outputs/full151105_boundary_component_final_evaluation_preparation_CPU_v2';core.require(not out.exists(),'Fresh toy preparation');out.mkdir()
    fixture=ROOT/'scripts/check_full151_boundary_impulse_component_cpu_v1.py';core.require(core.pin(fixture)['sha256']=='365d624d8a4fcb8a191459cc5fabe31539027f08b0c4d0919735fc95ac3256a0','Existing Toy fixture source')
    actual_protocol_path=ROOT/'protocols/full151105_boundary_impulse_matched_training_CPU_run_v1.json'
    pins={k:core.pin(v) for k,v in {**core.OWN,**core.DEPENDENCIES,'toy_fixture':fixture,'scientific_contract':core.CONTRACT,'actual_training_protocol_metadata':actual_protocol_path}.items()};core.c._write_json(out/'source_freeze_before_tests.json',pins)
    for name,value in pins.items():(out/(name+('_source.md' if name=='scientific_contract' else '_source.json' if name=='actual_training_protocol_metadata' else '_source.py'))).write_bytes(Path(value['path']).read_bytes())
    checks={};modules={}
    def eq(v):core.require(bool(v),'Toy final-eval contract')
    def reject(fn):
        try:fn()
        except (ValueError,KeyError):return
        raise AssertionError('Expected strict rejection')
    try:
        spec=importlib.util.spec_from_file_location('_frozen_final_eval_toy',fixture);toy=importlib.util.module_from_spec(spec);spec.loader.exec_module(toy)
        component=toy.core;raw=toy.ToyWM().eval().requires_grad_(False);xi=torch.arange(46,dtype=torch.float32)/100;scale=torch.maximum(raw.delta_norm.std(),raw.error_floor)
        ctx=component.fixed_context_from_rows(torch.tensor(2.,dtype=torch.float32),torch.eye(3,dtype=torch.float64),torch.zeros((152,3),dtype=torch.float64),torch.full((152,),.02,dtype=torch.float64),torch.eye(3)[None].expand(152,-1,-1).clone(),scale,xi)
        modules,receipt=component.make_matched_components(raw,ctx)
        actual_protocol=core.read(actual_protocol_path)
        eq(actual_protocol['schema']=='full151105_boundary_impulse_matched_training_CPU_protocol:1' and actual_protocol['role']=='matched_public_aux_vs_cross_target_consistency')
        eq('inputs' in actual_protocol and 'source_pins' in actual_protocol and 'input_pins' not in actual_protocol)
        dummy=out/'synthetic_training_protocol.json';core.c._write_json(dummy,{'synthetic_only':True});trainpin=core.pin(dummy)
        training={key:None for key in actual_protocol}
        training.update(schema=actual_protocol['schema'],role=actual_protocol['role'],recipe=trainpin,inputs={'toy':trainpin},source_pins={'toy':trainpin})
        eq(set(training)==set(actual_protocol) and 'input_pins' not in training)
        initial={'normalizer_host_configuration':core.c._norm_config(raw)};payloads={}
        for index,arm in enumerate(core.ARMS):
            module=modules[arm];weights={k:v.detach().clone() for k,v in module.wm.state_dict().items()};weights['net.2.bias'][index]+=.1
            payloads[arm]={'schema':component.ARTIFACT_SCHEMA,'pair_schema':'full151105_boundary_impulse_matched_pair:1','arm':arm,'iteration':1000,'checkpoint_tag':'AFTER_OPTIMIZER_I',
                'protocol':trainpin,'recipe':training['recipe'],'input_pins':training['inputs'],'source_pins':training['source_pins'],'WM_state':weights,'aux_head_state':{'weight':torch.full((14,512),.001),'bias':torch.full((14,),.05)},
                'physical_context_state':{k:v.clone() for k,v in module.context.state_dict().items()},'normalizer_host_configuration':core.c._norm_config(module.wm)}
        bad=copy.deepcopy(payloads[core.ARMS[0]]);bad['physical_context_state']['DeltaT_F64']+=.001
        before=component.state_digest(modules[core.ARMS[0]]);reject(lambda:core.load_final_into(modules[core.ARMS[0]],bad,initial,trainpin,training,torch));eq(component.state_digest(modules[core.ARMS[0]])==before)
        bad=copy.deepcopy(payloads[core.ARMS[0]]);bad['WM_state']['norm.mean'][0]+=1.;reject(lambda:core.load_final_into(modules[core.ARMS[0]],bad,initial,trainpin,training,torch))
        for arm in core.ARMS:core.load_final_into(modules[arm],payloads[arm],initial,trainpin,training,torch);modules[arm].assert_frozen()
        checks['strict_own_final_state_head_context_norm_no_repair_no_wrapped_copy']=True

        x,u,dynamics=toy.make_inputs(152,xi);counts={'native_FW_attempted':0,'native_FW_returned':0,'aux_head_FW':0,'predicted_rows':0}
        states={name:component.state_digest(m) for name,m in {'raw':raw,**modules}.items()};outputs=core.forward_three(raw,modules,x,u,dynamics,torch,counts)
        eq(counts=={'native_FW_attempted':3,'native_FW_returned':3,'aux_head_FW':2,'predicted_rows':456})
        eq(set(outputs)==set(core.ROLES) and torch.equal(outputs['unwrapped_baseline']['z12_FP32'].view(torch.int32),torch.zeros((152,2,6),dtype=torch.int32)) and torch.equal(outputs['unwrapped_baseline']['mode_logits2_FP32'].view(torch.int32),torch.zeros((152,2),dtype=torch.int32)))
        eq(any(not torch.equal(outputs[arm]['prediction58_FP32'],outputs['unwrapped_baseline']['prediction58_FP32']) for arm in core.ARMS))
        eq(all(not value.requires_grad for record in outputs.values() for value in record.values()) and all(m.counts['hook_calls']==1 and m._cache is None for m in modules.values()))
        eq(states=={name:component.state_digest(m) for name,m in {'raw':raw,**modules}.items()})
        checks['three_B152_native_toy_calls_two_heads_baseline_explicit_algebraic_zero']=True

        delta=torch.full((152,52),2.,dtype=torch.float32);s=torch.full((52,),2.,dtype=torch.float32)
        captures=core.capture_point_probability_proxy(delta,s,torch.zeros((152,2)),torch.zeros((152,15)),torch.zeros((152,2),dtype=torch.bool),torch)
        eq(all(v.dtype==torch.float32 for v in captures.values()) and torch.equal(captures['native_point_squared_scaled52_FP32'],torch.ones((152,52))))
        eq(torch.equal(captures['mode_probability2_FP32'],torch.full((152,2),.5)) and torch.equal(captures['self15_probability15_FP32'],torch.full((152,15),.5)))
        eq(torch.equal(captures['self15_penalty_rate_FP32'],torch.full((152,),-7.5)) and torch.equal(captures['self15_penalty_component_FP32'],torch.full((152,),-.14999999105930328)))
        eq(torch.equal(captures['mode_BCE_elements_FP32'],torch.full((152,2),.6931471824645996)))
        eq(not torch.cuda.is_initialized() and 'mujoco' not in sys.modules and 'warp' not in sys.modules and 'dex_supermimic.rl.dex_supermimic.networks' not in sys.modules)
        checks['native_FP32_point_sigmoid_BCE_penalty_capture_distinct_from_F64_rescore']=True
        for value in pins.values():core.checked(value)
        report={'schema':'full151105_boundary_component_final_eval_CPU_contract:1','status':'PASS_THREE_TOY_FINAL_LOADING_FORWARD_AND_CAPTURE_CONTRACTS','checks':checks,'passed':len(checks),'total':3,'source_pins':pins,
            'synthetic_scope':{'Toy_WM_constructors':1,'Toy_B152_native_FW':3,'Toy_head_FW':2,'AD_optimizer':0,'actual16_model_qualification':False},'actual_training_protocol_metadata_ABI':{'pin':core.pin(actual_protocol_path),'schema':actual_protocol['schema'],'top_level_keys':sorted(actual_protocol),'checkpoint_input_field':'input_pins','protocol_input_field':'inputs'},'actual_pt_NPZ_reads_native_WM_ctor_FW_MJB_physics_GPU':0}
        core.c._write_json(out/'report.json',report)
        ready={'schema':'full151105_boundary_component_final_eval_CPU_READY:1','status':report['status'],'source_pins':pins,'cpu_report':core.pin(out/'report.json'),'cpu_checks':{'passed':len(checks),'total':3},
            'api':{'entry':str(core.OWN['CLI']),'module':str(core.OWN['consumer']),'metadata':'metadata_preflight(protocol_path,output=None)','run':'run_evaluation(protocol_path,output)'},'actual_evaluation_authorized':False}
        core.c._write_json(out/'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':core.pin(out/'READY.json')}))
    except BaseException as error:
        core.c._write_json(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_data_weights_models_MJB_GPU':0});raise
    finally:
        for module in modules.values():module.close()


if __name__=='__main__':main()
