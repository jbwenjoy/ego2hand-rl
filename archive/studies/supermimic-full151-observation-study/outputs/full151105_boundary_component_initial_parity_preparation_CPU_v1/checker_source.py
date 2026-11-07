"""Exactly3 focused toy parity contracts; no real weights/data/native ctor."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import boundary_component_initial_parity105_v1 as core


def main():
    torch.set_num_threads(4);out=ROOT/'outputs/full151105_boundary_component_initial_parity_preparation_CPU_v1';core.require(not out.exists(),'Fresh toy proof');out.mkdir()
    toy_path=ROOT/'scripts/check_full151_boundary_impulse_component_cpu_v1.py'
    core.require(core.pin(toy_path)['sha256']=='365d624d8a4fcb8a191459cc5fabe31539027f08b0c4d0919735fc95ac3256a0','Frozen own Toy fixture source')
    pins={k:core.pin(v) for k,v in {**core.OWN,**core.DEPENDENCIES,'toy_fixture_source':toy_path}.items()};core.c._write_json(out/'source_freeze_before_tests.json',pins)
    for name,value in pins.items():(out/(name+'_source.py')).write_bytes(Path(value['path']).read_bytes())
    checks={};components={}
    def eq(v):core.require(bool(v),'Synthetic parity contract')
    def reject(fn):
        try:fn()
        except (ValueError,KeyError,TypeError):return
        raise AssertionError('Expected source/header rejection')
    try:
        # Explicit synthetic152 context; this is not any existing dataset tensor.
        data={key:np.zeros(shape,np.float32 if key.endswith('FP32') else np.float64) for key,shape in {
            'x58_FP32':(152,58),'u20_FP32':(152,20),'target58_FP32':(152,58),'physical_xi46_FP32':(46,),'palm_rot_raw_FP32':(152,3,3),
            'gravity_palm_F64':(152,3),'DeltaT_F64':(152,),'mass_actual_F64':(),'I_body_F64':(3,3),'J_palm_F64':(152,2,3),'K_palm_F64':(152,2,3),
            'r_v_saved_m_s_F64':(152,3),'r_w_saved_rad_s_F64':(152,3)}.items()}
        data['boundary_constraint_witness']=np.zeros((152,2),bool);data['mass_actual_F64'][...]=2.;data['DeltaT_F64'].fill(.02);data['palm_rot_raw_FP32'][:]=np.eye(3);data['I_body_F64'][:]=np.diag([2.,3.,4.])
        prepared=core.prepare_arrays(data,np);eq(set(prepared)==set(data) and all(a.tobytes()==data[k].tobytes() for k,a in prepared.items()))
        bad=copy.deepcopy(data);bad['mass_actual_F64'][...]=.1;reject(lambda:core.prepare_arrays(bad,np))
        bad=copy.deepcopy(data);bad['physical_xi46_FP32']=np.zeros(45,np.float32);reject(lambda:core.prepare_arrays(bad,np))
        header={'schema':core.SCHEMA,'role':core.ROLE,'CPU_run_authorized':False,'GPU_run_authorized':False,'evaluate_auxiliary_losses':True,'accuracy_gate':None,'output_prefix':core.PREFIX,'inputs':{k:None for k in core.INPUTS},'source_pins':{'toy':'declared'}}
        eq(core.validate_protocol(header) is header);reject(lambda:core.validate_protocol(header,True));reject(lambda:core.validate_protocol({**header,'GPU_run_authorized':True}))
        checks['synthetic152_operand_shape_source_mass_and_explicit_execution_boundary']=True

        spec=importlib.util.spec_from_file_location('_frozen_boundary_toy_fixture',toy_path);toy=importlib.util.module_from_spec(spec);spec.loader.exec_module(toy)
        component=toy.core;raw=toy.ToyWM().eval();xi=torch.arange(46,dtype=torch.float32)/100;std=torch.maximum(raw.delta_norm.std(),raw.error_floor)
        ctx=component.fixed_context_from_rows(torch.tensor(2.,dtype=torch.float32),torch.diag(torch.tensor([2.,3.,4.],dtype=torch.float64)),torch.tensor([[0.,0.,-10.]],dtype=torch.float64).expand(152,-1).clone(),
            torch.full((152,),.02,dtype=torch.float64),torch.eye(3)[None].expand(152,-1,-1).clone(),std,xi)
        components,receipt=component.make_matched_components(raw,ctx);x,u,dynamics=toy.make_inputs(152,xi)
        initial=component.state_digest(raw);before=core.c._tensor_digest({'x':x,'u':u,'dynamics':dynamics},torch)
        counts={'WM_forward_attempted':0,'WM_forward_returned':0,'predicted_rows':0};outputs=core.triplet_forward(raw,components,x,u,dynamics,torch,counts);masks=core.parity_masks(outputs,torch)
        eq(all(bool(mask.all()) for rows in masks.values() for mask in rows.values()) and counts=={'WM_forward_attempted':3,'WM_forward_returned':3,'predicted_rows':456})
        eq(raw.calls==1 and all(module.wm.calls==1 and module.counts['hook_calls']==1 and module._cache is None for module in components.values()))
        inventory,disjoint=core.named_storage_inventory({'unwrapped_baseline':raw,**components});eq(disjoint and len(inventory)==3 and all(rows for rows in inventory.values()))
        eq(all(not value.requires_grad for output in outputs.values() for value in output.values()) and all(parameter.grad is None for module in (raw,*components.values()) for parameter in module.parameters()))
        bad={name:{k:v.clone() for k,v in output.items()} for name,output in outputs.items()};arm=component.ARMS[1]
        bad[arm]['prediction58_FP32'][0,0]=torch.nextafter(bad[arm]['prediction58_FP32'][0,0],torch.tensor(float('inf')))
        bad[arm]['mode_logits2_FP32'][149,1]=torch.tensor(-0.,dtype=torch.float32)
        failed=core.parity_masks(bad,torch)
        eq(not failed[arm]['prediction58_FP32_bit_equal'][0] and not failed[arm]['mode_logits2_FP32_exact_positive_zero'][149])
        eq(component.state_digest(raw)==initial and before==core.c._tensor_digest({'x':x,'u':u,'dynamics':dynamics},torch) and receipt['disjoint_model_head_buffer_storage'])
        checks['three_B152_toy_inference_samefunction_headzero_and_perrow_bit_detection']=True

        a=core.c._rng(torch,np);b=copy.deepcopy(a);eq(all(core.raw_rng_equality(a,b,np,torch).values()))
        b['torch_CPU'][0]^=1;eq(not core.raw_rng_equality(a,b,np,torch)['TorchCPU_raw_bytes_equal'])
        b=copy.deepcopy(a);state=list(b['numpy']);state[4]=-0. if np.asarray(a['numpy'][4],np.float64).tobytes()==np.asarray(0.,np.float64).tobytes() else 0.;b['numpy']=tuple(state)
        eq(not core.raw_rng_equality(a,b,np,torch)['NumPy_MT_header_equal'])
        eq(all(core.raw_rng_equality(a,core.c._rng(torch,np),np,torch).values()))
        eq(not torch.cuda.is_initialized() and 'mujoco' not in sys.modules and 'warp' not in sys.modules and 'dex_supermimic.rl.dex_supermimic.networks' not in sys.modules)
        checks['raw_RNG_values_bytes_headers_no_pickle_identity_or_restore']=True
        for module in components.values():module.close()
        for value in pins.values():core.checked(value)
        report={'schema':'full151105_boundary_component_initial_parity_CPU_contract:1','status':'PASS_THREE_TOY_PARITY_AND_RAW_RNG_CONTRACTS','checks':checks,'passed':len(checks),'total':3,'source_pins':pins,
            'synthetic_scope':{'Toy_WM_constructors':1,'Toy_B152_forwards':3,'Toy_predicted_rows':456,'AD_optimizer_steps':0},'actual_dataset_reads_initial_loads_native_ctor_FW_MJB_physics_GPU':0}
        core.c._write_json(out/'report.json',report)
        ready={'schema':'full151105_boundary_component_initial_parity_CPU_READY:1','status':report['status'],'source_pins':pins,'cpu_report':core.pin(out/'report.json'),'cpu_checks':{'passed':len(checks),'total':3},
            'preregistration':core.pin(core.PREREG),'api':{'entry':str(core.OWN['CLI']),'module':str(core.OWN['consumer']),'metadata':'metadata_preflight(protocol_path,output=None)','run':'run_analysis(protocol_path,output)'},'actual_execution_authorized':False}
        core.c._write_json(out/'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':core.pin(out/'READY.json')}))
    except BaseException as error:
        core.c._write_json(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_data_initial_native_MJB_GPU':0});raise
    finally:
        for module in components.values():module.close()


if __name__=='__main__':main()
