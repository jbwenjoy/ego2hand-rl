"""Exactly3 synthetic trainer contracts; no actual data/native WM or training."""
import datetime,hashlib,importlib.util,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
TORCH_PREIMPORTED='torch' in sys.modules
if TORCH_PREIMPORTED:raise RuntimeError('Fresh masked toy process required before Torch import')
import torch
import numpy as np
CUDA_OBSERVATIONS=[{'stage':'after_Torch_import','CUDA_initialized':torch.cuda.is_initialized()}]

def cuda_point(stage):
    value=torch.cuda.is_initialized();CUDA_OBSERVATIONS.append({'stage':stage,'CUDA_initialized':value})
    if value:raise ValueError('Masked CPU toy unexpectedly initialized CUDA')

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import boundary_impulse_training105_v2 as core


def main():
    torch.set_num_threads(4);out=ROOT/'outputs/full151105_boundary_impulse_training_preparation_CPU_v2';core.require(not out.exists(),'Fresh synthetic trainer preparation');out.mkdir()
    fixture=ROOT/'scripts/check_full151_boundary_impulse_component_cpu_v1.py';core.require(core.pin(fixture)['sha256']=='365d624d8a4fcb8a191459cc5fabe31539027f08b0c4d0919735fc95ac3256a0','Frozen own Toy fixture')
    pins={k:core.pin(v) for k,v in {**core.OWN,**core.DEPENDENCIES,'toy_fixture':fixture}.items()};pins['frozen_recipe']=core.pin(core.RECIPE)
    core.c._write_json(out/'source_freeze_before_tests.json',pins)
    for name,value in pins.items():(out/(name+('_source.json' if name=='frozen_recipe' else '_source.py'))).write_bytes(Path(value['path']).read_bytes())
    checks={};modules={}
    def eq(v):core.require(bool(v),'Toy training contract')
    try:
        cuda_point('before_Toy_constructor')
        spec=importlib.util.spec_from_file_location('_frozen_boundary_training_toy',fixture);toy=importlib.util.module_from_spec(spec);spec.loader.exec_module(toy)
        component=toy.core;raw=toy.ToyWM().eval();xi=torch.arange(46,dtype=torch.float32)/100;scale=torch.maximum(raw.delta_norm.std(),raw.error_floor);n=152
        ctx=component.fixed_context_from_rows(torch.tensor(2.,dtype=torch.float32),torch.diag(torch.tensor([2.,3.,4.],dtype=torch.float64)),torch.tensor([[0.,0.,-10.]],dtype=torch.float64).expand(n,-1).clone(),
            torch.full((n,),.02,dtype=torch.float64),torch.eye(3)[None].expand(n,-1,-1).clone(),scale,xi)
        x,u,dyn=toy.make_inputs(n,xi)
        with torch.no_grad():baseline=raw(x,u,dyn)
        raw_before=component.state_digest(raw);modules,receipt=component.make_matched_components(raw,ctx)
        labels={'J_palm_F64':torch.zeros((n,2,3),dtype=torch.float64),'K_palm_F64':torch.zeros((n,2,3),dtype=torch.float64),'boundary_constraint_witness':torch.zeros((n,2),dtype=torch.bool),
                'r_v_saved_m_s_F64':torch.full((n,3),.1,dtype=torch.float64),'r_w_saved_rad_s_F64':torch.full((n,3),.2,dtype=torch.float64)}
        names={};params={};optimizers={};offsets={};extra={};logs=[];snapshots={};rng_setup=core.c._rng(torch,np)
        cuda_point('before_optimizer_construction')
        for arm,module in modules.items():
            names[arm],params[arm]=core.ordered_parameters(module);optimizers[arm]=torch.optim.AdamW(params[arm],lr=1e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8,foreach=False)
            cuda_point('after_optimizer_constructor_'+arm)
            offsets[arm]=dict(module.counts);extra[arm]={'backward_calls':0,'optimizer_steps':0}
        initial_state={name:core.snapshot_state(module.state_dict(),torch) for name,module in modules.items()}
        eq(all(core.parity.raw_rng_equality(rng_setup,core.c._rng(torch,np),np,torch).values()))
        eq(all(component.bit_equal(initial_state[component.ARMS[0]][k],initial_state[component.ARMS[1]][k]) for k in initial_state[component.ARMS[0]]))
        for iteration in (1,2):
            rows={}
            for arm,module in modules.items():
                opt=optimizers[arm];opt.zero_grad(set_to_none=True);pred,self15,z,mode=module(x,u,dyn)
                if iteration==1:eq(component.bit_equal(pred.detach(),baseline[0]) and component.bit_equal(self15.detach(),baseline[1]) and bool((z==0).all()) and bool((mode==0).all()))
                # Own synthetic point scalar; native LogR is not run in fixture.
                point=pred[:,:40].square().mean()
                losses=component.component_losses(module,x,pred,z,mode,point,labels,lambda_mode=1.,lambda_imp=1.,lambda_cons=0. if arm==component.ARMS[0] else 1.)
                losses['L_total'].backward();extra[arm]['backward_calls']+=1;snapshots[arm]=core.clip_gradients(module,iteration,torch,True)
                cuda_point('before_optimizer_step_'+arm+'_'+str(iteration));opt.step();cuda_point('after_optimizer_step_'+arm+'_'+str(iteration));extra[arm]['optimizer_steps']+=1
                eq(core.optimizer_steps_exact(opt,params[arm],iteration,torch));module.assert_frozen()
                counts=core.observed_counts(module,extra[arm],offsets[arm]);eq(set(counts)==set(core.COUNTS) and all(v==iteration for v in counts.values()))
                rows[arm]={'counts':counts,'losses':{k:float(losses[v].detach()) for k,v in (('state','L_state_FP32'),('mode','L_mode_FP32'),('impulse','L_imp_F64'),('consistency','L_cons_F64'),('public','L_public'),('total','L_total'))},'update_seconds':.01}
            logs.append({'iteration':iteration,'arms':rows})
        eq(component.state_digest(raw)==raw_before and len(logs)==2 and modules[component.ARMS[0]].counts['forward_returned']==2)
        checks['toy_fullbatch_shared_initial_firstFW_and_exact_two_arm_step_counts']=True

        # Snapshot stage: grads belong to PRE2 / weights after1; params AFTER2.
        dummy=out/'synthetic_protocol.json';core.c._write_json(dummy,{'synthetic_fixture_only':True});common={'protocol':core.pin(dummy),'recipe':core.pin(core.RECIPE),'input_pins':{'initial_component':core.pin(dummy)},'source_pins':pins,'fit_input_digest':'synthetic-input','fit_label_digest':'synthetic-label'}
        logpath=out/'toy_updates.jsonl'
        with logpath.open('xb') as f:
            for row in logs:f.write((json.dumps(row,allow_nan=False)+'\n').encode())
            f.flush();os.fsync(f.fileno())
        rawbytes=logpath.read_bytes();prefix={'path':str(logpath),'records':2,'bytes':len(rawbytes),'sha256':hashlib.sha256(rawbytes).hexdigest()}
        a=component.ARMS[0];shared=modules[a]
        initial={'schema':core.INITIAL_SCHEMA,'pair_schema':core.PAIR_SCHEMA,**common,'source_parent':core.pin(dummy),
            'WM_state':{k[3:]:v for k,v in initial_state[a].items() if k.startswith('wm.')},'aux_head_state':{k[9:]:v for k,v in initial_state[a].items() if k.startswith('aux_head.')},
            'physical_context_state':{k[8:]:v for k,v in initial_state[a].items() if k.startswith('context.')},'normalizer_host_configuration':core.c._norm_config(shared.wm),
            'ordered_parameter_names':names[a],'RNG_this_run':core.snapshot_state(rng_setup,torch),'construction_receipt':receipt,'synthetic_fixture_only':True}
        initial_path=out/'toy_initial_pair_state.pt';core.c._atomic_torch_save(initial_path,initial,torch)
        initial_loaded=torch.load(initial_path,map_location='cpu',weights_only=False);cuda_point('after_Toy_initial_readback')
        eq(initial_loaded['schema']==core.INITIAL_SCHEMA and 'arm' not in initial_loaded and 'iteration' not in initial_loaded and all(bool((v==0).all()) for v in initial_loaded['aux_head_state'].values()))
        # Pure state serialization of a Tiny checkpoint is not actual16 health.
        payload=core.checkpoint_payload(shared,optimizers[a],core.observed_counts(shared,extra[a],offsets[a]),snapshots[a],2,common,core.c._rng(torch,np),prefix,torch)
        path=out/'toy_update_0002.pt';core.c._atomic_torch_save(path,payload,torch);loaded=torch.load(path,map_location='cpu',weights_only=False);cuda_point('after_Toy_checkpoint_readback')
        eq(loaded['schema']==core.ARTIFACT_SCHEMA and loaded['pair_schema']==core.PAIR_SCHEMA and loaded['checkpoint_tag']=='AFTER_OPTIMIZER_I' and loaded['iteration']==2)
        g=loaded['gradient_snapshot'];eq(g['tag']=='PRE_OPTIMIZER_I' and g['weights_after_updates']==1 and g['iteration']==2 and g['parameter_names']==names[a])
        eq(set(g['pre_clip'])==set(g['post_clip'])==set(names[a]) and loaded['log_prefix']==prefix and loaded['ordered_parameter_names']==names[a])
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g['pre_clip'][k],2) for k in names[a]]),2);coef=torch.clamp(1./(norm+1e-6),max=1.)
        eq(component.bit_equal(norm,g['pre_clip_returned_norm_FP32']) and all(component.bit_equal(g['post_clip'][k],g['pre_clip'][k]*coef) for k in names[a]))
        eq(all(float(st['step'])==2 for st in loaded['optimizer_state']['state'].values()) and all(core.parity.raw_rng_equality(initial_loaded['RNG_this_run'],loaded['RNG'],np,torch).values()))
        with logpath.open('ab') as f:f.write(b'{"later_append_not_in_prefix":true}\n')
        eq(hashlib.sha256(logpath.read_bytes()[:prefix['bytes']]).hexdigest()==prefix['sha256'])
        checks['toy_checkpoint_PRE_gradient_AFTER_state_rawAdam_clip_and_prefix_IO_ABI']=True

        now=datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc);eta=core.estimate_eta_100([.1]*100,2.,15.,now)
        eq(eta['remaining_pairs']==900 and abs(eta['estimated_remaining_seconds']-112.)<1e-10 and abs(eta['estimated_total_elapsed_seconds']-127.)<1e-10)
        eq(eta['planning_remaining_range_seconds'][0]<112<eta['planning_remaining_range_seconds'][1] and eta['explicit_unmeasured_finalization_overhead_seconds']==20.)
        for module in modules.values():module.assert_frozen();module.close()
        eq(not torch.cuda.is_initialized() and 'mujoco' not in sys.modules and 'warp' not in sys.modules and 'dex_supermimic.rl.dex_supermimic.networks' not in sys.modules)
        checks['paired100_ETA_counts900pairs_setup_and_explicit_unmeasured_overhead']=True
        cuda_point('final_proof');eq(not TORCH_PREIMPORTED and all(v['CUDA_initialized'] is False for v in CUDA_OBSERVATIONS))
        for value in pins.values():core.checked(value)
        report={'schema':'full151105_boundary_impulse_training_CPU_contract:1','status':'PASS_THREE_TOY_TRAINING_PAYLOAD_AND_ETA_CONTRACTS','checks':checks,'passed':len(checks),'total':3,'source_pins':pins,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'Torch_preimported':TORCH_PREIMPORTED,'CUDA_state_observations':CUDA_OBSERVATIONS,
            'synthetic_scope':{'Toy_WM_constructors':1,'Toy_training_pairs':2,'Toy_training_FW_backward_steps':4,'Toy_validation_FW':1,'own_Toy_serialized_loads':2,'actual16_parameter_qualification':False},
            'actual_dataset_initial_reads_native_WM_constructor_FW_AD_optimizer_MJB_Env_physics_GPU':0}
        core.c._write_json(out/'report.json',report)
        ready={'schema':'full151105_boundary_impulse_training_CPU_READY:1','status':report['status'],'source_pins':pins,'cpu_report':core.pin(out/'report.json'),'cpu_checks':{'passed':len(checks),'total':3},'recipe':core.pin(core.RECIPE),
            'api':{'entry':str(core.OWN['CLI']),'module':str(core.OWN['consumer']),'metadata':'metadata_preflight(protocol_path,output=None)','run':'run_training(protocol_path,output,emit=None)'},'actual_training_authorized':False}
        core.c._write_json(out/'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':core.pin(out/'READY.json')}))
    except BaseException as error:
        core.c._write_json(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_data_initial_native_models_MJB_GPU':0,'CUDA_state_observations':CUDA_OBSERVATIONS});raise
    finally:
        for module in modules.values():module.close()


if __name__=='__main__':main()
