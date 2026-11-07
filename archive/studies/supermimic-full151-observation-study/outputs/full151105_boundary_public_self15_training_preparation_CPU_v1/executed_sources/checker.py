"""Exactly3 synthetic public15 trainer contracts; never actual labels/weights/WM."""
import copy,datetime,hashlib,importlib.util,json,math,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
TORCH_PREIMPORTED='torch' in sys.modules
if TORCH_PREIMPORTED:raise RuntimeError('Fresh masked Toy process required before Torch import')
import torch
import numpy as np
CUDA_OBSERVATIONS=[]


def cuda_point(stage):
    state=torch.cuda.is_initialized();CUDA_OBSERVATIONS.append({'stage':stage,'CUDA_initialized':state})
    if state:raise ValueError('CPU Toy must not initialize CUDA')


ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import boundary_public_self15_training105_v1 as core


def synthetic_json_labels():
    plan=[];joined=[]
    for bid in range(152):
        s,off=divmod(bid,19);pi,si=divmod(s,4);k=(0,49,99,149)[si];group=226*pi+150+19*si+off
        g={'row_id':bid,'snapshot_id':s,'path':('A0','A2')[pi],'mode':'baseline' if off<3 else 'direction','k':k,
           'branch_id':bid,'group_id':group,'physical_rows':[4*group+j for j in range(4)],'terminal_observation':k+1}
        found=[int((bid+j)%5==0) for j in range(25)]
        target=[float(found[j]>0) for j in (1,2,6,7,11,12,16,17,21,22)]+[float(found[d]>0 or found[t]>0) for d,t in zip((3,8,13,18,23),(4,9,14,19,24))]
        row={**g,'contact15':target,'raw_found25':found,'source_frame':k+200,'label_time':'lastsolver_preintegration~kdt+.015',
             'state_time':'postintegration_kdt+.020','valid_next_policy_decision':k<149,'no_u150':True}
        plan.append(g);joined.append(row)
    return plan,joined


def main():
    torch.set_num_threads(4);cuda_point('after_import_before_fixture')
    out=ROOT/'outputs/full151105_boundary_public_self15_training_preparation_CPU_v1'
    core.require(not out.exists(),'Fresh once-only public15 preparation');out.mkdir()
    fixture=ROOT/'scripts/check_full151_boundary_impulse_component_cpu_v1.py'
    real_protocol=ROOT/'protocols/full151105_boundary_impulse_matched_training_CPU_run_v1.json'
    core.require(core.pin(fixture)['sha256']=='365d624d8a4fcb8a191459cc5fabe31539027f08b0c4d0919735fc95ac3256a0','Frozen Toy definitions, not old checker execution')
    core.require(core.pin(real_protocol)['sha256']=='9f1a74a553f23d9ecd097a7e94a5381220860434a4701b995af936323fd0ccbe','Actual completed protocol metadata ABI, no fabricated input_pins field')
    pins={k:core.pin(v) for k,v in {**core.OWN,**core.DEPENDENCIES,'toy_fixture':fixture,'real_training_protocol_metadata':real_protocol,
        'frozen_recipe':core.RECIPE,'source_contract':core.SOURCE_CONTRACT}.items()}
    core.require(core.pin(core.HEALTH_READY)['sha256']==core.HEALTH_READY_SHA,'New namespace reader READY, never old5 health')
    health=core.read(core.HEALTH_READY);health_proof=core.read(core.checked(health['cpu_report']));core.c.allchecks(health_proof)
    pins.update(health_READY=core.pin(core.HEALTH_READY),health_cpu_report=health['cpu_report'])
    pins.update({'health_'+k:value for k,value in health['source_pins'].items()})
    core.c._write_json(out/'source_freeze_before_tests.json',pins)
    snapshots=out/'executed_sources';snapshots.mkdir()
    for name,value in pins.items():(snapshots/(name+Path(value['path']).suffix)).write_bytes(Path(value['path']).read_bytes())
    checks={};modules={}
    def eq(value,message='Synthetic public15 contract'):core.require(bool(value),message)
    def reject(fn):
        try:fn()
        except (ValueError,KeyError,TypeError):return
        raise AssertionError('Expected fail-closed synthetic rejection')
    try:
        plan,joined=synthetic_json_labels();values=core.cached_label_rows(plan,joined)
        eq(len(values)==152 and len(values[-1])==15 and plan[-1]['physical_rows'][-1]==1807 and plan[-1]['terminal_observation']==150 and joined[-1]['valid_next_policy_decision'] is False)
        gids=np.asarray([g['group_id'] for g in plan],dtype=np.int64);subrows=np.asarray([g['physical_rows'] for g in plan],dtype=np.int64)
        core.validate_target_identity_arrays(plan,gids,subrows,np)
        wrong_gids=gids.copy();wrong_gids[76]+=1;reject(lambda:core.validate_target_identity_arrays(plan,wrong_gids,subrows,np))
        for modify in ('reverse','group','phase','binary','mapping'):
            wrong=copy.deepcopy(joined)
            if modify=='reverse':wrong.reverse()
            if modify=='group':wrong[76]['group_id']+=1
            if modify=='phase':wrong[-1]['state_time']='pretend_samephase_geometry'
            if modify=='binary':wrong[7]['contact15'][4]=.5
            if modify=='mapping':wrong[11]['contact15'][0]=1.-wrong[11]['contact15'][0]
            reject(lambda wrong=wrong:core.cached_label_rows(plan,wrong))
        old=core.read(real_protocol)
        eq(old['schema']=='full151105_boundary_impulse_matched_training_CPU_protocol:1' and 'inputs' in old and 'source_pins' in old and 'input_pins' not in old)
        dummy=out/'synthetic_pin.json';core.c._write_json(dummy,{'synthetic_only':True});dummy_pin=core.pin(dummy)
        # Preserve every real protocol top-level name, extend the new source
        # prerequisite fields, and retain inputs vs payload.input_pins mapping.
        protocol={k:None for k in old}
        protocol.update(schema=core.SCHEMA,role=core.ROLE,CPU_run_authorized=False,GPU_run_authorized=False,automatic_retry=False,
            background_continue_to_final=True,output_prefix=core.PREFIX,output=str(out/'never_execute'),
            inputs={k:dummy_pin for k in core.INPUTS},source_pins={'synthetic':dummy_pin},recipe=core.pin(core.RECIPE),
            source_contract=core.pin(core.SOURCE_CONTRACT),source_review=dummy_pin)
        eq(set(protocol)==set(old)|{'source_contract','source_review'} and 'input_pins' not in protocol)
        core.validate_protocol(protocol)
        wrong=dict(protocol);wrong['input_pins']=wrong.pop('inputs');reject(lambda:core.validate_protocol(wrong))
        wrong={**protocol,'inputs':{k:v for k,v in protocol['inputs'].items() if k!='target_identities'}};reject(lambda:core.validate_protocol(wrong))
        core.c._write_json(out/'synthetic_protocol_real_keyset.json',protocol)
        checks['original152_label_mapping_terminal_and_real_protocol_inputs_namespace']=True

        spec=importlib.util.spec_from_file_location('_frozen_public15_toy',fixture);toy=importlib.util.module_from_spec(spec);spec.loader.exec_module(toy)
        component=toy.core;raw=toy.ToyWM().eval();n=152;xi=torch.arange(46,dtype=torch.float32)/100
        scale=torch.maximum(raw.delta_norm.std(),raw.error_floor)
        ctx=component.fixed_context_from_rows(torch.tensor(2.,dtype=torch.float32),torch.diag(torch.tensor([2.,3.,4.],dtype=torch.float64)),
            torch.tensor([[0.,0.,-10.]],dtype=torch.float64).expand(n,-1).clone(),torch.full((n,),.02,dtype=torch.float64),
            torch.eye(3)[None].expand(n,-1,-1).clone(),scale,xi)
        x,u,dyn=toy.make_inputs(n,xi);u=u+torch.arange(n,dtype=torch.float32)[:,None]*1e-4
        with torch.no_grad():baseline=raw(x,u,dyn)
        raw_before=component.state_digest(raw);modules,construction=component.make_matched_components(raw,ctx)
        target15=torch.tensor(values,dtype=torch.float32)
        labels={'J_palm_F64':torch.zeros((n,2,3),dtype=torch.float64),'K_palm_F64':torch.zeros((n,2,3),dtype=torch.float64),
                'boundary_constraint_witness':torch.zeros((n,2),dtype=torch.bool),'r_v_saved_m_s_F64':torch.full((n,3),.1,dtype=torch.float64),
                'r_w_saved_rad_s_F64':torch.full((n,3),.2,dtype=torch.float64)}
        r=core.fixed_recipe();names={};params={};optimizers={};offsets={};extra={};logs=[];gradients={};loss_snaps={}
        initial_states={arm:core.snapshot_state(m.state_dict(),torch) for arm,m in modules.items()};rng=core.c._rng(torch,np)
        for arm,module in modules.items():
            names[arm],params[arm]=core.ordered_parameters(module);cuda_point('before_optimizer_'+arm)
            optimizers[arm]=torch.optim.AdamW(params[arm],lr=r['lr'],weight_decay=r['weight_decay'],betas=tuple(r['betas']),eps=r['optimizer_eps'],foreach=False)
            cuda_point('after_optimizer_'+arm);offsets[arm]=dict(module.counts);extra[arm]={'backward_calls':0,'optimizer_steps':0}
        eq(all(component.bit_equal(initial_states[component.ARMS[0]][k],initial_states[component.ARMS[1]][k]) for k in initial_states[component.ARMS[0]]))
        last15_grad_descriptors={}
        for iteration in (1,2):
            row={}
            for arm,module in modules.items():
                opt=optimizers[arm];opt.zero_grad(set_to_none=True);pred,logits,z,mode=module(x,u,dyn)
                if iteration==1:eq(component.bit_equal(pred,baseline[0]) and component.bit_equal(logits,baseline[1]) and bool((z==0).all()) and bool((mode==0).all()))
                # Synthetic difference only. No native integrate/Log or actual
                # target arrays are used by this proof.
                Rdiff=(pred[:,43:52]-x[:,43:52]).reshape(n,3,3).diagonal(dim1=-2,dim2=-1)
                delta=torch.cat((pred[:,:43]-x[:,:43],Rdiff,pred[:,52:58]-x[:,52:58]),-1)
                losses,ops=core.public_self15_losses(module,x,pred,logits,z,mode,delta,scale,target15,labels,r,torch)
                eq(losses['L_state_FP32'].dtype==losses['L_self15_sum_FP32'].dtype==losses['L_native_pair_FP32'].dtype==torch.float32 and losses['L_public'].dtype==torch.float64)
                if iteration==1:
                    gl,gfinal=torch.autograd.grad(losses['L_self15_sum_FP32'],(logits,module.wm.net[-1].weight),retain_graph=True)
                    # Independent closed-form derivative: sum15 then mean rows,
                    # so no division by15. Check original native final15 rows.
                    expected=(logits.detach().sigmoid()-target15)/n
                    eq(torch.allclose(gl,expected,rtol=1e-6,atol=1e-9) and bool((gfinal[:52]==0).all()) and bool(torch.isfinite(gfinal).all()) and bool(gfinal[52:].abs().sum()>0))
                    eq(not torch.allclose(gl,expected/15,rtol=1e-3,atol=1e-9))
                    last15_grad_descriptors[arm]={'synthetic_nonzero_last15_weight_gradient':True,'actual_health_requires_nonzero_last15':False}
                    zero=torch.zeros_like(logits);zero_bce=torch.nn.functional.binary_cross_entropy_with_logits(zero,target15,reduction='none')
                    eq(abs(float(zero_bce.sum(-1).mean())-15*math.log(2))<2e-6)
                expected_pair=(ops['point_per_row_FP32']+torch.tensor(.1)*ops['self15_sum_per_row_FP32']).mean()
                eq(component.bit_equal(expected_pair,losses['L_native_pair_FP32']) and component.bit_equal(ops['point_per_row_FP32'].mean(),losses['L_state_FP32']))
                loss_snaps[arm]=core.loss_snapshot_before_update(losses,ops,iteration,torch)
                losses['L_total'].backward();extra[arm]['backward_calls']+=1;gradients[arm]=core.clip_gradients(module,iteration,torch,True)
                cuda_point('before_step_'+arm+'_'+str(iteration));opt.step();cuda_point('after_step_'+arm+'_'+str(iteration));extra[arm]['optimizer_steps']+=1
                eq(core.optimizer_steps_exact(opt,params[arm],iteration,torch));module.assert_frozen()
                counts=core.observed_counts(module,extra[arm],offsets[arm]);eq(set(counts)==set(core.COUNTS) and all(v==iteration for v in counts.values()))
                row[arm]={'counts':counts,'losses':{k:float(losses[v].detach()) for k,v in core.LOSS_FIELDS.items()},'update_seconds':.01}
            logs.append({'iteration':iteration,'arms':row})
        eq(component.state_digest(raw)==raw_before and raw.calls==1 and all(m.counts['forward_returned']==2 for m in modules.values()))
        checks['two_arm_Toy_single_forward_native15_sum_gradient_shared_start_and_steps']=True

        logpath=out/'toy_updates.jsonl'
        with logpath.open('xb') as f:
            for row in logs:f.write((json.dumps(row,allow_nan=False)+'\n').encode())
            f.flush();os.fsync(f.fileno())
        b=logpath.read_bytes();prefix={'path':str(logpath),'records':2,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        dig15=core.c._tensor_digest({'cached_self15_target_FP32':target15},torch);old_digest=core.c._tensor_digest(labels,torch)
        combined=core.c._tensor_digest({**labels,'cached_self15_target_FP32':target15},torch)
        common={'protocol':core.pin(out/'synthetic_protocol_real_keyset.json'),'recipe':protocol['recipe'],'input_pins':protocol['inputs'],'source_pins':protocol['source_pins'],
            'fit_input_digest':'synthetic_input','fit_label_digest':old_digest,'fit_self15_label_digest':dig15,'fit_combined_label_digest':combined}
        arm=component.ARMS[0];module=modules[arm];state=initial_states[arm]
        initial={'schema':core.INITIAL_SCHEMA,'pair_schema':core.PAIR_SCHEMA,**common,'source_parent':dummy_pin,
            'WM_state':{k[3:]:v for k,v in state.items() if k.startswith('wm.')},'aux_head_state':{k[9:]:v for k,v in state.items() if k.startswith('aux_head.')},
            'physical_context_state':{k[8:]:v for k,v in state.items() if k.startswith('context.')},'normalizer_host_configuration':core.c._norm_config(module.wm),
            'ordered_parameter_names':names[arm],'RNG_this_run':core.snapshot_state(rng,torch),'construction_receipt':construction,
            'cached_self15_target_FP32':target15.clone(),'source_identity_fields':plan,'synthetic_fixture_only':True}
        initpath=out/'toy_initial_pair_state.pt';core.c._atomic_torch_save(initpath,initial,torch)
        il=torch.load(initpath,map_location='cpu',weights_only=False);cuda_point('after_synthetic_initial_readback')
        eq(il['schema']==core.INITIAL_SCHEMA and 'arm' not in il and 'iteration' not in il and il['source_identity_fields']==plan)
        payload=core.checkpoint_payload(module,optimizers[arm],core.observed_counts(module,extra[arm],offsets[arm]),gradients[arm],2,common,core.c._rng(torch,np),prefix,loss_snaps[arm],torch)
        path=out/'toy_update_0002.pt';core.c._atomic_torch_save(path,payload,torch)
        saved=torch.load(path,map_location='cpu',weights_only=False);cuda_point('after_synthetic_checkpoint_readback')
        eq(saved['schema']==core.ARTIFACT_SCHEMA and saved['checkpoint_tag']=='AFTER_OPTIMIZER_I' and saved['pair_schema']==core.PAIR_SCHEMA)
        eq(set(saved['input_pins'])==set(protocol['inputs'])==core.INPUTS and all(core.same(v,protocol['inputs'][k]) for k,v in saved['input_pins'].items()) and 'input_pins' not in protocol)
        ls=saved['loss_snapshot'];g=saved['gradient_snapshot']
        eq(ls['iteration']==g['iteration']==2 and ls['tag']==g['tag']=='PRE_OPTIMIZER_I' and ls['weights_after_updates']==g['weights_after_updates']==1)
        eq(set(ls['scalars'])==set(core.LOSS_FIELDS) and component.bit_equal(ls['self15_target_FP32'],il['cached_self15_target_FP32']))
        eq(component.bit_equal(ls['point_squared52_FP32'].mean(-1),ls['point_per_row_FP32']) and component.bit_equal(ls['self15_BCE_elements_FP32'].sum(-1),ls['self15_sum_per_row_FP32']))
        eq(component.bit_equal(ls['mode_BCE_elements_FP32'].mean(),ls['scalars']['mode']))
        eq(component.bit_equal((ls['point_per_row_FP32']+torch.tensor(.1)*ls['self15_sum_per_row_FP32']).mean(),ls['scalars']['native_pair']))
        eq(core.c._tensor_digest({'cached_self15_target_FP32':il['cached_self15_target_FP32']},torch)==saved['fit_self15_label_digest'])
        eq(all(not v.requires_grad and v.device.type=='cpu' for k,v in ls.items() if isinstance(v,torch.Tensor)))
        eq(all(float(s['step'])==2 for s in saved['optimizer_state']['state'].values()) and saved['log_prefix']==prefix)
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g['pre_clip'][k],2) for k in names[arm]]),2);coef=torch.clamp(1./(norm+1e-6),max=1.)
        eq(component.bit_equal(norm,g['pre_clip_returned_norm_FP32']) and all(component.bit_equal(g['post_clip'][k],g['pre_clip'][k]*coef) for k in names[arm]))
        with logpath.open('ab') as f:f.write(b'{"not_in_checkpoint_prefix":true}\n')
        eq(hashlib.sha256(logpath.read_bytes()[:prefix['bytes']]).hexdigest()==prefix['sha256'])
        eq(all(core.parity.raw_rng_equality(il['RNG_this_run'],saved['RNG'],np,torch).values()))
        eta=core.estimate_eta_100([.1]*100,2.,15.,datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc))
        eq(eta['remaining_pairs']==900 and abs(eta['estimated_remaining_seconds']-112.)<1e-10 and eta['planning_remaining_range_seconds'][0]<112<eta['planning_remaining_range_seconds'][1])
        checks['new_payload_loss_snapshot_PRE_AFTER_Adam_logprefix_rawRNG_and_paired100_ETA']=True
        cuda_point('final')
        eq(not TORCH_PREIMPORTED and all(v['CUDA_initialized'] is False for v in CUDA_OBSERVATIONS) and 'mujoco' not in sys.modules and 'warp' not in sys.modules and 'dex_supermimic.rl.dex_supermimic.networks' not in sys.modules)
        for value in pins.values():core.checked(value)
        report={'schema':'full151105_boundary_public_self15_training_CPU_contract:1','status':'PASS_THREE_TOY_PUBLIC15_REDUCTION_LABEL_AND_CHECKPOINT_CONTRACTS',
            'checks':checks,'passed':len(checks),'total':3,'source_pins':pins,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'Torch_preimported':TORCH_PREIMPORTED,
            'CUDA_state_observations':CUDA_OBSERVATIONS,'synthetic_last15_gradient_descriptors':last15_grad_descriptors,
            'synthetic_scope':{'Toy_WM_constructors':1,'Toy_reference_FW':1,'Toy_training_FW_backward_optimizer_steps':4,'Toy_self15_gradient_queries':2,'own_Toy_checkpoint_loads':2,'actual16_parameter_qualification':False},
            'actual_dataset_labels_weights_reads_native_WM_constructor_FW_AD_optimizer_MJB_Env_physics_GPU':0}
        core.c._write_json(out/'report.json',report)
        ready={'schema':'full151105_boundary_public_self15_training_CPU_READY:1','status':report['status'],'source_pins':pins,'cpu_report':core.pin(out/'report.json'),
            'cpu_checks':{'passed':len(checks),'total':3},'recipe':core.pin(core.RECIPE),'source_contract':core.pin(core.SOURCE_CONTRACT),
            'api':{'entry':str(core.OWN['CLI']),'module':str(core.OWN['consumer']),'metadata':'metadata_preflight(protocol_path,output=None)','run':'run_training(protocol_path,output,emit=None)'},'actual_training_authorized':False}
        core.c._write_json(out/'READY.json',ready)
        print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':core.pin(out/'READY.json')}))
    except BaseException as error:
        core.c._write_json(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_data_weights_native_models_FW_AD_optimizer_MJB_GPU':0,'CUDA_state_observations':CUDA_OBSERVATIONS});raise
    finally:
        for module in modules.values():module.close()


if __name__=='__main__':main()
