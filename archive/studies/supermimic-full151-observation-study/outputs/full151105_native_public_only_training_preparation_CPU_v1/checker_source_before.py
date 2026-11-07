"""UNEXECUTED DRAFT: exactly3 synthetic native-only training contracts.

Never load actual weights/data or instantiate nativeWorldModel. Root must first
approve the formed source and NEW reader. Tiny Torch AD/optimizer is explicit.
"""
from pathlib import Path
import copy,datetime,hashlib,json,math,os,sys,time,traceback
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
OUT=ROOT/'outputs/full151105_native_public_only_training_preparation_CPU_v1'
FILES={'consumer':ROOT/'src/full151_gradient/native_public_only_training105_v1.py','CLI':ROOT/'scripts/train_full151_native_public_only_cpu_v1.py','checker':Path(__file__)}
DEPS={'parent_trainer':ROOT/'src/full151_gradient/boundary_public_self15_training105_v1.py','parent_eval':ROOT/'src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py','component':ROOT/'src/full151_gradient/boundary_impulse_component105_v1.py','recipe':ROOT/'protocols/full151105_native_public_only_training_recipe_v1.json','source_contract':ROOT/'docs/public_native_only_training_contract_v1.md'}

def pin(path):return {'path':str(Path(path).resolve(strict=True)),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    OUT.mkdir();pins={key:pin(value) for key,value in FILES.items()};deps={key:pin(value) for key,value in DEPS.items()}
    for name,path in FILES.items():(OUT/(name+'_source_before.py')).write_bytes(path.read_bytes())
    os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(ROOT/'src'));started=time.perf_counter();checks={};modules={}
    try:
        import numpy as np
        import torch
        from full151_gradient import native_public_only_training105_v1 as h
        from full151_gradient import boundary_impulse_component105_v1 as component
        torch.set_num_threads(4);cuda_after_import=torch.cuda.is_initialized();r=h.fixed_recipe()
        def reject(fn):
            try:fn()
            except ValueError:return True
            return False
        # 1: Closed-form native reduction and sum15 gradient, no aux arguments.
        delta=torch.ones((152,52),requires_grad=True);scale=torch.ones(52);logits=torch.zeros((152,15),requires_grad=True);y15=torch.ones_like(logits)
        loss,ops=h.native_only_losses(delta,scale,logits,y15,r,torch)
        dgrad,lgrad=torch.autograd.grad(loss['L_native_pair_FP32'],(delta,logits))
        expected=1.+.1*15*math.log(2.)
        checks['native_row_reduction_sum15_closedform_gradient_no_aux_loss']=bool(set(loss)==set(h.LOSS_FIELDS.values()) and len(loss)==3 and len(ops)==9
            and float(loss['L_state_FP32'])==1. and abs(float(loss['L_self15_sum_FP32'])-15*math.log(2.))<3e-6 and abs(float(loss['L_native_pair_FP32'])-expected)<1e-6
            and torch.equal(ops['native_pair_per_row_FP32'].mean(),loss['L_native_pair_FP32']) and torch.allclose(lgrad,torch.full_like(lgrad,-.05/152),atol=1e-10,rtol=1e-6) and bool(torch.isfinite(dgrad).all()))
        # 2: Own Tiny architecture has native names14/512->67, not native math.
        class Norm(torch.nn.Module):
            def __init__(self,n):
                super().__init__();self.register_buffer('mean',torch.zeros(n));self.register_buffer('variance',torch.ones(n));self.register_buffer('count',torch.tensor(0.,dtype=torch.float64));self.eps=1e-5;self.max_count=1000000;self._host_count=0
            def std(self):return self.variance.sqrt()
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.amp=False;self.net=torch.nn.Sequential(torch.nn.Linear(78,8),torch.nn.LayerNorm(8),torch.nn.ReLU(),torch.nn.Linear(8,8),torch.nn.LayerNorm(8),torch.nn.ReLU(),torch.nn.Linear(8,512),torch.nn.LayerNorm(512),torch.nn.ReLU(),torch.nn.Linear(512,67));self.norm=Norm(278);self.delta_norm=Norm(52);self.register_buffer('error_floor',torch.full((52,),.01))
            def forward(self,x,u,dynamics):
                a=self.net(torch.cat((x,u),-1));return torch.cat((x[:,:52]+a[:,:52],x[:,52:]),-1),a[:,52:]
        with torch.random.fork_rng(devices=[]):raw=Tiny().eval().requires_grad_(False)
        xi=torch.zeros(46);ctx=component.fixed_context_from_rows(torch.tensor(.03),torch.eye(3,dtype=torch.float64),torch.zeros((152,3),dtype=torch.float64),torch.full((152,),.02,dtype=torch.float64),torch.eye(3).expand(152,-1,-1).clone(),torch.ones(52),xi)
        modules,construction=component.make_matched_components(raw,ctx);module=modules[h.INTERNAL_ARM];unused=modules[component.ARMS[1]];unused.eval().requires_grad_(False)
        names,params=h.parent_trainer.ordered_parameters(module,h.PARAMETER_NAMES);unused_digest=component.state_digest(unused);head_digest=component.state_digest(module.aux_head)
        opt=torch.optim.AdamW(params,lr=r['lr'],weight_decay=r['weight_decay'],betas=tuple(r['betas']),eps=r['optimizer_eps'],foreach=False)
        cuda_before_optimizer_steps=torch.cuda.is_initialized();x=torch.ones((152,58))*.01;u=torch.zeros((152,20));dyn=torch.cat((x,xi[None].expand(152,-1),torch.zeros((152,174))),-1);target=torch.zeros((152,58));labels=torch.ones((152,15))
        gradient=None;snapshot=None;extra={'backward_calls':0,'optimizer_steps':0};offset=dict(module.counts)
        initial_wm=h.snapshot_state(module.wm.state_dict(),torch);initial_head=h.snapshot_state(module.aux_head.state_dict(),torch)
        for iteration in (1,2,3):
            opt.zero_grad(set_to_none=True);pred,self15,z,mode=module(x,u,dyn)
            loss,operands=h.native_only_losses(target[:,:52]-pred[:,:52],torch.ones(52),self15,labels,r,torch)
            snapshot=h.loss_snapshot_before_update(loss,operands,iteration,torch);loss['L_native_pair_FP32'].backward();extra['backward_calls']+=1
            gradient=h.clip_native_gradients(module,iteration,torch,True);opt.step();extra['optimizer_steps']+=1
            if not h.optimizer_steps_exact(opt,params,iteration,torch):raise ValueError('Tiny14 active Adam state/step failed')
        cuda_after_optimizer_steps=torch.cuda.is_initialized();counts=h.observed_counts(module,extra,offset)
        saved_none=all(gradient['pre_clip'][key] is None and gradient['post_clip'][key] is None for key in names[14:])
        bad_before=params[-1].grad;params[-1].grad=torch.zeros_like(params[-1]);bad_head_rejected=reject(lambda:h.clip_native_gradients(module,3,torch,True));params[-1].grad=bad_before
        checks['actual16_namedgroup_14_Tiny_grad_Adam_headsNone_unusedcopy_zero']=bool(tuple(names)==h.PARAMETER_NAMES and counts=={key:3 for key in h.COUNTS} and len(opt.state)==14 and len(opt.param_groups[0]['params'])==16 and saved_none
            and all(p.requires_grad for p in params) and all(p.grad is None and p not in opt.state for p in params[14:]) and component.state_digest(module.aux_head)==head_digest
            and all(component.bit_equal(value,initial_head[name]) for name,value in module.aux_head.state_dict().items()) and component.state_digest(unused)==unused_digest and not any(unused.counts.values()) and bad_head_rejected
            and all(value.dtype==torch.float32 and bool(torch.isfinite(value).all()) for key,value in gradient['pre_clip'].items() if value is not None))
        # 3: Real JSON keyset shape, own state-only checkpoint I/O, PRE/AFTER/ETA.
        parent_path=ROOT/'protocols/full151105_boundary_public_self15_matched_training_CPU_run_v1.json';parent=json.loads(parent_path.read_text(encoding='utf-8-sig'))
        own_inputs={key:{'path':str(OUT/(key+'.json')),'sha256':'0'*64} for key in h.INPUTS}
        common={'protocol':{'path':str(OUT/'synthetic_protocol.json'),'sha256':'0'*64},'recipe':h.pin(h.RECIPE),'input_pins':own_inputs,'parent_training_inputs':copy.deepcopy(parent['inputs']),'parent_training_protocol':pin(parent_path),'source_pins':pins,
            'first_training_forward_parity':{'status':'SYNTHETIC_ONLY_NOT_ACTUAL_PARITY'},'fit_input_digest':'0'*64,'fit_label_digest':'1'*64,'fit_self15_label_digest':'2'*64,'fit_combined_label_digest':'3'*64,'active_gradient_parameter_names':names[:14],'inactive_gradient_parameter_names':names[14:]}
        rng=h.c._rng(torch,np);prefix={'path':str(OUT/'updates.jsonl'),'records':3,'bytes':0,'sha256':hashlib.sha256(b'').hexdigest()}
        payload=h.checkpoint_payload(module,opt,counts,gradient,3,common,rng,prefix,snapshot,torch)
        path=OUT/'synthetic_checkpoint.pt';h.c._atomic_torch_save(path,payload,torch);readback=torch.load(path,map_location='cpu',weights_only=False)
        eta=h.estimate_eta_100([.2]*100,.1,30.,datetime.datetime(2026,11,6,tzinfo=datetime.timezone.utc))
        checks['real_protocol_inputs_own3_parent16_serialization_phase_and_single100_ETA']=bool('inputs' in parent and 'input_pins' not in parent and set(parent['inputs'])==h.parent_trainer.INPUTS and set(readback['input_pins'])==h.INPUTS
            and readback['parent_training_inputs']==parent['inputs'] and 'pair_schema' not in readback and readback['schema']==h.ARTIFACT_SCHEMA and readback['manifest_schema']==h.MANIFEST_SCHEMA and readback['arm']==h.ARM
            and readback['checkpoint_tag']=='AFTER_OPTIMIZER_I' and readback['gradient_snapshot']['tag']==readback['loss_snapshot']['tag']=='PRE_OPTIMIZER_I' and readback['gradient_snapshot']['weights_after_updates']==2
            and set(readback['loss_snapshot']['scalars'])==set(h.LOSS_FIELDS) and len(readback['optimizer_state']['state'])==14 and len(readback['optimizer_state']['param_groups'][0]['params'])==16
            and all(readback['gradient_snapshot']['pre_clip'][name] is None for name in names[14:]) and eta['remaining_updates']==900 and abs(eta['estimated_remaining_seconds']-200.1)<1e-10)
        if not all(checks.values()):raise ValueError('Synthetic contract failed '+str(checks))
        if cuda_after_import or cuda_before_optimizer_steps or cuda_after_optimizer_steps or torch.cuda.is_initialized() or any(k in sys.modules for k in ('mujoco','warp')):raise ValueError('CPU isolation failure')
        if any(pin(path)!=pins[key] for key,path in FILES.items()) or any(pin(path)!=deps[key] for key,path in DEPS.items()):raise ValueError('Proof sources changed')
        proof={'schema':'full151105_native_public_only_training_CPU_contract:1','status':'PASS_SYNTHETIC_NATIVE_ONLY_LOSS_14_ACTIVE_AND_CHECKPOINT_ABI','source_pins':pins,'dependency_pins':deps,'checks':checks,'passed':sum(checks.values()),'total':len(checks),
            'scope':{'Tiny_forward_backward_optimizer_steps':3,'synthetic_loss_autograd_calls':1,'own_synthetic_checkpoint_loads':1,'actual_NPZ_label_JSON_weight_reads':0,'native_WM_constructor_actual_FW':0,'actual_training_updates':0,'MJB_Env_physics_GPU':0},
            'CUDA_initialized':{'after_import':cuda_after_import,'before_Tiny_optimizer_steps':cuda_before_optimizer_steps,'after_Tiny_optimizer_steps':cuda_after_optimizer_steps},'real_parent_protocol_JSON_keyset':sorted(parent),'real_parent_protocol':pin(parent_path),'elapsed_seconds':time.perf_counter()-started}
        write(OUT/'report.json',proof);ready={'schema':'full151105_native_public_only_training_CPU_READY:1','status':proof['status'],'source_pins':pins,'dependency_pins':deps,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':proof['passed'],'total':proof['total']},'actual_training_authorized':False}
        write(OUT/'READY.json',ready);print(json.dumps({'status':ready['status'],'passed':proof['passed'],'total':proof['total'],'READY':pin(OUT/'READY.json')}));return 0
    except BaseException as error:
        write(OUT/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'source_pins':pins,'checks':checks,'automatic_retry':False,'actual_data_nativeModels_MJB_physics':0});raise
    finally:
        for module in modules.values():module.close()

if __name__=='__main__':raise SystemExit(main())
