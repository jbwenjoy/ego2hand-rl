"""Strict original104 WorldModel CPU diagnostics, never resume or simulation.

Physical xi46 is distinct from privileged task/critic context368. Parameters
and normalizers remain frozen; control inputs retain ordinary Torch autograd.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .io import canonical,checked,loadnp,pin,read
from .checkpoint_v2 import load_actor_checkpoint,_tensor_digest
from .bundle_v2 import FIELDS

HEALTH_V3_SHA='7a6a4e21d24f5acad0eccc6908bef03fd7e1c4acaec7196edf296cc75772850f'
STATE_DIM=58
XI_DIM=46
DISTURBANCE_DIM=174
DYNAMICS_DIM=278
CONTROL_DIM=20
CONTACT_DIM=15
DELTA_DIM=52


def _require(value,message):
    if not value:raise ValueError(message)


def _physical_xi(context,bundle_context,worlds):
    """Validate saved actual8 and bitwise physical reader context per world."""
    xi=context['reader_context'];expected=bundle_context['reader_context']
    _require(xi.dtype==np.float32 and xi.shape==(worlds,XI_DIM) and np.isfinite(xi).all(),'Saved actual physical xi must be finite FP32 worlds x46, not taskcontext368')
    _require(expected.dtype==np.float32 and expected.ndim==2 and expected.shape[1]==XI_DIM and np.isfinite(expected).all(),'Bundle physical xi46 ABI mismatch')
    wanted=np.broadcast_to(expected[:1],xi.shape)
    _require(np.array_equal(xi.view(np.uint32),wanted.view(np.uint32)),'Actual physical xi differs bitwise from frozen bundle')
    for field in FIELDS:
        got=context[field];source=bundle_context[field]
        _require(got.shape[0]==worlds and np.isfinite(got).all() and np.array_equal(got,np.broadcast_to(source[0].astype(got.dtype),got.shape)),'Actual8 context differs: '+field)
    return xi[0].copy()


def _wm_payload(state):
    """Static native WM architecture/normalizer guard before strict load."""
    import torch
    _require(state.get('format_version')==104 and state.get('contact_dim')==CONTACT_DIM,'Original104 native contact15 checkpoint required')
    _require(canonical(state['config']['wm_hidden'])==canonical([512,512,512]),'Only frozen512x3 WM architecture is supported')
    weights=state['modules']['wm']
    shapes={'norm.mean':(298,),'norm.var':(298,),'delta_norm.mean':(52,),'delta_norm.var':(52,),'error_floor':(52,),'net.0.weight':(512,298),'net.3.weight':(512,512),'net.6.weight':(512,512),'net.9.weight':(67,512),'net.9.bias':(67,)}
    for name,shape in shapes.items():
        value=weights[name]
        _require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and tuple(value.shape)==shape and bool(torch.isfinite(value).all()),'WM input/delta/contact tensor ABI mismatch: '+name)
    for name in ('norm.var','delta_norm.var'):_require(bool((weights[name]>=0).all()),'Negative WM normalization variance')
    for name in ('norm.count','delta_norm.count'):
        value=weights[name];_require(value.dtype==torch.float64 and value.numel()==1 and bool(torch.isfinite(value).all()) and float(value)>=0,'Invalid native WM normalization count')
    return weights


def load_world_model_checkpoint(checkpoint,bundle,expected_method,training_run,health_report):
    """Return (frozen_cpu_wrapper, receipt); forward(state58,u20) is grad-enabled.

    This explicitly consumes actualu, without projecting/clamping it again.
    Caller owns state58 palm-frame encoding and lawful-control interpretation.
    """
    import torch
    from dex_supermimic.rl.dex_supermimic.networks import WorldModel
    from dex_supermimic.rl.dex_supermimic.state import STATE_DIM as native_state,DELTA_DIM as native_delta
    _require(not torch.cuda.is_initialized(),'Use a fresh CPU-only diagnostic process')
    _require((native_state,native_delta)==(STATE_DIM,DELTA_DIM),'Native state/delta ABI changed')
    path=Path(checkpoint).resolve(strict=True);run=Path(training_run).resolve(strict=True);health_path=Path(health_report).resolve(strict=True)
    _require(path==run/'train/model_final.pt','Only declared actual final2000 file in training_run is supported')
    before=pin(path);health=read(health_path)
    _require(health.get('schema')=='rough104_checkpoint_CPU_health:1' and health.get('status')=='PASS_SAVED_ENGINEERING_HEALTH' and type(health.get('checks')) is dict and health['checks'] and all(value is True for value in health['checks'].values()),'Actual v3 final checkpoint health is not qualified')
    _require(type(health.get('passed')) is int and type(health.get('total')) is int and health['passed']==health['total']==len(health['checks']) and type(health.get('seed')) is int,'Malformed v3 health counts/seed')
    _require(health.get('checkpoint')==before and health.get('method')==expected_method and type(health.get('iteration')) is int and health['iteration']==2000 and health.get('final_iteration_scope') is True and Path(health['run_dir']).resolve()==run,'Health/run/method/final2000/checkpoint mismatch')
    auditor=checked(health['auditor']);_require(health['auditor']['sha256']==HEALTH_V3_SHA and auditor.name=='audit_rough_checkpoint_cpu_v3.py','Require frozen v3 health producer')
    training_pin=health['config'];training=read(checked(training_pin))
    _require(pin(run/'config.json')['sha256']==training_pin['sha256'] and Path(training['output']).resolve()==run and training['role']=='primary_matched' and training['method']==expected_method and type(training['seed']) is int and training['seed']==health['seed'] and type(training['training']['max_iterations']) is int and training['training']['max_iterations']==2000,'Health is not the actual primary training configuration')
    # Reuse all Actor104 root/controller/data/velocity/source/MJB qualifications;
    # the Actor object itself is not used or retained by this loader.
    actor,actor_receipt=load_actor_checkpoint(path,bundle,expected_method,device='cpu');del actor
    state=torch.load(path,map_location='cpu',weights_only=False)
    _require(type(state['iteration']) is int and state['iteration']==2000 and state['rough_controller']['root_descriptor']['config_sha256']==training_pin['sha256'],'Saved final/config identity differs')
    _require(training['input_bundle']==bundle['descriptor']['input_bundle'],'WorldModel data/source bundle differs')
    worlds=training['training']['num_envs'];_require(type(worlds) is int and worlds==128,'Frozen primary world count differs')
    context_path=run/'fixed_physics_context.npz';context=loadnp(context_path)
    # Actor validation reloads the authoritative bundle. Resolve its context
    # pin again rather than trusting a caller-mutated in-memory xi dictionary.
    authoritative_spec=read(checked(actor_receipt['bundle']))
    source_context_pin=authoritative_spec['context'];source_context=loadnp(checked(source_context_pin))
    physical_xi=_physical_xi(context,source_context,worlds)
    weights=_wm_payload(state);weight_sha=_tensor_digest(weights)
    with torch.random.fork_rng(devices=[]):
        wm=WorldModel(dynamics_dim=DYNAMICS_DIM,contact_dim=CONTACT_DIM,hidden=(512,512,512),amp=False)
        wm.load_state_dict(weights,strict=True);wm.eval().requires_grad_(False)
    _require(_tensor_digest(wm.state_dict())==weight_sha,'WM weights/normalizers changed during loading')
    class FrozenCPUWorldModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.wm=wm;self.register_buffer('physical_xi',torch.from_numpy(physical_xi.copy()));self.eval().requires_grad_(False)
        def forward(self,x,u):
            if not isinstance(x,torch.Tensor) or not isinstance(u,torch.Tensor):raise ValueError('WM CPU diagnostics require Torch tensors')
            _require(x.ndim==2 and x.shape[1]==STATE_DIM and x.shape[0]>0 and u.shape==(x.shape[0],CONTROL_DIM),'WM state58/control20 batch mismatch')
            _require(x.device.type=='cpu' and u.device.type=='cpu' and x.dtype==torch.float32 and u.dtype==torch.float32,'WM diagnostic inputs must be FP32 CPU')
            _require(not self.training and not self.wm.training and bool(torch.isfinite(x).all()) and bool(torch.isfinite(u).all()),'Finite evaluation-mode inputs required')
            xi=self.physical_xi[None].expand(x.shape[0],-1)
            disturbance=x.new_zeros((x.shape[0],DISTURBANCE_DIM))
            dynamics=torch.cat((x,xi,disturbance),-1)
            x1,contact=self.wm(x,u,dynamics)
            _require(x1.shape==x.shape and contact.shape==(x.shape[0],CONTACT_DIM) and bool(torch.isfinite(x1).all()) and bool(torch.isfinite(contact).all()),'Nonfinite or malformed native WorldModel output')
            return x1,contact
    wrapper=FrozenCPUWorldModel()
    _require(pin(path)==before,'Checkpoint bytes changed during diagnostic loading')
    receipt={'schema':'rough104_world_model_CPU_evaluation_load:1','status':'PASS_FROZEN_CPU_WORLD_MODEL_LOAD','checkpoint':before,'training_run':str(run),'health':pin(health_path),'health_auditor':health['auditor'],'training_config':training_pin,'fixed_physics_context':pin(context_path),'source_physical_context':source_context_pin,'bundle':bundle['descriptor']['input_bundle'],'loader':pin(__file__),'actor104_validation':actor_receipt,'method':expected_method,'seed':health['seed'],'iteration':2000,'WM_state_sha256':weight_sha,'WM_hidden':[512,512,512],'state_dim':STATE_DIM,'physical_xi_dim':XI_DIM,'native_task_context_dim':state['context_dim'],'disturbance_dim':DISTURBANCE_DIM,'dynamics_dim':DYNAMICS_DIM,'WM_total_input_dim':298,'delta_dim':DELTA_DIM,'self_collision_logits_dim':CONTACT_DIM,'physical_context_worlds_bitexact':worlds,'parameter_requires_grad':False,'input_control_autograd_supported':True,'amp_enabled_for_CPU_diagnostic':False,'training_amp_source_value':state['config']['amp'],'resume_supported':False,'GPU_used':False,'Env_or_Warp_created':False,'physics_steps':0,'scope':'Strict frozen native FP32CPU WM, original state58+actualu20 and physicalxi46+zero174 disturbance. Contact15 is native self-collision logits, not hand-object contact/force prediction. No GPUAMP parity, physical gradient direction or dynamics-accuracy qualification.'}
    return wrapper,receipt
