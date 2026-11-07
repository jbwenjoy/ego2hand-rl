"""Strict format105 frozen WorldModel CPU consumer; no training or simulation.

No real full151 checkpoint/105 health auditor exists in this A0-only stage.
This consumer declares a fail-closed future health contract instead of accepting
old104 health or relabelling old Actor/WM payloads. Contact15 remains self-collision.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import canonical,checked,loadnp,pin,read
from .metadata import FORMAT_VERSION,MODE,VARIANT
from .checkpoint import load_actor_checkpoint,_tensor_digest,validate_checkpoint_namespace
from .bundle import FIELDS

STATE_DIM=58
XI_DIM=46
DISTURBANCE_DIM=174
DYNAMICS_DIM=278
CONTROL_DIM=20
CONTACT_DIM=15
DELTA_DIM=52
HEALTH_SCHEMA='full151105_checkpoint_CPU_health:1'


def _require(value,message):
    if not value:raise ValueError(message)


def _physical_xi(context,bundle_context,worlds):
    """Actual8 and physicalxi46; never substitute privileged taskcontext368."""
    _require(type(worlds) is int and worlds>0,'World count must be a positive native integer')
    xi=context['reader_context'];expected=bundle_context['reader_context']
    _require(xi.dtype==np.float32 and xi.shape==(worlds,XI_DIM) and np.isfinite(xi).all(),'Saved actual physical xi must be finite FP32 worlds x46')
    _require(expected.dtype==np.float32 and expected.ndim==2 and expected.shape[1]==XI_DIM and np.isfinite(expected).all(),'Bundle physical xi46 ABI mismatch')
    _require(np.array_equal(xi.view(np.uint32),np.broadcast_to(expected[:1],xi.shape).view(np.uint32)),'Actual physical xi differs bitwise from frozen bundle')
    for field in FIELDS:
        got=context[field];source=bundle_context[field]
        _require(got.dtype==source.dtype and got.ndim==source.ndim and got.shape[0] in (1,worlds) and got.shape[1:]==source.shape[1:] and np.isfinite(got).all(),'Actual8 storage ABI differs: '+field)
        target=np.broadcast_to(source[:1],got.shape)
        _require(got.tobytes()==np.ascontiguousarray(target).tobytes(),'Actual8 context differs bitwise: '+field)
    return xi[0].copy()


def _wm_payload(state):
    """Static exact native WM architecture/normalizer guard, independently testable."""
    import torch
    validate_checkpoint_namespace(state)
    _require(type(state.get('contact_dim')) is int and state['contact_dim']==CONTACT_DIM,'Native self-collision15 head required')
    _require(canonical(state['config']['wm_hidden'])==canonical([512,512,512]),'Only frozen512x3 WM architecture is supported')
    weights=state['modules']['wm']
    shapes={'norm.mean':(298,),'norm.var':(298,),'delta_norm.mean':(52,),'delta_norm.var':(52,),'error_floor':(52,),'net.0.weight':(512,298),'net.3.weight':(512,512),'net.6.weight':(512,512),'net.9.weight':(67,512),'net.9.bias':(67,)}
    for name,shape in shapes.items():
        value=weights[name]
        _require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and tuple(value.shape)==shape and bool(torch.isfinite(value).all()),'WM input/delta/contact tensor ABI mismatch: '+name)
    for name in ('norm.var','delta_norm.var'):_require(bool((weights[name]>=0).all()),'Negative WM normalization variance')
    for name in ('norm.count','delta_norm.count'):
        value=weights[name];_require(value.dtype==torch.float64 and value.numel()==1 and bool(torch.isfinite(value).all()) and float(value)>=0,'Invalid native WM normalization count')
    from .checkpoint import _finite_tree
    _finite_tree(weights,torch)
    return weights


def validate_health_metadata(health,*,checkpoint_pin,expected_method,training_run,expected_iteration,expected_health_auditor):
    """Require a future independently frozen105 auditor, never104 proof."""
    _require(type(expected_health_auditor) is dict,'New105 health producer pin must be explicit; this stage has none')
    producer=checked(expected_health_auditor)
    _require(producer.name not in ('audit_rough_checkpoint_cpu.py','audit_rough_checkpoint_cpu_v2.py','audit_rough_checkpoint_cpu_v3.py'),'Old104 auditor cannot qualify105')
    _require(type(expected_iteration) is int and expected_iteration>0,'Explicit positive final iteration required')
    checks=health.get('checks')
    _require(health.get('schema')==HEALTH_SCHEMA and health.get('status')=='PASS_SAVED_ENGINEERING_HEALTH' and type(checks) is dict and bool(checks) and all(type(k) is str and v is True for k,v in checks.items()),'Actual new105 final checkpoint health is not qualified')
    _require(type(health.get('passed')) is int and type(health.get('total')) is int and health['passed']==health['total']==len(checks) and type(health.get('seed')) is int,'Malformed105 health counts/seed')
    _require(health.get('auditor')==expected_health_auditor and health.get('checkpoint')==checkpoint_pin and health.get('method')==expected_method and type(health.get('iteration')) is int and health['iteration']==expected_iteration and health.get('final_iteration_scope') is True and Path(health['run_dir']).resolve()==Path(training_run).resolve(),'105 health producer/run/method/final/checkpoint mismatch')
    return health


def _make_frozen_wrapper(wm,physical_xi):
    import torch
    class FrozenCPUWorldModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.wm=wm;self.register_buffer('physical_xi',torch.from_numpy(physical_xi.copy()));self.eval().requires_grad_(False)
        def forward(self,x,u):
            if not isinstance(x,torch.Tensor) or not isinstance(u,torch.Tensor):raise ValueError('WM CPU diagnostics require Torch tensors')
            _require(x.ndim==2 and x.shape[1]==STATE_DIM and x.shape[0]>0 and u.shape==(x.shape[0],CONTROL_DIM),'WM state58/control20 batch mismatch')
            _require(x.device.type=='cpu' and u.device.type=='cpu' and x.dtype==torch.float32 and u.dtype==torch.float32,'WM diagnostic inputs must be FP32 CPU')
            _require(not self.training and not self.wm.training and bool(torch.isfinite(x).all()) and bool(torch.isfinite(u).all()),'Finite evaluation-mode inputs required')
            dynamics=torch.cat((x,self.physical_xi[None].expand(x.shape[0],-1),x.new_zeros((x.shape[0],DISTURBANCE_DIM))),-1)
            x1,contact=self.wm(x,u,dynamics)
            _require(x1.shape==x.shape and contact.shape==(x.shape[0],CONTACT_DIM) and bool(torch.isfinite(x1).all()) and bool(torch.isfinite(contact).all()),'Nonfinite or malformed native WorldModel output')
            return x1,contact
    return FrozenCPUWorldModel()


def load_world_model_checkpoint(checkpoint,bundle,expected_method,training_run,health_report,*,expected_health_auditor,expected_iteration):
    """Freeze actual new105 WM only after a new trusted final health audit.

    forward(state58, actualu20) keeps input autograd, adds physicalxi46 and zero
    disturbance174, never repeats C2. No health producer is supplied by default.
    """
    import torch
    from dex_supermimic.rl.dex_supermimic.networks import WorldModel
    from dex_supermimic.rl.dex_supermimic.state import STATE_DIM as native_state,DELTA_DIM as native_delta
    _require(not torch.cuda.is_initialized(),'Use a fresh CPU-only diagnostic process')
    _require((native_state,native_delta)==(STATE_DIM,DELTA_DIM),'Native state/delta ABI changed')
    path=Path(checkpoint).resolve(strict=True);run=Path(training_run).resolve(strict=True)
    before=pin(path);state=torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint_namespace(state)
    _require(path==run/'train/model_final.pt','Only actual final file in declared training_run is supported')
    health_path=Path(health_report).resolve(strict=True)
    health=validate_health_metadata(read(health_path),checkpoint_pin=before,expected_method=expected_method,training_run=run,expected_iteration=expected_iteration,expected_health_auditor=expected_health_auditor)
    training_pin=health['config'];training=read(checked(training_pin))
    _require(pin(run/'config.json')['sha256']==training_pin['sha256'] and Path(training['output']).resolve()==run and training['role']=='primary_matched' and training['method']==expected_method and type(training['seed']) is int and training['seed']==health['seed'] and type(training['training']['max_iterations']) is int and training['training']['max_iterations']==expected_iteration,'105 health is not actual primary final training config')
    actor,actor_receipt=load_actor_checkpoint(path,bundle,expected_method,device='cpu');del actor
    _require(type(state['iteration']) is int and state['iteration']==expected_iteration and state['full151_controller']['root_descriptor']['config_sha256']==training_pin['sha256'],'Saved final/config identity differs')
    _require(training['input_bundle']==bundle['descriptor']['input_bundle'],'WorldModel data/source bundle differs')
    worlds=training['training']['num_envs'];_require(type(worlds) is int and worlds==128,'Frozen primary world count differs')
    context_path=run/'fixed_physics_context.npz';context=loadnp(context_path)
    authoritative_spec=read(checked(actor_receipt['bundle']));source_context_pin=authoritative_spec['context'];source_context=loadnp(checked(source_context_pin))
    physical_xi=_physical_xi(context,source_context,worlds)
    weights=_wm_payload(state);weight_sha=_tensor_digest(weights)
    with torch.random.fork_rng(devices=[]):
        wm=WorldModel(dynamics_dim=DYNAMICS_DIM,contact_dim=CONTACT_DIM,hidden=(512,512,512),amp=False)
        wm.load_state_dict(weights,strict=True);wm.eval().requires_grad_(False)
    _require(_tensor_digest(wm.state_dict())==weight_sha,'WM weights/normalizers changed during loading')
    wrapper=_make_frozen_wrapper(wm,physical_xi)
    _require(pin(path)==before,'Checkpoint bytes changed during diagnostic loading')
    receipt={'schema':'full151_105_world_model_CPU_evaluation_load:1','status':'PASS_FROZEN_CPU_WORLD_MODEL_LOAD','checkpoint':before,'training_run':str(run),'health':pin(health_path),'health_auditor':health['auditor'],'training_config':training_pin,'fixed_physics_context':pin(context_path),'source_physical_context':source_context_pin,'bundle':actor_receipt['bundle'],'loader':pin(__file__),'actor105_validation':actor_receipt,'mode':MODE,'retarget_variant_id':VARIANT,'method':expected_method,'seed':health['seed'],'iteration':expected_iteration,'WM_state_sha256':weight_sha,'WM_hidden':[512,512,512],'state_dim':STATE_DIM,'physical_xi_dim':XI_DIM,'native_task_context_dim':state['context_dim'],'disturbance_dim':DISTURBANCE_DIM,'dynamics_dim':DYNAMICS_DIM,'WM_total_input_dim':298,'delta_dim':DELTA_DIM,'self_collision_logits_dim':CONTACT_DIM,'physical_context_worlds_bitexact':worlds,'actual8_storage_shapes':{field:list(context[field].shape) for field in FIELDS},'parameter_requires_grad':False,'input_control_autograd_supported':True,'amp_enabled_for_CPU_diagnostic':False,'resume_supported':False,'GPU_used':False,'Env_or_Warp_created':False,'physics_steps':0,'scope':'Frozen native FP32CPU WM: state58+actualu20, physicalxi46+zero174 disturbance. Contact15 is self-collision logits, not hand-object contact/force. Loading does not qualify known-rough geometry/dynamics or physical gradients.'}
    return wrapper,receipt
