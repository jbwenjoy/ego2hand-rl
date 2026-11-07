"""Phone physical WM components, independent of experiment directories.

This is the world-left state58/current366 -> physical178+Fn22+coverage22
model, not the legacy pen chart52 model in ``dex_supermimic``. No data or
checkpoint is discovered implicitly; callers supply geometry and tensors.
Torch is imported only when a numerical API is explicitly called.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Mapping
import hashlib,math,os,random
import numpy as np

INPUT_SCHEMA='phone_raw80_current_relative_pose22x12_directional22_366:1'
OUTPUT_SCHEMA='phone_physical178_meanFn22_coverage_logits22:1'
INPUT_DIM=366;PHYSICAL_HEAD_DIM=178;OUTPUT_DIM=222

@dataclass(frozen=True)
class WorldModelConfig:
    hidden:tuple[int,...]=(128,128)
    geometry_chunk_rows:int=4
    input_schema:str=INPUT_SCHEMA
    output_schema:str=OUTPUT_SCHEMA
    def __post_init__(self):
        if not self.hidden or any(type(n) is not int or n<=0 for n in self.hidden):raise ValueError('Positive hidden layer widths')
        if type(self.geometry_chunk_rows) is not int or self.geometry_chunk_rows<1:raise ValueError('Positive geometry chunk rows')
        if self.input_schema!=INPUT_SCHEMA or self.output_schema!=OUTPUT_SCHEMA:raise ValueError('Phone world-left schema required; old pen shape is incompatible')
    def to_dict(self):return asdict(self)

@dataclass(frozen=True)
class LossConfig:
    absolute_heads:float=1.
    decoded_velocity:float=1.
    meanFn:float=1.
    coverage:float=1.
    counterfactual:float=0.
    force_reference_N:float=1.
    def __post_init__(self):
        if any(not math.isfinite(v) or v<0 for v in [self.absolute_heads,self.decoded_velocity,self.meanFn,self.coverage,self.counterfactual]):raise ValueError('Finite nonnegative loss coefficients')
        if not math.isfinite(self.force_reference_N) or self.force_reference_N<=0:raise ValueError('Positive force reference in N')

def physical_blocks():
    return [np.arange(0,20),np.arange(20,40),np.arange(40,43),np.arange(43,46),
        np.array([46+6*p+j for p in range(22) for j in range(3)]),np.array([46+6*p+j for p in range(22) for j in range(3,6)])]

def physical_floors(*,q_rad=.001,qd_rad_s=.01,COM_residual_m=.0001,world_dR_rad=.001,force_impulse_Ns=.0001,torque_impulse_Nms=.00001):
    values=[q_rad,qd_rad_s,COM_residual_m,world_dR_rad,force_impulse_Ns,torque_impulse_Nms]
    if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Positive unit-specific scale floors')
    result=np.zeros(178)
    for indices,value in zip(physical_blocks(),values):result[indices]=value
    return result

def current_features(x,u,h,n,geometry,*,context=None):
    """Differentiable current-only features; no future contact/force argument.

    ``x`` is phone COM/world state58 with BODY rotation, ``u`` absolute20,
    ``h`` native timestep and ``n`` integer substeps. Geometry is injected via
    the runtime provider; this API does not load a native model or assets.
    An explicit context belongs to this exact current tensor and autograd
    mode. Sharing its FK/descriptor graph does not detach any state input.
    """
    import torch
    if x.shape[-1]!=58 or u.shape!=x.shape[:-1]+(20,) or h.shape!=x.shape[:-1] or n.shape!=h.shape:raise ValueError('state58/action20/h/n with equal batch shape')
    if not h.is_floating_point() or n.dtype not in (torch.int32,torch.int64) or torch.any((n!=64)&(n!=128)) or not torch.isfinite(h).all() or torch.any(h<=0):raise ValueError('Explicit native timestep and64/128 integer ticks')
    if torch.any(abs(h*n-.02)>4*torch.finfo(h.dtype).eps*.02):raise ValueError('Phone control period20ms; no accumulated-clock inference')
    context=geometry.bundle(x) if context is None else context
    context.require(geometry,x);pose=context.poses();gap=context.directional()
    if not gap['directional_gap_known'].all():raise ValueError('Unknown geometry must not become zero model inputs')
    rel=torch.cat([pose['pair_relative_translation_BODY_COM_m'],pose['pair_relative_rotation_BODY9']],dim=-1).flatten(-2)
    raw=torch.cat([x,u,h[...,None].to(x.dtype),n[...,None].to(x.dtype)],dim=-1)
    result=torch.cat([raw,rel,gap['directional_gap_m']],dim=-1)
    if result.shape[-1]!=366 or not torch.isfinite(result).all():raise ValueError('Finite current-only366 features required')
    return result

def precompute_features(data,geometry,*,chunk_rows=4):
    """Cache exact current-state bytes, preserving all row/action weights.

    ``data`` is a caller-selected/validated partition with x/u/h/n NumPy
    arrays. This function never opens a file, selects a split or fits norms.
    """
    import torch
    if type(chunk_rows) is not int or chunk_rows<1:raise ValueError('Positive chunk size')
    X=np.asarray(data['x']);U=np.asarray(data['u']);hs=np.asarray(data['h']);ns=np.asarray(data['n'])
    if X.ndim!=2 or X.shape[1]!=58 or U.shape!=(len(X),20):raise ValueError('Batch state/action/time shapes')
    if hs.shape!=(len(X),) or ns.shape!=(len(X),) or not np.isfinite(np.c_[X,U,hs,ns]).all():raise ValueError('Finite explicitly selected current inputs')
    if hs.dtype.kind!='f' or ns.dtype.kind not in 'iu' or not np.all(np.isin(ns,[64,128])) or np.any(hs<=0) or not np.allclose(hs*ns,.02,rtol=0,atol=4*np.finfo(hs.dtype).eps*.02):raise ValueError('Actual phone h and64/128 substeps must declare20ms')
    if not len(X):return np.empty((0,366),np.float64),dict(unique_current_states=0,rows=0,pure_geometry_FK_calls=0,geometry_is_current_only=True,caching_does_not_change_training_row_weight=True)
    unique=[];ids=[];lookup={}
    for x in X:
        key=x.tobytes()
        if key not in lookup:lookup[key]=len(unique);unique.append(x)
        ids.append(lookup[key])
    relgap=[]
    with torch.no_grad():
        for i in range(0,len(unique),chunk_rows):
            x=torch.tensor(np.array(unique[i:i+chunk_rows]),dtype=torch.float64);u=torch.zeros((len(x),20),dtype=torch.float64)
            # Only geometry80: is cached; actual h/n and u are reinstated below.
            v=current_features(x,u,torch.full((len(x),),.00015625,dtype=torch.float64),torch.full((len(x),),128,dtype=torch.int64),geometry)
            relgap.append(v[:,80:].cpu().numpy())
    G=np.concatenate(relgap)[ids];raw=np.column_stack([X,U,hs,ns])
    return np.column_stack([raw,G]),dict(unique_current_states=len(unique),rows=len(X),pure_geometry_FK_calls=math.ceil(len(unique)/chunk_rows),geometry_is_current_only=True,caching_does_not_change_training_row_weight=True)

def normalize_features(features,norms):
    import torch
    mean=norms['input_mean'];std=norms['input_std']
    if features.shape[-1]!=366 or mean.shape!=(366,) or std.shape!=(366,) or not torch.isfinite(std).all() or torch.any(std<=0):raise ValueError('Frozen366 normalizers, positive std')
    return ((features-mean)/std).to(torch.float32)

def make_model(config:WorldModelConfig|Mapping|None=None,*,device='cpu',dtype='float32'):
    """Construct the actual MLP, keeping legacy physical ``layers.*`` keys."""
    import torch
    cfg=config if isinstance(config,WorldModelConfig) else WorldModelConfig(**(dict(config) if config is not None else {}))
    if dtype!='float32':raise ValueError('FP32 NN/F64 physical decoding is the declared contract')
    dev=torch.device(device)
    if dev.type not in ('cpu','cuda') or dev.type=='cuda' and dev.index is None:raise ValueError('CPU or explicit cuda:index')
    class GeometryMomentumNet(torch.nn.Module):
        def __init__(self):
            super().__init__();layers=[];width=366
            for hidden in cfg.hidden:layers.extend([torch.nn.Linear(width,hidden),torch.nn.SiLU()]);width=hidden
            layers.append(torch.nn.Linear(width,222));self.layers=torch.nn.Sequential(*layers);self.config=cfg
        def forward(self,normalized_features366):
            if normalized_features366.shape[-1]!=366:raise ValueError('Phone current366 required, not raw78/80 or old chart')
            return self.layers(normalized_features366)
        def forward_from_raw(self,x,u,h,n,norms,geometry,*,context=None):return self(normalize_features(current_features(x,u,h,n,geometry,context=context),norms))
    previous_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device('cpu'):model=GeometryMomentumNet()
    finally:torch.set_default_dtype(previous_dtype)
    return model.to(device=dev,dtype=torch.float32)

def heads_from_physical(v):
    if v.shape[-1]!=178:raise ValueError('Physical178 required, not native chart52')
    return dict(joint_delta40=v[...,:40],COM_ballistic_residual3_m=v[...,40:43],world_rotation_delta3_rad=v[...,43:46],pair_impulse_source_BODY_moving_COM22x6=v[...,46:].reshape(v.shape[:-1]+(22,6)))

def block_mse(error,blocks=None):
    import torch
    return torch.stack([error[...,index].square().mean() for index in (physical_blocks() if blocks is None else blocks)]).mean()

def decode_output(raw,x,h,n,norms,physical,*,codec=None):
    """Exact source-BODY/moving-COM impulse decoder; no zero-defect labels."""
    import torch
    if codec is None:from supermimic_research.runtime import structured as codec
    head=raw[...,:178].double()*norms['head_std']+norms['head_mean']
    y=codec.decode_physical_heads(x.double(),heads_from_physical(head),(h*n).double(),physical['mass'],physical['I'],physical['g'])
    return dict(next_state58=y,physical178=head,meanFn22_N=torch.nn.functional.softplus(raw[...,178:200]),coverage_logits22=raw[...,200:222],coverage_probability22_F64=torch.sigmoid(raw[...,200:222].double()))

def absolute_loss(model,batch,norms,scales,physical,*,config=LossConfig(),codec=None):
    """Configurable actual WM supervision, with no hidden fit/split policy.

    Required batch x/y/T/head/Fn/cov/feature_F64 and explicit physical-unit
    scales head/velocity. Optional cf_side/cf_base/du/cf_target only when the
    counterfactual coefficient is explicitly positive. Unknown/partial rows
    must be masked by the data loader, never silently finite-filtered here.
    """
    import torch
    if codec is None:from supermimic_research.runtime import structured as codec
    raw=model(normalize_features(batch['feature_F64'],norms));head=raw[:,:178].double()*norms['head_std']+norms['head_mean']
    pred=codec.decode_physical_heads(batch['x'],heads_from_physical(head),batch['T'],physical['mass'],physical['I'],physical['g'])
    lh=block_mse((head-batch['head'])/scales['head'])
    err=(pred[:,52:58]-batch['y'][:,52:58])/scales['velocity'];lv=(err[:,:3].square().mean()+err[:,3:].square().mean())*.5
    fn=torch.nn.functional.softplus(raw[:,178:200])*config.force_reference_N
    lf=(torch.log1p(fn/config.force_reference_N)-torch.log1p(batch['Fn']/config.force_reference_N)).square().mean()
    lc=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,200:222],batch['cov']);total=config.absolute_heads*lh+config.decoded_velocity*lv
    terms=dict(physical_heads=lh,decoded_velocity=lv,meanFn=lf,coverage=lc)
    if config.counterfactual:
        du=batch['du']
        if not torch.isfinite(du).all() or torch.any(abs(du)<1e-6):raise ValueError('Explicit finite actual CF denominator >=1e-6, no epsilon substitute')
        sec=(head[batch['cf_side']]-head[batch['cf_base']])/du[:,None]
        terms['counterfactual']=block_mse((sec-batch['cf_target'])/scales['CF']);total=total+config.counterfactual*terms['counterfactual']
    total=total+config.meanFn*lf+config.coverage*lc
    return total,terms,dict(raw222=raw,physical178=head,next_state58=pred)

def tensor_digest(values):
    import torch
    result=hashlib.sha256()
    for name,value in sorted(values.items()):
        value=value.detach().cpu().contiguous();result.update((name+str(value.dtype)+str(tuple(value.shape))).encode());result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()

def raw_tree_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(raw_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(raw_tree_equal(x,y) for x,y in zip(a,b))
    return a==b

def capture_rng(*,device='cpu'):
    import torch
    dev=torch.device(device);return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state(dev).clone() if dev.type=='cuda' else None)
def move_tensors(value,device):
    import torch
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move_tensors(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move_tensors(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move_tensors(v,device) for v in value)
    return value
def restore_rng(state,*,device='cpu'):
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.device(device).type=='cuda':torch.cuda.set_rng_state(state['cuda'],device)
def atomic_checkpoint(path,payload):
    import torch
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
