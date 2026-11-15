"""New Fn6 pen schema, reusing maintained DIRECT/impulse response math."""
import numpy as np
import torch
from .compound_response import fit_scales
from .impulse_response import ImpulseResponseWM

INPUT_SCHEMA='controlled_pen_objectFirst_state55_engineWarm26_actualPD20_nativeDt1:1'
OUTPUT_SCHEMA='world_left_delta52_nextWarm26_sourceBODY_wrench6_Fn6_H20:1'
HEAD_SCHEMA='pen_joint40_COM3_worlddR3_WORLD_defect6_nextWarm26_sourceBODY_wrench6_Fn6:1'
CV_HEAD_SCHEMA='pen_CV_residual_jointQ20_COM3_worldRotvec3_other_external90_H20:1'


def pen_families(config):
    if type(config.get('schema',1)) is not int:raise ValueError('Integer checkpoint family schema required')
    if config.get('schema',1)==1 and 'model_comparison' not in config:return ('DIRECT','IMPULSE')
    if config.get('schema')==2 and config.get('model_comparison')=='DIRECT_VS_DIRECT_CV':return ('DIRECT','DIRECT_CV')
    raise ValueError('Explicit legacy impulse or v2 constant-velocity comparison required')


def head_schema(mode):
    return {'DIRECT':OUTPUT_SCHEMA,'IMPULSE':HEAD_SCHEMA,'DIRECT_CV':CV_HEAD_SCHEMA}[mode]


def kinematic_baseline(x):
    """H20 baseline in the existing WORLD-left rotation-vector coordinates."""
    if x.shape[-1]!=102:raise ValueError('Explicit state55/warm26/PD20/nativeDt1 input')
    s=x.double()
    return torch.cat([.02*s[...,20:40],torch.zeros_like(s[...,:20]),.02*s[...,49:52],.02*s[...,52:55],
                      torch.zeros(s.shape[:-1]+(44,),dtype=s.dtype,device=s.device)],dim=-1)


def kinematic_baseline_numpy(x):
    # Match the model's F32 inputs, F64 physical baseline arithmetic exactly.
    s=np.asarray(x,dtype=np.float32).astype(np.float64)
    if s.shape[-1]!=102:raise ValueError('Explicit102-input baseline')
    return np.concatenate([.02*s[...,20:40],np.zeros(s.shape[:-1]+(20,)),.02*s[...,49:52],.02*s[...,52:55],
                           np.zeros(s.shape[:-1]+(44,))],axis=-1)


def training_head_scales(scales,coupled_scales,x,y,modes):
    result={}
    for mode in modes:
        if mode=='IMPULSE':result[mode]=coupled_scales
        else:
            mean=scales['y_mean'].copy()
            if mode=='DIRECT_CV':mean[:46]=(y-kinematic_baseline_numpy(x))[:,:46].mean(0)
            result[mode]=dict(head_mean=mean,head_scale=scales['y_scale'].copy())
    return result


def training_cache(data,device,modes):
    mask=data['valid'];x=data['x'][mask];y=data['y'][mask]
    result=dict(x=torch.as_tensor(x,dtype=torch.float32,device=device),y=torch.as_tensor(y,dtype=torch.float64,device=device))
    for mode in modes:
        target=data['heads'][mask] if mode=='IMPULSE' else y-kinematic_baseline_numpy(x) if mode=='DIRECT_CV' else y
        result[mode]=torch.as_tensor(target,dtype=torch.float64,device=device)
    return result


def validate_arm_schema(saved):
    version=saved['config'].get('schema',1);mode=saved['mode']
    if (version not in [1,2] or saved['schema']!=f'controlled_pen_response_arm:{version}'
            or mode not in pen_families(saved['config']) or saved['input_schema']!=INPUT_SCHEMA
            or saved['output_schema']!=OUTPUT_SCHEMA or saved['head_schema']!=head_schema(mode)
            or saved['coupled']!=(mode=='IMPULSE')
            or version==2 and saved.get('pose_baseline')!=('CONSTANT_VELOCITY' if mode=='DIRECT_CV' else 'ZERO')):
        raise ValueError('Explicit matching family/pose prior/schema required; no legacy optimizer reinterpretation')


def model_from_arm(saved,device):
    validate_arm_schema(saved)
    model=PenResponseWM(saved['scales'],saved['head_scales'],saved['physics'],width=saved['width'],coupled=saved['coupled'],
                        pose_baseline=saved.get('pose_baseline','ZERO')).to(device)
    model.load_state_dict(saved['model'],strict=True)
    return model


def fit_pen_scales(x,y,heads):
    x,y,heads=[np.asarray(v,float) for v in [x,y,heads]]
    if x.shape!=(len(x),102) or y.shape!=(len(x),90) or heads.shape!=y.shape or len(x)<2 or not np.isfinite(np.r_[x.ravel(),y.ravel(),heads.ravel()]).all() or np.any(y[:,84:]<0):
        raise ValueError('Finite explicitly supplied valid TRAIN102/90/90 matrices and nonnegative Fn6')
    base=fit_scales(x,y[:,:87]);scales={k:v.copy() for k,v in base.items()}
    scales['y_mean']=np.r_[base['y_mean'][:84],np.zeros(6)]
    scales['y_scale']=np.r_[base['y_scale'][:84],np.maximum(y[:,84:].std(0),.1)]
    hm=scales['y_mean'].copy();hs=scales['y_scale'].copy()
    hm[46:52]=heads[:,46:52].mean(0);hs[46:52]=np.maximum(heads[:,46:52].std(0),np.r_[np.full(3,1e-6),np.full(3,1e-8)])
    return scales,dict(head_mean=hm,head_scale=hs)


class PenResponseWM(ImpulseResponseWM):
    def __init__(self,scales,head_scales,physics,*,width=512,coupled=True,pose_baseline='ZERO'):
        if len(scales['y_mean'])!=90 or len(head_scales['head_mean'])!=90:raise ValueError('Explicit pen Fn6 schema; old87-output checkpoints incompatible')
        if pose_baseline not in ['ZERO','CONSTANT_VELOCITY'] or coupled and pose_baseline!='ZERO':raise ValueError('Declared independent-head pose baseline')
        super().__init__(scales,head_scales,physics,width=width,coupled=coupled)
        self.pose_baseline=pose_baseline
        self.net[-1]=torch.nn.Linear(width,90)
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def decode_heads(self,x,h):
        p=super().decode_heads(x,h)
        return p+kinematic_baseline(x) if self.pose_baseline=='CONSTANT_VELOCITY' else p
