"""New Fn6 pen schema, reusing maintained DIRECT/impulse response math."""
import numpy as np
import torch
from .compound_response import fit_scales
from .impulse_response import ImpulseResponseWM

INPUT_SCHEMA='controlled_pen_objectFirst_state55_engineWarm26_actualPD20_nativeDt1:1'
OUTPUT_SCHEMA='world_left_delta52_nextWarm26_sourceBODY_wrench6_Fn6_H20:1'
HEAD_SCHEMA='pen_joint40_COM3_worlddR3_WORLD_defect6_nextWarm26_sourceBODY_wrench6_Fn6:1'


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
    def __init__(self,scales,head_scales,physics,*,width=512,coupled=True):
        if len(scales['y_mean'])!=90 or len(head_scales['head_mean'])!=90:raise ValueError('Explicit pen Fn6 schema; old87-output checkpoints incompatible')
        super().__init__(scales,head_scales,physics,width=width,coupled=coupled)
        self.net[-1]=torch.nn.Linear(width,90)
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
