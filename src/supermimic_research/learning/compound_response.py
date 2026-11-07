"""New compound H20 response baseline; incompatible with frozen phone schemas.

Only explicit TRAIN arrays fit frozen scales. Hot forwards contain no geometry,
normalizer update, native call, host scalar check or device transfer.
"""
import numpy as np
import torch
from torch import nn

INPUT_SCHEMA='compound_state55_warm26_actualU20_nativeDt1:1'
OUTPUT_SCHEMA='world_left_delta52_nextwarm26_meanwrench_source_BODY6_Fn3:1'


def fit_scales(train_x,train_y):
    x,y=np.asarray(train_x,float),np.asarray(train_y,float)
    if x.ndim!=2 or x.shape[1]!=102 or y.shape!=(len(x),87) or len(x)<2 or not np.isfinite(np.r_[x.ravel(),y.ravel()]).all() or np.any(y[:,-3:]<0):raise ValueError('Finite explicitly supplied TRAIN matrices102/87 and nonnegative native Fn')
    x_floor=np.r_[np.full(20,.01),np.full(20,.05),np.full(3,.005),np.full(6,.05),np.full(3,.05),np.full(3,.2),np.ones(26),np.full(20,.02),.0001]
    y_floor=np.r_[np.full(20,.001),np.full(20,.05),np.full(3,.001),np.full(3,.01),np.full(3,.02),np.full(3,.1),np.ones(26),np.full(3,.1),np.full(3,.005),np.full(3,.1)]
    xm=x.mean(0);xs=np.maximum(x.std(0),x_floor);ym=y.mean(0);ys=np.maximum(y.std(0),y_floor);ym[-3:]=0
    return dict(x_mean=xm,x_scale=xs,y_mean=ym,y_scale=ys)


class CompoundResponseWM(nn.Module):
    def __init__(self,scales,*,width=512):
        super().__init__()
        for key,value in scales.items():self.register_buffer(key,torch.as_tensor(value,dtype=torch.float32).clone())
        self.net=nn.Sequential(nn.Linear(102,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,87))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        raw=self.net((x-self.x_mean)/self.x_scale)
        return torch.cat([raw[...,:84],torch.nn.functional.softplus(raw[...,84:])],dim=-1)

    def prediction(self,x):return self(x)*self.y_scale+self.y_mean

    def loss(self,x,target):
        error=(self(x)-(target-self.y_mean)/self.y_scale).square()
        return error[...,:52].mean()+.1*error[...,52:78].mean()+.2*error[...,78:84].mean()+.1*error[...,84:].mean()
