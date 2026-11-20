"""Forecast one control interval from state, target and optional previous contact."""
import numpy as np
import torch
from torch import nn
from .transition_records import state_rotation


def rotate_load(load,rotation):
    return np.r_[rotation@load[:3],rotation@load[3:6],load[6:12]]


def features(state,pd,history):
    return np.r_[state,pd-state[:20],history] if history is not None else np.r_[state,pd-state[:20]]


class LoadForecast(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        for name,v in [('center',x.mean(0)),('scale',x.std(0,unbiased=False).clamp_min(.001)),
                       ('out_center',y.mean(0)),('out_scale',y.std(0,unbiased=False).clamp_min(.00001))]:
            self.register_buffer(name,v)
        self.net=nn.Sequential(nn.Linear(x.shape[1],128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,y.shape[1]))

    def forward(self,x):
        return self.net((x-self.center)/self.scale)*self.out_scale+self.out_center


def examples(x,y,valid,use_history):
    rows=[];labels=[]
    for i in range(1,len(x)):
        if not(valid[i-1] and valid[i]):continue
        s=x[i,:55]
        hist=rotate_load(y[i-1,78:90],state_rotation(s).T@state_rotation(x[i-1,:55]))
        rows.append(features(s,x[i,81:101],hist if use_history else None))
        labels.append(np.r_[y[i,78:90],y[i,:40]])
    return np.array(rows),np.array(labels)
