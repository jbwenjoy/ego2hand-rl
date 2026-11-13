"""New explicit impulse-head representation with unchanged external102->87 chart."""
import numpy as np
import torch
from .compound_response import CompoundResponseWM
from ..mechanics.impulse_codec import close_velocities

HEAD_SCHEMA='compound_joint40_COM3_worlddR3_WORLD_impulse_defect6_warm26_sourceBODY_wrench6_Fn3:1'


def register_physics(module,physics):
    for name,value in [('mass',physics['mass']),('inertia_BODY',physics['inertia_BODY']),('inverse_inertia_BODY',np.linalg.inv(physics['inertia_BODY'])),('gravity',physics['gravity'])]:
        module.register_buffer(name,torch.as_tensor(value,dtype=torch.float64).clone())


class ImpulseView(torch.nn.Module):
    """Read-only zero-defect reinterpretation of a frozen direct-velocity model."""
    def __init__(self,base,physics):
        super().__init__();self.base=base;register_physics(self,physics)
    @property
    def y_scale(self):return self.base.y_scale
    @property
    def y_mean(self):return self.base.y_mean
    def prediction(self,x):return close_velocities(x,self.base.prediction(x),self.mass,self.inertia_BODY,self.inverse_inertia_BODY,self.gravity)


class ImpulseResponseWM(CompoundResponseWM):
    def __init__(self,scales,head_scales,physics,*,width=512,coupled=True):
        super().__init__(scales,width=width);self.coupled=bool(coupled)
        for name in ['head_mean','head_scale']:self.register_buffer(name,torch.as_tensor(head_scales[name],dtype=torch.float32).clone())
        register_physics(self,physics)
    def normalized_heads(self,x):
        raw=self.net((x-self.x_mean)/self.x_scale)
        return torch.cat([raw[...,:84],torch.nn.functional.softplus(raw[...,84:])],dim=-1)
    def physical_heads(self,x):return self.normalized_heads(x)*self.head_scale+self.head_mean
    def prediction(self,x):
        h=self.physical_heads(x)
        return close_velocities(x,h,self.mass,self.inertia_BODY,self.inverse_inertia_BODY,self.gravity,h[...,46:52].double()) if self.coupled else h.double()
    def forward(self,x):return (self.prediction(x)-self.y_mean)/self.y_scale
    def supervised_loss(self,x,y,head_target,*,auxiliary_weight=.1):
        h=self.physical_heads(x);p=close_velocities(x,h,self.mass,self.inertia_BODY,self.inverse_inertia_BODY,self.gravity,h[...,46:52].double()) if self.coupled else h.double()
        error=((p-y)/self.y_scale).square()
        absolute=error[...,:52].mean()+.1*error[...,52:78].mean()+.2*error[...,78:84].mean()+.1*error[...,84:].mean()
        auxiliary=((h[...,46:52]-head_target[...,46:52])/self.head_scale[46:52]).square().mean()
        return absolute+auxiliary_weight*auxiliary,torch.stack([absolute.detach(),auxiliary.detach()])
