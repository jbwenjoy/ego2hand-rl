"""Reference-conditioned actuator correction with an exact zero-error fixed point."""
import numpy as np
import torch
from torch import nn
from .mug_local_feedback import tangent_errors


def observation(model,nominal,actual,cmd,previous_delta):
    dx=tangent_errors(model,nominal,actual[None])[0]
    # Relative rotational coordinates avoid sign flips in the observed quaternion.
    # The nominal trace itself has a fixed, recorded quaternion convention.
    context=np.r_[nominal[1:],cmd].astype(np.float32)
    error=np.r_[dx,previous_delta].astype(np.float32)
    return context,error


class MugResidualActor(nn.Module):
    def __init__(self,context_dim,error_dim):
        super().__init__()
        self.register_buffer('mean',torch.zeros(context_dim))
        self.register_buffer('scale',torch.ones(context_dim))
        self.register_buffer('error_scale',torch.ones(error_dim))
        self.mlp=nn.Sequential(nn.Linear(context_dim+error_dim,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,20))

    def forward(self,context,error):
        c=((context-self.mean)/self.scale).clamp(-6,6)
        e=(error/self.error_scale).clamp(-6,6)
        observed=self.mlp(torch.cat([c,e],dim=-1))
        center=self.mlp(torch.cat([c,torch.zeros_like(e)],dim=-1))
        return .12*torch.tanh(observed-center)
