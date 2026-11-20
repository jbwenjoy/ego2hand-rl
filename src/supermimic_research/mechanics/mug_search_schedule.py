"""Predict the relative benefit of the four permitted search slots in a control block."""
import torch
from torch import nn


class MugSlotHead(nn.Module):
    def __init__(self,width,observe_error):
        super().__init__()
        self.use_error=observe_error
        self.register_buffer('center',torch.zeros(width))
        self.register_buffer('scale',torch.ones(width))
        self.net=nn.Sequential(nn.Linear(width,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,4))

    def forward(self,x):
        v=((x-self.center)/self.scale).clamp(-5,5)
        if not self.use_error:
            # Nominal context 73, measured state/last action error 72, future reference 32.
            v=torch.cat([v[...,:73],torch.zeros_like(v[...,73:145]),v[...,145:]],dim=-1)
        value=self.net(v)
        return value-value.mean(dim=-1,keepdim=True)
