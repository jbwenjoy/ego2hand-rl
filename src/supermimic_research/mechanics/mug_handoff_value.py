"""Candidate-specific estimate of continuous valid controls through a frozen-policy handoff."""
import torch
from torch import nn


class HandoffValue(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.register_buffer('mean',torch.zeros(width))
        self.register_buffer('scale',torch.ones(width))
        self.mlp=nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,x):
        return 6*torch.sigmoid(self.mlp(((x-self.mean)/self.scale).clamp(-6,6)).squeeze(-1))
