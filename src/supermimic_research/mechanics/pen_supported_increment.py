"""Freeze unidentifiable normalized input coefficients; preserve the measured load skip path."""
import torch
from .pen_load_increment import LoadIncrement


class SupportedIncrement(LoadIncrement):
    def __init__(self,x,y):
        super().__init__(x,y)
        self.register_buffer('learnable_input',x.amax(0)!=x.amin(0))

    def forward(self,x):
        z=(x-self.center)/self.scale
        residual=self.net(z*self.learnable_input)*self.out_scale+self.out_center
        return torch.cat((x[...,-12:]+residual[...,:12],residual[...,12:]),dim=-1)
