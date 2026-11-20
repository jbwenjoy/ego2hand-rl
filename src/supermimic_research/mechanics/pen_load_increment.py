"""Predict an additive change to measured previous BODY load; keep absolute-target scaling."""
import torch
from .pen_load_forecast import LoadForecast


class LoadIncrement(LoadForecast):
    def __init__(self,x,y):
        if x.shape[-1]!=87 or y.shape[-1]!=52:
            raise ValueError('Expected 75 state/control and 12 past load inputs, 52 output quantities')
        super().__init__(x,y)
        self.out_center[:12].sub_(x[:,-12:].mean(0))

    def forward(self,x):
        residual=super().forward(x)
        return torch.cat([residual[...,:12]+x[...,-12:],residual[...,12:]],dim=-1)
