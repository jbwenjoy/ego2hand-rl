"""Short-horizon head for simulated peak pen and self penetration in millimetres."""
import torch
from torch import nn


def inputs(state55, warm26, boundary_mm, pd20):
    return torch.cat((state55, warm26, boundary_mm, pd20 - state55[..., :20]), dim=-1)


class ActionDepthHead(nn.Module):
    def __init__(self, action_conditioned=True):
        super().__init__()
        self.action_conditioned = action_conditioned
        self.register_buffer('center', torch.zeros(109))
        self.register_buffer('scale', torch.ones(109))
        self.register_buffer('gates', torch.tensor([2., .2]))
        self.net = nn.Sequential(nn.Linear(109,96), nn.SiLU(), nn.Linear(96,96), nn.SiLU(),
                                 nn.Linear(96,2), nn.Softplus())

    def forward(self, x):
        z = ((x - self.center) / self.scale).clamp(-8,8)
        if not self.action_conditioned:
            z = torch.cat((z[..., :-20], torch.zeros_like(z[..., -20:])), dim=-1)
        return self.net(z)

    def predict_mm(self, x):
        return torch.expm1(self(x)) * self.gates
