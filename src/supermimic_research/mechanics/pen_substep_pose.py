"""Predict native pre-substep pen poses as residuals on a constant-velocity path."""
import torch
from torch import nn
from dex_supermimic.rl.dex_supermimic.rotations import mat_to_quat
from .impulse_codec import exp_world_unchecked, rotation55_torch

SCHEMA='pen_substep_pose_cv_residual:1'
NATIVE_DT=.0025


def features(state,pd,warm):
    if state.shape[-1]!=55 or pd.shape!=(*state.shape[:-1],20) or warm.shape!=(*state.shape[:-1],26):
        raise ValueError('Expected matched state55, held PD20 and warmstart26')
    return torch.cat([state,warm,pd-state[...,:20]],dim=-1)


def cv_qpos(state,correction=None):
    """Correction is (...,7,26) in rad/m/world-left rotvec at times 1..7."""
    dtype,device=state.dtype,state.device
    t=torch.arange(8,dtype=dtype,device=device).unsqueeze(-1)*NATIVE_DT
    hand=state[...,:20].unsqueeze(-2)+t*state[...,20:40].unsqueeze(-2)
    pos=state[...,40:43].unsqueeze(-2)+t*state[...,49:52].unsqueeze(-2)
    orientation=exp_world_unchecked(t*state[...,52:55].unsqueeze(-2))@rotation55_torch(state).unsqueeze(-3)
    if correction is not None:
        if correction.shape!=(*state.shape[:-1],7,26):
            raise ValueError('Expected seven pre-substep corrections')
        correction=torch.cat([torch.zeros_like(correction[...,:1,:]),correction],dim=-2)
        hand=hand+correction[...,:20];pos=pos+correction[...,20:23]
        orientation=exp_world_unchecked(correction[...,23:26])@orientation
    return torch.cat([pos,mat_to_quat(orientation),hand],dim=-1)


class SubstepPose(nn.Module):
    """Seven time-specific 26D residuals, with a hard zero residual at the source."""
    def __init__(self):
        super().__init__()
        self.register_buffer('center',torch.zeros(101))
        self.register_buffer('scale',torch.ones(101))
        self.register_buffer('varying',torch.ones(101,dtype=torch.bool))
        self.register_buffer('out_center',torch.zeros(7,26))
        self.register_buffer('out_scale',torch.ones(7,26))
        self.net=nn.Sequential(nn.Linear(101,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,7*26))

    @torch.no_grad()
    def normalize_from_train(self,x,y):
        self.center.copy_(x.mean(0));self.scale.copy_(x.std(0,unbiased=False).clamp_min(.001))
        self.varying.copy_(x.amax(0)!=x.amin(0))
        self.out_center.copy_(y.mean(0));self.out_scale.copy_(y.std(0,unbiased=False).clamp_min(1e-7))

    def forward(self,x):
        raw=self.net((x-self.center)/self.scale*self.varying)
        return raw.reshape(*x.shape[:-1],7,26)*self.out_scale+self.out_center

    def predict_qpos(self,state55,warmstart26,pd20):
        d=self.center.dtype
        x=features(state55.to(d),pd20.to(d),warmstart26.to(d))
        return cv_qpos(state55,self(x).to(state55.dtype))


def load_substep_pose(checkpoint_dict,dtype=None,device=None):
    if checkpoint_dict['schema']!=SCHEMA:
        raise ValueError('Unsupported substep predictor checkpoint')
    model=SubstepPose()
    model.load_state_dict(checkpoint_dict['state_dict'],strict=True)
    return model.to(dtype=dtype,device=device).eval()
