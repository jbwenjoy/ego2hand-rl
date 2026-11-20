"""Local native action Jacobian about a state-indexed nominal servo target."""
import torch
from torch import nn
from .pen_substep_pose import SubstepPose, cv_qpos,features

SCHEMA='pen_substep_pose_local_secant:1'


class LocalSecantPose(nn.Module):
    def __init__(self, absolute, anchors, offsets, slopes, selected):
        super().__init__()
        self.absolute=absolute
        self.register_buffer('anchors',anchors)
        self.register_buffer('scale',anchors.std(0,unbiased=False).clamp_min(.001))
        self.register_buffer('offsets',offsets)
        self.register_buffer('slopes',slopes)
        self.register_buffer('selected',selected)

    def predict_qpos(self,state55,warmstart26,pd20):
        i=self.anchor_index(state55)
        # Center loaded fingers at nearest TRAIN nominal servo offsets. The other
        # twelve command coordinates continue through the absolute forecaster.
        pd_center=pd20.clone()
        pd_center[...,self.selected]=state55[...,self.selected]+self.offsets[i]
        base=self.absolute(features(state55,pd_center,warmstart26))
        delta=pd20[...,self.selected]-pd_center[...,self.selected]
        change=torch.einsum('...j,...jtk->...tk',delta,self.slopes[i])
        return cv_qpos(state55,base+change)

    def anchor_index(self,state):
        return (((state.unsqueeze(-2)-self.anchors)/self.scale).square().mean(-1)).argmin(-1)


def load_local_secant(checkpoint):
    if checkpoint['schema']!=SCHEMA:raise ValueError('Wrong local secant pose format')
    w=checkpoint['state_dict']
    model=LocalSecantPose(SubstepPose().double(),w['anchors'],w['offsets'],w['slopes'],w['selected'])
    model.load_state_dict(w)
    return model.eval().requires_grad_(False)
