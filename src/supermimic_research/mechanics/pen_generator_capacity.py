"""Capacity predictions as the support function of twelve learned 6D points and zero."""
from itertools import combinations, product

import torch

from .pen_capacity_forecast import CapacityHead
from .pen_wrench_capacity import balanced_directions


GENERATOR_SCHEMA = 'pen_body72_generator_capacity:1'


def generator_support(generators, directions):
    """Support of a convex hull in arbitrary BODY directions."""
    if generators.shape[-2:] != (12,6) or directions.ndim != 2 or directions.shape[1] != 6:
        raise ValueError('Expected twelve BODY wrench generators and (N,6) directions')
    projections = generators @ directions.T
    # Include zero explicitly. Max is differentiable almost everywhere, with
    # a subgradient at a tie between generators or with the origin.
    return torch.cat((torch.zeros_like(projections[..., :1, :]),projections),dim=-2).amax(dim=-2)


class GeneratorCapacityHead(CapacityHead):
    def __init__(self):
        super().__init__(use_history=False)
        self.register_buffer('directions',torch.from_numpy(balanced_directions()).float())
        self.register_buffer('generator_center',torch.zeros(12,6))
        self.register_buffer('generator_scale',torch.ones(6))

    @torch.no_grad()
    def set_training_stats(self,x,y):
        super().set_training_stats(x,y)
        # Initialize distinct +/- axis generators using TRAIN axis capacities.
        # Signed pair probes are included in the loss along with the axes.
        self.generator_center.copy_(self.directions[:12]*y[:,:12].mean(0)[:,None])
        self.generator_scale.copy_((self.target_scale[:6]+self.target_scale[6:12])*.5)

    def generators(self,x):
        if x.shape[-1:] != (75,):
            raise ValueError('Expected state55 plus relative PD20')
        normalized=((x-self.center)/self.scale)*self.varying
        return self.net(normalized).unflatten(-1,(12,6))*self.generator_scale+self.generator_center

    def forward_features(self,x):
        return generator_support(self.generators(x),self.directions)


def load_generator_capacity_head(checkpoint,dtype=None,device=None):
    if checkpoint['schema']!=GENERATOR_SCHEMA or checkpoint['use_history'] is not False:
        raise ValueError('Unsupported generator-capacity checkpoint')
    net=GeneratorCapacityHead()
    net.load_state_dict(checkpoint['state_dict'],strict=True)
    return net.to(dtype=dtype,device=device).eval()


def pair_bound_violations(h):
    """Excesses of upper and lower subadditivity bounds, in unit-direction units.

    For signed axes u,v the pair value is h((u+v)/sqrt(2)). Subadditivity
    bounds it above by (h(u)+h(v))/sqrt(2), and below by each of
    (h(u)-h(-v))/sqrt(2), (h(v)-h(-u))/sqrt(2), and zero.
    """
    if h.shape[-1:]!=(72,):
        raise ValueError('Expected the balanced 72-probe support ordering')
    ip=[];jp=[]
    for i,j in combinations(range(6),2):
        for si,sj in product((-1,1),repeat=2):
            ip.append(i+(6 if si<0 else 0))
            jp.append(j+(6 if sj<0 else 0))
    ni=[(p+6)%12 for p in ip];nj=[(p+6)%12 for p in jp]
    upper=(h[...,ip]+h[...,jp])*(2**-.5)
    lower=torch.maximum(h[...,ip]-h[...,nj],h[...,jp]-h[...,ni]).clamp_min(0)*(2**-.5)
    return (h[...,12:]-upper).clamp_min(0),(lower-h[...,12:]).clamp_min(0)
