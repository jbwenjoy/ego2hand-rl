"""Matched response models with an observed interval soft-geometry head.

The scalar head predicts the recorded geometry LOSS,not a penetration distance,
endpoint surrogate,hard-failure probability or simulator error flag.
"""
import numpy as np
import torch
from .pen_response import PenResponseWM
from .pen_contrast_update import finite_contrast_loss

ARMS=['RESPONSE_ONLY','PATH_POINT','PATH_POINT_RESPONSE']
PATH_SCHEMA='H20_interval_log1p_squared_geometry_ratio_sum_nonnegative:1'
WEIGHTS={'RESPONSE_ONLY':(0.,0.),'PATH_POINT':(1.,0.),'PATH_POINT_RESPONSE':(1.,1.)}


def fit_path_units(labels,mask,groups,*,role):
    labels=np.asarray(labels,dtype=np.float64);mask=np.asarray(mask)
    if role!='TRAIN_CONTROLLED_PHYSICAL_RESPONSE' or labels.ndim!=1 or mask.dtype!=np.bool_ or mask.shape!=labels.shape or mask.sum()<2 or not np.isfinite(labels).all() or (labels<0).any():
        raise ValueError('Finite nonnegative aligned TRAIN geometry losses and original mask required')
    point=labels[mask];contrasts=[];indices=[]
    for gi,g in enumerate(groups):
        p,m=g['rows']['1'],g['rows']['-1'];complete=len(p)==len(m)==2;eligible=bool(complete and mask[p+m].all())
        if g['complete']!=complete or g['qualified']!=eligible:raise ValueError('Original complete physical pairs required')
        if eligible:contrasts.append((labels[p].mean()-labels[m].mean())/(2*g['epsilon']));indices.append(gi)
    if len(indices)<2 or point.mean()<=0:raise ValueError('Observed nonzero TRAIN geometry and eligible pairs required')
    point_scale=max(float(point.std()),1e-6);rms=float(np.sqrt(np.mean(np.square(contrasts))))
    return dict(output_scale=float(point.mean()/np.log(2)),point_scale=point_scale,response_RMS=rms,
        response_floor=.001*point_scale,response_scale=max(rms,.001*point_scale),TRAIN_group_indices=np.asarray(indices,np.int64))


class PathResponseWM(PenResponseWM):
    def __init__(self,scales,physics,units,*,width=512):
        super().__init__(scales,dict(head_mean=scales['y_mean'],head_scale=scales['y_scale']),physics,width=width,coupled=False)
        self.path_head=torch.nn.Linear(width,1)
        torch.nn.init.zeros_(self.path_head.weight);torch.nn.init.zeros_(self.path_head.bias)
        self.register_buffer('path_output_scale',torch.tensor(units['output_scale'],dtype=torch.float64))
        self.register_buffer('path_point_scale',torch.tensor(units['point_scale'],dtype=torch.float64))
        self.register_buffer('path_response_scale',torch.tensor(units['response_scale'],dtype=torch.float64))

    def full_prediction(self,x):
        h=(x-self.x_mean)/self.x_scale
        for layer in list(self.net.children())[:-1]:h=layer(h)
        raw=self.net[-1](h);heads=torch.cat([raw[...,:84],torch.nn.functional.softplus(raw[...,84:])],-1)*self.head_scale+self.head_mean
        path=torch.nn.functional.softplus(self.path_head(h)[...,0]).double()*self.path_output_scale
        return heads.double(),path

    def prediction(self,x):return self.full_prediction(x)[0]

    def point_terms(self,x,y,path_target,auxiliary_weight):
        p,path=self.full_prediction(x);error=((p-y)/self.y_scale).square()
        absolute=error[...,:52].mean()+.1*error[...,52:78].mean()+.2*error[...,78:84].mean()+.1*error[...,84:90].mean()
        # Direct head46:52 is the external target as in the original DIRECT arm.
        auxiliary=((p[...,46:52]-y[...,46:52])/self.head_scale[46:52]).square().mean()
        path_loss=((path-path_target)/self.path_point_scale).square().mean()
        return absolute+auxiliary_weight*auxiliary,path_loss


def update_path(arm,cache,cfg,*,capture_gradient=False):
    if arm['contract']!=cache['contract']:raise ValueError('Exact path/data/units identity required before sampling')
    model=arm['model'];device=cache['x'].device
    point_ids=torch.randint(len(cache['x']),(cfg['batch_size'],),device=device,generator=arm['sampler'])
    pair_ids=torch.randint(len(cache['x_plus']),(cfg['pair_batch_size'],),device=device,generator=arm['pair_sampler'])
    arm['optimizer'].zero_grad(set_to_none=True)
    point,path_point=model.point_terms(cache['x'][point_ids],cache['y'][point_ids],cache['path'][point_ids],cfg['auxiliary_weight'])
    pred,path=model.full_prediction(torch.cat([cache['x_plus'][pair_ids],cache['x_minus'][pair_ids]]));plus,minus=pred.chunk(2);pp,pm=path.chunk(2)
    response=finite_contrast_loss(plus,minus,cache['finite_contrast'][pair_ids],cache['epsilon'][pair_ids],cache['response_scale'])
    path_response=(((pp-pm)/(2*cache['epsilon'][pair_ids])-cache['path_contrast'][pair_ids])/model.path_response_scale).square().mean()
    a,b=WEIGHTS[arm['name']];loss=point+response+a*path_point+b*path_response
    # All three arms compute both terms,including explicit zero weights. The
    # unused head gets zero gradients/owned Adam moments,not an absent cursor.
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=False)
    gradient=dict(post_clip={k:p.grad.detach().clone() for k,p in model.named_parameters()},clip_norm=norm.detach().clone()) if capture_gradient else None
    arm['optimizer'].step()
    return torch.stack([loss.detach(),point.detach(),response.detach(),path_point.detach(),path_response.detach()]),point_ids,pair_ids,gradient
