"""Finite-amplitude response supervision with separate point/pair RNG streams."""
import torch

from .impulse_stage import update_impulse
from .pen_stage import restore_arm


def finite_contrast_loss(plus,minus,target,epsilon,scale):
    predicted=(plus-minus)/(2*epsilon[:,None])
    error=((predicted-target)/scale).square()
    return error[:,40:52].mean()+error[:,78:84].mean()


def update_contrast(arm,cache,cfg,*,capture_gradient=False):
    if arm['contrast_weight']==0:
        values,point_ids,gradient=update_impulse(arm,cache,cfg,capture_gradient=capture_gradient)
        return torch.cat([values,values.new_zeros(1)]),point_ids,None,gradient
    model,opt=arm['model'],arm['optimizer'];device=cache['x'].device
    point_ids=torch.randint(len(cache['x']),(cfg['batch_size'],),generator=arm['sampler'],device=device)
    pair_ids=torch.randint(len(cache['x_plus']),(cfg['pair_batch_size'],),generator=arm['pair_sampler'],device=device)
    opt.zero_grad(set_to_none=True)
    point_loss,terms=model.supervised_loss(cache['x'][point_ids],cache['y'][point_ids],cache[arm['mode']][point_ids],auxiliary_weight=cfg['auxiliary_weight'])
    inputs=torch.cat([cache['x_plus'][pair_ids],cache['x_minus'][pair_ids]])
    predictions=model.prediction(inputs);plus,minus=predictions.chunk(2)
    response=finite_contrast_loss(plus,minus,cache['finite_contrast'][pair_ids],cache['epsilon'][pair_ids],model.y_scale)
    loss=point_loss+arm['contrast_weight']*response;loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=False)
    gradient=dict(post_clip={k:p.grad.detach().clone() for k,p in model.named_parameters()},clip_norm=norm.detach().clone()) if capture_gradient else None
    opt.step()
    return torch.cat([loss.detach()[None],terms,response.detach()[None]]),point_ids,pair_ids,gradient


def restore_contrast_arm(saved,device):
    arm=restore_arm(saved,device)
    arm['pair_sampler']=torch.Generator(device=device);arm['pair_sampler'].set_state(saved['pair_sampler'])
    arm['contrast_weight']=saved['contrast_weight']
    return arm
