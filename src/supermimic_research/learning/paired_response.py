"""Matched compound-response updates with optional actual-control secant supervision."""
import numpy as np
import torch
from .response_diagnostics import BLOCKS,vector_comparison


def secant_loss(predicted_minus,predicted_plus,target_minus,target_plus,du,reference_rad=.025):
    # Inputs/targets are normalized87. Denominator is actual u-plus minus u-minus,
    # validated positive on CPU once, before caching. No epsilon/clamp/kink fiction.
    error=((predicted_plus-predicted_minus)-(target_plus-target_minus))*(reference_rad/du[:,None])
    square=error.square()
    return .5*square[:,:40].mean()+.5*square[:,40:52].mean()+.1*square[:,52:78].mean()+.2*square[:,78:84].mean()+.1*square[:,84:].mean()


def paired_cache(raw,model,device):
    p=raw['pairs'];lo,hi=p[:,:2].astype(np.int64).T
    return dict(x=torch.as_tensor(raw['x'],dtype=torch.float32,device=device),y=torch.as_tensor(raw['y'],dtype=torch.float32,device=device),
        low=torch.as_tensor(raw['x'][lo],dtype=torch.float32,device=device),high=torch.as_tensor(raw['x'][hi],dtype=torch.float32,device=device),
        ylow=(torch.as_tensor(raw['y'][lo],dtype=torch.float32,device=device)-model.y_mean)/model.y_scale,
        yhigh=(torch.as_tensor(raw['y'][hi],dtype=torch.float32,device=device)-model.y_mean)/model.y_scale,
        du=torch.as_tensor(p[:,3],dtype=torch.float32,device=device))


def paired_update(model,optimizer,sampler,absolute,paired,config,*,coefficient,capture_gradient=False):
    batch=torch.randint(len(absolute['x']),(config['batch_size'],),generator=sampler,device=absolute['x'].device)
    pair_batch=torch.randint(len(paired['du']),(config['pair_batch_size'],),generator=sampler,device=absolute['x'].device)
    optimizer.zero_grad(set_to_none=True);absolute_loss=model.loss(absolute['x'][batch],absolute['y'][batch])
    predictions=model(torch.cat([paired['low'][pair_batch],paired['high'][pair_batch]]));lo,hi=predictions.chunk(2)
    response_loss=secant_loss(lo,hi,paired['ylow'][pair_batch],paired['yhigh'][pair_batch],paired['du'][pair_batch],config['response_reference_rad'])
    loss=absolute_loss+coefficient*response_loss;loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=False)
    gradients=dict(post_clip={k:p.grad.detach().clone() for k,p in model.named_parameters()},clip_norm=norm.detach().clone()) if capture_gradient else None
    optimizer.step();return torch.stack([loss.detach(),absolute_loss.detach(),response_loss.detach()]),batch,pair_batch,gradients


def intervention_metrics(model,raw,device):
    pairs=raw['pairs'];lo,hi=pairs[:,:2].astype(np.int64).T;du=pairs[:,3];joint=pairs[:,2].astype(np.int64)
    center=(raw['x'][lo]+raw['x'][hi])*.5;direction=np.zeros_like(center);direction[np.arange(len(center)),81+joint]=.025
    with torch.no_grad():
        _,ad=torch.func.jvp(model.prediction,(torch.as_tensor(center,dtype=torch.float32,device=device),),(torch.as_tensor(direction,dtype=torch.float32,device=device),))
        prediction=model.prediction(torch.as_tensor(raw['x'],dtype=torch.float32,device=device)).cpu().numpy()
    scale=model.y_scale.detach().cpu().numpy();native=(raw['y'][hi]-raw['y'][lo])*(.025/du[:,None])/scale
    model_secant=(prediction[hi]-prediction[lo])*(.025/du[:,None])/scale;ad=ad.cpu().numpy()/scale
    amplitudes=raw['amplitude'][hi];metrics={}
    for amp in sorted(set(amplitudes.tolist())):
        selected=amplitudes==amp
        metrics[str(amp)]={name:dict(AD=vector_comparison(ad[selected,a:b],native[selected,a:b]),secant=vector_comparison(model_secant[selected,a:b],native[selected,a:b])) for name,(a,b) in BLOCKS.items()}
    return dict(pairs=len(pairs),by_amplitude=metrics,
        derivative='dy/du times.025rad; model AD at midpoint of ACTUAL executed controls. Includes asymmetric action-to-control room correctly; not da derivative at a kink.',
        scope='Finite contact/solver secants at distinct prefix states; not smooth native Jacobian, deployment or multi-step validity.'),dict(prediction=prediction,model_AD_normalized=ad,model_secant_normalized=model_secant,native_secant_normalized=native)
