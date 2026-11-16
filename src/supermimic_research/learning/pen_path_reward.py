"""One-step soft-path reward proxy and actual-PD/RAW chain;no hard-risk model."""
import torch
from ..mechanics.pen_reward_chain import controller,observable_proxy

SCORES=['path_residual','path_support']


def scores(model,x,raw,context,reward,objective):
    prediction,path=model.full_prediction(x);proxy=observable_proxy(x,prediction,raw,context,reward,objective)
    return torch.stack([proxy['observable_residual']-objective['geometry_soft_weight']*path,proxy['observable_support']-objective['geometry_soft_weight']*path],-1),prediction,path


def gradients(model,x,raw,context,reward,objective):
    x=x.detach().clone().requires_grad_(True);raw=raw.detach().clone().requires_grad_(True);PD,_,_=controller(raw,context)
    diagonal=torch.autograd.grad(PD.sum(),raw,retain_graph=True)[0];values,_,_=scores(model,x,raw,context,reward,objective)
    chained=[];partials=[];direct=[]
    for k in range(2):
        gx,gr=torch.autograd.grad(values[:,k].sum(),(x,raw),retain_graph=True);partials.append(gx[:,81:101]);direct.append(gr);chained.append(gx[:,81:101]*diagonal+gr)
    return dict(values=values.detach(),RAW=torch.stack(chained,1).detach(),PD_partial=torch.stack(partials,1).detach(),RAW_direct_partial=torch.stack(direct,1).detach(),controller_diagonal=diagonal.detach(),computed_PD=PD.detach())
