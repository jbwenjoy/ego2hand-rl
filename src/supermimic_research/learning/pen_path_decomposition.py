"""Read-only within-model separation of observable and learned path scores."""
import torch
from ..mechanics.pen_reward_chain import controller,observable_proxy

MODES=['OBSERVABLE','PATH_ONLY','FULL']


def values(model,x,raw,context,reward,objective):
    prediction,path=model.full_prediction(x);proxy=observable_proxy(x,prediction,raw,context,reward,objective)
    observable=torch.stack([proxy['observable_residual'],proxy['observable_support']],-1)
    geometry=(-objective['geometry_soft_weight']*path)[:,None].expand(-1,2)
    return torch.stack([observable,geometry,observable+geometry],1)


def gradients(model,x,raw,context,reward,objective):
    x=x.detach().clone().requires_grad_(True);raw=raw.detach().clone().requires_grad_(True);PD,_,_=controller(raw,context)
    diagonal=torch.autograd.grad(PD.sum(),raw,retain_graph=True)[0];v=values(model,x,raw,context,reward,objective)
    chained=[];partials=[];direct=[]
    for mode in range(3):
        cg=[];pg=[];dg=[]
        for k in range(2):
            gx,gr=torch.autograd.grad(v[:,mode,k].sum(),(x,raw),retain_graph=True,allow_unused=True)
            if gx is None:gx=torch.zeros_like(x)
            if gr is None:gr=torch.zeros_like(raw)
            pg.append(gx[:,81:101]);dg.append(gr);cg.append(gx[:,81:101]*diagonal+gr)
        chained.append(torch.stack(cg,1));partials.append(torch.stack(pg,1));direct.append(torch.stack(dg,1))
    return dict(values=v.detach(),RAW=torch.stack(chained,1).detach(),PD_partial=torch.stack(partials,1).detach(),RAW_direct_partial=torch.stack(direct,1).detach(),controller_diagonal=diagonal.detach(),computed_PD=PD.detach())
