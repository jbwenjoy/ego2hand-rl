"""Read-only old/new comparison; identical-source CHECK pairs only for AD."""
import numpy as np
import torch

from .pen_diagnostics import decode_pen_torch,endpoint_errors
from .pen_native_response import vector_comparison


def classify_pair(xp,xm,valid,center,direction,epsilon):
    xp,xm,center,direction=[np.asarray(v,float) for v in [xp,xm,center,direction]]
    same_source=bool(np.array_equal(xp[:81],xm[:81]) and xp[-1]==xm[-1])
    error=max(abs(xp[81:101]-center-epsilon*direction).max(),abs(xm[81:101]-center+epsilon*direction).max())
    return dict(exact_source_warm=same_source,unclipped_PD=bool(error<2e-7),valid=bool(valid),
        actual_PD_offset_error_rad=float(error),eligible=bool(same_source and error<2e-7 and valid))


def common_one_step(model,data,common_scale,device):
    valid=data['valid'];x=torch.tensor(data['x'][valid],dtype=torch.float32,device=device)
    y=torch.tensor(data['y'][valid],dtype=torch.float64,device=device)
    with torch.no_grad():
        prediction=model.prediction(x)
        predicted_state=decode_pen_torch(x[:,:55].double(),prediction)
        target_state=decode_pen_torch(x[:,:55].double(),y)
    p=prediction.cpu().numpy();target=y.cpu().numpy()
    normalized={name:float(np.mean(((p[:,a:b]-target[:,a:b])/common_scale[a:b])**2)) for name,a,b in [
        ('delta52',0,52),('object12',40,52),('wrench6',78,84),('Fn6',84,90)]}
    return dict(rows=len(p),common_v1_scale_MSE=normalized,
        physical_endpoint=endpoint_errors(predicted_state.cpu().numpy(),target_state.cpu().numpy())),p


def physical_pair_derivative(model,xplus,xminus,yplus,yminus,direction,epsilon,common_scale,device):
    x=torch.tensor(((xplus+xminus)/2)[None],dtype=torch.float32,device=device)
    tangent=torch.zeros_like(x);tangent[:,81:101]=torch.tensor(direction,dtype=x.dtype,device=device)
    _,gradient=torch.autograd.functional.jvp(model.prediction,x,tangent,create_graph=False)
    ad=gradient[0].detach().cpu().numpy();fd=(yplus-yminus)/(2*epsilon)
    result={}
    for name,sl in [('object12',slice(40,52)),('wrench6',slice(78,84))]:
        result[name]=vector_comparison(ad[sl]/common_scale[sl],fd[sl]/common_scale[sl])
    return result
