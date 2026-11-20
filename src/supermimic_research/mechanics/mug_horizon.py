"""Direct multi-control mug response in a common source BODY frame."""
import numpy as np
from scipy.spatial.transform import Rotation
import torch.nn as nn
from .mug_step_response import MugStepHead


def continuations(first,direct,phase,limits):
    """Hold the first correction with 0.8 decay on the next two reference targets."""
    correction=first-direct[phase]
    return np.stack([np.clip(direct[min(phase+j,len(direct)-1)]+(.8**j)*correction,*limits.T)
                     if j else first for j in range(3)],axis=1)


def features(chart,state,warm,commands):
    return np.column_stack([chart.features(state,warm,commands[:,0]),
        (commands[:,1:]-commands[:,:-1]).reshape(len(commands),-1)]).astype(np.float32)


def targets(chart,state,end,mean_load,H):
    p,R,v,w=chart.motion(state);ep,eR,_,_=chart.motion(end)
    cv=R*Rotation.from_rotvec(.02*H*w)
    return np.column_stack([R.inv().apply(ep-p-.02*H*v)/.01,(cv.inv()*eR).as_rotvec()/np.deg2rad(10),
        R.inv().apply(mean_load[:,:3])/chart.weight,R.inv().apply(mean_load[:,3:])/(chart.weight*.05)]).astype(np.float32)


def decode(chart,state,y,H):
    p,R,v,w=chart.motion(state)
    return (p+H*.02*v+R.apply(.01*y[:,:3]),R*Rotation.from_rotvec(H*.02*w)*Rotation.from_rotvec(np.deg2rad(10)*y[:,3:6]),
            np.column_stack([R.apply(chart.weight*y[:,6:9]),R.apply(chart.weight*.05*y[:,9:])]))


class MugHorizonHead(MugStepHead):
    def __init__(self,n):
        super().__init__(n)
        self.ymean=self.ymean.new_zeros(36)
        self.yscale=self.yscale.new_ones(36)
        self.net[-1]=nn.Linear(128,36)
