"""Bounded physical residual EMA, shared algebra for NumPy and Torch arrays."""
import numpy as np


def residual_ema(action,previous,negative_room,positive_room,beta=.5):
    # Pure array algebra: no scalar extraction, synchronization or hidden clamp.
    proposed=(action+abs(action))*.5*positive_room+(action-abs(action))*.5*negative_room
    return (1-beta)*previous+beta*proposed


class ResidualEMAController:
    def __init__(self,base,lower,upper,*,limit_rad=.05,beta=.5):
        self.base,self.lower,self.upper=[np.array(x,dtype=np.float64,copy=True) for x in [base,lower,upper]]
        if any(x.shape!=(20,) for x in [self.base,self.lower,self.upper]) or not np.isfinite(np.r_[self.base,self.lower,self.upper]).all() or np.any(self.base<self.lower) or np.any(self.base>self.upper):raise ValueError('Finite legal20D base and control bounds')
        if not np.isfinite(limit_rad) or limit_rad<=0 or not np.isfinite(beta) or not 0<beta<=1:raise ValueError('Positive physical residual limit and EMA beta')
        self.limit=float(limit_rad);self.beta=float(beta)
        self.negative=np.minimum(limit_rad,self.base-self.lower);self.positive=np.minimum(limit_rad,self.upper-self.base)
        self.memory=np.zeros(20);self.updates=0

    def step(self,action):
        a=np.asarray(action,dtype=np.float64)
        if a.shape!=(20,) or not np.isfinite(a).all() or np.any(abs(a)>1):raise ValueError('Raw finite bounded action20; no projection')
        before=self.memory.copy();after=residual_ema(a,before,self.negative,self.positive,self.beta);control=self.base+after
        if np.any(control<self.lower) or np.any(control>self.upper):raise ValueError('EMA generated illegal control')
        self.memory=after.copy();self.updates+=1
        return dict(action20=a.copy(),EMA_before20=before,EMA_after20=after.copy(),actual_control20=control,control_update=self.updates)

    def state_dict(self):
        return dict(schema='physical_residual_EMA20:1',base=self.base.copy(),lower=self.lower.copy(),upper=self.upper.copy(),limit_rad=self.limit,beta=self.beta,memory=self.memory.copy(),updates=self.updates)

    def restore(self,state):
        if state['schema']!='physical_residual_EMA20:1' or any(not np.array_equal(state[k],getattr(self,k)) for k in ['base','lower','upper']) or state['limit_rad']!=self.limit or state['beta']!=self.beta:raise ValueError('Controller contract changed')
        memory=np.array(state['memory'],dtype=np.float64)
        if memory.shape!=(20,) or not np.isfinite(memory).all() or np.any(memory< -self.negative) or np.any(memory>self.positive) or type(state['updates']) is not int or state['updates']<0:raise ValueError('Invalid saved EMA state')
        self.memory=memory.copy();self.updates=state['updates']
