"""50Hz task callback for the shared original native evaluation loop.

This adapter loads no checkpoint and constructs no NN or native model. Pass a
PhoneTask-compatible task (optionally the canonical F64-history wrapper),
current geometry and an already loaded Actor callable or actor=None for zero.
The outer native evaluator independently checks that callbacks mutate no
integration state and returned absolute controls are finite and legal.
"""
from . import state_codec as canonical_state_codec
from .phone_task import PhoneTask

class TaskController:
    """Injected deterministic Actor or zero action, no native observer forwards.

    This class loads no NN. A trained Actor must be explicitly supplied later.
    Memory advances when the command is issued; next callback is at the next H.
    """
    def __init__(self,task,geometry,state_codec=canonical_state_codec,dt=None,phone=None,*,actor=None,device='cpu'):
        if dt is None or phone is None:raise ValueError('Explicit native dt and phone body ID')
        self.task=task;self.geometry=geometry;self.codec=state_codec;self.dt=dt;self.phone=phone;self.actor=actor;self.device=device;self.memory=None;self.records=[];self.NN_FW=0
    def __call__(self,tick,dt,m,d):
        import numpy as np
        import torch
        assert dt==self.dt
        x=torch.tensor(self.codec.actual_state(d.qpos,d.qvel,m.body_ipos[self.phone]),dtype=torch.float64)
        if self.memory is None:self.memory=self.task.initial_memory(x,native_h_s=dt,n_substeps=round(.02/dt))
        assert int(self.memory['step'])==tick//round(.02/dt)
        with torch.no_grad():
            obs=self.task.observation(x,self.memory,self.geometry)
            if self.actor is None:action=torch.zeros(20,dtype=torch.float32)
            else:
                action=self.actor(obs.to(device=self.device,dtype=torch.float32).unsqueeze(0)).squeeze(0).to('cpu');self.NN_FW+=1
                if action.shape!=(20,):raise ValueError('Explicit singleton Actor action20')
            residual,u=self.task.control_target(action,self.memory)
        self.records.append(dict(pre_tick=int(tick),time_s=tick*dt,source_state58=x.numpy().tolist(),memory83=self.task.flatten_memory(self.memory).numpy().tolist(),observation222_FP32=obs.numpy().tolist(),action20_FP32=action.numpy().tolist(),ctrl20_F64=u.numpy().tolist()))
        self.memory=self.task.advance_memory(self.memory,action,residual,u)
        return u.numpy()


