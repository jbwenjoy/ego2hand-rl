"""Privileged native branch oracle; never mutates the active plant or physics."""
import numpy as np
from .compound_transition import observe55
from .compound_contacts import CompoundContacts
from .residual_control import ResidualEMAController


class NativeBranchOracle:
    def __init__(self,reader,*,max_steps=4000000):
        self.reader=reader;self.mj=reader.mj;self.model=reader.model;self.data=self.mj.MjData(self.model);self.max_steps=int(max_steps);self.steps=0;self.calls=0
        c=reader.controller;self.controller=ResidualEMAController(c.base,c.lower,c.upper,limit_rad=c.limit,beta=c.beta)
        self.contacts=CompoundContacts(self.model,reader.binding['mug_part_geom_ids'],reader.oid,reader.groups,self.model.geom('mug_task_table').id)
    def rollout(self,snapshot,action,horizon):
        if snapshot['physics']!=self.reader.physics or snapshot['dt']!=self.reader.dt or horizon<1 or snapshot['k']+horizon>100:raise ValueError('Explicit same model/dt/source and bounded horizon')
        required=horizon*self.reader.n
        if self.steps+required>self.max_steps:raise RuntimeError('Native oracle integration budget exhausted')
        d=self.data;m=self.model;mj=self.mj;mj.mj_resetData(m,d);mj.mj_setState(m,d,snapshot['integration_state'],self.reader.signature);self.controller.restore(snapshot['controller'])
        if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise ValueError('No applied force aid')
        states=[];contacts=[];controls=[];full=[]
        for _ in range(horizon):
            decision=self.controller.step(action);d.ctrl[:]=decision['actual_control20'];mj.mj_step(m,d,nstep=self.reader.n);self.steps+=self.reader.n
            if np.any(d.warning.number) or not np.isfinite(np.r_[d.qpos,d.qvel,d.qacc]).all():raise ValueError('Native branch numerical failure; no candidate reset/filtering')
            states.append(observe55(d.qpos,d.qvel,self.reader.COM));contacts.append(self.contacts.read(d));controls.append(decision['actual_control20'])
            values=np.empty(self.reader.size);mj.mj_getState(m,d,values,self.reader.signature);full.append(values)
        self.calls+=1
        return dict(states55=np.array(states),contacts=contacts,controls20=np.array(controls),native_states=np.array(full),EMA_after20=self.controller.memory.copy(),steps=required,
            label_scope='Endpoint solver-PRE sampled contacts, not interval means; full per-tick gates belong to separate active execution.')
