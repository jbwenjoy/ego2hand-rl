"""Analytic active-contact gap penalty at the commanded moving mug pose."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class ContactGapPenalty:
    def __init__(self,model,qinit,mug,hand):
        self.model=model;self.data=mujoco.MjData(model);self.qinit=qinit.copy();self.mug=mug
        self.hand=sorted(hand);self.index={g:i for i,g in enumerate(self.hand)}
        self.table=model.geom('mug_task_table').id
        self.adr=model.jnt_qposadr[model.body_jntadr[mug]]
        self.last=None

    def update(self,q,pose):
        key=q.tobytes(),pose.tobytes()
        if self.last==key:return self.values,self.jac
        d=self.data;m=self.model;d.qpos[:]=self.qinit;d.qpos[:20]=q
        quat=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
        d.qpos[self.adr:self.adr+7]=np.r_[pose[:3,3],quat]
        mujoco.mj_forward(m,d)
        size=len(self.hand)+2
        values=np.zeros(size);jac=np.zeros((size,20))
        for c in d.contact:
            g,h=map(int,c.geom);bs=m.geom_bodyid[[g,h]]
            if self.mug in bs and (g in self.index or h in self.index):
                idx=self.index[g if g in self.index else h];band=.000075
            elif g in self.index and h in self.index:idx=size-2;band=0.
            elif self.table in (g,h) and (g in self.index or h in self.index):idx=size-1;band=0.
            else:continue
            violation=(-c.dist-band)/.0001
            if violation<=values[idx]:continue
            values[idx]=violation
            J=[]
            for body in bs:
                jp=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,None,c.pos,int(body));J.append(jp[:,:20])
            jac[idx]=-(c.frame[:3]@(J[1]-J[0]))/.0001
        self.values,self.jac,self.last=values,jac,key
        return values,jac
