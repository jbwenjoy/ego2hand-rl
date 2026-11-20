"""Twenty-millisecond mug response model in the current mug BODY chart."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn


class MugStepChart:
    def __init__(self,m,binding):
        self.model=m;self.data=mujoco.MjData(m);self.body=binding['mug_body_id']
        self.qa=binding['mug_free_qpos'];self.da=m.jnt_dofadr[m.body_jntadr[self.body]]
        self.com=m.body_ipos[self.body].copy()
        self.weight=m.body_mass[self.body]*np.linalg.norm(m.opt.gravity)
        self.sites=[m.site(f'right_finger{k}_tip').id for k in range(1,6)]
        self.joints=m.actuator_trnid[:,0]
        self.qids=m.jnt_qposadr[self.joints];self.dofs=m.jnt_dofadr[self.joints]

    def motion(self,state):
        q=state[...,1:1+self.model.nq];v=state[...,1+self.model.nq:]
        rot=Rotation.from_quat(q[...,self.qa+np.array([4,5,6,3])])
        lever=rot.apply(self.com)
        wb=v[...,self.da+3:self.da+6];omega=rot.apply(wb)
        pos=q[...,self.qa:self.qa+3]+lever
        vel=v[...,self.da:self.da+3]+np.cross(omega,lever)
        return pos,rot,vel,wb

    def features(self,state,warm,commands):
        m=self.model;d=self.data
        pos,rot,v,wb=self.motion(state)
        d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:]
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        jac=[]
        for site in self.sites:
            jp=np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jp,None,site)
            jac.append(rot.as_matrix().T@jp[:,self.dofs])
        jac=np.concatenate(jac)
        site=rot.inv().apply(d.site_xpos[self.sites]-pos)
        site_vel=(jac@d.qvel[self.dofs]).reshape(5,3)-rot.inv().apply(v)-np.cross(wb,site)
        joint=d.qpos[self.qids]
        context=np.r_[joint,d.qvel[self.dofs],pos,rot.as_matrix()[:,:2].T.ravel(),rot.inv().apply(v),wb,
                      site.ravel(),site_vel.ravel(),warm]
        delta=commands-joint
        return np.column_stack([np.broadcast_to(context,(len(commands),len(context))),delta,delta@jac.T]).astype(np.float32)

    def targets(self,state,end,wrench):
        pos,R,v,wb=self.motion(state);ep,eR,_,_=self.motion(end)
        cv=R*Rotation.from_rotvec(.02*wb)
        y=np.column_stack([R.inv().apply(ep-pos-.02*v)/.01,(cv.inv()*eR).as_rotvec()/np.deg2rad(10),
            R.inv().apply(wrench[:,:3])/self.weight,R.inv().apply(wrench[:,3:])/(self.weight*.05)])
        return y.astype(np.float32)

    def decode(self,state,y):
        pos,R,v,wb=self.motion(state)
        ep=pos+.02*v+R.apply(.01*y[:,:3])
        eR=R*Rotation.from_rotvec(.02*wb)*Rotation.from_rotvec(np.deg2rad(10)*y[:,3:6])
        return ep,eR,np.column_stack([R.apply(y[:,6:9]*self.weight),R.apply(y[:,9:]*self.weight*.05)])


class MugStepHead(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.register_buffer('mean',torch.zeros(n))
        self.register_buffer('scale',torch.ones(n))
        self.register_buffer('ymean',torch.zeros(12))
        self.register_buffer('yscale',torch.ones(12))
        self.net=nn.Sequential(nn.Linear(n,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,12))

    def forward(self,x):
        return self.net(((x-self.mean)/self.scale).clamp(-8,8))*self.yscale+self.ymean
