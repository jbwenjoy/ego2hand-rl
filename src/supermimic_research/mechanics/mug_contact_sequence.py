"""Track grasp material points on both bodies alongside relative human fingertip motion."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from .retargeting import HandTipIK


def solve_contact_sequence(model,qinit,targets,poses,mug,hand_ids):
    base=HandTipIK(model,qinit,targets[0],regularization=0.)
    d=base.data;q0=qinit[:20].copy()
    d.qpos[:]=qinit;mujoco.mj_forward(model,d)
    binds=[]
    for c in d.contact:
        g,h=map(int,c.geom);on=model.geom_bodyid[[g,h]]==mug
        if c.dist>0 or on.sum()!=1:continue
        other=h if on[0] else g
        if other not in hand_ids:continue
        body=int(model.geom_bodyid[other]);local=lambda b:d.xmat[b].reshape(3,3).T@(c.pos-d.xpos[b])
        binds.append((body,local(body),local(mug),model.geom(other).name))
    if not binds:raise ValueError('No robot/mug contact point at start')
    def points():
        return np.stack([d.xpos[b]+d.xmat[b].reshape(3,3)@local for b,local,_,_ in binds])
    mug_points=np.stack([p for _,_,p,_ in binds])
    weights=np.array([1/np.sqrt(sum(row[0]==b for row in binds)) for b,_,_,_ in binds])
    seq=[q0];tip=[base.state(q0)];errors=[np.zeros(len(binds))];ok=[True]
    for i in range(1,len(poses)):
        previous=seq[-1];pose=poses[i];goal=mug_points@pose[:3,:3].T+pose[:3,3]
        base.targets=targets[i]
        def residual(q):
            return np.r_[base.residual(q)[:15],((points()-goal)*weights[:,None]/.0003).flatten(),.05*(q-q0),.1*(q-previous)]
        def jac(q):
            J=base.jacobian(q)[:15];blocks=[]
            for point,(body,_,_,_),w in zip(points(),binds,weights):
                jp=np.zeros((3,model.nv));mujoco.mj_jac(model,d,jp,None,point,body)
                blocks.append(jp[:,:20]*w/.0003)
            return np.vstack([J,*blocks,.05*np.eye(20),.1*np.eye(20)])
        result=least_squares(residual,previous,jac=jac,bounds=(base.lo,base.hi),max_nfev=100)
        q=result.x.copy();seq.append(q);tip.append(base.state(q));errors.append(np.linalg.norm(points()-goal,axis=1));ok.append(bool(result.success))
    return np.array(seq),np.array(tip),ok,np.array(errors),[r[3] for r in binds]
