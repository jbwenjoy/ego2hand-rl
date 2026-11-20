"""Hand contact wrench about a free mug's current COM, in world axes."""
import mujoco
import numpy as np


class MugContactImpulse:
    def __init__(self, model, binding):
        self.body=binding['mug_body_id']
        joint=model.body_jntadr[self.body]
        if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE or model.neq or model.body_gravcomp[self.body]:
            raise ValueError('Expected an unconstrained free mug')
        self.dof=model.jnt_dofadr[joint]
        self.is_mug=model.geom_bodyid==self.body
        self.is_hand=np.isin(np.arange(model.ngeom),[model.geom(n).id for n in binding['hand_collision_names']])
        self.table=model.geom('mug_task_table').id
        self.local=np.empty(6)

    def hand_wrench(self,model,data,*,verify=False):
        # Read the contact solve immediately after mj_step, before refreshing FK.
        force=data.qfrc_constraint[self.dof:self.dof+3]
        rot=data.xmat[self.body].reshape(3,3)
        torque=rot@data.qfrc_constraint[self.dof+3:self.dof+6]-np.cross(data.xipos[self.body]-data.xpos[self.body],force)
        wrench=np.r_[force,torque]
        a,b=data.contact.geom.T
        involved=self.is_mug[a]^self.is_mug[b]
        hand=involved & (self.is_hand[a]|self.is_hand[b])
        indices=np.flatnonzero(involved if verify else involved & ~hand)
        directly_summed=np.zeros(6)
        peak_table=0.
        for i in indices:
            c=data.contact[i]
            mujoco.mj_contactForce(model,data,i,self.local)
            sign=1. if self.is_mug[b[i]] else -1.
            basis=c.frame.reshape(3,3).T
            f=sign*(basis@self.local[:3])
            moment=sign*(basis@self.local[3:])+np.cross(c.pos-data.xipos[self.body],f)
            if hand[i]:
                directly_summed+=np.r_[f,moment]
            else:
                wrench-=np.r_[f,moment]
                if self.table in (a[i],b[i]):peak_table=max(peak_table,self.local[0])
        error=abs(directly_summed-wrench).max() if verify else None
        return wrench,peak_table,error
