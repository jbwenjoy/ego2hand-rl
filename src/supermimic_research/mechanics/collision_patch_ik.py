"""Jointly penalize collision and source patch mismatch using shared primitives."""
import time
import numpy as np
from scipy.optimize import least_squares
from .contact_ik import ContactAwareIK,ProposalLimit
from .surface_patch_ik import SurfacePatchIK


class CollisionPatchIK:
    def __init__(self,model,qpos,audit,anchors,points_world,inward_world,source_groups,known):
        import mujoco
        data=mujoco.MjData(model);data.qpos[:]=qpos;mujoco.mj_kinematics(model,data)
        targets=np.array([data.site_xpos[model.site(f'right_finger{i}_tip').id] for i in range(1,6)])
        self.collision=ContactAwareIK(model,qpos,targets,audit)
        self.patch=[]
        for group,anchor in enumerate(anchors,1):
            ids=np.flatnonzero(known&(source_groups==group));indices=np.arange((group-1)*4,group*4)
            self.patch.append((indices,SurfacePatchIK(model,qpos,indices,indices,anchor,points_world[ids],inward_world[ids],ids)))
        self.start=np.asarray(qpos[:20]).copy();self.best=None;self.log=None
        self.cached=None;self.maximum_seconds=float('inf');self.tick=time.perf_counter();self.calls=0

    def evaluate(self,q):
        if self.cached is not None and q.tobytes()==self.cached[0]:return self.cached[1:]
        if time.perf_counter()-self.tick>self.maximum_seconds:raise ProposalLimit('Declared coupled proposal wall limit')
        c,j=self.collision.evaluate(q)
        # Discard the generic IK's tip-fit and regularization rows; preserve its
        # object/self/table and intent-distance penalties without copying them.
        residual=[c[35:]];rows=[j[35:]];patches=[]
        for indices,problem in self.patch:
            r,J,record=problem.evaluate(q[indices]);expanded=np.zeros((len(r),20));expanded[:,indices]=J
            residual.append(r);rows.append(expanded);patches.append(record)
        residual.append(.005*(q-self.start));rows.append(.005*np.eye(20))
        r,J=np.concatenate(residual),np.concatenate(rows);cost=float(r@r);self.calls+=1
        record=dict(q20=q.tolist(),squared_cost=cost,patches=patches,
            native_collision_squared_cost=float(c[35:]@c[35:]),elapsed_s=time.perf_counter()-self.tick)
        if self.log is not None:
            import json
            self.log.write(json.dumps(record,default=lambda x:x.tolist())+'\n');self.log.flush()
        if self.best is None or cost<self.best['squared_cost']:self.best=record
        self.cached=(q.tobytes(),r,J);return r,J

    def solve(self,log,*,maximum_seconds=30.,max_nfev=160):
        self.log=log;self.maximum_seconds=maximum_seconds;self.tick=time.perf_counter();self.collision.tick=self.tick
        try:
            answer=least_squares(lambda q:self.evaluate(q)[0],self.start,jac=lambda q:self.evaluate(q)[1],
                bounds=(self.collision.lo,self.collision.hi),max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-8,x_scale='jac')
            success,message=bool(answer.success),str(answer.message)
        except ProposalLimit as error:success,message=False,str(error)
        if self.best is None:raise RuntimeError('No complete coupled proposal')
        return dict(best=self.best,solver_success=success,solver_message=message,coupled_calls=self.calls,
            native_collision_counts=self.collision.counts,patch_kinematics_calls=sum(p.calls for _,p in self.patch),
            scope='Piecewise selected source material-anchor and native collision penalties. Full20q legal bounds, no root/object change. Solver status is not independent clearance, source contact truth or physical holding.')
