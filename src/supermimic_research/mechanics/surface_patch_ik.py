"""Bounded material-point/normal proposals, with no contact or force claim.

An anchor is a fixed point and outward normal on one rigid hand collision geom.
Matching it to an estimated source patch does not establish native contact.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares


def material_anchor(model,data,geom,world_point,world_normal_hint):
    """Recover a true hull/capsule anchor near a native distance endpoint."""
    R=data.geom_xmat[geom].reshape(3,3);p=R.T@(np.asarray(world_point)-data.geom_xpos[geom])
    hint=R.T@np.asarray(world_normal_hint);hint/=np.linalg.norm(hint)
    kind=int(model.geom_type[geom])
    if kind==3:
        axis=np.array([0.,0.,np.clip(p[2],-model.geom_size[geom,1],model.geom_size[geom,1])])
        n=p-axis
        if np.linalg.norm(n)<1e-12:raise ValueError('Capsule axis has no surface normal')
        n/=np.linalg.norm(n);point=axis+model.geom_size[geom,0]*n;method='capsule_radial'
    elif kind==7:
        mid=int(model.geom_dataid[geom]);start=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid])
        vertices=np.asarray(model.mesh_vert[start:start+count],float);planes=ConvexHull(vertices).equations
        gaps=planes[:,:3]@p+planes[:,3];eligible=np.flatnonzero(np.abs(gaps)<=2e-5)
        if not len(eligible):raise ValueError('Native witness is not near the full hand hull boundary')
        chosen=eligible[np.argmax(planes[eligible,:3]@hint)];n=planes[chosen,:3];point=p-gaps[chosen]*n
        if np.max(planes[:,:3]@point+planes[:,3])>2e-5:raise ValueError('Projected anchor outside hull')
        method='nearest_supporting_plane_with_normal_hint'
    else:raise ValueError('Only declared convex mesh/capsule hand shapes')
    movement=float(np.linalg.norm(point-p))
    if movement>2e-5:raise ValueError('Surface anchor correction exceeds20um')
    return dict(geom=int(geom),point_local=point,normal_local=n,method=method,
                endpoint_projection_m=movement,normal_dot_native_hint=float(n@hint))


class SurfacePatchIK:
    def __init__(self,model,qpos,qpos_ids,dof_ids,anchor,points_world,inward_world,source_ids,
                 *,stand_off_m=.00005,position_scale_m=.003,normal_scale=.5):
        import mujoco
        self.mj,self.model,self.data=mujoco,model,mujoco.MjData(model)
        self.base=np.asarray(qpos,float).copy();self.qids=np.asarray(qpos_ids,int);self.dids=np.asarray(dof_ids,int)
        self.anchor=anchor;self.points=np.asarray(points_world,float);self.normals=np.asarray(inward_world,float)
        self.ids=np.asarray(source_ids,int);self.gap=float(stand_off_m);self.ps=position_scale_m;self.ns=normal_scale
        if self.base.shape!=(model.nq,) or self.qids.shape!=self.dids.shape or self.points.shape!=self.normals.shape or self.points.shape!=(len(self.ids),3) or not len(self.ids):raise ValueError('Explicit finite nonempty geometry and joint maps')
        if not np.isfinite(np.r_[self.base,self.points.ravel(),self.normals.ravel()]).all() or not np.allclose(np.linalg.norm(self.normals,axis=1),1,atol=1e-7):raise ValueError('Finite patches with unit inward normals')
        if not self.ps>0 or not self.ns>0 or self.gap<0:raise ValueError('Declared proposal scales and stand-off')
        self.origin=self.base[self.qids].copy();self.cached=None;self.calls=0

    def evaluate(self,x):
        x=np.asarray(x,float)
        if x.shape!=self.qids.shape or not np.isfinite(x).all():raise ValueError('Finite raw joint proposal')
        if self.cached is not None and x.tobytes()==self.cached[0]:return self.cached[1:]
        self.data.qpos[:]=self.base;self.data.qpos[self.qids]=x
        self.mj.mj_kinematics(self.model,self.data);self.mj.mj_comPos(self.model,self.data);self.calls+=1
        gid=self.anchor['geom'];R=self.data.geom_xmat[gid].reshape(3,3)
        p=self.data.geom_xpos[gid]+R@self.anchor['point_local'];n=R@self.anchor['normal_local']
        # Source outward=-inward; a positive stand-off lies outside the object.
        positions=self.points-self.gap*self.normals
        costs=np.sum(((p-positions)/self.ps)**2,axis=1)+np.sum(((n-self.normals)/self.ns)**2,axis=1)
        selected=int(np.argmin(costs));jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        self.mj.mj_jac(self.model,self.data,jp,jr,p,int(self.model.geom_bodyid[gid]))
        residual=np.r_[(p-positions[selected])/self.ps,(n-self.normals[selected])/self.ns,.005*(x-self.origin)]
        J=np.r_[jp[:,self.dids]/self.ps,np.cross(jr[:,self.dids].T,n).T/self.ns,.005*np.eye(len(x))]
        record=dict(point_world=p,normal_world=n,source_index=selected,source_id=int(self.ids[selected]),
            source_distance_m=float(np.linalg.norm(p-self.points[selected])),normal_dot=float(n@self.normals[selected]),
            proposal_cost=float(residual@residual))
        self.cached=(x.tobytes(),residual,J,record)
        return residual,J,record

    def solve(self,start,lower,upper,*,max_nfev=120):
        start,lower,upper=map(lambda a:np.asarray(a,float),[start,lower,upper])
        if np.any(start<lower) or np.any(start>upper):raise ValueError('Legal explicit start; never clamp')
        answer=least_squares(lambda x:self.evaluate(x)[0],start,jac=lambda x:self.evaluate(x)[1],
            bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-9)
        record=self.evaluate(answer.x)[2];q=self.base.copy();q[self.qids]=answer.x
        return dict(**record,qpos=q,solver_success=bool(answer.success),solver_message=str(answer.message),
            nfev=int(answer.nfev),kinematics_calls=self.calls,
            material_point_patch_gate=record['source_distance_m']<=.003 and record['normal_dot']>=.5,
            native_contact_or_force_certified=False)
