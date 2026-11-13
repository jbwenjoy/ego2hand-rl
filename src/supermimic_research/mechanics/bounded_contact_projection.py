"""Bounded native convex-distance proposals, with explicit object/joint indexing."""
import time
import numpy as np
from scipy.optimize import least_squares
from .compound_collision import eligible_pairs


class ProjectionBudget(RuntimeError):
    pass


class BoundedContactProjection:
    def __init__(self,model,data,qids,lower,upper,parts,hand,groups,config):
        import mujoco
        self.mj=mujoco;self.model=model;self.data=data;self.qids=np.asarray(qids,int)
        self.base=data.qpos.copy();self.start=self.base[self.qids].copy()
        self.lower=np.asarray(lower,float);self.upper=np.asarray(upper,float);self.config=config
        if self.start.shape!=self.lower.shape or self.start.shape!=self.upper.shape or np.any(self.lower>=self.upper) or np.any(self.start<self.lower) or np.any(self.start>self.upper):
            raise ValueError('Explicit feasible indexed joint bounds')
        self.parts=set(map(int,parts));self.hand=list(map(int,hand));self.groups={int(k):set(map(int,v)) for k,v in groups.items()}
        self.allowed=set().union(*self.groups.values());allpairs=eligible_pairs(model,list(parts)+self.hand)
        self.object_pairs=[(a,b) for a,b in allpairs if (a in self.parts)!=(b in self.parts)]
        self.self_pairs=eligible_pairs(model,self.hand)
        self.pairs=self.object_pairs+self.self_pairs
        self.group_indices={k:[i for i,(a,b) in enumerate(self.object_pairs) if (b if a in self.parts else a) in group] for k,group in self.groups.items()}
        if any(not v for v in self.group_indices.values()):raise ValueError('Nonempty declared contact groups')
        self.radius={g:self.geom_radius(g) for g in set(parts)|set(hand)}
        self.counts=dict(kinematics=0,native_distance=0,sphere_pruned=0,residual=0,integration_steps=0)
        self.best=None;self.cached=None;self.deadline=float('inf');self.log=None

    def geom_radius(self,g):
        m=self.model;t=int(m.geom_type[g]);size=m.geom_size[g]
        if t==2:return float(size[0])
        if t==3:return float(size[0]+size[1])
        if t==5:return float(np.linalg.norm(size[:2]))
        if t==6:return float(np.linalg.norm(size))
        if t==7:
            mesh=int(m.geom_dataid[g]);a,n=int(m.mesh_vertadr[mesh]),int(m.mesh_vertnum[mesh])
            return float(np.linalg.norm(m.mesh_vert[a:a+n],axis=1).max())
        raise ValueError('Explicit convex primitive/compiled mesh collider only')

    def gaps(self,q):
        q=np.asarray(q,float)
        if q.shape!=self.start.shape or np.any(q<self.lower-1e-12) or np.any(q>self.upper+1e-12):raise ValueError('Raw bounded projection, no clamp')
        self.data.qpos[:]=self.base;self.data.qpos[self.qids]=q
        self.mj.mj_kinematics(self.model,self.data);self.mj.mj_comPos(self.model,self.data);self.counts['kinematics']+=1
        limit=self.config['distance_limit_m'];result=[]
        for a,b in self.pairs:
            lower=np.linalg.norm(self.data.geom_xpos[a]-self.data.geom_xpos[b])-self.radius[a]-self.radius[b]
            if lower>limit:gap=limit;self.counts['sphere_pruned']+=1
            else:
                witness=np.zeros(6);gap=float(self.mj.mj_geomDistance(self.model,self.data,a,b,limit,witness));self.counts['native_distance']+=1
            if not np.isfinite(gap):raise ValueError('Nonfinite native distance')
            result.append(gap)
        return np.array(result)

    def residual(self,q):
        if self.cached is not None and np.array_equal(q,self.cached[0]):return self.cached[1].copy()
        if time.perf_counter()>self.deadline or self.counts['residual']>=self.config['maximum_residual_calls']:raise ProjectionBudget('Fixed proposal budget')
        gaps=self.gaps(q);c=self.config;n=len(self.object_pairs)
        floor=np.array([c['contact_floor_m'] if (b if a in self.parts else a) in self.allowed else c['other_clearance_m'] for a,b in self.object_pairs]+[c['self_clearance_m']]*len(self.self_pairs))
        contact=np.array([np.min(gaps[indices])-c['contact_target_m'] for indices in self.group_indices.values()])
        r=np.r_[c['joint_regularization']*(q-self.start)/c['joint_scale_rad'],np.maximum(floor-gaps,0)/c['collision_scale_m'],contact/c['contact_scale_m']]
        cost=float(r@r);self.counts['residual']+=1
        row=dict(call=self.counts['residual'],cost=cost,q=q.tolist(),max_object_penetration_m=max(0.,float(-gaps[:n].min())),max_self_penetration_m=max(0.,float(-gaps[n:].min())) if len(gaps)>n else 0.,group_gaps_m=(contact+c['contact_target_m']).tolist())
        if self.log is not None:
            import json
            self.log.write(json.dumps(row)+'\n');self.log.flush()
        if self.best is None or cost<self.best['cost']:self.best=dict(cost=cost,q=q.copy(),gaps=gaps.copy())
        self.cached=(np.array(q).copy(),r.copy());return r

    def solve(self,log=None):
        self.log=log;start=time.perf_counter();self.deadline=start+self.config['maximum_wall_s'];success=False;nfev=None
        try:
            result=least_squares(self.residual,self.start,bounds=(self.lower,self.upper),jac='2-point',max_nfev=self.config['max_nfev'],diff_step=1e-4,ftol=1e-9,xtol=1e-9,gtol=1e-8)
            success=bool(result.success);message=str(result.message);nfev=int(result.nfev)
        except ProjectionBudget as error:message=str(error)
        if self.best is None:raise ValueError('No completed projection proposal')
        q=self.best['q'];gaps=self.gaps(q);n=len(self.object_pairs);group=np.array([gaps[v].min() for v in self.group_indices.values()])
        c=self.config;checks=dict(object_penetration=bool(gaps[:n].min()>=-c['qualification_penetration_m']),self_penetration=bool(np.all(gaps[n:]>=-c['qualification_self_penetration_m'])),
            intended_contacts=bool(np.all(group>=-c['qualification_penetration_m']) and np.all(group<=c['qualification_contact_gap_m'])),
            joint_bounds=bool(np.all(q>=self.lower) and np.all(q<=self.upper)),object_fixed=bool(np.array_equal(np.delete(self.data.qpos,self.qids),np.delete(self.base,self.qids))))
        return dict(q=q,delta=q-self.start,gaps=gaps,group_gaps=group,checks=checks,geometry_qualified=all(checks.values()),
            solver_success=success,message=message,nfev=nfev,wall_s=time.perf_counter()-start,counts=self.counts.copy(),
            scope='Native ideal-convex geometry proposal and bounded clearance only; no independent CAD certificate or dynamic support qualification.')
