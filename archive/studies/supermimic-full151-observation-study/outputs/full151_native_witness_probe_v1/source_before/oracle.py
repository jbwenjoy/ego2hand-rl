"""Guarded, original-options native geometry and every generated witness J.

Kinematics/collision/point Jacobians only. No dynamics, forces, gap query or
optimizer. Native contact index is an inventory identity, never a persistent
feature identity across different configurations.
"""
from pathlib import Path
import importlib.util,sys
import numpy as np
from .io import checked
from .contact_row_oracle_v2 import distance_jacobian_from_point_jacobians


class FrozenNativeOracle:
    def __init__(self,model,context,native_geometry,kinematics):
        self.pins=[dict(p) for p in (model,context,native_geometry,kinematics)]
        self.paths=[checked(p) for p in self.pins]
        self.allowed={};self.forbidden={};self.original={};self.query=None

    def __enter__(self):
        if self.original:raise ValueError('Geometry guard context cannot be nested')
        for p in self.pins:checked(p)
        import mujoco
        import scipy.optimize as opt
        self.mj=mujoco
        def deny(name):
            def fail(*a,**kw):self.forbidden[name]+=1;raise AssertionError('Geometry oracle forbids '+name)
            return fail
        def count(name,f):
            def call(*a,**kw):self.allowed[name]+=1;return f(*a,**kw)
            return call
        for ns,names in [(mujoco,['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce']),
                         (opt,['linprog','minimize','least_squares'])]:
            for name in names:
                self.original[(ns,name)]=getattr(ns,name);self.forbidden.setdefault(name,0);setattr(ns,name,deny(name))
        for name in ['mj_kinematics','mj_comPos','mj_collision','mj_jac']:
            self.original[(mujoco,name)]=getattr(mujoco,name);self.allowed.setdefault(name,0);setattr(mujoco,name,count(name,getattr(mujoco,name)))
        try:
            if self.query is None:
                sys.path.insert(0,str(self.paths[3].parent))
                name='_full151_native_all_witness_geometry'
                spec=importlib.util.spec_from_file_location(name,self.paths[2]);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
                if Path(sys.modules[mod.RobotKinematics.__module__].__file__).resolve()!=self.paths[3]:raise ValueError('Wrong frozen kinematics import')
                self.query=mod.NativeFilteredContactGeometry(self.paths[0],self.paths[1])
                if len(self.query.pairs)!=273:raise ValueError('Expected fixed273pair catalog')
                body=self.query.model.body('object/cube').id
                if np.any(self.query.model.body_ipos[body]):raise ValueError('Only zero-COM-offset object supported')
        except Exception:
            self.__exit__(None,None,None);raise
        return self

    def __exit__(self,*args):
        for (ns,name),f in self.original.items():setattr(ns,name,f)
        self.original={}

    def evaluate(self,q,pose,with_jacobian=True):
        g=self.query
        if g is None or not self.original:raise ValueError('Use a context-managed oracle')
        q=np.asarray(q,dtype=float);pose=np.asarray(pose,dtype=float)
        # Each ledger row requires a fresh query, never a cached contact list.
        g.set_object_pose(pose);g.last_q=None;g.cached=None;g.robot.last_q=None
        before=g.collision_query_count;v=g.evaluate(q)
        if g.collision_query_count!=before+1:raise AssertionError('Expected one fresh collision query')
        m,d=g.model,g.data;raw=v.native_contact_records
        M=len(raw);J=np.empty((M,20));pair=np.empty(M,np.int32);native=np.empty(M,np.int32)
        order=np.empty((M,2),np.int32);dist=np.empty(M);pos=np.empty((M,3));norm=np.empty((M,3))
        for k,(p,idx,a,b,distance,point,normal) in enumerate(raw):
            pair[k]=p;native[k]=idx;order[k]=(a,b);dist[k]=distance;pos[k]=point;norm[k]=normal
            if with_jacobian:
                ja,jra=np.zeros((3,m.nv)),np.zeros((3,m.nv));jb,jrb=np.zeros((3,m.nv)),np.zeros((3,m.nv))
                self.mj.mj_jac(m,d,ja,jra,pos[k],int(m.geom_bodyid[a]));self.mj.mj_jac(m,d,jb,jrb,pos[k],int(m.geom_bodyid[b]))
                J[k]=distance_jacobian_from_point_jacobians(norm[k],ja,jb,g.robot.va)
            else:J[k]=np.nan
        if with_jacobian and not np.isfinite(J).all():raise ValueError('Nonfinite witness Jacobian; cannot fabricate or omit it')
        if d.time!=0 or np.any(d.qvel):raise ValueError('Geometry query modified clock/velocity')
        if v.uncatalogued_native_contact_count:raise ValueError('Uncatalogued contact inventory cannot be silently omitted')
        return {'q':q.copy(),'object_pose_w':pose.copy(),'pair_index':pair,'native_contact_index':native,'geom_order':order,
          'contactdist_m':dist,'position_w_m':pos,'normal_a_to_b_w':norm,'distance_jacobian_q_m_per_rad':J,
          'selected_distance_m':v.signed_distance_m.copy(),'generated_mask':v.generated_mask.copy(),
          'tie_count':v.tie_count.copy(),'contact_count':v.contact_count.copy(),
          'selected_native_index':v.selected_contact_index.copy(),'selected_distance_jacobian_q':v.selected_distance_jacobian.copy(),
          'selected_normal_canonical_w':v.selected_normal_a_to_b_w.copy(),'native_ncon':v.native_contact_count}

    def grid(self,q601,poses601,with_jacobian=True):
        q=np.asarray(q601,float);poses=np.asarray(poses601,float)
        if q.shape!=(601,20) or poses.shape!=(601,7) or not np.isfinite(q).all() or not np.isfinite(poses).all():raise ValueError('Complete finite601 grid required')
        rows=[]
        for i in range(601):rows.append(self.evaluate(q[i],poses[i],with_jacobian))
        keys=('pair_index','native_contact_index','geom_order','contactdist_m','position_w_m','normal_a_to_b_w','distance_jacobian_q_m_per_rad')
        out={k:np.concatenate([r[k] for r in rows],axis=0) for k in keys}
        out['sample_index']=np.concatenate([np.full(len(r['pair_index']),i,np.int32) for i,r in enumerate(rows)])
        for k in ('selected_distance_m','generated_mask','tie_count','contact_count'):out[k]=np.stack([r[k] for r in rows])
        out['witness_ids']=[{'sample':int(i),'pair_index':int(pair),'native_contact_index':int(idx),'native_geom_order':list(map(int,order))} for i,pair,idx,order in zip(out['sample_index'],out['pair_index'],out['native_contact_index'],out['geom_order'])]
        out['q601_rad']=q.copy();out['object_pose601_w']=poses.copy()
        return out


def geometry_summary(grid,pair_kind,target_depth=.002999):
    mask=np.asarray(grid['generated_mask']);distance=np.asarray(grid['selected_distance_m']);kind=np.asarray(pair_kind)
    depth=np.where(mask,np.maximum(-distance,0),0.)
    by_kind={k:depth[:,kind==k].max(axis=1) for k in ('hand_object','hand_self')}
    maximum=float(depth.max())
    return {'maximum_generated_depth_m':maximum,'planning_max_violation_m':max(0.,maximum-target_depth),
      'hand_object_max_m':float(by_kind['hand_object'].max()),'hand_self_max_m':float(by_kind['hand_self'].max()),
      'hand_object_samples_over3mm':int(np.count_nonzero(by_kind['hand_object']>.003)),
      'hand_self_samples_over3mm':int(np.count_nonzero(by_kind['hand_self']>.003)),
      'hand_object_absent_unknown_samples':int(np.count_nonzero(~mask[:,kind=='hand_object'].any(axis=1))),
      'tied_samples':int(np.count_nonzero(np.any(grid['tie_count']>1,axis=1))),
      'sampled_geometry_gate':bool(maximum<=.003),'witness_count':len(grid['contactdist_m'])}
