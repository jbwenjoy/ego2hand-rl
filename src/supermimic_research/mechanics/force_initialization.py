"""Bounded zero-velocity native equilibrium proposals with explicit geometry.

Native finite secants optimize initial q/object pose/PD commands, not a learned
model or certified contact derivatives. Qualification is a separate operation.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .compound_contacts import CompoundContacts


class NativeEquilibrium:
    def __init__(self,model,qpos,ctrl,binding,groups):
        import mujoco
        self.mj,self.model,self.binding=mujoco,model,binding
        self.q0=np.array(qpos,float);self.u0=np.array(ctrl,float);self.data=mujoco.MjData(model)
        self.oid=binding['mug_body_id'];self.qadr=binding['mug_free_qpos'];self.dadr=binding['mug_free_dof']
        if self.q0.shape!=(27,) or self.u0.shape!=(20,) or self.qadr!=20 or self.dadr!=20:
            raise ValueError('Explicit20 hinge + free rigid object layout')
        self.R0=Rotation.from_quat(self.q0[23:27][[1,2,3,0]]).as_matrix()
        self.scale=np.r_[np.full(20,.03),np.full(3,.002),np.full(3,.04),np.full(20,.3)]
        self.lo=np.full(46,-1.);self.hi=np.full(46,1.)
        self.lo[:20]=np.maximum(-1.,(model.jnt_range[:20,0]+.001-self.q0[:20])/.03)
        self.hi[:20]=np.minimum(1.,(model.jnt_range[:20,1]-.001-self.q0[:20])/.03)
        self.lo[26:]=np.maximum(-1.,(model.actuator_ctrlrange[:,0]-self.u0)/.3)
        self.hi[26:]=np.minimum(1.,(model.actuator_ctrlrange[:,1]-self.u0)/.3)
        if not np.all(self.lo<0) or not np.all(self.hi>0):raise ValueError('Interior legal parent state/target required')
        self.reader=CompoundContacts(model,binding['mug_part_geom_ids'],self.oid,groups,model.geom('mug_task_table').id)
        self.groups=groups;self.counts=dict(native_forward=0,residual=0,secant_columns=0,integration_steps=0)
        base=self.state(np.zeros(46));d=self.data
        self.palm_position=d.xpos[model.body('right_palm_link').id].copy();self.palm_rotation=d.xmat[model.body('right_palm_link').id].copy()
        self.anchors=[];parts=set(binding['mug_part_geom_ids'])
        for group,geoms in groups.items():
            choices=[]
            for row in base['raw23']:
                a,b=map(int,row[:2]);other=a if b in parts else b
                if (a in parts)!=(b in parts) and other in geoms:
                    choices.append((row[17],row,other))
            if not choices:raise ValueError('Each canonical finger requires actual native contact for equilibrium seed')
            _,row,gid=max(choices,key=lambda t:t[0]);position=row[5:8];sign=1 if int(row[1]) in parts else -1
            normal=sign*row[8:17].reshape(3,3)[0]
            self.anchors.append(dict(gid=gid,hand_local=d.geom_xmat[gid].reshape(3,3).T@(position-d.geom_xpos[gid]),
                object_local=self.R0.T@(position-self.q0[20:23]),normal_BODY=self.R0.T@normal))

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(46,) or not np.isfinite(z).all() or np.any(z<self.lo-1e-13) or np.any(z>self.hi+1e-13):
            raise ValueError('Raw bounded equilibrium proposal, never clamp')
        delta=z*self.scale;q=self.q0.copy();q[:20]+=delta[:20];q[20:23]+=delta[20:23]
        if np.any(delta[23:26]):
            R=Rotation.from_rotvec(delta[23:26]).as_matrix()@self.R0
            q[23:27]=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        return q,self.u0+delta[26:]

    def state(self,z):
        m,d,mj=self.model,self.data,self.mj;q,u=self.decode(z)
        mj.mj_resetData(m,d);d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=u;d.qacc_warmstart[:]=0
        mj.mj_forward(m,d);self.counts['native_forward']+=1
        if np.any(d.warning.number) or np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise ValueError('Native equilibrium warning or aid')
        s=self.reader.read(d,raw=True);net=d.qfrc_actuator+d.qfrc_passive+d.qfrc_constraint-d.qfrc_bias
        inertia=np.empty(m.nv);mj.mj_mulM(m,d,inertia,d.qacc)
        if np.max(np.abs(inertia-net))>1e-9:raise ValueError('Native mass-times-acceleration identity')
        s.update(qpos27=q,ctrl20=u,net_generalized26=net.copy(),qacc26=d.qacc.copy(),
            net_object_COM_wrench_world=np.r_[s['force_world_N']+m.body_mass[self.oid]*m.opt.gravity,s['torque_about_COM_world_Nm']])
        return s

    def residual(self,z,s):
        d=self.data;R=d.xmat[self.oid].reshape(3,3);gaps=[]
        for anchor in self.anchors:
            hand=d.geom_xpos[anchor['gid']]+d.geom_xmat[anchor['gid']].reshape(3,3)@anchor['hand_local']
            point=s['qpos27'][20:23]+R@anchor['object_local'];normal=R@anchor['normal_BODY']
            gaps.append(normal@(point-hand))
        gaps=np.array(gaps);net=s['net_generalized26'];normal=s['normal_by_finger_N'][:3]
        return np.r_[net[:20]/.01,net[20:23]/.05,net[23:26]/.003,
            np.maximum(.2-normal,0)/.1,s['other_normal_N']/.01,s['table_normal_N']/.01,
            max(s['max_native_penetration_m']-.000075,0)/.00002,s['max_self_penetration_m']/.00002,
            np.maximum(gaps-.0002,0)/.0001,np.maximum(-.0001-gaps,0)/.0001,
            .002*normal,.003*np.asarray(z)]

    def search(self,log,*,maximum_calls=2400,maximum_seconds=35.):
        import json
        tick=time.perf_counter();calls=0;best=None;last=None
        class Limit(RuntimeError):pass
        def fun(z):
            nonlocal calls,best,last
            if calls>=maximum_calls or time.perf_counter()-tick>=maximum_seconds:raise Limit('Declared native equilibrium budget')
            s=self.state(z);r=self.residual(z,s);cost=float(r@r);calls+=1;self.counts['residual']+=1
            record=dict(call=calls,z46=np.asarray(z).tolist(),squared_cost=cost,
                net_generalized26=s['net_generalized26'].tolist(),net_object_COM_wrench_world=s['net_object_COM_wrench_world'].tolist(),
                finger_normal_N=s['normal_by_finger_N'][:3].tolist(),penetration_m=s['max_native_penetration_m'],self_penetration_m=s['max_self_penetration_m'])
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            if best is None or cost<best['squared_cost']:best=dict(record)
            last=(np.asarray(z).copy(),r.copy());return r
        def jac(z):
            base=last[1] if last is not None and np.array_equal(z,last[0]) else fun(z)
            result=np.empty((len(base),46))
            for j in range(46):
                step=.001 if z[j]+.001<=self.hi[j] else -.001
                changed=np.asarray(z).copy();changed[j]+=step
                result[:,j]=(fun(changed)-base)/step;self.counts['secant_columns']+=1
            return result
        try:
            result=least_squares(fun,np.zeros(46),jac=jac,bounds=(self.lo,self.hi),max_nfev=100,
                ftol=1e-9,xtol=1e-9,gtol=1e-8)
            stop=dict(kind='optimizer',status=int(result.status),message=str(result.message),nfev=int(result.nfev))
        except Limit as error:stop=dict(kind='budget',message=str(error))
        if best is None:raise ValueError('No completed equilibrium proposal')
        final=self.state(np.array(best['z46']))
        if not np.array_equal(self.data.xpos[self.model.body('right_palm_link').id],self.palm_position) or not np.array_equal(self.data.xmat[self.model.body('right_palm_link').id],self.palm_rotation):
            raise ValueError('Fixed robot root changed')
        return dict(best=best,stop=stop,calls=calls,wall_s=time.perf_counter()-tick,counts=self.counts.copy(),final=final,
            fresh_final_squared_cost=float(self.residual(best['z46'],final)@self.residual(best['z46'],final)))
