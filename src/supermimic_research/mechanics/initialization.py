"""Bounded joint/phone/PD-target proposals using actual reset constraint solves.

No integration, learned model, action-interface change or applied aid. Native
finite differences are search proposals, not certified contact derivatives.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class SearchBudget(RuntimeError):pass


class NativeInitialization:
    size=38
    def __init__(self,model,q0,u0):
        import mujoco
        self.mj=mujoco;self.model=model;self.data=mujoco.MjData(model)
        self.q0=np.array(q0,dtype=float);self.u0=np.array(u0,dtype=float)
        self.R0=Rotation.from_quat(self.q0[23:27][[1,2,3,0]]).as_matrix()
        self.scale=np.r_[np.full(12,.2),np.full(3,.015),np.full(3,.25),np.full(20,.3)]
        self.lo=np.full(38,-1.);self.hi=np.full(38,1.)
        self.lo[:12]=np.maximum(-1.,(model.jnt_range[:12,0]+.002-self.q0[:12])/.2)
        self.hi[:12]=np.minimum(1.,(model.jnt_range[:12,1]-.002-self.q0[:12])/.2)
        self.lo[18:]=np.maximum(-1.,(model.actuator_ctrlrange[:,0]-self.u0)/.3)
        self.hi[18:]=np.minimum(1.,(model.actuator_ctrlrange[:,1]-self.u0)/.3)
        if not np.all(self.lo<0) or not np.all(self.hi>0):raise ValueError('Interior qualified parent required')
        self.counts=dict(MjData=1,static_forward=0,resetData=0,contact_force_reads=0,finite_difference_columns=0,integration_steps=0,NN_FW=0,GPU_calls=0)
        self.patch=[];self.base=self.state(np.zeros(38))
        phone=model.body('hot3d_phone24').id
        for gid in (15,28,41):
            row=next(row for row in self.base['contacts24'] if (int(row[1]),int(row[2])) in ((gid,71),(71,gid)))
            position=row[6:9];sign=1 if int(row[2])==71 else -1
            frame=row[9:18].reshape(3,3);R=self.data.geom_xmat[gid].reshape(3,3);p=self.data.geom_xpos[gid]
            self.patch.append(dict(gid=gid,hand_local=R.T@(position-p),phone_local=self.R0.T@(position-self.q0[20:23]),normal_BODY=self.R0.T@(sign*frame[0])))
        self.palm_position=self.data.xpos[model.body('right_palm_link').id].copy()
        self.palm_rotation=self.data.xmat[model.body('right_palm_link').id].copy()

    def decode(self,z):
        z=np.asarray(z,dtype=float)
        if z.shape!=(38,) or not np.isfinite(z).all() or np.any(z<self.lo-1e-14) or np.any(z>self.hi+1e-14):raise ValueError('Raw legal proposal parameters, no control clamp')
        delta=z*self.scale;q=self.q0.copy();q[:12]+=delta[:12];q[20:23]+=delta[12:15]
        if np.any(delta[15:18]):
            R=Rotation.from_rotvec(delta[15:18]).as_matrix()@self.R0
            quat=Rotation.from_matrix(R).as_quat();q[23:27]=quat[[3,0,1,2]]
        u=self.u0+delta[18:]
        if np.any(u<self.model.actuator_ctrlrange[:,0]) or np.any(u>self.model.actuator_ctrlrange[:,1]):raise ValueError('Illegal proposal target')
        return q,u

    def state(self,z):
        q,u=self.decode(z);m,d,mj=self.model,self.data,self.mj
        mj.mj_resetData(m,d);self.counts['resetData']+=1
        d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=u;d.qacc_warmstart[:]=0
        mj.mj_forward(m,d);self.counts['static_forward']+=1
        if np.any(d.warning.number) or np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise ValueError('Native reset warning or aid')
        contacts=[];pad=np.zeros(3);other=table=self_pen=pad_pen=0.
        for i in range(d.ncon):
            con=d.contact[i];g1,g2=map(int,con.geom);force=np.empty(6)
            mj.mj_contactForce(m,d,i,force);self.counts['contact_force_reads']+=1
            contacts.append(np.r_[-1,g1,g2,int(con.dim),int(con.efc_address),con.dist,con.pos,con.frame,force])
            if 71 in (g1,g2):
                gid=g1 if g2==71 else g2
                if gid in (15,28,41):pad[(15,28,41).index(gid)]+=max(0.,force[0]);pad_pen=max(pad_pen,-con.dist)
                elif gid==m.geom('stage1_single_table').id:table+=max(0.,force[0])
                else:other+=max(0.,force[0])
            elif m.geom('stage1_single_table').id not in (g1,g2):self_pen=max(self_pen,-con.dist)
        net=(d.qfrc_actuator+d.qfrc_passive+d.qfrc_constraint-d.qfrc_bias).copy()
        if not np.isfinite(np.r_[net,d.qacc]).all():raise ValueError('Nonfinite static native response')
        return dict(qpos27=q,ctrl20=u,net_generalized26=net,qacc26=d.qacc.copy(),contacts24=np.array(contacts).reshape(-1,24),
            pad_force_N=pad,nonpad_force_N=float(other),table_force_N=float(table),pad_penetration_m=float(pad_pen),self_penetration_m=float(self_pen),
            qfrc_bias26=d.qfrc_bias.copy(),qfrc_passive26=d.qfrc_passive.copy(),qfrc_constraint26=d.qfrc_constraint.copy(),qfrc_actuator26=d.qfrc_actuator.copy())

    def residual(self,z,s):
        d=self.data;R=Rotation.from_quat(s['qpos27'][23:27][[1,2,3,0]]).as_matrix();normal_gaps=[]
        for patch in self.patch:
            gid=patch['gid'];hand=d.geom_xpos[gid]+d.geom_xmat[gid].reshape(3,3)@patch['hand_local']
            point=s['qpos27'][20:23]+R@patch['phone_local'];normal=R@patch['normal_BODY']
            normal_gaps.append(float(normal@(point-hand)))
        gaps=np.array(normal_gaps)
        net=s['net_generalized26']
        # Fixed units/scales are declared in the intent; no data-fitted weights.
        return np.r_[net[:20]/.01,net[20:23]/.1,net[23:26]/.005,
            np.maximum(.1-s['pad_force_N'],0)/.1,s['nonpad_force_N']/.01,s['table_force_N']/.01,
            max(s['pad_penetration_m']-.0001,0)/.00005,max(s['self_penetration_m'],0)/.00005,
            np.maximum(gaps-.0002,0)/.001,np.maximum(-.0001-gaps,0)/.001,.003*np.asarray(z)]

    def search(self,start,path,*,maximum_calls=1800,maximum_seconds=35.):
        """All proposals logged; best total score selected before qualification."""
        log=Path(path).open('x',encoding='utf8');tick=time.perf_counter();calls=0;best=None;last=None;stop=None
        def fun(z):
            nonlocal calls,best,last
            if calls>=maximum_calls or time.perf_counter()-tick>=maximum_seconds:raise SearchBudget('Declared static search budget')
            s=self.state(z);r=self.residual(z,s);cost=float(r@r);calls+=1
            if best is None or cost<best['cost']:best=dict(z=np.asarray(z).copy(),cost=cost,call=calls)
            last=(np.asarray(z).copy(),r.copy())
            log.write(json.dumps(dict(call=calls,z38=np.asarray(z).tolist(),cost=cost,net_generalized26=s['net_generalized26'].tolist(),
                pad_force_N=s['pad_force_N'].tolist(),pad_penetration_m=s['pad_penetration_m'],self_penetration_m=s['self_penetration_m'],
                nonpad_force_N=s['nonpad_force_N'],table_force_N=s['table_force_N']),allow_nan=False)+'\n')
            return r
        def jac(z):
            if last is not None and np.array_equal(last[0],z):base=last[1]
            else:base=fun(z)
            result=np.empty((len(base),38))
            for j in range(38):
                # Explicit 1e-3 scaled one-sided secants, not optimizer-relative
                # eps at zero and not claims of smooth physical derivatives.
                step=.001 if z[j]+.001<=self.hi[j] else -.001
                p=np.asarray(z).copy();p[j]+=step;result[:,j]=(fun(p)-base)/step
                self.counts['finite_difference_columns']+=1
            return result
        try:
            result=least_squares(fun,np.asarray(start),jac=jac,bounds=(self.lo,self.hi),method='trf',max_nfev=150,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            stop=dict(kind='optimizer_returned',status=int(result.status),message=result.message,nfev=int(result.nfev))
        except SearchBudget as error:stop=dict(kind='budget',message=str(error))
        finally:log.close()
        if best is None:raise ValueError('No completed proposal')
        final=self.state(best['z'])
        if not np.array_equal(self.data.xpos[self.model.body('right_palm_link').id],self.palm_position) or not np.array_equal(self.data.xmat[self.model.body('right_palm_link').id],self.palm_rotation):raise ValueError('Fixed wrist changed')
        return dict(best=best,stop=stop,calls=calls,elapsed_s=time.perf_counter()-tick,final=final,
            fresh_final_cost=float(self.residual(best['z'],final)@self.residual(best['z'],final)))

    def deviation(self,q):
        R=Rotation.from_quat(q[23:27][[1,2,3,0]]).as_matrix();deltaR=R@self.R0.T;deltap=q[20:23]-deltaR@self.q0[20:23]
        return dict(q20_delta_rad=(q[:20]-self.q0[:20]).tolist(),phone_root_delta_world_m=(q[20:23]-self.q0[20:23]).tolist(),
            phone_rotation_delta_world_rad=Rotation.from_matrix(deltaR).as_rotvec().tolist(),
            proposed_reference_left_transform_R=deltaR.tolist(),proposed_reference_left_transform_t_m=deltap.tolist(),
            task_bundle_created=False,Actor_or_training_compatible=False,
            scope='Candidate initialization deviation. Transform is only a proposal for a distinct retargeted reference, not measured ego gravity, a rebuilt task, or permission to reuse the old observation/memory bundle.')
