"""Native fixed-wrist phone reader: POST state58 and complete solver-PRE traces.

One reset forward, then exactly one native step per integer tick. Observation
never refreshes MuJoCo's derived fields. Absolute u20 is held for H=20ms.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np

MODEL_SHA='ff88ccc33fd1190328639925d82c6c11550b44dea67eec0f51c0a3cfd4339826'
from . import state_codec as codec


def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def _bits(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

class NativePhoneReader:
    def __init__(self,model_path,*,expected_model_sha256=MODEL_SHA,dt_s):
        import mujoco
        self.mj=mujoco
        if mujoco.__version__!='3.11.0':raise ValueError('Pinned native MuJoCo3.11')
        self.model_pin=pin(model_path)
        if self.model_pin['sha256']!=expected_model_sha256 or expected_model_sha256!=MODEL_SHA:
            raise ValueError('Original fixed physical model only')
        self.plan=codec.state.TickPlan(float(dt_s));self.h=float(dt_s);self.n=self.plan.substeps
        self.m=mujoco.MjModel.from_binary_path(str(model_path));m=self.m
        if (m.nq,m.nv,m.nu,m.na,m.neq,m.nmocap,m.nflex,m.nplugin,m.ntendon)!=(27,26,20,0,0,0,0,0,0):
            raise ValueError('Original no-aid/no-plugin layout')
        for name in ['control','passive','contactfilter','sensor','act_dyn','act_gain','act_bias']:
            if getattr(mujoco,'get_mjcb_'+name)() is not None:raise ValueError('No opaque native callbacks')
        if np.any(m.body_gravcomp):raise ValueError('No gravity compensation')
        self.phone=m.body('hot3d_phone24').id;self.phone_geom=m.geom('phone24_collision_convex').id
        self.table=m.geom('stage1_single_table').id
        self.pads=[m.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)]
        if self.pads!=[15,28] or self.phone_geom!=71 or m.body_mass[self.phone]!=.2:
            raise ValueError('Same compiled phone/pad identities and nominal mass')
        if (m.opt.tolerance,m.opt.ls_tolerance,m.opt.iterations,m.opt.ls_iterations)!=(1e-6,.01,100,20):
            raise ValueError('Original solver budget')
        if (int(m.opt.integrator),int(m.opt.solver),int(m.opt.cone))!=(
                int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),int(mujoco.mjtSolver.mjSOL_NEWTON),
                int(mujoco.mjtCone.mjCONE_PYRAMIDAL)):
            raise ValueError('Original solver algorithms')
        if not np.array_equal(m.actuator_trnid[:,0],np.arange(20)) or not np.array_equal(m.actuator_gear[:,0],np.ones(20)):
            raise ValueError('Unit direct joint actuators')
        m.opt.timestep=self.h
        self.ipos=m.body_ipos[self.phone].copy();self.qrange=m.jnt_range[:20].copy()
        self.urange=m.actuator_ctrlrange.copy();self.initial_digest=self.model_digest()
        self.shared_physics_digest=self.model_digest(include_dt=False)
        self.signature=int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.state_size=mujoco.mj_stateSize(m,self.signature)
        self.counts=dict(MJB_reads=1,MjData=0,initial_forward=0,integration_steps=0,
                         contact_force_reads=0,initial_contact_force_reads=0,step_contact_force_reads=0,
                         receipt_reads=0,observer_forward=0,NN_training_GPU=0)
        self.d=None;self.tick=0;self.terminated=False

    def model_digest(self,include_dt=True):
        h=hashlib.sha256()
        for label,owner in [('model',self.m),('option',self.m.opt)]:
            for name in sorted(n for n in dir(owner) if not n.startswith('_')):
                if label=='option' and name=='timestep' and not include_dt:continue
                value=getattr(owner,name)
                if isinstance(value,np.ndarray):
                    a=np.ascontiguousarray(value);h.update((label+'.'+name+a.dtype.str+str(a.shape)).encode());h.update(a.tobytes())
                elif isinstance(value,(float,int,np.integer,np.floating)):
                    h.update((label+'.'+name+'='+repr(value)).encode())
        return h.hexdigest()

    def _control(self,u):
        u=np.asarray(u,dtype=np.float64)
        if u.shape!=(20,) or not np.isfinite(u).all() or np.any(u<self.urange[:,0]) or np.any(u>self.urange[:,1]):
            raise ValueError('Actual finite absolute u20 within real ranges, no clamp')
        return u.copy()

    def state58(self):return codec.actual_state(self.d.qpos,self.d.qvel,self.ipos)

    def receipt(self):
        state=np.empty(self.state_size,dtype=np.float64)
        self.mj.mj_getState(self.m,self.d,state,self.signature);self.counts['receipt_reads']+=1
        return dict(signature=self.signature,size=self.state_size,sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    integration_state=state,global_tick=self.tick,native_time_s=float(self.d.time))

    def reset(self,qpos27,base_ctrl20):
        q=np.asarray(qpos27,dtype=np.float64);u=self._control(base_ctrl20)
        if q.shape!=(27,) or not np.isfinite(q).all() or np.any(q[:20]<self.qrange[:,0]) or np.any(q[:20]>self.qrange[:,1]):
            raise ValueError('Exact legal finite starting pose')
        if abs(q[23:27]@q[23:27]-1)>1e-12:raise ValueError('Source unit quaternion, no repair')
        d=self.mj.MjData(self.m);self.counts['MjData']+=1
        d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=u
        self.mj.mj_forward(self.m,d);self.counts['initial_forward']+=1
        self.d=d;self.tick=0;self.terminated=False
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or np.any(d.warning.number):
            raise ValueError('Known no-aid initial native state')
        self.initial_qpos=q.copy();self.initial_state58=self.state58()
        initial_contacts=[];pad=np.zeros(2);other=table=pen=0.
        for ci in range(d.ncon):
            c=d.contact[ci];g1,g2=map(int,c.geom);f=np.empty(6)
            self.mj.mj_contactForce(self.m,d,ci,f)
            self.counts['contact_force_reads']+=1;self.counts['initial_contact_force_reads']+=1
            row=np.r_[0,g1,g2,int(c.dim),int(c.efc_address),c.dist,c.pos,c.frame,f]
            if not np.isfinite(row).all():raise ValueError('Known finite initial contacts')
            initial_contacts.append(row);pen=max(pen,-float(c.dist))
            if self.phone_geom in (g1,g2):
                otherg=g2 if g1==self.phone_geom else g1
                if otherg in self.pads:pad[self.pads.index(otherg)]+=max(0.,float(f[0]))
                elif otherg==self.table:table+=max(0.,float(f[0]))
                else:other+=max(0.,float(f[0]))
        return dict(state58=self.initial_state58.copy(),receipt=self.receipt(),qpos27=d.qpos.copy(),qvel26=d.qvel.copy(),
            ctrl20=d.ctrl.copy(),initial_contact_rows24=np.asarray(initial_contacts,dtype=np.float64).reshape(-1,24),
            initial_pad_normal_N2=pad,initial_phone_nonpad_normal_N=other,initial_table_normal_N=table,
            initial_penetration_m=pen,contact_phase='RESET_FORWARD at time0 with base_ctrl; separate from first action solver PRE')

    def step_control(self,ctrl20,*,max_substeps=None,hard_stops=None):
        if self.d is None or self.terminated:raise ValueError('Reset required before a fresh live episode')
        u=self._control(ctrl20);m,d,mj=self.m,self.d,self.mj
        n=self.n if max_substeps is None else int(max_substeps)
        if not 1<=n<=self.n:raise ValueError('Positive bounded native interval')
        stops=dict(COM_drift_m=.15,angle_rad=np.pi/2,penetration_m=.005,joint_excursion_rad=.05)
        if hard_stops is not None:
            if set(hard_stops)!=set(stops):raise ValueError('Explicit full stop contract')
            stops={k:float(v) for k,v in hard_stops.items()}
        start=self.tick;source=self.state58();before=self.receipt();d.ctrl[:]=u
        after_action=self.receipt()
        rows={k:[] for k in ['pre_tick','post_tick','pre_native_time_s','post_native_time_s',
            'post_qpos27','post_qvel26','post_state58','pre_pad_normal_N2','pre_phone_nonpad_normal_N',
            'pre_table_normal_N','pre_penetration_m','pre_self_penetration_m','pre_energy2',
            'pre_contact_wrench_about_COM_world6','pre_actuator_force20','pre_qfrc_actuator20']}
        contact=[];termination='COMPLETE_INTERVAL';finite=True;pre_state=source
        for local in range(n):
            tick=self.tick;native_pre=float(d.time)
            mj.mj_step(m,d);self.counts['integration_steps']+=1;self.tick+=1
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or not _bits(d.ctrl,u):
                raise ValueError('No external aid or hidden control mutation')
            finite=all(np.isfinite(a).all() for a in [d.qpos,d.qvel,d.qacc,d.energy,d.actuator_force,d.qfrc_actuator])
            pad=np.zeros(2);other=table=pen=selfpen=0.;wrench=np.zeros(6)
            for ci in range(d.ncon):
                c=d.contact[ci];g1,g2=map(int,c.geom);f=np.empty(6)
                mj.mj_contactForce(m,d,ci,f);self.counts['contact_force_reads']+=1;self.counts['step_contact_force_reads']+=1
                row=np.r_[tick,g1,g2,int(c.dim),int(c.efc_address),c.dist,c.pos,c.frame,f]
                finite=finite and bool(np.isfinite(row).all());contact.append(row)
                pen=max(pen,-float(c.dist))
                if self.phone_geom in (g1,g2):
                    otherg=g2 if g1==self.phone_geom else g1
                    if otherg in self.pads:pad[self.pads.index(otherg)]+=max(0.,float(f[0]))
                    elif otherg==self.table:table+=max(0.,float(f[0]))
                    else:other+=max(0.,float(f[0]))
                    sign=1. if g2==self.phone_geom else -1.
                    R=np.asarray(c.frame).reshape(3,3)
                    F=sign*(R.T@f[:3]);T=sign*(R.T@f[3:])
                    wrench[:3]+=F;wrench[3:]+=np.cross(c.pos-pre_state[40:43],F)+T
                elif self.table not in (g1,g2):selfpen=max(selfpen,-float(c.dist))
            post=self.state58() if finite else np.full(58,np.nan)
            vals=[tick,tick+1,native_pre,float(d.time),d.qpos.copy(),d.qvel.copy(),post.copy(),pad,other,table,
                  pen,selfpen,d.energy.copy(),wrench,d.actuator_force.copy(),d.qfrc_actuator[:20].copy()]
            for key,value in zip(rows,vals):rows[key].append(value)
            if not finite:termination='ABORT_NONFINITE'
            elif np.any(d.warning.number):termination='ABORT_NATIVE_WARNING'
            else:
                drift=np.linalg.norm(post[40:43]-self.initial_state58[40:43])
                angle=2*np.arccos(np.clip(abs(d.qpos[23:27]@self.initial_qpos[23:27]),0,1))
                exc=np.maximum(self.qrange[:,0]-d.qpos[:20],0)+np.maximum(d.qpos[:20]-self.qrange[:,1],0)
                if drift>stops['COM_drift_m']:termination='ABORT_ACQUISITION_COM_BOUND'
                elif angle>stops['angle_rad']:termination='ABORT_ACQUISITION_ANGLE_BOUND'
                elif pen>stops['penetration_m']:termination='ABORT_ACQUISITION_PENETRATION_BOUND'
                elif np.max(exc)>stops['joint_excursion_rad']:termination='ABORT_ACQUISITION_JOINT_BOUND'
            pre_state=post
            if termination!='COMPLETE_INTERVAL':self.terminated=True;break
        arrays={k:np.asarray(v) for k,v in rows.items()}
        arrays['contact_rows24']=np.asarray(contact,dtype=np.float64).reshape(-1,24)
        after=self.receipt();returned=len(rows['pre_tick']);complete=returned==self.n
        coverage=np.mean(arrays['pre_pad_normal_N2']>.1,axis=0)
        first=np.array([next((int(t) for t,f in zip(arrays['pre_tick'],arrays['pre_pad_normal_N2'][:,i]) if f<=.1),-1) for i in range(2)])
        summary=dict(pad_coverage2=coverage,pad_min_normal_N2=np.min(arrays['pre_pad_normal_N2'],axis=0),
            pad_endpoint_normal_N2=arrays['pre_pad_normal_N2'][-1].copy(),
            phone_nonpad_max_normal_N=float(arrays['pre_phone_nonpad_normal_N'].max()),
            first_pad_le_0p1N_global_PRE_tick2=first,
            contact_impulse_world6=np.sum(arrays['pre_contact_wrench_about_COM_world6'],axis=0)*self.h,
            endpoint_force_global_PRE_tick=int(arrays['pre_tick'][-1]))
        if not finite:
            for key in ['pad_coverage2','pad_min_normal_N2','pad_endpoint_normal_N2']:
                summary[key]=np.full(2,np.nan)
            summary['phone_nonpad_max_normal_N']=float('nan')
            summary['contact_impulse_world6']=np.full(6,np.nan)
        target=arrays['post_state58'][-1]
        # This is a complete finite physics target, not a safety/holding gate.
        # A hard-bound event exactly at H is retained with its termination label.
        eligible=bool(complete and finite and not np.any(d.warning.number))
        delta=codec.delta52(source,target) if eligible else np.full(52,np.nan)
        return dict(source_state58=source,actual_control20=u,target_state58=target,
            physical_world_delta52=delta,source_receipt=before,source_after_action_receipt=after_action,target_receipt=after,
            source_global_tick=start,target_global_tick=self.tick,dt_s=self.h,substeps=self.n,
            returned_substeps=returned,complete_H20ms=complete,finite=finite,warning_numbers=np.array(d.warning.number),
            training_eligible=eligible,termination=termination,summary=summary,trace=arrays,
            force_phase='solver_PRE integer ticks; qpos/qvel/state58 POST; no observer forward',
            state_schema='phone_COM_world_state58:1',delta_schema='phone_physical_world_delta52:2')

    def audit(self):
        digest=self.model_digest()
        if digest!=self.initial_digest:raise ValueError('Physical model/options changed during acquisition')
        return dict(passed=True,model_pin=self.model_pin,dt_s=self.h,model_digest=digest,
                    shared_physics_digest_without_dt=self.shared_physics_digest,counts=dict(self.counts))
