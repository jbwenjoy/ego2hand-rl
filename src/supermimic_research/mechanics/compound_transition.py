"""50Hz compound-object transitions with POST observations and solver-PRE labels.

This new state55/controller contract does not reuse frozen phone state58 or its
checkpoints. Full integration state is retained for exact finite-solver replay.
"""
import numpy as np
from .compound_contacts import CompoundContacts
from .compound_holding import object_state
from .residual_control import ResidualEMAController
from ..artifacts import verify_file_pin
from ..runtime.native_evaluation import model_digest_except_dt,DTS


def observe55(qpos,qvel,COM_BODY):
    import mujoco
    center,quat,velocity,omega=object_state(None,qpos,qvel,20,20,COM_BODY)
    matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,quat);R=matrix.reshape(3,3)
    state=np.r_[qpos[:20],qvel[:20],center,R[:,0],R[:,1],velocity,omega]
    if not np.isfinite(state).all():raise ValueError('Nonfinite actual POST state55')
    return state


class CompoundTransitionReader:
    def __init__(self,binding,qpos,base_control,*,dt_s):
        import mujoco
        self.mj=mujoco;self.binding=binding;self.model=mujoco.MjModel.from_binary_path(str(verify_file_pin(binding['model'])));m=self.model
        if dt_s not in DTS or (m.nq,m.nv,m.nu,m.na,m.neq,m.nmocap,m.nplugin,m.nflex,m.ntendon)!=(27,26,20,0,0,0,0,0,0):raise ValueError('Declared compound no-aid layout/timestep')
        for name in ['control','passive','contactfilter','sensor','act_dyn','act_gain','act_bias']:
            if getattr(mujoco,'get_mjcb_'+name)() is not None:raise ValueError('No opaque native callback')
        expected=[f'right_finger{i}_joint{j}' for i in range(1,6) for j in range(1,5)]
        if [m.joint(i).name for i in range(20)]!=expected or not np.array_equal(m.actuator_trnid[:,0],np.arange(20)) or not np.array_equal(m.actuator_gear[:,0],np.ones(20)):raise ValueError('Named direct joint order')
        if binding['mug_free_qpos']!=20 or binding['mug_free_dof']!=20 or np.any(m.body_gravcomp):raise ValueError('Explicit free BODY and no gravity aid')
        self.oid=binding['mug_body_id'];self.groups={i:[m.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]}
        self.contacts=CompoundContacts(m,binding['mug_part_geom_ids'],self.oid,self.groups,m.geom('mug_task_table').id)
        self.COM=m.body_ipos[self.oid].copy();self.dt=float(dt_s);self.n=round(.02/self.dt);m.opt.timestep=self.dt
        if self.n*self.dt!=.02:raise ValueError('Exact integer50Hz native tick plan')
        self.physics=model_digest_except_dt(m,np);self.controller=ResidualEMAController(base_control,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        self.initial_q=np.array(qpos,float);self.d=mujoco.MjData(m);mujoco.mj_resetData(m,self.d)
        if self.initial_q.shape!=(27,) or not np.isfinite(self.initial_q).all() or np.any(self.initial_q[:20]<m.jnt_range[:20,0]) or np.any(self.initial_q[:20]>m.jnt_range[:20,1]):raise ValueError('Exact legal initial q')
        self.d.qpos[:]=self.initial_q;self.d.qvel[:]=0;self.d.ctrl[:]=self.controller.base;self.d.qacc_warmstart[:]=0;mujoco.mj_forward(m,self.d)
        self.initial_COM,self.initial_quat,*_=object_state(m,self.initial_q,np.zeros(26),20,20,self.COM)
        self.signature=int(mujoco.mjtState.mjSTATE_INTEGRATION);self.size=mujoco.mj_stateSize(m,self.signature);self.tick=0;self.k=0;self.terminated=False

    def observe(self):return observe55(self.d.qpos,self.d.qvel,self.COM)

    def verify_physics(self):
        if model_digest_except_dt(self.model,np)!=self.physics:raise ValueError('Physical model changed')

    def snapshot(self):
        state=np.empty(self.size);self.mj.mj_getState(self.model,self.d,state,self.signature)
        return dict(schema='compound_transition_full_state:1',physics=self.physics,dt=self.dt,integration_state=state,controller=self.controller.state_dict(),tick=self.tick,k=self.k,terminated=self.terminated,initial_q=self.initial_q.copy())

    def restore(self,snapshot):
        if snapshot['schema']!='compound_transition_full_state:1' or snapshot['physics']!=self.physics or snapshot['dt']!=self.dt or not np.array_equal(snapshot['initial_q'],self.initial_q):raise ValueError('Reader restore contract mismatch')
        values=np.asarray(snapshot['integration_state'])
        if values.shape!=(self.size,) or not np.isfinite(values).all():raise ValueError('Invalid full native integration state')
        self.mj.mj_resetData(self.model,self.d);self.mj.mj_setState(self.model,self.d,values,self.signature)
        self.controller.restore(snapshot['controller']);self.tick=int(snapshot['tick']);self.k=int(snapshot['k']);self.terminated=bool(snapshot['terminated'])
        # No mj_forward: the next step1 refreshes derived fields, and observe55
        # reads only qpos/qvel. A restore must not alter saved solver warmstart.

    def step(self,action):
        if self.terminated or self.k>=100:raise ValueError('Terminal/reference-end state cannot be advanced')
        m,d,mj=self.model,self.d,self.mj;source=self.observe();warm_before=d.qacc_warmstart.copy();full_before=self.snapshot();decision=self.controller.step(action)
        u=decision['actual_control20'];source_R=np.empty(9);mj.mju_quat2Mat(source_R,d.qpos[23:27]);source_R=source_R.reshape(3,3)
        trace={k:[] for k in ['pre_qpos','pre_qvel','post_qpos','post_qvel','post_COM','post_quat','post_COM_velocity','post_omega_world',
            'pre_group_normal_N','pre_other_normal_N','pre_table_normal_N','pre_penetration_m','pre_self_penetration_m','pre_wrench_BODY_COM','pre_object_COM','post_limit_excursion_rad']}
        raw=[];world_wrenches=[];Fn=[];reason=None;stride=round(.01/self.dt);rne_errors=[]
        for _ in range(self.n):
            mj.mj_step1(m,d);preq,prev=d.qpos.copy(),d.qvel.copy();d.ctrl[:]=u
            if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise ValueError('Applied aid prohibited')
            mj.mj_step2(m,d);s=self.contacts.read(d,raw=True)
            if len(s['raw23']):raw.extend(np.c_[np.full(len(s['raw23']),self.tick),s['raw23']])
            if self.tick%stride==0:
                mj.mj_rnePostConstraint(m,d);error=float(abs(np.r_[s['torque_about_COM_world_Nm'],s['force_world_N']]-d.cfrc_ext[self.oid]).max());rne_errors.append(error)
                if error>1e-8:raise ValueError('Native external wrench/COM mismatch')
            center,quat,velocity,omega=object_state(m,d.qpos,d.qvel,20,20,self.COM)
            excursion=float(max(0.,np.max(m.jnt_range[:20,0]-d.qpos[:20]),np.max(d.qpos[:20]-m.jnt_range[:20,1])))
            values=[preq,prev,d.qpos.copy(),d.qvel.copy(),center,quat,velocity,omega,s['normal_by_finger_N'][:3],s['other_normal_N'],s['table_normal_N'],
                s['max_native_penetration_m'],s['max_self_penetration_m'],s['wrench_BODY_about_COM'],s['object_COM_world_m'],excursion]
            for name,value in zip(trace,values):trace[name].append(value)
            world_wrenches.append(np.r_[s['force_world_N'],s['torque_about_COM_world_Nm']]);Fn.append(s['normal_by_finger_N'][:3]);self.tick+=1
            if not np.isfinite(np.r_[d.qpos,d.qvel,d.qacc]).all():reason='NONFINITE'
            elif np.any(d.warning.number):reason='NATIVE_WARNING'
            elif np.linalg.norm(center-self.initial_COM)>.05:reason='COM_DRIFT_OVER_50MM'
            elif 2*np.arccos(np.clip(abs(quat@self.initial_quat),0,1))>np.pi/4:reason='ORIENTATION_OVER_45DEG'
            if reason is not None:break
        self.k+=1;self.terminated=reason is not None;trace={k:np.array(v) for k,v in trace.items()};count=len(world_wrenches)
        world=np.mean(world_wrenches,axis=0);body=np.r_[source_R.T@world[:3],source_R.T@world[3:]]
        return dict(schema='compound_H20_transition:1',source55=source,next55=self.observe(),warmstart26_before=warm_before,warmstart26_after=d.qacc_warmstart.copy(),
            decision=decision,k=self.k-1,source_native_state=full_before['integration_state'],next_native_state=self.snapshot()['integration_state'],
            complete_H=count==self.n,terminated=self.terminated,termination=reason,actual_duration_s=count*self.dt,executed_native_steps=count,
            mean_native_wrench_world_COM=world,mean_native_wrench_source_BODY_COM=body,mean_native_group_Fn_N=np.mean(Fn,axis=0),
            raw_contact_tick_plus23=np.array(raw).reshape(-1,24),trace=trace,rne_external_wrench_errors=np.array(rne_errors),
            source_contact_truth_known=False,native_simulated_contact_labels_known=True)
