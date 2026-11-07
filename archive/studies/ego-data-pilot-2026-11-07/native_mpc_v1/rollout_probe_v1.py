"""SOURCE DRAFT: real native H2 planning branches, never a chosen/main advance.

Each branch mj_copyData-clones its complete boundary state. No forward is used
to restore a branch. Each of two20ms controls is held for128 native steps.
Contact/force/energy after mj_step describe solver PRE, while observe derives
POST kinematics directly from qpos/qvel. Missing contact is not clearance proof.
"""
from pathlib import Path
import hashlib
import json
import time

PILOT=Path(__file__).resolve().parent.parent
DT=.00015625
CONTROL_INTERVAL=.02
STEPS_PER_CONTROL=128
HORIZON=2
SCHEMA='native_H2_planning_rollout_branch:1'
CONTACT_COLUMNS=['pre_tick','geom1','geom2','dim','efc_address','distance_m',
                 'position3','frame9','local_wrench6']


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


class RolloutProbe:
    """Own an immutable model; every branch owns fresh independent MjData."""

    @classmethod
    def from_existing_pilot(cls,*,model_digest_period=128):
        import mujoco
        p=PILOT/'hand2_holding_v1/compiled_nominal_v2/binding.json'
        binding=json.loads(p.read_text(encoding='utf8'))
        if pin(binding['model']['path'])!=binding['model'] or pin(binding['source']['path'])!=binding['source']:
            raise ValueError('Original compiled nominal model/source changed')
        model=mujoco.MjModel.from_binary_path(binding['model']['path'])
        obj=cls(model,model_digest_period=model_digest_period)
        obj.counts['MJB_reads']=1
        obj.model_source={'binding':pin(p),'model':binding['model']}
        return obj

    def __init__(self,model,*,model_digest_period=128):
        import mujoco
        import numpy as np
        self.mj,self.np,self.m=mujoco,np,model
        if mujoco.__version__!='3.11.0' or (model.nq,model.nv,model.nu,model.na,model.neq,model.nmocap,model.nflex,model.nplugin)!=(27,26,20,0,0,0,0,0):
            raise ValueError('Original rigid fixed-wrist20/free-phone native3.11 model only')
        if model.opt.timestep!=DT or CONTROL_INTERVAL/DT!=STEPS_PER_CONTROL:
            raise ValueError('Original fixed fine dt and128 steps/20ms')
        if np.any(model.body_gravcomp):raise ValueError('No gravity compensation')
        for name in ('control','passive','contactfilter','sensor','act_dyn','act_gain','act_bias'):
            if getattr(mujoco,'get_mjcb_'+name)() is not None:
                raise ValueError('No opaque native callbacks')
        self.oid=model.body('hot3d_phone24').id;self.phone=model.geom('phone24_collision_convex').id
        self.table=model.geom('stage1_single_table').id;self.palm=model.body('right_palm_link').id
        self.pads=[model.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)]
        if (int(model.body_parentid[self.oid])!=0 or int(model.body_weldid[self.palm])!=0
                or int(model.jnt_qposadr[model.body_jntadr[self.oid]])!=20
                or int(model.jnt_dofadr[model.body_jntadr[self.oid]])!=20):
            raise ValueError('World-parent free-phone and truly fixed palm layout')
        if type(model_digest_period) is not int or model_digest_period<1:
            raise ValueError('Explicit model hash audit period')
        self.model_digest_period=model_digest_period
        self.counts={'MJB_reads':0,'branches_attempted':0,'branches_returned':0,
            'branch_MjData':0,'mj_copyData':0,'mj_getState':0,
            'planning_steps_attempted':0,'planning_steps_returned':0,'planning_steps_finite':0,
            'contact_force_reads':0,'extra_forward_calls':0,'actual_chosen_steps':0,
            'neural_training_GPU':0,'full_model_digest_checks':1}
        self.model_source=None
        self.model_digest=self._model_digest()

    def _model_digest(self):
        np=self.np;h=hashlib.sha256()
        for scope,obj in (('model',self.m),('option',self.m.opt)):
            for name in sorted(n for n in dir(obj) if not n.startswith('_')):
                value=getattr(obj,name)
                if isinstance(value,np.ndarray):
                    a=np.ascontiguousarray(value);h.update((scope+name+a.dtype.str+str(a.shape)).encode());h.update(a.tobytes())
                elif scope=='option':
                    try:value=float(value)
                    except (TypeError,ValueError):continue
                    h.update((scope+name+repr(value)).encode())
        return h.hexdigest()

    def audit_model(self):
        self.counts['full_model_digest_checks']+=1
        if self._model_digest()!=self.model_digest:raise ValueError('Constant physical model changed')
        return {'passed':True,'model_digest':self.model_digest,'counts':dict(self.counts),
                'model_digest_period':self.model_digest_period,
                'between_audits_no_model_writes':'source-backed, not per-branch raw model proof'}

    def integration_state(self,d):
        np,mj=self.np,self.mj
        sig=int(mj.mjtState.mjSTATE_INTEGRATION)
        v=np.empty(mj.mj_stateSize(self.m,sig),dtype=np.float64)
        mj.mj_getState(self.m,d,v,sig);self.counts['mj_getState']+=1
        return v

    def source_digest(self,d):
        """Full integration state plus selected physical derived/solver bytes.

        No contactForce/forward call; native allocator bookkeeping is excluded.
        Includes integration HISTORY, warmstart, user inputs, eq and time.
        """
        np=self.np;h=hashlib.sha256();v=self.integration_state(d);h.update(v.tobytes())
        for name in ('qacc','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint',
                     'actuator_force','xpos','xmat','xipos','ximat','geom_xpos','geom_xmat',
                     'site_xpos','site_xmat','efc_force','efc_type','efc_id','solver_niter','solver_nnz'):
            a=np.ascontiguousarray(getattr(d,name));h.update((name+a.dtype.str+str(a.shape)).encode());h.update(a.tobytes())
        for a in (np.asarray(d.warning.number),np.asarray(d.warning.lastinfo)):
            h.update(np.ascontiguousarray(a).tobytes())
        for con in d.contact:
            h.update(np.r_[con.geom,con.dim,con.efc_address,con.dist,con.pos,con.frame,con.adhesion].tobytes())
        return {'sha256':h.hexdigest(),'integration_sha256':hashlib.sha256(v.tobytes()).hexdigest(),
                'integration_size':int(v.size),'state_signature':int(self.mj.mjtState.mjSTATE_INTEGRATION)}

    def observe(self,d):
        """Pure POST qpos/qvel observer, never derived PRE body arrays."""
        np=self.np;R=np.empty(9,dtype=np.float64)
        self.mj.mju_quat2Mat(R,d.qpos[23:27]);R=R.reshape(3,3)
        root=d.qpos[20:23].copy();r=R@self.m.body_ipos[self.oid]
        omega=R@d.qvel[23:26]
        return {'COM_world_m':root+r,'R_body_world':R.copy(),
                'v_COM_world_m_s':d.qvel[20:23]+np.cross(omega,r),
                'omega_world_rad_s':omega,'q20':d.qpos[:20].copy(),'qd20':d.qvel[:20].copy(),
                'phone_root_world_m':root,'phone_quaternion_wxyz':d.qpos[23:27].copy(),
                'v_root_world_m_s':d.qvel[20:23].copy()}

    def evaluate(self,source_data,u_controls2x20):
        np,mj,m=self.np,self.mj,self.m
        controls=np.asarray(u_controls2x20,dtype=np.float64).copy()
        if controls.shape!=(HORIZON,20) or not np.isfinite(controls).all():
            raise ValueError('Two finite held absolute control20 vectors')
        limited=m.actuator_ctrllimited.astype(bool)
        if np.any(controls[:,limited]<m.actuator_ctrlrange[limited,0]) or np.any(controls[:,limited]>m.actuator_ctrlrange[limited,1]):
            raise ValueError('Never clamp an unreachable planning control')
        if (m.opt.timestep!=DT or np.any(source_data.qfrc_applied) or np.any(source_data.xfrc_applied)
                or np.any(source_data.warning.number) or not np.isfinite(self.integration_state(source_data)).all()):
            raise ValueError('Clean finite boundary, no applied aid, unchanged dt')
        self.counts['branches_attempted']+=1;branch=self.counts['branches_attempted']
        model_audited=branch==1 or branch%self.model_digest_period==0
        if model_audited:self.audit_model()
        source_before=self.source_digest(source_data);initial=self.observe(source_data)
        start_time=float(source_data.time);palm_pos=source_data.xpos[self.palm].copy();palm_R=source_data.xmat[self.palm].copy()
        d=mj.MjData(m);self.counts['branch_MjData']+=1
        mj.mj_copyData(d,m,source_data);self.counts['mj_copyData']+=1
        clone_equal=self.source_digest(d)==source_before
        if not clone_equal:raise ValueError('Complete physical boundary clone mismatch')
        contacts=[];snapshots=[];unknown=[];attempted=returned=finite_steps=0
        trace={k:[] for k in ('pre_time_nominal_s','post_time_native_s','post_qpos27','post_qvel26',
            'pre_pad_normal_N','pre_phone_nonpad_normal_N','pre_table_normal_N','pre_penetration_m',
            'pre_self_penetration_m','pre_energy','pre_ncon','pre_qfrc_actuator26','pre_solver_niter')}
        aggregates={'max_native_penetration_m':0.,'max_self_penetration_m':0.,
            'max_phone_nonpad_normal_N':0.,'max_table_normal_N':0.,
            'max_joint_limit_excursion_rad':0.,'max_COM_drift_m':0.,
            'minimum_pad_normal_N':np.full(2,np.inf),'maximum_pad_normal_N':np.zeros(2),
            'warning_count':0}
        started=time.perf_counter()
        for control_index in range(HORIZON):
            d.ctrl[:]=controls[control_index]
            for local_tick in range(STEPS_PER_CONTROL):
                tick=control_index*STEPS_PER_CONTROL+local_tick
                attempted+=1;self.counts['planning_steps_attempted']+=1
                mj.mj_step(m,d)
                returned+=1;self.counts['planning_steps_returned']+=1
                if (not all(np.isfinite(v).all() for v in (d.qpos,d.qvel,d.qacc,d.energy,d.qfrc_actuator,d.actuator_force))):
                    unknown.append({'pre_tick':tick,'reason':'NONFINITE_NATIVE_AFTER_PLANNING_STEP'});break
                if (np.any(d.qfrc_applied) or np.any(d.xfrc_applied)
                        or not np.array_equal(d.ctrl,controls[control_index])
                        or not np.array_equal(d.xpos[self.palm],palm_pos)
                        or not np.array_equal(d.xmat[self.palm],palm_R)):
                    unknown.append({'pre_tick':tick,'reason':'APPLIED_FORCE_CONTROL_OR_FIXED_WRIST_MUTATION'});break
                pad=np.zeros(2);other=table=penetration=self_pen=0.
                for ci in range(d.ncon):
                    con=d.contact[ci];g1,g2=map(int,con.geom)
                    force=np.empty(6);mj.mj_contactForce(m,d,ci,force);self.counts['contact_force_reads']+=1
                    contacts.append(np.r_[tick,g1,g2,int(con.dim),int(con.efc_address),con.dist,con.pos,con.frame,force])
                    if not np.isfinite(np.r_[con.dist,con.pos,con.frame,force]).all():
                        unknown.append({'pre_tick':tick,'reason':'NONFINITE_NATIVE_CONTACT'});break
                    penetration=max(penetration,-float(con.dist))
                    if self.phone in (g1,g2):
                        other_geom=g2 if g1==self.phone else g1
                        if other_geom in self.pads:pad[self.pads.index(other_geom)]+=max(0.,float(force[0]))
                        elif other_geom==self.table:table+=max(0.,float(force[0]))
                        else:other+=max(0.,float(force[0]))
                    elif self.table not in (g1,g2):self_pen=max(self_pen,-float(con.dist))
                if unknown:break
                state=self.observe(d)
                excursion=np.maximum(m.jnt_range[:20,0]-d.qpos[:20],0)+np.maximum(d.qpos[:20]-m.jnt_range[:20,1],0)
                aggregates['max_native_penetration_m']=max(aggregates['max_native_penetration_m'],penetration)
                aggregates['max_self_penetration_m']=max(aggregates['max_self_penetration_m'],self_pen)
                aggregates['max_phone_nonpad_normal_N']=max(aggregates['max_phone_nonpad_normal_N'],other)
                aggregates['max_table_normal_N']=max(aggregates['max_table_normal_N'],table)
                aggregates['max_joint_limit_excursion_rad']=max(aggregates['max_joint_limit_excursion_rad'],float(excursion.max()))
                aggregates['max_COM_drift_m']=max(aggregates['max_COM_drift_m'],float(np.linalg.norm(state['COM_world_m']-initial['COM_world_m'])))
                aggregates['minimum_pad_normal_N']=np.minimum(aggregates['minimum_pad_normal_N'],pad)
                aggregates['maximum_pad_normal_N']=np.maximum(aggregates['maximum_pad_normal_N'],pad)
                for key,value in zip(trace,(start_time+tick*DT,float(d.time),d.qpos.copy(),d.qvel.copy(),pad.copy(),other,table,penetration,self_pen,d.energy.copy(),int(d.ncon),d.qfrc_actuator.copy(),np.asarray(d.solver_niter).copy())):
                    trace[key].append(value)
                finite_steps+=1;self.counts['planning_steps_finite']+=1
                if np.any(d.warning.number):
                    unknown.append({'pre_tick':tick,'reason':'NATIVE_WARNING_OR_AUTORESET','warning_number':np.asarray(d.warning.number).copy()});break
                if abs(float(d.time)-(start_time+(tick+1)*DT))>1e-10:
                    unknown.append({'pre_tick':tick,'reason':'NATIVE_CLOCK_RESET_OR_DRIFT'});break
                if local_tick==STEPS_PER_CONTROL-1:
                    boundary_integration=self.integration_state(d)
                    snapshots.append({'post_tick':tick+1,'time_nominal_s':start_time+(tick+1)*DT,
                        'elapsed_time_nominal_s':(tick+1)*DT,'native_time_s':float(d.time),
                        'state':state,'pad_normal_N':pad.copy(),
                        'pad_force_phase':'PRE of final solver tick, not POST geometry force',
                        'pad_force_pre_time_s':start_time+tick*DT,
                        'max_penetration_m':penetration,'max_nonpad_fn':other,'table_normal_N':table,
                        'self_penetration_m':self_pen,
                        'integration_state':boundary_integration,
                        'integration_state_sha256':hashlib.sha256(boundary_integration.tobytes()).hexdigest(),
                        'integration_state_signature':int(mj.mjtState.mjSTATE_INTEGRATION)})
            if unknown:break
        source_after=self.source_digest(source_data)
        if source_after!=source_before:raise ValueError('Planning branch changed physical source state')
        if m.opt.timestep!=DT:raise ValueError('Planning branch changed model timestep')
        if model_audited:self.audit_model()
        aggregates['warning_count']=int(np.sum(d.warning.number))
        if not finite_steps:aggregates['minimum_pad_normal_N']=None
        self.counts['branches_returned']+=1
        final_state=self.integration_state(d)
        return {'schema':SCHEMA,'branch_index':branch,
            'status':'COMPLETE_H2_PLANNING_ONLY' if not unknown and finite_steps==256 else 'UNKNOWN_OR_INCOMPLETE_PLANNING_BRANCH',
            'source_state_before':source_before,'source_state_after':source_after,'source_unchanged':True,
            'complete_clone_equal':clone_equal,'controls2x20':controls,'initial_state':initial,
            'snapshots':snapshots,'aggregates':aggregates,'trace':{k:np.asarray(v) for k,v in trace.items()},
            'contact_rows24':np.asarray(contacts,dtype=np.float64).reshape(-1,24),'contact_row_columns':CONTACT_COLUMNS,
            'final_integration_state':final_state,'final_state_digest':hashlib.sha256(final_state.tobytes()).hexdigest(),
            'final_observation':self.observe(d),'actual_planning_steps':returned,
            'planning_steps_attempted':attempted,'planning_steps_finite':finite_steps,
            'actual_chosen_steps':0,'unknown':unknown,'counts':dict(self.counts),
            'native_time_elapsed_s':float(d.time)-start_time,'wall_seconds':time.perf_counter()-started,
            'model_hash_checked_this_branch':model_audited,
            'scope':'Privileged true-model H2 planning; no chosen/main advance, no trained method benefit or holding qualification.'}
