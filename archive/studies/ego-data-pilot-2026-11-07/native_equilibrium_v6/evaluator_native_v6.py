"""SOURCE DRAFT: independent zero-speed native forward probes, never integration.

Construction explicitly loads the unchanged nominal explicit-pair model through
the frozen static evaluator. Each evaluate owns a fresh MjData, zero warmstart,
and exactly one mj_forward. Generated contacts are not global clearance proof.
No backend/model loading occurs on import. Caller owns the frozen protocol.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

PILOT = Path(__file__).resolve().parent.parent
SCHEMA = 'native_zero_speed_equilibrium_probe:6'
STATIC_SOURCE_SHA = '090cf6535f5249de4377a0e022ed7c87b555ad69ed7b570f995badb4d2245d57'
CONTACT_COLUMNS = ['probe_index', 'geom1', 'geom2', 'dim', 'efc_address',
                   'distance_m', 'position3', 'frame9', 'local_wrench6']


def pin(path):
    path = Path(path).resolve(strict=True)
    b = path.read_bytes()
    return {'path': str(path), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


class NativeEquilibriumEvaluator:
    """Fixed model, fresh native state per call; final geometry is a separate call."""

    @classmethod
    def from_existing_pilot(cls, *, max_heavy_queries=20000, deadline_monotonic=None,
                            digest_check_period=1):
        path = PILOT/'retarget_gravity_v3/evaluator_v3.py'
        if pin(path)['sha256'] != STATIC_SOURCE_SHA:
            raise ValueError('Frozen independent geometry source changed')
        spec = importlib.util.spec_from_file_location('frozen_static_geometry_for_native_v6', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        geometry = module.StaticRetargetEvaluator.from_existing_pilot(
            max_heavy_queries=max_heavy_queries, deadline_monotonic=deadline_monotonic)
        return cls(geometry,digest_check_period=digest_check_period)

    def __init__(self, geometry_evaluator, *, digest_check_period=1):
        import mujoco
        import numpy as np
        from scipy.spatial.transform import Rotation
        self.mj, self.np, self.Rotation = mujoco, np, Rotation
        self.geometry = geometry_evaluator
        self.m = geometry_evaluator.m
        self.initial = {k:np.asarray(v).copy() for k,v in geometry_evaluator.initial.items()}
        self.Rp, self.pp = geometry_evaluator.Rp.copy(), geometry_evaluator.pp.copy()
        self.oid, self.pid = geometry_evaluator.oid, geometry_evaluator.pid
        self.phone, self.table = geometry_evaluator.phone, geometry_evaluator.table
        self.pads = list(geometry_evaluator.pads)
        if type(digest_check_period) is not int or digest_check_period<1:
            raise ValueError('Explicit positive integer model digest period')
        self.digest_check_period=digest_check_period
        m = self.m
        if mujoco.__version__ != '3.11.0' or (m.nq,m.nv,m.nu,m.na,m.neq,m.nmocap,m.nflex,m.nplugin,m.ntendon) != (27,26,20,0,0,0,0,0,0):
            raise ValueError('Original native3.11 rigid fixed-wrist20/free-phone layout')
        if np.any(m.body_gravcomp):
            raise ValueError('No body gravity compensation')
        self._assert_callbacks_absent()
        binding_path = PILOT/'hand2_holding_v1/compiled_nominal_v2/binding.json'
        self.binding = json.loads(binding_path.read_text(encoding='utf8'))
        support = json.loads((PILOT/'hand2_holding_v1/support_preparation_v1/actual_v1/report.json').read_text(encoding='utf8'))
        if (self.binding['model'] != support['input_pins']['compiled_model']
                or pin(binding_path) != support['input_pins']['binding']
                or pin(self.binding['model']['path']) != self.binding['model']):
            raise ValueError('Same exact explicit-pair nominal model identity')
        self.counts = {'MJB_reads':1, 'geometry_MjData_construction':1,
            'native_evaluations_attempted':0, 'native_evaluations_returned':0,
            'native_MjData':0, 'mj_forward_attempted':0, 'mj_forward_returned':0,
            'COM_mj_jac':0, 'contact_force_reads':0, 'final_geometry_evaluations':0,
            'integration_steps':0, 'training_GPU':0, 'full_model_digest_checks':1}
        descendants=[]
        for b in range(1,m.nbody):
            parent=b
            while parent and parent!=self.oid:
                parent=int(m.body_parentid[parent])
            if parent==self.oid and b!=self.oid and m.body_mass[b]>0:
                descendants.append(b)
        if descendants or np.any(m.dof_armature[20:26]) or int(m.jnt_type[m.body_jntadr[self.oid]])!=int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError('Isolated massive free-phone body with zero six-DOF armature')
        self.model_configuration = {'nq':int(m.nq),'nv':int(m.nv),'nu':int(m.nu),
            'na':int(m.na),'neq':int(m.neq),'nmocap':int(m.nmocap),'nflex':int(m.nflex),
            'nplugin':int(m.nplugin),'ntendon':int(m.ntendon),
            'phone_massive_descendants':descendants,'phone_dof_armature6':m.dof_armature[20:26].copy(),
            'phone_mass_kg':float(m.body_mass[self.oid]),'phone_body_inertia3':m.body_inertia[self.oid].copy(),
            'phone_body_iquat_wxyz':m.body_iquat[self.oid].copy(),
            'options':{name:float(getattr(m.opt,name)) for name in ('timestep','tolerance','ls_tolerance','iterations','ls_iterations','integrator','solver','cone','disableflags','enableflags')}}
        self.model_digest = self._model_digest()
        self.last_data = None

    def _assert_callbacks_absent(self):
        for name in ('passive','control','contactfilter','sensor','act_dyn','act_gain','act_bias'):
            getter = getattr(self.mj, 'get_mjcb_'+name, None)
            if getter is None or getter() is not None:
                raise ValueError('Absent standard native callback required: '+name)

    def _model_digest(self):
        """All exposed model ndarray bytes and numeric options, without mutation."""
        np = self.np
        h = hashlib.sha256()
        for scope, obj in (('model',self.m), ('option',self.m.opt)):
            for name in sorted(n for n in dir(obj) if not n.startswith('_')):
                value = getattr(obj, name)
                if isinstance(value, np.ndarray):
                    a = np.ascontiguousarray(value)
                    h.update((scope+':'+name+':'+a.dtype.str+':'+str(a.shape)).encode())
                    h.update(a.tobytes())
                elif scope == 'option':
                    try:
                        scalar = float(value)
                    except (TypeError,ValueError):
                        continue
                    h.update((scope+':'+name+':'+repr(scalar)).encode())
        return h.hexdigest()

    def _check_model(self, probe_index, *, force=False):
        checked=bool(force or self.digest_check_period==1 or probe_index==1
                     or probe_index%self.digest_check_period==0)
        if checked:
            if self._model_digest()!=self.model_digest:
                raise ValueError('Constant model changed')
            self.counts['full_model_digest_checks']+=1
        return checked

    def audit_model(self):
        """Mandatory caller endpoint audit for periodic-hash optimization use."""
        self._check_model(self.counts['native_evaluations_attempted'],force=True)
        return {'model_digest':self.model_digest,'counts':dict(self.counts),
            'digest_check_period':self.digest_check_period,
            'between_hashes_no_model_writes':'source-backed only; not per-probe saved raw hash proof'}

    def _pose(self, q20, t3, rv3, ctrl20):
        np = self.np
        q,t,rv,ctrl = [np.asarray(v,dtype=np.float64).copy() for v in (q20,t3,rv3,ctrl20)]
        if (q.shape!=(20,) or t.shape!=(3,) or rv.shape!=(3,) or ctrl.shape!=(20,)
                or not np.isfinite(np.r_[q,t,rv,ctrl]).all()):
            raise ValueError('Finite20q/3translation_m/3rotvec_rad/20ctrl')
        if np.any(q<self.m.jnt_range[:20,0]) or np.any(q>self.m.jnt_range[:20,1]):
            raise ValueError('Original hard joint limits')
        R = self.Rp@self.Rotation.from_rotvec(rv).as_matrix()
        p = self.pp+self.Rp@t
        qpos = self.initial['qpos'].copy()
        qpos[:20], qpos[20:23] = q,p
        qpos[23:27] = self.Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        return q,t,rv,ctrl,qpos

    def evaluate(self, q20, phone_t_palm3, phone_rotvec3, ctrl20):
        """One fresh-data native forward. No geometry/LP/step is hidden here.

        All accelerations/forces/contact frames refer to this same zero-speed
        native state. body root xpos differs from inertial COM xipos. World
        COM/angular acceleration uses native mj_jac at COM and Jdot*v=0.
        """
        np,mj,m = self.np,self.mj,self.m
        checked_before=self._check_model(self.counts['native_evaluations_attempted']+1)
        self._assert_callbacks_absent()
        q,t,rv,ctrl,qpos = self._pose(q20,phone_t_palm3,phone_rotvec3,ctrl20)
        self.counts['native_evaluations_attempted'] += 1
        token = self.counts['native_evaluations_attempted']
        d = mj.MjData(m); self.counts['native_MjData'] += 1
        d.qpos[:] = qpos; d.qvel[:] = 0; d.ctrl[:] = ctrl; d.time = 0
        d.act[:] = 0; d.qacc[:] = 0; d.qacc_warmstart[:] = 0
        d.qfrc_applied[:] = 0; d.xfrc_applied[:] = 0
        warm_before = d.qacc_warmstart.copy()
        self.counts['mj_forward_attempted'] += 1
        mj.mj_forward(m,d)
        self.counts['mj_forward_returned'] += 1
        if (not np.array_equal(d.qpos,qpos) or np.any(d.qvel) or np.any(d.qfrc_applied)
                or np.any(d.xfrc_applied) or not np.array_equal(d.ctrl,ctrl)
                or not np.array_equal(d.xpos[self.pid],self.pp)
                or not np.array_equal(d.xmat[self.pid].reshape(3,3),self.Rp)):
            raise ValueError('Fixed pose/wrist/zero speed/applied force/controller invariant')
        COM = d.xipos[self.oid].copy()
        jp,jr = np.empty((3,m.nv)),np.empty((3,m.nv))
        mj.mj_jac(m,d,jp,jr,COM,self.oid); self.counts['COM_mj_jac'] += 1
        clampctrl = not (int(m.opt.disableflags)&int(mj.mjtDisableBit.mjDSBL_CLAMPCTRL))
        ctrl_effective = ctrl.copy()
        limited = m.actuator_ctrllimited.astype(bool)
        if clampctrl:
            ctrl_effective[limited] = np.clip(ctrl_effective[limited],m.actuator_ctrlrange[limited,0],m.actuator_ctrlrange[limited,1])
        offset = m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*d.actuator_length+m.actuator_biasprm[:,2]*d.actuator_velocity
        raw_pd = m.actuator_gainprm[:,0]*ctrl_effective+offset
        actuator_limited = raw_pd.copy(); mask = m.actuator_forcelimited.astype(bool)
        actuator_limited[mask] = np.clip(actuator_limited[mask],m.actuator_forcerange[mask,0],m.actuator_forcerange[mask,1])
        joint_limited = actuator_limited.copy(); mask = m.jnt_actfrclimited[:20].astype(bool)
        joint_limited[mask] = np.clip(joint_limited[mask],m.jnt_actfrcrange[:20][mask,0],m.jnt_actfrcrange[:20][mask,1])
        rows,details = [],[]
        aggregate = {k:{'contact_count':0,'normal_force_signed_N':0.,'force_world_N':np.zeros(3),
                       'direct_torque_world_Nm':np.zeros(3),'torque_about_COM_world_Nm':np.zeros(3),
                       'minimum_generated_distance_m':None}
                     for k in ('thumb_pad','index_pad','phone_nonpad','phone_table','self_or_other')}
        aggregate['self_or_other'].update(force_world_N=None,direct_torque_world_Nm=None,torque_about_COM_world_Nm=None)
        for ci in range(d.ncon):
            con = d.contact[ci]; g1,g2 = map(int,con.geom)
            force = np.empty(6); mj.mj_contactForce(m,d,ci,force)
            self.counts['contact_force_reads'] += 1
            rows.append(np.r_[token,g1,g2,int(con.dim),int(con.efc_address),con.dist,con.pos,con.frame,force])
            object_boundary = self.phone in (g1,g2)
            if object_boundary:
                other = g2 if g1==self.phone else g1
                key = ('thumb_pad' if other==self.pads[0] else 'index_pad' if other==self.pads[1]
                       else 'phone_table' if other==self.table else 'phone_nonpad')
                sign = 1. if g2==self.phone else -1.
                f = sign*con.frame.reshape(3,3).T@force[:3]
                direct = sign*con.frame.reshape(3,3).T@force[3:]
                torque = np.cross(con.pos-COM,f)+direct
            else:
                key='self_or_other'; sign=None; f=direct=torque=None
            a = aggregate[key]; a['contact_count'] += 1
            a['normal_force_signed_N'] += float(force[0])
            a['minimum_generated_distance_m'] = float(con.dist) if a['minimum_generated_distance_m'] is None else min(a['minimum_generated_distance_m'],float(con.dist))
            if object_boundary:
                a['force_world_N'] += f; a['direct_torque_world_Nm'] += direct; a['torque_about_COM_world_Nm'] += torque
            details.append({'contact_index':ci,'geom1_name':m.geom(g1).name,'geom2_name':m.geom(g2).name,
                'category':key,'object_boundary':object_boundary,'force_on_object_sign':sign,
                'constraint_address_nonnegative':bool(con.efc_address>=0),
                'native_exclude':int(con.exclude),'native_adhesion':float(con.adhesion),'force_on_object_world_N':f,
                'direct_torque_on_object_world_Nm':direct,'torque_about_COM_world_Nm':torque})
        contacts = np.asarray(rows,dtype=np.float64).reshape(-1,24)
        raw = {'qpos27':d.qpos.copy(),'qvel26':d.qvel.copy(),'ctrl20':d.ctrl.copy(),
            'qacc26':d.qacc.copy(),'qacc_warmstart_before26':warm_before,
            'qacc_warmstart_after26':d.qacc_warmstart.copy(),
            'qfrc_bias26':d.qfrc_bias.copy(),'qfrc_actuator26':d.qfrc_actuator.copy(),
            'qfrc_passive26':d.qfrc_passive.copy(),'qfrc_constraint26':d.qfrc_constraint.copy(),
            'qfrc_smooth26':d.qfrc_smooth.copy(),
            'qfrc_applied26':d.qfrc_applied.copy(),'xfrc_applied':d.xfrc_applied.copy(),
            'actuator_force20':d.actuator_force.copy(),'actuator_length20':d.actuator_length.copy(),
            'actuator_velocity20':d.actuator_velocity.copy(),'phone_root_world_m':d.xpos[self.oid].copy(),
            'phone_body_R_world':d.xmat[self.oid].reshape(3,3).copy(),
            'phone_COM_world_m':COM,'phone_inertial_R_world':d.ximat[self.oid].reshape(3,3).copy(),
            'phone_I_world_3x3':d.ximat[self.oid].reshape(3,3)@np.diag(m.body_inertia[self.oid])@d.ximat[self.oid].reshape(3,3).T,
            'COM_Jp3x26':jp,'COM_Jr3x26':jr,'a_COM_world_m_s2':jp@d.qacc,
            'alpha_world_rad_s2':jr@d.qacc,'contact_rows24':contacts,
            'efc_type':d.efc_type[:d.nefc].copy(),'efc_id':d.efc_id[:d.nefc].copy(),
            'efc_force':d.efc_force[:d.nefc].copy()}
        finite = all(np.isfinite(v).all() for v in raw.values())
        warnings = np.asarray(d.warning.number).copy()
        unknown = [] if finite and not np.any(warnings) else ['NONFINITE_NATIVE_OPERANDS_OR_NATIVE_WARNING']
        if int(d.efm_active)!=0:
            unknown.append('ACTIVE_EFFECTIVE_METRIC_OUTSIDE_RIGID_ZERO_METRIC_CONTRACT')
        checked_after=self._check_model(token)
        self.counts['native_evaluations_returned'] += 1
        self.last_data = d
        return {'schema':SCHEMA,'status':'FINITE_NATIVE_FORWARD_NOT_EQUILIBRIUM_CERTIFICATE' if not unknown else 'UNKNOWN_NATIVE_EVIDENCE',
            'probe_index':token,'phone_t_palm_m':t,'phone_rotvec_rad':rv,'raw':raw,
            'native_contact_details':details,'native_contact_aggregates':aggregate,
            'native_contact_scope':'Generated native contacts only; missing pair has unknown global clearance, never zero distance/force proof. No positive Fn filter.',
            'contact_row_columns':CONTACT_COLUMNS,'PD':{'requested_ctrl20':ctrl,'ctrl_clamp_enabled':clampctrl,
                'effective_ctrl20':ctrl_effective,'raw_affine_force20':raw_pd,'actuator_limited_force20':actuator_limited,
                'joint_limited_generalized_force20':joint_limited,
                'actual_actuator_force20':d.actuator_force.copy(),'actual_qfrc_actuator20':d.qfrc_actuator[:20].copy(),
                'actuator_formula_difference20':d.actuator_force-actuator_limited,
                'joint_formula_difference20':d.qfrc_actuator[:20]-joint_limited},
            'solver_niter':np.asarray(d.solver_niter).copy(),'solver_nnz':np.asarray(d.solver_nnz).copy(),
            'warnings_number':warnings,'warnings_lastinfo':np.asarray(d.warning.lastinfo).copy(),
            'native_contact_count':int(d.ncon),'native_constraint_count':int(d.nefc),
            'native_constraint_partition':{'ne':int(d.ne),'nf':int(d.nf),'nl':int(d.nl)},
            'efm_active':int(d.efm_active),'model_configuration':self.model_configuration,
            'qfrc_constraint_scope':'All native constraints including contact, joint limits and frictionloss, not an HO-only wrench',
            'zero_velocity_Jdot_v_term':True,'cacc_used':False,'unknown':unknown,
            'model_parameter_digest':self.model_digest,'counts':dict(self.counts),
            'model_hash_evidence':{'period':self.digest_check_period,'before_probe_checked':checked_before,
                'after_probe_checked':checked_after,'between_checks_no_writes':'source-backed only'},
            'equilibrium_or_holding_qualified':False,'integration_training_GPU':0}

    def final_geometry(self, q20, phone_t_palm3, phone_rotvec3, *, pad_gap_window_m,
                       nonpad_buffer_m=0., self_buffer_m=0.):
        """Independent full22 phone / full189 eligible-self check; no native step.

        All phone pairs forced exact; far self pairs retain separation lower-
        bound certificates. Any unknown fails closed. Pad compression window is
        caller's new hypothesis, never a hardware/compliance qualification.
        """
        self._check_model(self.counts['native_evaluations_attempted'],force=True)
        e=self.geometry
        s=e.evaluate(q20,phone_t_palm3,phone_rotvec3,exact_all=False)
        for row in s['phone_geometry_rows']:
            if row['status']=='DISJOINT_AABB_CERTIFICATE':
                identity={k:row[k] for k in ('geom_id','name','is_pad')}
                exact=e._describe_pair({'kind':'convex','v':e.phone_vertices},
                    e._shape(row['geom_id'],s['phone_R_world'],s['phone_p_world_m']),phone=True,force_exact=True)
                row.clear();row.update(identity);row.update(exact)
        self.counts['final_geometry_evaluations']+=1
        complete=len(s['phone_geometry_rows'])==22 and len(s['self_geometry_rows'])==189 and not s['unknown']
        constraints=e.constraint_vector(s,pad_gap_window_m=pad_gap_window_m,
            nonpad_buffer_m=nonpad_buffer_m,self_buffer_m=self_buffer_m)
        self._check_model(self.counts['native_evaluations_attempted'],force=True)
        return {'state':s,'constraints':constraints,'complete22_189':complete,
            'geometry_gate_pass':bool(complete and constraints['all_satisfied']),
            'new_pad_gap_hypothesis_m':list(pad_gap_window_m),'geometry_counts':dict(e.counts),
            'native_counts':dict(self.counts),'holding_qualified':False,'integration_steps':0}
