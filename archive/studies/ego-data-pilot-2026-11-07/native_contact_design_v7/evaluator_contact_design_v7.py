"""SOURCE DRAFT: same-forward native contact Jacobians and raw PD operands.

The frozen v6 evaluator owns native state construction, one forward, zero-speed
semantics, warnings, model auditing, and final independent22/189 geometry.
This subclass only adds read-only mj_jac calls at each actual contact con.pos.
Ideal point-wrench design is not native soft-contact equilibrium or holding.
"""
from pathlib import Path
import hashlib
import importlib.util
import re

PILOT=Path(__file__).resolve().parent.parent
BASE_PATH=PILOT/'native_equilibrium_v6/evaluator_native_v6.py'
BASE_SHA='0a7a15cf16d98066920de3a95dc7b0b1a9faf61d387c24cf43a1b38f1aa2a16f'


def pin(path):
    path=Path(path).resolve();b=path.read_bytes()
    return {'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


if pin(BASE_PATH)['sha256']!=BASE_SHA:
    raise ValueError('Frozen v6 evaluator changed')
_spec=importlib.util.spec_from_file_location('frozen_native_equilibrium_base_for_contact_v7',BASE_PATH)
_base=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class NativeContactDesignEvaluator(_base.NativeEquilibriumEvaluator):
    """Inherited factory works through cls(geometry,digest_check_period=...)."""

    def __init__(self,geometry_evaluator,*,digest_check_period=1):
        super().__init__(geometry_evaluator,digest_check_period=digest_check_period)
        self.counts.update(phone_hand_contact_jac_attempted=0,
                           phone_hand_contact_jac_returned=0,
                           native_contact_design_exports=0)

    def evaluate(self,q20,phone_t_palm3,phone_rotvec3,ctrl20):
        out=super().evaluate(q20,phone_t_palm3,phone_rotvec3,ctrl20)
        if out['unknown']:
            out['contact_design']={'schema':'same_forward_native_phone_hand_contact_design:7',
                'available':False,'phone_hand_contacts':None,'PD_raw':None,
                'reason':'Unknown native evidence retained; no Jacobian or zero-support inference.',
                'new_forward_or_step_calls':0}
            return out
        np,mj,m,d=self.np,self.mj,self.m,self.last_data
        # No new state construction/forward/geometry query is performed here.
        if not np.array_equal(d.qpos,out['raw']['qpos27']) or not np.array_equal(d.ctrl,out['raw']['ctrl20']):
            raise ValueError('Same native forward state must supply contact operands')
        contacts=[]
        hand_geoms=set(self.geometry.hand)
        for ci,detail in enumerate(out['native_contact_details']):
            con=d.contact[ci];g1,g2=map(int,con.geom)
            if self.phone not in (g1,g2):
                continue
            hand_geom=g2 if g1==self.phone else g1
            if hand_geom not in hand_geoms:
                continue
            body=int(m.geom_bodyid[hand_geom])
            jp,jr=np.empty((3,m.nv)),np.empty((3,m.nv))
            position=con.pos.copy()
            self.counts['phone_hand_contact_jac_attempted']+=1
            mj.mj_jac(m,d,jp,jr,position,body)
            self.counts['phone_hand_contact_jac_returned']+=1
            if not np.isfinite(jp).all() or not np.isfinite(jr).all():
                raise ValueError('Finite same-contact hand-body Jacobian required')
            geom_name=m.geom(hand_geom).name;body_name=m.body(body).name
            finger_match=re.match(r'^right_finger([1-5])_',body_name or '')
            finger=int(finger_match.group(1)) if finger_match else None
            sign=1. if g2==self.phone else -1.
            if sign!=detail['force_on_object_sign']:
                raise ValueError('Inherited native contact force convention mismatch')
            contacts.append({'native_contact_index':ci,'geom1':g1,'geom2':g2,
                'phone_geom_id':self.phone,'hand_geom_id':hand_geom,
                'hand_geom_name':geom_name,'hand_body_id':body,'hand_body_name':body_name,
                'finger_index':finger,'finger_group':f'finger{finger}' if finger is not None else None,
                'group_scope':'Source robot body-name anatomical mapping; palm/mount are nonfinger, never silently assigned to a finger budget.',
                'position_world_m':position,'frame_rows_normal_t1_t2_world':con.frame.reshape(3,3).copy(),
                'force_on_object_sign':sign,'normal_force_on_object_world':sign*con.frame[:3],
                'native_friction5':con.friction.copy(),'native_condim':int(con.dim),
                'native_solref2':con.solref.copy(),'native_solreffriction2':con.solreffriction.copy(),
                'native_solimp5':con.solimp.copy(),'native_distance_m':float(con.dist),
                'native_efc_address':int(con.efc_address),'native_exclude':int(con.exclude),
                'native_adhesion':float(con.adhesion),
                'hand_Jp3x20_world':jp[:,:20].copy(),'hand_Jp3x26_world':jp.copy(),
                'jacobian_point_scope':'Actual con.pos world point on the hand geom body; not a closest-surface witness or arbitrary patch centroid.',
                'actual_force_on_object_world_N':detail['force_on_object_world_N'],
                'actual_direct_torque_on_object_world_Nm':detail['direct_torque_on_object_world_Nm']})
        PD_raw={
            'actuator_gainprm20x10':m.actuator_gainprm.copy(),
            'actuator_biasprm20x10':m.actuator_biasprm.copy(),
            'actuator_gear20x6':m.actuator_gear.copy(),
            'actuator_trnid20x2':m.actuator_trnid.copy(),
            'actuator_trntype20':m.actuator_trntype.copy(),
            'actuator_gaintype20':m.actuator_gaintype.copy(),
            'actuator_biastype20':m.actuator_biastype.copy(),
            'actuator_dyntype20':m.actuator_dyntype.copy(),
            'actuator_group20':m.actuator_group.copy(),
            'actuator_ctrllimited20':m.actuator_ctrllimited.copy(),
            'actuator_ctrlrange20x2':m.actuator_ctrlrange.copy(),
            'actuator_forcelimited20':m.actuator_forcelimited.copy(),
            'actuator_forcerange20x2':m.actuator_forcerange.copy(),
            'joint_actfrclimited20':m.jnt_actfrclimited[:20].copy(),
            'joint_actfrcrange20x2':m.jnt_actfrcrange[:20].copy(),
            'joint_range20x2':m.jnt_range[:20].copy(),
            'actuator_length20':d.actuator_length.copy(),
            'actuator_velocity20':d.actuator_velocity.copy(),
            'PD_gain20':m.actuator_gainprm[:,0].copy(),
            'PD_offset20':(m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*d.actuator_length
                           +m.actuator_biasprm[:,2]*d.actuator_velocity),
            'requested_ctrl20':d.ctrl.copy(),
            'actual_actuator_force20':d.actuator_force.copy(),
            'actual_generalized_actuator_force26':d.qfrc_actuator.copy(),
            'qfrc_bias26':d.qfrc_bias.copy(),'qfrc_passive26':d.qfrc_passive.copy(),
            'qfrc_constraint26':d.qfrc_constraint.copy(),
            'dof_armature26':m.dof_armature.copy(),'dof_damping26':m.dof_damping.copy(),
            'dof_frictionloss26':m.dof_frictionloss.copy(),
            'native_disableflags':int(m.opt.disableflags),
            'native_disabled_actuator_group_bitmask':int(m.opt.disableactuator),
            'direct_unit_joint_mapping_verified_by_frozen_factory':True,
            'scope':'Raw actual arrays; caller must distinguish unclamped PD range, actuator force clipping and generalized joint-force clipping. No static friction force is assumed available.'}
        if not np.array_equal(d.qpos,out['raw']['qpos27']) or not np.array_equal(d.ctrl,out['raw']['ctrl20']):
            raise ValueError('Read-only Jacobian export mutated native physical input')
        self.counts['native_contact_design_exports']+=1
        out['contact_design']={'schema':'same_forward_native_phone_hand_contact_design:7',
            'available':True,
            'phone_hand_contacts':contacts,'phone_hand_contact_count':len(contacts),
            'phone_COM_world_m':out['raw']['phone_COM_world_m'].copy(),
            'PD_raw':PD_raw,'positive_normal_force_filter_applied':False,
            'new_forward_or_step_calls':0,
            'scope':'Same-forward actual contact points/frames/friction and hand Jacobians for an ideal wrench candidate; no implicit law, torque-feasibility or holding qualification.'}
        out['counts']=dict(self.counts)
        return out
