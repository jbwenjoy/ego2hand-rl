"""Read-only native packets for owned-runtime same-source control probes."""
import copy
import hashlib
import numpy as np
import mujoco
import mujoco_warp as mw
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.warp_contact_labels import read_warp_contact_labels
from supermimic_research.learning.pen_support_ppo import static_contract
from . import owned_pen_state as native
from .pen_reference_cache import tensor_digest
from .pen_portable_state import scalar_equal


def perturbed_actor_raw(base,direction,epsilon,sign):
    base=np.asarray(base);direction=np.asarray(direction)
    if base.shape!=(20,) or direction.shape!=(20,) or epsilon<0 or sign not in [-1,0,1]:raise ValueError('Explicit raw20/unit direction/amplitude/sign')
    if epsilon and abs(np.linalg.norm(direction)-1)>1e-8:raise ValueError('Unit direction required')
    raw=(base+4*sign*epsilon*direction).astype(np.float32)
    if not np.isfinite(raw).all() or np.any(abs(raw)>1):raise ValueError('Probe would clip total raw action; preserve exclusion')
    return raw


def source_frame_gap(qpos,qa,cached_R,cached_COM):
    R=Rotation.from_quat(np.asarray(qpos)[qa+np.array([4,5,6,3])]).as_matrix();COM=np.asarray(qpos)[qa:qa+3].copy()
    return dict(qpos_R=R,qpos_COM=COM,cached_rotation_gap_rad=float(Rotation.from_matrix(R.T@cached_R).magnitude()),
        cached_COM_gap_m=float(np.linalg.norm(COM-cached_COM)))


def restore_source(runtime,source):
    if source['schema']!='owned_trained_actor_probe_source:1':raise ValueError('Explicit probe-source schema')
    contract=static_contract(runtime,native.inventory(runtime))
    if contract!=source['task_static']:raise ValueError('Immutable task contract differs')
    p=source['native']['python']
    for path,digest in contract['tensors'].items():
        if tensor_digest(p['tensors'][path])!=digest:raise ValueError('Saved probe target tensor corrupted')
    for path,digest in contract['arrays'].items():
        v=p['arrays'][path]
        if hashlib.sha256(str((v.shape,v.dtype)).encode()+v.tobytes()).hexdigest()!=digest:raise ValueError('Saved probe target array corrupted')
    for path,value in contract['scalars'].items():
        if not scalar_equal(p['scalars'][path],value):raise ValueError('Saved probe task scalar corrupted')
    return native.restore(runtime,source['native'])


def save_physics(runtime,folder):
    sim=runtime.env.sim;model=copy.copy(sim.mj_model);physical={}
    for key in ['body_mass','body_inertia','body_ipos','body_iquat','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp',
                'actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange','jnt_range','dof_damping','dof_armature']:
        value=getattr(sim.wp_model,key).numpy().copy();physical[key]=value;target=getattr(model,key);target[:]=value if value.shape==target.shape else value[0]
    physical['gravity']=sim.wp_model.opt.gravity.numpy().copy()
    mujoco.mj_setConst(model,mujoco.MjData(model));mujoco.mj_saveModel(model,str(folder/'model.mjb'));np.savez_compressed(folder/'physical_context.npz',**physical)
    b=runtime.features.body;I=physical['body_inertia'][0,b];Q=Rotation.from_quat(physical['body_iquat'][0,b,[1,2,3,0]]).as_matrix()
    return dict(mass=float(physical['body_mass'][0,b]),inertia_BODY=(Q@np.diag(I)@Q.T).tolist(),gravity=physical['gravity'][0].tolist())


class OwnedProbeRecorder:
    """External output observer, never attached to physical runtime state."""
    def __init__(self,runtime):
        self.runtime=runtime;self.sim=runtime.env.sim;self.model=self.sim.mj_model;self.host=mujoco.MjData(self.model);self.enabled=False
        self.body=runtime.features.body;self.qids=runtime.features.qids.cpu().numpy();jids=[self.model.joint('robot/'+n).id for n in runtime.motion.joint_names]
        self.ranges=self.sim.wp_model.jnt_range.numpy()[0,jids].copy()

    def begin(self,raw):
        d=self.sim.wp_data;mw.get_data_into(self.host,self.model,d);sig=mujoco.mjtState.mjSTATE_INTEGRATION
        self.initial=np.empty(mujoco.mj_stateSize(self.model,sig));mujoco.mj_getState(self.model,self.host,self.initial,sig)
        self.initial_warm=d.qacc_warmstart.numpy()[0].copy();self.raw=np.asarray(raw).copy();self.phase=int(self.runtime.motion.time_steps[0])
        self.trace={k:[] for k in ['pre_qpos','pre_qvel','post_qpos','post_qvel','post_ctrl','post_qacc','post_qacc_warmstart','pre_object_COM',
            'post_qfrc_constraint','post_qfrc_applied','post_xfrc_applied','post_GPU_time','overflow','post_mocap_pos','post_mocap_quat',
            'pre_penetration_m','pre_self_penetration_m','post_joint_limit_excursion_rad']}
        self.rows=[];self.addresses=[];self.efc=[];self.offsets=[0];self.legacy=[];self.adhesion=[];self.enabled=True

    def before_step(self):
        d=self.sim.wp_data;self.preq=d.qpos.numpy()[0].copy();self.prev=d.qvel.numpy()[0].copy()

    def after_step(self):
        d=self.sim.wp_data;m=self.model;mw.get_data_into(self.host,m,d);h=self.host;labels=read_warp_contact_labels(self.sim.wp_model,d)
        if not np.array_equal(labels['geom'],h.contact.geom):raise ValueError('GPU/host contact order mismatch')
        tick=len(self.trace['post_qpos']);pen=selfpen=0.
        for i,(g,k) in enumerate(labels['geom']):
            c=h.contact[i];old=np.zeros(6);mujoco.mj_contactForce(m,h,i,old)
            self.rows.append(np.r_[tick,g,k,labels['dim'][i],labels['efc_address'][i,0],c.dist,c.pos,c.frame,labels['local_force6'][i]])
            self.addresses.append(labels['efc_address'][i]);self.legacy.append(old);self.adhesion.append(float(c.adhesion))
            on=m.geom_bodyid[[g,k]]==self.body
            if on.sum()==1:pen=max(pen,-float(c.dist))
            elif (m.geom(int(g)).name or '').startswith('robot/') and (m.geom(int(k)).name or '').startswith('robot/'):selfpen=max(selfpen,-float(c.dist))
        self.efc.append(labels['efc_force']);self.offsets.append(self.offsets[-1]+len(self.efc[-1]))
        excursion=float(np.maximum(np.maximum(self.ranges[:,0]-h.qpos[self.qids],h.qpos[self.qids]-self.ranges[:,1]),0).max())
        values=dict(pre_qpos=self.preq,pre_qvel=self.prev,post_qpos=h.qpos.copy(),post_qvel=h.qvel.copy(),post_ctrl=h.ctrl.copy(),post_qacc=h.qacc.copy(),
            post_qacc_warmstart=d.qacc_warmstart.numpy()[0].copy(),pre_object_COM=h.xipos[self.body].copy(),post_qfrc_constraint=h.qfrc_constraint.copy(),
            post_qfrc_applied=h.qfrc_applied.copy(),post_xfrc_applied=h.xfrc_applied.copy(),post_GPU_time=float(h.time),overflow=bool(labels['overflow'] or d.overflow.numpy()[0]),
            post_mocap_pos=h.mocap_pos.copy(),post_mocap_quat=h.mocap_quat.copy(),pre_penetration_m=pen,pre_self_penetration_m=selfpen,post_joint_limit_excursion_rad=excursion)
        for key,value in values.items():self.trace[key].append(value)

    def finish(self):
        self.enabled=False
        return dict(**{k:np.array(v) for k,v in self.trace.items()},initial_native_state=self.initial,initial_warmstart=self.initial_warm,actions20=self.raw[None],
            pre_action_phase=np.array([self.phase]),raw_contact_tick_plus23=np.array(self.rows).reshape(-1,24),
            GPU_contact_efc_addresses=np.array(self.addresses).reshape(-1,self.sim.wp_data.contact.efc_address.shape[-1]),GPU_efc_force_flat=np.concatenate(self.efc),
            GPU_efc_ranges=np.array(self.offsets),legacy_host_local_force6=np.array(self.legacy).reshape(-1,6),legacy_host_adhesion=np.array(self.adhesion))
