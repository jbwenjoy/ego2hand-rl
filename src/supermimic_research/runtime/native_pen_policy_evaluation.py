"""Original strict task evaluation of a fresh-start full-pose Gaussian mean policy."""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.learning.native_pen_ppo import CHECKPOINT_SCHEMA
from supermimic_research.mechanics.compound_holding import object_state
from supermimic_research.mechanics.warp_contact_labels import read_warp_contact_labels
from .pen_native_policy import NativePenPolicyAdapter
from .pen_pose_observation import require_full_pose_contract
from .pen_portable_state import reference_signature,host_model_digest
from .controlled_evaluation import summarize_trace


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def cpu(value):return value.detach().cpu().numpy().copy()


@torch.no_grad()
def evaluate(checkpoint,adapter_config,gates,folder):
    """Weights-only FRESH evaluation, not continuation from a training episode."""
    folder=Path(folder);folder.mkdir();state=torch.load(verify_file_pin(checkpoint),map_location='cpu',weights_only=False)
    if state['schema']!=CHECKPOINT_SCHEMA:raise ValueError('Native161/188 PPO checkpoint required')
    require_full_pose_contract(state['metadata']);verify_source_snapshot(state['native']['sources'],live=True)
    adapter=NativePenPolicyAdapter(adapter_config,1);env=adapter.env;m=env.sim.mj_model;f=adapter.features
    if state['metadata']['joint_order']!=list(adapter.motion.joint_names) or state['native']['config']!=file_pin(adapter.config_path):
        raise ValueError('Evaluation joint/config mismatch')
    if reference_signature(adapter)!=state['native']['reference_signature'] or host_model_digest(m)!=state['native']['host_model_digest']:
        raise ValueError('Fresh evaluation reference/host model mismatch')
    cfg=state['config'];actor=GaussianPPOActor(161,20,init_std=cfg['ppo_init_std'],std_range=(cfg['ppo_std_min'],cfg['ppo_std_max'])).to(adapter.device)
    actor.load_state_dict(state['modules']['actor']);actor.eval()
    physical={};physical_checks={};render_model=copy.copy(m)
    for key in ['body_mass','body_inertia','body_ipos','body_iquat','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp',
        'actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange','jnt_range','dof_damping','dof_armature']:
        value=getattr(env.sim.wp_model,key).numpy().copy();old=state['native']['native']['model']['arrays'][key]
        physical_checks[key]=bool(np.array_equal(np.broadcast_to(value,old.shape),old))
        physical[key]=value;target=getattr(render_model,key)
        target[:]=value if value.shape==target.shape else value[0]
    if not all(physical_checks.values()):raise ValueError('Evaluation dynamics differ from training')
    mujoco.mj_setConst(render_model,mujoco.MjData(render_model));mujoco.mj_saveModel(render_model,str(folder/'model.mjb'))
    np.savez_compressed(folder/'physical_context.npz',**physical)
    from mjlab.utils.lab_api.math import combine_frame_transforms
    frames=adapter.motion._frames;pos,quat=combine_frame_transforms(frames['wrist_pose_w'][:,:3],frames['wrist_pose_w'][:,3:],frames['object_pose'][:,:3],frames['object_pose'][:,3:])
    reference=dict(time_s=np.arange(369)*.02,COM_world_m=cpu(pos),quaternion_wxyz=cpu(quat))
    body=f.body;qa=f.qa;da=int(m.jnt_dofadr[m.body_jntadr[body]]);qids=cpu(f.qids);jointids=np.array([m.joint('robot/'+name).id for name in adapter.motion.joint_names])
    com=physical['body_ipos'][0,body];trace={key:[] for key in ['pre_qpos','pre_qvel','post_qpos','post_qvel','post_ctrl',
        'post_COM','post_quaternion_wxyz','post_COM_velocity','post_omega_world','pre_penetration_m','pre_self_penetration_m',
        'post_joint_limit_excursion_rad','pre_object_nonhand_normal_N','pre_hand_normal_N','post_qfrc_applied','post_xfrc_applied','overflow',
        'post_qacc_warmstart','post_actuator_force','post_GPU_time']}
    initial_qpos=env.sim.wp_data.qpos.numpy()[0].copy();initial_mocap_pos=env.sim.wp_data.mocap_pos.numpy().copy();initial_mocap_quat=env.sim.wp_data.mocap_quat.numpy().copy()
    aids=cpu(adapter.action._position_ctrl_ids);act=[];phases=[];observations=[];native_steps=0;control_steps=0;fixed_mount=True
    contact_rows={key:[] for key in ['tick','geom','dim','dist','pos','frame','friction','efc_address','local_force6']}
    efc=[];efc_offsets=[0]
    original=env.sim.step
    def step():
        nonlocal native_steps,fixed_mount
        d=env.sim.wp_data;trace['pre_qpos'].append(d.qpos.numpy()[0].copy());trace['pre_qvel'].append(d.qvel.numpy()[0].copy())
        original();native_steps+=1
        q=d.qpos.numpy()[0].copy();v=d.qvel.numpy()[0].copy();center,Q,velocity,omega=object_state(m,q,v,qa,da,com)
        labels=read_warp_contact_labels(env.sim.wp_model,d);pen=selfpen=hand=nonhand=0.
        efc.extend(labels['efc_force']);efc_offsets.append(len(efc))
        for i in range(len(labels['geom'])):
            contact_rows['tick'].append(native_steps-1)
            for key in contact_rows:
                if key!='tick':contact_rows[key].append(labels[key][i])
        for i,(g,h) in enumerate(labels['geom']):
            g,h=int(g),int(h);on=m.geom_bodyid[[g,h]]==body;gn=m.geom(g).name or '';hn=m.geom(h).name or ''
            if on.sum()==1:
                pen=max(pen,max(0.,-float(labels['dist'][i])));name=hn if on[0] else gn
                if name.startswith('robot/'):hand+=max(0.,float(labels['local_force6'][i,0]))
                else:nonhand+=max(0.,float(labels['local_force6'][i,0]))
            elif gn.startswith('robot/') and hn.startswith('robot/'):selfpen=max(selfpen,max(0.,-float(labels['dist'][i])))
        ranges=physical['jnt_range'][0,jointids];excursion=float(np.max(np.maximum(np.maximum(ranges[:,0]-q[qids],q[qids]-ranges[:,1]),0)))
        row=dict(post_qpos=q,post_qvel=v,post_ctrl=d.ctrl.numpy()[0].copy(),post_COM=center,post_quaternion_wxyz=Q,
            post_COM_velocity=velocity,post_omega_world=omega,pre_penetration_m=pen,pre_self_penetration_m=selfpen,
            post_joint_limit_excursion_rad=excursion,pre_object_nonhand_normal_N=nonhand,pre_hand_normal_N=hand,
            post_qfrc_applied=d.qfrc_applied.numpy()[0].copy(),post_xfrc_applied=d.xfrc_applied.numpy()[0].copy(),
            overflow=int(d.overflow.numpy()[0]) or int(labels['overflow']),post_qacc_warmstart=d.qacc_warmstart.numpy()[0].copy(),
            post_actuator_force=d.actuator_force.numpy()[0].copy(),post_GPU_time=float(d.time.numpy()[0]))
        for key,value in row.items():trace[key].append(value)
        fixed_mount=fixed_mount and np.array_equal(d.mocap_pos.numpy(),initial_mocap_pos) and np.array_equal(d.mocap_quat.numpy(),initial_mocap_quat)
    env.sim.step=step;terminated=truncated=False
    try:
        obs=adapter.observation()
        for _ in range(368):
            observations.append(cpu(obs['policy'])[0]);phases.append(int(adapter.motion.time_steps[0]))
            action=actor.distribution(obs['policy'],validate_args=False).mean
            act.append(cpu(action)[0]);obs,_,term,trunc,_=adapter.step(action);control_steps+=1
            terminated=bool(term[0]);truncated=bool(trunc[0])
            if terminated or truncated:break
        arrays={key:np.array(value) for key,value in trace.items()};complete=control_steps==368 and not terminated
        summary=summarize_trace(arrays,reference,.0025,complete,gates)
        raw_contacts={key:np.array(value).reshape((-1,)+({'geom':(2,),'pos':(3,),'frame':(3,3),'friction':(5,),'efc_address':(6,),'local_force6':(6,)}.get(key,()))) for key,value in contact_rows.items()}
        np.savez_compressed(folder/'raw_contacts.npz',**raw_contacts,efc_force=np.array(efc),efc_offsets=np.array(efc_offsets))
        np.savez_compressed(folder/'trace.npz',**arrays,initial_qpos=initial_qpos,initial_mocap_pos=initial_mocap_pos,
            initial_mocap_quat=initial_mocap_quat,pre_action_observations=np.array(observations),pre_action_phase=np.array(phases),
            sampled_mean_raw=np.array(act),applied_mean_raw=np.clip(np.array(act),-1,1),**reference)
        result=dict(schema='native_pen_full_pose_policy_strict_evaluation:1',checkpoint=checkpoint,iteration=state['iteration'],
            summary=summary,controls=control_steps,native_integrations=native_steps,dt=.0025,terminated=terminated,truncated=truncated,
            fixed_mount=fixed_mount,post_start_resets=0,physical_matches=physical_checks,model=file_pin(folder/'model.mjb'),
            physical_context=file_pin(folder/'physical_context.npz'),trace=file_pin(folder/'trace.npz'),
            raw_contacts=file_pin(folder/'raw_contacts.npz'),cone=int(env.sim.wp_model.opt.cone),
            scope='Fresh original369frame/7.36s no-aid mean-policy evaluation on the training reference. Perfect full pose, no ego/generalization/deterministic W32/CHORD-gradient or A1/A2 claim.')
        write(folder/'report.json',result);return result
    finally:
        env.sim.step=original;adapter.close()
