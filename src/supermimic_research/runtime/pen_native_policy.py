"""Explicit full-pose pen interface, with post-integration rewards and owned resets.

This is an actual-simulator baseline adapter. It never invokes a learned world
model or reinterprets the upstream 155-dimensional policy/checkpoints.
"""
from pathlib import Path
import json
import random
import tomllib

import numpy as np
import torch

from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat
from .pen_pose_observation import pose9_world_in_wrist,pack_full_pose_policy,OBSERVATION_SCHEMA,ACTION_CONTRACT

CRITIC_SCHEMA='pen_full_pose188_privileged_ema20_wrist_velocity6_phase1:1'
TRANSITION_SCHEMA='pen_native_postphysics_used_reference_phase:1'


def orientation_errors(q,reference_q):
    """Full SO(3) and directed shaft angles. Invalid quaternions stay invalid."""
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    r=reference_q/torch.linalg.vector_norm(reference_q,dim=-1,keepdim=True)
    full=2*torch.acos((q*r).sum(-1).abs().clamp(0,1))
    a=quat_to_mat(q)[..., :,0];b=quat_to_mat(r)[..., :,0]
    shaft=torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(a,b),dim=-1),(a*b).sum(-1))
    return full,shaft


def tracking_reward(position,quaternion,joints,reference_pose,reference_joints,applied_raw,cfg):
    """Reward evaluated after physics against the reference used by that action."""
    distance=torch.linalg.vector_norm(position-reference_pose[:,:3],dim=-1)
    angle,shaft=orientation_errors(quaternion,reference_pose[:,3:])
    hand=(joints-reference_joints).square().mean(-1)
    components=dict(COM=torch.exp(-(distance/cfg['COM_scale_m']).square()),
        full_SO3=torch.exp(-(angle/cfg['angle_scale_rad']).square()),
        hand=torch.exp(-hand/cfg['hand_scale_rad']**2),action=applied_raw.square().mean(-1))
    reward=(cfg['COM_weight']*components['COM']+cfg['angle_weight']*components['full_SO3']
        +cfg['hand_weight']*components['hand']-cfg['action_weight']*components['action'])
    return reward,dict(COM_error_m=distance,full_angle_rad=angle,shaft_angle_rad=shaft,**components)


class NativePenFeatures:
    """Resolved indices and resident reference arrays, shared by the observation term."""
    def __init__(self,env):
        self.env=env;self.motion=env.command_manager.get_term('motion')
        self.action=env.action_manager.get_term('joint_pos');self.robot=env.scene['robot']
        m=env.sim.mj_model;device=env.device
        self.body=m.body('object/cube').id;self.palm=m.body('robot/right_palm_link').id
        self.qa=int(m.jnt_qposadr[m.body_jntadr[self.body]])
        jids=np.array([m.joint('robot/'+name).id for name in self.motion.joint_names])
        self.qids=torch.as_tensor(m.jnt_qposadr[jids].copy(),device=device,dtype=torch.long)
        self.dofs=torch.as_tensor(m.jnt_dofadr[jids].copy(),device=device,dtype=torch.long)
        self.local_joints=self.action._target_ids
        self.reference=self.motion._frames

    def reference_pose_world(self):
        from mjlab.utils.lab_api.math import combine_frame_transforms
        w=self.motion.wrist_pose_w;p=self.motion.object_pose
        pos,quat=combine_frame_transforms(w[:,:3],w[:,3:],p[:,:3],p[:,3:])
        return torch.cat((pos,quat),-1)

    def mount(self):
        d=self.env.sim.data
        return torch.cat((d.xpos[:,self.palm],d.xquat[:,self.palm]),-1)

    def pose(self):
        return self.env.sim.data.qpos[:,self.qa:self.qa+7]

    def policy(self):
        d=self.env.sim.data;data=self.robot.data;ids=self.local_joints
        q=d.qpos[:,self.qids]+data.encoder_bias[:,ids]-data.default_joint_pos[:,ids]
        qd=(d.qvel[:,self.dofs]-data.default_joint_vel[:,ids])*.25
        mount=self.mount()
        return pack_full_pose_policy(self.motion.joint_pos,
            pose9_world_in_wrist(self.reference_pose_world(),mount),self.motion.link_pos.flatten(1),
            q,qd,pose9_world_in_wrist(self.pose(),mount),self.action._raw_actions)

    def critic(self,policy):
        # Free-joint translation velocity is WORLD; angular velocity is BODY.
        d=self.env.sim.data;da=int(self.env.sim.mj_model.jnt_dofadr[
            self.env.sim.mj_model.body_jntadr[self.body]])
        WT=quat_to_mat(self.mount()[:,3:]).transpose(-1,-2)
        R=quat_to_mat(self.pose()[:,3:])
        v=(WT@d.qvel[:,da:da+3,None]).squeeze(-1)
        w=(WT@R@d.qvel[:,da+3:da+6,None]).squeeze(-1)
        phase=self.motion.frame_ids.float()/(len(self.reference['joint_pos'])-1)
        return torch.cat((policy,self.action._processed_actions,v,w,phase[:,None]),-1)


def full_pose_policy_observation(env):
    if not hasattr(env,'_native_full_pose_features'):
        env._native_full_pose_features=NativePenFeatures(env)
    return env._native_full_pose_features.policy()


class NativePenPolicyAdapter:
    """Own terminal transitions before explicit partial reset, no auto-reset loss.

    The upstream environment performs only actions/physics/reference advancement.
    Its stale-derived reward/termination hooks are disabled for this new schema.
    The adapter evaluates original native thresholds on refreshed post-state;
    this is a declared timing change, not byte-parity with old controllers.
    """
    def __init__(self,config_path,num_envs):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.managers import ObservationGroupCfg,ObservationTermCfg
        from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
        from supermimic_research.artifacts import ArtifactRegistry,verify_file_pin
        from .controlled_assets import bind_controlled_assets
        from .numerical_options import set_model_numerical_options
        from .collision_numerics import configure_collision_numerics,set_collision_numerics
        self.config_path=Path(config_path).resolve();self.cfg=tomllib.loads(self.config_path.read_text())
        cfg=self.cfg;basepath=(self.config_path.parent/cfg['base_config']).resolve()
        base=tomllib.loads(basepath.read_text());self.num_envs=num_envs
        if num_envs not in [1,32]:raise ValueError('Predeclared integration worlds1/32')
        random.seed(base['seed']);np.random.seed(base['seed']);torch.manual_seed(base['seed'])
        ec=make_env_cfg(num_envs,50,play=True,task='hand2-pen-spin',
            recipe_path=str((basepath.parent/base['pen_recipe']).resolve()),
            clips_root=str((basepath.parent/base['clips_root']).resolve()))
        ec.seed=base['seed'];ec.auto_reset=False;ec.sim.mujoco.timestep=cfg['physics_dt']
        ec.decimation=round(.02/cfg['physics_dt'])
        for key,value in cfg['numerical_options'].items():setattr(ec.sim.mujoco,key,value)
        configure_collision_numerics(ec.sim.mujoco,cfg['collision_numerics'])
        self.assets=bind_controlled_assets(ec,'hand2-pen-spin',ArtifactRegistry(basepath.parent/base['artifact_manifest']))
        ec.commands['motion'].first_frame_hold_prob=0.;ec.commands['motion'].debug_vis=False
        ec.observations={'policy':ObservationGroupCfg({'full_pose':ObservationTermCfg(full_pose_policy_observation)},
            concatenate_terms=True,enable_corruption=False)}
        ec.rewards={};ec.terminations={};ec.metrics={};ec.curriculum={}
        if any(e.mode not in ['reset','startup'] for e in ec.events.values()):raise ValueError('No force/interval events')
        self.env=ManagerBasedRlEnv(ec,device='cuda:0');self.device=self.env.device
        set_model_numerical_options(self.env.sim.mj_model,self.env.sim.wp_model,cfg['numerical_options'])
        set_collision_numerics(self.env.sim.mj_model,self.env.sim.wp_model,cfg['collision_numerics'])
        self.env.reset();self.features=self.env._native_full_pose_features;self.motion=self.features.motion;self.action=self.features.action
        m=self.env.sim.mj_model;dm=self.env.sim.wp_model
        if m.neq or np.any(m.body_gravcomp) or not np.all(dm.body_ipos.numpy()[:,self.features.body]==0):
            raise ValueError('Free unassisted COM-at-origin asset required')
        if self.motion.group.num_trajectories!=1 or len(self.motion._frames['joint_pos'])!=369:
            raise ValueError('Original single369-frame clip')
        if self.action.cfg.scale!=.5 or self.action.cfg.alpha!=.5 or self.action.cfg.mask_prob:
            raise ValueError('Declared unmasked scale/EMA contract')
        proposal=json.loads((self.config_path.parent/cfg['initialization_proposal']).resolve().read_text())
        s=dict(np.load(verify_file_pin(proposal['artifacts']['selected'])))
        d=self.env.sim.data
        expected=torch.as_tensor(s['source_qpos'],device=self.device,dtype=torch.float32)
        if not torch.equal(d.qpos,expected.expand_as(d.qpos)):raise ValueError('Fresh source reset differs from initializer configuration')
        self.initial={key:getattr(d,key)[0].clone() for key in ['qpos','qvel','ctrl','qacc_warmstart','time','mocap_pos','mocap_quat']}
        projected=torch.as_tensor(s['qpos'],device=self.device,dtype=torch.float32)
        other=[i for i in range(m.nq) if i not in self.features.qids.tolist()]
        if not torch.equal(projected[other],expected[other]):raise ValueError('Only initial hand joints may be projected')
        self.initial['qpos']=projected
        self.initial_ema=torch.as_tensor(s['delta20'],device=self.device,dtype=torch.float32)
        self.initial_raw=2*self.initial_ema
        self.pending=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.control_transitions=0;self.physics_integrations=0;self.explicit_reset_worlds=0
        self.reset(torch.arange(num_envs,device=self.device))

    @torch.no_grad()
    def reset(self,ids):
        ids=ids.to(device=self.device,dtype=torch.long)
        self.env.reset(env_ids=ids)
        d=self.env.sim.data
        for key,value in self.initial.items():getattr(d,key)[ids]=value
        # Reference command is phase1 after reset; targets are explicitly frame0
        # plus initial EMA, exactly the declared pre-start physical configuration.
        a=self.action
        a._processed_actions[ids]=self.initial_ema;a._raw_actions[ids]=self.initial_raw
        a._targets[ids]=a.limit_target(self.motion._frames['joint_pos'][0][None]+self.initial_ema)[0]
        a._targets_valid[ids]=True
        self.env.sim.data.ctrl[ids[:,None],a._position_ctrl_ids[None,:]]=a._targets[ids]
        self.env.sim.forward();self.env.sim.sense()
        self.pending[ids]=False;self.explicit_reset_worlds+=len(ids)
        return self.observation()

    @torch.no_grad()
    def observation(self):
        p=self.features.policy()
        return dict(policy=p,critic=self.features.critic(p))

    @torch.no_grad()
    def step(self,sampled_raw):
        if sampled_raw.shape!=(self.num_envs,20):raise ValueError('World-by20 sample')
        if bool(self.pending.any()):raise ValueError('Explicitly reset terminal worlds before next step')
        applied=sampled_raw.clamp(-1,1)
        phase=self.motion.time_steps.clone();refpose=self.features.reference_pose_world().clone();refq=self.motion.joint_pos.clone()
        # Post-step obs points to the NEXT reference. Reward uses the saved one.
        self.env.step(applied)
        pose=self.features.pose();q=self.env.sim.data.qpos[:,self.features.qids]
        reward,parts=tracking_reward(pose[:,:3],pose[:,3:],q,refpose,refq,applied,self.cfg['reward'])
        finite=torch.isfinite(pose).all(-1)&torch.isfinite(self.env.sim.data.qvel).all(-1)&torch.isfinite(reward)
        position=parts['COM_error_m']>.05;shaft=parts['shaft_angle_rad']>np.pi/3
        reference_end=phase>=len(self.motion._frames['joint_pos'])-1
        terminated=position|shaft|~finite;truncated=reference_end&~terminated
        self.pending.copy_(terminated|truncated)
        self.control_transitions+=self.num_envs;self.physics_integrations+=self.num_envs*self.env.cfg.decimation
        return self.observation(),reward,terminated,truncated,dict(sampled_raw=sampled_raw.clone(),applied_raw=applied,
            reference_phase=phase,reference_pose_world=refpose,reference_joints=refq,
            termination_reasons=dict(position=position,shaft=shaft,nonfinite=~finite,reference_end=reference_end),**parts)

    def metadata(self):
        return dict(observation_schema=OBSERVATION_SCHEMA,policy_input_dim=161,critic_schema=CRITIC_SCHEMA,
            critic_input_dim=188,action_dim=20,action_contract=ACTION_CONTRACT,transition_schema=TRANSITION_SCHEMA,
            scope='Fresh controlled native full-pose interface; no WM, trained policy, ego camera or portable-physics-resume qualification.')

    def close(self):self.env.close()
