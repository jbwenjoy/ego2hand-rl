"""Opt-in immutable reference prefix for the fixed-mount native161 contract.

Dynamic hand/object/velocity/action features still use the current state. The
caller validates immutable inputs at block boundaries. This cache is NOT part
of existing native checkpoint schema3 and must not be attached silently.
"""
import hashlib
import torch
from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat
from .pen_pose_observation import pose9_world_in_wrist

SCHEMA='frozen_native_reference_prefix92_fixed_frames:2'


def tensor_digest(t):
    x=t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(x.shape),x.dtype)).encode()+x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class FrozenPenReferenceCache:
    @torch.no_grad()
    def __init__(self,features):
        from mjlab.utils.lab_api.math import combine_frame_transforms
        frames=features.motion._frames;self.device=features.env.device
        self.worlds=features.env.num_envs;self.frames=len(frames['joint_pos'])
        if frames['joint_pos'].shape!=(self.frames,20) or frames['link_pos'].shape!=(self.frames,21,3):raise ValueError('Native20 joints/21 links required')
        self.mount=features.mount().detach().clone();self.origins=features.env.scene.env_origins.detach().clone()
        if self.mount.shape!=(self.worlds,7) or self.origins.shape!=(self.worlds,3):raise ValueError('World-owned mount/origins')
        if not bool(torch.isfinite(self.mount).all()) or not bool(torch.isfinite(self.origins).all()):raise ValueError('Finite frozen mount/origins')
        # Preserve the two original normalization chains exactly. Policy
        # explicitly normalizes before quat_to_mat;critic normalizes only there.
        q=self.mount[:,3:]
        self.policy_WT=quat_to_mat(q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)).transpose(-1,-2)
        self.critic_WT=quat_to_mat(q).transpose(-1,-2)
        self.frame_signature={k:tensor_digest(v) for k,v in sorted(frames.items())}
        self.world_ids=torch.arange(self.worlds,device=self.device)
        # Batch every phase once. Preserve the original world->wrist transform
        # chain instead of replacing it with a guessed wrist-local convention.
        w=frames['wrist_pose_w'][None].expand(self.worlds,-1,-1).clone();w[:,:,:3]+=self.origins[:,None]
        p=frames['object_pose'][None].expand(self.worlds,-1,-1)
        pos,quat=combine_frame_transforms(w[:,:,:3].reshape(-1,3),w[:,:,3:].reshape(-1,4),p[:,:,:3].reshape(-1,3),p[:,:,3:].reshape(-1,4))
        self.world_pose=torch.cat((pos,quat),-1).reshape(self.worlds,self.frames,7)
        local=pose9_world_in_wrist(self.world_pose,self.mount[:,None].expand(-1,self.frames,-1))
        self.prefix=torch.cat((frames['joint_pos'][None].expand(self.worlds,-1,-1),local,
            frames['link_pos'].flatten(1)[None].expand(self.worlds,-1,-1)),-1).contiguous()
        self.table_signature={k:tensor_digest(getattr(self,k)) for k in ['prefix','world_pose','policy_WT','critic_WT']}

    def validate(self,features):
        """Explicit cold-path guard. No CPU reads are hidden in policy()."""
        frames=features.motion._frames
        checks=dict(worlds=features.env.num_envs==self.worlds,device=torch.device(features.env.device)==torch.device(self.device),
            mount=torch.equal(features.mount(),self.mount),origins=torch.equal(features.env.scene.env_origins,self.origins),
            reference={k:tensor_digest(v) for k,v in sorted(frames.items())}==self.frame_signature,
            table={k:tensor_digest(getattr(self,k)) for k in ['prefix','world_pose','policy_WT','critic_WT']}==self.table_signature)
        if not all(checks.values()):raise ValueError('Frozen reference cache input changed: '+str([k for k,v in checks.items() if not v]))
        return checks

    def reference_pose_world(self,frame_ids):return self.world_pose[self.world_ids,frame_ids]

    def policy(self,features,*,fixed_frames=False):
        """Same161 columns; only the immutable92-column prefix is cached."""
        d=features.env.sim.data;robot=features.robot.data;ids=features.local_joints
        q=d.qpos[:,features.qids]+robot.encoder_bias[:,ids]-robot.default_joint_pos[:,ids]
        qd=(d.qvel[:,features.dofs]-robot.default_joint_vel[:,ids])*.25
        current=pose9_world_in_wrist(features.pose(),self.mount,wrist_rotation_transpose=self.policy_WT) if fixed_frames else pose9_world_in_wrist(features.pose(),features.mount())
        return torch.cat((self.prefix[self.world_ids,features.motion.frame_ids],q,qd,current,features.action._raw_actions),-1)

    def metadata(self):
        return dict(schema=SCHEMA,worlds=self.worlds,frames=self.frames,prefix_columns=92,policy_columns=161,
            frame_signature=self.frame_signature,table_signature=self.table_signature,
            resident_bytes=sum(t.numel()*t.element_size() for t in [self.prefix,self.world_pose,self.world_ids,self.mount,self.origins,self.policy_WT,self.critic_WT]),
            guard='Caller must validate fixed mount, origins, reference content and rebuilt tables at block boundaries. Not portable schema3 state.',
            scope='Computational cache with the original world->wrist reference convention; no physics, action, reward or termination change.')
