"""Explicit fixed-wrist full-pose161 features; no legacy checkpoint reinterpretation."""
import torch
from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat

OBSERVATION_SCHEMA='pen_fixed_wrist_full_pose161_H20:1'
ACTION_CONTRACT='RAW_CLAMP_MINUS1_PLUS1_SCALE0P5_EMA0P5_PD_BOUNDS'


def pose9_world_in_wrist(pose,mount,*,position_bound=.15,wrist_rotation_transpose=None):
    """WXYZ world poses; COM at object free-body origin, matrix columns in wrist."""
    if pose.shape[-1]!=7 or mount.shape[-1]!=7 or position_bound<=0:
        raise ValueError('Declared WXYZ poses and positive wrist-local bound required')
    # No epsilon/identity fallback: invalid zero poses propagate NaN for the
    # collector's explicit finite-observation rejection at its boundary.
    q=pose[...,3:]/torch.linalg.vector_norm(pose[...,3:],dim=-1,keepdim=True)
    if wrist_rotation_transpose is None:w=mount[...,3:]/torch.linalg.vector_norm(mount[...,3:],dim=-1,keepdim=True)
    R=quat_to_mat(q)
    if wrist_rotation_transpose is None:
        WT=quat_to_mat(w).transpose(-1,-2)
    else:
        WT=wrist_rotation_transpose
        if WT.shape[-2:]!=(3,3) or WT.device!=pose.device or WT.dtype!=pose.dtype:raise ValueError('Matching fixed wrist WORLD-to-wrist matrix required')
    local=(WT@(pose[...,:3]-mount[...,:3]).unsqueeze(-1)).squeeze(-1)
    relative=WT@R
    return torch.cat([position_bound*torch.tanh(local/position_bound),relative[..., :,0],relative[..., :,1]],dim=-1)


def pack_full_pose_policy(reference_q,reference_pose9,reference_links,biased_joint_relative,
                          scaled_joint_velocity,current_pose9,previous_applied_raw):
    pieces=[reference_q,reference_pose9,reference_links,biased_joint_relative,scaled_joint_velocity,current_pose9,previous_applied_raw]
    sizes=[20,9,63,20,20,9,20]
    if any(p.shape[-1]!=size or p.shape[:-1]!=pieces[0].shape[:-1] for p,size in zip(pieces,sizes)):
        raise ValueError('Explicit reference/current full-pose161 feature layout required')
    return torch.cat(pieces,dim=-1)


def require_full_pose_contract(metadata):
    if (metadata.get('observation_schema')!=OBSERVATION_SCHEMA or metadata.get('policy_input_dim')!=161
            or metadata.get('action_dim')!=20 or metadata.get('action_contract')!=ACTION_CONTRACT):
        raise ValueError('New full-pose161/action metadata required; legacy155/cube weights cannot be reinterpreted')
