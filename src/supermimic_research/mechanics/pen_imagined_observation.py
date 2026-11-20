"""Construct the native pen Actor's full-pose observation from an imagined state."""
import torch

from dex_supermimic.rl.dex_supermimic.rotations import mat_to_quat, quat_to_mat
from ..runtime.pen_pose_observation import pack_full_pose_policy, pose9_world_in_wrist
from .impulse_codec import rotation55_torch


def pen_imagined_observation(state, previous_applied_raw, reference_q, reference_pose_wxyz,
                             reference_links, mount_wxyz, encoder_bias, default_q, default_qd):
    """Inputs describe the reference frame used by the upcoming control.

    state has shape (...,55), in the native pen state chart with object COM
    at the free-body origin. Both 7D poses are WORLD [xyz,wxyz]. The 21 XYZ
    link targets are already expressed in the wrist frame. Joint vectors follow
    the transmission-resolved 20-joint order. The final feature is the previous
    applied RAW action (or the initialized native RAW state at reset).

    Static features and individual reference frames may broadcast to the batch
    dimensions of state. Inputs should share its floating-point dtype/device.
    """
    if state.shape[-1:] != (55,):
        raise ValueError("Expected the native 55-dimensional state chart")
    batch = state.shape[:-1]

    def vector(x, width):
        if x.shape[-1:] != (width,):
            raise ValueError(f"Expected feature width {width}, got {tuple(x.shape)}")
        return x.expand(*batch, width)

    mount = vector(mount_wxyz, 7)
    mount_quat = mount[..., 3:]
    wrist_T = quat_to_mat(mount_quat / torch.linalg.vector_norm(
        mount_quat, dim=-1, keepdim=True)).mT
    if reference_links.shape[-2:] == (21, 3):
        reference_links = reference_links.flatten(-2)
    current_quat = mat_to_quat(rotation55_torch(state))
    current_pose = torch.cat((state[..., 40:43], current_quat), dim=-1)
    reference_pose9 = pose9_world_in_wrist(vector(reference_pose_wxyz, 7), mount,
                                           wrist_rotation_transpose=wrist_T)
    current_pose9 = pose9_world_in_wrist(current_pose, mount, wrist_rotation_transpose=wrist_T)
    return pack_full_pose_policy(
        vector(reference_q, 20), reference_pose9, vector(reference_links, 63),
        state[..., :20] + vector(encoder_bias, 20) - vector(default_q, 20),
        (state[..., 20:40] - vector(default_qd, 20)) * .25, current_pose9,
        vector(previous_applied_raw, 20),
    )
