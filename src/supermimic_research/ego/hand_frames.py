"""Explicit human wrist/palm frames and pinned annotation FK, not retargeting.

The palm convention is an application choice, not an upstream pose field.
All transforms map column vectors from the named source frame into the target.
"""
import dataclasses
import json
import tarfile

import numpy as np


def pose_matrix(value):
    p = np.asarray(value['translation_xyz'], dtype=np.float64)
    q = np.asarray(value['quaternion_wxyz'], dtype=np.float64)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
        raise ValueError('Finite xyz / wxyz source pose required')
    if abs(np.linalg.norm(q) - 1) > 1e-6:
        raise ValueError('Nonunit source quaternion; do not silently repair')
    w, x, y, z = q
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = p
    if np.max(np.abs(result[:3, :3].T @ result[:3, :3] - np.eye(3))) > 3e-6:
        raise ValueError('Source rotation is not orthogonal')
    return result


def transform_points(matrix, points):
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def palm_frame(wrist, bases):
    """Rigid rest-palm convention: origin mean four nonthumb bases.

    +X wrist->mean base (distal), +Y little->index orthogonalized (radial),
    +Z=X cross Y. This is a proper frame for either hand, never a reflection.
    ``bases`` order is index, middle, ring, little; no scale fit is performed.
    """
    wrist, bases = np.asarray(wrist, float), np.asarray(bases, float)
    if wrist.shape != (3,) or bases.shape != (4, 3) or not np.isfinite(np.r_[wrist, bases.ravel()]).all():
        raise ValueError('Finite named wrist and four base anchors required')
    center = bases.mean(0)
    x = center-wrist
    if np.linalg.norm(x) < 1e-5:
        raise ValueError('Degenerate distal palm axis')
    x /= np.linalg.norm(x)
    y = bases[0]-bases[3]
    y -= x*np.dot(x, y)
    if np.linalg.norm(y) < 1e-5:
        raise ValueError('Degenerate radial palm axis')
    y /= np.linalg.norm(y)
    result = np.eye(4)
    result[:3, :3] = np.stack([x, y, np.cross(x, y)], axis=1)
    result[:3, 3] = center
    return result


def read_hand_window(path, frames, side, bop_id, uid):
    if side not in ('left', 'right') or not frames or sorted(set(frames)) != list(frames):
        raise ValueError('Explicit ordered unique frame IDs and side required')
    angles, wrist, objects, records, raw = [], [], [], [], []
    with tarfile.open(path) as tar:
        def read(name):
            return json.load(tar.extractfile(name))
        profile = read('__hand_shapes.json__')['umetrack']
        for frame in frames:
            info = read(f'{frame:06d}.info.json')
            hand = read(f'{frame:06d}.hands.json')[side]['umetrack_pose']
            matches = [x for x in read(f'{frame:06d}.objects.json').get(str(bop_id), [])
                       if str(x['object_uid']) == str(uid)]
            if len(matches) != 1:
                raise ValueError('Exactly one declared object instance required')
            ob = matches[0]
            if str(ob['object_bop_id']) != str(bop_id):
                raise ValueError('Conflicting BOP identity')
            angles.append(hand['joint_angles'])
            wrist.append(pose_matrix(hand['T_world_from_wrist']))
            objects.append(pose_matrix(ob['T_world_from_object']))
            records.append(dict(frame_id=frame, **{k: info[k] for k in
                ('ref_timestamp_ns', 'image_timestamps_ns', 'sequence_id', 'participant_id')}))
            raw.append(dict(frame_id=frame, umetrack_pose=hand,
                object={k: ob[k] for k in ('object_bop_id', 'object_uid', 'T_world_from_object')}))
    angles = np.asarray(angles, float)
    if angles.shape != (len(frames), 22) or not np.isfinite(angles).all():
        raise ValueError('Preserve all22 finite source parameters')
    stamps = np.array([r['ref_timestamp_ns'] for r in records], dtype=np.int64)
    if np.any(np.diff(stamps) <= 0) or len({(r['sequence_id'], r['participant_id']) for r in records}) != 1:
        raise ValueError('One sequence/participant and strictly increasing source timestamps')
    return dict(profile=profile, angles22=angles, wrist=np.array(wrist),
                objects=np.array(objects), timestamps_ns=stamps, records=records, raw=raw)


def reconstruct(profile, angles22, wrist, side):
    """Unmodified upstream CPU FP32 FK, with all21 raw landmarks retained."""
    import torch
    from .vendor_umetrack.hand_models import umetrack_hand_model as official
    if side not in ('left', 'right'):
        raise ValueError('Explicit hand side required')
    n = len(angles22)
    if np.shape(angles22) != (n, 22) or np.shape(wrist) != (n, 4, 4):
        raise ValueError('Batched22 angle and wrist transform shapes')
    if not np.isfinite(angles22).all() or not np.isfinite(wrist).all():
        raise ValueError('Finite annotations required')
    model = official.from_json(profile)
    batched = official.UmeTrackHandModelData(**{
        f.name: (getattr(model, f.name).unsqueeze(0).expand(n, *getattr(model, f.name).shape).clone()
                 if isinstance(getattr(model, f.name), torch.Tensor) else getattr(model, f.name))
        for f in dataclasses.fields(model)})
    pose = official.UmeTrackHandPose(torch.full((n,), int(side == 'right'), dtype=torch.int64),
        torch.tensor(angles22, dtype=torch.float32), torch.tensor(wrist, dtype=torch.float32))
    with torch.inference_mode():
        canonical, vertices, faces = official.forward_kinematics(pose, batched)
        mirrored = pose.wrist_xform.clone()
        if side == 'right':
            mirrored[:, :, 0] *= -1
        raw21 = official.skin_landmarks(batched, pose.joint_angles, mirrored)
    if not torch.equal(canonical, raw21[:, :20]):
        raise ValueError('Unexpected canonical mapping')
    if not torch.isfinite(vertices).all() or not torch.isfinite(raw21).all():
        raise ValueError('Nonfinite source FK')
    return dict(vertices=vertices.numpy(), landmarks21=raw21.numpy(), faces=faces.numpy())


def numpy_skin(profile, angles22, wrist, side, *, mesh=True):
    """Independent FP64 linear-blend reconstruction for numerical checking.

    Uses explicit rotations about rest pivots and chains, no Torch/vendor call.
    Not an independent estimator of the human hand or a physical certificate.
    """
    from scipy.spatial.transform import Rotation
    if side not in ('left', 'right'):
        raise ValueError('Explicit side')
    q = np.asarray(angles22, float)
    rest = np.asarray(profile['joint_rest_positions'], float)[:20]*.001
    axes = np.asarray(profile['joint_rotation_axes'], float)[:20]
    points = np.asarray(profile['mesh_vertices' if mesh else 'landmark_rest_positions'], float)*.001
    if mesh:
        weights = np.asarray(profile['dense_bone_weights'], float)
    else:
        weights = np.zeros((len(points), 17))
        for row, (indices, values) in enumerate(zip(profile['landmark_rest_bone_indices'], profile['landmark_rest_bone_weights'])):
            for index, value in zip(indices, values):
                if value != 0:
                    weights[row, index] = value
    out = []
    for angles, root in zip(q, wrist):
        root = np.array(root, dtype=float, copy=True)
        if side == 'right':
            root[:, 0] *= -1
        local = np.repeat(np.eye(4)[None], 20, axis=0)
        local[:, :3, :3] = Rotation.from_rotvec(axes*angles[:20, None]).as_matrix()
        local[:, :3, 3] = rest - np.einsum('nij,nj->ni', local[:, :3, :3], rest)
        frames = [root, root]
        for digit in range(5):
            current = root.copy()
            for j in range(4):
                current = current @ local[digit*4+j]
                if j >= 1:
                    frames.append(current)
        homo = np.c_[points, np.ones(len(points))]
        out.append(np.einsum('vbr,vb->vr', np.einsum('brs,vs->vbr', np.array(frames), homo), weights)[:, :3])
    return np.array(out)
