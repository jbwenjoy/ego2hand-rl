"""Export derived robot goals to the native pen-reader ABI, never source labels.

The palm-root IK's 21 macro points and the native reader's 21 BODY origins have
different meanings. This module recomputes the latter from derived q. Export
compatibility is not physical initialization, tracking, or dataset qualification.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np


NATIVE_LINK_NAMES = ('right_palm_link',) + tuple(
    f'right_finger{finger}_link{link}' for finger in range(1, 6) for link in range(1, 5))
JOINT_NAMES = tuple(f'right_finger{finger}_joint{joint}'
                    for finger in range(1, 6) for joint in range(1, 5))
NATIVE_KEYS = frozenset(('reference_qpos', 'wrist_pose_w', 'object_pose_w', 'link_pos_w'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_native_names(constants_path):
    """Check the real source assignment without importing a simulation env."""
    tree = ast.parse(Path(constants_path).read_text(encoding='utf-8-sig'))
    assigned = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for name in statement.targets:
                if isinstance(name, ast.Name): assigned[name.id] = statement.value
    expressions = {
        'RIGHT_HAND_LINK_NAMES': '("right_palm_link",) + tuple(f"right_finger{finger}_link{link}" for finger in range(1,6) for link in range(1,5))',
        'RIGHT_FINGER_JOINT_NAMES': 'tuple(f"right_finger{finger}_joint{joint}" for finger in range(1,6) for joint in range(1,5))',
    }
    for name, expression in expressions.items():
        actual = assigned.get(name)
        if actual is None or ast.dump(actual, include_attributes=False) != ast.dump(ast.parse(expression, mode='eval').body, include_attributes=False):
            raise ValueError('Native order changed; review this ABI explicitly: ' + name)
    return {'constants_path': str(constants_path), 'constants_sha256': sha(constants_path),
            'native_body_link_names': list(NATIVE_LINK_NAMES), 'native_joint_names': list(JOINT_NAMES)}


def build_native_arrays(q, object_pose, model_path, constants_path):
    names_contract = verify_native_names(constants_path)
    q = np.array(q, dtype=np.float64, copy=True)
    pose = np.array(object_pose, dtype=np.float64, copy=True)
    if q.ndim != 2 or q.shape[1] != 20 or len(q) < 2 or pose.shape != (len(q), 7):
        raise ValueError('Expected at least two derived q20/object-pose7 samples')
    if not np.isfinite(q).all() or not np.isfinite(pose).all(): raise ValueError('Nonfinite derived reference')
    if np.max(np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1)) > 1e-6:
        raise ValueError('Object wxyz quaternion unit error; no silent renormalization')
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    joints = np.array([model.joint('robot/' + name).id for name in JOINT_NAMES])
    qa = model.jnt_qposadr[joints]
    lo, hi = model.jnt_range[joints].T
    if np.any(q < lo) or np.any(q > hi): raise ValueError('Derived q violates hard bounds; do not project')
    bodies = np.array([model.body('robot/' + name).id for name in NATIVE_LINK_NAMES])
    object_address = int(model.joint('object/cube_freejoint').qposadr[0])
    links, wrist = [], []
    initial_mocap_pos, initial_mocap_quat = data.mocap_pos.copy(), data.mocap_quat.copy()
    for joint_position, object_position in zip(q, pose):
        data.qpos[qa] = joint_position
        data.qpos[object_address:object_address + 7] = object_position
        mujoco.mj_kinematics(model, data)
        links.append(data.xpos[bodies].copy())
        wrist.append(np.r_[data.xpos[bodies[0]], data.xquat[bodies[0]]])
    links, wrist = np.array(links), np.array(wrist)
    if not np.array_equal(data.mocap_pos, initial_mocap_pos) or not np.array_equal(data.mocap_quat, initial_mocap_quat):
        raise ValueError('Default robot wrist/mocap was modified')
    if np.max(np.abs(wrist - wrist[:1])) > 1e-12:
        raise ValueError('This ABI requires the declared fixed robot palm frame')
    arrays = {'reference_qpos': q, 'wrist_pose_w': wrist, 'object_pose_w': pose, 'link_pos_w': links}
    assert set(arrays) == NATIVE_KEYS
    info = {**names_contract, 'model_path': str(model_path), 'model_sha256': sha(model_path),
            'frames': len(q), 'kinematics_calls': len(q), 'physics_steps': 0,
            'wrist_source': 'Original MJB default fixed robot palm, not demonstrator wrist',
            'link_source': 'Fresh BODY FK of derived q in exact native order; not IK macro points or source link labels',
            'q_source': 'Caller-supplied observation-derived command goal, not source proprioception',
            'object_pose_changed_by_exporter': False, 'q_projected_by_exporter': False,
            'physical_reference_qualified': False}
    return arrays, info


def export_clip(output, clip_id, time_s, q, object_pose, *, model_path, constants_path, provenance):
    if not re.fullmatch(r'[A-Za-z0-9_]+', clip_id): raise ValueError('Safe local clip ID required')
    time = np.asarray(time_s, dtype=np.float64)
    if time.shape != (len(q),) or not np.isfinite(time).all() or np.max(np.abs(np.diff(time) - .020)) > 1e-8:
        raise ValueError('Native ABI currently supports declared50Hz without hidden resampling')
    if provenance.get('is_real_ego') is not False or provenance.get('source_robot_labels_used') is not False:
        raise ValueError('This controlled prototype requires explicit no-source-label provenance')
    if not provenance.get('observation_source') or not provenance.get('geometric_derivative'):
        raise ValueError('Observation and derived-geometry provenance are required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / (clip_id + '.npz')
    sidecar = output / (clip_id + '.provenance.json')
    if target.exists() or sidecar.exists(): raise FileExistsError('Never overwrite a derived reference')
    arrays, info = build_native_arrays(q, object_pose, model_path, constants_path)
    np.savez_compressed(target, **arrays)
    report = {'schema': 'observation_derived_native_reference_ABI:1', **info,
              'clip_id': clip_id, 'source_time_start_s': float(time[0]), 'frame_dt_s': .020,
              'resampling': False, 'training_authorized': False, 'physical_reference_qualified': False,
              'native_npz_sha256': sha(target), 'native_npz_path': str(target),
              'provenance': provenance, 'exporter_sha256': sha(__file__)}
    sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return target, sidecar, report
