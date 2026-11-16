"""Explicit derived collision scenes; never a physics checkpoint migration."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def tensor_field(state, name, *, entity=None):
    matches = [(p, v) for p, v in state['native']['python']['tensors'].items()
               if p and p[-1] == ('attr', name) and (entity is None or ('item', entity) in p)]
    if len(matches) != 1:
        raise ValueError('One canonical retained tensor required: ' + repr((name, entity, [p for p, _ in matches])))
    return matches[0][1]


def direct_joint_contract(state):
    for name in ['encoder_bias', 'default_joint_pos']:
        t = tensor_field(state, name, entity='robot')
        if t.shape != (32, 20) or bool(t.any()):
            raise ValueError('Direct retained policy joints require exact zero bias/default')
    qids = tensor_field(state, 'qids')
    if qids.shape != (20,) or len(torch.unique(qids)) != 20:
        raise ValueError('Explicit20 unique native qpos indices required')
    return qids.numpy().copy()


def audit_boundary(state, memory, row, world):
    """Compare retained current policy/time and source-body contact frame."""
    if row < 1 or bool(memory['terminated'][row - 1, world]) or bool(memory['truncated'][row - 1, world]):
        raise ValueError('Adjacent non-reset boundary required')
    policy = memory['obs'][row, world]
    mount = tensor_field(state, 'mount')[world]
    wt = tensor_field(state, 'policy_WT')[world]
    center = memory['contact_source_C'][row, world]
    matrix = memory['contact_source_R'][row, world]
    local = wt @ (center - mount[:3])
    packed_position = .15 * torch.tanh(local / .15)
    relative = wt @ matrix
    packed_rotation = torch.cat((relative[:, 0], relative[:, 1]))
    result = dict(row=row, world=world,
        current_policy_bytes_equal_previous_terminal=torch.equal(policy, memory['terminal_next_policy'][row - 1, world]),
        time_equal=float(memory['contact_clock'][row, world, 0]) == float(memory['next_time'][row - 1, world]),
        phase_equal=int(memory['phase_source'][row, world]) == int(memory['phase_next'][row - 1, world]),
        position_feature_max_abs=float((packed_position - policy[132:135]).abs().max()),
        rotation_feature_max_abs=float((packed_rotation - policy[135:141]).abs().max()),
        rotation_orthogonality_max_abs=float((matrix.T @ matrix - torch.eye(3)).abs().max()),
        rotation_determinant=float(torch.linalg.det(matrix)))
    # These are only frame-consistency tolerances for a derived diagnostic;
    # no task geometry/overflow gate is changed.
    result['admitted_for_derived_query'] = all(result[k] for k in [
        'current_policy_bytes_equal_previous_terminal', 'time_equal', 'phase_equal']) and (
        result['position_feature_max_abs'] < 1e-5 and result['rotation_feature_max_abs'] < 1e-5
        and result['rotation_orthogonality_max_abs'] < 1e-5 and abs(result['rotation_determinant'] - 1) < 1e-5)
    return result


def derive_qpos(state, memory, row, world, object_qadr):
    audit = audit_boundary(state, memory, row, world)
    if not audit['admitted_for_derived_query']:
        raise ValueError('Retained boundary is not temporally/frame consistent: ' + repr(audit))
    qids = direct_joint_contract(state)
    if set(qids) & set(range(object_qadr, object_qadr + 7)):
        raise ValueError('Object free joint overlaps robot qpos indices')
    qpos = state['native']['native']['data']['arrays']['qpos'].copy()
    qpos[world, qids] = memory['obs'][row, world, 92:112].numpy()
    qpos[world, object_qadr:object_qadr + 3] = memory['contact_source_C'][row, world].numpy()
    source = memory['contact_source_R'][row, world].numpy()
    # SciPy projects the finite near-rotation matrix; this is explicitly a
    # derived quaternion, not the unavailable original qpos quaternion bytes.
    xyzw = Rotation.from_matrix(source.astype(np.float64)).as_quat()
    qpos[world, object_qadr + 3:object_qadr + 7] = np.roll(xyzw, 1).astype(np.float32)
    return qpos, audit
