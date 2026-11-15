"""Coverage, explicit oracle interventions, and independently executed action coverage."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .pen_diagnostics import decode_pen_torch, endpoint_errors
from ..mechanics.impulse_codec import exp_world_unchecked, rotation55_torch
from ..mechanics.transition_records import state_rotation


STATE_BLOCKS = dict(hand_q=slice(0, 20), hand_qd=slice(20, 40), COM=slice(40, 43),
                    rotation_columns=slice(43, 49), velocity=slice(49, 52), omega=slice(52, 55),
                    engine_memory=slice(55, 81), PD=slice(81, 101))
Y_BLOCKS = dict(object12=slice(40, 52), warm26=slice(52, 78), wrench6=slice(78, 84), Fn6=slice(84, 90))


def source_coverage(train, check, scales):
    def parents(d):
        return np.array([np.flatnonzero(d['parent_id'] == p)[0] for p in np.unique(d['parent_id'])])
    ti, ci = parents(train), parents(check)
    rows = []
    for i in ci:
        same = ti[train['x'][ti, -1] == check['x'][i, -1]]
        if not len(same):
            raise ValueError('No same-dt TRAIN source comparator')
        delta = (train['x'][same] - check['x'][i]) / scales['x_scale']
        distance = np.sqrt(np.mean(delta[:, :81]**2, axis=1))
        nearest = int(same[np.argmin(distance)])
        normalized = (train['x'][nearest] - check['x'][i]) / scales['x_scale']
        own = check['parent_id'] == check['parent_id'][i]
        pd_delta = (train['x'][train['valid'], 81:101] - check['x'][i, 81:101]) / scales['x_scale'][81:101]
        rows.append(dict(parent_id=int(check['parent_id'][i]), phase=int(check['phase'][i]), dt=float(check['x'][i, -1]),
            nearest_TRAIN_parent=int(train['parent_id'][nearest]), nearest_TRAIN_phase=int(train['phase'][nearest]),
            nearest_source_RMS=float(distance.min()),
            nearest_source_blocks={k: float(np.sqrt(np.mean(normalized[s]**2))) for k, s in STATE_BLOCKS.items() if k != 'PD'},
            zero_candidate_nearest_any_valid_TRAIN_PD_RMS=float(np.sqrt(np.mean(pd_delta**2, axis=1)).min()),
            distinct_actual_PD=int(len(np.unique(check['x'][own, 81:101], axis=0)))))
    counts = {role: dict(parents=len(np.unique(d['parent_id'])),
        exact_state55=len(np.unique(d['x'][:, :55], axis=0)), exact_source81=len(np.unique(d['x'][:, :81], axis=0)),
        distinct_PD=len(np.unique(d['x'][:, 81:101], axis=0)), attempted_rows=len(d['x']), valid_rows=int(d['valid'].sum()))
        for role, d in [('TRAIN', train), ('CHECK', check)]}
    return dict(counts=counts, CHECK_parents=rows,
        scope='RMS distances under frozen TRAIN input scales, same native dt. Euclidean rotation-column distances are chart features, not SO3 angles. Distances/correlations do not prove causal OOD failure.')


def constant_velocity(state, duration):
    R = exp_world_unchecked(state[:, 52:55] * duration) @ rotation55_torch(state)
    return torch.cat([state[:, :20] + duration * state[:, 20:40], state[:, 20:40],
        state[:, 40:43] + duration * state[:, 49:52], R[:, :, 0], R[:, :, 1], state[:, 49:]], dim=-1)


def oracle_rollout(model, data, ids, mode, device):
    if mode not in ['FULL', 'OBSERVED_WARM', 'OBSERVED_STATE', 'OBSERVED_BOTH']:
        raise ValueError('Explicit model-only or privileged diagnostic mode')
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    state = tensor(data['x'][ids[:, 0], :55])
    warm = tensor(data['x'][ids[:, 0], 55:81])
    states, memories, outputs = [state], [warm], []
    with torch.no_grad():
        for k in range(ids.shape[1]):
            observed = tensor(data['x'][ids[:, k]])
            if mode in ['OBSERVED_STATE', 'OBSERVED_BOTH']:
                state = observed[:, :55]
            if mode in ['OBSERVED_WARM', 'OBSERVED_BOTH']:
                warm = observed[:, 55:81]
            pred = model.prediction(torch.cat([state, warm, observed[:, 81:]], dim=-1).float())
            state, warm = decode_pen_torch(state, pred), pred[:, 52:78]
            states.append(state)
            memories.append(warm)
            outputs.append(pred)
    return torch.stack(states, 1).cpu().numpy(), torch.stack(memories, 1).cpu().numpy(), torch.stack(outputs, 1).cpu().numpy()


def rollout_block_errors(data, ids, states, memories, scales):
    records = []
    with torch.no_grad():
        actual = decode_pen_torch(torch.as_tensor(data['x'][ids, :55]), torch.as_tensor(data['y'][ids])).numpy()
    for k in range(ids.shape[1]):
        s, t, warm = states[:, k + 1], actual[:, k], memories[:, k + 1]
        finite = np.isfinite(s).all(1) & np.isfinite(warm).all(1)
        row = dict(step=k + 1, physical_time_s=(k + 1) * .02, windows=len(ids), finite=int(finite.sum()))
        if finite.any():
            row['physical'] = endpoint_errors(s[finite], t[finite])
            row['state_block_normalized_RMS'] = {name: float(np.sqrt(np.mean(
                ((s[finite, sl] - t[finite, sl]) / scales['x_scale'][sl])**2)))
                for name, sl in STATE_BLOCKS.items() if name not in ['PD', 'engine_memory']}
            row['next_memory_normalized_RMS'] = float(np.sqrt(np.mean(
                ((warm[finite] - data['y'][ids[finite, k], 52:78]) / scales['y_scale'][52:78])**2)))
        records.append(row)
    return records


def actual_action_coverage(data, parent, case, reference, gates):
    """All three independent repeats must satisfy unchanged quality/pose/geometry."""
    pid, k = parent['parent_id'], parent['k']
    ids = np.flatnonzero(data['parent_id'] == pid)
    lookup = {(int(data['repeat'][i]), int(data['candidate'][i])): i for i in ids}
    if len(lookup) != 171:
        raise ValueError('All actual three-repeat candidate responses required')
    ordered = np.array([[lookup[r, a] for a in range(57)] for r in range(3)])
    with torch.no_grad():
        state = decode_pen_torch(torch.as_tensor(data['x'][ordered, :55]), torch.as_tensor(data['y'][ordered])).numpy()
    COM = np.linalg.norm(state[:, :, 40:43] - reference['COM_world_m'][k + 1], axis=-1)
    actual_R = state_rotation(state)
    target_R = Rotation.from_quat(reference['quaternion_wxyz'][k + 1][[1, 2, 3, 0]]).as_matrix()
    angle = Rotation.from_matrix((actual_R @ target_R.T).reshape(-1, 3, 3)).magnitude().reshape(3, 57)
    geometry = np.all(data['geometry'][ordered] <= np.array([
        gates['max_penetration_m'], gates['max_self_penetration_m'], gates['max_joint_limit_excursion_rad']]), axis=-1)
    quality = np.all(data['valid'][ordered], axis=0)
    safe = np.all(geometry, axis=0) & quality
    position_pass = np.all(COM <= gates['COM_max_m'], axis=0)
    rotation_pass = np.all(angle <= gates['angle_max_rad'], axis=0)
    both = safe & position_pass & rotation_pass
    decision = case['planning'][k]
    if decision['k'] != k or decision['phase'] != k + 1:
        raise ValueError('Original reference/control clock required')
    selected = decision['selected']
    return dict(parent_id=pid, role=parent['role'], mode=parent['mode'], dt=parent['dt'], k=k,
        quality_candidates=int(quality.sum()), geometry_and_quality_candidates=int(safe.sum()),
        COM_candidates=int((safe & position_pass).sum()), full_orientation_candidates=int((safe & rotation_pass).sum()),
        joint_pose_candidates=int(both.sum()), selected=selected,
        selected_quality=bool(quality[selected]), selected_joint_pose=bool(both[selected]),
        min_worst_repeat_COM_m=float(COM[:, safe].max(0).min()) if safe.any() else None,
        min_worst_repeat_full_angle_deg=float(np.rad2deg(angle[:, safe].max(0).min())) if safe.any() else None,
        scope='Finite table, independently executed responses; original endpoint max pose/geometry gates, no RMS/duration or global infeasibility claim.')
