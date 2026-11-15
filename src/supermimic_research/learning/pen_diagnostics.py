"""Recorded-control recursive Fn6 evaluation; no optimizer or native simulation."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..mechanics.impulse_codec import exp_world_unchecked, rotation55_torch
from ..mechanics.transition_records import state_rotation


def window_indices(data, horizon):
    """All contiguous, numerically valid windows, never crossing episode IDs."""
    if horizon < 1: raise ValueError('Positive horizon')
    windows = []
    for case_id in np.unique(data['case_id']):
        ids = np.flatnonzero(data['case_id'] == case_id)
        ids = ids[np.argsort(data['control_index'][ids])]
        for start in range(len(ids) - horizon + 1):
            w = ids[start:start + horizon]
            if (np.all(np.diff(data['control_index'][w]) == 1)
                    and np.all(data['valid'][w])
                    and np.all(data['x'][w, -1] == data['x'][w[0], -1])):
                windows.append(w)
    return np.array(windows, dtype=int).reshape(-1, horizon)


def decode_pen_torch(state, prediction):
    """The Fn6 suffix has no effect on the shared state55 kinematic chart."""
    R = exp_world_unchecked(prediction[...,43:46]) @ rotation55_torch(state)
    return torch.cat([state[...,:43] + prediction[...,:43], R[..., :, 0], R[..., :, 1],
                      state[...,49:] + prediction[...,46:52]], dim=-1)


def recursive_prediction(model, state, warm, controls, native_dt):
    """Controls [batch,horizon,20]; only the first state/warm are observed."""
    states = [state]; outputs = []
    for k in range(controls.shape[1]):
        x = torch.cat([state, warm, controls[:,k], native_dt], dim=-1).float()
        prediction = model.prediction(x)
        state = decode_pen_torch(state, prediction); warm = prediction[:,52:78]
        outputs.append(prediction); states.append(state)
    return torch.stack(states, dim=1), torch.stack(outputs, dim=1), warm


def distribution(values):
    a = np.asarray(values, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p95=float(np.quantile(a, .95)), max=float(a.max()))


def endpoint_errors(predicted, target):
    return dict(COM_error_m=distribution(np.linalg.norm(predicted[:,40:43] - target[:,40:43], axis=1)),
        angle_error_deg=distribution(np.rad2deg(Rotation.from_matrix(
            state_rotation(predicted) @ state_rotation(target).transpose(0,2,1)).magnitude())),
        joint_position_RMS_rad=distribution(np.sqrt(np.mean((predicted[:,:20]-target[:,:20])**2, axis=1))),
        COM_velocity_error_m_s=distribution(np.linalg.norm(predicted[:,49:52]-target[:,49:52], axis=1)),
        omega_error_rad_s=distribution(np.linalg.norm(predicted[:,52:55]-target[:,52:55], axis=1)))


def rollout_diagnostics(model, data, device, horizons=(1,2,5,10)):
    results = {}; arrays = {}
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    for horizon in horizons:
        ids = window_indices(data, horizon)
        if not len(ids):
            results[str(horizon)] = dict(windows=0, duration_s=horizon*.02)
            continue
        source = tensor(data['x'][ids[:,0], :55]); warm = tensor(data['x'][ids[:,0],55:81])
        controls = tensor(data['x'][ids,81:101]); dt = tensor(data['x'][ids[:,0],-1:])
        with torch.no_grad():
            states, output, _ = recursive_prediction(model, source, warm, controls, dt)
            target = decode_pen_torch(tensor(data['x'][ids[:,-1],:55]), tensor(data['y'][ids[:,-1]]))
        states, output, target = [a.cpu().numpy() for a in (states, output, target)]
        finite = np.isfinite(states).all(axis=(1,2)) & np.isfinite(output).all(axis=(1,2))
        row = dict(windows=len(ids), finite_windows=int(finite.sum()), duration_s=horizon*.02,
                   episode_count=len(np.unique(data['case_id'][ids[:,0]])),
                   failed_or_nonfinite_windows=int((~finite).sum()))
        if finite.any():
            row['model'] = endpoint_errors(states[finite,-1], target[finite])
            row['persistence'] = endpoint_errors(states[finite,0], target[finite])
            # Rotate each predicted/actual mean wrench from its own SOURCE frame.
            pred_R = state_rotation(states[finite,:-1])
            actual_R = state_rotation(data['x'][ids[finite],:55])
            for label, sl in [('world_force_RMS_N', slice(78,81)), ('world_torque_RMS_Nm', slice(81,84))]:
                pw = np.einsum('...ij,...j->...i', pred_R, output[finite,:,sl])
                tw = np.einsum('...ij,...j->...i', actual_R, data['y'][ids[finite],sl])
                row[label] = float(np.sqrt(np.mean((pw-tw)**2)))
        arrays[f'H{horizon}_row_indices'] = ids
        arrays[f'H{horizon}_states'] = states
        arrays[f'H{horizon}_predictions'] = output
        arrays[f'H{horizon}_targets'] = target
        results[str(horizon)] = row
    return results, arrays


def network_derivative_diagnostics(model, data, device, seed=7461):
    """Numerical derivative self-consistency ONLY, not physical accuracy."""
    mask = data['valid']; values = data['x'][mask]
    directions = np.random.default_rng(seed).normal(size=(len(values),20))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    v = torch.zeros_like(x); v[:,81:101] = torch.as_tensor(directions, dtype=x.dtype, device=device)
    def normalized(z): return model.prediction(z) / model.y_scale
    _, jvp = torch.autograd.functional.jvp(normalized, x, v, create_graph=False)
    result = dict(rows=len(values), seed=seed, automatic_derivative_finite=bool(torch.isfinite(jvp).all()),
        contract='Unit PD-target20 directions at fixed state/warm/dt; normalized physical outputs. Not native physics gradients.')
    arrays = dict(directions=directions, automatic=jvp.detach().cpu().numpy())
    for epsilon in (1e-3, 3e-4, 1e-4):
        with torch.no_grad(): finite_difference = (normalized(x+epsilon*v)-normalized(x-epsilon*v))/(2*epsilon)
        row = {}
        for label, sl in [('object12',slice(40,52)), ('wrench6',slice(78,84))]:
            ad, fd = jvp[:,sl], finite_difference[:,sl]
            row[label] = dict(relative_L2=float(torch.linalg.vector_norm(ad-fd)/torch.linalg.vector_norm(ad).clamp_min(1e-12)),
                             max_abs_error=float((ad-fd).abs().max()))
        result[str(epsilon)] = row
        arrays[f'FD_{epsilon}'] = finite_difference.cpu().numpy()
    return result, arrays
