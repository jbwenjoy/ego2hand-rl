"""Independent NumPy-only original-gate and all-row descriptive arithmetic."""
import numpy as np

BLOCKS = {'q_rad': (0, 20), 'qd_rad_s': (20, 40), 'p_m': (40, 43),
          'R_dimensionless': (43, 52), 'v_m_s': (52, 55), 'omega_rad_s': (55, 58)}


def finite_f32(a, shape):
    a = np.asarray(a)
    if a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Saved array must have exact finite FP32 ABI')
    return a


def old_rule(actual, reference, state58):
    a, b = np.asarray(actual), np.asarray(reference)
    if a.ndim != 2 or a.shape != b.shape or a.dtype != b.dtype or a.dtype != np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Malformed saved comparison')
    if state58 and a.shape[1] != 58:
        raise ValueError('Mandatory full58 comparison')
    delta = a.astype('f8')-b.astype('f8')
    bounds = 8*np.finfo('f4').eps*np.maximum(1., np.maximum(abs(a.astype('f8')), abs(b.astype('f8'))))
    flags = abs(delta) <= bounds
    if state58:
        flags[:, :40] = a[:, :40].view('u4') == b[:, :40].view('u4')
    return {'difference_F64': delta, 'old_bound_F64': bounds, 'old_flags': flags,
            'all_passed': bool(flags.all()), 'failed_elements': int((~flags).sum()),
            'failed_rows': int((~flags).any(-1).sum())}


def all2260_gpu_cpu_description(gpu, cpu):
    """No tolerance/classification: CUDA+B1 versus CPU+B2260 is confounded."""
    gpu, cpu = finite_f32(gpu, (2260, 58)), finite_f32(cpu, (2260, 58))
    delta = gpu.astype('f8')-cpu.astype('f8')
    summary = {}
    for name, (start, stop) in BLOCKS.items():
        d = delta[:, start:stop]
        maximum = float(abs(d).max())
        summary[name] = {'rows': 2260, 'component_count': stop-start,
            'maximum_absolute': maximum, 'mean_absolute': float(abs(d).mean()),
            'RMS': float(np.sqrt(np.mean(d*d))),
            'FP32_byte_different_elements': int((gpu[:, start:stop].view('u4') != cpu[:, start:stop].view('u4')).sum()),
            'maximum_row_component_indices': np.argwhere(abs(d) == maximum).tolist() if maximum > 0 else [],
            'scope': 'Descriptive only; backend and batch shape change together.'}
    return delta, summary
