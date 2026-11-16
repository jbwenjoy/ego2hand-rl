"""Fixed-radius proposals and native-screened selection,without model updates."""
import numpy as np

RAW_RADIUS=.008
REWARD_MARGIN=.0001


def proposal(base_raw,direction,*,prediction_finite=True):
    """Retain original sampled RAW. Never project or reduce the fixed radius."""
    base=np.asarray(base_raw)
    if base.shape!=(20,) or base.dtype!=np.float32 or not np.isfinite(base).all():raise ValueError('Finite original float32 RAW20 required')
    v=np.asarray(direction,dtype=np.float64)
    if v.shape!=(20,):raise ValueError('Explicit RAW direction20 required')
    reason=None;norm=float(np.linalg.norm(v))
    if not prediction_finite:reason='NONFINITE_PREDICTION'
    elif not np.isfinite(v).all() or not np.isfinite(norm):reason='NONFINITE_DIRECTION'
    elif norm==0:reason='ZERO_DIRECTION'
    candidate=base.copy() if reason else (base.astype(np.float64)+RAW_RADIUS*v/norm).astype(np.float32)
    # A pre-existing clipped coordinate can remain byte-identical. Any changed
    # coordinate must be in bounds. This rule is shared by WM and random arms.
    if reason is None and np.any((candidate!=base)&(abs(candidate)>1)):reason='CHANGED_COORDINATE_CLIPS';candidate=base.copy()
    if reason is None and np.array_equal(candidate,base):reason='NO_FLOAT32_CHANGE'
    return dict(raw=candidate,duplicate_BASE=reason is not None,reason=reason,direction_norm=norm if np.isfinite(norm) else None,actual_raw_delta=(candidate.astype(np.float64)-base).copy())


def random_direction(seed,repetition,teacher_index,phase):
    if any(type(x) is not int or x<0 for x in [seed,repetition,teacher_index,phase]):raise ValueError('Nonnegative stateless random indices required')
    return np.random.default_rng(np.random.SeedSequence([seed,repetition,teacher_index,phase])).normal(size=20)


def choose(base,candidate):
    """Physical invalidity ends BASE; failures differ from reference completion."""
    for branch in [base]+([] if candidate is None else [candidate]):
        if any(type(branch[k]) is not bool for k in ['physical_valid','terminated','truncated']):raise ValueError('Explicit native branch outcome required')
        if branch['physical_valid'] and (branch['reward'] is None or not np.isfinite(branch['reward'])):raise ValueError('Physically valid native reward must be finite')
        if branch['terminated'] and branch['truncated']:raise ValueError('Failure and completion must be distinguished')
    if not base['physical_valid']:return dict(selected='BASE',accepted=False,stop=True,reason='BASE_PHYSICAL_INVALID')
    if candidate is None:raise ValueError('Valid screened BASE requires declared candidate query')
    accept=bool(candidate['physical_valid'] and not candidate['terminated'] and candidate['reward']>base['reward']+REWARD_MARGIN)
    selected=candidate if accept else base
    return dict(selected='ALT' if accept else 'BASE',accepted=accept,stop=bool(selected['terminated'] or selected['truncated']),
        reason='IMPROVED_VALID_NONFAILURE' if accept else 'BASE_RETAINED',selected_failure=selected['terminated'],selected_completion=selected['truncated'])
