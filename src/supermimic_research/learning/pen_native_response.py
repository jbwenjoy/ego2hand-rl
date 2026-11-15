"""Finite-amplitude response comparisons with explicit execution spread."""
import numpy as np


def vector_comparison(prediction, target):
    prediction, target = np.asarray(prediction,float), np.asarray(target,float)
    pn, tn = np.linalg.norm(prediction), np.linalg.norm(target)
    return dict(prediction_L2=float(pn), target_L2=float(tn),
        relative_L2_error=float(np.linalg.norm(prediction-target)/max(tn,1e-12)),
        cosine=float(prediction@target/(pn*tn)) if pn*tn > 1e-24 else None)


def compare_native_response(plus, minus, derivative, scales, epsilon):
    """Rows are execution repeats at the same fixed inputs, not IID samples."""
    plus, minus, derivative, scales = [np.asarray(a,float) for a in (plus,minus,derivative,scales)]
    if (plus.shape != minus.shape or plus.ndim != 2 or plus.shape[0] < 2
            or derivative.shape != scales.shape or plus.shape[1:] != scales.shape
            or np.any(scales<=0) or epsilon<=0
            or not all(np.isfinite(a).all() for a in (plus,minus,derivative,scales))):
        raise ValueError('Finite paired repeated responses and positive scales/amplitude')
    p, m = plus/scales, minus/scales
    contrast = p.mean(0)-m.mean(0); fd = contrast/(2*epsilon)
    spread = max(np.linalg.norm(p-p.mean(0),axis=1)) + max(np.linalg.norm(m-m.mean(0),axis=1))
    result = vector_comparison(derivative/scales,fd)
    result.update(execution_repeats=len(p), central_response_normalized_L2=float(np.linalg.norm(contrast)),
        observed_contrast_spread_bound_L2=float(spread),
        contrast_exceeds_observed_spread=bool(np.linalg.norm(contrast)>spread),
        central_derivative_normalized_L2=float(np.linalg.norm(fd)),
        caveat='Observed3-repeat spread is not a probabilistic bound or proof of differentiability.')
    return result, fd
