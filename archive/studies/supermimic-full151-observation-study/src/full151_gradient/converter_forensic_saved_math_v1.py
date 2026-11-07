"""Forensic-only saved-array arithmetic. No converter or model is imported.

Never turns old8eps failure into pass. Every row/component is preserved; the
new diagnostic may complete while its original arithmetic masks remain false.
"""
import numpy as np

BLOCKS58 = {'q': (0, 20), 'qd': (20, 40), 'p_reader': (40, 43),
            'R_reader_rowmajor': (43, 52), 'v_reader': (52, 55),
            'omega_reader_from_BODY': (55, 58)}
EXPECTED = {'converted': 2260, 'pre': 1808, 'endpoint': 452, 'anchor_comparisons': 160,
            'solver_pose_comparisons': 1808, 'factory': 0, 'proxy_forwards': 0}


def error_arrays(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32 or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Require same-shaped finite savedFP32 actual/reference')
    error = a.astype('f8')-b.astype('f8')
    scale = np.maximum(1., np.maximum(abs(a.astype('f8')), abs(b.astype('f8'))))
    bound = 8*np.finfo('f4').eps*scale
    return {'signed_error_F64': error, 'abs_error_F64': abs(error), 'bound_original8eps32': bound,
            'original8eps32_pass_mask': abs(error) <= bound,
            'abs_error_over_original_bound': abs(error)/bound,
            'FP32_exact_bit_mask': a.view('u4') == b.view('u4')}


def state58_errors(actual, reference, expected_rows):
    if np.asarray(actual).shape != (expected_rows, 58):
        raise ValueError('All expected state58 rows are mandatory')
    arrays = error_arrays(actual, reference)
    mask = arrays['original8eps32_pass_mask']
    exact = arrays['FP32_exact_bit_mask']
    summary = {}
    for name, (start, stop) in BLOCKS58.items():
        errors = arrays['abs_error_F64'][:, start:stop]
        passed = mask[:, start:stop]
        summary[name] = {'rows': expected_rows, 'components': stop-start,
            'failed_arithmetic_components': int((~passed).sum()),
            'failed_arithmetic_rows': int((~passed).any(axis=1).sum()),
            'maximum_abs_error': float(errors.max()),
            'maximum_ratio_to_original_bound': float(arrays['abs_error_over_original_bound'][:, start:stop].max()),
            'exact_bit_differences': int((~exact[:, start:stop]).sum())}
    return arrays, {'blocks': summary, 'q_qd_exact_original_rule': bool(exact[:, :40].all()),
        'other18_original_arithmetic_rule': bool(mask[:, 40:].all()),
        'original_gate_qualified': bool(exact[:, :40].all() and mask[:, 40:].all()),
        'meaning': 'Forensic diagnostic completion is independent of original gate qualification.'}


def transform_cancellation_inventory(A, x):
    """A-posteriori arithmetic scale diagnostic, never an acceptance criterion.

    Inputs are already saved matrices/vectors in the same frame. This does not
    identify a backend error cause, regenerate a pose, or propose a new bound.
    """
    A, x = np.asarray(A), np.asarray(x)
    if A.shape[-2:] != (3, 3) or x.shape != A.shape[:-2]+(3,) or not np.isfinite(A).all() or not np.isfinite(x).all():
        raise ValueError('Saved3x3/3 transform inventory mismatch')
    products = A.astype('f8')*x.astype('f8')[..., None, :]
    result = products.sum(axis=-1)
    sum_abs = abs(products).sum(axis=-1)
    return {'F64_saved_product_sum': result, 'F64_sum_absolute_products': sum_abs,
            'cancellation_amount_not_error_bound': sum_abs-abs(result),
            'input_vector_norm': np.linalg.norm(x.astype('f8'), axis=-1)}
