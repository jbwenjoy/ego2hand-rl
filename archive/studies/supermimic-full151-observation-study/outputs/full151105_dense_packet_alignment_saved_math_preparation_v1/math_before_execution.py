"""Independent saved-pose-only supplemental alignment arithmetic.

Caller must first bind the complete matched native full58 replay qualification.
This extra guard never replaces q/qd/velocity gates or authorizes proxy scoring.
No state/quat/FK/native/model method is invoked: worldR is saved forensic data.
"""
import numpy as np


def alignment(packet_x, palm_pos, palm_R, solver_world_pos, solver_world_R, expected_rows=1808):
    shape_values = [(packet_x, (expected_rows, 58)), (palm_pos, (expected_rows, 3)),
        (palm_R, (expected_rows, 3, 3)), (solver_world_pos, (expected_rows, 3)),
        (solver_world_R, (expected_rows, 3, 3))]
    arrays = []
    for value, shape in shape_values:
        a = np.asarray(value)
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('Require every original finite FP32 saved alignment row')
        arrays.append(a)
    x, pp, R, pos, objR = arrays
    inverse = R.astype('f8').swapaxes(-1, -2)
    world_delta = pos.astype('f8')-pp.astype('f8')
    local_p = (inverse*world_delta[:, None, :]).sum(-1)
    local_R = (inverse[..., :, :, None]*objR.astype('f8')[..., None, :, :]).sum(-2)
    reference = np.concatenate((local_p, local_R.reshape(expected_rows, 9)), -1)
    actual = x[:, 40:52].astype('f8')
    difference = actual-reference
    bound = 8*np.finfo('f4').eps*np.maximum(1., np.maximum(abs(actual), abs(reference)))
    flags = abs(difference) <= bound
    return {'actual_pose12_F64': actual, 'reference_pose12_F64': reference,
        'difference_F64': difference, 'old_bound_F64': bound, 'old_flags': flags,
        'alignment_all_passed': bool(flags.all()),
        'position_maximum_absolute_difference_m': float(abs(difference[:, :3]).max()),
        'rotation_maximum_absolute_difference': float(abs(difference[:, 3:]).max()),
        'failed_rows': int((~flags).any(-1).sum()), 'failed_elements': int((~flags).sum()),
        'scope': 'Supplemental saved world-to-original-Reader-palm alignment only; no replacement full58 qualification.'}
