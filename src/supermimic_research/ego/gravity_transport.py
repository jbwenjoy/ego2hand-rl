"""Conditional gravity transport under a time-varying fixed-palm reanchoring."""
import numpy as np
from supermimic_research.runtime.ego_reference import validate_transform


def gravity_transport_diagnostic(source_world_from_palm, robot_world_from_palm,
                                 assumed_source_gravity, robot_gravity):
    human = validate_transform(source_world_from_palm)
    robot = validate_transform(robot_world_from_palm)
    g = np.asarray(assumed_source_gravity, dtype=np.float64)
    gr = np.asarray(robot_gravity, dtype=np.float64)
    if human.ndim != 3 or len(human) < 2 or robot.shape != (4, 4):
        raise ValueError('Batched source palm poses and one fixed robot palm required')
    if g.shape != (3,) or gr.shape != (3,) or not np.isfinite(np.r_[g, gr]).all() or min(np.linalg.norm(g), np.linalg.norm(gr)) <= 0:
        raise ValueError('Explicit finite nonzero assumed/robot gravity vectors required')
    A = robot[:3, :3] @ human[:, :3, :3].transpose(0, 2, 1)
    mapped = A @ g
    D = A - A[:1]
    # min_|u|=1 mean_i |(A_i-A_0)u|^2 = lambda_min(mean D_i.T D_i).
    Q = np.einsum('nji,njk->ik', D, D) / len(D)
    eigenvalues, directions = np.linalg.eigh(Q)
    if eigenvalues[0] < -1e-12:
        raise ValueError('Unexpected non-PSD transport quadratic form')
    best = directions[:, 0]
    best_change = D @ best
    bound = float(np.sqrt(max(0., eigenvalues[0])))
    svd_bound = float(np.linalg.svd(D.reshape(-1, 3), compute_uv=False)[-1] / np.sqrt(len(D)))
    normalized = mapped / np.linalg.norm(g)
    angles_robot = np.arccos(np.clip(normalized @ (gr / np.linalg.norm(gr)), -1, 1))
    angles_initial = np.arccos(np.clip(normalized @ normalized[0], -1, 1))
    return dict(source_to_robot_rotation=A, conditional_mapped_gravity=mapped,
                conditional_angle_to_robot_rad=angles_robot,
                conditional_angle_to_initial_rad=angles_initial,
                best_unit_source_direction=best,
                metrics=dict(
                    minimum_RMS_unit_gravity_change_from_initial=bound,
                    independent_SVD_minimum=svd_bound,
                    minimizing_direction_RMS_change=float(np.sqrt(np.mean(np.sum(best_change ** 2, axis=1)))),
                    conditional_RMS_gravity_change_m_s2=float(np.sqrt(np.mean(np.sum((mapped - mapped[:1]) ** 2, axis=1)))),
                    conditional_angle_to_robot_range_deg=(angles_robot[[np.argmin(angles_robot), np.argmax(angles_robot)]] * 180 / np.pi).tolist(),
                    conditional_angle_to_initial_max_deg=float(angles_initial.max() * 180 / np.pi)),
                source_gravity_verified=False, source_force_ground_truth_created=False,
                scope='Gravity vector is an explicit uncalibrated assumption. The all-direction minimum is a geometric diagnostic,not gravity estimation. Even zero change is insufficient for dynamic equivalence:time-dependent translations/rotations remove wrist motion and introduce inertial-frame terms. No source force labels or fixed-wrist feasibility claim.')
