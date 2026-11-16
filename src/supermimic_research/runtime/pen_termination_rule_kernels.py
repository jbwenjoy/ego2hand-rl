"""Prospective solver-rule comparator; not admitted as an old physical backend."""
import warp as wp
from mujoco_warp._src.solver import _rescale


@wp.kernel
def candidate_solve_done(
    nv: int,
    opt_tolerance: wp.array[float],
    opt_iterations: int,
    stat_meaninertia: wp.array[float],
    ctx_grad_dot_in: wp.array[float],
    ctx_newton_decrement_in: wp.array[float],
    ctx_improvement_in: wp.array[float],
    ctx_done_in: wp.array[bool],
    solver_niter_out: wp.array[int],
    nsolving_out: wp.array[int],
    ctx_done_out: wp.array[bool],
):
    worldid = wp.tid()
    if ctx_done_in[worldid]:
        return
    solver_niter_out[worldid] += 1
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]
    improvement = _rescale(nv, meaninertia, ctx_improvement_in[worldid])
    gradient = _rescale(nv, meaninertia, wp.sqrt(ctx_grad_dot_in[worldid]))
    model_improvement = _rescale(nv, meaninertia, 0.5 * ctx_newton_decrement_in[worldid])
    done = (gradient < tolerance) or (model_improvement < tolerance)
    if done or solver_niter_out[worldid] == opt_iterations:
        ctx_done_out[worldid] = True
        wp.atomic_add(nsolving_out, 0, -1)
