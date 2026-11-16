"""Same-tick force partition; no solver intervention or feasibility-gate change."""
import numpy as np


def mass_product(matrix, acceleration, rowadr=None, col=None, madr=None):
    """F64 independent M*qacc from captured dense or packed matrix/mappings.

    MuJoCo-Warp can retain packed M even when model.is_sparse is false. Choose
    from the actual array layout, never that solver-factorization setting.
    """
    a = np.asarray(acceleration, np.float64)
    m = np.asarray(matrix, np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(m).all():
        raise ValueError('Finite aligned tick-by-DOF acceleration and mass matrix required')
    ticks, nv = a.shape
    if m.shape == (ticks, nv, nv):
        return np.einsum('tij,tj->ti', m, a)
    if m.ndim != 2 or m.shape[0] != ticks or any(v is None for v in [rowadr, col, madr]):
        raise ValueError('Packed M requires explicit multiplication mappings')
    r, c, p = [np.asarray(v) for v in [rowadr, col, madr]]
    if (any(v.dtype.kind not in 'iu' or v.ndim != 1 for v in [r,c,p])
            or r.shape != (nv+1,) or r[0] != 0 or r[-1] != len(c)
            or len(c) != len(p) or np.any(np.diff(r) < 0)
            or np.any(c < 0) or np.any(c >= nv) or np.any(p < 0) or np.any(p >= m.shape[1])):
        raise ValueError('Valid packed M row/column/address mappings required')
    out = np.zeros_like(a)
    for i in range(nv):
        sl = slice(r[i], r[i+1])
        out[:, i] = np.sum(m[:, p[sl]] * a[:, c[sl]], axis=1)
    return out


def partition(mass, dt, dofadr, gravity, fields, context, direct_Ma):
    """Separate actual scaled gradient from Ma drift and force-recovery error.

    All fields must come from the same solve exit, before integration/forward.
    This function cannot verify temporal ownership; the recorder must prove it.
    The quantity partitioned is the acceleration-force part of the original
    linear impulse audit, not the complete contact-label/velocity residual.
    """
    keys = ['qacc', 'efc.Ma', 'qfrc_smooth', 'qfrc_constraint']
    a, ma, smooth, constraint = [np.asarray(fields[k], np.float64) for k in keys]
    grad = np.asarray(context['grad'], np.float64)
    scale = np.asarray(context['grad_scale'], np.float64)
    direct = np.asarray(direct_Ma, np.float64)
    g = np.asarray(gravity, np.float64)
    if (a.ndim != 2 or dofadr < 0 or a.shape[1] < dofadr+3
            or any(v.shape != a.shape for v in [ma,smooth,constraint,grad,direct])
            or scale.shape != (a.shape[0],) or g.shape != (3,)
            or mass <= 0 or dt <= 0 or not np.isfinite([mass,dt]).all()
            or not all(np.isfinite(v).all() for v in [a,ma,smooth,constraint,grad,scale,direct,g])):
        raise ValueError('Finite same-tick fields, actual grad_scale and free-body linear DOFs required')
    scaled = grad * scale[:, None]
    sl = slice(dofadr,dofadr+3)
    terms = dict(free_mass_chart=dt*(mass*a[:,sl]-direct[:,sl]),
                 Ma_accumulation=dt*(direct[:,sl]-ma[:,sl]),
                 actual_scaled_gradient=dt*scaled[:,sl],
                 gradient_force_recovery=dt*(ma-smooth-constraint-scaled)[:,sl],
                 smooth_minus_gravity=dt*(smooth[:,sl]-mass*g))
    total = dt*(mass*a[:,sl]-constraint[:,sl]-mass*g)
    reconstructed = sum(terms.values())
    budget = 32*np.finfo(np.float64).eps*(1+sum(abs(v) for v in terms.values()))
    return dict(total_acceleration_force_Ns=total, **terms, reconstructed_Ns=reconstructed,
                scaled_gradient_N=scaled, direct_mass_balance_N=direct-smooth-constraint,
                stored_Ma_balance_N=ma-smooth-constraint,
                gradient_force_recovery_N=ma-smooth-constraint-scaled,
                recomposition_max_abs=float(abs(total-reconstructed).max()),
                recomposition_within_float64=bool(np.all(abs(total-reconstructed)<=budget)))


def terminal_predicates(nv, meaninertia, tolerance, iteration_cap, fields):
    """Report terminal eligibility predicates, never claim a unique stop cause.

    Several OR predicates may be true. F64 re-evaluation near a float32 branch
    boundary is descriptive; actual iteration-local branches must be recorded
    separately if the precise terminating predicate matters.
    """
    values = {k:np.asarray(fields[k],np.float64) for k in
              ['grad_dot','newton_decrement','improvement','solver_niter']}
    shape = values['grad_dot'].shape
    if (len(shape)!=1 or any(v.shape!=shape or not np.isfinite(v).all() for v in values.values())
            or nv<=0 or meaninertia<=0 or tolerance<0 or iteration_cap<0
            or not np.isfinite([meaninertia,tolerance]).all() or np.any(values['grad_dot']<0)):
        raise ValueError('Finite aligned terminal scalars and explicit solver settings required')
    denominator = meaninertia*nv
    return dict(improvement_below_tolerance=values['improvement']/denominator<tolerance,
                gradient_below_tolerance=np.sqrt(values['grad_dot'])/denominator<tolerance,
                newton_decrement_below_tolerance=.5*values['newton_decrement']/denominator<tolerance,
                at_iteration_cap=values['solver_niter']==iteration_cap)
