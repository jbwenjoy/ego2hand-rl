"""UNVALIDATED scalar design reference; not connected to training or physics.

Future production code must be one shared FP32 tensor kernel for real and
imagined paths. This reference has not been executed or tested. Parameters
must come from an explicitly frozen calibration artifact before enablement.
"""
import math


def proposed_filter_one(q, qd, proposed_u, *, hard_lower, hard_upper,
                        ctrl_lower, ctrl_upper, kp, kd_signed, force_limit,
                        dt, delta0=None, a_brake_empirical=None):
    """Return only a command; q/qd are inputs and are never changed."""
    if delta0 is None or a_brake_empirical is None:
        raise ValueError("Draft parameters are uncalibrated; controller disabled")
    values=(q,qd,proposed_u,hard_lower,hard_upper,ctrl_lower,ctrl_upper,kp,kd_signed,force_limit,dt,delta0,a_brake_empirical)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite controller input")
    if not (kp>0 and kd_signed<=0 and force_limit>0 and dt>0 and a_brake_empirical>0):
        raise ValueError("Requires verified unit-gear affine position actuator and positive calibration")
    cmd_lo,cmd_hi=max(hard_lower,ctrl_lower),min(hard_upper,ctrl_upper)
    if cmd_lo>cmd_hi or not 0<=delta0<(hard_upper-hard_lower)/2:
        raise ValueError("Invalid command domain or engineering reserve")
    vp,vm=max(qd,0.),max(-qd,0.)
    dp=dt*vp+vp*vp/(2*a_brake_empirical)
    dm=dt*vm+vm*vm/(2*a_brake_empirical)
    upper_risk=q+dp>=hard_upper-delta0
    lower_risk=q-dm<=hard_lower+delta0
    lo,hi=max(cmd_lo,hard_lower+delta0),min(cmd_hi,hard_upper-delta0)
    if upper_risk:hi=min(hi,q-delta0-dt*vp)
    if lower_risk:lo=max(lo,q+delta0+dt*vm)
    infeasible=lo>hi
    if not infeasible:
        u=min(max(proposed_u,lo),hi)
        fallback="none"
    else:
        # Empirical braking fallback, NOT a feasibility/safety certificate.
        # Native actuator saturation remains unchanged in the simulator.
        up_excess=q+dp-(hard_upper-delta0)
        low_excess=(hard_lower+delta0)-(q-dm)
        if up_excess>low_excess:tau_desired=-force_limit
        elif low_excess>up_excess:tau_desired=force_limit
        else:tau_desired=-force_limit if qd>=0 else force_limit
        brake_u=q+(tau_desired-kd_signed*qd)/kp
        u=min(max(brake_u,cmd_lo),cmd_hi)
        fallback="bounded_inward_PD_target; reserved domain may be violated"
    tau_affine=kp*(u-q)+kd_signed*qd
    tau_native_clip=min(max(tau_affine,-force_limit),force_limit)
    return u,{"proposed_u":proposed_u,"executed_u":u,"q_input":q,"qd_input":qd,
              "reserve_upper_rad":dp,"reserve_lower_rad":dm,"upper_risk":upper_risk,"lower_risk":lower_risk,
              "candidate_lower":lo,"candidate_upper":hi,"infeasible_interval":infeasible,
              "fallback":fallback,"affine_PD_force_estimate":tau_affine,"native_clipped_force_estimate":tau_native_clip,
              "state_was_projected":False,"rate_limit_imposed":False}
