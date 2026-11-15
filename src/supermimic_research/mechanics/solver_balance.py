"""Read-only free-body linear balance diagnostics; no feasibility gate changes."""
import numpy as np


def linear_solver_balance(packet,fields,mass,gravity,dofadr,native_dt):
    """Use recorded post-solver fields,without assuming solver convergence.

    Ma-smooth-constraint is a public-data force-balance proxy. It is not the
    private,scaled SolverContext gradient or the solver termination reason.
    """
    gravity=np.asarray(gravity,dtype=np.float64)
    if mass<=0 or native_dt<=0 or gravity.shape!=(3,) or dofadr<0:raise ValueError('Positive mass/dt and explicit WORLD gravity/free-body DOFs')
    keys=['qacc','qacc_smooth','qfrc_smooth','qfrc_constraint','qfrc_bias','qfrc_passive','qfrc_actuator','efc.Ma']
    values={key:np.asarray(fields[key],dtype=np.float64) for key in keys}
    shape=values['qacc'].shape
    if len(shape)!=2 or shape[1]<dofadr+3 or any(v.shape!=shape or not np.isfinite(v).all() for v in values.values()) or not np.isfinite([mass,native_dt,*gravity]).all():raise ValueError('Finite aligned post-solver arrays')
    sl=slice(dofadr,dofadr+3);acc=values['qacc'][:,sl];Ma=values['efc.Ma'][:,sl];smooth=values['qfrc_smooth'][:,sl];constraint=values['qfrc_constraint'][:,sl]
    before=np.asarray(packet['pre_qvel'],dtype=np.float64);after=np.asarray(packet['post_qvel'],dtype=np.float64)
    if before.shape!=shape or after.shape!=shape or not np.isfinite(before).all() or not np.isfinite(after).all():raise ValueError('Aligned finite velocity epochs')
    return dict(velocity_update_residual_m_s=after[:,sl]-before[:,sl]-native_dt*acc,
        mass_qacc_minus_smooth_constraint_N=mass*acc-smooth-constraint,
        public_Ma_balance_proxy_N=Ma-smooth-constraint,Ma_minus_mass_qacc_N=Ma-mass*acc,
        smooth_minus_gravity_N=smooth-mass*gravity,passive_N=values['qfrc_passive'][:,sl],actuator_N=values['qfrc_actuator'][:,sl],
        bias_plus_gravity_N=values['qfrc_bias'][:,sl]+mass*gravity)
