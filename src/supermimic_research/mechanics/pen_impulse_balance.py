"""Recorded free-COM linear impulse partition;no contact/solver threshold change."""
import numpy as np


def partition(mass,dt,pre_velocity,post_velocity,acceleration,constraint_force,contact_force,native_gravity,decoder_gravity):
    v0,v1,a,fc,fl=[np.asarray(x,np.float64) for x in [pre_velocity,post_velocity,acceleration,constraint_force,contact_force]]
    if v0.ndim!=2 or v0.shape[1]!=3 or any(x.shape!=v0.shape for x in [v1,a,fc,fl]) or mass<=0 or dt<=0:raise ValueError('Explicit positive mass/dt and aligned world-linear vectors required')
    gn,gd=np.asarray(native_gravity,np.float64),np.asarray(decoder_gravity,np.float64)
    if gn.shape!=(3,) or gd.shape!=(3,) or not all(np.isfinite(x).all() for x in [v0,v1,a,fc,fl,gn,gd]):raise ValueError('Finite gravity/recorded vectors required')
    dv=v1-v0;total=mass*dv-dt*(fl+mass*gd)
    terms=dict(velocity_update=mass*(dv-dt*a),acceleration_force=dt*(mass*a-fc-mass*gn),constraint_contact=dt*(fc-fl),gravity_representation=np.broadcast_to(dt*mass*(gn-gd),v0.shape).copy())
    reconstructed=sum(terms.values());error=abs(reconstructed-total);budget=32*np.finfo(np.float64).eps*(1+sum(abs(x) for x in terms.values()))
    return dict(total=total,**terms,reconstructed=reconstructed,recomposition_max_abs=float(error.max()),recomposition_within_float64=bool(np.all(error<=budget)))
