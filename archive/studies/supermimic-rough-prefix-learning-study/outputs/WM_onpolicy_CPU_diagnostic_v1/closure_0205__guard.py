"""Geometry-only guard; allowed native kinematics/collision/Jacobians remain."""
from contextlib import contextmanager

FORBIDDEN=('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce')

@contextmanager
def geometry_only_guard(mujoco):
    counts={name:0 for name in FORBIDDEN};original={name:getattr(mujoco,name) for name in FORBIDDEN}
    def deny(name):
        def fail(*args,**kwargs):
            counts[name]+=1
            raise AssertionError('Geometry-only worker forbids '+name)
        return fail
    try:
        for name in FORBIDDEN:setattr(mujoco,name,deny(name))
        yield counts
    finally:
        for name,value in original.items():setattr(mujoco,name,value)
