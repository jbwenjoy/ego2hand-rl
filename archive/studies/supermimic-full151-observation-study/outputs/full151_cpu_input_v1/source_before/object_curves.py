"""Full151 object Hermite curves using the frozen prefix SO(3) primitives.

All150 intervals are constructed from their own allowed mapped poses and the
previously declared full151 designed FD derivatives. These are reference
derivatives, never measured velocities or inferred forces. No extrapolation.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,importlib.util,sys
import numpy as np

PRIMITIVES=Path('D:/workspace/supermimic-observation-prefix-study/src/prefix_curves.py')
PRIMITIVES_SHA='aa03e4c8584c600875a7a6737f1c2f6d72fe63d4dd36be2e31f858ceb0bde384'


def primitive_math():
    if hashlib.sha256(PRIMITIVES.read_bytes()).hexdigest()!=PRIMITIVES_SHA:raise ValueError('Frozen SO3 primitives changed')
    name='_full151_frozen_prefix_math'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,PRIMITIVES)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


class ObjectCurves151:
    def __init__(self,time_s,pose,v,a,omega,alpha):
        m=primitive_math();t=m.finite(time_s,(151,));p=m.finite(pose,(151,7))
        if not np.array_equal(t,np.arange(151)*.02):raise ValueError('Exact full151 clock required')
        v,a,omega,alpha=(m.finite(x,(151,3)) for x in (v,a,omega,alpha))
        rotations=m.quaternion_matrices(p[:,3:]);pc=[];rc=[]
        for i,h in enumerate(np.diff(t)):
            pc.append(m.coefficients(p[i,:3],p[i+1,:3],v[i],v[i+1],a[i],a[i+1],h))
            phi=m.so3_log(rotations[i].T@rotations[i+1]);jr=m.right_jacobian(phi)
            d0=rotations[i].T@omega[i];d1=np.linalg.solve(jr,rotations[i+1].T@omega[i+1])
            a0=rotations[i].T@alpha[i]
            a1=np.linalg.solve(jr,rotations[i+1].T@alpha[i+1]-m.right_jacobian_directional(phi,d1)@d1)
            rc.append(m.coefficients(np.zeros(3),phi,d0,d1,a0,a1,h))
        self.time_s=t.copy();self.pose=p.copy();self.rotations=rotations.copy()
        self.p_coefficients=np.asarray(pc);self.phi_coefficients=np.asarray(rc)

    def sample(self,times):
        m=primitive_math();times=m.finite(times).reshape(-1)
        if not len(times) or np.any(times<0) or np.any(times>3):raise ValueError('No extrapolation or missing times')
        keys=('object_COM_world_m','object_v_world_m_s','object_a_world_m_s2',
              'object_rotation_w','object_omega_world_rad_s','object_alpha_world_rad_s2')
        result={key:[] for key in keys}
        for t in times:
            i=min(int(np.searchsorted(self.time_s,t,side='right')-1),149)
            h=self.time_s[i+1]-self.time_s[i];s=(t-self.time_s[i])/h
            for order,key in enumerate(keys[:3]):result[key].append(m.evaluate(self.p_coefficients[i],s,h,order))
            phi,dphi,ddphi=(m.evaluate(self.phi_coefficients[i],s,h,order) for order in range(3))
            R=self.rotations[i]@m.so3_exp(phi);jr=m.right_jacobian(phi)
            result['object_rotation_w'].append(R)
            result['object_omega_world_rad_s'].append(R@(jr@dphi))
            result['object_alpha_world_rad_s2'].append(R@(jr@ddphi+m.right_jacobian_directional(phi,dphi)@dphi))
        return {k:np.asarray(v) for k,v in result.items()}
