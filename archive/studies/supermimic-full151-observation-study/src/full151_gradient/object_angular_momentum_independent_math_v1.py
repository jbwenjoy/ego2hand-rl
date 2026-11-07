"""Scalar/vector angular-momentum observer; independent of producer matrices.

NumPy is used only for typed storage. Rotations use quaternion/vector cross
products; force rotations/sums use scalar fsum. No model/native imports.
"""
import math
import numpy as np


def require(v,message):
    if not v:raise ValueError(message)


def finite(a,shape,label,dtype=None):
    a=np.asarray(a);require(a.shape==shape and np.isfinite(a).all() and (dtype is None or a.dtype==np.dtype(dtype)),'Finite typed '+label)
    return a


def vector3(a):return [float(x) for x in finite(a,(3,),'vector3')]


def cross(a,b):
    a,b=vector3(a),vector3(b)
    return [math.fsum((a[1]*b[2],-a[2]*b[1])),math.fsum((a[2]*b[0],-a[0]*b[2])),math.fsum((a[0]*b[1],-a[1]*b[0]))]


def normalize_q32(q):
    raw=finite(q,(4,),'original wxyz',np.float32);values=[float(x) for x in raw]
    norm=math.sqrt(math.fsum(x*x for x in values))
    return norm,None if norm==0 else [x/norm for x in values]


def rotate_unit_q(q,v):
    """Active wxyz rotation v+2*w*(u cross v)+2*u cross(u cross v)."""
    q=[float(x) for x in finite(q,(4,),'unit quaternion')];v=vector3(v)
    uv=cross(q[1:],v);uuv=cross(q[1:],uv)
    return [math.fsum((v[i],2*q[0]*uv[i],2*uuv[i])) for i in range(3)]


def endpoint_world_L(raw_q32,omega_body32,inertia32,inertial_q32):
    inertia=finite(inertia32,(3,),'actual principal inertia',np.float32);require((inertia>0).all(),'Positive actual inertia')
    omega=finite(omega_body32,(3,),'raw body omega',np.float32)
    inertial_norm,qi=normalize_q32(inertial_q32);require(qi is not None,'Nonzero actual inertial quaternion prerequisite')
    norm,qb=normalize_q32(raw_q32)
    if qb is None:return {'endpoint_valid':False,'body_quaternion_norm_F64':norm,'inertial_quaternion_norm_F64':inertial_norm,
      'world_L_F64':np.full(3,np.nan),'body_L_F64':np.full(3,np.nan),'body_unit_quaternion_F64':np.full(4,np.nan),'inertial_unit_quaternion_F64':np.asarray(qi)}
    # Principal coordinates Qi^T*omega, scale diag(I), then return to body and
    # world. No production 3x3 rotation or inertia-matrix chain is called.
    principal_omega=rotate_unit_q([qi[0],-qi[1],-qi[2],-qi[3]],omega)
    principal_L=[float(inertia[j])*principal_omega[j] for j in range(3)]
    body_L=rotate_unit_q(qi,principal_L);world_L=rotate_unit_q(qb,body_L)
    return {'endpoint_valid':True,'body_quaternion_norm_F64':norm,'inertial_quaternion_norm_F64':inertial_norm,
      'world_L_F64':np.asarray(world_L,np.float64),'body_L_F64':np.asarray(body_L,np.float64),
      'body_unit_quaternion_F64':np.asarray(qb,np.float64),'inertial_unit_quaternion_F64':np.asarray(qi,np.float64)}


def rotation_observer_from_q32(q):
    norm,unit=normalize_q32(q)
    if unit is None:return norm,np.full((3,3),np.nan)
    columns=[rotate_unit_q(unit,v) for v in ([1.,0.,0.],[0.,1.,0.],[0.,0.,1.])]
    return norm,np.asarray(columns,np.float64).T


def signed_contact_torque(force6,frame,pos,com,sign):
    require(type(sign) in (int,float) and sign in (-1,1),'One object-boundary sign')
    f=finite(force6,(6,),'saved decoded force6');basis=finite(frame,(3,3),'saved frame');point=vector3(pos);origin=vector3(com)
    force=[sign*math.fsum(float(basis[k,axis])*float(f[k]) for k in range(3)) for axis in range(3)]
    direct=[sign*math.fsum(float(basis[k,axis])*float(f[k+3]) for k in range(3)) for axis in range(3)]
    lever=cross([math.fsum((point[i],-origin[i])) for i in range(3)],force)
    total=[math.fsum((lever[i],direct[i])) for i in range(3)]
    return {'force_world_N_F64':np.asarray(force),'lever_torque_COM_Nm_F64':np.asarray(lever),
      'direct_torque_Nm_F64':np.asarray(direct),'total_torque_COM_Nm_F64':np.asarray(total)}


def sum_vectors(vectors):
    v=[vector3(row) for row in vectors]
    return np.asarray([math.fsum(row[j] for row in v) for j in range(3)],np.float64)


def angular_balance(pre_L,post_L,dt,tau):
    pre,post=vector3(pre_L),vector3(post_L);force=vector3(tau);h=float(np.float32(dt));require(math.isfinite(h) and h>0,'Positive effective WP FP32 dt')
    delta=np.asarray([math.fsum((post[j],-pre[j])) for j in range(3)],np.float64)
    impulse=np.asarray([h*x for x in force],np.float64)
    residual=np.asarray([math.fsum((float(delta[j]),-float(impulse[j]))) for j in range(3)],np.float64)
    return {'delta_L_Nms_F64':delta,'contact_angular_impulse_Nms_F64':impulse,'unassigned_angular_impulse_Nms_F64':residual,
      'equivalent_torque_residual_Nm_F64':residual/h,'observer':'Normalized-F64 quaternion observer, not native GPU bit replay; no gyroscopic external-torque addition.'}
