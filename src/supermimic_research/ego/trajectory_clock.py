"""Integer-clock rigid BODY-pose sampling with explicit COM kinematics."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def resample_body_poses(timestamps_ns,poses,COM_BODY_m,*,period_ns=20_000_000):
    t=np.asarray(timestamps_ns);T=np.asarray(poses,float);COM=np.asarray(COM_BODY_m,float)
    if t.dtype!=np.int64 or t.ndim!=1 or len(t)<2 or T.shape!=(len(t),4,4) or COM.shape!=(3,) or not np.isfinite(np.r_[T.ravel(),COM]).all():raise ValueError('Explicit int64ns clock and finite rigid BODY poses/COM')
    span=int(t[-1])-int(t[0])
    if not 0<span<=np.iinfo(np.int64).max or any(int(b)<=int(a) for a,b in zip(t[:-1],t[1:])) or not isinstance(period_ns,int) or period_ns<=0:raise ValueError('Strict increasing clock and integer positive period; no time repair')
    R=T[:,:3,:3];orth=float(np.max(abs(np.swapaxes(R,1,2)@R-np.eye(3))))
    if orth>1e-10 or np.max(abs(np.linalg.det(R)-1))>1e-10 or not np.array_equal(T[:,3,:],np.tile([0.,0,0,1],(len(t),1))):raise ValueError('Proper source rigid poses; no rotation repair')
    offsets=np.arange(span//period_ns+1,dtype=np.int64)*period_ns;grid=t[0]+offsets
    elapsed=(t-t[0]).astype(np.float64)*1e-9;seconds=offsets.astype(np.float64)*1e-9
    left=np.minimum(np.searchsorted(t,grid,side='right')-1,len(t)-2)
    fraction=(grid-t[left]).astype(np.float64)/(t[left+1]-t[left])
    source_rotation=Rotation.from_matrix(R);sample_rotation=Slerp(elapsed,source_rotation)(seconds);sample_R=sample_rotation.as_matrix()
    position=T[left,:3,3]+fraction[:,None]*(T[left+1,:3,3]-T[left,:3,3])
    interval_s=(t[left+1]-t[left]).astype(np.float64)*1e-9;linear=(T[left+1,:3,3]-T[left,:3,3])/interval_s[:,None]
    local_rotvec=Rotation.from_matrix(np.swapaxes(R[left],1,2)@R[left+1]).as_rotvec()
    omega=np.einsum('nij,nj->ni',R[left],local_rotvec)/interval_s[:,None]
    arm=np.einsum('nij,j->ni',sample_R,COM);sample_T=np.tile(np.eye(4),(len(grid),1,1));sample_T[:,:3,:3]=sample_R;sample_T[:,:3,3]=position
    quaternion=sample_rotation.as_quat()[:,[3,0,1,2]]
    for i in range(1,len(quaternion)):
        if quaternion[i]@quaternion[i-1]<0:quaternion[i]*=-1
    return dict(timestamps_ns=grid,elapsed_s=seconds,T_world_from_BODY=sample_T,quaternion_wxyz=quaternion,
        COM_world_m=position+arm,root_velocity_world_m_s=linear,COM_velocity_world_m_s=linear+np.cross(omega,arm),angular_velocity_world_rad_s=omega,
        source_left_index=left,source_interval_alpha=fraction,
        metadata=dict(source_span_ns=span,period_ns=period_ns,excluded_tail_ns=int(t[-1])-int(grid[-1]),
            extrapolated=False,time_stretched=False,source_rotation_orthogonality_error=orth,
            rotation_conversion_roundtrip_error=float(np.max(abs(source_rotation.as_matrix()-R))),
            interpolation='Piecewise linear BODY-origin translation and shortest-arc SLERP; exact integer clock before float offsets',
            velocity='Analytic within source interval; right interval at internal knots, left at final source endpoint; COM lever arm included'))
