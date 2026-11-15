"""Read-only fixed-wrist reference and full-pose task diagnostics."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .pen_contact_groups import normal_group


def pose_components(actual_wxyz, reference_wxyz):
    actual=Rotation.from_quat(np.asarray(actual_wxyz)[...,[1,2,3,0]])
    reference=Rotation.from_quat(np.asarray(reference_wxyz)[...,[1,2,3,0]])
    relative=reference.inv()*actual
    q=relative.as_quat()
    axis=np.clip(relative.as_matrix()[...,0,0],-1.,1.)
    norm=np.hypot(q[...,3],q[...,0]);defined=norm>1e-12
    twist=np.where(defined,2*np.arctan2(np.abs(q[...,0]),np.abs(q[...,3])),np.nan)
    return dict(full_angle_rad=relative.magnitude(),shaft_angle_rad=np.arccos(axis),
                reference_axis_twist_rad=twist,twist_defined=defined)


def tracking_series(trace,dt):
    t=np.arange(1,len(trace['post_qpos'])+1)*dt
    target=np.stack([np.interp(t,trace['time_s'],trace['COM_world_m'][:,j]) for j in range(3)],axis=1)
    R=Slerp(trace['time_s'],Rotation.from_quat(trace['quaternion_wxyz'][:,[1,2,3,0]]))(np.minimum(t,trace['time_s'][-1]))
    target_q=R.as_quat()[:,[3,0,1,2]]
    return dict(nominal_time_s=t,actual_GPU_time_s=trace['post_GPU_time'],
                COM_error_m=np.linalg.norm(trace['post_COM']-target,axis=1),
                **pose_components(trace['post_quaternion_wxyz'],target_q))


def first_exceed(values,limit,nominal,actual):
    v=np.asarray(values);bad=np.flatnonzero(~np.isfinite(v)|(v>limit))
    if not len(bad):return None
    i=int(bad[0])
    return dict(index=i,nominal_time_s=float(nominal[i]),actual_GPU_time_s=float(actual[i]),
                value=float(v[i]) if np.isfinite(v[i]) else None,limit=float(limit))


def rebase_fixed_wrist(wrist,object_world,mount):
    wrist,object_world,mount=[np.asarray(x,float) for x in [wrist,object_world,mount]]
    W=Rotation.from_quat(wrist[:,[4,5,6,3]]);O=Rotation.from_quat(object_world[:,[4,5,6,3]])
    M=Rotation.from_quat(mount[[4,5,6,3]])
    local=W.inv().apply(object_world[:,:3]-wrist[:,:3])
    return np.c_[mount[:3]+M.apply(local),(M*W.inv()*O).as_quat()[:,[3,0,1,2]]]


def normal_loads(trace,model,body):
    N=len(trace['post_qpos']);groups=np.zeros((N,6));nonhand=np.zeros(N)
    for row in trace['raw_contact_tick_plus23']:
        tick,g,h=map(int,row[:3])
        if not 0<=tick<N or not 0<=g<model.ngeom or not 0<=h<model.ngeom:raise ValueError('Contact ownership/index contract')
        a,b=model.geom_bodyid[g]==body,model.geom_bodyid[h]==body
        if a==b:continue
        digit=normal_group(model.geom(h if a else g).name or '')
        normal=max(0.,float(row[18]))
        if digit<0:nonhand[tick]+=normal
        else:groups[tick,digit]+=normal
    return groups,nonhand


def candidate_pose_coverage(packet,reference,k,n,COM_gate,angle_gate,qa):
    q=packet['Q'][:,n-1::n,qa:qa+7];H=q.shape[1]
    if q.shape[0]!=57 or k+H>=len(reference['COM_world_m']):raise ValueError('57 actual-control forecasts within original clock')
    distance=np.linalg.norm(q[:,:,:3]-reference['COM_world_m'][None,k+1:k+H+1],axis=-1)
    good=np.isfinite(q[:,:,3:]).all(-1)&(np.linalg.norm(q[:,:,3:],axis=-1)>0)
    angle=np.full((57,H),np.inf)
    target=np.broadcast_to(reference['quaternion_wxyz'][None,k+1:k+H+1],(57,H,4))
    angle[good]=pose_components(q[:,:,3:][good],target[good])['full_angle_rad']
    safe=packet['score_feasible'].astype(bool)
    first=safe&(distance[:,0]<=COM_gate)&(angle[:,0]<=angle_gate)
    all_steps=safe&np.all((distance<=COM_gate)&(angle<=angle_gate),axis=1)
    return first,all_steps,distance,angle
