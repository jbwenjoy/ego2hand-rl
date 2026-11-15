"""Indexed/world-frame native replay comparison with caller-declared gates."""
import numpy as np
from scipy.spatial.transform import Rotation


def compare_response(predicted,actual,predicted_wrench,actual_wrench,qids,dofs,qa,da,gates):
    p,a=predicted['post_qpos'],actual['post_qpos'];pv,av=predicted['post_qvel'],actual['post_qvel']
    Rp=Rotation.from_quat(p[:,qa+3:qa+7][:,[1,2,3,0]]).as_matrix()
    Ra=Rotation.from_quat(a[:,qa+3:qa+7][:,[1,2,3,0]]).as_matrix()
    omega_p=np.einsum('nij,nj->ni',Rp,pv[:,da+3:da+6]);omega_a=np.einsum('nij,nj->ni',Ra,av[:,da+3:da+6])
    metrics=dict(hand_position_rad=float(abs(p[:,qids]-a[:,qids]).max()),
        object_position_m=float(np.linalg.norm(p[:,qa:qa+3]-a[:,qa:qa+3],axis=1).max()),
        object_rotation_rad=float(Rotation.from_matrix(Rp@Ra.transpose(0,2,1)).magnitude().max()),
        hand_velocity_rad_s=float(abs(pv[:,dofs]-av[:,dofs]).max()),
        object_velocity_m_s=float(np.linalg.norm(pv[:,da:da+3]-av[:,da:da+3],axis=1).max()),
        object_omega_rad_s=float(np.linalg.norm(omega_p-omega_a,axis=1).max()),
        control_rad=float(abs(predicted['post_ctrl']-actual['post_ctrl']).max()),
        EFC_force_N=float(np.linalg.norm(predicted_wrench[:,:3]-actual_wrench[:,:3],axis=1).max()),
        EFC_torque_Nm=float(np.linalg.norm(predicted_wrench[:,3:]-actual_wrench[:,3:],axis=1).max()))
    checks={k:bool(value<=gates[k]) for k,value in metrics.items()}
    checks['time_exact']=np.array_equal(predicted['post_GPU_time'],actual['post_GPU_time'])
    return metrics,checks
