"""Pure FP32 reference-damping compensation and C2 target export.

The Torch path is algebraic/differentiable only, with no live state or WM.
No function here runs physics, an optimizer, inverse dynamics or force solve.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

GROUPING = {
    'schema':'prefix_reference_PD_FP32:1',
    'static_prior':'kp*u_PDU + (bias0 + bias1*q0); qd=0',
    'proposal':'(((tau_ff_static0-bias0)-bias1*q_goal)-bias2*q_curve_qd)/kp',
    'affine_prediction':'kp*u + ((bias0+bias1*q)+bias2*qd)',
    'difference_prediction':'(kp*(u-q)+bias0)+bias2*qd; only when bias1==-kp',
    'C2':'minimum(maximum(u_pre,outer_lo32+float32(.020)),outer_hi32-float32(.020))',
    'scope':'Pure NumPy/Torch FP32 grouping parity; no Warp FMA, actuator measurement, dynamics or WM-gradient qualification',
}


def f32(a,shape=None):
    a=np.asarray(a,dtype=np.float32)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():raise ValueError('Nonfinite/wrongshape FP32 control input')
    return a


@dataclass(frozen=True)
class PDParameters:
    kp:np.ndarray
    bias:np.ndarray
    target_lower:np.ndarray
    target_upper:np.ndarray
    actuator_force_bounds:np.ndarray
    joint_force_bounds:np.ndarray

    def __post_init__(self):
        for key in ('kp','bias','target_lower','target_upper','actuator_force_bounds','joint_force_bounds'):
            a=np.ascontiguousarray(getattr(self,key),dtype=np.float32)
            object.__setattr__(self,key,np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape))
        self.validate()

    def validate(self):
        kp=f32(self.kp,(20,));bias=f32(self.bias,(20,3));lo=f32(self.target_lower,(20,));hi=f32(self.target_upper,(20,))
        act=f32(self.actuator_force_bounds,(20,2));joint=f32(self.joint_force_bounds,(20,2))
        if np.any(kp<=0) or not np.array_equal(bias[:,1],-kp) or np.any(lo+np.float32(.020)>hi-np.float32(.020)) or np.any(act[:,0]>=act[:,1]) or np.any(joint[:,0]>=joint[:,1]):raise ValueError('Unsupported PD or empty limits')
        return self


def load_actual_pd(model_path,context_path,*,model_sha256,context_sha256):
    import mujoco
    for path,expected in [(model_path,model_sha256),(context_path,context_sha256)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('PD input SHA mismatch')
    m=mujoco.MjModel.from_binary_path(str(model_path));ids=np.array([m.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
    aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j).item() for j in ids])
    checks={'gear1_joint':np.array_equal(m.actuator_gear[aids],np.tile([1.,0,0,0,0,0],(20,1))) and np.all(m.actuator_trntype[aids]==mujoco.mjtTrn.mjTRN_JOINT),
        'fixed_gain':np.all(m.actuator_gaintype[aids]==mujoco.mjtGain.mjGAIN_FIXED),'affine_bias':np.all(m.actuator_biastype[aids]==mujoco.mjtBias.mjBIAS_AFFINE),'dynamics_none':np.all(m.actuator_dyntype[aids]==mujoco.mjtDyn.mjDYN_NONE),'hinges':np.all(m.jnt_type[ids]==mujoco.mjtJoint.mjJNT_HINGE),'qpos0_zero':not np.any(m.qpos0[m.jnt_qposadr[ids]]),'actuator_and_joint_force_caps':np.all(m.actuator_forcelimited[aids]) and np.all(m.jnt_actfrclimited[ids]),'ctrl_caps':np.all(m.actuator_ctrllimited[aids]),'no_plugins':np.all(m.actuator_plugin[aids]<0)}
    if not all(checks.values()):raise ValueError('Unsupported actual actuation contract: '+str(checks))
    with np.load(context_path,allow_pickle=False) as z:
        gain=z['actuator_gainprm'];bias=z['actuator_biasprm']
        if not np.array_equal(gain,np.broadcast_to(gain[:1],gain.shape)) or not np.array_equal(bias,np.broadcast_to(bias[:1],bias.shape)):raise ValueError('Context is not fixed shared parameters')
        if np.any(bias[0,aids,3:]):raise ValueError('Unexpected affine bias columns')
        kp=gain[0,aids,0].copy();b=bias[0,aids,:3].copy()
    lo=np.maximum(m.jnt_range[ids,0],m.actuator_ctrlrange[aids,0]);hi=np.minimum(m.jnt_range[ids,1],m.actuator_ctrlrange[aids,1])
    p=PDParameters(f32(kp),f32(b),f32(lo),f32(hi),f32(m.actuator_forcerange[aids]),f32(m.jnt_actfrcrange[ids])).validate()
    return p,{'model_checks':{k:bool(v) for k,v in checks.items()},'joint_ids':ids.tolist(),'actuator_ids':aids.tolist(),'outer_lower64':lo.tolist(),'outer_upper64':hi.tolist(),'gear_scalar_units':'Nm for unit-gear hinge actuators','physical_steps':0}


def static_prior_numpy(q0,u_PDU,p):
    p.validate();q=f32(q0,(20,));u=f32(u_PDU,(20,))
    return p.kp*u+(p.bias[:,0]+p.bias[:,1]*q)


def proposal_numpy(q_goal,qd_curve,tau_ff,p):
    p.validate();q,qd=f32(q_goal),f32(qd_curve);tau=f32(tau_ff,(20,))
    if q.shape!=qd.shape or q.shape[-1:]!=(20,):raise ValueError('Reference q/continuous qd shape mismatch')
    return (((tau-p.bias[:,0])-p.bias[:,1]*q)-p.bias[:,2]*qd)/p.kp


def c2_numpy(proposed,p):
    p.validate();u=f32(proposed)
    if u.shape[-1:]!=(20,):raise ValueError('Expected actual target20')
    return np.minimum(np.maximum(u,p.target_lower+np.float32(.020)),p.target_upper-np.float32(.020))


def affine_predictions_numpy(q,qd,u,p):
    q,qd,u=f32(q),f32(qd),f32(u)
    if q.shape!=qd.shape or q.shape!=u.shape or q.shape[-1:]!=(20,):raise ValueError('PD phase/shape mismatch')
    affine=p.kp*u+((p.bias[:,0]+p.bias[:,1]*q)+p.bias[:,2]*qd)
    difference=(p.kp*(u-q)+p.bias[:,0])+p.bias[:,2]*qd
    return affine,difference


def torch_parameters(p,device='cpu'):
    import torch
    p.validate()
    return {k:torch.tensor(getattr(p,k),dtype=torch.float32,device=device) for k in ('kp','bias','target_lower','target_upper')}


def static_prior_torch(q0,u_PDU,p):
    import torch
    with torch.autocast(device_type=q0.device.type,enabled=False):
        q,u=q0.to(torch.float32),u_PDU.to(torch.float32)
        return p['kp']*u+(p['bias'][:,0]+p['bias'][:,1]*q)


def proposal_torch(q_goal,qd_curve,tau_ff,p):
    import torch
    with torch.autocast(device_type=q_goal.device.type,enabled=False):
        q,qd,tau=q_goal.to(torch.float32),qd_curve.to(torch.float32),tau_ff.to(torch.float32)
        if q.shape!=qd.shape or q.shape[-1:]!=(20,) or tau.shape!=(20,) or not bool(torch.isfinite(q).all() and torch.isfinite(qd).all() and torch.isfinite(tau).all()):raise ValueError('Torch nonfinite/reference shape mismatch')
        return (((tau-p['bias'][:,0])-p['bias'][:,1]*q)-p['bias'][:,2]*qd)/p['kp']


def c2_torch(proposed,p):
    import torch
    with torch.autocast(device_type=proposed.device.type,enabled=False):
        u=proposed.to(torch.float32)
        if u.shape[-1:]!=(20,) or not bool(torch.isfinite(u).all()):raise ValueError('Torch nonfinite/target shape')
        return torch.minimum(torch.maximum(u,p['target_lower']+.020),p['target_upper']-.020)


def export_control(curves,p,q0,u_PDU):
    tau=static_prior_numpy(q0,u_PDU,p);knots=curves.sample(curves.time_s);ticks=curves.sample(np.arange(61)*.005)
    pre=proposal_numpy(knots['q_goal_rad'],knots['q_curve_qd_rad_s'],tau,p);executed=c2_numpy(pre,p);schedule=executed[:15].copy();hold=np.repeat(schedule[:1],15,axis=0)
    # There are15 commands and14 actual post-initialization target changes.
    rates=np.diff(schedule.astype(np.float64),axis=0)/.02
    solver_index=np.arange(60)//4;u60=schedule[solver_index];a60,d60=affine_predictions_numpy(ticks['q_goal_rad'][:60],ticks['q_curve_qd_rad_s'][:60],u60,p)
    ha60,hd60=affine_predictions_numpy(ticks['q_goal_rad'][:60],ticks['q_curve_qd_rad_s'][:60],hold[solver_index],p)
    ak,dk=affine_predictions_numpy(knots['q_goal_rad'],knots['q_curve_qd_rad_s'],executed,p)
    def within(x,b):return bool(np.all(x>=b[:,0]) and np.all(x<=b[:,1]))
    force_ok=all(within(x,b) for x in (a60,d60) for b in (p.actuator_force_bounds,p.joint_force_bounds))
    hold_force_ok=all(within(x,b) for x in (ha60,hd60) for b in (p.actuator_force_bounds,p.joint_force_bounds))
    flags={'actual15_C2_bounds':bool(np.all(schedule>=p.target_lower+np.float32(.020)) and np.all(schedule<=p.target_upper-np.float32(.020))),
        'actual14_updates_rate_le15':bool(np.max(np.abs(rates))<=15.),'actual60_ref_PD_predictions_within_both_caps':force_ok,
        'hold60_ref_PD_predictions_within_both_caps':hold_force_ok,
        'hold_and_track_same_u0':bool(np.array_equal(hold[0].view(np.uint32),schedule[0].view(np.uint32))),
        'tracking_FP32_intervention_nonconstant':bool(np.any(schedule[1:]!=schedule[0]))}
    return {'tau_ff_static0_FP32_Nm':tau,'u_pre_16_FP32_rad':pre,'u_exec_16_diagnostic_FP32_rad':executed,'tracking_u_exec_15_FP32_rad':schedule,'hold_u_exec_15_FP32_rad':hold,'actual_target_rate_14_rad_s':rates,'solver_prestate_control_interval':solver_index,'reference_PD_affine_60_FP32_Nm':a60,'reference_PD_difference_60_FP32_Nm':d60,'reference_PD_affine_16_diagnostic_FP32_Nm':ak,'reference_PD_difference_16_diagnostic_FP32_Nm':dk,
        'hold_reference_PD_affine_60_FP32_Nm':ha60,'hold_reference_PD_difference_60_FP32_Nm':hd60,
        'gates':flags,'max_actual_target_rate_rad_s':float(np.abs(rates).max()),'clamped_components_actual15':int(np.count_nonzero(pre[:15]!=schedule)),
        'scope':'15 targets/14 updates; terminal16th command is diagnostic and never executed. Force gates predict only reference q/qd at60solver times; actual simulation q/qd/forces unqualified.'}
