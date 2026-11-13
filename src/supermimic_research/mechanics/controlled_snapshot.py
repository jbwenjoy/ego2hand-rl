"""Native diagnostic copy of a pinned controlled GPU initial state/parameters."""
import numpy as np
from ..artifacts import verify_file_pin


def native_controlled_snapshot(case):
    import mujoco
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['host_topology'])))
    trace=dict(np.load(verify_file_pin(case['trace'])))
    physical=dict(np.load(verify_file_pin(case['GPU_physical_context'])))
    for key,value in physical.items():
        target=getattr(m,key)
        if target.shape!=value[0].shape:raise ValueError('Explicit physical table shape '+key)
        target[:]=value[0]
    d=mujoco.MjData(m);mujoco.mj_setConst(m,d)
    mujoco.mj_setState(m,d,trace['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied) or m.neq or np.any(m.body_gravcomp):
        raise ValueError('No applied aid, equality or gravity compensation')
    names=[f'robot/right_finger{i}_joint{j}' for i in range(1,6) for j in range(1,5)]
    joints=np.array([m.joint(name).id for name in names]);qids=m.jnt_qposadr[joints];dofs=m.jnt_dofadr[joints]
    actuators=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in joints])
    return m,d,trace,dict(joints=joints,qpos=qids,dofs=dofs,actuators=actuators)
