"""Actual independent native packet -> maintained102/90 response chart."""
import numpy as np
from .controlled_pen_records import native_state55
from .transition_records import transition_xy,state_rotation
from .response_decode import momentum_defect


def packet_xy(source,packet,world_wrench,Fn,qids,dofs,aids,qa,da,dt,physics):
    start=native_state55(source['qpos'][0],source['qvel'][0],qids,dofs,qa,da)
    end=native_state55(packet['post_qpos'][-1],packet['post_qvel'][-1],qids,dofs,qa,da)
    R=state_rotation(start);mean=world_wrench.mean(0)
    x,y=transition_xy(dict(source55=start,next55=end,warmstart26_before=source['qacc_warmstart'][0],
        warmstart26_after=packet['post_qacc_warmstart'][-1],decision=dict(actual_control20=packet['post_ctrl'][0,aids]),
        mean_native_wrench_source_BODY_COM=np.r_[R.T@mean[:3],R.T@mean[3:]],mean_native_group_Fn_N=Fn.mean(0)),dt)
    head=y.copy();head[46:52]=momentum_defect(start,end,y,physics['mass'],np.asarray(physics['inertia_BODY']),np.asarray(physics['gravity']))
    return x,y,head
