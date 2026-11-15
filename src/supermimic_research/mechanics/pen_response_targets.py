"""Same-source held-PD response chart, reusing validated state/frame semantics."""
import numpy as np

from .controlled_pen_records import native_state55
from .transition_records import transition_xy,state_rotation


def response_xy(packet,world_wrench,model,source,qids,dofs,aids,body,qa,da,dt):
    start=native_state55(source['qpos'][0],source['qvel'][0],qids,dofs,qa,da)
    end=native_state55(packet['post_qpos'][-1],packet['post_qvel'][-1],qids,dofs,qa,da)
    R=state_rotation(start);mean=world_wrench.mean(0);Fn=np.zeros(6);n=len(packet['post_qpos'])
    for row in packet['raw_contact_tick_plus23']:
        g,h=map(int,row[1:3]);ob=model.geom_bodyid[[g,h]]==body
        if ob.sum()!=1:continue
        name=model.geom(h if ob[0] else g).name or ''
        if name.startswith('robot/'):
            digit=next((j-1 for j in range(1,6) if name.startswith(f'robot/right_finger{j}_')),5)
            Fn[digit]+=max(0.,row[18])/n
    return transition_xy(dict(source55=start,next55=end,warmstart26_before=source['qacc_warmstart'][0],
        warmstart26_after=packet['post_qacc'][-1],decision=dict(actual_control20=packet['post_ctrl'][0,aids]),
        mean_native_wrench_source_BODY_COM=np.r_[R.T@mean[:3],R.T@mean[3:]],mean_native_group_Fn_N=Fn),dt)
