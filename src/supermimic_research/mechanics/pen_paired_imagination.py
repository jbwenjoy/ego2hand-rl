"""Three-step paired imagined Actor loss. The two arms share plant, reward and start distribution."""
import torch

from .pen_imagined_load import pen_load_step,pen_rotation
from .pen_imagined_observation import pen_imagined_observation
from .pen_capacity_forecast import capacity_mismatch
from ..runtime.pen_native_policy import tracking_reward
from dex_supermimic.rl.dex_supermimic.rotations import mat_to_quat


def imagined_objectives(actor,load_head,capacity_head,rows,reward_cfg,mass,inertia,gravity):
    s=rows['state55'];last=rows['previous_raw'];ema=rows['previous_ema']
    tracking=[];mismatch=[];smooth=[];loads=[]
    for j in range(3):
        q=rows['ref_joints'][:,j];pose=rows['ref_pose_wxyz'][:,j]
        obs=pen_imagined_observation(s,last,q,pose,rows['ref_links'][:,j],rows['mount_wxyz'],
                                     rows['encoder_bias'],rows['default_joint_pos'],rows['default_joint_vel'])
        raw=actor.distribution(obs,validate_args=False).mean
        applied=raw.clamp(-1,1)
        ema=.5*ema+.25*applied
        pd=(q+ema-rows['encoder_bias']).clamp(rows['lower'],rows['upper'])
        cap=capacity_head(s,pd)
        s,_,wrench=pen_load_step(s,pd,None,load_head,mass,inertia,gravity)
        track,_=tracking_reward(s[:,40:43],mat_to_quat(pen_rotation(s)),s[:,:20],pose,q,applied,reward_cfg)
        tracking.append(track)
        smooth.append((applied-last).square().mean(-1))
        loads.append((wrench[:,6:].relu()/(mass*gravity.norm())).square().mean(-1))
        mismatch.append(capacity_mismatch(cap,rows['capacity_start'][:,j])*rows['capacity_eligible'][:,j])
        last=applied
    tr,sm,ld,mis=[torch.stack(v,dim=1).mean() for v in [tracking,smooth,loads,mismatch]]
    return dict(track=tr,smooth=sm,load=ld,capacity_error=mis,base=tr-.001*sm-.0001*ld)
