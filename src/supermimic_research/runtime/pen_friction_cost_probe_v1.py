"""Fixed-input arithmetic probe only; never bound into the simulator."""
import warp as wp
from mujoco_warp._src.solver import _eval_frictionloss_pt,_eval_frictionloss_cost,_shift_cost,math


@wp.kernel
def probe_friction_cost(
    jaref:wp.array[float],jv:wp.array[float],d:wp.array[float],loss:wp.array[float],alpha:wp.array[float],
    original:wp.array2d[float],same_regime:wp.array2d[float],handled:wp.array[int],
):
    i=wp.tid(); advance=alpha[i]*jv[i]; x=jaref[i]+advance; radius=math.safe_div(loss[i],d[i])
    old=_shift_cost(_eval_frictionloss_pt(x,loss[i],radius,jv[i],d[i]),_eval_frictionloss_cost(jaref[i],loss[i],radius,d[i]))
    delta=old[0]; supported=int(0)
    if -radius<jaref[i] and jaref[i]<radius and -radius<x and x<radius:
        delta=advance*(d[i]*jaref[i]+0.5*d[i]*advance); supported=1
    elif jaref[i]<=-radius and x<=-radius:
        delta=-loss[i]*advance; supported=1
    elif jaref[i]>=radius and x>=radius:
        delta=loss[i]*advance; supported=1
    original[i,0]=old[0]; original[i,1]=old[1]; original[i,2]=old[2]
    same_regime[i,0]=delta; same_regime[i,1]=old[1]; same_regime[i,2]=old[2]
    handled[i]=supported
