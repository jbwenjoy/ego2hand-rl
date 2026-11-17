"""Candidate cost-only replacement; derivative/Hessian and fallback stay original."""
import warp as wp
from mujoco_warp._src.solver import math


@wp.func
def with_stable_cost(point:wp.vec3,jaref:float,jv:float,d:float,loss:float,alpha:float)->wp.vec3:
    advance=alpha*jv; x=jaref+advance; radius=math.safe_div(loss,d); delta=point[0]
    if -radius<jaref and jaref<radius and -radius<x and x<radius:
        delta=advance*(d*jaref+0.5*d*advance)
    elif jaref<=-radius and x<=-radius:
        delta=-loss*advance
    elif jaref>=radius and x>=radius:
        delta=loss*advance
    return wp.vec3(delta,point[1],point[2])
