"""Production wrappers: unsupported rows return through an untouched old call."""
import warp as wp
from mujoco_warp._src.solver import _compute_efc_eval_pt_pyramidal as original_single
from mujoco_warp._src.solver import _compute_efc_eval_pt_3alphas_pyramidal as original_triple
from .pen_stable_friction_cost_v3 import with_stable_cost


@wp.func
def same_regime(j:float,v:float,d:float,f:float,a:float)->bool:
    if d<=0.0 or f<0.0: return False
    j64=wp.float64(j); x=j64+wp.float64(a)*wp.float64(v); r=wp.float64(f)/wp.float64(d)
    return (-r<j64 and j64<r and -r<x and x<r) or (j64<=-r and x<=-r) or (j64>=r and x>=r)


@wp.func
def guarded_single(i:int,a:float,ne:int,nf:int,d:float,f:wp.array[float],j:float,v:float)->wp.vec3:
    if i>=ne and i<ne+nf and same_regime(j,v,d,f[i],a):
        return with_stable_cost(original_single(i,a,ne,nf,d,f,j,v),j,v,d,f[i],a)
    return original_single(i,a,ne,nf,d,f,j,v)


@wp.func
def guarded_triple(i:int,a:float,b:float,c:float,ne:int,nf:int,d:float,f:wp.array[float],j:float,v:float)->tuple[wp.vec3,wp.vec3,wp.vec3]:
    if i>=ne and i<ne+nf:
        if same_regime(j,v,d,f[i],a) or same_regime(j,v,d,f[i],b) or same_regime(j,v,d,f[i],c):
            x,y,z=original_triple(i,a,b,c,ne,nf,d,f,j,v)
            return with_stable_cost(x,j,v,d,f[i],a),with_stable_cost(y,j,v,d,f[i],b),with_stable_cost(z,j,v,d,f[i],c)
    return original_triple(i,a,b,c,ne,nf,d,f,j,v)
