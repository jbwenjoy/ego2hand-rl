"""Fixed-input diagnostics, including a guarded wrapper experiment,not a backend."""
import warp as wp
from mujoco_warp._src.solver import _compute_efc_eval_pt_pyramidal as original_single
from mujoco_warp._src.solver import _compute_efc_eval_pt_3alphas_pyramidal as original_triple
from .pen_stable_friction_cost_v3 import with_stable_cost

vec12=wp.types.vector(length=12,dtype=wp.float32)


@wp.func
def same_regime(j:float,v:float,d:float,f:float,a:float)->bool:
    if d<=0.0 or f<0.0: return False
    j64=wp.float64(j); x=j64+wp.float64(a)*wp.float64(v); r=wp.float64(f)/wp.float64(d)
    return (-r<j64 and j64<r and -r<x and x<r) or (j64<=-r and x<=-r) or (j64>=r and x>=r)


@wp.func
def identity(point:wp.vec3,j:float,v:float,d:float,f:float,a:float)->wp.vec3:
    return point


@wp.func
def identity_single(i:int,a:float,ne:int,nf:int,d:float,f:wp.array[float],j:float,v:float)->wp.vec3:
    return identity(original_single(i,a,ne,nf,d,f,j,v),j,v,d,f[i],a)


@wp.func
def identity_triple(i:int,a:float,b:float,c:float,ne:int,nf:int,d:float,f:wp.array[float],j:float,v:float)->tuple[wp.vec3,wp.vec3,wp.vec3]:
    x,y,z=original_triple(i,a,b,c,ne,nf,d,f,j,v)
    return identity(x,j,v,d,f[i],a),identity(y,j,v,d,f[i],b),identity(z,j,v,d,f[i],c)


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


@wp.func_native('''
    auto mul_rn=[](float u,float v)->float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(u,v);
    #else
        volatile float result=u*v; return result;
    #endif
    };
    auto sub_rn=[](float u,float v)->float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(u,v);
    #else
        volatile float result=u-v; return result;
    #endif
    };
    auto fma_rn=[](float u,float v,float w)->float {
    #if defined(__CUDA_ARCH__)
        return __fmaf_rn(u,v,w);
    #else
        return (float)((double)u*(double)v+(double)w);
    #endif
    };
    float x=fma_rn(a,v,j);
    float s=mul_rn(a,v); float x_unfused=sub_rn(j,-s);
    float r=f/d;
    bool q0=(-r<j && j<r),q1=(-r<x && x<r);
    float p0=q0?mul_rn(mul_rn(0.5f,d),j):f;
    float t0=q0?j:(j<=-r?sub_rn(mul_rn(-0.5f,r),j):fma_rn(-0.5f,r,j));
    float p1=q1?mul_rn(mul_rn(0.5f,d),x):f;
    float t1=q1?x:(x<=-r?sub_rn(mul_rn(-0.5f,r),x):fma_rn(-0.5f,r,x));
    float c0=mul_rn(p0,t0),c1=mul_rn(p1,t1);
    wp::vec_t<12,float> result;
    result[0]=x; result[1]=x_unfused; result[2]=r;
    result[3]=p0; result[4]=t0; result[5]=c0;
    result[6]=p1; result[7]=t1; result[8]=c1;
    result[9]=sub_rn(c1,c0);
    result[10]=fma_rn(-p0,t0,c1);
    result[11]=fma_rn(p1,t1,-c0);
    return result;
''')
def explicit_rounding(j:float,v:float,d:float,f:float,a:float)->vec12:
    ...
