"""Full151 cubic geometry and tracking-only linear reference-PD contracts.

This module constructs sparse LPs and evaluates saved controls; it never calls a
solver, FK, collision query, Env, physics, Torch or a world model. Geometric q,
analytic qd and planned targets are not measured physical states or forces.
"""
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline

DEGREE=3
NCONTROL=153
NFREE=151
NJOINT=20
NLP=453
H=.02
BREAKS=np.arange(151,dtype=np.float64)*H
CONTROL_MARGIN_RAD=2e-6
FORCE_MARGIN_NM=2e-6
RATE_MARGIN_RAD_S=1e-4
EXECUTION_RATE_RAD_S=15.
PLANNING_TOL=1e-12
QHULL_TOL_RAD=1e-12
FROZEN_CONTROL_SHA256='76026dbdcdfdecf4680820b24da088e1d123e86ceb8b42a2bbd60a5412103b6d'


def _a(x,shape=None):
    a=np.asarray(x,dtype=np.float64)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError('Nonfinite or wrong-shape continuous-control input')
    return a


def _ro(x):
    a=np.ascontiguousarray(x)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


class ClampedCubicBasis:
    def __init__(self):
        self.degree=3
        self.ncontrol=153
        self.breaks=_ro(BREAKS)
        self.knots=_ro(np.r_[np.repeat(0.,4),BREAKS[1:-1],np.repeat(3.,4)])
        self._spline=BSpline(self.knots,np.eye(153),3,extrapolate=False)

    def evaluate(self,t,derivative=0):
        t=np.atleast_1d(_a(t))
        if t.ndim!=1 or not t.size or derivative not in (0,1,2) or np.any(t<0) or np.any(t>3.):
            raise ValueError('Unsupported derivative or curve extrapolation')
        return np.asarray(self._spline(t,nu=derivative),dtype=np.float64)


@lru_cache(maxsize=1)
def basis():
    return ClampedCubicBasis()


def eval_controls(P,t,order=0):
    """Source64-centered grouping; P0 is never clipped or replaced."""
    P=_a(P,(153,20))
    delta=P[1:]-P[0]
    v=basis().evaluate(t,order)[:,1:]@delta
    return v+P[0] if order==0 else v


@lru_cache(maxsize=1)
def bezier_matrices():
    """Cubic q and physical-seconds quadratic qd, elevated to cubic."""
    b=basis();left,right=BREAKS[:-1],BREAKS[1:]
    h=np.diff(BREAKS)[:,None]
    l,r=b.evaluate(left),b.evaluate(right)
    dl,dr=b.evaluate(left,1),b.evaluate(right,1)
    q=np.stack((l,l+h*dl/3,r-h*dr/3,r),axis=1)
    d2=3/h[:,None,:]*np.diff(q,axis=1)
    d3=np.stack((d2[:,0],d2[:,0]/3+2*d2[:,1]/3,
                 2*d2[:,1]/3+d2[:,2]/3,d2[:,2]),axis=1)
    return {'q_cubic':_ro(q),'qd_quadratic':_ro(d2),
            'qd_elevated_cubic':_ro(d3),'interval_seconds':_ro(np.diff(BREAKS))}


def _parameters(params):
    """Old PDParameters fields, or a pure dict containing those same fields."""
    if hasattr(params,'validate'):
        params.validate()
    get=(lambda k:params[k]) if isinstance(params,dict) else (lambda k:getattr(params,k))
    kp=_a(get('kp'),(20,));bias=_a(get('bias'),(20,3))
    lo=_a(get('target_lower'),(20,));hi=_a(get('target_upper'),(20,))
    act=_a(get('actuator_force_bounds'),(20,2))
    joint=_a(get('joint_force_bounds'),(20,2))
    if np.any(kp<=0) or not np.array_equal(bias[:,1],-kp) or np.any(lo>=hi):
        raise ValueError('Requires positive actual gain, affine bias1=-kp and legal outer domain')
    if np.any(act[:,0]>=act[:,1]) or np.any(joint[:,0]>=joint[:,1]):
        raise ValueError('Invalid actuator or jointactforce caps')
    return kp,bias,lo,hi,act,joint


def _affine(mat,p0,p1):
    """Return matrix/base in absolute free variables P2..152.

    Evaluation is mathematically the same curve as eval_controls. Derivative
    bases have zero row sums; LP arithmetic is FP64, not bitwise evaluation.
    """
    free=mat[...,2:]
    return free,mat[...,:2]@np.array([p0,p1],dtype=np.float64)


def build_joint_lp(j,target151,P0,P1,u0_F32,tau_ff32,params,qlo,qhi,lambda_coeff=4.):
    """Return linprog kwargs/contract, without solving; 453 variables/joint.

    Variable order: absolute P2..152 (151), fit abs epigraphs (151), coefficient
    second-difference abs epigraphs (151). P1 must be the exact saved source P1;
    callers must not reconstruct or optimize it. Targets may be [151] or [151,20].
    """
    if isinstance(j,(bool,np.bool_)) or not isinstance(j,(int,np.integer)) or not 0<=int(j)<20:
        raise ValueError('Joint index must be integer0..19')
    j=int(j)
    target=_a(target151)
    if target.shape==(151,20):target=target[:,j]
    if target.shape!=(151,):raise ValueError('Fit target must contain all151 frames')
    P0,P1=_a(P0,(20,)),_a(P1,(20,))
    u0,tau=_a(u0_F32,(20,)),_a(tau_ff32,(20,))
    if not np.array_equal(u0,u0.astype(np.float32).astype(float)) or not np.array_equal(tau,tau.astype(np.float32).astype(float)):
        raise ValueError('u0 and tau must represent the exact saved FP32 values')
    qlo,qhi=_a(qlo,(20,)),_a(qhi,(20,))
    if np.any(qlo>=qhi) or np.any(P0<qlo-QHULL_TOL_RAD) or np.any(P0>qhi+QHULL_TOL_RAD) or np.any(P1<qlo-QHULL_TOL_RAD) or np.any(P1>qhi+QHULL_TOL_RAD):
        raise ValueError('Fixed P0/P1 outside hard domain; no clipping or new initialization')
    if isinstance(lambda_coeff,(bool,np.bool_)) or not np.isfinite(lambda_coeff) or float(lambda_coeff)!=4.:
        raise ValueError('Frozen coefficient-difference regularizer is4')
    kp,bias,outerlo,outerhi,act,joint=_parameters(params)
    innerlo=(outerlo.astype(np.float32)+np.float32(.020)).astype(float)
    innerhi=(outerhi.astype(np.float32)-np.float32(.020)).astype(float)
    ulo,uhi=innerlo[j]+CONTROL_MARGIN_RAD,innerhi[j]-CONTROL_MARGIN_RAD
    flo=max(act[j,0],joint[j,0])+FORCE_MARGIN_NM
    fhi=min(act[j,1],joint[j,1])-FORCE_MARGIN_NM
    if ulo>uhi or flo>fhi or not ulo<=u0[j]<=uhi:
        raise ValueError('Empty tightened limits or fixedu0 outside planning domain')
    b=basis();Q,qbase=_affine(b.evaluate(BREAKS),P0[j],P1[j])
    D,dbase=_affine(b.evaluate(BREAKS,1),P0[j],P1[j])
    U=-(bias[j,1]*Q[:150]+bias[j,2]*D[:150])/kp[j]
    ub=(tau[j]-bias[j,0]-bias[j,1]*qbase[:150]-bias[j,2]*dbase[:150])/kp[j]
    U[0]=0.;ub[0]=u0[j]  # first interval is the pinned executed target, not an equality repair
    bez=bezier_matrices()
    Bq,bq=_affine(bez['q_cubic'],P0[j],P1[j])
    Bd,bd=_affine(bez['qd_elevated_cubic'],P0[j],P1[j])
    F=kp[j]*U[:,None,:]+bias[j,1]*Bq+bias[j,2]*Bd
    fb=kp[j]*ub[:,None]+bias[j,0]+bias[j,1]*bq+bias[j,2]*bd
    dU=np.diff(U,axis=0);dub=np.diff(ub)
    second=np.diff(np.eye(153),n=2,axis=0)
    S,sbase=_affine(second,P0[j],P1[j])
    zero=sparse.csr_matrix((151,151));eye=sparse.eye(151,format='csr')
    pieces=[];rhs=[];groups=[];cursor=0
    def add(name,A,bound,units,shape):
        nonlocal cursor
        A=sparse.csr_matrix(A);bound=np.asarray(bound,dtype=float).reshape(-1)
        if A.shape!=(len(bound),453):raise ValueError('LP row shape mismatch')
        pieces.append(A);rhs.append(bound)
        groups.append({'name':name,'start':cursor,'stop':cursor+len(bound),'units':units,'tensor_shape':list(shape)})
        cursor+=len(bound)
    def geometry_rows(name,A,bound,units,shape):
        A=sparse.csr_matrix(np.asarray(A).reshape(-1,151))
        add(name,sparse.hstack((A,sparse.csr_matrix((A.shape[0],302)))),bound,units,shape)
    geometry_rows('target_upper',U,uhi-ub,'rad',(150,))
    geometry_rows('target_lower',-U,ub-ulo,'rad',(150,))
    delta_bound=H*(EXECUTION_RATE_RAD_S-RATE_MARGIN_RAD_S)
    geometry_rows('target_rate_upper',dU,delta_bound-dub,'rad/update',(149,))
    geometry_rows('target_rate_lower',-dU,delta_bound+dub,'rad/update',(149,))
    geometry_rows('tracking_rawforce_upper',F,fhi-fb,'Nm',(150,4))
    geometry_rows('tracking_rawforce_lower',-F,fb-flo,'Nm',(150,4))
    add('fit_abs_positive',sparse.hstack((sparse.csr_matrix(Q),-eye,zero)),target-qbase,'rad',(151,))
    add('fit_abs_negative',sparse.hstack((-sparse.csr_matrix(Q),-eye,zero)),qbase-target,'rad',(151,))
    add('coefficient_second_difference_positive',sparse.hstack((sparse.csr_matrix(S),zero,-eye)),-sbase,'rad coefficient difference',(151,))
    add('coefficient_second_difference_negative',sparse.hstack((-sparse.csr_matrix(S),zero,-eye)),sbase,'rad coefficient difference',(151,))
    A=sparse.vstack(pieces,format='csr');rhs=np.concatenate(rhs)
    cost=np.r_[np.zeros(151),np.full(151,1/151),np.full(151,4/151)]
    kwargs={'c':cost,'A_ub':A,'b_ub':rhs,
            'bounds':[(float(qlo[j]),float(qhi[j]))]*151+[(0,None)]*302,
            'method':'highs','options':{'primal_feasibility_tolerance':1e-9,
             'dual_feasibility_tolerance':1e-9,'time_limit':5.,'maxiter':100000}}
    contract={'schema':'full151_joint_linear_tracking_contract:1','joint_index':j,
      'variable_layout':'P2..152 abs151,fit_abs151,coefficient_secondDiff_abs151',
      'nvariables':453,'row_groups':groups,'P0':float(P0[j]),'P1':float(P1[j]),
      'target151':target.copy(),'q_matrix':Q,'q_base':qbase,'qd_matrix':D,'qd_base':dbase,
      'command_matrix':U,'command_base':ub,'force_Bernstein_matrix':F,'force_Bernstein_base':fb,
      'second_difference_matrix':S,'second_difference_base':sbase,
      'planning_target_bounds':np.array([ulo,uhi]),'planning_force_bounds':np.array([flo,fhi]),
      'fixed_u0_FP32_as_real':float(u0[j]),'static_tau_FP32_as_real':float(tau[j]),
      'margins':{'control_rad':2e-6,'rawforce_Nm':2e-6,'rate_rad_s':1e-4,'planning_rate_rad_s':14.9999,
                 'original_rate_gate_rad_s':15.,'planning_certificate_tolerance':1e-12,'qhull_tolerance_rad':1e-12},
      'objective':'mean151 absolute q fit +4*mean151 absolute coefficient secondDifference',
      'scope':'Tracking-only sufficient reference-PD surrogate. Coefficient secondDifference is not an exact physical acceleration integral. No hold counterfactual, no hardware qdot bound, no actual force/geometry/dynamics qualification.'}
    return {'linprog_kwargs':kwargs,'contract':contract}


def unpack_joint_results(free151x20,P0,P1):
    return np.vstack((_a(P0,(20,)),_a(P1,(20,)),_a(free151x20,(151,20))))


def sample_controls(P):
    P=_a(P,(153,20));times=np.arange(601,dtype=float)*.005
    q=eval_controls(P,BREAKS);q32=q.astype(np.float32)
    fd=np.diff(q,axis=0)/H;fd32=np.diff(q32,axis=0)/np.float32(H)
    return {'time_s':BREAKS.copy(),'q_goal_rad':q,'analytic_qd151_rad_s':eval_controls(P,BREAKS,1),
      'analytic_qdd151_rad_s2':eval_controls(P,BREAKS,2),
      'native_forward_FD64_rad_s':np.r_[fd,fd[-1:]],'native_forward_FD32_rad_s':np.r_[fd32,fd32[-1:]],
      'planned_time601_s':times,'planned_q601_rad':eval_controls(P,times),
      'planned_qd601_rad_s':eval_controls(P,times,1),'planned_qdd601_rad_s2':eval_controls(P,times,2)}


def audit_post_C2(P,params,tau_ff32,u0_F32,qlo,qhi,frozen_control_module):
    """No solve; parent supplies the hash-checked frozen760 FP32 module.

    Keep all failed quantities. Do not replace the generated first target by the
    pinned u0 to hide an initialization mismatch. U150 is never executed.
    """
    P=_a(P,(153,20));tau=_a(tau_ff32,(20,));fixedu0=_a(u0_F32,(20,))
    kp,bias,lo,hi,act,joint=_parameters(params)
    qlo,qhi=_a(qlo,(20,)),_a(qhi,(20,));data=sample_controls(P)
    old=frozen_control_module
    pre=old.proposal_numpy(data['q_goal_rad'][:150],data['analytic_qd151_rad_s'][:150],tau,params)
    U=old.c2_numpy(pre,params)
    rates=np.diff(U.astype(float),axis=0)/H
    bez=bezier_matrices();delta=P[1:]-P[0]
    qbez=P[0]+np.einsum('irc,cj->irj',bez['q_cubic'][:,:,1:],delta)
    vbez=np.einsum('irc,cj->irj',bez['qd_elevated_cubic'][:,:,1:],delta)
    f=kp*U.astype(float)[:,None,:]+bias[:,0]+bias[:,1]*qbez+bias[:,2]*vbez
    flo=np.maximum(act[:,0],joint[:,0]);fhi=np.minimum(act[:,1],joint[:,1])
    excess=np.maximum(np.maximum(flo-f,f-fhi),0)
    index=np.minimum(np.arange(601)//4,149)
    aff,diff=old.affine_predictions_numpy(data['planned_q601_rad'],data['planned_qd601_rad_s'],U[index],params)
    within=lambda x:bool(np.all(x>=flo) and np.all(x<=fhi))
    qexcess=np.maximum(np.maximum(qlo-P,P-qhi),0)
    initial_exact=bool(np.array_equal(U[0].view(np.uint32),fixedu0.astype(np.float32).view(np.uint32)))
    innerlo=lo.astype(np.float32)+np.float32(.020);innerhi=hi.astype(np.float32)-np.float32(.020)
    data.update({'u_pre150_FP32_rad':pre,'u_postC2_150_FP32_rad':U,'planned_rate149_rad_s':rates,
      'postC2_real_affine_rawforce_Bernstein_Nm':f,'reference_affine601_FP32_Nm':aff,
      'reference_difference601_FP32_Nm':diff,'solver_prestate_interval600':np.arange(600)//4,
      'reference_diagnostic_interval601':index,'initial_ctrl_to_first_target_jump_rad':U[0].astype(float)-fixedu0,
      'gates':{'q_controlpoint_hull':bool(qexcess.max()<=QHULL_TOL_RAD),
       'C2_domain':bool(np.all(U>=innerlo) and np.all(U<=innerhi)),
       'C2_projection_identity':bool(np.array_equal(pre.view(np.uint32),U.view(np.uint32))),
       'same_fixed_initial_u0':initial_exact,'target149_updates_le15':bool(np.abs(rates).max()<=15.),
       'postC2_wholeinterval_real_affine_Bernstein_caps':bool(not np.any(excess>0)),
       'FP32_affine601_reference_caps':within(aff),'FP32_difference601_reference_caps':within(diff)},
      'max_controlpoint_hard_excess_rad':float(qexcess.max()),
      'max_postC2_Bernstein_rawforce_excess_Nm':float(excess.max()),
      'max_planned_target_rate_rad_s':float(np.abs(rates).max()),
      'C2_changed_components150':int(np.count_nonzero(pre!=U)),
      'scope':'150 planned targets/149 updates. State601 is terminal diagnostic, no u150 actuation. Both closed interval endpoint limits certified as exact-real affine polynomials at final U32, not a whole-interval FP32/FMA or physical-force certificate. FP32 samples and native FD are distinct diagnostic fields. No hold arm.'})
    return data
