"""Clamped cubic geometry and sufficient linear raw-PD planning contracts.

No LP, nonlinear solve, physics or geometry query is executed here. Variables
are absolute P[1:18,20] in control-major C order; P0 is source64 q0, never clipped.
"""
from pathlib import Path
import hashlib,importlib.util,sys
import numpy as np
from scipy.interpolate import BSpline
from scipy import sparse

DEGREE=3
NCONTROL=18
NVARIABLE=340
BREAKS=np.arange(16,dtype=np.float64)*.02
CONTROL_MARGIN_RAD=2e-6
FORCE_MARGIN_NM=2e-6
RATE_MARGIN_RAD_S=1e-4
EXECUTION_RATE_LIMIT_RAD_S=15.
Q_CERT_ATOL_RAD=1e-12
PROTOCOL_SHA256='0798ea7aa6da6f56684067992906bbba3136f9e15b413f743674f54159ad2e6d'
FROZEN_CONTROL=Path('D:/workspace/supermimic-observation-prefix-study/src/prefix_control.py')
FROZEN_CONTROL_SHA='76026dbdcdfdecf4680820b24da088e1d123e86ceb8b42a2bbd60a5412103b6d'


def _a(x,shape=None):
    x=np.asarray(x,dtype=np.float64)
    if (shape is not None and x.shape!=shape) or not np.isfinite(x).all():raise ValueError('Shape/nonfinite contract violation')
    return x


def _ro(x):
    x=np.ascontiguousarray(x)
    return np.frombuffer(x.tobytes(),dtype=x.dtype).reshape(x.shape)


class ClampedCubicBasis:
    def __init__(self):
        self.breaks=_ro(BREAKS);self.degree=3;self.ncontrol=18
        self.knots=_ro(np.r_[np.repeat(BREAKS[0],4),BREAKS[1:-1],np.repeat(BREAKS[-1],4)])
        self._spline=BSpline(self.knots,np.eye(18),3,extrapolate=False)

    def evaluate(self,t,derivative=0):
        t=np.atleast_1d(_a(t))
        if t.ndim!=1 or not len(t) or derivative not in (0,1,2) or np.any(t<0) or np.any(t>BREAKS[-1]):raise ValueError('No extrapolation/unsupported derivative')
        return np.asarray(self._spline(t,nu=derivative),dtype=np.float64)


def make_basis():return ClampedCubicBasis()


def basis_at(t,derivative=0):return make_basis().evaluate(t,derivative)


def unpack_controls(x,q0):
    return np.vstack((_a(q0,(20,)),_a(x,(340,)).reshape(17,20)))


def evaluate_controls(x,q0,t,derivative=0):
    q0=_a(q0,(20,));delta=_a(x,(340,)).reshape(17,20)-q0
    value=basis_at(t,derivative)[:,1:]@delta
    return value+q0 if derivative==0 else value


def bezier_basis():
    """q cubic Bezier basis and qd quadratic/elevated cubic, physical seconds."""
    b=make_basis();left,right=BREAKS[:-1],BREAKS[1:];h=np.diff(BREAKS)[:,None]
    l,r=b.evaluate(left),b.evaluate(right);dl,dr=b.evaluate(left,1),b.evaluate(right,1)
    eq=np.stack((l,l+h*dl/3,r-h*dr/3,r),axis=1)
    ev2=(3/h[:,None,:])*np.diff(eq,axis=1)
    ev3=np.stack((ev2[:,0],ev2[:,0]/3+2*ev2[:,1]/3,2*ev2[:,1]/3+ev2[:,2]/3,ev2[:,2]),axis=1)
    return {'q_cubic':_ro(eq),'qd_quadratic':_ro(ev2),'qd_elevated_cubic':_ro(ev3),'interval_seconds':_ro(np.diff(BREAKS))}


def bernstein3(s):
    s=_a(s)
    if np.any(s<0) or np.any(s>1):raise ValueError('Bernstein parameter outside interval')
    return np.stack(((1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3),axis=-1)


def control_point_certificate(x,q0,lower,upper):
    p=unpack_controls(x,q0);lower,upper=_a(lower,(20,)),_a(upper,(20,))
    if np.any(lower>=upper):raise ValueError('Empty hard domain')
    excess=np.maximum(np.maximum(lower-p,p-upper),0)
    e=bezier_basis();delta=p[1:]-q0;velocity=np.einsum('irc,cj->irj',e['qd_quadratic'][:,:,1:],delta)
    return {'schema':'cubic_controlpoint_convexhull_certificate:1','passed':bool(excess.max()<=Q_CERT_ATOL_RAD),'max_controlpoint_excess_rad':float(excess.max()),'numerical_q_evaluation_tolerance_rad':Q_CERT_ATOL_RAD,'controlpoint_min_rad':p.min(axis=0).tolist(),'controlpoint_max_rad':p.max(axis=0).tolist(),'derivative_Bernstein_min_rad_s':velocity.min(axis=(0,1)).tolist(),'derivative_Bernstein_max_rad_s':velocity.max(axis=(0,1)).tolist(),'velocity_hardware_limit_assumed':None,'scope':'Hard controlpoint box is a sufficient fullcurve convexhull restriction, not necessary for every legal curve. Derivative ranges are descriptive; 15rad/s is the target-update limit, not an invented hardware qdot bound.'}


def _joint_matrix(coeff):
    """coeff[...,17,20] maps centered absolute controls to outputs[...,20]."""
    coeff=_a(coeff);shape=coeff.shape[:-2]
    if coeff.shape[-2:]!=(17,20):raise ValueError('Expected free-control/joint coefficient axes')
    count=int(np.prod(shape));v=coeff.reshape(count,17,20)
    row=np.broadcast_to(np.arange(count)[:,None,None]*20+np.arange(20)[None,None,:],v.shape)
    col=np.broadcast_to(np.arange(17)[None,:,None]*20+np.arange(20)[None,None,:],v.shape)
    result=sparse.csr_matrix((v.ravel(),(row.ravel(),col.ravel())),shape=(count*20,340));result.eliminate_zeros()
    return result


def _parameters(params):
    if hasattr(params,'validate'):params.validate()
    kp=_a(params.kp,(20,));bias=_a(params.bias,(20,3));lo=_a(params.target_lower,(20,));hi=_a(params.target_upper,(20,));act=_a(params.actuator_force_bounds,(20,2));joint=_a(params.joint_force_bounds,(20,2))
    if np.any(kp<=0) or not np.array_equal(bias[:,1],-kp) or np.any(lo>=hi) or np.any(act[:,0]>=act[:,1]) or np.any(joint[:,0]>=joint[:,1]):raise ValueError('Unsupported actual affine PD parameters')
    return kp,bias,lo,hi,act,joint


def build_linear_contract(q0,params,tau_ff_static0,hard_lower,hard_upper):
    q0=_a(q0,(20,));qlo,qhi=_a(hard_lower,(20,)),_a(hard_upper,(20,));tau=_a(tau_ff_static0,(20,))
    kp,bias,outerlo,outerhi,act,joint=_parameters(params)
    if np.any(qlo>=qhi) or np.any(q0<qlo-Q_CERT_ATOL_RAD) or np.any(q0>qhi+Q_CERT_ATOL_RAD):raise ValueError('Fixed source64 q0 outside hard domain; no clipping')
    # Reproduce the OLD C2 FP32 domain before adding planning-only tightening.
    innerlo=(outerlo.astype(np.float32)+np.float32(.020)).astype(float)
    innerhi=(outerhi.astype(np.float32)-np.float32(.020)).astype(float)
    ulo,uhi=innerlo+CONTROL_MARGIN_RAD,innerhi-CONTROL_MARGIN_RAD
    flo=np.maximum(act[:,0],joint[:,0])+FORCE_MARGIN_NM;fhi=np.minimum(act[:,1],joint[:,1])-FORCE_MARGIN_NM
    if np.any(ulo>uhi) or np.any(flo>fhi):raise ValueError('Tightening produced empty planning limits')
    b,d=basis_at(BREAKS[:15])[:,1:],basis_at(BREAKS[:15],1)[:,1:]
    uc=-(bias[:,1]/kp)[None,None,:]*b[:,:,None]-(bias[:,2]/kp)[None,None,:]*d[:,:,None]
    command=_joint_matrix(uc);baseu=(tau-bias[:,0]-bias[:,1]*q0)/kp
    commandbase=np.tile(baseu,15)
    e=bezier_basis();eq,ev=e['q_cubic'][:,:,1:],e['qd_elevated_cubic'][:,:,1:]
    trackcoeff=kp[None,None,None,:]*uc[:,None,:,:]+bias[:,1][None,None,None,:]*eq[:,:,:,None]+bias[:,2][None,None,None,:]*ev[:,:,:,None]
    holdcoeff=kp[None,None,None,:]*uc[0][None,None,:,:]+bias[:,1][None,None,None,:]*eq[:,:,:,None]+bias[:,2][None,None,None,:]*ev[:,:,:,None]
    trackforce,holdforce=_joint_matrix(trackcoeff),_joint_matrix(holdcoeff)
    forcebase=np.tile(kp*baseu+bias[:,0]+bias[:,1]*q0,60)
    rate=command[20:]-command[:-20]
    planning_rate=EXECUTION_RATE_LIMIT_RAD_S-RATE_MARGIN_RAD_S
    pieces=[];rhs=[];groups=[];cursor=0
    def add(name,matrix,bound,units,shape):
        nonlocal cursor
        bound=np.asarray(bound,float).reshape(-1)
        if matrix.shape[0]!=len(bound):raise ValueError('Linear row shape')
        pieces.append(matrix);rhs.append(bound);groups.append({'name':name,'start':cursor,'stop':cursor+len(bound),'units':units,'tensor_shape':list(shape)});cursor+=len(bound)
    add('target_upper',command,np.tile(uhi,15)-commandbase,'rad',(15,20))
    add('target_lower',-command,commandbase-np.tile(ulo,15),'rad',(15,20))
    add('target_rate_upper',rate,np.full(280,.02*planning_rate),'rad/update',(14,20))
    add('target_rate_lower',-rate,np.full(280,.02*planning_rate),'rad/update',(14,20))
    for name,matrix in [('tracking',trackforce),('hold',holdforce)]:
        add(name+'_force_upper',matrix,np.tile(fhi,60)-forcebase,'Nm',(15,4,20))
        add(name+'_force_lower',-matrix,forcebase-np.tile(flo,60),'Nm',(15,4,20))
    aub=sparse.vstack(pieces,format='csr');centered=np.concatenate(rhs);x0=np.tile(q0,17)
    absolute=centered+aub@x0
    return {'schema':'clamped_cubic_linear_planning:1','protocol_sha256':PROTOCOL_SHA256,'degree':3,'breaks':BREAKS.copy(),'knots':make_basis().knots.copy(),'ncontrol':18,'nvariable':340,'variable_layout':'P[1:18,20].reshape340 C-order/control-major; P0 fixed source64 q0',
        'q0':q0.copy(),'x0':x0,'lower':np.tile(qlo,17),'upper':np.tile(qhi,17),'A_ub':aub,'b_ub':absolute,'b_centered':centered,'row_groups':groups,
        'command_matrix':command,'command_baseline':commandbase,'tracking_force_matrix':trackforce,'hold_force_matrix':holdforce,'force_baseline':forcebase,
        'q_Bezier_basis':e['q_cubic'],'qd_Bezier_basis':e['qd_quadratic'],'qd_degree_elevated_Bezier_basis':e['qd_elevated_cubic'],
        'planning_target_lower':ulo,'planning_target_upper':uhi,'planning_force_lower':flo,'planning_force_upper':fhi,
        'margins':{'target_domain_rad':CONTROL_MARGIN_RAD,'raw_force_Nm':FORCE_MARGIN_NM,'rate_rad_s':RATE_MARGIN_RAD_S,'planning_rate_rad_s':planning_rate,'execution_rate_gate_rad_s':15.,'q_evaluation_atol_rad':Q_CERT_ATOL_RAD},
        'phase0':'One HiGHS L1 nearest repeatq0,340abs epigraph, no second tie-break; only matrices generated here',
        'scope':'Controlpoint boxes and Bernstein rawforce constraints are sufficient restrictions, not necessary conditions. Each closed polynomial interval includes both one-sided held-target endpoint limits. Hold force checks use a moving-qref counterfactual, not actual hold safety. This FP64 linear surrogate is not a FP32/dynamics/WM certificate; final post-C2 audit required.'}


def phase0_l1_problem(contract):
    """Return linprog kwargs only; does NOT execute an LP or choose a solution."""
    n=340;a=contract['A_ub'];zero=sparse.csr_matrix((a.shape[0],n));eye=sparse.eye(n,format='csr')
    augmented=sparse.vstack((sparse.hstack((a,zero)),sparse.hstack((eye,-eye)),sparse.hstack((-eye,-eye))),format='csr')
    return {'c':np.r_[np.zeros(n),np.ones(n)],'A_ub':augmented,'b_ub':np.r_[contract['b_ub'],contract['x0'],-contract['x0']],'bounds':list(zip(contract['lower'],contract['upper']))+[(0,None)]*n,'method':'highs','options':{'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}}


def planned_values(contract,x):
    delta=_a(x,(340,))-contract['x0']
    return {'commands15':(contract['command_matrix']@delta+contract['command_baseline']).reshape(15,20),
        'tracking_rawforce_Bernstein':(contract['tracking_force_matrix']@delta+contract['force_baseline']).reshape(15,4,20),
        'hold_rawforce_Bernstein':(contract['hold_force_matrix']@delta+contract['force_baseline']).reshape(15,4,20),
        'centered_constraint_residual':contract['A_ub']@delta-contract['b_centered']}


def load_frozen_fp32_control():
    if hashlib.sha256(FROZEN_CONTROL.read_bytes()).hexdigest()!=FROZEN_CONTROL_SHA:raise ValueError('Old FP32 controller source drift')
    name='_frozen_prefix_fp32_control_76026d';spec=importlib.util.spec_from_file_location(name,FROZEN_CONTROL);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def audit_post_C2_controls(x,q0,params,u_PDU,hard_lower,hard_upper):
    """Reuse frozen FP32 export, then re-certify ACTUAL PLANNED U32, no solve."""
    old=load_frozen_fp32_control();x=_a(x,(340,));q0=_a(q0,(20,));kp,bias,lo,hi,act,joint=_parameters(params)
    class CurveView:
        time_s=BREAKS
        def sample(self,t):return {'q_goal_rad':evaluate_controls(x,q0,t),'q_curve_qd_rad_s':evaluate_controls(x,q0,t,1)}
    result=old.export_control(CurveView(),params,q0,u_PDU)
    p=unpack_controls(x,q0);delta=p[1:]-q0;e=bezier_basis()
    qbez=q0+np.einsum('irc,cj->irj',e['q_cubic'][:,:,1:],delta)
    vbez=np.einsum('irc,cj->irj',e['qd_elevated_cubic'][:,:,1:],delta)
    force_cert={};coefficients={}
    for name,commands in [('tracking',result['tracking_u_exec_15_FP32_rad']),('hold',result['hold_u_exec_15_FP32_rad'])]:
        # Real affine coefficients with the FINAL post-C2 FP32 values represented
        # exactly as reals. This is not a whole-interval FP32/FMA proof.
        f=kp*commands.astype(float)[:,None,:]+bias[:,0]+bias[:,1]*qbez+bias[:,2]*vbez
        violation=np.maximum(np.maximum(np.maximum(act[:,0],joint[:,0])-f,f-np.minimum(act[:,1],joint[:,1])),0)
        coefficients[name]=f
        force_cert[name]={'passed':bool(not np.any(violation>0)),'maximum_rawforce_coefficient_excess_Nm':float(violation.max()),'sufficient_not_necessary':True,'hold_moving_reference_counterfactual':name=='hold'}
    times=np.arange(61)*.005;idx=np.minimum(np.arange(61)//4,14);q=evaluate_controls(x,q0,times);v=evaluate_controls(x,q0,times,1)
    samples={};sample_gates={}
    for name,commands in [('tracking',result['tracking_u_exec_15_FP32_rad']),('hold',result['hold_u_exec_15_FP32_rad'])]:
        a,d=old.affine_predictions_numpy(q,v,commands[idx],params)
        samples[name+'_rawPD_affine61_FP32_Nm']=a;samples[name+'_rawPD_difference61_FP32_Nm']=d
        sample_gates[name]=all(np.all(f>=bound[:,0]) and np.all(f<=bound[:,1]) for f in (a,d) for bound in (act,joint))
    q16=evaluate_controls(x,q0,BREAKS);native=np.diff(q16,axis=0)/.02;native=np.r_[native,native[-1:]]
    q32=q16.astype(np.float32);native32=np.diff(q32,axis=0)/np.float32(.02);native32=np.r_[native32,native32[-1:]]
    return {'fp32_old_export':result,'q_goal16_rad':q16,'analytic_qd16_rad_s':evaluate_controls(x,q0,BREAKS,1),'analytic_qdd16_rad_s2':evaluate_controls(x,q0,BREAKS,2),'native_forward_FD64_rad_s':native,'native_forward_FD32_rad_s':native32,'q_convexhull_certificate':control_point_certificate(x,q0,hard_lower,hard_upper),'postC2_real_affine_Bernstein_coefficients':coefficients,'postC2_real_affine_Bernstein_certificate':force_cert,'FP32_reference61_samples':samples,'FP32_reference61_caps_pass':{k:bool(v) for k,v in sample_gates.items()},'proposal_postC2_identity_actual15':bool(np.array_equal(result['u_pre_16_FP32_rad'][:15].view(np.uint32),result['tracking_u_exec_15_FP32_rad'].view(np.uint32))),
        'scope':'Analytic BSpline derivatives; native FD is separate. Real-affine Bernstein sufficiency at final U32 differs from61FP32 samples; neither proves actual dynamics, force realization, hardware velocity bounds or WM gradients. Final61st sample is reference-state diagnostic, not a61st solver actuation.'}
