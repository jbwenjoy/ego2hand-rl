"""Frozen prefix16 kinematic curves; no optimizer, physics or controller state.

SO(3): R=Ri Exp(phi), omega_body=Jr(phi) phi_dot. All derivatives use
physical seconds. Native forward-FD qd is deliberately a separate field.
"""
from dataclasses import dataclass
import numpy as np

Q_STENCILS = {
    'schema': 'prefix16_q_derivative_stencils:1', 'knots': 16, 'dt_s': .02,
    'continuous_qd_interior': {'offsets': [-1, 0, 1], 'weights': [-.5, 0., .5], 'divide_by': 'h'},
    'continuous_qd_first': {'indices': [0, 1, 2], 'weights': [-1.5, 2., -.5], 'divide_by': 'h'},
    'continuous_qd_last': {'offsets_from_last': [0, -1, -2], 'weights': [1.5, -2., .5], 'divide_by': 'h'},
    'continuous_qdd_interior': {'offsets': [-1, 0, 1], 'weights': [1., -2., 1.], 'divide_by': 'h**2'},
    'continuous_qdd_first': {'indices': [0, 1, 2, 3], 'weights': [2., -5., 4., -1.], 'divide_by': 'h**2'},
    'continuous_qdd_last': {'offsets_from_last': [0, -1, -2, -3], 'weights': [2., -5., 4., -1.], 'divide_by': 'h**2'},
    'native_FD_qd': 'forward difference at0..14; row15 repeats row14; source64 and nativeFP32 are exported separately',
    'FP64_stencil_grouping': 'First differences then second differences; algebraically identical declared coefficients and exact zero for constant q',
    'range_check': 'all real derivative-polynomial roots in each complete interval plus endpoints; no clipping; zero range slack; near-real ambiguous roots fail closed',
}


def finite(a, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError('Unexpected shape or nonfinite kinematic input')
    return a


def immutable(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def hat(v):
    x, y, z = finite(v, (3,))
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def coefficients(y0, y1, d0, d1, a0, a1, h):
    y0, y1, d0, d1, a0, a1 = (finite(x) for x in (y0, y1, d0, d1, a0, a1))
    if not all(x.shape == y0.shape for x in (y1, d0, d1, a0, a1)) or not np.isfinite(h) or h <= 0:
        raise ValueError('Invalid Hermite endpoints or interval')
    delta = y1 - y0
    return np.stack((y0, h*d0, .5*h*h*a0,
        10*delta-6*h*d0-4*h*d1-1.5*h*h*a0+.5*h*h*a1,
        -15*delta+8*h*d0+7*h*d1+1.5*h*h*a0-h*h*a1,
        6*delta-3*h*(d0+d1)-.5*h*h*a0+.5*h*h*a1))


def evaluate(c, s, h, order=0):
    if order not in (0, 1, 2) or not np.isfinite(s) or s < -1e-14 or s > 1+1e-14:
        raise ValueError('Invalid curve evaluation')
    c = np.asarray(c)
    answer = np.zeros(c.shape[1:])
    for k in range(5, order-1, -1):
        factor = 1 if order == 0 else k if order == 1 else k*(k-1)
        answer = answer*s + factor*c[k]
    return answer / h**order


def q_derivatives(q):
    q = finite(q, (16, 20)); h = .02
    qd = np.empty_like(q); qdd = np.empty_like(q)
    delta = np.diff(q,axis=0); second = np.diff(delta,axis=0)
    qd[1:-1] = (delta[1:]+delta[:-1])/(2*h)
    qd[0] = (3*delta[0]-delta[1])/(2*h)
    qd[-1] = (3*delta[-1]-delta[-2])/(2*h)
    qdd[1:-1] = second/h**2
    qdd[0] = (2*second[0]-second[1])/h**2
    qdd[-1] = (2*second[-1]-second[-2])/h**2
    native = np.diff(q, axis=0)/h
    native = np.concatenate((native, native[-1:]), axis=0)
    q32 = q.astype(np.float32); native32 = np.diff(q32, axis=0)/np.float32(h)
    native32 = np.concatenate((native32, native32[-1:]), axis=0)
    return qd, qdd, native, native32


def _abc(phi):
    rho = float(phi@phi)
    if rho < 1e-6:
        b = .5-rho/24+rho*rho/720-rho**3/40320
        c = 1/6-rho/120+rho*rho/5040-rho**3/362880
        db_drho = -1/24+rho/360-rho*rho/13440
        dc_drho = -1/120+rho/2520-rho*rho/120960
    else:
        t = np.sqrt(rho)
        b = (1-np.cos(t))/rho; c = (t-np.sin(t))/(rho*t)
        db_drho = (t*np.sin(t)-2*(1-np.cos(t)))/(2*rho*rho)
        dc_drho = (t*(1-np.cos(t))-3*(t-np.sin(t)))/(2*rho*rho*t)
    return b, c, db_drho, dc_drho


def so3_exp(phi):
    phi = finite(phi, (3,)); rho = float(phi@phi); k = hat(phi)
    a = 1-rho/6+rho*rho/120-rho**3/5040 if rho < 1e-6 else np.sin(np.sqrt(rho))/np.sqrt(rho)
    b = _abc(phi)[0]
    return np.eye(3)+a*k+b*(k@k)


def so3_log(rotation):
    r = finite(rotation, (3, 3))
    if np.max(np.abs(r.T@r-np.eye(3))) > 1e-8 or abs(np.linalg.det(r)-1) > 1e-8:
        raise ValueError('Expected proper SO3 rotation')
    theta = float(np.arccos(np.clip((np.trace(r)-1)/2, -1, 1)))
    if theta >= np.pi-1e-5:
        raise ValueError('Near-pi branch is unqualified; no silent axis choice')
    v = np.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])/2
    factor = 1+theta**2/6+7*theta**4/360 if theta < 1e-3 else theta/np.sin(theta)
    return factor*v


def right_jacobian(phi):
    phi = finite(phi, (3,)); k = hat(phi); b, c, _, _ = _abc(phi)
    return np.eye(3)-b*k+c*k@k


def right_jacobian_directional(phi, direction):
    phi = finite(phi, (3,)); v = finite(direction, (3,)); k, dk = hat(phi), hat(v)
    b, c, br, cr = _abc(phi); drho = 2*float(phi@v)
    return -br*drho*k-b*dk+cr*drho*(k@k)+c*(dk@k+k@dk)


def quaternion_matrices(q):
    q = finite(q)
    if q.shape[-1] != 4 or np.max(np.abs(np.linalg.norm(q, axis=-1)-1)) > 1e-6:
        raise ValueError('Expected finite unit wxyz quaternions')
    q = q/np.linalg.norm(q, axis=-1, keepdims=True)
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


@dataclass(frozen=True)
class PrefixCurves:
    time_s: np.ndarray
    q_coefficients: np.ndarray
    p_coefficients: np.ndarray
    phi_coefficients: np.ndarray
    rotation_knots: np.ndarray
    q_goal_rad: np.ndarray
    q_curve_qd_rad_s: np.ndarray
    q_curve_qdd_rad_s2: np.ndarray
    native_FD_goal_qd_rad_s: np.ndarray
    native_FD_goal_qd_FP32_rad_s: np.ndarray

    def sample(self, time_s):
        times = finite(time_s).reshape(-1)
        if np.any(times < self.time_s[0]) or np.any(times > self.time_s[-1]):
            raise ValueError('No extrapolation or appended old frame')
        result = {k: [] for k in ('q_goal_rad','q_curve_qd_rad_s','q_curve_qdd_rad_s2','object_COM_world_m','object_v_world_m_s','object_a_world_m_s2','object_rotation_w','object_omega_world_rad_s','object_alpha_world_rad_s2')}
        for t in times:
            i = min(int(np.searchsorted(self.time_s,t,side='right')-1),14); h = self.time_s[i+1]-self.time_s[i]; s = (t-self.time_s[i])/h
            for order,key in enumerate(('q_goal_rad','q_curve_qd_rad_s','q_curve_qdd_rad_s2')): result[key].append(evaluate(self.q_coefficients[i],s,h,order))
            for order,key in enumerate(('object_COM_world_m','object_v_world_m_s','object_a_world_m_s2')): result[key].append(evaluate(self.p_coefficients[i],s,h,order))
            phi,phid,phidd = (evaluate(self.phi_coefficients[i],s,h,k) for k in range(3))
            rotation = self.rotation_knots[i]@so3_exp(phi); jr = right_jacobian(phi)
            result['object_rotation_w'].append(rotation)
            result['object_omega_world_rad_s'].append(rotation@(jr@phid))
            result['object_alpha_world_rad_s2'].append(rotation@(jr@phidd+right_jacobian_directional(phi,phid)@phid))
        return {k: np.asarray(v) for k,v in result.items()}

    def q_range_check(self, lower, upper):
        lower,upper = finite(lower,(20,)),finite(upper,(20,))
        if np.any(lower>=upper): raise ValueError('Empty joint domain')
        rows=[]; ambiguous=[]; mins=np.full(20,np.inf);maxs=np.full(20,-np.inf)
        for i in range(15):
            for j in range(20):
                c=self.q_coefficients[i,:,j]; roots=np.polynomial.polynomial.polyroots(c[1:]*np.arange(1,6)) if np.any(c[1:]) else np.array([])
                candidates=[0.,1.]
                for root in roots:
                    if abs(root.imag)<=1e-10*(1+abs(root.real)) and 0<root.real<1: candidates.append(float(root.real))
                    elif abs(root.imag)<=1e-7*(1+abs(root.real)) and 0<root.real<1: ambiguous.append({'segment':i,'joint':j,'root_real':float(root.real),'root_imag':float(root.imag)})
                values=np.polynomial.polynomial.polyval(np.asarray(candidates),c)
                lo,hi=float(values.min()),float(values.max());mins[j]=min(mins[j],lo);maxs[j]=max(maxs[j],hi)
                rows.append({'segment':i,'joint':j,'candidate_s':candidates,'candidate_values_rad':values.tolist(),'min_rad':lo,'max_rad':hi,'hard_excess_rad':max(lower[j]-lo,hi-upper[j],0.)})
        excess=np.maximum(np.maximum(lower-mins,maxs-upper),0)
        # The canonical q0 has a known numerical boundary. Outward initial
        # derivative is independently flagged; the polynomial is never clipped.
        near_lower=np.abs(self.q_goal_rad[0]-lower)<=1e-12;near_upper=np.abs(self.q_goal_rad[0]-upper)<=1e-12
        outward=np.flatnonzero((near_lower&(self.q_curve_qd_rad_s[0]<0))|(near_upper&(self.q_curve_qd_rad_s[0]>0)))
        return {'schema':'whole_quintic_joint_range:1','passed':not ambiguous and not np.any(excess>0) and len(outward)==0,'joint_min_rad':mins.tolist(),'joint_max_rad':maxs.tolist(),'max_hard_excess_rad':float(excess.max()),'numeric_domain_slack_rad':0.,'root_classification_ambiguous':ambiguous,'initial_outward_boundary_joint_indices':outward.tolist(),'all300_segments_joints':rows,'scope':'Every endpoint/stationary root of each stored FP64 polynomial; conservative floating numerical audit, not a formal interval-arithmetic proof.'}


def build_prefix_curves(time_s,q_goal_rad,mapped_object_pose_w,object_v_world_m_s,object_a_world_m_s2,object_omega_world_rad_s,object_alpha_world_rad_s2):
    t=finite(time_s,(16,));q=finite(q_goal_rad,(16,20));pose=finite(mapped_object_pose_w,(16,7))
    if not np.array_equal(t,np.arange(16)*.02):raise ValueError('Expected frozen16-knots50Hz local clock')
    v,a,omega,alpha=(finite(x,(16,3)) for x in (object_v_world_m_s,object_a_world_m_s2,object_omega_world_rad_s,object_alpha_world_rad_s2))
    rotations=quaternion_matrices(pose[:,3:]);qd,qdd,native,native32=q_derivatives(q)
    qc=[];pc=[];rc=[]
    for i,h in enumerate(np.diff(t)):
        qc.append(coefficients(q[i],q[i+1],qd[i],qd[i+1],qdd[i],qdd[i+1],h));pc.append(coefficients(pose[i,:3],pose[i+1,:3],v[i],v[i+1],a[i],a[i+1],h))
        phi=so3_log(rotations[i].T@rotations[i+1]);j1=right_jacobian(phi)
        d0=rotations[i].T@omega[i];d1=np.linalg.solve(j1,rotations[i+1].T@omega[i+1]);a0=rotations[i].T@alpha[i]
        a1=np.linalg.solve(j1,rotations[i+1].T@alpha[i+1]-right_jacobian_directional(phi,d1)@d1)
        rc.append(coefficients(np.zeros(3),phi,d0,d1,a0,a1,h))
    return PrefixCurves(*(immutable(x) for x in (t,np.asarray(qc),np.asarray(pc),np.asarray(rc),rotations,q,qd,qdd,native,native32)))
