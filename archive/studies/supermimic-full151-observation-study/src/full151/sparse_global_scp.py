"""Pure sparse construction for a bounded full151 elastic contact SCP.

No solver, geometry oracle, model, FK, collision, Env, WM or GPU is called.
The external oracle supplies *all* generated contact-witness linearizations at
one current601 grid. An absent native pair is unknown, not a clearance row.
"""
import numpy as np
from scipy import sparse

NJOINT=20
NFREE=151
JOINT_LP_VARIABLES=453
NBASE=9060
NVARIABLE=9061
INITIAL_TRUST_RAD=.15
ROW_SCALE=1e4
ETA_PENALTY_RAD_PER_M=1e4
GEOMETRY_PLANNING_DEPTH_M=.002999
ORIGINAL_DEPTH_GATE_M=.003
RAW_PRIMAL_TOL=1e-12
ETA_QUALIFICATION_GATE_M=1e-9
MAJOR_LIMIT=8
PER_MAJOR_SECONDS=10.


def _finite(x,shape=None):
    a=np.asarray(x,dtype=float)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError('Wrong shape or nonfinite SCP array')
    return a


def _scalar(x,name,expected):
    if isinstance(x,(bool,np.bool_)) or not np.isscalar(x) or not np.isfinite(x) or float(x)!=expected:
        raise ValueError(name+' differs from the proposed fixed construction')
    return float(x)


def geometry_to_global_map():
    """Oracle joint-major j*151+c -> global joint-LP j*453+c."""
    rows=np.arange(3020)
    cols=np.repeat(np.arange(20)*453,151)+np.tile(np.arange(151),20)
    return sparse.csr_matrix((np.ones(3020),(rows,cols)),shape=(3020,9060))


def _joint_parts(joint_problems,center_P):
    if len(joint_problems)!=20:raise ValueError('Exactly20 frozen joint contracts required')
    center=_finite(center_P,(153,20))
    A=[];b=[];original_b=[];cost=[];bounds=[];shift=[];center_vector=[];groups=[];cursor=0
    for j,problem in enumerate(joint_problems):
        kw,c=problem['linprog_kwargs'],problem['contract']
        if c['schema']!='full151_joint_linear_tracking_contract:1' or c['joint_index']!=j or c['nvariables']!=453:
            raise ValueError('Joint contract identity/order differs')
        if center[0,j]!=c['P0'] or center[1,j]!=c['P1']:
            raise ValueError('Current curve changed fixed source P0/P1')
        aj=sparse.csr_matrix(kw['A_ub'],dtype=float)
        bj=_finite(kw['b_ub']);cj=_finite(kw['c'],(453,))
        if aj.shape!=(2402,453) or bj.shape!=(2402,) or len(kw['bounds'])!=453 or not np.isfinite(aj.data).all():
            raise ValueError('Unexpected original joint LP dimensions/value')
        expected=np.r_[np.zeros(151),np.full(151,1/151),np.full(151,4/151)]
        if not np.array_equal(cj,expected):raise ValueError('Original fit/smooth relative weights changed')
        offsets=np.r_[np.full(151,c['P0']),np.zeros(302)]
        # This is an algebraic variable change, not a relaxation or an edit of
        # the frozen original source. Raw original units remain available.
        A.append(aj);b.append(bj-aj@offsets);original_b.append(bj.copy());cost.append(cj);shift.append(offsets)
        bounds.extend([(None if low is None else float(low-offsets[k]),None if high is None else float(high-offsets[k])) for k,(low,high) in enumerate(kw['bounds'])])
        absolute=center[2:,j]
        fit=np.abs(c['q_matrix']@absolute+c['q_base']-c['target151'])
        smoothing=np.abs(c['second_difference_matrix']@absolute+c['second_difference_base'])
        center_vector.extend(np.r_[absolute-c['P0'],fit,smoothing])
        for g in c['row_groups']:
            groups.append({'joint':j,'name':g['name'],'start':cursor+g['start'],'stop':cursor+g['stop'],'units':g['units']})
        cursor+=2402
    return {'A':sparse.block_diag(A,format='csr'),'b':np.concatenate(b),'original_b':np.concatenate(original_b),'cost':np.concatenate(cost),
            'bounds':bounds,'absolute_shift':np.concatenate(shift),'center_vector':np.asarray(center_vector),
            'row_groups':groups,'center_P':center.copy()}


def build_scp_lp(joint_problems,center_P,geometry,trust_radius_rad=.15,row_scale=1e4,eta_penalty=1e4):
    """Construct one joint-coupled elastic LP, never solve it.

    geometry = {'A_step':CSR[M,3020], 'rhs_m':float[M], 'units':'m',
                'column_order':'joint-major:j*151+c', 'witness_ids':list[M], ...}
    A_step is -Jdistance composed with B_free; its step-row is
       A_step @ (Pfree_trial-Pfree_center) <= native_dist +2.999mm.
    Every actual generated witness, including ties, is a separate row. No row
    may be silently dropped for missing rank, ties, sign or high penetration.
    """
    # Radius may only shrink after rejection in the external deterministic
    # driver; it never expands beyond the initial .15rad in this proposal.
    if isinstance(trust_radius_rad,(bool,np.bool_)) or not np.isfinite(trust_radius_rad) or not 0<float(trust_radius_rad)<=.15:
        raise ValueError('Trust radius must be in(0,.15]rad')
    _scalar(row_scale,'row_scale',1e4);_scalar(eta_penalty,'eta_penalty',1e4)
    if geometry.get('units')!='m' or geometry.get('column_order')!='joint-major:j*151+c':
        raise ValueError('Geometry units/order must be explicit meters and joint-major')
    if geometry.get('target_depth_m')!=GEOMETRY_PLANNING_DEPTH_M:
        raise ValueError('Geometry planning depth must be2.999mm')
    G=sparse.csr_matrix(geometry['A_step'],dtype=float)
    rhs=_finite(geometry['rhs_m'])
    if G.shape!=(len(rhs),3020) or not np.isfinite(G.data).all() or len(geometry['witness_ids'])!=len(rhs):
        raise ValueError('Geometry witness row dimensions/value/identity mismatch')
    parts=_joint_parts(joint_problems,center_P)
    select=geometry_to_global_map();center=parts['center_vector']
    mapped=G@select
    raw_control=sparse.hstack((parts['A'],sparse.csr_matrix((48040,1))),format='csr')
    raw_geometry=sparse.hstack((mapped,-np.ones((len(rhs),1))),format='csr')
    bgeom=rhs+mapped@center
    Araw=sparse.vstack((raw_control,raw_geometry),format='csr')
    braw=np.r_[parts['b'],bgeom]
    bounds=list(parts['bounds'])
    for j in range(20):
        for c in range(151):
            k=j*453+c;low,high=bounds[k]
            low=max(center[k]-trust_radius_rad,-np.inf if low is None else low)
            high=min(center[k]+trust_radius_rad,np.inf if high is None else high)
            if low>high:raise ValueError('Hard bound/trust intersection empty; no clipping')
            bounds[k]=(float(low),float(high))
    bounds.append((0,None))
    cost=np.r_[parts['cost'],eta_penalty]
    kwargs={'c':cost,'A_ub':Araw*row_scale,'b_ub':braw*row_scale,'bounds':bounds,'method':'highs',
            'options':{'time_limit':10.,'maxiter':100000,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}}
    center_eta=float(max(0.,float(np.max(-rhs)))) if rhs.size else 0.
    contract={'schema':'full151_sparse_elastic_SCP_subproblem:1','nvariables':9061,
       'variable_layout':'joint blocks[j]: centered freeP2..152151,fit_abs151,coefficient_secondDiff_abs151; last eta_m',
       'A_raw':Araw,'b_raw':braw,'A_original_absolute':parts['A'],'b_original_absolute':parts['original_b'],
       'absolute_shift':parts['absolute_shift'],
       'center_vector':np.r_[center,center_eta],'center_P':parts['center_P'],
       'geometry_map':select,'geometry_row_start':48040,'geometry_witness_ids':list(geometry['witness_ids']),
       'control_row_groups':parts['row_groups'],'trust_radius_rad':float(trust_radius_rad),'row_scale':1e4,
       'objective':'sum20(original mean L1 fit +4mean coefficient secondDiff) +1e4rad/m*eta_m',
       'eta_penalty_rad_per_m':1e4,'eta_qualification_gate_m':1e-9,'raw_primal_tolerance':1e-12,
       'geometry_planning_depth_m':.002999,'original_geometry_gate_m':.003,
       'generated_witness_rows':len(rhs),'missing_pair_gap_semantics':'UNKNOWN; absent rows are not a clearance certificate',
       'scope':'Elastic phaseI local model; eta is not permitted physical penetration. Positive row scaling/centering changes arithmetic, not scientific constraints. Actual fresh601 geometry+finalFP32/control audits are mandatory; LP success is insufficient.'}
    return {'linprog_kwargs':kwargs,'contract':contract}


def unpack_trial(solution,contract):
    z=_finite(solution,(9061,));old=_finite(contract['center_P'],(153,20))
    P=old.copy()
    for j in range(20):P[2:,j]=z[j*453:j*453+151]+old[0,j]
    return P,float(z[-1])


def audit_subproblem_solution(solution,contract,bounds):
    """Audit unscaled raw rows/bounds; never certify actual native geometry."""
    z=_finite(solution,(9061,))
    if len(bounds)!=9061:raise ValueError('Bounds length differs')
    residual=contract['A_raw']@z-contract['b_raw']
    original_absolute=contract['A_original_absolute']@(z[:9060]+contract['absolute_shift'])-contract['b_original_absolute']
    low=np.array([-np.inf if a is None else a for a,b in bounds]);high=np.array([np.inf if b is None else b for a,b in bounds])
    excess=np.maximum(np.maximum(low-z,z-high),0)
    geometry=residual[48040:]
    maximum=float(residual.max());boundmax=float(excess.max())
    return {'schema':'full151_SCP_raw_linear_audit:1','max_raw_Ax_minus_b':maximum,
       'max_centered_control_raw_Ax_minus_b':float(residual[:48040].max()),
       'max_original_absolute_raw_Ax_minus_b':float(original_absolute.max()),
       'max_geometry_linear_raw_residual_m':None if not geometry.size else float(geometry.max()),
       'max_bound_excess':boundmax,'raw_primal_pass':bool(maximum<=1e-12 and float(original_absolute.max())<=1e-12 and boundmax<=1e-12),
       'eta_m':float(z[-1]),'eta_gate_pass':bool(0<=z[-1]<=1e-9),
       'raw_residuals':residual,'original_absolute_raw_residuals':original_absolute,'bound_excesses':excess,'actual_geometry_qualified':False,
       'scope':'Only an unscaled linear subproblem audit. All unqueried/changed features and nonlinear curvature remain unqualified.'}
