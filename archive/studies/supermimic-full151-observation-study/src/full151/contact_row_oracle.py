"""Pure all-witness native-distance row algebra; no native query or solver.

The caller supplies one derivative for every actually generated native contact.
This module cannot infer those derivatives from saved normals or distances and
cannot certify that a caller supplied a complete native contact inventory.
"""
import copy
import numpy as np
from scipy import sparse

NFREE_PER_JOINT=151
NJOINT=20
NVARIABLES=3020
TARGET_DEPTH_M=.002999

def distance_jacobian_from_point_jacobians(normal_a_to_b_w,J_A,J_B,robot_columns):
    """ddist/dq20 = n_A_to_B @ (J_B(point)-J_A(point)).

    Both point Jacobians must be evaluated at the same current native contact.pos
    on the corresponding bodies. Native geom order and its normal must be kept
    together. Object columns disappear only because robot_columns omit its free
    DOFs, not because this helper silently removes a body contribution.
    """
    n=np.asarray(normal_a_to_b_w,np.float64);a=np.asarray(J_A,np.float64);b=np.asarray(J_B,np.float64);cols=np.asarray(robot_columns)
    if n.shape!=(3,) or a.ndim!=2 or a.shape[0]!=3 or a.shape!=b.shape:raise ValueError('Normal/point-Jacobian shape differs')
    if not np.isfinite(n).all() or not np.isfinite(a).all() or not np.isfinite(b).all() or abs(np.linalg.norm(n)-1)>1e-6:raise ValueError('Finite point Jacobians and unit native normal required')
    if cols.shape!=(20,) or cols.dtype.kind not in 'iu' or len(set(cols.tolist()))!=20 or np.any(cols<0) or np.any(cols>=a.shape[1]):raise ValueError('Explicit unique20 robot velocity columns required')
    return n@(b[:,cols]-a[:,cols])

def geometry_rows(Bfree,sample_index,contactdist_m,Jdist20,target_depth_m=TARGET_DEPTH_M,*,witness_ids=None,source_fields=None):
    """Return raw-metre step rows A_step*delta_free <= rhs_m.

    Bfree = position basis B601[:,2:] after fixed P0/P1 removal. Variables are
    joint-major: flat[j*151+c] is free coefficient P[c+2,j].
    A_step[r,j*151+c] = -Jdist20[r,j]*Bfree[sample_index[r],c].
    rhs_m[r] = contactdist_m[r] + target_depth_m.

    There is no implicit /1mm scaling, contact attraction, selected-min reduction,
    tie averaging or Jacobian fabrication. Empty rows carry no clearance claim.
    """
    B=np.asarray(Bfree,np.float64);si=np.asarray(sample_index);d=np.asarray(contactdist_m,np.float64)
    if Jdist20 is None:raise ValueError('Actual witness distance Jacobians are missing; saved normals/distances cannot substitute')
    J=np.asarray(Jdist20,np.float64)
    if B.ndim!=2 or B.shape[1]!=151 or not B.shape[0] or not np.isfinite(B).all():raise ValueError('Finite position basis with151 free columns required')
    if np.any(B < -1e-12) or np.any(B.sum(1)>1+1e-12):raise ValueError('Position B-spline free basis violates convex partition bounds')
    if si.ndim!=1 or si.dtype.kind not in 'iu' or np.any(si<0) or np.any(si>=B.shape[0]):raise ValueError('Integer in-range sample indices required')
    m=len(si)
    if d.shape!=(m,) or J.shape!=(m,20) or not np.isfinite(d).all() or not np.isfinite(J).all():raise ValueError('Finite M distances and actual Mx20 witness Jacobians required')
    if isinstance(target_depth_m,(bool,np.bool_)) or not np.isfinite(target_depth_m) or float(target_depth_m)!=TARGET_DEPTH_M:raise ValueError('Frozen planning depth2.999mm required')
    if witness_ids is not None and len(witness_ids)!=m:raise ValueError('Witness IDs must retain every input row')
    row=[];col=[];data=[]
    for r in range(m):
        nonzero=np.flatnonzero(B[si[r]]!=0)
        for j in np.flatnonzero(J[r]!=0):
            row.extend([r]*len(nonzero));col.extend((int(j)*151+nonzero).tolist());data.extend((-J[r,j]*B[si[r],nonzero]).tolist())
    A=sparse.csr_matrix((data,(row,col)),shape=(m,NVARIABLES));rhs=d+float(target_depth_m)
    A.sort_indices()
    zero=np.diff(A.indptr)==0
    return {'A_step':A,'rhs_m':rhs,'current_violation_m':np.maximum(-rhs,0),
            'column_order':'joint-major:j*151+c','target_depth_m':TARGET_DEPTH_M,
            'sample_index':si.copy(),'contactdist_m':d.copy(),'distance_jacobian_q_m_per_rad':J.copy(),
            'zero_control_jacobian':zero,'fixed_sample_no_free_position_support':np.all(B[si]==0,axis=1),
            'witness_ids':copy.deepcopy(witness_ids),'source_fields':copy.deepcopy(source_fields),
            'metadata':{'schema':'full151_native_all_witness_step_rows:1','inequality':'A_step * delta_free <= rhs_m','units':{'A_step':'metre per radian coefficient','rhs_m':'metre'},
                        'variables':'joint-major j*151+c, c=P2..P152; P0/P1 excluded','nvariables':3020,'rows':m,'target_depth_m':TARGET_DEPTH_M,
                        'no_implicit_millimetre_scaling':True,'each_generated_witness_retained':True,'selected_min_or_tie_reduction':False,
                        'empty_rows_semantics':'No generated witness rows is not positive clearance or a no-contact certificate. Absent catalogue gaps remain UNKNOWN outside this algebra helper.',
                        'source_completeness_verified_by_helper':False,'native_or_physical_derivative_qualified':False,
                        'qualification':'PENDING_SOURCE_BOUND_NATIVE_BRANCH_PROBE; finite or zero rows are not a derivative certificate.'}}
