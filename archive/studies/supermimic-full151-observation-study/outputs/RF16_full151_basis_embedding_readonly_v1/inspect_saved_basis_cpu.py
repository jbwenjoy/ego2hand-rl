"""Saved-basis algebra only. No model, query, solve, trajectory release or u."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.interpolate import BSpline
import scipy

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
OUT=Path(__file__).resolve().parent
paths={
    'old_RF16':Path('D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restored_prefix_control_export_v1/reference_bspline.npz'),
    'full151_terminal':ROOT/'outputs/full151_contact_coupled_scp_v1/reference_candidate151.npz',
    'saved_hard_limits':ROOT/'outputs/full151_cpu_input_v1/actual_PD_parameters.npz',
    'old_basis_source':Path('D:/workspace/supermimic-observation-constrained-prefix-study/src/bspline_control.py'),
    'full_basis_source':ROOT/'src/full151/continuous_control.py',
}
def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

before={k:pin(p) for k,p in paths.items()}
rf=load(paths['old_RF16'],['control_points_rad','Bspline_knots_s','q_goal_rad','analytic_qd16_rad_s','analytic_qdd16_rad_s2'])
full=load(paths['full151_terminal'],['control_points_rad','Bspline_knots_s'])
bounds=load(paths['saved_hard_limits'],['q_lower_rad','q_upper_rad'])
pk,fk=rf['Bspline_knots_s'],full['Bspline_knots_s']
R=rf['control_points_rad'];P=full['control_points_rad']
bp=BSpline(pk,np.eye(18),3,extrapolate=False)
bf=BSpline(fk,np.eye(153),3,extrapolate=False)
t=np.arange(301,dtype=float)*.001;t[-1]=.3
knots=np.arange(16,dtype=float)*.02
# Algebraic endpoint-unclamping restriction map only. No full153 candidate is
# created: P18..152 and t>.3 are intentionally undefined by this diagnostic.
T=np.eye(18)
T[16]=0.;T[16,16]=1.5;T[16,15]=-.5
T[17]=0.;T[17,17]=6.;T[17,16]=-6.;T[17,15]=1.
E=T@R
op_errors={};curve_errors={};naive_errors={};first_operator_diff={}
for order,label in enumerate(('q','qd','qdd')):
    A=bp(t,nu=order);B=bf(t,nu=order)[:,:18]
    op_errors[label]=float(np.max(np.abs(B@T-A)))
    old=A[:,1:]@(R[1:]-R[0]);naive=B[:,1:]@(R[1:]-R[0]);embedded=B[:,1:]@(E[1:]-E[0])
    if order==0:old+=R[0];naive+=R[0];embedded+=E[0]
    curve_errors[label]=float(np.max(np.abs(embedded-old)))
    naive_errors[label]=float(np.max(np.abs(naive-old)))
    changed=np.flatnonzero(np.max(np.abs(B-A),axis=1)>1e-10)
    first_operator_diff[label]=None if not len(changed) else float(t[changed[0]])
old_knots=bp(knots);naive_knots=bf(knots)[:,:18]
raw_naive_q=(naive_knots[:,1:]@(R[1:]-R[0])+R[0])-(old_knots[:,1:]@(R[1:]-R[0])+R[0])
changed=np.flatnonzero(np.max(np.abs(raw_naive_q),axis=1)>1e-10)
first_changed_knot=None if not len(changed) else {'local_index':int(changed[0]),'time_s':float(knots[changed[0]])}
excess=np.maximum(np.maximum(bounds['q_lower_rad']-E,E-bounds['q_upper_rad']),0.)
outside=[{'transformed_control_index':int(i),'joint_index':int(j),'excess_rad':float(excess[i,j]),'value_rad':float(E[i,j]),'lower_rad':float(bounds['q_lower_rad'][j]),'upper_rad':float(bounds['q_upper_rad'][j])} for i,j in np.argwhere(excess>1e-12)]
first_knots_diff=np.flatnonzero(pk!=fk[:len(pk)])
qfull=bf(t)[:,1:]@(P[1:]-P[0])+P[0]
qrf=bp(t)[:,1:]@(R[1:]-R[0])+R[0]
result={
    'schema':'RF16_full151_basis_restriction_readonly:1',
    'status':'COMPLETED_SAVED_ALGEBRA_NOT_A_NEW_REFERENCE_OR_WARMSTART_RELEASE',
    'source_pins':before,'libraries':{'numpy':np.__version__,'scipy':scipy.__version__},
    'facts':{
        'old_controls':list(R.shape),'full_controls':list(P.shape),'P0_P1_current_equal_oldRF':exact(P[:2],R[:2]),
        'first_knot_vector_difference':{'zero_based_index':int(first_knots_diff[0]),'prefix_value_s':float(pk[first_knots_diff[0]]),'full_value_s':float(fk[first_knots_diff[0]])},
        'common_initial_knot_vector_indices_inclusive':[0,int(first_knots_diff[0]-1)],
        'prefix_endpoint_multiplicity':int(np.count_nonzero(pk==.3)),'full_same_time_multiplicity':int(np.count_nonzero(fk==.3)),
        'naive_copy_basis_difference_first_dense_time_by_derivative':first_operator_diff,
        'naive_copy_actual_oldcurve_first_changed_reference_knot':first_changed_knot,
        'naive_copy_max_curve_difference_0_to0p3':naive_errors,
        'exact_algebraic_tail_map':{'P0_to15':'RF0_to15','P16':'1.5*RF16-0.5*RF15','P17':'6*RF17-6*RF16+RF15','P18_to152':'UNDEFINED_NOT_CONSTRUCTED'},
        'restriction_operator_max_abs_F64_difference':op_errors,
        'oldcurve_vs_embedded_restriction_F64_max_error':curve_errors,
        'transformed_first18_hard_hull_max_excess_rad':float(excess.max()),
        'transformed_first18_violating_entries':outside,
        'transformed_first18_pass_old_full_coefficient_box':not bool(outside),
        'current_full_terminal_prefix_vs_RF16_max_q_difference_rad':float(np.max(np.abs(qfull-qrf))),
    },
    'interpretation':[
        'Copying P0/P1 does not preserve prefix motion. Copying all18 RF controls unchanged also does not preserve the last prefix spans because endpoint clamping differs.',
        'The cubic RF curve is mathematically representable as a restriction of the full basis through q/qd/qdd on[0,.3] via the displayed analytic endpoint-unclamping map. This is not tail construction or a physics certificate.',
        'If transformed coefficients violate the full hard-control-point box, exact prefix preservation is excluded by that sufficient convex-hull coefficient certificate, not proved physically impossible. Never clip transformed coefficients and still call it the same prefix.',
        'Any extension/warmstart must explicitly handle remaining controls, C2 join, wholeinterval q/control/force/rate and allfullgeometry. Do not splice sampledq, paste an incompatibleqtail or inherit oldprefix physics.',
        'A future distinct basis/join design may preserve a prefix scaffold, but changes source/ABI/certification. Native151 knownrough integration can advance independently under explicit freshengineering contracts.'
    ],
    'scope':{'only_saved_arrays_generic_BSpline_evaluation_matrix_products':True,'linalg_solve_LP_optimizer':False,'P153_or_U150_new_candidate_constructed_or_saved':False,'source_qtail_or_labels_loaded':False,'model_FK_query_Env_WM_physics_GPU':False,'framework_imported':any(x in sys.modules for x in ('mujoco','torch','warp')),'old_sources_or_data_written':False},
    'review_source':pin(__file__),
}
result['source_pins_unchanged_pre_post']=before=={k:pin(p) for k,p in paths.items()}
target=OUT/'report.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
np.savez_compressed(OUT/'restriction_operators_only.npz',RF_knots_s=pk,full_knots_s=fk,restriction_time_s=t,algebraic_endpoint_map18=T)
print(json.dumps({'report':pin(target),'facts':result['facts']}))
if not result['source_pins_unchanged_pre_post']:raise SystemExit(2)
