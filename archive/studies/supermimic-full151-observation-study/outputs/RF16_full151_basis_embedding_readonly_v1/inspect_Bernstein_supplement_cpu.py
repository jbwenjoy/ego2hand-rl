"""Saved-prefix range-certificate algebra, no new candidate/solver/model."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.interpolate import BSpline

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
OUT=Path(__file__).resolve().parent
def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
paths={
    'basis_review':OUT/'report.json','operators':OUT/'restriction_operators_only.npz',
    'RF16':Path('D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restored_prefix_control_export_v1/reference_bspline.npz'),
    'full_knots':ROOT/'outputs/full151_contact_coupled_scp_v1/reference_candidate151.npz',
    'hard_limits':ROOT/'outputs/full151_cpu_input_v1/actual_PD_parameters.npz',
}
before={k:pin(p) for k,p in paths.items()}
o=load(paths['operators'],['RF_knots_s','full_knots_s','algebraic_endpoint_map18'])
r=load(paths['RF16'],['control_points_rad']);limits=load(paths['hard_limits'],['q_lower_rad','q_upper_rad'])
T=o['algebraic_endpoint_map18'];R=r['control_points_rad'];E=T@R
bp=BSpline(o['RF_knots_s'],np.eye(18),3,extrapolate=False)
bf=BSpline(o['full_knots_s'],np.eye(153),3,extrapolate=False)
t=np.arange(16)*.02;h=np.diff(t)[:,None]
def bezier(spline,n):
    L=spline(t[:-1])[:,:n];V=spline(t[1:])[:,:n]
    DL=spline(t[:-1],nu=1)[:,:n];DV=spline(t[1:],nu=1)[:,:n]
    return np.stack((L,L+h*DL/3,V-h*DV/3,V),axis=1)
B_old=bezier(bp,18);B_full=bezier(bf,18)
old=R[0]+np.einsum('ikc,cj->ikj',B_old[:,:,1:],R[1:]-R[0])
embedded=E[0]+np.einsum('ikc,cj->ikj',B_full[:,:,1:],E[1:]-E[0])
lo,hi=limits['q_lower_rad'],limits['q_upper_rad']
oldex=np.maximum(np.maximum(lo-old,old-hi),0.)
newex=np.maximum(np.maximum(lo-embedded,embedded-hi),0.)
cp_ex=np.maximum(np.maximum(lo-E,E-hi),0.)
report={
    'schema':'RF16_Bernstein_restriction_range_algebra_supplement:1',
    'status':'COMPLETED_READONLY_STRUCTURE_DIAGNOSTIC_NO_REQUALIFICATION',
    'source_pins':before,
    'facts':{'intervals':15,'range_coefficients_per_interval':4,
        'restricted_basis_Bezier_operator_error_max':float(np.max(np.abs(B_full@T-B_old))),
        'RF_vs_embedded_prefix_Bezier_value_error_rad':float(np.max(np.abs(embedded-old))),
        'old_RF_prefix_Bezier_hard_excess_rad':float(oldex.max()),
        'embedded_restriction_Bezier_hard_excess_rad':float(newex.max()),
        'embedded_global_controlpoint_hard_excess_rad':float(cp_ex.max()),
        'original_joint_limits_preserved':True},
    'interpretation':'An identical prefix polynomial can satisfy the per-interval Bernstein range sufficient certificate even when its endpoint-unclamped full-basis controlpoints violate the stronger globalCP box. CP violation is not q(t) violation or physical impossibility. Replacing certificates is a new formulation requiring new protocol, solver and independentpostaudit; it does not waive oldgate or release a candidate.',
    'proposal_only':'Future singlemethod can replace CPbox by per-interval4cubicBernstein bounds (or an independently qualified stationary-root range audit) at the same original physical joint limits. No threshold relaxation. Wholefull151 tail, C2 join, controls and geometry remain unspecified/unqualified.',
    'scope':{'saved_array_BSpline_evaluation_and_products_only':True,'newP153_or_U150_candidate_constructed_or_saved':False,'solve_LP_optimizer':False,'model_FK_query_Env_WM_physics_GPU':False,'framework_imported':any(x in sys.modules for x in ('mujoco','torch','warp')),'old_sources_written':False},
    'review_source':pin(__file__),'source_unchanged_pre_post':before=={k:pin(p) for k,p in paths.items()},
}
target=OUT/'Bernstein_range_supplement.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(target),'facts':report['facts']}))
