"""Saved-array/static-source review only; no fitting or source-module imports."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np

OUT=Path(__file__).resolve().parent
if (OUT/'report.json').exists():
    raise FileExistsError('Review output is immutable; use a new version')
W=Path('D:/workspace')
R=W/'supermimic-observation-restoration-prefix-study'
P=W/'supermimic-observation-prefix-study'
I=W/'supermimic-observation-contact-intent-study'
pins={
 'bspline_source':W/'supermimic-observation-constrained-prefix-study/src/bspline_control.py',
 'fp32_control_source':P/'src/prefix_control.py',
 'object_curve_source':P/'src/prefix_curves.py',
 'restored_export_source':R/'scripts/export_restored_prefix_cpu.py',
 'restoration_math_source':R/'src/restoration_prefix/math.py',
 'restored_reference':R/'outputs/restored_prefix_control_export_v1/reference_bspline.npz',
 'restored_init':R/'outputs/restored_prefix_control_export_v1/init_requested.npz',
 'restored_control':R/'outputs/restored_prefix_control_export_v1/planned_control.npz',
 'restored_export_receipt':R/'outputs/restored_prefix_control_export_v1/receipt.json',
 'rough_bundle':W/'supermimic-rough-prefix-learning-study/outputs/input_bundle_v1/bundle.json',
 'full151_geometry':W/'supermimic-observation-retarget-study/outputs/native_contact_refined_geometry_v1/geometric_retarget.npz',
 'full151_geometry_report':W/'supermimic-observation-retarget-study/outputs/native_contact_refined_geometry_v1/report.json',
 'full151_raw_object_FD':W/'supermimic-observation-dynamic-wrench-study/outputs/dynamic_wrench_v2/designed_wrench_151.npz',
 'H123_candidate':I/'outputs/frame0_h123_seeded_baseline_v3/initialization_candidate.npz',
 'H123_PDU_design':I/'outputs/h123_preload_cpu_v1/baseline/designed_target_u0.json',
 'H123_PDU_saved_matrices':I/'outputs/h123_preload_cpu_v1/baseline/static_matrices.npz',
 'actual_context':W/'supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz',
 'nominal_model':W/'supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb',
 'frozen_pure_C2':W/'supermimic-observation-warmseed-grounding-study/vendor/target_inset.py',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_pins={k:{'path':str(p),'sha256':sha(p)} for k,p in pins.items()}
def load(k):
    with np.load(pins[k],allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
ref,init,control,full,raw,ctx,static=[load(k) for k in ('restored_reference','restored_init','restored_control','full151_geometry','full151_raw_object_FD','actual_context','H123_PDU_saved_matrices')]
design=json.loads(pins['H123_PDU_design'].read_text('utf-8'))
oldreceipt=json.loads(pins['restored_export_receipt'].read_text('utf-8'))
fullreport=json.loads(pins['full151_geometry_report'].read_text('utf-8'))
bundle=json.loads(pins['rough_bundle'].read_text('utf-8'))
q0=ref['q_goal_rad'][0]
v0=ref['analytic_qd16_rad_s'][0]
u0=init['u_planned0_FP32_rad']
tau=control['control__fp32_old_export__tau_ff_static0_FP32_Nm']
kp=ctx['actuator_gainprm'][0,:,0]
bias=ctx['actuator_biasprm'][0,:,:3]
lo,hi=full['joint_lower_rad'],full['joint_upper_rad']
outerlo=np.asarray(oldreceipt['PD_model']['outer_lower64'])
outerhi=np.asarray(oldreceipt['PD_model']['outer_upper64'])
innerlo=outerlo.astype(np.float32)+np.float32(.020)
innerhi=outerhi.astype(np.float32)-np.float32(.020)
p1=ref['control_points_rad'][1].copy()
p1_formula=q0+(.02/3)*v0
v0_reconstructed=(3/.02)*(p1-q0)
q32,v32=q0.astype(np.float32),v0.astype(np.float32)
proposal32=(((tau-bias[:,0])-bias[:,1]*q32)-bias[:,2]*v32)/kp
c2_32=np.minimum(np.maximum(proposal32,innerlo),innerhi)
u64=(tau.astype(float)-bias[:,0].astype(float)-bias[:,1].astype(float)*q0-bias[:,2].astype(float)*v0)/kp.astype(float)
force64=kp.astype(float)*u0.astype(float)+bias[:,0].astype(float)+bias[:,1].astype(float)*q0+bias[:,2].astype(float)*v0
flo=np.maximum(static['actuator_force_bounds'][:,0],static['joint_force_bounds'][:,0])
fhi=np.minimum(static['actuator_force_bounds'][:,1],static['joint_force_bounds'][:,1])
checks={
 'current_reference16_shape':ref['q_goal_rad'].shape==(16,20),
 'derived_geometry151_shape':full['joint_pos_rad'].shape==(151,20),
 'full_time_0_to3_uniform20ms':bool(np.max(np.abs(full['time_s']-np.arange(151)*.02))<1e-14),
 'old_q0_exact_H123_design_source64':bool(np.array_equal(q0,np.asarray(design['q0_rad']))),
 'old_q0_cast_exact_requested_FP32':bool(np.array_equal(q32,init['q_init_requested_FP32_rad'])),
 'old_analytic_qd0_cast_exact_requested_FP32':bool(np.array_equal(v32,init['qd_init_requested_FP32_rad_s'])),
 'P1_reconstructed_qd0_cast_exact_requested_FP32':bool(np.array_equal(v0_reconstructed.astype(np.float32),v32)),
 'current_u0_exact_old_control_first':bool(np.array_equal(u0,control['control__fp32_old_export__tracking_u_exec_15_FP32_rad'][0])),
 'old_actual_FP32_proposal_recomputed_bitexact':bool(np.array_equal(proposal32.view(np.uint32),control['control__fp32_old_export__u_pre_16_FP32_rad'][0].view(np.uint32))),
 'old_actual_FP32_C2_recomputed_bitexact_u0':bool(np.array_equal(c2_32.view(np.uint32),u0.view(np.uint32))),
 'old_u0_requires_no_C2_clipping':bool(np.array_equal(proposal32.view(np.uint32),c2_32.view(np.uint32))),
 'old_u0_inside_C2_planning_domain_2e6':bool(np.all(u0.astype(float)>=innerlo.astype(float)+2e-6) and np.all(u0.astype(float)<=innerhi.astype(float)-2e-6)),
 'old_t0_raw_reference_force_inside_tightened_caps':bool(np.all(force64>=flo+2e-6) and np.all(force64<=fhi-2e-6)),
 'P0_hardlegal_source64_not_clipped':bool(np.all(q0>=lo-1e-12) and np.all(q0<=hi+1e-12)),
 'fixed_P1_hardlegal':bool(np.all(p1>=lo) and np.all(p1<=hi)),
 'derived151_knots_hardlegal':bool(np.all(full['joint_pos_rad']>=lo) and np.all(full['joint_pos_rad']<=hi)),
 'derived539d_q0_different_from_H123_q0':bool(not np.array_equal(q0,full['joint_pos_rad'][0])),
 'object_pose_prefix16_exact_full539d':bool(np.array_equal(ref['mapped_object_pose_w'],full['mapped_object_pose_w'][:16])),
 'rawFD_object_pose151_exact_derived539d':bool(np.array_equal(raw['object_pose_mapped_w'],full['mapped_object_pose_w'])),
 'actual_bias1_exact_negative_kp':bool(np.array_equal(bias[:,1],-kp)),
 'actual_bias0_zero':bool(not np.any(bias[:,0])),
 'actual_bias_velocity_all_nonzero':bool(np.all(bias[:,2]!=0)),
 'saved_model_actuation_checks_all_true':bool(all(oldreceipt['PD_model']['model_checks'].values())),
 'saved_actuator_order0through19':oldreceipt['PD_model']['actuator_ids']==list(range(20)),
 'source_snapshot_hashes_unchanged_after_reads':all(sha(p)==source_pins[k]['sha256'] for k,p in pins.items()),
 'no_Torch_Warp_MuJoCo_SciPy_imported':not any(n in sys.modules for n in ('torch','warp','mujoco','scipy')),
}
diagnostics={
 'q0_vs_full539d_q0_max_abs_rad':float(np.abs(q0-full['joint_pos_rad'][0]).max()),
 'q0_to_derived_frame1_implicit_20ms_max_rate_rad_s':float(np.abs(full['joint_pos_rad'][1]-q0).max()/.02),
 'analytic_qd0_max_abs_rad_s':float(np.abs(v0).max()),
 'analytic_qd0_to_requested_FP32_max_abs_rad_s':float(np.abs(v0-v32.astype(float)).max()),
 'fixed_P1_min_hard_margin_rad':float(np.minimum(p1-lo,hi-p1).min()),
 'source_P1_vs_reconstructed_formula_max_abs_rad':float(np.abs(p1-p1_formula).max()),
 'P1_reconstructed_qd0_max_abs_error_rad_s':float(np.abs(v0_reconstructed-v0).max()),
 'current_u0_vs_static_PDU_max_abs_rad':float(np.abs(u0.astype(float)-np.asarray(design['u0_rad'])).max()),
 'exact_real_affine_u0_vs_fixed_FP32_u0_max_abs_rad':float(np.abs(u64-u0.astype(float)).max()),
 'fixed_FP32_u0_raw_real_force_vs_static_tau_max_abs_Nm':float(np.abs(force64-tau.astype(float)).max()),
 'u0_min_FP32_inset_margin_rad':float(np.minimum(u0-innerlo,innerhi-u0).min()),
 'actual_kd_over_kp_min_max_s':[float((-bias[:,2]/kp).min()),float((-bias[:,2]/kp).max())],
 'actual_kp_min_max':[float(kp.min()),float(kp.max())],
 'source64_q0_lower_boundary_joint_indices':np.flatnonzero(np.abs(q0-lo)<=1e-12).tolist(),
 'source64_q0_upper_boundary_joint_indices':np.flatnonzero(np.abs(q0-hi)<=1e-12).tolist(),
 'derived_geometry_unconverged_frames_retained':fullreport['optimizer_unconverged_frames'],
 'derived_geometry_saved_HO_max_depth_m':fullreport['native_contact_summaries']['hand_object']['after']['max_generated_depth_m'],
 'derived_geometry_saved_self_max_depth_m':fullreport['native_contact_summaries']['hand_self']['after']['max_generated_depth_m'],
}
api={}
for k in ('bspline_source','fp32_control_source','object_curve_source','restored_export_source','restoration_math_source'):
    tree=ast.parse(pins[k].read_text('utf-8'))
    api[k]={n.name:{'line':n.lineno,'end_line':n.end_lineno} for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
np.savez_compressed(OUT/'anchor_linear_diagnostics.npz',q0_source64=q0,analytic_qd0_source64=v0,fixed_u0_FP32=u0,static_u_PDU_FP32=np.asarray(design['u0_rad'],np.float32),static_tau_FP32=tau,required_P1_source64=p1,P1_reconstructed_qd0=v0_reconstructed,actual_kp_FP32=kp,actual_bias_FP32=bias,hard_lower64=lo,hard_upper64=hi,C2_inner_lower_FP32=innerlo,C2_inner_upper_FP32=innerhi,source32_proposal=proposal32,source32_postC2=c2_32,t0_real_affine_rawforce=force64)
report={
 'schema':'full151_continuous_control_interface_readonly_review:1',
 'status':'READ_ONLY_REVIEW_COMPLETE_NOT_SOLVED_NOT_QUALIFIED',
 'source_pins':source_pins,'review_source':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
 'checks':checks,'passed':sum(checks.values()),'total':len(checks),'diagnostics':diagnostics,'API_symbols':api,
 'array_receipt':{k:{n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in z.items()} for k,z in [('reference16',ref),('init',init),('control16',control),('derived151',full),('raw_object_FD151',raw)]},
 'proposed_construction':{
  'status':'REVIEW_ONLY_PENDING_ROOT_PROTOCOL_NO_PRODUCTION_IMPLEMENTATION',
  'basis':{'degree':3,'breaks':'arange(151)*.02','knots':'0 repeated4; .02..2.98 simple;3 repeated4','ncontrol':153,'fixed_P0':'exact H123 source64 q0','fixed_P1':'exact saved restored reference control_points_rad[1]; algebraically q0+.02*analytic_qd0/3, not reconstructed to avoid rounding change','free_per_joint':151,'free_all20':3020},
  'objective':'Per joint mean_i0..150 abs(q(t_i)-derived539d_q[i]) + 4*mean_r0..150 abs(P[r+2]-2P[r+1]+P[r]); first fit residual is constant because P0 is fixed; no frame dropped.',
  'regularizer_interpretation':'4=(.04/.02)^2 is a declared coefficient-difference smoothing weight, not an exact integral of qdd squared/absolute or hardware acceleration limit. Clamped endpoint physical derivatives have nonuniform knot factors.',
  'LP':'20 separable single HiGHS L1 fits; each has151 free controls+151 fit epigraph+151 secondDifference epigraph=453 variables. No parameter search, fallback, random start, tie-break or geometry solve. Parent freezes time budget before run.',
  'target_formula':'u[0]=fixed existing FP32 u0 as real constant. For k=1..149, u[k]=(tau_ff_static0-b0-b1*q(t_k)-b2*qdot(t_k))/kp; b1=-kp,b2=-kd. No u150 execution.',
  'planning_constraints':['all153 controlpoints hard joint boxes; fixedP0/P1 must first pass','all150 targets inside actual FP32 C2 inner bounds tightened2e-6rad','149 consecutive updates abs(u[k]-u[k-1])<=.02*(15-1e-4)rad for k1..149; explicitly includes u1-fixedu0','all150 intervals four exact-real affine raw-PD Bernstein coefficients inside intersection(actuator,jointactforce) tightened2e-6Nm; interval0 uses constant fixedu0','No hold/moving-reference counterfactual force rows in this new tracking-only stage; no invented qdot15 bound'],
  'force_polynomial':'interval k: kp*u[k]+b0+b1*q_cubic_Bezier[k,r]+b2*qd_quadratic_degreeElevatedToCubic[k,r], r0..3; includes both closed endpoint limits of held target.',
  'initial_rounding_contract':'Do not require exact-real u(q0,qd0,tau)==fixed FP32 u0 (known existing8.081e-8rad difference). Require final grouped FP32 proposal/C2 to reproduce fixedu0 bitexact, and newcurve qd(0) cast32 to reproduce requested analytic init qd32. If not, FAIL not implicit new init.',
  'final_saved_export':['q151/analyticqd151/analyticqdd151 plus601time diagnostics; native forward FD151 separately named','150 original grouped FP32 proposals/postC2 targets; q32/qd32 consumed exactly as frozen primitive; expected noC2change or explicitly fail qualification','149 target rates recomputed in FP64 from final U32; startup initctrl-to-U0 exactly0 with bitidentity required','all600 interval Bernstein coefficients recomputed using final postC2 U32 as exact reals; separate601 FP32 affine/difference grouping reference-force checks','q0/qd0/u0/objectpose/worldtwist/mocap/sourceidentity pinned; no repeatedprefix or frame splice'],
  'sufficient_not_necessary':True,'physical_reference_qualified':False,'training_authorized':False,
 },
 'scope':{
  'lineage':'existing legacy controlled observation source frames[200,351); not shared007[0,151), not independent source or robot joint label',
  'derived_q151':'robot-specific geometric inferred goals from539d; legal knots and old selected-contact depth data do not certify the new interpolated/smoothed curve',
  'baseline_u150':'planned feedforward absolute targets derived from goal q/analyticqd/static tau; neither actualq nor measuredctrl/force',
  'actual_state':'only future independent physics can provide q/qd/object/contacts/actuator force and original13 physical gates',
  'object_curve':'reuse frozen per-segment quintic/SO3 primitive with full151 frozen60b1 raw v/a/worldomega/worldalpha; no re-FD prefix or joint-source substitution',
  'unknown':'Global LP feasibility, fullcurve geometry/contact intent/force realization/dynamics/WM gradient/generalization remain untested.',
  'forbidden_counts':{'optimization_calls':0,'LP_calls':0,'FK_calls':0,'collision_calls':0,'Env_constructions':0,'physics_steps':0,'WM_forward':0,'GPU_calls':0},
  'module_imports':'stdlib+NumPy only; source AST/hash and saved-array ordinary algebra; nominalMJB only byte-hashed, not instantiated',
 },
 'artifacts':{'anchor_diagnostics':{'path':str(OUT/'anchor_linear_diagnostics.npz'),'sha256':sha(OUT/'anchor_linear_diagnostics.npz')}},
}
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n','utf-8')
print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':str(OUT/'report.json'),'report_sha256':sha(OUT/'report.json'),'diagnostics':diagnostics},indent=2))
