"""Static producer/protocol review; no studied source import or array load."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
OUT = Path(__file__).resolve().parent

def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def all_pins(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}: yield value
        else:
            for v in value.values(): yield from all_pins(v)
    elif isinstance(value,list):
        for v in value: yield from all_pins(v)

producer=ROOT/'scripts/build_full151_input_cpu_v2.py'
protocol_path=ROOT/'protocols/full151-input-cpu-v2.json'
source=producer.read_text(encoding='utf-8-sig')
proto=json.loads(protocol_path.read_text(encoding='utf-8-sig'))
tree=ast.parse(source)
old_source=(ROOT/'scripts/build_full151_input_cpu_v1.py').read_text(encoding='utf-8-sig')
restored_source=source.replace('import argparse,datetime,hashlib,importlib.util,json,os,platform,sys,time,traceback','import argparse,datetime,hashlib,importlib.util,json,platform,sys,time,traceback').replace(",\'CUDA_VISIBLE_DEVICES\':os.environ.get('CUDA_VISIBLE_DEVICES')",'').replace("            report['cost']['LP_calls']+=1  # count attempts even if the backend raises\n",'').replace('            total_solver+=elapsed\n',"            total_solver+=elapsed;report['cost']['LP_calls']+=1\n")
old_proto=json.loads((ROOT/'protocols/full151-input-cpu-v1.json').read_text(encoding='utf-8-sig'))
restored_proto=json.loads(json.dumps(proto));restored_proto.pop('prelaunch_revision');restored_proto['sources']['producer']=old_proto['sources']['producer']
pins=list(all_pins(proto))
checked=[{'declared':p,'actual':pin(p['path']),'matches':p['sha256']==pin(p['path'])['sha256']} for p in pins]
checks={
    'v2_source_only_count_attempt_and_visibility_metadata_delta':restored_source==old_source,
    'v2_protocol_only_producer_route_and_prelaunch_note_delta':restored_proto==old_proto,
    'producer_protocol_source_identity_current':proto['sources']['producer']==pin(producer),
    'all_protocol23_source_and_input_pin_occurrences_current':len(checked)==23 and all(x['matches'] for x in checked),
    'protocol_legacy151_not_prefix_or_shared':proto['source_identity']['legacy_source_frames_half_open']==[200,351] and proto['source_identity']['local_states']==151 and proto['source_identity']['planned_targets']==150 and proto['source_identity']['shared_source_0_151'] is False,
    'full20LP_5s_no_retry_noGPUphysics_training':proto['execution']=={'joint_LP_calls':20,'per_joint_time_limit_s':5.,'retries':0,'lambda_coefficient':4.,'GPU':False,'physics':False,'training':False},
    'fresh_output_no_overwrite_before_execution':"out.mkdir(parents=True,exist_ok=False)" in source,
    'all_pins_before_import_scientific_execution':source.index('for p in all_pins(protocol):checked(p)')<source.index('import scipy'),
    'all_data_and_object_checks_block_before_LP':source.index("raise ValueError('Pre-solve input gate failure; no LP permitted')")<source.index('result=linprog') and source.index("raise ValueError('Object reference gate failure; no LP permitted')")<source.index('result=linprog'),
    'P0_P1_exact_saved_controls_not_reconstructed':"P0=anchor['control_points_rad'][0].copy();P1=anchor['control_points_rad'][1].copy()" in source,
    'initial_q_and_qd_saved_FP32_exact_checked_before_LP':"exact(P0.astype(np.float32),init['q_init_requested_FP32_rad'])" in source and "exact((150*(P1-P0)).astype(np.float32),init['qd_init_requested_FP32_rad_s'])" in source,
    'initial_u0_and_old_FP32_proposal_verified':"checks['first_target_saved_exact']" in source and "checks['initial_proposal_saved_exact']" in source,
    'initial_object_pose_worldvelocity_bodyangular_checked_not_replaced':all(s in source for s in ("checks['object_initial_pose_exact']", "checks['object_initial_velocity_exact']", "checks['object_initial_world_omega_exact']", "checks['object_initial_body_omega_exact']", "np.savez_compressed(out/'init_requested.npz',**init)")),
    'zeroCOM_and_model_jointdomains_bound':"checks['zero_COM_offset']" in source and "checks['hard_joint_bounds_from_actual_model']" in source,
    'new_object_derivatives_and_wholeprefix_same_law_checked':"objerr[k]<=tol[k]" in source and "prefixerr[k]<=tol[k]" in source,
    'LP_single_source_call_in_fixed20loop':sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='linprog' for n in ast.walk(tree))==1 and 'for j in range(20):' in source,
    'perjoint_raw_matrix_and_result_saved':all(s in source for s in ("'_input.npz'", "'_result.npz'", 'raw_Ax_minus_b=residual', "row['arithmetic_1e12_gate']")),
    'raw_math_gate_separate_from_solver_success':"checks['all20_solver_success']" in source and "checks['all20_raw_arithmetic_1e12']" in source,
    'failed_raw_primal_retained_not_repaired':"if result.success and np.isfinite(x).all():free[:,j]=x[:151]" in source and "checks.update(export['gates'])" in source,
    'actual_760_FP32_Torch_and_original_C2_compared':all(s in source for s in ("checks['CPU_Torch_proposal150_bitexact']", "checks['CPU_Torch_C2_150_bitexact']", "checks['original_C2_150_bitexact']")),
    'final_init_qd_and_P0P1_rechecked':"checks['analytic_init_qd_FP32_exact']" in source and "checks['same_P0_P1_source64']" in source,
    'LPfailure_exception_never_READY':"report['status']='CPU_INPUT_EXCEPTION_PRESERVED'" in source and "if report['status']=='CPU_INPUT_GATES_PASS_PHYSICS_PENDING':" in source,
    'all_sources_checked_again_before_success':"checks['all_input_source_pins_unchanged']" in source,
    'READY_denies_physics_geometry_target_training':all(s in source for s in ("'physics_qualified':False", "'geometry_qualified':False", "'contact_target_ready':False", "'training_authorized_by_this_receipt':False")),
    'no_FK_collision_Env_WM_or_training_call_sites':not any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in ('mj_forward','mj_kinematics','mj_step','ManagerBasedRlEnv','WorldModel')) or (isinstance(n.func,ast.Attribute) and n.func.attr in ('mj_forward','mj_kinematics','mj_step','mj_collision','mj_geomDistance','learn','step'))) for n in ast.walk(tree)),
}
report={
    'schema':'full151_input_producer_source_independent_review:1',
    'status':'STATIC_NO_SCIENTIFIC_BLOCKING_IMPLEMENTATION_NOT_EXECUTED',
    'producer':pin(producer),'protocol':pin(protocol_path),
    'module_review':pin(OUT/'modules_review.json'),
    'checks':checks,'all_checks_pass':all(checks.values()),'checked_protocol_pins':checked,
    'no_blocking_scope':'Source/metadata review only. Not CPU solver execution, output feasibility, native ABI, reset, physical or training qualification.',
    'observations':[{'id':'unused_source_wrist_value','severity':'nonblocking_source_minimization','present': 'wrist_pose_source_w' in proto['selected_fields']['observations'], 'note':'Selected unused wrist_pose_source_w is loaded, never used for mapping/control. Remove before execution if asserting no source-wrist values read; do not falsely describe current allowlist as only3 arrays.'},
                    {'id':'solver_invocation_count_exception','severity':'CLOSED_BEFORE_EXECUTION_IN_V2','present':source.index("report['cost']['LP_calls']+=1")>source.index('result=linprog'), 'note':'V2 increments before backend call, preserving attempted cost even if solver raises. V1 source/protocol retained and never executed. Numerical formulas unchanged.'}],
    'execution_scope':{'source_JSON_AST_hash_only':True,'production_modules_imported':False,'numpy_scipy_Torch_Warp_MuJoCo_imported':any(x in sys.modules for x in ('numpy','scipy','torch','warp','mujoco')),'NPZ_arrays_loaded':False,'LP_FK_Env_physics_WM_GPU':False,'source_modified':False},
    'review_source':pin(__file__),
}
report['sourcepins_unchanged_after_review']=all(p['sha256']==pin(p['path'])['sha256'] for p in pins)
target=OUT/'producer_v2_review.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(target),'all_checks_pass':report['all_checks_pass'],'checks':len(checks),'sources':len(checked)}))
if not report['all_checks_pass'] or not report['sourcepins_unchanged_after_review']:raise SystemExit(2)
