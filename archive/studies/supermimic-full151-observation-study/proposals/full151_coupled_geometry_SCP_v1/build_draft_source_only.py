"""Build a non-executable protocol draft and source receipt, using stdlib only."""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/full151_coupled_static_design_v1'

def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

paths={
    'original_scheme':Path('C:/Users/ASUS/Documents/Codex/2026-10-31/qin/outputs/Wuji-Hand2-SuperMimic-CHORD-Demo-实施方案.md'),
    'full151_findings':ROOT/'docs/full151_input_findings_2026-11-04.md',
    'new_mode_contract':ROOT/'docs/full151_mode_contract_draft_v1.md',
    'parent_protocol':ROOT/'protocols/full151-input-cpu-v2.json',
    'parent_failed_receipt':ROOT/'outputs/full151_cpu_input_v1/receipt.json',
    'parent_reference':ROOT/'outputs/full151_cpu_input_v1/reference_bspline151.npz',
    'parent_control':ROOT/'outputs/full151_cpu_input_v1/planned_control150.npz',
    'parent_init':ROOT/'outputs/full151_cpu_input_v1/init_requested.npz',
    'parent_fits':ROOT/'outputs/full151_cpu_input_v1/fit_solutions.npz',
    'geometry_receipt':ROOT/'outputs/full151_failed_candidate_geometry_v1/receipt.json',
    'geometry_raw_records':ROOT/'outputs/full151_failed_candidate_geometry_v1/native_contacts.jsonl',
    'geometry_raw_independent_review':ROOT/'outputs/full151_geometry_saved_independent_v1/report_v2.json',
    'frozen_joint_linear_source':ROOT/'src/full151/continuous_control.py',
    'global_SCP_builder_candidate':ROOT/'src/full151/sparse_global_scp.py',
    'all_witness_row_algebra_v2':ROOT/'src/full151/contact_row_oracle_v2.py',
    'all_witness_units_proof':ROOT/'outputs/contact_row_oracle_units_CPU_v2/report.json',
    'native_filtered_geometry_source':Path('D:/workspace/supermimic-observation-retarget-study/scripts/native_filtered_contact_geometry.py'),
    'RobotKinematics_source':Path('D:/workspace/supermimic-observation-retarget-study/scripts/retarget_palmroot_cpu.py'),
    'object_curve_source':ROOT/'src/full151/object_curves.py',
    'frozen_FP32_control':Path('D:/workspace/supermimic-observation-prefix-study/src/prefix_control.py'),
    'original_C2':Path('D:/workspace/supermimic-observation-warmseed-grounding-study/vendor/target_inset.py'),
    'local_native_J_probe_protocol':ROOT/'protocols/full151-native-witness-probe-v1.json',
    'local_native_J_probe_producer':ROOT/'scripts/probe_native_witness_cpu_v1.py',
    'new_native_witness_oracle':ROOT/'src/full151/native_witness.py',
}
before={k:pin(p) for k,p in paths.items()}
parent=read(paths['parent_failed_receipt']);geometry=read(paths['geometry_receipt'])
source={k:paths[k].read_text(encoding='utf-8-sig') for k in ('frozen_joint_linear_source','global_SCP_builder_candidate','all_witness_row_algebra_v2','native_filtered_geometry_source')}
proto={
    'schema':'full151_coupled_geometry_SCP_draft:1',
    'status':'DRAFT_NOT_AUTHORIZED_NOT_EXECUTED',
    'authorizations':{'CPU_solver_run':False,'FK_or_geometry_query':False,'Env_physics_GPU':False,'WM_training':False},
    'purpose':'One bounded full151 generated-contact-geometry-coupled reference repair attempt, not a force/dynamics teacher or full CHORD target.',
    'inputs_and_source_readiness_evidence':before,
    'source_identity':{'legacy_source_window_half_open':[200,351],'local_states':151,'duration_s':3.,'control_intervals':150,'last_real_action':149,'terminal_observation':150,'no_u150':True,'shared007_0_151_enrollment':False,'independent_ego_N':0,'source_lineage_N':1},
    'immutable_scientific_fields':{
        'qfit_target':'All151 observation-derived539d goals via the pinned parent protocol; never original sourceproprioception.',
        'fixed_source64_P0_P1':'Exact parent source control points; no clipping, reconstruction, optimization or q0 splice.',
        'requested_init':'All7 parent FP32 fields byte-identical, including nonzeroanalyticqd0, object world/bodytwist and actualu0. Geometric query qvel0/time0 is scratch state only.',
        'object_motion':'Entire existing parent mappedobject151/quintic601 position, SO3 rotations and designed velocities unchanged.',
        'joint_and_controller_contract':'Original hard-coefficient box, hard-intersect-ctrl, C2 .020 source,2e-6rad target margin,2e-6Nm force margin,14.9999rad/s planning and15 original target-rate gate.',
        'reference_PD':'All150 ZOH interval cubic Bernstein inequalities, actualgain/bias/gear1 and both actuator/joint force caps, plus final601 FP32 groupings. Sufficient reference-PD surrogate, not actualforces or wholeintervalFP32/FMA certificate.',
        'original_joint_linear_rows':'All20 pinned2402-row LPs, including fit/smooth epigraphs; no raw1e-12 tolerance relaxation.',
    },
    'geometry':{
        'sample_times':'k*.005,k=0..600; all151knots and450interior samples retained, never shorten to prefix.',
        'original_diagnostic_depth_m':.003,'planning_depth_m':.002999,'planning_margin_m':1e-6,
        'planning_arithmetic_tolerance_m':1e-12,'original3mm_gate_tolerance_m':0.,
        'scope':'Generated native catalogue hand-object and hand-self overlaps at601 samples. No continuous-clearance/positivegap/absence/force/physical validity claim.',
        'witnesses':'Every actually generated native contact, including all contacts within tied/samepair buckets. No selected-min, sign/gradient/penetration-based deletion.',
        'native_options':'OriginalMJB filtered collision options and pinnedactualgeomsize/pos; no geomDistance, contactForce, forward, step, inverse.',
        'unknown_pairs':'No generated witness => gapUNKNOWN, no clearance row. Never turn it into known human no-contact or a positiveforce/support contact.',
        'all_witness_distance_J':'n_geom0_to_geom1.T*(Jbody1(point)-Jbody0(point)), explicit robot20dof columns, then Jdist20 @ B601_free. Native order+normal must stay paired.',
        'linearized_inequality':'-Jdist20*Bfree*(Pfree_trial-Pfree_current) - eta_m <= contactdist_current + .002999',
        'fixed_init_conflict':'If fixedt0 prescribed geometry violates planning depth and no free basis support, record restricted fixed-init conflict; never alterq0/objectpose/u0.',
        'feature_and_tie_status':'Exact witnesses/feature membership may change discontinuously. Local164-query proof cannot establish a globalJacobian; actual601 audits decide primal values.',
    },
    'subproblem':{
        'type':'Elastic penalty SCP, not a pure phase-I minimumslack guarantee.',
        'physical_free_variables':3020,'epigraph_variables':6040,'elastic_eta_variables':1,'total_variables':9061,
        'layout':'20 joint blocks453: centeredP2..152(151),fit_abs151,secondDiff_abs151; finaleta_m.',
        'old_rows':48040,'additional_rows':'One per complete current native generated witness; row count variable and fully recorded.',
        'centering':'Pfree=P0+y; actual curve/control definition unchanged. Verify both centered and reconstructed originalabsolute rows.',
        'positive_row_scale':1e4,
        'objective':'F(P)+1e4rad/m*eta_m, where F=sum_j[mean151|q(P)-target|+4mean151|secondDifference(P)|]. No extra divide20.',
        'eta_nonnegative_intermediate':True,'eta_m_final_additional_gate':1e-9,
        'eta_gate_is_NOT_intermediate_trial_eligibility':True,
        'raw_primal_tolerance_original_native_row_units':1e-12,'bounds_tolerance_rad':1e-12,
        'max_LP_calls':8,'each_LP_time_limit_s':10.,'LP_method':'highs','solver_primal_dual_tolerance':1e-9,'maxiter':100000,
        'normal_terminal_required':'successTrue/status0 plus finite full9061vector and all unscaled old/new linear/trust/bound audits. Solver success alone is never accepted.',
    },
    'trust_and_acceptance':{
        'initial_and_max_radius_rad':.15,'accepted_radius_update':'unchanged','rejected_radius_multiplier':.5,'expansion':False,'line_search':False,
        'major_count_rule':'Every attempted LP, accepted or rejected, consumes one of8 major slots. Numerical/API/solver failure stops; not repeated at a smaller radius.',
        'trial_preconditions':'Normalterminal, all original/centered+elasticlinear/trust/bounds raw<=1e-12, FP32postC2/control gates pass. Do not require eta<=1e-9 at an intermediate restoration trial.',
        'actual_merit':'Phi(P)=directF(P)+1e4*max(0,max(all601actual_generateddepth_m)-.002999). DirectF usesP, not solverepigraph variables.',
        'predicted_merit':'DirectF(P_trial)+1e4*max(0,max_current_witness(-d_current-Jdistance_B_delta-.002999)); do not substitute solvereta or epigraph objective.',
        'ratio':'actual_reduction/predicted_reduction; require finite predicted_reduction>0, actual_reduction>0 and ratio>=.1.',
        'accept':'Only that ratio+strictmerit rule updates lastaccepted. Reuse raw/Jacobians only when bound to exact acceptedP hash and sameposes/options.',
        'reject':'Preserve trial arrays/geometry/rawrows and reason, keep currentP, halve radius, consume major. No alternativealpha/damping/solver/weight/seed search.',
        'no_best_history':'Final source is chronological lastaccepted, not best quality/lowest penetration/anyrejected trial.',
    },
    'finalization':{
        'normal_end':'After accepted sufficientfullplanning-feasible state or8major budget; exact fixed early-stop criterion must be sourcebound.',
        'fresh_final_grid_required':True,'fresh_final_source':'lastaccepted only; if zeroaccepted then explicitly parentFAILEDseed, not improved result.',
        'final_gates':['normal complete execution','all601freshsample generateddepth<=.002999+1e-12m','independent original3mm exact diagnostic','allold and new rawlinear/bounds<=1e-12 nativeunits','lastacceptedLPeta>=0 and<=1e-9m','unchangedP0/P1+all7initfields','all original postC2 FP32/control/wholeintervalPD gates','source/input/options hash unchanged'],
        'interruption_or_API_failure':'Partialarrays+lastaccepted provenance only, no canonicalREADY or inferredgeometryqualification. Do not launch extraquery/LP after deadline or retry worker.',
        'if_feasibility_gates_pass':'At most SAMPLE_GEOMETRY_AND_PLANNED_CONTROL_QUALIFIED; physical_reference_qualified=False, native151integration=False, training_authorized=False.',
        'if_any_gate_fails':'Full151failedcandidate retained; noREADY. Not globalmotioninfeasible and not evidence SuperMimic/CHORDcannotcorrectroughreference.',
    },
    'finite_cost':{
        'local_J_probe_separate_queries':164,'local_J_probe_description':'4states*(1baseline+20DOF*2signs eps1e-5rad), as the actual frozen localprobe protocol; local stablebranch only. Must receive its own sourcebound proof, no global derivativepass.',
        'baseline_grids':1,'maximum_trial_grids':8,'fresh_final_grids':1,'points_per_grid':601,'max_scheduled_SCP_collision_queries':6010,
        'max_including_separate164probe':6174,'constructor_FK_and_anyother_setup':'separately instrumented; not silently folded into6010 wholeprocess claim.',
        'proposed_worker_soft_s':220,'proposed_owned_process_hard_s':240,'wholeprocess_limit_status':'NEW_PROPOSED_ROOT_EXECUTION_PROTOCOL_MUST_CONFIRM',
        'count_attempts_before_backend':True,'no_automatic_retry':True,
    },
    'readiness_checklist':[
        {'id':'frozen_sources_inputs','status':'PENDING_FINAL_ROOT_IMPLEMENTATION','need':'Allcandidate/target/time/object/context/model/primitive/sourcepins; actual producer/supervisor not yet part of this draft.'},
        {'id':'units_interface','status':'CLOSED_BY_NEW_HELPER_V2_METADATA_PROOF_ONLY','need':'contact_row_oracle_v2 supplies top-levelunits=m;6unitcontractproof retained, no native qualification.'},
        {'id':'local_native_J164','status':'PENDING_SOURCE_BOUND_RESULT','need':'Exact native geom/order/branchmatching/absent and tie exclusions from directionalaccuracy only; save everynegative/unsupported case.'},
        {'id':'global_sparse_chain','status':'PENDING_REVIEW_PROOF','need':'3020jointmajor→9060jointblocks, centered/absolute equivalence, everywitness row retained includingzero/tie rows, exactP0/P1 zero free support att0.'},
        {'id':'merit_and_update','status':'PENDING_ROOT_DRIVER','need':'Syntheticaccept/rejectratio/nonfinite/negativepred/eta-positiveintermediate/terminaleta/lastaccepted/no-history tests without realquery.'},
        {'id':'original_controller_FP32','status':'PENDING_NEW_TRIAL_CHECKS','need':'Original760 proposal, singleC2.020, U150/149neighborupdates/startupjump0, hardintersection, bothforcecaps and initbitexact.'},
        {'id':'failure_preservation','status':'PENDING_ROOT_DRIVER','need':'EveryLP count beforecall, inputCSR/rhs/bounds/source/Jac rowinventory/trialgrid saved; finiteowneddeadline including setup; failure/noREADY states tested.'},
        {'id':'resource_caps','status':'PENDING_ROOT_DECLARE','need':'Declare finitewitness/memory/output limits beforeexecution; overflow/unknown/truncation stops as engineeringcensor, never dropsconstraints.'},
        {'id':'independent_intent','status':'NOT_PRODUCED_BY_SCP','need':'HumanlabelsUNKNOWN, currentobservationcues notloadableA2target. A future independent151intenthypothesis must be separatelyfrozen, no policycontact labels/old16tiling.'},
        {'id':'future_roughlearning','status':'SEPARATE_MODE_CONTRACT_PENDING','need':'Do not require perfect openloopdynamics as eternalgate. Independentexplicitroughmode may study correction after source/control/reset/nativeA0 engineering checks and explicitauthorization, retaining any geometry/physicalfailures. This draft does not waive currentfailed gates or authorize training.'},
    ],
    'limitations':[
        'Strongpenalty cannot guarantee eta0, nonlinear feasibility or convergence in8steps. Rejectshrink is a predeclared algorithm, not an unlogged retry/search.',
        '601sampled overlap and wholeinterval referenceforce are different quantifiers. Neither certifies the other or actual q/qd/contactforces.',
        'Hard q/target/force/rate planning restrictions are sufficient restrictedsurrogates, not necessary conditions for humanmotion or residuallearning feasibility.',
        'No contact attraction/phase support/normal force positive-event labels are produced by collisionavoidance. This is not full CHORD integration.',
        'Retain complete3s objectmotion and151 goals. No automaticprefixnarrowing, objectassistance, wristmotion, qprojection or sourceproprioception fallback.',
    ],
}
HERE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
target=HERE/'protocol.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(proto,indent=2,allow_nan=False)+'\n',encoding='utf-8')
checks={
    'parent_failed_retained_noREADY':parent['status']=='CPU_CANDIDATE_GATES_FAIL_NO_QUALIFICATION' and not (paths['parent_failed_receipt'].parent/'READY.json').exists(),
    'geometry_failure_evidence_not_erased':geometry['summary']['sampled_depth3mm_pass'] is False,
    'current_oldlinear_source_retained_3c32':before['frozen_joint_linear_source']['sha256']=='3c32f3022d5bade3b6acfd0423c94b4f0dcee3b8f4832a6f0cbd45c9c2da4d17',
    'current_SCP_builder9061_and_sum20objective': 'NVARIABLE=9061' in source['global_SCP_builder_candidate'] and 'sum20(original mean L1 fit' in source['global_SCP_builder_candidate'],
    'new_helper_v2_units_interface_matches_builder': "'units':'m'" in source['all_witness_row_algebra_v2'] and "geometry.get('units')!='m'" in source['global_SCP_builder_candidate'],
    'distance_sign_and_sparse_mapping_documented': 'return n@(b[:,cols]-a[:,cols])' in source['all_witness_row_algebra_v2'] and 'return sparse.csr_matrix((np.ones(3020)' in source['global_SCP_builder_candidate'],
    '8major6010arithmetic':(1+8+1)*601==6010 and 4*(1+2*20)==164,
    'newdraft_no_authorizations':all(x is False for x in proto['authorizations'].values()),
}
for key in ('frozen_joint_linear_source','global_SCP_builder_candidate','all_witness_row_algebra_v2'):ast.parse(source[key])
report={
    'schema':'full151_coupled_static_design_receipt:1','status':'DRAFT_COMPLETE_IMPLEMENTATION_AND_RUN_AUTHORIZATION_PENDING',
    'draft_protocol':pin(target),'source_pins':before,'checks':checks,'all_checks_pass':all(checks.values()),
    'document':pin(ROOT/'docs/full151_coupled_feasibility_contract_draft_v1.md'),
    'actual_source_interface_findings':['Originalall-witnesshelper lacked top-levelunits=m; newv2metadata-only closure read, not rerun.','eta_gate_pass must not be an intermediate restoration eligibility condition.','Globalbuilder objective already keeps sum20, not globalmean20.','Rootdriver merit/acceptance/finitefailure rules require implementation proof; not implied by sparsebuilder.'],
    'scope':{'source_JSON_AST_hash_only':True,'arrays_loaded_or_computed':False,'native_or_newmathmodules_imported':False,'numpy_scipy_mujoco_torch_warp_imported':any(x in sys.modules for x in ('numpy','scipy','mujoco','torch','warp')),'solver_FK_query_Env_WM_physics_GPU':False,'old_sources_modified':False},
    'source_unchanged_pre_post':before=={k:pin(p) for k,p in paths.items()},'receipt_source':pin(__file__),
}
out=OUT/'report.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'protocol':pin(target),'report':pin(out),'checks':len(checks),'all_pass':all(checks.values())}))
if not all(checks.values()) or not report['source_unchanged_pre_post']:raise SystemExit(2)
