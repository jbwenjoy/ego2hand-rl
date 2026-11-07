"""Final evaluator/source receipt only; no production import or real arrays."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'producer_core':('src/full151_gradient/boundary_component_final_evaluation105_v1.py','8c652a01128117f7ced84d3ae09e8aad1aafd41aa66caf51c8a408d3e6e75cf8'),
 'producer_CLI':('scripts/evaluate_full151_boundary_components_final_cpu_v1.py','811a3c22175fc383d39285078978e4f5e23edd54220fc7bf159d164abc3c6a42'),
 'producer_checker':('scripts/check_full151_boundary_components_final_cpu_v1.py','0ba292abca6c49f0239db7ace3513cc4f69a6ca8e9287acdf8e815802e292015'),
 'statistics':('scripts/describe_boundary_component_predictions_saved_cpu_v1.py','7487e16b819271fd3b6d3d6562d4802fd97216afa4e3f666f705948f88cce036'),
 'scientific_contract':('docs/full151_boundary_component_final_evaluation_contract_v1.md','3ad8c12324e37795fc13217d699f323ac99a88719bdb8cfa6f9bfada01b899f6'),
 'frozen_component':('src/full151_gradient/boundary_impulse_component105_v1.py','003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317'),
 'initial_factory':('src/full151_gradient/component_H8_donor105_v1.py','0c77dc5255ffd2d7294dabdc7bbaf8ed09e80f3f2bc0eb89fee3954d268a6b13'),
 'stats_pure_READY':('outputs/full151105_boundary_component_statistics_contract_preparation_v1/READY.json','6e9cabefc49dd6d3d847d13eb500771d15fa1ccdb52e54b92f9780af5ce3ea42'),
 'producer_READY':('outputs/full151105_boundary_component_final_evaluation_preparation_CPU_v1/READY.json','92306387c3e8a8c578f68f05f45d73b4e19aca88f2456c322140cc35598d8caa')}
def require(x,s):
    if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

def main():
    out=ROOT/'outputs/full151105_boundary_component_final_evaluation_source_independent_v1';require(out.is_dir() and {p.name for p in out.iterdir()}=={'preserved_reviewer_v1.py','source_hash_race_v1_failure.json'},'Only preserved prior pre-review failure exists');pins={}
    for k,(name,h) in FILES.items():
        p=ROOT/name;require(sha(p)==h,'Exact final reviewed source '+k);pins[k]=pin(p);(out/(k+p.suffix)).write_bytes(p.read_bytes())
    pins['reviewer']=pin(__file__);(out/'executed_reviewer.py').write_bytes(Path(__file__).read_bytes())
    core=(ROOT/FILES['producer_core'][0]).read_text();stats=(ROOT/FILES['statistics'][0]).read_text();component=(ROOT/FILES['frozen_component'][0]).read_text();cli=(ROOT/FILES['producer_CLI'][0]).read_text()
    tree=ast.parse(core);funcs={v.name:v for v in tree.body if isinstance(v,ast.FunctionDef)};ready=read(ROOT/FILES['stats_pure_READY'][0]);p=ready['cpu_report'];require(sha(p['path'])==p['sha256'],'Pure proof pin');proof=read(p['path'])
    require(proof['passed']==proof['total']==len(proof['checks'])==1 and all(v is True for v in proof['checks'].values()),'Single closed pure statistics proof')
    for v in ready['source_pins'].values():require(sha(v['path'])==v['sha256'],'Current pure proof source')
    producer_ready=read(ROOT/FILES['producer_READY'][0]);producer_proof_pin=producer_ready['cpu_report'];require(sha(producer_proof_pin['path'])==producer_proof_pin['sha256'],'Final producer proof pin');producer_proof=read(producer_proof_pin['path'])
    require(producer_proof['passed']==producer_proof['total']==len(producer_proof['checks'])==3 and all(v is True for v in producer_proof['checks'].values()),'Final producer focused3 proof')
    for v in producer_ready['source_pins'].values():require(sha(v['path'])==v['sha256'],'Final focused proof current sources')
    checks={
      'final_sources_AST_valid_default_metadata_explicit_execute_and14aliases':all(ast.parse((ROOT/FILES[k][0]).read_text()) for k in ('producer_core','producer_CLI','producer_checker','statistics')) and "len(p['inputs'])" not in core and "set(p['inputs'])==INPUTS" in core and 'if a.execute:' in cli,
      'own_initial_plus_two_AFTER1000_strict_namespace_health5_and_no_initialpair_load':"'checkpoint_tag']=='AFTER_OPTIMIZER_I'" in core and "health['final_iteration_scope'] is True" in core and "torch.load(checked(i['original_initial_component'])" in core and "health['initial_pair']" in core and 'torch.load(checked(health' not in core,
      'fixed_all152_identity_group_substeps_absoluteu_and_xi_zero174_target_outside_forward':[a.arg for a in funcs['forward_three'].args.args]==['raw','modules','x','u','dynamics','torch','counts'] and "gids.dtype==np.int64" in core and "substeps.shape==(152,4)" in core and '4*gids[:,None]+np.arange(4)' in core and 'torch.zeros((152,174),dtype=torch.float32)' in core,
      'one_native_constructor_two_rawcopies_three_B152_nativeFW_two_heads_and_baseline_algebra0':"'own_trusted_torch_loads':3,'native_WM_constructors':1,'raw_deepcopies':2,'native_FW_attempted':3,'native_FW_returned':3,'aux_head_FW':2,'predicted_rows':456" in core and 'with torch.inference_mode():' in core and "torch.zeros((152,2,6),dtype=torch.float32)" in core and "'baseline_head_scope'" in core,
      'strict_initial_context_norm_effective_scale_frozen_final_load_no_repairs':"module.wm.load_state_dict(weights,strict=True);module.aux_head.load_state_dict(savedhead,strict=True)" in core and "payload['physical_context_state']" in core and "all(component.bit_equal(saved[k],v) for k,v in context.items())" in core and 'module.eval().requires_grad_(False);module.assert_frozen()' in core,
      'native52_capture_chart_003_same_nonzero_defect_formula_and_F32_public_order':"delta=difference(values['prediction58_FP32'],target)" in core and 'delta,.01,torch)' in core and "module=modules[ARMS[0] if role==ROLES[0] else role]" in core and 'public=native_point_loss+lambda_mode*L_mode+lambda_imp*L_imp' in component and '(rv-rv_saved)/sv' in component and 'I1=R1@Ib@R1.transpose' in component,
      'FP32_point_probability_BCE_and_self15_proxy_captured_distinct_from_F64_stats':all(k in core and k in stats for k in ('native_point_squared_scaled52_FP32','mode_probability2_FP32','self15_probability15_FP32','self15_penalty_rate_FP32','self15_penalty_component_FP32')) and "'mode_BCE_elements_FP32'" in core and 'not claimed bit-identical' in stats,
      'all13cohorts_456rows_strict_mode_ties_unitblocks_no_mixed_norm_or_winner':"[152, 76, 76] + [19]*8 + [24, 128]" in stats and 'guess = logits > 0' in stats and 'assert len(rows) == 456' in stats and "data['source_group_id152']" in stats and 'signed_metric_differences_not_a_composite_winner' in stats and "k.startswith('per_feature_')" in stats and proof['checks']['closed52_units_mode_ties_sources_cancellation_13cohorts_and456rows'] is True,
      'source_RNG_cache_frozen_inputs_failure_preservation_and_cleanup_limited_qualification':"save_raw_rng(out,'before'" in core and "save_raw_rng(out,'after'" in core and "'all_hook_caches_empty'" in core and 'finally:' in core and 'for module in modules.values():module.close()' in core and "'physical_force_or_gradient_qualified':False" in core,
      'raw_norm_before_after_common_context_host_maps_saved_for_independent_read':all(k in core for k in ('normalizer_buffers_before.npz','normalizer_buffers_after.npz','physical_context_buffers.npz',"'normalizer_host_configuration_before':host_before","'normalizer_host_configuration_after':host_after")) and producer_proof['passed']==3,
      'reviewer_source_JSON_AST_and_prior_synthetic_only_no_real_arrays_or_models':not any(k.split('.')[0] in ('numpy','torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules)}
    require(all(checks.values()),'Static source receipt predicate failed')
    for v in pins.values():require(sha(v['path'])==v['sha256'],'Reviewed source changed')
    report={'schema':'full151105_boundary_component_final_evaluation_source_independent:1','status':'NO_KNOWN_SOURCE_BLOCKER_FOR_FIXED_FINAL_TRAINING_SET_DESCRIPTION','source_review_qualified':True,'passed':len(checks),'total':len(checks),'checks':checks,'reviewed_source_pins':pins,'statistics_pure_fixture':{'READY':pin(ROOT/FILES['stats_pure_READY'][0]),'report':p,'passed':1,'total':1},'producer_focused_tests_scope':{'READY':pin(ROOT/FILES['producer_READY'][0]),'report':producer_proof_pin,'passed':3,'total':3,'reviewer_reexecution':False},'preserved_prior_source_hash_race':pin(out/'source_hash_race_v1_failure.json'),'known_source_blockers':[],
      'limits':['No actual evaluation arrays, weights, payloads, model/native imports or FW/AD performed by this review.',
        'Full parameter unchanged digest remains a producer runtime assertion, not an independent fresh weight reload. Final source now additionally saves raw norm before/after arrays, host maps and shared context buffers for the future saved audit; this source receipt does not read their actual values.',
        'Baseline z/mode is algebraic proven positive-zero initialization, not a third head FW. Initial native58/self15 is genuinely forwarded.',
        'Captured native52 Log is not independently reimplemented and is not a rawR9 SO3 geodesic certificate. FP32 captured reductions and F64 descriptive rescores have separate names.',
        'All152 are known training rows; 13cohorts overlap. No independent trajectory, CI, composite winner, physical, gradient, hidden-state or Actor/ego qualification.',
        'Failure retains raw outputs before loss arithmetic and stops the whole job; no automatic retries, repair or selected successful subset.'],
      'actual_evaluation_authorized_by_this_receipt':False,'actual_model_accuracy_or_physical_qualified':False,'reviewer_actual_arrays_weights_models_FW_AD_MJB_Env_physics_GPU':0}
    write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'passed':len(checks),'total':len(checks),'source_review_qualified':True},indent=2))
if __name__=='__main__':main()
