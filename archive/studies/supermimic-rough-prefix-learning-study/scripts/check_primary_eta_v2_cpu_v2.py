"""Saved JSONL ETA routing/replay proof. Never run a P/A2 primary ETA."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/estimate_primary_eta_cpu_v2.py'
FROZEN='50bc470d2337bfa536eebd96c03ac7f69369bd28364ec2d0e37f669f6902fd27'
spec=importlib.util.spec_from_file_location('eta_v2',SOURCE);eta=importlib.util.module_from_spec(spec);spec.loader.exec_module(eta)

def rejected(fn):
    try:fn()
    except ValueError:return True
    return False

def main():
    out=ROOT/'outputs/primary_eta_v2_cpu_proof_v2';out.mkdir(parents=True,exist_ok=False);checks={}
    old_source=ROOT/'scripts/estimate_primary_eta_cpu.py';old=eta.read(ROOT/'outputs/primary_a1l_seed0_v1_eta_at_pause/report.json');new=eta.read(ROOT/'outputs/primary_eta_v2_A1L199_replay_v1/report.json');config=ROOT/'configs/a1l-primary_matched-authorized_v1.json';health=ROOT/'outputs/health_primary_a1l_100_v1/report.json';snapshot=ROOT/'outputs/primary_a1l_seed0_v1_eta_at_pause/metrics_prefix_snapshot.jsonl';p,h=eta.validate_health(config,health)
    checks['source_frozen_before_proof']=eta.pin(SOURCE)['sha256']==FROZEN==new['source']['sha256']
    checks['executed_V1_source_preserved']=eta.pin(old_source)==old['source']
    function=lambda path:ast.unparse(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='policy_cost'))
    checks['policy_cost_formula_AST_unchanged']=function(old_source)==function(SOURCE)
    checks['A1L199_point_remaining_exact_replay']=new['point_remaining_seconds']==old['point_remaining_seconds']==1531.705229113698
    checks['same_WM_window_and_measured_base']=new['base_measurement_window']==old['base_measurement_window']=={'first_iteration':148,'last_iteration':198,'intervals':50} and new['base_WM_collect_log_holdout_seconds_per_iteration']==old['base_WM_collect_log_holdout_seconds_per_iteration']
    checks['same20second_allowance_and_half_to1p8_range']=new['unmeasured_future_checkpoint_and_phase_overhead_allowance_seconds']==20. and new['rough_remaining_range_seconds']==old['rough_remaining_range_seconds']==[new['point_remaining_seconds']*.5,new['point_remaining_seconds']*1.8]
    checks['snapshot_is_historical_not_current_finish_forecast']=new['mode']=='historical_snapshot_replay_not_current_ETA' and new['point_finish_local_UTCplus8'] is None and new['rough_finish_range_local_UTCplus8'] is None
    checks['exact_health_config_and_metrics_pins']=new['config']==old['config']==eta.pin(config) and new['health100']==old['health100']==eta.pin(health) and new['metrics_committed_prefix_sha256']==old['metrics_committed_prefix_sha256']==eta.pin(snapshot)['sha256']
    proxy={}
    for method in eta.METHODS:
        # Routing fixture only: this does not assert a real P/A2 healthy100 run.
        routed=copy.deepcopy(p);routed['method']=method;cost,source=eta.smoke_policy_proxy(routed);expected=ROOT/f'outputs/smoke_{method.lower()}_v2/metrics.jsonl';proxy[method]={'cost':cost,'source':source,'routing_only_not_primary_ETA':True}
        checks[method+'_fixed_same_arm_proxy_path']=eta.smoke_proxy_path(method)==expected and source['file']==eta.pin(expected) and source['method']==method and source['cross_arm_proxy'] is False
        checks[method+'_own_smoke_identity_and_steady_5calls']=eta.read(expected.parent/'config.json')['method']==method and cost['first_iteration']==4 and cost['last_iteration']==9 and cost['measured_policy_calls']==5 and cost['seconds_per_policy_update']>0
    checks['unknown_arm_rejected']=rejected(lambda:eta.smoke_proxy_path('BOGUS'))
    checks['noncanonical_arm_spelling_rejected']=rejected(lambda:eta.smoke_proxy_path('p_only'))
    smoke_a1l=eta.read(ROOT/'outputs/smoke_a1l_v2/config.json');routed=copy.deepcopy(p);routed['method']='P_ONLY'
    checks['cross_arm_smoke_metadata_rejected']=rejected(lambda:eta.validate_smoke_identity(routed,smoke_a1l))
    wrong=copy.deepcopy(p);wrong['seed']=1;checks['wrong_seed_smoke_rejected']=rejected(lambda:eta.validate_smoke_identity(wrong,smoke_a1l))
    wrong=copy.deepcopy(p);wrong['input_bundle']={**p['input_bundle'],'sha256':'0'*64};checks['wrong_source_smoke_rejected']=rejected(lambda:eta.validate_smoke_identity(wrong,smoke_a1l))
    wrong_health=copy.deepcopy(h);wrong_health['method']='P_ONLY';hp=out/'wrong_arm_health_fixture.json';hp.write_text(json.dumps(wrong_health));checks['health_wrong_arm_rejected_despite_matching_config_pin']=rejected(lambda:eta.validate_health(config,hp))
    cp=out/'wrong_config_fixture.json';cp.write_text(json.dumps(p));checks['health_wrong_exact_config_pin_rejected']=rejected(lambda:eta.validate_health(cp,health))
    rows,prefix=eta.complete_rows(snapshot)
    checks['snapshot_wrong_config_pin_rejected']=rejected(lambda:eta.validate_snapshot(snapshot,cp,health,rows,prefix))
    checks['snapshot_wrong_health_pin_rejected']=rejected(lambda:eta.validate_snapshot(snapshot,config,hp,rows,prefix))
    checks['snapshot_altered_prefix_rejected']=rejected(lambda:eta.validate_snapshot(snapshot,config,health,rows,prefix+b'\n'))
    f=out/'partial_line_fixture.jsonl';f.write_bytes(b'{"iteration":0}\n{"uncommitted":');rr,bb=eta.complete_rows(f);checks['uncommitted_last_JSONL_line_ignored']=rr==[{'iteration':0}] and bb==b'{"iteration":0}\n'
    primary_rows,_=eta.complete_rows(Path(p['output'])/'metrics.jsonl');primary_cost=eta.policy_cost(primary_rows);checks['real_primary_policy_window_available_for_priority_branch']=primary_cost is not None and primary_cost['first_iteration']>=p['training']['wm_only_iters'] and primary_cost['seconds_per_policy_update']>0
    checks['no_GPU_Env_or_Torch_imports']=not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
    checks['source_remains_frozen']=eta.pin(SOURCE)['sha256']==FROZEN
    result={'schema':'primary_ETA_v2_CPU_saved_proof:1','status':'PASS_JSONL_REPLAY_AND_SAME_ARM_ROUTING' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source':eta.pin(SOURCE),'checker':eta.pin(__file__),'A1L199_replay_report':eta.pin(ROOT/'outputs/primary_eta_v2_A1L199_replay_v1/report.json'),'preserved_V1_report':eta.pin(ROOT/'outputs/primary_a1l_seed0_v1_eta_at_pause/report.json'),'proxy_routing':proxy,'actual_A1L_primary_policy_cost_helper_only':primary_cost,'actual_P_ONLY_primary_ETA_executed':False,'actual_A2_primary_ETA_executed':False,'GPU_used':False,'physics_steps':0,'training_started':False,'scope':'Source-frozen historical replay and helper routing only. P/A2 descriptors are in-memory fixtures for own-smoke selection, not healthy100 claims or actual long-run forecasts.'}
    path=out/'report.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'report':eta.pin(path)},indent=2));return int(not all(checks.values()))
if __name__=='__main__':raise SystemExit(main())
