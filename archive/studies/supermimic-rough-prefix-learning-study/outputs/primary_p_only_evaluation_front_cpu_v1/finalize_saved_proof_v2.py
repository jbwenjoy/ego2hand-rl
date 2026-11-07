"""Saved-only finalization of successful CPU children; no subprocess rerun."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def main():
    out=ROOT/'finalization_review_v2';out.mkdir(exist_ok=False);(out/'finalizer_source.py').write_bytes(Path(__file__).read_bytes())
    prior=read(ROOT/'report.json');config=read(prior['config']['path']);protocol=read(prior['source_pins']['protocol']['path']);checks={}
    checks['old_harness_failure_retained']=prior['status']=='FAIL_OR_INCOMPLETE' and prior['checks']['shared_six_consumers_exact_unchanged'] is False and prior['passed']==12 and prior['total']==13
    checks['actual_three_commands_success']=len(prior['commands'])==3 and [row['name'] for row in prior['commands']]==['builder','entry','supervisor'] and all(row['exit_code']==0 and '--run' not in row['command'] for row in prior['commands'])
    checks['config_bytes_match_builder_stdout']=pin(prior['config']['path'])==prior['config']==read(ROOT/'builder.stdout.json')
    checks['actual_Gfalse_final_Pcheckpoint']=config['GPU_run_authorized'] is False and config['method']=='P_ONLY' and config['seed']==0 and config['checkpoint_iteration']==2000 and config['checkpoint']['sha256']=='c73044119386d8041eaed3e28ab456b5b0a973e220773f1fddc406cb72b22bbe'
    for name in ('entry','supervisor'):
        proof=read(ROOT/(name+'.stdout.json'))
        checks[name+'_actual_default_saved_proof']=proof['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and proof['config_sha256']==prior['config']['sha256'] and proof['checkpoint']==config['checkpoint'] and proof['method']=='P_ONLY' and proof['Torch_imported'] is False and proof['Warp_imported'] is False and proof['GPU_started'] is False and proof['physics_steps']==0
    expected={'scripts/evaluate_rough_prefix_pen_v2.py','scripts/run_rough_evaluation_v2.py','src/rough_prefix/evaluation_v2.py','src/rough_prefix/policy_probe.py','src/rough_prefix/checkpoint_v2.py','scripts/audit_native_policy_saved_cpu.py'}
    shared=protocol['shared_executed_consumer']
    checks['actual_six_consumer_keyset']=set(shared)==expected and len(shared)==6
    for key,value in shared.items():checks['consumer_'+key]=config['source_pins']['protocol_shared_consumer__'+key]==value==pin(value['path']) and Path(value['path']).resolve()==(STUDY/key).resolve()
    checks['all_execution_dependencies_current']=all(pin(value['path'])['sha256']==value['sha256'] and Path(value['path']).is_absolute() for value in config['source_pins'].values())
    checks['execution_source_snapshots_match']=all(pin(ROOT/'source'/Path(value['path']).name)['sha256']==value['sha256']==pin(value['path'])['sha256'] for value in prior['source_pins'].values())
    train=read(config['training_config']['path']);health=read(config['health_report']['path']);load=read(config['actor_load_report']['path'])
    checks['primary_training_protocol4_alias_exact']=config['source_pins']['training_protocol']==train['protocol']==protocol['training_protocol']
    checks['actual_final_health_bound']=health['status']=='PASS_SAVED_ENGINEERING_HEALTH' and all(v is True for v in health['checks'].values()) and health['iteration']==2000 and health['final_iteration_scope'] is True and health['checkpoint']==config['checkpoint'] and health['config']==config['training_config']
    checks['actual_Actor_load_bound']=load['status']=='PASS_ACTUAL_EVALUATION_ACTOR_CPU_LOAD' and load['checkpoint']==config['checkpoint'] and load['receipt']['iteration']==2000 and all(row['passed'] is True for row in load['checks'])
    checks['same_original_budget_no_retry']=config['hard_process_seconds']==protocol['hard_process_seconds']==120 and config['soft_worker_seconds']==protocol['soft_worker_seconds']==110 and config['automatic_retry'] is False and config['compile_fallback'] is False and config['worlds']==1 and config['evaluation_gates']==protocol['evaluation_gates']
    checks['no_physics_or_watchdog_output_created']=not Path(config['output']).exists() and not Path(config['output']+'_watchdog').exists()
    report={'schema':'rough_P_ONLY_final_evaluation_saved_CPU_finalization:1','status':'PASS_SAVED_ONLY_CPU_FINALIZATION' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'config':prior['config'],'source_pins':prior['source_pins'],'actual_actor_load':prior['actual_actor_load'],'health':prior['health'],'old_report':pin(ROOT/'report.json'),'positive_commands_source':pin(ROOT/'positive_commands.ps1'),'finalizer':pin(__file__),'inputs':{name:pin(ROOT/(name+'.stdout.json')) for name in ('builder','entry','supervisor')},'correction':'Original PowerShell harness used PSObject.Properties.Count as scalar6; that collection expression caused a false count gate. Original failed report and source retained. Python dict keyset and all six current file pins independently recomputed from saved data. A first saved finalizer also rejected equivalent forward/backslash spelling on inherited path pins (all SHAs matched); that failure/source are preserved. Current-file checks use absolute supplied paths and actual SHA, matching the frozen checked() contract. No builder/default child command repeated.','builder_entry_supervisor_processes_repeated':0,'GPU_run_authorized':False,'physics_steps':0,'Env_or_Runner_constructed':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v]},indent=2))
    raise SystemExit(report['status']!='PASS_SAVED_ONLY_CPU_FINALIZATION')
if __name__=='__main__':main()
