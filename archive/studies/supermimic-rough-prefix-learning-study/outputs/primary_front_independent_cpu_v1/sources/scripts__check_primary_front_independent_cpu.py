"""Independent protocol/closure review and one actual default primary preflight."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os,subprocess,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/primary_front_independent_cpu_v1'
PYTHON=Path(r'D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe')
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def main():
 OUT.mkdir(exist_ok=False);snap=OUT/'sources';snap.mkdir();checks={}
 def ck(k,v):checks[k]=bool(v)
 files=['scripts/train_rough_prefix_pen_v5.py','scripts/run_rough_prefix_v6.py','scripts/build_primary_config_cpu.py','scripts/audit_rough_checkpoint_cpu_v3.py','protocols/rough-prefix-v4.json','scripts/check_primary_front_independent_cpu.py']
 sourcepins={k:pin(ROOT/k) for k in files}
 for k in files:(snap/k.replace('/','__')).write_bytes((ROOT/k).read_bytes())
 protocol=read(ROOT/'protocols/rough-prefix-v4.json');proofpath=ROOT/'outputs/primary_front_cpu_v2/report.json';proof=read(proofpath)
 ck('owner_CPU42_source_bound',proof['status']=='PASS_CPU_ONLY' and proof['passed']==proof['check_count']==42 and all(c['passed'] is True for c in proof['checks']) and all(pin(p['path'])['sha256']==p['sha256'] for p in proof['source_pins'].values()) and all(pin(p['path'])['sha256']==p['sha256'] for p in proof['config_pins'].values()))
 ck('owner_CPU_real_defaults_no_watch_or_framework',proof['watch_called'] is False and proof['GPU_run_authorized'] is False and proof['Torch_imported'] is False and proof['Warp_imported'] is False and proof['Env_or_Runner_constructed'] is False and len(proof['actual_direct_child_commands'])==9 and all(c['exit_code']==0 for c in proof['actual_direct_child_commands']))
 ck('protocol4_exact_role_seed_and_lineage',protocol['protocol_revision']==4 and protocol['allowed_roles']==['primary_matched'] and protocol['allowed_seeds']==[0] and protocol['source_window_half_open']==[200,216] and protocol['source_lineage_N']==1 and protocol['independent_ego_N']==0 and protocol['physical_reference_qualified'] is False and protocol['shared10_enrollment'] is False)
 cfgs={method:read(ROOT/f'configs/{method.lower()}-primary_matched-v1.json') for method in ('A1L','P_ONLY','A2')}
 cfgpins={method:pin(ROOT/f'configs/{method.lower()}-primary_matched-v1.json') for method in cfgs}
 source_counts={};prior_files=0
 for method,cfg in cfgs.items():
  ck(method+'_GPUfalse_strict_scope_and_budget',cfg['GPU_run_authorized'] is False and cfg['role']=='primary_matched' and cfg['method']==method and cfg['seed']==0 and cfg['training']==protocol['primary_matched_training'] and cfg['protocol']==pin(ROOT/'protocols/rough-prefix-v4.json') and cfg['controller']=={'kind':'ROUGH_PREFIX_BASELINE_EMA_C2','format_version':104,'margin_rad':.02,'resume':'from_scratch_only'} and cfg['automatic_retry'] is False and cfg['compile_fallback'] is False)
  ck(method+'_sources_all_current',all(pin(p['path'])['sha256']==p['sha256'] for p in cfg['source_pins'].values()))
  ck(method+'_scope_remains_fresh_output',not Path(cfg['output']).exists() and not Path(cfg['output']+'_watchdog').exists())
  ck(method+'_old_checkpoint_arguments_absent',not {'checkpoint','model_checkpoint'} & set(cfg['training']) and cfg['training_authorized'] is True)
  values=list(cfg['source_pins'].values());source_counts[method]=len(values)
  ck(method+'_all27_prerequisite_pins_in_archive',all(v in values for item in protocol['primary_matched_prerequisites'].values() for v in item.values()))
  closed=True
  for prior_method,item in protocol['primary_matched_prerequisites'].items():
   for kind in ('smoke_config','evaluation_config'):
    previous=read(item[kind]['path']);prior_files+=len(previous['source_pins']);closed=closed and all(v in values for v in previous['source_pins'].values())
  ck(method+'_all_prior_executed_source_closure_preserved',closed)
 ck('all_three_methods_matched_train_budget',all(c['training']==cfgs['A1L']['training'] for c in cfgs.values()))
 all27=[p for method in protocol['primary_matched_prerequisites'].values() for p in method.values()]
 ck('all27_receipt_bytes_current',len(all27)==27 and all(pin(p['path'])['sha256']==p['sha256'] for p in all27))
 outcomes={};engineering={}
 for method,item in protocol['primary_matched_prerequisites'].items():
  audit=read(item['evaluation_saved_audit']['path']);evaluation=read(item['evaluation']['path']);cap=read(item['evaluation_capacity']['path']);outcomes[method]=evaluation['native_policy_pass'];engineering[method]=audit['engineering_qualified']
  ck(method+'_same_checkpoint_eval_audit_and_capacity_binding',audit['engineering_qualified'] is True and all(v is True for v in audit['checks'].values()) and audit['config']==item['evaluation_config'] and audit['checkpoint']==item['smoke_checkpoint'] and audit['evaluation']==item['evaluation'] and cap['status']=='OK' and cap['flags_by_world']==[0] and cap['read_count']==1 and cap['read_after_step']==60 and evaluation['complete'] is True and evaluation['physical_steps']==60 and evaluation['control_intervals']==15)
 ck('actual_three_qualityfalse_not_selection_gate',all(v is False for v in outcomes.values()) and all(engineering.values()) and protocol['fixed_campaign']['quality_selection'] is False)
 t=protocol['primary_matched_training'];acc=0.;per=t['warmup_steps'];first=None
 for i in range(1,t['max_iterations']+1):
  acc+=1./t['utd'];n=int(acc);acc-=n;per+=n
  if i==100:first=per*128
 active=t['max_iterations']-t['wm_only_iters'];computed={'collect_calls':t['max_iterations']+1,'control_steps_per_world':per,'real_transitions_including_holdout':per*128,'training_worlds':128-max(1,int(128*t['holdout_fraction'])),'WM_holdout_worlds':max(1,int(128*t['holdout_fraction'])),'actual_optimizer_steps':{'wm':2000*2,'fail_head':2000*2,'actor':active,'critic':active*2*2},'recorded_training_batched_physics_substeps':per*4,'recorded_training_world_substeps':per*128*4}
 ck('independent_full_budget_and_holdout_arithmetic',all(protocol['expected_counts_per_arm'][k]==v for k,v in computed.items()) and first==protocol['first_checkpoint']['real_transitions']==40192)
 ck('first100_WM_only_prefix_not_final_or_policy_qualification',protocol['first_checkpoint']['actual_optimizer_steps']=={'wm':200,'fail_head':200,'actor':0,'critic':0} and protocol['first_checkpoint']['collect_calls']==101 and 'NOT_RUN' in protocol['first_checkpoint']['policy_status'] and 'finalizedfalse' in protocol['first_checkpoint']['capacity'])
 counter=read(ROOT/'outputs/rough_optimizer_counter_cpu_v3/report.json');ck('new_v3_lazyzero_gate28_frozen',counter['passed']==counter['total']==28 and pin(counter['source_pins']['new_v3']['path'])==counter['source_pins']['new_v3'] and 'reject_missing_expected_positive' in counter['checks'] and counter['checks']['reject_missing_expected_positive'] is True)
 ck('native_holdout_scope_honestly_distinct_from_generalization',protocol['holdout_scope']=='NativeWM holdout12 worlds excluded fromWM training/torque normalization. NativeActor observation norm usesallfiniteworld observations; noindependenttrajectory/generalizationclaim fromthisWMholdout.')
 life_path=Path(r'D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py');spec=importlib.util.spec_from_file_location('reviewed_primary_CPU_cleanup',life_path);life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
 args=[str(PYTHON),str(ROOT/'scripts/run_rough_prefix_v6.py'),'--config',cfgpins['A1L']['path']];tick=time.monotonic();child=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
 try:stdout,stderr=child.communicate(timeout=20)
 except subprocess.TimeoutExpired:
  (OUT/'CPU_timeout_cleanup.json').write_text(json.dumps(life.stop_owned(child)));raise RuntimeError('Default-only CPU test timed out; no GPU or retry')
 (OUT/'actual_A1L_default_supervisor_stdout.json').write_bytes(stdout);(OUT/'actual_A1L_default_supervisor_stderr.txt').write_bytes(stderr)
 actual=json.loads(stdout);ck('independent_actual_A1L_default_supervisor_CPU_pass',child.returncode==0 and actual['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and actual['config_sha256']==cfgpins['A1L']['sha256'] and actual['role']=='primary_matched' and actual['method']=='A1L' and actual['Torch_imported'] is False and actual['Warp_imported'] is False and actual['GPU_started'] is False and actual['physics_steps']==0)
 ck('no_longtraining_output_from_independent_default',not Path(cfgs['A1L']['output']).exists() and not Path(cfgs['A1L']['output']+'_watchdog').exists())
 ck('all_reviewed_sources_and_configs_preserved',all(pin(p['path'])['sha256']==p['sha256'] for p in sourcepins.values()) and all(pin(p['path'])['sha256']==p['sha256'] for p in cfgpins.values()))
 report={'schema':'rough_primary_front_independent_CPU:1','status':'PASS_CPU_PRIMARY_GATE_NOT_GPU_AUTHORIZATION' if all(checks.values()) else 'BLOCKING_OR_FAILED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_pins':sourcepins,'config_pins':cfgpins,'owner_CPU42_proof':pin(proofpath),'actual_independent_default_child':{'args':args,'exit_code':child.returncode,'elapsed_seconds':time.monotonic()-tick,'scope':'Default supervisor only; it launches one default entry internally, no watch/run.'},'source_count_per_arm':source_counts,'actual_smoke_quality':outcomes,'independent_budget_arithmetic':computed,'first100_transitions':first,'ETA_and_user_boundary':'After independent healthy100 WM-prefix, report WM/collect measured primary rates with separate200warmup/1800policy phases; same-versionA1L8policy smoke is only uncertain proxy before primarypolicy observations exist. Do not10iterx200, create new benchmark, waitpost200 or assert guaranteedtime/GPUbusy. Pause Goal after healthy100+ETA while explicitly authorized A1L background worker continues.','fairness_limit':'Matched real transitions/optimizer updates only;7200s is engineering guard, not registered equal-wall secondary endpoint. Reference/setup/failed attempt/teacher costs remain separate.','authorization_boundary':'All3 prepared configurations remain GPUfalse; only Root may separately freeze and authorize A1L once. P_ONLY/A2 cannot auto-start and no queue is created.','scope':'Source/metadata/ownerCPUproof review and one independent real default supervisor preflight only. No builder rerun, no primary source modification, no Env/physics/GPU/optimizer/longtraining.','GPU_used':False,'physics_steps':0,'optimizer_steps':0,'experimental_source_modified':False}
 p=OUT/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(p)}));return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
