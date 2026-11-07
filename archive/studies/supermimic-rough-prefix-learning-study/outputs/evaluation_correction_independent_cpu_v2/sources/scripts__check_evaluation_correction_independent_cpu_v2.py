"""Actual default CLI checks for the configuration-only evaluation-v2 repair."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/evaluation_correction_independent_cpu_v2'
PYTHON=Path(r'D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe')
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def main():
 OUT.mkdir(exist_ok=False);snap=OUT/'sources';snap.mkdir();checks={};runs=[]
 def ck(k,v):checks[k]=bool(v)
 files=['src/rough_prefix/evaluation.py','src/rough_prefix/evaluation_v2.py','scripts/evaluate_rough_prefix_pen.py','scripts/evaluate_rough_prefix_pen_v2.py','scripts/run_rough_evaluation.py','scripts/run_rough_evaluation_v2.py','scripts/build_evaluation_config_cpu.py','scripts/build_evaluation_config_cpu_v2.py','protocols/rough-evaluation-v1.json','protocols/rough-evaluation-v2.json','scripts/check_evaluation_correction_independent_cpu_v2.py']
 before={k:pin(ROOT/k) for k in files}
 for k in files:(snap/k.replace('/','__')).write_bytes((ROOT/k).read_bytes())
 life_file=Path(r'D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py');s=importlib.util.spec_from_file_location('reviewed_cpu_only_cleanup',life_file);life=importlib.util.module_from_spec(s);s.loader.exec_module(life)
 assert pin(life_file)['sha256']=='40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'
 def child(label,args):
  tick=time.monotonic();p=subprocess.Popen([str(PYTHON),*map(str,args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  try:out,err=p.communicate(timeout=20)
  except subprocess.TimeoutExpired:
   (OUT/(label+'_cleanup.json')).write_text(json.dumps(life.stop_owned(p)));raise RuntimeError('CPU preflight timeout; no GPU or retry')
  (OUT/(label+'_stdout.txt')).write_bytes(out);(OUT/(label+'_stderr.txt')).write_bytes(err);runs.append({'label':label,'args':list(map(str,args)),'returncode':p.returncode,'elapsed_seconds':time.monotonic()-tick})
  if p.returncode:raise RuntimeError(label+' failed: '+err.decode(errors='replace'))
  return ast.literal_eval(out.decode()) if label=='build_config_CPU' else json.loads(out)
 cfgpath=ROOT/'configs/a1l-policy-evaluation-check-v2.json'
 built=child('build_config_CPU',[ROOT/'scripts/build_evaluation_config_cpu_v2.py','--method','A1L','--version','check-v2','--health',ROOT/'outputs/health_smoke_a1l_v2_checker_v2/report.json','--actor-load',ROOT/'outputs/actual_actor_load_a1l_v2/report.json','--evidence',ROOT/'outputs/evaluation_termination_cpu_v2/report.json','--evidence',ROOT/'outputs/evaluation_v1_failure_independent_saved_review_v1/report.json'])
 cfg=read(cfgpath);ck('new_config_GPU_false_and_builder_SHA',cfg['GPU_run_authorized'] is False and built==pin(cfgpath))
 for label,script in [('actual_default_entry','evaluate_rough_prefix_pen_v2.py'),('actual_default_supervisor','run_rough_evaluation_v2.py')]:
  p=child(label,[ROOT/'scripts'/script,'--config',cfgpath]);ck(label+'_CPU_boundary',p['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and p['Torch_imported'] is False and p['Warp_imported'] is False and p['GPU_started'] is False and p['physics_steps']==0 and p['config_sha256']==pin(cfgpath)['sha256'])
 ck('no_experimental_or_watch_output_created',not Path(cfg['output']).exists() and not Path(cfg['output']+'_watchdog').exists())
 ck('all_config_sources_current_and_v2_route_pinned',all(pin(v['path'])['sha256']==v['sha256'] for v in cfg['source_pins'].values()) and all(any(Path(v['path']).resolve()==(ROOT/k).resolve() for v in cfg['source_pins'].values()) for k in ['src/rough_prefix/evaluation_v2.py','scripts/evaluate_rough_prefix_pen_v2.py','scripts/run_rough_evaluation_v2.py','scripts/build_evaluation_config_cpu_v2.py']))
 ck('training_protocol_preserved_in_archive',cfg['source_pins']['training_protocol']==read(cfg['training_config']['path'])['protocol'])
 reportfile=ROOT/'outputs/evaluation_termination_cpu_v2/report.json';proof=read(reportfile)
 ck('actual_factory_owner20_source_bound',proof['status']=='PASS' and proof['passed']==proof['check_count']==20 and all(x['passed'] for x in proof['checks']) and all(pin(v['path'])['sha256']==v['sha256'] for v in proof['source_pins'].values()))
 ck('actual_factory_calls_and_nativeA0_semantics',proof['actual_factory_calls']==4 and proof['factory_mode']=={'num_envs':1,'control_hz':50,'play':True} and proof['native_A0_config_AST_statements_executed']==3 and proof['final_termination_semantics']==proof['native_A0_expected_semantics'] and set(proof['final_termination_semantics'])=={'time_out','unstable','reference_end'} and proof['CUDA_initialized'] is False and proof['Env_constructed'] is False and proof['physics_steps']==0)
 def normalized(path,new):
  t=ast.parse(path.read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='run_evaluation');f.body=[n for n in f.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ((new and isinstance(n.value.func,ast.Name) and n.value.func.id=='configure_full_horizon_terminations') or (not new and 'env_cfg.terminations.pop(' in ast.unparse(n))))];return ast.dump(f,include_attributes=False)
 ck('all_other_orchestration_AST_exact',normalized(ROOT/'src/rough_prefix/evaluation.py',False)==normalized(ROOT/'src/rough_prefix/evaluation_v2.py',True))
 helper=ast.unparse(next(n for n in ast.parse((ROOT/'src/rough_prefix/evaluation_v2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='configure_full_horizon_terminations'))
 ck('strict_actual4keys_to3_only_position_removed',"{'time_out', 'unstable', 'object_position_error', 'reference_end'}" in helper and "env_cfg.terminations.pop('object_position_error')" in helper and "pop('object_orientation_error')" not in helper)
 old=(ROOT/'scripts/evaluate_rough_prefix_pen.py').read_text().replace("'policy_probe.py','evaluation.py','checkpoint_v2.py'","'policy_probe.py','evaluation_v2.py','checkpoint_v2.py'").replace('from rough_prefix.evaluation import run_evaluation','from rough_prefix.evaluation_v2 import run_evaluation')
 ck('front_only_sourceclosure_and_module_route_changed',old==(ROOT/'scripts/evaluate_rough_prefix_pen_v2.py').read_text())
 old=(ROOT/'scripts/run_rough_evaluation.py').read_text().replace("ENTRY=ROOT/'scripts/evaluate_rough_prefix_pen.py'","ENTRY=ROOT/'scripts/evaluate_rough_prefix_pen_v2.py'").replace('(evaluate_rough_prefix_pen|','(evaluate_rough_prefix_pen(?:_v[0-9]+)?|')
 ck('supervisor_only_v2_route_and_advisory_regex_changed',old==(ROOT/'scripts/run_rough_evaluation_v2.py').read_text())
 p1=read(ROOT/'protocols/rough-evaluation-v1.json');p2=read(ROOT/'protocols/rough-evaluation-v2.json');common={k:v for k,v in p1.items() if k!='status'}
 ck('protocol_controls_budgets_gates_selection_unchanged',all(p2[k]==v for k,v in common.items()) and p2['engineering_correction']['probe_solver_samples']==0 and p2['engineering_correction']['prior_quality']=='NOT_ASSESSED')
 ck('old_failed_attempt_current_pins_preserved',all(pin(v['path'])['sha256']==v['sha256'] for v in read(ROOT/'outputs/eval_smoke_a1l_v1/source/inventory.json').values()))
 ck('no_quality_selection_or_retraining',p2['engineering_correction']['new_implementation_attempts_per_arm']==1 and p2['engineering_correction']['prior_quality']=='NOT_ASSESSED' and 'No priorphysicalquality' in p2['engineering_correction']['selection'])
 ck('review_sources_unchanged',all(pin(v['path'])['sha256']==v['sha256'] for v in before.values()))
 report={'schema':'rough_evaluation_configuration_correction_independent_CPU:2','status':'PASS_NEW_IMPLEMENTATION_CPU_QUALIFICATION' if all(checks.values()) else 'BLOCKING_OR_FAILED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'actual_CPU_subprocesses':runs,'config':pin(cfgpath),'source_pins':before,'actual_factory_CPU_proof':pin(reportfile),'old_failed_outcome_independent':pin(ROOT/'outputs/evaluation_v1_failure_independent_saved_review_v1/report.json'),'scope':'Independent static AST/diff plus builder/default-entry/default-supervisor CPU checks. Actual installed factory coverage is the inspected owner20 source-bound proof, not repeated. Old failed attempt retained; v2 is a new explicitly declared implementation qualification. No Env/physics/GPU/optimizer or training rerun by this audit.','GPU_used':False,'physical_task_quality':'NOT_ASSESSED','physical_reference_qualified':False,'experimental_source_modified':False}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')}));return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
