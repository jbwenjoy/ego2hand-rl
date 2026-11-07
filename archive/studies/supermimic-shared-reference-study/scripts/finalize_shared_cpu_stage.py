"""Create non-overwriting CPU-stage index and draft cost/status contract."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 report=read(ROOT/'outputs/shared_registry_cpu_v1/report.json');checks=read(ROOT/'outputs/shared_registry_contract_checks_v1/report.json')
 assert checks['status']=='PASS' and checks['count']==29
 for p in report['sources'].values():assert sha(p['path'])==p['sha256']
 assert sha(ROOT/'README.md')=='909f5bea6e70adb7d280fbd448a9243f80fb358af9cced4ef5df5771db9adae3'
 contract={'schema':'shared_CPU_stage_contract:1','status':'CPU_ONLY_NOT_TRAINING_AUTHORITY',
  'G2':'Split-specific native catalogs resolved/read on CPU; runtime init and trainer integration PENDING',
  'G3':'Target identity/index contract CPU tested; 10 real independent target bundles MISSING, gather fails closed',
  'G4_G5':'Shared native training, physics/reset parity, fixed seeds and actual gradient/runtime qualification PENDING',
  'G6':'Replay provenance/window contract CPU tested; native replay/normalizer/actor-start integration PENDING',
  'G7':'Exact metadata transfer identity guard CPU tested; actual checkpoint transfer/resume PENDING, old2/3/103 rejected',
  'G8':'No trained evaluation; future N=training seeds, 3 processes/checkpoint, primary all3-pass, task failures retained',
  'budgets':{'status':'DRAFT_NOT_EXECUTED','per_reference_aggregate':{'real_transitions':648192,'WM_optimizer_steps':4000,'failure_head_optimizer_steps':4000,'actor_optimizer_steps':1800,'critic_optimizer_steps':7200,'teacher_transitions':0},
   'N':[1,5,8,10],'formal_seeds_and_engineering_smoke':'PENDING_NEW_EXPLICIT_AUTHORIZATION_AND_PHYSICS_INIT_QUALIFICATION',
   'secondary_wall_seconds_per_reference':1200,'secondary_selection':'latest completed checkpoint before inclusive total method deadline; no quality selection'},
  'cost_ledger_required':['stage','method','training_seed','trajectory_exposure_source_ids','reference_registry_sha256','start_OS_creation_UTC','end_UTC',
   'process_wall_seconds','nested_simulator_seconds','real_transitions','WM_optimizer_steps','failure_head_optimizer_steps','actor_optimizer_steps','critic_optimizer_steps',
   'teacher_transitions','teacher_generation_wall_seconds','pretraining_cost','per_new_trajectory_adaptation_cost','evaluation_cost','setup_compile_cost','failed_attempts_cost'],
  'cost_rules':['Pretraining/adaptation/teacher/setup/failed attempts count toward aggregate primary and wall ledgers as applicable',
   'No free all10 exposure before N1','Do not add nested wall timers','Shared efficiency or transfer benefit remains unmeasured'],
  'source_scope':'one conservative lineage, not independent ego demonstration generalization','training_executed':False,'physics_steps':0}
 cp=ROOT/'configs/shared_cpu_stage_contract_v1.json';write(cp,contract)
 ready={'schema':'shared_CPU_registry_READY:1','status':'COMPLETE_CPU_STAGE_WITH_PENDING_TRAINING_PHYSICS_TARGETS','registry':report['registry'],
  'pending_targets':report['pending_targets'],'CPU_native_reader_report':pin(ROOT/'outputs/shared_registry_cpu_v1/report.json'),
  'CPU_contract_checks':pin(ROOT/'outputs/shared_registry_contract_checks_v1/report.json'),'stage_contract':pin(cp),
  'documentation':pin(ROOT/'docs/shared_cpu_registry_v1.md'),'source_pins':{str(p.relative_to(ROOT)):pin(p) for p in [*sorted((ROOT/'src/shared_reference').glob('*.py')),
   ROOT/'scripts/build_shared_registry_cpu.py',ROOT/'scripts/check_shared_registry_cpu.py',Path(__file__)]},
  'all10_init_measured':None,'all10_independent_contact_targets':'MISSING','native_training_entry_integration':'PENDING',
  'training_authorized':False,'physical_reference_qualified':False,'GPU_used':False,'root_README_unchanged_sha256':sha(ROOT/'README.md')}
 out=ROOT/'outputs/shared_registry_cpu_v1/READY.json';write(out,ready);print(json.dumps({'READY':pin(out),'registry':report['registry'],'checks':ready['CPU_contract_checks']},indent=2))
if __name__=='__main__':main()
