"""Independent static routing/AST/source review. Never import production code."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve(strict=True);return {'path':p.as_posix(),'sha256':sha(p)}
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


old_inventory_path=ROOT/'outputs/current104_physical_FD_v1/source/inventory.json'
inv=read(old_inventory_path)
specs=[
('worker','src/rough_prefix/physical_fd.py','src/rough_prefix/physical_fd_v2.py',[
    ('from .probe_transaction_v2 import ProbeTransaction','from .probe_transaction_v3 import ProbeTransaction')]),
('entry','scripts/probe_current104_physical_fd.py','scripts/probe_current104_physical_fd_v2.py',[
    ("protocol['protocol_revision']!=2","protocol['protocol_revision']!=3"),
    ("'physical_fd.py','probe_transaction_v2.py'","'physical_fd_v2.py','probe_transaction_v3.py'"),
    ("ROOT/'scripts/run_current104_physical_fd.py'","ROOT/'scripts/run_current104_physical_fd_v2.py'"),
    ('outputs/current104_physical_FD_v1','outputs/current104_physical_FD_v2'),
    ('from rough_prefix.physical_fd import run_physical_probe','from rough_prefix.physical_fd_v2 import run_physical_probe')]),
('supervisor','scripts/run_current104_physical_fd.py','scripts/run_current104_physical_fd_v2.py',[
    ('scripts/probe_current104_physical_fd.py','scripts/probe_current104_physical_fd_v2.py'),
    ('(probe_current104_physical_fd|','(probe_current104_physical_fd(?:_v[0-9]+)?|')]),
('builder','scripts/build_current104_physical_probe_config_cpu.py','scripts/build_current104_physical_probe_config_cpu_v2.py',[
    ('protocols/current104-physical-fd-execution-v2.json','protocols/current104-physical-fd-execution-v3.json'),
    ("p['protocol_revision']==2","p['protocol_revision']==3"),
    ('revision2 physical-FD protocol','revision3 physical-FD protocol'),
    ('Revision2 native FP32 reward term ABI missing','Revision3 native FP32 reward term ABI missing'),
    ('configs/current104-physical-FD-v1.json','configs/current104-physical-FD-v2.json'),
    ('outputs/current104_physical_FD_v1','outputs/current104_physical_FD_v2'),
    ('scripts/probe_current104_physical_fd.py','scripts/probe_current104_physical_fd_v2.py'),
    ('scripts/run_current104_physical_fd.py','scripts/run_current104_physical_fd_v2.py'),
    ('src/rough_prefix/physical_fd.py','src/rough_prefix/physical_fd_v2.py'),
    ('src/rough_prefix/probe_transaction_v2.py','src/rough_prefix/probe_transaction_v3.py')]),
]
checks={};files=[]
def check(name,v):checks[name]=bool(v)
for role,old_name,new_name,replacements in specs:
    old_path=ROOT/old_name;new_path=ROOT/new_name
    old=old_path.read_text(encoding='utf-8');new=new_path.read_text(encoding='utf-8')
    check(role+'_baseline_is_actual_executed_old_pin',sha(old_path)==inv[role]['sha256'] and Path(inv[role]['path']).resolve()==old_path.resolve())
    expected=old;reverse=new;counts=[]
    for before,after in replacements:
        counts.append({'old':before,'new':after,'old_count':expected.count(before),'new_reverse_count':reverse.count(after)})
        if expected.count(before)!=1 or reverse.count(after)!=1:raise AssertionError('Nonunique declared routing change '+role)
        expected=expected.replace(before,after,1);reverse=reverse.replace(after,before,1)
    check(role+'_exact_text_delta_only',expected==new)
    check(role+'_reverse_routing_AST_identity',ast.dump(ast.parse(old),include_attributes=False)==ast.dump(ast.parse(reverse),include_attributes=False))
    files.append({'role':role,'old':pin(old_path),'new':pin(new_path),'declared_changes':counts})
oldp_path=ROOT/'protocols/current104-physical-fd-execution-v2.json';newp_path=ROOT/'protocols/current104-physical-fd-execution-v3.json'
oldp=read(oldp_path);newp=read(newp_path)
normalized=dict(newp);history=normalized.pop('revision_history');routing=normalized.pop('implementation_routing');normalized['protocol_revision']=2
check('protocol3_only_revision_history_routing_changed',canon(normalized)==canon(oldp) and type(newp['protocol_revision']) is int and newp['protocol_revision']==3)
check('protocol3_correct_old_attempt_and_helper21_bound',history['previous_execution_protocol']['sha256']==sha(oldp_path)
      and history['preserved_failed_attempt_config']['sha256']==sha(history['preserved_failed_attempt_config']['path'])
      and history['preserved_failed_attempt_saved_audit']['sha256']==sha(history['preserved_failed_attempt_saved_audit']['path'])
      and history['new_empty_array_helper']['sha256']==sha(ROOT/'src/rough_prefix/probe_transaction_v3.py')=='43378a259abf3a38c9f06bca98cf50e755de1f99089ddfe28229d86ac1012851'
      and history['new_empty_array_real_Warp_CPU_proof']['sha256']==sha(history['new_empty_array_real_Warp_CPU_proof']['path'])=='987ec351e8a4ad42292ea3ac562db8981f460e9f148a7f90d2b44e647e6f4719')
check('same_ce9a_proposal_directions_caps_reward_criteria',newp['parent_proposal']['sha256']=='ce9a9575b6aa7dc9b5b535bfe1cf7ff5be4db6065dc122ba4860561d40096e09'
      and newp['directions']==oldp['directions'] and newp['hard_process_seconds']==240 and newp['soft_worker_seconds']==220
      and newp['fresh_baseline']==oldp['fresh_baseline'] and newp['object_reward_arithmetic']==oldp['object_reward_arithmetic'])
check('routing_has_only_fresh_v2_output_and_helper_v3',Path(routing['output']).resolve()==ROOT/'outputs/current104_physical_FD_v2'
      and Path(routing['transaction_helper']).resolve()==ROOT/'src/rough_prefix/probe_transaction_v3.py')
check('protocol_still_GPU_false_until_separate_authorization',newp['GPU_run_authorized_by_protocol'] is False and newp['automatic_retry'] is False and newp['compile_fallback'] is False)
# Recompute the conflict-pattern behavior without importing/calling supervisor.
sup=ast.parse((ROOT/'scripts/run_current104_physical_fd_v2.py').read_text())
fn=next(n for n in sup.body if isinstance(n,ast.FunctionDef) and n.name=='reject_other_workers')
pattern=next(ast.literal_eval(n.value) for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pattern' for t in n.targets))
positive=[r'D:\workspace\probe_current104_physical_fd.py --run',r'D:\workspace\probe_current104_physical_fd_v2.py --run',
          'D:/workspace/probe_current104_physical_fd_v10.py" --run',r'D:\workspace\evaluate_rough_prefix_pen_v2.py --run',
          r'D:\workspace\train_rough_prefix_pen_v5.py --run']
negative=[r'D:\workspace\probe_current104_physical_fd_v2_extra.py --run',r'D:\workspace\probe_current104_physical_fd_vx.py --run',
          r'D:\workspace\probe_current104_physical_fd_v2.pyc --run',r'D:\workspace\run_current104_physical_fd_v2.py --run']
check('supervisor_regex_rejects_old_and_versioned_workers',all(re.search(pattern,x,re.I) is not None for x in positive))
check('supervisor_regex_retains_file_boundary_scope',all(re.search(pattern,x,re.I) is None for x in negative))
bad=[]
for label,p in inv.items():
    if sha(p['path'])!=p['sha256']:bad.append(label)
check('all411_original_execution_pins_still_exact',len(inv)==411 and not bad)
check('new_physical_output_and_watchdog_still_fresh',not (ROOT/'outputs/current104_physical_FD_v2').exists()
      and not (ROOT/'outputs/current104_physical_FD_v2_watchdog').exists())
check('review_no_framework_or_production_import',not any(name in sys.modules for name in ('torch','warp','mujoco','mujoco_warp','rough_prefix.physical_fd_v2')))
result={'schema':'current104_physical_v2_independent_static_delta_review:1','status':'PASS_STATIC_DELTA_NO_BLOCKING_FINDING' if all(checks.values()) else 'FAIL_RETAINED',
        'checks':checks,'passed':sum(checks.values()),'total':len(checks),'files':files,'old_actual_execution_inventory':pin(old_inventory_path),
        'old_pins_count':len(inv),'old_pin_mismatches':bad,'protocol3':pin(newp_path),'reviewer':pin(__file__),
        'blocking_findings':[] if all(checks.values()) else [k for k,v in checks.items() if not v],
        'unchanged_scientific_and_runtime_requirements':['complete native inventory and explicit sidecars','INTEGRATION plus full-array bitexact restore',
            'three fresh baseline comparisons before each perturbation block','external OR capacity ledger','original source action/EMA/actualtarget checks',
            '15ms sensor-force vs20ms poststate phase','same frozen geometry/reference/context/directions','no retry/fallback/quality selection',
            '220s soft/240s owned deadline and same PID+creation/shared-lock lifecycle'],
        'runtime_status':'PENDING_CPU_DEFAULT_AND_NEGATIVES_THEN_SEPARATE_AUTHORIZED_PHYSICAL_ATTEMPT',
        'qualification':'Static equivalence/routing only. Real native restore, baseline and physical directional comparisons not established.',
        'new_source_imports':0,'production_builder_or_entry_runs':0,'helper40_or21_tests_repeated':False,'WM_calls':0,'Env_created':False,'physics_steps':0,'GPU_used':False}
target=OUT/'report.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'failed':result['blocking_findings'],'report':pin(target)},indent=2))
raise SystemExit(0 if all(checks.values()) else 2)
