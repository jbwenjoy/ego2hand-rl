"""Small stdlib queue/identity contracts, not simulator or task tests."""
import ast
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q',ROOT/'scripts/run_c2_portfolio_queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
path=ROOT/'configs/c2-confirmation-evaluation-v1.json'
p,w,_=q.preflight(path)
checks={'stdlib_only_no_gpu_import':not any(k in sys.modules for k in ('torch','warp','mujoco_warp'))}
def rejected(f):
    try:f()
    except (ValueError,KeyError):return True
    return False
checks['9_models_seeds456_3arms']={(m['method'],m['training_seed']) for m in p['models']}=={(a,s) for a in ('A1L','P_ONLY','A2') for s in (4,5,6)}
checks['all27_margin20'] = all(j['margin_rad']==.020 and j['margin_mrad']==20 for j in p['jobs'])
checks['3_fresh_processes_per_model']=all(sum(j['model_id']==m['model_id'] for j in p['jobs'])==3 for m in p['models'])
positive=True
for job in p['jobs']:
    m=w.model_for_job(p,job)
    meta={'checkpoint_sha256':m['checkpoint_sha256'],'seed':0,'checkpoint_training_config':m['expected']['native_config'],
      'checkpoint_iteration':2000,'checkpoint_format_version':103,'checkpoint_contract_sha256':m['contract_file']['sha256'],
      'checkpoint_controller':m['expected']['controller'],'runtime_reference_signature':m['expected']['reference_signature']}
    record={'schema':'C2_trained_confirmation:1','training_checkpoint_format':103,'checkpoint_contract_sha256':m['contract_file']['sha256'],
      'trained_method':m['method'],'model_id':m['model_id'],'training_seed':m['training_seed'],'checkpoint_sha256':m['checkpoint_sha256']}
    q.validate_policy_records(p,job,w,meta,record,record)
checks['27_positive_record_contracts']=positive
for name,change in [('wrong_format',{'checkpoint_format_version':3}),('wrong_checkpoint',{'checkpoint_sha256':'wrong'}),
 ('wrong_training_seed',{'checkpoint_training_config':{'seed':1}}),('wrong_reference_signature',{'runtime_reference_signature':'wrong'}),
 ('wrong_contract_file',{'checkpoint_contract_sha256':'wrong'})]:
    checks[name+'_rejected']=rejected(lambda c=change:q.validate_policy_records(p,job,w,{**meta,**c},record,record))
checks['wrong_arm_record_rejected']=rejected(lambda:q.validate_policy_records(p,job,w,meta,{**record,'trained_method':'wrong'},record))
cap={'status':'OK','overflow_bitmask':0,'flags_by_world':[0],'resets_during_rollout':0,'read_count':1,'read_at_control_intervals':150,
 'actual_world_count':1,'job_id':job['job_id'],'plan_sha256':w.sha(path),'worker_sha256':p['source_pins']['worker']['sha256'],
 'checkpoint_sha256':m['checkpoint_sha256'],'training_seed':m['training_seed'],'trained_method':m['method'],'model_id':m['model_id'],
 'training_checkpoint_format':103,'checkpoint_contract_sha256':m['contract_file']['sha256']}
runtimeplan={**p,'plan_sha256_runtime':w.sha(path)}
q.validate_capacity(runtimeplan,job,w,{'resets_during_rollout':0},cap)
checks['capacity_positive']=True
for name,change in [('overflow',{'status':'OVERFLOW','overflow_bitmask':1,'flags_by_world':[1]}),('unknown',{'status':'UNKNOWN','overflow_bitmask':None}),
 ('wrong_capacity_contract',{'checkpoint_contract_sha256':'wrong'}),('wrong_capacity_seed',{'training_seed':1})]:
    checks[name+'_rejected']=rejected(lambda c=change:q.validate_capacity(runtimeplan,job,w,{'resets_during_rollout':0},{**cap,**c}))
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);bad=copy.deepcopy(p);bad['jobs'][0],bad['jobs'][1]=bad['jobs'][1],bad['jobs'][0]
    wrong=root/'wrong.json';wrong.write_text(json.dumps(bad))
    checks['frozen_schedule_change_rejected']=rejected(lambda:w.validate(wrong))
    class Life:
        def __init__(self,fail):self.calls=0;self.fail=fail
        def gpu_lock(self):return contextlib.nullcontext()
        def reject_project_gpu_workers(self):pass
        def write(self,p,v):Path(p).write_text(json.dumps(v))
        def watch(self,*a,**kw):
            assert 0<kw['hard_seconds']<=45
            self.calls+=1;return {'status':'failed' if self.fail else 'completed','exit_code':7 if self.fail else 0}
    saved=q.health;q.health=lambda *_:{'task_success':False,'quality_used_for_control':False}
    try:
        for fail in (True,False):
            pp={**p,'output_root':str(root/str(fail))};life=Life(fail)
            try:q.run(path,pp,w,life)
            except ValueError:assert fail
            state=w.read(Path(pp['output_root'])/'state.json')
            checks['failstop_no_retry' if fail else 'taskfalse_all27_continue']=life.calls==(1 if fail else 27) and state['status']==('engineering_failed' if fail else 'completed_27_jobs')
    finally:q.health=saved
def factory(p):return ast.dump(next(n for n in ast.walk(ast.parse(Path(p).read_text())) if isinstance(n,ast.FunctionDef) and n.name=='factory'),include_attributes=False)
checks['same_factory_AST']=factory(w.__file__)==factory('D:/workspace/supermimic-c2-020-portfolio-study/scripts/evaluate_c2_portfolio.py')
checks['pure_bytes_same']=w.sha(ROOT/'src/target_inset.py')==p['source_pins']['pure_projection']['sha256']=='2d88859c62d2833633f8b09a3be42d11c943e0c3422095e1d941100cb1bf86f4'
out=ROOT/'outputs/preparation_v1';out.mkdir(parents=True,exist_ok=True)
report={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'count':len(checks),'GPU_started':False,
 'scope':'stdlib fakequeue/metadata/order/capacity contracts only; actual103 header tests independent',
 'plan_sha256':w.sha(path),'worker_sha256':w.sha(w.__file__),'queue_sha256':w.sha(q.__file__)}
(out/'queue_cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
