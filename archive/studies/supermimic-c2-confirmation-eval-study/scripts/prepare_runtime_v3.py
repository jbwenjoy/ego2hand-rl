"""CPU-only separate metadata-I/O hardening attempt; no process execution."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    old=ROOT/'configs/c2-confirmation-evaluation-v2.json';assert sha(old)=='99dd2b646d18ed1efc0a386bdd5e09706b5fe6e207b57c067bf91b3f4e8d22e3'
    p=copy.deepcopy(read(old));out=ROOT/'outputs/runtime_v3_preparation';out.mkdir(parents=True,exist_ok=True)
    oldworker=ROOT/'scripts/evaluate_c2_portfolio_v2.py';oldqueue=ROOT/'scripts/run_c2_portfolio_queue_v2.py'
    worker=oldworker.read_text();assert worker.count('    started = time.perf_counter()')==1
    worker=worker.replace('    started = time.perf_counter()',
      '    started = time.perf_counter()\n    replace_with_bounded_retry=import_path(ROOT/"src/runtime_io.py","bounded_runtime_io").replace_with_bounded_retry')
    worker=worker.replace('        os.replace(temporary,metadata_dir/"runtime.json")','        replace_with_bounded_retry(temporary,metadata_dir/"runtime.json",max_seconds=.75)')
    newworker=ROOT/'scripts/evaluate_c2_portfolio_v3.py';newworker.write_text(worker,encoding='utf-8')
    queue=oldqueue.read_text().replace('configs/c2-confirmation-evaluation-v2.json','configs/c2-confirmation-evaluation-v3.json').replace('scripts/evaluate_c2_portfolio_v2.py','scripts/evaluate_c2_portfolio_v3.py')
    queue=queue.replace('    lifecycle.ROOT=STUDY','    lifecycle.ROOT=STUDY\n    bounded_io=import_path(plan["source_pins"]["runtime_io_helper"]["path"],"bounded_supervisor_io")\n    lifecycle.write=bounded_io.write_json')
    newqueue=ROOT/'scripts/run_c2_portfolio_queue_v3.py';newqueue.write_text(queue,encoding='utf-8')
    p['source_pins'].update(worker=pin(newworker),queue=pin(newqueue),runtime_io_helper=pin(ROOT/'src/runtime_io.py'),runtime_io_CPU_checker=pin(ROOT/'scripts/check_runtime_io_cpu.py'))
    p.update(status='CPU_prepared_bounded_metadata_IO_GPU_not_authorized',output_root=str(ROOT/'outputs/confirmation_evaluation_v3'),
      engineering_IO={'retry_scope':'Only atomic metadata rename PermissionError with winerrorNone/5/32; same serialized payload and temporary file',
        'max_retry_seconds':.75,'max_replace_calls':32,'process_or_physics_retry':False,
        'supervisor_metadata':'Only this queue module replaces imported lifecycle.write with the same bounded atomic JSON primitive; original lifecycle source/timer/cleanup unchanged',
        'previous_attempt':pin(ROOT/'outputs/confirmation_evaluation_v2/state.json'),
        'incident_cause':'Unresolved; Windows reader/delete-sharing race is a reproduced possible mechanism, not a proven sole cause',
        'previous_first_job_role':'INVALID engineering attempt;26NOT_RUN; never pooled or selected into this new cohort'})
    path=ROOT/'configs/c2-confirmation-evaluation-v3.json'
    for j in p['jobs']:
        target=Path(p['output_root'])/j['job_id'];j['output_dir']=str(target);j['wrapper_metadata_dir']=str(target)+'_wrapper'
        args=j['original_evaluator_args'];args[args.index('--output')+1]=str(target)
        cmd=j['command'];cmd[2]=str(newworker);cmd[cmd.index('--plan')+1]=str(path);cmd[cmd.index('--output')+1]=str(target)
    checker=ROOT/'scripts/check_confirmation_queue_cpu_v3.py'
    test=(ROOT/'scripts/check_confirmation_queue_cpu_v2.py').read_text().replace('scripts/run_c2_portfolio_queue_v2.py','scripts/run_c2_portfolio_queue_v3.py').replace('configs/c2-confirmation-evaluation-v2.json','configs/c2-confirmation-evaluation-v3.json').replace("ROOT/'outputs/budget_v2_preparation'","ROOT/'outputs/runtime_v3_preparation'")
    test=test.replace("checks['9_models_seeds456_3arms']", "checks['supervisor_writer_bounded_and_lifecycle_original']=_.write.__name__=='write_json' and w.sha(p['source_pins']['lifecycle']['path'])=='40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'\nchecks['9_models_seeds456_3arms']")
    checker.write_text(test,encoding='utf-8')
    p['source_pins']['queue_cpu_checker']=pin(checker);write(path,p)
    for kind,before,after in [('worker',oldworker,newworker),('queue',oldqueue,newqueue)]:
        (out/f'{kind}.diff').write_text(''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile=str(before),tofile=str(after))),encoding='utf-8')
    def node(p,name):return ast.dump(next(n for n in ast.walk(ast.parse(Path(p).read_text())) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    checks={'factory_AST_exact_qualified':node(newworker,'factory')==node(ROOT/'scripts/evaluate_c2_portfolio.py','factory'),
      'step_AST_exact':node(newworker,'step')==node(oldworker,'step'),
      'process_AST_exact':node(newworker,'process')==node(oldworker,'process'),
      'same_models_expected':p['models']==read(old)['models'],
      'same1500_child45_reserve45':p['whole_queue_budget_seconds']==1500 and p['per_child_cap_seconds']==p['cleanup_reserve_seconds']==45,
      'old_worker2_unchanged':sha(oldworker)=='1ee9a8a79f47ddf6c6cb019c5d06a5c873879c9df1b853a59f40933f41a4575e',
      'old_queue2_unchanged':sha(oldqueue)=='6b5e6da306f90db77f613995140c6768d3241f497764f68e9fd482b8ef3b5719',
      'old_plan2_unchanged':sha(old)=='99dd2b646d18ed1efc0a386bdd5e09706b5fe6e207b57c067bf91b3f4e8d22e3'}
    write(out/'runtime_copy_contract.json',{'status':'passed' if all(checks.values()) else 'failed','checks':checks,'plan':pin(path),
      'sources':{k:p['source_pins'][k] for k in ('worker','queue','runtime_io_helper','evaluator','checkpoint103_contract','pure_projection','confirmation_analysis')},'GPU_started':False})
    print(json.dumps({'plan':pin(path),'checks':checks}))

if __name__=='__main__':main()
