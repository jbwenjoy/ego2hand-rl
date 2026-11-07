"""CPU-only authorized budget1500 copy; preserve the qualified v1 sources."""
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
    original=ROOT/'configs/c2-confirmation-evaluation-v1.json'
    assert sha(original)=='884559c40735111ee52fd630c981ecd517bfc041b3c1e94b5ce2e420c66249ee'
    p=copy.deepcopy(read(original));before={k:pin(ROOT/'scripts'/v) for k,v in {'worker':'evaluate_c2_portfolio.py','queue':'run_c2_portfolio_queue.py'}.items()}
    newnames={'worker':'evaluate_c2_portfolio_v2.py','queue':'run_c2_portfolio_queue_v2.py'}
    out=ROOT/'outputs/budget_v2_preparation';out.mkdir(parents=True,exist_ok=True)
    for kind,spec in before.items():
        text=Path(spec['path']).read_text(encoding='utf-8');updated=text.replace('600','1500')
        if kind=='queue':updated=updated.replace('configs/c2-confirmation-evaluation-v1.json','configs/c2-confirmation-evaluation-v2.json').replace('scripts/evaluate_c2_portfolio.py','scripts/evaluate_c2_portfolio_v2.py')
        new=ROOT/'scripts'/newnames[kind];new.write_text(updated,encoding='utf-8');p['source_pins'][kind]=pin(new)
        (out/f'{kind}.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),updated.splitlines(True),fromfile=spec['path'],tofile=str(new))),encoding='utf-8')
    p.update(whole_queue_budget_seconds=1500,output_root=str(ROOT/'outputs/confirmation_evaluation_v2'),
       status='CPU_prepared_budget1500_GPU_not_authorized',
       whole_budget_enforcement='1500s scheduling/stop-request budget with45s cleanup reserve; independent child timer<=45s; not OS hard realtime',
       budget_change_evidence={'qualification_state':pin(ROOT/'outputs/format103_qualification_v1/state.json'),
         'qualified_v1_plan':pin(ROOT/'configs/format103-qualification-v1.json'),'unrun_v1_plan':pin(original),
         'reason':'Measured26.25s single child;27x45s+45s reserve=1260s;1500s finite guard retains startup/health room',
         'scientific_inputs_changed':False})
    path=ROOT/'configs/c2-confirmation-evaluation-v2.json'
    for j in p['jobs']:
        target=Path(p['output_root'])/j['job_id'];j['output_dir']=str(target);j['wrapper_metadata_dir']=str(target)+'_wrapper'
        args=j['original_evaluator_args'];args[args.index('--output')+1]=str(target)
        cmd=j['command'];cmd[2]=p['source_pins']['worker']['path'];cmd[cmd.index('--plan')+1]=str(path);cmd[cmd.index('--output')+1]=str(target)
    checker=ROOT/'scripts/check_confirmation_queue_cpu_v2.py'
    test=(ROOT/'scripts/check_confirmation_queue_cpu.py').read_text().replace('scripts/run_c2_portfolio_queue.py','scripts/run_c2_portfolio_queue_v2.py').replace('configs/c2-confirmation-evaluation-v1.json','configs/c2-confirmation-evaluation-v2.json').replace("ROOT/'outputs/preparation_v1'","ROOT/'outputs/budget_v2_preparation'")
    test=test.replace('def factory(p):', '''checks['effective_budget1500_child45_reserve45']=p['whole_queue_budget_seconds']==1500 and p['per_child_cap_seconds']==p['cleanup_reserve_seconds']==45
with tempfile.TemporaryDirectory() as tmp:
    bad=copy.deepcopy(p);bad['whole_queue_budget_seconds']=600
    wrong=Path(tmp)/'wrong_budget.json';wrong.write_text(json.dumps(bad))
    checks['stale600_budget_rejected']=rejected(lambda:w.validate(wrong))
    bad=copy.deepcopy(p);bad['per_child_cap_seconds']=90
    wrong=Path(tmp)/'wrong_child.json';wrong.write_text(json.dumps(bad))
    checks['child90_rejected']=rejected(lambda:w.validate(wrong))
def factory(p):''')
    checker.write_text(test,encoding='utf-8');p['source_pins']['queue_cpu_checker']=pin(checker)
    write(path,p)
    def factory(p):return ast.dump(next(n for n in ast.walk(ast.parse(Path(p).read_text())) if isinstance(n,ast.FunctionDef) and n.name=='factory'),include_attributes=False)
    checks={'worker_diff_exact_budget_only':Path(p['source_pins']['worker']['path']).read_text()==Path(before['worker']['path']).read_text().replace('600','1500'),
      'factory_AST_exact_qualified':factory(p['source_pins']['worker']['path'])==factory(before['worker']['path']),
      'original_worker_unchanged':sha(before['worker']['path'])==before['worker']['sha256'],
      'original_queue_unchanged':sha(before['queue']['path'])==before['queue']['sha256'],
      'original_plan_unchanged':sha(original)=='884559c40735111ee52fd630c981ecd517bfc041b3c1e94b5ce2e420c66249ee',
      'qualified_plan_unchanged':sha(ROOT/'configs/format103-qualification-v1.json')=='1c2dd5dcdb33f519eeb4c60044606f51ef2308cde1b4aa44acc778421138621b',
      'same_models_and_expected_contracts':p['models']==read(original)['models'],
      'same_frozen27_schedule':all(all(j[k]==v for k,v in old.items() if k not in ('output_dir','wrapper_metadata_dir','original_evaluator_args','command')) for j,old in zip(p['jobs'],read(original)['jobs'])),
      'single_truth_budget1500':p['whole_queue_budget_seconds']==1500 and p['per_child_cap_seconds']==p['cleanup_reserve_seconds']==45}
    write(out/'budget_copy_contract.json',{'status':'passed' if all(checks.values()) else 'failed','checks':checks,
      'main_plan':pin(path),'old_sources':before,'new_sources':{k:p['source_pins'][k] for k in before},
      'unchanged_operations':{k:p['source_pins'][k] for k in ('evaluator','checkpoint103_contract','pure_projection','lifecycle','confirmation_analysis')},
      'GPU_started':False})
    print(json.dumps({'plan':pin(path),'checks':checks}))

if __name__=='__main__':main()
