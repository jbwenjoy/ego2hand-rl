"""Real installed play-config construction and native-A0 termination parity.

Builds configuration objects only. No Manager/Env, simulator, reset or step.
"""
from __future__ import annotations
import argparse,ast,copy,dataclasses,difflib,hashlib,inspect,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def semantic(value):
    if value is None or type(value) in (str,bool,int,float):return value
    if callable(value):return {'callable':value.__module__+'.'+value.__qualname__}
    if dataclasses.is_dataclass(value):return {'dataclass':type(value).__module__+'.'+type(value).__qualname__,'fields':{f.name:semantic(getattr(value,f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value,dict):return {str(k):semantic(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return {type(value).__name__:[semantic(v) for v in value]}
    if isinstance(value,slice):return {'slice':[value.start,value.stop,value.step]}
    raise TypeError('Unexpected config semantic value '+type(value).__name__)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);snapshot=out/'source';snapshot.mkdir()
    before=ROOT/'src/rough_prefix/evaluation.py';after=ROOT/'src/rough_prefix/evaluation_v2.py';native=ROOT/'src/rough_prefix/native_v2.py'
    files=[before,after,native,Path(__file__)];pins={}
    for p in files:(snapshot/p.name).write_bytes(p.read_bytes());pins[p.name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (out/'evaluation_v1_to_v2.diff').write_text(''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile='evaluation.py_frozen_v1',tofile='evaluation_v2.py')),encoding='utf-8')
    import torch
    from rough_prefix.evaluation_v2 import configure_full_horizon_terminations
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin import mdp
    from mjlab.managers import TerminationTermCfg,SceneEntityCfg
    from rough_prefix.io import read
    bundle=read(ROOT/'outputs/input_bundle_v1/bundle.json');ref=bundle['native_reference']
    def cfg():return make_env_cfg(1,50,play=True,recipe_path=Path(ref['recipe_path']),clips_root=Path(ref['clips_root']))
    eval_cfg=cfg();a0_cfg=cfg();checks=[]
    def check(name,value):checks.append({'name':name,'passed':bool(value)})
    original_keys=set(eval_cfg.terminations)
    check('actual_play_config_has_exact_four_terms',original_keys=={'time_out','unstable','object_position_error','reference_end'})
    check('actual_play_already_removed_orientation','object_orientation_error' not in original_keys)
    # Execute only the frozen native-v2 config mutation statements, not its
    # run function, imports/device choice, Env constructor or probe.
    native_tree=ast.parse(native.read_text());run=next(n for n in native_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_native')
    selected=[]
    for node in run.body:
        text=ast.unparse(node)
        if isinstance(node,ast.Assign) and text.startswith("env_cfg.terminations['object_"):
            selected.append(copy.deepcopy(node))
        elif isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='is_a0' and any('env_cfg.terminations.pop' in ast.unparse(child) for child in node.body):
            selected.append(copy.deepcopy(node))
    check('native_A0_only_three_config_statements_selected',len(selected)==3)
    allowed=ast.Module(body=selected,type_ignores=[]);ast.fix_missing_locations(allowed)
    (out/'native_A0_config_statements.py').write_text(ast.unparse(allowed)+'\n')
    exec(compile(allowed,str(native),'exec'),{'env_cfg':a0_cfg,'TerminationTermCfg':TerminationTermCfg,'SceneEntityCfg':SceneEntityCfg,'mdp':mdp,'is_a0':True})
    configure_full_horizon_terminations(eval_cfg)
    final=semantic(eval_cfg.terminations);expected=semantic(a0_cfg.terminations)
    check('final_three_terms_exact',set(eval_cfg.terminations)=={'time_out','unstable','reference_end'})
    check('native_A0_final_full_dataclass_semantics_equal',final==expected)
    for name in ('time_out','unstable','reference_end'):
        x,y=eval_cfg.terminations[name],a0_cfg.terminations[name]
        check(name+'_function_identity',x.func is y.func)
        check(name+'_params_equal',semantic(x.params)==semantic(y.params))
        check(name+'_timeout_semantics_equal',type(x.time_out) is bool and x.time_out is y.time_out)
    check('timeout_and_reference_end_are_timeout_only',eval_cfg.terminations['time_out'].time_out and eval_cfg.terminations['reference_end'].time_out and not eval_cfg.terminations['unstable'].time_out)
    unknown=cfg();unknown.terminations['unexpected']=copy.deepcopy(unknown.terminations['unstable']);keys=set(unknown.terminations)
    try:configure_full_horizon_terminations(unknown)
    except ValueError:check('unexpected_term_rejected_without_partial_mutation',set(unknown.terminations)==keys)
    else:check('unexpected_term_rejected_without_partial_mutation',False)
    missing=cfg();missing.terminations.pop('object_position_error');keys=set(missing.terminations)
    try:configure_full_horizon_terminations(missing)
    except ValueError:check('missing_position_rejected_without_partial_mutation',set(missing.terminations)==keys)
    else:check('missing_position_rejected_without_partial_mutation',False)
    def run_without_termination_calls(path,new):
        tree=ast.parse(path.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_evaluation')
        run.body=[n for n in run.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ((new and isinstance(n.value.func,ast.Name) and n.value.func.id=='configure_full_horizon_terminations') or (not new and 'env_cfg.terminations.pop(' in ast.unparse(n))))]
        return ast.dump(run,include_attributes=False)
    check('all_other_evaluation_run_AST_unchanged',run_without_termination_calls(before,False)==run_without_termination_calls(after,True))
    check('old_evaluation_source_unchanged',hashlib.sha256(before.read_bytes()).hexdigest()==pins['evaluation.py']['sha256'])
    config_source=Path(inspect.getsourcefile(make_env_cfg));pins['installed_make_env_cfg']={'path':str(config_source),'sha256':hashlib.sha256(config_source.read_bytes()).hexdigest()}
    check('CUDA_uninitialized_no_environment_run',not torch.cuda.is_initialized())
    report={'schema':'rough_policy_evaluation_termination_CPU_contract:2','status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'source_pins':pins,'actual_factory_calls':4,'factory_mode':{'num_envs':1,'control_hz':50,'play':True},'native_A0_config_AST_statements_executed':3,'final_termination_semantics':final,'native_A0_expected_semantics':expected,'Torch_imported':'torch' in sys.modules,'Warp_imported_transitively':'warp' in sys.modules,'CUDA_initialized':torch.cuda.is_initialized(),'Env_constructed':False,'physics_steps':0,'optimizer_objects_created':0,'scope':'Actual installed config factory + three frozen native-v2 A0 configuration statements only; no Env/Manager/device/physics/probe execution.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'checks':report['check_count'],'passed':report['passed'],'failed':[c for c in checks if not c['passed']]},indent=2));raise SystemExit(report['status']!='PASS')


if __name__=='__main__':main()
