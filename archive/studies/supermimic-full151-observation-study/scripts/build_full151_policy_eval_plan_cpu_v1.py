"""Freeze fixed-final evaluation of an independently healthy actual new105 run."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import checked,pin,read,write
from full151_native.probe import GATES

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-config',type=Path,required=True);ap.add_argument('--health',type=Path,required=True);ap.add_argument('--tag',required=True);ap.add_argument('--authorize-gpu',action='store_true');a=ap.parse_args()
    if not a.tag.replace('_','').isalnum():raise ValueError('Simple study tag required')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('Metadata-only builder')
    training=read(a.training_config);health=read(a.health)
    if health['schema']!='full151105_checkpoint_CPU_health:1' or health['status']!='PASS_SAVED_ENGINEERING_HEALTH' or health['final_iteration_scope'] is not True:raise ValueError('Completed actual saved health prerequisite')
    config_path=ROOT/f'configs/full151-a1l-policy-eval-{a.tag}-v1.json'
    if config_path.exists():raise FileExistsError('Freeze once; no overwrite')
    source=dict(training['source_pins']);source.pop('protocol',None)
    for key,path in {'evaluation_worker':ROOT/'scripts/execute_full151_policy_eval_v1.py','evaluation_supervisor':ROOT/'scripts/run_full151_policy_eval_v1.py','evaluation_plan_builder':Path(__file__),'policy_CPU_READY':ROOT/'outputs/READY_full151_policy_evaluation_v1.json'}.items():source[key]=pin(path)
    source['training_protocol']=training['protocol'];source['training_config']=pin(a.training_config);source['independent_health']=pin(a.health);source['health_auditor']=health['auditor']
    for path in sorted((ROOT/'src/full151_native').glob('*.py')):
        if not any(Path(v['path']).resolve()==path.resolve() and v['sha256']==pin(path)['sha256'] for v in source.values()):source['eval_full151__'+path.name]=pin(path)
    for value in source.values():checked(value)
    config={'schema':'full151_native_policy_evaluation_run:1','role':'native_policy_eval','method':'A1L','seed':training['seed'],'worlds':1,
        'input_bundle':training['input_bundle'],'output':str(ROOT/f'outputs/full151_native_A1L_policy_eval_{a.tag}_v1'),
        'checkpoint':health['checkpoint'],'checkpoint_iteration':health['iteration'],'training_config':pin(a.training_config),
        'health':pin(a.health),'expected_health_auditor':health['auditor'],'source_pins':source,
        'GPU_run_authorized':bool(a.authorize_gpu),'hard_process_seconds':240,'soft_worker_seconds':180,
        'automatic_retry':False,'compile_fallback':False,'evaluation_gates':GATES}
    from full151_native.bundle import load_bundle
    from full151_native.evaluation import validate_evaluation_inputs
    bundle=load_bundle(checked(training['input_bundle']),training['input_bundle']['sha256'])
    validate_evaluation_inputs(config,bundle)
    write(config_path,config)
    print(json.dumps({'config':pin(config_path),'pins':len(source),'GPU_started':False,'Torch_imported':False,'Warp_imported':False}))

if __name__=='__main__':main()
