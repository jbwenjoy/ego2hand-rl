"""CPU-only default preflight; explicitly supervised format104 physical evaluation."""
from __future__ import annotations
import argparse, json, os, shutil, sys, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
STARTED=time.perf_counter(); ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'vendor'))
from rough_prefix.io import read, checked, pin, sha, jsonable
from rough_prefix.bundle_v2 import load_bundle
from rough_prefix.metadata import write


def preflight(path):
    p=read(path)
    required={'schema','role','method','seed','input_bundle','output','checkpoint','checkpoint_iteration',
              'training_config','health_report','actor_load_report','source_pins','protocol',
              'GPU_run_authorized','worlds','hard_process_seconds','soft_worker_seconds',
              'automatic_retry','compile_fallback','evaluation_gates'}
    if set(p)!=required or p['schema']!='rough_prefix_policy_evaluation_run:1' or p['role']!='native_policy_eval':
        raise ValueError('Separate strict evaluation schema required')
    if p['method'] not in ('A1L','P_ONLY','A2') or type(p['seed']) is not int or p['seed']<0:
        raise ValueError('Wrong arm/seed')
    if type(p['worlds']) is not int or p['worlds']!=1 or p['automatic_retry'] is not False or p['compile_fallback'] is not False:
        raise ValueError('Only one-world no-retry/no-fallback evaluation')
    for value in p['source_pins'].values(): checked(value)
    for value in (p['checkpoint'],p['training_config'],p['health_report'],p['actor_load_report']): checked(value)
    protocol=read(checked(p['protocol']))
    if p['source_pins']['protocol']!=p['protocol'] or protocol['schema']!='rough_prefix_policy_evaluation_protocol:1':
        raise ValueError('Evaluation protocol not source-bound')
    if p['hard_process_seconds']!=protocol['hard_process_seconds'] or p['soft_worker_seconds']!=protocol['soft_worker_seconds']:
        raise ValueError('Finite evaluation wall budget differs')
    if p['checkpoint_iteration'] not in protocol['checkpoint_iterations'] or p['seed'] not in protocol['allowed_seeds']:
        raise ValueError('Undeclared checkpoint selection')
    from rough_prefix.probe import GATES
    if p['evaluation_gates']!=GATES or protocol['evaluation_gates']!=GATES:
        raise ValueError('Original numerical physical criteria changed')
    b=load_bundle(checked(p['input_bundle']),p['input_bundle']['sha256'])
    if protocol['input_bundle']!=p['input_bundle']:
        raise ValueError('Evaluation data/source differs')
    train=read(checked(p['training_config'])); health=read(checked(p['health_report']))
    load=read(checked(p['actor_load_report']))
    if p['source_pins'].get('training_protocol')!=train['protocol']:
        raise ValueError('Executed checkpoint-loader training protocol missing from archive')
    if train['method']!=p['method'] or train['seed']!=p['seed'] or train['input_bundle']!=p['input_bundle']:
        raise ValueError('Training source/data identity differs')
    if Path(train['output']).resolve()/'train/model_final.pt' != Path(p['checkpoint']['path']).resolve():
        raise ValueError('Only declared final checkpoint is selected')
    if train['training']['max_iterations']!=p['checkpoint_iteration']:
        raise ValueError('Wrong declared final iteration')
    if health.get('status')!='PASS_SAVED_ENGINEERING_HEALTH' or not health.get('checks') or not all(health['checks'].values()):
        raise ValueError('Actual checkpoint health not qualified')
    if load.get('status')!='PASS_ACTUAL_EVALUATION_ACTOR_CPU_LOAD' or load['passed']!=load['check_count']:
        raise ValueError('Actual Actor CPU consumption not qualified')
    # Both prerequisite receipts must name the selected bytes, not merely pass.
    def contains_pin(value,wanted):
        if isinstance(value,dict):
            if value==wanted:return True
            return any(contains_pin(v,wanted) for v in value.values())
        return isinstance(value,list) and any(contains_pin(v,wanted) for v in value)
    if not contains_pin(health,p['checkpoint']) or not contains_pin(load,p['checkpoint']):
        raise ValueError('Prerequisite receipt not bound to selected checkpoint')
    closure=('io.py','metadata.py','bundle.py','bundle_v2.py','native_v2.py','runtime_v2.py',
             'controller.py','probe.py','policy_probe.py','evaluation.py','checkpoint_v2.py')
    for source in [Path(__file__),*[ROOT/'src/rough_prefix'/name for name in closure],
                   ROOT/'vendor/runtime_io.py',ROOT/'vendor/measured_contacts.py']:
        if not any(Path(v['path']).resolve()==source.resolve() and v['sha256']==sha(source) for v in p['source_pins'].values()):
            raise ValueError('Unfrozen evaluation dependency '+str(source))
    output=Path(p['output']).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Owned output required')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):
        raise AssertionError('CPU preflight imported GPU frameworks')
    return p,b,{'status':'CPU_INPUT_SOURCE_PREFLIGHT_ONLY','config_sha256':sha(path),
                'entry_sha256':sha(__file__),'role':p['role'],'method':p['method'],
                'checkpoint':p['checkpoint'],'Torch_imported':False,'Warp_imported':False,
                'GPU_started':False,'physics_steps':0,'pinned_files':len(p['source_pins'])}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    path=a.config.resolve(strict=True);p,b,proof=preflight(path)
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['GPU_run_authorized'] is not True:raise ValueError('Separate hash-bound GPU authorization absent')
    output=Path(p['output']).resolve()
    if a.output is None or a.output.resolve()!=output or output.exists():raise ValueError('Exact fresh output required')
    output.mkdir(parents=True,exist_ok=False);source=output/'source';source.mkdir()
    with zipfile.ZipFile(source/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for i,(key,value) in enumerate(p['source_pins'].items()):
            file=checked(value);z.write(file,f'{i:04d}_{file.name}')
    write(source/'inventory.json',p['source_pins']);shutil.copy2(path,output/'config.json')
    write(output/'CPU_preflight.json',proof)
    runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),
             'started_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(path),
             'source_sha256':sha(__file__),'role':p['role'],'method':p['method'],
             'physics_steps':None,'physics_step_scope':'Recorded evaluation solver steps; UNKNOWN before probe counters; excludes startup graph capture/forwards',
             'optimizer_updates':0,'automatic_retry':False}
    write(output/'runtime.json',runtime)
    def emit(**updates):
        runtime.update(updates);runtime['worker_elapsed_seconds']=time.perf_counter()-STARTED
        write(output/'runtime.json',runtime)
    try:
        from rough_prefix.evaluation import run_evaluation
        result=run_evaluation({**p,'_worker_started_perf':STARTED},b,output,emit)
        write(output/'evaluation_completion.json',{'schema':'rough_prefix_policy_evaluation_completion:1',
              'config_sha256':sha(path),'result':result})
        runtime.update(status='completed_native_policy_evaluation',result_status=result['status'],
                       physics_steps=result['evaluation']['physical_steps'])
    except BaseException as exc:
        runtime.update(status='failed_engineering_partial',error=repr(exc))
        write(output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'no_retry':True})
        raise
    finally:
        runtime.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-STARTED)
        write(output/'runtime.json',runtime)
    print(json.dumps(jsonable(result),indent=2))

if __name__=='__main__':main()
