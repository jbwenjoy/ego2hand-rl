"""Separate one-world new105 learned-policy evaluation; CPU metadata by default."""
from pathlib import Path
import argparse,datetime,json,os,shutil,sys,time,traceback,zipfile
STARTED=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from full151_native.io import read,checked,pin,sha,canonical,jsonable
from full151_native.metadata import write

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def preflight(path):
    p=read(path)
    allowed={'schema','role','method','seed','worlds','input_bundle','output','checkpoint','checkpoint_iteration','training_config','health','expected_health_auditor','source_pins','GPU_run_authorized','hard_process_seconds','soft_worker_seconds','automatic_retry','compile_fallback','evaluation_gates'}
    if set(p)!=allowed or p['schema']!='full151_native_policy_evaluation_run:2' or p['role']!='native_policy_eval' or p['method'] not in ('A1L','P_ONLY','A2'):raise ValueError('Separate learned A1L evaluation only')
    if type(p['GPU_run_authorized']) is not bool or p['automatic_retry'] is not False or p['compile_fallback'] is not False or (p['soft_worker_seconds'],p['hard_process_seconds'])!=(180,240):raise ValueError('Finite explicit no-retry policy evaluation required')
    for value in p['source_pins'].values():checked(value)
    required=[Path(__file__),ROOT/'scripts/run_full151_policy_eval_v2.py',*(ROOT/'src/full151_native'/n for n in ('__init__.py','bundle.py','controller.py','runtime.py','native.py','probe.py','io.py','metadata.py','checkpoint.py','evaluation_v2.py','policy_probe.py')),ROOT/'vendor/runtime_io.py',ROOT/'vendor/measured_contacts.py']
    for source in required:
        if not any(Path(value['path']).resolve()==source.resolve() and value['sha256']==sha(source) for value in p['source_pins'].values()):raise ValueError('Unpinned evaluation dependency '+str(source))
    for name in ('training_config','health','expected_health_auditor'):
        checked(p[name])
        if not any(p[name]==v for v in p['source_pins'].values()):raise ValueError('Unarchived evaluation prerequisite '+name)
    from full151_native.bundle import load_bundle
    from full151_native.evaluation_v2 import validate_evaluation_inputs
    from full151_native.probe import GATES
    if canonical(p['evaluation_gates'])!=canonical(GATES):raise ValueError('Original fullhorizon criteria differ')
    bundle=load_bundle(checked(p['input_bundle']),p['input_bundle']['sha256'])
    evidence=validate_evaluation_inputs(p,bundle)
    if not Path(p['output']).resolve().is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Study-owned fresh evaluation output required')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('Default metadata preflight imported device frameworks')
    return p,bundle,{'status':'CPU_INPUT_SOURCE_PREFLIGHT_ONLY','config_sha256':sha(path),'entry_sha256':sha(__file__),'role':p['role'],'method':p['method'],'checkpoint':p['checkpoint'],'checkpoint_iteration':p['checkpoint_iteration'],'health':p['health'],'pinned_files':len(p['source_pins']),'Torch_imported':False,'Warp_imported':False,'GPU_started':False,'physics_steps':0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();path=a.config.resolve(strict=True)
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    p,bundle,proof=preflight(path)
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['GPU_run_authorized'] is not True:raise ValueError('Separate explicit hash-bound GPU plan required')
    out=Path(p['output']).resolve()
    if a.output is None or a.output.resolve()!=out or out.exists():raise ValueError('Exact fresh output required, no restart')
    out.mkdir(parents=True,exist_ok=False);dest=out/'source';dest.mkdir()
    with zipfile.ZipFile(dest/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for i,(name,value) in enumerate(p['source_pins'].items()):archive.write(checked(value),f'{i:04d}_{Path(value["path"]).name}')
    write(dest/'inventory.json',p['source_pins']);shutil.copy2(path,out/'config.json');write(out/'CPU_preflight.json',proof)
    runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(path),'source_sha256':sha(__file__),'role':p['role'],'method':p['method'],'optimizer_updates':0,'physics_steps':None,'physics_step_scope':'Episode API returns only; constructor/reset/graph UNKNOWN','automatic_retry':False}
    write(out/'runtime.json',runtime)
    def emit(**updates):runtime.update(updates);runtime['worker_elapsed_seconds']=time.perf_counter()-STARTED;write(out/'runtime.json',runtime)
    try:
        from full151_native.evaluation_v2 import run_evaluation
        result=run_evaluation({**p,'_config_sha256':sha(path),'_worker_started_perf':STARTED},bundle,out,emit)
        evaluation=read(out/'evaluation.json')
        if evaluation.get('complete') is not True or result['cleanup_errors']:raise ValueError('Full151 learned-policy episode engineering incomplete')
        write(out/'completion.json',{'schema':'full151_native_policy_evaluation_completion:2','config_sha256':sha(path),'result':result,'quality_is_not_engineering_gate':True})
        runtime.update(status='completed_native_policy_evaluation',result_status=result['status'],physics_steps=evaluation['physical_steps'],control_intervals=evaluation['control_intervals'])
    except BaseException as e:
        runtime.update(status='failed_engineering_partial',error=repr(e));write(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'no_retry':True});raise
    finally:
        runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(out/'runtime.json',runtime)
    print(json.dumps(jsonable(result),indent=2))

if __name__=='__main__':main()

