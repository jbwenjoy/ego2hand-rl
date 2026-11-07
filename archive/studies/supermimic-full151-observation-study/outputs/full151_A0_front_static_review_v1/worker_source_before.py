"""Default CPU preflight; explicit frozen --run starts one native full151 A0."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,sys,time,traceback,zipfile
STARTED=time.perf_counter()
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from full151_native.io import read,checked,pin,sha,canonical,jsonable
from full151_native.metadata import write,MODE,VARIANT,FORMAT_VERSION


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def preflight(path):
    p=read(path)
    allowed={'schema','role','method','seed','input_bundle','output','training','contact_binding','source_pins','protocol','GPU_run_authorized','training_authorized','hard_process_seconds','soft_worker_seconds','automatic_retry','compile_fallback','controller','evaluation_gates'}
    if set(p)!=allowed or p['schema']!='full151_knownrough_native_A0_run:1' or p['role']!='native_a0' or p['method']!='A0':raise ValueError('This entry supports only one newfull151 A0, not training')
    if type(p['seed']) is not int or p['seed']!=0 or p['automatic_retry'] is not False or p['compile_fallback'] is not False or type(p['GPU_run_authorized']) is not bool:raise ValueError('Seed0/no retry/no compile fallback/frozen GPU boolean required')
    if p['training']!={'num_envs':1} or type(p['training']['num_envs']) is not int or p['contact_binding'] is not None or p['training_authorized'] is not False:raise ValueError('A0 must not instantiate an optimizer or contact prior')
    if p['hard_process_seconds']!=240 or p['soft_worker_seconds']!=180:raise ValueError('Fixed finite A0 lifecycle contract changed')
    if p['controller']!={'kind':'FULL151_KNOWNROUGH_BASELINE_EMA_C2','format_version':105,'margin_rad':.020,'resume':'from_scratch_only'} or type(p['controller']['format_version']) is not int:raise ValueError('Wrong controller/checkpoint ABI')
    for value in p['source_pins'].values():checked(value)
    required=[Path(__file__),ROOT/'scripts/run_full151_a0_v1.py',*(ROOT/'src/full151_native'/name for name in ('__init__.py','bundle.py','controller.py','runtime.py','native.py','probe.py','io.py','metadata.py')),ROOT/'vendor/runtime_io.py',ROOT/'vendor/measured_contacts.py']
    for source in required:
        if not any(Path(value['path']).resolve()==source.resolve() and value['sha256']==sha(source) for value in p['source_pins'].values()):raise ValueError('Unfrozen execution dependency '+str(source))
    if p['protocol']!=p['source_pins']['protocol']:raise ValueError('Protocol alias differs')
    protocol=read(checked(p['protocol']))
    if protocol['schema']!='full151_knownrough_native_A0_protocol:1' or protocol['input_bundle']!=p['input_bundle'] or protocol['reference_states']!=151 or protocol['control_intervals']!=150 or protocol['allowed_roles']!=['native_a0'] or protocol['allowed_seeds']!=[0]:raise ValueError('New151 protocol differs')
    if protocol['training_authorized'] is not False or protocol['physical_reference_qualified'] is not False:raise ValueError('Input/source cannot self-qualify physics or training')
    from full151_native.bundle import load_bundle
    from full151_native.probe import GATES
    if canonical(p['evaluation_gates'])!=canonical(GATES):raise ValueError('Original fullhorizon numeric evaluation criteria differ')
    bundle=load_bundle(checked(p['input_bundle']),p['input_bundle']['sha256'])
    if bundle['descriptor']['mode']!=MODE or bundle['descriptor']['retarget_variant_id']!=VARIANT or FORMAT_VERSION!=105:raise ValueError('Wrong freshfull151 descriptor')
    output=Path(p['output']).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Study-owned fresh output required')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('Default preflight must not import device frameworks')
    return p,bundle,{'status':'CPU_INPUT_SOURCE_PREFLIGHT_ONLY','config_sha256':sha(path),'entry_sha256':sha(__file__),'role':p['role'],'method':p['method'],
      'input_signature_sha256':bundle['descriptor']['input_signature_sha256'],'pinned_files':len(p['source_pins']),'Torch_imported':False,'Warp_imported':False,'GPU_started':False,'physics_steps':0}


def freeze_sources(output,p):
    dest=output/'source';dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(dest/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for i,(name,value) in enumerate(p['source_pins'].items()):archive.write(checked(value),f'{i:04d}_{Path(value["path"]).name}')
    write(dest/'inventory.json',p['source_pins'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path);parser.add_argument('--run',action='store_true');a=parser.parse_args();path=a.config.resolve(strict=True)
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    p,bundle,proof=preflight(path)
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['GPU_run_authorized'] is not True:raise ValueError('Explicit hash-bound Root GPU plan required')
    out=Path(p['output']).resolve()
    if a.output is None or a.output.resolve()!=out or out.exists():raise ValueError('Exact fresh output; never rerun/overwrite a worker')
    out.mkdir(parents=True,exist_ok=False)
    runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(path),'source_sha256':sha(__file__),
      'role':p['role'],'method':p['method'],'optimizer_updates':0,'physics_steps':None,'physics_step_scope':'Actual scheduled A0 solver steps; constructor/graph/startup forwards and steps separately recorded by native adapter','automatic_retry':False}
    write(out/'runtime.json',runtime);shutil.copy2(path,out/'config.json');freeze_sources(out,p);write(out/'CPU_preflight.json',proof)
    config={**p,'_config_sha256':sha(path),'_worker_started_perf':STARTED}
    def emit(**updates):runtime.update(updates);runtime['worker_elapsed_seconds']=time.perf_counter()-STARTED;write(out/'runtime.json',runtime)
    try:
        from full151_native.native import run_native
        result=run_native(config,bundle,out,emit)
        evaluation=read(out/'evaluation.json') if (out/'evaluation.json').is_file() else {}
        if not evaluation.get('complete'):raise ValueError('Full151 native A0 engineering incomplete; quality is separately recorded')
        write(out/'completion.json',{'schema':'full151_native_A0_completion:1','config_sha256':sha(path),'result':result,'quality_is_not_engineering_gate':True})
        runtime.update(status='completed_native_A0',result_status=result['status'],physics_steps=evaluation.get('physical_steps'),control_intervals=evaluation.get('control_intervals'),optimizer_updates=0)
    except BaseException as e:
        runtime.update(status='failed_engineering_partial',error=repr(e));write(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'no_retry':True});raise
    finally:
        runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(out/'runtime.json',runtime)
    print(json.dumps(jsonable(result),indent=2))


if __name__=='__main__':main()
