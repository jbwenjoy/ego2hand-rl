"""CPU-only preflight by default; one explicitly pinned from-scratch full151 run."""
from pathlib import Path
import argparse,datetime,json,os,shutil,sys,time,traceback,zipfile
STARTED=time.perf_counter()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from full151_native.io import read,checked,pin,sha,canonical,jsonable
from full151_native.metadata import write,MODE,VARIANT,FORMAT_VERSION

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def preflight(path):
    p=read(path)
    allowed={'schema','role','method','seed','input_bundle','output','training','contact_binding','source_pins','protocol','GPU_run_authorized','training_authorized','hard_process_seconds','soft_worker_seconds','automatic_retry','compile_fallback','controller','evaluation_gates','readiness'}
    if set(p)!=allowed or p['schema']!='full151_knownrough_native_training_run:1' or p['role'] not in ('engineering_smoke','primary_matched') or p['method']!='A1L':raise ValueError('Fresh full151 A1L training roles only')
    if type(p['seed']) is not int or p['seed']!=0 or p['automatic_retry'] is not False or p['compile_fallback'] is not False or type(p['GPU_run_authorized']) is not bool or p['training_authorized'] is not True:raise ValueError('Seed0/explicit authorization/no retry required')
    bounds=(540,600) if p['role']=='engineering_smoke' else (7100,7200)
    if (p['soft_worker_seconds'],p['hard_process_seconds'])!=bounds:raise ValueError('Fixed finite lifecycle changed')
    if p['controller']!={'kind':'FULL151_KNOWNROUGH_BASELINE_EMA_C2','format_version':105,'margin_rad':.020,'resume':'from_scratch_only'}:raise ValueError('Wrong new105 controller ABI')
    for value in p['source_pins'].values():checked(value)
    required=[Path(__file__),ROOT/'scripts/run_full151_learning_v1.py',*(ROOT/'src/full151_native'/name for name in ('__init__.py','bundle.py','controller.py','runtime.py','native.py','probe.py','io.py','metadata.py','learning.py','training_capacity.py')),ROOT/'vendor/runtime_io.py',ROOT/'vendor/measured_contacts.py']
    for source in required:
        if not any(Path(value['path']).resolve()==source.resolve() and value['sha256']==sha(source) for value in p['source_pins'].values()):raise ValueError('Unfrozen execution dependency '+str(source))
    if p['protocol']!=p['source_pins']['protocol']:raise ValueError('Protocol alias differs')
    protocol=read(checked(p['protocol']))
    key='engineering_smoke_training' if p['role']=='engineering_smoke' else 'primary_matched_training'
    if protocol['schema']!='full151_knownrough_native_training_protocol:1' or protocol['input_bundle']!=p['input_bundle'] or protocol['reference_states']!=151 or protocol['control_intervals']!=150 or p['role'] not in protocol['allowed_roles'] or p['seed'] not in protocol['allowed_seeds'] or canonical(protocol[key])!=canonical(p['training']):raise ValueError('Training protocol input/role/budget differs')
    if protocol['physical_reference_qualified'] is not False or protocol['contact_objective'] is not None or protocol['allowed_methods']!=['A1L']:raise ValueError('Knownrough A1L-only contract required')
    if p['readiness']!=protocol['readiness']:raise ValueError('External readiness aliases differ')
    for name,value in p['readiness'].items():
        checked(value)
        if not any(value==v for v in p['source_pins'].values()):raise ValueError('Unarchived readiness '+name)
    a0=read(checked(p['readiness']['native_A0_engineering_enrollment']))
    if a0['native_A0_episode_engineering_qualified'] is not True or a0['input_bundle']!=p['input_bundle'] or a0['training_pipeline_qualified'] is not False or a0['physical_reference_qualified'] is not False:raise ValueError('A0 episode prerequisite/scope differs')
    ready=read(checked(p['readiness']['training_CPU_READY']))
    if ready['status']!='READY_CPU_ONLY_NOT_GPU_AUTHORIZATION' or ready['cpu_checks']!={'passed':57,'total':57} or ready['checkpoint_format']!=105:raise ValueError('Training CPU prerequisite differs')
    for value in [*ready['source_pins'].values(),ready['cpu_report']]:
        checked(value)
        if not any(value==v for v in p['source_pins'].values()):raise ValueError('CPU prerequisite source/report not bound to current archive')
    if p['role']=='primary_matched':
        health=read(checked(p['readiness']['engineering_smoke_health']))
        if health['schema']!='full151105_checkpoint_CPU_health:1' or health['status']!='PASS_SAVED_ENGINEERING_HEALTH' or health['final_iteration_scope'] is not True or health['role']!='engineering_smoke' or health['method']!='A1L' or health['input_bundle']!=p['input_bundle'] or health['iteration']!=40 or not all(v is True for v in health['checks'].values()):raise ValueError('Actual independently healthy smoke final required')
        checked(health['checkpoint']);checked(health['config']);checked(health['auditor'])
        hchecks=health['checks'];smoke=read(checked(health['config']))
        if not hchecks or type(health['passed']) is not int or type(health['total']) is not int or health['passed']!=health['total'] or health['total']!=len(hchecks) or type(health['seed']) is not int or health['seed']!=p['seed'] or health['mode']!=MODE or health['retarget_variant_id']!=VARIANT:raise ValueError('Complete same-seed new105 smoke health required')
        if health['auditor']!=pin(ROOT/'scripts/audit_full151_checkpoint_cpu_v1.py') or health['source_pins']!=smoke['source_pins'] or smoke['input_bundle']!=p['input_bundle'] or smoke['method']!='A1L' or smoke['role']!='engineering_smoke' or smoke['training']['max_iterations']!=40 or smoke['seed']!=p['seed'] or Path(health['run_dir']).resolve()!=Path(smoke['output']).resolve() or Path(health['checkpoint']['path']).resolve()!=Path(smoke['output']).resolve()/'train/model_final.pt':raise ValueError('Smoke health auditor/config/source/fixed-final binding differs')
        for value in health['source_pins'].values():checked(value)
    from full151_native.bundle import load_bundle
    from full151_native.learning import validate_training_inputs
    from full151_native.probe import GATES
    if canonical(p['evaluation_gates'])!=canonical(GATES):raise ValueError('Original fullhorizon gates differ')
    bundle=load_bundle(checked(p['input_bundle']),p['input_bundle']['sha256'])
    contract=validate_training_inputs(p,bundle)
    if bundle['descriptor']['mode']!=MODE or bundle['descriptor']['retarget_variant_id']!=VARIANT or FORMAT_VERSION!=105:raise ValueError('Wrong freshfull151 mode')
    output=Path(p['output']).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Study-owned fresh output required')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('Default preflight imported device frameworks')
    return p,bundle,{'status':'CPU_INPUT_SOURCE_PREFLIGHT_ONLY','config_sha256':sha(path),'entry_sha256':sha(__file__),'role':p['role'],'method':p['method'],'contract':contract,'input_signature_sha256':bundle['descriptor']['input_signature_sha256'],'pinned_files':len(p['source_pins']),'Torch_imported':False,'Warp_imported':False,'GPU_started':False,'physics_steps':0}

def freeze_sources(output,p):
    dest=output/'source';dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(dest/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for i,(name,value) in enumerate(p['source_pins'].items()):archive.write(checked(value),f'{i:04d}_{Path(value["path"]).name}')
    write(dest/'inventory.json',p['source_pins'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();path=a.config.resolve(strict=True)
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    p,bundle,proof=preflight(path)
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['GPU_run_authorized'] is not True:raise ValueError('Explicit hash-bound GPU plan required')
    out=Path(p['output']).resolve()
    if a.output is None or a.output.resolve()!=out or out.exists():raise ValueError('Exact fresh output; no worker restart/overwrite')
    out.mkdir(parents=True,exist_ok=False)
    runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(path),'source_sha256':sha(__file__),'role':p['role'],'method':p['method'],'from_scratch':True,'physical_reference_qualified':False,'automatic_retry':False}
    write(out/'runtime.json',runtime);shutil.copy2(path,out/'config.json');freeze_sources(out,p);write(out/'CPU_preflight.json',proof)
    config={**p,'_config_sha256':sha(path),'_worker_started_perf':STARTED}
    def emit(**updates):runtime.update(updates);runtime['worker_elapsed_seconds']=time.perf_counter()-STARTED;write(out/'runtime.json',runtime)
    try:
        from full151_native.learning import run_training
        result=run_training(config,bundle,out,emit)
        if result['status']!='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY' or result['full151_capacity_qualified_final'] is not True:raise ValueError('Training/final capacity incomplete')
        write(out/'training_completion.json',{'schema':'full151_native_worker_training_completion:1','config_sha256':sha(path),'result':result})
        runtime.update(status='completed_full151_training',result_status=result['status'],iteration=result['final_iteration'],actual_optimizer_steps=result['actual_optimizer_steps'])
    except BaseException as e:
        runtime.update(status='failed_engineering_partial',error=repr(e));write(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'no_retry':True});raise
    finally:
        runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(out/'runtime.json',runtime)
    print(json.dumps(jsonable(result),indent=2))

if __name__=='__main__':main()
