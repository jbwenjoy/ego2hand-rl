"""Pinned full151 three-arm experiment; metadata-only unless explicit --run."""
from pathlib import Path
import argparse,datetime,json,os,shutil,sys,time,traceback,zipfile
STARTED=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from full151_native.io import read,checked,pin,sha,canonical,jsonable
from full151_native.metadata import write,MODE,VARIANT
METHODS=('A1L','P_ONLY','A2')
_ARCHIVE_INDEX={}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def archived(value,p):
    checked(value)
    # The first full source validation still hashes every pin. This index only
    # avoids repeated Windows filesystem resolution in membership queries.
    key=id(p)
    if key not in _ARCHIVE_INDEX:
        _ARCHIVE_INDEX[key]=(p,{(str(Path(v['path']).resolve()).casefold(),v['sha256']) for v in p['source_pins'].values()})
    return (str(Path(value['path']).resolve()).casefold(),value['sha256']) in _ARCHIVE_INDEX[key][1]
def all_true_report(value):
    r=read(checked(value));c=r.get('checks')
    if type(c) is not dict or not c or not all(type(v) is bool and v for v in c.values()) or type(r.get('passed')) is not int or r['passed']!=r.get('total') or r['total']!=len(c):raise ValueError('Incomplete independent CPU readiness/health report')
    return r
def same_pin(a,b):
    return a['sha256']==b['sha256'] and Path(checked(a)).samefile(checked(b))
def cpu_ready_proof(p,protocol,name,critical_sources):
    ready=read(checked(p['readiness'][name+'_READY']))
    report=all_true_report(p['readiness'][name+'_report'])
    if not same_pin(ready['cpu_report'],p['readiness'][name+'_report']) or ready['cpu_checks']!={'passed':report['passed'],'total':report['total']}:raise ValueError('CPU READY/report aliases differ: '+name)
    expected=protocol['CPU_report_contracts'][name]
    if report['schema']!=expected['schema'] or report['status']!=expected['status']:raise ValueError('Wrong CPU proof schema/status '+name)
    pins=[*ready['source_pins'].values(),*report['source_pins'].values()]
    for value in pins:
        if not archived(value,p):raise ValueError('CPU source/report no longer matches execution archive '+name)
    for source in critical_sources:
        actual=pin(source)
        if not any(same_pin(actual,v) for v in pins):raise ValueError('CPU report did not test current execution source '+str(source))
    return ready,report
def preflight(path):
    p=read(path)
    allowed={'schema','role','method','seed','input_bundle','output','training','contact_binding','source_pins','protocol','GPU_run_authorized','training_authorized','hard_process_seconds','soft_worker_seconds','automatic_retry','compile_fallback','controller','evaluation_gates','readiness'}
    if set(p)!=allowed or p['schema']!='full151_knownrough_native_training_run:2' or p['role'] not in ('engineering_smoke','primary_matched') or p['method'] not in METHODS:raise ValueError('Explicit new three-arm full151 config required')
    if type(p['seed']) is not int or p['seed']!=0 or type(p['GPU_run_authorized']) is not bool or p['training_authorized'] is not True or p['automatic_retry'] is not False or p['compile_fallback'] is not False:raise ValueError('Seed0/explicit scope/no retry required')
    if (p['method']=='A1L')!=(p['contact_binding'] is None):raise ValueError('Only P_ONLY/A2 consume the frozen independent objective spec')
    bounds=(540,600) if p['role']=='engineering_smoke' else (7100,7200)
    if (p['soft_worker_seconds'],p['hard_process_seconds'])!=bounds:raise ValueError('Finite original budget changed')
    if p['controller']!={'kind':'FULL151_KNOWNROUGH_BASELINE_EMA_C2','format_version':105,'margin_rad':.020,'resume':'from_scratch_only'}:raise ValueError('Controller105 unchanged across arms')
    for v in p['source_pins'].values():checked(v)
    for source in [Path(__file__),ROOT/'scripts/run_full151_learning_v3.py',ROOT/'src/full151_native/learning_v2.py',ROOT/'src/full151_native/prior_hook_v1.py',ROOT/'src/full151_native/training_capacity.py',ROOT/'src/full151_native/controller.py',ROOT/'src/full151_native/runtime.py',ROOT/'src/full151_intent/binding_v3.py']:
        if not archived(pin(source),p):raise ValueError('Execution dependency unarchived '+str(source))
    if p['protocol']!=p['source_pins']['protocol']:raise ValueError('Protocol alias differs')
    protocol=read(checked(p['protocol']))
    key='engineering_smoke_training' if p['role']=='engineering_smoke' else 'primary_matched_training'
    if protocol['schema']!='full151_knownrough_matched_training_protocol:2' or protocol['allowed_methods']!=list(METHODS) or p['role'] not in protocol['allowed_roles'] or protocol['allowed_seeds']!=[0] or protocol['input_bundle']!=p['input_bundle'] or canonical(protocol[key])!=canonical(p['training']) or protocol['readiness']!=p['readiness']:raise ValueError('Shared matched protocol differs')
    if protocol['physical_reference_qualified'] is not False or protocol['contact_targets_are_human_truth'] is not False:raise ValueError('No physical/reference/human qualification upgrade')
    for name,v in p['readiness'].items():
        if not archived(v,p):raise ValueError('Unarchived prerequisite '+name)
    a0=read(checked(p['readiness']['native_A0_engineering_enrollment']))
    if a0['input_bundle']!=p['input_bundle'] or a0['native_A0_episode_engineering_qualified'] is not True or a0['physical_reference_qualified'] is not False:raise ValueError('A0 episode prerequisite differs')
    cpu_ready_proof(p,protocol,'learning_CPU',[ROOT/'src/full151_native/learning_v2.py',ROOT/'src/full151_native/prior_hook_v1.py'])
    binding_ready,_=cpu_ready_proof(p,protocol,'binding_CPU',[ROOT/'src/full151_intent/binding_v3.py'])
    cpu_ready_proof(p,protocol,'health_auditor_CPU',[ROOT/'scripts/audit_full151_checkpoint_cpu_v2.py'])
    all_true_report(p['readiness']['matched_cone_saved_numeric_report'])
    cone=read(checked(p['readiness']['matched_cone_report']))
    if cone['status']!='COMPLETED_MATCHED_CONE_CPU_SCREEN' or cone['LP_calls']!=302 or cone['condition_counts']['H123']!={'FEASIBLE_CONDITIONAL_WITNESS':151} or cone['physical_reference_qualified'] is not False:raise ValueError('Fixed H123 conditional151 witness evidence required; not actual contact qualification')
    binding=read(checked(protocol['objective_spec']))
    if binding['schema']!='full151_declared_H123_objective_spec:2' or binding['input_bundle']!=p['input_bundle'] or binding['execution_scope']!='EXPLORATORY_MATCHED_TRAINING' or binding['CPU_authorized'] is not True or binding['GPU_authorized'] is not True or binding['training_authorized'] is not True:raise ValueError('Explicit pinned exploratory objective specification differs')
    if not archived(protocol['objective_spec'],p):raise ValueError('Shared objective spec missing from source archive')
    tested=read(checked(binding_ready['CPU_spec']))
    expected_gpu_spec={**tested,'GPU_authorized':True}
    if tested['GPU_authorized'] is not False or canonical(binding)!=canonical(expected_gpu_spec):raise ValueError('Shared GPU spec must match actual CPU-tested spec except explicit GPU authorization')
    for value in binding['expected_code_pins'].values():
        if not archived(value,p):raise ValueError('Objective code not bound to actual tested source')
    if p['contact_binding'] is not None and p['contact_binding']!=protocol['objective_spec']:raise ValueError('Same frozen target/spec required for P_ONLY/A2')
    if p['role']=='primary_matched':
        smoke=read(checked(p['readiness']['paired_smoke_enrollment']))
        if smoke['schema']!='full151_matched_three_arm_smoke_enrollment:1' or smoke['status']!='PASS_ENGINEERING_THREE_ARM_SMOKE_NOT_TASK_QUALITY' or smoke['methods']!=list(METHODS) or smoke['same_initial_four_parameter_fingerprints'] is not True:raise ValueError('Actual healthy equal-initialization three-arm smoke required')
        pair=all_true_report(smoke['saved_pair_review'])
        if pair['schema']!='full151_matched_three_arm_smoke_saved_review:1' or pair['same_initial_four_parameter_fingerprints'] is not True or pair['health']!=smoke['health']:raise ValueError('Independent raw-initialization pair proof missing')
        for method in METHODS:
            h=all_true_report(smoke['health'][method]);checked(h['checkpoint']);checked(h['config']);checked(h['auditor'])
            if h['schema']!='full151105_checkpoint_CPU_health:1' or h['status']!='PASS_SAVED_ENGINEERING_HEALTH' or h['method']!=method or h['seed']!=p['seed'] or h['role']!='engineering_smoke' or h['iteration']!=40 or h['final_iteration_scope'] is not True or h['input_bundle']!=p['input_bundle']:raise ValueError('Fixed final40 same-seed smoke health differs')
            sc=read(checked(h['config']))
            if sc['schema']!=p['schema'] or sc['method']!=method or sc['role']!='engineering_smoke' or sc['seed']!=p['seed'] or sc['input_bundle']!=p['input_bundle'] or not same_pin(sc['protocol'],smoke['smoke_protocol']) or canonical(sc['training'])!=canonical(protocol['engineering_smoke_training']):raise ValueError('Smoke health belongs to another method/config/budget family')
            if sc['contact_binding']!=(None if method=='A1L' else protocol['objective_spec']) or Path(h['run_dir']).resolve()!=Path(sc['output']).resolve() or Path(h['checkpoint']['path']).resolve()!=Path(sc['output']).resolve()/'train/model_final.pt' or not same_pin(h['auditor'],pin(ROOT/'scripts/audit_full151_checkpoint_cpu_v2.py')):raise ValueError('Smoke fixed-final/objective/auditor identities differ')
            for value in sc['source_pins'].values():
                checked(value)
                if Path(value['path']).suffix=='.py' and not archived(value,p):raise ValueError('Primary executed Python source differs from validated smoke')
    from full151_native.bundle import load_bundle
    from full151_native.learning_v2 import validate_training_inputs
    from full151_native.probe import GATES
    if canonical(p['evaluation_gates'])!=canonical(GATES):raise ValueError('Original full3s quality criteria changed')
    bundle=load_bundle(checked(p['input_bundle']),p['input_bundle']['sha256']);contract=validate_training_inputs(p,bundle)
    if not Path(p['output']).resolve().is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Fresh study-owned output required')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('Metadata preflight imported device frameworks')
    return p,bundle,{'status':'CPU_INPUT_SOURCE_PREFLIGHT_ONLY','config_sha256':sha(path),'entry_sha256':sha(__file__),'role':p['role'],'method':p['method'],'contract':contract,'pinned_files':len(p['source_pins']),'Torch_imported':False,'Warp_imported':False,'GPU_started':False,'physics_steps':0}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();path=a.config.resolve(strict=True)
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    p,bundle,proof=preflight(path)
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['GPU_run_authorized'] is not True:raise ValueError('Explicit frozen GPU run required')
    out=Path(p['output']).resolve()
    if a.output is None or a.output.resolve()!=out or out.exists():raise ValueError('Fresh exact output, never restart/overwrite')
    out.mkdir(parents=True,exist_ok=False);runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(path),'source_sha256':sha(__file__),'role':p['role'],'method':p['method'],'from_scratch':True,'automatic_retry':False,'physical_reference_qualified':False}
    write(out/'runtime.json',runtime);shutil.copy2(path,out/'config.json');write(out/'CPU_preflight.json',proof);dest=out/'source';dest.mkdir()
    with zipfile.ZipFile(dest/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for i,(name,v) in enumerate(p['source_pins'].items()):archive.write(checked(v),f'{i:04d}_{Path(v["path"]).name}')
    write(dest/'inventory.json',p['source_pins'])
    def emit(**updates):runtime.update(updates);runtime['worker_elapsed_seconds']=time.perf_counter()-STARTED;write(out/'runtime.json',runtime)
    try:
        from full151_native.learning_v2 import run_training
        result=run_training({**p,'_config_sha256':sha(path),'_worker_started_perf':STARTED},bundle,out,emit)
        if result['status']!='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY' or result['full151_capacity_qualified_final'] is not True:raise ValueError('Final training/capacity incomplete')
        write(out/'training_completion.json',{'schema':'full151_native_worker_training_completion:2','config_sha256':sha(path),'result':result})
        runtime.update(status='completed_full151_training_v2',iteration=result['final_iteration'],result_status=result['status'],actual_optimizer_steps=result['actual_optimizer_steps'])
    except BaseException as exc:
        runtime.update(status='failed_engineering_partial',error=repr(exc));write(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'no_retry':True});raise
    finally:runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(out/'runtime.json',runtime)
    print(json.dumps(jsonable(result),indent=2))
if __name__=='__main__':main()
