"""CPU-only default preflight for a separately authorized current104 probe."""
from __future__ import annotations
import argparse,json,os,shutil,sys,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
STARTED=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from rough_prefix.io import read,checked,pin,sha
from rough_prefix.bundle_v2 import load_bundle
from rough_prefix.metadata import write


def all_pins(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:yield value
        else:
            for v in value.values():yield from all_pins(v)
    elif isinstance(value,list):
        for v in value:yield from all_pins(v)


def preflight(path):
    p=read(path)
    required={'schema','role','stage','evaluation_seed','input_bundle','output','directions','datasets','models',
              'source_pins','protocol','parent_proposal','GPU_run_authorized','worlds',
              'hard_process_seconds','soft_worker_seconds','automatic_retry','compile_fallback'}
    if set(p)!=required or p['schema']!='rough_current104_physical_FD_run:1' or p['role']!='current104_same_Warp_physical_FD':raise ValueError('Separate current104 probe schema required')
    if p['stage']!='GPU_physics_then_separateCPU' or type(p['evaluation_seed']) is not int or p['evaluation_seed']!=0:raise ValueError('Wrong process stage or seed')
    if type(p['GPU_run_authorized']) is not bool or type(p['worlds']) is not int or p['worlds']!=1 or p['automatic_retry'] is not False or p['compile_fallback'] is not False:raise ValueError('One world/no retry/no fallback required')
    for v in all_pins(p):checked(v)
    protocol=read(checked(p['protocol']));proposal=read(checked(p['parent_proposal']))
    if protocol['schema']!='rough_current104_physical_FD_execution_protocol:1' or protocol['protocol_revision']!=3:raise ValueError('Only revised FP32 execution ABI supported')
    if protocol['parent_proposal']!=p['parent_proposal'] or p['parent_proposal']!=pin(ROOT/'proposals/current104_samewarp_physical_fd_v2/protocol.json'):raise ValueError('Wrong parent proposal')
    for key in ('stage','input_bundle','directions','worlds','evaluation_seed','hard_process_seconds','soft_worker_seconds','automatic_retry','compile_fallback'):
        if p[key]!=protocol[key]:raise ValueError('Protocol alias differs '+key)
    if any(type(p[k]) is not int for k in ('hard_process_seconds','soft_worker_seconds')) or p['hard_process_seconds']!=240 or p['soft_worker_seconds']!=220:raise ValueError('Wrong fixed process budget')
    if p['datasets']!=proposal['inputs']['saved_trajectories'] or p['models']!=proposal['inputs']['models']:raise ValueError('Source/model selection changed')
    if [x['name'] for x in p['datasets']]!=['A0','A1L','P_ONLY','A2']:raise ValueError('Wrong path order')
    for value in all_pins(proposal):checked(value)
    b=load_bundle(checked(p['input_bundle']),p['input_bundle']['sha256'])
    closure=['io.py','metadata.py','bundle.py','bundle_v2.py','native_v2.py','runtime_v2.py',
             'controller.py','capacity.py','probe.py','policy_probe.py','evaluation_v2.py','physical_fd_v2.py','probe_transaction_v3.py']
    needed=[Path(__file__),ROOT/'scripts/run_current104_physical_fd_v2.py',
            *[ROOT/'src/rough_prefix'/name for name in closure],ROOT/'vendor/runtime_io.py',ROOT/'vendor/measured_contacts.py']
    for source in needed:
        if not any(Path(v['path']).resolve()==source.resolve() and v['sha256']==sha(source) for v in p['source_pins'].values()):raise ValueError('Unpinned probe dependency '+str(source))
    for key,value in [('entry',pin(__file__)),('protocol',p['protocol']),('parent_proposal',p['parent_proposal']),('input_bundle',p['input_bundle'])]:
        if p['source_pins'].get(key)!=value:raise ValueError('Missing exact source alias '+key)
    prep=read(checked(p['directions']['report']))
    if prep['status']!='PREPARED_NOT_PHYSICALLY_QUALIFIED' or prep['valid_domain_pairs']!=256 or prep['zero_dimensional_states']!=0 or prep['dataset']!=p['directions']['arrays']:raise ValueError('Directions not fixed/qualified for this contract')
    abi=protocol['producer_result_schema']
    for key,n in [('branch_object_terms_k',560),('branch_object_terms_kplus1',560),('donor_object_terms_k',16),('donor_object_terms_kplus1',16)]:
        container=abi['branch_arrays'] if n==560 else abi['snapshot_arrays']
        if container.get(key)!=[n,2]:raise ValueError('Missing original FP32 reward terms ABI')
    output=Path(p['output']).resolve()
    if output!=(ROOT/'outputs/current104_physical_FD_v2').resolve():raise ValueError('Exact owned single-attempt output required')
    if p['GPU_run_authorized']:
        auth=read(checked(p['source_pins']['Root_authorization']))
        if auth['schema']!='current104_physical_FD_authorization:1' or auth['GPU_run_authorized'] is not True or Path(auth['output']).resolve()!=output or auth['automatic_retry'] is not False:raise ValueError('Separate byte-bound authorization absent')
        base=read(checked(auth['base_config']));expected={**p,'GPU_run_authorized':False,'source_pins':{k:v for k,v in p['source_pins'].items() if k!='Root_authorization'}}
        if base!=expected:raise ValueError('Authorization changed scientific contract')
        checked(auth['independent_review'])
    if any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')):raise AssertionError('Metadata preflight imported simulation frameworks')
    return p,b,dict(status='CPU_INPUT_SOURCE_PREFLIGHT_ONLY',config_sha256=sha(path),entry_sha256=sha(__file__),
        role=p['role'],Torch_imported=False,Warp_imported=False,GPU_started=False,physics_steps=0,pinned_files=len(p['source_pins']))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    path=a.config.resolve(strict=True);p,b,proof=preflight(path)
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['GPU_run_authorized'] is not True:raise ValueError('Separate GPU authorization required')
    output=Path(p['output']).resolve()
    if a.output is None or a.output.resolve()!=output or output.exists():raise ValueError('Exact fresh output required')
    output.mkdir(parents=True,exist_ok=False);source=output/'source';source.mkdir()
    with zipfile.ZipFile(source/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for i,v in enumerate(p['source_pins'].values()):
            f=checked(v);z.write(f,f'{i:04d}_{f.name}')
    write(source/'inventory.json',p['source_pins']);shutil.copy2(path,output/'config.json');write(output/'CPU_preflight.json',proof)
    runtime=dict(status='initializing',pid=os.getpid(),parent_pid=os.getppid(),started_utc=datetime.now(timezone.utc).isoformat(),
                 plan_sha256=sha(path),source_sha256=sha(__file__),role=p['role'],physics_steps=None,optimizer_updates=0,automatic_retry=False)
    def emit(**updates):
        runtime.update(updates);runtime['worker_elapsed_seconds']=time.perf_counter()-STARTED;write(output/'runtime.json',runtime)
    emit()
    try:
        from rough_prefix.physical_fd_v2 import run_physical_probe
        result=run_physical_probe({**p,'_worker_started_perf':STARTED},b,output,emit)
        for value in all_pins(p):checked(value)
        if sha(path)!=proof['config_sha256']:raise ValueError('Actual config changed during physical probe')
        write(output/'source_and_input_preservation.json',dict(all_pins_unchanged=True,config_sha256=sha(path),pin_occurrences=sum(1 for _ in all_pins(p))))
        write(output/'worker_completion.json',dict(schema='current104_physical_producer_completion:1',config_sha256=sha(path),report=pin(output/'physical_report.json')))
        runtime.update(status='completed_physical_probe',physics_steps=result['scheduled_substeps'])
    except BaseException as exc:
        runtime.update(status='failed_engineering_partial',error=repr(exc))
        write(output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),no_retry=True));raise
    finally:
        runtime.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-STARTED)
        write(output/'runtime.json',runtime)


if __name__=='__main__':main()
