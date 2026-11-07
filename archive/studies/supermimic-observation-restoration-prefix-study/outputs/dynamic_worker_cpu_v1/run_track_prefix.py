"""Default CPU preflight; explicit Root authorization permits one hidden45s worker.

No queue, retry, historical-rest fallback or quality-based selection is provided.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json
LIFECYCLE=Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
LIFECYCLE_SHA='40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'
IO_SHA='159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1'
HIDDEN=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
WORKER=ROOT/'scripts/track_prefix_warp.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def lifecycle():
    if sha(LIFECYCLE)!=LIFECYCLE_SHA or sha(ROOT/'vendor/runtime_io.py')!=IO_SHA:raise ValueError('Reviewed lifecycle/atomicIO changed')
    spec=importlib.util.spec_from_file_location('dynamic_prefix_reviewed_lifecycle',LIFECYCLE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # Keep old40b immutable; explicitly replace metadataIO and processcwd only.
    m.write=write_json;m.ROOT=ROOT
    return m


def preflight(plan,python):
    p=read(plan)
    for pin in p['source_pins'].values():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Frozen source/input drift')
    for source in (WORKER,Path(__file__),ROOT/'vendor/runtime_io.py',LIFECYCLE,ROOT/'vendor/measured_contacts.py'):
        if not any(Path(pin['path']).resolve()==source.resolve() and sha(source)==pin['sha256'] for pin in p['source_pins'].values()):raise ValueError('Execution dependency is not frozen: '+str(source))
    if p['hard_process_seconds']!=45 or p['soft_worker_seconds']!=40 or p['automatic_retry'] is not False:raise ValueError('Fixed finite/no-retry process contract changed')
    result=subprocess.run([str(python),str(WORKER),'--plan',str(plan),'--validate-only'],capture_output=True,text=True,timeout=20,creationflags=HIDDEN,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
    if result.returncode:raise ValueError('CPU preflight failed: '+result.stderr+result.stdout)
    report=json.loads(result.stdout)
    if report['status']!='passed_CPU_new_dynamic_inputs_FK_FP32_control_contract' or report['Warp_imported'] or report['Torch_imported']:raise ValueError('CPU proof did not preserve import/physics boundary')
    return {'status':'passed_CPU_preflight_no_GPU','plan_sha256':sha(plan),'worker_sha256':sha(WORKER),'supervisor_sha256':sha(__file__),'lifecycle_sha256':LIFECYCLE_SHA,'runtime_IO_sha256':IO_SHA,'input_contract':report,'GPU_started':False}


def reject_gpu_work():
    pattern=r'[\\/](train_controlled_pen|evaluate_pen_checkpoint|evaluate_c2_portfolio|evaluate_confirmation_checkpoint|probe_world_model_gradients|braking_microprobe|controlled_pen_rollout|rest_hold_warp|track_prefix_warp)\.py(?:[\s\"\x27]|$)'
    query="Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.ProcessId -ne "+str(os.getpid())+" -and $_.CommandLine -match '"+pattern.replace("'","''")+"' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,check=True,timeout=10,creationflags=HIDDEN)
    if result.stdout.strip():raise ValueError('Another project GPU worker exists: '+result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--python',type=Path,default=Path('D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe'));parser.add_argument('--run',action='store_true');args=parser.parse_args()
    plan=args.plan.resolve(strict=True);proof=preflight(plan,args.python)
    if not args.run:print(json.dumps(proof,indent=2));return
    p=read(plan)
    if p.get('GPU_run_authorized') is not True:raise ValueError('Root has not authorized this hash-bound plan')
    output=Path(p['output']).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()) or output.exists():raise ValueError('New owned output required; no overwrite/retry')
    logs=output.with_name(output.name+'_watchdog');logs.mkdir(parents=True,exist_ok=False);life=lifecycle();life.write(logs/'preflight.json',proof);shutil.copy2(plan,logs/'plan_snapshot.json');shutil.copy2(__file__,logs/'supervisor_source.py')
    with life.gpu_lock():
        reject_gpu_work()
        if sha(plan)!=proof['plan_sha256']:raise ValueError('Plan changed after preflight')
        command=[str(args.python),'-u',str(WORKER),'--plan',str(plan),'--output',str(output),'--run']
        observer=lambda pid,parent:life.observe_actual_pid(pid,parent,WORKER,plan,output)
        result=life.watch(command,output,logs,hard_seconds=45,observer=observer)
        if result['status'].endswith('cleanup_unverified'):
            result['shared_lock_release_scope']='Bounded cleanup attempted; descendants/GPU idleness unconfirmed. No retry.';life.write(logs/'state.json',result)
    print(json.dumps(result,indent=2));raise SystemExit(result['status']!='completed')
if __name__=='__main__':main()
