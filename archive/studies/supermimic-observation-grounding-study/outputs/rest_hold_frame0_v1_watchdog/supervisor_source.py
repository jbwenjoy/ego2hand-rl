"""CPU preflight by default; explicit reviewed authorization permits one hidden45s worker."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json
LIFECYCLE=Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
LIFECYCLE_SHA='40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'
HIDDEN=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lifecycle():
    assert sha(LIFECYCLE)==LIFECYCLE_SHA,'Reviewed bounded lifecycle changed'
    spec=importlib.util.spec_from_file_location('rest_hold_reviewed_lifecycle',LIFECYCLE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.write=write_json;return m
def preflight(plan,python):
    p=read(plan)
    for pin in p['source_pins'].values():assert sha(pin['path'])==pin['sha256'],'Source/input drift'
    assert p['hard_process_seconds']==45 and p['soft_worker_seconds']==40 and p['automatic_retry'] is False
    result=subprocess.run([str(python),str(ROOT/'scripts/rest_hold_warp.py'),'--plan',str(plan),'--validate-only'],capture_output=True,text=True,timeout=20,creationflags=HIDDEN,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
    assert result.returncode==0,result.stderr+result.stdout
    report=json.loads(result.stdout);assert report['status']=='passed_CPU_inputs_FK_FP32_C2_contract' and not report['torch_CUDA_initialized'] and not report['Warp_imported']
    return {'status':'passed_CPU_preflight_no_GPU','plan_sha256':sha(plan),'worker_sha256':sha(ROOT/'scripts/rest_hold_warp.py'),'supervisor_sha256':sha(__file__),'input_contract':report,'GPU_started':False}
def reject_gpu_work():
    pattern=r'[\\/](train_controlled_pen|evaluate_pen_checkpoint|evaluate_c2_portfolio|evaluate_confirmation_checkpoint|probe_world_model_gradients|braking_microprobe|controlled_pen_rollout|rest_hold_warp)\.py(?:[\s\"\x27]|$)'
    query="Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.ProcessId -ne "+str(os.getpid())+" -and $_.CommandLine -match '"+pattern.replace("'","''")+"' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    r=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,check=True,timeout=10,creationflags=HIDDEN)
    assert not r.stdout.strip(),'Another project GPUworker exists: '+r.stdout
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=ROOT/'configs/rest-hold-frame0-v1.json');parser.add_argument('--python',type=Path,default=Path('D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe'));parser.add_argument('--run',action='store_true');a=parser.parse_args();plan=a.plan.resolve(strict=True);proof=preflight(plan,a.python)
    if not a.run:print(json.dumps(proof,indent=2));return
    p=read(plan);assert p.get('GPU_run_authorized') is True,'Root has not authorized GPU execution'
    output=Path(p['output']);assert output.resolve().is_relative_to((ROOT/'outputs').resolve()) and not output.exists(),'No overwrite/retry'
    logs=output.with_name(output.name+'_watchdog');logs.mkdir(parents=True,exist_ok=False);life=lifecycle();life.write(logs/'preflight.json',proof);shutil.copy2(plan,logs/'plan_snapshot.json');shutil.copy2(__file__,logs/'supervisor_source.py')
    with life.gpu_lock():
        reject_gpu_work();assert sha(plan)==proof['plan_sha256'],'Plan changed afterpreflight'
        command=[str(a.python),'-u',str(ROOT/'scripts/rest_hold_warp.py'),'--plan',str(plan),'--output',str(output),'--run']
        observer=lambda pid,parent:life.observe_actual_pid(pid,parent,ROOT/'scripts/rest_hold_warp.py',plan,output)
        result=life.watch(command,output,logs,hard_seconds=45,observer=observer)
        if result['status'].endswith('cleanup_unverified'):
            result['shared_lock_release_scope']='Boundedcleanup attempted; descendants/GPUidleness unconfirmed. No retry.';life.write(logs/'state.json',result)
    print(json.dumps(result,indent=2));raise SystemExit(result['status']!='completed')
if __name__=='__main__':main()
