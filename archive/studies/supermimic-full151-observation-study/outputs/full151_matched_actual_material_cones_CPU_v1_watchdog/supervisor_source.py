"""Owned hidden CPU-only45s watchdog; default metadata preflight only.

Uses the frozen generic lifecycle/watch/stop functions without GPU lock,
nvidia-smi, GPU-worker rejection or any device/framework import.
"""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'scripts/screen_full151_matched_material_cones_cpu_v1.py'
DRIVER=Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py')
DRIVER_SHA='f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475'
HIDDEN=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import read,sha,pin
from full151_native.metadata import write

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--python',type=Path,default=Path('D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe'));ap.add_argument('--run',action='store_true');a=ap.parse_args();protocol=a.protocol.resolve(strict=True);p=read(protocol)
    for v in p['source_pins'].values():
        if sha(v['path'])!=v['sha256']:raise ValueError('Frozen source/input changed')
    for path in (WORKER,Path(__file__),DRIVER):
        if not any(Path(v['path']).resolve()==path.resolve() and v['sha256']==sha(path) for v in p['source_pins'].values()):raise ValueError('Watchdog/worker/identity source unpinned')
    if sha(DRIVER)!=DRIVER_SHA:raise ValueError('Qualified lifecycle/identity driver changed')
    spec=importlib.util.spec_from_file_location('_matched_cone_CPU_owned_identity',DRIVER);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d);d.ROOT=ROOT;d.WORKER=WORKER
    identity_python=d.selected_python_identity(a.python)
    process=subprocess.Popen([str(a.python),str(WORKER),'--protocol',str(protocol)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=HIDDEN,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
    try:stdout,stderr=process.communicate(timeout=15)
    except BaseException as exc:
        cleanup={'status':'UNVERIFIED','retained_CPU_launcher_pid':process.pid}
        try:cleanup['owned_tree_cleanup']=d.lifecycle().stop_owned(process)
        except BaseException as error:cleanup['cleanup_error']=repr(error)
        try:process.communicate(timeout=1)
        except BaseException:
            if process.stdout:process.stdout.close()
            if process.stderr:process.stderr.close()
        raise RuntimeError('CPU preflight interrupted/timeout; no LP permitted; retained owned cleanup '+repr(cleanup)) from exc
    if process.returncode:raise ValueError('CPU array preflight rejected: '+stderr+stdout)
    proof=json.loads(stdout)
    if proof['status']!='CPU_METADATA_ARRAY_PREFLIGHT_ONLY' or proof['LP_calls']!=0 or proof['Torch_Warp_Env_imported']:raise ValueError('Default CPU metadata boundary failed')
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['CPU_run_authorized'] is not True or p['soft_worker_seconds']!=30 or p['hard_process_seconds']!=45:raise ValueError('Separate Root finiteCPU authorization required')
    output=Path(p['output']).resolve()
    if output.exists() or not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Fresh owned output required')
    logs=output.with_name(output.name+'_watchdog');logs.mkdir(parents=True,exist_ok=False);write(logs/'preflight.json',proof);write(logs/'protocol_snapshot.json',p);(logs/'supervisor_source.py').write_bytes(Path(__file__).read_bytes())
    life=d.lifecycle();command=[str(a.python),'-u',str(WORKER),'--protocol',str(protocol),'--output',str(output),'--run'];error=None
    # Intentionally no gpu_lock, GPU discovery, nvidia-smi or training action.
    try:result=life.watch(command,output,logs,hard_seconds=45,observer=lambda pid,parent:life.observe_actual_pid(pid,parent,WORKER,protocol,output))
    except BaseException as exc:
        error=exc;result=read(logs/'state.json') if (logs/'state.json').exists() else {'status':'MISSING_TERMINAL_METADATA'}
    post={'status':'UNVERIFIED','queried_pids':[],'actual_CIM_rows':None}
    try:post=d.terminal_processes([result.get('launcher_pid'),result.get('actual_pid')])
    except BaseException as exc:post['error']=repr(exc)
    write(logs/'post_terminal_CIM.json',post)
    try:runtime=read(output/'runtime.json')
    except BaseException as exc:runtime={'read_error':repr(exc)}
    identity=d.qualify_terminal_identity(result,runtime,post,plan=protocol,output=output,command=command,plan_sha=sha(protocol),worker_sha=sha(WORKER),python_identity=identity_python)
    identity.update(CPU_only=True,supervisor=pin(__file__),scope='Owned process identity/exit only. No GPU lock/device discovery/physics. A complete numerical report and its saved primal witnesses remain separate.')
    write(logs/'engineering_identity.json',identity)
    print(json.dumps({'status':result.get('status'),'identity':identity},indent=2))
    if error is not None:raise error
    raise SystemExit(0 if identity['engineering_identity_qualified'] else 1)

if __name__=='__main__':main()
