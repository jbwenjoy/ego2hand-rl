"""One constrained CPU candidate using the reviewed Windows owned-tree watchdog.

The venv executable is a launcher on this host. Actual worker identity comes
from runtime.pid plus CIM; timeout cleanup owns the launcher and its tree.
No simulator, optimization, GPU or worker code is imported by this supervisor.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE = Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
LIFECYCLE_SHA = '40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'
IO = Path('D:/workspace/supermimic-c2-confirmation-eval-study/src/runtime_io.py')
IO_SHA = '159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load_pinned(path, expected, name):
    if sha(path) != expected: raise ValueError('Reviewed dependency changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    life = load_pinned(LIFECYCLE, LIFECYCLE_SHA, 'prefix_cpu_reviewed_lifecycle')
    io = load_pinned(IO, IO_SHA, 'prefix_cpu_bounded_metadata_IO')
    # Explicit wrapper bindings only: cwd and metadata writer. No edits to the
    # frozen lifecycle or timeout/tree/process-observation functions.
    life.ROOT = ROOT
    life.write = io.write_json
    return life, io


def validate(plan_path):
    p = read(plan_path)
    required = {'schema':'constrained_prefix_geometry_cpu_plan:1', 'optimization_authorized':True,
                'GPU_authorized':False, 'training_authorized':False,
                'frames':16, 'joints':20, 'geometry_samples':61, 'n_control':18,
                'free_variables':340, 'maxiter':80, 'max_evaluations':250,
                'soft_wall_seconds':220, 'owned_process_seconds':240, 'automatic_retry':False}
    for k, v in required.items():
        if p.get(k) != v or type(p.get(k)) is not type(v): raise ValueError('Plan contract differs: '+k)
    if p['source_pins']['supervisor'] != {'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}:
        raise ValueError('Supervisor source binding differs')
    for name, pin in {**p['inputs'], **p['source_pins'], 'protocol':p['protocol']}.items():
        if sha(pin['path']) != pin['sha256']: raise ValueError('Source/input drift: '+name)
    for key, path, digest in [('reviewed_lifecycle',LIFECYCLE,LIFECYCLE_SHA),('bounded_IO',IO,IO_SHA)]:
        if p['source_pins'][key] != {'path':str(path),'sha256':digest}:
            raise ValueError('Reviewed lifecycle/IO binding differs')
    protocol = read(p['protocol']['path'])
    if protocol.get('schema') != 'constrained_prefix_protocol:1' or p['protocol']['sha256'] != '0798ea7aa6da6f56684067992906bbba3136f9e15b413f743674f54159ad2e6d':
        raise ValueError('New frozen constrained protocol required')
    if p['inputs'] != protocol['inputs']: raise ValueError('Input catalog differs from frozen proposal')
    if p['parent_protocol'] != protocol['parent_protocol']:
        raise ValueError('Parent source-window protocol binding differs')
    if sha(p['parent_protocol']['path']) != p['parent_protocol']['sha256']:
        raise ValueError('Parent protocol source drift')
    for key, value in [('control_inner_margin_rad',2e-6),('force_margin_Nm',2e-6),('rate_margin_rad_s',1e-4),('generated_depth_max_m',.002999)]:
        if p['planning_margins'][key] != value or protocol['planning_margins'][key] != value:
            raise ValueError('Frozen numerical planning margin differs: '+key)
    if protocol['GPU_authorized'] is not False or protocol['training_authorized'] is not False:
        raise ValueError('Only the CPU proposal is accepted')
    worker = Path(p['source_pins']['worker']['path']).resolve()
    worker.relative_to(ROOT/'scripts')
    output = Path(p['output']).resolve()
    output.relative_to(ROOT/'outputs')
    if output == ROOT/'outputs': raise ValueError('Output must be a new child directory')
    logs = output.with_name(output.name+'_watchdog')
    if output.exists() or logs.exists(): raise FileExistsError('Existing outcome retained; no retry/overwrite')
    if Path(p['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Use the pinned virtual-environment Python')
    return p, worker, output, logs


def terminal_processes(pids):
    clean = sorted(set(int(pid) for pid in pids if pid is not None))
    if not clean: return []
    query = ('$ids=@('+','.join(map(str,clean))+'); @('
        "Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in $ids } | "
        "ForEach-Object { [PSCustomObject]@{pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;"
        "creation_utc=$_.CreationDate.ToUniversalTime().ToString('o');command_line=$_.CommandLine} }"
        ') | ConvertTo-Json -Compress')
    r = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],
                       capture_output=True,text=True,timeout=5,check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    if not r.stdout.strip(): return []
    value = json.loads(r.stdout)
    return value if isinstance(value,list) else [value]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--run',action='store_true')
    a=ap.parse_args(); plan_path=a.plan.resolve()
    p,worker,output,logs=validate(plan_path)
    life,io=dependencies()
    env_fields={'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    proof={'status':'SUPERVISOR_PLAN_PREFLIGHT_PASS','plan_sha256':sha(plan_path),
           'supervisor_sha256':sha(__file__),'worker_sha256':sha(worker),
           'lifecycle_sha256':LIFECYCLE_SHA,'bounded_IO_sha256':IO_SHA,
           'environment':env_fields,'optimization_started':False,'LP_started':False,'GPU_used':False,
           'physics_steps':0,'venv_redirector_handling':'actual runtime PID+CIM; launcher tree cleanup'}
    if not a.run:
        print(json.dumps(proof,indent=2)); return
    logs.mkdir(parents=True,exist_ok=False)
    io.write_json(logs/'preflight.json',proof)
    shutil.copy2(plan_path,logs/'plan_snapshot.json')
    shutil.copy2(__file__,logs/'supervisor_source.py')
    # Inherited by the single hidden child tree; no machine-wide settings.
    os.environ.update(env_fields)
    if sha(plan_path) != proof['plan_sha256']: raise ValueError('Plan changed after preflight')
    for pin in p['source_pins'].values():
        if sha(pin['path']) != pin['sha256']: raise ValueError('Source changed after preflight')
    command=[str(Path(sys.executable).resolve()),'-u',str(worker),
             '--plan',str(plan_path),'--output',str(output),'--run']
    observer=lambda pid,parent:life.observe_actual_pid(pid,parent,worker,plan_path,output)
    watch_error=None
    try:
        result=life.watch(command,output,logs,hard_seconds=240,observer=observer)
    except BaseException as exc:
        watch_error=exc
        result=read(logs/'state.json') if (logs/'state.json').is_file() else {'status':'missing_terminal_metadata'}
    post={'status':'UNVERIFIED','source_sha256':sha(__file__),'new_worker_attempts':1,
          'automatic_retry':False,'GPU_used':False,'physics_steps':0}
    try:
        rows=terminal_processes([result.get('launcher_pid'),result.get('actual_pid')])
        post.update(actual_CIM_rows=rows,status='NO_OWNED_PID_REMAINING' if not rows else 'PID_PRESENT_REQUIRES_IDENTITY_REVIEW')
    except Exception as exc:
        post['error']=repr(exc)
    io.write_json(logs/'post_terminal_CIM.json',post)
    qualified=(result['status']=='completed' and result.get('exit_code')==0
        and result.get('actual_process_OS_observation',{}).get('status')=='exact_OS_identity_observed'
        and post['status']=='NO_OWNED_PID_REMAINING')
    print(json.dumps({'status':result['status'],'engineering_identity_qualified':qualified,
        'launcher_pid':result.get('launcher_pid'),'actual_pid':result.get('actual_pid'),
        'timeout':result.get('timeout'),'watchdog_elapsed_seconds':result.get('elapsed_s'),
        'post_terminal_CIM_status':post['status']},indent=2))
    if watch_error is not None: raise watch_error
    raise SystemExit(0 if qualified else 1)


if __name__=='__main__': main()
