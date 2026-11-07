"""Default CPU preflight; explicit Root authorization permits one hidden45s worker.

No queue, retry, historical-rest fallback or quality-based selection is provided.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys
from datetime import datetime
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


def terminal_processes(pids):
    """Exact known PID query only, after the retained Popen tree is terminal."""
    clean=sorted(set(int(pid) for pid in pids if pid is not None))
    if not clean:raise ValueError('Missing owned PIDs; cannot claim no remaining worker')
    query=('$ids=@('+','.join(map(str,clean))+'); @('
        "Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in $ids } | "
        "ForEach-Object { [PSCustomObject]@{pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;"
        "creation_utc=$_.CreationDate.ToUniversalTime().ToString('o');command_line=$_.CommandLine} }"
        ') | ConvertTo-Json -Compress')
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,timeout=5,check=True,creationflags=HIDDEN)
    value=json.loads(result.stdout) if result.stdout.strip() else []
    rows=value if isinstance(value,list) else [value]
    return {'status':'NO_OWNED_PID_REMAINING' if not rows else 'PID_PRESENT_REQUIRES_IDENTITY_REVIEW','queried_pids':clean,'actual_CIM_rows':rows,'scope':'Retained launcher/runtime actual PIDs only; not a machine-wide GPU idleness certificate'}


def command_executable(command_line):
    """Parse Windows' actual command line, including quoted executable paths."""
    if not isinstance(command_line,str) or not command_line.strip():raise ValueError('Missing OS command line')
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        shell=ctypes.WinDLL('shell32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        shell.CommandLineToArgvW.argtypes=(wintypes.LPCWSTR,ctypes.POINTER(ctypes.c_int));shell.CommandLineToArgvW.restype=ctypes.POINTER(wintypes.LPWSTR)
        kernel.LocalFree.argtypes=(ctypes.c_void_p,);kernel.LocalFree.restype=ctypes.c_void_p
        count=ctypes.c_int();argv=shell.CommandLineToArgvW(command_line,ctypes.byref(count))
        if not argv:raise OSError(ctypes.get_last_error(),'CommandLineToArgvW failed')
        try:
            if count.value<1:raise ValueError('Empty parsed OS command')
            return argv[0]
        finally:kernel.LocalFree(argv)
    import shlex
    return shlex.split(command_line)[0]


def selected_python_identity(chosen_python):
    """This supervisor must run in the chosen venv; resolve its actual base file.

    sys.executable may be the venv redirector. The worker OS executable must be
    compared to sys._base_executable, not to that redirector's distinct file.
    """
    chosen=Path(chosen_python).resolve(strict=True)
    if not os.path.samefile(chosen,sys.executable):raise ValueError('Run supervisor through the same selected virtual-environment Python')
    raw=Path(sys._base_executable)
    base=raw.resolve(strict=True)
    return {'chosen_python':str(chosen),'supervisor_sys_executable':sys.executable,'base_executable_raw':str(raw),'base_executable_resolved':str(base),'base_executable_sha256':sha(base),'venv_selected_samefile':True,'base_alias_resolved_samefile':os.path.samefile(raw,base),'scope':'Supervisor is running in the selected venv; its actual base Python file is independently compared to the observed worker OS executable'}


def qualify_terminal_identity(result,runtime,post,*,plan,output,command,plan_sha,worker_sha,python_identity):
    """Pure fail-closed identity gate; exit0 alone never qualifies a run."""
    checks={}
    def need(name,value):checks[name]=bool(value)
    def positive_pid(value):return type(value) is int and value>0
    def norm(value):return str(value).replace('/','\\').lower()
    try:
        observed=result.get('actual_process_OS_observation',{});pid=result.get('actual_pid');launcher=result.get('launcher_pid');supervisor=result.get('supervisor_pid')
        need('watchdog_completed_exit0_no_timeout',result.get('status')=='completed' and result.get('exit_code')==0 and result.get('timeout') is False)
        need('exact_OS_identity_observed',observed.get('status')=='exact_OS_identity_observed')
        need('owned_PID_fields_present',all(positive_pid(k) for k in (pid,launcher,supervisor)))
        expected_parent=supervisor if pid==launcher else launcher
        need('runtime_actual_PID_parent_match',positive_pid(pid) and runtime.get('pid')==pid and observed.get('pid')==pid and runtime.get('parent_pid')==expected_parent and observed.get('parent_pid')==expected_parent)
        need('watch_command_equals_launched_command',result.get('command')==list(map(str,command)) and norm(result.get('worker_output'))==norm(output))
        raw_line=observed.get('command_line','');line=norm(raw_line);first=command_executable(raw_line);exe=observed.get('executable_path','')
        need('chosen_venv_supervisor_identity_samefile',os.path.samefile(python_identity['chosen_python'],python_identity['supervisor_sys_executable']) and python_identity.get('venv_selected_samefile') is True)
        need('OS_command_exe_expected_base_samefile',os.path.samefile(first,exe) and os.path.samefile(exe,python_identity['base_executable_raw']) and os.path.samefile(exe,python_identity['base_executable_resolved']) and sha(exe)==python_identity['base_executable_sha256'])
        need('OS_command_worker_plan_output_bound',all(norm(Path(path).resolve()) in line for path in (WORKER,plan,output)) and '--run' in line)
        need('runtime_source_plan_bound',runtime.get('plan_sha256')==plan_sha and runtime.get('source_sha256')==worker_sha)
        captured=result.get('runtime_metadata',{})
        need('watched_runtime_matches_terminal_identity',all(captured.get(k)==runtime.get(k) for k in ('pid','parent_pid','started_utc','plan_sha256','source_sha256')) and captured.get('pid')==pid)
        def stamp(value):
            date=datetime.fromisoformat(value.replace('Z','+00:00'))
            if date.utcoffset() is None:raise ValueError('Naive timestamp is not exact OS UTC')
            return date
        creation=stamp(observed['creation_date_utc']);prelaunch=stamp(result['Popen_prelaunch_utc']);started=stamp(runtime['started_utc']);seen=stamp(observed['observed_utc'])
        need('exact_creation_time_order_bound',prelaunch<=creation<=started<=seen)
        expected_pids=sorted(set(k for k in (pid,launcher) if positive_pid(k)))
        need('postterminal_CIM_exact_owned_PIDs_absent',post.get('status')=='NO_OWNED_PID_REMAINING' and post.get('actual_CIM_rows')==[] and post.get('queried_pids')==expected_pids and len(expected_pids)>0)
    except Exception as exc:
        checks['identity_parse_error']=False;error=repr(exc)
    else:error=None
    qualified=bool(checks) and all(checks.values())
    return {'status':'ENGINEERING_OS_IDENTITY_QUALIFIED' if qualified else 'ENGINEERING_OS_IDENTITY_UNQUALIFIED_UNKNOWN','engineering_identity_qualified':qualified,'checks':checks,'parse_error':error,'actual_pid':result.get('actual_pid'),'launcher_pid':result.get('launcher_pid'),'selected_python_identity':python_identity,'physical_outcome_qualified_by_supervisor':False,'next_condition_permitted_by_identity':qualified,'scope':'Process identity/exit gate only. Separate raw/full-horizon/capacity/controller/quality audit is still required; no automatic next condition or physics retry.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--python',type=Path,default=Path('D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe'));parser.add_argument('--run',action='store_true');args=parser.parse_args()
    plan=args.plan.resolve(strict=True);python_identity=selected_python_identity(args.python);proof=preflight(plan,args.python);proof['selected_python_identity']=python_identity
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
        watch_error=None
        try:result=life.watch(command,output,logs,hard_seconds=45,observer=observer)
        except BaseException as exc:
            watch_error=exc;result=read(logs/'state.json') if (logs/'state.json').is_file() else {'status':'missing_terminal_metadata'}
        if result.get('status','').endswith('cleanup_unverified'):
            result['shared_lock_release_scope']='Bounded cleanup attempted; descendants/GPU idleness unconfirmed. No retry or next condition.';life.write(logs/'state.json',result)
        post={'status':'UNVERIFIED','queried_pids':[],'actual_CIM_rows':None}
        try:post=terminal_processes([result.get('launcher_pid'),result.get('actual_pid')])
        except Exception as exc:post['error']=repr(exc)
        life.write(logs/'post_terminal_CIM.json',post)
        try:terminal_runtime=read(output/'runtime.json')
        except Exception as exc:terminal_runtime={'read_error':repr(exc)}
        identity=qualify_terminal_identity(result,terminal_runtime,post,plan=plan,output=output,command=command,plan_sha=proof['plan_sha256'],worker_sha=proof['worker_sha256'],python_identity=python_identity)
        identity.update(supervisor_sha256=sha(__file__),plan_sha256=proof['plan_sha256'],worker_sha256=proof['worker_sha256'])
        life.write(logs/'engineering_identity.json',identity)
    print(json.dumps({'watchdog_status':result.get('status'),**identity},indent=2))
    if watch_error is not None:raise watch_error
    raise SystemExit(0 if identity['engineering_identity_qualified'] else 1)
if __name__=='__main__':main()
