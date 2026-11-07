"""After the existing four-case evaluation, run two fixed WM diagnostics once.

No learning, model selection, physics call or restart occurs in this supervisor.
Default validates source/JSON only. Actual models are loaded only by the two
reviewed CPU diagnostic programs after the original training has completed.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess, sys, time, traceback

H=Path(__file__).resolve().parent;P=H.parent
TRAIN=P/'phone_tripod_supermimic_v1'
EVAL=P/'phone_tripod_policy_evaluation_v1'
EVAL_PROCESS=EVAL/'learned_process_v2_run1'
PROCESS=H/'process_run1'
JOBS=[
    ('distribution',P/'phone_tripod_final_WM_diagnostics_v1/protocol_CPUFalse_v2.json','a9092a56011156ee0aaf361148b27e51aead195d7fdcfc2d956badc9ad3e65a5','81ddf2cbacf994884e455532373229f4035c94578e3d834e561bf0b28741886e'),
    ('action_response',P/'phone_tripod_action_response_evaluation_v1/protocol_CPUFalse_v1.json','ba469e53221363da4d8ea64700f1da738c9071a73f3b91a26ee17532492e4fe0','1aa281633dbc7aafa4c4ac4ce08fc98574762ed58f66f6f3f4f445a572fcda88')]

def need(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text('utf8'))

def pin(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def checked(value):
    need(pin(value['path'])==value,'Current immutable input/source pin');return Path(value['path'])

def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush()

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def native_supervisor():
    cmd=('Get-CimInstance Win32_Process -Filter "ProcessId=41220" '
         '| Select-Object ProcessId,@{Name="CreationDate";Expression={$_.CreationDate.ToUniversalTime().ToString("o")}},CommandLine | ConvertTo-Json -Depth 3')
    r=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,encoding='utf8',timeout=15,check=True)
    if not r.stdout.strip():return None
    item=json.loads(r.stdout)
    creation=datetime.datetime.fromisoformat(item['CreationDate'].replace('Z','+00:00'))
    expected=datetime.datetime.fromisoformat('2026-11-07T11:45:16.842558+00:00')
    need(abs((creation-expected).total_seconds())<.000001 and 'execute_final_native_after_training_v1.py --run' in item['CommandLine'],'Same native supervisor, not reused PID')
    return item

def preflight():
    need(not PROCESS.exists(),'Fresh single diagnostic supervisor run')
    for label,base,protocol_sha,source_sha in JOBS:
        p=read(base);need(pin(base)['sha256']==protocol_sha,'Frozen pending '+label)
        source=p['source'] if 'source' in p else p['sources']['evaluator']
        need(source['sha256']==source_sha,'Frozen reviewed diagnostic source '+label);checked(source)
        for v in p['sources'].values():checked(v)
        for v in p['inputs'].values():
            if v is not None:checked(v)
        need(p['CPU_run_authorized'] is False and p['GPU_run_authorized'] is False,'Pending CPU/GPU flags')
        need(all(p['inputs'][k] is None for k in ['final_training_protocol','final_result','final_health','final_terminal']),'Four actual final inputs still pending')
        need(not Path(p['output']).exists(),'Fresh '+label+' results')
        need(not any((base.parent/n).exists() for n in ['protocol_postfinal_CPUFalse_run1.json','protocol_postfinal_CPUTrue_run1.json']),'Fresh final bound protocols')

def execute():
    preflight();observed=native_supervisor()
    need(observed is not None or (EVAL_PROCESS/'terminal.json').exists(),'Existing native supervisor or its terminal; no restart')
    PROCESS.mkdir();start=utc();began=time.perf_counter();phase='wait_existing_native_evaluation';results=[]
    write(PROCESS/'start.json',dict(UTC=start,pid=os.getpid(),launcher=pin(__file__),waiting_for=observed,no_new_learning=True,training_restarts=0,jobs=[dict(label=k,base=pin(b)) for k,b,_,_ in JOBS]))
    print(json.dumps(dict(pid=os.getpid(),waiting_for_native_supervisor=41220,process=str(PROCESS),new_learning=False)),flush=True)
    try:
        deadline=time.monotonic()+3*3600
        with (PROCESS/'wait_observations.jsonl').open('x',encoding='utf8') as f:
            while not (EVAL_PROCESS/'terminal.json').exists():
                need(not (EVAL_PROCESS/'supervisor_failure.json').exists(),'Inspect failed native supervisor before any diagnostic launch')
                need(time.monotonic()<deadline,'Observation deadline, inspect without restarting')
                try:item=native_supervisor()
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError,json.JSONDecodeError) as error:
                    f.write(json.dumps(dict(UTC=utc(),transient_observation_error=repr(error)))+'\n');f.flush();time.sleep(10);continue
                f.write(json.dumps(dict(UTC=utc(),process=item))+'\n');f.flush()
                if item is None:
                    time.sleep(2);need((EVAL_PROCESS/'terminal.json').exists(),'Native supervisor stopped without terminal; preserve, do not restart')
                else:time.sleep(15)
        native=read(EVAL_PROCESS/'terminal.json')
        need(native['returncode']==0 and native['source']['sha256']=='bf154e3810bfc4a587c03e70281ff250a5079fb5ff382c54a733662663fa2c43','All four native cases recorded successfully; physical failures remain results')
        training_terminal=TRAIN/'process_resume1/terminal.json';terminal=read(training_terminal)
        need(terminal['pid']==28104 and terminal['returncode']==0,'Original fixed training successful')
        final_files=dict(final_training_protocol=pin(TRAIN/'protocol_CPU_GPU_True_resume1.json'),final_result=pin(TRAIN/'gpu_resume1/result.json'),final_health=pin(TRAIN/'gpu_resume1/health_1000.json'),final_terminal=pin(training_terminal))
        need(terminal['protocol']==final_files['final_training_protocol'],'Actual resumed training protocol')
        prepared=[]
        for label,base,protocol_sha,source_sha in JOBS:
            phase=label+'_bind';need(pin(base)['sha256']==protocol_sha,'Pending protocol unchanged while waiting')
            p=read(base);p['inputs'].update(final_files)
            source=p['source'] if 'source' in p else p['sources']['evaluator'];checked(source)
            p['root_review']=dict(parent_protocol=pin(base),launcher=pin(__file__),native_evaluation_terminal=pin(EVAL_PROCESS/'terminal.json'),scope='Fixed final1000 two-model CPU diagnostic once; no fit, checkpoint selection, retry or new learning. Existing native failures remain unfiltered.')
            bound_false=base.parent/'protocol_postfinal_CPUFalse_run1.json';write(bound_false,p)
            meta=subprocess.run([sys.executable,'-X','utf8',source['path'],'--protocol',str(bound_false)],capture_output=True,text=True,encoding='utf8')
            write(PROCESS/(label+'_metadata.json'),dict(returncode=meta.returncode,stdout=meta.stdout,stderr=meta.stderr,protocol=pin(bound_false)))
            need(meta.returncode==0,'Bound metadata must pass before model loading')
            prepared.append((label,base,p,source,bound_false))
        # Both metadata checks must finish before either diagnostic loads a model.
        for label,base,p,source,bound_false in prepared:
            p['CPU_run_authorized']=True;p['root_review']['bound_false_protocol']=pin(bound_false)
            bound_true=base.parent/'protocol_postfinal_CPUTrue_run1.json';write(bound_true,p)
            phase=label+'_CPU_diagnostic';cmd=[sys.executable,'-X','utf8',source['path'],'--protocol',str(bound_true),'--run'];tick=time.perf_counter()
            stdout=PROCESS/(label+'.stdout.log');stderr=PROCESS/(label+'.stderr.log')
            with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
                child=subprocess.Popen(cmd,stdout=so,stderr=se,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));print(json.dumps(dict(phase=phase,pid=child.pid,UTC=utc())),flush=True);rc=child.wait()
            result=dict(label=label,pid=child.pid,returncode=rc,command=cmd,protocol=pin(bound_true),source=source,elapsed_s=time.perf_counter()-tick,stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
            write(PROCESS/(label+'_terminal.json'),result);results.append(result)
            need(rc==0,'Diagnostic failed; preserve and stop without retry')
        write(PROCESS/'terminal.json',dict(schema='phone_tripod_fixed_posttraining_diagnostics_terminal:1',returncode=0,started_UTC=start,ended_UTC=utc(),elapsed_s=time.perf_counter()-began,results=results,launcher=pin(__file__),new_learning_started=False,automatic_retry=False))
        print(json.dumps(dict(terminal=pin(PROCESS/'terminal.json'),returncode=0)),flush=True);return 0
    except BaseException as error:
        write(PROCESS/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),results=results,UTC=utc(),automatic_retry=False,training_restarts=0));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',action='store_true');args=a.parse_args()
    if args.run:raise SystemExit(execute())
    preflight();print(json.dumps(dict(status='SOURCE_METADATA_NO_BACKEND_OR_WAIT',jobs=[k for k,_,_,_ in JOBS],new_learning=False,existing_native_supervisor=native_supervisor())))
