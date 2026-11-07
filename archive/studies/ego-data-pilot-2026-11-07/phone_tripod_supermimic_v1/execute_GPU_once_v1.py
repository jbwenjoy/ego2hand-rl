"""Root's first matched GPU learning run with visible hardware sampling."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time,statistics
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def GPU():
    try:q=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
    except (OSError,subprocess.TimeoutExpired) as e:return dict(error=repr(e))
    if q.returncode:return dict(error=q.stderr.strip())
    lines=q.stdout.strip().splitlines()
    if len(lines)!=1:return dict(error='Expected one enumerated GPU',raw=q.stdout.strip())
    v=[s.strip() for s in lines[0].split(',')]
    if len(v)!=5:return dict(error='Unexpected GPU sample',raw=q.stdout.strip())
    result=dict(name=v[0])
    for key,value in zip(['GPU_util_pct','memory_controller_util_pct','memory_used_MiB','memory_total_MiB'],v[1:]):
        try:result[key]=float(value)
        except ValueError:result[key]=None
    return result
a=argparse.ArgumentParser();a.add_argument('--false-protocol',type=Path,required=True);a.add_argument('--expected-sha',required=True);a.add_argument('--source-review',type=Path,required=True);args=a.parse_args()
base=args.false_protocol.resolve(strict=True);p=read(base);review=read(args.source_review)
assert base.parent==H and pin(base)['sha256']==args.expected_sha
assert p['CPU_run_authorized'] is False and p['GPU_run_authorized'] is False and p['native_online_authorized'] is False and p['device']=='cuda:0'
assert p['recipe']['paired_rounds']==1000 and p['recipe']['common_WM_warmup_updates']==100 and p['recipe']['first_healthy_round']==100 and p['recipe']['stop_after_first_healthy'] is True
assert review['passed']==review['total'] and all(review['checks'].values())
for v in [*p['sources'].values(),*p['inputs'].values(),p['recipe_pin']]:assert v is not None and pin(v['path'])==v
observed=GPU();assert '5090' in observed.get('name','')
protocol=H/'protocol_CPU_GPU_True_run1.json';process=H/'process_run1'
assert not any(v.exists() for v in [protocol,process,Path(p['output'])])
p['CPU_run_authorized']=True;p['GPU_run_authorized']=True
p['root_review']=dict(parent_protocol=pin(base),independent_SOURCE_review=pin(args.source_review),launcher=pin(__file__),GPU_before_run=observed,
    scope='First actual matched offline A1/A2 learning: commonWM100 then1000paired rounds target. Fresh Actor/Critic1733, old WM5000 weights+frozen modelnormalizers with newoptimizers. First healthy paired100+ETA exits; Root archives and pausesGoal. No native online, no method/task benefit claim. Hardware samples whole-machine; torch peak is worker-owned.')
write(protocol,p);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log';samples=[]
command=[sys.executable,'-X','utf8',p['sources']['trainer']['path'],'--protocol',str(protocol),'--run'];start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();last_progress=start
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se,(process/'GPU_samples.jsonl').open('x',encoding='utf8') as gs:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,process=str(process),output=p['output'])),flush=True)
    while child.poll() is None:
        row=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,**GPU());samples.append(row);gs.write(json.dumps(row)+'\n');gs.flush()
        if time.perf_counter()-last_progress>=15:
            current=None;log=Path(p['output'])/'paired_updates.jsonl'
            if log.exists():
                raw=log.read_bytes();lines=raw.splitlines()
                if not raw.endswith(b'\n'):lines=lines[:-1]
                if lines:
                    try:current=json.loads(lines[-1]).get('paired_round')
                    except (ValueError,KeyError):pass
            print(json.dumps(dict(elapsed_s=row['elapsed_s'],latest_paired_round=current,GPU_util_pct=row.get('GPU_util_pct'),machine_memory_used_MiB=row.get('memory_used_MiB'))),flush=True);last_progress=time.perf_counter()
        try:child.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
    rc=child.returncode
valid=[v for v in samples if isinstance(v.get('GPU_util_pct'),(int,float))]
stats=dict(samples=len(samples),valid_samples=len(valid),sample_interval_target_s=2,scope='Whole-machine nvidia-smi snapshots including setup/health, not attributed kernel occupancy or per-process VRAM.')
for key in ['GPU_util_pct','memory_controller_util_pct','memory_used_MiB']:
    values=[v[key] for v in valid if isinstance(v.get(key),(int,float))];stats[key]=dict(min=min(values),median=statistics.median(values),max=max(values),mean=statistics.mean(values)) if values else None
write(process/'GPU_summary.json',stats)
write(process/'terminal.json',dict(schema='phone_tripod_matched_GPU_training_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,protocol=pin(protocol),source=pin(p['sources']['trainer']['path']),launcher=pin(__file__),stdout=pin(stdout),stderr=pin(stderr),GPU_samples=pin(process/'GPU_samples.jsonl'),GPU_summary=pin(process/'GPU_summary.json'),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
