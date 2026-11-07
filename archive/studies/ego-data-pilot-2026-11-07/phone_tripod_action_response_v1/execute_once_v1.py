"""Root's one-shot held-out native action-response acquisition, never learning."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys, time
H=Path(__file__).resolve().parent

def pin(path):
    path=Path(path).resolve(strict=True);data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)

a=argparse.ArgumentParser();a.add_argument('--false-protocol',type=Path,required=True);a.add_argument('--expected-sha',required=True);a.add_argument('--peer-review',type=Path,required=True);args=a.parse_args()
base=args.false_protocol.resolve(strict=True);p=read(base);review=read(args.peer_review)
assert base.parent==H and pin(base)['sha256']==args.expected_sha
assert p['schema']=='phone_tripod_heldout_action_response_CPU_protocol:1'
assert p['CPU_run_authorized'] is False and p['NN_training_GPU_authorized'] is False
assert p['rules']['contexts']==8 and p['rules']['planned_branches']==328 and p['rules']['maximum_native_steps']==31488
assert p['rules']['all8_baselines_before_any_new_side'] is True and p['permanent_data_role']=='ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE'
assert review['known_blocker'] is False and review['reviewed_runner']==p['sources']['runner'] and review['reviewed_common']==p['sources']['common']
for value in [*p['inputs'].values(),*p['sources'].values()]:assert pin(value['path'])==value
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(path.exists() for path in [protocol,process,Path(p['output'])])
meta=subprocess.run([sys.executable,'-X','utf8',p['sources']['runner']['path'],'--protocol',str(base)],capture_output=True,text=True,encoding='utf8')
assert meta.returncode==0, meta.stderr
p['CPU_run_authorized']=True
p['root_review']=dict(parent_protocol=pin(base),peer_review=pin(args.peer_review),launcher=pin(__file__),scope='Run predeclared8baseline global barrier and available bounded finite sides once. No NN, parameter change, data fitting, pose aid, safety recentering or retry. All branches permanently heldout; partial/unavailable retained.')
write(protocol,p);process.mkdir();write(process/'metadata_preflight.json',dict(returncode=meta.returncode,stdout=meta.stdout,stderr=meta.stderr,parent_protocol=pin(base)))
command=[sys.executable,'-X','utf8',p['sources']['runner']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
stdout=process/'stdout.log';stderr=process/'stderr.log'
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps(dict(pid=child.pid,process=str(process),maximum_native_steps=31488,NN_learning_calls=0)),flush=True)
    rc=child.wait()
write(process/'terminal.json',dict(schema='phone_tripod_heldout_native_action_response_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,protocol=pin(protocol),source=p['sources']['runner'],launcher=pin(__file__),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False,new_learning_started=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
