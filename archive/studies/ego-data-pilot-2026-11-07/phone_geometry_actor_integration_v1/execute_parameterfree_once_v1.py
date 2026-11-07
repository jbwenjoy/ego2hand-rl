"""Root runs the parameter-free task/replay/proxy contract, zero NN/native."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'parameterfree_CPUFalse_v1.json';c=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='6ad9d775a83f5143fa9ab4edbc03169c78bc2c858e0f6d7399d7067c272554c3'
assert c['parameterfree_cpu_run_authorized'] is False and c['NN_native_training_authorized'] is False
assert c['checker']==pin(c['checker']['path']) and c['checker']['sha256']=='12c1693455074b887b65a45f12992e4e310c1c1d8e134ce9324557d8d7d77df4'
assert c['budget']['Actor_WM_Critic_constructors']==c['budget']['NN_forward_backward']==c['budget']['optimizer_updates']==c['budget']['native_model_FK_forward_steps']==c['budget']['GPU_calls']==0
for r in [*c['sources'].values(),*c['inputs'].values()]:assert pin(r['path'])==r
protocol=H/'parameterfree_CPUTrue_run1.json';process=H/'parameterfree_process_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['parameterfree_cpu_run_authorized']=True;c['root_review']=dict(source_protocol=pin(base),launcher=pin(__file__),scope='Parameter-free CPU contract only, actual3TensorFK/3inputAD, FakeActor plaincallable. Zero NN/native/training; synthetic packets are not physical receipt proof. Preserve any failure, no retry.')
write(protocol,c);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['checker']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,process=str(process))),flush=True);rc=child.wait()
write(process/'terminal.json',dict(schema='phone_actor_parameterfree_contract_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,source=pin(c['checker']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
