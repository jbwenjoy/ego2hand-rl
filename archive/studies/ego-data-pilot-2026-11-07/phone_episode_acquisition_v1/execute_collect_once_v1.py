"""One reviewed acquisition process. No automatic retries or old-output reuse."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys,time
from native_phone_reader import pin

ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);a=ap.parse_args()
p=json.loads(a.protocol.read_text(encoding='utf8'));root=Path(__file__).resolve().parent
assert p['CPU_run_authorized'] is True and p['new_training_authorized'] is False
assert not Path(p['output']).exists()
runner=p['source_pins']['collector'];assert pin(runner['path'])==runner
folder=root/'process_run1';folder.mkdir()
cmd=[sys.executable,'-X','utf8',runner['path'],'--protocol',str(a.protocol.resolve()),'--run']
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with (folder/'stdout.log').open('x',encoding='utf8') as o,(folder/'stderr.log').open('x',encoding='utf8') as e:
    child=subprocess.Popen(cmd,cwd=root,stdout=o,stderr=e,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps(dict(status='STARTED_ONCE',pid=child.pid,started_UTC=start)),flush=True)
    code=child.wait()
j=dict(schema='phone_episode_acquisition_process_terminal:1',pid=child.pid,returncode=code,
    started_UTC=start,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-t,
    command=cmd,source=pin(__file__),runner=runner,protocol=pin(a.protocol),stdout=pin(folder/'stdout.log'),
    stderr=pin(folder/'stderr.log'),automatic_retry=False)
with (folder/'terminal.json').open('x',encoding='utf8') as f:json.dump(j,f,indent=2)
print(json.dumps(j),flush=True);sys.exit(code)
