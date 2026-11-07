"""One CPU runner invocation with durable PID, logs and terminal receipt."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys,time
from evaluator_contact_design_v7 import pin

r=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--prefix',required=True)
a=ap.parse_args();runner=(r/a.runner).resolve();protocol=(r/a.protocol).resolve()
if runner.parent!=r or protocol.parent!=r or not a.prefix.replace('_','').isalnum():raise ValueError('Local named stage only')
cmd=[sys.executable,'-X','utf8',str(runner),'--protocol',str(protocol),'--run']
t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
stdout=r/(a.prefix+'_stdout.log');stderr=r/(a.prefix+'_stderr.log');terminal=r/(a.prefix+'_terminal.json')
if terminal.exists():raise ValueError('No automatic repeat/overwrite')
with stdout.open('x',encoding='utf8') as o,stderr.open('x',encoding='utf8') as e:
    child=subprocess.Popen(cmd,cwd=r.parent,stdout=o,stderr=e)
    print(json.dumps({'status':'STARTED_ONCE','pid':child.pid,'started_UTC':start}),flush=True)
    code=child.wait()
record={'schema':'native_contact_design_CPU_terminal:7','pid':child.pid,'command':cmd,
 'started_UTC':start,'ended_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'wall_seconds':time.perf_counter()-t,'returncode':code,'automatic_retry':False,
 'source':pin(runner),'protocol':pin(protocol),'stdout':pin(stdout),'stderr':pin(stderr)}
with terminal.open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
sys.exit(code)
