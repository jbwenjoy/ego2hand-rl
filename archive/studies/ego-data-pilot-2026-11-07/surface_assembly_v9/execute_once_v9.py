"""One frozen staged CPU run, preserving actual PID, logs and terminal status."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time

r=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
cmd=[sys.executable,'-X','utf8',str(r/'solve_assembly_support_v9.py'),'--protocol',str(r/'protocol_v1.json'),'--run']
t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/'run1_stdout.log').open('x',encoding='utf8') as o,(r/'run1_stderr.log').open('x',encoding='utf8') as e:
    child=subprocess.Popen(cmd,cwd=r.parent,stdout=o,stderr=e)
    print(json.dumps({'status':'STARTED_ONCE','pid':child.pid,'started_UTC':start}),flush=True)
    code=child.wait()
record={'schema':'surface_assembly_process_terminal:9','pid':child.pid,'returncode':code,'command':cmd,
 'started_UTC':start,'ended_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.perf_counter()-t,
 'source':pin(r/'solve_assembly_support_v9.py'),'protocol':pin(r/'protocol_v1.json'),
 'stdout':pin(r/'run1_stdout.log'),'stderr':pin(r/'run1_stderr.log'),'automatic_retry':False}
with (r/'run1_terminal.json').open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
sys.exit(code)
