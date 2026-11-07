"""Run the frozen CPU search once and save its true process terminal status."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time

r=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
cmd=[sys.executable,'-X','utf8',str(r/'solve_surface_anchor_pose_v8.py'),'--protocol',str(r/'protocol_v1.json'),'--run']
t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/'run1_stdout.log').open('x',encoding='utf8') as o,(r/'run1_stderr.log').open('x',encoding='utf8') as e:
    child=subprocess.Popen(cmd,cwd=r.parent,stdout=o,stderr=e)
    print(json.dumps({'status':'STARTED_ONCE','pid':child.pid,'started_UTC':start}),flush=True)
    code=child.wait()
record={'schema':'surface_anchor_search_process_terminal:8','pid':child.pid,'command':cmd,'returncode':code,
 'started_UTC':start,'ended_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.perf_counter()-t,
 'source':pin(r/'solve_surface_anchor_pose_v8.py'),'protocol':pin(r/'protocol_v1.json'),
 'stdout':pin(r/'run1_stdout.log'),'stderr':pin(r/'run1_stderr.log'),'automatic_retry':False}
with (r/'run1_terminal.json').open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
sys.exit(code)
