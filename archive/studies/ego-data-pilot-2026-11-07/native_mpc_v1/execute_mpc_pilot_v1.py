"""One locked subprocess invocation, preserving PID, stdout, stderr and exit."""
from pathlib import Path
import datetime
import json
import subprocess
import sys
import time
from rollout_probe_v1 import pin

root=Path(__file__).resolve().parent
started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
command=[sys.executable,'-X','utf8',str(root/'run_mpc_feasibility_v1.py'),
         '--protocol',str(root/'mpc_feasibility_protocol_v1.json'),'--run']
with (root/'pilot_stdout_v1.log').open('x',encoding='utf8') as stdout, (root/'pilot_stderr_v1.log').open('x',encoding='utf8') as stderr:
    process=subprocess.Popen(command,cwd=root.parent,stdout=stdout,stderr=stderr)
    print(json.dumps({'status':'STARTED_ONCE','pid':process.pid,'started_UTC':started}),flush=True)
    code=process.wait()
terminal={'schema':'locked_MPC_process_terminal:1','pid':process.pid,'command':command,
 'started_UTC':started,'ended_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'elapsed_s':time.perf_counter()-t,'exit_code':code,'automatic_retry':False,
 'source':pin(root/'run_mpc_feasibility_v1.py'),'protocol':pin(root/'mpc_feasibility_protocol_v1.json'),
 'stdout':pin(root/'pilot_stdout_v1.log'),'stderr':pin(root/'pilot_stderr_v1.log')}
with (root/'pilot_process_terminal_v1.json').open('x',encoding='utf8') as f:json.dump(terminal,f,indent=2)
print(json.dumps(terminal),flush=True)
sys.exit(code)
