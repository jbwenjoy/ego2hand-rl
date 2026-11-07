"""Capture one CPUFalse metadata call or one explicitly authorized actual call."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def pin(p):
    p=p.resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['metadata','actual'],required=True);a=p.parse_args()
stage=a.stage
terminal=ROOT/(stage+'_process_terminal_v1.json')
if terminal.exists():raise ValueError('Once only; preserve existing terminal receipt')
protocol=ROOT/('metadata_CPUFalse_protocol_v1.json' if stage=='metadata' else 'cpu_protocol_v1.json')
c=json.loads(protocol.read_text())
assert c['CPU_run_authorized'] is (stage=='actual')
if stage=='actual':
    prior=json.loads((ROOT/'metadata_process_terminal_v1.json').read_text())
    assert prior['returncode']==0
    meta=json.loads((ROOT/'metadata_stdout_v1.log').read_text())
    assert meta['variants']==['matched_constant_u_seed0_rank0'] and meta['maximum_steps']==19200
cmd=[sys.executable,str(ROOT/'run_cpu_constant_u_baseline_v1.py'),'--protocol',str(protocol)]
if stage=='actual':cmd.append('--run')
out=ROOT/(stage+'_stdout_v1.log');err=ROOT/(stage+'_stderr_v1.log')
if out.exists() or err.exists():raise ValueError('Preserve prior logs')
started=time.perf_counter()
with out.open('xb') as so,err.open('xb') as se:
    process=subprocess.Popen(cmd,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
    pid=process.pid
    code=process.wait(timeout=120)
record={'stage':stage,'returncode':code,'PID':pid,'elapsed_s':time.perf_counter()-started,
    'command':cmd,'protocol':pin(protocol),'stdout':pin(out),'stderr':pin(err),'no_automatic_retry':True}
with terminal.open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
print(json.dumps({'terminal':pin(terminal),'returncode':code,'PID':pid,'elapsed_s':record['elapsed_s']}))
if stage=='metadata' and code==0:print(out.read_text())
raise SystemExit(code)
