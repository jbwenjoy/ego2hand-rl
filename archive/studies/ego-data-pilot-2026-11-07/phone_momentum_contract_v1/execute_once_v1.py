"""Pinned, bounded saved-data/analytic contract execution; no retry."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,datetime

ROOT=Path(__file__).resolve().parent;P=ROOT.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
protocol=ROOT/'protocol_run1.json';process=ROOT/'process_run1';output=ROOT/'completed_run1'
assert not any(p.exists() for p in [protocol,process,output]),'one actual execution, no overwrite'
record=dict(schema='phone_momentum_saved_TRAIN_and_analytic_contract_protocol:1',
    source={'verifier':pin(ROOT/'verify_contract_v1.py'),'core':pin(ROOT/'momentum_contract.py'),'launcher':pin(__file__)},
    inputs={'physical_parameters':pin(P/'phone_object_wrench_labels_v1/completed_v1/physical_parameters.json'),
        'train_npz':pin(P/'phone_episode_acquisition_v1/acquired_v1/train.npz'),
        'published_all960_labels':pin(P/'phone_object_wrench_labels_v1/completed_v1/all960_interval_wrench_labels_and_residuals.npz'),
        'published_row_identities':pin(P/'phone_object_wrench_labels_v1/completed_v1/all960_row_identities.json'),
        'label_interface':pin(P/'phone_object_wrench_labels_v1/label_interface_v1.json')},
    output=str(output),neural_training_authorized=False,native_physics_authorized=False,
    primary='all 684 original eligible TRAIN rows, no fit or model selection; both impulse residual units separate',
    fixtures='rotated anisotropic independent NumPy closure; world covariance; ballistic; analytic impulse AD; wrong axes/origin',
    partial_rows='original 36 TRAIN partial targets excluded by original eligibility, never replaced with H targets',
    scope='nominal momentum-only reusable loss/velocity closure, no position/R integration/contact causality qualification')
write(protocol,record);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',str(ROOT/'verify_contract_v1.py'),'--protocol',str(protocol)]
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);rc=child.wait()
terminal=dict(schema='bounded_saved_contract_process_terminal:1',pid=child.pid,returncode=rc,command=command,
    started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,
    source=pin(ROOT/'verify_contract_v1.py'),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',terminal)
print(json.dumps(dict(terminal=pin(process/'terminal.json'),returncode=rc,elapsed_s=terminal['elapsed_s'])))
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
