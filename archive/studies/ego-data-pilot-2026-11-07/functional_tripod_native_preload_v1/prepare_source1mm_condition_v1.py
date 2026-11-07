"""Bind the separately repaired1mm geometry to unchanged native preload rules."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
old=H/'protocol_CPUFalse_v2.json';c=json.loads(old.read_text(encoding='utf8'))
c['inputs']['candidate_report']=pin(P/'functional_tripod_loaded_entry_v1/completed_run2/report.json')
c['output']=str(H/'completed_source1mm_run1')
c['root_condition_revision']=dict(parent_protocol=pin(old),builder=pin(__file__),
    scope='Only separately qualified1mm functional geometry after declared2um search interior. Original negative-margin candidate remains failed. Unchanged core/runner/40FW/rules/physics/target; no exact-case repeat.')
write(H/'protocol_source1mm_CPUFalse_v2.json',c)
src=H/'execute_once_v2.py';s=src.read_text(encoding='utf8')
s=s.replace('protocol_CPUFalse_v2.json','protocol_source1mm_CPUFalse_v2.json').replace('protocol_CPUTrue_run1.json','protocol_source1mm_CPUTrue_run1.json').replace("process_run1'","process_source1mm_run1'")
with (H/'execute_source1mm_once_v2.py').open('x',encoding='utf8') as f:f.write(s)
print(json.dumps(dict(protocol=pin(H/'protocol_source1mm_CPUFalse_v2.json'),launcher=pin(H/'execute_source1mm_once_v2.py'))))
