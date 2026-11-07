"""Preserve v1; add frozen full native dt-pair endpoint rule, metadata only."""
from pathlib import Path
import hashlib,json,runpy,sys
H=Path(__file__).resolve().parent
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
m=runpy.run_path(str(H/'evaluate_learned_actors_v2.py'));p=read(H/'learned_CPUFalse_v1.json')
p.update(schema=m['SCHEMA'],source=pin(H/'evaluate_learned_actors_v2.py'),output=str(H/'learned_fixed1000_run2'),native_dt_pair_gates=m['PAIR_GATES'],native_dt_pair_rule=dict(source=p['sources']['kernel_frozen_loop'],constant='PAIR_GATES',function='dt_pair_summary',cases=4,unique_variants=2,scope='Original endpoint COM2mm/fullorientation2deg AND bothcomplete2s, all abort records retained'),preparation=pin(__file__),retained_v1_protocol=pin(H/'learned_CPUFalse_v1.json'))
write(H/'learned_CPUFalse_v2.json',p);m['metadata_preflight'](H/'learned_CPUFalse_v2.json');assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
write(H/'learned_metadata_v2.json',dict(schema='phone_tripod_final_Actor_metadata:2',status='PASS_METADATA_PENDING_FINAL1000_NOT_EXECUTABLE',protocol=pin(H/'learned_CPUFalse_v2.json'),source=p['source'],Torch_imported=False,MuJoCo_imported=False,NN_arrays_native_calls=0))
write(H/'learned_revision_v2.json',dict(schema='phone_tripod_final_Actor_revision:2',reason='Root SOURCE review identified omitted endpoint dt-pair rule in v1 aggregation, before any NN/native run',retained_v1_source=pin(H/'evaluate_learned_actors_v1.py'),new_v2_source=p['source'],exact_diff=pin(H/'learned_v1_to_v2.diff'),protocol=pin(H/'learned_CPUFalse_v2.json'),formula_scope='Direct frozen dt_pair_summary; no new gates/tolerance, no native or NN budget change',actual_NN_native_calls=0))
print(json.dumps(dict(protocol=pin(H/'learned_CPUFalse_v2.json'),source=p['source'],metadata=pin(H/'learned_metadata_v2.json'),revision=pin(H/'learned_revision_v2.json'))))
