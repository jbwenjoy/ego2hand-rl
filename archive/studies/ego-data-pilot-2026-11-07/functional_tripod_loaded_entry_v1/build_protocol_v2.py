"""SOURCE/JSON only: one registered original eligibility subset; no model call."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
oldpath=H/'protocol_CPUFalse_v1.json';cfg=read(oldpath)
assert cfg['CPU_static_run_authorized'] is False
parent=read(cfg['inputs']['original_geometry_report']['path'])
matches=[]
for item in parent['candidates']:
    c=read(item['candidate']['path'])
    if item['geometry_pass'] and c['source_entry']['name']=='source1mm__index_material_toward_COM_30mm':matches.append(item['candidate'])
assert len(matches)==1 and matches[0] in cfg['selected_parent_candidates']
previous=H/'completed_run1/report.json';vr=read(previous)
boundary=[v['candidate'] for v in vr['candidates'] if read(v['candidate']['path'])['source_entry']['name']=='source1mm__index_material_toward_COM_30mm']
assert len(boundary)==1
cfg['selected_parent_candidates']=matches
cfg['output']=str(H/'completed_run2')
cfg['sources']['runner']=pin(H/'solve_compression_geometry_v2.py')
cfg['sources']['protocol_builder']=pin(Path(__file__))
cfg['inputs']['retained_compression_v1_report']=pin(previous)
cfg['inputs']['retained_numeric_boundary_candidate']=boundary[0]
cfg['proposal']['interior_target_margin_m']=2e-6
cfg['retained_CPUFalse_v1_protocol']=pin(oldpath)
cfg['numeric_feasibility_revision']='Original eligible starting state only, explicit one-state subset and2um proposal interior. Original hard>=0 /[-100,+200]um and compression[-90,-60]um unchanged. The v1 negative margin remains failed; no warmstart from it, no valid-exact repeat.'
assert cfg['budget']==read(oldpath)['budget']
assert {k:v for k,v in cfg['proposal'].items() if k!='interior_target_margin_m'}=={k:v for k,v in read(oldpath)['proposal'].items() if k!='interior_target_margin_m'}
out=H/'protocol_CPUFalse_v2.json'
with out.open('x') as f:json.dump(cfg,f,indent=2,allow_nan=False)
print(json.dumps(pin(out)))
