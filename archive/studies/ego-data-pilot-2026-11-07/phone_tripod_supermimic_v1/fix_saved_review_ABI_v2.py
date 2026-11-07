"""Map the review's old local role labels to the final saved stateactor ABI."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
old=H/'verify_saved_first100_v1.py';s=old.read_text(encoding='utf8')
a="arms=ck['arms'];modes=['A1_tracking','A2_engineered_material_proxy'];opt=ck['optimizers'];grads=ck['gradient_snapshots']"
b="""modes=['A1_tracking','A2_engineered_material_proxy'];fields={'actor':'stateactor','critic':'statecritic','target_critic':'statecritic_target'}
armsp=ck['arms'];arms={mode:{role:armsp[mode][key] for role,key in fields.items()} for mode in modes}
initial['arms']={mode:{role:initial['arms'][mode][key] for role,key in fields.items()} for mode in modes}
opt=ck['optimizers'];grads=ck['gradient_snapshots']"""
assert s.count(a)==1;s=s.replace(a,b).replace('root_saved_first100_review_v1','root_saved_first100_review_v2')
new=H/'verify_saved_first100_v2.py'
with new.open('x',encoding='utf8',newline='\n') as f:f.write(s)
with (H/'saved_review_ABI_revision_v2.json').open('x',encoding='utf8') as f:json.dump(dict(schema='paired_saved_review_ABI_correction:2',old=pin(old),new=pin(new),source=pin(__file__),correction='Review-only labels mapped to frozen trainer final stateactor/statecritic/statecritic_target. No checkpoint, training source, data or physics changed.',prior_failed_review_counts=dict(own_CPU_payload_loads=2,NN=0,GPU=0,native=0),prior_error="KeyError actor after two own payload reads; no report produced",new_expected_CPU_payload_loads=2),f,indent=2)
