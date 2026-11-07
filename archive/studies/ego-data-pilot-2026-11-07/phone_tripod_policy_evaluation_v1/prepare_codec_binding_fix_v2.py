"""Preserve failed v1 and bind the exact actual_state codec used by the Reader."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
old=H/'check_zero_controller_v1.py';s=old.read_text(encoding='utf8')
oldcode="P/'phone_state_time_contract_v1/state_contract.py'";newcode="P/'phone_transition_dataset_v1/transition_codec_v1.py'"
assert s.count(oldcode)==1
s=s.replace(oldcode,newcode).replace("H/'zero_CPUFalse_v1.json'","H/'zero_CPUFalse_v2.json'").replace("H/'zero_qualified_run1'","H/'zero_qualified_run2'")
new=H/'check_zero_controller_v2.py'
with new.open('x',encoding='utf8',newline='\n') as f:f.write(s)
manifest=dict(schema='scheduled_controller_actual_state_codec_binding_fix:2',source=pin(__file__),prior_source=pin(old),corrected_source=pin(new),
    prior_terminal=pin(H/'zero_process_run1/terminal.json'),prior_failure=pin(H/'zero_qualified_run1/failure.json'),unchanged_kernel=pin(H/'controlled_native_kernel_v1.py'),
    correction='Bind phone_transition_dataset_v1.transition_codec_v1.actual_state, exactly the frozen Reader877 implementation. The state_contract module provides named conversions, not actual_state. Only this dependency and protocol/output paths changed.',
    previous_failure_scope='AttributeError in first callback after first RESET_FORWARD, before first mj_step; no completed integrations or NN updates. Native counters were not serialized on this failure; this scope is derived from saved traceback and executed source, not a counter receipt.',
    no_gate_or_model_or_timestep_or_controller_mapping_change=True)
with (H/'codec_binding_revision_v2.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2)
print(json.dumps(dict(source=pin(new),revision=pin(H/'codec_binding_revision_v2.json'))))
