"""Saved raw-array check that v2 changed no numeric label output."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
paths=[H/f'completed_v{i}/all2952_pair22_physical_labels.npz' for i in [1,2]]
with np.load(paths[0],allow_pickle=False) as old,np.load(paths[1],allow_pickle=False) as new:
    assert old.files==new.files
    checks={k:old[k].dtype==new[k].dtype and old[k].shape==new[k].shape and old[k].tobytes()==new[k].tobytes() for k in old.files}
identical=(H/'completed_v1/row_identities.json').read_bytes()==(H/'completed_v2/row_identities.json').read_bytes()
report=dict(schema='causal_impulse_serialization_numeric_preservation:2',source=pin(__file__),checks=checks,
    all_arrays_bit_identical=all(checks.values()),row_identities_bytes_identical=identical,
    old_output=pin(paths[0]),new_output=pin(paths[1]),correction=pin(H/'serialization_correction_v2.json'),
    verification_counts=dict(saved_export_NPZ_reads=2,new_native_NN_AD_calls=0),
    actual_both_exports_saved_branch_NPZ_reads=5884,actual_both_exports_native_NN_calls=0)
with (H/'numeric_preservation_v2.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(report=pin(H/'numeric_preservation_v2.json'),all_arrays_bit_identical=report['all_arrays_bit_identical'])))
assert all(checks.values()) and identical
