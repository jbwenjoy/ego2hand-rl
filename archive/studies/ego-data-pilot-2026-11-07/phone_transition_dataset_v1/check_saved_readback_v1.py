"""Serialized dataset readback only, not an independent dynamics experiment."""
from pathlib import Path
import hashlib,json
import numpy as np
import transition_codec_v1 as codec
H=Path(__file__).resolve().parent
D=H/'dataset_v1'
pin=lambda p:dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
r=json.loads((D/'report.json').read_text())
for record in r['outputs'].values():assert pin(Path(record['path']))==record
with np.load(D/'transitions322_F64.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
X,Y,U,delta=(a[k] for k in ['input_state58','next_state58','control20','physical_world_delta52'])
assert X.shape==Y.shape==(322,58) and U.shape==(322,20) and delta.shape==(322,52)
assert all(np.isfinite(v).all() for v in a.values())
assert np.array_equal(a['source_boundary_index'],np.repeat(np.arange(7),46))
assert np.array_equal(a['branch_id'],np.repeat(np.arange(1,162),2)) and np.array_equal(a['H2_step'],np.tile([0,1],161))
assert np.array_equal(X[1::2],Y[::2]) and np.array_equal(U[::2],a['branch_controls2x20'][:,0]) and np.array_equal(U[1::2],a['branch_controls2x20'][:,1])
for row in np.arange(0,322,2):assert np.array_equal(X[row],a['actual_boundary_state58'][a['source_boundary_index'][row]])
maximum=max(float(np.max(abs(codec.apply_delta52(x,d)-y))) for x,y,d in zip(X,Y,delta));assert maximum<=2e-12
report=dict(schema='phone_transition_saved_serialization_readback:1',source=pin(Path(__file__)),dataset_report=pin(D/'report.json'),NPZ=pin(D/'transitions322_F64.npz'),passed=True,rows=322,all7x46_cohorts_and161_H2_pairs_preserved=True,all_input_controls_and_second_step_state_link_bytes_match=True,maximum_decoded_delta_state58_abs=maximum,new_model_FK_steps_LP_NN_training=0,scope='Serialized shapes/finites/order/source-boundary/H2 links and saved-codec arithmetic; not independent physics, Markov or learning qualification.')
out=H/'saved_readback_v1.json'
with out.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'report':pin(out),'rows':322,'max_decode_error':maximum}))
