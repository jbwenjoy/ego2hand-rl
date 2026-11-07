"""One closed noncommuting world-rotation fixture; no actual data/model."""
from pathlib import Path
import hashlib,json
import numpy as np
import transition_codec_v1 as c
H=Path(__file__).resolve().parent
pin=lambda p:dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
R0=c.exp_world(np.array([.47,-.21,.16]));dr=np.array([-.14,.09,.07]);R1=c.exp_world(dr)@R0
x=np.r_[np.arange(20)*.01,np.arange(20)*.02,[.3,-.2,.5],R0.ravel(),[.2,.1,-.4],[.4,-.3,.1]]
y=x.copy();y[:43]+=.003;y[43:52]=R1.ravel();y[52:]+=.02
d=c.delta52(x,y);recovered=c.apply_delta52(x,d);raw=c.codec_roundtrip(x,[.01,-.002,.003])
err=float(np.max(abs(recovered-y)));codec=float(np.max(abs(raw-x)));wrong=float(np.max(abs(R0@c.exp_world(d[43:46])-R1)))
assert err<2e-12 and codec<2e-12 and np.max(abs(d[43:46]-dr))<2e-12 and wrong>.01
report=dict(schema='physical_world_delta52_codec_fixture:1',source=pin(Path(__file__)),codec_source=pin(H/'transition_codec_v1.py'),passed=True,fixture='Noncommuting R1=Exp(world_dr)R0 plus translational/joint/velocity deltas and nonzero COM offset',Exp_delta_Rcur_roundtrip_max_abs=err,state58_COM_inverse_roundtrip_max_abs=codec,wrong_right_multiplication_error=wrong,no_actual_data_model_FK_LP_physics_or_training=True)
with (H/'codec_fixture_v1.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
