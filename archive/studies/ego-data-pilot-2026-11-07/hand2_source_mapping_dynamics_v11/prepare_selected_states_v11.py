"""Pure saved JSON→NPZ preparation; no model/forward/step/LP call."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('source_mapping_dynamic_metadata',H/'run_source_mapping_dynamics_v11.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--protocol-output',type=Path,required=True);a=ap.parse_args()
plan=m.read(a.plan)
if a.output.exists() or a.protocol_output.exists():raise ValueError('Fresh selected-state preparation and protocol only')
if plan['CPU_run_authorized'] is not False:raise ValueError('Preparation only emits CPUFalse protocol')
binding,choices,screening=m.source_choices(plan)
a.output.mkdir();selections=[]
for v in choices:
    path=a.output/(v['branch']+'_state.npz')
    np.savez_compressed(path,qpos27=np.asarray(v['verified_qpos27'],dtype=float),qvel26=np.asarray(v['verified_qvel26'],dtype=float),ctrl20=np.asarray(v['verified_ctrl20'],dtype=float),x46=np.asarray(v['verified_x46'],dtype=float))
    selections.append(dict(branch=v['branch'],record=v['record'],state=m.pin(path),seed=v['seed'],rank=v['rank'],control_origin=v['control_origin']))
receipt=dict(schema='source_mapping_dynamic_selected_states:11',source=m.pin(Path(__file__)),runner=m.pin(H/'run_source_mapping_dynamics_v11.py'),plan=m.pin(a.plan),selections=selections,screening=screening,new_native_model_FK_LP_physics_calls=0)
with (a.output/'prepared_selections.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
prepared_protocol={**plan,'selections':selections,'prepared_state_receipt':m.pin(a.output/'prepared_selections.json')}
with a.protocol_output.open('x') as f:json.dump(prepared_protocol,f,indent=2,allow_nan=False)
print(json.dumps({'receipt':m.pin(a.output/'prepared_selections.json'),'protocol':m.pin(a.protocol_output),'selections':selections}))
