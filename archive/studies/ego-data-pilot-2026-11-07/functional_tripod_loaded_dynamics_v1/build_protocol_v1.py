"""Create unauthorized JSON/hash-only loaded dynamic plan."""
from pathlib import Path
import hashlib,importlib.util,json
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
    spec=importlib.util.spec_from_file_location('loaded_dynamic_metadata_builder',H/'run_loaded_dynamics_v1.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);f=r.physics()
    c=dict(schema=r.SCHEMA,CPU_run_authorized=False,integration_training_GPU_other_than_declared_CPU_steps_authorized=False,output=str(H/'completed_run1'),
        dt_s=f.DTS,duration_s=f.DURATION,gates=f.GATES,old_two_pad_gates=f.OLD_GATES,dt_pair_gates=f.PAIR_GATES,routes=r.ROUTES,
        budget=dict(candidates=1,control_routes=4,dt_cases=2,maximum_cases=8,maximum_integration_steps=76800,preload_preparation_forward=0,maximum_initial_forward=8),
        inputs={k:pin(P/v) for k,v in dict(model_mjb='hand2_holding_v1/compiled_nominal_v2/model.mjb',model_binding='hand2_holding_v1/compiled_nominal_v2/binding.json',
            preload_report='functional_tripod_native_preload_v1/completed_run1/report.json',preload_protocol='functional_tripod_native_preload_v1/protocol_CPUTrue_run1.json',
            preload_saved_review='functional_tripod_native_preload_v1/saved_review_v1/report.json').items()},
        source_pins=dict(runner=pin(H/'run_loaded_dynamics_v1.py'),builder=pin(__file__),frozen_dynamicv3=pin(r.FROZEN)),automatic_retry=False)
    p=H/'protocol_CPUFalse_v1.json'
    with p.open('x',encoding='utf8') as out:json.dump(c,out,indent=2,allow_nan=False)
    print(json.dumps(dict(protocol=pin(p),CPU_run_authorized=False)))
if __name__=='__main__':main()
