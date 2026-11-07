"""Create CPUFalse two-source measured-control plan; no backend execution."""
from pathlib import Path
import hashlib,importlib.util,json
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
    spec=importlib.util.spec_from_file_location('loaded_dynamic_v2_metadata_builder',H/'run_loaded_dynamics_v2.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);f=r.physics()
    sources=[]
    for sid,directory,protocol,review in [('exact','completed_run1','protocol_CPUTrue_run1.json','saved_review_v1/report.json'),
                                         ('source1mm','completed_source1mm_run1','protocol_source1mm_CPUTrue_run1.json','source1mm_saved_review_v1/report.json')]:
        base=P/'functional_tripod_native_preload_v1'
        sources.append(dict(source_id=sid,preload_report=pin(base/directory/'report.json'),preload_protocol=pin(base/protocol),preload_saved_review=pin(base/review)))
    c=dict(schema=r.SCHEMA,CPU_run_authorized=False,integration_training_GPU_other_than_declared_CPU_steps_authorized=False,output=str(H/'completed_run1'),
        dt_s=f.DTS,duration_s=f.DURATION,gates=f.GATES,old_two_pad_gates=f.OLD_GATES,dt_pair_gates=f.PAIR_GATES,routes=r.ROUTES,
        budget=dict(candidates=2,control_routes_per_source=4,dt_cases=2,maximum_cases=16,maximum_integration_steps=153600,preload_preparation_forward=0,maximum_initial_forward=16),
        inputs=dict(model_mjb=pin(P/'hand2_holding_v1/compiled_nominal_v2/model.mjb'),model_binding=pin(P/'hand2_holding_v1/compiled_nominal_v2/binding.json')),
        preload_sources=sources,source_pins=dict(runner=pin(H/'run_loaded_dynamics_v2.py'),builder=pin(__file__),frozen_dynamicv3=pin(r.FROZEN)),automatic_retry=False)
    path=H/'protocol_CPUFalse_v2.json'
    with path.open('x',encoding='utf8') as out:json.dump(c,out,indent=2,allow_nan=False)
    print(json.dumps(dict(protocol=pin(path),CPU_run_authorized=False)))
if __name__=='__main__':main()
