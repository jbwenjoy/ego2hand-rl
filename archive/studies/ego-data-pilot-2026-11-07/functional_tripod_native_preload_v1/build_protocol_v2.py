"""One source/JSON-only CPUFalse preload protocol; no native execution."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate-report',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not a.output.is_absolute() or a.output.parent.resolve()!=H.resolve() or a.output.exists():raise ValueError('Fresh own output')
    source=H/'run_native_preload_v2.py';sp=importlib.util.spec_from_file_location('preload_runner_metadata_constants',source);r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
    c=dict(schema=r.SCHEMA,native_forward_authorized=False,integration_training_GPU_authorized=False,output=str(a.output),budget=r.BUDGET,rules=r.RULES,
        inputs=dict(model_mjb=pin(P/'hand2_holding_v1/compiled_nominal_v2/model.mjb'),model_binding=pin(P/'hand2_holding_v1/compiled_nominal_v2/binding.json'),candidate_report=pin(a.candidate_report)),
        source_pins=dict(core=pin(H/'native_contact_preload_v1.py'),runner=pin(source),builder=pin(__file__),frozen_physics_helper=pin(P/'functional_tripod_dynamics_v1/run_functional_tripod_dynamics_v3.py')),
        target_wrench_world6_rule='Default [-mass*gravity_world,0 torque about actual body COM]; API accepts explicit finite WORLD6 for future bounded feedback. Native_loaded_entry does not imply target wrench achieved.',
        fixed_pose_rule='Candidate qpos27/zero qvel; no pose changes, actuation constraints and physical materials unchanged. Same freshData/zero warmstart for everyFW.',
        reference_routes='ctrl=q then same-forward native contact normal total1N/3N PER PAD, evenly split over each actual generated pad contact. Actual verification is separate.',
        actual_interpretation='Only actual native Fn/wrench/ctrl are measurements. Material anchors never enter force proposals. Ideal normals/finite-action Jacobian/GN remain proposals. No all-qacc0, idealLP or wrench-error accuracy gate certifies holding.',automatic_retry=False)
    with a.protocol.open('x',encoding='utf8') as f:json.dump(c,f,indent=2,allow_nan=False)
    print(json.dumps(dict(protocol=pin(a.protocol),native_forward_authorized=False)))
if __name__=='__main__':main()
