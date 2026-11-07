"""Pin the execution binding before any CPU integration, then run the reviewed bench."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

import run_phone_cpu_bench_v1 as bench

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/phone_physics_stage1_v1.json'
ASSET = ROOT / 'asset_stage1/phone_asset.xml'
SOURCE = Path(bench.__file__)
EXPECTED_SOURCE = '3e3303f09d710c5d9de9e61324fcfc84776cb648b5d8cf3a8d32972953aa858c'
EXPECTED_PROTOCOL = '8981e89aec0993887aad65db1722f80d27470d6553a7bd25760a6ba7556d0505'

def main():
    if bench.pin(SOURCE)['sha256'] != EXPECTED_SOURCE or bench.pin(PROTOCOL)['sha256'] != EXPECTED_PROTOCOL:
        raise ValueError('Reviewed source/protocol changed')
    model, geometry = bench.make_model(ASSET, 0.2, 0.6, 0.0025)
    pins = [PROTOCOL, SOURCE, Path(__file__), ASSET,
            ROOT / 'asset_stage1/geometry_report.json',
            ROOT / 'asset_stage1/aabb_geometry_audit_v1/report.json',
            ROOT / 'asset_stage1/meshes/phone24_visual_m.obj',
            ROOT / 'hot3d/object_models/obj_000024.glb',
            ROOT / 'selected_models_manifest_v1.json',
            ROOT / 'protocols/CPU_SOURCE_REVIEW.json', Path(bench.mujoco.__file__)]
    binding = {'schema': 'phone_CPU_execution_binding:1',
               'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'pins': [bench.pin(path) for path in pins],
               'nominal_compiled_options': bench.compiled_options(model),
               'nominal_box_half_m': geometry['half_m'].tolist(),
               'COM_body_m': geometry['center_m'].tolist(),
               'nominal_uniform_box_inertia_kg_m2': geometry['inertia_kg_m2'].tolist(),
               'case_mass_and_inertia_rule': '9 declared priors; exact case compiled options and uniform-box inertia saved per case',
               'mujoco_version': bench.mujoco.__version__, 'python': sys.version,
               'CPU_only': True, 'integration_steps_executed_at_binding': 0,
               'geometry_qualified': False, 'real_parameters_measured': False}
    bench.write(ROOT / 'simulation_stage1/execution_binding_cpu_v1.json', binding)
    print(json.dumps({'status': 'PREINTEGRATION_BINDING_SAVED'}), flush=True)
    result = subprocess.run([sys.executable, '-X', 'utf8', str(SOURCE)], timeout=180, check=False)
    bench.write(ROOT / 'simulation_stage1/launch_cpu_v1_result.json',
                {'returncode': result.returncode,
                 'result': bench.pin(ROOT / 'simulation_stage1/cpu_v1/result.json') if result.returncode == 0 else None})
    if result.returncode:
        raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
