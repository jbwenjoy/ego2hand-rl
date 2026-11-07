"""Execute a versioned fixed grid with unchanged convex-model/capture functions."""
import argparse
import datetime
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'simulation_convex_v2'))
import phone_convex_common_v2 as common
from run_cpu_v2 import run_case

TESTS = ('drop', 'table_settling', 'force_push_release')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    if p['schema'] != 'phone_convex_fixedgrid_protocol:4':
        raise ValueError('Unsupported fixed-grid contract')
    if common.pin(common.__file__)['sha256'] != '4460ccba501e46a033e304460796103afb7b2ef4ef822e6d8653f1a496228ae9' or common.pin(ROOT/'simulation_convex_v2/run_cpu_v2.py')['sha256'] != '1b6889af2d5d8f1d4366532eab745ddd55aa9d0246aa1a410a5566d89e172965':
        raise ValueError('Frozen physical model or capture changed')
    grid = p['fixed_grid']
    masses, mus = grid['cartesian_grid']['mass_kg'], grid['cartesian_grid']['sliding_friction']
    dts, tau = grid['timesteps_s'], grid['timeconst_s']
    if (masses, mus, dts, tau, grid['test_names']) != ([.1, .2, .3], [.3, .6, .9], [.0003125, .00015625], .0025, list(TESTS)):
        raise ValueError('Unexpected registered fixed grid')
    pairs = [(.2, .6)] + [pair for pair in itertools.product(masses, mus) if pair != (.2, .6)]
    params = [(mass, mu, dt, test) for mass, mu in pairs for dt in dts for test in TESTS]
    expected_steps = sum(round(3/dt) for _, _, dt, _ in params)
    if len(params) != grid['case_count'] or expected_steps != grid['integration_steps']:
        raise ValueError('Registered budget mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    binding = {'schema': 'phone_convex_fixedgrid_CPU_binding:4',
               'protocol': common.pin(args.protocol), 'runner': common.pin(__file__),
               'common': common.pin(common.__file__),
               'capture_source': common.pin(ROOT/'simulation_convex_v2/run_cpu_v2.py'),
               'legacy_helpers': common.pin(ROOT/'simulation_stage1/run_phone_cpu_bench_v1.py'),
               'asset': common.pin(common.ASSET), 'geometry_report': common.pin(common.ASSET.parent/'report.json'),
               'collision_OBJ': common.pin(common.ASSET.parent/'meshes/phone24_collision_convex_m.obj'),
               'visual_OBJ': common.pin(common.ASSET.parent/'meshes/phone24_visual_m.obj'),
               'parameters': params, 'timeconst_s': tau, 'integration_steps_before_binding': 0,
               'mujoco_version': common.mujoco.__version__}
    common.write(args.output/'execution_binding.json', binding)
    started, rows = time.perf_counter(), []
    for mass, mu, dt, test in params:
        model, g = common.make_model(common.ASSET, mass, mu, dt, tau)
        case = f'{test}_mass{mass:g}_mu{mu:g}_tau{tau:g}_dt{dt:g}'
        xml = args.output/(case+'.xml')
        with xml.open('x') as stream:
            stream.write(g['xml'])
        options = common.compiled_options(model)
        support = float(((g['local_collision_vertices_m']-g['center_m'])@common.ROT.T)[:, 2].min())
        com = np.array([0., 0., (.1 if test == 'drop' else .0005)-support])
        common.write(args.output/(case+'_before_steps.json'), {'compiled_options': options,
                     'model_xml': common.pin(xml), 'inertia_kg_m2': g['inertia_kg_m2'].tolist(),
                     'COM_body_m': g['center_m'].tolist(),
                     'local_collision_vertices_m': g['local_collision_vertices_m'].tolist(),
                     'initial_minimum_collision_height_relative_to_COM_m': support,
                     'initial_requested_qpos': np.concatenate((com-common.ROT@g['center_m'], common.QUAT)).tolist()})
        a, metrics = run_case(model, g, test)
        path = args.output/(case+'.npz')
        np.savez_compressed(path, **a)
        rows.append({'case': case, 'backend': 'CPU_MuJoCo', 'test': test, 'mass_kg': mass,
                     'sliding_friction': mu, 'timeconst_s': tau, 'dt_s': dt,
                     'metrics': metrics, 'trajectory': common.pin(path), 'compiled_options': options,
                     'model_xml': common.pin(xml)})
        print(json.dumps({'status': 'FIXEDGRID_CPU_RECORDED', 'case': case,
                          'max_penetration_mm': 1000*metrics['max_penetration_m']}), flush=True)
    common.write(args.output/'result.json', {'schema': 'phone_convex_fixedgrid_CPU:4',
                 'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'mode': 'fixedgrid', 'timeconst_s': tau, 'timesteps_s': dts,
                 'binding': common.pin(args.output/'execution_binding.json'), 'protocol': common.pin(args.protocol),
                 'cases': rows, 'case_count': len(rows),
                 'integration_steps': sum(row['metrics']['completed_steps'] for row in rows),
                 'elapsed_seconds': time.perf_counter()-started,
                 'capture_phase': 'unchanged pinned v2 capture; native contacts/qacc at explicit POST forward',
                 'thresholds_assessed': False, 'real_parameters_calibrated': False,
                 'Hand2_or_training_executed': False})
    print(json.dumps({'status': 'FIXEDGRID_CPU_COMPLETE_NOT_ASSESSED', 'cases': len(rows), 'output': str(args.output)}), flush=True)

if __name__ == '__main__':
    main()
