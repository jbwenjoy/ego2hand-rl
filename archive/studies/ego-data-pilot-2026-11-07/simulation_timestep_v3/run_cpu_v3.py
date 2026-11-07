"""Finer-timestep convergence / confirmation using unchanged v2 physics capture."""
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
PAIRS = [[.000625, .0003125], [.0003125, .00015625]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('screen', 'confirm'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--protocol', type=Path, default=ROOT/'protocols/phone_convex_timestep_v3.json')
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    if p['schema'] != 'phone_convex_timestep_protocol:3':
        raise ValueError('Wrong protocol')
    if common.pin(common.__file__)['sha256'] != '4460ccba501e46a033e304460796103afb7b2ef4ef822e6d8653f1a496228ae9' or common.pin(ROOT/'simulation_convex_v2/run_cpu_v2.py')['sha256'] != '1b6889af2d5d8f1d4366532eab745ddd55aa9d0246aa1a410a5566d89e172965':
        raise ValueError('Frozen v2 model or integration capture changed')
    s = p['screening']
    if (s['timeconst_s'], s['mass_kg'], s['sliding_friction'], s['timesteps_s'], s['test_names']) != (.0025, .2, .6, [.000625, .0003125, .00015625], list(TESTS)):
        raise ValueError('Frozen convergence ladder mismatch')
    selection_pin = None
    if args.mode == 'screen':
        params = [(.2, .6, dt, test) for dt in s['timesteps_s'] for test in TESTS]
    else:
        if args.selection is None:
            raise ValueError('A complete saved selection is required')
        selection = json.loads(args.selection.read_text())
        pair = selection['selected_timesteps_s']
        if pair not in PAIRS or selection['selected_timeconst_s'] != .0025 or not selection['screening_complete'] or not selection['selection_complete'] or not selection['audit_integrity_qualified']:
            raise ValueError('No qualified step pair selected')
        if selection['protocol_pin']['sha256'] != common.pin(args.protocol)['sha256']:
            raise ValueError('Selection belongs to another protocol')
        selection_pin = common.pin(args.selection)
        pairs = [(.2, .6)] + [a for a in itertools.product((.1, .2, .3), (.3, .6, .9)) if a != (.2, .6)]
        params = [(mass, mu, dt, test) for mass, mu in pairs for dt in pair for test in TESTS]
    args.output.mkdir(parents=True, exist_ok=False)
    binding = {'schema': 'phone_convex_timestep_CPU_binding:3', 'mode': args.mode,
               'protocol': common.pin(args.protocol), 'selection': selection_pin,
               'runner': common.pin(__file__), 'common': common.pin(common.__file__),
               'capture_source': common.pin(ROOT/'simulation_convex_v2/run_cpu_v2.py'),
               'legacy_helpers': common.pin(ROOT/'simulation_stage1/run_phone_cpu_bench_v1.py'),
               'asset': common.pin(common.ASSET), 'geometry_report': common.pin(common.ASSET.parent/'report.json'),
               'collision_OBJ': common.pin(common.ASSET.parent/'meshes/phone24_collision_convex_m.obj'),
               'visual_OBJ': common.pin(common.ASSET.parent/'meshes/phone24_visual_m.obj'),
               'parameters': params, 'timeconst_s': .0025, 'integration_steps_before_binding': 0,
               'mujoco_version': common.mujoco.__version__}
    common.write(args.output/'execution_binding.json', binding)
    started, rows = time.perf_counter(), []
    for mass, mu, dt, test in params:
        model, geometry = common.make_model(common.ASSET, mass, mu, dt, .0025)
        case = f'{test}_mass{mass:g}_mu{mu:g}_tau0.0025_dt{dt:g}'
        xml = args.output/(case+'.xml')
        with xml.open('x') as stream:
            stream.write(geometry['xml'])
        options = common.compiled_options(model)
        support = float(((geometry['local_collision_vertices_m']-geometry['center_m'])@common.ROT.T)[:, 2].min())
        com = np.array([0., 0., (.1 if test == 'drop' else .0005)-support])
        common.write(args.output/(case+'_before_steps.json'), {'compiled_options': options,
                     'model_xml': common.pin(xml), 'inertia_kg_m2': geometry['inertia_kg_m2'].tolist(),
                     'COM_body_m': geometry['center_m'].tolist(),
                     'local_collision_vertices_m': geometry['local_collision_vertices_m'].tolist(),
                     'initial_minimum_collision_height_relative_to_COM_m': support,
                     'initial_requested_qpos': np.concatenate((com-common.ROT@geometry['center_m'], common.QUAT)).tolist()})
        a, metrics = run_case(model, geometry, test)
        path = args.output/(case+'.npz')
        np.savez_compressed(path, **a)
        rows.append({'case': case, 'backend': 'CPU_MuJoCo', 'test': test, 'mass_kg': mass,
                     'sliding_friction': mu, 'timeconst_s': .0025, 'dt_s': dt,
                     'metrics': metrics, 'trajectory': common.pin(path),
                     'compiled_options': options, 'model_xml': common.pin(xml)})
        print(json.dumps({'status': 'V3_CPU_RECORDED', 'case': case,
                          'max_penetration_mm': 1000*metrics['max_penetration_m']}), flush=True)
    common.write(args.output/'result.json', {'schema': ('phone_convex_timestep_CPU_screening:3' if args.mode == 'screen' else 'phone_convex_timestep_CPU_confirmation:3'),
                 'mode': args.mode, 'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'binding': common.pin(args.output/'execution_binding.json'), 'protocol': common.pin(args.protocol),
                 'cases': rows, 'case_count': len(rows),
                 'integration_steps': sum(row['metrics']['completed_steps'] for row in rows),
                 'elapsed_seconds': time.perf_counter()-started,
                 'capture_phase': 'unchanged pinned v2 capture; native contacts and qacc at explicit POST forward',
                 'thresholds_assessed': False, 'real_parameters_calibrated': False,
                 'Hand2_or_training_executed': False})
    print(json.dumps({'status': 'V3_CPU_COMPLETE_NOT_ASSESSED', 'cases': len(rows), 'output': str(args.output)}), flush=True)

if __name__ == '__main__':
    main()
