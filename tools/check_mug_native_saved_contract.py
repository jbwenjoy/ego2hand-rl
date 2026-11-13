"""Additional independent saved chronology/EMA checks, no rerun or fitting."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.native_saved_contract import saved_chain_checks


def main(args):
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh unified output required')
    report = json.loads(args.report.read_text(encoding='utf8'))
    binding = json.loads(verify_file_pin(report['inputs']['binding']).read_text(encoding='utf8'))
    rr = json.loads(verify_file_pin(report['inputs']['reference']).read_text(encoding='utf8'))
    ref = np.load(verify_file_pin(rr['output']))
    base = ref['feedforward_PD_target20'][0]
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(binding['model'])))
    negative = np.minimum(.05, base - model.actuator_ctrlrange[:, 0])
    positive = np.minimum(.05, model.actuator_ctrlrange[:, 1] - base)
    cases = []
    for case in report['cases']:
        a = dict(np.load(verify_file_pin(case['execution']['transitions'])))
        z = dict(np.load(verify_file_pin(case['execution']['trace'])))
        checks = saved_chain_checks(a, z, case['dt'], base, negative, positive)
        checks['initial_pose_matches_reference'] = np.array_equal(z['pre_qpos'][0], ref['initial_qpos27'])
        checks['initial_velocity_zero'] = not np.any(z['pre_qvel'][0])
        cases.append(dict(task=case['task'], controller=case['controller'], dt=case['dt'], checks=checks))
    result = dict(schema='native_saved_chronology_contract:1', run=file_pin(args.report),
        sources=[file_pin(__file__), file_pin(ROOT / 'src/supermimic_research/mechanics/native_saved_contract.py')],
        cases=cases, passed=sum(sum(c['checks'].values()) for c in cases),
        total=sum(len(c['checks']) for c in cases), neural_updates=0, native_integration_steps=0)
    args.output.mkdir()
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))
    if result['passed'] != result['total']:
        raise ValueError('Saved chronology/EMA failed; preserve failure')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
