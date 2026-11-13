"""Validate frozen H6 protocol pins before shared independent raw readback."""
from pathlib import Path
import argparse
import json
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.native_saved_review import review_saved_cases


def main(args):
    report = json.loads((args.run / 'report.json').read_text(encoding='utf8'))
    extension = tomllib.loads(verify_file_pin(report['config']).read_text(encoding='utf8'))
    base = tomllib.loads(verify_file_pin(report['base_config']).read_text(encoding='utf8'))
    effective = {**base, **{k: extension[k] for k in
                          ['horizon', 'tasks', 'controllers', 'max_total_oracle_steps']}}
    if (extension['horizon'], extension['tasks'], extension['controllers']) != (6, ['REFERENCE'], ['NATIVE_MPC']):
        raise ValueError('Frozen H6 cohort required')
    if effective != report['effective_config'] or report['neural_updates'] != 0:
        raise ValueError('Protocol drift')
    baseline = json.loads(verify_file_pin(report['baseline']).read_text(encoding='utf8'))
    baseline_review = json.loads(verify_file_pin(report['baseline_review']).read_text(encoding='utf8'))
    if baseline_review['run'] != report['baseline'] or baseline['config'] != report['base_config']:
        raise ValueError('Baseline pin mismatch')
    if (baseline_review['passed'], baseline_review['total']) != (8, 8):
        raise ValueError('Baseline readback incomplete')
    for key in ['binding', 'reference', 'oracle_parity']:
        if baseline['inputs'][key] != report['inputs'][key]:
            raise ValueError('Baseline physical/reference input mismatch')
        verify_file_pin(report['inputs'][key])
    if any(c['horizon'] != 6 for c in report['cases']):
        raise ValueError('Horizon drift')
    if (sum(c['oracle_steps'] for c in report['cases']) != report['total_oracle_steps']
        or report['total_oracle_steps'] > effective['max_total_oracle_steps']
        or sum(c['execution']['native_steps'] for c in report['cases']) != report['total_active_steps']):
        raise ValueError('Cost accounting mismatch')
    result = review_saved_cases(args.run, args.output, root=ROOT,
        expected_cases={('REFERENCE', 'NATIVE_MPC', dt) for dt in base['dt']}, horizon=6)
    result['protocol_reviewer'] = file_pin(__file__)
    result['baseline_review'] = report['baseline_review']
    (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
