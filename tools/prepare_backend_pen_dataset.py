"""Register and freeze reviewed new-backend TRAIN-only point/pair products."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.backend_probe_review import require_response_probe
from supermimic_research.learning.backend_response_dataset import backend_response_data, ROLES
from supermimic_research.learning.pen_response import INPUT_SCHEMA, OUTPUT_SCHEMA
from supermimic_research.runtime.explicit_collision_backend import load_contract


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main(args):
    cp = args.config.resolve(); cfg = tomllib.loads(cp.read_text())
    if cfg['schema'] != 1 or cfg['normalizer_role'] != 'VALID_NEW_BACKEND_TRAIN_ONLY':
        raise ValueError('Explicit new-backend normalization contract required')
    paths = {k: (cp.parent/cfg[k]).resolve() for k in [
        'TRAIN', 'VALIDATE', 'data_review', 'backend_manifest', 'response_registration',
        'protocol', 'registration', 'output']}
    contract = load_contract(paths['backend_manifest']); review = read(paths['data_review'])
    response_reg = read(paths['response_registration'])
    inputs = {k: file_pin(paths[k]) for k in paths if k not in ['registration', 'output']}
    if (review.get('schema') != 'backend_response_data_review:1'
            or review['collision_backend'] != contract or response_reg['backend_id'] != contract['backend_id']
            or review['preparations'] != {k: inputs[k] for k in ROLES}
            or review['inputs']['response_registration'] != inputs['response_registration']
            or review['inputs']['backend_manifest'] != inputs['backend_manifest']
            or review['passed'] != review['total'] or not all(review['checks'].values())
            or review['actual_native_integrations'] or review['optimizer_steps']
            or review['normalizer_fits'] or review['validation_rows_used_for_TRAIN']):
        raise ValueError('Completed same-backend independent physical review required before registration')
    verify_source_snapshot(read(verify_file_pin(review['source_receipt'])), live=True)
    verify_source_snapshot(read(verify_file_pin(response_reg['source_receipt'])), live=True)
    reports = {}; data = {}; directions = {}; raw = {}; physics = None
    for role in ROLES:
        report = reports[role] = read(paths[role]); cohort = raw[role] = read(verify_file_pin(report['probe']))
        spec_path = verify_file_pin(cohort['config']); spec = tomllib.loads(spec_path.read_text())
        require_response_probe(cohort, spec, contract, inputs['response_registration'], file_pin(spec_path))
        if (response_reg['specs'][ROLES[role]] != file_pin(spec_path)
                or report['registration'] != inputs['response_registration']
                or report['directions'] != cohort['directions']
                or report['config'] != cohort['config'] or spec['role'] != ROLES[role]):
            raise ValueError('Exact registered raw/prepared cohort and role required')
        for entry in [report, cohort]:
            verify_source_snapshot(read(verify_file_pin(entry['source_receipt'])), live=True)
        data[role] = dict(np.load(verify_file_pin(report['rows'])))
        directions[role] = np.load(verify_file_pin(report['directions']))['directions']
        for source in cohort['sources']:
            if not source['available']: continue
            if physics is None: physics = source['physics']
            if source['physics'] != physics:
                raise ValueError('One unchanged plant physical context required')
    for field in ['noise_seed', 'phase']:
        if {s[field] for s in raw['TRAIN']['sources']} & {s[field] for s in raw['VALIDATE']['sources']}:
            raise ValueError('TRAIN and validation noise/phase identities overlap')
    if physics is None: raise ValueError('No actual physical source exists')
    source_paths = sorted((ROOT/'src').rglob('*.py'))+[Path(__file__), cp, paths['protocol']]
    def freeze(out):
        receipt = freeze_sources(source_paths, out/'source_snapshot', ROOT,
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        write(out/'source_receipt.json', receipt); return receipt
    if args.register:
        out = paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'): raise ValueError('Fresh dataset registration required')
        out.mkdir(); freeze(out)
        write(paths['registration'], dict(schema='backend_response_dataset_registration:1',
            config=file_pin(cp), inputs=inputs, backend_id=contract['backend_id'],
            source_receipt=file_pin(out/'source_receipt.json'), normalizer_fits=0,
            actual_native_integrations=0, optimizer_steps=0,
            scope='Reviewed actual masks frozen before new TRAIN-only normalization;not a model fitting registration.'))
        print(json.dumps(dict(registration=file_pin(paths['registration']))), flush=True); return
    reg = read(paths['registration'])
    if (reg.get('schema') != 'backend_response_dataset_registration:1' or reg['config'] != file_pin(cp)
            or reg['inputs'] != inputs or reg['backend_id'] != contract['backend_id']):
        raise ValueError('Registered actual data inputs changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])), live=True)
    out = paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'): raise ValueError('Fresh new-backend dataset output required')
    out.mkdir(); receipt = freeze(out)
    try:
        built = backend_response_data(reports, data, directions, contract)
        altered = {r: {k: v.copy() for k, v in arrays.items()} for r, arrays in data.items()}
        for key in ['x', 'y', 'heads']:
            altered['VALIDATE'][key] += 1e6
            altered['TRAIN'][key][~data['TRAIN']['valid']] += 1e7
        negative = backend_response_data(reports, altered, directions, contract)
        checks = {key+'_no_validation_or_invalid_TRAIN': all(np.array_equal(v, negative[key][k]) for k, v in built[key].items())
                  for key in ['scales', 'head_scales', 'pairs', 'pair_subsets']}
        checks['eligible_groups_exact'] = all(built['pairs']['TRAIN_group_index'][indices].tolist() == review['TRAIN_eligible_pair_group_indices'][key]
                                             for key, indices in built['pair_subsets'].items())
        for role in ROLES: np.savez_compressed(out/(role+'.npz'), **data[role])
        for key in ['scales', 'head_scales', 'pairs', 'pair_subsets']: np.savez_compressed(out/(key+'.npz'), **built[key])
        verify_source_snapshot(receipt, live=True)
        report = dict(schema='backend_pen_response_common_data:1', config=file_pin(cp),
            registration=file_pin(paths['registration']), inputs=inputs, collision_backend=contract,
            source_receipt=file_pin(out/'source_receipt.json'), preparations={k: inputs[k] for k in ROLES},
            outputs={r: file_pin(out/(r+'.npz')) for r in ROLES},
            products={k: file_pin(out/(k+'.npz')) for k in ['scales', 'head_scales', 'pairs', 'pair_subsets']},
            directions={k: reports[k]['directions'] for k in ROLES}, physics=physics,
            input_schema=INPUT_SCHEMA, output_schema=OUTPUT_SCHEMA, pair_audit=built['pair_audit'],
            rows={k: dict(total=reports[k]['total_rows'], valid=reports[k]['qualified_rows'], rejected=reports[k]['rejected_rows']) for k in ROLES},
            pair_counts={k: len(v) for k, v in built['pair_subsets'].items()},
            scale_input_role=cfg['normalizer_role'], validation_role=ROLES['VALIDATE'],
            checks=checks, passed=sum(checks.values()), total=len(checks),
            normalizer_fits=2, validation_rows_used_for_TRAIN=0, CHECK_response_rows_used=0,
            actual_native_integrations=0, optimizer_steps=0,
            scope='Dynamic actual new-backend counts,all masks/empty groups retained. TRAIN-only point/head scales and finite pairs;second normalization is deliberate leakage check. No old fitted scales,NN/Actor updates or physical-gradient/task/ego admission.')
        write(out/'report.json', report)
        if not all(checks.values()): raise ValueError('Preserve dataset isolation/eligibility failure')
        print(json.dumps(dict(report=file_pin(out/'report.json'), checks=[report['passed'], report['total']],
                             rows={k: [v['valid'], v['total']] for k, v in report['rows'].items()}, pair_counts=report['pair_counts'])), flush=True)
    except Exception as error:
        write(out/'failure.json', dict(type=type(error).__name__, message=str(error), actual_native_integrations=0, optimizer_steps=0)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--register', action='store_true'); main(parser.parse_args())
