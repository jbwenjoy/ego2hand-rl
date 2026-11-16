"""Archive completed TRAIN responses and dataset readiness,without fitting."""
import json
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.backend_probe_review import require_response_probe
from supermimic_research.learning.response_coverage_review import review_coverage
from supermimic_research.mechanics.probe_row_review import response_groups
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.runtime.explicit_collision_backend import load_contract
from archive_pen_source_ccd_evidence import pins


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    out = ROOT/'runs/pen_backend_train_response_review_v1'
    archive = ROOT/'provenance/pen_backend_train_response_v1'
    if out.exists() or archive.exists(): raise ValueError('Fresh TRAIN review/archive required')
    rp = ROOT/'runs/pen_backend_response_train_v1/report.json'
    pp = ROOT/'runs/pen_backend_response_train_preparation_v1/report.json'
    cp = ROOT/'configs/pen_backend_response_train_v1.toml'; cfg = tomllib.loads(cp.read_text())
    reg_path = ROOT/'runs/pen_backend_response_registration_v1/registration.json'
    raw, prep, reg = read(rp), read(pp), read(reg_path)
    contract = load_contract(ROOT/'configs/collision_backend_face_normal_v1.json')
    require_response_probe(raw, cfg, contract, file_pin(reg_path), file_pin(cp))
    receipts = [v['source_receipt'] for v in [raw, prep, reg]]
    for pin in receipts: verify_source_snapshot(read(verify_file_pin(pin)), live=True)
    data = dict(np.load(verify_file_pin(prep['rows'])))
    checks = dict(exact_preparation=prep['schema']=='backend_pen_response_preparation:1'
        and prep['probe']==file_pin(rp) and prep['config']==file_pin(cp)
        and prep['collision_backend']==contract and prep['role']==cfg['role']
        and prep['registration']==file_pin(reg_path),
        preparation_completed=prep['passed']==prep['total']==69 and all(prep['checks'].values()),
        unchanged_masks=np.array_equal(data['valid'],[m['valid'] for m in prep['metadata']])
        and np.flatnonzero(~data['valid']).tolist()==prep['rejected_rows']
        and int(data['valid'].sum())==prep['qualified_rows'],
        all_available_rows=len(data['x'])==prep['total_rows']==raw['available_cases']
        and len(prep['metadata'])==len(data['x']),
        all_declared_groups=response_groups(prep['metadata'],raw['sources'],cfg)==prep['response_groups'],
        no_fit=not prep['scales_fitted'] and not prep['optimizer_steps'] and not prep['actual_native_integrations']
        and not prep['CHECK_response_rows_used'] and not prep['validation_rows_used_for_TRAIN'])
    for source in raw['sources']:
        state = torch.load(verify_file_pin(source['source']), map_location='cpu', weights_only=False)
        require_backend_probe(state, contract)
        checks[source['name']+'_owned_source'] = state['role']==raw['role'] and state['actor_checkpoint']==source['actor_checkpoint']
        verify_source_snapshot(state['native']['sources'], live=True)
        for pin in state['native']['engine_sources']: verify_file_pin(pin)
    rows_exact = []; bad = Counter(); rejected_sources = Counter()
    for i, row in enumerate(prep['metadata']):
        case = raw['cases'][row['case_index']]
        rows_exact.append(row['row_index']==i and row['source_name']==case['source_name']
            and row['profile']==case['profile'] and row['name']==case['name']
            and row['online_quality']==case['qualified']
            and row['valid']==bool(case['qualified'] and row['packet_quality']
                                  and row['geometry_legal'] and all(row['checks'].values())))
        if not row['valid']:
            bad.update(k for k,v in row['checks'].items() if not v)
            bad.update(k for k in ['online_quality','packet_quality','geometry_legal'] if not row[k])
            rejected_sources.update([row['source_name']])
    checks['every_original_case_row_and_mask'] = all(rows_exact)
    checks['row_gate_accounting'] = (sum(sum(m['checks'].values()) for m in prep['metadata'])==prep['row_gate_passed']
        and sum(len(m['checks']) for m in prep['metadata'])==prep['row_gate_total'])
    old_path = ROOT/'runs/pen_contact_response_common_data_v1/report.json'; old = read(old_path)
    scale = np.load(verify_file_pin(old['scales']))['y_scale']
    if old['scale_input_role']!='VALID_TRAIN_ONLY': raise ValueError('Pinned descriptive old units required')
    directions = np.load(verify_file_pin(raw['directions']))['directions']
    coverage = review_coverage(data,prep['metadata'],prep['response_groups'],directions,
                              [s['name'] for s in raw['sources']],scale)
    checks['qualified_pair_count'] = len(coverage['eligible_pair_groups']['rank20'])==prep['qualified_response_groups']
    tests = ROOT/'runs/pen_backend_dataset_tests495_v1.log'
    negative = ROOT/'runs/pen_backend_dataset_precondition_v2.log'
    checks['full495_tests'] = 'Ran 495 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    checks['production_missing_review_guard'] = ('FileNotFoundError' in negative.read_text()
        and 'pen_backend_data_review_v1' in negative.read_text()
        and not (ROOT/'runs/pen_backend_common_data_registration_v1').exists()
        and not (ROOT/'runs/pen_backend_common_data_v1').exists())
    if not all(checks.values()): raise ValueError('TRAIN archive review failed: '+repr([k for k,v in checks.items() if not v]))
    docs = [ROOT/'docs'/name for name in ['pen-backend-train-response-results-2026-11-16.md',
        'pen-backend-dataset-readiness-results-2026-11-16.md','pen-backend-common-data-protocol-v1.md','research-plan-v1.33.md']]
    source_paths = sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__),
        ROOT/'tools/prepare_backend_pen_dataset.py', ROOT/'configs/pen_backend_common_data_v1.toml',
        ROOT/'tests/test_backend_response_dataset.py']
    out.mkdir(); receipt = freeze_sources(source_paths,out/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(out/'source_receipt.json',receipt)
    summary = dict(raw=file_pin(rp),preparation=file_pin(pp),checks=[sum(checks.values()),len(checks)],
        declared_sources=len(raw['sources']),qualified_sources=raw['qualified_sources'],declared_cases=len(raw['cases']),
        executed_rows=prep['total_rows'],qualified_rows=prep['qualified_rows'],rejected_rows=prep['rejected_rows'],
        failed_gate_counts=dict(bad),rejected_source_counts=dict(rejected_sources),declared_groups=len(prep['response_groups']),
        qualified_pairs=prep['qualified_response_groups'],pair_counts={k:len(v) for k,v in coverage['eligible_pair_groups'].items()},
        producer_native=raw['producer_native_integrations'],branch_native=8*raw['available_cases'],
        actual_native_integrations=raw['actual_native_integrations'],attempted_native_integrations=raw['attempted_native_integrations'],
        optimizer_steps=0,collection_wall_s=raw['wall_s'],added_review_native_integrations=0,added_review_optimizer_steps=0,
        production_normalizer_fits=0,scope='Completed new TRAIN only;all exclusions/masks retained. Coverage in old descriptive units is not a new fitted scale or physical-gradient admission. Synthetic unit normalizations are test operations,not production supervision. Validation remains live at last handle poll;no aggregate cohort cost or completed combined review claim.')
    write(out/'report.json',dict(schema='backend_TRAIN_response_archive_review:1',summary=summary,checks=checks,
        source_receipt=file_pin(out/'source_receipt.json'),coverage=coverage,old_descriptive_scale=old['scales'],
        dataset_readiness=dict(full_tests=file_pin(tests),negative_CLI=file_pin(negative),
        success_production_path_executed=False),actual_native_integrations=0,optimizer_steps=0,production_normalizer_fits=0))
    prior = ROOT/'provenance/pen_backend_control_results_v1/ledger.json'
    todo = [file_pin(p) for p in [rp,pp,reg_path,out/'report.json',out/'source_receipt.json',tests,negative,prior,old_path]+docs]+receipts
    roots = [rp.parent,pp.parent,out,reg_path.parent]; verified = {}
    while todo:
        pin = todo.pop(); key = json.dumps(pin,sort_keys=True)
        if key in verified: continue
        path = verify_file_pin(pin); verified[key] = pin
        if path.suffix=='.json' and any(path.is_relative_to(p) for p in roots): todo.extend(pins(read(path)))
    write(out/'verified_pins.json',dict(records=list(verified.values())))
    archive.mkdir(); write(archive/'summary.json',summary)
    write(archive/'ledger.json',dict(schema='backend_TRAIN_response_results_ledger:1',prior_evidence=file_pin(prior),
        summary=file_pin(archive/'summary.json'),roots=dict(raw=file_pin(rp),preparation=file_pin(pp),review=file_pin(out/'report.json')),
        source_receipts=receipts+[file_pin(out/'source_receipt.json')],documents=[file_pin(p) for p in docs],
        verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),
        added_review_native_integrations=0,added_review_optimizer_steps=0,production_normalizer_fits=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),
        TRAIN=[prep['qualified_rows'],prep['total_rows']],pairs=summary['pair_counts'])),flush=True)


if __name__=='__main__': main()
