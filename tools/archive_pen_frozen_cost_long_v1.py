"""Verify and archive the continuous frozen policy evidence."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from archive_pen_source_ccd_evidence import pins
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.strict_json import write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(pin):
    return tuple(pin[key] for key in ('path', 'bytes', 'sha256'))


def main():
    runs = ROOT / 'runs'
    out = runs / 'pen_frozen_cost_long_checkpoint_v1'
    dest = ROOT / 'provenance/pen_frozen_cost_long_v1'
    if out.exists() or dest.exists():
        raise ValueError('Fresh frozen policy archive required')
    prior = file_pin(ROOT / 'provenance/pen_guarded_cost_native_v1/ledger.json')
    parent = read(verify_file_pin(prior))
    manifests = parent['inherited_verified_manifests'] + [parent['additional_verified_manifest']]
    known = {identity(pin) for manifest in manifests for pin in pins(read(verify_file_pin(manifest)))}
    roots = {name: file_pin(runs / path) for name, path in {
        'worker_v1': 'pen_frozen_cost_long_registration_v1/registration.json',
        'reader_v1': 'pen_frozen_cost_long_reader_registration_v1/registration.json',
        'failure_v1': 'pen_frozen_cost_long_v1/failure.json',
        'intent_v1': 'pen_frozen_cost_long_v1/intent.json',
        'source_v1': 'pen_frozen_cost_long_v1/source_receipt.json',
        'worker_v2': 'pen_frozen_cost_long_registration_v2/registration.json',
        'reader_v2': 'pen_frozen_cost_long_reader_registration_v2/registration.json',
        'admission': 'pen_frozen_cost_actor_admission_v1/report.json',
        'producer': 'pen_frozen_cost_long_v2/report.json',
        'reader': 'pen_frozen_cost_long_review_v2/report.json',
        'summary': 'pen_frozen_cost_long_summary_v1/report.json',
        'video': 'pen_frozen_cost_long_video_v1/report.json',
    }.items()}
    reports = {key: read(verify_file_pin(pin)) for key, pin in roots.items()}
    checks = {key: value['passed'] == value['total'] and all(value['checks'].values())
              for key, value in reports.items() if 'checks' in value}
    failed, producer, reader, summary = (reports[key] for key in ('failure_v1', 'producer', 'reader', 'summary'))
    checks.update(
        retained_failure=failed['type'] == 'TypeError' and len(failed['missing']) == 18 and not failed['rows']
        and failed['actual_native_integrations'] == failed['attempted_native_integrations'] == 0,
        exact_integrations=producer['actual_native_integrations'] == producer['attempted_native_integrations'] == 11616
        and producer['completed_call_accounting_exact'],
        independent_coverage=len(producer['rows']) == len(reader['rows']) == 18 and reader['controls_recomputed'] == 1452,
        complete_task_result=len(summary['rows']) == 18 and not any(row['full_task_success'] for row in summary['rows']),
        no_updates=all(value.get('optimizer_steps', 0) == value.get('normalizer_fits', 0) == 0 for value in reports.values()),
    )
    if not all(checks.values()):
        raise ValueError(f'Incomplete long evaluation: {checks}')

    documents = [ROOT / 'docs' / name for name in (
        'pen-frozen-cost-long-results-2026-11-20.md', 'research-plan-v1.57.md',
        'pen-frozen-cost-long-protocol-v1.md', 'pen-frozen-cost-long-protocol-v2.md',
    )]
    execution_tools = sorted((ROOT / 'tools').glob('*pen_frozen_cost*.py'))
    tests = sorted((ROOT / 'tests').glob('test_pen_frozen_cost*.py')) + [ROOT / 'tests/test_pen_continuous_math_view_v1.py']
    configs = sorted((ROOT / 'configs').glob('pen_frozen_cost_long*.toml'))
    logs = sorted(runs.glob('pen_frozen_cost*.log'))
    partial_files = sorted((runs / 'pen_frozen_cost_long_v1/R0_INSTALLED_ORIGINAL_OR_INITIAL').glob('*'))
    frames = sorted((runs / 'pen_frozen_cost_long_video_v1').glob('*/*.png'))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    out.mkdir()
    receipt = freeze_sources(sorted((ROOT / 'src').rglob('*.py')) + documents + execution_tools + tests + configs,
                             out / 'source_snapshot', ROOT, revision)
    write_json(out / 'source_receipt.json', receipt)

    todo = list(roots.values()) + [file_pin(path) for path in logs + partial_files + frames + [out / 'source_receipt.json']]
    verified, inherited = {}, set()
    while todo:
        pin = todo.pop()
        key = identity(pin)
        path = Path(pin['path']).resolve()
        if path == runs / 'pen_native_screened_rollout_v1/work.json':
            raise ValueError('Historical progress file is mutable')
        if key in known:
            inherited.add(key)
        elif key not in verified:
            verify_file_pin(pin)
            verified[key] = pin
            if path.suffix == '.json' and path.is_relative_to(runs):
                todo.extend(pins(read(path)))

    verify_source_snapshot(receipt, live=True)
    write_json(out / 'additional_verified_pins.json', dict(records=list(verified.values()), inherited_reached=len(inherited)))
    write_json(out / 'review.json', dict(checks=checks, passed=sum(checks.values()), total=len(checks)))
    dest.mkdir()
    ledger = dest / 'ledger.json'
    write_json(ledger, dict(schema='pen_frozen_cost_long_checkpoint:1', prior_evidence=prior,
        inherited_verified_manifests=manifests, roots=roots, documents=[file_pin(path) for path in documents],
        retained_logs=[file_pin(path) for path in logs], source_receipt=file_pin(out / 'source_receipt.json'),
        review=file_pin(out / 'review.json'), additional_verified_manifest=file_pin(out / 'additional_verified_pins.json'),
        additional_verified_pins=len(verified), inherited_pins_reached=len(inherited),
        native_integrations_in_producers=11616, archival_native_integrations=0, optimizer_steps=0, normalizer_fits=0,
        scope='18 frozen W1 cases on one controlled reference, two videos from fixed repeat zero, zero complete task successes.'))
    print(json.dumps(dict(ledger=file_pin(ledger), checks=[sum(checks.values()), len(checks)], new_pins=len(verified))), flush=True)


if __name__ == '__main__':
    main()
