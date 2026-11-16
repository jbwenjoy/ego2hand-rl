"""Seal closed full54 and actual A/B results; inherit explicit old manifests."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from archive_pen_source_ccd_evidence import pins
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    out = ROOT / 'runs/pen_full54_solve_exit_checkpoint_v1'
    dest = ROOT / 'provenance/pen_full54_solve_exit_v1'
    if out.exists() or dest.exists():
        raise ValueError('Fresh checkpoint required')
    parent_pin = file_pin(ROOT / 'provenance/pen_context_layout_v2/ledger.json')
    parent = json.loads(verify_file_pin(parent_pin).read_text())
    inherited = parent['inherited_verified_manifests'] + [parent['additional_verified_manifest']]
    known = set()
    identity = lambda p: json.dumps(p, sort_keys=True)
    for pin in inherited:
        known.update(identity(p) for p in pins(json.loads(verify_file_pin(pin).read_text())))
    names = dict(
        full54='pen_native_screened_rollout_v1/report.json',
        review54='pen_native_screened_review_v3/report.json',
        fixed_index54='pen_native_screened_review_v3/indices/completed_54.json',
        summary54='pen_native_screened_summary_54_v1/report.json',
        taxonomy54='pen_screened_failure_taxonomy_54_v1/report.json',
        videos='pen_native_screened_video_v1/report.json',
        workerA='pen_solve_exit_replay_v2/A/report.json',
        workerB='pen_solve_exit_replay_v2/B/report.json',
        readerA='pen_solve_exit_saved_review_v2/A/report.json',
        readerB='pen_solve_exit_saved_review_v2/B/report.json',
        all120='pen_solve_exit_summary_v2/report.json',
        pairs60='pen_solve_exit_scope_pairs_v1/report.json')
    roots = {k: file_pin(ROOT / 'runs' / v) for k, v in names.items()}
    reports = {k: json.loads(verify_file_pin(v).read_text()) for k, v in roots.items()}
    checks = {k: d['passed'] == d['total'] and all(d['checks'].values())
              for k, d in reports.items() if 'checks' in d}
    checks['full54_budget'] = (reports['full54']['actual_native_integrations'] ==
                              reports['full54']['attempted_native_integrations'] == 68840)
    checks['full54_no_success'] = reports['summary54']['summary']['full_task_successes'] == 0
    checks['diagnostic960'] = sum(reports[k]['actual_native_integrations'] for k in ['workerA', 'workerB']) == 960
    checks['all_masks_retained'] = reports['all120']['new_impulse_failures'] == 48
    checks['all_pairs_disclosed'] = (reports['pairs60']['all_trace_bytes_equal'] == 49 and
                                   reports['pairs60']['physical_masks_equal'] == 60)
    checks['zero_updates'] = all(d.get('optimizer_steps', 0) == 0 and d.get('normalizer_fits', 0) == 0
                                 for d in reports.values())
    if not all(checks.values()):
        raise ValueError('Closed-cohort evidence mismatch')
    documents = [ROOT / 'docs/research-plan-v1.50.md',
                 ROOT / 'docs/pen-full54-and-solve-exit-results-2026-11-17.md']
    source_files = sorted((ROOT / 'src').rglob('*.py')) + documents + [Path(__file__)]
    source_files += [ROOT / 'tools' / n for n in [
        'summarize_pen_solve_exit_v2.py', 'review_pen_solve_exit_scope_pairs_v1.py',
        'summarize_pen_native_screened_rollout.py', 'classify_pen_screened_stopping.py',
        'run_pen_solve_exit_replay_v2.py', 'review_pen_solve_exit_saved_v2.py']
        if (ROOT / 'tools' / n).exists()]
    out.mkdir(); dest.mkdir()
    receipt = freeze_sources(source_files, out / 'source_snapshot', ROOT,
                             subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write_json(out / 'source_receipt.json', receipt)
    attestations = [file_pin(ROOT / f'runs/{base}/{replica}/context_layout_adapter.json')
                    for base in ['pen_solve_exit_replay_v2', 'pen_solve_exit_saved_review_v2']
                    for replica in ['A', 'B']]
    todo = list(roots.values()) + attestations + [file_pin(out / 'source_receipt.json')]
    forbidden = {ROOT / 'runs/pen_native_screened_rollout_v1/work.json'}
    verified = {}; skipped = set()
    while todo:
        pin = todo.pop(); key = identity(pin); path = Path(pin['path']).resolve()
        if path in forbidden or (path.parent == ROOT / 'runs' and path.name.startswith('pen_native_screened_rollout') and path.suffix == '.log'):
            raise ValueError('Mutable producer progress/log must not be frozen')
        if key in known:
            skipped.add(key); continue
        if key in verified:
            continue
        verify_file_pin(pin); verified[key] = pin
        if path.suffix == '.json' and path.is_relative_to(ROOT / 'runs'):
            todo.extend(pins(json.loads(path.read_text())))
        if len(verified) % 1000 == 0:
            print(json.dumps(dict(new_verified=len(verified), inherited_reached=len(skipped))), flush=True)
    for pin in list(roots.values()) + attestations + [parent_pin] + inherited:
        verify_file_pin(pin)
    verify_source_snapshot(receipt, live=True)
    write_json(out / 'additional_verified_pins.json', dict(records=list(verified.values()),
        inherited_reached=len(skipped), scope='Only new pins cold-verified here; prior pins inherited through exact manifest identities.'))
    write_json(out / 'review.json', dict(checks=checks, passed=sum(checks.values()), total=len(checks),
        actual_native_integrations=0, optimizer_steps=0, normalizer_fits=0))
    ledger = dict(schema='pen_full54_solve_exit_checkpoint:1', prior_evidence=parent_pin,
        inherited_verified_manifests=inherited, roots=roots, attestations=attestations,
        documents=[file_pin(p) for p in documents], source_receipt=file_pin(out / 'source_receipt.json'),
        review=file_pin(out / 'review.json'), additional_verified_manifest=file_pin(out / 'additional_verified_pins.json'),
        additional_verified_pins=len(verified), inherited_pins_reached=len(skipped),
        full54_native_integrations=68840, diagnostic_native_integrations=960,
        archival_native_integrations=0, optimizer_steps=0, normalizer_fits=0,
        scope='Closed54 and both actual V2 processes/readers. Original failures, nonreproductions, observer differences and censored outcomes retained. '
              'Sources/results pinned; mutable producer progress/root log excluded. Earlier manifests inherited, not all cold-rehashed. '
              'No termination-rule comparator yet, no Actor/ego/speed admission, full research goal incomplete.')
    write_json(dest / 'ledger.json', ledger)
    print(json.dumps(dict(ledger=file_pin(dest / 'ledger.json'), checks=[sum(checks.values()), len(checks)],
                          additional_verified_pins=len(verified), inherited_reached=len(skipped))), flush=True)


if __name__ == '__main__':
    main()
