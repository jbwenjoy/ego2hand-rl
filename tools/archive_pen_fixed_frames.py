"""Archive both fixed-frame cache cohorts without pooling source revisions."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import torch
from archive_pen_source_ccd_evidence import pins, write
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import (
    freeze_sources, verify_source_pin, verify_source_snapshot,
)


def main():
    out = ROOT / 'provenance/pen_fixed_frame_results'
    analysis = ROOT / 'runs/pen_fixed_frame_archive_v1'
    if out.exists() or analysis.exists():
        raise ValueError('Fresh immutable fixed-frame evidence required')
    prior_path = ROOT / 'provenance/pen_ordinary_performance_results/ledger.json'
    prior = json.loads(prior_path.read_text())
    previous = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    old, counts, cursor = set(), [], prior
    while True:
        manifest = json.loads(verify_file_pin(cursor['verified_manifest']).read_text())
        counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key = json.dumps(entry['observed'], sort_keys=True)
            if key not in old:
                verify_source_pin(entry['observed'], previous)
                old.add(key)
        if cursor['schema'] == 'pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:
            break
        cursor = json.loads(verify_file_pin(cursor['prior_evidence']).read_text())

    names = dict(
        failed='pen_fixed_frame_qualification_v1',
        initial_qualification='pen_fixed_frame_qualification_v2',
        final_qualification='pen_fixed_frame_qualification_v3',
        initial_comparison='pen_fixed_frame_comparison_v1',
        final_comparison='pen_fixed_frame_comparison_v2',
        initial_gradients='pen_fixed_frame_gradients_v1',
        final_gradients='pen_fixed_frame_gradients_v2',
        review='pen_fixed_frame_review_v1',
    )
    reports = {k: ROOT / 'runs' / v / ('failure.json' if k == 'failed' else 'report.json')
               for k, v in names.items()}
    data = {k: json.loads(p.read_text()) for k, p in reports.items()}
    expected = dict(initial_qualification=[386, 386], final_qualification=[388, 388],
                    initial_comparison=[4, 4], final_comparison=[4, 4],
                    initial_gradients=[9, 9], final_gradients=[9, 9], review=[37, 37])
    if any([data[k]['passed'], data[k]['total']] != v for k, v in expected.items()):
        raise ValueError('Observed qualification checks differ')
    native = sum(d['actual_native_integrations'] for d in data.values())
    updates = sum(d['optimizer_steps'] for d in data.values())
    if (native, updates) != (331776, 640):
        raise ValueError('Preserve both diagnostic timing cohorts and their work')
    review = data['review']
    if (review['performance_status'] != 'OBSERVED_COMPONENT_TIMING_GAIN' or
            review['mechanism_status'] != 'EXACT_TESTED_FEATURES_GRADIENTS_AND_DECLARED_SOURCE_FORKS'):
        raise ValueError('Preserve bounded measured timing/mechanism scope')
    for key in ['initial_qualification', 'final_qualification']:
        if not data[key]['all_feature_rows_exact'] or any(data[key]['max_abs'].values()):
            raise ValueError('Fixed-input exactness evidence changed')
    log = ROOT / 'runs/pen_fixed_frame_tests374_v1.log'
    test_text = log.read_text(encoding='utf-8').replace('\r\n', '\n')
    if 'Ran 374 tests in ' not in test_text or '\nOK\n' not in test_text:
        raise ValueError('Full374-test evidence required')
    docs = [ROOT / 'docs' / n for n in [
        'pen-fixed-frame-cache-protocol-v1.md', 'pen-fixed-frame-gradient-protocol-v1.md',
        'pen-fixed-frame-cache-results-2026-11-16.md', 'research-plan-v1.12.md']]
    configs = [ROOT / 'configs' / n for n in [
        'pen_fixed_frame_qualification_v1.toml', 'pen_fixed_frame_comparison_v1.toml',
        'pen_fixed_frame_comparison_v2.toml']]
    scripts = [ROOT / 'tools' / n for n in [
        'profile_ordinary_pen_ppo.py', 'qualify_pen_fixed_frames.py',
        'check_pen_fixed_frame_gradients.py', 'review_pen_fixed_frames.py',
        'archive_pen_fixed_frames.py', 'archive_pen_source_ccd_evidence.py']]
    sources = sorted({p for pkg in ['supermimic_research', 'dex_supermimic', 'chord_study']
                      for p in (ROOT / 'src' / pkg).rglob('*.py')})
    sources += sorted((ROOT / 'tests').glob('*.py')) + docs + configs + scripts
    sources += [ROOT / 'README.md', ROOT / 'AGENTS.md']
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    analysis.mkdir()
    receipt = freeze_sources(sources, analysis / 'source_snapshot', ROOT, revision)
    write(analysis / 'source_receipt.json', receipt)
    receipts = [p.parent / 'source_receipt.json' for p in reports.values()]
    receipts += [analysis / 'source_receipt.json']
    aliases = {json.dumps(e['original'], sort_keys=True): e for e in previous['records']}
    for path in receipts:
        source = json.loads(path.read_text())
        verify_source_snapshot(source)
        for entry in source['records']:
            aliases.setdefault(json.dumps(entry['observed'], sort_keys=True), dict(
                original=entry['observed'], archived=entry['snapshot'],
                reason='Exact original full path/bytes/SHA;preserve failed probe and separate pre/post saved-mode-guard cohorts.'))
    resolutions = dict(schema='explicit_source_relocations:1', records=list(aliases.values()))
    for entry in resolutions['records']:
        verify_source_pin(entry['original'], resolutions)

    engines, fixtures = [], []
    for key in ['initial_comparison', 'final_comparison', 'final_qualification']:
        path = reports[key].parent / ('native_source.pt' if key == 'final_qualification' else 'fixture.pt')
        fixtures.append(file_pin(path))
        state = torch.load(path, map_location='cpu', weights_only=False)
        state = state if key == 'final_qualification' else state['native']
        verify_source_snapshot(state['sources'])
        for pin in state['engine_sources']:
            verify_source_pin(pin, resolutions)
        engines += state['engine_sources']
    logs = [log] + [ROOT / 'runs' / n for n in [
        'pen_fixed_frame_qualification_console_v1.log', 'pen_fixed_frame_qualification_console_v2.log',
        'pen_fixed_frame_qualification_console_v3.log', 'pen_fixed_frame_comparison_console_v1.log',
        'pen_fixed_frame_comparison_console_v2.log', 'pen_fixed_frame_gradients_console_v1.log',
        'pen_fixed_frame_gradients_console_v2.log', 'pen_fixed_frame_review_console_v1.log']]
    todo = [file_pin(p) for p in list(reports.values()) + receipts + docs + configs + logs + [prior_path]]
    todo += fixtures + engines
    allowed = {p.parent for p in reports.values()} | {analysis}
    verified = {}
    skip_json = {'trace.json', 'torch_cpu_trace.json', 'python_calls.json',
                 'torch_cpu_operators.json', 'key_averages.json'}
    while todo:
        pin = todo.pop()
        key = json.dumps(pin, sort_keys=True)
        if key in verified:
            continue
        resolved = verify_source_pin(pin, resolutions)
        verified[key] = dict(observed=pin, resolved=file_pin(resolved))
        if (resolved.suffix == '.json' and resolved.name not in skip_json and
                any(resolved.is_relative_to(folder) for folder in allowed)):
            todo += list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt, live=True)
    summary = dict(
        schema='pen_fixed_frame_cache_summary:1', roots={k: file_pin(p) for k, p in reports.items()},
        checks=expected, failed_attempt=data['failed'], timing=review['timing'],
        collect_throughput_ratio=review['collect_throughput_ratio'],
        collect_wall_reduction_fraction=review['collect_wall_reduction_fraction'],
        whole_phase_throughput_ratio=review['whole_phase_throughput_ratio'],
        performance_status=review['performance_status'], mechanism_status=review['mechanism_status'],
        operators=review['operators'], cohorts_pooled=False, default_mode='LIVE',
        explicit_cold_cached_mode_available=True, main_policy_promoted=False, unit_tests=374,
        actual_native_integrations=native, optimizer_steps=updates,
        scope='Separate source-version cohorts;final early-training32-world collector timing improved with exact tested features/JVP and declared three-field source forks. Not universal/full-training speed,identical long trajectories,physical WM gradients,main-policy/task/ego success.')
    out.mkdir()
    write(out / 'summary.json', summary)
    write(out / 'source_resolutions.json', resolutions)
    shutil.copyfile(log, out / log.name)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(
        schema='pen_fixed_frame_cache_ledger:1', analysis_revision=revision,
        summary=file_pin(out / 'summary.json'), roots={k: file_pin(p) for k, p in reports.items()},
        source_resolutions=file_pin(out / 'source_resolutions.json'),
        source_receipts=[file_pin(p) for p in receipts], documents=[file_pin(p) for p in docs],
        prior_evidence=file_pin(prior_path), prior_manifest_counts=counts,
        prior_unique_reverified_pins=len(old), verified_pins=len(verified),
        exact_source_aliases=len(aliases), verified_manifest=file_pin(analysis / 'verified_pins.json'),
        test_log=file_pin(out / log.name), actual_native_integrations=native, optimizer_steps=updates,
        performance_status=review['performance_status'], mechanism_status=review['mechanism_status'],
        scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), verified_pins=len(verified),
                          source_aliases=len(aliases), prior_reverified=len(old))), flush=True)


if __name__ == '__main__':
    main()
