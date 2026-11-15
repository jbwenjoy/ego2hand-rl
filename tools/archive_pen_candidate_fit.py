"""Compact tracked candidate-fit evidence; exact large artifacts stay local."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from archive_pen_source_ccd_evidence import pins, write
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot, verify_source_pin


def main():
    out = ROOT / 'provenance/pen_candidate_fit_results'
    analysis = ROOT / 'runs/pen_candidate_fit_analysis_v1'
    if out.exists() or analysis.exists():
        raise ValueError('Fresh immutable evidence directory required')
    reports = dict(
        first100=ROOT / 'runs/pen_candidate_fit_first100_v1/health_0100.json',
        final=ROOT / 'runs/pen_candidate_fit_continuation_v1/health_2000.json',
        physical=ROOT / 'runs/pen_candidate_fit_evaluation_v1/report.json',
        recursive=ROOT / 'runs/pen_candidate_fit_rollout_v1/report.json',
        figure=ROOT / 'runs/pen_candidate_fit_figure_v1/report.json')
    receipt_paths = [ROOT / ('runs/' + folder + '/source_receipt.json') for folder in [
        'pen_candidate_fit_first100_sources_v1', 'pen_candidate_fit_continuation_sources_v1',
        'pen_candidate_fit_evaluation_sources_v1', 'pen_candidate_fit_rollout_sources_v1']]
    analysis.mkdir()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = freeze_sources([Path(__file__), ROOT / 'tools/archive_pen_source_ccd_evidence.py',
                              ROOT / 'tools/plot_pen_candidate_fit.py'], analysis / 'snapshot', ROOT, revision)
    write(analysis / 'source_receipt.json', manifest)
    receipt_paths.append(analysis / 'source_receipt.json')
    prior = ROOT / 'provenance/pen_candidate_response_results/source_resolutions.json'
    aliases = {json.dumps(e['original'], sort_keys=True): e for e in json.loads(prior.read_text())['records']}
    for path in receipt_paths:
        receipt = json.loads(path.read_text())
        verify_source_snapshot(receipt)
        for e in receipt['records']:
            aliases.setdefault(json.dumps(e['observed'], sort_keys=True), dict(
                original=e['observed'], archived=e['snapshot'], reason='Exact observed source-byte receipt; no basename or hash-only fallback'))
    resolutions = dict(schema='explicit_source_relocations:1', records=list(aliases.values()))
    for e in resolutions['records']:
        verify_source_pin(e['original'], resolutions)
    docs = [ROOT / ('docs/' + name) for name in ['research-plan-v0.87.md', 'research-plan-v0.88.md',
        'research-plan-v0.89.md', 'research-plan-v0.90.md', 'pen-candidate-fit-results-2026-11-15.md']]
    test = ROOT / 'runs/pen_candidate_fit_final_tests229.log'
    todo = [file_pin(p) for p in list(reports.values()) + receipt_paths + docs + [test]]
    allowed = {p.parent for p in reports.values()} | {p.parent for p in receipt_paths}
    verified = {}
    while todo:
        pin = todo.pop()
        key = json.dumps(pin, sort_keys=True)
        if key in verified:
            continue
        resolved = verify_source_pin(pin, resolutions)
        verified[key] = dict(observed=pin, resolved=file_pin(resolved))
        if resolved.suffix == '.json' and any(resolved.is_relative_to(folder) for folder in allowed):
            todo += list(pins(json.loads(resolved.read_text())))
    h, f, e, r = [json.loads(reports[k].read_text()) for k in ['first100', 'final', 'physical', 'recursive']]
    for health, expected in [(h, 43), (f, 44)]:
        if health['passed'] != expected or health['total'] != expected or not all(health['checks'].values()) or health['training_running']:
            raise ValueError('Terminal healthy checkpoints required')
    summary = dict(schema='candidate_fit_evidence_summary:1', roots={k: file_pin(p) for k, p in reports.items()},
        health=dict(first100=[h['passed'], h['total']], final=[f['passed'], f['total']]),
        checkpoints=dict(first100=h['checkpoint'], final=f['checkpoint']), CHECK_curve=f['CHECK_curve'],
        one_step={mode: m['one_step'] for mode, m in e['models'].items()},
        physical=dict(attempted=len(e['pairs']), eligible=sum(p['eligible'] for p in e['pairs']),
            failed_flags={k: sum(not p['checks'][k] for p in e['pairs']) for k in e['pairs'][0]['checks']},
            models={mode: m['physical_finite_response']['summary'] for mode, m in e['models'].items()}),
        recursive=dict(cases=r['cases'], coverage=r['window_coverage'], models=r['models']),
        paired_compute_and_metrics_s=f['paired_compute_and_metric_wall_s'], training_peak_bytes=f['training_peak_bytes'],
        GPU_telemetry=f['GPU_telemetry'], updates_per_arm=2000, live_optimizer_updates=4000,
        clone_optimizer_updates=h['extra_clone_optimizer_updates'] + f['extra_clone_optimizer_updates'],
        Actor_optimizer_updates=0, additional_native_integrations=0,
        prior_data_evidence=file_pin(ROOT / 'provenance/pen_candidate_response_results/ledger.json'),
        scope='Fixed-budget WM components remain unready for Actor gradients/control. Correlated same-clip development data, failed source tasks, uncertain excluded labels; no ego/task/acceleration claim. Full Goal incomplete and active.')
    out.mkdir()
    write(out / 'summary.json', summary)
    write(out / 'source_resolutions.json', resolutions)
    shutil.copyfile(test, out / test.name)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='candidate_fit_evidence_ledger:1', analysis_revision=revision,
        roots={k: file_pin(p) for k, p in reports.items()}, summary=file_pin(out / 'summary.json'),
        source_resolutions=file_pin(out / 'source_resolutions.json'), previous_resolutions=file_pin(prior),
        source_receipts=[file_pin(p) for p in receipt_paths], verified_pins=len(verified), exact_source_aliases=len(aliases),
        verified_manifest=file_pin(analysis / 'verified_pins.json'), test_log=file_pin(out / test.name),
        documents=[file_pin(p) for p in docs], live_optimizer_updates=4000, clone_optimizer_updates=8,
        Actor_updates=0, additional_native_integrations=0,
        scope='Detailed nested evidence verified in this cohort; older data are exact boundary inputs backed by their existing7342-pin ledger. No rewriting of historical source or artifacts.'))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), verified=len(verified), aliases=len(aliases))))


if __name__ == '__main__':
    main()
