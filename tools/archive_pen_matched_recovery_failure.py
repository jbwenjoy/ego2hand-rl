"""Seal both startup and measured recovery failures without double charging work."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from archive_pen_source_ccd_evidence import pins, write
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_pin, verify_source_snapshot


def main():
    out = ROOT / 'provenance/pen_matched_recovery_failure_v2'
    analysis = ROOT / 'runs/pen_matched_recovery_failure_archive_v1'
    if out.exists() or analysis.exists():
        raise ValueError('Fresh terminal archive required')
    prior_path = ROOT / 'provenance/pen_long_support206_failure/ledger.json'
    prior = json.loads(prior_path.read_text())
    aliases = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    for item in json.loads(verify_file_pin(prior['verified_manifest']).read_text())['records']:
        if file_pin(verify_source_pin(item['observed'], aliases)) != item['resolved']:
            raise ValueError('Historical resolution changed')
    root_names = dict(
        startup_registration='pen_matched_recovery_registration_v1/registration.json',
        registration='pen_matched_recovery_registration_v2/registration.json',
        startup_evaluation_registration='pen_matched_recovery_evaluation_registration_v1/registration.json',
        evaluation_registration='pen_matched_recovery_evaluation_registration_v2/registration.json',
        W1_admission='pen_matched_recovery_W1_admission_v1/report.json',
        startup_failure='pen_matched_recovery_training_v1/failure.json',
        parent_load_qualification='pen_matched_recovery_parent_loads_v1/report.json',
        measured_failure='pen_matched_recovery_training_v2/failure.json',
        first_checkpoint_health='pen_matched_recovery_training_v2/RESIDUAL/health_0201.json',
        parent_reload='pen_matched_recovery_training_v2/RESIDUAL/parent_reload.json',
        failed_health='pen_matched_recovery_training_v2/RESIDUAL/collector_health_0252.json',
        overflow_review='pen_matched_recovery_overflow_review_v1/report.json')
    roots = {k: file_pin(ROOT / 'runs' / name) for k, name in root_names.items()}
    parsed = {k: json.loads(verify_file_pin(p).read_text()) for k, p in roots.items()}
    f = parsed['measured_failure']
    if (f['phase'], f['actual_native_integrations'], f['optimizer_steps'], f['completed_call_accounting_exact']) != ('RESIDUAL_consistency_252', 851968, 1632, True):
        raise ValueError('Observed failed recovery differs')
    for name in ['pen_matched_recovery_training_v1', 'pen_matched_recovery_training_v2']:
        if (ROOT / 'runs' / name / 'report.json').exists():
            raise ValueError('Failed cohort must not have a completed report')
    if any(parsed['startup_failure'][k] for k in ['actual_native_integrations', 'optimizer_steps']):
        raise ValueError('Startup work differs')
    if (parsed['parent_load_qualification']['passed'], parsed['overflow_review']['passed']) != (21, 13):
        raise ValueError('Observed diagnostic checks differ')
    if not all(parsed['first_checkpoint_health']['first_CPU_readback'].values()):
        raise ValueError('First checkpoint byte readback failed')
    docs = [ROOT / 'docs' / name for name in ['pen-matched-recovery-startup-and-overflow-2026-11-16.md', 'research-plan-v1.22.md']]
    scripts = [Path(__file__), ROOT / 'tools/review_pen_recovery_overflow.py']
    analysis.mkdir()
    receipt = freeze_sources(scripts + docs + [ROOT / 'README.md'], analysis / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(analysis / 'source_receipt.json', receipt)
    receipts = sorted({p.parent.parent / 'source_receipt.json' if p.parent.name == 'RESIDUAL' else p.parent / 'source_receipt.json'
                       for p in [Path(pin['path']) for pin in roots.values()]})
    receipts = [p for p in receipts if p.exists()] + [analysis / 'source_receipt.json']
    merged = {json.dumps(v['original'], sort_keys=True): v for v in aliases['records']}
    verification = []
    for path in receipts:
        manifest = json.loads(path.read_text())
        # V1 used the pre-fix callback CLI. Its frozen bytes remain authoritative.
        historical = path.parent.name in {'pen_matched_recovery_registration_v1', 'pen_matched_recovery_training_v1',
                                           'pen_matched_recovery_evaluation_registration_v1'}
        verify_source_snapshot(manifest, live=not historical)
        verification.append(dict(receipt=file_pin(path), live=not historical))
        for record in manifest['records']:
            merged.setdefault(json.dumps(record['observed'], sort_keys=True), dict(original=record['observed'],
                archived=record['snapshot'], reason='Exact historical recovery source: full original path, bytes and SHA.'))
    aliases = dict(schema='explicit_source_relocations:1', records=list(merged.values()))
    logs = [ROOT / 'runs' / name for name in ['pen_matched_recovery_training_console_v1.log',
        'pen_matched_recovery_training_console_v2.log', 'pen_matched_recovery_tests431_v2.log']]
    todo = list(roots.values()) + [file_pin(p) for p in receipts + docs + scripts + logs + [prior_path]]
    allowed = {Path(pin['path']).parent for pin in roots.values()} | {analysis}
    verified = {}
    while todo:
        pin = todo.pop()
        key = json.dumps(pin, sort_keys=True)
        if key in verified:
            continue
        resolved = verify_source_pin(pin, aliases)
        verified[key] = dict(observed=pin, resolved=file_pin(resolved))
        if resolved.suffix == '.json' and any(resolved.is_relative_to(p) for p in allowed):
            todo.extend(pins(json.loads(resolved.read_text())))
    out.mkdir()
    summary = dict(roots=roots, source_verification=verification,
        added_actual_work=dict(native_integrations=851968, optimizer_steps=1632),
        startup_and_load_qualification_work=dict(native_integrations=0, optimizer_steps=0),
        named_lineage_actual_work=dict(native_integrations=37098304, optimizer_steps=72384),
        current_live_training=False, failed_batch_optimized=False, final_equal_budget_result=False,
        raw_incident_native_mask_available=False, original_SUPPORT206_cause_proven=False,
        scope='Terminal measured failure; historical original and replay costs are not charged again. Named lineage is not all historical research cost. No package/driver/CUDA change or physics gate relaxation.')
    write(out / 'summary.json', summary)
    write(out / 'source_resolutions.json', aliases)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='pen_matched_recovery_failure_ledger:1', prior_evidence=file_pin(prior_path),
        roots=roots, summary=file_pin(out / 'summary.json'), documents=[file_pin(p) for p in docs],
        source_receipts=[file_pin(p) for p in receipts], source_resolutions=file_pin(out / 'source_resolutions.json'),
        verified_manifest=file_pin(analysis / 'verified_pins.json'), verified_pins=len(verified),
        actual_native_integrations=851968, optimizer_steps=1632, current_live_training=False))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), verified_pins=len(verified))), flush=True)


if __name__ == '__main__':
    main()
