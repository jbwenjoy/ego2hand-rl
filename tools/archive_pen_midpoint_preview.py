"""Retain an actual failed midpoint episode, outside the running training cost."""
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
    out = ROOT / 'provenance/pen_long_control_midpoint_diagnostic_v1'
    analysis = ROOT / 'runs/pen_long_midpoint_archive_v1'
    if out.exists() or analysis.exists():
        raise ValueError('Fresh archive required')
    prior_path = ROOT / 'provenance/pen_long_control_evaluation_readiness_v1/ledger.json'
    prior = json.loads(prior_path.read_text())
    aliases = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    manifest = json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for record in manifest['records']:
        if file_pin(verify_source_pin(record['observed'], aliases)) != record['resolved']:
            raise ValueError('Prior evidence changed')
    run = ROOT / 'runs/pen_long_control_midpoint_diagnostic_v1'
    report_path = run / 'report.json'
    report = json.loads(report_path.read_text())
    registration = json.loads((run / 'registration.json').read_text())
    case = report['case']
    if (case['iteration'], case['controls'], report['actual_native_integrations'], report['optimizer_steps']) != (1000, 93, 744, 0):
        raise ValueError('Retain the declared actual diagnostic work')
    if not registration['checkpoint_absent_at_registration'] or case['summary']['qualified'] or not case['termination_reasons']['position']:
        raise ValueError('Preserve preregistration and observed unsuccessful task')
    doc = ROOT / 'docs/pen-long-midpoint-results-2026-11-16.md'
    analysis.mkdir()
    receipt = freeze_sources([Path(__file__), doc], analysis / 'source_snapshot', ROOT,
                             subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(analysis / 'source_receipt.json', receipt)
    receipts = [run / 'source_receipt.json', analysis / 'source_receipt.json']
    merged = {json.dumps(a['original'], sort_keys=True): a for a in aliases['records']}
    for path in receipts:
        source = json.loads(path.read_text())
        verify_source_snapshot(source, live=True)
        for record in source['records']:
            merged.setdefault(json.dumps(record['observed'], sort_keys=True), dict(original=record['observed'],
                archived=record['snapshot'], reason='Exact midpoint diagnostic source path/bytes/SHA.'))
    aliases = dict(schema='explicit_source_relocations:1', records=list(merged.values()))
    todo = [file_pin(p) for p in [report_path, run / 'registration.json', doc, prior_path] + receipts]
    allowed = {run, analysis}
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
    verify_source_snapshot(json.loads((ROOT / 'runs/pen_long_control_training_v1/source_receipt.json').read_text()), live=True)
    out.mkdir()
    write(out / 'summary.json', dict(registration=report['registration'], report=file_pin(report_path),
        checkpoint=case['checkpoint'], summary=case['summary'], termination_reasons=case['termination_reasons'],
        video=report['rendering']['video'], actual_native_integrations=744, optimizer_steps=0,
        main_policy_promoted=False, final_training_actual_work=None, scope=report['scope']))
    write(out / 'source_resolutions.json', aliases)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='pen_long_control_midpoint_diagnostic_ledger:1',
        prior_evidence=file_pin(prior_path), summary=file_pin(out / 'summary.json'), report=file_pin(report_path),
        documents=[file_pin(doc)], source_receipts=[file_pin(p) for p in receipts],
        source_resolutions=file_pin(out / 'source_resolutions.json'), verified_manifest=file_pin(analysis / 'verified_pins.json'),
        verified_pins=len(verified), actual_native_integrations=744, optimizer_steps=0,
        final_training_actual_work=None, scope=report['scope']))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), verified_pins=len(verified))), flush=True)


if __name__ == '__main__':
    main()
