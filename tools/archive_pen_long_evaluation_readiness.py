"""Freeze verified W1 admission without claiming a completed long PPO trial."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from archive_pen_source_ccd_evidence import pins, write
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_pin, verify_source_snapshot


def main():
    out = ROOT / 'provenance/pen_long_control_evaluation_readiness_v1'
    analysis = ROOT / 'runs/pen_long_control_evaluation_readiness_v1'
    if out.exists() or analysis.exists():
        raise ValueError('Fresh evidence paths required')
    prior_path = ROOT / 'provenance/pen_long_control_registration_v1/ledger.json'
    prior = json.loads(prior_path.read_text())
    aliases = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    manifest = json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for record in manifest['records']:
        if file_pin(verify_source_pin(record['observed'], aliases)) != record['resolved']:
            raise ValueError('Prior source resolution changed')
    report_path = ROOT / 'runs/pen_long_control_W1_admission_v1/report.json'
    report = json.loads(report_path.read_text())
    audit = report['audit']
    if (report['actual_native_integrations'], report['optimizer_steps'], report['policy_performance_evaluated']) != (0, 0, False):
        raise ValueError('Admission-only zero-work scope required')
    if (audit['full_W1_model_arrays'], audit['full_W1_model_scalars'], audit['evaluation_features']) != (404, 195, 'LIVE'):
        raise ValueError('Full LIVE W1 compatibility differs')
    if not audit['full_W1_model_bytes_equal'] or not audit['actor_normalizer_bytes_equal'] or audit['old_native_optimizer_critic_RNG_counters_loaded']:
        raise ValueError('Actor-only admission differs')
    log = ROOT / 'runs/pen_fn_long_W1_tests415_v1.log'
    text = log.read_text(encoding='utf-8').replace('\r\n', '\n')
    if 'Ran 415 tests in ' not in text or '\nOK\n' not in text:
        raise ValueError('Full415-test evidence required')
    doc = ROOT / 'docs/pen-long-control-evaluation-readiness-2026-11-16.md'
    sources = [Path(__file__), doc, ROOT / 'README.md', ROOT / 'tests/test_pen_w1_evaluation.py']
    analysis.mkdir()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    receipt = freeze_sources(sources, analysis / 'source_snapshot', ROOT, revision)
    write(analysis / 'source_receipt.json', receipt)
    receipts = [Path(report['source_receipt']['path']), analysis / 'source_receipt.json']
    merged = {json.dumps(r['original'], sort_keys=True): r for r in aliases['records']}
    for path in receipts:
        source = json.loads(path.read_text())
        verify_source_snapshot(source, live=True)
        for record in source['records']:
            merged.setdefault(json.dumps(record['observed'], sort_keys=True), dict(
                original=record['observed'], archived=record['snapshot'], reason='Exact full path/bytes/SHA;LIVE W1 admission preparation.'))
    aliases = dict(schema='explicit_source_relocations:1', records=list(merged.values()))
    registration = verify_file_pin(report['registration'])
    todo = [file_pin(p) for p in [report_path, registration, doc, log, prior_path] + receipts]
    expand = {str(p) for p in [report_path, registration] + receipts}
    verified = {}
    while todo:
        pin = todo.pop()
        key = json.dumps(pin, sort_keys=True)
        if key in verified:
            continue
        resolved = verify_source_pin(pin, aliases)
        verified[key] = dict(observed=pin, resolved=file_pin(resolved))
        if pin['path'] in expand:
            todo.extend(pins(json.loads(resolved.read_text())))
    # Verify the active trainer's captured bytes without importing its runtime.
    training_receipt = ROOT / 'runs/pen_long_control_training_v1/source_receipt.json'
    verify_source_snapshot(json.loads(training_receipt.read_text()), live=True)
    out.mkdir()
    shutil.copyfile(log, out / log.name)
    write(out / 'admission_summary.json', report)
    write(out / 'source_resolutions.json', aliases)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='pen_long_control_evaluation_readiness_ledger:1',
          prior_evidence=file_pin(prior_path), registration=file_pin(registration),
          admission_report=file_pin(report_path), summary=file_pin(out / 'admission_summary.json'),
          documents=[file_pin(doc)], test_log=file_pin(out / log.name),
          source_receipts=[file_pin(p) for p in receipts], source_resolutions=file_pin(out / 'source_resolutions.json'),
          verified_manifest=file_pin(analysis / 'verified_pins.json'), verified_pins=len(verified),
          preparation_native_integrations=0, preparation_optimizer_steps=0,
          long_training_final_actual_work=None, policy_performance_evaluated=False,
          scope='Unstepped admission only;active long training,final evaluation/video and original broader goals remain incomplete.'))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), verified_pins=len(verified))), flush=True)


if __name__ == '__main__':
    main()
