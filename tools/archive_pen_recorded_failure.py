"""Close offline diagnosis and recorder QA;long training stays a separate cohort."""
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
    out = ROOT / 'provenance/pen_recorded_failure_and_evaluation_v2'
    analysis = ROOT / 'runs/pen_recorded_failure_archive_v1'
    if out.exists() or analysis.exists():
        raise ValueError('Fresh closed-cohort archive required')
    prior_path = ROOT / 'provenance/pen_long_control_midpoint_diagnostic_v1/ledger.json'
    prior = json.loads(prior_path.read_text())
    aliases = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    manifest = json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for item in manifest['records']:
        if file_pin(verify_source_pin(item['observed'], aliases)) != item['resolved']:
            raise ValueError('Prior archived source resolution differs')
    names = dict(diagnosis_v1='pen_long_midpoint_failure_diagnosis_v1', diagnosis_v2='pen_long_midpoint_failure_diagnosis_v2',
                 recording_QA='pen_long_control_recording_qualification_v1', QA_diagnosis='pen_long_recording_QA_diagnosis_v1')
    roots = {k: ROOT / 'runs' / v / 'report.json' for k, v in names.items()}
    data = {k: json.loads(p.read_text()) for k, p in roots.items()}
    for key, count in [('diagnosis_v1', 11), ('diagnosis_v2', 11), ('QA_diagnosis', 17)]:
        if data[key]['passed'] != count or data[key]['total'] != count or data[key]['actual_native_integrations'] or data[key]['optimizer_steps']:
            raise ValueError('Retain offline arithmetic and zero new native/optimizer work')
    qa = data['recording_QA']
    control = qa['executed_control_audit']
    if (qa['actual_native_integrations'], qa['optimizer_steps'], control['passed'], control['total'],
            control['max_abs_GPU_ctrl_reconstruction_error_rad']) != (88, 0, 6, 6, 0.):
        raise ValueError('Actual recorder QA work/audit differs')
    log = ROOT / 'runs/pen_long_recording_tests419_v1.log'
    text = log.read_text(encoding='utf-8').replace('\r\n', '\n')
    if 'Ran 419 tests in ' not in text or '\nOK\n' not in text:
        raise ValueError('Full419-test evidence required')
    docs = [ROOT / 'docs' / n for n in ['pen-long-recorded-failure-and-evaluation-v2-2026-11-16.md', 'research-plan-v1.19.md']]
    sources = [Path(__file__), ROOT / 'README.md', ROOT / 'tests/test_pen_executed_control_audit.py'] + docs
    analysis.mkdir()
    receipt = freeze_sources(sources, analysis / 'source_snapshot', ROOT,
                             subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(analysis / 'source_receipt.json', receipt)
    receipts = [p.parent / 'source_receipt.json' for p in roots.values()] + [analysis / 'source_receipt.json']
    merged = {json.dumps(v['original'], sort_keys=True): v for v in aliases['records']}
    for path in receipts:
        source = json.loads(path.read_text())
        verify_source_snapshot(source)
        for record in source['records']:
            merged.setdefault(json.dumps(record['observed'], sort_keys=True), dict(original=record['observed'],
                archived=record['snapshot'], reason='Exact path/bytes/SHA;preserved offline views and recorder-only amendment.'))
    aliases = dict(schema='explicit_source_relocations:1', records=list(merged.values()))
    registration = ROOT / 'runs/pen_long_control_evaluation_registration_v2/registration.json'
    todo = [file_pin(p) for p in list(roots.values()) + receipts + docs + [log, registration, prior_path]]
    allowed = {p.parent for p in roots.values()} | {analysis, registration.parent}
    verified = {}
    while todo:
        pin = todo.pop()
        key = json.dumps(pin, sort_keys=True)
        if key in verified:
            continue
        resolved = verify_source_pin(pin, aliases)
        verified[key] = dict(observed=pin, resolved=file_pin(resolved))
        if resolved.suffix == '.json' and any(resolved.is_relative_to(parent) for parent in allowed):
            todo.extend(pins(json.loads(resolved.read_text())))
    for folder in ['pen_long_control_training_v1', 'pen_long_control_evaluation_registration_v2']:
        verify_source_snapshot(json.loads((ROOT / 'runs' / folder / 'source_receipt.json').read_text()), live=True)
    verify_source_snapshot(receipt, live=True)
    out.mkdir()
    shutil.copyfile(log, out / log.name)
    write(out / 'summary.json', dict(roots={k: file_pin(p) for k, p in roots.items()},
        events=data['diagnosis_v2']['events'], reference=data['diagnosis_v2']['reference'],
        descriptive_intervals=data['diagnosis_v2']['descriptive_intervals'], raw_action=data['diagnosis_v2']['raw_action'],
        executed_recording_QA=control, actual_native_integrations=88, optimizer_steps=0,
        long_training_final_actual_work=None, final_evaluation_registration=file_pin(registration),
        scope='Single-episode observational failure diagnosis plus first-checkpoint recording QA. No causal,final policy,WM Actor or ego success claim. Added88 native only;prior744/4096 and ongoing long training are separate.'))
    write(out / 'source_resolutions.json', aliases)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='pen_recorded_failure_and_evaluation_v2_ledger:1',
        prior_evidence=file_pin(prior_path), roots={k: file_pin(p) for k, p in roots.items()},
        summary=file_pin(out / 'summary.json'), final_evaluation_registration=file_pin(registration),
        documents=[file_pin(p) for p in docs], test_log=file_pin(out / log.name),
        source_receipts=[file_pin(p) for p in receipts], source_resolutions=file_pin(out / 'source_resolutions.json'),
        verified_manifest=file_pin(analysis / 'verified_pins.json'), verified_pins=len(verified),
        actual_native_integrations=88, optimizer_steps=0, long_training_final_actual_work=None))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), verified_pins=len(verified))), flush=True)


if __name__ == '__main__':
    main()
