"""Close the no-new-training audit;embedded live-job milestones are nonadditive."""
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
    out = ROOT / 'provenance/ego_gravity_and_long_arm_transition'
    analysis = ROOT / 'runs/ego_gravity_transition_archive_v1'
    if out.exists() or analysis.exists():
        raise ValueError('Fresh immutable audit archive required')
    prior_path = ROOT / 'provenance/pen_recorded_failure_and_evaluation_v2/ledger.json'
    prior = json.loads(prior_path.read_text())
    aliases = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    manifest = json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for record in manifest['records']:
        if file_pin(verify_source_pin(record['observed'], aliases)) != record['resolved']:
            raise ValueError('Prior manifest resolution differs')
    roots = dict(gravity=ROOT / 'runs/mug_gravity_transport_audit_v1/report.json',
                 transition=ROOT / 'runs/pen_long_control_second_arm_milestone_v2/report.json',
                 transition_inline=ROOT / 'runs/pen_long_control_second_arm_milestone_v1/report.json')
    data = {k: json.loads(p.read_text()) for k, p in roots.items()}
    for key, expected in [('gravity', 8), ('transition', 14), ('transition_inline', 13)]:
        if data[key]['passed'] != expected or data[key]['total'] != expected:
            raise ValueError('Preserve observed audit checks')
    if data['gravity']['actual_native_integrations'] or data['gravity']['optimizer_steps']:
        raise ValueError('Offline audit may not add native/optimizer work')
    if data['transition']['extra_native_integrations'] or data['transition']['extra_optimizer_steps'] or data['transition']['long_training_final_actual_work'] is not None:
        raise ValueError('Embedded training milestones must remain nonadditive/not final')
    log = ROOT / 'runs/ego_gravity_tests423_v2.log'
    text = log.read_text(encoding='utf-8').replace('\r\n', '\n')
    if 'Ran 423 tests in ' not in text or '\nOK\n' not in text:
        raise ValueError('Full423 pass required;retain previous failed fixture log')
    docs = [ROOT / 'docs' / name for name in ['ego-gravity-contract-and-long-arm-transition-2026-11-16.md', 'research-plan-v1.20.md']]
    scripts = [ROOT / 'tools' / name for name in ['archive_ego_gravity_and_arm_transition.py', 'audit_mug_gravity_transport.py', 'review_pen_long_arm_transition.py']]
    paths = sorted((ROOT / 'src').rglob('*.py')) + [ROOT / 'tests/test_ego_gravity_transport.py', ROOT / 'README.md'] + scripts + docs
    analysis.mkdir()
    source = freeze_sources(paths, analysis / 'source_snapshot', ROOT,
                            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(analysis / 'source_receipt.json', source)
    receipts = [p.parent / 'source_receipt.json' for k, p in roots.items() if k != 'transition_inline'] + [analysis / 'source_receipt.json']
    merged = {json.dumps(r['original'], sort_keys=True): r for r in aliases['records']}
    for path in receipts:
        saved = json.loads(path.read_text())
        verify_source_snapshot(saved)
        for record in saved['records']:
            merged.setdefault(json.dumps(record['observed'], sort_keys=True), dict(original=record['observed'],
                archived=record['snapshot'], reason='Exact path/bytes/SHA;conditional ego gravity and nonadditive long-job milestone.'))
    aliases = dict(schema='explicit_source_relocations:1', records=list(merged.values()))
    logs = [log, ROOT / 'runs/ego_gravity_tests423_v1.log', ROOT / 'runs/mug_gravity_transport_audit_console_v1.log',
            ROOT / 'runs/pen_long_arm_transition_review_console_v2.log']
    todo = [file_pin(p) for p in list(roots.values()) + receipts + docs + scripts + logs + [prior_path]]
    allowed = {p.parent for p in roots.values()} | {analysis}
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
    verify_source_snapshot(source, live=True)
    out.mkdir()
    shutil.copyfile(log, out / log.name)
    write(out / 'summary.json', dict(roots={k: file_pin(p) for k, p in roots.items()},
        gravity_metrics=data['gravity']['metrics'], physical_status=data['gravity']['physical_status'],
        transition_checks=data['transition']['checks'], actual_native_integrations=0, optimizer_steps=0,
        long_training_final_actual_work=None, embedded_milestone_work_is_nonadditive=True,
        scope='Conditional gravity/kinematic frame audit and CPU saved-checkpoint milestone only. No source gravity calibration,force ground truth,task/ego success or new training work.'))
    write(out / 'source_resolutions.json', aliases)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='ego_gravity_and_long_arm_transition_ledger:1', prior_evidence=file_pin(prior_path),
        roots={k: file_pin(p) for k, p in roots.items()}, summary=file_pin(out / 'summary.json'),
        documents=[file_pin(p) for p in docs], test_log=file_pin(out / log.name),
        source_receipts=[file_pin(p) for p in receipts], source_resolutions=file_pin(out / 'source_resolutions.json'),
        verified_manifest=file_pin(analysis / 'verified_pins.json'), verified_pins=len(verified),
        actual_native_integrations=0, optimizer_steps=0, long_training_final_actual_work=None,
        embedded_milestone_work_is_nonadditive=True))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), verified_pins=len(verified))), flush=True)


if __name__ == '__main__':
    main()
