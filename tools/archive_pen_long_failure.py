"""Archive terminal failure and nonreproducing diagnostic as separate work."""
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
    out = ROOT / 'provenance/pen_long_support206_failure'
    analysis = ROOT / 'runs/pen_long_failure_archive_v1'
    if out.exists() or analysis.exists():
        raise ValueError('Fresh terminal-evidence archive required')
    prior_path = ROOT / 'provenance/ego_gravity_and_long_arm_transition/ledger.json'
    prior = json.loads(prior_path.read_text())
    aliases = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    for record in json.loads(verify_file_pin(prior['verified_manifest']).read_text())['records']:
        if file_pin(verify_source_pin(record['observed'], aliases)) != record['resolved']:
            raise ValueError('Prior source resolution changed')
    run = ROOT / 'runs/pen_long_control_training_v1'
    failure_path = run / 'failure.json'
    failure = json.loads(failure_path.read_text())
    replay_path = ROOT / 'runs/pen_support206_failure_replay_v1/report.json'
    replay = json.loads(replay_path.read_text())
    if (failure['phase'], failure['actual_native_integrations'], failure['optimizer_steps']) != ('SUPPORT_206', 36143104, 70560) or (run / 'report.json').exists():
        raise ValueError('Preserve incomplete failed original trial')
    if (replay['actual_native_integrations'], replay['optimizer_steps'], replay['stopped_on_failed_gate'], len(replay['rows'])) != (98304, 192, False, 6):
        raise ValueError('Retain fixed nonreproducing replay work')
    comparisons = []
    for row in replay['rows']:
        index = row['iteration']
        if row['health']['passed'] != 25 or row['health']['total'] != 25:
            raise ValueError('Observed replay checks differ')
        if index <= 205:
            original = json.loads((run / 'SUPPORT' / f'health_{index:04d}.json').read_text())['collector']
            comparisons.append(dict(iteration=index, original_mean_raw_reward=original['mean_raw_reward'],
                replay_mean_raw_reward=row['health']['mean_raw_reward'], original_terminated=original['terminated'], replay_terminated=row['health']['terminated']))
    docs = [ROOT / 'docs' / name for name in ['pen-long-support206-failure-2026-11-16.md', 'research-plan-v1.21.md', 'pen-support206-replay-protocol-v1.md']]
    scripts = [Path(__file__), ROOT / 'tools/replay_pen_support_failure.py']
    analysis.mkdir()
    source = freeze_sources(scripts + docs + [ROOT / 'README.md'], analysis / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(analysis / 'source_receipt.json', source)
    receipts = [run / 'source_receipt.json', replay_path.parent / 'source_receipt.json', analysis / 'source_receipt.json']
    merged = {json.dumps(v['original'], sort_keys=True): v for v in aliases['records']}
    for path in receipts:
        manifest = json.loads(path.read_text())
        verify_source_snapshot(manifest, live=True)
        for record in manifest['records']:
            merged.setdefault(json.dumps(record['observed'], sort_keys=True), dict(original=record['observed'], archived=record['snapshot'], reason='Exact terminal failure/replay source path,bytes,SHA.'))
    aliases = dict(schema='explicit_source_relocations:1', records=list(merged.values()))
    roots = dict(original_failure=file_pin(failure_path), replay=file_pin(replay_path))
    todo = list(roots.values()) + [file_pin(p) for p in receipts + docs + scripts + [prior_path,
        ROOT / 'runs/pen_long_control_training_console_v1.log', ROOT / 'runs/pen_support206_failure_replay_console_v1.log']]
    allowed = {replay_path.parent, analysis}
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
    out.mkdir()
    write(out / 'summary.json', dict(roots=roots, comparisons=comparisons,
        original_trial_work=dict(actual_native_integrations=36143104, optimizer_steps=70560, completed=False),
        added_diagnostic_work=dict(actual_native_integrations=98304, optimizer_steps=192),
        original_failed_batch_or_flags_available=False, original_failure_cause_proven=False,
        original_job_terminal_exit=1, replay_job_terminal_exit=0, current_live_training=False,
        scope='Original failed work is historical and not charged again in added diagnostic work. No final equal-budget success or source/geometry gate relaxation. Replay does not reproduce original trajectories/failure.'))
    write(out / 'source_resolutions.json', aliases)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='pen_long_support206_failure_ledger:1', prior_evidence=file_pin(prior_path),
        roots=roots, summary=file_pin(out / 'summary.json'), documents=[file_pin(p) for p in docs],
        source_receipts=[file_pin(p) for p in receipts], source_resolutions=file_pin(out / 'source_resolutions.json'),
        verified_manifest=file_pin(analysis / 'verified_pins.json'), verified_pins=len(verified),
        actual_native_integrations=98304, optimizer_steps=192,
        original_trial_final_actual_work=dict(native_integrations=36143104, optimizer_steps=70560, completed=False),
        original_failure_cause_proven=False, current_live_training=False))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), verified_pins=len(verified))), flush=True)


if __name__ == '__main__':
    main()
