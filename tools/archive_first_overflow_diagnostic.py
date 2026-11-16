"""Independent terminal diagnostic audit and exact evidence sealing."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
import torch
from archive_pen_source_ccd_evidence import pins, write
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_pin, verify_source_snapshot


def main():
    out = ROOT / 'provenance/pen_first_overflow_diagnostic_v1'
    analysis = ROOT / 'runs/pen_first_overflow_archive_v1'
    if out.exists() or analysis.exists(): raise ValueError('Fresh evidence archive required')
    prior_path = ROOT / 'provenance/pen_matched_recovery_failure_v2/ledger.json'
    prior = json.loads(prior_path.read_text())
    aliases = json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    for row in json.loads(verify_file_pin(prior['verified_manifest']).read_text())['records']:
        if file_pin(verify_source_pin(row['observed'], aliases)) != row['resolved']:
            raise ValueError('Prior evidence resolution changed')
    names = dict(diagnostic='pen_first_overflow_diagnostic_v1/report.json',
        registration='pen_first_overflow_registration_v1/registration.json',
        GPU_qualification='pen_first_overflow_observer_qualification_v1/report.json',
        EPA_helper='pen_epa_horizon_helper_probe_v1/report.json')
    roots = {k: file_pin(ROOT / 'runs' / name) for k, name in names.items()}
    reports = {k: json.loads(verify_file_pin(p).read_text()) for k, p in roots.items()}
    d = reports['diagnostic']; run = Path(roots['diagnostic']['path']).parent
    packet = torch.load(verify_file_pin(d['observer_packet']), map_location='cpu', weights_only=False)
    memory = torch.load(verify_file_pin(d['terminal_evidence']['failed_memory']), map_location='cpu', weights_only=False)
    policy = torch.load(verify_file_pin(d['terminal_evidence']['failed_policy']), map_location='cpu', weights_only=False)
    native_model = torch.load(verify_file_pin(d['model_inputs']), map_location='cpu', weights_only=False)
    parent = torch.load(verify_file_pin(d['parent']), map_location='cpu', weights_only=False)
    pm = parent['native']['native']['model']
    all_model_equal = native_model['scalars'] == pm['scalars'] and native_model['arrays'].keys() == pm['arrays'].keys()
    all_model_equal = all_model_equal and all(a.dtype == pm['arrays'][k].dtype and a.shape == pm['arrays'][k].shape
        and a.tobytes() == pm['arrays'][k].tobytes() for k, a in native_model['arrays'].items())
    old202 = json.loads((ROOT / 'runs/pen_matched_recovery_training_v2/RESIDUAL/collector_health_0202.json').read_text())
    rows = d['rows']
    checks = dict(
        fixed99_collections=[r['iteration'] for r in rows] == list(range(202, 301)),
        all_original_health25=all(r['health']['passed'] == r['health']['total'] == 25 for r in rows),
        health_files_match_report=all(json.loads((run / f"health_{r['iteration']:04d}.json").read_text()) == r for r in rows),
        exact98_updates=sum(r['optimized'] for r in rows) == 98 and not rows[-1]['optimized'],
        independent_native_budget=d['actual_native_integrations'] == 99 * 64 * 32 * 8 == 1622016,
        independent_optimizer_budget=d['optimizer_steps'] == 98 * 32 == 3136,
        no_partial_calls=d['completed_call_accounting_exact'],
        all_step_and_forward_paths_observed=d['diagnostic_native_step_calls'] == 50688 and d['diagnostic_forward_calls'] == 10014,
        no_event_recorded=not bool(packet['raw_mask'].any()) and not d['captured_worlds'] and not d['stopped_on_observed_event'],
        terminal_memory_complete=memory['rollout_complete'] and memory['captured_contact_rows'] == 64,
        final_memory_no_overflow=not bool(memory['memory']['engine_overflow'].any()),
        final_collection_not_optimized=(policy['iteration'], policy['optimizer_steps'], policy['training_transitions']) == (299, 9568, 614400),
        complete_model_inputs_equal_parent=all_model_equal,
        future_stats_differ_at202=rows[0]['health']['mean_raw_reward'] != old202['mean_raw_reward'],
        GPU_qualification10=reports['GPU_qualification']['passed'] == reports['GPU_qualification']['total'] == 10,
        EPA_helper8=reports['EPA_helper']['passed'] == reports['EPA_helper']['total'] == 8,
        helper_actual_collision_queries3=reports['EPA_helper']['isolated_collision_queries'] == 3,
        full_test_log435='Ran 435 tests in 11.516s' in (ROOT / 'runs/pen_first_overflow_tests435_v1.log').read_text()
            and '\nOK\n' in (ROOT / 'runs/pen_first_overflow_tests435_v1.log').read_text())
    if not all(checks.values()): raise ValueError('Observed diagnostic audit differs: ' + repr(checks))
    docs = [ROOT / 'docs' / name for name in ['pen-first-overflow-diagnostic-results-2026-11-16.md', 'research-plan-v1.23.md',
        'pen-first-overflow-diagnostic-protocol-v1.md']]
    scripts = [ROOT / 'tools' / name for name in ['capture_pen_first_overflow.py', 'qualify_first_overflow_observer.py',
        'probe_epa_horizon_regression.py']] + [Path(__file__), ROOT / 'tests/test_first_overflow_observer.py']
    analysis.mkdir()
    receipt = freeze_sources(scripts + docs + [ROOT / 'README.md'], analysis / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(analysis / 'source_receipt.json', receipt)
    receipts = [Path(p['path']).parent / 'source_receipt.json' for p in roots.values()] + [analysis / 'source_receipt.json']
    merged = {json.dumps(v['original'], sort_keys=True): v for v in aliases['records']}
    for path in receipts:
        manifest = json.loads(path.read_text()); verify_source_snapshot(manifest, live=True)
        for record in manifest['records']:
            merged.setdefault(json.dumps(record['observed'], sort_keys=True), dict(original=record['observed'],
                archived=record['snapshot'], reason='Exact first-overflow diagnostic source: original path,bytes,SHA.'))
    aliases = dict(schema='explicit_source_relocations:1', records=list(merged.values()))
    logs = [ROOT / 'runs' / name for name in ['pen_first_overflow_observer_qualification_console_v1.log',
        'pen_first_overflow_diagnostic_console_v1.log', 'pen_epa_horizon_helper_probe_console_v1.log',
        'pen_first_overflow_tests435_v1.log', 'pen_first_overflow_observer_tests4_v1.log']]
    todo = list(roots.values()) + [file_pin(p) for p in receipts + docs + scripts + logs + [prior_path]]
    todo += [file_pin(p) for p in run.glob('health_*.json')]
    allowed = {Path(p['path']).parent for p in roots.values()} | {analysis}
    verified = {}
    while todo:
        pin = todo.pop(); key = json.dumps(pin, sort_keys=True)
        if key in verified: continue
        resolved = verify_source_pin(pin, aliases)
        verified[key] = dict(observed=pin, resolved=file_pin(resolved))
        if resolved.suffix == '.json' and any(resolved.is_relative_to(p) for p in allowed):
            todo.extend(pins(json.loads(resolved.read_text())))
    out.mkdir()
    summary = dict(roots=roots, checks=checks, passed=sum(checks.values()), total=len(checks),
        added_actual_work=dict(native_integrations=1622016, optimizer_steps=3136, isolated_collision_queries=3,
                               official_fixture_initializations=1),
        named_lineage_actual_work=dict(native_integrations=38720320, optimizer_steps=75520),
        original202_mean_reward=old202['mean_raw_reward'], replay202_mean_reward=rows[0]['health']['mean_raw_reward'],
        current_live_training=False, pen_incident_reproduced=False, production_capacity_fix_demonstrated=False,
        scope='Observer admission, bounded negative replay and official helper control. Prior costs not charged twice; named lineage is not all research cost. No task success, final comparison, ego or acceleration claim.')
    write(out / 'summary.json', summary); write(out / 'source_resolutions.json', aliases)
    write(analysis / 'verified_pins.json', dict(records=list(verified.values())))
    write(out / 'ledger.json', dict(schema='first_overflow_diagnostic_ledger:1', prior_evidence=file_pin(prior_path),
        roots=roots, summary=file_pin(out / 'summary.json'), documents=[file_pin(p) for p in docs],
        source_receipts=[file_pin(p) for p in receipts], source_resolutions=file_pin(out / 'source_resolutions.json'),
        verified_manifest=file_pin(analysis / 'verified_pins.json'), verified_pins=len(verified),
        actual_native_integrations=1622016, optimizer_steps=3136, isolated_collision_queries=3, current_live_training=False))
    print(json.dumps(dict(ledger=file_pin(out / 'ledger.json'), checks=[sum(checks.values()), len(checks)], verified_pins=len(verified))), flush=True)


if __name__ == '__main__': main()
