"""Offline review of retained invalid rollout;do not decode a bool as raw bits."""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import write_metadata as write


def main(a):
    run = ROOT / 'runs/pen_matched_recovery_training_v2'
    fp = run / 'failure.json'
    failure = json.loads(fp.read_text())
    health_path = run / 'RESIDUAL/collector_health_0252.json'
    health = json.loads(health_path.read_text())
    packet = torch.load(verify_file_pin(failure['evidence']['failed_memory']), map_location='cpu', weights_only=False)
    policy = torch.load(verify_file_pin(failure['evidence']['failed_policy']), map_location='cpu', weights_only=False)
    native = torch.load(verify_file_pin(failure['evidence']['failed_native']), map_location='cpu', weights_only=False)['native']
    m = packet['memory']
    bad_checks = [k for k, v in health['checks'].items() if not v]
    overflow = torch.nonzero(m['engine_overflow'], as_tuple=False).tolist()
    errors = torch.nonzero(m['contact_errors'], as_tuple=False).tolist()
    checks = dict(
        recorded_failure_stage=failure['phase'] == 'RESIDUAL_consistency_252',
        independent_actual_native=failure['actual_native_integrations'] == 52 * 16384 == 851968,
        independent_actual_optimizer=failure['optimizer_steps'] == 51 * 32 == 1632,
        no_partial_call_uncertainty=failure['completed_call_accounting_exact'],
        failed_batch_NOT_updated=(policy['iteration'], policy['optimizer_steps']) == (251, 8032),
        complete_retained_rollout=packet['rollout_complete'] and packet['captured_contact_rows'] == 64,
        exact_failed_checks=bad_checks == ['no_overflow', 'contact_errors_zero'],
        finite_recorded_memory=all(bool(torch.isfinite(v).all()) for v in m.values() if v.is_floating_point()),
        exact_overflow_locations=overflow == [[62, 31], [63, 31]],
        exact_metric_error_location=errors == [[63, 31]] and int(m['contact_errors'][63, 31]) == 128,
        no_global_metric_error=not bool(m['contact_global_error'].any()),
        native_end_flag_cleared=not bool(native['native']['data']['arrays']['overflow'].any()),
        affected_world_terminal_reset=bool(m['terminated'][63, 31]) and int(m['phase_after_reset'][63, 31]) == 1)
    if not all(checks.values()):
        raise ValueError('Observed retained failure differs: ' + repr(checks))
    rows = []
    for row in [61, 62, 63]:
        world = 31
        rows.append(dict(row=row, world=world, source_phase=int(m['phase_source'][row, world]),
            phase_after_reset=int(m['phase_after_reset'][row, world]), terminated=bool(m['terminated'][row, world]),
            native_overflow_boolean=bool(m['engine_overflow'][row, world]), resident_error=int(m['contact_errors'][row, world]),
            geometry_max=m['contact_geometry_max'][row, world].tolist(), counts=m['contact_counts'][row, world].tolist()))
    package = Path(torch.__file__).resolve().parent.parent / 'mujoco_warp/_src'
    engine_paths = [package / name for name in ['types.py', 'collision_convex.py', 'collision_gjk.py', 'collision_gjk_test.py']]
    types = engine_paths[0].read_text()
    gjk = engine_paths[2].read_text()
    horizon = int(re.search(r'^MJ_MAX_EPAHORIZON\s*=\s*(\d+)', types, re.M).group(1))
    if horizon != 24 or 'EPA_HORIZON = 1 << 8' not in types or 'wp.atomic_or(overflow_out, worldid, OverflowType.EPA_HORIZON)' not in gjk:
        raise ValueError('Installed collision-source evidence changed')
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh offline review required')
    out.mkdir()
    receipt = freeze_sources([Path(__file__)], out / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    (out / 'engine_source_snapshots').mkdir()
    engine = []
    for path in engine_paths:
        pin = file_pin(path)
        target = out / 'engine_source_snapshots' / path.name
        shutil.copyfile(path, target)
        copied = file_pin(target)
        if (pin['bytes'], pin['sha256']) != (copied['bytes'], copied['sha256']):
            raise ValueError('Changed engine source during read-only snapshot')
        engine.append(dict(installed=pin, snapshot=copied))
    console = ROOT / 'runs/pen_matched_recovery_training_console_v2.log'
    result = dict(schema='pen_matched_recovery_overflow_review:1', source_receipt=file_pin(out / 'source_receipt.json'),
        failure=file_pin(fp), health=file_pin(health_path), retained_evidence=failure['evidence'],
        checks=checks, passed=sum(checks.values()), total=len(checks), affected_rows=rows,
        installed_engine_sources=engine, installed_EPA_horizon=horizon,
        namespaces=dict(resident_metric_OVERFLOW_bit=128, native_EPA_HORIZON_bit=256,
                        saved_engine_overflow_is_boolean=True, raw_native_bit_at_incident_available=False),
        EPA_warning_count=console.read_text(encoding='utf-8').count("Warning: EPA horizon = 24 isn't large enough."), console=file_pin(console),
        actual_native_integrations=0, optimizer_steps=0,
        inspected_failed_cohort_work=dict(native_integrations=851968, optimizer_steps=1632),
        scope='Offline real failed-rollout evidence. Console and installed source establish EPA horizon warning branch,but resident128 is generic overflow and boolean engine flags do not identify raw native bits/geom pair. End native flags were cleared after reset;do not call the incident resolved or assign original SUPPORT206 this cause.')
    verify_source_snapshot(receipt, live=True)
    write(out / 'report.json', result)
    print(json.dumps(dict(report=file_pin(out / 'report.json'), checks=[result['passed'], result['total']], affected_rows=rows,
                         namespaces=result['namespaces'], EPA_warning_count=result['EPA_warning_count'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
