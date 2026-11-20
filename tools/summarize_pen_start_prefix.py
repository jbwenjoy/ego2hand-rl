"""Diagnose matched control endpoints and stopping gates from saved W1 traces."""
import argparse
import json
from pathlib import Path
import tomllib

import numpy as np

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.pen_task_contracts import pose_components
from supermimic_research.strict_json import write_json


PHASES = (1, 8, 9, 16)
TICKS = 8
GEOMETRY = (
    ('object_penetration_m', 'pre_penetration_m'),
    ('self_penetration_m', 'pre_self_penetration_m'),
    ('joint_excursion_rad', 'post_joint_limit_excursion_rad'),
    ('nonhand_normal_N', 'pre_object_nonhand_normal_N'),
)
SCHEMA = 'pen_start_prefix_W1_diagnostics:1'


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def endpoint(trace, phase):
    tick = phase * TICKS - 1
    p, q = trace['post_COM'][tick], trace['post_quaternion_wxyz'][tick]
    qref = trace['quaternion_wxyz'][phase]
    xyz = np.linalg.norm(p - trace['COM_world_m'][phase]) * 1000
    angles = (pose_components(q, qref) if np.isfinite(q).all() and np.linalg.norm(q) > 0 else None)
    return dict(phase=phase, time_s=trace['time_s'][phase], COM_mm=xyz if np.isfinite(xyz) else None,
                full_deg=np.rad2deg(angles['full_angle_rad']) if angles else None,
                shaft_deg=np.rad2deg(angles['shaft_angle_rad']) if angles else None,
                twist_deg=np.rad2deg(angles['reference_axis_twist_rad']) if angles and angles['twist_defined'] else None)


def failure(case, trace, limits, phase, stop):
    check = case['checks'][phase-1]
    reasons = [k for k, flag in case['reasons'].items() if flag and k != 'reference_end']
    failed_checks = [k for k, flag in check['metrics']['checks'].items() if not flag]
    ticks = slice((phase-1)*TICKS, phase*TICKS)
    exceeded = []
    for idx, (name, key) in enumerate(GEOMETRY):
        values = trace[key][ticks]
        if idx < 3:
            np.testing.assert_allclose(values.max(), check['geometry'][idx], rtol=0., atol=1e-10)
        indices = np.flatnonzero(values > limits[name])
        if len(indices):
            i = int(indices[0])
            exceeded.append(dict(name=name, limit=limits[name], interval_peak=values.max(),
                                 first_native_tick_in_control=i+1, first_above_limit=values[i]))
    end = endpoint(trace, phase)
    task_thresholds = []
    for name, field, limit in [('position', 'COM_mm', 50.), ('shaft', 'shaft_deg', 60.)]:
        if name in reasons:
            task_thresholds.append(dict(name=field, value=end[field], limit=limit))
    return dict(phase=phase, stop=stop, native_reasons=reasons, geometry_thresholds_exceeded=exceeded,
                failed_dynamics_checks=failed_checks, task_thresholds=task_thresholds,
                unidentified_packet_failure=not check['ok'] and not (failed_checks or exceeded),
                endpoint=end)


def diagnose(entry, arm, limits):
    case = read(entry['case'])
    total, stop = case['controls'], case['stop']
    if (case['arm'] != arm or case['rep'] != entry['rep'] or total != entry['controls'] or
            len(case['checks']) != total or stop != entry['stop']):
        raise ValueError('Pinned case identity or stop mismatch')
    physical_failure = next((i for i, check in enumerate(case['checks'], 1) if not check['ok']), None)
    if physical_failure != entry['first_failed_physical_phase'] or physical_failure not in (None, total):
        raise ValueError('Unexpected physical failure timing')
    if stop not in ('PHYSICAL_AUDIT_STOP', 'TASK_FAILURE', 'REFERENCE_COMPLETION'):
        raise ValueError(f'Unknown stop reason {stop}')
    if (stop == 'PHYSICAL_AUDIT_STOP') != (physical_failure is not None):
        raise ValueError('Stop reason disagrees with interval audit')
    last_legal = total - int(stop != 'REFERENCE_COMPLETION')
    with np.load(verify_file_pin(case['trace']), allow_pickle=False) as data:
        keys = ['post_COM','post_quaternion_wxyz','COM_world_m','quaternion_wxyz','time_s',
                *(name for _, name in GEOMETRY)]
        trace = {k: data[k] for k in keys}
    if len(trace['post_COM']) != total*TICKS or total >= len(trace['time_s']):
        raise ValueError('Trace length or reference phase mismatch')
    np.testing.assert_allclose(trace['time_s'][:total+1], np.arange(total+1)*.02, rtol=0., atol=1e-8)
    return dict(rep=entry['rep'], case=entry['case'], trace=case['trace'], controls=total,
                strict_success=entry['strict_success'], last_legal_phase=last_legal,
                matched={str(p): endpoint(trace,p) if p <= last_legal else None for p in PHASES},
                last_legal=endpoint(trace,last_legal) if last_legal else None,
                first_failure=failure(case,trace,limits,total,stop) if stop != 'REFERENCE_COMPLETION' else None)


def summarize(report_path, out):
    pin = file_pin(report_path)
    source = read(pin)
    arms = {}
    for name, group in source['arms'].items():
        native = read(group['native_report'])
        arm_intent = read(native['intent'])
        limits = tomllib.loads(verify_file_pin(arm_intent['objective']).read_text('utf-8'))
        arms[name] = [diagnose(case, name, limits) for case in group['cases']]
    counts = {str(p): {name: sum(case['matched'][str(p)] is not None for case in cases)
                       for name, cases in arms.items()} for p in PHASES}
    if out.exists() and (out.suffix.lower() != '.json' or json.loads(out.read_bytes()).get('schema') != SCHEMA):
        raise ValueError('Select a separate diagnostic output file')
    write_json(out, dict(schema=SCHEMA, source=pin, clip=source['clip'], matched_valid_counts=counts,
        semantics='Control k uses the eighth post-step state and reference frame k. A terminal failure control is '
                  'excluded from all valid endpoints. Null matched endpoints were not reached legally. '
                  'twist_deg is unsigned twist about the reference x axis; undefined twist is null.',
        arms=arms))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.report, args.out)
