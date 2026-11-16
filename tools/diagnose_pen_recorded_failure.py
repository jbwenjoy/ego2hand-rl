"""Offline pose/contact/action timelines from a retained native execution."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from supermimic_research.runtime.pen_executed_control_audit import audit_executed_control


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def first_event(mask, times):
    indices = np.flatnonzero(mask)
    return None if not len(indices) else dict(index=int(indices[0]), time_s=float(times[indices[0]]))


def main(a):
    case_path = a.case_report.resolve()
    case = json.loads(case_path.read_text())
    source_path = a.source_receipt.resolve()
    producer_source = json.loads(source_path.read_text())
    verify_source_snapshot(producer_source)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh diagnostic output required')
    inputs = {k: case[k] for k in ['checkpoint', 'model', 'trace', 'raw_contacts', 'control_windows']}
    arrays = {}
    for key in ['trace', 'raw_contacts', 'control_windows']:
        with np.load(verify_file_pin(inputs[key])) as saved:
            arrays[key] = dict(saved)
    verify_file_pin(inputs['checkpoint'])
    z, raw, windows = (arrays[k] for k in ['trace', 'raw_contacts', 'control_windows'])
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    n = len(z['post_qpos'])
    controls = len(z['sampled_mean_raw'])
    dt = case['dt']
    if dt != .0025 or controls * 8 != n or n != case['native_integrations']:
        raise ValueError('Recorded native/control clock differs')
    if not np.array_equal(z['pre_action_phase'], np.arange(1, controls + 1)):
        raise ValueError('Single uninterrupted reference clock required')
    if not np.all(windows['counts'][:, 0] == 8) or np.any(windows['errors']) or np.any(windows['global_error']):
        raise ValueError('Retain incomplete/error windows instead of diagnosing as valid')
    post_t = (np.arange(n) + 1) * dt
    contact_t = np.arange(n) * dt
    mid_t = (np.arange(controls) + .5) * .02
    ref = np.stack([np.interp(post_t, z['time_s'], z['COM_world_m'][:, j]) for j in range(3)], -1)
    error = np.linalg.norm(z['post_COM'] - ref, axis=-1)
    ref_R = Slerp(z['time_s'], Rotation.from_quat(z['quaternion_wxyz'][:, [1, 2, 3, 0]]))(post_t)
    full_angle = (Rotation.from_quat(z['post_quaternion_wxyz'][:, [1, 2, 3, 0]]) * ref_R.inv()).magnitude()
    object_body = model.body('object/cube').id
    hand = np.zeros((n, 6))
    nonhand = np.zeros(n)
    penetration = np.zeros(n)
    for i, tick in enumerate(raw['tick']):
        if tick < 0 or tick >= n:
            raise ValueError('Raw contact tick outside recorded trace')
        g, h = raw['geom'][i]
        on = model.geom_bodyid[[g, h]] == object_body
        if on.sum() != 1:
            continue
        other = h if on[0] else g
        group = normal_group(model.geom(int(other)).name or '')
        load = max(0., float(raw['local_force6'][i, 0]))
        penetration[tick] = max(penetration[tick], -float(raw['dist'][i]))
        if group >= 0:
            hand[tick, group] += load
        else:
            nonhand[tick] += load
    force_world = np.einsum('nij,nj->ni', windows['source_R'], windows['objective_measured_source'][:, :3])
    target_world = windows['objective_target_world'][:, :3]
    grouped = hand.reshape(controls, 8, 6).mean(1)
    stored_grouped = windows['Fn6_sum'] / windows['counts'][:, :1]
    source_target = np.concatenate([
        np.einsum('nji,nj->ni', windows['source_R'], windows['objective_target_world'][:, :3]),
        np.einsum('nji,nj->ni', windows['source_R'], windows['objective_target_world'][:, 3:])], -1)
    checks = dict(
        native_count=n == controls * 8,
        raw_hand_matches_trace=bool(np.allclose(hand.sum(1), z['pre_hand_normal_N'], rtol=0, atol=1e-10)),
        raw_nonhand_matches_trace=bool(np.allclose(nonhand, z['pre_object_nonhand_normal_N'], rtol=0, atol=1e-10)),
        raw_penetration_matches_trace=bool(np.allclose(penetration, z['pre_penetration_m'], rtol=0, atol=1e-12)),
        raw_groups_match_resident=bool(np.allclose(grouped, stored_grouped, rtol=0, atol=1e-10)),
        measured_source_mean=bool(np.allclose(windows['objective_measured_source'], windows['wrench_sum'][:, 5] / 8, rtol=0, atol=1e-10)),
        source_target_rotation=bool(np.allclose(source_target, windows['objective_target_source'], rtol=0, atol=1e-10)),
        no_metric_error=not bool(windows['errors'].any() or windows['global_error'].any()),
        no_aid=not bool(z['post_qfrc_applied'].any() or z['post_xfrc_applied'].any()),
        finite=all(bool(np.isfinite(v).all()) for group in arrays.values() for v in group.values()))
    gates_path = ROOT / 'configs/controlled_evaluation_v4.toml'
    import tomllib
    gates = tomllib.loads(gates_path.read_text())
    reference = {k: z[k] for k in ['time_s', 'COM_world_m', 'quaternion_wxyz']}
    recalculated = summarize_trace({k: v for k, v in z.items() if k.startswith(('post_', 'pre_')) and k != 'pre_action_phase'} | {'overflow': z['overflow']},
                                   reference, dt, False, gates)
    checks['saved_metrics_reproduced'] = all(abs(recalculated['metrics'][k] - v) <= 1e-12 for k, v in case['summary']['metrics'].items())
    control_audit = audit_executed_control(z, case) if 'post_ctrl' in z else None
    if control_audit is not None:
        checks.update({'executed_' + k: v for k, v in control_audit['checks'].items()})
    if not all(checks.values()):
        raise ValueError('Recorded arithmetic mismatch: ' + repr(checks))
    # These are descriptive intervals chosen to expose the recorded failure,
    # not held-out tests, causes, optimizer inputs or policy selection criteria.
    bounds = sorted(set([0.] + [x for x in [1., 1.5, 1.6, 1.7, 1.8] if x < post_t[-1]] + [float(post_t[-1])]))
    rows = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        mask = (post_t > start + 1e-12) & (post_t <= end + 1e-12)
        cmask = (mid_t >= start - 1e-12) & (mid_t < end - 1e-12)
        rows.append(dict(interval_s=[start, end],
            COM_RMS_mm=float(np.sqrt(np.mean(error[mask] ** 2)) * 1000), COM_max_mm=float(error[mask].max() * 1000),
            full_angle_RMS_deg=float(np.sqrt(np.mean(full_angle[mask] ** 2)) * 180 / np.pi),
            mean_hand_normal_N=float(hand[mask].sum(1).mean()), hand_load_over_0p1N_fraction=float((hand[mask].sum(1) > .1).mean()),
            Fn6_control_mean_N=grouped[cmask].mean(0).tolist(),
            measured_force_world_mean_N=force_world[cmask].mean(0).tolist(),
            target_force_world_mean_N=target_world[cmask].mean(0).tolist()))
    action = z['sampled_mean_raw']
    clipped = np.argwhere(np.abs(action) > 1)
    frame_dt = np.diff(z['time_s'])
    frame_distance = np.linalg.norm(np.diff(z['COM_world_m'], axis=0), axis=1)
    frame_angle = (Rotation.from_quat(z['quaternion_wxyz'][1:, [1, 2, 3, 0]]) *
                   Rotation.from_quat(z['quaternion_wxyz'][:-1, [1, 2, 3, 0]]).inv()).magnitude()
    late_start = min(1.5, max(0., float(post_t[-1]) - .36))
    late = (z['time_s'][1:] >= late_start) & (z['time_s'][1:] <= post_t[-1] + .02)
    events = {f'COM_above_{mm}mm': first_event(error > mm / 1000, post_t) for mm in [5, 10, 20, 50]}
    events.update({f'full_angle_above_{deg}deg': first_event(full_angle > deg * np.pi / 180, post_t) for deg in [5, 10]})
    events['hand_load_at_most_0p1N'] = first_event(hand.sum(1) <= .1, contact_t)
    out.mkdir()
    source = freeze_sources([Path(__file__), gates_path, ROOT / 'src/supermimic_research/mechanics/pen_contact_groups.py',
                            ROOT / 'src/supermimic_research/runtime/controlled_evaluation.py',
                            ROOT / 'src/supermimic_research/runtime/pen_executed_control_audit.py'], out / 'source_snapshot', ROOT,
                            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', source)
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    axes[0].plot(post_t, error * 1000)
    for value in [5, 10]:
        axes[0].axhline(value, color='gray', linestyle='--', linewidth=.8, label=f'{value} mm threshold')
    axes[0].set_ylabel('COM error (mm)')
    axes[1].plot(post_t, full_angle * 180 / np.pi)
    for value in [5, 10]:
        axes[1].axhline(value, color='gray', linestyle='--', linewidth=.8)
    axes[1].set_ylabel('Full angle error (deg)')
    for j in range(6):
        axes[2].plot(mid_t, grouped[:, j], label=f'Finger{j + 1}' if j < 5 else 'Other robot', linewidth=.9)
    axes[2].plot(mid_t, grouped.sum(1), color='black', linewidth=1.4, label='Total normal')
    axes[2].set_ylabel('Mean contact normal (N)')
    axes[2].legend(loc='upper left', ncol=4, fontsize=8)
    axes[3].plot(mid_t, force_world[:, 2], label='Measured hand Fz')
    axes[3].plot(mid_t, target_world[:, 2], label='Nominal reference Fz', linestyle='--')
    axes[3].set_ylabel('World vertical force (N)')
    axes[3].legend(loc='upper left', fontsize=8)
    axes[4].plot(mid_t, np.abs(action).max(1), label='Max abs mean raw action')
    axes[4].axhline(1., color='gray', linestyle='--', label='Raw clamp boundary')
    axes[4].set_ylabel('Max abs raw action')
    axes[4].set_xlabel('Nominal physical time (s)')
    axes[4].legend(loc='upper left', fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xlim(0, post_t[-1])
        for mm, color in [(5, '#aa6600'), (10, '#b22222')]:
            if events[f'COM_above_{mm}mm'] is not None:
                axis.axvline(events[f'COM_above_{mm}mm']['time_s'], color=color, linewidth=.8, alpha=.7)
    fig.suptitle(f'{case["arm"]} iteration {case["iteration"]}: one recorded fresh W1 execution\n'
                 'Intermediate diagnostic, not final / ego / causal evidence', fontsize=13)
    fig.savefig(out / 'failure_timeline.png', dpi=170)
    plt.close(fig)
    np.savez_compressed(out / 'timeline_arrays.npz', post_time_s=post_t, contact_pre_time_s=contact_t,
                        control_mid_time_s=mid_t, COM_error_m=error, full_angle_rad=full_angle,
                        raw_rebuilt_Fn6_N=hand, Fn6_control_mean_N=grouped,
                        measured_force_world_N=force_world, target_force_world_N=target_world)
    result = dict(schema='recorded_pen_failure_diagnosis:1', case_report=file_pin(case_path),
        producer_source_receipt=file_pin(source_path), source_receipt=file_pin(out / 'source_receipt.json'), inputs=inputs,
        checks=checks, passed=sum(checks.values()), total=len(checks), events=events, descriptive_intervals=rows,
        raw_action=dict(max_abs=float(np.abs(action).max()), total_components=int(action.size),
                        clipped_components=int(len(clipped)), clipped_locations_control_component=clipped.tolist(),
                        transmitted_PD_targets_recorded=control_audit is not None),
        executed_control_audit=control_audit,
        reference=dict(max_frame_displacement_m=float(frame_distance.max()), max_frame_angle_rad=float(frame_angle.max()),
                       largest_five_rotation_intervals=[dict(interval_s=z['time_s'][i:i + 2].tolist(),
                           angle_rad=float(frame_angle[i]), position_step_m=float(frame_distance[i]))
                           for i in np.argsort(frame_angle)[-5:][::-1]],
                       late_start_s=late_start, late_to_end_plus_one_frame_max_displacement_m=float(frame_distance[late].max()),
                       late_max_frame_angle_rad=float(frame_angle[late].max()), frame_dt_s_range=[float(frame_dt.min()), float(frame_dt.max())]),
        nominal_clock_max_GPU_drift_s=float(np.max(np.abs(windows['clock'][:, 1] - np.arange(1, controls + 1) * .02))),
        figure=file_pin(out / 'failure_timeline.png'), arrays=file_pin(out / 'timeline_arrays.npz'),
        actual_native_integrations=0, optimizer_steps=0, inspected_source_integrations=n,
        timing='Pose is post-integration;instant contact metrics are from that step pre-integration native forward/solve. Window means represent original20ms intervals at midpoint;nominal clock,raw GPU float32 drift retained.',
        scope='Offline observational reconstruction of one retained episode. Failure-time intervals are descriptive,not unseen validation or causal identification. Raw action clipping is distinct from executed PD target limiting;the latter is audited only when native control fields exist. No physics/Actor updates,checkpoint selection or changes to the running trial.')
    verify_source_snapshot(source, live=True)
    for pin in inputs.values():
        verify_file_pin(pin)
    write(out / 'report.json', result)
    print(json.dumps(dict(report=file_pin(out / 'report.json'), checks=[result['passed'], result['total']], events=events,
                         intervals=rows, raw_action=result['raw_action'], reference=result['reference'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-report', type=Path, required=True)
    parser.add_argument('--source-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
