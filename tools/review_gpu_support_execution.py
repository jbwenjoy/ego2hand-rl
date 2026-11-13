"""Saved no-reset support-proposal control, cone primal, tracking and EFC audit."""
from pathlib import Path
import argparse
import json
import sys
import tomllib
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts


def main(args):
    if args.output.exists():
        raise ValueError('Fresh execution readback')
    r = json.loads(args.report.read_text())
    for pin in r['sources'] + r['task_sources']:
        verify_file_pin(pin)
    p = json.loads(verify_file_pin(r['proposal']).read_text())
    selected = np.load(verify_file_pin(p['artifacts']['selected']))
    qref = np.load(verify_file_pin(p['clip']))['reference_qpos']
    config = tomllib.loads(verify_file_pin(r['base_config']).read_text())
    args.output.mkdir()
    expected = {('SUPPORT_AWARE_CONSTANT_RESIDUAL', dt) for dt in [.005, .0025]}
    efc = review_saved_warp_contacts(args.report, args.output / 'EFC', expected)
    cases = []
    physical_tables = []
    for c in r['cases']:
        z = dict(np.load(verify_file_pin(c['trace'])))
        snap = np.load(verify_file_pin(c['initialization']['snapshot']))
        physical = dict(np.load(verify_file_pin(c['GPU_physical_context'])))
        physical_tables.append(physical)
        m = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        qids, aids = selected['qids'], selected['aids']
        jids = m.actuator_trnid[aids, 0]
        lo = np.maximum(physical['jnt_range'][0, jids, 0], physical['actuator_ctrlrange'][0, aids, 0])
        hi = np.minimum(physical['jnt_range'][0, jids, 1], physical['actuator_ctrlrange'][0, aids, 1])
        delta = selected['delta20'].astype(np.float32)
        memory = delta.copy()
        targets = []
        for phase, raw in zip(z['pre_action_phase'], z['actions20']):
            memory = np.float32(.5) * np.clip(np.float32(.5) * raw, -.6, .6) + np.float32(.5) * memory
            targets.append(np.clip(qref[int(phase)] + memory, lo, hi))
        repeat = round(.02 / c['dt'])
        control_error = float(abs(np.repeat(targets, repeat, axis=0) - z['post_ctrl'][:, aids]).max())
        pre = mujoco.MjData(m)
        mujoco.mj_setState(m, pre, z['initial_native_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        static = mujoco.MjData(m)
        mujoco.mj_setState(m, static, snap['static_native_integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        old = json.loads(verify_file_pin(p['run']).read_text())
        prior = next(x for x in old['cases'] if x['name'] == 'PROJECTED_CONSTANT_RESIDUAL' and x['dt'] == c['dt'])
        source = np.load(verify_file_pin(prior['trace']))
        parent = mujoco.MjData(m)
        mujoco.mj_setState(m, parent, source['initial_native_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
        summary = summarize_trace(z, z, c['dt'], False, config)
        metric_error = max(abs(summary['metrics'][k] - c['summary']['metrics'][k]) for k in summary['metrics'])
        W = snap['wrench_BODY']
        coeff = snap['static_ray_coefficients_N']
        dyn = snap['dynamic_ray_coefficients_N']
        tau = snap['torque_offset'] + snap['loads20'] @ coeff
        checks = dict(
            selected_initial_pose=np.array_equal(z['initial_qpos'], selected['qpos'].astype(np.float32).astype(float)),
            original_source_velocity=np.array_equal(z['initial_qvel'], source['initial_qvel']),
            fixed_nonjoint=np.array_equal(np.delete(z['initial_qpos'], qids), np.delete(selected['source_qpos'], qids)),
            pose_carry=np.array_equal(z['pre_qpos'][1:], z['post_qpos'][:-1]),
            velocity_carry=np.array_equal(z['pre_qvel'][1:], z['post_qvel'][:-1]),
            original_phase=np.array_equal(z['pre_action_phase'], np.arange(1, len(z['actions20']) + 1)),
            original_decimation=len(z['post_qpos']) == repeat * len(z['actions20']),
            raw_action=np.array_equal(z['actions20'], np.tile(2 * delta, (len(z['actions20']), 1))),
            original_EMA_PD=control_error < 2e-7,
            first_target=float(abs(static.ctrl[aids] - selected['ctrl20']).max()) < 2e-7,
            static_probe_same_pose=np.array_equal(snap['static_qpos'], z['initial_qpos']) and not np.any(snap['static_qvel']),
            mocap_warmstart=np.array_equal(pre.mocap_pos, parent.mocap_pos) and np.array_equal(pre.mocap_quat, parent.mocap_quat) and np.array_equal(pre.qacc_warmstart, parent.qacc_warmstart),
            first_terminal=not np.any(z['terminated_truncated'][:-1]) and bool(z['terminated_truncated'][-1, 0]),
            no_force_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
            static_full_wrench_primal=np.min(coeff) >= -1e-12 and np.max(abs(W @ coeff - snap['target_static_BODY'])) < 1e-8,
            static_torque_bounds=np.all(tau >= snap['torque_lower'] - 1e-8) and np.all(tau <= snap['torque_upper'] + 1e-8),
            dynamic_full_wrench_primal=np.min(dyn) >= -1e-12 and np.max(abs(W @ dyn - snap['target_dynamic_BODY'])) < 1e-8,
            all_robot_contact_scope=c['initialization']['support_scope'] == 'all robot collision geoms',
            metrics=metric_error < 1e-12,
            task_gates=summary['gates'] == c['summary']['gates'] and not c['summary']['qualified'])
        cases.append(dict(dt=c['dt'], checks={k: bool(v) for k, v in checks.items()}, control_error_rad=control_error,
                          metric_error=metric_error, actual_duration_s=c['summary']['metrics']['actual_duration_s']))
    tables_equal = all(np.array_equal(physical_tables[0][k], physical_tables[1][k]) for k in physical_tables[0])
    result = dict(schema='GPU_support_pen_execution_review:1', source=file_pin(__file__), run=file_pin(args.report),
        cases=cases, passed=sum(sum(c['checks'].values()) for c in cases), total=sum(len(c['checks']) for c in cases),
        physical_tables_equal=tables_equal, EFC_review=file_pin(args.output / 'EFC/report.json'),
        EFC_checks=[efc['passed'], efc['total']], neural_updates=0, simulation_integration_steps=0,
        scope='Saved initialization, source clock/control/EMA, LP primals, tracking and EFC review of two failed physical trajectories, not new successful manipulation or independent nominal-inertia calibration.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if result['passed'] != result['total'] or not tables_equal:
        raise ValueError('Preserve failed execution checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
