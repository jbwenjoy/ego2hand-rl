"""Fixed-final three-arm saved summary; wait for all three physical audits.

No model/Env import, prediction, rollout, candidate selection or new grading.
Original per-gate outcomes are preserved from independently audited raw data.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('A1L', 'P_ONLY', 'A2')
GATE_NAMES = ('object_position_151', 'directed_shaft_151', 'actual_q_all601',
              'penetration_solver600', 'no_nonhand_object_contacts', 'no_exogenous',
              'all_actual_u_exact', 'actual_target_rate', 'clock', 'load_ratio',
              'load_charge', 'drop', 'capacity')
COUNT = {'real_transitions': 648192,
         'optimizer_steps': {'wm': 4000, 'fail_head': 4000, 'actor': 1800, 'critic': 7200},
         'collect_calls': 2001, 'metric_rows': 2000}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(value):
    path = Path(value['path'])
    if set(value) != {'path', 'sha256'} or not path.is_absolute() or pin(path)['sha256'] != value['sha256']:
        raise ValueError('Missing/changed pinned saved evidence')
    return path.resolve(strict=True)


def same_pin(a, b):
    return a['sha256'] == b['sha256'] and checked(a).samefile(checked(b))


def full_pass(report):
    return (type(report.get('checks')) is dict and bool(report['checks'])
            and all(v is True for v in report['checks'].values())
            and type(report.get('passed')) is int and type(report.get('total')) is int
            and report['passed'] == report['total'] == len(report['checks']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/full151_schema2_three_arm_primary2000_summary_v1')
    a = ap.parse_args(); out = a.output.resolve()
    if out.exists(): raise FileExistsError('Preserve every summary attempt')
    # Qualify all three saved physical reports before creating any summary or plot.
    review_paths = {m: ROOT / f'outputs/full151_v3_{m}_primary2000_policy_saved_review_v1/report.json' for m in METHODS}
    reviews = {m: read(p) for m, p in review_paths.items()}
    for method, review in reviews.items():
        if (review.get('schema') != 'full151105_policy_saved_independent:4'
                or review.get('status') != 'PASS_SAVED_AUDIT_ORIGINAL_QUALITY_PRESERVED'
                or review.get('method') != method or review.get('engineering_qualified') is not True
                or not full_pass(review) or set(review['recomputed_gates']) != set(GATE_NAMES)
                or review.get('checkpoint_iteration') != 2000 or review.get('training_role') != 'primary_matched'):
            raise ValueError('All3 fixed-final primary physical audits must exist and qualify first')
    out.mkdir(parents=True, exist_ok=False)
    own = pin(__file__); (out / 'summary_source.py').write_bytes(Path(__file__).read_bytes())
    checks = {}; details = {}; configs = {}; curves = {}; table = []; pins = {}
    def need(name, value): checks[name] = bool(value)
    initial_path = ROOT / 'outputs/full151_v3_P_ONLY_primary_smoke_initial_pairing_v1/report.json'
    initial = read(initial_path)
    need('actual_all3_primary_raw_initial_pairing43_bound',
         full_pass(initial) and initial['same_initial_four_parameter_fingerprints'] is True
         and initial['status'] == 'PASS_SAVED_FOUR_CASE_RAW_INITIAL_PAIRING_NOT_TASK_QUALITY')
    pins['raw_initial_pairing'] = pin(initial_path)
    for method in METHODS:
        review = reviews[method]
        hp = ROOT / f'outputs/full151_v3_primary_{method}_final2000_health_v1/report.json'
        health = read(hp); cfg = read(checked(health['config'])); configs[method] = cfg
        run = Path(cfg['output']).resolve()
        evaluation = ROOT / f'outputs/full151_v2_policy_eval_primary2000_{method}_seed0_v1'
        ecfg = read(evaluation / 'config.json')
        result = read(run / 'result.json'); runtime = read(run / 'runtime.json')
        watch = read(run.with_name(run.name + '_watchdog') / 'state.json')
        need(method + '_health_fixed_final2000_primary_and_complete_budget',
             full_pass(health) and health['status'] == 'PASS_SAVED_ENGINEERING_HEALTH'
             and health['method'] == method and health['seed'] == 0 and health['role'] == 'primary_matched'
             and health['iteration'] == 2000 and health['final_iteration_scope'] is True
             and health['budget_counts'] == COUNT and health['source_pins'] == cfg['source_pins']
             and cfg['schema'] == 'full151_knownrough_native_training_run:2')
        need(method + '_physical_health_checkpoint_same_fixed_final',
             same_pin(review['health'], pin(hp)) and same_pin(review['actual_checkpoint'], health['checkpoint'])
             and same_pin(ecfg['checkpoint'], health['checkpoint'])
             and same_pin(ecfg['training_config'], health['config'])
             and ecfg['method'] == method and ecfg['seed'] == 0 and ecfg['worlds'] == 1
             and ecfg['input_bundle'] == cfg['input_bundle'])
        for value in review['inputs'].values(): checked(value)
        for value in cfg['source_pins'].values(): checked(value)
        cpath = review_paths[method].parent / 'independent_numeric_arrays.npz'
        with np.load(cpath, allow_pickle=False) as z: curves[method] = {k: z[k].copy() for k in z.files}
        c = curves[method]; m = review['recomputed_metrics']; gates = review['recomputed_gates']
        shapes = {'position_error601_m': (601,), 'shaft_error601_deg': (601,),
                  'joint_excess601_rad': (601, 20), 'penetration600_m': (600,),
                  'category_depth600_m': (600, 3), 'target_rate150_rad_s': (150, 20),
                  'load_ratio600': (600, 20), 'applied150_FP32': (150, 20)}
        need(method + '_all_fixed_full_horizon_curve_shapes_finite',
             all(c[k].shape == v and np.isfinite(c[k]).all() for k, v in shapes.items()))
        derived = {'max_object_position_error_151_m': c['position_error601_m'][::4].max(),
                   'max_object_position_error_601_m': c['position_error601_m'].max(),
                   'max_shaft_angle_151_deg': c['shaft_error601_deg'][::4].max(),
                   'max_shaft_angle_601_deg': c['shaft_error601_deg'].max(),
                   'max_actual_joint_excess_rad': c['joint_excess601_rad'].max(),
                   'max_penetration_m': c['penetration600_m'].max(),
                   'max_actual_target_rate_rad_s': c['target_rate150_rad_s'].max(),
                   'max_actuator_load_ratio': c['load_ratio600'].max()}
        need(method + '_published_metric_peaks_match_saved_independent_arrays',
             all(np.isclose(m[k], value, rtol=1e-12, atol=1e-14) for k, value in derived.items()))
        need(method + '_original13_gates_preserved_no_total_score',
             all(type(gates[k]) is bool for k in GATE_NAMES)
             and review['failures'] == [k for k in gates if not gates[k]]
             and ecfg['evaluation_gates'] == cfg['evaluation_gates'])
        row = {'method': method, 'training_seed': 0, 'source_lineage_N': 1, 'independent_ego_N': 0,
               'checkpoint_iteration': 2000, 'all_original13_gates_pass': all(gates.values()),
               **m, **{'gate__' + k: gates[k] for k in GATE_NAMES},
               'failure_names': ';'.join(review['failures']),
               'real_training_transitions': 648192, 'WM_optimizer_steps': 4000, 'failure_optimizer_steps': 4000,
               'actor_optimizer_steps': 1800, 'critic_optimizer_steps': 7200,
               'worker_wall_seconds': runtime['wall_seconds'], 'watchdog_wall_seconds': watch['elapsed_s'],
               'native_training_seconds_excluding_imports': result['native_call_wall_seconds_excluding_imports']}
        table.append(row)
        details[method] = {'health': pin(hp), 'physical_saved_review': pin(review_paths[method]),
                           'numeric_curves': pin(cpath), 'final_checkpoint': health['checkpoint'],
                           'original_gates': gates, 'original_metrics': m, 'failure_names': review['failures'],
                           'task_pass': all(gates.values()), 'training_cost': {k: v for k, v in row.items() if 'seconds' in k},
                           'cost_scope': 'Nested worker/watch/native timing; not summed. Fixed original real transitions/updates, no extra teacher rollout in this summary.'}
        for key, path in {'health': hp, 'review': review_paths[method], 'curves': cpath,
                          'training_config': checked(health['config']), 'result': run / 'result.json',
                          'eval_config': evaluation / 'config.json'}.items(): pins[method + '__' + key] = pin(path)
    first = configs['A1L']
    common = ('schema', 'role', 'seed', 'input_bundle', 'training', 'source_pins', 'protocol',
              'controller', 'evaluation_gates', 'hard_process_seconds', 'soft_worker_seconds', 'readiness')
    need('same_protocol_source498_controller_input_and_training_budget',
         len(first['source_pins']) == 498 and all(configs[m][k] == first[k] for m in METHODS for k in common))
    need('same_four_module_primary_raw_fingerprints',
         initial['raw_initial_fingerprints']['P_ONLY_primary']
         == initial['raw_initial_fingerprints']['A1L_primary']
         == initial['raw_initial_fingerprints']['A2_primary'])
    if not all(checks.values()):
        (out / 'failure.json').write_text(json.dumps({'checks': checks, 'failed': [k for k, v in checks.items() if not v]}, indent=2) + '\n', encoding='utf-8')
        raise ValueError('Saved summary join/metric scope failed, preserve output')

    with (out / 'methods.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    with (out / 'original13_gates.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'gate', 'pass']); writer.writeheader()
        writer.writerows({'method': m, 'gate': k, 'pass': details[m]['original_gates'][k]} for m in METHODS for k in GATE_NAMES)

    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {'A1L': '#2166ac', 'P_ONLY': '#e08214', 'A2': '#1b7837'}
    panels = [
        ('Object COM error (mm)', 'position_error601_m', 1000., .005, 20.),
        ('Directed shaft error (deg)', 'shaft_error601_deg', 1., .005, 20.),
        ('Actual joint excess, max20 (mrad)', 'joint_excess601_rad', 1000., .005, 1.),
        ('Generated penetration, any category (mm)', 'penetration600_m', 1000., .005, 3.),
        ('Executed target update rate, max20 (rad/s)', 'target_rate150_rad_s', 1., .02, 15.),
        ('Solver actuator load ratio, max20', 'load_ratio600', 1., .005, 1.01)]
    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
    peak_values = {}
    for ax, (title, field, scale, step, threshold) in zip(axes, panels):
        peak_values[field] = {}
        for method in METHODS:
            y = curves[method][field] * scale
            if y.ndim == 2: y = y.max(axis=1)
            # Post-integration solver samples are labelled by their interval end;
            # target-rate table includes startup->learnedU0 at t0.
            t = np.arange(len(y)) * step
            if len(y) == 600: t = t + step
            ax.plot(t, y, color=colors[method], linewidth=1.5)
            i = int(np.argmax(y)); ax.scatter(t[i], y[i], color=colors[method], s=20, zorder=4)
            peak_values[field][method] = {'peak': float(y[i]), 'canonical_time_s': float(t[i])}
        ax.axhline(threshold, color='#a50026', linestyle='--', linewidth=1.)
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylim(bottom=0); ax.margins(y=.09); ax.grid(alpha=.22)
    axes[-1].set_xlabel('Canonical episode time (s)'); axes[-1].set_xlim(0, 3)
    fig.suptitle('Fixed final2000, full151 known-rough reference\nTraining seed0: Nseed=1 per method, Nsource=1, Nego=0; one evaluation per model', fontsize=12, y=.995)
    handles = [Line2D([0], [0], color=colors[m], label=m, linewidth=2) for m in METHODS]
    handles += [Line2D([0], [0], color='#a50026', linestyle='--', label='Original gate')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .96), ncol=4, frameon=False)
    fig.subplots_adjust(top=.915, bottom=.055, left=.09, right=.98, hspace=.36)
    fig.savefig(out / 'tracking_original_gates.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / 'tracking_original_gates.svg', bbox_inches='tight'); plt.close(fig)
    report = {'schema': 'full151_schema2_three_arm_primary2000_summary:1',
              'status': 'COMPLETE_DESCRIPTIVE_FIXED_FINAL_MATCHED_SINGLE_SEED_SUMMARY',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'source': own, 'input_pins': pins, 'methods': details, 'plot_peak_values': peak_values,
              'N_training_seed_per_method': 1, 'N_source_lineages': 1, 'N_independent_ego_sources': 0,
              'evaluation_processes_per_model': 1, 'fixed_final_iteration': 2000,
              'no_quality_checkpoint_selection': True, 'no_weighted_or_composite_score': True,
              'training_run_order': ['A2', 'A1L', 'P_ONLY'],
              'limits': ['Descriptive single-seed, single-source controlled experiment; no CI, statistical method-benefit, real-ego or independent held-out trajectory claim.',
                         'Original13 task criteria are preserved individually. An all-gates task boolean is the original conjunction, not a new total score.',
                         'Primary rawinitial equality, common source/controller/bundle and equal real-transition/update budget are bound to saved audits. Wall costs can differ and nested timing scopes are not added.',
                         'Reference geometry/physical failures and exploratory prior/compiled-gradient limitations remain; one-step CPU AD and cone witnesses do not qualify physical control gradients.',
                         'Poststate tracking and lastsolver contact/force phases are distinct; plotted600solver values use interval-end timestamps only for display. Primary tracking acceptance uses151 knots;601curves also reveal inter-knot behavior.',
                         'No old schema1 cohort, smoke outcome, best repeat/dose or old checkpoint is pooled. No model/Env/WM/Actor/optimizer/physics/GPU is constructed by this summary.'],
              'artifacts': {k: pin(out / k) for k in ('methods.csv', 'original13_gates.csv', 'tracking_original_gates.png', 'tracking_original_gates.svg')},
              'visual_QA': 'PENDING_ACTUAL_VIEW_AFTER_FIRST_CPU_EXECUTION'}
    for value in pins.values(): checked(value)
    if pin(__file__) != own: raise ValueError('Summary source changed during execution')
    (out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'comparison': pin(out / 'comparison.json'), 'checks': [report['passed'], report['total']], 'all13_task_pass': {m: details[m]['task_pass'] for m in METHODS}}, indent=2))


if __name__ == '__main__':
    main()
