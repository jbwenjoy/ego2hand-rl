"""Independent saved pairing of three actual schema2 full151 smoke40 runs.

No producer, model, environment or objective is constructed. Only the three
raw initial CPU tensor artifacts are torch-loaded; final checkpoint health is
bound to the independently audited fixed-final artifact, not rerun here.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('A1L', 'P_ONLY', 'A2')
MODULES = ('wm', 'fail_head', 'actor', 'critic')
COUNTERS = ('policy_update_calls', 'objective_forward_calls', 'active_rows',
            'inactive_rows', 'dt_applications')
EXPECTED_HEALTH_SHA = '5950b9947b00ed0cfce6a8403a65877db6d71be1d7b01b8c3ad1626bdd5de792'
EXPECTED_COUNTS = {'real_transitions': 20992,
                   'optimizer_steps': {'wm': 80, 'fail_head': 80, 'actor': 20, 'critic': 80},
                   'collect_calls': 41, 'metric_rows': 40}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def checked(value):
    if type(value) is not dict or set(value) != {'path', 'sha256'}:
        raise ValueError('Exact path/SHA pin required')
    path = Path(value['path'])
    actual=pin(path) if path.is_absolute() else None
    if actual is None or actual['sha256'] != value['sha256'] or not Path(actual['path']).samefile(path):
        raise ValueError('Changed/missing pinned artifact: ' + str(path))
    return path.resolve(strict=True)


def same_pin(a,b):
    return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Partial log is not a complete fixed-final smoke: ' + str(path))
    return [json.loads(v) for v in raw.splitlines() if v.strip()]


def passed_report(value):
    report = read(checked(value))
    checks = report.get('checks')
    if (type(checks) is not dict or not checks
            or not all(type(v) is bool and v for v in checks.values())
            or type(report.get('passed')) is not int or type(report.get('total')) is not int
            or report['passed'] != report['total'] or report['total'] != len(checks)):
        raise ValueError('Incomplete health report')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configs', nargs=3, type=Path, required=True)
    ap.add_argument('--health', nargs=3, type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    own = pin(__file__)
    (out / 'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    checks = {}
    detail = {}
    configs = {}; health = {}; config_pins = {}; health_pins = {}
    artifact_pins = {}; initial_fingerprints = {}; initial_parameter_layout = {}
    same_initial = None
    raw_initial_loads = 0
    source_pins = {'pairing_auditor': own}

    def need(name, value):
        checks[name] = bool(value)

    try:
        # Reject absent/old/partial health before importing Torch or reading tensors.
        for path in a.configs:
            cfg = read(path); method = cfg.get('method')
            if method not in METHODS or method in configs:
                raise ValueError('Exactly one config for each new arm is required')
            if cfg.get('schema') != 'full151_knownrough_native_training_run:2':
                raise ValueError('Old schema1 smoke is not a substitute')
            configs[method] = cfg; config_pins[method] = pin(path)
        for path in a.health:
            hp = pin(path); h = passed_report(hp); method = h.get('method')
            if method not in METHODS or method in health:
                raise ValueError('Exactly one health report for each arm is required')
            if (h.get('schema') != 'full151105_checkpoint_CPU_health:1'
                    or h.get('status') != 'PASS_SAVED_ENGINEERING_HEALTH'
                    or h.get('role') != 'engineering_smoke' or h.get('iteration') != 40
                    or h.get('final_iteration_scope') is not True):
                raise ValueError('Require actual schema2 fixed-final40 health, not quality selection')
            health[method] = h; health_pins[method] = hp
        if set(configs) != set(METHODS) or set(health) != set(METHODS):
            raise ValueError('All three actual smoke results must exist first')

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import torch
        torch.set_num_threads(1)
        raw = {}; common_controllers = {}; contexts = {}; resets = {}; priors = {}
        for method in METHODS:
            cfg = configs[method]; h = health[method]; run = Path(cfg['output']).resolve(strict=True)
            need(method + '_actual_schema2_healthy_final40_identity',
                 same_pin(h['config'],config_pins[method]) and Path(h['run_dir']).resolve() == run
                 and h['source_pins'] == cfg['source_pins'] and h['input_bundle'] == cfg['input_bundle']
                 and type(cfg['seed']) is int and cfg['seed'] == h['seed'] == 0
                 and cfg['role'] == 'engineering_smoke' and cfg['GPU_run_authorized'] is True)
            need(method + '_new_expected_health_auditor', checked(h['auditor'])
                 == ROOT / 'scripts/audit_full151_checkpoint_cpu_v2.py'
                 and h['auditor']['sha256'] == EXPECTED_HEALTH_SHA)
            checked(h['checkpoint'])
            need(method + '_fixed_final_artifact_and_budget',
                 Path(h['checkpoint']['path']).resolve() == run / 'train/model_final.pt'
                 and h['budget_counts'] == EXPECTED_COUNTS
                 and cfg['training']['max_iterations'] == 40
                 and cfg['training']['warmup_steps'] == 64
                 and cfg['training']['wm_only_iters'] == 20
                 and cfg['training']['horizon'] == 8 and cfg['training']['pi_batch'] == 128)
            for value in cfg['source_pins'].values():
                checked(value)
            checked(cfg['input_bundle']); checked(cfg['protocol'])
            for name in ('learning_v2.py', 'prior_hook_v1.py', 'controller.py', 'runtime.py',
                         'training_capacity.py'):
                source_pins[name] = pin(ROOT / 'src/full151_native' / name)
            source_pins['binding_v3'] = pin(ROOT / 'src/full151_intent/binding_v3.py')
            source_pins['health_auditor_v2'] = h['auditor']

            initial_path = run / 'initial_parameters.pt'
            raw_initial_loads += 1
            raw[method] = torch.load(initial_path, map_location='cpu', weights_only=False)
            initial = raw[method]; fp = read(run / 'initial_parameters.json')
            hi = h['details']['initial_parameters']
            need(method + '_raw_initial_capture_schema_and_scope',
                 initial.get('schema') == 'full151_native_initial_parameters:1'
                 and initial.get('capture_point') == 'after_native_DexRunner_init_before_first_collect_or_optimizer'
                 and set(initial.get('modules', {})) == set(MODULES)
                 and initial.get('source_pins') and same_pin(hi['raw_initial_tensor_artifact'],pin(initial_path))
                 and hi['capture_point'] == initial['capture_point'])
            for value in initial['source_pins'].values():
                checked(value)
                need(method + '_raw_capture_source_' + Path(value['path']).name,
                     any(value == v for v in cfg['source_pins'].values()))
            recomputed = {}; layout = {}
            for name in MODULES:
                data = initial['modules'][name]; values = data['parameters']; names = data['parameter_names']
                digest = hashlib.sha256()
                valid = (type(names) is list and len(values) == len(names) == len(set(names))
                         and bool(values) and data['shapes'] == [list(v.shape) for v in values]
                         and data['dtypes'] == [str(v.dtype) for v in values])
                for value in values:
                    valid = valid and isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                    valid = valid and bool(torch.isfinite(value).all())
                    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
                    digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
                recomputed[name] = {'sha256': digest.hexdigest(), 'tensors': len(values),
                                    'parameters': sum(v.numel() for v in values)}
                layout[name] = {'parameter_names': names, 'shapes': data['shapes'], 'dtypes': data['dtypes']}
                need(method + '_raw_' + name + '_matches_initial_JSON_and_health',
                     valid and recomputed[name] == fp[name]
                     == hi['independently_recomputed_fingerprints'][name])
            initial_fingerprints[method] = recomputed
            initial_parameter_layout[method] = layout

            marker = read(run / 'training_completion.json'); result = read(run / 'result.json')
            need(method + '_schema2_completion_counts_and_health_delta_binding',
                 marker['schema'] == 'full151_native_worker_training_completion:2'
                 and marker['config_sha256'] == config_pins[method]['sha256']
                 and marker['result'] == result
                 and result['schema'] == 'full151_native_training_completion:2'
                 and result['method'] == method and result['final_iteration'] == 40
                 and result['actual_optimizer_steps'] == EXPECTED_COUNTS['optimizer_steps']
                 and result['sim_steps'] == 20992 and result['phase_calls']['collect'] == 41
                 and result['full151_capacity_qualified_final'] is True
                 and result['cleanup_errors'] == [] and same_pin(result['checkpoint'],h['checkpoint'])
                 and all(hi['independent_final_parameter_delta_l2'][k] > 0
                         and math.isclose(hi['independent_final_parameter_delta_l2'][k],
                                          result['parameter_delta_l2'][k], rel_tol=1e-10, abs_tol=1e-12)
                         for k in MODULES))
            ctl = read(run / 'controller_descriptor.json')
            root = ctl['root_descriptor']
            need(method + '_trained_root_controller_identity',
                 root['trained_method'] == method and root['training_seed'] == 0
                 and root['run_role'] == 'engineering_smoke'
                 and root['config_sha256'] == config_pins[method]['sha256']
                 and root['execution_source_pins'] == cfg['source_pins']
                 and ctl['format_version'] == 105 and ctl['margin_rad'] == .020)
            extras = {'trained_method', 'training_seed', 'run_role', 'config_sha256', 'execution_source_pins'}
            common_controllers[method] = {k: v for k, v in ctl.items()
                                          if k not in ('root_descriptor', 'root_descriptor_sha256',
                                                       'reference_controller_signature', 'latest_actualu_evidence')}
            common_controllers[method]['root_descriptor_common'] = {k: v for k, v in root.items() if k not in extras}
            with np.load(run / 'fixed_physics_context.npz', allow_pickle=False) as z:
                contexts[method] = {k: z[k].copy() for k in z.files}
            resets[method] = read(run / 'runtime_reset_readback.json')

            e = result['full151_prior_evidence']; priors[method] = e
            initial_prior = read(run / 'prior_metadata.json')
            rng = read(run / 'objective_build_rng.json')
            need(method + '_prior_scope_and_metadata_prefix_binding',
                 e['schema'] == 'full151_imagination_prior_checkpoint_evidence:1'
                 and e['method'] == method and e['dt_s'] == .02
                 and e['imagined_soft_required'] is True
                 and e['real_collect_shaped'] is False and e['WMfit_shaped'] is False
                 and e['dead_end_rule'] == 'unchanged native full151 phase<150 hardmask after reward'
                 and e['logging_errors'] == [] and all(type(e[k]) is int and e[k] >= 0 for k in COUNTERS)
                 and all(initial_prior[k] == 0 for k in COUNTERS)
                 and {k: v for k, v in e.items() if k not in COUNTERS}
                 == {k: v for k, v in initial_prior.items() if k not in COUNTERS}
                 and e['build_rng'] == rng)
            log = rows(run / 'full151_prior_evidence.jsonl')
            if method == 'A1L':
                need('A1L_no_objective_spec_import_or_prior_counts',
                     cfg['contact_binding'] is None and e['status'] == 'NONE'
                     and e['spec_pin'] is None and e['objective_metadata'] is None
                     and all(e[k] == 0 for k in COUNTERS) and log == []
                     and rng['status'] == 'NOT_CONSTRUCTED' and rng['objective'] == 'NONE')
            else:
                checked(cfg['contact_binding'])
                rng_names = {'Python_exact', 'NumPy_exact', 'TorchCPU_exact',
                             'CUDA_initialization_unchanged', 'already_initialized_CUDA_exact'}
                need(method + '_CUDA_preinitialized_all5_RNG_checks_and_raw_digest_exact',
                     rng['status'] == 'PRESERVED' and rng['factory_error'] is None
                     and same_pin(rng['spec_pin'],cfg['contact_binding']) and same_pin(e['spec_pin'],cfg['contact_binding'])
                     and set(rng['checks']) == rng_names and all(v is True for v in rng['checks'].values())
                     and rng['before'] == rng['after'] and rng['before']['cuda_initialized'] is True
                     and len(rng['before']['torch_cuda']) > 0
                     and all(type(v) is str and len(v) == 64 for v in rng['before']['torch_cuda']))
                need(method + '_prior_20policy_160H8_160dt_20480rows',
                     e['status'] == 'IMAGINATION_ONLY' and e['policy_update_calls'] == 20
                     and e['objective_forward_calls'] == e['dt_applications'] == 160
                     and e['active_rows'] + e['inactive_rows'] == 20480)
                cumulative = {k: 0 for k in COUNTERS}; valid = len(log) == 20
                for index, row in enumerate(log):
                    valid = valid and row['schema'] == 'full151_imagination_prior_policy_log:1'
                    valid = valid and row['method'] == method and row['complete'] is True
                    valid = valid and row['iteration'] == 20 + index and row['policy_update_call'] == index + 1
                    valid = valid and row['objective_forward_calls'] == row['dt_applications'] == 8
                    valid = valid and all(type(row[k]) is int and row[k] >= 0 for k in COUNTERS[1:])
                    valid = valid and row['active_rows'] + row['inactive_rows'] == 1024
                    cumulative['policy_update_calls'] += 1
                    for k in COUNTERS[1:]: cumulative[k] += row[k]
                    valid = valid and all(row['cumulative'][k] == cumulative[k] for k in COUNTERS)
                    valid = valid and {k: v for k, v in row['cumulative'].items() if k not in COUNTERS} == {k: v for k, v in e.items() if k not in COUNTERS}
                need(method + '_all20_saved_policy_logs_match_final_prefix', valid and all(e[k] == cumulative[k] for k in COUNTERS))
                need(method + '_prior_final_counts_and_log_bytes_bound_to_health',
                     h['details']['prior_checkpoint_prefix']['checkpoint_counters'] == {k: e[k] for k in COUNTERS}
                     and h['details']['prior_checkpoint_prefix']['prefix_policy_calls'] == 20
                     and h['details']['prior_checkpoint_prefix']['prefix_SHA256']
                     == sha(run / 'full151_prior_evidence.jsonl'))
                policy = h['details']['runtime_saved_evidence']['policy']
                mask = np.asarray(policy['active_mask_H_B']); starts = np.asarray(policy['start_phases']).reshape(-1)
                expected_mask = starts[None] + np.arange(8)[:, None] < 150
                need(method + '_latest_H8_phase150_hardmask_example_matches_prior_rows',
                     mask.dtype == np.bool_ and mask.shape == (8, 128) and starts.shape == (128,)
                     and np.array_equal(mask, expected_mask)
                     and log[-1]['active_rows'] == int(mask.sum())
                     and log[-1]['inactive_rows'] == int(mask.size - mask.sum()))
            artifact_pins[method] = {name: pin(run / name) for name in (
                'initial_parameters.pt', 'initial_parameters.json', 'training_completion.json',
                'result.json', 'controller_descriptor.json', 'fixed_physics_context.npz',
                'runtime_reset_readback.json', 'prior_metadata.json', 'objective_build_rng.json')}
            if (run / 'full151_prior_evidence.jsonl').exists():
                artifact_pins[method]['full151_prior_evidence.jsonl'] = pin(run / 'full151_prior_evidence.jsonl')
            detail[method] = {'run_dir': str(run), 'budget_counts': h['budget_counts'],
                              'health_final_parameter_delta_l2': hi['independent_final_parameter_delta_l2'],
                              'prior_counts': {k: e[k] for k in COUNTERS},
                              'phase_coverage': h['details']['runtime_saved_evidence'].get('phase_coverage'),
                              'actual_checkpoint': h['checkpoint']}

        anchor = configs['A1L']
        shared_keys = ('schema', 'role', 'seed', 'input_bundle', 'training', 'source_pins',
                       'protocol', 'hard_process_seconds', 'soft_worker_seconds', 'automatic_retry',
                       'compile_fallback', 'controller', 'evaluation_gates', 'readiness')
        need('shared_config_fields_budget_protocol_source_and_input_exact',
             all(configs[m][k] == anchor[k] for m in METHODS for k in shared_keys))
        need('shared_controller_common_core_exact', all(common_controllers[m] == common_controllers['A1L'] for m in METHODS))
        need('shared_actual8_reader_xi46_dtype_shape_bytes_exact',
             all(set(contexts[m]) == set(contexts['A1L'])
                 and all(v.dtype == contexts[m][k].dtype and v.shape == contexts[m][k].shape
                         and v.tobytes() == contexts[m][k].tobytes() for k, v in contexts['A1L'].items())
                 for m in METHODS))
        need('shared_initial_reset_readbacks_exact', all(resets[m] == resets['A1L'] for m in METHODS))
        need('shared_P_ONLY_A2_spec_and_different_formula_only',
             same_pin(configs['P_ONLY']['contact_binding'],configs['A2']['contact_binding'])
             and priors['P_ONLY']['objective_metadata']['formula'] == '-.1*penetration_prior'
             and priors['P_ONLY']['objective_metadata']['support_coefficient'] == 0.
             and priors['A2']['objective_metadata']['formula'] == 'phase_weight*official_support_match-.1*penetration_prior'
             and priors['A2']['objective_metadata']['scoring']['support_coefficient'] == 1.)

        same_initial = True
        for method in METHODS[1:]:
            for name in MODULES:
                first = raw['A1L']['modules'][name]; other = raw[method]['modules'][name]
                exact = (first['parameter_names'] == other['parameter_names']
                         and first['shapes'] == other['shapes'] and first['dtypes'] == other['dtypes']
                         and len(first['parameters']) == len(other['parameters'])
                         and all(x.shape == y.shape and x.dtype == y.dtype
                                 and x.contiguous().view(torch.uint8).numpy().tobytes()
                                 == y.contiguous().view(torch.uint8).numpy().tobytes()
                                 for x, y in zip(first['parameters'], other['parameters'])))
                need(method + '_vs_A1L_raw_' + name + '_name_order_shape_dtype_bytes_exact', exact)
                same_initial = same_initial and exact
        need('same_initial_four_parameter_fingerprints_all3', same_initial
             and all(initial_fingerprints[m] == initial_fingerprints['A1L'] for m in METHODS))
        need('no_CUDA_native_framework_model_or_environment_import',
             not torch.cuda.is_initialized() and not any(k.split('.')[0] in
             ('warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic') for k in sys.modules))
        need('all_source_and_raw_input_artifact_bytes_unchanged', pin(__file__) == own
             and all(checked(v) for table in artifact_pins.values() for v in table.values())
             and all(checked(v) for v in source_pins.values())
             and all(checked(v) for v in config_pins.values())
             and all(checked(v) for v in health_pins.values()))
    except BaseException as exc:
        need('audit_completed_without_missing_or_malformed_saved_evidence', False)
        detail['failure'] = {'error': repr(exc), 'traceback': traceback.format_exc()}

    ok = bool(checks) and all(checks.values())
    report = {'schema': 'full151_matched_three_arm_smoke_saved_review:1',
              'status': 'PASS_SAVED_THREE_ARM_SMOKE_PAIRING_NOT_TASK_QUALITY' if ok else 'FAIL_OR_INCOMPLETE_THREE_ARM_SMOKE_PAIRING',
              'methods': list(METHODS), 'health': health_pins, 'configs': config_pins,
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'same_initial_four_parameter_fingerprints': same_initial,
              'raw_initial_assessment': 'NOT_ASSESSED' if same_initial is None else ('EXACT_MATCH' if same_initial else 'MISMATCH'),
              'raw_initial_fingerprints': initial_fingerprints,
              'raw_initial_parameter_layout': initial_parameter_layout,
              'source_pins': source_pins, 'artifact_pins': artifact_pins, 'details': detail,
              'shared_protocol': configs.get('A1L', {}).get('protocol'),
              'source_lineage_N': 1, 'independent_ego_N': 0,
              'physical_task_quality': 'NOT_ASSESSED', 'physical_reference_qualified': False,
              'scope': [
                  'Only three raw initial CPU tensor artifacts loaded. Final checkpoint identity and parameter deltas are linked to independently audited health and completion, not independently recomputed from final tensors here.',
                  'Full-prefix prior counts derive from saved logs; latest H8 phase mask is recomputed from health-published checkpoint start phases. No raw whole-prefix imagined tensor or physical Jacobian is claimed.',
                  'No Env, objective, Actor, WM, optimizer, physics, native query, GPU or training construction. No old schema1 substitution, quality selection, target/mask update or altered result.'
              ],
              'execution': {'raw_initial_torch_loads_attempted': raw_initial_loads, 'final_checkpoint_torch_loads': 0,
                            'model_forward': 0, 'Env': 0, 'optimizer': 0, 'physics': 0, 'GPU': 0},
              'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(out / 'report.json'), 'status': report['status'],
                      'passed': report['passed'], 'total': report['total']}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
