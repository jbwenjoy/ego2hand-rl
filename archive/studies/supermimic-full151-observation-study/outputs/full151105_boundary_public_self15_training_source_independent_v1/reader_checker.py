"""One Tiny public15 payload fixture; NO real checkpoint/model/data loads."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT/'scripts/audit_boundary_public_self15_checkpoint_saved_cpu_v1.py'
OUT = ROOT/'outputs/full151105_boundary_public_self15_checkpoint_auditor_preparation_CPU_v1'


def main():
    OUT.mkdir(exist_ok=False)
    for p in (AUDITOR, Path(__file__)):
        ast.parse(p.read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('_boundary_health_auditor_synthetic', AUDITOR)
    a = importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    source_pins = {k: a.pin(v) for k, v in a.OWN.items()}
    for k in ('auditor', 'checker'):
        (OUT/('frozen_'+k+'.py')).write_bytes(Path(source_pins[k]['path']).read_bytes())
    a.write(OUT/'FREEZE.json', {'schema': 'full151105_boundary_checkpoint_fixture_source_freeze:1',
                              'source_pins': source_pins, 'fixture_parameter_count': 2,
                              'actual16_qualification': False, 'real_checkpoint_or_data_reads': 0})
    checks = {};details = {}
    try:
        default = subprocess.run([sys.executable, str(AUDITOR)], text=True, capture_output=True, check=True)
        declaration = json.loads(default.stdout)
        checks['default_metadata_no_Torch_or_checkpoint'] = declaration['status'] == 'METADATA_ONLY_NO_CHECKPOINT_OR_TORCH_READ' and 'torch' not in sys.modules
        import torch
        import numpy as np
        require = a.require
        torch.set_num_threads(4)
        require(not torch.cuda.is_initialized(), 'CPU-only synthetic fixture')
        helper = a.load_helpers()
        def rng():
            return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_CPU': torch.get_rng_state().clone(), 'CUDA_initialized': False}
        before = rng()
        load_attempts = []
        real_load = torch.load
        def deny_load(*args, **kwargs):
            load_attempts.append(True)
            raise AssertionError('This fixture cannot load any checkpoint')
        torch.load = deny_load
        try:
            # These are tiny anonymous synthetic tensor dictionaries, not Models.
            specs = (('wm.net.0.weight', (2, 2)), ('aux_head.weight', (2, 2)))
            names = [name for name, _ in specs]
            wm = {'net.0.weight': torch.arange(4, dtype=torch.float32).reshape(2, 2),
                  'norm.mean': torch.zeros(298), 'norm.var': torch.ones(298), 'norm.count': torch.tensor(42., dtype=torch.float64),
                  'delta_norm.mean': torch.zeros(52), 'delta_norm.var': torch.ones(52), 'delta_norm.count': torch.tensor(99., dtype=torch.float64),
                  'error_floor': torch.full((52,), .1)}
            host = {'norm': {'eps': .01, 'max_count': 2e8, '_host_count': 42.},
                    'delta_norm': {'eps': .0001, 'max_count': 2e8, '_host_count': 99.}}
            ctx = {'mass_actual_FP32': torch.tensor(1.), 'I_body_F64': torch.eye(3, dtype=torch.float64),
                   'gravity_palm_F64': torch.zeros(3, dtype=torch.float64), 'DeltaT_F64': torch.tensor(.02, dtype=torch.float64),
                   'palm_rot_FP32': torch.eye(3), 'effective_s52_FP32': torch.ones(52), 'physical_xi46_FP32': torch.arange(46, dtype=torch.float32),
                   'I_body_antisymmetry_maxabs_F64': torch.tensor(0., dtype=torch.float64)}
            fixture_dir = OUT/'fixture';fixture_dir.mkdir()
            for name in ('protocol.json', 'recipe.json', 'parent.txt'):
                (fixture_dir/name).write_text('{}\n', encoding='utf-8')
            pp, rp = a.pin(fixture_dir/'protocol.json'), a.pin(fixture_dir/'recipe.json')
            inputs = {'initial_component': a.pin(fixture_dir/'parent.txt')}
            sources = {'synthetic_auditor': a.pin(AUDITOR)}
            construction = {'raw_WM_unchanged': True, 'disjoint_model_head_buffer_storage': True,
                            'constructor_Python_TorchCPU_RNG_unchanged': True, 'feature_hooks_independent': True,
                            'raw_clone_state_identity': dict.fromkeys(a.ARMS, True)}
            cached15 = torch.zeros((152, 15), dtype=torch.float32);cached15[0, 0] = 1.;cached15[1, 10] = 1.
            identities = []
            for bid in range(152):
                s, off = divmod(bid, 19);pi, si = divmod(s, 4);k = (0, 49, 99, 149)[si];group = 226*pi+150+19*si+off
                identities.append({'row_id': bid, 'snapshot_id': s, 'path': ('A0', 'A2')[pi], 'mode': 'baseline' if off < 3 else 'direction',
                                   'k': k, 'branch_id': bid, 'group_id': group, 'physical_rows': [4*group+j for j in range(4)], 'terminal_observation': k+1})
            initial = {'schema': a.INITIAL_SCHEMA, 'pair_schema': a.PAIR_SCHEMA, 'protocol': pp, 'recipe': rp,
                       'input_pins': inputs, 'source_pins': sources, 'source_parent': inputs['initial_component'],
                       'WM_state': wm, 'aux_head_state': {'weight': torch.zeros(2, 2)}, 'physical_context_state': ctx,
                       'normalizer_host_configuration': host, 'ordered_parameter_names': names,
                       'RNG_this_run': rng(), 'construction_receipt': construction,
                       'training_counts': {arm: dict.fromkeys(a.COUNTS, 0) for arm in a.ARMS},
                       'fit_input_digest': '1'*64, 'fit_label_digest': '2'*64,
                       'cached_self15_target_FP32': cached15, 'source_identity_fields': identities,
                       'fit_self15_label_digest': helper.tensor_digest({'cached_self15_target_FP32': cached15}, torch),
                       'fit_combined_label_digest': '4'*64}
            def loss_snapshot(arm):
                delta = ((torch.arange(152*52, dtype=torch.float32).reshape(152, 52)%17)+1)/100
                scale = ctx['effective_s52_FP32'];squared = (delta/scale).square();point_row = squared.mean(-1)
                full67 = torch.zeros((152, 67), dtype=torch.float32)
                full67[:, 52:] = (torch.arange(15, dtype=torch.float32)[None]-7)/2
                ell_native_slice = full67[:, 52:]  # Same (67,1) layout as native.
                bce = torch.nn.functional.binary_cross_entropy_with_logits(ell_native_slice, cached15, reduction='none')
                self_row = bce.sum(-1);pair_row = point_row+.1*self_row;mode_elements = torch.full((152, 2), .5)
                scalars = {'state': point_row.mean(), 'self15': self_row.mean(), 'native_pair': pair_row.mean(), 'mode': mode_elements.mean(),
                           'impulse': torch.tensor(2., dtype=torch.float64), 'consistency': torch.tensor(.25, dtype=torch.float64)}
                scalars['public'] = (scalars['native_pair']+scalars['mode']).double()+scalars['impulse']
                scalars['total'] = scalars['public']+a.FIXED_RECIPE['lambda_cons_by_arm'][arm]*scalars['consistency']
                return {'iteration': 100, 'tag': 'PRE_OPTIMIZER_I', 'weights_after_updates': 99,
                        'native_D52_FP32': delta, 'effective_s52_FP32': scale.clone(), 'point_squared52_FP32': squared,
                        'point_per_row_FP32': point_row, 'self15_logits_FP32': ell_native_slice.clone(), 'self15_target_FP32': cached15.clone(),
                        'self15_BCE_elements_FP32': bce, 'self15_sum_per_row_FP32': self_row, 'native_pair_per_row_FP32': pair_row,
                        'mode_BCE_elements_FP32': mode_elements, 'scalars': scalars}
            loss_snapshots = {arm: loss_snapshot(arm) for arm in a.ARMS}
            payloads = {};logs = []
            for i in range(1, 101):
                row = {'iteration': i, 'arms': {}}
                for arm in a.ARMS:
                    row['arms'][arm] = {'counts': dict.fromkeys(a.COUNTS, i), 'update_seconds': .001,
                        'losses': {key: float(value) for key, value in loss_snapshots[arm]['scalars'].items()}}
                logs.append(row)
            log_bytes = b''.join((json.dumps(r)+'\n').encode() for r in logs)
            log_path = fixture_dir/'updates.jsonl';log_path.write_bytes(log_bytes)
            prefix = {'path': str(log_path), 'records': 100, 'bytes': len(log_bytes), 'sha256': hashlib.sha256(log_bytes).hexdigest()}
            for arm in a.ARMS:
                pre = {names[0]: torch.full((2, 2), 3.), names[1]: torch.full((2, 2), 4.)}
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(pre[n], 2) for n in names]), 2)
                coef = torch.clamp(1./(norm+1e-6), max=1.)
                post = {k: v*coef for k, v in pre.items()}
                post_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(post[n], 2) for n in names]), 2)
                gradient = {'iteration': 100, 'tag': 'PRE_OPTIMIZER_I', 'weights_after_updates': 99, 'parameter_names': names,
                            'pre_clip': pre, 'post_clip': post, 'max_norm': 1., 'pre_clip_returned_norm_FP32': norm,
                            'post_clip_total_norm_FP32': post_norm}
                optimizer = {'param_groups': [{'params': [0, 1], 'lr': 1e-4, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': 0., 'foreach': False}],
                             'state': {i: {'step': torch.tensor(100.), 'exp_avg': torch.zeros(2, 2), 'exp_avg_sq': torch.ones(2, 2)} for i in range(2)}}
                payloads[arm] = {'schema': a.SCHEMA, 'pair_schema': a.PAIR_SCHEMA, 'arm': arm, 'iteration': 100,
                                'checkpoint_tag': 'AFTER_OPTIMIZER_I', 'protocol': pp, 'recipe': rp, 'input_pins': inputs, 'source_pins': sources,
                                'WM_state': copy.deepcopy(wm), 'aux_head_state': {'weight': torch.full((2, 2), .25)},
                                'physical_context_state': copy.deepcopy(ctx), 'normalizer_host_configuration': copy.deepcopy(host),
                                'ordered_parameter_names': list(names), 'optimizer_state': optimizer, 'counts': dict.fromkeys(a.COUNTS, 100),
                                'RNG': copy.deepcopy(initial['RNG_this_run']), 'gradient_snapshot': gradient, 'loss_snapshot': loss_snapshots[arm], 'log_prefix': dict(prefix),
                                **{key: initial[key] for key in ('fit_input_digest', 'fit_label_digest', 'fit_self15_label_digest', 'fit_combined_label_digest')}}
            def evaluate(p=payloads, it=initial, ls=logs):
                protocol = {'inputs': inputs, 'source_pins': sources}
                protocol_inputs, protocol_sources = a.protocol_maps(protocol)
                return a.validate_payload_pair(it, p, ls, 100, pp, rp, protocol_inputs, protocol_sources, torch, helper, fixture_specs=specs)
            result, evidence = evaluate()
            checks['one_Tiny_payload_pair_positive_raw_health'] = all(result.values()) and evidence['fixture_only'] is True and evidence['actual16_parameter_contract_verified'] is False and evidence['parameter_count'] == 2
            def rejects(call):
                try:call()
                except (ValueError, KeyError):return True
                return False
            checks['Tiny_cannot_qualify_actual16_namespace'] = rejects(lambda: a.validate_payload_pair(initial, payloads, logs, 100, pp, rp, inputs, sources, torch, helper))
            bad_recipe = copy.deepcopy(a.FIXED_RECIPE);bad_recipe['lambda_mode'] = True
            checks['Root_canonical_recipe_numeric_and_types'] = a.recipe_matches(a.read(a.RECIPE_FILE)['recipe']) and not a.recipe_matches(bad_recipe)
            old_schema = copy.deepcopy(payloads);old_schema[a.ARMS[0]]['schema'] = 'full151105_boundary_impulse_WM_component:1'
            checks['old_component_namespace_cannot_be_relabelled_public15_health'] = rejects(lambda: evaluate(old_schema))
            checks['protocol_inputs_not_artifact_input_pins_alias'] = rejects(lambda: a.protocol_maps({'input_pins': inputs, 'source_pins': sources}))
            mutations = {}
            for key in ('Adam_step', 'moment_shape', 'context', 'RNG_raw', 'gradient_clip', 'gradient_phase', 'fit_digest', 'counter',
                        'self15_mean_instead_of_sum', 'hoisted_scalar_pair_rows', 'BCE_outside_fixed_envelope', 'cached_label_bits'):
                bad = copy.deepcopy(payloads);p = bad[a.ARMS[0]]
                if key == 'Adam_step':p['optimizer_state']['state'][0]['step'].fill_(99.)
                elif key == 'moment_shape':p['optimizer_state']['state'][0]['exp_avg'] = torch.zeros(1)
                elif key == 'context':p['physical_context_state']['physical_xi46_FP32'][0] += 1
                elif key == 'RNG_raw':p['RNG']['torch_CPU'][0] ^= 1
                elif key == 'gradient_clip':p['gradient_snapshot']['post_clip'][names[0]][0, 0] += 1
                elif key == 'gradient_phase':p['gradient_snapshot']['tag'] = 'AFTER_OPTIMIZER_I'
                elif key == 'fit_digest':p['fit_label_digest'] = '3'*64
                elif key == 'counter':p['counts']['optimizer_steps'] = 99
                elif key == 'self15_mean_instead_of_sum':p['loss_snapshot']['self15_sum_per_row_FP32'] /= 15
                elif key == 'hoisted_scalar_pair_rows':p['loss_snapshot']['native_pair_per_row_FP32'].fill_(float(p['loss_snapshot']['scalars']['native_pair']))
                elif key == 'BCE_outside_fixed_envelope':p['loss_snapshot']['self15_BCE_elements_FP32'][0, 0] += .1
                else:p['loss_snapshot']['self15_target_FP32'][0, 0] = 0
                mutations[key] = rejects(lambda bad=bad: evaluate(bad))
            bad_logs = copy.deepcopy(logs);bad_logs[31]['arms'][a.ARMS[0]]['losses']['total'] += 1
            mutations['public_weight_confounded_log'] = rejects(lambda: evaluate(ls=bad_logs))
            checks['focused_new_payload_mutations_rejected'] = all(mutations.values())
            after = rng();checks['raw_RNG_unchanged_without_repair_and_zero_real_loads'] = helper.raw_rng_comparison(before, after, torch)['all_raw_values_equal'] and not load_attempts and not torch.cuda.is_initialized()
            # Persist the ONE synthetic fixture after the source freeze. Never
            # load it here; a real auditor rejects its two-parameter contract.
            torch.save({'scope': 'SYNTHETIC_TWO_PARAMETER_ABI_ONLY_NOT_ACTUAL16', 'fixture_specs': specs,
                        'initial': initial, 'payloads': payloads, 'logs': logs}, OUT/'synthetic_payload_fixture.pt')
            checks['raw_RNG_unchanged_without_repair_and_zero_real_loads'] = helper.raw_rng_comparison(before, rng(), torch)['all_raw_values_equal'] and not load_attempts and not torch.cuda.is_initialized()
            require(not any(k in sys.modules for k in ('warp', 'mujoco', 'mujoco_warp', 'dex_supermimic', 'mjlab')), 'No native package import in fixture')
            details = {'focused_mutations': mutations, 'fixture_positive_details': evidence,
                       'fixture': a.pin(OUT/'synthetic_payload_fixture.pt'), 'fixture_parameter_count': 2,
                       'actual16_parameter_contract_qualified': False, 'actual_training_checkpoint_readback': False}
        finally:
            torch.load = real_load
        require(all(checks.values()), 'Focused synthetic payload proof failed')
        require(all(a.same(a.pin(v), source_pins[k]) for k, v in a.OWN.items()), 'Frozen proof sources unchanged')
        report = {'schema': 'full151105_boundary_public_self15_checkpoint_auditor_CPU_contract:1',
                  'status': 'PASS_SYNTHETIC_PUBLIC_SELF15_PAYLOAD_AND_RAW_HEALTH_NO_REAL_LOAD', 'checks': checks,
                  'passed': sum(checks.values()), 'total': len(checks), 'source_pins': source_pins, 'details': details,
                  'actual_counts': {'actual_checkpoint_loads': 0, 'actual_array_reads': 0,
                                    'actual_model_constructors_FW_AD': 0, 'MJB_Env_physics_GPU': 0}}
        a.write(OUT/'report.json', report)
        ready = {'schema': 'full151105_boundary_public_self15_checkpoint_auditor_CPU_READY:1', 'status': report['status'],
                 'source_pins': source_pins, 'cpu_report': a.pin(OUT/'report.json'),
                 'cpu_checks': {'passed': report['passed'], 'total': report['total']},
                 'api': {'entry': str(AUDITOR), 'actual': '--protocol PATH --run-dir PATH --iteration 100|1000 --output FRESH --run'},
                 'actual_checkpoint_readback_qualified': False, 'actual16_parameter_contract_qualified': False}
        a.write(OUT/'READY.json', ready)
        print(json.dumps({'status': report['status'], 'passed': report['passed'], 'total': report['total'], 'READY': a.pin(OUT/'READY.json')}))
        return 0
    except BaseException as error:
        if not (OUT/'failure.json').exists():
            a.write(OUT/'failure.json', {'error': repr(error), 'source_pins': source_pins, 'actual_checkpoints_models_FW_AD_physics': 0})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
