"""Synthetic format104 loader contracts with the actual native Actor, CPU only."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys, traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, required=True); a = ap.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False); source = out / 'source'; source.mkdir()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from rough_prefix.bundle_v2 import load_bundle
    from rough_prefix.checkpoint_v2 import load_actor_checkpoint
    from rough_prefix.controller import _digest
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
    b = load_bundle(ROOT / 'outputs/input_bundle_v1/bundle.json')
    base_controller = json.loads((ROOT / 'outputs/native_a0_v2/controller_descriptor.json').read_text())
    c = copy.deepcopy(base_controller); c['root_descriptor']['trained_method'] = 'A1L'; c['root_descriptor']['run_role'] = 'engineering_smoke'
    def resign(state):
        c = state['rough_controller']; c['root_descriptor_sha256'] = _digest(c['root_descriptor'])
        c['reference_controller_signature'] = _digest({k: v for k, v in c.items() if k not in ('reference_controller_signature', 'latest_actualu_evidence')})
        state['reference_signature'] = c['reference_controller_signature']
    with torch.random.fork_rng(devices=[]): actor = Actor(155)
    cfg = DexSuperMimicCfg(task='hand2-pen-spin', recipe_path=b['native_reference']['recipe_path'], clips_root=b['native_reference']['clips_root'], seed=0)
    state = {'format_version': 104, 'rough_controller': c, 'iteration': 10, 'sim_steps': 5248,
             'config': cfg.to_dict(), 'context_dim': 368, 'contact_dim': 15,
             'joint_names': [f'right_finger{f}_joint{j}' for f in range(1, 6) for j in range(1, 5)],
             'context_names': [('friction_ratio', 1), ('object_mass_ratio', 1), ('kp_ratio', 1), ('kd_ratio', 1), ('joint_damping_ratio', 1), ('object_size_ratio', 1), ('actuator_kp_ratios', 20), ('actuator_kd_ratios', 20)],
             'modules': {'actor': actor.state_dict()}, 'scope': 'SYNTHETIC_CHECKPOINT_NOT_ACTUAL_TRAINING'}
    resign(state); path = out / 'synthetic_104.pt'; torch.save(state, path)
    checks = []; cases = []; pins = {}
    for p in (Path(__file__), ROOT / 'src/rough_prefix/checkpoint_v2.py'):
        (source / p.name).write_bytes(p.read_bytes()); pins[p.name] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    def check(name, ok): checks.append({'name': name, 'passed': bool(ok)})
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest(); rng = torch.get_rng_state().clone(); np_rng = np.random.get_state(); py_rng = random.getstate()
    try:
        loaded, receipt = load_actor_checkpoint(path, b, 'A1L')
        check('synthetic104_real_nativeActor_load_complete_norm_and_weights', all(torch.equal(value, loaded.state_dict()[key]) for key, value in actor.state_dict().items()))
        check('load_preserves_cpu_numpy_python_RNG', torch.equal(rng, torch.get_rng_state()) and np.array_equal(np_rng[1], np.random.get_state()[1]) and np_rng[2:] == np.random.get_state()[2:] and py_rng == random.getstate())
        with torch.no_grad(): result = loaded(torch.zeros(4, 155))
        check('actual_Actor155_to20_forward_finite', tuple(result.shape) == (4, 20) and bool(torch.isfinite(result).all()))
        check('evaluation_frozen_not_resume', not loaded.training and all(not p.requires_grad for p in loaded.parameters()) and receipt['evaluation_only'] and not receipt['training_resume_supported'])
        check('checkpoint_bytes_not_changed', hashlib.sha256(path.read_bytes()).hexdigest() == original_sha)
        (out / 'synthetic_load_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    except BaseException as exc:
        checks.append({'name': 'positive_loader_complete', 'passed': False, 'error': repr(exc), 'traceback': traceback.format_exc()})
    def negative(name, change, *, expected='A1L', rebalance=False, bundle=None):
        altered = copy.deepcopy(state); change(altered)
        if rebalance: resign(altered)
        file = out / (name + '.pt'); torch.save(altered, file)
        before = hashlib.sha256(file.read_bytes()).hexdigest(); before_rng = torch.get_rng_state().clone()
        try: load_actor_checkpoint(file, b if bundle is None else bundle, expected)
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            check('reject_' + name, True); cases.append({'case': name, 'rejected': repr(exc)})
        else: check('reject_' + name, False)
        check('rejected_file_and_RNG_unchanged_' + name, before == hashlib.sha256(file.read_bytes()).hexdigest() and torch.equal(before_rng, torch.get_rng_state()))
    negative('old3', lambda s: s.update(format_version=3))
    negative('old103', lambda s: s.update(format_version=103))
    negative('wrong_expected_arm', lambda s: None, expected='A2')
    negative('wrong_controller_signature', lambda s: s.update(reference_signature='0' * 64))
    negative('resigned_wrong_input_signature', lambda s: s['rough_controller']['root_descriptor'].update(input_signature_sha256='1' * 64), rebalance=True)
    negative('resigned_wrong_baseline', lambda s: s['rough_controller'].update(baseline_u15_FP32_sha256='2' * 64), rebalance=True)
    negative('resigned_wrong_bounds', lambda s: s['rough_controller']['target_lower_rad'].__setitem__(0, -999.), rebalance=True)
    negative('resigned_wrong_velocity_authority', lambda s: s['rough_controller']['root_descriptor'].update(goal_velocity_authority='nativeFD'), rebalance=True)
    negative('resigned_wrong_source_pin', lambda s: next(iter(s['rough_controller']['root_descriptor']['execution_source_pins'].values())).update(sha256='3' * 64), rebalance=True)
    negative('resigned_wrong_native_reference', lambda s: s['rough_controller'].update(native_reference_signature='4' * 64), rebalance=True)
    negative('wrong_seed', lambda s: s['config'].update(seed=19))
    negative('wrong_hz', lambda s: s['config'].update(control_hz=20))
    negative('physical_xi46_is_not_task_context', lambda s: s.update(context_dim=46))
    negative('wrong_joint_order', lambda s: s['joint_names'].reverse())
    negative('wrong_context_order', lambda s: s['context_names'].reverse())
    negative('missing_normalizer', lambda s: s['modules']['actor'].pop('norm._std'))
    negative('wrong_actor_input', lambda s: s['modules']['actor'].__setitem__('net.0.weight', torch.zeros(512, 154)))
    negative('nonfinite_actor', lambda s: s['modules']['actor']['net.6.bias'].__setitem__(0, float('nan')))
    negative('nonfinite_other_savedtensor', lambda s: s.update(other=torch.tensor([float('inf')])))
    negative('negative_normalizer_std', lambda s: s['modules']['actor']['norm._std'].__setitem__((0, 0), -1.))
    changed_bundle = copy.deepcopy(b); changed_bundle['arrays']['reference']['analytic_qd16_rad_s'][1, 0] += 1.
    negative('caller_changed_bundle_arrays', lambda s: None, bundle=changed_bundle)
    changed_bundle = copy.deepcopy(b); changed_bundle['descriptor']['source_identity']['primary_pose_source_frame_range_half_open'] = [0, 16]
    negative('caller_relabels_legacy_window', lambda s: None, bundle=changed_bundle)
    # Save-time actualu metadata is explicitly outside the immutable signature.
    dynamic = copy.deepcopy(state); dynamic['rough_controller']['latest_actualu_evidence'] = {'collect_call': 11, 'CPU_fixture': True}
    file = out / 'synthetic_dynamic_evidence.pt'; torch.save(dynamic, file)
    try:
        _, r = load_actor_checkpoint(file, b, 'A1L'); check('save_time_latestactualu_excluded_only_as_frozen_controller_defines', r['reference_signature'] == state['reference_signature'])
    except BaseException as exc:
        checks.append({'name': 'save_time_latestactualu_excluded_only_as_frozen_controller_defines', 'passed': False, 'error': repr(exc)})
    check('CPU_CUDA_uninitialized_no_physics', not torch.cuda.is_initialized())
    report = {'schema': 'rough104_actor_loader_CPU_contract:1', 'status': 'PASS' if all(x['passed'] for x in checks) else 'FAIL', 'checks': checks, 'check_count': len(checks), 'passed': sum(x['passed'] for x in checks), 'negative_cases': cases, 'source_pins': pins, 'actual_training_checkpoint_loaded': False, 'fixtures': 'Synthetic actor/checkpoint; actual validated bundle/native model/source signatures. Existing native A0 controller metadata is template only, relabeled fixture is not a training output.', 'native_Actor_constructed': True, 'native_Env_or_Runner_constructed': False, 'CUDA_initialized': torch.cuda.is_initialized(), 'physics_steps': 0}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'checks': report['check_count'], 'passed': report['passed'], 'failed': [x for x in checks if not x['passed']]}, indent=2))
    raise SystemExit(report['status'] != 'PASS')


if __name__ == '__main__': main()
