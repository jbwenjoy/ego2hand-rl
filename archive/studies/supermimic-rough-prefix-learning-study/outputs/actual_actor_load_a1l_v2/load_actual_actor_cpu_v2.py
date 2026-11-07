"""Saved-only actual104 evaluation Actor loading, with no Env or GPU."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys, traceback
from pathlib import Path
from collections import Counter
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
FROZEN_LOADER_SHA = '9f88ea6cd21a071d826dab6a890e1466160a8c619260cae15abd330700df3aca'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, default=ROOT / 'outputs/input_bundle_v1/bundle.json')
    ap.add_argument('--method', choices=('A1L', 'P_ONLY', 'A2'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from rough_prefix.bundle_v2 import load_bundle
    from rough_prefix.checkpoint_v2 import load_actor_checkpoint, _tensor_digest
    from rough_prefix.io import pin, read
    loader = ROOT / 'src/rough_prefix/checkpoint_v2.py'
    if pin(loader)['sha256'] != FROZEN_LOADER_SHA: raise ValueError('Executed fixture-qualified loader changed')
    for p in (Path(__file__), loader): (out / p.name).write_bytes(p.read_bytes())
    checks = []
    def check(name, value): checks.append({'name': name, 'passed': bool(value)})
    before = pin(a.checkpoint); before_stat = a.checkpoint.stat(); bundle = load_bundle(a.bundle)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    saved_versions = {key: value._version for key, value in saved['modules']['actor'].items()}
    saved_digest = _tensor_digest(saved['modules']['actor'])
    cpu_rng = torch.get_rng_state().clone(); np_rng = np.random.get_state(); py_rng = random.getstate()
    receipt = None; capacity_review = None; error = None
    try:
        actor, receipt = load_actor_checkpoint(a.checkpoint, bundle, a.method, device='cpu')
        loaded_digest = _tensor_digest(actor.state_dict())
        weights_before_forward = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        # Deterministic observation probes do not claim actual rollout states.
        probe = torch.stack([torch.zeros(155), torch.linspace(-1., 1., 155), torch.linspace(1., -1., 155)])
        with torch.no_grad(): action = actor(probe)
        np.savez_compressed(out / 'actor_CPU_forward.npz', observation=probe.numpy(), action=action.numpy())
        check('actual104_file_load_succeeded', receipt['format_version'] == 104 and receipt['method'] == a.method)
        check('complete_actor_weights_and_norm_exact', loaded_digest == saved_digest == receipt['actor_state_sha256'])
        check('actor_forward_finite_3x155_to3x20', action.shape == (3, 20) and bool(torch.isfinite(action).all()) and bool((action.abs() <= 1).all()))
        check('forward_did_not_mutate_weights_or_norm', all(torch.equal(value, actor.state_dict()[key]) for key, value in weights_before_forward.items()))
        check('saved_tensor_payload_not_mutated', all(value._version == saved_versions[key] for key, value in saved['modules']['actor'].items()) and _tensor_digest(saved['modules']['actor']) == saved_digest)
        check('CPU_numpy_python_RNG_preserved', torch.equal(cpu_rng, torch.get_rng_state()) and np.array_equal(np_rng[1], np.random.get_state()[1]) and np_rng[2:] == np.random.get_state()[2:] and py_rng == random.getstate())
        check('checkpoint_file_bytes_size_mtime_preserved', pin(a.checkpoint) == before and a.checkpoint.stat().st_size == before_stat.st_size and a.checkpoint.stat().st_mtime_ns == before_stat.st_mtime_ns)
        run = a.checkpoint.resolve().parents[1]
        final = read(run / 'capacity_observation.json'); prefix = saved['rough_capacity_evidence']
        mask = 0; unknown = 0
        for record in final['records']:
            if record['read_status'] == 'UNKNOWN': unknown += 1
            if record['raw_flags_values'] is not None:
                raw = np.asarray(record['raw_flags_values']); current = 0
                for value in raw.reshape(-1): current |= int(value)
                mask |= current
                check('capacity_record_' + str(record['sequence']) + '_raw_OR', current == record['raw_bitmask'] and record['cumulative_OR_bitmask'] == mask and record['raw_flags_shape'] == list(raw.shape))
        events = Counter(row['event'] for row in final['records'])
        check('checkpoint_prefix_is_exact_final_record_prefix', final['records'][:len(prefix['records'])] == prefix['records'])
        check('checkpoint_not_finalized', prefix['evidence_kind'] == 'checkpoint_prefix' and prefix['prefix_at_checkpoint_capacity_qualified'] and not prefix['entire_run_finalized'] and not prefix['finalized'] and prefix['final_read_count'] == 0)
        check('entire_run_finalized_once_after_checkpoint', final['finalized'] and final['entire_run_finalized'] and final['engineering_capacity_qualified'] and final['records'][-1]['event'] == 'final' and events['final'] == final['final_read_count'] == 1)
        check('all_saved_capacity_reads_qualified_no_world_attribution', mask == final['cumulative_OR_bitmask'] == 0 and unknown == final['unknown_read_count'] == 0 and final['status'] == 'OK' and not final['flag_shape_is_world_count'])
        check('reset_and_checkpoint_events_have_distinct_labels', events['before_reset'] == final['before_reset_read_count'] and events['checkpoint'] >= 1 and all(r['selected_reset']['kind'] == 'not_a_reset' for r in final['records'] if r['event'] == 'checkpoint'))
        capacity_review = {'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL', 'source': pin(run / 'capacity_observation.json'), 'event_counts': dict(events), 'stage_counts': dict(Counter(r['stage'] for r in final['records'])), 'cumulative_OR': mask, 'unknown': unknown, 'prefix_records': len(prefix['records']), 'final_records': len(final['records']), 'scope': 'Saved actual flag records and prefix linkage only. No physical reset re-executed and no budget health audit replacement.'}
    except BaseException as exc:
        error = {'error': repr(exc), 'traceback': traceback.format_exc()}
        checks.append({'name': 'actual_actor_load_and_saved_review_complete', 'passed': False, **error})
    check('CUDA_not_initialized_no_Env_Warp_import', not torch.cuda.is_initialized() and 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules)
    report = {'schema': 'rough104_actual_actor_CPU_load:1', 'status': 'PASS_ACTUAL_EVALUATION_ACTOR_CPU_LOAD' if all(c['passed'] for c in checks) else 'FAIL_OR_UNKNOWN', 'check_count': len(checks), 'passed': sum(c['passed'] for c in checks), 'checks': checks, 'actual_training_checkpoint_loaded': receipt is not None, 'receipt': receipt, 'capacity_saved_review': capacity_review, 'source': pin(__file__), 'checkpoint': before, 'bundle': pin(a.bundle), 'physics_steps': 0, 'Env_or_Runner_constructed': False, 'CUDA_initialized': torch.cuda.is_initialized(), 'physical_task_success': 'NOT_ASSESSED', 'error': error}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if receipt is not None: (out / 'actor_load_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'checks': report['check_count'], 'passed': report['passed'], 'failed': [c for c in checks if not c['passed']]}, indent=2))
    raise SystemExit(report['status'] != 'PASS_ACTUAL_EVALUATION_ACTOR_CPU_LOAD')


if __name__ == '__main__': main()
