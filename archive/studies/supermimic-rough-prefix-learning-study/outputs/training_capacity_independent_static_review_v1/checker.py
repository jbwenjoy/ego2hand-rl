"""CPU-only capacity integration and native atomic-save method contracts.

Env/backend instances are fixtures; the actual DexRunner.save and frozen
controller serialization wrapper are exercised without Env/Runner construction.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    snap = out / 'source'; snap.mkdir()
    sources = [Path(__file__), ROOT / 'src/rough_prefix/training.py', ROOT / 'src/rough_prefix/capacity.py', ROOT / 'src/rough_prefix/native_v2.py']
    pins = {}
    for p in sources:
        (snap / p.name).write_bytes(p.read_bytes())
        pins[p.name] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    checks = []
    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
    def case(name, fn):
        try: fn()
        except BaseException as exc:
            checks.append({'name': name, 'passed': False, 'error': repr(exc), 'traceback': traceback.format_exc()})
    def fresh(name):
        p = out / name; p.mkdir(); return p
    import numpy as np
    import torch
    from rough_prefix.training import TrainingCapacityScope, run_training, CAPACITY_SHA, ALLOWED_ROLES
    from rough_prefix.capacity import CapacityOverflowError, CapacityUnknownError
    from rough_prefix.controller import rough_controller
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner

    class Flag:
        def __init__(self): self.value = np.zeros(2, np.int32); self.reads = 0; self.fail = False
        def numpy(self):
            self.reads += 1
            if self.fail: raise OSError('fixture unreadable backend')
            return self.value
    class Manager:
        def __init__(self, cfg):
            self.cfg = cfg; self.flag = Flag(); self.sim = SimpleNamespace(wp_data=SimpleNamespace(overflow=self.flag))
            self.reset_calls = 0; self.close_calls = 0; self.close_output = None; self.close_error = None
        def reset(self, *, env_ids=None):
            self.reset_calls += 1; self.flag.value[:] = 0; return 'native_reset_result'
        def close(self):
            self.close_calls += 1
            if self.close_output is not None:
                data = json.loads((self.close_output / 'capacity_observation.json').read_text())
                self.snapshot_existed_before_close = data['finalized'] and data['final_read_count'] == 1
            self.flag.fail = True  # Reading after release would be UNKNOWN.
            if self.close_error is not None: raise self.close_error
    cfg = SimpleNamespace(sim=SimpleNamespace(nconmax=192, njmax=4096))
    originals = (Manager.__init__, torch.save)
    payload = {'format_version': 104, 'rough_controller': {'fixture': True},
               'rough_runtime_evidence': {'preserve': 1},
               'rough_initial_parameter_fingerprints': {'preserve': 2},
               'modules': {'wm': torch.tensor([3.])}, 'iteration': 10, 'sim_steps': 5248}

    def normal():
        folder = fresh('normal')
        with TrainingCapacityScope(Manager, torch, folder, stage='engineering_smoke') as scope:
            env = Manager(cfg); env.close_output = folder
            check('observer_installed_before_first_native_reset', hasattr(env, '_rough_capacity_observer') and env.reset_calls == 0)
            rng = torch.get_rng_state().clone()
            check('native_reset_return_preserved', env.reset(env_ids=np.array([1], np.int64)) == 'native_reset_result')
            observer = scope.entries[0]['observer']; read_count = env.flag.reads
            original_payload = payload.copy()
            torch.save(payload, folder / 'model.pt')
            state = torch.load(folder / 'model.pt', weights_only=False, map_location='cpu')
            cap = state['rough_capacity_evidence']
            check('checkpoint_reads_once_without_finalize', env.flag.reads == read_count + 1 and not observer.finalized and cap['final_read_count'] == 0)
            check('checkpoint_event_is_not_before_reset', cap['records'][-1]['event'] == 'checkpoint' and cap['records'][-1]['selected_reset']['kind'] == 'not_a_reset')
            check('checkpoint_prefix_qualified_whole_run_not_qualified', cap['prefix_at_checkpoint_capacity_qualified'] and not cap['entire_run_finalized'] and not cap['engineering_capacity_qualified'])
            check('checkpoint_preserves_runtime_initial_and_model_payload', state['rough_runtime_evidence'] == payload['rough_runtime_evidence'] and state['rough_initial_parameter_fingerprints'] == payload['rough_initial_parameter_fingerprints'] and torch.equal(state['modules']['wm'], payload['modules']['wm']) and set(payload) == set(original_payload))
            check('checkpoint_selection_dtype_shape_preserved', cap['records'][0]['selected_reset']['dtype'] == 'int64' and cap['records'][0]['selected_reset']['values'] == [1] and cap['records'][0]['raw_flags_shape'] == [2] and not cap['flag_shape_is_world_count'])
            other = {'format_version': 3, 'tensor': torch.tensor([7.])}
            torch.save(other, folder / 'unrelated.pt')
            got = torch.load(folder / 'unrelated.pt', weights_only=False)
            check('non104_save_exact_passthrough_no_capacity_read', set(got) == set(other) and torch.equal(got['tensor'], other['tensor']) and env.flag.reads == read_count + 1)
            check('observation_and_save_no_RNG_change', torch.equal(rng, torch.get_rng_state()))
            env.close()
            check('final_snapshot_saved_before_native_close', env.snapshot_existed_before_close)
            check('normal_final_qualified', scope.qualified_final())
            before_exit_reads = env.flag.reads
        check('final_read_once_not_repeated_after_close', env.flag.reads == before_exit_reads and observer.snapshot()['final_read_count'] == 1)
        check('class_instance_torch_scope_restored', (Manager.__init__, torch.save) == originals and 'reset' not in vars(env) and 'close' not in vars(env) and not hasattr(env, '_rough_capacity_observer'))
    case('normal_contract_completed', normal)

    def overflow():
        folder = fresh('overflow')
        try:
            with TrainingCapacityScope(Manager, torch, folder, stage='engineering_smoke') as scope:
                env = Manager(cfg); env.flag.value[:] = [0, 8]
                env.reset(env_ids=[0])
        except CapacityOverflowError:
            saved = json.loads((folder / 'capacity_observation.json').read_text())
            check('pre_reset_overflow_latched_before_native_clear', env.reset_calls == 0 and env.flag.value.tolist() == [0, 8] and saved['cumulative_OR_bitmask'] == 8)
            check('overflow_snapshot_finalized_not_qualified', saved['finalized'] and not saved['engineering_capacity_qualified'] and saved['status'] == 'OVERFLOW')
        else: check('overflow_must_raise', False)
    case('overflow_contract_completed', overflow)

    class BodyError(RuntimeError): pass
    def body_error():
        folder = fresh('body_error_unknown_cleanup'); sentinel = BodyError('original body exception')
        try:
            with TrainingCapacityScope(Manager, torch, folder, stage='engineering_smoke'):
                env = Manager(cfg); env.flag.fail = True; raise sentinel
        except BodyError as exc:
            saved = json.loads((folder / 'capacity_observation.json').read_text())
            check('body_exception_not_replaced_by_cleanup_UNKNOWN', exc is sentinel and saved['status'] == 'UNKNOWN' and env.close_calls == 1)
        check('exception_scope_restored', (Manager.__init__, torch.save) == originals and 'reset' not in vars(env) and 'close' not in vars(env))
    case('body_exception_contract_completed', body_error)

    def checkpoint_failure():
        folder = fresh('checkpoint_unknown')
        try:
            with TrainingCapacityScope(Manager, torch, folder, stage='engineering_smoke'):
                env = Manager(cfg); env.flag.fail = True
                torch.save(payload, folder / 'not_written.pt')
        except CapacityUnknownError:
            check('UNKNOWN_checkpoint_rejects_save_without_retry', not (folder / 'not_written.pt').exists() and env.reset_calls == 0 and env.close_calls == 1)
        else: check('UNKNOWN_checkpoint_must_raise', False)
        check('checkpoint_error_restores_torch_and_manager', (Manager.__init__, torch.save) == originals)
    case('checkpoint_failure_contract_completed', checkpoint_failure)

    def write_error():
        folder = fresh('save_IO_failure')
        try:
            with TrainingCapacityScope(Manager, torch, folder, stage='engineering_smoke'):
                env = Manager(cfg)
                torch.save(payload, folder / 'missing' / 'model.pt')
        except RuntimeError:
            check('native_serialization_error_kept_and_final_capacity_saved', (folder / 'capacity_observation.json').is_file() and env.close_calls == 1)
        else: check('native_save_IO_error_expected', False)
        check('save_IO_error_scope_restored', (Manager.__init__, torch.save) == originals)
    case('write_error_contract_completed', write_error)

    # Actual native atomic save, actual controller3->104 wrapper and the same
    # runtime evidence ordering as frozen RecordedRunner.save. No Runner init.
    class SaveEnv:
        def __init__(self): pass
        def step(self, *args): pass
    class SaveMDP:
        def __init__(self): pass
        def control_target(self, *args): pass
    class SaveRunner:
        save = DexSuperMimicRunner.save
        def __init__(self, env):
            self.env = env; self.device = 'cpu'; self.iteration = 9; self.sim_steps = 5248
            self.cfg = SimpleNamespace(algorithm='world_model', to_dict=lambda: {'task': 'hand2-pen-spin'})
            self.mdp = SimpleNamespace(reference_signature='CPU_fixture_signature', rough_controller_metadata={'CPU_fixture': True})
            self.torque_norm = SimpleNamespace(state_dict=lambda: {'mean': torch.tensor([0.])})
            self.context_dim = 46; self.contact_dim = 15; self.optimizers = {'fixture': SimpleNamespace(state_dict=lambda: {'CPU_fixture': True})}
            self.reader = SimpleNamespace(context_names=['CPU_fixture'], robot=SimpleNamespace(joint_names=['CPU_fixture']))
            self.wall_offset = 0.; self.started_at = time.perf_counter(); self.env.common_step_counter = 41
            module = torch.nn.Module(); module.register_parameter('weight', torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3)))
            self._modules = {'wm': module}; self.rough_initial_parameter_fingerprints = {'kept': True}
        def modules(self): return self._modules
        def format_version(self): return 3
        def collect(self, *args): raise AssertionError('No collect permitted in CPU save test')
        def load(self, *args): pass
        def load_model_only(self, *args): pass
    class Recorded(SaveRunner):
        def save(self, path, next_iteration=None):
            native_save = torch.save
            def serialize(state, *args, **kwargs):
                if state.get('format_version') != 104: raise ValueError('Runtime must follow controller')
                return native_save({**state, 'rough_runtime_evidence': {'fixture_runtime': 1}, 'rough_initial_parameter_fingerprints': self.rough_initial_parameter_fingerprints}, *args, **kwargs)
            torch.save = serialize
            try: super().save(path, next_iteration)
            finally: torch.save = native_save
    def real_native_save():
        folder = fresh('real_native_atomic_save'); native_save_method = SaveRunner.save
        with TrainingCapacityScope(Manager, torch, folder, stage='engineering_smoke', runner_class=SaveRunner) as scope:
            env = Manager(cfg); rng = torch.get_rng_state().clone()
            with rough_controller(SaveEnv, SaveMDP, SaveRunner, np.zeros((15, 20), np.float32), {'CPU_fixture_only': True}):
                runner = Recorded(env)
                initial = torch.load(folder / 'initial_parameter_tensors.pt', weights_only=False)
                check('initial_raw_tensors_saved_before_any_collect_or_update', initial['schema'] == 'rough_prefix_initial_parameters:1' and initial['modules']['wm']['parameter_names'] == ['weight'] and torch.equal(initial['modules']['wm']['parameters'][0], runner._modules['wm'].weight) and initial['modules']['wm']['shapes'] == [[2, 3]])
                check('initial_raw_capture_no_rng_change', torch.equal(rng, torch.get_rng_state()))
                runner.save(folder / 'model_10.pt', next_iteration=10)
                state = torch.load(folder / 'model_10.pt', weights_only=False)
                check('actual_DexRunner_atomic_save_iteration_optimizer_payload', state['iteration'] == 10 and state['sim_steps'] == 5248 and state['optimizers'] == {'fixture': {'CPU_fixture': True}} and not (folder / 'model_10.tmp').exists())
                check('all_three_serialization_layers_coexist', state['format_version'] == 104 and state['rough_controller']['CPU_fixture'] and state['rough_runtime_evidence'] == {'fixture_runtime': 1} and state['rough_initial_parameter_fingerprints'] == {'kept': True} and state['rough_capacity_evidence']['prefix_at_checkpoint_capacity_qualified'])
                check('raw_init_artifact_has_no_checkpoint_metadata', 'rough_capacity_evidence' not in initial and 'format_version' not in initial)
                cap = state['rough_capacity_evidence']; env.close()
                final = json.loads((folder / 'capacity_observation.json').read_text())
                check('checkpoint_capacity_is_exact_prefix_of_final', final['records'][:len(cap['records'])] == cap['records'] and final['records'][-1]['event'] == 'final')
            check('controller_save_restored_after_native_method_test', SaveRunner.save is native_save_method)
        check('runner_init_restored', 'runner_initialize' not in SaveRunner.__init__.__qualname__)
    case('actual_native_atomic_save_contract_completed', real_native_save)

    for role, method in [('native_a0', 'A0'), ('engineering_smoke', 'A0'), ('primary_matched_arm', 'A1L')]:
        try: run_training({'role': role, 'method': method}, None, out / 'must_not_create', None)
        except ValueError: check('invalid_role_or_A0_rejected_' + role + '_' + method, not (out / 'must_not_create').exists())
        else: check('invalid_role_or_A0_rejected_' + role + '_' + method, False)
    check('actual_frontschema_roles', ALLOWED_ROLES == ('engineering_smoke', 'primary_matched'))
    check('frozen_capacity_7380_unchanged', pins['capacity.py']['sha256'] == CAPACITY_SHA)
    check('CUDA_not_initialized_no_Env_physics_optimizer', not torch.cuda.is_initialized())
    result = {'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL', 'check_count': len(checks), 'passed': sum(c['passed'] for c in checks), 'checks': checks, 'source_pins': pins, 'actual_native_method': 'DexSuperMimicRunner.save, atomic temp+replace and frozen rough_controller serialization', 'actual_native_Env_or_Runner_initialized': False, 'physics_steps': 0, 'optimizer_steps': 0, 'CUDA_initialized': torch.cuda.is_initialized(), 'scope': 'Fake capacity backend/Manager. Actual native save method only; no model training, Manager physics or CUDA.'}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': result['check_count'], 'passed': result['passed'], 'failed': [c for c in checks if not c['passed']]}, indent=2))
    raise SystemExit(result['status'] != 'PASS')


if __name__ == '__main__': main()
