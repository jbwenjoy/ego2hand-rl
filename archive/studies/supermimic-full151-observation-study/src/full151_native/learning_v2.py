"""From-scratch three-arm native training for the separately named full151 mode.

Import is metadata/CPU only. An explicit authorized caller must invoke
run_training to construct the actual Manager, Runner, CUDA WM and optimizers.
All old controller/runtime/native files are used unchanged, never rewritten.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import random
import time
from pathlib import Path
import numpy as np

from .io import checked, jsonable, pin, sha, read, canonical
from .metadata import FORMAT_VERSION, MODE, VARIANT, write

ALLOWED_ROLES = ('engineering_smoke', 'primary_matched')
METHODS = ('A1L', 'P_ONLY', 'A2')
MODULE_KEYS = ('wm', 'fail_head', 'actor', 'critic')
FROZEN_SOURCES = {
    'controller.py': 'b81c5376caaed93941078600983afa6356df60f01be50dcc2b9c259b19a257e0',
    'runtime.py': 'd41057c2f849fe645a52c7ffa8b175bac60eb6aa1ed616d756a9df64a286f8c9',
    'native.py': '142cd9b710786d1973c6c518165e1d08bfd3a38b7a65b425819b538e3d025a2f',
}


def source_pins():
    result = {'learning': pin(__file__),
              'training_capacity': pin(Path(__file__).with_name('training_capacity.py'))}
    for name, expected in FROZEN_SOURCES.items():
        value = pin(Path(__file__).with_name(name))
        if value['sha256'] != expected:
            raise ValueError('Frozen full151 training dependency changed: ' + name)
        result[name[:-3]] = value
    return result


def validate_training_inputs(config, bundle):
    """No Torch/Warp imports, simulator, optimizer, fallback or old-mode migration."""
    if config.get('schema') != 'full151_knownrough_native_training_run:2' or config.get('role') not in ALLOWED_ROLES or config.get('method') not in METHODS:
        raise ValueError('Only new full151 schema2 from-scratch A1L/P_ONLY/A2 training is accepted')
    validate_prior_spec(config,bundle)
    if any(config.get(k) not in (None, False, '') for k in ('resume', 'checkpoint', 'load_checkpoint', 'model_only', 'warm_start')):
        raise ValueError('Full151 training is from scratch only, with no old checkpoint loading')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        raise ValueError('Explicit nonnegative integer training seed required')
    d = bundle['descriptor']
    if (d.get('mode') != MODE or d.get('retarget_variant_id') != VARIANT
            or d.get('schema') != 'full151_knownrough_controller_input_descriptor:1'
            or d.get('physical_reference_qualified') is not False
            or d.get('control_intervals') != 150 or d.get('reference_states') != 151):
        raise ValueError('New known-rough full151 descriptor required; prefix/padding rejected')
    u = np.asarray(bundle['baseline_u150'])
    if u.shape != (150, 20) or u.dtype != np.float32 or not np.isfinite(u).all():
        raise ValueError('Exactly150 finite FP32 baseline targets required')
    if not np.array_equal(u[0], bundle['arrays']['init']['u_planned0_FP32_rad']):
        raise ValueError('Initial requested target differs from baseline0')
    t = config['training']
    common = dict(num_envs=128, buffer_steps=256, warmup_steps=64, wm_window=8,
                  wm_batches=2, wm_batch=256, pi_batch=128, horizon=8,
                  max_horizon=8, critic_epochs=2, critic_minibatches=2,
                  holdout_fraction=.1, log_interval=1, amp=True, compile=False,
                  compile_policy=False, compile_policy_dynamic=False,
                  action_noise=.1, utd=.4)
    for key, expected in common.items():
        if key not in t or type(t[key]) is not type(expected) or t[key] != expected:
            raise ValueError('Native matched training hyperparameter differs: ' + key)
    if list(t.get('wm_hidden', ())) != [512, 512, 512]:
        raise ValueError('Native WM hidden512x3 required')
    for key in ('max_iterations', 'wm_only_iters', 'save_interval'):
        if type(t.get(key)) is not int:
            raise ValueError('Integer finite budget required: ' + key)
    if not 0 <= t['wm_only_iters'] < t['max_iterations'] or not 1 <= t['save_interval'] <= min(100, t['max_iterations']):
        raise ValueError('A finite WM-and-policy budget with checkpoints is required')
    if config['role'] == 'primary_matched' and (t['max_iterations'], t['wm_only_iters'], t['save_interval']) != (2000, 200, 100):
        raise ValueError('Primary from-scratch budget must remain2000/WM-only200/save100')
    if config['role'] == 'engineering_smoke' and (t['max_iterations'], t['wm_only_iters'], t['save_interval']) != (40, 20, 40):
        raise ValueError('Engineering-only budget must remain40/WM-only20/save40')
    for value in config['source_pins'].values():
        checked(value)
    return {'schema': 'full151_native_training_input_CPU_contract:1', 'method': config['method'],
            'role': config['role'], 'from_scratch': True, 'contact_objective': 'NONE' if config['method']=='A1L' else config['method'],
            'physical_reference_qualified': False, 'format_version': FORMAT_VERSION,
            'training': jsonable(t)}


def validate_prior_spec(config,bundle):
    method=config['method'];value=config.get('contact_binding')
    if method=='A1L':
        if value is not None:raise ValueError('A1L must not construct or bind a contact objective')
        return None
    if type(value) is not dict:raise ValueError('P_ONLY/A2 require an exact new spec:2 pin')
    spec=read(checked(value))
    if spec.get('schema')!='full151_declared_H123_objective_spec:2' or spec.get('execution_scope')!='EXPLORATORY_MATCHED_TRAINING':raise ValueError('New training spec:2 required')
    if spec.get('CPU_authorized') is not True or spec.get('training_authorized') is not True or type(spec.get('GPU_authorized')) is not bool:raise ValueError('Explicit new objective authorization fields required')
    if config.get('GPU_run_authorized') is True and spec['GPU_authorized'] is not True:raise ValueError('CPU-only objective spec cannot enter actual GPU training')
    if spec['input_bundle']!=bundle['descriptor']['input_bundle'] or spec['hyperparameters']!={'bandwidth_m':.001,'tolerance':.1,'var':.1}:raise ValueError('Objective source/budget hyperparameter binding differs')
    for key in ('binding_manifest','target_receipt','independent_target_review','prior_CPU_binding_review','actual_material_contract','matched_material_screen','matched_material_saved_review','legacy_proxy_dynamic_cone_review','matched_material_lifecycle_review'):
        checked(spec[key])
    for value in spec['expected_code_pins'].values():checked(value)
    return spec


def objective_rng_snapshot():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state().clone(),
            'cuda_initialized':torch.cuda.is_initialized(),
            'torch_cuda':([v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])}


def objective_rng_comparison(before,after):
    import torch,pickle
    def digest(value):return hashlib.sha256(value).hexdigest()
    def hashes(value):
        n=value['numpy']
        return {'python':digest(pickle.dumps(value['python'],protocol=4)),
                'numpy':digest(pickle.dumps(n,protocol=4)),
                'torch_cpu':digest(value['torch_cpu'].numpy().tobytes()),
                'torch_cuda':[digest(v.cpu().numpy().tobytes()) for v in value['torch_cuda']],
                'cuda_initialized':value['cuda_initialized']}
    checks={'Python_exact':before['python']==after['python'],
            'NumPy_exact':before['numpy'][0]==after['numpy'][0] and np.array_equal(before['numpy'][1],after['numpy'][1]) and before['numpy'][2:]==after['numpy'][2:],
            'TorchCPU_exact':torch.equal(before['torch_cpu'],after['torch_cpu']),
            'CUDA_initialization_unchanged':before['cuda_initialized']==after['cuda_initialized'],
            'already_initialized_CUDA_exact':len(before['torch_cuda'])==len(after['torch_cuda']) and all(torch.equal(a,b) for a,b in zip(before['torch_cuda'],after['torch_cuda']))}
    return {'schema':'full151_objective_build_rng:1','status':'PRESERVED' if all(checks.values()) else 'RNG_CHANGED_ABORT',
            'checks':checks,'before':hashes(before),'after':hashes(after),
            'scope':'Entire binding_v3 import+factory+device_to range, after initial4 capture and before firstcollect. Observation only; learner never resets/restores RNG to conceal changes. CUDA states read only if already initialized.'}


def construct_prior(config,output,device):
    # Called after raw initial capture and fingerprints; never by default CPU
    # metadata preflight and never during native Runner construction.
    from .prior_hook_v1 import ImaginationPrior
    method=config['method'];output=Path(output)
    if method=='A1L':
        prior=ImaginationPrior(method,None,None,output)
        prior.build_rng={'schema':'full151_objective_build_rng:1','status':'NOT_CONSTRUCTED','method':'A1L','objective':'NONE','RNG_capture_required':False}
        write(output/'objective_build_rng.json',prior.build_rng)
        return prior
    spec=read(checked(config['contact_binding']))
    before=objective_rng_snapshot();objective=None;body_error=None
    try:
        from full151_intent.binding_v3 import build_contact_objective,build_penalty_only_objective
        objective=(build_penalty_only_objective if method=='P_ONLY' else build_contact_objective)(spec,device=device)
    except BaseException as exc:
        body_error=exc
        raise
    finally:
        report=objective_rng_comparison(before,objective_rng_snapshot())
        report.update(method=method,spec_pin=config['contact_binding'],factory_error=None if body_error is None else repr(body_error))
        write(output/'objective_build_rng.json',report)
        if body_error is None and report['status']!='PRESERVED':raise RuntimeError('Objective import/build changed sampling RNG; abort without repair')
    prior=ImaginationPrior(method,objective,config['contact_binding'],output);prior.build_rng=report
    return prior


def parameter_fingerprints(modules):
    """Native parameter order; CPU clones and hashes consume no random draws."""
    import torch
    if set(modules) != set(MODULE_KEYS):
        raise ValueError('Native module identity differs')
    before = {name: [p.detach().cpu().clone() for p in module.parameters()]
              for name, module in modules.items()}
    result = {}
    for name, values in before.items():
        h = hashlib.sha256()
        for value in values:
            h.update(str((str(value.dtype), tuple(value.shape))).encode())
            h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        result[name] = {'sha256': h.hexdigest(), 'tensors': len(values),
                        'parameters': sum(value.numel() for value in values)}
    return before, result


def collect_phase_evidence(runner, start_ptr, steps):
    """Read completed native ring rows; never add transitions or alter sampling."""
    import torch
    if type(steps) is not int or not 0 < steps <= runner.buffer.capacity:
        raise ValueError('Bounded native collect required for unambiguous ring evidence')
    data = runner.buffer.data['full151_phase']
    rows = (start_ptr + torch.arange(steps, device=data.device)) % runner.buffer.capacity
    phases = data[rows].reshape(-1)
    if bool(((phases < 0) | (phases >= 150)).any()):
        raise ValueError('Replay contains an unexecuted full151 phase')
    histogram = torch.bincount(phases.long(), minlength=150).cpu().tolist()
    old = getattr(runner, 'full151_phase_hist150', [0] * 150)
    runner.full151_phase_hist150 = [a + b for a, b in zip(old, histogram)]
    return {'latest_collect_phase_hist150': histogram,
            'phase_hist150': list(runner.full151_phase_hist150),
            'observed_transition_count': sum(runner.full151_phase_hist150),
            'all150_phases_observed': all(n > 0 for n in runner.full151_phase_hist150)}


def runtime_evidence(runner, runtime_api):
    policy = runtime_api['policy_evidence'](runner)
    return jsonable({'replay_counts': getattr(runner, '_full151_runtime_replay_counts', None),
        'latest_window': getattr(runner, '_full151_runtime_latest_window', None),
        'policy': policy, 'policy_not_run_is_not_a_pass': policy.get('status') == 'NOT_RUN',
        'phase_hist150': getattr(runner, 'full151_phase_hist150', [0] * 150),
        'phase_coverage': getattr(runner, 'full151_latest_phase_evidence', None)})


@contextmanager
def checkpoint_evidence_scope(runner, runtime_api, torch_module):
    """After controller3->105, before outer capacity hook and native atomic IO."""
    previous_save = torch_module.save
    evidence = runtime_evidence(runner, runtime_api)
    def serialize(state, *args, **kwargs):
        if (not isinstance(state, dict) or type(state.get('format_version')) is not int
                or state['format_version'] != FORMAT_VERSION or 'full151_controller' not in state):
            raise ValueError('Runtime evidence must follow full151105 controller serialization')
        if any(k in state for k in ('full151_runtime_evidence', 'full151_initial_parameter_fingerprints', 'full151_prior_evidence')):
            raise ValueError('Duplicate full151 runtime/fingerprint evidence')
        fingerprints = runner.full151_initial_parameter_fingerprints
        if set(fingerprints) != set(MODULE_KEYS):
            raise ValueError('Four from-scratch parameter fingerprints required')
        return previous_save({**state, 'full151_runtime_evidence': evidence,
            'full151_initial_parameter_fingerprints': fingerprints,
            'full151_prior_evidence':runner.full151_prior.evidence()}, *args, **kwargs)
    torch_module.save = serialize
    try: yield
    finally: torch_module.save = previous_save


def run_training(config, bundle, output, emit):
    """Explicit device entry, scoped capacity outside both frozen class contexts."""
    validate_training_inputs(config, bundle)
    pins = source_pins()
    if config.get('GPU_run_authorized') is not True:
        raise ValueError('Separate explicit Root GPU training authorization required')
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from .training_capacity import TrainingCapacityScope
    with TrainingCapacityScope(ManagerBasedRlEnv, torch, output,
                              stage='full151_' + config['role'], source_pins=pins,
                              runner_class=DexSuperMimicRunner) as scope:
        result = _run_native(config, bundle, Path(output), emit)
        if not scope.qualified_final():
            raise RuntimeError('Full151 training final capacity evidence failed; no retry')
        result.update(full151_capacity_observation=pin(Path(output) / 'capacity_observation.json'),
                      full151_capacity_qualified_final=True)
    write(Path(output) / 'result.json', result)
    return result


def _run_native(config, bundle, output, emit):
    import torch
    import warp as wp
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg, SceneEntityCfg
    from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP, PEN_MEMORY_DIMS
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand
    from dex_supermimic.tasks.pen_spin import mdp
    from .runtime import full151_runtime
    from .controller import full151_controller
    from .native import install_actual_context, ACTUAL_FIELDS
    if not output.is_dir():
        raise ValueError('Authorized worker must reserve a fresh output directory')
    started = time.perf_counter(); seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.suppress_errors = False
    if not torch.cuda.is_available():
        raise RuntimeError('Explicitly authorized isolated NVIDIA environment required')
    wp.set_device('cuda:0')
    settings = {**config['training'], 'wm_hidden': tuple(config['training']['wm_hidden']),
        'task': 'hand2-pen-spin', 'algorithm': 'world_model',
        'recipe_path': bundle['native_reference']['recipe_path'],
        'clips_root': bundle['native_reference']['clips_root'], 'device': 'cuda:0',
        'seed': seed, 'logger': 'none', 'contact_weight': .1, 'failure_head': False,
        'profile': True}
    cfg = DexSuperMimicCfg(**settings)
    write(output / 'effective_training_config.json', cfg.to_dict())
    env_cfg = make_env_cfg(cfg.num_envs, 50, play=True, recipe_path=Path(cfg.recipe_path), clips_root=Path(cfg.clips_root))
    if set(env_cfg.terminations) != {'time_out', 'unstable', 'object_position_error', 'reference_end'}:
        raise ValueError('Native play factory termination ABI differs')
    env_cfg.seed = seed; env_cfg.auto_reset = False
    motion = env_cfg.commands['motion']; motion.first_frame_hold_prob = 0.
    motion.first_frame_hold_s = (0., 0.); motion.trajectory_sampling_mode = 'uniform'; motion.debug_vis = False
    env_cfg.actions['joint_pos'].mask_prob = 0.
    env_cfg.sim.nconmax = 192; env_cfg.sim.njmax = 4096
    # These are unchanged native-v2 training termination thresholds, not the
    # separately measured physical-evaluation20mm/20degree quality criteria.
    env_cfg.terminations['object_position_error'].params['threshold'] = .05
    env_cfg.terminations['object_orientation_error'] = TerminationTermCfg(
        func=mdp.object_shaft_direction_error_exceeded,
        params={'robot_cfg': SceneEntityCfg('robot', body_names=('right_palm_link',)),
                'object_cfg': SceneEntityCfg('object'), 'threshold_deg': 60.}, time_out=False)
    if env_cfg.decimation != 4 or env_cfg.sim.mujoco.timestep != .005:
        raise ValueError('Native4x5ms cadence differs')
    if set(env_cfg.events) != {'reset_to_reference_state', 'pen_dimensions'} or any(g.enable_corruption for g in env_cfg.observations.values()):
        raise ValueError('Unexpected DR, noise, disturbance or exogenous event')
    for key in ('diameter_range_m', 'total_length_range_m'):
        lo, hi = env_cfg.events['pen_dimensions'].params[key]
        if lo != hi:
            raise ValueError('Pen geometry must be fixed')
    descriptor = {**bundle['descriptor'], 'trained_method': config['method'], 'training_seed': seed,
        'run_role': config['role'], 'config_sha256': config['_config_sha256'],
        'execution_source_pins': config['source_pins']}
    deadline = config['_worker_started_perf'] + config['soft_worker_seconds']
    result = {'schema': 'full151_native_training_completion:2', 'status': 'INITIALIZING',
        'method': config['method'], 'role': config['role'], 'mode': MODE, 'retarget_variant_id': VARIANT,
        'format_version': FORMAT_VERSION, 'physical_reference_qualified': False,
        'source_lineage_N': 1, 'independent_ego_N': 0, 'from_scratch': True,
        'contact_objective': config.get('contact_binding'), 'old_checkpoint_migration': False}
    env = runner = runtime_api = None

    class RecordedRunner(DexSuperMimicRunner):
        def __init__(self, *args, **kwargs):
            self.phase_seconds = defaultdict(float); self.phase_calls = defaultdict(int)
            self.actual_optimizer_steps = defaultdict(int)
            self.full151_prior = None
            super().__init__(*args, **kwargs)
            for name, optimizer in self.optimizers.items():
                optimizer.register_step_post_hook(lambda _o, _a, _k, name=name:
                    self.actual_optimizer_steps.__setitem__(name, self.actual_optimizer_steps[name] + 1))
            base = self.writer
            class Writer:
                def log_scalars(self, values, step):
                    base.log_scalars(values, step)
                    import json
                    with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(jsonable({'iteration': step, **values}), allow_nan=False) + '\n')
                def close(self): base.close()
            self.writer = Writer()

        @contextmanager
        def phase(self, name):
            if time.perf_counter() > deadline:
                raise TimeoutError('Finite full151 training worker deadline; no retry')
            torch.cuda.synchronize(); tick = time.perf_counter()
            emit(status='running', phase=name, iteration=self.iteration,
                 actual_optimizer_steps=dict(self.actual_optimizer_steps))
            try: yield
            finally:
                torch.cuda.synchronize(); self.phase_seconds[name] += time.perf_counter() - tick
                self.phase_calls[name] += 1

        def collect(self, steps):
            with self.phase('collect'):
                start = self.buffer.ptr
                value = super().collect(steps)
                self.full151_latest_phase_evidence = collect_phase_evidence(self, start, steps)
                write(output / 'runtime_replay_evidence_latest.json', runtime_evidence(self, runtime_api))
                return value

        def fit_world_model(self, holdout=False):
            with self.phase('wm_holdout' if holdout else 'wm_train'):
                return super().fit_world_model(holdout)

        def policy_update(self):
            with self.phase('policy'):
                with self.full151_prior.policy_scope(self):
                    stats=super().policy_update()
                stats.update(self.full151_prior.last_metrics)
                return stats

        def save(self, path, next_iteration=None):
            with self.phase('checkpoint'):
                with checkpoint_evidence_scope(self, runtime_api, torch):
                    super().save(path, next_iteration)
                record = {'checkpoint': str(Path(path).resolve()), 'sha256': sha(path),
                    'iteration': self.iteration if next_iteration is None else next_iteration,
                    'sim_steps': self.sim_steps, 'actual_optimizer_steps': dict(self.actual_optimizer_steps),
                    'worker_elapsed_seconds': time.perf_counter() - started,
                    'selection': 'declared iteration only; no quality selection'}
                write(Path(path).with_suffix('.meta.json'), record)
                import json
                with (output / 'checkpoint_index.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')
                emit(status='running', phase='checkpoint_committed', iteration=record['iteration'], checkpoint=record)

    try:
        with full151_runtime(ManagerBasedRlEnv, SingleHandTrackingCommand, PenMDP,
                             DexSuperMimicRunner, bundle['arrays'], descriptor) as runtime_api:
            with full151_controller(ManagerBasedRlEnv, PenMDP, DexSuperMimicRunner,
                                    bundle['baseline_u150'], descriptor):
                emit(status='initializing', phase='full151_training_environment')
                env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
                mujoco.mj_saveModel(env.sim.mj_model, str(output / 'nominal_model.mjb'), None)
                if sha(output / 'nominal_model.mjb') != bundle['model']['sha256']:
                    raise ValueError('Native nominal model differs from frozen source')
                install_actual_context(env, bundle, output)
                emit(status='initializing', phase='full151_runner_and_first_reset')
                runner = RecordedRunner(env, cfg, output / 'train'); reader = runner.reader
                if runner.contact_dim != 15 or not bool((runner.observations.disturbance(runner.observations.auxiliary()) == 0).all()):
                    raise ValueError('Native self15/exogenous input contract differs')
                xi = reader.xi.detach().cpu().numpy(); expected = np.asarray(bundle['context']['reader_context'])[0]
                if xi.shape != (128, 46) or not np.array_equal(xi, np.broadcast_to(expected, xi.shape)):
                    raise ValueError('All128 native reader xi46 rows must match frozen actual8')
                np.savez_compressed(output / 'fixed_physics_context.npz',
                    **{k: getattr(env.sim.wp_model, k).numpy() for k in ACTUAL_FIELDS}, reader_context=xi)
                write(output / 'context_names.json', {'reader_context': reader.context_names,
                    'memory_dims': PEN_MEMORY_DIMS, 'selfcollision_head_dim': 15})
                write(output / 'controller_descriptor.json', reader.mdp.full151_controller_metadata)
                write(output / 'runtime_reset_readback.json', env._full151_runtime_last_reset)
                before, fingerprints = parameter_fingerprints(runner.modules())
                runner.full151_initial_parameter_fingerprints = fingerprints
                write(output / 'initial_parameters.json', fingerprints)
                runner.full151_prior=construct_prior(config,output,cfg.device)
                write(output/'prior_metadata.json',runner.full151_prior.evidence())
                emit(status='running', phase='learn'); runner.learn(); torch.cuda.synchronize()
                changes = {name: sum(float((p.detach().cpu() - old).double().square().sum())
                    for p, old in zip(module.parameters(), before[name])) ** .5
                    for name, module in runner.modules().items()}
                for name in MODULE_KEYS:
                    if changes[name] == 0 or runner.actual_optimizer_steps[name] == 0:
                        raise ValueError('Missing final actual optimizer change: ' + name)
                if not all(bool(torch.isfinite(p).all()) for module in runner.modules().values() for p in module.parameters()):
                    raise FloatingPointError('Nonfinite final parameters')
                checkpoint = output / 'train/model_final.pt'
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if (state['format_version'] != FORMAT_VERSION or state['iteration'] != cfg.max_iterations
                        or any(key not in state for key in ('full151_controller', 'full151_runtime_evidence',
                            'full151_initial_parameter_fingerprints', 'full151_capacity_evidence','full151_prior_evidence'))):
                    raise ValueError('Full151105 final checkpoint evidence missing')
                result.update(status='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY',
                    checkpoint=pin(checkpoint), final_iteration=state['iteration'],
                    reference_signature=state['reference_signature'], sim_steps=runner.sim_steps,
                    parameter_delta_l2=changes, runtime_evidence=runtime_evidence(runner, runtime_api),
                    torch_peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                    full151_prior_evidence=runner.full151_prior.evidence())
                return result
    except BaseException as exc:
        result.update(status='FAILED_ENGINEERING_PARTIAL', error=repr(exc)); raise
    finally:
        result['native_call_wall_seconds_excluding_imports'] = time.perf_counter() - started
        cleanup_errors = []
        if runner is not None:
            result.update(phase_seconds=dict(runner.phase_seconds), phase_calls=dict(runner.phase_calls),
                          actual_optimizer_steps=dict(runner.actual_optimizer_steps))
            try: runner.close()
            except BaseException as exc: cleanup_errors.append({'stage': 'runner.close', 'error': repr(exc)})
        if env is not None:
            try: env.close()
            except BaseException as exc: cleanup_errors.append({'stage': 'env.close', 'error': repr(exc)})
        result['cleanup_errors'] = cleanup_errors
        if cleanup_errors: result['status'] = 'FAILED_ENGINEERING_CLEANUP_UNQUALIFIED'
        write(output / 'result.json', result)
        if cleanup_errors and 'error' not in result:
            raise RuntimeError('Full151 native cleanup failed; retained result, no retry')
