"""Explicit full151 known-rough, one-world native A0 orchestration.

Importing this module is CPU/metadata-only. run_native is a device entry point
and must only be called by Root's separately authorized, source-pinned worker.
There is deliberately no training, checkpoint load, Actor or Runner instance.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib.util
import random
import time
import numpy as np

from .io import checked, jsonable, pin, sha
from .metadata import FORMAT_VERSION, MODE, VARIANT, write

ACTUAL_FIELDS = ('body_mass', 'body_inertia', 'body_iquat', 'geom_size',
                 'geom_pos', 'geom_friction', 'actuator_gainprm', 'actuator_biasprm')
CAPACITY_SOURCE = Path('D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/capacity.py')
CAPACITY_SHA256 = '7380bef76a4c6af5c65204d0bf9f5921d68826072a71a77656f1b4b2fda3844a'


def validate_a0_inputs(config, bundle):
    """Fail before any framework import for a wrong role/mode or old horizon."""
    if config.get('role') != 'native_a0' or config.get('method') != 'A0':
        raise ValueError('This full151 entry implements only native A0, never training or learned evaluation')
    if config.get('training') != {'num_envs': 1} or type(config['training']['num_envs']) is not int:
        raise ValueError('Exactly one A0 world, no optimizer/training configuration')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        raise ValueError('Explicit nonnegative integer seed required')
    d = bundle['descriptor']
    if d.get('mode') != MODE or d.get('retarget_variant_id') != VARIANT or d.get('physical_reference_qualified') is not False:
        raise ValueError('Distinct full151 physically unqualified known-rough descriptor required')
    u = np.asarray(bundle['baseline_u150'])
    if u.shape != (150, 20) or u.dtype != np.float32 or not np.isfinite(u).all():
        raise ValueError('Exact finite FP32 baseline150 required; old15/padded151 rejected')
    r = bundle['arrays']['reference']
    required = {'q_goal_rad': (151, 20), 'analytic_qd151_rad_s': (151, 20),
                'mapped_object_pose_w': (151, 7), 'planned_time601_s': (601,),
                'object_COM_world_m': (601, 3), 'object_rotation_w': (601, 3, 3),
                'object_v_world_m_s': (601, 3), 'object_omega_world_rad_s': (601, 3)}
    for name, shape in required.items():
        if np.asarray(r[name]).shape != shape or not np.isfinite(r[name]).all():
            raise ValueError('Wrong full151 reference field ' + name)
    from .probe import CLOCK_CONTRACT
    if d.get('clock_contract') != CLOCK_CONTRACT:
        raise ValueError('Root-frozen full151 FP32 clock contract is required')
    if not np.array_equal(u[0], bundle['arrays']['init']['u_planned0_FP32_rad']):
        raise ValueError('Baseline first target differs from requested initialization')


def configure_a0_env(make_env_cfg, bundle, seed):
    """Real native factory, with exact reviewed play-termination semantics."""
    native = bundle['native_reference']
    cfg = make_env_cfg(1, 50, play=True, recipe_path=Path(native['recipe_path']),
                       clips_root=Path(native['clips_root']))
    if set(cfg.terminations) != {'time_out', 'unstable', 'object_position_error', 'reference_end'}:
        raise ValueError('Installed play factory termination ABI changed')
    cfg.terminations.pop('object_position_error')
    # play=True already removes orientation. Do not pop it a second time.
    if set(cfg.terminations) != {'time_out', 'unstable', 'reference_end'}:
        raise ValueError('A0 must retain only hard instability and natural timeouts')
    if cfg.decimation != 4 or cfg.sim.mujoco.timestep != .005:
        raise ValueError('Native cadence must remain four5ms steps at50Hz')
    cfg.seed = seed
    cfg.auto_reset = False
    motion = cfg.commands['motion']
    motion.first_frame_hold_prob = 0.
    motion.first_frame_hold_s = (0., 0.)
    motion.trajectory_sampling_mode = 'uniform'
    motion.debug_vis = False
    cfg.actions['joint_pos'].mask_prob = 0.
    cfg.sim.nconmax = 192
    cfg.sim.njmax = 4096
    if set(cfg.events) != {'reset_to_reference_state', 'pen_dimensions'} or any(g.enable_corruption for g in cfg.observations.values()):
        raise ValueError('Unexpected DR, noise, disturbance or exogenous event')
    for key in ('diameter_range_m', 'total_length_range_m'):
        lo, hi = cfg.events['pen_dimensions'].params[key]
        if lo != hi:
            raise ValueError('Pen geometry must be fixed')
    return cfg


def install_actual_context(env, bundle, output):
    """Nominal host model stays nominal; actual8 copied once, double readback."""
    import warp as wp
    import mujoco_warp as mjwarp
    expected = {}
    for name in ACTUAL_FIELDS:
        value = np.asarray(bundle['context'][name])
        if value.ndim < 1 or not len(value) or not np.isfinite(value).all() or not np.array_equal(value, np.broadcast_to(value[:1], value.shape)):
            raise ValueError('Fixed shared actual-context source required: ' + name)
        expected[name] = value[0].copy()
    reads = {}
    def verify(stage):
        result = {}
        for name, value in expected.items():
            got = getattr(env.sim.wp_model, name).numpy()
            wanted = np.broadcast_to(value.astype(got.dtype), got.shape)
            if not np.array_equal(got, wanted):
                raise ValueError('Actual8 readback differs: ' + stage + '/' + name)
            reads[stage + '__' + name] = got.copy()
            result[name] = {'exact_expected_FP32_all_storage_rows': True, 'storage_rows': len(got)}
        return result
    for name, value in expected.items():
        dest = getattr(env.sim.wp_model, name)
        source = np.broadcast_to(value, (dest.shape[0],) + value.shape).copy()
        wp.copy(dest, wp.array(source, dtype=dest.dtype, device=env.device))
    result = {'after_copy': verify('after_copy')}
    mjwarp.set_const_0(env.sim.wp_model, env.sim.wp_data, restore=True)
    result['after_set_const_0'] = verify('after_set_const_0')
    np.savez_compressed(output / 'expected_context_fields.npz', **expected)
    np.savez_compressed(output / 'actual_context_readbacks.npz', **reads)
    write(output / 'actual8field_copy.json', result)
    return result


@contextmanager
def observed_capacity(env):
    """Hash-bound existing observer; it never clears flags or changes capacity."""
    if sha(CAPACITY_SOURCE) != CAPACITY_SHA256:
        raise ValueError('Qualified read-only capacity helper changed')
    spec = importlib.util.spec_from_file_location('_full151_frozen_capacity', CAPACITY_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.capacity_observer(env, stage='full151_A0_constructor_actual8_then_reset',
                                  fixed_capacities={'nconmax': 192, 'njmax': 4096}) as observer:
        env._full151_capacity_observer = observer
        # Keep a host-side handle even if constructor flag rejection occurs
        # before this context yields to its caller. Native finally persists it.
        env._full151_capacity_last_observer = observer
        try:
            # This is extra setup accounting, not a physical step or flag clear.
            record = observer._read('after_constructor_before_actual8')
            observer._raise_if_unqualified(record)
            yield observer
        finally:
            delattr(env, '_full151_capacity_observer')


def run_native(config, bundle, output, emit):
    """Actual A0 entry. No old rough contexts, model load, Actor or optimizer."""
    validate_a0_inputs(config, bundle)
    if config.get('GPU_run_authorized') is not True:
        raise ValueError('Explicit Root GPU authorization is required before native framework imports')
    import torch
    import warp as wp
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP, PenSimReader, PEN_MEMORY_DIMS
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand
    from .runtime import full151_runtime
    from .controller import full151_controller
    from .probe import run_probe
    output = Path(output)
    if not output.is_dir():
        raise ValueError('Root worker must reserve its fresh output before native entry')
    started = time.perf_counter()
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError('Explicitly authorized isolated NVIDIA environment required')
    wp.set_device('cuda:0')
    cfg = configure_a0_env(make_env_cfg, bundle, seed)
    descriptor = {**bundle['descriptor'], 'trained_method': 'A0', 'training_seed': seed,
                  'run_role': 'native_a0', 'config_sha256': config['_config_sha256'],
                  'execution_source_pins': config['source_pins']}
    deadline = config['_worker_started_perf'] + config['soft_worker_seconds']
    env = reader = observer = None
    result = {'status': 'INITIALIZING', 'method': 'A0', 'role': 'native_a0',
              'mode': MODE, 'retarget_variant_id': VARIANT, 'checkpoint_format_reserved_for_new_mode': FORMAT_VERSION,
              'physical_reference_qualified': False, 'source_lineage_N': 1, 'independent_ego_N': 0,
              'Actor_instances': 0, 'Runner_instances': 0, 'optimizer_objects_created': 0,
              'training_updates': 0, 'old_checkpoint_migration': False,
              'counter_scope': 'Probe episode only; constructor, actual8 setup and reset API work is UNKNOWN_NOT_WHOLEPROCESS_OBSERVED.'}
    write(output / 'effective_probe_config.json', jsonable({**result, 'num_envs': 1,
          'control_hz': 50, 'physics_dt_s': .005, 'control_intervals': 150,
          'raw_state_count': 601, 'solver_steps': 600, 'native_postinterval_forwards': 150}))
    try:
        with full151_runtime(ManagerBasedRlEnv, SingleHandTrackingCommand, PenMDP,
                             DexSuperMimicRunner, bundle['arrays'], descriptor):
            with full151_controller(ManagerBasedRlEnv, PenMDP, DexSuperMimicRunner,
                                    bundle['baseline_u150'], descriptor):
                emit(status='initializing', phase='full151_native_environment')
                env = ManagerBasedRlEnv(cfg, device='cuda:0')
                with observed_capacity(env) as observer:
                    mujoco.mj_saveModel(env.sim.mj_model, str(output / 'nominal_model.mjb'), None)
                    if sha(output / 'nominal_model.mjb') != bundle['model']['sha256']:
                        raise ValueError('Native nominal model differs from frozen source')
                    install_actual_context(env, bundle, output)
                    emit(status='initializing', phase='full151_first_reset')
                    env.reset(seed=seed)
                    reader = PenSimReader(env)
                    reader.attach_capture()
                    expected = np.asarray(bundle['context']['reader_context'])[0]
                    xi = reader.xi.detach().cpu().numpy()
                    if xi.shape != (1, 46) or not np.array_equal(xi, np.broadcast_to(expected, xi.shape)):
                        raise ValueError('Native reader physical xi46 differs from fixed actual8')
                    np.savez_compressed(output / 'fixed_physics_context.npz',
                        **{k: getattr(env.sim.wp_model, k).numpy() for k in ACTUAL_FIELDS}, reader_context=xi)
                    write(output / 'context_names.json', {'reader_context': reader.context_names,
                          'memory_dims': PEN_MEMORY_DIMS, 'selfcollision_head_dim': 15})
                    write(output / 'controller_descriptor.json', reader.mdp.full151_controller_metadata)
                    write(output / 'runtime_reset_readback.json', env._full151_runtime_last_reset)
                    emit(status='running', phase='full151_native_A0_episode')
                    result.update(run_probe(env, reader, bundle, output, emit, deadline))
                    return result
    except BaseException as exc:
        result.update(status='FAILED_ENGINEERING_PARTIAL', error=repr(exc))
        raise
    finally:
        result['native_call_wall_seconds_excluding_imports'] = time.perf_counter() - started
        cleanup_errors = []
        if observer is None and env is not None:
            observer = getattr(env, '_full151_capacity_last_observer', None)
        if observer is not None:
            try:
                if not observer.finalized:
                    observer.final_read()
            except BaseException as exc:
                cleanup_errors.append({'stage': 'capacity.final_read', 'error': repr(exc)})
            try: write(output / 'capacity_observer_final.json', observer.snapshot())
            except BaseException as exc: cleanup_errors.append({'stage': 'capacity.snapshot', 'error': repr(exc)})
        if env is not None and hasattr(env, '_full151_capacity_last_observer'):
            delattr(env, '_full151_capacity_last_observer')
        if reader is not None:
            try: reader.detach_capture()
            except BaseException as exc: cleanup_errors.append({'stage': 'reader.detach_capture', 'error': repr(exc)})
        if env is not None:
            try: env.close()
            except BaseException as exc: cleanup_errors.append({'stage': 'env.close', 'error': repr(exc)})
        result['cleanup_errors'] = cleanup_errors
        if cleanup_errors: result['status'] = 'FAILED_ENGINEERING_CLEANUP_UNQUALIFIED'
        write(output / 'result.json', result)
        if cleanup_errors and 'error' not in result:
            raise RuntimeError('Native cleanup failed; retained result, no retry')
