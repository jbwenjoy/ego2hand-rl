"""Fresh one-world format104 Actor evaluation; no training Runner or optimizer.

Importing this module does not start a device. Evaluation retains the trained
controller descriptor and installs the same rough runtime, nominal model,
actual8, analytic reset, and native Manager cadence as the qualified A0.
"""
from __future__ import annotations
import random
import time
from pathlib import Path
import numpy as np
from .io import checked, pin, jsonable
from .metadata import write


def configure_full_horizon_terminations(env_cfg):
    """Validate installed play configuration before removing quality-only stop.

    play=True has already removed object_orientation_error. Reject unknown or
    missing terms instead of silently accepting a different termination model.
    """
    expected = {'time_out', 'unstable', 'object_position_error', 'reference_end'}
    if set(env_cfg.terminations) != expected:
        raise ValueError('Unexpected installed play termination keys')
    env_cfg.terminations.pop('object_position_error')
    if set(env_cfg.terminations) != {'time_out', 'unstable', 'reference_end'}:
        raise ValueError('Full-horizon evaluation termination contract differs')

def run_evaluation(config, bundle, output, emit):
    import torch
    import warp as wp
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP, PenSimReader, PEN_MEMORY_DIMS
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand
    from .runtime_v2 import rough_runtime
    from .controller import rough_controller, _canonical
    from .checkpoint_v2 import load_actor_checkpoint, _tensor_digest
    from .native_v2 import install_actual_context
    from .policy_probe import run_policy_probe

    if config['role'] != 'native_policy_eval' or config['worlds'] != 1:
        raise ValueError('Only a fresh one-world evaluation is supported')
    output = Path(output)
    checkpoint = checked(config['checkpoint'])
    actor, load_receipt = load_actor_checkpoint(checkpoint, bundle,
                                               expected_method=config['method'], device='cpu')
    write(output / 'actor_load.json', load_receipt)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    descriptor = state['rough_controller']['root_descriptor']
    if state['iteration'] != config['checkpoint_iteration'] or descriptor['training_seed'] != config['seed']:
        raise ValueError('Evaluation checkpoint selection/seed differs')
    if descriptor['config_sha256'] != config['training_config']['sha256']:
        raise ValueError('Checkpoint training configuration differs')
    trained_controller = _canonical({k:v for k,v in state['rough_controller'].items()
                                     if k != 'latest_actualu_evidence'})
    del state
    weights_before = _tensor_digest(actor.state_dict())
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.suppress_errors = False
    if not torch.cuda.is_available():
        raise RuntimeError('Selected isolated NVIDIA environment required')
    wp.set_device('cuda:0'); actor.to('cuda:0').eval().requires_grad_(False)
    env_cfg = make_env_cfg(1, 50, play=True,
        recipe_path=Path(bundle['native_reference']['recipe_path']),
        clips_root=Path(bundle['native_reference']['clips_root']))
    env_cfg.seed = seed; env_cfg.auto_reset = False
    motion = env_cfg.commands['motion']
    motion.first_frame_hold_prob = 0.; motion.first_frame_hold_s = (0., 0.)
    motion.trajectory_sampling_mode = 'uniform'; motion.debug_vis = False
    env_cfg.actions['joint_pos'].mask_prob = 0.
    env_cfg.sim.nconmax = 192; env_cfg.sim.njmax = 4096
    # The same complete finite horizon as A0. Quality is scored afterward.
    configure_full_horizon_terminations(env_cfg)
    if set(env_cfg.events) != {'reset_to_reference_state', 'pen_dimensions'} or any(
            group.enable_corruption for group in env_cfg.observations.values()):
        raise ValueError('Unexpected exogenous event, DR, or observation noise')
    for key in ('diameter_range_m', 'total_length_range_m'):
        value = env_cfg.events['pen_dimensions'].params[key]
        if value[0] != value[1]: raise ValueError('Pen geometry must be fixed')
    deadline = config['_worker_started_perf'] + config['soft_worker_seconds']
    env = None; reader = None
    result = {'status':'INITIALIZING_POLICY_EVALUATION', 'method':config['method'],
              'source_lineage_N':1, 'independent_ego_N':0, 'optimizer_updates':0,
              'physical_reference_qualified':False, 'checkpoint':config['checkpoint']}
    started = time.perf_counter()
    try:
        with rough_runtime(ManagerBasedRlEnv, SingleHandTrackingCommand, PenMDP,
                           DexSuperMimicRunner, bundle['arrays'], descriptor):
            with rough_controller(ManagerBasedRlEnv, PenMDP, DexSuperMimicRunner,
                                  bundle['baseline_u15'], descriptor):
                emit(status='initializing', phase='native_policy_environment')
                env = ManagerBasedRlEnv(env_cfg, device='cuda:0')
                mujoco.mj_saveModel(env.sim.mj_model, str(output/'nominal_model.mjb'), None)
                if pin(output/'nominal_model.mjb')['sha256'] != bundle['model']['sha256']:
                    raise ValueError('Native nominal MJB differs from qualified A0')
                install_actual_context(env, bundle, output)
                initial_obs, _ = env.reset(seed=seed)
                reader = PenSimReader(env); reader.attach_capture()
                wanted = np.broadcast_to(bundle['context']['reader_context'][0], reader.xi.shape)
                if not np.array_equal(reader.xi.cpu().numpy(), wanted):
                    raise ValueError('Reader actual physical context differs')
                controller = reader.mdp.rough_controller_metadata
                if _canonical({k:v for k,v in controller.items() if k != 'latest_actualu_evidence'}) != trained_controller:
                    raise ValueError('Evaluation controller does not match trained descriptor/signature')
                write(output/'controller_descriptor.json', controller)
                write(output/'context_names.json', {'reader_context':reader.context_names,
                                                     'memory_dims':PEN_MEMORY_DIMS})
                np.savez_compressed(output/'reader_context.npz', reader_context=reader.xi.cpu().numpy())
                result.update(run_policy_probe(env, reader, actor, bundle, output,
                                               emit, deadline, initial_obs=initial_obs))
                if _tensor_digest(actor.state_dict()) != weights_before:
                    raise ValueError('Actor weights/normalization changed during evaluation')
                if pin(checkpoint) != config['checkpoint']:
                    raise ValueError('Evaluation changed checkpoint file')
                write(output/'actor_unchanged.json', {'status':'PASS', 'actor_state_sha256':weights_before,
                    'checkpoint':pin(checkpoint), 'normalization_updated':False, 'optimizer_objects_created':0})
                return result
    except BaseException as exc:
        result.update(status='FAILED_POLICY_EVALUATION_ENGINEERING_PARTIAL', error=repr(exc))
        raise
    finally:
        errors=[]
        if reader is not None:
            try: reader.detach_capture()
            except BaseException as exc: errors.append({'stage':'detach_capture','error':repr(exc)})
        if env is not None:
            try: env.close()
            except BaseException as exc: errors.append({'stage':'env.close','error':repr(exc)})
        result.update(cleanup_errors=errors, native_seconds_excluding_imports=time.perf_counter()-started)
        if errors: result['status']='FAILED_POLICY_EVALUATION_CLEANUP_UNQUALIFIED'
        write(output/'result.json', jsonable(result))
        if errors and 'error' not in result: raise RuntimeError('Evaluation cleanup failed; no retry')
