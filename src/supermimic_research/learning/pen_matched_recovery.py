"""Explicit parent200 budgets and failure evidence;no change to PPO or physics."""
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from .native_pen_ppo import cpu_tree, tensor_tree_equal
from .pen_support_ppo import CENTERED_SCHEMA

SCHEMA = 'pen_long_control_matched_recovery_training:1'
ARMS = ['RESIDUAL', 'SUPPORT']


def saved_bytes_equal(a, b):
    if isinstance(a, (float, np.floating)):
        return type(a) == type(b) and np.asarray(a).tobytes() == np.asarray(b).tobytes()
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, torch.Tensor):
        return tensor_tree_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(saved_bytes_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(saved_bytes_equal(x, y) for x, y in zip(a, b))
    from supermimic_research.runtime.pen_portable_state import scalar_equal
    return scalar_equal(a, b)


def json_metadata(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    if isinstance(value, dict):
        return {k: json_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metadata(v) for v in value]
    return value


def write_metadata(path, value):
    Path(path).write_text(json.dumps(json_metadata(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def budget(cfg):
    if (cfg['schema'], cfg['study'], cfg['start_iteration'], cfg['final_iteration'],
            cfg['num_envs'], cfg['rollout_controls'], cfg['checkpoint_interval'], cfg['arm_order']) != (
            1, 'MATCHED_RESUME_200', 200, 2000, 32, 64, 100, ARMS):
        raise ValueError('Registered matched200->2000/32-world/64-control budget required')
    return dict(remaining_iterations_per_arm=1800, added_native_per_arm=29491200,
                added_optimizer_per_arm=57600, added_transitions_per_arm=3686400,
                added_native_total=58982400, added_optimizer_total=115200,
                reused_prefix_native_total=6553600, reused_prefix_optimizer_total=12800,
                causal_policy_native_total=65536000, causal_policy_optimizer_total=128000)


def validate_parent(state, arm, objective_pin):
    if (state['schema'], state['metadata']['arm'], state['iteration'], state['optimizer_steps'],
            state['training_transitions'], state['native']['worlds']) != (CENTERED_SCHEMA, arm, 200, 6400, 409600, 32):
        raise ValueError('Explicit original arm200 state required')
    if (state['metadata']['objective_config'] != objective_pin or state['metadata']['support_weight'] != (0. if arm == 'RESIDUAL' else .2)
            or state['metadata']['fixed_frame_features'] != 'CACHED' or not state['normalizers_ready']
            or not state['optimizer']['state'] or any(int(v['step']) != 6400 for v in state['optimizer']['state'].values())):
        raise ValueError('Parent objective/arm/features/Adam differs')
    verify_source_snapshot(state['native']['sources'], live=True)


def validate_initial_pair(residual, support):
    metadata = copy.deepcopy(residual['metadata'])
    metadata.update(arm='SUPPORT', support_weight=.2)
    if (metadata != support['metadata'] or residual['config'] != support['config']
            or not tensor_tree_equal(residual['modules'], support['modules'])
            or not tensor_tree_equal(residual['optimizer'], support['optimizer'])
            or residual['optimizer']['state'] or any(s['iteration'] != 0 for s in [residual, support])):
        raise ValueError('Original common untrained pair differs')


class WorkCounter:
    def __init__(self):
        self.native_attempted = self.native = self.optimizer_attempted = self.optimizer = 0

    def native_call(self, original, sim):
        self.native_attempted += sim.num_envs
        result = original(sim)
        self.native += sim.num_envs
        return result

    def bind_optimizer(self, optimizer):
        original = optimizer.step

        def step(*args, **kwargs):
            self.optimizer_attempted += 1
            result = original(*args, **kwargs)
            self.optimizer += 1
            return result

        optimizer.step = step

    def metadata(self):
        return dict(actual_native_integrations=self.native, optimizer_steps=self.optimizer,
                    attempted_native_integrations=self.native_attempted, attempted_optimizer_steps=self.optimizer_attempted,
                    completed_call_accounting_exact=self.native == self.native_attempted and self.optimizer == self.optimizer_attempted)


def checked_health(learner, path):
    health = learner.consistency()
    # Persist all flags BEFORE a caller may decide to reject this rollout.
    write_metadata(path, health)
    if health['passed'] != health['total'] or not all(health['checks'].values()):
        raise ValueError('Collector failed before optimizer: ' + repr([k for k, v in health['checks'].items() if not v]))
    return health


def persist_failed_learner(learner, folder, receipt, *, native_capture=None):
    """Diagnostic tensors may be nonfinite/incomplete;none are resume checkpoints."""
    result = {}
    if learner is None:
        return result
    memory = learner.memory
    # On collection exceptions even the prefix may have incomplete base fields.
    # Keep raw bytes explicitly untrusted instead of inventing completed rows.
    complete = bool(getattr(learner, '_rollout_complete', False))
    rows = int(getattr(learner, '_captured_rows', 0))
    if memory is not None:
        path = folder / 'failed_memory_NOT_RESUMABLE.pt'
        prefix = {k: (v[:rows] if not complete and isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == 64 else v)
                  for k, v in memory.items()}
        torch.save(dict(schema='failed_collector_memory_NOT_RESUMABLE:1', memory=cpu_tree(prefix),
                        rollout_complete=complete, captured_contact_rows=rows,
                        incomplete_fields_may_include_uninitialized_storage=not complete), path)
        result['failed_memory'] = file_pin(path)
    path = folder / 'failed_policy_NOT_RESUMABLE.pt'
    torch.save(dict(schema='failed_policy_NOT_RESUMABLE:1',
                    modules=dict(actor=cpu_tree(learner.actor.state_dict()), critic=cpu_tree(learner.critic.state_dict())),
                    optimizer=cpu_tree(learner.optimizer.state_dict()), observations=cpu_tree(learner.observations),
                    iteration=learner.iteration, optimizer_steps=learner.optimizer_steps,
                    training_transitions=learner.training_transitions), path)
    result['failed_policy'] = file_pin(path)
    try:
        if native_capture is None:
            from supermimic_research.runtime.owned_pen_state import capture as native_capture
        path = folder / 'failed_native_NOT_RESUMABLE.pt'
        torch.save(dict(schema='failed_native_diagnostic_NOT_RESUMABLE:1', native=native_capture(learner.adapter, receipt)), path)
        result['failed_native'] = file_pin(path)
    except Exception as error:
        result['failed_native_capture_error'] = dict(type=type(error).__name__, message=str(error))
    return result


def validate_recovery_result(report, objective_pin, registration_pin):
    if (report['schema'] != SCHEMA or report['config'] != objective_pin or report['registration'] != registration_pin
            or report['passed'] != 5 or report['total'] != 5 or not all(report['checks'].values())
            or report['actual_native_integrations'] != 58982400 or report['optimizer_steps'] != 115200
            or [a['arm'] for a in report['arms']] != ARMS):
        raise ValueError('Complete registered matched-recovery result required')
    registration = json.loads(verify_file_pin(registration_pin).read_text())
    if (registration['schema'] != 'pen_matched_recovery_registration:1'
            or report['recovery_config'] != registration['config']
            or registration['inputs']['objective_config'] != objective_pin
            or report['causal_policy_budget'] != registration['budgets']
            or report['unique_named_lineage_total'] != dict(native_integrations=95228736, optimizer_steps=185952)):
        raise ValueError('Recovery registration,causal budget or unique lineage accounting differs')
    for arm in report['arms']:
        if (arm['parent'] != registration['parents'][arm['arm']]
                or (arm['start_iteration'], arm['iterations'], arm['added_iterations'], arm['actual_native_integrations'],
                    arm['added_optimizer_steps'], arm['optimizer_steps'], arm['training_transitions']) != (
                    200, 2000, 1800, 29491200, 57600, 64000, 4096000)):
            raise ValueError('Recovery parent or cumulative/additional arm budget differs')
