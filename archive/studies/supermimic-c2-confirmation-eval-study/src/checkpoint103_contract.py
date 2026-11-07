"""Read-only header guard for the frozen C2-trained confirmation cohort.

Expected values come from separately frozen run/config/model evidence supplied
by the caller, never from the candidate checkpoint itself. This guard does not
load a network, rewrite a format, validate physical success, or resume training.
"""
from __future__ import annotations

import json
import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_copy(value):
    # Native to_dict retains tuples; frozen JSON config snapshots contain lists.
    # JSON also keeps False distinct from0 and .020 distinct from an arbitrary
    # numeric string. Reject nonfinite or non-JSON metadata rather than guessing.
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (ValueError, TypeError) as error:
        raise ValueError('Checkpoint metadata must be finite JSON data') from error


def _same(left, right):
    return json.dumps(_json_copy(left), sort_keys=True, separators=(',', ':')) == json.dumps(
        _json_copy(right), sort_keys=True, separators=(',', ':'))


def validate_checkpoint(state, model):
    """Return a JSONable validated descriptor without mutating either argument."""
    _require(isinstance(state, dict) and isinstance(model, dict), 'Expected checkpoint/model dictionaries')
    _require(state.get('format_version') == 103, 'Only original format103 C2 checkpoints are accepted; no format3 coercion')
    _require(state.get('iteration') == 2000, 'Only frozen final2000 is evaluated')
    expected = model.get('expected')
    _require(isinstance(expected, dict) and {'native_config', 'reference_signature', 'joint_names', 'controller'} <= set(expected),
             'Independent frozen expected header is required')
    _require(model.get('method') in {'A1L', 'P_ONLY', 'A2'} and model.get('training_seed') in {4, 5, 6},
             'Wrong confirmation method/training seed')
    config = state.get('config')
    _require(isinstance(config, dict) and _same(config, expected['native_config']), 'Native config differs from frozen effective config')
    _require(config.get('task') == 'hand2-pen-spin' and config.get('control_hz') == 50, 'Task/control clock mismatch')
    _require(config.get('seed') == model['training_seed'], 'Actual checkpoint training seed mismatch')
    _require(config.get('checkpoint') is None and config.get('model_checkpoint') is None, 'Warmstart/resume is forbidden')
    _require(isinstance(state.get('reference_signature'), str) and state['reference_signature'] and
             state['reference_signature'] == expected['reference_signature'], 'Reference signature mismatch')
    names = [f'right_finger{finger}_joint{joint}' for finger in range(1, 6) for joint in range(1, 5)]
    _require(state.get('joint_names') == expected['joint_names'] == names, 'Canonical20 joint order mismatch')
    controller = state.get('c2_controller')
    required = {'kind', 'margin_rad', 'format_version', 'exact_resume_supported', 'trained_method', 'base_method',
                'training_seed', 'run_role', 'config_sha256', 'paired_protocol_sha256', 'study_protocol_sha256',
                'native_source_inventory_sha256', 'adapter_sha256', 'pure_projection_sha256', 'entry_sha256',
                'objective', 'target_lower_rad', 'target_upper_rad'}
    wanted = expected['controller']
    _require(isinstance(controller, dict) and isinstance(wanted, dict) and required <= set(wanted),
             'Incomplete independently frozen controller expectation')
    for key, value in wanted.items():
        _require(key in controller and _same(controller[key], value), 'C2 controller mismatch: ' + key)
    _require(controller['kind'] == 'C2_TARGET_INSET' and controller['margin_rad'] == .020 and
             controller['format_version'] == 103 and controller['exact_resume_supported'] is False,
             'Wrong C2 controller kind/margin/format/resume semantics')
    _require(controller['trained_method'] == controller['base_method'] == model['method'] and
             controller['training_seed'] == model['training_seed'] and controller['run_role'] == 'primary_matched_arm',
             'Controller arm/seed/role mismatch')
    _require(controller.get('record_actual_u') is True, 'Training actual-u evidence was not enabled')
    for key in ('target_lower_rad', 'target_upper_rad'):
        values = controller[key]
        _require(isinstance(values, list) and len(values) == 20 and
                 all(type(v) in (int, float) and math.isfinite(v) for v in values), 'Invalid20-joint bounds: ' + key)
    _require(all(a + .020 <= b - .020 for a, b in zip(controller['target_lower_rad'], controller['target_upper_rad'])),
             'Empty inset controller domain')
    modules = state.get('modules')
    _require(isinstance(modules, dict) and isinstance(modules.get('actor'), dict), 'Native Actor state dictionary is required')
    actor = modules['actor']
    for key, shape in {'norm._mean': (1, 155), 'norm._std': (1, 155),
                       'net.0.weight': (512, 155), 'net.6.weight': (20, 128)}.items():
        _require(key in actor and tuple(getattr(actor[key], 'shape', ())) == shape, 'Native Actor155->20 header mismatch: ' + key)
    return {'schema': 'C2_trained_confirmation_checkpoint_header:1', 'validated': True,
            'original_format_version': 103, 'iteration': 2000, 'trained_method': model['method'],
            'training_seed': model['training_seed'], 'reference_signature': state['reference_signature'],
            'joint_names': list(names), 'actor_input_dimension': 155, 'actor_output_dimension': 20,
            'native_config': _json_copy(config), 'controller': _json_copy(controller),
            'scope': 'Frozen header/identity validation only; native Actor strict load and runtime bounds/control/physics checks remain required.'}
