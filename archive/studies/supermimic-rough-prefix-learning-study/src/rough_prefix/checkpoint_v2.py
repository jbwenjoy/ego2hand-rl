"""Strict format104 evaluation-only Actor loader; never training resume."""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
import numpy as np
from .io import canonical, checked, loadnp, pin, read
from .bundle_v2 import load_bundle
from types import SimpleNamespace
from .controller import _canonical, _digest, KIND, PURE_PATH, PURE_SHA256


def _require(ok, message):
    if not ok: raise ValueError(message)


def _tensor_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def _finite_tree(value, torch):
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), 'Nonfinite saved state tensor')
        return 1
    if isinstance(value, np.ndarray):
        _require(value.dtype.kind not in 'fc' or np.isfinite(value).all(), 'Nonfinite saved numpy payload')
    elif isinstance(value, dict):
        return sum(_finite_tree(v, torch) for v in value.values())
    elif isinstance(value, (tuple, list)):
        return sum(_finite_tree(v, torch) for v in value)
    elif isinstance(value, (float, np.floating)):
        _require(math.isfinite(value), 'Nonfinite saved scalar')
    return 0


def _validated_bundle(bundle):
    _require(type(bundle) is dict and 'descriptor' in bundle, 'Use a loaded validated new-mode bundle')
    source = bundle['descriptor']['input_bundle']
    actual = load_bundle(checked(source), source['sha256'])
    _require(_canonical(bundle['descriptor']) == _canonical(actual['descriptor']), 'Caller bundle descriptor changed')
    _require(bundle['model'] == actual['model'], 'Caller model alias changed')
    for group in ('reference', 'init', 'control'):
        a, b = bundle['arrays'][group], actual['arrays'][group]
        _require(set(a) == set(b) and all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in b), 'Caller bundle arrays changed')
    _require(np.array_equal(bundle['baseline_u15'], actual['baseline_u15']), 'Caller baseline changed')
    return actual


def load_actor_checkpoint(checkpoint, bundle, expected_method, device='cpu'):
    """Return (frozen native Actor, CPU receipt), without restoring any trainer.

    Full checkpoint SHA is recorded for the future evaluation plan to bind.
    Budget/optimizer/physical quality qualification belongs to separate audits.
    """
    _require(expected_method in ('A1L', 'P_ONLY', 'A2'), 'Unknown expected method')
    b = _validated_bundle(bundle)
    import torch
    import mujoco
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    from dex_supermimic.rl.dex_supermimic.observations import DR_PARAM_NAMES
    path = Path(checkpoint).resolve(strict=True); original_pin = pin(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    _require(type(state) is dict and type(state.get('format_version')) is int and state['format_version'] == 104, 'Only original format104 is evaluation-loadable; old3/103 rejected')
    tensor_count = _finite_tree(state, torch)
    c = state['rough_controller']; root = c['root_descriptor']
    _canonical(c)  # Ordinary finite JSON metadata only; tensors are not allowed.
    base = {k: v for k, v in c.items() if k not in ('latest_actualu_evidence', 'reference_controller_signature')}
    _require(c['reference_controller_signature'] == state['reference_signature'] == _digest(base), 'Controller canonical signature mismatch')
    _require(c['root_descriptor_sha256'] == _digest(root), 'Root descriptor digest mismatch')
    expected = b['descriptor']
    extra = {'trained_method', 'training_seed', 'run_role', 'config_sha256', 'execution_source_pins'}
    _require(set(root) == set(expected) | extra, 'Unexpected or missing root descriptor fields')
    _require(all(_canonical({'value': root[k]}) == _canonical({'value': v}) for k, v in expected.items()), 'Root data/reference/init/baseline/velocity/input signature differs from bundle')
    base_input = {k: v for k, v in expected.items() if k != 'input_signature_sha256'}
    _require(expected['input_signature_sha256'] == _digest(base_input), 'Bundle input signature mismatch')
    _require(root['trained_method'] == expected_method and type(root['training_seed']) is int and root['training_seed'] == state['config']['seed'], 'Checkpoint arm/seed mismatch')
    _require(root['run_role'] in ('engineering_smoke', 'primary_matched'), 'Checkpoint is not a training arm')
    _require(type(root['config_sha256']) is str and len(root['config_sha256']) == 64 and all(x in '0123456789abcdef' for x in root['config_sha256']), 'Missing source config digest')
    _require(type(root['execution_source_pins']) is dict and root['execution_source_pins'], 'Missing executed source pins')
    for value in root['execution_source_pins'].values(): checked(value)
    controller_pin = pin(Path(__file__).with_name('controller.py'))
    _require(c['controller_source_sha256'] == controller_pin['sha256'] and c['pure_projection_sha256'] == PURE_SHA256 == pin(PURE_PATH)['sha256'], 'Controller/projection source differs')
    _require(any(p == controller_pin for p in root['execution_source_pins'].values()), 'Executed controller source pin absent')
    _require(any(p['sha256'] == PURE_SHA256 and Path(p['path']).resolve() == PURE_PATH.resolve() for p in root['execution_source_pins'].values()), 'Executed pure projection pin absent')
    required = {'schema': 'rough_prefix_baseline_EMA_C2:1', 'kind': KIND,
                'format_version': 104, 'exact_resume_supported': False, 'margin_rad': .02,
                'baseline_shape': [15, 20], 'active_intervals': [0, 15], 'terminal_observation': 15,
                'baseline_u15_FP32_sha256': hashlib.sha256(b['baseline_u15'].tobytes()).hexdigest()}
    for key, value in required.items():
        _require(_canonical({'v': c[key]}) == _canonical({'v': value}), 'Controller ABI mismatch: ' + key)
    config = state['config']
    _require(config['task'] == 'hand2-pen-spin' and config['algorithm'] == 'world_model' and type(config['control_hz']) is int and config['control_hz'] == 50, 'Native task/algorithm/cadence mismatch')
    _require(Path(config['recipe_path']).resolve() == Path(b['native_reference']['recipe_path']).resolve() and Path(config['clips_root']).resolve() == Path(b['native_reference']['clips_root']).resolve(), 'Native recipe/source root mismatch')
    finalizer = read(checked(b['descriptor']['source_pins']['native_finalizer']))
    native_reader_pin = finalizer['input_pins']['saved_actual_native_reader_arrays']
    n = loadnp(checked(native_reader_pin)); digest = hashlib.sha256()
    for field in ('joint_pos', 'object_pose', 'link_pos'): digest.update(n[field].tobytes())
    digest.update(n['trajectory_lengths'].tobytes())
    _require(c['native_reference_signature'] == digest.hexdigest(), 'Native fixed-wrist reference signature mismatch')
    model = mujoco.MjModel.from_binary_path(str(checked(b['model'])))
    names = [f'right_finger{f}_joint{j}' for f in range(1, 6) for j in range(1, 5)]
    joints = np.array([model.joint('robot/' + name).id for name in names])
    actuators = np.array([np.flatnonzero((model.actuator_trnid[:, 0] == j) & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)).item() for j in joints])
    lo = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[actuators, 0]).astype(np.float32)
    hi = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[actuators, 1]).astype(np.float32)
    _require(c['target_lower_rad'] == lo.tolist() and c['target_upper_rad'] == hi.tolist(), 'Controller hard-intersect-ctrl bounds mismatch')
    # Native context_dim is the FailureHead/task context, not physical xi.
    # Execute the installed PenObservations constructor on a CPU reader stub;
    # its context_dim is xi46 + sum(base_terms)=301 + cage20/time1 = 368.
    from dex_supermimic.rl.dex_supermimic.pen import PenObservations
    physical_xi = np.asarray(b['context']['reader_context'])
    _require(physical_xi.ndim == 2 and physical_xi.shape[1] == 46 and np.isfinite(physical_xi).all(), 'Physical reader xi46 mismatch')
    native_observations = PenObservations(SimpleNamespace(xi=torch.as_tensor(physical_xi[:1]), mdp=None))
    expected_context_dim = native_observations.context_dim
    _require(state['joint_names'] == names and type(state['context_dim']) is int and state['context_dim'] == expected_context_dim and type(state['contact_dim']) is int and state['contact_dim'] == 15, 'Native joint/context/contact ABI mismatch')
    context_names = [(name, 1) for name in DR_PARAM_NAMES] + [('actuator_kp_ratios', 20), ('actuator_kd_ratios', 20)]
    _require(canonical(state['context_names']) == canonical(context_names), 'Saved context name/order mismatch')
    actor_state = state['modules']['actor']
    _require(type(actor_state) is dict or isinstance(actor_state, __import__('collections').OrderedDict), 'Actor state dict missing')
    for key, shape in {'net.0.weight': (512, 155), 'net.6.weight': (20, 128), 'norm._mean': (1, 155), 'norm._var': (1, 155), 'norm._std': (1, 155)}.items():
        _require(tuple(actor_state[key].shape) == shape and actor_state[key].dtype == torch.float32, 'Actor155->20/norm tensor ABI mismatch')
    _require(bool((actor_state['norm._var'] >= 0).all()) and bool((actor_state['norm._std'] >= 0).all()) and actor_state['norm.count'].dtype == torch.int64 and actor_state['norm.count'].numel() == 1 and int(actor_state['norm.count']) >= 0, 'Invalid empirical normalization state')
    before_weights = _tensor_digest(actor_state)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        actor = Actor(obs_dim=155)
        actor.load_state_dict(actor_state, strict=True)
        actor.to(device).eval().requires_grad_(False)
    _require(_tensor_digest(actor.state_dict()) == before_weights, 'Actor weights changed during evaluation load')
    _require(pin(path) == original_pin, 'Checkpoint file changed during load')
    receipt = {'schema': 'rough_prefix_104_actor_evaluation_load:1', 'status': 'PASS_EVALUATION_ACTOR_LOAD',
               'checkpoint': original_pin, 'loader': pin(__file__), 'controller': controller_pin,
               'bundle': b['descriptor']['input_bundle'], 'native_reader_arrays': native_reader_pin,
               'method': expected_method, 'seed': root['training_seed'], 'iteration': state['iteration'],
               'format_version': 104, 'reference_signature': state['reference_signature'],
               'input_signature_sha256': expected['input_signature_sha256'], 'actor_state_sha256': before_weights,
               'saved_state_tensor_count_finite': tensor_count, 'actor_observation_dim': 155, 'actor_action_dim': 20,
               'device': str(next(actor.parameters()).device), 'evaluation_only': True,
               'physical_reader_xi_dim': 46, 'native_task_context_dim': expected_context_dim,
               'native_task_frame_dim': native_observations.frame_dim,
               'training_resume_supported': False, 'physical_task_success': 'NOT_ASSESSED',
               'budget_health_audit': 'SEPARATE_NOT_REPLACED', 'physics_steps': 0}
    return actor, receipt
