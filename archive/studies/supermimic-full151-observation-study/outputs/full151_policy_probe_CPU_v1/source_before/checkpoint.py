"""Strict new-format105 evaluation Actor consumer; never resume or migration.

No actual full151 training/checkpoint has been created by this module. Future
saved payloads must carry the new namespace, signatures and from-scratch evidence.
"""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
import numpy as np
from .io import canonical, checked, loadnp, pin, read
from .metadata import FORMAT_VERSION,MODE,VARIANT,DESCRIPTOR_SCHEMA
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


def validate_checkpoint_namespace(state):
    """Early fail-closed header guard, independently CPU-fixture-testable."""
    _require(type(state) is dict and type(state.get('format_version')) is int and state['format_version']==FORMAT_VERSION==105,
             'Only new original105 is supported; old3/103/104 and integer-relabelled old checkpoints are rejected')
    _require(not any(k in state for k in ('rough_controller','c2_controller','rough_runtime_evidence','rough_initial_parameter_fingerprints')),
             'Legacy checkpoint namespaces cannot be migrated or relabelled')
    for k in ('full151_controller','full151_runtime_evidence','full151_initial_parameter_fingerprints'):
        _require(type(state.get(k)) is dict and bool(state[k]),'Missing future full151 saved provenance: '+k)
        _canonical(state[k])
    fp=state['full151_initial_parameter_fingerprints']
    _require(set(fp)=={'wm','fail_head','actor','critic'},'Require all4 from-scratch module fingerprints')
    for name,v in fp.items():
        _require(type(v) is dict and set(v)=={'sha256','tensors','parameters'},'Initial fingerprint schema differs: '+name)
        _require(type(v['sha256']) is str and len(v['sha256'])==64 and all(c in '0123456789abcdef' for c in v['sha256']),'Invalid initial fingerprint hash')
        _require(all(type(v[k]) is int and v[k]>0 for k in ('tensors','parameters')),'Initial fingerprint counts must be positive native integers')
    c=state['full151_controller']
    _require(c.get('schema')=='full151_knownrough_baseline_EMA_C2:1' and c.get('mode')==MODE and c.get('retarget_variant_id')==VARIANT,'Wrong105 controller schema/mode/variant')
    _require(type(c.get('format_version')) is int and c['format_version']==105 and c.get('exact_resume_supported') is False,'Wrong105 from-scratch-only ABI')
    root=c.get('root_descriptor');_require(type(root) is dict,'New root descriptor required')
    _require(root.get('schema')==DESCRIPTOR_SCHEMA and root.get('mode')==MODE and root.get('retarget_variant_id')==VARIANT and root.get('physical_reference_qualified') is False,'Old or physically upgraded descriptor is forbidden')
    _require(c.get('baseline_shape')==[150,20] and c.get('active_intervals')==[0,150] and type(c.get('terminal_observation')) is int and c['terminal_observation']==150,'Full151 interval/terminal ABI mismatch')
    _require(c.get('inactive_imagination')=='finite u149 placeholder; runtime must keep survival0 and mask dead losses','Full151 inactive target must be u149')
    return c


def validate_controller_metadata(state,expected,expected_method):
    """Pure signed105 controller/bundle-metadata contract, not a health audit."""
    _require(expected_method in ('A1L','P_ONLY','A2'),'Unknown expected training arm')
    c=validate_checkpoint_namespace(state);root=c['root_descriptor']
    base={k:v for k,v in c.items() if k not in ('latest_actualu_evidence','reference_controller_signature')}
    _require(c.get('reference_controller_signature')==state.get('reference_signature')==_digest(base),'Controller canonical signature mismatch')
    _require(c.get('root_descriptor_sha256')==_digest(root),'Root descriptor digest mismatch')
    extra={'trained_method','training_seed','run_role','config_sha256','execution_source_pins'}
    _require(set(root)==set(expected)|extra,'Unexpected or missing root descriptor fields')
    _require(all(_canonical({'value':root[k]})==_canonical({'value':v}) for k,v in expected.items()),'Root bundle/source/init/phase signature differs')
    _require(expected.get('input_signature_sha256')==_digest({k:v for k,v in expected.items() if k!='input_signature_sha256'}),'Bundle input signature mismatch')
    _require(root['trained_method']==expected_method and type(root['training_seed']) is int and type(state['config']['seed']) is int and root['training_seed']==state['config']['seed'],'Checkpoint arm/seed mismatch')
    _require(root['run_role'] in ('engineering_smoke','primary_matched'),'Future105 checkpoint must be a declared training arm')
    _require(type(root['config_sha256']) is str and len(root['config_sha256'])==64 and all(x in '0123456789abcdef' for x in root['config_sha256']),'Source config digest is missing')
    _require(type(root['execution_source_pins']) is dict and root['execution_source_pins'],'Executed new105 source pins are missing')
    for value in root['execution_source_pins'].values():checked(value)
    controller_pin=pin(Path(__file__).with_name('controller.py'))
    _require(c.get('controller_source_sha256')==controller_pin['sha256'] and c.get('pure_projection_sha256')==PURE_SHA256==pin(PURE_PATH)['sha256'],'New105 controller/projection source differs')
    _require(any(value==controller_pin for value in root['execution_source_pins'].values()),'Executed full151 controller pin absent')
    _require(any(value['sha256']==PURE_SHA256 and Path(value['path']).resolve()==PURE_PATH.resolve() for value in root['execution_source_pins'].values()),'Executed projection pin absent')
    required={'schema':'full151_knownrough_baseline_EMA_C2:1','kind':KIND,'mode':MODE,'retarget_variant_id':VARIANT,
              'format_version':105,'exact_resume_supported':False,'margin_rad':.02,'baseline_shape':[150,20],'active_intervals':[0,150],'terminal_observation':150}
    for key,value in required.items():_require(_canonical({'v':c[key]})==_canonical({'v':value}),'Controller ABI mismatch: '+key)
    return c,root


def _validated_bundle(bundle):
    from .bundle import load_bundle
    _require(type(bundle) is dict and 'descriptor' in bundle, 'Use a loaded validated new-mode bundle')
    source = bundle['descriptor']['input_bundle']
    actual = load_bundle(checked(source), source['sha256'])
    _require(actual['descriptor'].get('schema')==DESCRIPTOR_SCHEMA and actual['descriptor'].get('mode')==MODE and actual['descriptor'].get('retarget_variant_id')==VARIANT and actual['descriptor'].get('physical_reference_qualified') is False,'Exact known-rough full151 bundle is required')
    _require(_canonical(bundle['descriptor']) == _canonical(actual['descriptor']), 'Caller bundle descriptor changed')
    _require(bundle['model'] == actual['model'], 'Caller model alias changed')
    for group in ('reference', 'init', 'control'):
        a, b = bundle['arrays'][group], actual['arrays'][group]
        _require(set(a) == set(b) and all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in b), 'Caller bundle arrays changed')
    _require(bundle['baseline_u150'].shape==(150,20) and bundle['baseline_u150'].dtype==np.float32 and np.array_equal(bundle['baseline_u150'], actual['baseline_u150']), 'Caller baseline changed')
    return actual


def _actor_payload(state):
    """Exact installed Actor155 architecture and normalization, before load."""
    import torch
    from collections import OrderedDict
    actor_state=state['modules']['actor']
    _require(type(actor_state) in (dict,OrderedDict),'Actor state dict missing')
    for key,shape in {'net.0.weight':(512,155),'net.6.weight':(20,128),'norm._mean':(1,155),'norm._var':(1,155),'norm._std':(1,155)}.items():
        v=actor_state[key]
        _require(isinstance(v,torch.Tensor) and v.device.type=='cpu' and tuple(v.shape)==shape and v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'Actor155->20/norm tensor ABI mismatch: '+key)
    _require(bool((actor_state['norm._var']>=0).all()) and bool((actor_state['norm._std']>=0).all()) and actor_state['norm.count'].dtype==torch.int64 and actor_state['norm.count'].numel()==1 and int(actor_state['norm.count'])>=0,'Invalid empirical normalization state')
    _finite_tree(actor_state,torch)
    return actor_state


def load_actor_checkpoint(checkpoint, bundle, expected_method, device='cpu'):
    """Return (frozen native Actor, CPU receipt), without restoring any trainer.

    Full checkpoint SHA is recorded for the future evaluation plan to bind.
    Budget/optimizer/physical quality qualification belongs to separate audits.
    """
    _require(expected_method in ('A1L', 'P_ONLY', 'A2'), 'Unknown expected method')
    import torch
    path = Path(checkpoint).resolve(strict=True); original_pin = pin(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_namespace(state)
    b = _validated_bundle(bundle)
    import mujoco
    from dex_supermimic.rl.dex_supermimic.networks import Actor
    from dex_supermimic.rl.dex_supermimic.observations import DR_PARAM_NAMES
    tensor_count = _finite_tree(state, torch)
    expected = b['descriptor']
    c,root=validate_controller_metadata(state,expected,expected_method)
    controller_pin = pin(Path(__file__).with_name('controller.py'))
    required = {'schema': 'full151_knownrough_baseline_EMA_C2:1', 'kind': KIND,
                'mode':MODE,'retarget_variant_id':VARIANT,
                'format_version': 105, 'exact_resume_supported': False, 'margin_rad': .02,
                'baseline_shape': [150, 20], 'active_intervals': [0, 150], 'terminal_observation': 150,
                'baseline_u150_FP32_sha256': hashlib.sha256(b['baseline_u150'].tobytes()).hexdigest()}
    for key, value in required.items():
        _require(_canonical({'v': c[key]}) == _canonical({'v': value}), 'Controller ABI mismatch: ' + key)
    config = state['config']
    _require(config['task'] == 'hand2-pen-spin' and config['algorithm'] == 'world_model' and type(config['control_hz']) is int and config['control_hz'] == 50, 'Native task/algorithm/cadence mismatch')
    _require(Path(config['recipe_path']).resolve() == Path(b['native_reference']['recipe_path']).resolve() and Path(config['clips_root']).resolve() == Path(b['native_reference']['clips_root']).resolve(), 'Native recipe/source root mismatch')
    native_reader_pin = b['descriptor']['source_pins']['native_reader_arrays']
    n = loadnp(checked(native_reader_pin)); digest = hashlib.sha256()
    _require(n['joint_pos'].shape==(151,20) and n['object_pose'].shape==(151,7) and n['link_pos'].shape==(151,21,3) and np.array_equal(n['trajectory_lengths'],np.array([151],dtype=n['trajectory_lengths'].dtype)), 'Require new151-frame native Reader arrays; old16 cannot be relabelled')
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
    actor_state = _actor_payload(state)
    before_weights = _tensor_digest(actor_state)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        actor = Actor(obs_dim=155)
        actor.load_state_dict(actor_state, strict=True)
        actor.to(device).eval().requires_grad_(False)
    _require(_tensor_digest(actor.state_dict()) == before_weights, 'Actor weights changed during evaluation load')
    _require(pin(path) == original_pin, 'Checkpoint file changed during load')
    receipt = {'schema': 'full151_105_actor_evaluation_load:1', 'status': 'PASS_EVALUATION_ACTOR_LOAD',
               'checkpoint': original_pin, 'loader': pin(__file__), 'controller': controller_pin,
               'bundle': b['descriptor']['input_bundle'], 'native_reader_arrays': native_reader_pin,
               'method': expected_method, 'seed': root['training_seed'], 'iteration': state['iteration'],
               'format_version': 105, 'reference_signature': state['reference_signature'],
               'input_signature_sha256': expected['input_signature_sha256'], 'actor_state_sha256': before_weights,
               'saved_state_tensor_count_finite': tensor_count, 'actor_observation_dim': 155, 'actor_action_dim': 20,
               'device': str(next(actor.parameters()).device), 'evaluation_only': True,
               'physical_reader_xi_dim': 46, 'native_task_context_dim': expected_context_dim,
               'native_task_frame_dim': native_observations.frame_dim,
               'training_resume_supported': False, 'physical_task_success': 'NOT_ASSESSED',
               'mode':MODE,'retarget_variant_id':VARIANT,'parent_knownrough_status':'Authoritative bundle preserves source SCP overallFAIL and sampled-geometryFAIL; accepted engineering input is not physical qualification.',
               'from_scratch_evidence_scope':'New namespace and all4 initial fingerprint metadata required; raw initialization, optimizer/budget and actual runtime evidence remain a separate105 health audit, never inferred from this Actor load.',
               'budget_health_audit': 'SEPARATE_NEW105_REQUIRED_NOT_REPLACED', 'physics_steps': 0}
    return actor, receipt
