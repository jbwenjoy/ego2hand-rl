"""Fresh new105 full151 Actor evaluation, never a trainer or resume entry.

The caller supplies a source-pinned authorized configuration and a new105
health receipt. CPU metadata validation is safe on import; native construction
occurs only inside run_evaluation after explicit authorization.
"""
from __future__ import annotations
import random
import time
from pathlib import Path
import numpy as np
from .io import checked, canonical, pin, read
from .metadata import MODE, VARIANT, write


def _same_pin(a, b):
    return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def validate_evaluation_inputs(config, bundle):
    """No Actor/Env/device import; health is separate from policy quality."""
    if config.get('role')!='native_policy_eval' or config.get('method') not in ('A1L','P_ONLY','A2'):
        raise ValueError('This stage supports the explicit full151 three-arm v2 evaluation')
    if type(config.get('worlds')) is not int or config['worlds']!=1:
        raise ValueError('One evaluation world required, not a training batch')
    if type(config.get('seed')) is not int or config['seed']<0:
        raise ValueError('Nonnegative integer seed required')
    if type(config.get('checkpoint_iteration')) is not int or config['checkpoint_iteration']<=0:
        raise ValueError('Explicit positive native checkpoint iteration required')
    from .native import validate_a0_inputs
    validate_a0_inputs({'role':'native_a0','method':'A0','seed':config['seed'],'training':{'num_envs':1}},bundle)
    checkpoint=checked(config['checkpoint']);training=read(checked(config['training_config']))
    health=read(checked(config['health']));auditor=checked(config['expected_health_auditor'])
    if health.get('schema')!='full151105_checkpoint_CPU_health:1' or health.get('status')!='PASS_SAVED_ENGINEERING_HEALTH':
        raise ValueError('Require the independent new105 saved engineering health receipt')
    checks=health.get('checks')
    if type(checks) is not dict or not checks or any(type(v) is not bool or not v for v in checks.values()) or type(health.get('passed')) is not int or type(health.get('total')) is not int or health['passed']!=health['total'] or health['total']!=len(checks):
        raise ValueError('Incomplete or malformed new105 health checks')
    if not _same_pin(health['checkpoint'],config['checkpoint']) or not _same_pin(health['config'],config['training_config']) or not _same_pin(health['auditor'],config['expected_health_auditor']):
        raise ValueError('Health checkpoint/config/auditor do not identify actual evaluation inputs')
    if health.get('method')!=config['method'] or type(health.get('seed')) is not int or health['seed']!=config['seed'] or type(health.get('iteration')) is not int or health['iteration']!=config['checkpoint_iteration']:
        raise ValueError('Health method/seed/iteration differs from fixed selection')
    if health.get('role')!=training.get('role') or training['role'] not in ('engineering_smoke','primary_matched'):
        raise ValueError('New105 training role is missing or changed')
    if health.get('mode')!=MODE or health.get('retarget_variant_id')!=VARIANT or not _same_pin(health['input_bundle'],bundle['descriptor']['input_bundle']):
        raise ValueError('Health describes another input mode/bundle')
    if health.get('source_pins')!=training.get('source_pins') or not health.get('source_pins'):
        raise ValueError('Health executed source table differs from training configuration')
    for value in health['source_pins'].values():checked(value)
    if training.get('schema')!='full151_knownrough_native_training_run:2' or training.get('method')!=config['method'] or type(training.get('seed')) is not int or training['seed']!=config['seed'] or not _same_pin(training['input_bundle'],bundle['descriptor']['input_bundle']):
        raise ValueError('Training arm/seed/input-bundle aliases differ')
    if type(training['training'].get('max_iterations')) is not int or training['training']['max_iterations']!=config['checkpoint_iteration'] or health.get('final_iteration_scope') is not True or checkpoint.name!='model_final.pt':
        raise ValueError('Require the fixed final checkpoint, never a quality-selected checkpoint')
    if not Path(health['run_dir']).samefile(training['output']):
        raise ValueError('Health run directory differs from executed training output')
    return {'health':health,'training':training,'checkpoint_path':checkpoint,'auditor_path':auditor}


def run_evaluation(config, bundle, output, emit):
    evidence=validate_evaluation_inputs(config,bundle)
    if config.get('GPU_run_authorized') is not True:
        raise ValueError('Separate explicit Root authorization required before device imports')
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('Use a fresh worker; strict new105 Actor loads on CPU before CUDA initialization')
    from .checkpoint import load_actor_checkpoint, _tensor_digest
    from .controller import full151_controller, _canonical
    from .runtime import full151_runtime
    from .native import configure_a0_env, install_actual_context, observed_capacity, ACTUAL_FIELDS
    from .policy_probe import run_policy_probe
    output=Path(output)
    if not output.is_dir():raise ValueError('Worker must reserve its exact fresh output first')
    checkpoint=evidence['checkpoint_path']
    actor,load_receipt=load_actor_checkpoint(checkpoint,bundle,expected_method=config['method'],device='cpu')
    if load_receipt['device']!='cpu' or load_receipt['format_version']!=105:
        raise ValueError('New105 Actor must first be strictly loaded on CPU')
    write(output/'actor_load.json',load_receipt)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    saved_controller=state['full151_controller'];descriptor=saved_controller['root_descriptor']
    if type(state['iteration']) is not int or state['iteration']!=config['checkpoint_iteration'] or descriptor['training_seed']!=config['seed'] or descriptor['config_sha256']!=config['training_config']['sha256']:
        raise ValueError('Loaded checkpoint training selection differs from bound health/config')
    trained_controller=_canonical({k:v for k,v in saved_controller.items() if k!='latest_actualu_evidence'})
    del state
    weights_before=_tensor_digest(actor.state_dict())
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    if not torch.cuda.is_available():raise RuntimeError('Explicit isolated NVIDIA evaluation environment required')
    import warp as wp
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP,PenSimReader,PEN_MEMORY_DIMS
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand
    wp.set_device('cuda:0');actor.to('cuda:0').eval().requires_grad_(False)
    env_cfg=configure_a0_env(make_env_cfg,bundle,seed)
    deadline=config['_worker_started_perf']+config['soft_worker_seconds']
    env=reader=observer=None;started=time.perf_counter()
    result={'status':'INITIALIZING_POLICY_EVALUATION','method':config['method'],'role':'native_policy_eval',
        'mode':MODE,'retarget_variant_id':VARIANT,'source_lineage_N':1,'independent_ego_N':0,
        'optimizer_updates':0,'Runner_instances':0,'physical_reference_qualified':False,
        'checkpoint':config['checkpoint'],'counter_scope':'Episode only; construction/reset/graph API counts UNKNOWN_NOT_WHOLEPROCESS_OBSERVED'}
    try:
        with full151_runtime(ManagerBasedRlEnv,SingleHandTrackingCommand,PenMDP,DexSuperMimicRunner,bundle['arrays'],descriptor):
            with full151_controller(ManagerBasedRlEnv,PenMDP,DexSuperMimicRunner,bundle['baseline_u150'],descriptor):
                emit(status='initializing',phase='full151_policy_environment')
                env=ManagerBasedRlEnv(env_cfg,device='cuda:0')
                with observed_capacity(env) as observer:
                    mujoco.mj_saveModel(env.sim.mj_model,str(output/'nominal_model.mjb'),None)
                    if pin(output/'nominal_model.mjb')['sha256']!=bundle['model']['sha256']:
                        raise ValueError('Nominal MJB differs from the native151 A0 model')
                    install_actual_context(env,bundle,output)
                    initial_obs,_=env.reset(seed=seed)
                    reader=PenSimReader(env);reader.attach_capture()
                    xi=reader.xi.detach().cpu().numpy().copy()
                    if xi.shape!=(1,46) or not np.array_equal(xi,np.asarray(bundle['context']['reader_context'])[:1]):
                        raise ValueError('Actual reader xi46 differs from frozen actual8')
                    actual_controller=reader.mdp.full151_controller_metadata
                    if _canonical({k:v for k,v in actual_controller.items() if k!='latest_actualu_evidence'})!=trained_controller:
                        raise ValueError('Actual evaluation controller is not the trained105 controller/signature')
                    write(output/'controller_descriptor.json',actual_controller)
                    write(output/'runtime_reset_readback.json',env._full151_runtime_last_reset)
                    write(output/'context_names.json',{'reader_context':reader.context_names,'memory_dims':PEN_MEMORY_DIMS,'selfcollision_head_dim':15})
                    np.savez_compressed(output/'fixed_physics_context.npz',**{k:getattr(env.sim.wp_model,k).numpy() for k in ACTUAL_FIELDS},reader_context=xi)
                    result.update(run_policy_probe(env,reader,actor,bundle,output,emit,deadline,initial_obs=initial_obs))
                    # This is CPU-only saved evidence. The probe copied/checked
                    # Actor state before the final capacity read; no device read
                    # is performed here after that final observation.
                    actor_evidence=read(output/'actor_state_evidence.json')
                    if actor_evidence['all_weights_and_normalizer_unchanged'] is not True or actor_evidence['actor_state_before_sha256']!=weights_before or actor_evidence['actor_state_after_sha256']!=weights_before:
                        raise ValueError('Actor state digest differs from its strict CPU load')
                    if pin(checkpoint)!=config['checkpoint']:raise ValueError('Checkpoint file changed during evaluation')
                    write(output/'actor_unchanged.json',{'status':'PASS','actor_state_sha256':weights_before,'checkpoint':pin(checkpoint),'normalization_updated':False,'optimizer_objects_created':0,'evidence':'Probe CPU snapshots made before final sticky capacity read'})
                    return result
    except BaseException as exc:
        result.update(status='FAILED_POLICY_EVALUATION_ENGINEERING_PARTIAL',error=repr(exc));raise
    finally:
        errors=[]
        if observer is None and env is not None:observer=getattr(env,'_full151_capacity_last_observer',None)
        if observer is not None:
            try:
                if not observer.finalized:observer.final_read()
            except BaseException as exc:errors.append({'stage':'capacity.final_read','error':repr(exc)})
            try:write(output/'capacity_observer_final.json',observer.snapshot())
            except BaseException as exc:errors.append({'stage':'capacity.snapshot','error':repr(exc)})
        if env is not None and hasattr(env,'_full151_capacity_last_observer'):delattr(env,'_full151_capacity_last_observer')
        if reader is not None:
            try:reader.detach_capture()
            except BaseException as exc:errors.append({'stage':'reader.detach_capture','error':repr(exc)})
        if env is not None:
            try:env.close()
            except BaseException as exc:errors.append({'stage':'env.close','error':repr(exc)})
        result.update(cleanup_errors=errors,native_call_wall_seconds_excluding_imports=time.perf_counter()-started)
        if errors:result['status']='FAILED_POLICY_EVALUATION_CLEANUP_UNQUALIFIED'
        write(output/'result.json',result)
        if errors and 'error' not in result:raise RuntimeError('Native105 evaluation cleanup failed; no retry')

