"""Native Manager training entry for the separately versioned rough mode.

Nothing in this module starts a device until run_native is explicitly called.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
import dataclasses,hashlib,importlib.metadata as versions,os,random,time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from .io import sha,jsonable,pin,read,checked
from .metadata import write
from .bundle import FIELDS

def install_actual_context(env,bundle,output):
    """Keep CPU model nominal; copy actual8 on-device and retain both reads."""
    import warp as wp
    import mujoco_warp as mjwarp
    expected={k:bundle['context'][k][0].copy() for k in FIELDS};reads={}
    def verify(stage):
        report={}
        for key,value in expected.items():
            got=getattr(env.sim.wp_model,key).numpy();wanted=np.broadcast_to(value.astype(got.dtype),got.shape)
            if not np.array_equal(got,wanted):raise ValueError('Actual8 mismatch '+stage+'/'+key)
            reads[stage+'__'+key]=got.copy();report[key]={'exact_expected_FP32_all_worlds':True,'world_rows':len(got)}
        return report
    for key,value in expected.items():
        dest=getattr(env.sim.wp_model,key);count=dest.shape[0]
        source=np.broadcast_to(value,(count,)+value.shape).copy()
        wp.copy(dest,wp.array(source,dtype=dest.dtype,device=env.device))
    report={'after_copy':verify('after_copy')}
    mjwarp.set_const_0(env.sim.wp_model,env.sim.wp_data,restore=True)
    report['after_set_const_0']=verify('after_set_const_0')
    np.savez_compressed(output/'expected_context_fields.npz',**expected)
    np.savez_compressed(output/'actual_context_readbacks.npz',**reads)
    write(output/'actual8field_copy.json',report)
    return report

@contextmanager
def imagination_prior(runner,objective):
    original=runner.reward_step;collected=defaultdict(list)
    if objective is None:
        yield collected;return
    def shaped(x1,memory,action,residual,contact,torque,soft=False):
        reward,cage,terms=original(x1,memory,action,residual,contact,torque,soft=soft)
        bonus,extra=objective(x1,memory)
        if bonus.shape!=reward.shape:raise ValueError('Prior reward shape differs')
        for k,v in extra.items():collected[k].append(v.detach().float().mean())
        return reward+.02*bonus,cage,terms
    runner.reward_step=shaped
    try:yield collected
    finally:runner.reward_step=original

def run_native(config,bundle,output,emit):
    import torch
    import warp as wp
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg,SceneEntityCfg
    from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP,PenSimReader,PEN_MEMORY_DIMS
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand
    from dex_supermimic.tasks.pen_spin import mdp
    from .runtime import rough_runtime
    from .controller import rough_controller,FORMAT_VERSION
    started=time.perf_counter();output=Path(output);seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch._dynamo.config.suppress_errors=False
    if not torch.cuda.is_available():raise RuntimeError('Selected isolated NVIDIA environment required')
    wp.set_device('cuda:0')
    defaults=dict(num_envs=128,max_iterations=10,buffer_steps=256,warmup_steps=16,wm_only_iters=2,wm_window=8,wm_batches=2,wm_batch=256,wm_hidden=(512,512,512),pi_batch=128,horizon=8,max_horizon=8,critic_epochs=2,critic_minibatches=2,holdout_fraction=.1,save_interval=10,log_interval=1,profile=True,amp=True,compile=False,compile_policy=False,compile_policy_dynamic=False,action_noise=.1)
    defaults.update(config['training']);defaults.update(task='hand2-pen-spin',algorithm='world_model',recipe_path=bundle['native_reference']['recipe_path'],clips_root=bundle['native_reference']['clips_root'],device='cuda:0',seed=seed,logger='none',contact_weight=.1,failure_head=False,profile=True)
    defaults['wm_hidden']=tuple(defaults['wm_hidden']);is_a0=config['role']=='native_a0'
    if is_a0:
        if config['training']!={'num_envs':1}:raise ValueError('A0 uses one native Manager world and no training config/runner')
        cfg=SimpleNamespace(num_envs=1,recipe_path=defaults['recipe_path'],clips_root=defaults['clips_root'],device='cuda:0')
        write(output/'effective_probe_config.json',{'mode':'one_world_native_Manager_PenSimReader_no_training_runner','num_envs':1,'optimizer_objects_created':0,'source':config['training']})
    else:
        cfg=DexSuperMimicCfg(**defaults)
        if cfg.max_iterations<1 or not 0<=cfg.wm_only_iters<cfg.max_iterations or cfg.save_interval>100:raise ValueError('Invalid finite from-scratch training budget')
        write(output/'effective_training_config.json',cfg.to_dict())
    env_cfg=make_env_cfg(cfg.num_envs,50,play=True,recipe_path=Path(cfg.recipe_path),clips_root=Path(cfg.clips_root))
    env_cfg.seed=seed;env_cfg.auto_reset=False;motion=env_cfg.commands['motion'];motion.first_frame_hold_prob=0.;motion.first_frame_hold_s=(0.,0.);motion.trajectory_sampling_mode='uniform';motion.debug_vis=False
    env_cfg.actions['joint_pos'].mask_prob=0.
    env_cfg.sim.nconmax=192;env_cfg.sim.njmax=4096
    env_cfg.terminations['object_position_error'].params['threshold']=.05
    env_cfg.terminations['object_orientation_error']=TerminationTermCfg(func=mdp.object_shaft_direction_error_exceeded,params={'robot_cfg':SceneEntityCfg('robot',body_names=('right_palm_link',)),'object_cfg':SceneEntityCfg('object'),'threshold_deg':60.},time_out=False)
    if is_a0:
        # Record quality at the full horizon. Never clear pending resets or
        # write a state to continue after a quality-based early termination.
        env_cfg.terminations.pop('object_position_error')
        env_cfg.terminations.pop('object_orientation_error')
    if set(env_cfg.events)!={'reset_to_reference_state','pen_dimensions'} or any(g.enable_corruption for g in env_cfg.observations.values()):raise ValueError('Unexpected DR, noise or exogenous event')
    for key in ('diameter_range_m','total_length_range_m'):
        v=env_cfg.events['pen_dimensions'].params[key]
        if v[0]!=v[1]:raise ValueError('Pen geometry must be fixed')
    descriptor={**bundle['descriptor'],'trained_method':config['method'],'training_seed':seed,'run_role':config['role'],'config_sha256':config['_config_sha256'],'execution_source_pins':config['source_pins']}
    deadline=config['_worker_started_perf']+config['soft_worker_seconds'];env=None;runner=None;a0_reader=None;runtime_api=None;result={'status':'INITIALIZING','method':config['method'],'role':config['role'],'physical_reference_qualified':False,'source_lineage_N':1,'independent_ego_N':0,'no_old_checkpoint_migration':True}
    def runtime_evidence(value):
        policy=runtime_api['policy_evidence'](value) if runtime_api is not None else None
        return jsonable({'replay_counts':getattr(value,'_rough_runtime_replay_counts',None),'latest_window':getattr(value,'_rough_runtime_latest_window',None),'policy':policy,'policy_not_run_is_not_a_pass':policy is None or policy.get('status')=='NOT_RUN'})
    class RecordedRunner(DexSuperMimicRunner):
        def __init__(self,*args,**kwargs):
            self.phase_seconds=defaultdict(float);self.phase_calls=defaultdict(int);self.actual_optimizer_steps=defaultdict(int);self.contact_objective=None
            super().__init__(*args,**kwargs)
            for key,opt in self.optimizers.items():opt.register_step_post_hook(lambda _o,_a,_k,key=key:self.actual_optimizer_steps.__setitem__(key,self.actual_optimizer_steps[key]+1))
            base=self.writer;outer=self
            class Writer:
                def log_scalars(self,values,step):
                    base.log_scalars(values,step)
                    with (output/'metrics.jsonl').open('a',encoding='utf-8') as f:
                        import json
                        f.write(json.dumps(jsonable({'iteration':step,**values}),allow_nan=False)+'\n')
                def close(self):base.close()
            self.writer=Writer()
        @contextmanager
        def phase(self,name):
            if time.perf_counter()>deadline:raise TimeoutError('Finite rough-mode worker deadline; no retry')
            torch.cuda.synchronize();tick=time.perf_counter();emit(status='running',phase=name,iteration=self.iteration,actual_optimizer_steps=dict(self.actual_optimizer_steps))
            try:yield
            finally:
                torch.cuda.synchronize();self.phase_seconds[name]+=time.perf_counter()-tick;self.phase_calls[name]+=1
        def collect(self,steps):
            with self.phase('collect'):
                result=super().collect(steps)
                write(output/'runtime_replay_evidence_latest.json',runtime_evidence(self))
                return result
        def fit_world_model(self,holdout=False):
            with self.phase('wm_holdout' if holdout else 'wm_train'):return super().fit_world_model(holdout)
        def policy_update(self):
            with self.phase('policy'):
                if self.contact_objective is None:return super().policy_update()
                with imagination_prior(self,self.contact_objective) as terms:stats=super().policy_update()
                stats.update({'prior/'+k:torch.stack(v).mean() for k,v in terms.items()});return stats
        def save(self,path,next_iteration=None):
            with self.phase('checkpoint'):
                native_save=torch.save
                evidence=runtime_evidence(self)
                def serialize(state,*args,**kwargs):
                    if state.get('format_version')!=FORMAT_VERSION or 'rough_controller' not in state:raise ValueError('Runtime evidence must follow the new controller serialization boundary')
                    return native_save({**state,'rough_runtime_evidence':evidence,'rough_initial_parameter_fingerprints':getattr(self,'rough_initial_parameter_fingerprints',None)},*args,**kwargs)
                torch.save=serialize
                try:super().save(path,next_iteration)
                finally:torch.save=native_save
                record={'checkpoint':str(Path(path).resolve()),'sha256':sha(path),'iteration':self.iteration if next_iteration is None else next_iteration,'sim_steps':self.sim_steps,'actual_optimizer_steps':dict(self.actual_optimizer_steps),'worker_elapsed_seconds':time.perf_counter()-started,'selection':'declared iteration only; no quality selection'}
                write(Path(path).with_suffix('.meta.json'),record)
                with (output/'checkpoint_index.jsonl').open('a',encoding='utf-8') as f:
                    import json
                    f.write(json.dumps(record)+'\n')
                emit(status='running',phase='checkpoint_committed',iteration=record['iteration'],checkpoint=record)
    try:
        with rough_runtime(ManagerBasedRlEnv,SingleHandTrackingCommand,PenMDP,DexSuperMimicRunner,bundle['arrays'],descriptor) as runtime_api:
            with rough_controller(ManagerBasedRlEnv,PenMDP,DexSuperMimicRunner,bundle['baseline_u15'],descriptor):
                emit(status='initializing',phase='native_environment')
                env=ManagerBasedRlEnv(env_cfg,device=cfg.device)
                mujoco.mj_saveModel(env.sim.mj_model,str(output/'nominal_model.mjb'),None)
                if sha(output/'nominal_model.mjb')!=bundle['model']['sha256']:raise ValueError('Native nominal MJB differs from frozen model')
                install_actual_context(env,bundle,output)
                emit(status='initializing',phase='native_runner_and_first_reset')
                if is_a0:
                    env.reset(seed=seed);a0_reader=PenSimReader(env);a0_reader.attach_capture();reader=a0_reader
                else:
                    runner=RecordedRunner(env,cfg,output/'train');reader=runner.reader
                    if runner.contact_dim!=15 or not bool((runner.observations.disturbance(runner.observations.auxiliary())==0).all()):raise ValueError('Native contact-head/exogenous input contract differs')
                expected_context=bundle['context']['reader_context'][0]
                if not np.array_equal(reader.xi.cpu().numpy(),np.broadcast_to(expected_context,reader.xi.shape)):raise ValueError('Reader physical context differs from fixed actual8 source')
                np.savez_compressed(output/'fixed_physics_context.npz',**{k:getattr(env.sim.wp_model,k).numpy() for k in FIELDS},reader_context=reader.xi.cpu().numpy())
                write(output/'context_names.json',{'reader_context':reader.context_names,'memory_dims':PEN_MEMORY_DIMS})
                write(output/'controller_descriptor.json',reader.mdp.rough_controller_metadata)
                if is_a0:
                    from .probe import run_probe
                    result.update(run_probe(env,None,bundle,output,emit,deadline));return result
                if config['method']!='A1L':
                    from .contact_binding import build_contact_objective,build_penalty_only_objective
                    objective_spec=read(checked(config['contact_binding']))
                    runner.contact_objective=(build_penalty_only_objective if config['method']=='P_ONLY' else build_contact_objective)(objective_spec,device=cfg.device)
                    write(output/'contact_objective_metadata.json',runner.contact_objective.metadata)
                before={k:[p.detach().cpu().clone() for p in v.parameters()] for k,v in runner.modules().items()}
                fingerprints={}
                for k,values in before.items():
                    h=hashlib.sha256()
                    for v in values:h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
                    fingerprints[k]={'sha256':h.hexdigest(),'tensors':len(values),'parameters':sum(v.numel() for v in values)}
                write(output/'initial_parameters.json',fingerprints);runner.rough_initial_parameter_fingerprints=fingerprints
                emit(status='running',phase='learn');runner.learn();torch.cuda.synchronize()
                changes={k:sum(float((p.detach().cpu()-old).double().square().sum()) for p,old in zip(v.parameters(),before[k]))**.5 for k,v in runner.modules().items()}
                for key in ('actor','wm','critic'):
                    if changes[key]==0 or runner.actual_optimizer_steps[key]==0:raise ValueError('Missing actual optimizer change '+key)
                if not all(bool(torch.isfinite(p).all()) for v in runner.modules().values() for p in v.parameters()):raise FloatingPointError('Nonfinite final parameters')
                checkpoint=output/'train/model_final.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if state['format_version']!=FORMAT_VERSION or state['iteration']!=cfg.max_iterations or 'rough_controller' not in state:raise ValueError('New checkpoint contract missing')
                result.update(status='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY',checkpoint=pin(checkpoint),final_iteration=state['iteration'],reference_signature=state['reference_signature'],sim_steps=runner.sim_steps,parameter_delta_l2=changes,torch_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                result['runtime_evidence']=runtime_evidence(runner)
                return result
    except BaseException as exc:
        result.update(status='FAILED_ENGINEERING_PARTIAL',error=repr(exc));raise
    finally:
        result['native_call_wall_seconds_excluding_imports']=time.perf_counter()-started
        cleanup_errors=[]
        if runner is not None:
            result.update(phase_seconds=dict(runner.phase_seconds),phase_calls=dict(runner.phase_calls),actual_optimizer_steps=dict(runner.actual_optimizer_steps))
            try:runner.close()
            except BaseException as exc:cleanup_errors.append({'stage':'runner.close','error':repr(exc)})
        if a0_reader is not None:
            try:a0_reader.detach_capture()
            except BaseException as exc:cleanup_errors.append({'stage':'reader.detach_capture','error':repr(exc)})
        if env is not None:
            try:env.close()
            except BaseException as exc:cleanup_errors.append({'stage':'env.close','error':repr(exc)})
        result['cleanup_errors']=cleanup_errors
        if cleanup_errors:result['status']='FAILED_ENGINEERING_CLEANUP_UNQUALIFIED'
        write(output/'result.json',result)
        if cleanup_errors and 'error' not in result:raise RuntimeError('Native cleanup failed; retained result, no retry')
