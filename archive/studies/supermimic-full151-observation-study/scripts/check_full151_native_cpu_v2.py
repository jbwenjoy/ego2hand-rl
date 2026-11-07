"""Full151 native-method CPU contracts; fake Env/physical writers, no physics.

No ManagerBasedRlEnv is constructed. Native ActionTerm/MDP/Runner save/policy,
ReplayBuffer and installed observation buffers run on CPU fixtures. Tiny policy
clients backpropagate only: no optimizer.step, learned WM or actual rollout.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native import controller as cc,runtime as rt
from dex_supermimic.tasks.pen_spin.mdp.actions import SingleHandResidualEMAAction,SingleHandResidualEMAActionCfg
from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand as NativeCommand
from dex_supermimic.tasks.pen_spin.config.wuji_hand2.hand2_constants import RIGHT_HAND_LINK_NAMES
from dex_supermimic.rl.dex_supermimic.pen import PenMDP,PEN_MEMORY_DIMS
from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner as NativeRunner
from dex_supermimic.rl.dex_supermimic.networks import WorldModel
from dex_supermimic.rl.dex_supermimic.buffer import ReplayBuffer
from dex_supermimic.rl.dex_supermimic.common import RunningNorm
from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat
from mjlab.managers.action_manager import ActionManager
from mjlab.managers.observation_manager import ObservationManager,ObservationGroupCfg,ObservationTermCfg
from mjlab.utils.buffers import CircularBuffer,DelayBuffer
from mjlab.utils.lab_api.math import quat_apply_inverse
import mujoco

MODEL=None
BUNDLE=None
LOADED=None
OUTPUT=None
DESCRIPTOR={'schema':'full151_knownrough_controller_input_descriptor:1','mode':cc.MODE,'retarget_variant_id':cc.VARIANT,'physical_reference_qualified':False,'input_signature':'SYNTHETIC_CPU_ONLY_NOT_A_REAL_BUNDLE'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}


def signal(env):return torch.cat((env.sim.data.qvel[:,6:26],env.sim.data.ctrl,env.motion.time_steps.float()[:,None]),-1)
def instant(env):return signal(env)[:,:2]


def buffer_snapshot(manager):
    out={}
    for group,terms in manager._group_obs_term_history_buffer.items():
        for term,b in terms.items():out['history/'+group+'/'+term]={'data':b._buffer.clone() if b.is_initialized else None,'pointer':b._pointer,'pushes':b._num_pushes.clone()}
    for group,terms in manager._group_obs_term_delay_buffer.items():
        for term,b in terms.items():out['delay/'+group+'/'+term]={'data':b._buffer._buffer.clone() if b.is_initialized else None,'pointer':b._buffer._pointer,'pushes':b._buffer._num_pushes.clone(),'lags':b._current_lags.clone(),'steps':b._step_count.clone(),'phase_offsets':b._phase_offsets.clone()}
    return out


class Motion:
    _configure_fixed_wrist_reference=NativeCommand._configure_fixed_wrist_reference
    frame_ids=NativeCommand.frame_ids;trajectory_starts=NativeCommand.trajectory_starts;trajectory_ends=NativeCommand.trajectory_ends;trajectory_lengths=NativeCommand.trajectory_lengths;wrist_pose_w=NativeCommand.wrist_pose_w;_current=NativeCommand._current
    def __init__(self,env):
        self._env=env;self.device='cpu';self.num_envs=env.num_envs;self.joint_names=rt.JOINT_NAMES
        self.cfg=NS(joint_names=rt.JOINT_NAMES,link_names=RIGHT_HAND_LINK_NAMES,fixed_wrist_pose_w=tuple(LOADED['wrist_pose_w'][0]),trajectory_sampling_mode='uniform',first_frame_hold_prob=0,axis_point_radius_m=.1,object_position_tanh_scale_m=.15)
        q=torch.tensor(BUNDLE['reference']['q_goal_rad'],dtype=torch.float32)
        fd=torch.tensor(LOADED['joint_vel']).clone()
        self.group=NS(frame_dt=.02,trajectory_end=torch.tensor([151]),joint_vel=fd.clone())
        self._trajectory_starts=torch.tensor([0]);self._trajectory_ends=torch.tensor([151])
        self._frames={k:torch.tensor(LOADED[k]).clone() for k in ('joint_pos','joint_vel','wrist_pose_w','object_pose','link_pos')}
        self._configure_fixed_wrist_reference()
        self.trajectory_ids=torch.zeros(env.num_envs,dtype=torch.long);self.time_steps=torch.zeros(env.num_envs,dtype=torch.long);self._episode_start_steps=self.time_steps.clone()
    def prepare_reset(self,ids):self.trajectory_ids[ids]=0;self.time_steps[ids]=0;self._episode_start_steps[ids]=0
    @property
    def joint_pos(self):return self._current('joint_pos')


class Robot:
    def __init__(self,env):
        self.env=env;self.joint_names=list(rt.JOINT_NAMES)
        ids=np.asarray([MODEL.joint('robot/'+n).id for n in self.joint_names])
        aids=np.asarray([np.flatnonzero(MODEL.actuator_trnid[:,0]==j).item() for j in ids])
        self.indexing=NS(joint_ids=torch.tensor(ids),ctrl_ids=torch.tensor(aids[::-1].copy() if env.cfg.reverse_ctrl_order else aids),joint_q_adr=torch.tensor(MODEL.jnt_qposadr[ids].copy()),joint_v_adr=torch.tensor(MODEL.jnt_dofadr[ids].copy()))
        self.data=NS(joint_pos_limits=torch.tensor(MODEL.jnt_range[ids].copy(),dtype=torch.float32)[None].repeat(env.num_envs,1,1),default_joint_pos=torch.zeros(env.num_envs,20),encoder_bias=torch.zeros(env.num_envs,20))
        self.target=torch.zeros(env.num_envs,20)
    def find_joints_by_actuator_names(self,patterns):return list(range(20)),self.joint_names
    def find_joints(self,names,preserve_order=True):return list(range(20)),list(names)
    def set_joint_position_target(self,u,joint_ids):self.target[:]=u
    def write_joint_state_to_sim(self,q,qd,joint_ids,env_ids):
        self.env.sim.data.qpos[env_ids[:,None],self.indexing.joint_q_adr]=q
        self.env.sim.data.qvel[env_ids[:,None],self.indexing.joint_v_adr]=qd
    def write_mocap_pose_to_sim(self,pose,env_ids):self.env.sim.data.mocap_pos[env_ids,0]=pose[:,:3];self.env.sim.data.mocap_quat[env_ids,0]=pose[:,3:]


class Object:
    def __init__(self,env):self.env=env;self.indexing=NS(free_joint_q_adr=torch.arange(7),free_joint_v_adr=torch.arange(6));self.last_world_state=None
    def write_root_state_to_sim(self,state,env_ids):
        self.last_world_state=state.clone();self.env.sim.data.qpos[env_ids,:7]=state[:,:7];self.env.sim.data.qvel[env_ids,:3]=state[:,7:10]
        self.env.sim.data.qvel[env_ids,3:6]=quat_apply_inverse(state[:,3:7],state[:,10:13])


class Scene(dict):
    def __init__(self,env):super().__init__();self.env=env;self.env_origins=torch.arange(env.num_envs)[:,None].float()*torch.tensor([[0.,.5,0.]])
    def write_data_to_sim(self):
        term=self.env.action_manager.get_term('joint_pos');self.env.sim.data.ctrl[:,term._position_ctrl_ids]=self['robot'].target


class Actions:
    reset=ActionManager.reset
    def __init__(self,term,n):self._terms={'joint_pos':term};self._action=torch.zeros(n,20);self._prev_action=self._action.clone();self._prev_prev_action=self._action.clone()
    def get_term(self,name):return self._terms[name]
    @property
    def action(self):return self._action
    @property
    def prev_action(self):return self._prev_action


class CPUEnv:
    """Narrow process/apply and reset API fixture; forward/sense are no-ops."""
    def __init__(self,cfg):
        self.cfg=cfg;self.num_envs=4;self.device='cpu';self.common_step_counter=0;self.noop_forward_calls=0;self.noop_sense_calls=0
        d=NS(qpos=torch.zeros(4,MODEL.nq),qvel=torch.zeros(4,MODEL.nv),ctrl=torch.zeros(4,MODEL.nu),mocap_pos=torch.zeros(4,1,3),mocap_quat=torch.zeros(4,1,4),time=torch.zeros(4));d.qpos[:,3]=1;d.mocap_quat[:,:,0]=1
        self.sim=NS(data=d,mj_model=MODEL,model=NS(actuator_ctrlrange=torch.tensor(MODEL.actuator_ctrlrange.copy(),dtype=torch.float32)[None].repeat(4,1,1)),forward=self.noop_forward,sense=self.noop_sense)
        self.scene=Scene(self);self.scene['robot']=Robot(self);self.scene['object']=Object(self)
        self.motion=Motion(self);self.command_manager=NS(get_term=lambda n:self.motion)
        term=SingleHandResidualEMAAction(cfg.actions['joint_pos'],self);self.action_manager=Actions(term,4)
        self.reader=NS(env=self,robot=self.scene['robot'],kin=NS(names=[MODEL.body(i).name for i in range(MODEL.nbody)]),context_names=['CPU_fixture_context'+str(j) for j in range(46)])
        groups={'policy':ObservationGroupCfg(terms={'history':ObservationTermCfg(func=signal,history_length=3,flatten_history_dim=False),'instant':ObservationTermCfg(func=instant)},concatenate_terms=False,enable_corruption=False)}
        if cfg.delays:groups['critic']=ObservationGroupCfg(terms={'delayed':ObservationTermCfg(func=signal,delay_min_lag=0,delay_max_lag=3,delay_update_period=3,delay_per_env_phase=True,delay_hold_prob=.25,history_length=2,flatten_history_dim=False)},concatenate_terms=False,enable_corruption=False)
        self.observation_manager=ObservationManager(groups,self);self.reset_time_outs=torch.zeros(4,dtype=torch.bool)
    def noop_forward(self):self.noop_forward_calls+=1
    def noop_sense(self):self.noop_sense_calls+=1
    def reset(self,*,seed=None,env_ids=None,options=None):
        self.motion.prepare_reset(env_ids);self.action_manager.reset(env_ids);self.observation_manager.reset(env_ids=env_ids)
        self.motion.time_steps[env_ids]+=1  # deliberate stale native-reset observation to test selected correction
        obs=self.observation_manager.compute(update_history=True,env_ids=env_ids)
        self.native_reset_cache=obs;self.native_reset_buffers=buffer_snapshot(self.observation_manager);self.native_reset_rng=torch.get_rng_state().clone()
        return obs,{}
    def step(self,action):
        term=self.action_manager.get_term('joint_pos');self.action_manager._prev_action[:]=self.action_manager._action;self.action_manager._action[:]=action
        term.process_actions(action)
        for _ in range(4):term.apply_actions();self.scene.write_data_to_sim()
        self.reset_time_outs[:]=self.motion.time_steps>=149;self.motion.time_steps+=1;self.common_step_counter+=1
        return self.observation_manager.compute(update_history=True),None,torch.zeros(4,dtype=torch.bool),self.reset_time_outs.clone(),{}


class CPURunner:
    """Real ReplayBuffer and native save; constructor/collect are API fixtures."""
    save=NativeRunner.save;format_version=NativeRunner.format_version;policy_update=NativeRunner.policy_update
    def __init__(self,env,mdp):
        self.env=env;self.mdp=mdp;self.reader=env.reader;self.device='cpu';self.iteration=0;self.sim_steps=0;self.context_dim=368;self.contact_dim=15;self.is_pen=True
        self.log_dir=OUTPUT/'fixture_train'/'train';self.log_dir.mkdir(parents=True,exist_ok=True)
        self.buffer=ReplayBuffer(256,4,'cpu',holdout_fraction=0)
        self.unflatten_memory=lambda m:{'time':m[:,:1],'start':m[:,1:2],'end':m[:,2:3]}
        self.cfg=NS(algorithm='supermimic',task='pen',control_hz=50,to_dict=lambda:{'algorithm':'supermimic','task':'pen','utd':.4,'critic_minibatches':2,'control_hz':50})
        self.torque_norm=RunningNorm(20);self.optimizers={};self.wall_offset=0.;self.started_at=time.perf_counter()
        self._modules={name:nn.Linear(1,1) for name in ('actor','wm','fail_head','critic')}
    def modules(self):return self._modules
    def collect(self,steps):
        for _ in range(steps):
            phase=self.env.motion.time_steps.clone();memory=torch.stack((phase,torch.zeros_like(phase),torch.full_like(phase,151)),-1).float()
            _,_,terminated,timeout,_=self.env.step(torch.zeros(4,20));term=self.env.action_manager.get_term('joint_pos')
            self.buffer.add({'m':memory,'u':term._targets.clone(),'c':term.processed_action.clone(),'done':terminated|timeout,'terminated':terminated,'valid':torch.ones(4,dtype=torch.bool)})
            if bool(timeout.any()):self.env.reset(env_ids=timeout.nonzero().flatten())
        return {'fixture_collection_only':True}
    def load(self,*a,**kw):return 'unpatched_load'
    def load_model_only(self,*a,**kw):return 'unpatched_model_load'


def cfg(delays=False,reverse=False):
    action=SingleHandResidualEMAActionCfg(entity_name='robot',actuator_names=('right_finger.*_joint.*',),mask_prob=0.)
    return NS(auto_reset=False,reverse_ctrl_order=reverse,delays=delays,actions={'joint_pos':action},observations={'policy':NS(enable_corruption=False)},events={'reset_to_reference_state':NS(mode='reset')},terminations={'reference_end':NS(func=lambda env:torch.zeros(env.num_envs,dtype=torch.bool),time_out=True)})


def memory(phase,n=4):
    m={k:torch.zeros(n,s) for k,s in PEN_MEMORY_DIMS.items()};m['end'][:]=151;m['time'][:]=phase;m['age'][:]=1;m['c1'][:]=.1;m['c2'][:]=-.2;return m


def main(output,bundle_path):
    global MODEL,BUNDLE,OUTPUT,LOADED,DESCRIPTOR
    OUTPUT=output
    if output.exists():raise FileExistsError('Fresh output required')
    output.mkdir(parents=True);snap=output/'source_before';snap.mkdir()
    files={'controller':Path(cc.__file__),'runtime':Path(rt.__file__),'checker':Path(__file__),'native_runner':Path(sys.modules[NativeRunner.__module__].__file__),'native_action':Path(sys.modules[SingleHandResidualEMAAction.__module__].__file__),'native_command':Path(sys.modules[NativeCommand.__module__].__file__),'ObservationManager':Path(sys.modules[ObservationManager.__module__].__file__),'CircularBuffer':Path(sys.modules[CircularBuffer.__module__].__file__),'DelayBuffer':Path(sys.modules[DelayBuffer.__module__].__file__)}
    pins={k:pin(p) for k,p in files.items()}
    for k,p in files.items():(snap/(k+'.py')).write_bytes(p.read_bytes())
    checks=[];evidence={}
    def check(name,fn):
        try:
            if not bool(fn()):raise AssertionError('False contract')
            checks.append({'name':name,'passed':True})
        except Exception as e:checks.append({'name':name,'passed':False,'error':repr(e),'traceback':traceback.format_exc()})
    def raises(fn):
        try:fn()
        except (ValueError,AssertionError,FloatingPointError,RuntimeError):return True
        return False
    forbidden={};originals={}
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_kinematics','mj_collision','mj_jac'):
        originals[name]=getattr(mujoco,name);forbidden[name]=0
        def deny(*a,_name=name,**kw):forbidden[_name]+=1;raise AssertionError('CPU fixture forbids '+_name)
        setattr(mujoco,name,deny)
    try:
        MODEL=mujoco.MjModel.from_binary_path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb')
        from full151_native.bundle import load_bundle
        bound=load_bundle(bundle_path)
        BUNDLE=bound['arrays'];DESCRIPTOR=bound['descriptor'];baseline=bound['baseline_u150'];ref=BUNDLE['reference'];init=BUNDLE['init']
        native_reader_pin=bound['spec']['source_pins']['native_reader_arrays']
        if sha(native_reader_pin['path'])!=native_reader_pin['sha256']:raise ValueError('Saved native reader array source changed')
        with np.load(native_reader_pin['path'],allow_pickle=False) as z:LOADED={k:z[k].copy() for k in z.files}
        evidence['actual_bundle']=pin(bundle_path);evidence['saved_reader_arrays']=native_reader_pin
        env_originals=(CPUEnv.__init__,CPUEnv.reset,CPUEnv.step,PenMDP.__init__,PenMDP.control_target,CPURunner.save,CPURunner.load,CPURunner.load_model_only)
        native_termination=PenMDP.termination
        with rt.full151_runtime(CPUEnv,Motion,PenMDP,CPURunner,BUNDLE,DESCRIPTOR) as api:
            with cc.full151_controller(CPUEnv,PenMDP,CPURunner,baseline,DESCRIPTOR):
                env=CPUEnv(cfg());obs,_=env.reset();mdp=PenMDP(env.reader);term=env.action_manager.get_term('joint_pos')
                check('native151_velocity_authority_and_readerFD_diagnostics_separate',lambda:torch.equal(env.motion._frames['joint_vel'],torch.tensor(ref['analytic_qd151_rad_s'],dtype=torch.float32)) and not torch.equal(env.motion._frames['joint_vel'],env.motion.group.joint_vel))
                check('object_world_velocity151_tables',lambda:torch.equal(env.motion._frames['object_linvel_w'],torch.tensor(ref['object_v_world_m_s'][::4],dtype=torch.float32)) and env.motion._frames['object_angvel_w'].shape==(151,3))
                reset=api['reset_readback'](env)
                check('fake_physical_writers_nonzeroqd_u0_and_EMA0_readback',lambda:torch.equal(reset['joint_vel_rad_s'],torch.tensor(init['qd_init_requested_FP32_rad_s']).expand(4,-1)) and torch.equal(reset['actual_ctrl'],torch.tensor(baseline[0]).expand(4,-1)) and not bool(reset['EMA_residual'].any()))
                check('world_angular_Entity_API_not_double_rotated',lambda:torch.equal(env.scene['object'].last_world_state[:,10:],torch.tensor(init['object_angular_requested_FP32_world_rad_s']).expand(4,-1)))
                for phase in (0,14,15,75,148,149):
                    env.motion.time_steps[:]=phase;term._processed_actions[:]=.1;m=memory(phase);old1=m['c1'].clone();old2=m['c2'].clone();action=torch.full((4,20),8. if phase%2 else -8.)
                    residual,u=mdp.control_target(action,m,torch.zeros(4,20));env.step(action)
                    check(f'phase{phase}_actualAction_PenMDP_cache_ctrl_EMA_exact',lambda u=u,residual=residual,m=m,old1=old1,old2=old2:torch.equal(u,term._targets) and torch.equal(u,env.sim.data.ctrl[:,term._position_ctrl_ids]) and torch.equal(residual,term.processed_action) and torch.equal(m['c1'],old1) and torch.equal(m['c2'],old2) and torch.equal(m['frozen_target'],u))
                    nxt=NativeRunner.advance_memory(NS(mdp=mdp),torch.zeros(4,58),m,action,residual,torch.zeros(4,20),torch.eye(3)[None].repeat(4,1,1))
                    check(f'phase{phase}_advance_memory_c1c2_residual_not_absolute',lambda nxt=nxt,residual=residual,old1=old1:torch.equal(nxt['c1'],residual) and torch.equal(nxt['c2'],old1))
                    wm_input=WorldModel.inputs(None,torch.zeros(4,278),u)
                    mdp.kp_default=torch.ones(20);mdp.kd_default=-torch.ones(20);mdp.force_limit=torch.ones(20)*3
                    tau_actual_argument=mdp.pd_torque(torch.zeros(4,58),u,torch.ones(4,46))
                    check(f'phase{phase}_WM298_and_PD_use_executed20_u',lambda wm_input=wm_input,u=u,tau_actual_argument=tau_actual_argument:wm_input.shape==(4,298) and torch.equal(wm_input[:,-20:],u) and torch.equal(tau_actual_argument,u.clamp(-3,3)))
                check('real_terminal150_action_rejects',lambda:raises(lambda:env.step(torch.zeros(4,20))))
                env.reset();runner=CPURunner(env,mdp);runner.collect(151)
                check('full150_action_intervals_plus_next_reset_phase0',lambda:runner.full151_actualu_evidence['phase_histogram']==[8]+[4]*149 and runner.full151_actualu_evidence['steps']==151)
                check('allbatch_replay_actualu_and_residual_exact',lambda:runner.full151_actualu_evidence['all_batch_exact'] and all(v==0 for v in runner.full151_actualu_evidence['max_abs_errors'].values()))
                check('replay_natural149_kept_and_not_physicalfailure',lambda:runner.buffer.data['full151_phase'][:151,0].tolist()==list(range(150))+[0] and runner.buffer.data['full151_source_frame'][149,0].item()==349 and runner._full151_runtime_replay_counts['natural_end_rows']==4 and runner._full151_runtime_replay_counts['physical_terminated_rows']==0)
                checkpoint=output/'native_save_fixture105.pt';before_save=torch.save;runner.save(checkpoint,10);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                check('actual_native_save105_new_signature_metadata_atomic',lambda:state['format_version']==105 and state['full151_controller']['baseline_shape']==[150,20] and state['reference_signature']==state['full151_controller']['reference_controller_signature'] and 'rough_controller' not in state and not checkpoint.with_suffix('.tmp').exists() and torch.save is before_save)
                unsigned={k:v for k,v in state['full151_controller'].items() if k not in ('latest_actualu_evidence','reference_controller_signature')}
                check('canonical_signature_excludes_only_dynamic_and_selfsignature',lambda:cc._digest(unsigned)==state['full151_controller']['reference_controller_signature'])
                check('resume_and_modelwarmstart_failclosed',lambda:raises(lambda:runner.load('old104.pt')) and raises(lambda:runner.load_model_only('old103.pt')))
                check('real_negative_noninteger_phase_rejected',lambda:raises(lambda:mdp.control_target(torch.zeros(4,20),memory(-1),torch.zeros(4,20))) and raises(lambda:mdp.control_target(torch.zeros(4,20),memory(.5),torch.zeros(4,20))))
                m=memory(150);m['c1'].zero_();a=torch.zeros(4,20,requires_grad=True)
                residual,u=mdp.control_target(a,m,torch.zeros(4,20));u.sum().backward()
                check('imag_terminal150_u149_finite_and_zero_actiongrad',lambda:torch.equal(u,torch.tensor(baseline[149]).expand(4,-1)) and not bool(a.grad.any()))
                check('imag_later157_uses_same149_no_u150',lambda:torch.equal(mdp.control_target(torch.ones(4,20),memory(157),torch.zeros(4,20))[1],torch.tensor(baseline[149]).expand(4,-1)))
                terminal_memory=memory(0);terminal_memory['time'][:,0]=torch.tensor([0.,148.,149.,150.]);goal=mdp.reference(terminal_memory)[1]
                x=torch.zeros(4,58);x[:,40:43]=goal[:,:3];x[:,43:52]=quat_to_mat(goal[:,3:]).flatten(-2)
                native_d=native_termination(mdp,x,terminal_memory);new_d=mdp.termination(x,terminal_memory)
                check('real_PenMDP_soft_termination_kept_until149_reference_timeout',lambda:torch.equal(new_d[:2],native_d[:2]) and torch.equal(new_d[2:],torch.ones(2)) and rt.phase_active(terminal_memory).tolist()==[True,True,True,False])
                actual_policy_observation=mdp.policy_observation(x,terminal_memory,torch.zeros(4,12))
                check('real_PenMDP_policy_observation_remains155_at149_and150',lambda:actual_policy_observation.shape==(4,155) and bool(torch.isfinite(actual_policy_observation).all()))
                action=torch.zeros(4,20,requires_grad=True);m=memory(0);m['c1'].zero_()
                with torch.autocast('cpu',dtype=torch.bfloat16):_,activeu=mdp.control_target(action,m,torch.zeros(4,20))
                activeu.sum().backward()
                check('active_actor_to_C2_FP32_interior_gradient_preserved',lambda:activeu.dtype==torch.float32 and torch.equal(action.grad,torch.full((4,20),.25)))
                saved_torch_save=torch.save
                def denied_save(*a,**kw):raise OSError('intentional CPU serialization fault')
                try:
                    torch.save=denied_save
                    try:runner.save(output/'intentionally_failed_save.pt',11)
                    except OSError:pass
                    check('save_exception_restores_outer_torchsave',lambda:torch.save is denied_save)
                finally:torch.save=saved_torch_save
                evidence['controller']=runner.full151_actualu_evidence;evidence['metadata']=state['full151_controller']
        check('all_class_hooks_restored_after_composed_scopes',lambda:(CPUEnv.__init__,CPUEnv.reset,CPUEnv.step,PenMDP.__init__,PenMDP.control_target,CPURunner.save,CPURunner.load,CPURunner.load_model_only)==env_originals)
        check('native_format3_loader_rejects_new105_serialization',lambda:raises(lambda:NativeRunner.load(runner,checkpoint)))
        # Installed observation buffers with selected-row correction, not a
        # physical Env reset. Native reset's first compute is intentionally old.
        torch.manual_seed(97)
        with rt.full151_runtime(CPUEnv,Motion,PenMDP,CPURunner,BUNDLE,DESCRIPTOR):
            env=CPUEnv(cfg(delays=True));obs,_=env.reset();manager=env.observation_manager
            check('installed_ObservationManager_Delay_Circular_CPU',lambda:type(manager) is ObservationManager and all(isinstance(x,CircularBuffer) for v in manager._group_obs_term_history_buffer.values() for x in v.values()) and all(isinstance(x,DelayBuffer) for v in manager._group_obs_term_delay_buffer.values() for x in v.values()))
            check('corrected_cache_history_and_analytic_signal',lambda:manager._obs_buffer is obs and obs is not env.native_reset_cache and torch.equal(obs['policy']['history'],signal(env)[:,None].expand(-1,3,-1)))
            check('correction_no_second_RNG_lag_draw',lambda:torch.equal(torch.get_rng_state(),env.native_reset_rng))
            for _ in range(5):env.sim.data.qvel[:,6:26]+=.03;env.motion.time_steps+=1;manager.compute(update_history=True)
            before=buffer_snapshot(manager);beforestate=rt.reset_readback(env);ids=torch.tensor([1,3]);others=torch.tensor([0,2]);obs,_=env.reset(env_ids=ids);after=buffer_snapshot(manager);afterstate=rt.reset_readback(env)
            check('partial_reset_only_selected_state_generations',lambda:afterstate['generation'].tolist()==[1,2,1,2] and all(torch.equal(beforestate[k][others],afterstate[k][others]) for k in ('full_qpos','full_qvel','actual_ctrl','EMA_residual','mocap_pos','mocap_quat','phase')))
            check('partial_reset_nonselected_history_delay_storage_untouched',lambda:all(torch.equal(after[k]['data'][:,others],before[k]['data'][:,others]) and torch.equal(after[k]['pushes'][others],before[k]['pushes'][others]) and after[k]['pointer']==before[k]['pointer'] for k in before))
            check('partial_reset_no_extra_lag_schedule_or_RNG',lambda:torch.equal(torch.get_rng_state(),env.native_reset_rng) and all(torch.equal(after[k][f][others],before[k][f][others]) for k in before if k.startswith('delay/') for f in ('lags','steps','phase_offsets')))
            check('partial_selected_corrected_backfill',lambda:torch.equal(obs['policy']['history'][ids],signal(env)[ids,None].expand(-1,3,-1)) and torch.equal(obs['critic']['delayed'][ids],signal(env)[ids,None].expand(-1,2,-1)))
            env.motion.time_steps[:]=torch.tensor([0,148,149,150]);check('terminal149_to_obs150_timeout_semantics',lambda:rt.reference_timeout(env).tolist()==[False,False,True,True] and PenMDP.termination.__name__=='termination')
            check('negative_reset_ids_rejected',lambda:raises(lambda:env.reset(env_ids=torch.tensor([-1]))) and raises(lambda:env.reset(env_ids=torch.tensor([1,1]))))
        rawbuf=ReplayBuffer(8,1,'cpu',holdout_fraction=0)
        for phase,done,gen in ((148,False,1),(149,True,1),(0,False,2),(1,False,2)):
            rawbuf.add({'done':torch.tensor([done]),'valid':torch.tensor([True]),'full151_phase':torch.tensor([phase]),'full151_generation':torch.tensor([gen]),'full151_start':torch.tensor([0]),'full151_end':torch.tensor([151])})
        window=rawbuf.sample(1,window=4)
        check('native_ReplayBuffer_terminalretained_reset_suffixmask',lambda:window['mask'].tolist()==[[1.,1.,0.,0.]] and rt.validate_replay_window(window)['terminal_valid_entries']==1)
        bad=copy.deepcopy(window);bad['full151_phase'][0,2]=150
        check('unexecuted_replay_phase150_rejected',lambda:raises(lambda:rt.validate_replay_window(bad)))
        bad=copy.deepcopy(window);bad['done'][0,1]=False
        check('unmarked_reset_generation_rejected',lambda:raises(lambda:rt.validate_replay_window(bad)))
        badbundle=copy.deepcopy(BUNDLE);badbundle['reference']['q_goal_rad']=badbundle['reference']['q_goal_rad'][:16]
        check('old_prefix16_bundle_rejected',lambda:raises(lambda:rt._validated_bundle(badbundle,DESCRIPTOR)))
        bad=copy.deepcopy(DESCRIPTOR);bad['mode']='rough_reference_pilot'
        check('old104_descriptor_rejected',lambda:raises(lambda:rt._validated_bundle(BUNDLE,bad)))
        # Reuse only retained tiny-network fixture definitions; all full151
        # phase semantics are explicitly overridden here, not text-replaced.
        oldpath=Path('D:/workspace/supermimic-rough-prefix-learning-study/scripts/check_rough_runtime_cpu.py')
        spec=importlib.util.spec_from_file_location('_full151_retained_tiny_clients',oldpath);tiny=importlib.util.module_from_spec(spec);spec.loader.exec_module(tiny)
        class TinyMDP150(tiny.TinyMDP):
            def control_target(self,a,m,bias):
                phase=m['time'].squeeze(-1).long();res=.5*(.5*a).clamp(-.6,.6)+.5*m['c1'];u=self.owner.baseline[phase.clamp(max=149)]+res;u=torch.where((phase<150)[:,None],u,self.owner.baseline[149]);return res,u
        class Policy150(tiny.PolicyFixture):
            def __init__(self,phases,length=151):super().__init__(phases,baseline,length=length);self.mdp=TinyMDP150(self)
            def task_features(self,x,m,aux,sim):self.current_phase=m['time'].squeeze(-1);return x+((self.current_phase>=150).float()*self.perturb_dead)[:,None]
            def step_wm(self,x,u,dynamics):
                y,c=self.wm(x,u,dynamics);dead=self.current_phase>=150;y=y+dead[:,None]*self.perturb_dead
                if self.nonfinite_dead and bool(dead.any()):y=y.clone();y[dead,0]=float('nan')
                return y,c
            def reward_step(self,x,m,a,target,contact,tau,soft):return -(x[:,:20]-.2).square().mean(-1)+((m['time'].squeeze(-1)>=150).float()*self.perturb_dead),torch.zeros_like(m['cage']),{}
        policy=rt.make_policy_update(NativeRunner.policy_update)
        (output/'native_policy_source_before.py').write_text(policy.__full151_native_source__,'utf-8');(output/'full151_policy_source_after.py').write_text(policy.__full151_source__,'utf-8')
        torch.manual_seed(13);base=Policy150([0,0,0,0],length=1000);new=copy.deepcopy(base);rng=torch.get_rng_state();NativeRunner.policy_update(base);torch.set_rng_state(rng);policy(new)
        check('real_native_policy_noend_formula_loss_equivalence',lambda:max(abs(a[1]-b[1]) for a,b in zip(base.optimize_calls,new.optimize_calls))<1e-6 and len(base.optimize_calls)==len(new.optimize_calls)==9)
        check('real_native_policy_noend_grad_norm_equivalence',lambda:max(float((a[1]-b[1]).abs().max()) for a,b in zip(base.gradients,new.gradients))<1e-6 and all(torch.equal(base.critic.norm.state_dict()[k],new.critic.norm.state_dict()[k]) for k in base.critic.norm.state_dict()))
        torch.manual_seed(14);mixed=Policy150([0,142,149,149]);perturbed=copy.deepcopy(mixed);perturbed.perturb_dead=1000.;rng=torch.get_rng_state();policy(mixed);torch.set_rng_state(rng);policy(perturbed);pe=rt.policy_evidence(mixed)
        check('full151_mixedH8_activecounts8_8_1_1',lambda:np.asarray(pe['active_mask_H_B']).sum(0).tolist()==[8,8,1,1])
        check('dead_rows_cannot_change_actor_critic_losses_grad_norm',lambda:max(abs(a[1]-b[1]) for a,b in zip(mixed.optimize_calls,perturbed.optimize_calls))<1e-6 and max(float((a[1]-b[1]).abs().max()) for a,b in zip(mixed.gradients,perturbed.gradients))<1e-6 and all(torch.equal(mixed.critic.norm.state_dict()[k],perturbed.critic.norm.state_dict()[k]) for k in mixed.critic.norm.state_dict()))
        check('critic_norm19_and_update_calls1plus8_unchanged',lambda:pe['critic_norm_eligible']==19 and mixed.critic.norm.count.item()==19 and pe['optimizer_actor_calls']==1 and pe['optimizer_critic_calls']==8)
        bad=copy.deepcopy(mixed);bad.nonfinite_dead=True
        check('nonfinite_dead_output_rejected_not_NaNtimes0',lambda:raises(lambda:policy(bad)))
        evidence['policy']=pe;evidence['tiny_definition_source']=pin(oldpath)
        check('all_model_physics_calls_stayed_zero',lambda:not any(forbidden.values()))
        check('TorchCUDA_no_visible_device_uninitialized',lambda:not torch.cuda.is_initialized() and not torch.cuda.is_available() and torch.cuda.device_count()==0)
        check('frozen_source_snapshots_unchanged',lambda:all(sha(p['path'])==p['sha256'] for p in pins.values()))
        status='PASS' if all(x['passed'] for x in checks) else 'FAIL'
    except Exception as e:
        status='HARNESS_FAILURE';evidence['exception']=repr(e);evidence['traceback']=traceback.format_exc()
    finally:
        for name,fn in originals.items():setattr(mujoco,name,fn)
    report={'schema':'full151_native_methods_CPU_contract:1','status':status,'checks':checks,'passed':sum(x['passed'] for x in checks),'total':len(checks),'source_pins':pins,'evidence':evidence,'forbidden_native_calls':forbidden,'actual_Env_constructions':0,'actual_physics_steps':0,'actual_optimizer_steps':0,'GPU_used':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'scope':'Actual native ActionTerm/PenMDP/advance-memory/save/policy-update/ReplayBuffer and installed ObservationManager/Delay/Circular methods on CPU API fixtures. Env reset/Entity state writes/forward/sense and collection are fake. Tiny policy clients backpropagate only. No actual ManagerBasedRlEnv, native physics reset/rollout, learned WM forward or training qualification.'}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({'status':status,'passed':report['passed'],'total':report['total'],'report':pin(output/'report.json'),'exception':evidence.get('exception')},indent=2))
    if status!='PASS':print(json.dumps([x for x in checks if not x['passed']],indent=2))
    return 0 if status=='PASS' else 1


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--bundle',default=str(ROOT/'outputs/full151_native_input_v1/bundle.json'));args=ap.parse_args();raise SystemExit(main(Path(args.output).resolve(),Path(args.bundle).resolve()))
