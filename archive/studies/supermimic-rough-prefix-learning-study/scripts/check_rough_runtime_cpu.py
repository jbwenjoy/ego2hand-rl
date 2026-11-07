"""CPU method contracts; real command/policy/buffer methods, stub Env/Entity I/O.

No native Env construction, forward/step, CUDA, research optimizer or training
run. Policy probes backpropagate through tiny fixture networks without stepping
an optimizer; counters denote call sites, not actual optimizer updates.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,difflib,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import rough_prefix.runtime as rt
from rough_prefix.bundle import load_bundle
from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand as NativeCommand
from dex_supermimic.tasks.pen_spin.mdp.actions import SingleHandResidualEMAAction as NativeAction
from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner as NativeRunner
from dex_supermimic.rl.dex_supermimic.buffer import ReplayBuffer
from dex_supermimic.rl.dex_supermimic.common import RunningNorm
from mjlab.managers.action_manager import ActionManager as NativeActionManager
from mjlab.utils.lab_api.math import quat_apply_inverse
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Motion:
    _configure_fixed_wrist_reference=NativeCommand._configure_fixed_wrist_reference
    frame_ids=NativeCommand.frame_ids;trajectory_starts=NativeCommand.trajectory_starts;trajectory_ends=NativeCommand.trajectory_ends;trajectory_lengths=NativeCommand.trajectory_lengths;wrist_pose_w=NativeCommand.wrist_pose_w;_current=NativeCommand._current
    def __init__(self,env,loaded):
        self._env=env;self.device='cpu';self.num_envs=env.num_envs;self.cfg=NS(joint_names=rt.JOINT_NAMES,fixed_wrist_pose_w=tuple(loaded['wrist_pose_w'][0]),trajectory_sampling_mode='uniform',first_frame_hold_prob=0)
        self.group=NS(frame_dt=.02,trajectory_end=torch.tensor([16]),joint_vel=torch.tensor(loaded['joint_vel']).clone());self._trajectory_starts=torch.tensor([0]);self._trajectory_ends=torch.tensor([16]);self._frames={k:torch.tensor(loaded[k]).clone() for k in ('joint_pos','joint_vel','wrist_pose_w','object_pose','link_pos')}
        self._configure_fixed_wrist_reference();self.trajectory_ids=torch.zeros(env.num_envs,dtype=torch.long);self.time_steps=torch.zeros(env.num_envs,dtype=torch.long);self._episode_start_steps=self.time_steps.clone()
    def prepare_reset(self,ids):self.trajectory_ids[ids]=0;self.time_steps[ids]=0;self._episode_start_steps[ids]=0


class Term:
    reset=NativeAction.reset;apply_actions=NativeAction.apply_actions
    def __init__(self,env,robot):
        self._entity=robot;self._target_ids=torch.arange(20);self._position_ctrl_ids=torch.arange(20);self._raw_actions=torch.zeros(env.num_envs,20);self._processed_actions=torch.zeros_like(self._raw_actions);self._targets=torch.zeros_like(self._raw_actions);self._targets_valid=torch.zeros(env.num_envs,dtype=torch.bool);self._mask_active=torch.zeros_like(self._targets_valid);self._mask_remaining=torch.zeros(env.num_envs,dtype=torch.long);self._mask_dofs=torch.zeros_like(self._raw_actions,dtype=torch.bool)
    def limit_target(self,u):return u.clamp(-100,100)


class Actions:
    reset=NativeActionManager.reset
    def __init__(self,env,term):self._action=torch.zeros(env.num_envs,20);self._prev_action=self._action.clone();self._prev_prev_action=self._action.clone();self._terms={'joint_pos':term}
    def get_term(self,name):return self._terms[name]
    @property
    def action(self):return self._action
    @property
    def prev_action(self):return self._prev_action


class Robot:
    def __init__(self,env):self.env=env;self.indexing=NS(joint_q_adr=torch.arange(7,27),joint_v_adr=torch.arange(6,26));self.target=torch.zeros(env.num_envs,20)
    def find_joints(self,names,preserve_order=True):return list(range(20)),list(names)
    def write_joint_state_to_sim(self,q,qd,joint_ids,env_ids):self.env.sim.data.qpos[env_ids[:,None],self.indexing.joint_q_adr]=q;self.env.sim.data.qvel[env_ids[:,None],self.indexing.joint_v_adr]=qd
    def write_mocap_pose_to_sim(self,pose,env_ids):self.env.sim.data.mocap_pos[env_ids,0]=pose[:,:3];self.env.sim.data.mocap_quat[env_ids,0]=pose[:,3:]
    def set_joint_position_target(self,u,joint_ids):self.target[:]=u


class Object:
    def __init__(self,env):self.env=env;self.indexing=NS(free_joint_q_adr=torch.arange(7),free_joint_v_adr=torch.arange(6));self.last_world_state=None
    def write_root_state_to_sim(self,state,env_ids):
        self.last_world_state=state.clone();self.env.sim.data.qpos[env_ids,:7]=state[:,:7];self.env.sim.data.qvel[env_ids,:3]=state[:,7:10];self.env.sim.data.qvel[env_ids,3:6]=quat_apply_inverse(state[:,3:7],state[:,10:13])


class Scene(dict):
    def __init__(self,env):super().__init__();self.env=env;self.env_origins=torch.arange(env.num_envs)[:,None].float()*torch.tensor([[0.,.5,0.]])
    def write_data_to_sim(self):self.env.sim.data.ctrl[:]=self['robot'].target


class StubEnv:
    def __init__(self,cfg):
        self.cfg=cfg;self.num_envs=4;self.device='cpu';self.sim=NS(data=NS(qpos=torch.zeros(4,27),qvel=torch.zeros(4,26),ctrl=torch.zeros(4,20),mocap_pos=torch.zeros(4,1,3),mocap_quat=torch.zeros(4,1,4),time=torch.zeros(4)),forward=lambda:None,sense=lambda:None);self.sim.data.qpos[:,3]=1;self.sim.data.mocap_quat[:,:,0]=1
        self.scene=Scene(self);self.scene['robot']=Robot(self);self.scene['object']=Object(self);self.motion=Motion(self,LOADED);self.command_manager=NS(get_term=lambda n:self.motion);term=Term(self,self.scene['robot']);self.action_manager=Actions(self,term);self.reset_time_outs=torch.zeros(4,dtype=torch.bool);self.observation_manager=NS(compute=lambda **kw:{'policy':torch.cat((self.sim.data.qpos,self.sim.data.qvel),-1).clone()})
    def reset(self,*,seed=None,env_ids=None,options=None):
        self.motion.prepare_reset(env_ids);self.motion.time_steps[env_ids]+=1;self.action_manager.reset(env_ids);return self.observation_manager.compute(),{}


class StubMDP:
    def termination(self,x,memory):return torch.full((len(x),),.05)


class ScopeRunner:
    policy_update=NativeRunner.policy_update
    def __init__(self,env):self.env=env;self.buffer=ReplayBuffer(32,env.num_envs,'cpu',holdout_fraction=0);self.is_pen=True;self.unflatten_memory=lambda m:{'time':m[:,:1],'start':m[:,1:2],'end':m[:,2:3]}


class TinyCritic(nn.Module):
    def __init__(self):super().__init__();self.norm=RunningNorm(58);self.net=nn.Linear(58,1)
    def forward(self,x):return self.net(self.norm(x)).squeeze(-1)


class TinyWM(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Linear(20,58)
    def forward(self,x,u,dynamics):return x+.01*self.net(u),x.new_zeros(len(x),15)


class TinyMDP:
    dt=.02
    def __init__(self,owner):self.owner=owner
    def flatten_history(self,h):return h[:,-1]
    def control_target(self,a,m,bias):
        phase=m['time'].squeeze(-1).long();res=.5*(.5*a).clamp(-.6,.6)+.5*m['c1'];u=self.owner.baseline[phase.clamp(max=14)]+res;u=torch.where((phase<15)[:,None],u,self.owner.baseline[14]);return res,u
    def pd_torque(self,x,u,context,limits):return (u-x[:,:20])*.1
    def termination(self,x,m):return x.new_full((len(x),),.05)
    def policy_observation(self,x,m,auxiliary):return x


class PolicyFixture:
    def __init__(self,phases,baseline,length=16,perturb_dead=0.,nonfinite_dead=False):
        self.cfg=NS(horizon=8,gamma=.97,lam=.95,failure_head=False,wm_only_iters=200,failure_warmup_iters=0,critic_minibatches=4,critic_epochs=2);self.device='cpu';self.is_pen=True;self.iteration=0;self.env=NS(common_step_counter=0,max_common_steps=10000);self.actor=nn.Sequential(nn.Linear(58,20),nn.Tanh());self.wm=TinyWM();self.fail_head=nn.Linear(1,1);self.critic=TinyCritic();self.mdp=TinyMDP(self);self.baseline=torch.tensor(baseline);self.phases=torch.tensor(phases,dtype=torch.float32);self.B=len(phases);self.reader=NS(robot=NS(data=NS(encoder_bias=torch.zeros(self.B,20))));self.observations=NS(disturbance=lambda a:a[:,:1]);self.step_actor=self.actor;self.optimize_calls=[];self.gradients=[];self.perturb_dead=perturb_dead;self.nonfinite_dead=nonfinite_dead
        x=torch.randn(self.B,58)*.05;m={'time':self.phases[:,None],'start':torch.zeros(self.B,1),'end':torch.full((self.B,1),float(length)),'c1':torch.zeros(self.B,20),'a1':torch.zeros(self.B,20),'a2':torch.zeros(self.B,20),'cage':torch.zeros(self.B,20),'goal':torch.eye(3).flatten()[None].repeat(self.B,1),'age':torch.zeros(self.B,1),'limits':torch.ones(self.B,20)};aux=torch.zeros(self.B,12);aux[:,3:]=torch.eye(3).flatten();self.w={'x':x,'m':m,'obs':x.clone(),'aux':aux,'sim_context':torch.ones(self.B,46),'env':torch.arange(self.B)}
    def sample_starts(self):return copy.deepcopy(self.w)
    def unflatten_memory(self,m):return m
    def history_from_obs(self,o):return o[:,None]
    def task_features(self,x,m,aux,sim):self.current_phase=m['time'].squeeze(-1);return x+((self.current_phase>=15).float()*self.perturb_dead)[:,None]
    def critic_features(self,c):return c
    def wm_features(self,x,sim,dist):return x
    def step_wm(self,x,u,dynamics):
        y,c=self.wm(x,u,dynamics);dead=self.current_phase>=15;y=y+dead[:,None]*self.perturb_dead
        if self.nonfinite_dead and bool(dead.any()):y=y.clone();y[dead,0]=float('nan')
        return y,c
    def reward_step(self,x,m,a,target,contact,tau,soft):
        r=-(x[:,:20]-.2).square().mean(-1)+((m['time'].squeeze(-1)>=15).float()*self.perturb_dead);return r,torch.zeros_like(m['cage']),{}
    def advance_memory(self,x,m,a,target,cage,next_goal):return {**m,'time':m['time']+1,'c1':target,'a1':a,'a2':m['a1'],'age':m['age']+1,'cage':cage}
    def optimize(self,name,loss):
        module=self.actor if name=='actor' else self.critic;module.zero_grad(set_to_none=True);loss.backward();self.optimize_calls.append((name,float(loss.detach())));self.gradients.append((name,torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in module.parameters()]).clone()))


def main():
    global LOADED
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();start=time.perf_counter();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'runtime_source.py').write_bytes(Path(rt.__file__).read_bytes());(out/'checker_source.py').write_bytes(Path(__file__).read_bytes());b=load_bundle(ROOT/'outputs/input_bundle_v1/bundle.json');bundle=b['arrays'];descriptor=b['descriptor'];native_manifest=read_json= json.loads(Path(b['spec']['source_pins']['native_manifest']['path']).read_text());saved=Path(native_manifest['native']['path']).parents[2]/'native_reader_loaded_CPU.npz'
    with np.load(saved,allow_pickle=False) as z:LOADED={k:z[k].copy() for k in z.files}
    checks=[]
    def check(name,call):
        try:
            result=call()
            if result is not None and not bool(result):raise AssertionError('False result')
            checks.append({'name':name,'passed':True})
        except Exception as exc:checks.append({'name':name,'passed':False,'error':repr(exc)})
    def reject(call):
        try:call()
        except (ValueError,FloatingPointError):return True
        raise AssertionError('Invalid input accepted')
    cfg=NS(auto_reset=False,events={'reset_to_reference_state':NS(mode='reset')},terminations={'reference_end':NS(func=lambda env:torch.zeros(env.num_envs,dtype=torch.bool),time_out=True)})
    originals={(cls,name):getattr(cls,name) for cls,name in ((StubEnv,'__init__'),(StubEnv,'reset'),(Motion,'_configure_fixed_wrist_reference'),(Motion,'prepare_reset'),(StubMDP,'termination'),(ScopeRunner,'policy_update'),(ScopeRunner,'__init__'))}
    with rt.rough_runtime(StubEnv,Motion,StubMDP,ScopeRunner,bundle,descriptor):
        env=StubEnv(cfg);motion=env.motion;native_fd=torch.tensor(LOADED['joint_vel'])
        check('real_native_fixedwrist_then_analytic_joint_velocity',lambda:torch.equal(motion._frames['joint_vel'],torch.tensor(bundle['reference']['analytic_qd16_rad_s'],dtype=torch.float32)))
        check('native_reader_group_FD_not_modified',lambda:torch.equal(motion.group.joint_vel,native_fd) and torch.equal(motion._rough_runtime_native_velocity_diagnostics['joint_vel'],native_fd))
        check('object_world_velocity_tables_from_quintic',lambda:torch.equal(motion._frames['object_linvel_w'],torch.tensor(bundle['reference']['object_v_world_m_s'][::4],dtype=torch.float32)) and torch.equal(motion._frames['object_angvel_w'],torch.tensor(bundle['reference']['object_omega_world_rad_s'][::4],dtype=torch.float32)))
        check('natural_end_cfg_timeout',lambda:cfg.terminations['reference_end'].func is rt.reference_timeout and cfg.terminations['reference_end'].time_out is True)
        env.reset();snap=rt.reset_readback(env);check('fullreset_analytic_qd_and_u0_EMA0',lambda:torch.equal(snap['joint_vel_rad_s'],torch.tensor(bundle['init']['qd_init_requested_FP32_rad_s']).expand(4,-1)) and torch.equal(snap['actual_ctrl'],torch.tensor(bundle['init']['u_planned0_FP32_rad']).expand(4,-1)) and not bool(snap['EMA_residual'].any()))
        check('world_angular_API_used_without_double_conversion',lambda:torch.equal(env.scene['object'].last_world_state[:,10:],torch.tensor(bundle['init']['object_angular_requested_FP32_world_rad_s']).expand(4,-1)))
        env.motion.time_steps[:]=torch.tensor([3,5,7,14]);term=env.action_manager.get_term('joint_pos');term._processed_actions[:]=.123;term._targets[:]=.4;term.apply_actions();env.scene.write_data_to_sim();before=rt.reset_readback(env);env.reset(env_ids=torch.tensor([1,3]));after=rt.reset_readback(env)
        check('partial_reset_only_selected_phases',lambda:env.motion.time_steps.tolist()==[3,0,7,0])
        check('partial_reset_nonselected_q_qd_u_EMA_mocap_untouched',lambda:all(torch.equal(before[k][[0,2]],after[k][[0,2]]) for k in ('full_qpos','full_qvel','actual_ctrl','EMA_residual','mocap_pos','mocap_quat')))
        check('partial_generations_selected_only',lambda:after['generation'].tolist()==[1,2,1,2])
        env.motion.time_steps[:]=torch.tensor([0,7,13,14]);check('terminal_phase14_to_obs15',lambda:rt.reference_timeout(env).tolist()==[False,False,False,True] and StubMDP().termination(torch.zeros(4,58),{'time':torch.tensor([[0.],[7.],[14.],[15.]])}).tolist()==[.05000000074505806,.05000000074505806,1.,1.])
        runner=ScopeRunner(env);env.motion.time_steps[:]=0
        for k in range(16):
            phase=torch.full((4,1),float(k%15));natural=k%15==14;env.reset_time_outs[:]=natural
            if k==15:env._rough_runtime_episode_generation+=1
            runner.buffer.add({'m':torch.cat((phase,phase*0,phase*0+16),-1),'done':torch.full((4,),natural,dtype=torch.bool),'terminated':torch.zeros(4,dtype=torch.bool),'valid':torch.ones(4,dtype=torch.bool),'u':torch.zeros(4,20)})
        check('real_native_buffer_add_has_source_phase_generation',lambda:runner.buffer.data['rough_phase'][:16,0].tolist()==list(range(15))+[0] and runner.buffer.data['rough_source_frame'][14,0].item()==214 and runner._rough_runtime_replay_counts['natural_end_rows']==4 and runner._rough_runtime_replay_counts['physical_terminated_rows']==0)
    check('context_methods_restored',lambda:all(getattr(cls,name) is f for (cls,name),f in originals.items()))
    policy=rt.make_policy_update(NativeRunner.policy_update);(out/'native_policy_before.py').write_text(policy.__rough_native_source__);(out/'rough_policy_after.py').write_text(policy.__rough_source__);(out/'policy_diff.patch').write_text(''.join(difflib.unified_diff(policy.__rough_native_source__.splitlines(keepends=True),policy.__rough_source__.splitlines(keepends=True),fromfile='native_policy',tofile='rough_policy')))
    torch.manual_seed(13);base=PolicyFixture([0,0,0,0],b['baseline_u15'],length=151);new=copy.deepcopy(base);rng=torch.get_rng_state();NativeRunner.policy_update(base);torch.set_rng_state(rng);policy(new)
    check('real_native_runner_long_noend_loss_equivalence',lambda:max(abs(x[1]-y[1]) for x,y in zip(base.optimize_calls,new.optimize_calls))<1e-6 and len(base.optimize_calls)==len(new.optimize_calls)==9)
    check('real_native_runner_long_noend_gradient_equivalence',lambda:max(float((x[1]-y[1]).abs().max()) for x,y in zip(base.gradients,new.gradients))<1e-6)
    check('real_native_runner_long_noend_norm_equivalence',lambda:all(torch.equal(base.critic.norm.state_dict()[k],new.critic.norm.state_dict()[k]) for k in base.critic.norm.state_dict()))
    torch.manual_seed(14);mixed=PolicyFixture([0,7,14,14],b['baseline_u15']);perturbed=copy.deepcopy(mixed);perturbed.perturb_dead=1000.;rng=torch.get_rng_state();policy(mixed);torch.set_rng_state(rng);policy(perturbed);evidence=rt.policy_evidence(mixed)
    check('mixed_H8_ends_active_counts8_8_1_1',lambda:np.array(evidence['active_mask_H_B']).sum(axis=0).tolist()==[8,8,1,1])
    check('dead_placeholder_rewards_values_do_not_change_losses',lambda:max(abs(x[1]-y[1]) for x,y in zip(mixed.optimize_calls,perturbed.optimize_calls))<1e-6)
    check('dead_placeholder_no_actor_or_critic_gradient_effect',lambda:max(float((x[1]-y[1]).abs().max()) for x,y in zip(mixed.gradients,perturbed.gradients))<1e-6)
    check('dead_placeholder_no_critic_norm_effect',lambda:all(torch.equal(mixed.critic.norm.state_dict()[k],perturbed.critic.norm.state_dict()[k]) for k in mixed.critic.norm.state_dict()))
    check('critic_norm_counts_only_active_plus_live_bootstrap',lambda:evidence['critic_norm_eligible']==19 and mixed.critic.norm.count.item()==19)
    check('optimizer_call_budget_unchanged',lambda:evidence['optimizer_actor_calls']==1 and evidence['optimizer_critic_calls']==8 and len(mixed.optimize_calls)==9)
    nan=copy.deepcopy(mixed);nan.nonfinite_dead=True;check('nonfinite_inactive_output_rejected_not_NaNtimeszero',lambda:reject(lambda:policy(nan)))
    check('frozen_modules_restored_after_failure',lambda:all(p.requires_grad for module in (nan.wm,nan.fail_head,nan.critic) for p in module.parameters()))
    rawbuf=ReplayBuffer(8,1,'cpu',holdout_fraction=0)
    for phase,done,gen in [(13,False,1),(14,True,1),(0,False,2),(1,False,2)]:rawbuf.add({'done':torch.tensor([done]),'valid':torch.tensor([True]),'rough_phase':torch.tensor([phase]),'rough_generation':torch.tensor([gen]),'rough_start':torch.tensor([0]),'rough_end':torch.tensor([16])})
    window=rawbuf.sample(1,window=4);check('real_native_replay_retains_terminal_masks_reset_suffix',lambda:window['mask'].tolist()==[[1.,1.,0.,0.]] and rt.validate_replay_window(window)['terminal_valid_entries']==1)
    wrong=copy.deepcopy(window);wrong['done'][0,1]=False;check('unmarked_replay_reset_rejected',lambda:reject(lambda:rt.validate_replay_window(wrong)))
    wrong=copy.deepcopy(window);wrong['rough_phase'][0,2]=15;check('unexecuted_phase15_replay_rejected',lambda:reject(lambda:rt.validate_replay_window(wrong)))
    wrong=copy.deepcopy(window);wrong['mask'][0,2]=1;check('reset_suffix_mask_error_rejected',lambda:reject(lambda:rt.validate_replay_window(wrong)))
    check('Torch_CUDA_uninitialized_and_no_cuda_tensor',lambda:not torch.cuda.is_initialized() and all(not p.is_cuda for p in mixed.actor.parameters()))
    result={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'runtime_sha256':sha(rt.__file__),'checker_sha256':sha(__file__),'policy_evidence':evidence,'CPU_native_method_scope':['real SingleHandTrackingCommand fixed-wrist method','real residual action reset/apply methods','real ActionManager reset method','real DexSuperMimicRunner.policy_update method with tiny differentiable network clients','real ReplayBuffer add/sample'],'stub_scope':['Env init/reset and Entity world-state writers/sim-forward/sense are narrow CPU API fixtures, not physics','optimize callsites backpropagate only; no actual optimizer.step or research training'],'native_physics_steps':0,'GPU_started':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'actual_optimizer_steps':0,'wall_seconds':time.perf_counter()-start};(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));raise SystemExit(result['status']!='PASS')
if __name__=='__main__':main()
