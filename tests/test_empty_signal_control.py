import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationManager
from supermimic_research.runtime.empty_signal_control import empty_signal_control_step


def fixture():
    calls=[]
    def call(name):return lambda *args,**kwargs:calls.append((name,kwargs))
    term=object.__new__(TerminationManager);term._term_names=[];term._term_cfgs=[];term._truncated_buf=torch.zeros(2,dtype=torch.bool);term._terminated_buf=torch.zeros(2,dtype=torch.bool)
    env=NS(cfg=NS(auto_reset=False,terminations={},rewards={},metrics={},curriculum={},recorders={},decimation=8),device='cpu',_manual_reset_pending=torch.zeros(2,dtype=torch.bool),
        extras={},_sim_step_counter=0,episode_length_buf=torch.zeros(2,dtype=torch.long),common_step_counter=0,physics_dt=.0025,step_dt=.02,
        termination_manager=term,event_manager=NS(available_modes=['reset','startup']),
        action_manager=NS(process_action=call('process'),apply_action=call('apply')),
        scene=NS(write_data_to_sim=call('write'),update=call('scene_update')),
        sim=NS(step=call('native'),forward=call('forward'),sense=call('sense')),
        metrics_manager=NS(compute_substep=call('metrics_substep'),compute=call('metrics')),
        command_manager=NS(compute=call('command')),recorder_manager=NS(record_post_step=call('post_step')))
    def reward(**kwargs):calls.append(('reward',kwargs));return torch.zeros(2)
    def observation(**kwargs):calls.append(('observation',kwargs));return {'policy':torch.ones(2,3)}
    env.reward_manager=NS(compute=reward);env.observation_manager=NS(compute=observation)
    return env,calls


class EmptySignalControl(unittest.TestCase):
    def test_matches_installed_upstream_chronology_counters_and_outputs(self):
        left,lc=fixture();right,rc=fixture();action=torch.zeros(2,20)
        a=ManagerBasedRlEnv.step(left,action);b=empty_signal_control_step(right,action)
        self.assertEqual(lc,rc);self.assertEqual(left._sim_step_counter,right._sim_step_counter);self.assertEqual(left.common_step_counter,right.common_step_counter)
        torch.testing.assert_close(left.episode_length_buf,right.episode_length_buf)
        for key in a[0]:torch.testing.assert_close(a[0][key],b[0][key])
        for i in (1,2,3):torch.testing.assert_close(a[i],b[i])
        self.assertEqual(a[4],b[4]);self.assertEqual(sum(n=='native' for n,k in rc),8)

    def test_removes_exactly_the_known_empty_nonzero_query(self):
        original=torch.Tensor.nonzero;counts=[]
        def counted(tensor,*args,**kwargs):counts.append(tuple(tensor.shape));return original(tensor,*args,**kwargs)
        with patch.object(torch.Tensor,'nonzero',counted):
            env,_=fixture();ManagerBasedRlEnv.step(env,torch.zeros(2,20));self.assertEqual(len(counts),1)
            counts.clear();env,_=fixture();empty_signal_control_step(env,torch.zeros(2,20));self.assertEqual(counts,[])

    def test_unsupported_signals_or_events_reject_before_actions(self):
        for key in ['auto_reset','terminations','rewards','metrics','curriculum','recorders']:
            env,calls=fixture();setattr(env.cfg,key,True if key=='auto_reset' else {'unexpected':1})
            with self.assertRaises(ValueError):empty_signal_control_step(env,torch.zeros(2,20))
            self.assertEqual(calls,[])
        env,calls=fixture();env.event_manager.available_modes.append('step')
        with self.assertRaises(ValueError):empty_signal_control_step(env,torch.zeros(2,20))
        self.assertEqual(calls,[])

    def test_manual_pending_and_hidden_active_terms_are_not_bypassed(self):
        env,calls=fixture();env._manual_reset_pending[0]=True
        with self.assertRaises(RuntimeError):empty_signal_control_step(env,torch.zeros(2,20))
        self.assertEqual(calls,[])
        env,calls=fixture();env.termination_manager._term_names=['hidden']
        with self.assertRaises(ValueError):empty_signal_control_step(env,torch.zeros(2,20))
        self.assertEqual(calls,[])
