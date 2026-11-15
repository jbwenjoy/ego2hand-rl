import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from torch.distributions import Normal
from dex_supermimic.rl.dex_supermimic.ppo import gae_targets
from supermimic_research.learning.native_pen_ppo import NativePenPPO,timeout_reward,model_pair,validate_modules
from supermimic_research.learning.native_pen_ppo import cpu_tree,tensor_tree_equal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from review_native_pen_ppo import clone_training_modules


class CountNorm:
    def __init__(self):self.count=0
    def update(self,x):self.count+=len(x)


class ToyActor(nn.Module):
    def __init__(self):super().__init__();self.mean=nn.Parameter(torch.full((20,),.2));self.norm=CountNorm()
    def distribution(self,x,*,validate_args=None):return Normal(self.mean.expand(len(x),-1),torch.full((len(x),20),.5),validate_args=validate_args)


class ToyCritic(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(1.));self.norm=CountNorm()
    def forward(self,x):return x[:,0]*self.weight


class FakeAdapter:
    def __init__(self):
        self.num_envs=3;self.device='cpu';self.ticks=0
        self.motion=SimpleNamespace(time_steps=torch.ones(3,dtype=torch.long))
        self.action=SimpleNamespace(_processed_actions=torch.zeros(3,20),_targets=torch.zeros(3,20),_position_ctrl_ids=torch.arange(20))
        self.views=dict(time=torch.zeros(3),ctrl=torch.zeros(3,20),overflow=torch.zeros(3,dtype=torch.int32),qfrc_applied=torch.zeros(3,26),xfrc_applied=torch.zeros(3,1,6))
        self.env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace()))
        self.obs=dict(policy=torch.zeros(3,161),critic=torch.zeros(3,188));self.obs['critic'][:,0]=torch.tensor([1.,2,3])
    def reset(self,ids):
        self.motion.time_steps[ids]=1;self.action._processed_actions[ids]=0;self.action._targets[ids]=0
        p=self.obs['policy'].clone();c=self.obs['critic'].clone();p[ids]=0;c[ids]=0;c[ids,0]=-100
        self.obs=dict(policy=p,critic=c);return self.obs
    def step(self,sample):
        self.ticks+=1;used=self.motion.time_steps.clone();applied=sample.clamp(-1,1)
        ref=used[:,None].expand(-1,20).float()*.001
        self.action._processed_actions.mul_(.5).add_(applied,alpha=.25)
        self.action._targets.copy_(ref+self.action._processed_actions);self.views['ctrl'].copy_(self.action._targets)
        self.views['time'].add_(.02);self.motion.time_steps.add_(1)
        p=torch.zeros(3,161);p[:,141:]=applied;c=torch.zeros(3,188);c[:,0]=torch.tensor([10.,20,30])+self.ticks
        self.obs=dict(policy=p,critic=c)
        return self.obs,torch.tensor([1.,2,3]),torch.tensor([False,True,False]),torch.tensor([False,False,True]),dict(applied_raw=applied,reference_phase=used,reference_joints=ref)


def fake_runner():
    a=FakeAdapter();r=NativePenPPO.__new__(NativePenPPO);r.adapter=a;r.device='cpu';r.actor=ToyActor();r.critic=ToyCritic()
    r.cfg=SimpleNamespace(ppo_gamma=.9);r.observations=a.obs;r.memory=None;r.normalization_batch=None
    r.normalizers_ready=False;r.training_transitions=0;r.native_views=a.views
    return r


class NativePPOTests(unittest.TestCase):
    def test_timeout_uses_next_terminal_value_once_and_termination_wins(self):
        raw=torch.tensor([1.,2,3,4]);v=torch.tensor([10.,20,30,40])
        term=torch.tensor([False,True,False,True]);trunc=torch.tensor([False,False,True,True])
        torch.testing.assert_close(timeout_reward(raw,v,term,trunc,.9),torch.tensor([1.,2,30.,4]))
        torch.testing.assert_close(raw,torch.tensor([1.,2,3,4]))

    def test_mixed_native_collection_preserves_terminal_observation_before_reset(self):
        r=fake_runner();m=r.collect(2,forced_actions=torch.full((2,3,20),1.5))
        self.assertTrue(all(r.consistency()['checks'].values()))
        self.assertEqual(r.training_transitions,6)
        torch.testing.assert_close(m['terminal_next_value'][:,2],torch.tensor([31.,32.]))
        self.assertEqual(float(r.observations['critic'][2,0]),-100.)
        self.assertEqual(int(m['phase_source'][1,0]),2);self.assertEqual(int(m['phase_source'][1,2]),1)
        torch.testing.assert_close(m['actions'],torch.full((2,3,20),1.5));torch.testing.assert_close(m['applied_actions'],torch.ones(2,3,20))

    def test_hand_calculated_GAE_never_crosses_terminated_or_timeout_reset(self):
        r=fake_runner();m=r.collect(2,forced_actions=torch.zeros(2,3,20))
        adv,ret=gae_targets(m,r.critic(r.observations['critic']),gamma=.9,lam=.95,normalize=False)
        torch.testing.assert_close(adv[0],torch.tensor(10.584),atol=1e-5,rtol=0)
        torch.testing.assert_close(ret.reshape(2,3)[:,1],torch.tensor([2.,2.]))
        torch.testing.assert_close(ret.reshape(2,3)[:,2],torch.tensor([30.9,31.8]))

    def test_normalizers_update_only_between_rollout_blocks(self):
        r=fake_runner();r.collect(2,forced_actions=torch.zeros(2,3,20))
        self.assertEqual(r.actor.norm.count,3);self.assertEqual(r.critic.norm.count,3)
        r.collect(2,forced_actions=torch.zeros(2,3,20))
        self.assertEqual(r.actor.norm.count,9);self.assertEqual(r.critic.norm.count,9)
        self.assertTrue(r.consistency()['checks']['old_logprob_frozen'])

    def test_wrong_module_shape_rejected_before_mutating_parameters(self):
        c=SimpleNamespace(ppo_init_std=.5,ppo_std_min=.2,ppo_std_max=50.,ppo_lr=1e-4,weight_decay=0.)
        actor,critic,opt=model_pair('cpu',c)
        saved=dict(modules=dict(actor=copy.deepcopy(actor.state_dict()),critic=copy.deepcopy(critic.state_dict())),
            optimizer=opt.state_dict(),parameter_order=[prefix+'.'+name for prefix,mod in [('actor',actor),('critic',critic)] for name,_ in mod.named_parameters()])
        validate_modules(actor,critic,opt,saved);original=actor.net[0].weight.detach().clone()
        saved['modules']['actor']['net.0.weight']=saved['modules']['actor']['net.0.weight'][:,:155]
        with self.assertRaisesRegex(ValueError,'layout mismatch'):validate_modules(actor,critic,opt,saved)
        torch.testing.assert_close(actor.net[0].weight,original,atol=0,rtol=0)

    def test_legacy_cube_checkpoint_is_rejected_at_schema_boundary(self):
        r=NativePenPPO.__new__(NativePenPPO)
        with self.assertRaisesRegex(ValueError,'checkpoint/config'):r.load({'schema':'legacy_cube_format4'})

    def test_optimizer_clones_do_not_share_CPU_step_with_source_or_other_clone(self):
        c=dict(ppo_init_std=.5,ppo_std_min=.2,ppo_std_max=50.,ppo_lr=1e-4,weight_decay=0.)
        actor,critic,opt=model_pair('cpu',SimpleNamespace(**c))
        loss=actor.net[0].weight.square().mean()+critic.net[0].weight.square().mean();loss.backward();opt.step()
        state=dict(config=c,modules=dict(actor=cpu_tree(actor.state_dict()),critic=cpu_tree(critic.state_dict())),optimizer=cpu_tree(opt.state_dict()))
        source=copy.deepcopy(state['optimizer']);a,ac,ao=clone_training_modules(state,'cpu');b,bc,bo=clone_training_modules(state,'cpu')
        (a.net[0].weight.square().mean()+ac.net[0].weight.square().mean()).backward();ao.step()
        self.assertTrue(tensor_tree_equal(source,state['optimizer']))
        self.assertTrue(tensor_tree_equal(source,cpu_tree(bo.state_dict())))
        self.assertFalse(tensor_tree_equal(source,cpu_tree(ao.state_dict())))


if __name__=='__main__':unittest.main()
