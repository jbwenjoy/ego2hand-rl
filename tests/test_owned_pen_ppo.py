"""Collector ordering and update/schema guards without a physical simulator."""
from types import SimpleNamespace
import unittest
import torch
import warp as wp
from supermimic_research.runtime.owned_pen_runtime import OwnedPenRuntime
from supermimic_research.learning.native_pen_ppo import NativePenPPO
from supermimic_research.learning.owned_pen_ppo import OwnedPenPPO,SCHEMA


CONFIG=dict(seed=7411,ppo_rollout_steps=2,ppo_epochs=1,ppo_minibatches=1,ppo_clip=.2,ppo_entropy=.001,ppo_value_coef=.5,ppo_max_grad_norm=1.,
    ppo_normalize_advantage=True,ppo_lr=.0001,weight_decay=0.,ppo_init_std=.5,ppo_std_min=.2,ppo_std_max=50.,ppo_gamma=.996,lam=.95)


class FakeRuntime(OwnedPenRuntime):
    def __init__(self):
        wp.init();self.device='cpu';self.num_envs=1;self._collection_open=False;self.fail=False;self.resets=0
        def zeros(shape,dtype=float):return wp.zeros(shape,dtype=dtype,device='cpu')
        raw=SimpleNamespace(time=zeros(1),ctrl=zeros((1,20)),overflow=zeros(1,int),qfrc_applied=zeros((1,26)),xfrc_applied=zeros((1,3,6)))
        self.env=SimpleNamespace(sim=SimpleNamespace(wp_data=raw));self.motion=SimpleNamespace(time_steps=torch.ones(1,dtype=torch.long))
        self.action=SimpleNamespace(_processed_actions=torch.zeros(1,20),_targets=torch.zeros(1,20),_position_ctrl_ids=torch.arange(20))
        self.contact_views=dict(geometry_max=torch.zeros(1,3,dtype=torch.float64),Fn6_sum=torch.zeros(1,6,dtype=torch.float64),nonhand_max_sum=torch.zeros(1,2,dtype=torch.float64),
            wrench_sum=torch.zeros(1,8,6,dtype=torch.float64),counts=torch.zeros(1,2,dtype=torch.int32),errors=torch.zeros(1,dtype=torch.int32),global_error=torch.zeros(1,dtype=torch.int32),
            source_R=torch.eye(3)[None],source_C=torch.zeros(1,3),clock=torch.zeros(1,2,dtype=torch.float64))
    def begin_collection(self):self._collection_open=True
    def end_collection(self):self._collection_open=False
    def observation(self):return dict(policy=torch.zeros(1,161),critic=torch.zeros(1,188))
    def step(self,raw):
        if self.fail:raise ValueError('Synthetic collector failure')
        phase=self.motion.time_steps.clone();self.motion.time_steps+=1;v=self.contact_views
        v['counts'][0]=torch.tensor([8,2],dtype=torch.int32);v['Fn6_sum'][0,0]=4.;v['source_R'][0]=torch.eye(3);v['clock'][0]=torch.tensor([0.,.02],dtype=torch.float64)
        wp.to_torch(self.env.sim.wp_data.time).fill_(.02)
        return self.observation(),torch.zeros(1),torch.ones(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool),dict(applied_raw=raw.clamp(-1,1),reference_phase=phase,reference_joints=torch.zeros(1,20),contact_window=v)
    def reset(self,ids):
        self.resets+=1;self.motion.time_steps.fill_(1)
        for value in self.contact_views.values():value.zero_()
        return self.observation()


class OwnedPPOTests(unittest.TestCase):
    def test_contact_rows_are_owned_before_terminal_reset_erases_source(self):
        r=FakeRuntime();p=OwnedPenPPO(r,CONFIG);m=p.collect(2,forced_actions=torch.zeros(2,1,20))
        self.assertEqual(r.resets,2);self.assertFalse(r.contact_views['counts'].any())
        self.assertTrue((m['contact_counts'][...,0]==8).all());self.assertTrue((m['contact_Fn6_sum'][...,0]==4).all())
        self.assertNotEqual(m['contact_counts'].data_ptr(),r.contact_views['counts'].data_ptr())
        self.assertFalse(r._collection_open);self.assertNotIn('step',r.__dict__)

    def test_failed_rollout_unhooks_and_cannot_be_checkpointed(self):
        r=FakeRuntime();p=OwnedPenPPO(r,CONFIG);r.fail=True
        with self.assertRaisesRegex(ValueError,'Synthetic collector'):p.collect(2,forced_actions=torch.zeros(2,1,20))
        self.assertFalse(r._collection_open);self.assertFalse(p._collecting);self.assertNotIn('step',r.__dict__)
        with self.assertRaisesRegex(ValueError,'partially initialized'):p.checkpoint({})

    def test_one_complete_rollout_can_be_updated_only_once(self):
        r=FakeRuntime();p=OwnedPenPPO(r,CONFIG);p.collect(2,forced_actions=torch.zeros(2,1,20));p.update()
        self.assertEqual(p.iteration,1);self.assertEqual(p.optimizer_steps,1)
        with self.assertRaisesRegex(ValueError,'Exactly one update'):p.update()
        with self.assertRaisesRegex(ValueError,'before this rollout'):p.consistency()

    def test_legacy_and_owned_training_schemas_reject_each_other_immediately(self):
        owned=OwnedPenPPO.__new__(OwnedPenPPO);old=NativePenPPO.__new__(NativePenPPO)
        with self.assertRaisesRegex(ValueError,'training2'):owned.load({'schema':'pen_native_full_pose_PPO_training:1'})
        with self.assertRaisesRegex(ValueError,'checkpoint/config'):old.load({'schema':SCHEMA})


if __name__=='__main__':unittest.main()
