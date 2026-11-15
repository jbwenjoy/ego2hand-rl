"""Frozen-runner parity, target gradients, minibatch execution and optimizer resume."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor, Critic
from dex_supermimic.rl.dex_supermimic.ppo import gae_targets, ppo_update
from supermimic_research.frozen_sources import verify_source_snapshot

ROOT = Path(__file__).resolve().parents[1]


def runner_update_at(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DexSuperMimicRunner')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'ppo_update')
    module = ast.Module(body=[fn], type_ignores=[])
    ns = dict(torch=torch, nn=torch.nn)
    exec(compile(module, str(path), 'exec'), ns)
    return ns['ppo_update']


def frozen_legacy_update():
    receipt = json.loads((ROOT/'archive/pen_pre_shared_ppo_v1/source_receipt.json').read_text())
    verify_source_snapshot(receipt)
    path = next(Path(r['snapshot']['path']) for r in receipt['records'] if Path(r['observed']['path']).name == 'runner.py')
    return runner_update_at(path)


def fixture(*, minibatches=2, epochs=2, count=6):
    torch.manual_seed(1729)
    actor = GaussianPPOActor(161, 20, hidden=(16, 8))
    critic = Critic(17, amp=False, empirical_norm=True)
    optimizer = torch.optim.AdamW(tuple(actor.parameters())+tuple(critic.parameters()), lr=1e-4)
    T, N = count//2, 2
    obs = torch.randn(T, N, 161); co = torch.randn(T, N, 17)
    with torch.no_grad():
        actor.norm.update(obs.reshape(-1, 161)); critic.norm.update(co.reshape(-1, 17))
        dist = actor.distribution(obs); action = dist.sample()
        memory = dict(obs=obs, critic_obs=co, actions=action,
            log_prob=dist.log_prob(action).sum(-1), value=critic(co), reward=torch.randn(T, N),
            terminated=torch.zeros(T, N), truncated=torch.zeros(T, N))
    memory['terminated'][0, 0]=1; memory['truncated'][0, 1]=1
    # Collector has already applied its timeout reward correction.
    memory['reward'][0, 1] += .996*memory['value'][0, 1]
    final = torch.randn(N, 17)
    cfg=SimpleNamespace(ppo_gamma=.996, lam=.95, ppo_normalize_advantage=True,
        ppo_minibatches=minibatches, ppo_epochs=epochs, ppo_clip=.2,
        ppo_value_coef=.5, ppo_entropy=.001, ppo_max_grad_norm=1.)
    return actor, critic, optimizer, memory, final, cfg


def legacy_run(parts, *, detach_bootstrap=False):
    actor, critic, optimizer, memory, final, cfg = parts
    handle = critic.register_forward_hook(lambda module,args,result:
        result.detach() if args[0] is final else result) if detach_bootstrap else None
    runner = SimpleNamespace(cfg=cfg, ppo_memory=memory, device='cpu', actor=actor,
        optimizers={'ppo':optimizer}, _ensure_ppo_critic=lambda x:critic, _critic_observation=lambda:final)
    try:
        return frozen_legacy_update()(runner)
    finally:
        if handle: handle.remove()


def equal_nested(a,b):
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b


class SharedPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_frozen_original_reproduces_bootstrap_graph_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'backward through the graph a second time|modified by an inplace operation'):
            legacy_run(fixture())

    def test_frozen_original_single_update_numerical_parity_when_final_done(self):
        a=fixture(minibatches=1,epochs=1); b=fixture(minibatches=1,epochs=1)
        a[3]['terminated'][-1]=1; b[3]['terminated'][-1]=1
        rng=torch.get_rng_state(); old=legacy_run(a)
        torch.set_rng_state(rng); new=ppo_update(*b)
        for x,y in zip(a[:3],b[:3]):self.assertTrue(equal_nested(x.state_dict(),y.state_dict()))
        for key in old:self.assertTrue(equal_nested(old[key],new[key]),key)

    def test_multibatch_matches_frozen_algorithm_with_only_bootstrap_detached(self):
        a=fixture(); b=fixture()
        rng=torch.get_rng_state(); old=legacy_run(a,detach_bootstrap=True)
        torch.set_rng_state(rng); new=ppo_update(*b)
        for x,y in zip(a[:3],b[:3]):self.assertTrue(equal_nested(x.state_dict(),y.state_dict()))
        for key in old:self.assertTrue(equal_nested(old[key],new[key]),key)
        self.assertEqual(new['optimizer_steps'],4)
        self.assertEqual(new['omitted_transitions_per_epoch'],0)
        self.assertGreater(float(new['approx_kl']),0)
        self.assertEqual(float(new['kl']),0)  # Historical metric has this limitation.

    def test_gae_values_boundaries_and_no_target_gradients(self):
        value=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
        memory=dict(value=value,reward=torch.tensor([[2.,4.],[6.,8.]],requires_grad=True),
            terminated=torch.tensor([[1,0],[0,0]]),truncated=torch.tensor([[0,1],[0,0]]))
        adv,ret=gae_targets(memory,torch.tensor([5.,6.],requires_grad=True),gamma=.9,lam=.8,normalize=False)
        # End bootstrap: [6+.9*5-3, 8+.9*6-4]; first row terminal/timeout cuts recurrence.
        torch.testing.assert_close(adv,torch.tensor([1.,2.,7.5,9.4]))
        torch.testing.assert_close(ret,torch.tensor([2.,4.,10.5,13.4]))
        self.assertFalse(adv.requires_grad); self.assertFalse(ret.requires_grad)
        memory['terminated'][0,0]=0
        adv,_=gae_targets(memory,torch.tensor([5.,6.]),gamma=.9,lam=.8,normalize=False)
        self.assertAlmostEqual(float(adv[0]),2+.9*3-1+.9*.8*7.5,places=5)

    def test_optimizer_rng_resume_and_fixed_rollout_immutability(self):
        a=fixture(); original=copy.deepcopy(a[3]); ppo_update(*a)
        state=[copy.deepcopy(x.state_dict()) for x in a[:3]]; rng=torch.get_rng_state()
        expected=ppo_update(*a)
        b=fixture()
        for module,saved in zip(b[:3],state):module.load_state_dict(saved)
        torch.set_rng_state(rng); actual=ppo_update(*b)
        for x,y in zip(a[:3],b[:3]):self.assertTrue(equal_nested(x.state_dict(),y.state_dict()))
        self.assertTrue(equal_nested(expected,actual));self.assertTrue(equal_nested(original,a[3]))

    def test_legacy_remainder_is_explicit_and_invalid_sizes_rejected(self):
        parts=fixture(minibatches=4,count=6)
        result=ppo_update(*parts)
        self.assertEqual(result['used_transitions_per_epoch'],4)
        self.assertEqual(result['omitted_transitions_per_epoch'],2)
        parts[-1].ppo_minibatches=7
        with self.assertRaises(ValueError):ppo_update(*parts)

    def test_live_cube_wrapper_delegates_to_shared_update(self):
        a=fixture(); b=fixture()
        actor,critic,optimizer,memory,final,cfg=a
        runner=SimpleNamespace(actor=actor,optimizers={'ppo':optimizer},ppo_memory=memory,cfg=cfg,
            _ensure_ppo_critic=lambda x:critic,_critic_observation=lambda:final)
        # Compile only the live wrapper; importing its simulator is unnecessary.
        fn=runner_update_at(ROOT/'src/dex_supermimic/rl/dex_supermimic/runner.py')
        fn.__globals__['__package__']='dex_supermimic.rl.dex_supermimic'
        rng=torch.get_rng_state();actual=fn(runner)
        torch.set_rng_state(rng);expected=ppo_update(*b)
        self.assertTrue(equal_nested(actual,expected))
        for x,y in zip(a[:3],b[:3]):self.assertTrue(equal_nested(x.state_dict(),y.state_dict()))


if __name__=='__main__':unittest.main()
