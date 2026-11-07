"""CPU synthetic own-payload health contracts, zero learning/native calls.

Tiny two-parameter networks replace the constructors only in this fixture.
Adam moments and gradients are explicit fixture values, not trained artifacts.
No optimizer.step or backward is called; this does not qualify real model size.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

# Isolate CPU-only fixture before importing Torch/constructing CPU AdamW.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from supermimic_research.learning import checkpoint as C, world_model as WM, update_step as U


class TinyNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(2))
        self.register_buffer('scale', torch.ones(2))
        self.register_buffer('count', torch.tensor(7, dtype=torch.int64))

    def forward(self, x):
        return (x - self.mean) / self.scale


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 2))
        self.bias = torch.nn.Parameter(torch.randn(2))
        self.norm = TinyNorm()

    def forward(self, x):
        return self.norm(x) @ self.weight.T + self.bias


class CheckpointContracts(unittest.TestCase):
    def setUp(self):
        self.original_rng = WM.capture_rng()
        random.seed(12)
        np.random.seed(13)
        torch.manual_seed(14)
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.path = self.directory / 'synthetic.pt'
        self.counts = {'WM_updates': 10, 'paired_updates': 10}
        arms = {mode: dict(actor=TinyNet(), critic=TinyNet(), target_critic=TinyNet())
                for mode in ['A1', 'A2']}
        self.context = dict(torch=torch, device='cpu', model=TinyNet(), arms=arms,
            c={'model': {'hidden': [2, 2]}, 'training': {'seed': 1734}},
            norm={'input_std': torch.ones(2)}, scales={'head': torch.tensor([.1, .2])},
            physical={'mass': torch.tensor(.2, dtype=torch.float64)},
            generator=torch.Generator(device='cpu').manual_seed(1735),
            provenance={'source': 'synthetic_tiny_not_real_training'})
        self.modules = {'WM': self.context['model']}
        for mode, arm in arms.items():
            self.modules[mode + '_Actor'] = arm['actor']
            self.modules[mode + '_Critic'] = arm['critic']
        self.optimizers, self.gradients = {}, {}
        for key, module in self.modules.items():
            opt = torch.optim.AdamW(module.parameters(), lr=1e-4, foreach=False)
            for parameter in module.parameters():
                opt.state[parameter] = dict(step=torch.tensor(10.),
                    exp_avg=torch.full_like(parameter, .01),
                    exp_avg_sq=torch.full_like(parameter, .02))
                parameter.grad = torch.full_like(parameter, .03)
            self.optimizers[key] = opt
            self.gradients[key] = dict(tag='PRE_SYNTHETIC_10',
                pre_clip={name: p.grad.detach().clone() for name, p in module.named_parameters()},
                post_clip={name: p.grad.detach().clone() for name, p in module.named_parameters()},
                clip_returned_norm=torch.tensor(.1))
        self.probe = torch.tensor([[.25, -.5], [1., 2.]])
        self.parent_prefix = self._prefix('parent.jsonl', 1, 5)
        self.current_prefix = self._prefix('current.jsonl', 6, 10)
        self.payload = self._payload()
        self._save()
        self.wm_constructor = patch.object(C.WM, 'make_model', side_effect=lambda *a, **k: TinyNet())
        self.arm_constructor = patch.object(C.IM, 'make_fresh_actor_critic',
            side_effect=lambda **k: dict(actor=TinyNet(), critic=TinyNet()))
        self.mock_wm = self.wm_constructor.start()
        self.mock_arms = self.arm_constructor.start()

    def tearDown(self):
        self.wm_constructor.stop()
        self.arm_constructor.stop()
        self.temporary.cleanup()
        WM.restore_rng(self.original_rng)

    def _prefix(self, name, first, last):
        path = self.directory / name
        raw = b''.join((json.dumps({'update': i, 'loss': float(i)}) + '\n').encode()
                       for i in range(first, last + 1))
        path.write_bytes(raw)
        return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    records=last - first + 1, range=[first, last])

    def _payload(self):
        s = self.context
        return dict(schema='synthetic_own_checkpoint:1', update=10, counts=self.counts.copy(),
            configuration=copy.deepcopy(s['c']), provenance=copy.deepcopy(s['provenance']),
            shared_WM=copy.deepcopy(s['model'].state_dict()), arms=copy.deepcopy(U.arm_states(s['arms'])),
            normalizers=copy.deepcopy(s['norm']), loss_scales=copy.deepcopy(s['scales']),
            physical_context=copy.deepcopy(s['physical']),
            optimizers=copy.deepcopy({key: opt.state_dict() for key, opt in self.optimizers.items()}),
            gradient_snapshots=copy.deepcopy(self.gradients), RNG=WM.capture_rng(),
            sample_generator=s['generator'].get_state().clone(),
            parent_log_prefixes=[self.parent_prefix], log_prefix=self.current_prefix)

    def _save(self):
        torch.save(self.payload, self.path)

    def _health(self, **kwargs):
        return C.checkpoint_health(self.path, self.context, self.optimizers, self.gradients,
            self.counts, self.probe, self.probe, schema='synthetic_own_checkpoint:1', **kwargs)

    def test_healthy_tiny_readback_and_append_only_prefix(self):
        for prefix in [self.parent_prefix, self.current_prefix]:
            with Path(prefix['path']).open('ab') as stream:
                stream.write(b'{"later":"outside_immutable_prefix"}\n')
        before = WM.capture_rng()
        health, evidence = self._health(extra_checks=lambda ck: {'online_cursor': ck['update'] == 10})
        self.assertEqual(health['schema'], C.HEALTH_SCHEMA)
        self.assertEqual(health['checkpoint_schema'], 'synthetic_own_checkpoint:1')
        self.assertEqual(health['passed'], health['total'])
        self.assertTrue(all(health['checks'].values()))
        self.assertEqual((health['extra_NN_FW'], health['extra_NN_constructors']), (10, 5))
        self.assertEqual((self.mock_wm.call_count, self.mock_arms.call_count), (1, 2))
        self.assertEqual(len(evidence), 10)
        for name, value in evidence.items():
            if name.startswith('expected_'):
                actual = evidence['restored_' + name[len('expected_'):]]
                self.assertEqual(value.dtype, np.float32)
                self.assertEqual(value.tobytes(), actual.tobytes())
        self.assertTrue(WM.raw_tree_equal(before, WM.capture_rng()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_offline_count_policy_is_not_hardcoded(self):
        self.counts['WM_updates'] = 110
        for value in self.optimizers['WM'].state.values():
            value['step'] = torch.tensor(110.)
        self.payload = self._payload()
        self._save()
        health, _ = self._health(extra_checks=lambda ck: {'offline_cursor': ck['counts']['WM_updates'] == 100 + ck['update']})
        self.assertEqual(health['passed'], health['total'])

    def test_schema_and_caller_cursor_are_hard_gates(self):
        with self.assertRaisesRegex(ValueError, 'online_cursor'):
            self._health(extra_checks=lambda ck: {'online_cursor': False})
        with self.assertRaisesRegex(ValueError, 'Python bools'):
            self._health(extra_checks=lambda ck: {'online_cursor': np.bool_(True)})
        self.payload['schema'] = 'foreign_checkpoint:1'
        self._save()
        with self.assertRaisesRegex(ValueError, 'artifact schema'):
            self._health()

    def test_missing_Adam_state_and_wrong_steps_cannot_pass_by_self_comparison(self):
        key = 'A1_Actor'
        parameter = next(self.modules[key].parameters())
        del self.optimizers[key].state[parameter]
        self.payload = self._payload()
        self._save()
        with self.assertRaisesRegex(ValueError, 'all_Adam_steps_and_moments'):
            self._health()
        self.optimizers[key].state[parameter] = dict(step=torch.tensor(9.),
            exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.ones_like(parameter))
        self.payload = self._payload()
        self._save()
        with self.assertRaisesRegex(ValueError, 'all_Adam_steps_and_moments'):
            self._health()

    def test_nonfinite_matching_target_and_gradients_still_fail(self):
        target = self.context['arms']['A1']['target_critic']
        with torch.no_grad():
            target.weight[0, 0] = float('nan')
        self.payload = self._payload()
        self._save()
        with self.assertRaisesRegex(ValueError, 'all_weights_and_norms_finite'):
            self._health()
        with torch.no_grad():
            target.weight[0, 0] = 1.
        name = next(iter(self.gradients['WM']['pre_clip']))
        self.gradients['WM']['pre_clip'][name].reshape(-1)[0] = float('nan')
        self.payload = self._payload()
        self._save()
        with self.assertRaisesRegex(ValueError, 'named_pre_post_gradients'):
            self._health()

    def test_raw_weight_bits_sampler_and_saved_rng_are_checked(self):
        with torch.no_grad():
            self.context['model'].bias[0] = -0.
        self.payload = self._payload()
        self.payload['shared_WM']['bias'][0] = 0.
        self._save()
        with self.assertRaisesRegex(ValueError, 'all_networks_and_norms'):
            self._health()
        self.payload = self._payload()
        self.payload['sample_generator'][0] ^= 1
        self._save()
        with self.assertRaisesRegex(ValueError, 'sample_generator'):
            self._health()
        self.payload = self._payload()
        self.payload['RNG']['torch'][0] ^= 1
        self._save()
        with self.assertRaisesRegex(ValueError, 'saved_RNG'):
            self._health()

    def test_changed_prefix_and_live_post_gradient_fail(self):
        raw = bytearray(Path(self.parent_prefix['path']).read_bytes())
        raw[raw.index(b'1.0')] = ord('2')
        Path(self.parent_prefix['path']).write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'log_prefix'):
            self._health()
        self.parent_prefix = self._prefix('parent.jsonl', 1, 5)
        next(self.modules['A2_Critic'].parameters()).grad.add_(1.)
        self.payload = self._payload()
        self._save()
        with self.assertRaisesRegex(ValueError, 'named_pre_post_gradients'):
            self._health()


if __name__ == '__main__':
    unittest.main()
