import json
import tempfile
import unittest
import copy
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import torch
from supermimic_research.learning.pen_matched_recovery import (
    budget, checked_health, json_metadata, persist_failed_learner, WorkCounter,
    saved_bytes_equal, validate_recovery_result, SCHEMA)
from supermimic_research.artifacts import file_pin


class MatchedRecoveryTests(unittest.TestCase):
    def config(self):
        return dict(schema=1, study='MATCHED_RESUME_200', start_iteration=200, final_iteration=2000,
                    num_envs=32, rollout_controls=64, checkpoint_interval=100, arm_order=['RESIDUAL', 'SUPPORT'])

    def test_additional_causal_and_total_work_do_not_double_count_prefix(self):
        b = budget(self.config())
        self.assertEqual(b['reused_prefix_native_total'] + b['added_native_total'], b['causal_policy_native_total'])
        self.assertEqual(b['reused_prefix_optimizer_total'] + b['added_optimizer_total'], b['causal_policy_optimizer_total'])
        self.assertEqual(36143104 + 98304 + 4928 + b['added_native_total'], 95228736)
        self.assertEqual(70560 + 192 + b['added_optimizer_total'], 185952)
        cfg = self.config()
        cfg['start_iteration'] = 205
        with self.assertRaises(ValueError):
            budget(cfg)

    def test_failed_flags_are_written_before_rejection_and_update(self):
        health = dict(checks={'finite': False}, passed=0, total=1, mean_raw_reward=float('nan'))
        calls = []
        learner = SimpleNamespace(consistency=lambda: health, update=lambda: calls.append('updated'))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'health.json'
            with self.assertRaisesRegex(ValueError, 'finite'):
                checked_health(learner, path)
                learner.update()
            saved = json.loads(path.read_text())
            self.assertFalse(saved['checks']['finite'])
            self.assertEqual(saved['mean_raw_reward'], {'nonfinite_float': 'nan'})
            self.assertEqual(calls, [])

    def test_attempted_and_completed_steps_remain_distinct_on_exception(self):
        counter = WorkCounter()
        sim = SimpleNamespace(num_envs=32)
        counter.native_call(lambda s: None, sim)
        def fail(*args):
            raise RuntimeError('partial operation')
        with self.assertRaises(RuntimeError):
            counter.native_call(fail, sim)
        optimizer = SimpleNamespace(step=lambda: None)
        counter.bind_optimizer(optimizer)
        optimizer.step()
        broken = SimpleNamespace(step=fail)
        counter.bind_optimizer(broken)
        with self.assertRaises(RuntimeError):
            broken.step()
        self.assertEqual((counter.native, counter.native_attempted, counter.optimizer, counter.optimizer_attempted), (32, 64, 1, 2))
        self.assertFalse(counter.metadata()['completed_call_accounting_exact'])

    def test_failed_nan_payload_preserved_and_incomplete_tail_excluded(self):
        actor, critic = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        memory = torch.arange(64, dtype=torch.float32)
        memory[1] = float('nan')
        learner = SimpleNamespace(memory={'raw_reward': memory}, _rollout_complete=False, _captured_rows=2,
            actor=actor, critic=critic, optimizer=torch.optim.Adam(list(actor.parameters()) + list(critic.parameters())),
            observations={'policy': torch.tensor([float('nan')])}, iteration=200, optimizer_steps=6400,
            training_transitions=409664, adapter=object())
        with tempfile.TemporaryDirectory() as folder:
            result = persist_failed_learner(learner, Path(folder), {}, native_capture=lambda r, s: {'untrusted': True})
            saved = torch.load(result['failed_memory']['path'], weights_only=False)
            self.assertEqual(saved['memory']['raw_reward'].shape, (2,))
            self.assertTrue(torch.isnan(saved['memory']['raw_reward'][1]))
            self.assertEqual(saved['memory']['raw_reward'].view(torch.int32)[1], memory.view(torch.int32)[1])
            self.assertTrue(saved['incomplete_fields_may_include_uninitialized_storage'])
            self.assertIn('NOT_RESUMABLE', saved['schema'])
            self.assertIn('failed_policy', result)
            self.assertIn('failed_native', result)

    def test_nonfinite_metadata_remains_explicit(self):
        value = json_metadata({'nested': [float('inf'), float('-inf'), 1.]})
        self.assertEqual(value['nested'][:2], [{'nonfinite_float': 'inf'}, {'nonfinite_float': '-inf'}])
        json.dumps(value, allow_nan=False)

    def test_full_saved_state_compares_numpy_and_tensor_bytes(self):
        value = {'native': np.array([np.nan, -0.], dtype=np.float32), 'nn': torch.tensor([1.])}
        self.assertTrue(saved_bytes_equal(value, copy.deepcopy(value)))
        changed = copy.deepcopy(value)
        changed['native'][1] = 0.
        self.assertFalse(saved_bytes_equal(value, changed))
        self.assertFalse(saved_bytes_equal({'scalar': -0.}, {'scalar': 0.}))

    def report_fixture(self, folder):
        parents = {'RESIDUAL': {'sha256': 'R'}, 'SUPPORT': {'sha256': 'S'}}
        reg = dict(schema='pen_matched_recovery_registration:1', config={'sha256': 'config'},
                   inputs={'objective_config': {}}, budgets=budget(self.config()), parents=parents)
        path = folder / 'registration.json'
        path.write_text(json.dumps(reg))
        arm = dict(start_iteration=200, iterations=2000, added_iterations=1800,
                   actual_native_integrations=29491200, added_optimizer_steps=57600,
                   optimizer_steps=64000, training_transitions=4096000)
        report = dict(schema=SCHEMA, config={}, registration=file_pin(path), recovery_config=reg['config'],
                      passed=5, total=5, checks={str(i): True for i in range(5)}, actual_native_integrations=58982400,
                      optimizer_steps=115200, causal_policy_budget=reg['budgets'],
                      unique_named_lineage_total=dict(native_integrations=95228736, optimizer_steps=185952),
                      arms=[dict(arm, arm=name, parent=parents[name]) for name in ['RESIDUAL', 'SUPPORT']])
        return report, file_pin(path)

    def test_final_admission_rejects_wrong_parent_or_policy_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            original, reg = self.report_fixture(Path(directory))
            validate_recovery_result(original, {}, reg)
            for key, value in [('parent', {'sha256': 'other'}), ('optimizer_steps', 57600), ('added_iterations', 1795)]:
                report = copy.deepcopy(original)
                report['arms'][1][key] = value
                with self.assertRaises(ValueError):
                    validate_recovery_result(report, {}, reg)

    def test_final_admission_rejects_hidden_failed_cost_or_legacy_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            original, reg = self.report_fixture(Path(directory))
            report = copy.deepcopy(original)
            report['unique_named_lineage_total']['native_integrations'] = 65536000
            with self.assertRaises(ValueError):
                validate_recovery_result(report, {}, reg)
            report = copy.deepcopy(original)
            report['schema'] = 'pen_geometry_support_long_control_training:2'
            with self.assertRaises(ValueError):
                validate_recovery_result(report, {}, reg)


if __name__ == '__main__':
    unittest.main()
