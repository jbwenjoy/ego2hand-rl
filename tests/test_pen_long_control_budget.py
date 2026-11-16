import importlib.util
from pathlib import Path
import tomllib
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('long_control_entry',ROOT/'tools/train_pen_support_pilot.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class LongControlBudget(unittest.TestCase):
    def config(self):return tomllib.loads((ROOT/'configs/pen_long_control_training_v1.toml').read_text())

    def test_long_budget_requires_explicit_registration_not_historical_alias(self):
        c=self.config();self.assertTrue(module.registered_budget(c));c.pop('study')
        with self.assertRaises(ValueError):module.registered_budget(c)

    def test_first_reload_and_fixed_work_cannot_be_silently_changed(self):
        for change in [{'first_checkpoint_reload':False},{'pilot_iterations':1000},{'num_envs':64},{'checkpoint_interval':1},{'fixed_frame_features':'LIVE'}]:
            c=self.config();c.update(change)
            with self.assertRaises(ValueError):module.registered_budget(c)

    def test_legacy_cached_two_round_trial_remains_accepted(self):
        c=tomllib.loads((ROOT/'configs/pen_cached_support_training_v1.toml').read_text());self.assertFalse(module.registered_budget(c))
