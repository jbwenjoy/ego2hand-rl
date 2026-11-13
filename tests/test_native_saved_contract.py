import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from supermimic_research.mechanics.native_saved_contract import saved_chain_checks


class NativeSavedContractTests(unittest.TestCase):
    def fixture(self):
        lengths = np.array([64, 64, 3])
        ends = np.cumsum(lengths)
        a = dict(k=np.arange(3), actual_duration_s=lengths * .0003125,
                 complete_H=lengths == 64, terminated=np.array([False, False, True]))
        for field, width in [('action20', 20), ('EMA_before20', 20), ('EMA_after20', 20),
                             ('actual_control20', 20), ('source_native_state', 4),
                             ('next_native_state', 4), ('source55', 55), ('next55', 55),
                             ('warmstart26_before', 26), ('warmstart26_after', 26)]:
            a[field] = np.zeros((3, width))
        z = dict(tick_ranges=np.c_[np.r_[0, ends[:-1]], ends])
        for field in ['pre_qpos', 'post_qpos', 'pre_qvel', 'post_qvel']:
            z[field] = np.zeros((ends[-1], 2))
        return a, z

    def checks(self, a, z):
        return saved_chain_checks(a, z, .0003125, np.zeros(20), np.ones(20)*.05, np.ones(20)*.05)

    def test_preserves_partial_terminal_row(self):
        self.assertTrue(all(self.checks(*self.fixture()).values()))

    def test_rejects_plausible_but_discontinuous_saved_sequences(self):
        for field, check in [('source_native_state', 'native_state_carry'),
                             ('EMA_before20', 'EMA_carry'), ('source55', 'observed_state_carry'),
                             ('warmstart26_before', 'warmstart_carry')]:
            with self.subTest(field=field):
                a, z = self.fixture()
                a[field][1, 0] = .001
                self.assertFalse(self.checks(a, z)[check])

    def test_rejects_hidden_reset_and_relabelled_terminal_duration(self):
        a, z = self.fixture()
        z['pre_qpos'][80, 0] = .001
        self.assertFalse(self.checks(a, z)['per_tick_pose_carry'])
        a['complete_H'][-1] = True
        a['actual_duration_s'][-1] = .02
        checks = self.checks(a, z)
        self.assertFalse(checks['complete_flags'])
        self.assertFalse(checks['interval_duration'])


if __name__ == '__main__':
    unittest.main()
