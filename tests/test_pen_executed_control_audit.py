import copy
import unittest
import numpy as np
from supermimic_research.runtime.pen_executed_control_audit import audit_executed_control, ACTION_CONTRACT


class ExecutedControlTests(unittest.TestCase):
    def fixture(self):
        # Explicit values exercise raw clipping, EMA history, encoder bias and
        # absolute-target limiting separately rather than fitting an output.
        raw = np.zeros((2, 20), np.float32)
        raw[0, 0] = 2.
        raw[1, 0] = -1.
        reference = np.zeros((3, 20), np.float32)
        reference[1:, 0] = .1
        bias = np.zeros(20, np.float32)
        bias[0] = .05
        lower = np.full(20, -1., np.float32)
        upper = np.full(20, 1., np.float32)
        upper[0] = .2
        # First unconstrained target .1+.25-.05=.3 -> .2.
        # Second EMA=-.25+.125=-.125;target=.1-.125-.05=-.075.
        ctrl = np.zeros((16, 20), np.float32)
        ctrl[:8, 0] = .2
        ctrl[8:, 0] = -.075
        state = np.zeros((16, 27), np.float32)
        velocity = np.zeros((16, 26), np.float32)
        trace = dict(sampled_mean_raw=raw, pre_action_phase=np.array([1, 2]),
                     pre_qpos=state.copy(), post_qpos=state.copy(), pre_qvel=velocity.copy(), post_qvel=velocity.copy(),
                     initial_ema=np.zeros(20, np.float32), encoder_bias=bias,
                     target_position_lower=lower, target_position_upper=upper,
                     reference_joint_pos=reference, post_ctrl=ctrl,
                     post_actuator_force=np.zeros((16, 20)), post_qfrc_actuator=np.zeros((16, 26)))
        case = dict(recorded_executed_control_schema='pen_PD_and_native_drives:1', action_contract=ACTION_CONTRACT,
                    position_ctrl_ids=list(range(20)), joint_order=[str(i) for i in range(20)])
        return trace, case

    def test_raw_clip_and_PD_limit_have_separate_counts(self):
        trace, case = self.fixture()
        audit = audit_executed_control(trace, case)
        self.assertEqual((audit['passed'], audit['total']), (6, 6))
        self.assertEqual(audit['raw_clipped_components'], 1)
        self.assertEqual(audit['absolute_PD_target_limited_components'], 1)

    def test_changed_actual_ctrl_and_state_gap_fail(self):
        trace, case = self.fixture()
        trace['post_ctrl'][4, 0] += .01
        trace['pre_qpos'][8, 0] = .1
        result = audit_executed_control(trace, case)
        for key in ['EMA_reference_bias_and_limits_match_GPU_ctrl', 'ctrl_held_for_eight_native_substeps', 'qpos_native_continuity']:
            self.assertFalse(result['checks'][key])

    def test_wrong_bias_or_reference_is_detected(self):
        original, case = self.fixture()
        for field in ['encoder_bias', 'reference_joint_pos']:
            trace = copy.deepcopy(original)
            trace[field] += .01
            self.assertFalse(audit_executed_control(trace, case)['checks']['EMA_reference_bias_and_limits_match_GPU_ctrl'])

    def test_bad_binding_phase_or_drive_rejected(self):
        original, case = self.fixture()
        for mutation in ['binding', 'phase', 'drive']:
            trace, changed = copy.deepcopy(original), copy.deepcopy(case)
            if mutation == 'binding':
                changed['position_ctrl_ids'][1] = 0
            elif mutation == 'phase':
                trace['pre_action_phase'][1] = 1
            else:
                trace['post_actuator_force'][0, 0] = np.nan
            with self.assertRaises(ValueError):
                audit_executed_control(trace, changed)


if __name__ == '__main__':
    unittest.main()
