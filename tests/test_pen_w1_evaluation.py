"""Negative admission cases: these must fail before any native-state import."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from supermimic_research.learning.pen_w1_evaluation import validate_actor_contract, validate_W1_model
from supermimic_research.learning.pen_support_ppo import CENTERED_SCHEMA
from supermimic_research.runtime.pen_pose_observation import OBSERVATION_SCHEMA, ACTION_CONTRACT
from evaluate_pen_long_control import initialize_final_actor, validate_final_training


class W1AdmissionTests(unittest.TestCase):
    def fixture(self):
        metadata = dict(checkpoint_schema=CENTERED_SCHEMA, observation_schema=OBSERVATION_SCHEMA,
                        action_contract=ACTION_CONTRACT, policy_input_dim=161, action_dim=20,
                        joint_order=['j' + str(i) for i in range(20)], objective_config={'sha256': 'objective'},
                        arm='RESIDUAL', support_weight=0.)
        state = dict(schema=CENTERED_SCHEMA, metadata=metadata, normalizers_ready=True,
                     native=dict(worlds=32, config={'sha256': 'adapter'}),
                     modules=dict(actor={'norm.count': torch.tensor(288), 'weights': torch.ones(2)}))
        return state, copy.deepcopy(metadata)

    def validate(self, state, metadata):
        validate_actor_contract(state, metadata, {'sha256': 'objective'}, {'sha256': 'adapter'},
                                ['j' + str(i) for i in range(20)], {'RESIDUAL': 0., 'SUPPORT': .2})

    def test_incompatible_observation_action_joints_objective_rejected(self):
        original, metadata = self.fixture()
        self.validate(original, metadata)
        mutations = [lambda s: s['metadata'].update(policy_input_dim=155),
                     lambda s: s['metadata'].update(action_contract='different EMA'),
                     lambda s: s['metadata'].update(joint_order=list(reversed(s['metadata']['joint_order']))),
                     lambda s: s['metadata'].update(objective_config={'sha256': 'other'}),
                     lambda s: s['metadata'].update(support_weight=.2),
                     lambda s: s.update(normalizers_ready=False),
                     lambda s: s['native'].update(worlds=1)]
        for mutation in mutations:
            state = copy.deepcopy(original)
            mutation(state)
            with self.subTest(state=state['metadata']), self.assertRaises(ValueError):
                self.validate(state, metadata)

    def test_nonfinite_or_empty_normalizer_rejected(self):
        for name, value in [('weights', torch.tensor([float('nan')])), ('norm.count', torch.tensor(0))]:
            state, metadata = self.fixture()
            state['modules']['actor'][name] = value
            with self.assertRaises(ValueError):
                self.validate(state, metadata)

    def test_W1_full_array_bytes_and_scalars_not_broadcast(self):
        current = {'plant': torch.tensor([0., 1.], dtype=torch.float32)}
        saved = dict(arrays={'plant': np.array([0., 1.], dtype=np.float32)}, scalars={'worlds': 1})
        validate_W1_model(current, {'worlds': 1}, saved)
        for old in [np.array([-0., 1.], dtype=np.float32), np.array([[0., 1.]], dtype=np.float32),
                    np.array([0., 1.], dtype=np.float64), np.array([0., 2.], dtype=np.float32)]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                validate_W1_model(current, {'worlds': 1}, dict(saved, arrays={'plant': old}))
        with self.assertRaises(ValueError):
            validate_W1_model(current, {'worlds': 32}, saved)

    def test_wrong_checkpoint_cannot_be_substituted_for_final(self):
        state = dict(iteration=100, optimizer_steps=3200, training_transitions=204800,
                     metadata={'arm': 'RESIDUAL'})
        with self.assertRaisesRegex(ValueError, 'counters'):
            initialize_final_actor(None, state, {}, plant_context_pin={}, label='RESIDUAL')
        state.update(iteration=2000, optimizer_steps=64000, training_transitions=4096000)
        with self.assertRaisesRegex(ValueError, 'arm'):
            initialize_final_actor(None, state, {}, plant_context_pin={}, label='SUPPORT')

    def test_incomplete_training_not_admitted(self):
        arm = dict(iterations=2000, actual_native_integrations=32768000,
                   optimizer_steps=64000, training_transitions=4096000)
        report = dict(schema='pen_geometry_support_long_control_training:2', config={}, passed=5,
                      total=5, checks={'budget': True}, actual_native_integrations=65536000,
                      optimizer_steps=128000, arms=[dict(arm, arm='RESIDUAL'), dict(arm, arm='SUPPORT')])
        validate_final_training(report, {})
        report['arms'][1]['iterations'] = 64
        with self.assertRaises(ValueError):
            validate_final_training(report, {})


if __name__ == '__main__':
    unittest.main()
