"""CPU contracts for frozen WM inference and world-to-palm state construction."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from chord_study.world_model import FrozenPenWorldModel, rollout_palm_states


class PalmStateEvidence(unittest.TestCase):
    def test_rotating_wrist_frame_transforms_pose_and_both_velocities(self):
        qz = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        row = {"wrist_pose_w": np.array([[1., 2., 3., *qz]]),
               "object_pose_w": np.array([[1., 3., 3., *qz]]),
               "object_lin_vel_w_m_s": np.array([[0., 2., 0.]]),
               "object_ang_vel_w_rad_s": np.array([[0., 3., 0.]]),
               "joint_pos_rad": np.arange(20)[None] * .01,
               "joint_vel_rad_s": np.arange(20)[None] * .1}
        x = rollout_palm_states(row)
        self.assertEqual(x.shape, (1, 58))
        np.testing.assert_allclose(x[0, 40:43], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(x[0, 43:52].reshape(3, 3), np.eye(3), atol=1e-6)
        np.testing.assert_allclose(x[0, 52:55], [2, 0, 0], atol=1e-6)
        np.testing.assert_allclose(x[0, 55:58], [3, 0, 0], atol=1e-6)
        np.testing.assert_allclose(x[0, :20], row["joint_pos_rad"][0])

    def test_missing_velocity_is_not_silently_zero_filled(self):
        with self.assertRaisesRegex(ValueError, "missing state evidence"):
            rollout_palm_states({})


@unittest.skipUnless(importlib.util.find_spec("dex_supermimic"), "Dex environment required")
class FrozenWorldModelContract(unittest.TestCase):
    def setUp(self):
        from dex_supermimic.rl.dex_supermimic.networks import WorldModel
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        torch.manual_seed(37)
        model = WorldModel(278, 15, (8, 8), amp=False)
        with torch.no_grad():
            model.net[-1].weight.normal_(0, .02)
        self.checkpoint = self.root / "fixture.pt"
        torch.save({"format_version": 3, "config": {"task": "hand2-pen-spin", "wm_hidden": [8, 8]},
                    "modules": {"wm": model.state_dict()}, "contact_dim": 15, "iteration": 0,
                    "context_names": [["reference_summary", 6], ["actuator_kp_ratios", 20], ["actuator_kd_ratios", 20]],
                    "joint_names": [f"q{i}" for i in range(20)], "reference_signature": "synthetic-contract-test"}, self.checkpoint)
        self.context = self.root / "context.npz"
        np.savez(self.context, reader_context=np.ones((2, 46), dtype=np.float32))

    def tearDown(self):
        self.folder.cleanup()

    def test_target_gradients_flow_while_parameters_remain_frozen(self):
        wm = FrozenPenWorldModel(self.checkpoint, self.context, device="cpu")
        x = torch.zeros(2, 58)
        x[:, 43:52] = torch.eye(3).flatten()
        target = torch.zeros(2, 20, requires_grad=True)
        prediction, collision = wm(x, target)
        self.assertEqual(prediction.shape, (2, 58))
        self.assertEqual(collision.shape, (2, 15))
        loss = prediction[:, 40:43].square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(target.grad).all())
        self.assertGreater(float(target.grad.abs().sum()), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in wm.parameters()))
        self.assertEqual(wm.metadata["disturbance_dim"], 174)

    def test_wrong_input_layout_is_rejected(self):
        wm = FrozenPenWorldModel(self.checkpoint, self.context)
        with self.assertRaisesRegex(ValueError, "state58/target20"):
            wm(torch.zeros(2, 57), torch.zeros(2, 20))
        with self.assertRaisesRegex(ValueError, "state58/target20"):
            wm(torch.zeros(2, 58), torch.zeros(2, 21))

    def test_mixed_physics_context_is_rejected(self):
        c = np.ones((2, 46), dtype=np.float32)
        c[1, 0] = .4
        np.savez(self.context, reader_context=c)
        with self.assertRaisesRegex(ValueError, "fixed physical context"):
            FrozenPenWorldModel(self.checkpoint, self.context)

    def test_empty_physics_context_is_rejected(self):
        np.savez(self.context, reader_context=np.ones((0, 46), dtype=np.float32))
        with self.assertRaises(ValueError):
            FrozenPenWorldModel(self.checkpoint, self.context)

    def test_wrong_context_width_is_not_padded_as_disturbance(self):
        np.savez(self.context, reader_context=np.ones((2, 45), dtype=np.float32))
        with self.assertRaises(ValueError):
            FrozenPenWorldModel(self.checkpoint, self.context)


if __name__ == "__main__":
    unittest.main()
