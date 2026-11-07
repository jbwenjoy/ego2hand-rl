import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from chord_study.contact_objective import GeometricContactObjective, official_tolerance_support_match
from chord_study.contacts import support_reward
from chord_study.geometry import quat_matrix

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs" / "reference_contact_targets_v1"
REFERENCE_HASH = "bb61dbbfe97cdca5ed7453eeea372b6d11007999f463476eaab5cfcd07040f53"


class ControlledGeometry(nn.Module):
    """Analytic test fixture for masking/algebra, never used by the experiment."""
    def forward(self, state, bandwidth, return_geometry=False):
        support = (1+.2*state[..., :1]).expand(state.shape[:-1]+(64,))
        sdf = state[..., 1:2].expand(state.shape[:-1]+(2,))
        return {"support": support, "sampled_signed_distance_m": sdf,
                "minimum_signed_distance_m": sdf.amin(-1)}


def controlled_objective():
    objective = GeometricContactObjective.__new__(GeometricContactObjective)
    nn.Module.__init__(objective)
    objective.geometry_proxy = ControlledGeometry()
    objective.register_buffer("reference_support64", torch.ones(5, 1, 1, 64))
    objective.register_buffer("phase_weight", torch.tensor([0., 0., 1., 1., .5]))
    objective.register_buffer("phase", torch.arange(5))
    objective.bandwidth_m, objective.tolerance, objective.var = .001, .1, .1
    return objective


def memory(frames, *, start=None, end=5):
    time = torch.as_tensor(frames, dtype=torch.float32).reshape(-1, 1)
    return {"time": time, "start": torch.zeros_like(time) if start is None else torch.as_tensor(start, dtype=torch.float32).reshape(-1, 1),
            "end": torch.full_like(time, end)}


class GeometryIntentAlgebra(unittest.TestCase):
    def test_phase_masks_only_attraction_and_penetration_stays_active(self):
        objective = controlled_objective()
        state = torch.zeros(5, 58)
        state[:, 1] = -.005
        bonus, terms = objective(state, memory(range(5)))
        torch.testing.assert_close(terms["geometry_intent_support_match"], torch.ones(5))
        torch.testing.assert_close(terms["geometry_intent_penetration_prior"], torch.full((5,), 4.), atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(bonus, torch.tensor([-.4, -.4, .6, .6, .1]), atol=2e-6, rtol=1e-6)

    def test_no_contact_or_unknown_does_not_attract_to_contacts(self):
        state = torch.zeros(5, 58)
        state[:, 1] = -.002
        bonus, _ = controlled_objective()(state, memory(range(5)))
        torch.testing.assert_close(bonus, torch.tensor([0., 0., 1., 1., .5]))

    def test_penmdp_start_negative_time_and_end_indexing(self):
        _, terms = controlled_objective()(torch.zeros(3, 58), memory([-8, 100, 0], start=[1, 1, 2], end=4))
        torch.testing.assert_close(terms["geometry_intent_reference_frame"], torch.tensor([1., 3., 2.]))

    def test_input_gradient_flows_but_target_cannot_be_optimized(self):
        objective = controlled_objective()
        state = torch.zeros(1, 58)
        state[:, 0], state[:, 1] = -2., -.005
        state.requires_grad_()
        bonus, _ = objective(state, memory([2]))
        bonus.sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(state.grad[0, 0]), 0)
        self.assertGreater(float(state.grad[0, 1]), 0)
        self.assertFalse(objective.reference_support64.requires_grad)
        self.assertIsNone(objective.reference_support64.grad)
        self.assertEqual(list(objective.parameters()), [])

    def test_official_union_support_score_without_extra_contact_penalty(self):
        torch.manual_seed(81)
        target = torch.rand(3, 2, 3, 64)
        current = torch.rand_like(target)
        target[0], target[1, 0], current[1, 1, 0] = 0, 0, 0
        torch.testing.assert_close(official_tolerance_support_match(current, target), support_reward(current, target))
        self.assertEqual(float(official_tolerance_support_match(torch.ones(1, 1, 1, 64), torch.zeros(1, 1, 1, 64))), 0)


@unittest.skipUnless(BUNDLE.is_dir(), "requires the independently frozen local target bundle")
class FrozenBundleContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objective = GeometricContactObjective.from_bundle(BUNDLE, expected_reference_sha256=REFERENCE_HASH)

    def test_stride_subset_assets_metadata_and_no_mount(self):
        objective = self.objective
        torch.testing.assert_close(objective.reference_support64, objective.reference_support512[..., ::8])
        torch.testing.assert_close(objective.geometry_proxy.basis, objective.basis512[::8])
        self.assertEqual(objective.metadata["points_per_geom"], 64)
        self.assertEqual(objective.metadata["sampled_hand_surface_points"], 1344)
        self.assertFalse(objective.metadata["qualified_contact_predictor"])
        self.assertFalse(objective.metadata["dynamics_gradient_qualified"])
        self.assertFalse(objective.metadata["dt_applied_inside_module"])
        self.assertFalse(any("mount" in geom["name"] for geom in objective.metadata["geometry"]["hand_collision_geoms"]))
        self.assertTrue(objective.metadata["scoring"]["no_finger_identity_or_extra_contact_penalty"])

    def test_wrong_reference_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference hash"):
            GeometricContactObjective.from_bundle(BUNDLE, expected_reference_sha256="0"*64)

    def test_target_file_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copy2(BUNDLE / "manifest.json", target / "manifest.json")
            (target / "contact_targets.npz").write_bytes((BUNDLE / "contact_targets.npz").read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "target.*SHA256"):
                GeometricContactObjective.from_bundle(target)

    def test_explicit_unknown_and_known_absence_override_contact_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for filename in ("manifest.json", "rules.json", "raw_geometry.json", "time_windows.json"):
                shutil.copy2(BUNDLE / filename, target / filename)
            with np.load(BUNDLE / "contact_targets.npz", allow_pickle=False) as raw:
                arrays = {key: raw[key].copy() for key in raw.files}
            arrays["phase"][:2] = 3
            arrays["contact_known"][0] = False
            arrays["known_no_contact"][1] = True
            np.savez_compressed(target / "contact_targets.npz", **arrays)
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            manifest["target_npz_sha256"] = hashlib.sha256((target / "contact_targets.npz").read_bytes()).hexdigest()
            (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            objective = GeometricContactObjective.from_bundle(target)
            torch.testing.assert_close(objective.phase_weight[:2], torch.zeros(2))

    def test_real_geometry_cpu_fullgraph_compile_and_full512_diagnostic(self):
        objective = copy.deepcopy(self.objective)
        path = Path(objective.metadata["reference_file"])
        with np.load(path, allow_pickle=False) as raw:
            q = raw["joint_pos_rad"][7].copy()
            object_pose, wrist_pose = raw["object_pose_w"][7].copy(), raw["wrist_pose_w"][7].copy()
        inv = quat_matrix(wrist_pose[3:]).T
        state = np.r_[q, np.zeros(20), inv@(object_pose[:3]-wrist_pose[:3]),
                       (inv@quat_matrix(object_pose[3:])).reshape(-1), np.zeros(6)]
        x = torch.tensor(state[None], dtype=torch.float32, requires_grad=True)
        frame_memory = memory([7], end=objective.frame_count)
        plain, _ = objective(x, frame_memory)
        compiled = torch.compile(objective, backend="eager", fullgraph=True)
        result, terms = compiled(x, frame_memory)
        torch.testing.assert_close(result, plain)
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(value.shape == (1,) for value in terms.values()))
        full = objective.full_basis_support_for_evaluation(x.detach())
        subset = objective.geometry_proxy(x.detach(), objective.bandwidth_m)["support"]
        self.assertEqual(full.shape, (1, 512))
        torch.testing.assert_close(full[..., ::8], subset)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp_full = objective.full_basis_support_for_evaluation(x.detach())
        torch.testing.assert_close(amp_full, full, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
