"""Cache lifetime and input-gradient invariants; CPU-only, no simulation."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supermimic_research.runtime.geometry_context import CurrentGeometry
from supermimic_research.learning import update_step as U


class Owner:
    def __init__(self):
        self.provider = self
        self.counts = {"pure_Torch_FK_calls": 0}
        self.pose_calls = 0
        self.directional_calls = 0

    def poses(self, x):
        self.pose_calls += 1
        return {"p": x * 2}

    def _features_from_poses(self, x, poses, return_all_directions=False):
        self.directional_calls += 1
        return {"gap": poses["p"] * 3}


class GeometryContextTests(unittest.TestCase):
    def test_shared_graph_keeps_all_consumer_gradients(self):
        x = torch.tensor([[1., 2., 3.]], dtype=torch.float64, requires_grad=True)
        owner = Owner(); ctx = CurrentGeometry(owner, x)
        p = ctx.poses(); self.assertIs(p, ctx.poses())
        g = ctx.directional(); self.assertIs(g, ctx.directional())
        scalar = p["p"].sum() + 2*p["p"].sum() + g["gap"].sum()
        grad, = torch.autograd.grad(scalar, x)
        self.assertTrue(torch.equal(grad, torch.full_like(x, 12.)))
        self.assertEqual((owner.pose_calls, owner.directional_calls), (1, 1))

    def test_equal_valued_other_state_and_other_owner_are_rejected(self):
        owner = Owner(); x = torch.ones(1, 3, dtype=torch.float64)
        ctx = CurrentGeometry(owner, x)
        with self.assertRaises(ValueError): ctx.require(owner, x.clone())
        with self.assertRaises(ValueError): ctx.require(Owner(), x)

    def test_inplace_and_requires_grad_changes_are_rejected(self):
        owner = Owner(); x = torch.ones(1, 3, dtype=torch.float64)
        ctx = CurrentGeometry(owner, x); x.add_(1)
        with self.assertRaises(ValueError): ctx.poses()
        x = torch.ones(1, 3, dtype=torch.float64); ctx = CurrentGeometry(owner, x)
        x.requires_grad_(True)
        with self.assertRaises(ValueError): ctx.poses()

    def test_grad_lifecycle_is_not_crossed(self):
        owner = Owner(); x = torch.ones(1, 3, dtype=torch.float64)
        ctx = CurrentGeometry(owner, x)
        with torch.no_grad():
            with self.assertRaises(ValueError): ctx.poses()

    def test_nonfinite_gradient_fails_before_private_Adam_mutation(self):
        layer = torch.nn.Linear(2, 1)
        before = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        opt = U.make_optimizer(layer, learning_rate=.001)
        for parameter in layer.parameters(): parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaises(ValueError): U.finish_optimizer_step(layer, opt, phase="invalid-test")
        self.assertFalse(opt.state)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in layer.state_dict().items()))

    def test_deferred_logs_preserve_sequential_F64_mean(self):
        values = tuple(torch.tensor(v, dtype=torch.float32) for v in [1e10, 1., -1e10, 1.])
        expected = 0.
        for value in values: expected += float(value)/4
        result = U.materialize_metrics_tree({"actor": U._SequentialMeanMetric(values, 4)})
        self.assertEqual(result["actor"], expected)


if __name__ == "__main__": unittest.main()
