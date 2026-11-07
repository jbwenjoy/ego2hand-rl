"""CPU exact-value/gradient and isolation checks for the P_ONLY adapter.

Real frozen A2 geometry is checked separately from an explicitly controlled SDF
fixture that exercises A2.forward's boundary/phase algebra. Neither is a physical
performance result. No source, target, optimizer or simulator state is changed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / "supermimic-chord-study"
sys.path.insert(0, str(ROOT / "src"))
from penalty_study.penalty_only_objective import PenaltyOnlyPrior


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def rng_equal(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2]))


class ControlledGeometry(nn.Module):
    """Test-only SDF injection; not collision geometry or new training data."""
    def __init__(self, directions):
        super().__init__()
        self.directions = directions

    def forward(self, state, bandwidth, *, return_geometry=False):
        sdf = state[:, :4]
        support = (1 + state[:, 4:5]).expand(-1, self.directions)
        return {"sampled_signed_distance_m": sdf, "minimum_signed_distance_m": sdf.amin(-1),
                "support": support}


def memory(frames, end=5):
    frames = torch.as_tensor(frames, dtype=torch.float32).reshape(-1, 1)
    return {"start": torch.zeros_like(frames), "time": frames, "end": torch.full_like(frames, end)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-spec", type=Path, default=ROOT / "outputs/penalty_cpu_factory_v1/source/contact_objective/frozen_inputs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/penalty_cpu_checks_v1")
    parser.add_argument("--factory-only", action="store_true", help="Small recheck after a wrapper metadata-only hash update")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace CPU evidence: {args.output}")
    args.output.mkdir(parents=True)
    spec = json.loads(args.frozen_spec.read_text(encoding="utf-8-sig"))
    trainer_path = ROOT / "scripts/train_controlled_pen.py"
    module_spec = importlib.util.spec_from_file_location("penalty_train_cpu_checks", trainer_path)
    train = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(train)
    torch.set_num_threads(2)
    cuda_before = {"visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                   "is_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
                   "is_initialized": torch.cuda.is_initialized()}
    if cuda_before["is_available"] or cuda_before["device_count"] or cuda_before["is_initialized"]:
        raise RuntimeError("CPU checker requires an unavailable and uninitialized CUDA runtime")
    base = train.build_contact_objective(spec, "cpu")
    real_wrapper = PenaltyOnlyPrior(base).eval()
    controlled = copy.deepcopy(base)
    controlled.geometry_proxy = ControlledGeometry(base.reference_support64.shape[-1])
    # These five test labels exercise the exact unchanged A2.forward algebra;
    # the real target bundle and real base module remain untouched.
    controlled.phase = torch.arange(5)
    controlled.phase_weight = torch.tensor([0., 0., 1., 1., .5])
    controlled.frame_count = 5
    controlled.reference_support64 = controlled.reference_support64[:5].clone()
    adapted = PenaltyOnlyPrior(controlled).eval()
    trace_path = MAIN / "outputs/a1l_paired_v2_clip007_2000iter_seed0_restart1/compiled/replay_control_trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        real_state = torch.tensor(trace["x1"][0].copy(), dtype=torch.float32)
        packed = torch.tensor(trace["m"][0].copy(), dtype=torch.float32)
    from dex_supermimic.rl.dex_supermimic.pen import unflatten_pen_memory
    real_memory = unflatten_pen_memory(packed)
    deep = real_state[:1].clone()
    hand_point = base.geometry_proxy.hand_points_palm(deep[:, :20])[:, 0]
    deep[:, 40:43] = hand_point - torch.tensor([[0., .001, 0.]])
    deep[:, 43:52] = torch.eye(3).reshape(1, 9)
    deep_memory = {k:v[:1].clone() for k,v in real_memory.items()}

    def state(sdfs):
        sdfs = torch.tensor(sdfs, dtype=torch.float32)
        x = torch.zeros(len(sdfs), 58)
        x[:, :4] = sdfs
        return x

    class Checks(unittest.TestCase):
        def exact(self, original, wrapper, x, mem, *, amp=False):
            a, b = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
                _, original_terms = original(a, mem)
                expected = -.1 * original_terms["geometry_intent_penetration_prior"]
                actual, terms = wrapper(b, mem)
            eg = torch.autograd.grad(expected.sum(), a)[0]
            ag = torch.autograd.grad(actual.sum(), b)[0]
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(ag, eg))
            self.assertTrue(torch.isfinite(actual).all() and torch.isfinite(ag).all())
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(terms["geometry_intent_bonus_unscaled"], actual))
            self.assertTrue(torch.equal(terms["geometry_intent_support_coefficient"], torch.zeros_like(actual)))
            return actual.detach(), ag.detach(), terms

        def test_real_logged_states_exact_value_and_state_gradient(self):
            self.exact(base, real_wrapper, real_state, real_memory)

        def test_real_geometry_deep_penetration_exact_and_nonzero_gradient(self):
            value, gradient, terms = self.exact(base, real_wrapper, deep, deep_memory)
            self.assertGreater(float(terms["geometry_intent_penetration_prior"].detach().max()), 1.)
            self.assertLess(float(value.min()), -.1)
            self.assertGreater(float(gradient.abs().max()), 0.)

        def test_controlled_threshold_before_at_and_after_three_mm(self):
            x = state([[-.0029999,.01,.02,.03],[-.003,.01,.02,.03],[-.0030001,.01,.02,.03],[-.01,.01,.02,.03]])
            value, gradient, _ = self.exact(controlled, adapted, x, memory([2]*4))
            self.assertTrue(torch.equal(value[:2], torch.zeros(2)))
            self.assertTrue(torch.equal(gradient[:2], torch.zeros_like(gradient[:2])))
            self.assertLess(float(value[2]), 0.)
            self.assertGreater(float(gradient[2,0]), 0.)

        def test_controlled_tied_maximum_keeps_exact_A2_subgradient(self):
            x = state([[-.006,-.006,.01,.02]])
            value, gradient, _ = self.exact(controlled, adapted, x, memory([2]))
            self.assertEqual(float(gradient[0,0]), float(gradient[0,1]))
            self.assertGreater(float(gradient[0,0]), 0.)

        def test_all_five_phases_penalty_independent_of_attraction_masks(self):
            x = state([[-.006,-.001,.01,.02]]*5)
            value, gradient, terms = self.exact(controlled, adapted, x, memory(range(5)))
            self.assertTrue(torch.equal(value, value[:1].expand_as(value)))
            self.assertTrue(torch.equal(gradient, gradient[:1].expand_as(gradient)))
            self.assertTrue(torch.equal(terms["geometry_intent_phase"], torch.arange(5,dtype=torch.float32)))
            self.assertTrue(torch.equal(terms["geometry_intent_phase_weight"], torch.tensor([0.,0.,1.,1.,.5])))

        def test_support_input_is_diagnostic_only_no_reward_gradient(self):
            x = state([[-.006,-.001,.01,.02]]*3)
            x[:,4] = torch.tensor([-.9,0.,8.])
            value, gradient, _ = self.exact(controlled, adapted, x, memory([2]*3))
            self.assertTrue(torch.equal(value, value[:1].expand_as(value)))
            self.assertTrue(torch.equal(gradient[:,4], torch.zeros(3)))

        def test_controlled_outer_AMP_same_FP32_value_and_gradient(self):
            x = state([[-.006,-.006,.01,.02],[-.0030001,.01,.02,.03]])
            a = self.exact(controlled, adapted, x, memory([0,4]))
            b = self.exact(controlled, adapted, x, memory([0,4]), amp=True)
            self.assertTrue(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]))

        def test_real_geometry_outer_AMP_same_FP32_value_and_gradient(self):
            a = self.exact(base, real_wrapper, deep, deep_memory)
            b = self.exact(base, real_wrapper, deep, deep_memory, amp=True)
            self.assertTrue(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]))

        def test_frozen_targets_no_grad_and_base_metadata_not_mutated(self):
            self.assertFalse(list(real_wrapper.parameters()))
            self.assertTrue(all(not v.requires_grad for v in real_wrapper.buffers()))
            self.assertEqual(base.metadata["scoring"]["support_coefficient"],1.)
            self.assertEqual(real_wrapper.metadata["scoring"]["support_coefficient"],0.)
            self.assertEqual(real_wrapper.metadata["penetration_coefficient"],.1)
            self.assertIs(real_wrapper.metadata["dt_applied_inside_module"],False)
            self.assertEqual(real_wrapper.metadata["target_npz_sha256"],spec["expected_hashes"]["target_npz_sha256"])
            self.exact(base,real_wrapper,deep,deep_memory)
            self.assertTrue(all(v.grad is None for v in real_wrapper.buffers()))

        def test_constructor_preserves_python_numpy_torch_RNG(self):
            random.seed(901); np.random.seed(901); torch.manual_seed(901)
            before = rng(); PenaltyOnlyPrior(base)
            self.assertTrue(rng_equal(before,rng()))

        def test_factory_preserves_RNG_and_frozen_bindings(self):
            random.seed(902); np.random.seed(902); torch.manual_seed(902)
            before = rng(); module = train.build_penalty_only_objective(spec,"cpu")
            self.assertTrue(rng_equal(before,rng()))
            self.assertFalse(module.training)
            self.assertEqual(module.metadata["support_coefficient"],0.)
            self.assertEqual(module.metadata["formula"],"-.1*penetration_prior")
            self.assertEqual(module.metadata["target_npz_sha256"],spec["expected_hashes"]["target_npz_sha256"])
            self.exact(module.base,module,deep,deep_memory)
            (args.output/"constructed_metadata.json").write_text(json.dumps(module.metadata,indent=2)+"\n")

        def test_factory_rejects_tampered_wrapper_before_consuming_RNG(self):
            altered = copy.deepcopy(spec); altered["penalty_wrapper_sha256"]="0"*64
            before = rng()
            with self.assertRaisesRegex(ValueError,"wrapper"):
                train.build_penalty_only_objective(altered,"cpu")
            self.assertTrue(rng_equal(before,rng()))

        def test_native_reward_hook_adds_dt_exactly_once_and_restores(self):
            cage=torch.zeros(2,1); native_terms={"native":torch.ones(2)}
            def native(x,*unused,**kwargs):return x.square().sum(-1),cage,native_terms
            runner=SimpleNamespace(reward_step=native)
            x=state([[-.006,.01,.02,.03],[-.008,.01,.02,.03]]).requires_grad_();mem=memory([0,4])
            base_reward=native(x)[0];penalty=adapted(x,mem)[0];expected=base_reward+.02*penalty
            with train.imagination_contact_reward(runner,adapted,scale=.02):
                actual,actual_cage,actual_terms=runner.reward_step(x,mem,None,None,None,None)
                self.assertTrue(torch.equal(actual,expected));self.assertIs(actual_cage,cage);self.assertIs(actual_terms,native_terms)
                self.assertTrue(torch.equal(torch.autograd.grad(actual.sum(),x,retain_graph=True)[0],torch.autograd.grad(expected.sum(),x)[0]))
            self.assertIs(runner.reward_step,native)
            self.assertTrue(torch.equal(runner.reward_step(x)[0],base_reward))

        def test_native_reward_hook_exception_restores_future_collect_and_fit(self):
            def native(x,*unused,**kwargs):return x.sum(-1),x,{}
            runner=SimpleNamespace(reward_step=native)
            with self.assertRaisesRegex(RuntimeError,"intentional"):
                with train.imagination_contact_reward(runner,adapted,scale=.02):raise RuntimeError("intentional")
            self.assertIs(runner.reward_step,native)
            x=state([[-.006,.01,.02,.03]])
            self.assertTrue(torch.equal(runner.reward_step(x)[0],x.sum(-1)))

    names=unittest.defaultTestLoader.getTestCaseNames(Checks)
    if args.factory_only:names=[n for n in names if n.startswith("test_factory_")]
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(Checks(n) for n in names))
    record={"scope":"CPU exact shared-penalty adapter/factory/hook validation only; no simulator, training, CUDA or physical outcome",
            "gpu_executed":False,"tests_run":result.testsRun,"passed":result.wasSuccessful(),"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
            "source_sha256":{"checker":sha(__file__),"trainer":sha(trainer_path),"wrapper":sha(ROOT/'src/penalty_study/penalty_only_objective.py'),"frozen_spec":sha(args.frozen_spec),"real_trace":sha(trace_path)},
            "frozen_spec":str(args.frozen_spec.resolve()),"factory_only":args.factory_only,
            "fixture_boundary":"Real original A2 geometry checked on logged and deliberately penetrating state58; separate controlled-SDF/phase fixture runs unchanged A2.forward algebra, not real collision or dynamics data",
            "AMP_boundary":"CPU outer bfloat16 on identical FP32 state; no CUDA or end-to-end WM AMP claim", "log":stream.getvalue()}
    record["cuda_environment_before"] = cuda_before
    record["cuda_initialized_after"] = torch.cuda.is_initialized()
    record["environment_note"] = "Windows harness uses CUDA_VISIBLE_DEVICES=-1; earlier empty-string mask was not a reliable is_available guard. Earlier mathematical evidence is retained, without retroactive CUDA initialization claims."
    (args.output/'checks.json').write_text(json.dumps(record,indent=2)+"\n");(args.output/'tests.log').write_text(stream.getvalue());shutil.copy2(__file__,args.output/'check_penalty_only_cpu_source.py')
    print(json.dumps(record,indent=2));raise SystemExit(0 if result.wasSuccessful() and not result.skipped and not record["cuda_initialized_after"] else 1)


if __name__=="__main__":main()
