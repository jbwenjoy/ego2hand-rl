"""CPU-only isolation, gradient and frozen-input checks for the A2 reward hook."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("controlled_pen_train", ROOT / "scripts/train_controlled_pen.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def objective_config():
    bundle = ROOT / "outputs/reference_contact_targets_v1"
    manifest = json.loads((bundle / "manifest.json").read_text())
    files = {
        "manifest_sha256": bundle / "manifest.json", "target_npz_sha256": bundle / "contact_targets.npz",
        "reference_npz_sha256": Path(manifest["reference_npz_file"]),
        "reference_json_sha256": Path(manifest["reference_npz_file"]).with_suffix(".json"),
        "model_sha256": Path(manifest["model_file"]), "rules_sha256": bundle / "rules.json",
        "raw_geometry_sha256": bundle / "raw_geometry.json", "time_windows_sha256": bundle / "time_windows.json",
        "objective_module_sha256": ROOT / "src/chord_study/contact_objective.py",
        "geometry_module_sha256": ROOT / "src/chord_study/contact_geometry.py",
    }
    return {"kind": "CHORD-inspired geometric intent shaping", "bundle_dir": str(bundle),
            "reference_file": manifest["reference_npz_file"], "model_file": manifest["model_file"],
            "expected_hashes": {key: train.digest(path) for key, path in files.items()},
            "hyperparameters": {"bandwidth_m": .001, "tolerance": .1, "var": .1}, "per_control_step_scale": .02,
            "qualification_status": "local_proxy_qualification_failed_exploratory_ablation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = objective_config()
    train.write_json(args.output / "contact_objective_config.json", config)
    (args.output / "source").mkdir(exist_ok=True)
    frozen = train.freeze_contact_inputs(config, args.output)
    torch.set_num_threads(2)

    class Checks(unittest.TestCase):
        def test_a1_none_branch_identity_bits_and_rng_are_unchanged(self):
            random.seed(71); np.random.seed(71); torch.manual_seed(71)
            def native(x, *unused, **kwargs):
                return x.square().sum(-1), x, {"native": x}
            runner = SimpleNamespace(reward_step=native)
            x = torch.linspace(-1, 1, 12).reshape(3,4)
            expected = native(x)[0].clone()
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            with train.imagination_contact_reward(runner, None) as terms:
                self.assertIsNone(terms)
                self.assertIs(runner.reward_step, native)
                self.assertTrue(torch.equal(runner.reward_step(x)[0], expected))
            self.assertIs(runner.reward_step, native)
            self.assertEqual(random.getstate(), before[0])
            self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

        def test_a2_exact_once_scale_native_cage_terms_and_gradient(self):
            cage = torch.zeros(3,1); native_terms = {"native": torch.ones(3)}
            def native(x, memory, action, residual, contact, torque, soft=False):
                return x.square().sum(-1), cage, native_terms
            def objective(x, memory):
                bonus = x[:,0].sin() + 2*x[:,1]
                return bonus, {"test_bonus": bonus}
            runner = SimpleNamespace(reward_step=native)
            x = torch.linspace(-.3,.8,12).reshape(3,4).requires_grad_()
            expected = native(x,None,None,None,None,None)[0] + .02*objective(x,None)[0]
            with train.imagination_contact_reward(runner, objective) as terms:
                actual, actual_cage, actual_terms = runner.reward_step(x,None,None,None,None,None,soft=True)
                self.assertTrue(torch.equal(actual, expected))
                self.assertIs(actual_cage,cage); self.assertIs(actual_terms,native_terms)
                self.assertTrue(terms["test_bonus"])
                actual_grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected.sum(), x)[0]
                self.assertTrue(torch.equal(actual_grad,expected_grad))
            self.assertIs(runner.reward_step,native)
            self.assertTrue(torch.equal(runner.reward_step(x,None,None,None,None,None)[0],x.square().sum(-1)))

        def test_exception_restores_native_before_subsequent_fit_or_collect(self):
            def native(x,*args,**kwargs): return x.sum(-1),x,{}
            runner=SimpleNamespace(reward_step=native)
            with self.assertRaisesRegex(RuntimeError,"deliberate"):
                with train.imagination_contact_reward(runner,lambda x,m:(x.sum(-1),{})):
                    raise RuntimeError("deliberate policy failure")
            self.assertIs(runner.reward_step,native)

        def test_fixed_factory_preserves_rng_and_gradients_remain_on_inputs(self):
            random.seed(81); np.random.seed(81); torch.manual_seed(81)
            before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
            module=train.build_contact_objective(frozen,"cpu")
            self.assertEqual(random.getstate(),before[0])
            self.assertTrue(np.array_equal(np.random.get_state()[1],before[1][1]))
            self.assertTrue(torch.equal(torch.get_rng_state(),before[2]))
            self.assertFalse(module.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))
            trace=np.load(ROOT/'outputs/a1_pilot_clip007_f0200_2000iter_seed0/compiled/replay_control_trace.npz')
            from dex_supermimic.rl.dex_supermimic.pen import unflatten_pen_memory
            state=torch.tensor(trace['x1'][0,:2],requires_grad=True)
            memory=unflatten_pen_memory(torch.tensor(trace['m'][0,:2]))
            bonus,terms=module(state,memory)
            self.assertEqual(tuple(bonus.shape),(2,))
            self.assertTrue(torch.isfinite(bonus).all())
            gradient=torch.autograd.grad(bonus.sum(),state)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertFalse(module.metadata['qualified_contact_predictor'])
            self.assertFalse(module.metadata['dynamics_gradient_qualified'])
            self.assertFalse(module.metadata['target_defined_from_policy_contacts'])
            train.write_json(args.output/'constructed_metadata.json',module.metadata)

        def test_tampered_expected_hash_rejected_before_constructor(self):
            altered=json.loads(json.dumps(config))
            altered['expected_hashes']['manifest_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'manifest_sha256'):
                train.freeze_contact_inputs(altered,args.output)

    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    record={'scope':'CPU hook/factory isolation and differentiability only; no policy training or physical contact qualification',
            'tests_run':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),
            'log':stream.getvalue(),'training_script_sha256':train.digest(ROOT/'scripts/train_controlled_pen.py')}
    train.write_json(args.output/'checks.json',record)
    print(json.dumps(record,indent=2))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=='__main__': main()
