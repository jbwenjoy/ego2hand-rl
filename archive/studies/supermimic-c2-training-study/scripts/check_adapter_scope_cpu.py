"""Independent CPU stub checks of scoped adapter semantics, not physics parity."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from c2_training_adapter import training_controller


class Term:
    def __init__(self):
        self._mask_enabled = False
        self.target_position_limits = (torch.full((2, 20), -1.), torch.full((2, 20), 1.))
        self._processed_actions = torch.zeros(2, 20)
        self._targets = torch.zeros(2, 20)

    def process_actions(self, action):
        self._processed_actions.copy_(.5 * self._processed_actions + .5 * action)
        self._targets.copy_(self._processed_actions.clamp(-1, 1))


class Env:
    def __init__(self):
        self.term = Term()
        self.action_manager = SimpleNamespace(get_term=lambda _: self.term)
        self.term.process_actions(torch.zeros(2, 20))
        self.term._targets.fill_(1.)  # Mimic native initialization at boundary.

    def step(self, action):
        self.term.process_actions(action)
        self.ctrl = self.term._targets.clone()
        return self.ctrl


class MDP:
    def __init__(self, reader):
        self.target_lower = reader.env.term.target_position_limits[0][0].clone()
        self.target_upper = reader.env.term.target_position_limits[1][0].clone()
        self.mask_cfg = SimpleNamespace(mask_prob=0)

    def control_target(self, action, memory, bias):
        residual = .2 * action + .8 * memory["c1"]
        target = (residual + bias).clamp(self.target_lower, self.target_upper)
        memory["frozen_target"] = target
        return residual, target


class Runner:
    def __init__(self, mdp):
        self.mdp = mdp

    def save(self, path, next_iteration=None):
        torch.save({"format_version": 3, "modules": {"actor": torch.ones(1)}, "iteration": next_iteration}, path)

    def load(self, *args, **kwargs):
        return "old-load"

    def load_model_only(self, *args, **kwargs):
        return "old-model-load"


def main(output=None):
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        assert condition, name

    originals = (Env.__init__, Env.step, MDP.__init__, MDP.control_target, Runner.save, Runner.load, Runner.load_model_only)
    save_function = torch.save
    marker = {"base_method": "CPU_STUB_ONLY", "protocol_sha256": "synthetic", "role": "contract_check"}
    with training_controller(Env, MDP, Runner, marker):
        env = Env()
        check("native_initialization_unchanged", torch.equal(env.term._targets, torch.ones(2, 20)))
        ctrl = env.step(torch.full((2, 20), 10.))
        check("action_target_inset", torch.equal(ctrl, torch.full((2, 20), .98)))
        check("native_EMA_unchanged", torch.equal(env.term._processed_actions, torch.full((2, 20), 5.)))
        check("active_flag_restored", env._c2_action_active is False)
        env.term.process_actions(torch.full((2, 20), 10.))
        check("outside_step_native_process_preserved", torch.equal(env.term._targets, torch.ones(2, 20)))
        mdp = MDP(SimpleNamespace(env=env))
        check("controller_metadata_present", mdp.c2_controller_metadata["base_method"] == "CPU_STUB_ONLY")
        for action_value, target_value, gradient_value in ((.1, .42, .2), (10., .98, 0.)):
            action = torch.full((2, 20), action_value, requires_grad=True)
            memory = {"c1": torch.full((2, 20), .5), "c2": torch.full((2, 20), .3)}
            original_c1, original_c2 = memory["c1"].clone(), memory["c2"].clone()
            residual, target = mdp.control_target(action, memory, torch.zeros(20))
            target.sum().backward()
            check(f"imagined_target_{action_value}", torch.allclose(target, torch.full_like(target, target_value)))
            check(f"piecewise_gradient_{action_value}", torch.allclose(action.grad, torch.full_like(action, gradient_value)))
            check(f"residual_memory_{action_value}", torch.equal(residual, .2 * action + .8 * original_c1) and torch.equal(memory["c1"], original_c1) and torch.equal(memory["c2"], original_c2))
            check(f"frozen_target_{action_value}", memory["frozen_target"] is target)
        runner = Runner(mdp)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            runner.save(path, 10)
            saved = torch.load(path, weights_only=False)
            check("serialized_format103_and_identity", saved["format_version"] == 103 and saved["c2_controller"]["base_method"] == "CPU_STUB_ONLY" and saved["iteration"] == 10)
            try:
                runner.save(Path(tmp) / "missing" / "checkpoint.pt")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected serialization failure")
            check("torch_save_restored_after_exception", torch.save is save_function)
        for name in ("load", "load_model_only"):
            try:
                getattr(runner, name)("unused")
            except ValueError:
                check(name + "_fromscratch_reject", True)
            else:
                raise AssertionError("Resume should reject")
    check("classes_restored_after_scope", originals == (Env.__init__, Env.step, MDP.__init__, MDP.control_target, Runner.save, Runner.load, Runner.load_model_only))
    try:
        with training_controller(Env, MDP, Runner, marker):
            raise LookupError("synthetic context failure")
    except LookupError:
        pass
    check("classes_restored_after_context_exception", originals == (Env.__init__, Env.step, MDP.__init__, MDP.control_target, Runner.save, Runner.load, Runner.load_model_only))
    check("cuda_uninitialized", not torch.cuda.is_initialized())
    out = output or ROOT / "outputs/root_adapter_scope_cpu_v1"
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "passed", "scope": "Independent synthetic CPU stubs. Not native env/control/WM parity or physical qualification.", "checks": checks, "adapter_sha256": hashlib.sha256((ROOT / "src/c2_training_adapter.py").read_bytes()).hexdigest(), "pure_sha256": hashlib.sha256((ROOT / "src/target_inset.py").read_bytes()).hexdigest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "output": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args().output)
