"""Export a frozen Hand2 Actor and compare ONNX reference execution on CPU.

Only the network is exported. Observation construction, reference lookup,
EMA residual memory, target limits/insets and physical actuation remain outside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from dex_supermimic.rl.dex_supermimic.networks import Actor


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(checkpoint: Path, output: Path) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["format_version"] != 3 or state["config"]["task"] != "hand2-pen-spin":
        raise ValueError("Expected frozen format3 Hand2 pen Actor")
    weights = state["modules"]["actor"]
    dimension = weights["net.0.weight"].shape[-1]
    actor = Actor(dimension).eval()
    actor.load_state_dict(weights, strict=True)
    saved = {key: value.clone() for key, value in actor.state_dict().items()}
    model_path = output / "actor.onnx"
    # Match the pinned upstream CLI exporter: legacy export, opset17 and
    # dynamic batch, while running the entire verification on masked CPU.
    torch.onnx.export(actor, torch.zeros((1, dimension)), str(model_path),
                      input_names=["policy"], output_names=["action"],
                      dynamic_axes={"policy": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model, full_check=True)
    evaluator = ReferenceEvaluator(model)
    mean = weights["norm._mean"].numpy()
    std = weights["norm._std"].numpy()
    generator = np.random.default_rng(4701)
    cases = {"zero_batch1": np.zeros((1, dimension), dtype=np.float32),
             "training_normalizer_mean_batch1": mean.copy()}
    for batch, scale in ((1, 1.), (7, 1.), (128, 3.)):
        cases[f"normalizer_domain_batch{batch}_scale{scale:g}"] = (
            mean + std * (generator.standard_normal((batch, dimension)).astype(np.float32) * np.float32(scale)))
    arrays, checks = {}, []
    with torch.inference_mode():
        for name, observations in cases.items():
            observations = np.asarray(observations, dtype=np.float32)
            expected = actor(torch.from_numpy(observations)).numpy()
            actual = evaluator.run(["action"], {"policy": observations})[0]
            error = actual.astype(np.float64) - expected
            absolute = float(np.abs(error).max())
            relative = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))
            finite = bool(np.isfinite(actual).all())
            passed = finite and actual.shape == (len(observations), 20) and absolute <= 1e-5 and relative <= 1e-5
            checks.append({"case": name, "batch": len(observations), "finite": finite,
                           "shape": list(actual.shape), "max_abs_error": absolute,
                           "relative_l2_error": relative, "passed": passed})
            arrays[name + "_input"] = observations
            arrays[name + "_torch"] = expected
            arrays[name + "_onnx_reference"] = actual
    unchanged = all(torch.equal(saved[key], value) for key, value in actor.state_dict().items())
    operators = sorted({node.op_type for node in model.graph.node})
    no_random = not any(name.startswith("Random") or name == "Dropout" for name in operators)
    np.savez_compressed(output / "equivalence_cases.npz", **arrays)
    report = {"status": "passed" if all(c["passed"] for c in checks) and unchanged and no_random else "failed",
              "schema": "hand2_actor_onnx_cpu_equivalence:1", "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha(checkpoint), "checkpoint_training_seed": state["config"]["seed"],
              "iteration": state["iteration"], "onnx_sha256": sha(model_path), "script_sha256": sha(Path(__file__)),
              "input_dimension": dimension, "output_dimension": 20, "control_hz": state["config"]["control_hz"],
              "joint_names": state["joint_names"], "opset": 17, "dynamic_batch_checked": [1, 7, 128],
              "runtime": "onnx.reference.ReferenceEvaluator CPU; not ONNX Runtime provider benchmarking",
              "torch_version": torch.__version__, "onnx_version": onnx.__version__,
              "thresholds_frozen_before_execution": {"max_abs": 1e-5, "relative_l2": 1e-5},
              "sample_scope": "Synthetic zeros/normalizer mean and seeded normalizer-domain samples; not replayed physical or ego observations",
              "cases": checks, "actor_state_unchanged": unchanged, "operators": operators,
              "no_random_or_dropout_nodes": no_random, "cuda_initialized": torch.cuda.is_initialized(),
              "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "wall_seconds": time.perf_counter() - started,
              "action_semantics": "tanh normalized20D residual action, not absolute joint targets",
              "outside_graph": ["observation construction and history", "reference phase", "EMA residual state",
                                "absolute target computation/limits/inset", "actuator control and state feedback"],
              "qualification": "Specific frozen actor mathematical export equivalence only; not closed-loop or hardware deployment qualification"}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "input_dimension", "cases", "actor_state_unchanged", "cuda_initialized", "wall_seconds")}))
    if report["status"] != "passed":
        raise RuntimeError("ONNX equivalence gate failed; retain artifacts")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint.resolve(strict=True), args.output.resolve())
