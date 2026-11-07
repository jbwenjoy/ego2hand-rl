"""Bounded CUDA engineering checks of the frozen geometric A2 prior.

This is not a contact/dynamics qualification. Actual recurrent policy/WM
training is checked by the separate, unchanged ten-iteration training entry.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEX = Path("D:/workspace/dex-supermimic-research")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(DEX / ".cache/torchinductor/a2_geometry_probe")
    os.environ["TRITON_CACHE_DIR"] = str(DEX / ".cache/triton/controlled_pen")
    import numpy as np
    import torch
    from dex_supermimic.rl.dex_supermimic.networks import WorldModel
    from dex_supermimic.rl.dex_supermimic.pen import unflatten_pen_memory

    spec = importlib.util.spec_from_file_location("controlled_pen_train", ROOT / "scripts/train_controlled_pen.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    ready = ROOT / "outputs/a2_hook_final_cpu_20261102"
    frozen = json.loads((ready / "source/contact_objective/frozen_inputs.json").read_text())
    cfg = json.loads((ROOT / "outputs/training_configs/a2_bounded_10iter_smoke_seed0.json").read_text())
    assert frozen["expected_hashes"] == cfg["contact_objective"]["expected_hashes"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(41)
    result = {"status": "running", "scope": "CUDA geometry/prior engineering only; failed physical proxy qualification remains",
              "script_sha256": train.digest(Path(__file__)), "trainer_sha256": train.digest(ROOT / "scripts/train_controlled_pen.py"),
              "expected_hashes": frozen["expected_hashes"], "cache_paths": {k: os.environ[k] for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
              "cache_before": train.cache_inventory(os.environ["TORCHINDUCTOR_CACHE_DIR"]), "phases": []}
    train.write_json(output / "runtime.json", {"pid": os.getpid(), "status": "running"})

    def measure(name, function):
        gc.collect(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        tick = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        record = {"name": name, "seconds": time.perf_counter()-tick, "baseline_allocated_bytes": baseline,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "after_allocated_bytes": torch.cuda.memory_allocated(), "free_total_bytes": list(torch.cuda.mem_get_info())}
        result["phases"].append(record)
        train.write_json(output / "progress.json", record)
        print(json.dumps(record), flush=True)
        return value

    try:
        assert torch.cuda.is_available()
        result["device"] = torch.cuda.get_device_name()
        module = train.build_contact_objective(frozen, "cuda:0")
        metadata = module.metadata
        assert metadata["sampled_hand_surface_points"] == 1344 and metadata["basis"]["training_directions"] == 64
        assert not list(module.parameters())
        assert all(not tensor.requires_grad for tensor in module.buffers())
        assert module.geometry_proxy.sample_points_body.dtype == torch.float32
        train.write_json(output / "metadata.json", metadata)
        a1 = ROOT / "outputs/a1l_paired_v2_clip007_2000iter_seed0_restart1/compiled"
        data = np.load(a1 / "replay_control_trace.npz")

        def batch(key, t=0):
            value = torch.tensor(data[key][t], device="cuda:0")
            return value.repeat((32,)+(1,)*(value.ndim-1))

        x = batch("x1")
        memory = unflatten_pen_memory(batch("m"))
        assert tuple(x.shape) == (128, 58)

        def evaluate(forward, outer_amp=False):
            state = x.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=outer_amp):
                bonus, terms = forward(state, memory)
            assert bonus.dtype == torch.float32
            assert torch.isfinite(bonus).all() and all(torch.isfinite(v).all() for v in terms.values())
            gradient = torch.autograd.grad(bonus.sum(), state)[0]
            assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
            return bonus.detach().cpu(), gradient.detach().cpu()

        fp = measure("eager_B128_forward_backward_FP32", lambda: evaluate(module))
        amp = measure("eager_B128_forward_backward_outer_BF16", lambda: evaluate(module, True))
        assert torch.equal(fp[0], amp[0]) and torch.equal(fp[1], amp[1])
        result["same_FP32_input_outer_AMP_exact"] = True
        compiled = torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)
        actual = measure("inductor_fullgraph_B128_compile_forward_backward", lambda: evaluate(compiled, True))
        gradient_relative = float(torch.linalg.vector_norm(actual[1]-fp[1])/torch.linalg.vector_norm(fp[1]).clamp_min(1e-12))
        result["compiled_eager_parity"] = {"bonus_max_abs": float((actual[0]-fp[0]).abs().max()),
                                          "gradient_max_abs": float((actual[1]-fp[1]).abs().max()), "gradient_relative_l2": gradient_relative}
        assert result["compiled_eager_parity"]["bonus_max_abs"] < 1e-3 and gradient_relative < .01

        def retained_horizon(forward):
            states = [batch("x1", t).detach().clone().requires_grad_() for t in range(8)]
            total = states[0].new_zeros(())
            for t, state in enumerate(states):
                mem = unflatten_pen_memory(batch("m", t))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    bonus, terms = forward(state, mem)
                    total = total + .02*bonus.mean()
                assert torch.isfinite(bonus).all() and all(torch.isfinite(v).all() for v in terms.values())
            gradients = torch.autograd.grad(total, states)
            assert all(torch.isfinite(g).all() for g in gradients)
            return float(total), max(float(g.abs().max()) for g in gradients)

        result["geometry_only_B128_H8_eager"] = measure("eager_B128_H8_retained_forward_backward", lambda: retained_horizon(module))
        result["geometry_only_B128_H8_compiled"] = measure("compiled_B128_H8_retained_forward_backward", lambda: retained_horizon(compiled))
        measure("compiled_B128_H8_repeat", lambda: retained_horizon(compiled))
        result["horizon_scope"] = "Eight retained prior evaluations; actual recurrent actor/WM path must pass the separate real ten-iteration smoke."

        points = x.new_tensor([[-.089,0,0],[-.088,0,0],[-.090,0,0],[-.0888,.0082,.001],[-.107,.012,.013]])[None].requires_grad_()
        sdf, normal = module.geometry_proxy.pen_sdf_normal(points)
        gradient = torch.autograd.grad(sdf.sum(), points)[0]
        assert torch.isfinite(gradient).all()
        # The seam-axis points intentionally have tied nearest patches; compare
        # derivatives only at the two stable exposed-face/corner points.
        assert torch.allclose(gradient[:,3:], normal[:,3:], atol=2e-5, rtol=2e-5)
        assert abs(float(sdf[0,0])+.00745) < 1e-6
        result["seam_and_exposed_sdf_m"] = sdf.detach().cpu().tolist()
        result["seam_penetration_prior"] = float(((-sdf[0,0]-.003).clamp_min(0)/.001).square())

        checkpoint = torch.load(a1 / "train/model_final.pt", map_location="cpu", weights_only=False)
        wm_state = checkpoint["modules"]["wm"]
        dynamics_dim = wm_state["norm.mean"].numel()-20
        wm = WorldModel(dynamics_dim, 15, hidden=(512,512,512), amp=True).to("cuda:0")
        wm.load_state_dict(wm_state); wm.eval().requires_grad_(False)
        state, control = batch("x"), batch("u")
        context = torch.tensor(data["reader_context"], device="cuda:0").repeat(32,1)
        dynamics = torch.cat((state,context,batch("aux")[:,21:]),-1)
        assert dynamics.shape[-1] == dynamics_dim
        def wm_geometry(amp_enabled):
            wm.amp = amp_enabled
            u = control.detach().clone().requires_grad_()
            predicted, _ = wm(state,u,dynamics)
            bonus,_ = module(predicted,memory)
            g = torch.autograd.grad(bonus.sum(),u)[0]
            assert predicted.dtype == torch.float32 and torch.isfinite(predicted).all() and torch.isfinite(g).all()
            return predicted.detach().cpu(),bonus.detach().cpu(),g.detach().cpu()
        low = measure("actual_frozen_WM_AMP_to_geometry", lambda: wm_geometry(True))
        high = measure("actual_frozen_WM_FP32_to_geometry", lambda: wm_geometry(False))
        result["WM_to_prior_precision_diagnostic"] = {"WM_x1_dtype": "torch.float32 after BF16 network output cast/integration",
            "x1_max_abs_AMP_FP32":float((low[0]-high[0]).abs().max()),"bonus_max_abs_AMP_FP32":float((low[1]-high[1]).abs().max()),
            "control_gradient_max_abs_AMP_FP32":float((low[2]-high[2]).abs().max()),
            "scope":"Reported numerical sensitivity, not exact end-to-end AMP parity or validated physical gradient."}
        result["compiler_counters"] = {group:{key:int(value) for key,value in values.items()} for group,values in torch._dynamo.utils.counters.items()}
        assert not result["compiler_counters"].get("graph_break")
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        result["total_wall_seconds"] = time.perf_counter()-started
        train.write_json(output / "result.json", result)
        train.write_json(output / "runtime.json", {"pid":os.getpid(),"status":result["status"]})
        print(json.dumps({"status":result["status"],"output":str(output),"total_wall_seconds":result["total_wall_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
