"""Small real-data CUDA training run; writes auditable validation results."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--run-name", default="windows_eager_smoke")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    cfg = TrainConfig(run_name=args.run_name, seed=1, resume=args.resume)
    cfg.data.clips = ["walk1_subject1", "dance1_subject1", "jumps1_subject1"]
    cfg.env.num_envs = args.num_envs
    cfg.env.replay_capacity = 128
    cfg.wm_trainer.batch_size = args.batch_size
    cfg.wm_trainer.batches_per_update = 2
    cfg.policy_trainer.batch_size = args.batch_size
    cfg.policy_trainer.critic_epochs = 2
    cfg.policy_trainer.critic_minibatches = 2
    cfg.wm_trainer.compile = "default" if args.compiled else None
    cfg.policy_trainer.compile = "default" if args.compiled else None
    cfg.schedule.total_iterations = args.steps
    cfg.schedule.log_interval = 20
    cfg.schedule.checkpoint_interval = 80
    cfg.schedule.warmup_iterations = 40
    started = time.perf_counter()
    trainer = Trainer(cfg)
    modules = {"world_model": trainer.wm, "policy": trainer.policy, "critic": trainer.critic}
    before = {
        name: [p.detach().cpu().clone() for p in mod.parameters()] for name, mod in modules.items()
    }
    initial_updates = trainer.num_updates
    initial_iteration = trainer.iteration
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.run()
        changes = {}
        for name, mod in modules.items():
            parameters = list(mod.parameters())
            assert all(bool(torch.isfinite(p).all()) for p in parameters), name
            delta = sum(
                float((p.detach().cpu() - old).double().square().sum())
                for p, old in zip(parameters, before[name])
            )
            assert delta > 0, f"{name} did not update"
            changes[name] = delta**0.5
        checkpoint = trainer.log_dir / "ckpt_latest.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["iteration"] == args.steps
        assert trainer.num_updates > initial_updates
        result = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compiled": args.compiled,
            "start_iteration": initial_iteration,
            "iteration": trainer.iteration,
            "new_updates": trainer.num_updates - initial_updates,
            "parameter_delta_l2": changes,
            "checkpoint_reload": True,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "elapsed_seconds": time.perf_counter() - started,
            "note": "Pipeline validation on three clips; not a convergence or paper-metric reproduction.",
        }
        (trainer.log_dir / "validation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()


if __name__ == "__main__":
    main()
