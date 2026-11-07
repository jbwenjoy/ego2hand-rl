"""Bounded CUDA bring-up, parameter-update and checkpoint-resume validation.

This is an installation/pipeline check, not a learning-quality experiment.
The controlled environment uses playback configuration (no training DR/noise).
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["hand2-reorient", "hand2-pen-spin"], default="hand2-reorient")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--simulation-steps", type=int, default=100)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--clips-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--sim-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA unavailable"
    from mjlab.envs import ManagerBasedRlEnv
    import warp as wp
    from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    wp.set_device("cuda:0")
    cfg = DexSuperMimicCfg(
        task=args.task, num_envs=args.num_envs, logger="none", seed=42,
        recipe_path=str(args.recipe.resolve()) if args.recipe else None,
        clips_root=str(args.clips_root.resolve()) if args.clips_root else None,
        max_iterations=args.iterations, buffer_steps=64, warmup_steps=16,
        wm_only_iters=2, wm_window=4, wm_batches=1, wm_batch=32,
        wm_hidden=(128, 128), pi_batch=16, horizon=4, max_horizon=4,
        critic_epochs=1, critic_minibatches=1, save_interval=4, log_interval=2,
        compile=args.compiled, compile_policy=args.compiled, amp=False,
        contact_weight=0.0, failure_head=False, holdout_fraction=0.125,
        checkpoint=str(args.resume.resolve()) if args.resume else None,
    )
    env_cfg = make_env_cfg(args.num_envs, 50, play=True, task=args.task,
                           recipe_path=cfg.recipe_path, clips_root=cfg.clips_root)
    env_cfg.seed = 42
    env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
    runner = None
    mj_model = env.sim.mj_model
    palm = mj_model.body("robot/right_palm_link")
    assert int(palm.jntnum.item()) == 0, "Expected a fixed wrist"
    assert mj_model.nq == 27 and mj_model.nv == 26 and mj_model.nu == 20
    result = {
        "purpose": "pipeline smoke only; no task-success, speedup, or video-to-robot claim",
        "platform": platform.platform(), "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
        "versions": {p: metadata.version(p) for p in ["torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "wuji-description", "triton-windows"]},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_has_local_changes": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "task": args.task, "compiled": args.compiled, "num_envs": args.num_envs,
        "controlled_play_configuration": True, "contact_loss_weight": 0.0, "failure_head_enabled": False,
        "model_dimensions": {k: int(getattr(env.sim.mj_model, k)) for k in ["nq", "nv", "nu", "nbody", "ngeom"]},
        "joint_names": list(env.scene["robot"].joint_names),
        "fixed_wrist": {"palm_joint_count": int(palm.jntnum.item()), "palm_pos_w": palm.pos.tolist(), "palm_quat_w_wxyz": palm.quat.tolist()},
        "model_joint_contract": [{"name": mj_model.joint(i).name, "qpos_address": int(mj_model.jnt_qposadr[i]),
                                  "dof_address": int(mj_model.jnt_dofadr[i]), "type": int(mj_model.jnt_type[i]),
                                  "range_radians": mj_model.jnt_range[i].tolist()} for i in range(mj_model.njnt)],
        "model_actuator_contract": [{"name": mj_model.actuator(i).name,
                                     "joint_id": int(mj_model.actuator_trnid[i, 0]),
                                     "control_range": mj_model.actuator_ctrlrange[i].tolist(),
                                     "force_range": mj_model.actuator_forcerange[i].tolist()} for i in range(mj_model.nu)],
        "events": list(env_cfg.events),
    }
    try:
        obs, _ = env.reset()
        finite_steps = 0
        reward_sum = 0.0
        terminations = 0
        for _ in range(args.simulation_steps):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(args.num_envs, 20, device="cuda:0"))
            assert bool(torch.isfinite(obs["policy"]).all())
            assert bool(torch.isfinite(reward).all())
            assert bool(torch.isfinite(env.sim.data.qpos).all())
            reward_sum += float(reward.mean())
            finite_steps += 1
            terminations += int(terminated.sum())
            done = terminated | truncated
            if bool(done.any()):
                env.reset(env_ids=done.nonzero(as_tuple=False).flatten())
        result.update(simulation_finite_steps=finite_steps, mean_reward_sum=reward_sum,
                      simulation_terminated_transitions=terminations)
        if not args.sim_only:
            runner = DexSuperMimicRunner(env, cfg, args.output / "train")
            start_iteration = runner.iteration
            modules = {name: mod for name, mod in runner.modules().items() if name in ["actor", "wm", "critic"]}
            if args.resume:
                saved = torch.load(args.resume, weights_only=False, map_location="cuda:0")
                assert saved["iteration"] == start_iteration
                for name, module in modules.items():
                    assert all(torch.equal(value, saved["modules"][name][key]) for key, value in module.state_dict().items())
                result["resume_exact_parameter_load"] = True
            before = {name: [p.detach().cpu().clone() for p in module.parameters()] for name, module in modules.items()}
            torch.cuda.reset_peak_memory_stats()
            runner.learn()
            changes = {}
            for name, module in modules.items():
                params = list(module.parameters())
                assert all(bool(torch.isfinite(p).all()) for p in params), name
                delta = sum(float((p.detach().cpu() - old).double().square().sum()) for p, old in zip(params, before[name])) ** 0.5
                assert delta > 0, f"{name} unchanged"
                changes[name] = delta
            checkpoint = args.output / "train/model_final.pt"
            restored = torch.load(checkpoint, weights_only=False, map_location="cpu")
            assert restored["iteration"] == args.iterations
            result.update(start_iteration=start_iteration, final_iteration=restored["iteration"],
                          checkpoint=str(checkpoint.resolve()), checkpoint_readable=True,
                          parameter_delta_l2=changes, collected_sim_steps=runner.sim_steps,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["status"] = "passed"
        (args.output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), elapsed_seconds=time.perf_counter() - started)
        (args.output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        if runner is not None:
            runner.close()
        env.close()


if __name__ == "__main__":
    main()
