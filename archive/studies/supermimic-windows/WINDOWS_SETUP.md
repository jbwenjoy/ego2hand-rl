# Native Windows setup

This checkout is at `D:\workspace\supermimic-windows`. It uses a project-local
Python 3.12 installation, `.venv`, and uv cache. NVIDIA drivers and the system
CUDA installation are not modified.

## Commands (PowerShell)

```powershell
Set-Location D:\workspace\supermimic-windows
.\uv-local.ps1 sync --locked --group dev
.\train-windows.ps1 -Eager -Steps 160 -RunName my_eager_check
.\train-windows.ps1 -Steps 160 -RunName my_compiled_check
# Steps is the absolute target iteration, including when resuming:
.\train-windows.ps1 -Steps 240 -RunName my_resume_check -Resume logs/my_compiled_check/ckpt_latest.pt
.\uv-local.ps1 run --locked tensorboard --logdir logs
```

The short validation runner uses three real LAFAN1 G1 clips, 128 environments,
128-sample minibatches, the original model sizes and a 32-step imagined horizon.
It reduces world-model batches and critic epochs for a quick pipeline check.
It verifies updates to the world model, actor and critic, finite parameters,
checkpoint deserialization, and records `logs/<run>/validation.json`.
Short-run returns do not establish convergence or reproduction of paper metrics.

For the original trainer with the normal optimization schedule:

```powershell
.\uv-local.ps1 run --locked sm-train --run-name windows_walk --data.clips walk1_subject1 --env.num-envs 128 --env.replay-capacity 128 --wm-trainer.batch-size 128 --policy-trainer.batch-size 128 --schedule.total-iterations 1000
```

## Windows-specific dependency changes

- Added the Windows-only `triton-windows==3.8.0.post29` dependency for PyTorch 2.14.
- PyTorch is the original official `2.14.1+cu130` Windows wheel, verified against
  the upstream lockfile SHA-256. Its download-r2 endpoint stalled here, so the
  identical wheel was downloaded from download.pytorch.org using ranged requests.
- The Windows uv source points to this wheel under `.cache/downloads/` to avoid
  downloading it again. **Keep that wheel**: it is a dependency source, not a
  disposable compilation cache. To recreate it from the official server, run
  `python scripts/windows_download_torch.py` from this directory, then uv sync.
  The downloader uses only Python's standard library and validates SHA-256.
- `uv-local.ps1` selects the local Python/cache directories and configures local
  Hugging Face, Triton and TorchInductor caches. Warp uses its normal per-user
  cache under `%LOCALAPPDATA%\NVIDIA\warp\Cache`.
- No changes to reward functions, policy observations or model architecture.
- Single-process CUDA training only; native Windows multi-GPU NCCL is not tested.

## Data

Three G1 clips (walk1_subject1, dance1_subject1, jumps1_subject1) were downloaded
from https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1
and converted to 50 Hz npz files. The source dataset's license remains applicable.

```powershell
.\uv-local.ps1 run --locked python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30 --workers 3
```

Logs are in `.cache/*.log`, training results/checkpoints in `logs/`. Test logs
and per-run validation JSON files contain the measured results from this machine.

## Verified on 2026-11-01

- Windows 11 build 26200; RTX 5090 32 GB; unchanged NVIDIA driver 616.92.
- Python 3.12.15, PyTorch 2.14.1+cu130, Warp 1.18.0, MuJoCo/MJWarp 3.16.0,
  mjlab 1.6.0 at the upstream pinned commit, Triton Windows 3.8.0.post29.
- `pytest tests -q -k 'not training_smoke'`: **23 passed**, 1 deselected.
  The deselected short trainer test is covered by the larger runs below.
- Eager: 0 -> 160 iterations, 48 updates, 169.1 seconds, checkpoint verified.
- Compiled resume: 160 -> 320 iterations, another 48 updates, 502.3 seconds
  including first compilation. All three models updated with finite parameters.
- First compiled update interval spent about 38 s in the world model and 424 s
  in the policy (includes compilation). Later intervals reached 3.8-5.3 env
  iterations/s, versus about 0.8 eager, for this small configuration. These are
  indicative short-run measurements, not a controlled throughput benchmark.
- The original benchmark script loads the compiled-run checkpoint and evaluates
  all three clips. Mean normalized return 0.4272; **all three clips fail early**.
  This is a validated training pipeline, not a trained motion-tracking policy.
- `uv sync --locked --group dev` and `uv pip check` both pass after setup.
- A fresh process reused the compilation caches: 320 -> 400 iterations,
  16 additional updates in 40.1 seconds; finite parameter updates and checkpoint
  reload passed again (`logs/windows_warm_cache_resume/validation.json`).

To inspect the verified checkpoint:

```powershell
.\uv-local.ps1 run --locked python scripts/eval_benchmark.py --ckpt logs/windows_compiled_resume/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 32
```

Keep `.cache/torchinductor` and `.cache/triton` to reuse the expensive first
compilation. Changing model sizes, minibatch shapes or library versions can
trigger compilation again. PyTorch currently defaults to one compiler thread
on Windows; this setup preserves that default.
