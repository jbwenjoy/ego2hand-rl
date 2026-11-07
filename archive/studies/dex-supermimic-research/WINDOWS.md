# Windows research bring-up

This checkout targets independent Python 3.12 and the upstream mjlab 1.6.0 /
MuJoCo 3.11 task API. It does not reuse another project's virtual environment.
The RTX 5090 uses an official PyTorch 2.14.1+cu130 Windows wheel; its SHA256 is
verified by the bootstrap. System drivers and the system CUDA installation are
unchanged.

From PowerShell in this checkout:

```powershell
./uv-local.ps1 python install 3.12
.uv-python/cpython-3.12-windows-x86_64-none/python.exe scripts/windows_download_torch.py
git sparse-checkout add packages/wuji-description
./uv-local.ps1 sync --locked
./uv-local.ps1 run --locked python scripts/download_pen_minimal.py
```

The original local checkout was sparse and included only `src`. Expanding it
restores the upstream `packages/wuji-description` resource library, already
bundled at Wuji revision `829c9aa27cb12c3a49d5b8d63c9869dd8f656702` under MIT.
No additional model repository or Python wrapper is required. On a full clone,
the `sparse-checkout add` line is unnecessary. Physical models are unchanged.

`download_pen_minimal.py` extracts the entire original `01_single_axis` clip
group from the official v2026.10.22 release using byte ranges, preserving its
index and verifying both Zip CRCs and indexed SHA256 hashes.

Run bounded installation checks (not a task-success experiment):

```powershell
./uv-local.ps1 run --locked python scripts/windows_smoke.py --output outputs/windows_bringup_20261101/cube_eager
./uv-local.ps1 run --locked python scripts/windows_smoke.py --output outputs/windows_bringup_20261101/cube_resume --iterations 16 --resume outputs/windows_bringup_20261101/cube_eager/train/model_final.pt
./uv-local.ps1 run --locked python scripts/windows_smoke.py --task hand2-pen-spin --recipe src/dex_supermimic/tasks/pen_spin/data/recipe/01_single_axis.json --output outputs/windows_bringup_20261101/pen_eager
```

The check uses `play=True`, 16 environments, 100 finite simulation steps,
16 warmup steps, `wm_only_iters=2`, 12 training iterations, small model/batches,
no compilation by default, failure weighting disabled in imagination and zero contact-head loss. It asserts
actual world-model, actor and critic parameter updates and readable checkpoints.
Resume is tested in a separate invocation. Playback configuration has different
termination thresholds from training and is not an acceptance evaluator.

The pen contact head predicts finger self-collisions, not hand-object support.
Zero contact loss is used only for this pipeline check; policy rewards still
read that head. Restore its supervision (upstream `contact_weight=0.1`) when
running a faithful baseline, or explicitly remove the corresponding objective.
Do not interpret the upstream pen `task/rot_error` log as reference tracking
quality. A physics evaluator with object reference tracking, no reset and no
object assistance must determine experimental success.

Native Windows uses `MUJOCO_GL=wgl` for Viser playback instead of EGL.
Use `--compiled` only after the eager pipeline check passes.
