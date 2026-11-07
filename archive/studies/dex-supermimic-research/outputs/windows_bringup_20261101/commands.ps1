# Run from D:\workspace\dex-supermimic-research.
# Setup: independent interpreter and environment; no system driver/CUDA changes.
./uv-local.ps1 python install 3.12
.uv-python/cpython-3.12-windows-x86_64-none/python.exe scripts/windows_download_torch.py
git sparse-checkout add packages/wuji-description
./uv-local.ps1 sync --locked
.venv/Scripts/python.exe scripts/download_pen_minimal.py

# Bounded eager train + independent-process resume.
.venv/Scripts/python.exe scripts/windows_smoke.py --output outputs/windows_bringup_20261101/cube_eager
.venv/Scripts/python.exe scripts/windows_smoke.py --output outputs/windows_bringup_20261101/cube_resume --iterations 16 --resume outputs/windows_bringup_20261101/cube_eager/train/model_final.pt
.venv/Scripts/python.exe scripts/windows_smoke.py --task hand2-pen-spin --recipe src/dex_supermimic/tasks/pen_spin/data/recipe/01_single_axis.json --output outputs/windows_bringup_20261101/pen_eager
.venv/Scripts/dex-supermimic-play.exe --checkpoint outputs/windows_bringup_20261101/pen_eager/train/model_final.pt --steps 20 --num-envs 16 --no-viser

# Environment/source checks.
uv pip check --python .venv/Scripts/python.exe
.venv/Scripts/python.exe -m compileall -q scripts src/dex_supermimic/rl/dex_supermimic/cli.py
git diff --check

# Optional network acceleration, only if wheel single-stream downloads stall.
# The script verifies hashes against uv.lock; install by named requirement with
# find-links, then rerun locked sync. All versions remain those in uv.lock.
# .uv-python/cpython-3.12-windows-x86_64-none/python.exe scripts/prefetch_locked_windows.py
# uv pip install --python .venv/Scripts/python.exe --no-deps --no-index --find-links .cache/downloads scipy==1.17.1 imageio-ffmpeg==0.6.0 mujoco==3.11.0 tensorboard==2.20.0 matplotlib==3.10.8 onnx==1.20.1 wandb==0.25.0 triton-windows==3.8.0.post29 sympy==1.14.0 numpy==2.4.2 pillow==12.2.0
# ./uv-local.ps1 sync --locked
