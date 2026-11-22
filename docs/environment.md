# 运行环境

这项研究用到了两套物理环境。导出的纸盒控制包用 MuJoCo 3.11.0 在 CPU 上重放。策略训练则在 WSL 的 FlashCHORD 环境里进行，使用 Newton 和 MuJoCo-Warp 在 GPU 上步进。两边的模型格式和依赖版本各有用途。

## 从新检出的仓库开始

下面的 PowerShell 命令使用 Python 3.12，为共享代码建立一个本地环境。示例分别检查数据文件的定位与校验，以及手掌坐标在旋转和平移后是否一致。两项都可以在没有数据集的机器上运行。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/research.py --help
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_artifacts.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_hand_frames.py -k palm_convention
```

如果 3.12 是独立安装的，把第一行的 `py -3.12` 换成该解释器的路径即可。重放开发纸盒时，再运行 `.\.venv\Scripts\python.exe -m pip install "mujoco==3.11.0"`，按[回放示例](../tools/README.md)读取解压后的控制包。根项目的可选依赖见 [pyproject.toml](../pyproject.toml)。

## 训练用的 WSL 环境

留出实验使用 FlashCHORD v0.4.0、Python 3.11.15、Newton 1.2.0、Warp 1.14.0、JAX 0.10.2，以及 MuJoCo 和 MuJoCo-Warp 3.8.1。完整版本记录在[环境配置](../environments/profiles.toml)。从 [NVIDIA 对应版本](https://github.com/nvidia-isaac/video_to_data/tree/v0.4.0/robotic_grounding/flash_chord) 建立训练环境，把本仓库的 `src/` 加入 `PYTHONPATH`，将 `V2D_ASSETS_DIR` 指向准备好的资产根目录。Wuji 手的导出与单世界回放命令见[移植记录](flashchord-wuji-port.md)。

## 已有工作区的数据路径

早期转笔、手机与持杯实验通过 [configs/artifacts.toml](../configs/artifacts.toml) 查找外部数据。换机器时可以复制 [paths.example.toml](../paths.example.toml) 为根目录的 `paths.local.toml`，填写本机数据目录。运行 `python scripts/research.py doctor` 可查看解释器、包来源和缺少的文件，`python scripts/research.py artifacts --verify` 会进一步核对内容摘要。缺少必需数据时这两个检查会返回非零退出码。HO-Cap 的训练输出和重放包由各实验脚本单独读写，路径见相应实验记录。

已有 Windows Hand2 环境也可以直接用它的解释器运行 `scripts/research.py`。这个入口在当前进程中加载本仓库的 `src/`，无需更改旧工程的 editable 安装。历史项目自己的依赖与锁文件在 [archive/studies](../archive/studies)。
