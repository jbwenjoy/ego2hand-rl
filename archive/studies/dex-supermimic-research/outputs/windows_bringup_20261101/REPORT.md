# Windows Hand2 bring-up — 2026-11-01

已在独立环境 `.venv` 完成 Hand2 固定腕方块与转笔的 CUDA 链路验证。以下是安装、采样、更新和恢复检查；不是任务成功率、收敛结果、CHORD增强收益或真实视频重定向结论。

| 检查 | 实测结果 | 证据 |
|---|---|---|
| 方块 CUDA 仿真与短训练 | 16环境，100有限控制步；12训练迭代，WM/actor/critic参数均改变；可读检查点 | `cube_eager/validation.json` |
| 独立进程恢复 | 从12迭代精确加载参数，继续到16；三个模块再次更新 | `cube_resume/validation.json` |
| 转笔 CUDA 仿真与短训练 | 单轴分组，16环境，100有限控制步；12迭代，WM/actor/critic参数均改变；可读检查点 | `pen_eager/validation.json` |
| 转笔 headless 播放 | 加载上述检查点，20控制步，退出0 | `pen_play.log` |
| 包一致性 | 112个已安装包兼容 | `uv_pip_check.log` |
| 官方单轴数据 | 10条完整原始片段；仅传输1,784,632字节；Zip CRC和catalog SHA256通过 | `pen_minimal_download.json` |
| 手部资源 | 原始包68个资源文件与上游Git blob一致（XML按Git换行规则规范化） | `environment_and_assets.json` |

环境：Python 3.12.15；PyTorch 2.14.1+cu130；mjlab 1.6.0 (`0fb8a681`)，MuJoCo/MuJoCo-Warp 3.11.0，Warp 1.14.0，triton-windows 3.8.0.post29。手部资源回到原始打包revision `829c9aa27cb12c3a49d5b8d63c9869dd8f656702`。本机RTX5090，无驱动或系统CUDA修改。Python解释器也位于本项目 `.uv-python`，没有使用G1虚拟环境。

基线源码：`2becb7380fed86171056f9bf4338b2a6a0c14b01`，当前含本地Windows补丁。锁文件、安装清单、diff、源码哈希和资源哈希均已记录。

控制检查确认：固定掌根无关节，20维手指动作；整体 `nq=27,nv=26,nu=20`，唯一额外自由状态为物体。恢复和转笔验证JSON含关节顺序、关节限位、执行器控制/输出范围；这些是仿真契约，尚未与硬件SDK校准。

配置：play=True，50Hz，warmup=16，wm_only_iters=2，wm_window=4，WM hidden=(128,128)，每迭代一次WM batch32，policy batch16/horizon4，amp=False，compile=False。用于imagination的failure weighting关闭，contact loss=0。训练仍会记录辅助头损失；关闭其策略权重不等于删除辅助头。pen play配置还按硬件笔尺寸执行确定性的pen_dimensions启动项。训练使用的WM/actor/critic均通过有限性和参数变化断言。

重要口径：smoke仿真在终止后重置；转笔100步期间3次终止。`play=True` 也放宽终止条件，不能作为片段验收。contact loss=0仅为链路检查，正式上游A1需恢复自碰撞监督0.1或显式修改对应目标，因为奖励仍读取该头。CLI中的pen rot_error也不能当作参考跟踪误差。`peak_allocated_gib`只统计PyTorch allocator，排除了Warp和驱动，不能作为完整显存占用。

源码适配：加入win32/AMD64、Python3.12和官方CUDA torch wheel来源；独立uv配置/锁；Windows Viser的EGL默认改为WGL；play/export从Actor首层权重推断输入维度，修复当前EmpiricalNorm没有 `norm.mean` 的读取错误。

资产排查纠正：最初目录缺少packages是本地 sparse checkout 仅包含src，不是上游漏包。已扩展checkout并恢复原始资源包与MIT许可证；packages目录无源码修改。排查中曾下载新commit的右手资源，其33个共同物理文件与原打包资源Git blob全部相同；临时资源保存在本outputs下，不参与最终新进程加载。

安装网络排查：单连接wheel下载缓慢；用锁文件指定官方URL的并行range下载并核验每个SHA256，再用`uv sync --locked`完成安装，未改变已锁定版本。对应脚本为 `scripts/prefetch_locked_windows.py`；torch wheel单独验证官方SHA256。

未完成：编译训练、完整参考片段物理质量、视频产物、ONNX数值一致性、接触增强消融、多轨迹摊销、真实ego输入和真机闭环。这些应由独立物理评估与后续实验推进。
