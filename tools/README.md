# 实验脚本

## 重放已经求解的纸盒动作

手边有重放包时，这一步只需要 CPU、MuJoCo 3.11.0 和 NumPy。展开 `Wuji-HOCap-carton-replay.zip` 后，用仓库里的脚本读取解压目录。下面假设文件解压到了仓库根目录下的 `replay/`。

```sh
python tools/replay_exported_hand_controls.py replay
```

脚本读取包内的模型和腕、指控制，重新积分并逐帧核对关节状态和盒子位姿。当前开发纸盒版本的输出是 `141 frames, 5600 physics steps, position RMS 6.596 mm`。这台研究机器上的包位于 `runs/hocap_carton_table_margin_sensitivity_v1/export/`，同一实验的 `visual/` 目录里有原片与仿真的并排视频。

## 从一条新参考开始

| 环节 | 入口与用途 |
| --- | --- |
| 准备参考 | [prepare_hocap_flashchord.py](prepare_hocap_flashchord.py) 把源手物姿态、已求出的 Wuji 关节参考和物体资产整理成 FlashCHORD 输入。|
| 检查首帧 | [preflight_wuji_source_contacts.py](preflight_wuji_source_contacts.py) 在 CPU 上放好手、物体和桌面，读取初始碰撞距离。[输入和调用示例](../docs/hocap-heldout-initial-support.md) |
| 逐条求解 | [run_hocap_carton_baseline.py](run_hocap_carton_baseline.py) 从开发纸盒的运动学控制开始。[腕部间隙搜索](fit_hocap_wrist_table_clearance.py)和[余量比较](sweep_hocap_wrist_table_margin.py)展示了怎样修正接近和退出时的穿桌。|
| 训练策略 | [run_wuji_multiref_flashsac.py](run_wuji_multiref_flashsac.py) 运行单参考或多参考学习器。[开发集比较](../docs/hocap-flash-training-comparison.md)给出了数据、硬件和预算。 |
| 在新示范上评测 | [run_wuji_heldout.py](run_wuji_heldout.py) 负责冻结参考、训练和评分。[配对动力学迁移](run_wuji_recovered_paired_dynamics.py)复用同一个初始策略，比较从头拟合和迁移世界模型。|
| 核算结果 | [reconcile_wuji_heldout_cost.py](reconcile_wuji_heldout_cost.py) 汇总实际作业、重复使用的检查点和扰动记录。[plot_wuji_heldout_comparison.py](plot_wuji_heldout_comparison.py) 从评分记录生成[结果图](../docs/figures/heldout_cost_and_coverage.png)。 |

共享的手部控制在 [ego](../src/supermimic_research/ego)，物理步进与接触记录在 [runtime](../src/supermimic_research/runtime)，世界模型的想象更新在 [learning](../src/supermimic_research/learning/flash_imagination.py)。表中的拟合脚本沿用各自实验的输入目录。移植到新机器时，先准备对应的参考、模型和环境，再按[移植记录](../docs/flashchord-wuji-port.md)接入训练。

## 早期工具

[benchmark_gpu_v2.py](benchmark_gpu_v2.py) 在指定的测量区间采集 GPU 数据，结束后丢弃临时权重。[compare_gpu_benchmarks.py](compare_gpu_benchmarks.py) 用固定容差比较两次测量，并单列微批大小是否改变。旧的 [benchmark_gpu.py](benchmark_gpu.py) 仍对应第一版报告，它的整段 GPU 中位数包含 CPU 导出性能记录时的空闲时间。[consolidate_sources.py](consolidate_sources.py) 用于归档历史研究目录，默认先列出要复制的内容。
