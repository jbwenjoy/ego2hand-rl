# FlashCHORD 接触载荷与策略评测

2026-11-21。以下检查针对同一台 RTX 5090 上的单刚体 Wuji 训练环境。纸盒视频的无物体辅助名义轨迹此前已在独立的 CPU MuJoCo 方案中通过开发门限。本页记录移植到 FlashCHORD/MuJoCo-Warp 时发现的问题，以及如何评估新训练的策略。

## 求解器提前收敛

官方场景将桌面表示为运动学支撑体。其六个自由度使编译所得平均惯量统计量升至约 1.579 × 10⁹，实际 Wuji 手及 150 g 纸盒可动自由度在初态的平均广义惯量约为 0.07904。MuJoCo-Warp 用平均惯量对牛顿法的下降量、梯度和线搜索阈值进行归一化。实际原配置容差为 10⁻⁶，前 240 个子步每步只做 1 次迭代。六帧试验中，基于完整广义质量矩阵计算的纸盒竖直方向动力学方程残差 RMS 约 294 N，峰值约 2,914 N。直接读取约束行并解码摩擦锥与 Newton 报出的法向载荷吻合，错误来自不充分的求解。单纯增加迭代上限仍允许原判据提前退出。[原始接触读数](../runs/flashchord_solver_replay_diagnosis_v1/report.json) · [移动物体全质量矩阵对照](../runs/flashchord_dynamic_cpu_default_v1/env.npz) · [修改桌面表示与归一化的独立 2×2 因果试验](../runs/wuji_table_causal_cpu_v1/report.json)

修正通过版本化的 `WujiSimConfig` 启用。沿用原支撑网格和全部物理参数，装配完成后读取该场景的实际广义质量矩阵，对可动自由度的对角线求均值，仅替换求解器用于收敛判定的统计量。保留原容差、接触参数和最多 100 次迭代。批量世界共享同一统计量时，会核对每个世界的质量、质心、惯量与关节反射惯量。官方 v0.4.0 的 replay、RL 和检查点推理路径都经相同的类型化分派。旧配置仍使用旧求解器参数。训练时将配置清单中的 overrides 应用于新运行，避免接续旧物理条件训练出的优化器状态。[数值条件修正](../src/supermimic_research/runtime/flash_solver.py) · [新物理清单与证据指纹](../configs/flashchord_wuji_physics_v2.json) · [三个复制世界的构造检查](../runs/flashchord_conditioned_batch_cpu_v1/report.json)

## 静置、落体和动态手物接触

对三条开发参考各自的真实物体和名义质量，在保留同一手、支撑面及材料的场景中，将手移开，分别从桌面与离桌 50 mm 释放物体。纸盒的解析落下时间约 101 ms，修正后的 GPU 求解器测得 102.5 ms 初次承重。在 150 g 纸盒的最后 0.25 秒，静置及落体后桌面载荷分别为自重的 1.000、1.004 倍。整段自由刚体平移三个方向的完整方程残差 RMS 分别为 0.00016、0.00023 N。三种物体的 CPU 和 CUDA 文件均通过预设的载荷、落体时间与残差检查。另在 46 个控制间隔、含手物交互的纸盒轨迹中，CPU、CUDA 的竖直方程残差 RMS 分别为 0.00176、0.00158 N，求解器中位迭代次数为 4。这些数值验证了特定仿真条件下的载荷读数；纸盒实际质量和摩擦仍属于未实测参数。[GPU 静置与落体](../runs/flashchord_rest_drop_cuda_v1/report.json) · [CPU 同条件](../runs/flashchord_rest_drop_cpu_v1/report.json) · [动态 CUDA 状态与完整惯量项](../runs/flashchord_conditioned_dynamic_cuda46_v1/env.npz) · [动力学核验工具](../tools/validate_flash_wuji_contact_force.py)

## 保留原始失败与独立物理复算

修正前，32 环境单轨迹策略在 2,048 环境步后评测。由首帧按原时间关闭物体辅助，恢复训练原本采用的任务失败判据，在第 46/140 个区间因物体跟踪失败终止。另以各 384 环境步的多参考无模型策略、带世界模型策略作健康检查，三条开发参考的执行长度分别为 46/140、93/220、42/275 与 30/140、93/220、42/275。七次均未达到释放阶段。中止后的源状态没有补值，只有到访区间的误差有数值。旧物理对应的载荷及吞吐不能作为修正配置下的训练证据。[单轨迹原始评测](../runs/flashchord_learned_quality_v1/single_subject1_G04_2_r2/report.json) · [多参考纸盒](../runs/flashchord_learned_quality_v1/mf_subject1_G04_2/report.json) · [混合纸盒](../runs/flashchord_learned_quality_v1/hybrid_subject1_G04_2/report.json)

在前三个名义条件一致的独立求解器中重跑相同广义指令。CPU 上 1,840 个子步的全部位置、速度和控制缓冲均逐位吻合。CUDA 上即使只比较两个新建求解器，在初始条件和输入逐位相同的情况下，首子步手指速度已出现浮点差异，当时纸盒尚未接触桌面。这项对照无法将最初的差异进一步定位到某一内核。保留原始 10⁻⁹ 独立复算门限及实测差异。[CPU 同指令复算](../runs/flashchord_solver_replay_cpu_v2/report.json) · [GPU 首次分歧](../runs/flashchord_solver_replay_diagnosis_v1/report.json)

为避免把 GPU 逐位差异等同于动作成败，在出现任何达标的正式学习结果之前另行冻结物理重复性标准。原策略从首帧完整通过全部源动作和物理门限后，在两套独立初始化的环境中，仅重放相同手腕和手指逐子步指令。每次都必须完整通过原轨迹、接触负载、离手、穿透和关节等门限。报告三条轨迹两两之间的物体位置、转角、关节位置及速度差异。物理读数缺少与当前场景匹配的独立验证时，结论标为暂定。原精确复算结论另列。已在 CPU 中重放相同的六帧失败前缀，两次复算状态完全相同，完整轨迹和离手判据仍明确失败。[独立物理复算标准](../configs/hocap_carton_physical_repeat.json) · [统一评测入口](../tools/evaluate_flash_wuji_quality.py) · [失败前缀的双重复验证](../runs/flashchord_physical_repeat_cpu_v1/report.json)
