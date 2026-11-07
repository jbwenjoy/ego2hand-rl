# H0/T0 动态失败诊断：只读保存数据

两个 0.3 s 试验均保留原 **FAIL**，唯一未通过的主门槛是有向 shaft 误差。物体确实转动，失败涉及轴方向轨迹偏离，不能概括为“没有转”或仅仅转幅不足。20/20 数据相位与重建一致性检查通过；本审阅没有新增查询、求解、物理或参数调整。

| 保存结果 | H0 hold | T0 track |
|---|---:|---:|
| 主位置最大误差 | 11.072 mm | 7.209 mm |
| 主 shaft 最大误差 | **33.283°** | **32.807°** |
| 完整 SO(3) 最大误差，仅诊断 | 34.613° | 42.374° |
| q 对参考 RMS | 0.04640 rad | 0.02395 rad |
| qd 对参考 RMS | 1.6565 rad/s | 1.6281 rad/s |
| 实际状态 PD 模型越单关节力矩限幅 | 0/1200 joint-steps | **1/1200 joint-steps** |

末端 shaft 相对共同初态的夹角分别为 24.032°、25.378°，目标为 31.170°。在固定初始物体坐标系中，目标末端 shaft 的 y 分量为 **+0.4064**，H0/T0 为 **−0.1609/−0.1505**。这直接显示方向路径分离。两条实际 q/qd/object pose/twist 在 0–20 ms 逐位相同；第一次新控制更新之后才分叉。比较仍是已知历史下每条件 N1，不是训练 seed 或总体方法成功率。

## 实际 PD 与名义预测

所有 60 个力矩样本均使用 `rawstate[k−1]` 的 q/qd、**solver sample 的新 ctrl**，与 step k 的步前 actuator force 对齐。特别是 step5 使用刚写入的 u1，不能使用上一条 rawstate 中仍保留的 u0。startup-forward 样本不作为已积分的 solver step。

独立计算的实际状态未限幅 PD，经单 actuator 限幅后，与实测 actuator force 的最大差为 H0 **2.27e−7 Nm**、T0 **2.36e−7 Nm**。未限幅量是已知仿射 PD 的计算值，不是额外记录的测量。T0 唯一的限幅发生于 **step5 / solver pretime≈20 ms / finger3_joint3**：名义参考预测 +0.001167 Nm，而实际 q、qd 偏差分别贡献 −0.131203、−0.130924 Nm，合成未限幅 −0.260960 Nm；实测被限为 −0.250950009 Nm。

T0 实测 load ratio 峰值约 1.000000034，仍通过冻结的 1.01 门槛。只有一个 5 ms 的单关节限幅；H0 没有限幅也发生相似轴误差，故没有证据把终态失败单独归因于持续饱和。T0 的关节位置误差较小，但速度跟踪误差仍明显；参考状态下的力矩界限不能直接保证实际轨迹也处于同一状态。

## COM 扳手与相位

从所有保存的实际接触重建 force 与 direct torsion。若 object 是 `geom[1]`，取 s=+1；是 `geom[0]`，取 s=−1。对每条记录：`F=s*frame.T*f`，`τ_COM=(pos−solver_COM)×F+s*frame.T*τ_contact`。累计只包含 step1–60，使用各步 **preintegration solver COM/姿态**；q/qd 的 rawstate[k] 是步末量。

每步均有 1–4 个生成手物接触，没有完全失去接触力的步骤。平均垂直接触力 H0/T0 为 0.28883/0.29438 N，物体重力约 0.29430 N；这不等于动态抓持已合格。实际 COM 角冲量与名义需求明显不同：在固定初始坐标 z 方向累计为 H0 **−0.6796**、T0 **−0.8033 mNm·s**，参考需求 **−0.1109 mNm·s**。角冲量不是转角；实际与参考也处于不同姿态，不能只凭该差值推断唯一原因。

惯量使用 `I_body=R(actual_body_iquat) diag(actual_body_inertia) Rᵀ`。参考需求用相同 actual mass/I_body、Rref 和 analytic ω/α；实际离散检查用 native body-frame ω 的步间差、真实 raw-clock dt、prestate gyro 项，再由 **保存的 Rsolver** 转世界坐标。freejoint 线速度是 world COM 速度（body_ipos=0），角速度是 body-frame。该一步 bodyEuler/线动量余差见 report；它是含积分与 FP32 误差的保存数据一致性诊断，不是连续精确动力学或 solver 精度证书。

现有数据支持“名义运动/参考力矩界限不保证实际接触扳手和轴轨迹得到实现”，以及“预计算控制没有物体姿态误差反馈”。它不能独立分离滑动、滚动、解剖映射、被动手指动力学或接触参数的作用；相对手接触点速度和完整手部广义力分解未保存。没有证明完整 motion 不可行，也没有产生新接触标签、学习收益或硬件结论。

[report.json](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_saved_failure_diagnosis_v3/report.json) · [全部 40 关节诊断](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_saved_failure_diagnosis_v3/all40_joint_diagnostics.csv) · [全部 120 solver 步](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_saved_failure_diagnosis_v3/all120_solver_diagnostics.csv) · [归档执行源码](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_saved_failure_diagnosis_v3/analyzer_source.py)

![动态失败诊断](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_saved_failure_diagnosis_v3/dynamic_failure_diagnostics.png)

v1/v2 原产物保留；各版执行源码按原 report 中 SHA 校验后归档。v3 明确采用 Rsolver 做 bodyEuler 世界变换；相对 v2 rawpre quaternion 变换，余差变化不超过 3.8e−10 Nm。原物理结果、控制、主评分和未识别机制均未改变。
