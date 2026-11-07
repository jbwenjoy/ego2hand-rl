# Constrained prefix 终态审阅

唯一正常返回的终态候选 `64aa2cfe…ad9b43` 保留为本次 canonical；没有从历史迭代中选择替代。该候选满足保存的线性规划约束和控制点硬范围，但 **5/61 个时刻的手物最大重叠超过 3 mm，几何资格失败**。它未通过当前物理试验的前置门槛，也不能证明该动作不可行。本审阅仅重算保存的数组、原生接触记录、矩阵和日志；24/24 一致性检查通过，没有新增碰撞查询、LP、优化或物理。

## 剩余违规的位置

| 采样点 | 时间 | 最大重叠 | 该时刻最深几何对 |
|---|---:|---:|---|
| 28 | 0.140 s | 3.004905 mm | object/cube × finger1_tip_sensor_col |
| 29 | 0.145 s | 3.026379 mm | object/cube × finger1_tip_sensor_col |
| 53 | 0.265 s | 3.023741 mm | object/cube × finger1_tip_sensor_col |
| 59 | 0.295 s | 3.040943 mm | object/cube × finger2_tip_sensor_col |
| 60 | 0.300 s | **3.194716 mm** | object/cube × finger1_tip_sensor_col |

这些时刻也都超过 2.999 mm 的规划上界，最严重的规划超量为 0.195716 mm。按所有几何对逐项计数，共有 **7 个 pair/time 违规**：0.295 s 的 finger1 末端另有 3.012042 mm；0.300 s 的 finger2 末端另有 3.094675 mm。完整清单在 [violating_pair_times.csv](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_terminal_review_v1/violating_pair_times.csv)。仅检查 16 个拟合节点也会发现 frame7、15 失败；其余中间采样仍不能省略。

终态 61 个采样共保存 369 条手物原生几何接触；没有生成手自碰接触。未生成 pair 的距离仍为 NaN，不能据此宣称正间隙或整段无自碰。51 个采样存在同 pair 等深接触，合计 82 个 pair/time tie 槽位；最严重的末端拇指 pair 有两个等深接触。确定性分支和这些记录不构成全域光滑性证书。

## 拟合取舍

总 cost 为 312.543411。材料位置项占 **68.10%**、软重叠项 **24.90%**、骨方向项 **6.69%**、材料法向项 **0.309%**。相邻和二阶 q 差分合计仅占约 0.0000701%；它们仍是软偏好，不能代替显式控制约束。

F1/F2/F3 材料点平均误差分别为 **4.282 / 2.127 / 0.261 mm**，最大值为 **7.489 / 3.021 / 0.548 mm**。法向平均误差为 **61.19 / 52.62 / 15.49°**，最大值为 **77.76 / 55.53 / 28.42°**。骨方向平均误差为 **16.7307°**，相同目标的指尖平均误差为 **16.4134 mm**。在减少重叠并满足线性控制限制时，材料拟合仍有明显残差；即使材料点较近，也不等于整根手指或其末端碰撞体合法。法向项沿用原单位向量残差和 .25 权重，不是硬角度门槛。

保存的线性矩阵独立重算为所有 `A(x−x_repeat)−b_centered ≤ 0`，最大值 −9.9012e−10，控制点界限超量为零。最紧的组是 target_lower，5 行距边界小于 1e−7 rad；target-rate、track/hold raw-force 组在终态没有同等接近活跃。不同组有不同单位，不能把这个全矩阵最大数当作统一物理安全裕量；更不能把它解释为非线性几何可行或实际受力已通过。最终 FP32/C2 独立导出由 Root 单独完成，本报告没有代替该步骤。

## 对“收敛”的实际证据

一次 HiGHS L1 初始化成功；一次 SLSQP 在 80 次迭代后正常返回 **status9 / Iteration limit reached**。实际有 240 次 objective fun、81 次 objective Jacobian、241 次 constraint fun、81 次 constraint Jacobian、80 次 callback，未触及 250 次 objective 上限。第一条目标 cost 为 1895.272162，终态为 312.543411，下降约 83.51%。

保存的 240 条 cost 是线搜索/试探函数调用，**不是 80 条主迭代的完整诊断**。其最高试探 cost 为 200134.18；末 20 条仍在 312.54–1317.06 之间。图中的尖峰不能直接解释为主迭代发散；同时，status9、剩余违规和这些 cost 也不足以宣布收敛。原始目标梯度最大绝对值 396.11 不是带约束的 KKT 残差，不作最优性证书。

目前只保留了 latest 完整检查点，没有逐主迭代的最大/累计几何违规、活跃 pair 或 tie 分支轨迹。因此不能从这些日志判断历史可行性、证明分支切换导致失败，或据此选择某个旧状态。所有历史状态都不替代规定的 terminal result.x。

计数只有一个共享 `g`：**18,077 = fit16 的 3,616 + constraint61 的 14,461**。parent fit 中名为 `actual_collision_calls=18077` 的字段读取同一个共享对象，不能再加一次。保存的 18,077 个 event 索引连续且唯一；本次分析自己调用几何的次数为零。

## 下一步方法审阅

当前结果只建立了受限线性子问题的可行初始化，没有建立线性与非线性约束的联合可行性。下一次如另立有限 CPU 协议，最有区分力的改变是把 **非线性可行性恢复与拟合目标分开**：预先固定初始化、线性约束、阈值及总预算，先以明确的几何违规 slack 为目标，再仅在可行性成立后拟合。它应是一个公开的新方法条件，不能在本次失败后延长迭代、悄悄调权重或放宽 3 mm 门槛。局部恢复仍不能证明全局可行或不可行。

新协议还应逐主迭代保存 cost、最大/累计违规、选中 pair/contact、tie 和分支签名，以区分目标进展与可行性进展。活动约束的导数应单独做保留困难分支的局部验证；原生碰撞列表对缺失 pair 没有正间隙，不能将其补零或把确定性 argmin 当作光滑距离。

控制点凸包、Bernstein 原始力矩和 moving-reference hold 条件是充分性限制；它们可能排除其他合法曲线，但本终态也没有证据证明这些限制就是失败原因。若未来要改变它们，必须另声明方法，而不是将控制或资格门槛随结果移动。固定 H123 材料、持续贴附、手解剖对应与 fixed-palm/object 映射仍有未知置信度；滑动、滚动、释放或映射修改属于另一个假设条件，不能自动作为本次 fallback。

主目标仍是完整运动下的重定向及实际动力学资格。此 prefix 不是终点，不能接回旧 151 帧 q 尾段，不能从此前静止持有结果继承动态初态或跟踪资格；当前也没有真实 ego 数据、接触标签或测量力的新证据。

![终态几何、拟合与试探 cost](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_terminal_review_v1/terminal_review.png)

[机器可读报告](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_terminal_review_v1/report.json) · [全部 122 条约束](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_terminal_review_v1/all122_constraints.csv) · [目标试探日志表](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_terminal_review_v1/objective_trials.csv) · [只读复核脚本](D:/workspace/supermimic-observation-constrained-prefix-study/scripts/analyze_constrained_terminal_saved_cpu.py)
