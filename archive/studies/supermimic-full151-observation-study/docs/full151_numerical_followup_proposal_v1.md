# full151 数值合同后续提案：保持原资格门槛

状态：POSTHOC_ENGINEERING_DRAFT_NOT_AUTHORIZED_NOT_EXECUTED。原单次20LP、源码、输入、raw witness、失败结果和无 READY 状态全部保留。本提案不运行新LP、优化、模型或物理，不修改原 `1e-12` 资格阈值。

原结果20个HiGHS成功，只说明其已配置 `1e-9` solver feasibility 与退出条件；原34gate中 `all20_raw_arithmetic_1e12=false`，所以仍是输入候选未合格。这是数值资格失败，不是观测轨迹物理不可行证明。原始postC2控制/参考力门槛为pass也不能倒过来豁免raw LP gate。

## 必须分开的两类 witness

fit 和 coefficient-second-difference 的绝对值 epigraph 是辅助证据变量，不决定几何曲线。可在**另立的只读诊断**中根据保存的 P 重新计算其必要最小值，并向正无穷 `nextafter` 取外向舍入，重算同一原矩阵各row，保留原 x/result.fun 与新 canonical epigraph 的差异。如此只能证明该固定P是否存在更准确的 epigraph witness；不能说原HiGHS witness已通过，不能冒称目标值完全相同/新全局最优，也不能改写当前receipt或READY。

joint4另有真实 `target_upper` 原始row约 `+1.1025e-10 rad`，超过 `1e-12`。它涉及P，**无法由epigraph重建消除**。必须独立展示剩余物理控制row残差；禁止把所有失败都称为“仅辅助变量浮点误差”。原2µrad规划收缩与真正C2域资格是不同门槛；实际C2域pass不证明该更紧LP row通过。

## 若后续授权：一次固定的重新表述，而非参数搜索

推荐先冻结单一新 numerical formulation，再另开一次有界20joint运行，不从多个方法/权重/seed/平滑参数挑winner。其科学目标、目标数据、P0/P1、requestedinit、tau/u0、hard∩ctrl/C2/rate/force margin、目标函数权重4、原raw `1e-12`审计与postC2全部门槛保持；不换solver成功标准来替代外部检查。

唯一推荐组合的第一部分是 fixed centered variables `y_i=P_i−P0`（i=2..152）：从源basis直接构造 `q=P0+B1*(P1−P0)+B_free*y`，导数用 `B1'*(P1−P0)+B_free'*y`；hard box相应平移。这样减少绝对位置与导数零行和的抵消，P0/P1不变、初始qd不变。此为数学等价的新数值表达，仍需保存与旧原矩阵在同一 reconstructed P 上的双重检查，不能假定FP64运算逐位相同。

组合的第二部分是所有不等式row固定乘 `1e4`（单位随row保留，正数缩放不改数学可行域），保留原solver配置，不尝试多个scale。该factor来自原solver1e-9与资格1e-12之间的量级差再留10倍数值裕量，**不是精度保证**；HiGHS内部缩放/退化仍可能使原row残差失败。scale属于后验工程选择，必须在新运行前冻结、只算一次，不把该选择称为预先的科学确认。

上述两部分是一份唯一的工程提案，不构成执行授权。每joint记录原/新矩阵、变量变换、row units/scales、solver原始terminalx/状态、attempt/return次数、原单位 `Ax−b`/bounds、objective与辅助witness诊断。若新terminal仍失败，保留失败并停止，不改margin/门槛、不修P、不自动retry或选择历史轨迹。

即使新数值输入资格通过，也只是完整曲线/控制 scaffold：没有重新FK/碰撞、独立151contact意图、native151 reset/replay/新checkpoint/controller consumer或3s物理资格。旧539d的geometry及prefix104的学习结果不迁移为这些资格。

## 当前只读审计与成本

[独立生命周期/来源审阅](D:/workspace/supermimic-full151-observation-study/outputs/full151_input_saved_lifecycle_independent_v1/report.json)只核JSON/文件/源和失败语义；Contact另做raw数学复算，避免重复求解。producer没有采样PID/Creation；Root已保存 [工具执行观察](D:/workspace/supermimic-full151-observation-study/outputs/full151_cpu_input_v1/root_execution_observation.json)，明确exit1、once/no retry、CUDA mask−1及PID未测，不应补造OS身份账本。solver时间包含于worker时间，worker包含于外层tool时间，三者不能相加。
