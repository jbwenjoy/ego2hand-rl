# 同求解前状态的密集几何惩罚分析

1808 行分析已完成，独立 saved-only 审计 **35/35 通过**。原生生成手物接触深度严格大于
3 mm 的记录有 44 行，原训练 P_ONLY 代理在其中 42 行返回正惩罚，另外 2 行为零。
本批没有“代理正惩罚、原生已生成 HO 深度不超过 3 mm”的记录。
这些是相依保存记录的描述性计数，不能写成独立测试集召回率或方法收益。

| 路径/模式 | 行数 | 原生 HO >3 mm | P >0 | 原生超阈值且 P=0 |
|---|---:|---:|---:|---:|
| A0 donor | 600 | 30 | 29 | 1 |
| A0 restored baseline | 48 | 0 | 0 | 0 |
| A0 restored perturbation | 256 | 0 | 0 | 0 |
| A2 donor | 600 | 14 | 13 | 1 |
| A2 restored baseline | 48 | 0 | 0 | 0 |
| A2 restored perturbation | 256 | 0 | 0 | 0 |
| 全部 | 1808 | 44 | 42 | 2 |

## 它改变了哪项解释

此前 160 个局部 physical next20 与 8 个 WM next20 状态上的 P 都为零，本批正值全部来自
完整 donor 路径的其他求解前时刻。96 行 baseline 和 512 行扰动分支的同相位原生深度也均未超过
3 mm。因此不能从此前局部零值推论惩罚未实现、训练期间没有惩罚信号或当前所有物理状态均漏检。
已有完整日志还显示 P_ONLY 和 A2 各 1800 个有效更新的 P 均值都为正；想象状态与本批实际状态不同。

两处同位姿漏检则确实存在，均位于代理已覆盖的 `robot/right_finger2_link3_col`（geom 27）
与 `object/cube` 圆柱（geom 1）之间。完整最深并列集合均只有这一 pair：

| 原始行 | 路径/相位/子步 | 原生最大深度 mm | 全局代理最大采样深度 mm | 原生最大−全局代理 mm |
|---|---|---:|---:|---:|
| 47 | A0 donor / k11 / 4 | 3.030756 | 2.964228 | 0.066528 |
| 926 | A2 donor / k5 / 3 | 3.573335 | 2.971762 | 0.601572 |

上表的全局代理值取全部采样几何的最大深度，不能当作该原生最深 pair 的配对值。
A2 行的 geom 27 采样深度实际为 2.814320 mm，与同 geom 原生深度之差约 0.759015 mm；
全局 2.971762 mm 来自 geom 31，其对应原生次 pair 深度为 3.072037 mm。
A0 行最深 pair 与全局代理来自同一 geom 27。

该胶囊的原固定采样为 64 点。两行均属于已覆盖几何上的低敏感候选，本批没有最深 pair 的
几何集合缺失。当前证据没有唯一分离有限采样、物体外表面代理和原生窄相语义的作用；不能通过
减小 3 mm 门槛把两行改判，也不能把零报警写成全局几何真值资格。

## 坐标、时刻与算术资格

输入为同一批已独立 29 项核验的原生 GPU/B1 状态。保留完整 452 终点、160 锚点的 58 维原规则，
不再运行已失败的 CPU 状态转换。额外对齐全部 1808 行 GPU p/R 与对应求解前 world pose：
记录的 FP32 palm/solver 操作数经 NumPy F64 显式乘加生成派生参考，按原 8eps32 逐分量规则检查。
最大位置差为 `3.377581214714098e-8 m`，旋转元素差为 `5.052720553067047e-7`，失败元素/行均为 0。
该参考不是新测量或原 GPU FP32 真值。

全部计分使用同一 PRE q/v、对应整数控制相位 k 和同一原训练工厂，避免拿 20 ms 状态与前一
solver 时相的接触深度直接比较。原生标签只取完整 enabled-hand × object 生成接触集合，包含零力
候选；不把自碰、其他类别最大穿透或接触缺失替代为 HO 标签/全局间隙。

原 P 公式为 `square(max(clamp(-sampled_sdf - .003, min=0)) / .001)`，BP 为 `-.1*P`。
保持原 1344 点、21 个采样几何及 FP32 运算；主覆盖保留全部精确并列最深 pair，副覆盖保留全部
严格超过 3 mm 的 pair，二者是单独的重叠证据。名义 `.003` 与 FP32 标量 allowance
`0.003000000026077032` 的边界标志均保留。

## 成本与审计范围

唯一实际 CPU run 用时 62.002 秒：显式被动 MJB 常量读取 1 次、P_ONLY 工厂 1 次、前向 15 次
（14×128+16），计分全部 1808 行。工厂内部另有 1 次源码可追溯的被动 MJB 读取，未单独插桩。
新状态转换、WM/Actor 权重加载、AD、优化、Env、原生 FK/碰撞查询、物理与 GPU 调用均为 0。
RNG/参数/缓冲不变由执行源的前后断言支持，独立审计没有原始 RNG/权重快照可重新播放。

独立审计复算全部 `1808×1344` SDF 的 P/BP、行最小值、逐几何最小值、原生 HO depth、全部 ties、
覆盖分类、六组统计、15 个 chunk 与完整 CSV，并核对 12436 个源 pin/归档、actual8 参数、坐标和身份。
完整原生 catalog 由执行源复制，审计据保存常量独立重建 mask，没有重新加载 MJB。

重放的 runtime 保留最后周期进度 2176；完整 2260 raw mask、逐次调用记录、result 与 completion
为最终计数依据。其旧 core runtime pin 与终态 hash 的已知差异明示保留，其他不可变 pin 严格检查。
旧 CPU FAIL、新旧协议和所有已执行源码/证明均未修改。

## 后续顺序

本结果支持继续用该 P 代理做受控研究，但不赋予其硬接触约束资格，也不改变此前三臂整体任务失败。
下一步先以固定 A2 世界模型覆盖全部 452 个 20 ms 控制区间，按 q/qd/p/R/v/omega 的各自单位核对
58 维预测误差随相位的变化，避免又用少数局部状态概括整段。原 controlled GPU goal 仅直接保存了
8 个锚点，故下一诊断不重建 goal、不计算 task score，也不拿 native15 score 替代 controlled20。
只做 source/API 与预算审阅后再形成新实验；不在本轮补跑训练、
调 prior 权重或用两个漏检点挑选更优采样配置。

已知参考几何失败、WM 物理梯度局部符号一致 5/9，以及不同 arm 在 prepolicy100 就存在 WM/
normalizer 分歧，仍分别限制物理与因果解释。后续新训练需具备可行参考与真实相同 warmup 前缀。
真实 ego 数据仍为 0，未知人手接触和尺度/相机谱系保持未知。

## 证据入口

- [实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_solver_phase_penalty_GPU_qualified_CPU_v1/report.json)，SHA `416d8dfcfe01a910a406ba78cdddc28628717bce5a45a917b69f6af583699bf2`
- [独立 35 项审计](D:/workspace/supermimic-full151-observation-study/outputs/full151105_solver_phase_penalty_GPU_qualified_saved_independent_v1/report.json)，SHA `a4008bde96f5b178751ba454886fd6b646e68ff4c577de345383aa94f1f6485a`
- [全部逐行 CSV](D:/workspace/supermimic-full151-observation-study/outputs/full151105_solver_phase_penalty_GPU_qualified_CPU_v1/rows_all1808.csv)
- [完整深度图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_solver_phase_penalty_GPU_qualified_figures_v2/all1808_same_phase_depth.png)与[计数表](D:/workspace/supermimic-full151-observation-study/outputs/full151105_solver_phase_penalty_GPU_qualified_figures_v2/all1808_same_phase_counts.png)：全部数据保留，Root 已视检；v2 仅改善 v1 小分段标签的可读性。
- [计算账本](D:/workspace/supermimic-full151-observation-study/outputs/full151105_matched_replay_dense_compute_ledger_v1.json)
