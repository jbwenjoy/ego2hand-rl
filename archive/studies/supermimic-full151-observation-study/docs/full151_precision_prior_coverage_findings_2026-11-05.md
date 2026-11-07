# 数值精度、接触先验与诊断覆盖：2026-11-05 后续结果

本轮完成了 GPU 精度对照、训练中实际几何先验的局部响应比较，以及两臂全部3600条训练先验日志复核。结果将下一步问题收窄到诊断覆盖与代理可观测性：切换到原生 BF16 网络没有改变本批任务评分梯度5/9同号的结果；局部探针的穿透项全部为零，但完整训练日志证明穿透项均值在每次策略更新中都为正。

这不是CHORD收益或失败原因的唯一归因。三个固定final2000策略仍未通过全部任务门槛；共同seed和初始化并未保证共同的实际预热轨迹，原三臂比较的因果限制继续保留。

## 同一权重的 GPU 精度对照

训练配置为AMP=True、compile=False；学习源码明确设置matmul TF32=True，但历史运行时标志未保存。新进程严格通过原CPU加载器验证后，才将验证过的原生WM核心移到RTX 5090，按固定顺序执行两种模式：FP32存储且允许TF32，以及网络BF16、归一化与积分FP32且允许TF32。没有调用CPU包装器处理GPU输入，没有物理、Actor前向、优化器或编译。

两种模式各一次batch8前向、一次u AD；与既有CPU结果比较时不再执行CPU前向。两组输入、归一化特征及权重相同；实际网络输出dtype为FP32/BF16。每种模式前后记录了TF32、reduction、确定性与相关环境标志，均保持一致。CPU/CUDA及host RNG原始记录另经保存审计核对。

| 数值模式 | 同一9组稳定方向中同号数，两个epsilon分别检查 | 对CPU梯度最大绝对差，score/rad |
|---|---:|---:|
| 已存CPU FP32 | 5/9；5/9 | 0 |
| GPU FP32存储、TF32允许 | 5/9；5/9 | 0.00064114 |
| GPU原生BF16网络、TF32允许 | 5/9；5/9 | 0.01258123 |

GPU两模式之间的梯度最大绝对差为0.01241365 score/rad。它们保留相同64条物理比较记录、32个依赖方向组及3个裁剪稳定组。仅各运行一次，没有模式内变异估计；CPU与GPU还存在后端、硬件和TF32差别，不能把所有数值差唯一归因于精度。

[实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_A2_GPU_precision_v1/report.json) · [独立保存审计187/187](D:/workspace/supermimic-full151-observation-study/outputs/full151105_GPU_precision_saved_independent_v1/report.json) · [全部数值差热图与表](D:/workspace/supermimic-full151-observation-study/outputs/full151105_A2_GPU_precision_figures_v1/GPU_precision_gradient_differences.png)。操作系统身份12项通过，原两个PID已退出；图已由绘图agent和Root实际查看。

## 实际训练先验的局部响应

按执行前冻结协议，使用原binding_v3、实际R目录中的冻结数学、原训练spec和64个训练方向。分别实际构造P_ONLY与A2，评分全部160个保存物理下一状态与8个固定A2 WM预测；4次原目标前向中只读hook保存了现有SDF、软权重、support及COM点/法线，没有另行查询几何。一次batch8 WM前向和4次u VJP均完成。

原始穿透P、加权穿透BP=-0.1P、原始support S、加权support CS及实际A2总bonus分开保存。评分使用post20ms状态和preaction目标k；实际d_eff为主方向，裁剪与中心偏移保留。噪声、单位缩放、分量抵消与dt=.02规则在评分前固定。

| 分量 | 32方向组的稳定非平坦数 | 稳定组同号数，每个epsilon | 解释 |
|---|---:|---:|---|
| P与BP | 0 | NOT_ASSESSED | 所有160物理输入及8预测的值均为0，没有非零穿透信号 |
| S与CS | 10 | 8/10 | 包含3个裁剪组；其余22组至少一个尺度低于门槛 |
| 实际A2总bonus | 10 | 8/10 | 本批P为0，总bonus与support数值相同 |

全部320条epsilon比较、160个分组分类均保留。两个epsilon不是20个独立稳定样本；未裁剪稳定组5/7只是预声明的次要描述。此处是几何代理在物理下一状态上的分数，既不是solver测得的接触力，也不是原生GPU AMP先验梯度。

例如A2/k0的物理下一状态S约0.36021，WM预测S约0.59766；最近采样SDF分别约+1.508mm与+0.371mm，最大软权重约0.321与0.933。它们是实际保存的代理输出，不是从位置RMS推测的接触误差。A2后续远离手部的状态及零评分也完整保留，没有删除。

![全部物理与预测的先验值](D:/workspace/supermimic-full151-observation-study/outputs/full151105_trained_bonus_saved_description_v3/bonus_levels_all160_plus8.png)

[实际CPU报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_trained_bonus_CPU_v1/report.json) · [独立保存审计523/523](D:/workspace/supermimic-full151-observation-study/outputs/full151105_trained_bonus_saved_independent_v1/report.json) · [全部320条比较图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_trained_bonus_saved_description_v3/bonus_secant_AD_all320.png) · [全部采样表面诊断](D:/workspace/supermimic-full151-observation-study/outputs/full151105_trained_bonus_saved_description_v3/proxy_surface_all160_plus8.png)。独审从保存SDF、COM点/法线、basis64与target64重算P、softweight、support和S，最大相关算术差约1.19e-7；没有重新运行FK、几何、目标或模型。CPU权重、normalizer、RNG不变性仍是源支持的运行时断言。三张图均由作者及Root实际查看，作者描述与独立复核分开。

## 完整训练日志排除了“惩罚全程为零”

局部P为零不代表训练中的P始终为零。另一个执行前冻结的只读日志reader覆盖两臂全部1800次策略更新：raw iteration 200..1999，对应显示的已完成训练201..2000；每臂14400次H8目标前向及dt应用。未混入旧schema或其他cohort。

| 仅想象先验的记录量 | P_ONLY | A2 |
|---|---:|---:|
| 正P均值的更新次数 | 1800/1800 | 1800/1800 |
| phase<150的active行数 | 1,821,456 | 1,821,808 |
| 按active行加权的P均值 | 1.08114969 | 1.07391097 |
| 加权穿透项均值 | -0.10811497 | -0.10739110 |
| 加权support项均值 | 0 | 0.71024707 |
| 记录总bonus均值，未乘dt | -0.10811497 | 0.60285597 |

各均值用sum(active_rows×logged_active_mean)/sum(active_rows)计算，不是各次更新的简单平均。P_ONLY也计算S作诊断，但support系数为0。总bonus大小由不同公式决定，不能作为方法收益比较。

“每次更新的P均值为正”不能恢复每行P为正的比例，也不证明控制或Actor梯度非零。日志均值没有使用survival/discount权重，不能当成实际Actor损失贡献；它们也不证明真实接触或穿透。

[全部1800次曲线](D:/workspace/supermimic-full151-observation-study/outputs/full151_trained_prior_log_summary_v1/full1800_prior_histories.png) · [3600行CSV与报告](D:/workspace/supermimic-full151-observation-study/outputs/full151_trained_prior_log_summary_v1/report.json) · [独立原日志/CSV复核20/20](D:/workspace/supermimic-full151-observation-study/outputs/full151_training_prior_logs_saved_independent_v1/report.json)。独立流式重算最大数值差5.55e-17，日志与原final196项健康证据的byte-prefix、metrics及计数关联一致；没有重新执行健康审计或模型。图已由作者和Root实际查看。

## 下一步已冻结为同一solver时相的可观测性检查

下一项研究使用已保存的全部1808个pre-qpos/qvel，比较同一solver时相的原生HO生成接触深度与P代理。先以452个已知post20端点、160个首步anchor及1808个solver物体位姿验证原生state58转换，全部通过后才允许单个P_ONLY factory和15批CPU评分。

原始接触深度属于积分前状态，不能拿15ms接触深度与20ms代理值比较。HO深度按原geom集合逐接触重建，不用全类别最大穿透、自碰撞或正力过滤替代；同时保存代理实际采样geom IDs和逐geom SDF，区分几何集合覆盖缺口与同姿态代理欠敏感的候选现象。未生成原生接触不等于全局不存在穿透。

[同solver时相预注册V2](D:/workspace/supermimic-full151-observation-study/protocols/full151105_solver_phase_penalty_observability_CPU_prereg_v2.json)目前只有源码/格式可行性证据；实际转换器、密集评分和新的Root运行配置尚待完成。固定帧依据源结构与452个保存reader frame，不伪称保存了1808次mocap实测。

本轮新增物理子步和优化器更新均为0，没有新长训练。[计算账本](D:/workspace/supermimic-full151-observation-study/outputs/full151105_precision_bonus_followup_compute_ledger_v1.json)分别记录GPU两次与CPU一次WM前向、目标评分和梯度开销，不用不同任务耗时声称加速。Graph隐藏scratch仍UNKNOWN，物理唯一Jacobian、HO力梯度、参考物理可行性、真实ego泛化与多轨迹收益均未因本轮而取得资格。Goal保持active。
