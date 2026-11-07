# 角动量账本：有限步变化与 PRE 接触力矩并非严格相等

2026-11-06。原full05全部1808子步、9473条generated contact的一次实际保存算术已完成，独立保存复算9/9通过。0新增MJB、MjData、backend、模型、物理或训练；沿用上一轮唯一被动读取的静态元数据范围。全部行满足本轮解释前提，但没有残差准确性通过门，也没有因此取得完整扳手/参考/CHORD任务资格。

## 观测定义与来源

free关节后三维qvel是物体body轴角速度，实际主惯量为[1.1704360e-6, 1.9457894e-4, 1.9457894e-4]kg·m²，body_iquat为wxyz=[1,0,0,0]。新检查对完整body_inertia/iquat字段重核A0/A2的copy/set_const两个阶段，逐FP32字节绑定固定context，不能以先前质量检查代替。body与主惯性坐标的区分见[MuJoCo官方说明](https://mujoco.readthedocs.io/en/3.1.6/XMLreference.html#body-inertial)；自由体角速度坐标见[官方概览](https://mujoco.readthedocs.io/en/3.2.2/overview.html)。本次运行语义最终以冻结的本地smooth/forward/math源码为准。

采用明确声明的F64观察约定：实际body_iquat和每行pre/post的原始FP32 wxyz先转F64、再按F64范数归一化。保留原数据、范数、归一化值和R矩阵，不改写原qpos，也不冒称原GPU位级姿态重放。

`I_body = Q_i diag(I_actual) Q_i.T`

`L_world_endpoint = R_body_endpoint I_body omega_body_endpoint`

两端各用自己的R，计算`delta_L=L_post-L_pre`。不能用I_body乘角速度差代替，也不在世界角动量方程再把gyro项当外力矩加入。

物体外接触含HO及terrain，按原ci/有序geom、frame和已解码force6重建：`tau_COM = sum(cross(pos_solverPRE-COM_solverPRE,F_world)+direct_tau_world)`。物体在geom2取正、在geom1取负，F和direct torque同符号；direct只计一次。内部物体对抵消，其他非物体接触不计入物体总和。保留弱/零/负值，不按Fn、接触深度或误差筛行，不重新解码pyramidal EFC。

用原有效FP32步长h（F64值0.004999999888241291s）形成`e_L=delta_L-h*tau_PRE`及等效力矩`e_L/h`。COM0下均匀重力关于COM力矩为0；其他静态项沿既有账本，外施力只保留原源码零断言/max0的范围。它是离散账本，不强制连续h*tau与有限步delta_L精确相等。

## 全量结果

| 原总体 | 子步数 | 等效力矩残差向量RMS / mN·m |
|---|---:|---:|
| ALL | 1808 | 1.129187 |
| donor | 1200 | 1.205765 |
| baseline | 96 | 0.964757 |
| direction | 512 | 0.959451 |
| A0全部 | 904 | 0.896316 |
| A2全部 | 904 | 1.321644 |

事后全量描述：残差范数中位数0.00285335mN·m、p90=0.788820、p99=4.816041、最大14.518062mN·m。最大值位于原row929，A2 donor，phase6/substep2，未删去。接触总力矩向量RMS为18.978215mN·m，残差RMS与它的比值5.9499%仅为同单位量级描述，不是“准确率”或新的通过阈值。

线动量与角动量残差的单位不同，不能直接比较两个数值大小或拼成总分。当前角动量结果要求后续学习明确处理观测到的离散缺陷，不能以线动量残差较小就强制全部六维动量残差为0。

源相位描述保留全部1808行：solver-PRE位置和COM与同条pre_qpos位置的差为0；归一化F64的PRE姿态与pre_qpos姿态最大矩阵元素差6.30129e-8。原q范数偏离1的最大值1.45781e-7。重算HO的COM力矩与原保存聚合逐元素数值差0。这些描述支持本次时相/符号账本，不能证明全hidden状态或独立完整pyramid解码。

## 独立复算与可识别性

独立实现以四元数向量叉积旋转而非生产者矩阵链形成L，逐分量fsum计算力/力矩；核全部端点、原ci、raw字段/模式/路径、实际惯量字段、616个来源/ZIP/当前文件/输出pins，以及全部12项路径/模式汇总。最大实现间差：L端点6.23416e-19、delta_L/残差8.67362e-19N·m·s，总力矩2.77556e-17N·m，等效残差1.73472e-16N·m。F64复算容差事前固定，仅检验实现一致性，没有将残差接近0设为通过条件。

有限solver stationarity、implicitfast有效加速度、gyro/姿态有限步推进、FP32与F64 observer差异，以及未保存的完整passive/constraint分解仍混在残差中，不能唯一归因。静态元数据没有第二次MJB独立重读；独审明确沿用其源码支持范围。原human力未知、whole-hidden未知、原参考/三臂任务FAIL、单缓存谱系和ego N=0不改变。

## 对下一方法的约束

下一项进入WM监督目标的实现准备，停止追加Actor迭代：先仅把原152个restored的20ms组（608子步）中的HO/nonhand线冲量和COM角冲量分源汇聚，绑定已有精确native58输入/目标/实际控制和固定palm坐标。donor300与fresh152不混入训练或调参。

候选对照使用相同14维旁支、相同模式/冲量监督及同一实际初始权重/normalizer；method只增加状态变化与分源冲量之间的监督一致性项。若其中使用每行`r_saved`，必须称为**监督目标之间的一致性**：该残差也是target，不进入rollout输入，不视为推理时已知修正或严格零缺陷物理定律。接触模式target是solver边界约束见证，不能冒充human intent或保证未见输入的零HO梯度。

当前只完成新方法源码建议，尚无此目标dataset、recipe、训练或收益。下一步先落实精确组/坐标/时相合同与目标生产，不以“9/9 PASS”替代这些要求。

## 证据与成本

实际worker4.9086399秒；源审、3个生产者合成检查、独立数学/ABI检查、独审和绘图另计。两幅完整图已经逐一目视检查，保留全部1808行和两donor全部三轴。

- [实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_CPU_v1/report.json)，SHA `63f797255bc3f7418135d0cdb5a0d578d116774bead974a734695caa146cbf3f`
- [独立9项报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_independent_v1/report.json)，SHA `f208f626aea05797aec2337055cfa29a03bbe163db896303ae598276cae542c0`
- [全部残差与同单位描述](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_figures_v1/report.json)
- [全1808图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_figures_v1/all1808_angular_momentum_consistency.png)
- [两donor全部三轴力矩图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_figures_v1/both_donors_all3_moment_axes.png)
- [冻结协议](D:/workspace/supermimic-full151-observation-study/protocols/full151105_saved_object_angular_momentum_CPU_run_v1.json)，SHA `9b3e6b5eca3c184399af3cc0d9bf31e8a3458362b78161f3ea9137fa90b18490`
- [新WM方法源码建议](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_WM_method_source_feasibility_v1.md)
