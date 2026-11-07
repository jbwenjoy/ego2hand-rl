# 固定校准组件的任务score梯度：符号一致，幅值仍有取舍

2026-11-06。在原确认条件未通过后，固定同两个final1000，另做探索性任务score机制检查。唯一实际运行完成，独立保存复算27/27通过。没有新增训练、物理采样或模型选择；原ea22确认失败保持。

## 比较定义

使用原matched post20物体score及真实保存的goal_p/R[k]。每臂一次B8前向、一次对实际绝对控制u的VJP；只冻结的模型参数参与前向，未计算Actor参数梯度。物理端只用原GPU主score；CPU次级项和原native15ms reward不替代它。

全部8个中心、32个物理方向、64个epsilon记录均保留。两种口径分列：

- 实际坐标弦线：`delta_score / actual_delta_u[j]` 对照 `g dot actual_delta_u20 / actual_delta_u[j]`。
- legacy：`delta_score / (2*epsilon)` 对照 `g dot d_eff`，d_eff保持无量纲。

实际正分母及全部8个三baseline门为前置条件；零或负分母整体停止，不用删行或补零修复。clipped/asymmetric标志保留。局部中心梯度与有限、可能非对称的弦线不自动等同真实物理Jacobian。

## 全量与共享物理分类

沿旧规则以物理数据独立生成共享分类：score floor为`max(1e-5,10*max(三baseline范围,baseline-donor误差))`；两epsilon均超过floor、同号且nominal斜率相对差<=.5才标stable。它是有限重复的描述，不是噪声上界。两个模型共用分类，未按模型表现筛样。

32个方向中，6个stable nonflat，25个flat/below，1个unstable。两epsilon不把6个共享方向扩成12个独立样本，也不与旧实验的9方向混池。

| 探索描述 | pointwise only | pointwise + response |
|---|---:|---:|
| 全64行实际口径MAE / score·rad^-1 | 0.228291 | 0.174050 |
| 全64行实际口径RMSE / score·rad^-1 | 0.493487 | 0.471018 |
| 全64行符号一致记录数（含弱/零响应） | 40/64 | 40/64 |
| 共享6方向、epsilon=.002的legacy RMSE | 0.813261 | 0.970609 |
| 共享6方向、epsilon=.001的legacy RMSE | 0.819084 | 0.975687 |
| 共享6方向各epsilon符号一致数 | 6/6、6/6 | 6/6、6/6 |

全64的平均幅值误差略降，但在预先共享的6个stable方向上，响应校准臂的幅值误差反而增加。符号相同不代表幅值正确；全体指标改善也不能代替共享子集或原状态确认判据。

因此，本轮没有取得将该校准组件升级接入Actor的新增确认依据。它说明“状态响应normalized MSE降低”与“任务score梯度更贴近物理有限差分”需要分别检查。它不证明该损失普遍无效，也不确定幅值差异的唯一来源。

## 验证范围与成本

实际3个可信component文件读取、2次native WM构造、2次B8前向/16预测、2次uVJP；0 optimizer、Actor、原119f加载、MJB、环境/碰撞/物理或GPU。worker用时7.5733995秒，不包括源审、合成检查、独审或绘图成本。

保存全部8预测、两项score、8×20输入梯度、原物理64弦线/32分类、实际/legacy两投影、原8eps32逐项算术界及raw RNG。独审从保存数组重做score算术、物理floor、分类、F64投影与误差汇总，并核原始RNG、输入、源码/输出归档和原确认decision。它没有重新运行NN、native Log或AD；AD正确性及权重/normalizer不变仍保留运行源码/摘要证据的范围。

输入仍为直接捕获的snapshot_x8，分支PRE58并未独立直接保存；使用primitive/INTEGRATION/三baseline所支持的条件范围，whole graph scratch、真实Jacobian、HO力梯度及CHORD/Actor收益均不授予资格。当前已见状态/单缓存谱系/egoN0不改变。

## 下一项

先核验多步状态滚动。H8此前只做过源连续性检查，现解除park到源码准备：固定初始未校准A2及两final1000，在同两donor的全部286个重叠窗口上做三模型H8状态误差描述。每步x与dynamics278内state前缀同时更新为上一预测，沿原8个已记录控制，不teacher-force状态、不clamp预测。先实际绑定xi和所有298相邻state位链/300个四子步held控制，随后才允许拼窗口和前向。

这仍是已见donor上的探索诊断，不作新blind确认、精度门槛或学习加速结论；没有新的训练或物理安排。物理可行参考、接触扳手支持与统一任务对照等原目标继续保留。

## 证据入口

- [实际任务梯度报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_component_task_gradient_CPU_v1/report.json)，SHA `f27f257defb29b95532b1cc1b7faa26b7b3144cf73a38f6fd2da2c8a2cca9960`
- [独立27项报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_component_task_gradient_saved_independent_v1/report.json)，SHA `362ea1b688f0374f3653711952b2364e2960638b4a618035d7cc882d9bc3f646`
- [全64与共享物理子集图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_component_task_gradient_figures_v1/all64_task_score_gradient_comparison.png)
- [原确认失败结果](D:/workspace/supermimic-full151-observation-study/docs/full151_component_confirmation_findings_2026-11-06.md)
- [三模型H8源码提案](D:/workspace/supermimic-full151-observation-study/docs/full151_component_H8_three_model_rollout_proposal_v1.md)
