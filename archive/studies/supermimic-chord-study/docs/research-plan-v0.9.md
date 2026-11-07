# 研究方案 v0.9：保留两项取舍，检查多步模型行为

2026-11-06。Goal active，无新长训练或后台物理。原完整目标与失败证据保留。

已完成两个固定组件检查：

- 原确认评价独审24/24：normalized有限响应MSE降低22.603%，但平均点式状态损失增加10.180%，位置向量RMS从0.675mm增至1.060mm。原确认条件未通过。
- 后续探索性任务score梯度独审27/27：32物理方向中6 stable、25 below、1 unstable；两臂每epsilon均6/6符号一致。全64实际口径RMSE略降（0.49349→0.47102），共享6方向的幅值误差反增（.002时0.81326→0.97061，.001时0.81908→0.97569）。不扩大N、不覆盖原确认失败，不授予真实物理/HO/Actor梯度资格。

下一步解除H8的park到源码准备，按固定数据作三模型探索性滚动诊断：

1. 比较initial_component中未校准A2、pointwise final1000、pointwise+response final1000。初始模型为独立角色，严格加载其已验证WM字典；不伪造final1000 payload或旧105资格。
2. 原452组按(path,mode=donor,k)选两条150步donor，不能取数组前300行。运行前重新核300个u四子步位一致、298个完整58维target→next-pre位链，以及xi46绑定。
3. 两path各143个起点k0..142，共286个相依窗口；每窗口8个控制至k+7、终点观测<=150，无u150。每h用上一预测同时替换x及dynamics278内state，控制按原记录固定；不teacher-force状态、投影/clamp预测或更新normalizer。
4. 预算为3个可信component文件读取、3次native构造、24次B286前向/6864预测，0AD/optimizer/119f/MJB/Actor/新物理。每h报告全部窗口的六块原单位误差，R9仍是原矩阵量，不混合58单位、不新设准确性门或CI。

该步骤检验单步校准后的多步行为与相对未校准模型的代价；没有独立新状态/轨迹或ego泛化资格，不能唯一归因为遗忘。它与固定teacher控制的条件有关，不等于闭环Actor imagination或策略收益。

H8执行器当前仅source draft，尚未拼真实窗口或运行新模型。后续根据完整证据选择物理可行参考/模型结构与接触支持的改进，再推进共同真实学习初态的任务对照和多轨迹/真实ego/硬件契约。known-rough参考、三臂任务FAIL、warmup差异、代理漏检、旧5/9、graph scratch未知与egoN0均保留。新的长训练若启动，仍按首个健康checkpoint与ETA的约定暂停。

- [固定组件确认结果](D:/workspace/supermimic-full151-observation-study/docs/full151_component_confirmation_findings_2026-11-06.md)
- [任务score梯度结果](D:/workspace/supermimic-full151-observation-study/docs/full151_component_task_gradient_findings_2026-11-06.md)
- [三模型H8提案](D:/workspace/supermimic-full151-observation-study/docs/full151_component_H8_three_model_rollout_proposal_v1.md)
- [上一版方案](D:/workspace/supermimic-chord-study/docs/research-plan-v0.8.md)
