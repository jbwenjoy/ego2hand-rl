# 研究方案 v0.8：保留校准取舍，核验任务梯度机制

2026-11-06。用户已恢复Goal，状态active。原完整研究目标保持，首检查点暂停记录保留。

两臂final1000独立读回29/29通过，确认数据按原条件解封。唯一实际确认及独立24项复算均完成：全64对normalized响应MSE从4.367903降至3.380625（−22.603%），全152行native平均点式损失从0.077276升至0.085142（+10.180%）。位置向量RMS由0.675mm增至1.060mm。原“响应降低且点式损失不增”确认条件未满足；不调整权重、筛样、选checkpoint或改变判据。

全部8组的响应MSE都降低，但点式损失只有2组降低。总响应损失下降主要来自40维手部q/qd项，不能用它代替12维物体响应、物体任务梯度或CHORD策略收益。全6块原单位误差和全8×6响应表、全152/64图及长表保留，normalized和原单位指标分别解释。

下一项固定同两个final1000，做一个探索性任务score梯度机制检查：

1. 沿原matched post20物体score与真实保存goal_p/R[k]，每臂一次B8前向、一次绝对控制u的VJP；参数和normalizer冻结，CPU FP32，无Actor、优化或新物理。
2. 全32物理方向/64步长保留。实际口径为δscore/actualδu[j]对照g·Δu/actualδu[j]；legacy口径为δscore/(2epsilon)对照g·d_eff，保持d_eff无量纲。裁剪/非对称标志另列。
3. 原三baseline范围/donor误差及两epsilon物理稳定性只作共享探索描述，不按模型结果选组，不与旧9组混池，不赋予噪声上界或真实Jacobian资格。
4. 结果用于判断状态响应MSE改善是否伴随物体任务score敏感度改善；它不是新的blind确认，也不覆盖本次确认失败或授予HO力梯度、完整隐藏状态与Actor收益资格。

当前仅该检查源码及必要合成数学在准备，实际任务gradient FW/AD尚未执行。没有新的长训练安排，H8保持parked；后续若再启动长训练，仍按首个健康checkpoint与ETA的约定暂停。

物理可行参考修复、接触扳手支持目标、共同真实学习初态的任务对照、多轨迹共享/留出、真实ego接入和硬件契约继续作为原目标。known-rough参考、完整三臂任务FAIL、warmup差异、代理漏检、原5/9、graph scratch未知和egoN0均保留。新分支的PRE58没有独立直接保存，维持原工程primitive/INTEGRATION/三baseline的条件范围。

- [确认结果与全部证据](D:/workspace/supermimic-full151-observation-study/docs/full151_component_confirmation_findings_2026-11-06.md)
- [探索性任务梯度字段可行性](D:/workspace/supermimic-full151-observation-study/docs/full151_final_components_task_gradient_source_feasibility_v1.md)
- [上一版方案](D:/workspace/supermimic-chord-study/docs/research-plan-v0.7.md)
