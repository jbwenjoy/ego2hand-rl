# 研究方案 v0.14：首个冲量一致性配对训练与暂停交接

2026-11-06。已完成实际初始native三批152前向的位级验证与独立10/10；新recipe冻结，匹配的公共损失/一致性两臂实际启动并保存首个100次检查点，独立健康5/5通过，ETA已记录。按用户明确约定暂停Goal。本轮1000次已在记录时完成，实际剩余0；首100次ETA保留作历史测量。

两臂同WM/零14头/context/norm/152顺序、公共点/mode/impulse损失和AdamW1000预算，唯一科学差异为跨目标一致性系数0/1。r_saved仅作监督标签，不能进入WM输入；不强迫原始观测缺陷为零。无新增物理、Actor或GPU训练。训练健康不证明方法收益。

恢复后先读取现有进程/终态，不重启。对固定1000检查点另做独立saved健康，再预注册组件预测/冲量/mode/一致性与旧self15诊断，报告各项取舍。随后回到可行参考/接触支持、真实共同warmup初态和同预算任务对照。已见缓存/fresh仅探索性使用，不重新宣称blind或跨轨迹泛化。

原参考与旧任务FAIL、有限响应确认FAIL、H8取舍、whole-hidden未知均保留。单缓存谱系、ego N=0；共享多轨迹G2–G8、heldout009/010和硬件子版本/观测契约仍待完成，完整目标尚未完成。

- [本轮证据与限制](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_checkpoint100_findings_2026-11-06.md)
- [首100交接](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_impulse_checkpoint100_pause_handoff_v1.json)
- [上一版](D:/workspace/supermimic-chord-study/docs/research-plan-v0.13.md)
