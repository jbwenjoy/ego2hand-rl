# 研究方案 v0.7：校准已保存，按首检查点条件暂停

2026-11-06。首个100更新的两臂新component checkpoint独立读回29/29通过，ETA已记录，按用户约定暂停Goal。任务全目标保持不变。

新相位discovery1200独审77项、新full1808/152分支工程独审540项完成，确认响应77文件封存。两臂共同PRE52有限响应校准使用完全相同的原152/64数据、初态、normalizer和1000更新预算；总worker约33秒已结束，无后台作业。当前剩余时间0，确认收益尚未评价。

首100检查点的RNG原值一致。初版reader27/29误用跨序列化pickle摘要，失败保留；新版仅换为原始字段/字节比较，其余27门、训练及数据不变。final1000目前仅运行方健康记录，独立读回待恢复后执行。

恢复Goal后依次执行：

1. 独立验证两臂final1000的参数、Adam实际更新数、冻结统计、来源与完整性。
2. 满足原封存条件后创建单独解封记录，以冻结recipe上的全部64pair/152端点评价；响应MSE下降且pointwise loss不增加才取得局部候选确认，报告所有原单位块及失败，不调参/选checkpoint/筛接触类别。
3. 根据实测结果继续模型响应修正与物理可行参考修复，再推进共同真实初态的CHORD任务对照、多轨迹/真实ego接入和硬件契约。

本轮仍是固定腕、已见缓存谱系及父模型seed0的局部部件实验，不授予全隐藏状态、真Jacobian、HO力梯度、CHORD完整任务收益或真实ego泛化资格。原参考几何和完整三臂任务FAIL、warmup差异、代理漏检、旧5/9、egoN0等保留；H8保持parked。

- [本轮检查点、ETA及证据](D:/workspace/supermimic-full151-observation-study/docs/full151_common_pre_calibration_checkpoint100_findings_2026-11-06.md)
- [冻结recipe](D:/workspace/supermimic-full151-observation-study/protocols/full151105_WM_common_pre_calibration_recipe_v1.json)
- [上一版方案](D:/workspace/supermimic-chord-study/docs/research-plan-v0.6.md)
