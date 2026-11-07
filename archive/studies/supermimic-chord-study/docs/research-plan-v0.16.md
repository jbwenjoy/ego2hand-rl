# 研究方案 v0.16：共同native15监督已训练，首100健康后交接

2026-11-06。新的两臂共同native15监督实验已实际启动，配对100 checkpoint与新独立健康5/5、ETA已就绪，按用户约定暂停Goal。记录时两组已完成固定1000更新，实际剩余0；最终1000独审/效果评价仍待恢复后进行。

两臂同初始WM/零14头/context/norm/152顺序/1000预算；公共native_pair=mean152(point_row+.1*sum15 BCE)FP32，再+modeFP32+impF64，唯一方法差异仍cons0/1。标签仅为旧native15ms缓存sensor，不能视为post20几何或人类contact。checkpoint健康不能当准确性或CHORD收益。

恢复后先观察既有worker/终态，禁止重启消耗同一协议；对固定1000检查点做新saved独审和全152/13cohort预测/自15评价，保留既有无自15实验的负面结果。之后需response/H8、GPU精度/原norm安装、一次共同collect/replay/Actor/Critic/RNG及fresh-reset条件起点，使用新namespace，不冒105完整恢复。

原参考/任务FAIL、有限响应确认FAIL、旧H8/warmup差异、wholehidden未知、多轨迹/ego0/硬件合同仍未解决，完整目标保持。

- [本轮证据与限制](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_public_self15_checkpoint100_findings_2026-11-06.md)
- [交接](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_public_self15_checkpoint100_pause_handoff_v1.json)
- [上一版](D:/workspace/supermimic-chord-study/docs/research-plan-v0.15.md)
