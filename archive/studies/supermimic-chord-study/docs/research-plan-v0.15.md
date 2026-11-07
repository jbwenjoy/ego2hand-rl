# 研究方案 v0.15：保留一致性负面结果，修补旧接触监督后再接任务

2026-11-06。Goal active。两组固定1000更新的最终健康5/5、实际3×152预测与独立11项保存复算已完成；另已完成缓存self15标签评价及独立审阅。不是任务、CHORD或真实ego收益资格。

当前cons1方法在ALL的point/mode/impulse/consistency四项误差均高于公共损失对照，完整13组均保留，不自动增迭代或调权。旧15未入公共loss，两final BCE较初态恶化、Brier改善并出现低估软惩罚偏差；对照成为全负分类。新的2mode不能替代旧15。标签是原15ms缓存传感器，不能当post20几何truth。

下一项应固定新的共同公共native15监督配方（canonical原loss定义/权重，source明确），再做同initial/context/norm/order/budget的cons0/1对照，首健康checkpoint+ETA按用户约定暂停。尚未启动该新训练。反事实/响应与H8、GPU精度安装和共同fresh-reset/策略起点仍须验证；之后才开展共同warmup/replay/Actor/Critic/RNG的新namespace任务实验，不能假105恢复或换norm坐标系。

原151参考/任务FAIL、finite-response确认FAIL、H8取舍、whole-hidden未知、单谱系/ego0、多轨迹与硬件合同均未解决。完整目标保持，不以离线拟合结束研究。

- [本轮结果/证据/限制](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_components_final1000_findings_2026-11-06.md)
- [机器可查交接](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_components_final1000_research_update_v1.json)
- [上一版](D:/workspace/supermimic-chord-study/docs/research-plan-v0.14.md)
