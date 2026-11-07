# 研究方案 v0.19：纯native监督对照已训练，按首checkpoint条件暂停

2026-11-06。按用户约定，新增native52+self15-only对照的首100 checkpoint独立健康校验4/4通过，单臂ETA已保存，本Goal暂停。检查时同一已授权后台worker已经完成固定1000更新，producer总耗时19.34秒，剩余训练时间0；终态1000的独立读回和评价尚未进行。

新对照与既有两份public+mode/J/K及+consistency模型共用原initial、原152样本与缓存15标签、FP32逐row损失、AdamW和1000更新预算。保留同wrapper及16个参数组成员，实际14个WM梯度/Adam状态更新；辅助head的2个梯度为None、没有Adam状态、原positive-zero权重保持。只优化native_pair，不计算mode、J/K或consistency损失。首training FW兼初始函数parity，没有额外validation FW；unused第二copy为0 FW。没有Actor、Env、物理或GPU运行。

准备证据为reader2/2、trainer3/3与最终SOURCE10/10；真实100健康检查加载原ea051、新initial及100 checkpoint三文件，并复核一个缓存标签JSON。100健康不能替代1000终态健康或准确性评价。原100时ETA是包含20秒保守finalization余量的计划值；交接以已观察到的终态、19.34秒实际总时长和剩余0为准，不以单臂与旧两臂worker用时宣称加速。

恢复Goal后依次推进：

1. 独立验证新native-only固定AFTER1000序列化文件，检查14/16分区、PRE/AFTER时相、raw norm/RNG/context、预算和标签身份，再做同152/13cohort的公共native52+self15指标比较。各臂自己的优化total不作跨臂总分。
2. 在新namespace将native-only加入已固定fresh/H8/动作敏感度比较；可以复用经审的旧模型保存预测，但须锁同数据、控制、精度、人口与实际输出来源。保留v0.18的多步改善和局部动作敏感度取舍，不据结果调λ、筛组或改旧文件。
3. 继续逐branch PRE58资格、GPU精度及固定norm安装、共同collect/replay/Actor/Critic/RNG起点任务验证，以及接触可行性/力容量、多轨迹、真实ego接入与硬件契约。原参考/物理任务FAIL、旧响应确认FAIL、observer/whole-hidden未知、人类接触未知与ego N=0均保留。

本次首100暂停条件已经履行；用户恢复Goal后不得再次因这个历史checkpoint自动暂停。只有后续另一项新训练才重新适用首checkpoint条件。完整研究目标未完成。

- [本次checkpoint与ETA交接](D:/workspace/supermimic-full151-observation-study/outputs/full151105_native_public_only_checkpoint100_pause_handoff_v1.json)
- [首100独立健康报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_native_public_only_checkpoint100_saved_independent_v1/report.json)
- [上一版与已完成评价](D:/workspace/supermimic-chord-study/docs/research-plan-v0.18.md)
