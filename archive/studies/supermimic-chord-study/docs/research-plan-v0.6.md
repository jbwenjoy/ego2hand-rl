# 研究方案 v0.6：确认记录的接触差异，推进有限响应校准

2026-11-06。Goal保持active，按用户确认继续受控上游实验。v0.5及历史实验全部保留。

全部152恢复分支的608条solver PRE接触汇总与独立复算已完成，22/22通过。新增结果仍是部件诊断；原物理参考几何和三臂完整任务均未达标。

## 新证据及判断

2831条生成接触全部保留：HO1140、手部自接触475、other1216。按保存raw约束槽/type/world证据检查，2717条原decoded Fn>1e-6 N；这不构成完整原始力重解码。38条valid-empty样本及所有空类别的全局间隙仍UNKNOWN。

A0四个快照各76条PRE均有HO候选。A2/k0仅38/76有HO；A2/k49、99、149均0/76。k49仅记录手部自接触；k99/149各有608条物体–terrain候选及合格原正Fn证据。后三组48个物体p/R编码平坦端点中，42个实际控制改变，而固定A2 WM仍预测非零物体响应。

这支持把世界模型的有限动作响应单独校准。它不确定唯一原因，也不将无保存候选升级为全局间隙、把有限响应升级为真实Jacobian。先验P的2处同几何漏检、完整1808行的44处原生HO>3mm、452区间8.20mm位置向量RMS及baseline偏差继续保留；没有用本次局部记录重判这些结果。

## 选定的一个候选

从同一固定A2 final2000建立两个新的WM-only可训练副本，固定physical xi、所有normalizer及初始参数。均只拟合原152 restored transitions /64 pairs，排除300条donor。采用fullbatch152、1000更新、CPU FP32、AdamW lr1e-4/weight_decay0/clip1，不搜索超参、不挑最佳checkpoint。

对照沿用native52点式状态损失；方法在同一次152-row前向上增加lambda1的共同PRE52有限响应误差，除以固定native std/error_floor以及实际控制差/(1 rad)。保留全部clipped/asymmetric pairs。共同PRE的旋转Log chart统一两侧基准；native so3_log实际经mat_to_quat/quat_to_rotvec，后续guard须按该实现检查距pi的.01rad margin及finite值/梯度，raw trace仅诊断。保留native内部既有归一化，不额外投影R，不把它说成acos(trace)实现或旧rawR9描述的替代。

新artifact schema隔离校准参数/optimizer/norm/data/init指纹；不改119f frozen wrapper或旧105健康产物。当前仅源码草案和方法proposal，正式run协议、实际加载、优化与结果均待执行。

## 确认数据与下一项操作

1. 收口最小WM-only adapter、新物理worker源码及必要数学检查，Root冻结正式recipe、输入指纹和评价式。
2. 固定新扰动相位 `[24,74,124,148]`，沿用8 snapshot/152 branch/4 substep及原容量、时钟、三baseline门。新worker先discovery1200步，再审其真实8相位inventory，随后serialfull1808步；不复制旧标签继承隐藏状态资格。
3. 封存新增扰动响应，两个校准臂final1000冻结后统一解封全部64 pair。nominal状态已经看过，确认对象仅是预定相位上的新增受控扰动响应；仍是同缓存谱系、单seed，不称独立状态/轨迹或ego泛化。
4. 用全部64 pair的新归一化响应误差与全部152的点式损失比较；以响应误差下降且点式损失不增加作为局部候选确认条件，同时报告全部单位块、失败及undefined。不能按接触类别、旧stable9或当前误差筛选。

新的长训练若启动，仍按用户要求在两臂checkpoint100正常保存、健康核验并给出ETA后暂停Goal。当前没有新训练或后台物理；源码草案准备不触发这个暂停条件。H8保持parked。

## 原目标继续保留

部件校准不能替代物理可行参考；后续仍须修复接触/动力学可行性、建立共同真实学习初态与统一任务对照，再评估CHORD支持目标的收益。无机器人标签的ego输入/尺度/坐标与物体谱系、多轨迹共享与留出隔离、真实视频接入和硬件契约继续作为原目标。真实ego数量仍0，不声称已完成无本体重定向或学习加速。

- [全608接触结果与证据](D:/workspace/supermimic-full151-observation-study/docs/full151_saved_contact_regime608_findings_2026-11-06.md)
- [共同PRE有限响应校准proposal](D:/workspace/supermimic-full151-observation-study/docs/full151_WM_common_pre_response_calibration_proposal_v1.md)
- [上一版方案](D:/workspace/supermimic-chord-study/docs/research-plan-v0.5.md)
