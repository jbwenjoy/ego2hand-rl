# 新schema2三方法完整同预算结果（2026-11-05）

三方法固定final2000现已全部实际训练、保存并各完成一次完整3秒物理评价。共同seed0、同一151状态参考/150控制区间、actual8 context、105 EMA/C2控制器、498个共同源pin、128worlds/H8/优化器预算，四模块原始初始化逐参数字节一致。每组实际648192真实转移、WM4000/failure4000/Actor1800/critic7200，未warmstart或按质量选checkpoint。

| fixedfinal2000 | 原门槛通过 | 位置151最大误差 mm | 方向151最大误差 ° | 所有类别最大穿透 mm | 目标最大速率 rad/s | 掉落 |
|---|---:|---:|---:|---:|---:|---|
| A1L：无先验 | 7/13 | 566.7449 | 156.3288 | 23.3065 | 17.4890 | 是 |
| P_ONLY：穿透项 | 9/13 | 29.5693 | 33.4332 | 7.0256 | 18.9496 | 否 |
| A2：同穿透项+support | 7/13 | 553.9911 | 150.4940 | 28.4029 | 17.0143 | 是 |

三方法均不满足全部原门槛。P_ONLY在本受控样本中未掉落、没有非手物体接触，但位置>20mm、方向>20°、穿透>3mm、目标速率>15rad/s，任务仍失败。A1L/A2额外违反掉落与无非手物体接触门槛。不将门槛个数当加权总分，不把某项误差较小写成总体收益。

保存健康分别178/178、196/196、196/196；独立生命周期分别32/32、30/30、30/30；三次物理评价保存独审均188/188。每次固定150controls/600solver行/601状态、零优化器更新、Actor与normalizer不变、无物体辅助、不因质量失败提前终止。各原始质量失败与旧报告均保留。原始13门槛、全部峰值和完整曲线再作18/18汇总检查，图已由agent及Root实际查看。

[汇总JSON与pins](D:/workspace/supermimic-full151-observation-study/outputs/full151_schema2_three_arm_primary2000_summary_v1/comparison.json) · [三行指标CSV](D:/workspace/supermimic-full151-observation-study/outputs/full151_schema2_three_arm_primary2000_summary_v1/methods.csv) · [原13门槛CSV](D:/workspace/supermimic-full151-observation-study/outputs/full151_schema2_three_arm_primary2000_summary_v1/original13_gates.csv) · [完整曲线图](D:/workspace/supermimic-full151-observation-study/outputs/full151_schema2_three_arm_primary2000_summary_v1/tracking_original_gates.png)

## 对方案的含义

这是一个seed、一个来源的描述性消融，独立ego N=0；旧schema1 A1L结果不混入新三臂配对。当前没有成功任务或真实ego泛化证据。在此精确配置中，只用P保住物体，而额外加入support的A2掉落，足以要求诊断该support代理与真实接触响应的关系，不能据此外推CHORD普遍无效或P_ONLY普遍优越。

共享输入仍是knownrough SCP8候选：先前几何门失败，声明surface cue并非已观测人类接触。实际材料H123/H12345各151帧的条件扳手分解不包含实际接触存在、手部可达性、法向力幅、执行器或连续动力学约束；不能因静态分解可行就升级参考的物理资格。

本轮不继续扫support权重、换目标、挑更好checkpoint或追加seed来掩盖这些结果。先完成已提出的同Warp控制/接触诊断：真实105快照和warmstart恢复、三baseline重复、完整时钟和容量host OR账本资格，之后才比较有限控制差分与同标量的WM AD。预计152分支/1808episode-substeps的完整设计保留，phase0恢复smoke只是必要工程前置，不替代完整诊断。

新schema2两份WM的严格CPU加载/各一次forward和u AD已经15/15工程检查，但AD使用混单位sum验证连接，不是任务/物理梯度；self15也不是HO力head。新transaction105的42项CPU surrogate检查没有执行Native恢复。实际native probe尚未运行；未知foreign内部状态或closure必须先有明确源绑定审阅，不能以类型身份或CPU绿色检查替代恢复资格。

[WM接口工程证据](D:/workspace/supermimic-full151-observation-study/outputs/full151_schema2_pair_WM_loader_CPU_v1/report.json) · [事务组件CPU证据](D:/workspace/supermimic-full151-observation-study/outputs/full151105_transaction_preparation_CPU_v1/report.json) · [完整同Warp草案](D:/workspace/supermimic-full151-observation-study/protocols/full151_same_warp_control_gradient_proposal_v1.json)

## 当前交接

P_ONLY于北京时间18:20:03.785正常结束，实际worker wall2106.697s，原ETA点预测晚31.393s；原四PIDfresh CIM确认终态不在。三组后台训练已结束，无重启或新训练队列。Goal保持active，继续准备和实际验证上述物理探针；研究未完成，真实视频接入与真机子版本/观测控制契约仍按原Goal保留。
