# 三方法工程验证与 A2 首检查点交接（2026-11-05）

A1L、P_ONLY、A2 的新 schema2 同源短训练均完成，保存健康检查118/118、136/136、136/136，原始四模块初始化配对75/75。三次固定 final40 全3秒物理回放的保存独审574/574通过；任务门槛分别10/13、9/13、7/13，三组任务均失败，A2掉落且出现非手物体接触。目前没有接触先验收益证据。

实际材料的固定302项LP中，H123和H12345均151/151有条件可行；这是理想接触点与框架下的所需扳手分解，未证明实际接触、手部可达性、力幅上限或连续动力学可行。旧mu.1表示锥23/151结果仍保留，不能把其差异归因于增加手指。

Root一次启动 A2 的2000iter从零训练；共同协议也冻结A1L/P_ONLY配置，但未启动或加入后台队列。新v3入口仅优化元数据pin查找，训练仍使用实测三方法smoke中的 learning_v2/prior_hook/controller/runtime。完整预算为648192真实转移、WM4000/failure4000/Actor1800/critic7200。

首个 `model_100.pt` 健康审计116/116通过：40192真实转移、WM200/failure200，Actor/critic/prior政策更新仍为零；这是WM-only前缀，不能据此声称完整A2政策训练或任务已合格。最终容量与最终checkpoint尚待完成后检查。

四PID链 `[15644, 41436, 38532, 33452]` 通过创建时间、父子关系、命令、解释器和冻结source身份复核，后台worker为 `33452`。首次记录与当前CIM身份一致；运行仍由原有soft7100s/hard7200s watchdog管理，无自动重试。

以实际checkpoint100文件mtime `2026-11-04T19:45:46.416286+00:00` 为锚，预计北京时间 **2026-11-05T04:21:08.834340+08:00** 完成；启发式范围 **2026-11-05T04:13:48.594124+08:00 至 2026-11-05T04:32:19.194664+08:00**。估时使用本次前100步末50差分与同方法A2 smoke去前3次后的policy开销，不把native局部计时加到worker启动时间；范围不是置信区间或保证，后续策略、reset和机器负载可能改变速度。

用户已授权首健康checkpoint及ETA之后暂停Goal、保留后台job。此文与状态文件在调用暂停工具之前写入，实际Goal状态以随后工具返回为准；暂停不等于研究完成或停止此A2进程。

[首检查点健康](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_primary_A2_ckpt100_health_v1/report.json) · [实时身份与ETA交接](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A2_primary100_live_pause_v1/report.json) · [完整ETA](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A2_primary100_live_pause_v1/ETA.json) · [三方法回放保存独审](D:/workspace/supermimic-full151-observation-study/outputs/full151_v2_three_arm_smoke_policy_saved_review_v2/report.json) · [冻结A2配置](D:/workspace/supermimic-full151-observation-study/configs/full151-v3-primary2000-A2-seed0-v1.json)

恢复Goal时先读取原worker/watchdog终态与已有最终产物，禁止因观察超时重启。对固定final2000做独立健康/身份/完整3秒回放；随后执行已冻结的A1L/P_ONLY同预算实验并核原始初始化，再比较完整2000结果。仍需接触/物理梯度校验、多轨迹共享和独立ego验证。无指定ego数据，独立ego N=0；真机子版本及观测/控制契约在硬件部署前核对。
