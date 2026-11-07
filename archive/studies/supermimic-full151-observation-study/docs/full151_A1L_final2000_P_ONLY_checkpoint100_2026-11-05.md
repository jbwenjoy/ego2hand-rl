# A1L 完整2000、P_ONLY 首检查点与梯度诊断准备（2026-11-05）

上一Goal turn是实际progress：A2完整健康与物理评价、新A1L首100、同Warp草案与只读ABI25。本轮先核原A1L终态与fresh四PID CIM，没有重启A1L。A1L于北京时间14:11:54正常完成2000iter，worker wall 1703.528s，原ETA点预测比实际晚30.386901s；只描述本次估时。

## 新schema2固定final2000结果

A1L健康178/178、生命周期32/32、完整3秒保存独审188/188通过；498源pin与ZIP、actual8/controller/固定21个检查点、完整648192真实转移及WM4000/failure4000/Actor1800/critic7200均核对。A1L始终prior=None、无objective或prior JSONL、计数0。最终容量3222次观测/3199pre-reset，overflow/unknown均0；原启动记录错误与恢复来源仍保留，四PID终态独立核不在。

Root对预声明fixedfinal2000仅做一次150controls/600solver/601states回放，无Actor/norm更新、无物体辅助、无质量提前结束。A1L与A2均任务失败，失败项一致：位置、方向、穿透、非手接触、目标速率和掉落。

| 方法 / fixedfinal2000 | 通过门槛 | 位置151最大误差 mm | 方向151最大误差 ° | 所有类别最大穿透 mm | 目标最大速率 rad/s |
|---|---:|---:|---:|---:|---:|
| A1L | 7/13 | 566.7449 | 156.3288 | 23.3065 | 17.4890 |
| A2 | 7/13 | 553.9911 | 150.4940 | 28.4029 | 17.0143 |

这两组为同源、同seed0、同原始四模块参数、同预算的完整训练；P_ONLY final2000仍未完成，故三方法完整对照尚未闭合。二者都未达标，不能因某项误差略低就称先验有效，也不能从单条训练参考、单seed外推泛化。旧schema1实验没有进入新三方法配对，不替代本轮基线。

## P_ONLY：第三个冻结预算已启动

Root通过参数化launcher一次启动P_ONLY，UTC09:44:55.472（北京时间17:44:55）。本次预先计算固定SHA256与source pins，再Start-Process；launch.json与initial_CIM原生记录完成，没有记录恢复或重复启动。四PID为30280→33496→22700→37396。三组primary和对应smoke原始四模块逐name/order/shape/dtype/bytes配对一致。

固定model100健康116/116，40192真实转移、101collect/100metrics、WM200/failure200，Actor/Critic/prior调用均0；P_ONLY objective已构造且metadata/spec/RNG一致，无policy JSONL，final_scope=False。这只证明WM-only工程前缀；最终政策更新、容量和任务仍需终态审计。

实时身份/估时37/37通过，预计北京时间 **2026-11-05T18:20:35.178763+08:00** 完成，启发式范围 **2026-11-05T18:13:29.571728+08:00–2026-11-05T18:31:23.589316+08:00**。以model100实际mtime 2026-11-05T09:46:26.229503+00:00 为锚，使用本次末50步平均0.626059s与同P_ONLY smoke去前三次policy均值0.465846s。范围不是置信区间或保证。

按用户已授权条件，保存本地交接后调用暂停工具，Goal状态以工具返回为准。P_ONLY原后台训练继续由7100/7200s watchdog管理，无后续队列或自动重试。恢复时先核37396原worker/watchdog/已有终态，禁止因观察超时重复启动；固定final2000健康、完整3秒评价与独立保存核验后，才汇总三方法最终表。

[A1L final健康178](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_primary_A1L_final2000_health_v1/report.json) · [A1L物理保存审188](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A1L_primary2000_policy_saved_review_v1/report.json) · [A1L生命周期32](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A1L_primary2000_saved_lifecycle_v1/report.json) · [P_ONLY首100健康116](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_primary_P_ONLY_ckpt100_health_v1/report.json) · [P_ONLY实时身份与ETA](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_P_ONLY_primary100_live_pause_v1/report.json)

## 梯度验证仍按物理范围分层

本轮已完成两项边界明确的工程工作。新schema2 A1L/A2 fixedfinal各用冻结119f loader实际严格加载一次，每个只在同一保存A0 phase0 state58/actualu20上做一次forward和一次u AD，15/15工程检查通过。内部共4次torch.load、2次CPU Actor构造但无Actorforward、2次被动MjModel读；无Env/Warp/CUDA初始化/物理API。权重、normalizer、输入和RNG不变，state58/self15/u梯度均有限。AD标量是混单位输出sum，仅用于连通性检查，不能充当native任务标量或精度证据。

新增隔离transaction105组件与CPU surrogate契约42/42通过，覆盖缺字段、旧104、phase149/terminal150、恢复不完整、外部容量账本及别名、closure、RNG、失败分支不重试等拒绝场景。Root已阅读源码：真实PenSimReader.memory将c1/c2都复制当前processed_action，故相等guard只用于真实捕获边界；想象advance_memory之后两者可不同，禁止套用该profile。opaque foreign仅记录身份，内部状态资格仍PENDING。上述42项不是实际Native恢复试验。

冻结old observed_capacity将observer挂在env上，不适合可恢复donor graph；future worker必须把容量host OR/UNKNOWN账本放在事务外，任何restore都不能清除此账本，旧source不改。真实105完整inventory、warmstart、三baseline、长时钟、sameWarp控制差分及HO测量均需实际资格。

WM的self15仅是手自碰撞，不是HO接触/力head。CPU WM内部u AD有限只能证明模型自身可微；不是物理Jacobian、HO力梯度、GPU AMP或任务收益证据。152分支/1808步的sameWarp方案仍为proposal，未执行。几何候选SCP、真实接触/力幅/执行器/连续动力学资格、多轨迹共享和独立ego数据仍未解决，独立ego N=0；真机子版本与观测/控制契约在部署前核对。

[schema2_pair_WM_actual_CPU_check](D:/workspace/supermimic-full151-observation-study/outputs/full151_schema2_pair_WM_loader_CPU_v1/report.json)

[transaction105_CPU_preparation](D:/workspace/supermimic-full151-observation-study/outputs/full151105_transaction_preparation_CPU_v1/report.json)
