# 原生 full151 从零学习与首 checkpoint 暂停记录

2026-11-04，用户选择尚无指定 ego 数据，继续上游转笔受控实验。本轮实现实际学习链路，运行一次 smoke40、一次固定 final40 的完整策略评估，并启动独立 from-scratch 2000 次迭代 A1L。没有加载旧 104 或 smoke 权重作为长训练初值，没有修改驱动/CUDA。

本轮有实际进展；整个探索目标尚未完成。151 帧参考的 SCP 几何失败与 A0 三项质量失败继续保留。当前只接通 A1L；full151 的独立接触意图/扳手绑定仍缺失，不能用空目标冒充 P_ONLY/A2。独立真实 ego 数据为 0，shared 多轨迹训练仍待推进。

## 实现与验证

新增隔离的 learning/training_capacity、独立 105 健康审计、policy_probe/evaluation 和训练/评估入口。原 controller/runtime/native/A0 已执行源保持原字节。默认 preflight 无 Torch/Warp；显式运行使用独立环境及带 OS 身份记录的有限 watchdog，无自动重试或 fallback。

CPU 合同为训练 57/57、策略评估 49/49、健康审计 64/64；这些 fixture 不代替实际 CUDA 证据。健康 checker 首版有一项负例选择了本来就为 True 的 mask，失败源与结果保留；v2 只修 fixture，生产审计器未改。实际训练之后另做独立 artifact 审计。

保存沿用原原子写入：native format3 payload → 新 controller105 → runtime/四模块初始 fingerprint → capacity prefix。记录四模块原始初始化参数，独立比较后续参数变化、有限张量、优化器次数、实际控制、H8 mask、replay 相位与容量。capacity observer 从 Env 构造结束后开始；构造/graph/startup 之前的整个进程容量和 API 次数仍为 UNKNOWN。

## smoke40 实测

冻结配置 `c70352465ddb1bd955026cb676f628eab6f86ca3c1bd692f243b6ba350c2d1be`，145 个 source/input pins。128 environments，buffer256、warmup64、window8、horizon8、UTD0.4、WM512×3、AMP、eager。40 iterations，前20次 WM-only。一次实际 GPU 执行正常结束，worker9648/launcher21288；保留 session42443 已 exit0，不再启动。

实测 20,992 条转移，WM80/failure80/Actor20/critic80 次更新；41 collect、40 metrics。四模块参数变化 L2 分别为 WM3.8347895、failure3.1279380、Actor1.2976422、critic2.5849417。worker59.833405s，native54.399843s。初始参数、105 final、来源/ZIP、控制和容量独审 **103/103 PASS**。全150相位确实被访问，phase149与自然末端均71条；另有82条物理提前终止，不能视作质量通过。全部相位覆盖来自真实 histogram，而非用总164 worldsteps推断。

容量前缀53记录、final54记录，50次 reset 前读取，raw OR0/unknown0。watchdog/OS 身份12/12与终态精确 PID 消失检查通过。最新 replay-window generation 是源码约束的保存摘要，未保存所有原始窗口 generation 张量，独审不扩张为完整窗口重建。

- [smoke 独立健康报告](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_smoke40_health_v1/report.json)
- [实际结果](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_smoke40_v1/result.json)
- [固定 final40 checkpoint](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_smoke40_v1/train/model_final.pt)，SHA `5252e58e6eddb1b8a54c848d884a58b4ec1ec5bc708c486f0c98c3db2558d20e`。

另有 source/生命周期/actual8/初始化独立审阅78/78通过。其首版76/78两项是审计假设错误：路径 pin 的 slash 表示不同；natural 与 physical 终止不能假设互斥并简单求和。v1保留，v2仅改审计逻辑，未改训练源或结果。[78项报告](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_smoke40_lifecycle_independent_v2/report.json)。

## 固定 final40 完整物理评估

新 worker 在 CUDA 初始化之前严格 CPU 加载真实 105 Actor155→20，保留原训练 root descriptor/controller signature。随后同一实际8字段/xi46、无噪声、无辅助力、无额外 reset，完整150次控制、600solver、601raw states、150原 Manager forward/reward capture。Actor/norm/checkpoint 前后不变，optimizer0、Runner实例0；质量失败不 early stop。一次运行39.654724s，worker41348/launcher32156，session53480 exit0；OS12/12通过。

| 固定完整回放指标 | A0 | smoke40 Actor | 原阈值 |
|---|---:|---:|---:|
| 151点最大位置误差 | 38.002625mm | 565.710011mm | ≤20mm |
| 151点最大有向笔轴误差 | 60.249328° | 170.399547° | ≤20° |
| 601点实际关节越界 | 0.0000505mrad | 6.380567mrad | ≤1mrad |
| 600solver最大穿透 | 7.941423mm | 16.433617mm | ≤3mm |
| 实际目标最大变化率 | 14.999902rad/s | 15.339008rad/s | ≤15rad/s |
| 总质量通过 | 10/13 | 6/13 | 全部13项 |

smoke40 的7项失败包括位置、笔轴、关节、穿透、物体非手接触、目标变化率、掉落。无外力、actualu逐步精确、clock、负载比、charge、capacity通过。策略变化率包括初始 ctrl→learned U0 与后续149次，共150个差分；startup1.908726rad/s，最大超限出现在后续更新。A0只有149个非零更新，其初始差为0，不影响表中最大值。

这是一条工程 smoke 的负例，不是最终2000次策略成绩，也不能证明最终收益或多种方法的因果差异。原质量阈值与全部失败保持原样，没有按结果修改控制器或选择更好的 checkpoint。

- [完整原始评估](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_policy_eval_smoke40_v1/evaluation.json)
- [严格 CPU Actor 加载 receipt](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_policy_eval_smoke40_v1/actor_load.json)

保存后独立 NumPy 审计完成178/179，不能标全PASS：13原gate、9连续指标与150步FP32 EMA/C2/cache/rawctrl重算均吻合，source152/ZIP、OS、capacity、训练root/checkpoint/health与Actor保存digest均核对。唯一未关闭项是额外断言“policy155前20列应逐步与同步raw q完全相等”失败，首帧相等，后续的观测变换/时相原因尚未查明。未判断为harness或工程bug，也未重新运行模型/物理；此疑点需在恢复后优先核查，不能隐去。现有84项checkpoint健康不包含这项独立观测同步判断。[保留的178/179报告](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_smoke40_eval_saved_independent_v1/report.json)。

## 长训练首个健康 checkpoint 与估时

独立 from-scratch primary2000 配置 SHA `fe2c7bd04a7e27af248c97f71d8aa13a5bbd98e1bfd2023ab7b93b30983adac9`，147 pins；2000 iterations、WM-only200、save100，其余共同预算不变。全程预计648192 real / WM4000 / failure4000 / Actor1800 / critic7200，当前只是预算，不冒称已完成。

隐藏后台 supervisor 于北京时间15:39:14启动，retained launcher16444、actual supervisor29188、worker launcher30948、actual worker34468。Root 实际 CIM 核对四层 PID/父子链、worker creationUTC与原watchdog记录、命令/config/source一致。当前尚未终态，不能宣称进程已退出或 final capacity 已通过。

首 `model_100.pt` SHA `8c0aed2746c29136cda69d6f54e319cc1d9d31e5315ad39dd16e26e7711d12ea`，**84/84 独立前缀健康通过**：40192 real / WM200 / failure200 / Actor0 / critic0。Actor/critic raw参数未变，policy NOT_RUN。101 collect、100 metrics；容量137前缀记录 OR0/unknown0。`final_iteration_scope=False`，此健康范围不是 Actor 质量、最终预算或最终容量证明。

估时锚点为 checkpoint100 的15:40:38。实时最后50个 WM-only iteration平均0.634619s；smoke H8 policy剔除前3次后平均0.246565s。按 `1900×实时基线 + 1800×H8策略成本 + 20×保存 + 20s未测开销`，从锚点约27.84分钟，预计 **2026-11-04 16:08（北京时间）**结束；经验范围 **16:03–16:17**。范围不是置信区间，负载/接触/reset/温度会改变速度。

按用户的明确条件：首 checkpoint 正常落盘、独立校验无异常并估时后，暂停 Goal；保留已授权的隐藏后台训练继续。暂停不等于整个研究完成，不自动启动其它方法或评估。

- [首 checkpoint 独立健康](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_primary_ckpt100_health_v1/report.json)
- [实时进程证据与 ETA](D:/workspace/supermimic-full151-observation-study/outputs/full151_primary_first_healthy_pause_v1/report.json)
- [完整 ETA](D:/workspace/supermimic-full151-observation-study/outputs/full151_primary_first_healthy_pause_v1/ETA.json)
- [长训练实时状态](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_primary2000_v1/runtime.json)

恢复后先读同一个后台 job 的状态及终态身份，勿重复启动；完成后审计固定 final2000，统一完整回放，再判断是否值得改参考/方法。full151 独立目标绑定、同预算 P_ONLY/A2、多种随机种子、多轨迹 shared 与真实 ego 验证仍需继续。
