# 研究方案 v0.35：策略再采集与在线更新循环源码

2026-11-07，承接 [v0.34](research-plan-v0.34.md) 与完整 Goal。本回合为 progress：同一 GPU 实验继续运行；新增可执行的 collection/update 循环和最终 WM 诊断源码，并启动了训练完成后的原生策略评测等待进程。完整目标仍未完成。

## 当前实际运行与自动衔接

现有 GPU worker PID28104 / supervisor24420 沿原固定配方继续至 paired1000，未改数据、源码、batch、精度、优化器或物理参数。不能把本回合新增的 online 源码算成已经训练或采样。

[最终原生评测 supervisor](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_policy_evaluation_v1/execute_final_native_after_training_v1.py) 已启动，PID41220，工具 session42698。它核对现有 worker 的 PID、创建时间、父 PID 和命令，等待原训练 terminal0、固定 final1000 / WM1100 以及500和1000的保存健康，再绑定评测v2的四项最终输入。绑定后的元数据检查通过才执行两臂×双dt的四组原始两秒 native 评价；不重启训练、不选检查点、不修改物理，也不启动新学习实验。工具 session 丢失时先查 OS 进程及其 terminal，不重复启动。

等待进程的草稿元数据检查曾遇到 Windows PowerShell 与 pwsh 的日期 JSON 表达差异；在启动前改成显式 UTC ISO 和 aware datetime 比较后预检通过。已运行的最终 supervisor 源码保持不变。此问题未触发任何 NN、积分或训练重启。

## 新在线实验的可执行循环，尚未实际运行

[online runner](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_online_learning_v1/train_online_matched_v1.py) 及其协议准备/绑定脚本已经实现真实更新路径。源码顶部保留早期 draft 标签；最终交付清单以精确 SHA 标识待实测版本，不将其误称已验证 runtime。

新阶段固定 seed1734，以 offline final1000 的 shared WM、两臂 Actor/Critic/target、全部 normalizer、WM loss scales 和物理常量启动，第一次新建五个 AdamW。阶段内恢复才读回五个优化器、CPU/CUDA RNG、采样器、已采 pool、更新位置及历史日志字节前缀。旧 offline1000 配方与预算不变。

预声明三个 cycle，每个 cycle 先采16条 episode（每臂四 profile×双dt、每条最多20H），再完成100个 paired update。每个 cycle 的健康100更新模型供下一轮采集，因而代码确实形成策略→native→replay→WM/Actor/Critic→新策略的循环。每轮最多320H / 30720 native steps，三轮最多92160步；提前失持保留实际较短前缀，不补采或筛选，不能声称两臂实际样本数相同。

WM batch8 为累计 online4 + 原 TRAIN684 中2条 + 原三指 TRAIN153 中2条；两臂使用同累计 online pool、相同 batch8 行，各取自己的实际奖励列。Actor H2 使用四个 microbatch2，从完整 prealive 当前状态想象，不伪称每个起点都有原生两步真值。所有 normalizer/scales 均冻结，不用 MONITOR、CHECK、fresh 或 TEST 拟合。TD 对吸收失持和 reference timeout 断 bootstrap；采集预算截断与 partial 分开处理。

累计 pool 以 `(原pool SHA, 原row/episode ID)` 保留来源并重新编号全局身份，防止跨轮本地 episode0..15 混接。原 partial NaN、mask、两种奖励以及原 raw 保留；纯 NumPy 两个合成 pool 的身份/缺失值/奖励契约8/8通过，未加载 Torch、checkpoint 或原生模型。独立窄源码审阅未发现具体阻断；这不是实际学习、数据采集或物理成功的证据。

[collector v3 交付](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_online_collection_v1/SOURCE_DELIVERY_v3.json) 区分 exact offline1000 root 与本阶段自有 schema 的健康完整100更新模型，拒绝将首次10更新 checkpoint 用于下一轮采集。它保留原 Bridge、F64 控制历史、raw持久化、canonical接触标签、只读 norm/RNG 检查及独立输出。JSON/AST/元数据契约8/8。实际采集使用 CPU 子进程，训练仍在 CUDA 主进程，子进程环境不改父进程 CUDA 状态。

新实验第一次 cycle1/update10 保存、读回健康并给出 ETA 后退出；届时按用户既定规则暂停 Goal。恢复该点不会重新采集 cycle1，继续11→100，再以更新后的模型进入 cycle2。本回合没有启动这个新实验，因此尚无新暂停触发点。

## 固定最终 WM 的旧/新分布诊断

[WM诊断v2](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_final_WM_diagnostics_v1/compare_final_WM_v2.py) 待绑定最终四项训练文件。计划两次 CPU checkpoint 加载、两个 WM、四次前向及解码，共1804行：final模型预测新MON52、旧CHECK1472和旧fresh228，warm5000仅新增MON52预测；旧两组 warm 预测使用已保存结果。旧 normalized input 仅在 normalizer、物理常量和 geometry catalog 严格一致后复用。读取五个数据 NPZ，原 provider 另读取两个冻结几何 asset NPZ。

诊断保留旧CHECK四个未知和fresh十二个 partial，比较状态、已有真实22对J/Fn/coverage、1436条有限动作响应与原生精确零COM响应上的虚假预测；fresh只报告实际已有的状态和原两pad coverage，不发明22对扳手真值。该代码的元数据预检已通过，本回合未运行这四次模型前向。它不提供 infinitesimal native Jacobian、独立ego或持握收益证书，也不据此挑选更好的检查点。

## 仍需完成的工作

先完成现有 fixed1000 和原始四组策略物理评价，再执行固定最终 WM 诊断，随后实际验证上述 online 新实验与保存恢复。若实际失败，保留失败与所有终止分支，以实测决定后续改动。

更多独立ego、RGB手部重建及人手到机器人输入契约、真实碰撞资产和质量/COM/惯量/摩擦的校准与鲁棒性、源重力、完整61帧执行、多轨迹共享成本与匹配对照、真机观测/运控契约继续保留。目前仍无新增成功两秒持握、真实ego泛化或稳定CHORD收益证据。

本轮证据及具体进程入口：[里程碑v16](D:/workspace/ego-data-pilot-2026-11-07/outputs/goal_extension_tripod_online_loop_SOURCE_milestone_v16.json)。Goal 保持 active。
