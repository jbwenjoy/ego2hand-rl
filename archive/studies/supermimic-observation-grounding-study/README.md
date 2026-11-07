# 观测派生参考的物理 grounding

本工作区承接[几何重定向](D:/workspace/supermimic-observation-retarget-study/README.md)，先检查候选首帧在一个明确设计的静止保持干预中能否承载自由物体。输入来自受控缓存观测；未接入真实 ego 视频，也未读取上游机器人关节标签作为重定向输入。

## Frame0 rest-hold v1：工程完整，保持失败

2026-11-03 的唯一一次 Warp 尝试完整执行 60 个 5ms 步，保存 61 个积分状态和 60 个步前求解接触样本，无重试、重置或物体辅助。

| 固定诊断 | 实测 | 门槛结果 |
|---|---:|---|
| 物体最大位置变化 | 446.679685 mm | FAIL，门槛 20 mm |
| 有向笔轴最大角变化 | 0.352706° | PASS，门槛 20° |
| 实际关节限位超出 | 0 rad | PASS，容差 1 mrad |
| 生成接触最大穿透 | 0.504407 mm | PASS，门槛 3 mm |
| Sticky overflow | `[0]`，终末仅读一次 | PASS |

其余原定外力、非手支撑、固定实际控制量、时钟、有限性和覆盖门槛均通过。[原始评估](D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1/evaluation.json)保持 `rest_hold_pass=false`、`physical_reference_qualified=false`。`physics_inference_qualified=true`仅表示本次测量的工程/容量资格，不表示整条参考合格。

[独立 raw 重建](D:/workspace/supermimic-observation-grounding-study/outputs/independent_review_v1/report.json)已复算全部门槛、实际参数读回、初态/固定目标、接触分类与力/力矩符号、积分前后时序和进程身份，结论一致。1129 个逐采样布尔叶检查全部通过，另有 11 项 CPU 合成负例；它们不增加诊断的独立样本数。

候选 q0 只做 FP32 转换，物理初态不投影；关节和物体初速度明确设为零。这是新的设计干预，未声称从视频估计速度。PD 目标取固定候选 qref0，经同一 C2.020 一次处理后保持；本帧触发夹限的分量为 0，因此本次结果不能识别过滤器的处理效应。静止保持失败也不能单独证明整条运动轨迹不可行。

## 可核查的执行合同

- [冻结协议](D:/workspace/supermimic-observation-grounding-study/protocols/rest_hold_frame0_v1.json)、[CPU 审阅计划](D:/workspace/supermimic-observation-grounding-study/configs/rest-hold-frame0-v1.json)与[单次执行计划](D:/workspace/supermimic-observation-grounding-study/configs/rest-hold-frame0-authorized-v1.json)分别保存。源和输入 SHA、36 项 CPU 检查见[READY](D:/workspace/supermimic-observation-grounding-study/outputs/READY.json)及[CPU v3](D:/workspace/supermimic-observation-grounding-study/outputs/CPU_contract_v3/report.json)。
- 保持原 MJB，`put_model` 后复制实际八项物理参数，`set_const_0(restore=True)` 后仅做一次 startup forward；每步只执行一次 Warp step，不额外 forward。真正读回的参数保存于[actual_context_readbacks.npz](D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1/actual_context_readbacks.npz)，期望参数单列。
- [raw_states.npz](D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1/raw_states.npz)的 q/qd/时间属于积分后状态；[solver_samples.json](D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1/solver_samples.json)的接触、执行器力与物体求解姿态属于积分前状态，显式标出 pre/post index。Startup 样本单列；最后一个力样本是 0.295s，未刷新 0.3s 的接触几何。
- [raw_contacts.jsonl](D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1/raw_contacts.jsonl)保留 geom 对、世界接触点/坐标架和六维求解力；它们是本次仿真测量，不成为视频接触真值标签。
- [容量证据](D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1/capacity_audit.json)在第 60 步后读 sticky 标志一次，无清零；192 contacts / 4096 constraints 均未触顶。

实际 worker PID 42728，OS 创建时刻 `2026-11-03T04:34:56.1099330Z`。Worker 墙钟约 18.223 秒，watchdog 范围约 18.968 秒，二者嵌套；不相加为 GPU 计算时间。[Watchdog](D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1_watchdog/state.json)保留命令、身份、exit0 与 45 秒上限。该新 N=1 诊断不并入三臂 27 次确认评估。

## 同钟接触力诊断已完成

[CPU 诊断](D:/workspace/supermimic-observation-grounding-study/outputs/rest_contact_diagnostic_v1/INTERPRETATION.txt)从原始六维力重建合力与关于物体 COM 的力矩，计入直接接触力矩。最初三个 5ms 求解区间只有 finger2-link3 一处正力接触，向上力依次为 17.359/14.715/12.694mN；物体重量约 294.300mN，最高向上支撑仅 5.90%。从 15ms 的求解样本起，余下 57 个区间均无手物接触。该首帧几何接触对与实际求解接触吻合；本次没有证据将它判为 ghost contact。

`ΣF + mg` 与 `m Δv / Δt` 的最大分量残差为 5.50e-7N，测力积分的速度误差最高 1.64e-6m/s，支持时序/符号的一致性。手关节最大漂移 10.244mrad，峰值速度 0.06062rad/s，无执行器力饱和，不能据此归因于唯一控制器机制。具体数值、原始文件 SHA 及线速度约定见[report.json](D:/workspace/supermimic-observation-grounding-study/outputs/rest_contact_diagnostic_v1/report.json)与[逐区间 CSV](D:/workspace/supermimic-observation-grounding-study/outputs/rest_contact_diagnostic_v1/intervals.csv)。

实测垂直下落 446.603mm，同 5ms 半隐式离散自由落体解析参照为 448.8075mm，二者差约 2.205mm，与短暂向上冲量一致。实测三维位移范数 446.680mm是另一个量；连续时间自由落体 441.450mm也单列，未混作同积分参照。此处没有新物理运行。

[全程诊断图](D:/workspace/supermimic-observation-grounding-study/outputs/rest_contact_diagnostic_v1/rest_diagnostic.png)与[最初30ms放大图](D:/workspace/supermimic-observation-grounding-study/outputs/rest_contact_diagnostic_v1/contact_onset_zoomed.png)均已实际查看并通过[图形 QA](D:/workspace/supermimic-observation-grounding-study/outputs/rest_contact_diagnostic_v1/visual_qa.json)。

下一步优先明确来自观测几何的接触意图假设和初始承载/PD 预载设计，保留当前失败 baseline，再单独冻结运动初速与整轨迹 grounding 条件。未知置信度不改成视频接触或力真值；当前不依据保持结果调整原验收门槛或扩大长训练。
