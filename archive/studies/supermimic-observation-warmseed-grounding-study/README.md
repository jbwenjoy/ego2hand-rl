# Warmseed frame-0 rest diagnostic

2026-11-03最新：新增H123三指分支及同q静态预载实验，见文末。H123 baseline预载在原0.3s rest门槛下PASS；完整运动参考仍未取得资格。下文原A/B记录专指H23两指分支，原B仍未成立、未执行，未被新H123结果覆盖。

原H23分支保存一次零速静置检验。A（`QREF_C2`）完整执行，但物体位置、轴向和穿透三项均未通过。B（静力 PD 预载）没有成立的目标，未执行，不能记为 B 的任务失败。该H23候选、接触意图和物理参考均未获资格。

Canonical 固定为 [H23 baseline-derived 几何候选](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h23_seeded_baseline_v3/report.json)，其 [q0/物体姿态](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h23_seeded_baseline_v3/initialization_candidate.npz) SHA 为 `7e5c0da26ddadc16d3464e504b436d7dcc5254af37b35bd2460a8c9c7ffcdf2e`。候选来自 [受控 observation-only fixture](D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1/observations.npz)、H23 固定材料接触假设及单指解拼接的 warm start；没有使用上游机器人 q 标签。它不是已校准的真实 ego 数据。Midpoint-derived 解只保留为局部初值敏感性记录，不替换 canonical，也不增加独立样本数；几何优化收敛不等于材料位置/法向意图或静力平衡达成。

## A：唯一一次实际执行

按 [冻结协议](D:/workspace/supermimic-observation-warmseed-grounding-study/protocols/warmseed_rest_frame0_v1.json) 与 [单次授权计划](D:/workspace/supermimic-observation-warmseed-grounding-study/configs/warmseed-rest-qref-authorized-v1.json) 执行，诊断 N=1，不属于训练 seed 复制。协议原稿的 DRAFT 标签保留；实际授权由后者明确给出。初始物理 q、物体姿态分别由 canonical 一次 cast FP32；q 不做投影，腕固定为原模型默认值，关节速度和物体 twist 人为设为零，这些不是从视频测得的速度。

固定 qref0 经过同一 C2 `.020 rad` 投影一次后，actual u 在 60 个 5 ms 物理步保持不变；本次改变分量数为 0，不能据此推断 C2 的处理收益。完整保存 61 个积分状态、60 个 solver 样本及含 startup 的 92 条接触记录。外加力为零，没有 reset、额外支撑、后续状态重写或自动重试；接触力与 PD 力矩并非零。

| 固定质量门槛 | 实测最大值 | 结果 |
| --- | ---: | --- |
| 物体位置误差 ≤20 mm | 65.159069 mm | FAIL |
| 有向轴误差 ≤20° | 41.349539° | FAIL |
| 任意类别穿透 ≤3 mm | 3.871942 mm | FAIL |
| 实际关节越硬限 ≤1 mrad | 0 mrad | PASS |

[原始评估](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/evaluation.json) 的其余固定门槛通过。末步后只读一次 sticky overflow，bitmask=0，计数均未达容量、无初始化后 reset，因此本次窄容量门槛为 OK；这不证明求解器收敛或总体物理准确。接触/力矩来自各步 preintegration solver 时钟 0–0.295 s；积分状态覆盖 0–0.3 s。没有额外 forward 刷新末态接触，不能把末 solver 接触冒称 0.3 s 的新测量。

[Owner 完成记录](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/qref_owner_completion_v1/completion.json) 与 [Root 独立原始数据审计](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/qref_independent_review_v1/report.json) 均通过工程/身份/重算合同，物理保持结论仍是 FAIL。实际 worker PID `36172`，OS CreationDate `2026-11-03T05:25:37.2573190Z`，exit=0；终态核实无 owned/workspace Python 残留。嵌套成本分别为 worker **5.670619 s**、watchdog owned child **6.391000 s**、完整 exec 调用 **9.302296 s**，不能相加，也不是 GPU kernel busy 时间。

可复查 [worker](D:/workspace/supermimic-observation-warmseed-grounding-study/scripts/rest_hold_warp.py)、[监督器](D:/workspace/supermimic-observation-warmseed-grounding-study/scripts/run_rest_hold.py)、[runtime](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/runtime.json)、[容量 sidecar](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/capacity_audit.json)、[raw states](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/raw_states.npz)、[solver samples](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/solver_samples.json) 和 [raw contacts](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/raw_contacts.jsonl)。授权计划绑定全部源/输入 SHA；原实际 context 的 8 字段，在 copy 后与 `set_const_0` 后各核一次：[16 组实际 readbacks](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/actual_context_readbacks.npz)、[实际 applied 字段](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/actual_context_applied.npz) 与 [expected 字段](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/warmseed_rest_qref_v1/expected_context_fields.npz) 分开保存，不能把 expected 代称实测。

## 静力筛查与 B 的状态

[独立 q64 native cone 筛查](D:/workspace/supermimic-observation-grounding-study/outputs/warmseed_object_wrench_screen_v1/report.json) 使用两候选各自保存的实际 native 接触、材料和 18 条 pyramidal rays，不设任意法向力上限。两者仅平衡重力合力可行，但含 COM 力矩的完整六维平衡均不可行；canonical 在精确重力合力约束下，最小 COM 力矩无穷范数仍为 **4.526834 mNm**。报告另保存逐 ray 检验的分离证书，结论不是只有求解器的 infeasible 字符串。对应 [筛查源](D:/workspace/supermimic-observation-grounding-study/scripts/screen_warmseed_object_wrenches_cpu.py) 与 [native pyramid/LP 实现](D:/workspace/supermimic-observation-grounding-study/scripts/diagnose_rest_wrench_feasibility_cpu_v2.py) 均有 SHA 绑定。

[canonical q32 重建筛查](D:/workspace/supermimic-observation-contact-intent-study/outputs/seeded_preload_cpu_v1/report.json) 单列实际 FP32 初态合同：重新生成 3 个 native 接触后，六维物体 LP 仍不可行，不覆盖 q64 结果。保存了全体 body COM Jacobian、重力广义力、手接触映射、实际 PD 参数和 FP32 C2 边界；20 项非零 frictionloss 明确保留。由于物体前置条件已经失败，零 stiction 与 bounded-stiction 的 whole-hand LP 均未运行，也没有合法 λ 或预载见证可导出。

[B 目标状态文件](D:/workspace/supermimic-observation-contact-intent-study/outputs/seeded_preload_cpu_v1/designed_target_u0.json) 为 `BLOCKED_OBJECT_STATIC_WRENCH_INFEASIBLE`、`feasible=false`、`u0_rad=null`；[B 计划](D:/workspace/supermimic-observation-warmseed-grounding-study/configs/warmseed-rest-preload-DRAFT-v1.json) 保持未执行。静力 [源快照/矩阵/独立证据索引](D:/workspace/supermimic-observation-contact-intent-study/outputs/seeded_preload_cpu_v1/independent_links.json) 与 A 结果分开；[独立矩阵/符号审计](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/preload_independent_static_review_v1/report.json) 通过限定范围重算。未根据 A 的运动结果调整静力方程或选择目标。

## 同钟诊断与结论边界

[接触与净 wrench 诊断](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/qref_contact_diagnostic_v1/report.json)、[简报](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/qref_contact_diagnostic_v1/REPORT.txt)、[逐区间 CSV](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/qref_contact_diagnostic_v1/intervals.csv) 及 [图](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/qref_contact_diagnostic_v1/warmseed_rest_diagnostic.png) 只重算本次 raw。它们按相同 solver 时钟对齐接触合力/冲量与运动，COM 力矩包含力臂与 direct torque；不将线性平衡一致或旋转相关性写成唯一因果，也未宣称完成角动量 Euler 平衡资格。

该固定接触初态的精确静力平衡失败，不排除 20 mm/20° 有限窗口内先 settling 后形成其他接触，也不证明有动量的轨迹不可行。A 已直接检验的这一个零速、固定 qref 0.3 s 条件确实失败；当前尚无成立的 B 预载或合格物理参考。此前 [另一 q0 的约 447 mm rest 失败](D:/workspace/supermimic-observation-grounding-study/outputs/rest_hold_frame0_v1/evaluation.json) 只作背景，不并池、不构成同 q 控制对照；本次位置误差较小也不能盖过旋转和深度失败。这里没有建立全局平滑性、安全保证、真实 ego 转移、力标签或完整动态重定向资格。

## H123三指与静态预载：后续顺序探索

H123增加拇指材料意图，保留旧H23目标、固定物体与腕。baseline/midpoint两套几何初值分别保存，未选表现较好者替代canonical。两套六维物体扳手LP均可行，但仅代表不设任意法向上界的数学存在性。两次固定QREF实际保持均失败，随后仅canonical baseline导出合法全手静态PD预载；这是结果已知后的顺序探索，不追溯称为盲法或共同预注册比较。

| 0.3s条件，各N1 | 位移峰mm | 有向轴峰° | 任意穿透峰mm | 实际q越限峰mrad | rest |
| --- | ---: | ---: | ---: | ---: | --- |
| H123 baseline QREF | 10.293122 | 31.331353 | 2.183087 | 0.0000505 | FAIL |
| H123 midpoint QREF | 30.668902 | 42.818720 | 1.814682 | 0.0000505 | FAIL |
| H123 baseline静态PD预载 | 3.956322 | 16.287022 | 2.183087 | 0.626138 | PASS |

门槛维持20mm/20°/3mm/1mrad。三条均61状态/60 solver样本、sticky overflow0末步仅读一次、actual八参数copy后/set_const_0后两次exact读回。每条只有startup一次forward，之后60次step，无状态重写、reset、外力或额外支撑。baseline预载与baseline QREF的fullqpos/qvel/默认mocap初态逐位一致，19个固定控制目标分量不同；它们与物理q0的差异计数是另一口径。实际q没有投影，预载出现0.626mrad真实越限，不能写成完全不越限。

预载由[受限静力设计](D:/workspace/supermimic-observation-contact-intent-study/outputs/h123_preload_cpu_v1/baseline/designed_target_u0.json)与[55项独审](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_preload_independent_review_v2/report.json)绑定；C2处理前后逐位相同，clamp0，不能从本条件识别过滤效应。所选数学见证ζ=0不改变模型20项非零frictionloss。设计未强制软接触本构、排除关节限位反力，LP见证力不等于实际实现力。

实际[单次计划v2](D:/workspace/supermimic-observation-warmseed-grounding-study/configs/h123-rest-baseline-preload-authorized-v2.json) SHA `d94132eabb96ae5dadca32a4e3f861b21e5a24b7c672987637f100b881b4fedf`；[协议v2](D:/workspace/supermimic-observation-warmseed-grounding-study/protocols/h123_baseline_preload_rest_frame0_v2.json)只纠正选择时序措辞，未改数值设计/物理/门槛。worker PID3796，OS创建`2026-11-03T06:21:18.199063Z`，exit0。worker4.289236s、watchdog5.016s嵌套，GPU busy未知；唯一一次尝试，无重试，终态CIM无owned进程、GPU已释放。见[owner证据](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_preload_owner_execution_v1/completion.json)、[原始评估](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_rest_baseline_preload_v1/evaluation.json)和[Root独立raw重建](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_preload_independent_raw_review_v1/report.json)。

新[control-aware auditor](D:/workspace/supermimic-observation-warmseed-grounding-study/scripts/audit_rest_hold_controls_cpu.py)从hash绑定的设计读取预期u，而非信任raw的proposal header；原QREF审计器原样保留。实际同q QREF的CPU回归亦通过。回归v1误传plan_snapshot路径，被严格command_plan路径合同拒绝；v2传实际原计划路径通过，失败记录保留，未修改源/物理或放宽检查。

本次只支持这一固定首帧、零初速、0.3s有限窗的保持结果。其来源窗口是clip007[200,351)，不能迁移为新10clip[0,151)批次的初始化资格。完整151帧接触意图/速度一致性、动态跟踪、学习修正与真实ego仍待验证；不将N1静止PASS升级成力预测、连续安全或总体方法收益。

[同q只读比较](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_qref_preload_comparison_v1/REPORT.txt)确认两条60/60求解区间均有正力手物接触，向上力峰QREF/预载为0.317984/0.303463N，而COM总矩范数峰为8.50211/8.59658mNm，后者略升，并非全部wrench指标下降。实际执行器力用求解前q/qd和实际u的PD affine重建，最大误差约2.34e-7/2.10e-7Nm；错用积分后状态会达到9.389e-3/2.346e-3Nm。预载初始PD力矩接近静力见证，但随q/qd演化已偏离它，不能把固定u当固定力矩或把λ当实际力。该比较没有新增physics/LP/GPU。

更直接的[初始测力对照](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_qref_preload_comparison_v1/LP_witness_vs_measured_initial.json)：预载初始实际PD力矩与LP设计差仅1.375e-7Nm，但实际Σfn为3.465981N，LP为2.139425N；实际COM矩范数8.59658mNm，LP近0。这证明实际接触分配与静力见证不等价，不能据此作唯一机制归因。[最终图](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_qref_preload_comparison_v1/sameq_rest_comparison_readable.png)已真实查看，轴角到窗末仍在增加；有限窗PASS不是稳态平衡。全部数据与绘图版本见[最终pins](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_qref_preload_comparison_v1/FINAL_PINS.json)。
