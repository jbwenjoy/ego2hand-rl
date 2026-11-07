# 观测接触意图与初始抓持探索

本工作区承接[观测派生参考的物理保持失败](D:/workspace/supermimic-observation-grounding-study/README.md)。当前仅研究受控缓存观测到机器人初始化的映射，不使用上游机器人 q 标签，不接入真实 ego 数据。接触意图是待验证的几何假设，置信度保持 unknown。

最新H123三指扩展已导出受限静态PD预载，并在同q0单次0.3s真实保持通过原门槛；H23失败及未成立的旧B保持原样。该结果只支持有限静止窗口，完整动态参考仍未获资格，详见文末。

## 观测侧关系发生了什么

解析[笔的真实外表面](D:/workspace/supermimic-observation-contact-intent-study/scripts/pen_outer_surface.py)使用已验证的圆柱与两端 marker box 并集，排除内部端面，不调用有反例的 geomDistance。原始观测、纯刚性重映射、长度归一化目标和机器人 FK 四套数据分别保存；纯 R/t 对照保持点与物体的相对距离。

首帧 source 刚性重映射中的链2点7距物体表面 8.712mm，链3点11为 6.409mm，两处物体外法向夹角 173.125°。长度归一化后，同一序号点的夹角为 111.152°，其中链2外法向的世界 z 分量由 −0.551 变为 +0.639；已有几何候选 FK 对应两点夹角仅约 28.31°。这些数值揭示接触意图线索的改变，不构成接触标签或力闭合判定。骨架点到表面的距离也不是整个手碰撞体的间隙。

[逐点/逐帧报告](D:/workspace/supermimic-observation-contact-intent-study/outputs/contact_intent_v2/REPORT.txt)、[冻结首帧坐标与排序](D:/workspace/supermimic-observation-contact-intent-study/outputs/contact_intent_v2/frame0_rankings.json)保留全部点、最近外表面点、法向、特征及并列项。首帧排序 SHA 为 `03a22cbdb2b9b74e6582bc3acc83843d007f9c0a07f0537437e158c326391e33`。图中 C1–C5 的解剖名称带星号，表示默认 ordinal 拓扑假设，尚未做独立解剖标定。

![同序号点的表面关系](D:/workspace/supermimic-observation-contact-intent-study/outputs/contact_intent_v2/frame0_comparison.png)

## 固定物体的有限候选：均未取得物理资格

H23 取近表面且对向的链2/3；H123 明确加入下一近的链1，作为扩展假设。每个候选都从同一 q0 开始，固定物体位姿、腕与骨方向目标，最多120次函数求值。实际碰撞体上的材料点和材料支撑法向在优化前冻结，以点/法向目标吸引；原生全手碰撞只提供重叠残差，未知 gap 继续为 NaN。

| 分支 | 最大材料点误差 | 最大手物穿透 | 最大自碰穿透 | 结果 |
|---|---:|---:|---:|---|
| 最近碰撞体 v1，H23 | 20.290mm | 0.532mm | 未生成 | 未达材料意图，静力 screen 不可行 |
| 最近碰撞体 v1，H123 | 20.500mm | 1.518mm | 1.808mm | 未达材料意图，静力 screen 不可行 |
| ordinal 远端对应 v2，H23 | 10.925mm | 7.675mm | 未生成 | 未达材料意图，超过原3mm深度门槛 |
| ordinal 远端对应 v2，H123 | 9.319mm | 10.173mm | 5.126mm | 未达材料意图，超过原3mm深度门槛 |

v1 的跨连杆最近选择把链2点7绑定到近端 link2；v2 按既有 ordinal 对应限制到 link4，tip 点限制到 tip_sensor。两版源、配置、优化前 bindings、全部结果保留。它们改变了材料点假设，不是相同目标下的优化增益比较。四次优化均报告收敛，但收敛和每指有接触都不能证明意图达成。

结果目录：[H23-v1](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h23_v1/report.json)、[H123-v1](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h123_v1/report.json)、[H23-distal-v2](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h23_distal_v2/report.json)、[H123-distal-v2](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h123_distal_v2/report.json)。所有分支 `physical_reference_qualified=false`，未执行 GPU 或物理步骤，未生成 PD 目标。

## 接触方向的必要条件

原保持首帧的真实接触有效 condim 为4，允许绕法向的扭转力矩。它单独平衡重力需要负法向力 −0.101133N，且即便允许无限法向扭矩，仍有约1.922mNm垂直法向力矩无法抵消，见[严格必要条件](D:/workspace/supermimic-observation-grounding-study/outputs/rest_wrench_feasibility_v2/REPORT.txt)。

v1 两套新几何尽管有更多接触，其接触方向仍无法平衡重力。原生摩擦棱锥 LP 以及放宽为连续摩擦力锥后的解析分离证书均失败，见[候选静力 screen](D:/workspace/supermimic-observation-grounding-study/outputs/initial_candidate_wrench_screen_v1/REPORT.txt)。这是物体侧的必要条件检查，不预测接触力，也未证明机器人关节/PD/软接触动力学平衡。

## 实现与当前工作

[材料点辅助模块](D:/workspace/supermimic-observation-contact-intent-study/scripts/material_anchors.py)使用实际 capsule 解析支撑点或 compiled mesh 顶点的线性支撑极值，未替换碰撞几何。独立局部检查覆盖[五指材料点与法向 Jacobian](D:/workspace/supermimic-observation-contact-intent-study/outputs/material_anchor_cpu_v1/report.json)及[全部10个 enabled mesh 分支](D:/workspace/supermimic-observation-contact-intent-study/outputs/material_mesh_independent_v1/report.json)。这些仅证明所检查固定材料分支的坐标/导数，不证明全局光滑、可达或抓持。

[初始化设计审阅](D:/workspace/supermimic-observation-contact-intent-study/docs/initialization_design_review.md)保留局部选择、对应、并列项与失败条件。

单指分解随后完成：固定材料点的点目标各有合法近解；加法向项后的点误差也在微米量级，但链2仍有24.70°法向误差。全部[8个条件](D:/workspace/supermimic-observation-contact-intent-study/outputs/material_reachability_v1/report.json)和严格但未排除目标的球界保留，不能把先前全手局部结果判为不可达。

两个单指B解组成[复合初值](D:/workspace/supermimic-observation-contact-intent-study/outputs/material_reachability_v1/composite_seeds.json)。新caller仅替换优化初值，逐字保留全手残差/Jacobian与qbaseline正则，并在运行时要求材料bindings与远端v2完整相等。[静态独审](D:/workspace/supermimic-observation-contact-intent-study/outputs/seeded_caller_review_v1/report.json)和实际绑定均通过。两个复合初值仅差5.394e-6rad，结果几乎相同，不能作为独立样本N。

| 固定物体warmseed-v3 | 实测 |
|---|---:|
| 生成手物接触 | 3pairs，链2和3 |
| 最大生成手物深度 | 1.814715mm |
| 生成手自碰接触 | 0，未知gap仍为NaN |
| 链2/3材料点目标误差 | 1.591/.0435mm |
| 链2/3材料法向误差 | 51.85/8.40° |
| 骨方向均值/最大误差 | 12.416/54.692° |
| 物体位姿、腕、q硬限位 | 固定/固定/几何超限0 |

两套结果分别保存：[baseline-derived canonical](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h23_seeded_baseline_v3/report.json)、[midpoint-derived敏感性](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h23_seeded_midpoint_v3/report.json)。Canonical事前固定为前者。该初值变更使同目标下找到另一局部解，但不证明全局最优、精确意图或物理抓持；接下来检验静力与可执行PD预载，再比较同q0的两种固定目标保持。

物体平移修正分支另保留在[未执行的CPU准备](D:/workspace/supermimic-observation-contact-intent-study/outputs/objectshift_cpu_checks_final_v1/READY.json)：固定旋转/腕，在20mm硬球内联合优化q与物体位置，51项局部检查通过。该分支会改变source物体映射，置信度仍未知；目前未优化、未送物理，不与固定物体候选混为同条件。

## H123与完整扳手、执行器约束

新增拇指单指B点/法向解后，固定旧两指材料bindings，只增加拇指块组成两套全手初值。H123 baseline/midpoint分别生成4/5个手物接触，最大手物深度2.183/1.815mm、几何q硬限超限0，未生成自碰。拇指材料误差/法向误差分别为1.299mm/40.10°和0.0245mm/19.79°；仍不是精确观测意图。两套[几何结果](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h123_seeded_baseline_v3/report.json)与[敏感性结果](D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h123_seeded_midpoint_v3/report.json)均保留，六维物体扳手筛查的最小Σfn约1.174/1.083N，不等于实际力。

[全手预载设计](D:/workspace/supermimic-observation-contact-intent-study/outputs/h123_preload_cpu_v1/FINAL_PINS.json)明确区分q、u、手重力广义力、接触对手广义力及物体完整扳手，约束目标为关节/启用执行器交集的C2内域，并保留实际力上限与非零frictionloss。baseline的零ζ和有界ζ两LP均可行；按预先固定的数学优先规则选择零ζ，Σfn2.139425N，postcast手平衡残差6.36e-8Nm，u32经C2逐位不变。该零ζ只是见证，不关闭模型摩擦。

midpoint的CPU q32重建只有3个接触，与其保存q64/实际Warp startup的5个不同，物体LP不可行，u0=null；不能选择另一个backend重查获得成功。原whole-hand LP未运行的状态保留，另行明确补算的两个LP也不可行。backend差异未被唯一归因为cast。全手设计仍排除关节限位反力、未强制软接触本构，因此数学可行不保证动力学实现。

同一canonical baseline q0的QREF保持原轴角31.331°失败；随后唯一一次静态预载保持通过20mm/20°/3mm/1mrad门槛，实际3.956mm/16.287°/2.183mm/0.626mrad。两次是顺序探索N1，没有重复选择或放宽门槛。详细[执行/独立raw证据](D:/workspace/supermimic-observation-warmseed-grounding-study/README.md)与[动态需求分析](D:/workspace/supermimic-observation-dynamic-wrench-study/outputs/dynamic_wrench_v2/REPORT.txt)分开：旧151帧几何轨迹的71帧无完整需求扳手见证，另80帧无上界见证也可能要求数百N。H123首帧未拼接到旧全轨迹；下一步需整段接触意图与明确初速/控制一致性，不能由静止PASS推导完整运动成功。
