# 全151帧接触几何耦合：局部导数验证与首次8步SCP

本轮真实执行了固定的164次局部几何导数probe，以及一次最多8步的全151帧接触几何耦合求解。**最后接受点仍为FAIL，无READY：手物最大穿透降低，但自碰显著恶化，不能称总体修正成功。** 新物理积分、Env、WM前向、GPU和训练更新均为0，Goal保持active。

继续使用已有legacy source[200,351)、固定腕任务重设计、同一派生几何目标与物体曲线。没有新视频或独立ego样本，没有读取原始机器人答案。上轮[20独立LP与601点负例](D:/workspace/supermimic-full151-observation-study/docs/full151_input_findings_2026-11-04.md)及其失败状态完整保留。

## 先验证局部模型，不授予全局梯度资格

新[all-witness行代数](D:/workspace/supermimic-full151-observation-study/src/full151/contact_row_oracle_v2.py)为每个实际生成的接触保留独立行，不平均ties，不把未生成pair的未知gap填成正间隙。距离导数来自同一原生contact.pos、原geom顺序与两body point Jacobian：`n_A_to_B @ (J_B-J_A)`，再取20个机器人hinge列。该公式不等于所有接触分支已可微。

按[固定probe协议v2](D:/workspace/supermimic-full151-observation-study/protocols/full151-native-witness-probe-v2.json)，选择旧数据中最深HO17、最深self335、强tie HO489和self330四个样本，所有20关节分别±1e-5rad，仅执行一次164次query。没有根据新FD结果改选pose/eps。两条指定unique分支同时要求双侧生成、唯一、normal/count/selected-geom-order稳定及原1e-3m/rad门槛；这些只检测局部inventory/normal稳定，不是持续feature身份。

| 指定状态 | 局部中心FD最大误差 | 判定 |
|---|---:|---|
| worst HO，t=.085s | 1.5514885e-7 m/rad | 两侧全部20坐标的规定局部门槛通过 |
| worst self，t=1.675s | 4.0700645e-12 m/rad | 两侧全部20坐标的规定局部门槛通过 |
| tie HO，t=2.445s | 中心FD与selected之差仅2.4376e-5 m/rad | 有contact-count切换，不合格；单侧模型最大误差达.138440m/rad |
| tie self，t=1.65s | 中心FD差很小 | 仍为5个并列witness，不授予光滑资格 |

[实际probe receipt](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_witness_probe_v1/receipt.json) SHA `bf58c4a0a9b5617e7f7225faebee9df7cb92c09899c10a18f9024f56c69b3d4b`。165次FK/COM包含constructor一次，164次collision、1782次mj_jac，所有受禁动力学/力查询/优化调用0。[853项保存数据独审](D:/workspace/supermimic-full151-observation-study/outputs/native_witness_probe_saved_independent_v1/report.json)逐项复算库存、FD、稳定性和对应关系；它不重放未保存的body Jacobian原矩阵，也不把两处局部通过扩展为未来601点或tie梯度合格。

## 单一有界的全段SCP实验

[正式协议v2](D:/workspace/supermimic-full151-observation-study/protocols/full151-coupled-scp-cpu-v2.json)与[执行入口v2](D:/workspace/supermimic-full151-observation-study/scripts/run_contact_coupled_scp_cpu_v2.py)在运行前冻结。v1未执行且保留；v2加入fatal-stop禁止READY、finite merit检查，并明确规划buffer与最终标准。纯构造器35项synthetic检查与两份独立source审阅29/27项通过后，Root只执行一次，没有改参数或额外重启。

每步全局9061变量：原20个关节的453变量块，加一个米制几何弹性变量η。保留全部原关节硬界、C2/目标更新率、参考PD区间Bernstein约束、L1拟合与4倍系数二阶差分目标。固定原P0/P1、解析qd0、实际旧u0、tau及物体曲线。`y=Pfree-P0`只作旧矩阵的代数平移，所有row正乘1e4；同时按原单位检查新centered与旧absolute矩阵的1e-12数值证书。

几何行来自当前601点的全部生成witness，η只松弛几何局部模型，不能松弛控制约束。2.999mm是线性化规划buffer，最终仍按原3mm采样标准，未认证非线性实际达到2.999mm。η≤1e-9m只用于最终资格，不能提前堵住弹性修复步骤。

真实merit为：原20关节直接重算的fit+4smooth之和，加1e4乘全601点最大生成穿透超过2.999mm的正部。它不是两类碰撞各自不恶化的约束。每步LP normal退出且旧/新raw数值和FP32控制门全部通过后，重新执行完整601点原生几何；预测与真实下降均为正、比值≥.1才接受。拒绝则信赖半径减半，仍占8步预算，没有线搜、扩半径、择优历史或失败后补修P。最终只取最后接受点并做额外fresh601审计。

## 实测结果：改善一类，恶化另一类

| 601点生成接触指标 | 初始完整曲线 | 最后接受点major8 |
|---|---:|---:|
| 手物最大穿透 | 15.158818mm | 11.447693mm |
| 手物超过原3mm的采样数 | 350 | 338 |
| 自碰最大穿透 | 5.884525mm | 11.431469mm |
| 自碰超过原3mm的采样数 | 16 | 38 |
| 未生成HO、gap仍UNKNOWN的采样数 | 15 | 47 |
| 有tied pairs的采样数 | 257 | 264 |
| 原fit+4smooth直接值 | .411554 | .616162 |
| 优化merit Φ | 122.009731 | 85.103095 |

8次LP全部正常终态，旧absolute/新centered原单位raw与bounds证书均通过。第1–5、7、8步接受，第6步拒绝，信赖半径从.15降到.075rad；第7步中心确为第5步，没有使用被拒第6步作为中心。最后η=.0084213562628m，即8.421mm，远高于最终1e-9m门槛。

[原始receipt](D:/workspace/supermimic-full151-observation-study/outputs/full151_contact_coupled_scp_v1/receipt.json) SHA `c5c31f917e2ebd4024719632fd945bcda83bbeccb682b5f6f5aa6bf4c44668be`：26/28门通过，`last_accepted_eta_final_gate`和`final_fresh_sampled_geometry`为false，整体`CPU_COUPLED_CANDIDATE_FAIL_RETAINED`。原150条控制/C2/149计划更新率/FP32原clamp一致性及全区间参考PD caps通过，不等于接触动力学或完整运动成功。

[独立保存数据审计173/173](D:/workspace/supermimic-full151-observation-study/outputs/full151_scp_saved_independent_v1/report.json)重建每步原/新global CSR、geometry J→B行、bounds、直接F、预测/真实Φ、接受和半径更新、最后接受身份及10组原生接触库存统计，结论一致。没有重求解或重查询。native_ncon、clock/qvel未以raw arrays落盘，其完整性/零速度断言仍是source-bound运行证据，不冒称已离线重放原生状态。

[执行来源与终态身份独审32/32](D:/workspace/supermimic-full151-observation-study/outputs/full151_SCP_saved_lifecycle_independent_v1/report.json)核对24个source/input pin、11份执行源码、103份不可变artifact、最后接受点/7个init数组/全部固定物体曲线以及151/150/149身份。唯一mutable live_state旧pin过期单列，没有把它伪写为全pins一致；未重跑模型或几何。

![全151帧接触耦合结果](D:/workspace/supermimic-full151-observation-study/outputs/full151_contact_coupled_scp_summary_v2/contact_coupled_scp_summary.png)

Φ下降压低了原来占主导的HO最坏重叠，但没有防止self恶化，最终两者都远超3mm。这个负例揭示了本次有限L∞弹性目标的类别取舍与局部模型限制。它不证明更长迭代必然可行，也不证明原运动全局不可行，更不能作为CHORD/SuperMimic总体收益、动力学力预测或ego泛化结论。

## 前缀延续的结构诊断

另做[只读B样条限制映射分析](D:/workspace/supermimic-full151-observation-study/outputs/RF16_full151_basis_embedding_readonly_v1/report.json)，没有求解或生成新151曲线。旧RF16在.3s是4重端点，full151在.3s只是内部结点；直接复制18个旧控制点从>.26s开始改变曲线，q最大变化20.403mrad、qd最大6.599rad/s。通用解析映射可以在[0,.3]复现旧q/qd/qdd，但对应第17控制点的0-based joint3/7分别超过当前系数box 47.065/110.639mrad。

这说明“精确旧前缀+当前系数box”不兼容，不代表q(t)真的越原关节界或运动物理不可能。系数box是连续q范围的充分而非必要证书；未来可以另立方案，用逐段q Bernstein范围或严格极值检查保留同一物理关节界，研究较少保守的可行域。它不是本轮调低标准、拼接q尾或重新选择候选，也不应阻碍明确known-rough的native151真实接入。

## 成本、来源与下一步

SCP内部wall10.811131s，其中8次LP合计3.689296s；10组grid共6010次collision、6011次FK/COM（含一次constructor）、124122次mj_jac，动力学/力/geomDistance和geometry guard内优化API调用0。几何guard外8次真实LP单列，不把guard内0误读成全流程0优化。局部probe的164次query另计。以上仍不含全部审阅/导出/作图成本，不与CHORD训练时长作提速比较。

Root保留了前台session53563的启动和终态exit1工具观察，PID/Creation未采样、完整外部wall未测，不能将两次等待时间相加作为进程wall。原receipt的103个immutable artifact pin一致，唯一live_state在末写TERMINAL后旧pin过期；[终态inventory说明](D:/workspace/supermimic-full151-observation-study/outputs/full151_contact_coupled_scp_v1/terminal_artifact_inventory_v1.json)独立记录这一可变进度文件，不改原receipt、数值、源或失败判定。

后续重点不缩成“CPU几何必须完美后才能研究学习”：原任务本来就是在动力学中修正粗糙观测参考。应将最后接受状态以明确known-rough身份接入真正native151 controller、解析速度/初始化、partial reset、H8/replay/end及新checkpoint消费者，并先做完整150interval的A0工程验证。A0任务质量失败可如实保留，不等于工程身份/容量/完整性可以豁免；不加载旧104 checkpoint、不继承旧prefix物理成功、不暗改本次FAIL。最终学习/物理收益仍需实测。

几何路线并行做结构诊断：分别看HO与self可行性、局部模型和固定初态约束，明确新方案，不自动增加迭代/调权重去挑winner。全151独立接触意图/扳手目标仍PENDING，观测距离、未生成接触和旧16帧标签不能填作已知human-contact或释放；A2不能用退化的空目标冒充已验证接触增强。

当前没有长训练或GPU作业，因此没有触发首个健康checkpoint后估时暂停条件。shared多轨迹、真实ego与硬件验证仍未完成，Goal保持原范围active。

[独立结果评议与native151优先级](D:/workspace/supermimic-full151-observation-study/docs/full151_coupled_attempt_assessment_v1.md)。同一basis诊断的纯线代补充验证旧RF与限制映射的前15段q Bernstein系数一致且原关节范围越界0；这不构造新尾段或授予full151资格。
