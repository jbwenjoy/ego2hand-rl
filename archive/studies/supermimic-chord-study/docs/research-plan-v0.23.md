# 研究方案 v0.23：闭合、支持与接触面定义分开验证

2026-11-07。目标保持 active，完整范围承接 [v0.22](research-plan-v0.22.md) 及原 SuperMimic + CHORD 目标。上一目标回合和本回合均产生实际代码、静态求解、保存审查及动力学证据，属于 progress；完整研究目标尚未完成。没有启动新的学习训练，下一次训练首个健康 checkpoint、ETA 后暂停的约定继续保留。

## 当前结论

本回合把真实 HOT3D 手机参考推进到两个几何闭合姿态，并在不改变原固定腕、尺度、机器人参数、材料和控制范围的条件下恢复了其中一个姿态的合力平衡。但未取得可用持握：新一轮两步长无辅助测试均在约0.12秒达到50mm漂移终止阈值。不能把距离贴合、合力平衡或COM在接触线下方分别当作完整承载能力。该结论只属于当前模型、先验与有限搜索，不证明该机器人或所有姿态不可行。

手机仍是官方配套视觉资产及单凸碰撞体；质量0.2kg、手物摩擦0.7、桌物摩擦0.6、COM/惯量均为显式工程先验。上一版无手台架资格继续有效；真实参数、Hand2持握、ego轨迹、策略及泛化均未取得资格。真实ego数据已用于参考几何，尚未用于机器人策略训练。

## 新的支持与几何接口

新增 [StaticRetargetEvaluator](D:/workspace/ego-data-pilot-2026-11-07/retarget_gravity_v3/evaluator_v3.py)，每个姿态检查全部22个手机对及189个原生有效手部自碰撞对。远处采用有向AABB分离证书，近处使用已验证的独立凸查询；未知不通过。每次重算零速bias、passive及实际接触点Jacobians，避免沿用旧姿态的法向、位置或力臂。4次静态契约检查10/10通过，与原生forward参考的bias/passive/控制可达力矩界差均为0，没有执行积分。[静态检查](D:/workspace/ego-data-pilot-2026-11-07/retarget_gravity_v3/static_contract_v1/report.json)

新 [支持残差LP](D:/workspace/ego-data-pilot-2026-11-07/gravity_support_design_v3/support_residual_lp_v3.py) 保留真实力矩/PD命令可达范围及每个pad最多3N法向力。设计锥仍为实际摩擦0.7/√2的圆内接八边形，包含于任何切线朝向的原生四射线锥，不授予点接触任意扭转矩。主问题硬约束合力平衡，最小化COM力矩∞残差；主问题不可行时，另算仅用于恢复指导的合力残差，主失败状态不变。纯解析夹具4/4通过。

3N是静态规划预算，不能当成原生瞬态反力上限；原生反力完整保存，未裁剪成规划值。关节及控制的实际硬范围始终保留。

力学只使用当前真实、唯一的最近点配对：物体点计算力臂，手点计算Jacobian。正间隙是潜在闭合接触，仍非当前实际力。相交、点对不唯一、切线图奇点或数值异常均明确记录；不取任意支持质心充当真实接触。COM下方仅为必要侧别约束，不是动态稳定性证书。

**严格支持候选门**保持合力误差≤1e-8N、力矩∞残差≤1e-4N·m及全部几何、实际COM侧别、真实控制范围。小残差仍只是一种工程候选，不能代替原生持握。另行定义的**有限网格动力学诊断入组**允许力矩残差存在，要求几何、合力平衡和真实控制范围已核；该入组不授予严格静力或持握资格。实际测试继续使用原2秒、5mm、5°、接触覆盖、穿透、非目标接触、速度和warning门，不放宽通过标准。

## 已完成的三段求解

1. **v3原腕联合探索**：原几何合格起点与98°重力对齐但相交的起点均保留。阶段A用固定物体材料点连线的平滑重力必要条件恢复几何，阶段B使用当前真实接触LP。实际226.22秒，1456次静态native评估、3661次重型凸查询、59次LP，无积分/训练/GPU。三个已执行阶段分别20/19/11次callback，均触预定时间上限；第二起点无可评估几何而未运行阶段B。严格候选0。[结果](D:/workspace/ego-data-pilot-2026-11-07/retarget_gravity_v3/completed_v1/report.json)及[保存独审8/8](D:/workspace/ego-data-pilot-2026-11-07/retarget_gravity_v3/completed_v1_saved_recheck/report.json)。
2. **v4固定手机姿态闭合**：从上一失败前缀预先选定两个起点，手机姿态固定，仅优化原20关节，全部选择与失败保留。实际151.24秒，两个姿态均通过最终22/189几何门；两pad间隙分别约50.01/50.02µm、114.63/116.62µm。但合力硬约束都未通过，恢复力∞残差约0.4982/0.8702N，未进入物理。保存独审9/9通过。[结果](D:/workspace/ego-data-pilot-2026-11-07/retarget_contact_polish_v4/completed_v1/report.json)及[独审](D:/workspace/ego-data-pilot-2026-11-07/retarget_contact_polish_v4/completed_v1_saved_recheck/report.json)。
3. **v5从闭合姿态继续联合支持优化**：复用同一冻结v3驱动、真实范围/摩擦/3N上限，从v4两姿态重新求解；没有修改已执行协议。实际247.17秒，1684次静态评估、4372次重型查询、930次主LP及914次恢复LP。两个最终姿态仍通过几何；seed0恢复合力平衡，法向力1.649/2.803N、最小真实力矩余量约0.1006N·m，但力矩∞残差0.100113N·m，严格候选仍为0；seed1主合力约束仍失败。保存独审9/9通过。[结果](D:/workspace/ego-data-pilot-2026-11-07/retarget_gravity_v3/support_refinement_v5_run1/report.json)及[独审](D:/workspace/ego-data-pilot-2026-11-07/retarget_gravity_v3/support_refinement_v5_saved_recheck/report.json)。

上述都是序贯工程探索，不是盲测或同预算方法收益对照。所有有限时间停止都不能当作不可行证明。旧缓存登记对先以FD访问、后被优化器用作base的相同点可能保守漏登，故未声称最优或穷尽全部候选；最终几何/力学门与实际测试仍独立。下一版优化接口应显式处理候选登记，不改写已有结果。

## 第二次无物体辅助持握实测

新元数据适配只让v5的seed0入组，seed1的合力失败记录保留。原生积分循环、PRE接触/能量与POST状态、尾窗及原15项每例门与旧版本逐AST核对；新增2mm/2°的两步长末态比较，不能替代每例2秒通过。

固定命令、无物体外力/姿态重置/等式/mocap辅助，使用相同原机器人和显式pair材料。一次CPU运行两个步长0.3125/0.15625ms，共1153步，进程exit0、完整stderr为空、原生warning为0。粗/细分别在0.1203125/0.1200000秒触发50mm漂移终止，末角约35.16°/34.18°，最大原生穿透约0.0819/0.0693mm；未记录原生手部自碰穿透、非目标部件或桌面对手机的支撑。两例均失败，1.5–2秒尾窗未运行，末态差小也不给2秒步长资格。不能据此称较前一控制或方法改进。[完整物理报告](D:/workspace/ego-data-pilot-2026-11-07/hand2_holding_v3/cpu_nominal_v1/report.json)。

保存独审8/8通过，1153个状态和420条原始接触全部重算；两例均为首次COM漂移超过50mm而终止，角度尚未到45°提前终止门。反力已实际建立但未持续维持双pad接触，不能描述为全程无接触自由落体。运行时无外力/warning等守门保留其生产源码证据范围，未冒称保存中不存在的全量buffer独立重建。[保存审查](D:/workspace/ego-data-pilot-2026-11-07/hand2_holding_v3/saved_recheck_v1/report.json)及[实际前缀图](D:/workspace/ego-data-pilot-2026-11-07/hand2_holding_v3/saved_recheck_v1/holding_v3_actual_prefix.png)。

## 约束失败的解释与接触定义修订

v4第一个恢复解出现index joint2的PD可达上界约为0且饱和，但这只是共现。预先锁定的两个纯LP反事实只移除PD可达交集、保留真实actuator/joint力矩界和全部摩擦/3N条件，结果两例合力仍不可行。不能把失败唯一归因于PD范围，也不能将反事实当合法控制。[反事实报告](D:/workspace/ego-data-pilot-2026-11-07/retarget_contact_polish_v4/ideal_torque_counterfactual_v1/report.json)

本地pinned Hand2资产的 `tip_sensor_frame` 是固定零偏移子body，整体sensor STL作visual/collision；`sensor_pad`仅是材质名，未声明功能面、法向或软接触区域。现有canonical tip投影到整个凸包的anchor只定义位置，不能证明已选择真实指腹面或接触法向正确对置。[资产语义检查](D:/workspace/ego-data-pilot-2026-11-07/hand2_holding_v3/pad_semantics_source_review_v1.json)

对既有左手全部61×788保存网格顶点的unsigned扫描距离统计确认：全窗口只有thumb/index粗区域有≤5mm点，其他左手指节/掌腕区域没有第三近距证据。frame42各区域最小值约thumb0.0318mm、index0.4252mm、middle28.84mm、ring48.82mm、pinky55.72mm、root/wrist45.59mm。2mm离散近点图在thumb有两组件、index一组件；5mm图为两指各一组件。“两指”不能缩成“两个数学点”，离散组件也不是接触面积、力或独立接触证书。蒙皮混合权重、unsigned距离、扫描/重建误差及右手覆盖范围必须保留。[全部顶点审查](D:/workspace/ego-data-pilot-2026-11-07/contact_region_audit_v1/interpretation_v1.md)。

另以同一官方实现完成右手61帧重建及原扫描距离：1次官方人体FK batch61，49,288点nearest查询，无机器人FK/物理/学习。frame42右手最小距离15.394mm；整个窗口最小2.619mm，仅6帧右食指有≤5mm近点、没有≤2mm点。没有持续贴合或承重证据，但有限网格和重建误差也不能证明单手受力或右手作用为零。后续接触来源掩码应保留这些不确定帧，不用距离直接生成力真值。[右手审查](D:/workspace/ego-data-pilot-2026-11-07/right_hand_contact_audit_v1/interpretation_v1.md)。

下一次接触候选应显式定义机器人实际凸包的局部面片、法向锥和范围，与人体近距区域/物体材料面匹配，再核完整几何、实际力矩与无辅助动力学。没有CAD语义或实物校准时，这些只能称工程几何候选。不能为了通过而任意增加torsional friction、扩大法向力预算、添加没有观测支持的第三区域、缩放手机或移动原固定腕。若研究其他恒定安装姿态，必须另立scene与匹配对照，并保留原姿态结果。

具体下一最小路线：先修正base/FD缓存的候选登记；在新命名空间试用有尺度说明的六维合力/力矩联合残差，避免把“先可平衡合力、再处理力矩”的中间状态当完整支持。对非唯一最近特征，应构造经过几何核验的成对近接触面片/边界点集合，而非随意选一个质心；法向力上限按手指汇总，不按面片采样点倍增，也不凭面片模型授予任意torsional力矩。然后分离关节接触建立、预载和保持阶段，核实际持续反力与静态目标的差别。若使用仿真接触反馈作可行性控制，必须标为privileged诊断，不能冒部署可观测性或策略收益。任何新候选仍需原无辅助持握门实测。

正间隙reset并不等于已建立承载接触。应增加对原生柔顺接触静止平衡的直接核验，将原生接触反力/自由物体加速度与非目标非穿透、真实actuator/ctrl范围一起约束，避免只靠孤立的理想点力反解常量PD命令。若新方案允许指定pad的初始柔顺压缩，须另立协议固定其范围及原生预载证据；不能在旧协议中把负距重新解释为通过，更不能把非目标link穿透称pad预载。原2秒动力学门仍保持。

## 重力注册及学习接线的必要契约

官方论文说明MPS轨迹随后经7DoF对齐到OptiTrack frame，不能直接把MPS的−Z重力当成本clip的GT重力。当前尚未找到本序列的注册旋转或同world重力字段。原98.342°是当前机器人模型固定材料连线的重力矩消除角；99.862°等源/机器人重力差只在未验证的源−Z假设下成立，不作为实际量。[注册追查](D:/workspace/ego-data-pilot-2026-11-07/gravity_reference_audit_v1/registration_source_followup_v1/report.json)、[官方论文Appendix A.3](https://arxiv.org/html/2411.19167v2#A3)。这不阻止在明确机器人重力先验下做物理可行性研究，但限制源重力保持的宣称。

进入学习前还需独立的新phone合同，不能仅修改旧pen的dt：[完整接线审查](D:/workspace/ego-data-pilot-2026-11-07/temporal_bridge_audit_v1/report.md)。保留20ms控制周期，64/128个积分子步分别对应名义区间内19.6875/19.84375ms的lastsolver传感器相位，原生时钟仍需另行读回；WM仍每20ms预测一次。旧4×5ms入口、clock/index、J/K/残差及四子步标签均需参数化并重新生成。相同控制预算的物理步数增加16/32倍，同预算SM/CHORD必须同时报告控制区间、实际积分、WM/AD/策略更新及walltime，不能宣称旧预算或加速。

58/52可保维数但需明确root↔COM、速度与惯量/扳手坐标；手机body原点到COM有非零偏移，旧串行入口硬拒该情形。xi46也未编码全部mesh、绝对I/COM和求解器参数，不能按shape直接继承旧物理资格。旧pen部分actor观察与奖励仅使用长轴两端点，手机完整姿态/接触目标需新Reader/MDP与缓存。当前没有实施这些学习适配或启动训练，旧152训练行结果仍只属于原受控分支。

完整后续范围继续包括：接触面与参考可行性、名义持握和步长/参数鲁棒性、61帧时序重定向、SuperMimic/CHORD同预算学习、世界模型接触/梯度校验、独立轨迹、真实参数与硬件契约。上述范围不会因几何闭合或任何窄检查通过而缩减。
