# full151 接触几何耦合 SCP：有限可行性合同草案 v1

**DRAFT_NOT_AUTHORIZED_NOT_EXECUTED。** 本文和同目录 proposal 是下一次有限 CPU 尝试的设计，不执行 solver/FK/query/Env/WM/GPU。必须由 Root 的新 producer、协议与有界启动器明确授权，不能直接消费本草案作为执行配置。

原方案要求共享世界模型修正粗参考，并在物理仿真评价；参考本身不是动力学监督。现 full151 候选虽然计划控制和参考PD检查通过，仍有大面积穿透，且原始LP数值资格失败。下一次先保留完整 `[200,351)`/151状态/150目标/固定3秒物体运动，把全手接触几何耦合纳入一次有限修正，不缩回prefix，不用调门槛绕过失败。

## 固定输入与两类资格

固定父候选的 source64 P0/P1、requested FP32 q0/nonzero qd0/object pose/world-body twist/u0。全段物体曲线、539d派生fit目标、真实actual-context/MJB/几何、hard∩ctrl/C2.020、原始奖励/原力caps/15rad/s目标更新标准不变。原始记录机器人q/link/wrist不作为重定向答案；允许且已明确的wrist字段读取不等于把它用于机器人腕控制。新curve必须仍是完整151状态、u0..u149，terminal150没有u150。

规划穿透上限为 **2.999mm**，给原 **3mm** 评价标准留1µm数值裕量。建议规划算术容差1e-12m只说明矩阵/采样算术，不能被解释为几何准确度；原3mm门保持exact，无放宽。原始linear row/bounds的1e-12原单位、2µrad target margin、2µNm force margin、14.9999rad/s规划上限及15原标准分别保存。

通过时最多得到“601样本生成接触几何 + 计划控制输入资格”。`physical_reference_qualified`、完整动力学/真实接触力、native151 reset/replay/新checkpoint消费者与训练授权仍为false或pending。参考态wholeinterval PD力caps是**充分的受限surrogate**，不是实际控制负载；601点碰撞是**采样**约束，不能外推连续间隔无穿透。两者量词不同、不能互相替代。

这也不能成为无限等待“完美openloop参考”的门槛。未来独立明确的 rough-learning mode 可以研究残差修正尚未物理成功的粗参考；它仍需来源/数值/控制/初始化/真实native工程完整性、容量与授权，保留失败证据。不能把本次失败候选暗中改成READY，也不能从旧prefix104的学习成功继承full151资格。

## 全局局部模型

保留20个原jointLP的全部48040行与hard coefficient box。变量为20个453维joint block（151个centered freeP、151 fit epigraph、151二阶差分epigraph）加一个η_m，共9061维；物理自由系数3020。Pfree=P0+y只改数值表达，不改P0/P1或数学约束。正row缩放固定1e4，所有资格仍在未缩放原单位重算，并核 reconstructed absolute原矩阵。

在每个当前点，对601个5ms样本保留**全部实际生成的native contact witness**，包括同pair多个点与ties；不只选deepest、平均法向或丢掉零Jac/深穿透行。native normal方向须与原geom0/geom1次序配对：`Jdist = nᵀ(Jbody1(contactpos)−Jbody0(contactpos))`，明确机器人20dof列，再乘该时刻Bfree。对象运动固定，不能用移动对象/腕或投影q消除穿透。

每个witness一行：`−Jdist*Bfree*(Pfree_trial−Pfree_current)−η <= d_current+0.002999`，η>=0。未知/未生成pair不凭空加“安全gap”行。current set只对当前局部feature有效，trial必须重新生成全部native contacts再评价；所有新增/消失/tie/uncatalogued信息保留。初态q0固定，若t0有无法通过free basis改变的冲突，明确报告该受限初态冲突，不修初态。

目标保持原尺度：`F(P)=Σ_j[mean151|q(P)−target|+4mean151|Δ²P|]`，新LP目标 `F+1e4(rad/m)*η`。不是再除20，也不是单纯minη的phase-I。有限罚系数、solver成功或η接近零都不保证真实几何可行。

## 固定8步信赖域策略

最多8次LP，每次上限10s；每个attempt无论accept/reject都占一个major。r初始及最大=.15rad。不得扩张、线搜索、换solver/权重/种子或选择历史最优。API/非有限/非正常solverterminal/原始线性或postC2门禁失败直接停止，不以缩r重试数值失败。

正常LP trial先检查全部原/centered/new elastic row、hard/trust/bounds，再以原760 FP32 proposal+单次C2检查全部150目标/149邻接更新/u0/forcecaps。**η<=1e-9m是最终附加门槛，不是中间trial资格**，否则初始深穿透无法逐步恢复。中间必须η非负且弹性row审计通过，不能删除η来谎称hardgeometry已过。

valid trial执行完整601几何评价。真实 `Φ(P)=directF(P)+1e4*max(0,max601 generated depth−.002999)`；directF从P重算，不信solver epigraph。predictedΦ用同一个directF及当前全部witness affine distance的最大超量，不直接以solverη替代。要求predicted reduction>0、actual reduction>0且两者比>=.1才accept；严格下降不能被浮点同值替代。

accept更新 chronological lastaccepted，r不变；几何/merit不匹配则reject、原点不变、r*=.5，仍消耗major。这是执行前固定的算法内策略，不是隐藏外部参数搜索。所有trial保留。即便rejected trial恰好较低穿透，也不事后挑出；最后只评价lastaccepted。零accept要写明source仍为父FAILED seed，不能声称改进。

正常结束时对lastaccepted另做fresh601。最终要求完整原单位linear/bounds、当前η<=1e-9m、fresh规划2.999mm与原3mm分列检查、全部原postC2与FP32初始化/source不变等共同通过；任何一项不通过都无READY。中断/超时/API失败只保留partial/lastaccepted来源，不伪造final资格。

## 有界成本与结果解释

主SCP最多 `(baseline1+trial8+final1)*601=6010` 次计划碰撞查询；独立局部Jac probe另为 `4*(1+20*2)=164`，合计计划上限6174。构造FK、其他setup/metadata调用另实计，不把6174称为wholeprocess精确总数。建议worker soft220s、owned-process hard240s，但仍待 Root 正式协议确认；每个LP已开始即计attempt，异常/timeout保留输出且无自动retry。

局部164query proof只能评价所测稳定branch。ties、缺失/新增feature、zeroJac、接触集合切换都保留UNKNOWN边界；不能称全局接触Jacobian已认证。producer还要预声明witness/内存/输出上限；触限或截断为工程censor，不能丢行后继续声称完整约束。

失败类型必须分开：输入/source错误、固定初态冲突、数值solver/原始row失败、postC2控制失败、几何oracle错误/partial、8major后生成几何仍不合格。它们都不等于人类运动/所有机器人轨迹/未来残差学习全局不可行。

## 来源和实施检查清单

机器可读草案保存于 [protocol.json](D:/workspace/supermimic-full151-observation-study/proposals/full151_coupled_geometry_SCP_v1/protocol.json)，包括精确父数据与源pin、failure语义和pending项。旧`continuous_control.py`3c32源保留。新all-witness helper最初缺top-level units接口，v2仅补`units='m'`已由6项metadata proof闭合；新全局builder仍需Root实际driver对merit/accept/save/timeout做纯CPU负例。

重点负例：未知单位/列顺序、漏witness/平均ties、正常LP但raw1e-12失败、η>1nm中间应允许继续但final应失败、negative/nonfinite predicted reduction、actual worsening拒绝且r半减、零accept保父seed、最后一次reject不得替换lastaccepted、时间上限无额外LP/query、source/初态/U0漂移、追加u150、缺采样点当pass、旧16 contact意图tiling。它们不是本文件已执行的测试。

本步骤仅是**接触几何耦合**。碰撞约束不产生正法向力、接触支撑或换指阶段目标，也不把observed nearest-surface cues变成真实接触标签。CHORD式独立151意图/扳手支持仍需另立来源和资格；原UNKNOWN不能补成known-no-contact。未来真实动力学与WM方向校验仍另计，旧single-source/seed0或prefix结果不扩充为独立ego泛化证据。
