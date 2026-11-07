# 研究方案 v0.21：凸包接触步长收敛与真实ego参考几何

2026-11-07。目标保持active，完整范围承接[v0.20](research-plan-v0.20.md)及原始SuperMimic + CHORD目标；本版记录新增证据与下一进入条件，不以无手台架代替灵巧手重定向、学习对照、泛化或实物验证。上一目标回合完成真实资产与两后端基线，属于progress；本回合继续改变资产/参考/物理验证的权威状态。

## 物理配置与验证边界

固定使用同一手机凸包（352顶点/700面、原物体坐标、真实尺度），COM/惯量继续为显式均匀原AABB假设，质量/摩擦仍为工程先验。没有为通过而缩小物体、移动手腕或移除0.1kg/摩擦0.3等困难组合。原2mm穿透、2°末姿态步长差、静置/能量等门槛保留。

复核已安装MuJoCo-Warp 3.11源码发现put_model会将主求解容差限制为至少1e-6。v1实际CPU为1e-10、GPU为1e-6；当时记录保存了有效值，但对照审查漏检该算法差异。因此旧54项轨迹比较只能说明其数值差在当时容差内，不能声称算法设置完全相同。旧结果和失败不修改；新版本预先统一CPU/GPU主容差1e-6、线搜索容差0.01，并要求实际读回。详见[诊断源码记录](D:/workspace/ego-data-pilot-2026-11-07/simulation_convex_v2/warp_diagnostics_source_review_v1.json)。

Warp没有CPU式原生warning表。新诊断范围明确为全部支持的状态/qacc有限性、每步9位粘性溢出标志、容量及完整进程日志，不制造不存在的警告账本，也不把Python元数据检查称为GPU执行。qacc、原生能量与接触的PRE/POST相位单独标注。

## 已执行的接触与时间步探索

- **v2接触时间尺度筛选**：18个新鲜CPU测试、64800步；名义0.2kg/摩擦0.6，时间常数20/5/2.5ms，步长1.25/0.625ms。最硬配置最大穿透1.577mm，但推后末姿态步长差4.392°，全门选择为null，未进入确认或Warp。[选择记录](D:/workspace/ego-data-pilot-2026-11-07/simulation_convex_v2/cpu_screen_assessment_v2/selection.json)
- **v3更细步长探索**：固定时间常数2.5ms，三个步长0.625/0.3125/0.15625ms，9例100800步；原门全部通过，按预先规则选最粗合格相邻对0.625/0.3125ms。名义推移末姿态差依次降至0.1982°/0.0130°，不以瞬时角差另造门槛。[9例选择](D:/workspace/ego-data-pilot-2026-11-07/simulation_timestep_v3/cpu_screen_assessment_v3/selection.json)
- **v3全参数确认**：重新执行全部9个质量/摩擦组合，54例388800步。54例G1–G6通过；27个步长比较中，0.1kg/摩擦0.3的推移末姿态差2.809°，仍高于2°，确认未通过。不能在v3内部换成更细对或只报告26个通过组合；Warp元数据入口已正确拒绝执行。[完整失败报告](D:/workspace/ego-data-pilot-2026-11-07/simulation_timestep_v3/cpu_confirm_assessment_v3/report.json)
- **v4固定完整网格验证**：另立[新协议](D:/workspace/ego-data-pilot-2026-11-07/protocols/phone_convex_fixedgrid_v4.json)，固定2.5ms接触时间常数及0.3125/0.15625ms步长，所有9先验×3测试×2步长共54例、777600步。原几何/solver/输入/门槛不变；CPU完整保存审查通过后，才允许同网格Warp。此为序贯工程验证，不是盲测、实参辨识或方法效果独立确认。

v4 CPU完整独立保存审查已通过：54例G1–G6及27组G7全部通过，最大穿透1.397mm，步长末姿态差最大0.393°。[CPU报告](D:/workspace/ego-data-pilot-2026-11-07/simulation_fixedgrid/cpu_assessment_v4/report.json)。随后同54项Warp已一次完成777600 world steps/259200 batched steps与18次终态forward，进程session7040确认为exit0；完整stdout/stderr在进程结束后校验，保存结果仍需按全部原门及CPU配对审查，不能将终态exit0称为全阶段合格。[进程交接](D:/workspace/ego-data-pilot-2026-11-07/simulation_fixedgrid/warp_v4_process_terminal.json)

**v4保存审查已全部完成，当前无手台架合格。** Warp54例G1–G8和27组G7通过；最大穿透1.403mm。另一位未编写Warp记录器的读者额外复核54组G8全部通过：CPU/Warp末COM差最大0.104mm、末姿态差0.558°、共同tick COM RMS差0.601mm、首次接触tick一致。18组实际solver标量/FP32转换独立复核通过；GPU大模型数组只保存equal-cast守门结果而非原数组，证据范围仍为source-backed，未冒称完整隐藏状态等价。Warp累计FP32时钟偏差最多约409.8µs，逐次浮点加法位模式已核，比较使用整数tick而非按累计clock插值。原生接触力未导出，qacc/原生能量与接触相位按原协议区别。通过仅限已声明材料/质量/惯量假设的单物体无手台架。[Warp保存审查](D:/workspace/ego-data-pilot-2026-11-07/simulation_fixedgrid/warp_assessment_v4/report.json)、[额外独立G8](D:/workspace/ego-data-pilot-2026-11-07/simulation_fixedgrid/parity_independent_v4/report.json)。原失败和null选择保留，不能以本次合格替代实物校准、Hand2接触资格或ego方法效果。

当前证据指向积分分辨率及非光滑接触的敏感性。v2推移粗步长在释放附近出现更大倾斜瞬态，末态差几乎为偏航；接触候选切换与差异共现，不能据此认定唯一原因。完整曲线与失败保留：[姿态诊断](D:/workspace/ego-data-pilot-2026-11-07/simulation_convex_v2/push_orientation_diagnostic_v2.json)。

## 机器人及ego参考接入

精确装配Hand2、手机和一张支撑桌面，保留原固定腕、20关节/执行器与识别参数。MuJoCo默认XML写出会舍入数值，故保存精确MJB及可重建MjSpec接口；482个模型数组重建逐字节一致。手机仅停放在手碰撞包络外20mm；编译及零步前向通过不代表抓握。加载接口：[scene_loader_v1.py](D:/workspace/ego-data-pilot-2026-11-07/hand2_stage1/scene_loader_v1.py)。手部材质priority会影响有效手物摩擦，不能直接将无手物体—桌面先验当成手物摩擦。

实际HOT3D1856第42–102帧已用固定官方UmeTrack实现生成61帧左手几何：完整22角、20 canonical/21 raw关键点、788顶点/1544面，以及世界/手腕/手机坐标。官方该版本内部skin使用前20角，末两角本窗口全0；保留原22输入及这一实现事实，未将其映射成机器人20角。原始手顶点到手机扫描三角的无符号距离只作近距候选，不能证明接触方向、穿透、力或摩擦。[参考几何报告](D:/workspace/ego-data-pilot-2026-11-07/ego_hand_geometry_v1/report.json)

下一几何子阶段固定首帧42，以解剖指对应的关键点/接触候选生成Hand2几何解。人体左手与机器人右手的差异显式保留；仅允许手机正旋转SE3与20个限位内关节，不暗镜像手机、不移动腕、不缩放资产。有限确定性起点、固定目标与预算的优化结果只称静态几何候选，所有失败及碰撞残差保留。进入真实夹持动力学后须另行验证接触支持能力、实际执行器范围和无物体辅助回放。

首次几何拟合已按固定frame42、3起点、原限位执行，5890次实际静态API/22.4s，未作积分或训练。按预声明无穿透优先规则选seed2，tip RMS14.24mm、食指距手机7.16mm，尚未共同贴合；seed1目标较低但存在0.351mm穿透，作为失败/待修候选保留，不偷偷改选。全部三候选图和状态均可复查：[几何报告](D:/workspace/ego-data-pilot-2026-11-07/retarget_geometry_v1/report.json)。这一步暴露了左右手形态、接触目标与优化可行性问题，不能将人体FK完成或可视贴近写成已完成重定向/抓握。下一步先诊断具体穿透部位和求解终止，再在新协议中改进接触可行性目标；原失败与权重不改。

读回诊断进一步定位：seed1负距在食指中节capsule−0.351mm及食指sensor capsule−0.004mm，拇指pad为+0.267mm；不能将中节穿透改称允许的pad预载。seed0/1触API预算，并非收敛证明；seed2仅因xtol停止，返回值等于起点，约1.49e-8m的手机微扰曾使native gap残差剧烈跳变，不能把返回0当可靠贴合证书。下一最小动作是在新命名空间按固定±10µm扰动验证距离接口的稳定性/运动界限，再使用固定pad目标、显式接触约束及与几何容差相称的差分步或site Jacobian。非接触指保留低权重风格目标；掌部、非目标link和桌面不得穿透。Hand2动态前须显式指定phone–hand contact pair的摩擦/condim/solref/solimp，防止原手部priority覆盖新物体设置，再先验证重力/力矩支持能力和无物体辅助holding。[诊断记录](D:/workspace/ego-data-pilot-2026-11-07/retarget_geometry_v1/diagnostic_readback_v1.json)

新更细物理步长对旧20ms控制周期意味着更多子步，必须在Hand2联合场景中实测数值行为与吞吐；旧5ms步长、旧转笔策略/世界模型、旧观测资格不能自动迁移。原native-only AFTER1000保存健康/共同指标评价等旧待办继续保留；本轮尚未开始新学习训练，原首健康checkpoint及ETA后暂停约定仍仅在下一项新训练时适用。
