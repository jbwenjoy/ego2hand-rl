# 研究方案 v0.24：原生软接触、匹配基线与短时预测控制

2026-11-07。目标为 active；本回合属于 progress。完整范围承接 [v0.23](research-plan-v0.23.md) 与原 SuperMimic + CHORD 目标，包含已获用户同意的 ego 数据、碰撞资产、动力学先验及仿真验证。没有启动新学习训练；下次新训练的首个健康 checkpoint、ETA 后暂停约定继续有效。本版及其新实验不改写已有协议、失败或资格。

## 当前结果与边界

实际 HOT3D ego 数据及其官方手机资产已进入参考几何和机器人仿真探索，但尚未进入机器人策略/世界模型训练。官方视觉 mesh 保留；碰撞使用先前验证的单凸体。质量0.2kg、手物摩擦0.7、桌物摩擦0.6及均匀包围体 COM/惯量仍是工程先验，未获实物校准。无手台架资格仅在这些假设下成立，不能代替 Hand2 持握资格。

本回合新增原生软接触评估、预载、六维扳手残差及闭环可行性工具；未取得合格持握。相同初态下，恒定控制两步长都约0.1225秒失握；特权真模型 H2 预测控制在细步长约0.12844秒失握。三例均未运行到原2秒与尾窗，不能称方法改进、步长收敛、ego轨迹重定向成功或学习收益。

## 修正静止平衡门的适用范围

新原生 evaluator 每次使用独立 MjData、零速度、真实合法PD命令和一次 forward；直接输出实际接触、20关节与物体加速度、COM Jacobian、世界惯量及完整力项。四次 cold 契约检查重复一致；cold 没有生成接触，手机自由关节约束/被动/驱动/施加力均为零，COM加速度为重力，没有隐含物体支撑。[契约](D:/workspace/ego-data-pilot-2026-11-07/native_equilibrium_v6/four_forward_contract_v1/report.json)

随后采用预先固定的有限初始化：按实际手侧法向与Jacobian线性闭合到约−25µm，关节增量受真实范围及±0.02rad约束。指定pad初始压缩假设为最多0.1mm，另立新门，不能回写成旧正间隙协议通过。seed0两个pad建立约1.102/2.486N反力，但物体/手指加速度仍明显；seed1出现非pad相交并保留失败。[预载](D:/workspace/ego-data-pilot-2026-11-07/native_equilibrium_v6/preload_v1/report.json)

v6联合姿态/控制原生静态搜索实际420次fresh forward、1031次接触力读取、6次最终完整几何检查，0积分/训练。两seed全部保留，严格静止候选为0。seed0前三个候选通过22手机对/189自碰对、真实控制、加载及重复检查，但均不满足三项加速度门；rank0的COM/角/关节加速度∞值约3.9145m/s²、137.1408rad/s²、3.8225rad/s²。优化器的xtol状态不等于物理平衡或全局收敛。[搜索](D:/workspace/ego-data-pilot-2026-11-07/native_equilibrium_v6/completed_v1/report.json)

**严格全零速、全零加速度不是当前软摩擦持握的必要条件。** 固定MuJoCo3.11源码显示：有限正R、无surface velocity/adhesion及无NoSlip后处理时，condim3 pyramidal contact在全v=a=0时正负射线同值，切向合力为0；小滑移或相对加速度可通过软阻抗产生切向力。仅有v=0不意味着Ft=0，已保存preload恰有零速非零Ft与非零qacc。法向合成也可能托重，故不能据此说所有静止平衡都不可能。严格v6门与其失败原样保留；新动态入组只把三项静态加速度作为描述，原2秒最终门不变。[条件、推导及保存证据](D:/workspace/ego-data-pilot-2026-11-07/native_soft_friction_audit_v1/report.md)、[固定版原生源码](https://github.com/google-deepmind/mujoco/blob/b85fdca54f0e0038b804af146a0b4e94199e00d0/src/engine/engine_core_constraint.c#L1677)。

不能把静态搜索失败笼统归咎于数值噪声。对rank0的关节、控制、平移及旋转四个方向，三级中心差分共26次fresh forward，物理残差方向导数随步长缩小保持一致；这只覆盖这四个方向，不是全Jacobian光滑或可行性证明。[方向检查](D:/workspace/ego-data-pilot-2026-11-07/native_equilibrium_v6/directional_derivative_probe_v1/actual_v1/report.json)

## 六维支持接口与状态时间合同

新增同时处理合力和力矩的缩放∞残差LP：F尺度为mg=1.962N、M尺度为mg×0.077m=0.151074Nm，保留原真实力矩界、设计摩擦内接锥及按手指汇总的3N预算；不能按面片采样数量增加预算。严格力/矩门独立于有限缩放残差。解析夹具4/4通过；旧v5两个保存姿态的纯LP重算仍未通过严格六维支持，不以较小综合数值称可承载。[接口夹具](D:/workspace/ego-data-pilot-2026-11-07/wrench_residual_v6/preparation_v1/report.json)、[旧姿态诊断](D:/workspace/ego-data-pilot-2026-11-07/wrench_residual_v6/saved_v5_diagnostic_v1.json)

新增纯 COM/world state58 正逆合同：p_COM=p_root+R·ipos，ω_world=R·ω_body，v_COM=v_root+ω_world×R·ipos；保留完整R及20关节位置/速度。控制周期20ms对应64/128整数tick；最后PRE相位分别19.6875/19.84375ms，native clock独立记录。4/4解析检查通过，旧cold零速保存状态读回COM/Jacobian×速度一致；这些检查没有新模型或积分，也不授予旧WM/actor手机资格。[合同](D:/workspace/ego-data-pilot-2026-11-07/phone_state_time_contract_v1/report.md)

## 相同初态的恒定控制与闭环实测

两分支共同使用v6 seed0 rank0的原始qpos27、zero qvel26及ctrl20、同一冻结MJB。除三项静态加速度描述外，原加载/预算/控制/重复/模型与22/189几何门保留。原固定腕、手指/手机尺度、执行器、材料、求解器及步长均不调参；无物体外力、重力补偿、pose reset、等式或mocap辅助。

恒定控制两步长0.3125/0.15625ms共1176实际步，分别392/784步在约0.1225秒首次超过50mm提前终止门；末角约32.86/32.74°，最大穿透0.0196mm，无非pad/桌面支撑、自碰穿透或warning。两个末态差只是早停前缀描述，不能给2秒步长一致资格。保存算术复核9/9通过；旧复核v1的list字段适配失败另保留，未重跑物理。[基线实测](D:/workspace/ego-data-pilot-2026-11-07/hand2_constant_u_baseline_v1/cpu_v1/report.json)、[保存复核](D:/workspace/ego-data-pilot-2026-11-07/hand2_constant_u_baseline_v1/saved_recheck_v2/report.json)

预测分支使用mj_copyData克隆完整当前状态，包含warmstart/history，不在分支恢复时forward。首次相同命令的两个40ms分支共512规划步，轨迹、接触及20/40ms完整积分状态位级一致，42/42检查通过，实际主状态未推进。[分支契约](D:/workspace/ego-data-pilot-2026-11-07/native_mpc_v1/two_branch_contract_v1/report.json)

新的有限预算MPC是**特权真模型可行性诊断**。每20ms规划未来两个区间，只优化原thumb/index前8个位置命令，其余12个保持原命令。每区间trust相对上一执行命令±0.05rad并与真实控制界相交；有限差分0.01rad、最多2次bounded Gauss–Newton及精确line search。目标固定为原始COM/完整R，包含COM速度、世界角速度、弱姿态及持续接触项；执行候选须通过原碰撞/非pad/table/self/关节安全上限。没有安全局部incumbent只表示此有限搜索未选到安全控制，不证明全局不可行。3N只是软设计目标，实际反力未裁剪，原动态门没有3N瞬态上限。[锁定协议](D:/workspace/ego-data-pilot-2026-11-07/native_mpc_v1/mpc_feasibility_protocol_v1.json)

首跑进程正常退出、stderr空。实际执行822步、0.1284375秒后失握；末COM漂移50.1066mm、转角约41.15°，无非pad/table/self、关节越界或warning。另有161条规划分支、41,216规划步，不能计成实际控制步或学习预算。前六个完整20ms区间的实际完整积分状态与所选预测逐位相同；第七个区间仅54实际步，不能给完整区间资格。它排除了本次完整区间内的预测/执行接线不一致，但不验证真实物体参数。持握全部资格保持false，未运行粗步长确认。[完整结果](D:/workspace/ego-data-pilot-2026-11-07/native_mpc_v1/feasibility_fine_v1/report.json)

0.128秒与基线0.123秒的差别不足以构成稳定性或方法收益；角度表现也未改善。短H2、有限局部方向、8命令及固定其余指都是此oracle的限制。不能由失败断言所有姿态、参考或控制器都不可能持握，也不能据此直接启动SM/CHORD收益训练。

两份保存审查分别13/13通过，核对实际822步/419条接触、7×23条分支、全部真实控制范围/每区间trust/其余12命令、原门及首次早停。六个完整区间的预测300维向量SHA与记录的实际SHA一致；独立保存审查可逐位重建time/qpos/qvel并核显式POST观察，未另存的actual warmstart/HISTORY/solver字段仍以运行源码及日志声明为证，不冒称独立重建。[算术与对比图](D:/workspace/ego-data-pilot-2026-11-07/native_mpc_v1/saved_recheck_v1/report.json)、[来源计数审查](D:/workspace/ego-data-pilot-2026-11-07/native_mpc_v1/saved_count_review_v1/report.json)

保存机制分析显示，七次所选H2代价都比“相同当前状态继续上一命令”的分支低，但不是继续原始u0的同源反事实。前两实际20ms区间的双pad>0.1N覆盖仅18.75%/5.47%，40ms以后各区间为0；最初连续双pad加载仅约2.19ms，最后零星双pad在约29.53ms。因此cost降低没有维持接触。相同122.5ms时刻，MPC漂移约44.84mm、恒u约50.08mm，但转角约38.41°、恒u约32.74°，不能只选择漂移一项称改善。累计命令偏移可达0.2rad，仍满足每一步相对上一命令的0.05rad trust；它不是相对u0的总偏移上限。[机制分析](D:/workspace/ego-data-pilot-2026-11-07/native_mpc_v1/failure_mechanism_saved_v1/report.json)

## 当前接触几何的乐观受力上界

针对相同rank0初态，另行冻结并执行两次纯保存LP：保留全部3个原生generated contacts（thumb1、index2），使用各实际frame、物体侧符号及原生μ=0.7的四射线点力锥，不增加扭矩。忽略手部力矩/PD和软接触本构给出乐观理想静力能力；第一例每指总Fn≤3N，第二例取消该上限，其他点/法向/摩擦完全相同。没有重建模型、forward、积分、姿态搜索或挑选其他时刻。

两例最优六维缩放∞残差都约0.282215（差约4e−12）；最优每指Fn约1.540/2.648N，3N未激活。联合最优解仍有约[−0.5537,+0.5537,−0.5537]N合力残差、约−0.04264Nm的Mx/My残差；数值不等式违约≤1.4e−11。**在这组固定点/法向/μ的理想点力模型内，放宽3N并不能补足六维重力支持。** 这支持优先修改接触几何/参考候选，不能把失败唯一归因于3N预算或PD控制。它也不证明其他接触面、姿态、微运动或所有动态控制不可行；LP没有授予实际软模型平衡/持握资格。[两例原始结果](D:/workspace/ego-data-pilot-2026-11-07/native_equilibrium_v6/wrench_upper_bound_saved_v1/actual_v1/report.json)

## 后续推进顺序与完整目标

本回合已从保存轨迹明确接触早失、合法控制及短期cost与姿态的取舍，并完成当前接触几何上界诊断。下一步不再盲目延长同一失败初态或放宽2秒门。

接着在原固定腕下重审**参考的接触几何与支持扳手**：从实际CAD凸面/边界构造有范围和法向的工程面片候选，把ego的近距区域与机器人可用面片映射；每个候选核完整几何、真实力臂、每指总法向预算与6D支持残差。没有CAD语义/实物标定时仍标UNKNOWN，不把整个sensor mesh当已校准功能指腹，不凭空增加源中未支持的第三接触，也不授予任意点扭矩。需要改变安装姿态、任务接触来源或使用第三指时，必须另立具名条件及匹配对照，保留当前失败。

在新的几何可行候选上，再明确分开接触建立、预载与动态保持；用有限短rollout评价真实反力持续性，允许原生软接触自然建立小滑移。原始零速qacc继续保存为诊断，最终仍必须满足原2秒、5mm/5°、尾速度/覆盖率、穿透/非目标接触和warning门。通过名义持握后才做双步长、质量/摩擦/COM/惯量先验鲁棒性及61帧时序。

后续SM/CHORD还需新的手机Reader/完整姿态reward、20ms J/K/残差与子步标签、WM接触及梯度校验；xi46与旧pen长轴观察不能直接继承手机资格。同预算需同时报告控制区间、真实积分、WM/AD/策略更新及walltime。完整目标继续包括受控转笔、ego机器人参考、跨轨迹对照与独立泛化、真实资产/动力学校准及硬件观测契约；本回合没有缩减范围或宣称完成。
