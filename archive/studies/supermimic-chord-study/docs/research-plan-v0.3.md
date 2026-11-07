# 研究方案 v0.3：三种子证据后的推进顺序

本修订保留原目标：无机器人关节标签的ego演示，经感知、几何重定向与物理修正，得到遵守接触动力学的灵巧手轨迹，并检验共享模型能否降低新增轨迹成本。当前仍是原生Windows/RTX5090、右Wuji Hand2、固定腕、单笔的受控研究；真实ego与硬件阶段不因模拟结果而视为完成。

## 0. 2026-11-03 新种子确认后的当前结论

共同20mrad控制在真实采样与想象中均使用实际绝对目标，控制/回放/WM输入、初始化、EMA、来源和同预算合同已验证。新训练种子4/5/6的9次2000轮训练及27次固定独立进程评估现已完成，主终点仍为每个模型三次全过：

| 方法 | seed4 | seed5 | seed6 | 三次全过的训练seed数 |
|---|---:|---:|---:|---:|
| A1L | 3/3 | 0/3 | 0/3 | 1/3 |
| P_ONLY | 0/3 | 0/3 | 0/3 | 0/3 |
| A2 | 0/3 | 0/3 | 3/3 | 1/3 |

这是同一已见clip的前瞻训练种子复制，N=3；旧checkpoint的20mrad零样本结果与本轮单列。A2相对A1L三对shaft峰值过程均值变化为+11.274/−20.503/−23.669°，存在种子间权衡。A2九条50Hz采样关节记录均无hard-excess，但seed4/5仍因轴误差失败，未显示稳定整体任务收益；不能据此把20mrad称为连续状态安全保证。P_ONLY seed6仅手物穿透3.519mm失败，不能用良好跟踪覆盖该门槛。

训练队列实耗2h57m26s。A1L/P_ONLY/A2平均native STARTED墙钟892.086/1292.698/1348.856秒；等更新仍非等计算成本。详见[新队列原始复算](D:/workspace/supermimic-c2-confirmation-eval-study/outputs/analysis_v3/REPORT.txt)与[训练成本](D:/workspace/supermimic-c2-training-study/outputs/confirmation_training_cost_v1/ledger.json)。不按本轮质量继续搜索margin、替换seed或放宽20°/1mrad/3mm门槛。

下一阶段把重点放在观测输入与物理grounding。掌根几何候选加入手物soft非穿透后，独立碰撞复核发现手自碰仍13.551mm；最新候选同时约束原生过滤后的手物与手自碰，151帧的最大生成接触深度约0.522/0.505mm，但指尖拟合偏差升至11.45mm，8帧在80次函数求值上限内未收敛。未知gap保持NaN，未生成接触的优化残差0不代表间隙测量。它仍是未标定缓存数据下的几何折中，不能直接认作物理参考。

Frame0设计的零初速0.3s rest-hold已实际完成：61状态/60步前接触样本完整，overflow0仅读一次；物体最大位移446.680mm，超过20mm而失败，其余轴角0.353°、q超限0、穿透0.504mm及日志/外力门槛通过。它只检验新的静止干预，不是视频速度重建、完整3s任务成功，也不能单独否定运动轨迹可行性。先在CPU上对齐求解前接触力和随后状态，检查实际承载；再分别冻结运动初速与全轨迹无辅助动力学修正，记录实际q/qd/object twist/executed-u。见[grounding证据](D:/workspace/supermimic-observation-grounding-study/README.md)。A1L保留为成本基线，A2保留为待验证的接触意图项；真实ego、共享多轨迹协议与硬件契约范围不变。

同钟CPU诊断随后完成：最初三个求解区间仅一个finger2接触，最大向上力为重量5.90%，之后57个区间无手物接触；线动量平衡残差5.50e-7N，无执行器饱和。下一轮优先从观测几何提出显式接触意图假设，并设计初始承载/PD预载，保留本失败baseline；初速与运动条件分别冻结。未生成接触保持unknown，不把新假设当视频接触真值，也不从本次rest结果优先搜索阻尼或扩大训练。

该步骤现已有新证据：[观测表面诊断](D:/workspace/supermimic-observation-contact-intent-study/README.md)发现两指外法向夹角由source173.125°变为归一化目标111.152°，链2发生侧别变化。仅替换几何优化初值后，在固定物体和同一全手目标下找到两指3pairs候选，但其[真实0.3s保持](D:/workspace/supermimic-observation-warmseed-grounding-study/README.md)仍因65.159mm位移、41.350°轴角和3.872mm深度失败。当前q64与q32接触集都能平衡合力、不能同时平衡COM力矩；未导出静力预载，B未执行。完整扳手/接触位置条件应进入下一步候选设计，不能用接触数量替代。静力精确初态失败仅是诊断，后续还需依据设计初速和物体运动需求检验动态扳手与整段无辅助控制，不能把静止保持变成整个动态目标的替代品。

[共享多轨迹协议](shared-multitrajectory-protocol-v1.md)与[机器入口](D:/workspace/supermimic-chord-study/protocols/shared-multitrajectory-controlled-v1-bundle.json)也已落地：10条3987帧仅观测导出、同lineage分层/外部轨迹留出、真实源码接口和8项实现缺口、同aggregate更新与完整成本账均已CPU核验。它不等于共享实现或训练结果；三字段导出需新adapter，外部留出不得进入train replay/normalizer，逐clip初态/动态门槛与多bundle/新checkpoint迁移合同仍是下一阶段工作。

后续H123三指分支补上拇指材料约束，两套固定初值的六维物体扳手LP均可行，但实际固定qref保持仍分别因轴角31.331°、以及位置30.669mm/轴角42.819°失败。Canonical baseline的全手静力代理随后导出C2内合法PD目标，并经独立矩阵/符号审计。保持相同物理q0/零初速/物体/腕/实际八参数，仅改变固定u的唯一一次预载实验，通过原0.3s门槛：位置3.956mm、轴角16.287°、穿透2.183mm、实际关节越限0.626mrad。见[raw独立重建](D:/workspace/supermimic-observation-warmseed-grounding-study/outputs/h123_preload_independent_raw_review_v1/report.json)。这是已知QREF结果后提出的顺序探索，N1，不是盲法、稳定方法收益或完整动态参考资格；LP未强制软接触本构，不将见证力当实测力。

动态需求分析已按完整151帧旧几何候选计算固定掌任务的设计速度/加速度与所需扳手，原始有限差分和固定平滑敏感性分列。该重设计不是原视频惯性运动；71/151帧现有接触棱锥不能满足所需完整扳手，另外80帧的无上界力见证也不代表PD可实现。H123首帧没有拼入旧151帧轨迹。下一步应建立整段接触意图连续参考与明确初速，随后验证无辅助动态控制；静止通过不是整段训练的充分前提。见[动态需求与限制](D:/workspace/supermimic-observation-dynamic-wrench-study/outputs/dynamic_wrench_v2/REPORT.txt)。

严格三字段观测reader已落地，10条各[0,151)帧通过几何优化和真实原生SingleHandTrackingReader。原生link_pos_w取21个BODY原点，不能直接复用IK的21个宏骨架点（两种定义相差最多37.6mm）。1510帧中1508收敛，两帧未收敛保留；clip003/004骨方向峰值约94°、clip001参考q的有限差分峰39.19rad/s，因此只取得数据/ABI工程资格。该批窗口不同于既有物理clip007[200,351)，不能共享其首帧预载资格。多轨迹registry、逐轨迹目标、训练/留出隔离、reset/replay身份及迁移合同继续推进，未启动共享训练。见[完整批次证据](D:/workspace/supermimic-shared-reference-study/outputs/geometry_native_batch_v1/READY.json)。

新增[registry CPU阶段](D:/workspace/supermimic-shared-reference-study/docs/shared_cpu_registry_v1.md)进一步完成五套实际原生recipe加载和29项边界/负例检查。目标registry当前10/10缺少独立bundle，gather明确报错；请求初态、设计速度、原生goal FD、实际状态和u分开，旧窗口证据拒绝。Replay边界与精确身份/预算guard仅为CPU接口，未安装到原生collect/WM/normalizer/loader；旧训练入口仍保留单clip限制，尚未取得多轨迹训练资格。下一步先补物理参考与逐轨迹目标，再独立集成新训练入口，不能以接口测试替代真实训练证据。

已保存[下一轮短动态协议草案](D:/workspace/supermimic-observation-dynamic-wrench-study/proposals/short_dynamic_tracking_v1/README.md)：先构造一条16帧联合参考，q0硬固定、全段重复q0作初猜，禁止拼接旧q尾段。直接拼接的只读反例会产生55.393rad/s的qgoal跳变，并非实测u速率。下一CPU任务最多80次函数评估/120秒，不换参数补跑；连续曲线、非零初速、PD阻尼补偿、C2实际目标/15rad/s和独立窗口身份通过后，才考虑另审两条同新初态/同新u0的hold/track诊断。当前新参考、scheduler和物理资格均PENDING，GPU/训练未授权；0.3s新短任务不能替代原完整3s任务。

该CPU任务随后唯一执行完成，46次函数评估以ftol终止，300自由变量不含q0，16节点q硬限超量0。完整[新证据](D:/workspace/supermimic-observation-prefix-study/README.md)却显示：手物生成深度5.141mm、12/16帧超过3mm；二阶导数/quintic曲线节点间越限15.627mrad；计划C2目标更新峰28.485rad/s超过15；hold/track在参考状态上的原始PD力矩预测均超力限。它们不是实测物理失败或实际饱和，但足以使本协议的GPU前置合同失败。全部结果保留，无新GPU/训练、无改权重/导数/平滑补跑。下一设计应将连续范围与目标速率/力限纳入轨迹构造，并检查固定材料锚点是否与观测/机器人几何冲突；不能仅凭ftol、接口检查或较小点误差称可执行，也不能把该局部失败当整个运动或学习残差路线不可能。

随后另立[受约束三次B样条协议](D:/workspace/supermimic-observation-constrained-prefix-study/README.md)：P0固定、340个系数变量保留原拟合目标，显式系数盒、计划目标域/15rad/s、每ZOH区间PD Bernstein充分界，以及61时刻生成重叠≤2.999mm规划约束（原资格仍3mm）。只执行一次L1 LP→一次SLSQP，正常返回terminal作为唯一候选。LP成功，SLSQP到80轮上限；独立复算确认全曲线范围、最终U32控制速率11.449rad/s、零C2限幅和track/hold参考PD证书通过，几何仍5/61时刻超3mm，峰3.194716mm，primal未取得资格。没有新动态GPU/训练或替换候选。凸包/Bernstein/hold参考限制充分非必要，61点生成约束不保连续净空或实际动力学；下一版应区分非线性可行性恢复与拟合，记录每个major违规/活跃分支，不能仅延长本80轮或改权重重试。完整151帧、学习残差/CHORD和共享成本目标继续保留。

## 1. 已建立的证据

原方案、源码审阅和输入/控制契约见[v0.2](research-plan-v0.2.md)。独立环境已完成方块与笔的加载、短训练、保存恢复、播放和编译验证。原生PPO仅有方块，不能把笔的世界模型训练称作已验证的笔PPO。

正式控制契约同时限制真实与想象中的绝对目标，EMA残差保持原定义。最终任务用自由物体、固定初态、完整3秒、无重置/物体辅助的Warp评估。目标限幅不保证真实关节状态可行，二者分别检查。

[三种子报告](../outputs/replication_seeds123_comparison_v1/REPORT.md)使用训练种子1/2/3和共同评估种子0；seed0只作pilot，151个时刻不增加独立样本数。

| 口径 | A1L | A2 | 当前解释 |
|---|---:|---:|---|
| 2000轮、等真实采样/更新 | 1/3通过 | 0/3通过 | 未显示整体任务率收益 |
| 原2000轮实验中1200秒前可获得模型 | 1/3通过 | 1/3通过 | 成功种子不同，不能称稳定改善 |
| 最大笔轴误差的配对差值 | — | 三对均降，平均−7.760° | 描述性局部改善信号 |
| 平均笔轴误差的配对均差 | — | −0.0139° | 接近零，不能泛称所有跟踪改善 |
| 实际关节状态越界 | — | 1.46–3.31mrad | 三次均超过冻结1mrad容差 |

A2当前实现训练平均约为A1L的1.493倍，Torch峰值分配约7.62GiB对0.54GiB。等更新不是等计算成本；工程失败、预训练、评估和教师成本须另外保留。没有共享模型摊销或真实ego提速结论。

## 2. 接触目标的能力边界

上游15维head监督手指自碰撞，不是CHORD手物接触。独立冻结目标用物体COM、朝内法向和单位六维方向集表示几何扳手支持能力，半径只缩放力矩一次；它不估计真实接触力，也不证明动力学可行性。

A2为想象阶段的几何先验：`phase_weight * support_match - 0.1 * penetration_prior`，外层仅乘一次`dt=0.02`。训练64方向，独立真实接触评估512方向；不惩罚缺失的参考拇指身份，也不把unknown当无接触。几何代理仍有ghost接触及局部动力学梯度资格失败；原同Warp WM局部梯度probe只支持其特定checkpoint和少数状态，不能扩展为全部模型可靠性的证明。

内部接缝和最大值归约NaN已修复，但standalone编译先验的精度门槛仍失败。正式方法使用eager几何先验与编译的原生子网，不能把训练完成写成该编译精度门槛通过。

## 3. 近期先做机制消融，再研究新控制器

**P_ONLY**使用同一冻结A2返回的穿透tensor，直接返回`−0.1P`，不通过相减抵消支持项。保持几何、采样、阶段、原生训练、控制、数据与预算，只移除支持吸引。独立工作区为`D:/workspace/supermimic-chord-penalty-only-study`；它是在三种子结果已知后提出的探索性followup，须在新臂运行前另行冻结，不能倒称原实验的预注册部分。

P_ONLY−A1L比较无支持项时的穿透正则增量；A2−P_ONLY比较相同穿透背景下增加支持项的增量。缺少support-only臂，不能估计完整2×2交互或无条件支持主效应。

[三臂完整报告](D:/workspace/supermimic-chord-penalty-only-study/outputs/three_arm_seed123_comparison_v1/REPORT.md)现已完成。主预算通过数为A1L/P_ONLY/A2的1/2/0（各3个种子），次预算为1/2/1；P_ONLY两预算成功种子不同，不能合成3/3。主预算中A2相对P_ONLY三对最大笔轴误差均降低，同时三对实际关节超限均增大。P_ONLY所有实际关节门槛通过，但其penalty是手物穿透代理，不含实际关节barrier，不能从结果倒推直接因果。

这轮复用已经完成的A1L/A2作为历史对照，三臂执行时段和缓存顺序没有完全平衡；保持配置和源码一致不能消除该限制。因此该followup用于机制诊断，后续确认性实验仍需独立、平衡执行，不能将全部差异无条件归因奖励项。

在机制结果之外，另研究真实状态可行性控制。见[独立控制审阅](control-feasibility-followup.md)和[越界诊断](../outputs/replication_joint_limit_diagnostics_v1/README.txt)。原生奖励没有直接的实际q越界barrier；WM的3mrad loss归一化尺度不是精度下界或死区，不能据此归因越界。

新控制条件必须显式使用q/qd等因果可得量，真实和想象共享同一映射，replay与WM输入保存实际执行的绝对目标，并单独记录不可行情况。不得clamp物理状态、施加对象辅助力或放宽本轮门槛。新控制器不混入当前P_ONLY，须独立冻结、测试和对照；连续时间及硬件可行性另证，50Hz门控不能代替它们。

当前先实现[有限制动诊断](D:/workspace/supermimic-control-feasibility-draft/calibration_plan_v1.json)：三个seed1最终模型各取0.8/1.6/2.4秒自然状态，共9个快照；每个快照3次基线恢复和3条单关节反初速脉冲，共54条0.2秒分支。基线恢复和逐子步复现失败时，该快照不产生制动推断。固定upper侧定义与速度符号分开，初始向内、低速、无停转及非单调案例均保留。0.1rad脉冲不能证明较弱C1规则的制动能力，m0/a参数保持null；该诊断无训练更新。

v1已按固定门槛失败停止：只运行1条baseline、0条pulse，53条未运行；完整恢复、控制/时钟、外力及overflow检查通过，0.2秒回放偏差仍超门槛，故无制动资格，参数未启用。见[partial证据](D:/workspace/supermimic-control-feasibility-draft/outputs/braking_microprobe_analysis_v1/partial_terminal_review.json)。新增5ms采集的A1L donor还出现历史PASS→本次FAIL的关节门槛变化；它与原评估的采集流程不同，不直接混池。原三臂单次物理记录保持不变，先研究原uninstrumented evaluator的fresh-process重复性，再决定新控制器或扩量训练。

随后按新冻结协议完成原evaluator五次fresh-process复验，同一个A1L seed1 final/eval0/精确初态：F/F/F/F/T；前4仅实际q门槛失败，峰值1.45–1.69mrad，第5为0.73mrad。全部物体跟踪/目标限幅/不掉落及其余门槛通过；[独立CPU报告与图](D:/workspace/supermimic-control-feasibility-draft/outputs/plain_evaluator_repeats_analysis_v1/REPORT.txt)保留全部结果。相同输入仍有进程间轨迹变化，不能全归因5ms hook；原因未定位。训练seed N仍1，本次1/5不作为方法成功率，也不改原三臂历史记录。下一确认比较须预先定义每checkpoint重复和进程顺序，分别报告训练随机性与评估重复性，禁止best-of-repeat。

当前新增[C2固定目标内收零样本诊断](c2-target-inset-exploration.md)，与C1分开：固定检查点比较0/5/20mrad三个工程剂量，各3次、Latin顺序，全部条件同wrapper和采集，保持原t0与状态验收。它只改绝对目标，不使用未合格的制动常数、不训练，也不保证实际状态。

C2 v1与独立容量审计v2均已完成并分别保留；v1容量UNKNOWN，v2九次累计overflow均0且无重置。v2原任务通过数为0/3、3/3、3/3；六条内收轨迹在记录的50Hz状态上均无关节hard-excess，物体跟踪及其余原门槛通过。见[v2报告](D:/workspace/supermimic-target-inset-study/outputs/analysis_capacity_v2/REPORT.txt)。这只支持该已见A1L检查点的工程信号，不支持CHORD收益或连续时间/全域保证。

[共同5mrad控制器组合评估](D:/workspace/supermimic-c2-portfolio-study/outputs/analysis_v1/SUMMARY.txt)现已完成原9个最终模型各3次，共27条。身份、控制、容量及原指标重算均通过；A1L/P_ONLY/A2各有1/3、2/3、1/3训练种子三次全过，训练N保持3。A2最大shaft误差仍三对改善，但seed1/3的actualq失败未消失；5mrad并未从单检查点结果推广为通用状态可行性保护。未实现并验证real/imag训练一致性前，不把本零样本执行干预称为新的已训练方法。后续先固定补查原已声明的20mrad剂量，所有新结果单列，不搜索最好剂量/重复或改原门槛。

## 4. 无本体输入与多轨迹仍是后续交付

20mrad补查现已完成独立27条评估：A1L/P_ONLY/A2三次全过的训练seed数为1/2/3（各N3），A2九条均无采样关节hard-excess；相对5mrad，A2三个checkpoint的shaft峰值过程均值分别增加约0.668/0.794/0.733°。A1L seed3仍超限12.156–12.274mrad。见[分开保留的剂量报告](D:/workspace/supermimic-c2-020-portfolio-study/outputs/dose_comparison_v1/REPORT.txt)。这支持继续工程研究，但两轮顺序、已见结果及数值/缓存背景限制仍在，不取最好剂量或将54过程混池。

下一阶段为20mrad训练控制映射的CPU/GPU合同短测，再事前冻结新种子4–6的A1L/P_ONLY/A2比较。真实、想象、replay、WM输入和PD估计须用同一实际绝对u，EMA记忆保留原语义；旧checkpoint加载必须防止静默漏装控制器。短测使用已见seed0并实际触发策略/价值/H8先验，正式仍2000轮及原采样/更新预算，不修改WM损失或任务门槛。新工作区`D:/workspace/supermimic-c2-training-study`隔离原结果。

当前源为机器人运动学参考；MediaPipe关键点不是原始ego视频谱系证明。后续受控输入可隐藏机器人关节标签，只保留人手/物体观测，检查重定向链路；这仍不能称真实ego验证。真实数据接入需原视频父ID、相机坐标/尺度、物体几何与物理参数不确定性、原腕轨迹、接触已知/未知与置信度。

现已建立[单clip观测数据视图](../outputs/observation_only_fixture_v1/manifest.json)：151帧只保留source关键点、source物体/腕位姿与时间；机器人joint/link、phase、执行target和参考contact均不进入候选输入。未知置信度使用known=false，不补造数值；不把fixed-wrist重设计后的参考作为观测。CPU exact-allowlist、禁止标签全部替换为NaN/无意义值后输出不变、非法时间拒绝三项[合同检查](../outputs/observation_only_fixture_contract_v1/report.json)通过，脚本为`export_observation_fixture.py`与`check_observation_fixture.py`。这只是API数据边界，原文件仍在磁盘；相机谱系、坐标/尺度与landmark语义未验证，尚未形成物理重定向或真实无本体结果。

新增[landmarks-only几何IK原型](D:/workspace/supermimic-observation-retarget-study/README.md)：从该视图仅读time/21手点/object，不读cached wrist或源robotq；使用已知机器人运动学和显式掌basis固定腕重设计。151帧的平均骨方向误差6.13°，归一化目标指尖均误差16.2mm，仍不足以支持接触关系。8项数据边界/坐标变换检查与18项独立FK/误差复算通过，physical_reference_qualified保持false；它推进了几何接口，不能作为已完成动力学grounding或真实ego验证。

移动腕逐帧消除会改变惯性与重力关系；保留原轨迹并筛选固定腕适用性，不能满足时明确为任务重设计或转到机械臂版本。当前clip007[200:351]即任务重设计，只约64°可观测笔轴运动，不是完整转笔或轴向roll复现。

掌根v2几何分支修订了v1将移动link2当作固定链根的问题，使用link1固定原点和跨两关节的首宏段；其自身归一化目标的指尖均偏差4.99mm，平均骨方向6.36°，q有限差分峰12.49rad/s。新旧目标不同，不能据差值声称解剖标定或相同目标优化改善。8项合同检查通过，未推进物理，camera/scale/接触/任务重设计资格仍未解决，physical_reference_qualified=false。

新增几何筛查显示该候选151帧均有约5.05–15.11mm的最深手物成对重叠，最近负距离经过segment与原生碰撞查询复核；非最小非负距离的11处异常另行保留，不将完整距离表称为已合格oracle。该候选不得直接进入当前训练参考。下一步需同时约束观测拟合、手物相对位姿和非穿透，显式保留尺度/掌锚不确定性；未知接触仍unknown，不能通过吸引奖励补造成known contact。之后再验证无对象辅助的初始化与动力学grounding，并保存实际q/qd/object twist和executed targets。当前C2三臂训练继续使用原冻结reference/contact bundle，其结果不替该输入候选取得资格。

先建立足够可靠的单轨迹质量，再做1/5/10/20条轨迹的共享模型和策略成本曲线。原父ID未知的10个clip仍按同一保守split group处理，不以切片/扰动副本作独立泛化集。物理教师须保存真实初态q/qd/对象twist与实际目标，不能把q_actual直接当PD target；记录`target[k+1]`对应`x[k]→x[k+1]`，教师及失败成本均计入。

硬件子版本、观测、延迟、执行器标定与闭环物体感知在真机阶段单独核对，本研究不会自动控制实物或改系统驱动/CUDA。

## 5. 本轮交付状态

| 交付 | 状态 |
|---|---|
| 源码审阅、环境、数据/坐标/控制契约 | 已有可追溯产物 |
| 单轨迹无辅助物理评估与三种子A1L/A2 | 已完成，混合信号及失败完整保留 |
| 支持域数值实现、独立512评估、代理/梯度诊断 | 已完成当前检查，资格限制未消失 |
| P_ONLY机制消融 | 三种子训练、六个物理评估及三臂报告已完成；历史对照限制保留 |
| C1状态速度制动条件 | 未过回放资格，0pulse；m0/a未启用 |
| C2固定目标内收零样本条件 | 同A1L检查点v2中0/5/20mrad通过0/3、3/3、3/3；共同5mrad全27评估后方法训练seed全过数1/2/1（各N3），未证明整体CHORD收益 |
| 策略网络导出 | 三个seed1 Actor的ONNX参考执行CPU数值检查及独立复算通过；完整控制器/部署另证 |
| 无机器人标签几何接入、共享多轨迹成本 | 几何IK原型已测，物理参考与共享成本实验尚未合格/完成 |
| 真实ego与真机 | 待可用数据/硬件条件后推进 |

本方案不以改善辅助分数、单次成功或任意放宽门槛作为完成标准。训练启动后可按用户要求，在首检查点健康和ETA确定后暂停Goal，保留明确授权的独立后台队列。


2026-11-03执行更新：C2共同20mrad训练合同已过nativeCPU/Warp命令、CPU/CUDA投影AMP梯度及三臂seed0实际H8/WM/actor/critic短测；正式seed4–6九次同预算fromscratch训练已启动，首100的51核心+17actualu审计通过。Goal按用户条件暂停，detached队列继续。主质量评估仍待final2000+各三次过程，不能把首检查点健康当作任务成功。见[当日记录](progress-2026-11-03.md)。
