# Rough-prefix SuperMimic / CHORD 学习接入

当前新物理FD探针已完整执行：16个新donor快照、560个分支、2480个scheduled子步，48baseline过原门槛，物理独审194/194；独立CPU三个WM完成48个名义预测与768行共享物理方向对照，保存结果独审30/30。128方向中93稳定、30不稳定、5平坦/低信号；稳定方向同号率A1L83.9%/P_ONLY80.6%/A274.2%（两epsilon均如此），只作单源短窗口局部诊断。原空数组setup失败/消费配置路径拒绝保留，helper3及另存版本修正，没有覆盖或挑选旧物理成绩。Goal active，无GPU作业。[结果与边界](D:/workspace/supermimic-rough-prefix-learning-study/docs/current104_physical_gradient_findings_2026-11-04.md) · [图](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_WM_summary_v2/physical_secants_WM_summary.png) · [full151缺口](D:/workspace/supermimic-rough-prefix-learning-study/docs/full151_integration_gap_current104_v1.md)。

当前三臂 A1L/P_ONLY/A2 的协议4、seed0、2000iter 同预算训练均已结束，各自唯一一次最终原生评估通过该0.3秒窗口原13项物理门槛。A0及三臂10iter历史结果仍仅笔轴失败。新增3世界模型×4保存轨迹×15转移的180次CPU一步预测已完成，独立保存审计68/68通过；175/288内部差分因动作边界被跳过，真实物理梯度和完整运动资格尚未取得。Goal active，当前没有运行中的GPU任务。真实ego数据尚无，独立ego N=0。

同一legacy clip007[200,216)来源、retarget variant `H123_cubic_minimax_restoration_filtered_prefix16_v1`，独立ego N=0。上一阶段direct-Warp H0/T0各一次均轴误差失败，结果不变。本模式允许学习修正粗参考，仍保持physical_reference_qualified=false，不继承shared10或旧151帧任务资格，不迁移旧checkpoint。

最新证据：[三臂最终物理对照](D:/workspace/supermimic-rough-prefix-learning-study/outputs/matched_primary_three_arm_comparison_v1/report.json) · [180次世界模型预测与独立68项复算](D:/workspace/supermimic-rough-prefix-learning-study/outputs/WM_CPU_diagnostic_saved_independent_v1/README.md) · [当前104物理探针API边界](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_probe_interface_review_v1/REVIEW.md) · [合法方向CPU准备，尚无物理执行](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_directions_CPU_v1/report.json)。详细成本、失败记录和下一阶段见[11月4日进展](D:/workspace/supermimic-chord-study/docs/progress-2026-11-04.md)。

## 已完成的CPU证据

| 组件 | 证据 | 范围 |
|---|---|---|
| 真实/想象控制器 | [68项检查](D:/workspace/supermimic-rough-prefix-learning-study/outputs/controller_cpu_contract_v1/report.json) | baseline_u15 + 原EMA残差，重建后hard/control/C2；format104原子保存及旧格式拒绝 |
| Runtime | [READY与40项检查](D:/workspace/supermimic-rough-prefix-learning-study/outputs/READY_runtime_v1.json) | 解析速度权威、selected reset、15interval、replay、dead-row梯度/criticnorm隔离；Env/Entity I/O明确为CPU stub |
| 输入包v2 | [36项独立检查](D:/workspace/supermimic-rough-prefix-learning-study/outputs/input_bundle_independent_cpu_v2/report.json) | 三包/native/contact bridge来源和语义join，资格、速度、baseline及哈希负例 |
| 新接触目标 | [28项工厂检查](D:/workspace/supermimic-rough-prefix-learning-study/outputs/contact_binding_cpu_v1/report.json) | savedR几何和H123声明意图，human接触全unknown；unknown负例精确退化为P_ONLY值/梯度 |
| 原生A0测量和评分 | [36项独立CPU检查](D:/workspace/supermimic-rough-prefix-learning-study/outputs/independent_native_a0_probe_cpu_v3/report.json) | 60求解器步前相位、15 Manager forward、速率/原门槛、partial与容量UNKNOWN；未创建环境 |
| 默认CLI与监督器 | [30项独立检查](D:/workspace/supermimic-rough-prefix-learning-study/outputs/entry_supervisor_independent_cpu_v1/report.json) | CPU默认不导入Torch/Warp，不创建原生输出；参数/旧checkpoint/越权角色负例 |

监督器后续 [47项独立检查](D:/workspace/supermimic-rough-prefix-learning-study/outputs/entry_supervisor_independent_cpu_v2/report.json) 补齐了watch前无child失败终态、外部C2源码归档以及CPU预检retained-Popen超时树清理。检查器/未执行的提案不再通过全目录glob误绑入GPU执行依赖；显式执行源闭合后才生成新config。

## 数据与算法合同

[输入包](D:/workspace/supermimic-rough-prefix-learning-study/outputs/input_bundle_v1/bundle.json) 绑定原三包、独立native16、canonicalR、模型/context、contact参考桥和来源收据。解析qd0与nativeReaderFD0最大差11.938072rad/s；readerFD保留诊断，runtime在fixed-wrist配置后、首reset前安装解析joint qd和quintic world v/omega。真实状态及WM标签来自模拟测量，不用参考速度teacher-force。

真实与想象目标从baseline_u15 + 原processed residual重建，再作原hard/control交集与一次20mrad C2。c1/c2仍是残差，不能储存u0−q0。零残差不改变baseline；新的format104完整记录baseline/init/velocity/phase/source和控制signature，拒绝旧103/3或exact resume。

16结点只执行phase0…14的15个20ms区间；state15为终端观测，不执行u15。自然终点done+timeout但不是failure标签，terminal transition保留，后reset episode不进入同一监督窗口。H8结束后的有限占位不影响active actor梯度、critic归一化或loss；所有方法采用相同新语义，实际有效样本数及分母另记。

接触目标是声明的机器人几何意图，不是人类观测接触/实测力。A1L无prior，P_ONLY使用精确-.1P，A2用固定support−.1P，dt=.02仅在policy imagination外加一次。原代理的局部梯度/编译精度失败边界保留，CPU输入梯度通过不升级为动力学或力预测资格。

## 下一次原生检查

[冻结工程协议](D:/workspace/supermimic-rough-prefix-learning-study/protocols/rough-prefix-v1.json) 首先只允许一个world的native Manager+PenSimReader A0，无训练Cfg/Runner/optimizer。保持实际八字段、设计非零初态、默认mocap、零残差和15个baseline目标。每sim.step后、Manager末forward前记录61状态/60求解器样本，保留15个postinterval forward，末尾sticky overflow读一次；质量越限不截断，数值硬错误保留partial/UNKNOWN。API body-angular算术容差1e−6rad/s不是物理标准放宽。

新A0与后续训练/评估采用相同native cadence；历史directWarp T0单列。单次A0120s watchdog覆盖Popen/导入/setup/worker，soft110s，无retry/compile fallback。通过工程、容量与精确OS PID/创建时间后，才可另授权依序A1L/P_ONLY/A2各一次10iter smoke，质量结果不用于选择重跑。

正式长训练仍需新协议及完整checkpoint/evaluation合同；当前工程协议不允许primary_matched角色。启动新的长训练后，按用户要求在第一个健康checkpoint和ETA确认后暂停Goal，已授权队列继续。CPU与短时smoke不等于完整151帧/真实ego泛化或CHORD总体收益。

## 首次原生接口证据与修正

[首轮A0计划](D:/workspace/supermimic-rough-prefix-learning-study/configs/a0-native_a0-v1.json) SHA `df66d6ff…45d8360` 正常启动了原生Manager并通过nominal MJB及actual8两次读回，但在首reset最后的 `ObservationManager.compute(update_history=False, env_ids=ids)` 触发真实API断言。[失败记录](D:/workspace/supermimic-rough-prefix-learning-study/outputs/native_a0_v1/failure.json) 保留；尚未进入probe，0轨迹采样步/0优化器更新，物理质量 **NOT_ASSESSED**，不写成任务FAIL。

实际worker43088/launcher19912，OS创建10:27:37.161366Z；worker9.897090s/watch10.954s为嵌套口径，exit1/no timeout，终态两个owned PID CIM均为空。原runtime5736、CPU40证明及sourceZIP不变。该失败说明Env/Entity CPU stub的接口边界，不能把其通过扩大为实际Manager资格。

独立[真实API审阅](D:/workspace/supermimic-rough-prefix-learning-study/outputs/reset_observation_API_independent_review_v1/report.json) 确认：简单删除env_ids会返回旧cache；在原reset已初始化的历史/延迟buffer上，正确的True+selectedids走backfill/peek，不推进全局pointer/lag或未选world。新runtime_v2需以真实ObservationManager/CircularBuffer/DelayBuffer验证这个修正，不再调用manager.reset或丢弃缓存事实。

[修正协议v2](D:/workspace/supermimic-rough-prefix-learning-study/protocols/rough-prefix-v2.json) 显式保留上次setup失败，另立一次新实现资格A0，不改任何目标/物理数值标准、不作质量择优，且只允许native_a0角色；smokes仍未授权。未来训练还需安装并验证 [reset前锁存容量观察器](D:/workspace/supermimic-rough-prefix-learning-study/outputs/READY_capacity_v1.json)，避免episode reset清掉历史overflow后只凭末尾0推断全程无溢出。

## 原生A0 v2实际完成

实际安装的ObservationManager/Circular/Delay [25项CPU检查](D:/workspace/supermimic-rough-prefix-learning-study/outputs/runtime_observation_cpu_v2/report.json) 与 [独立修正审阅](D:/workspace/supermimic-rough-prefix-learning-study/outputs/engineering_correction_independent_cpu_v2/report.json) 通过后，按新[f69d…计划](D:/workspace/supermimic-rough-prefix-learning-study/configs/a0-native_a0-v2.json) 只执行一次新版本A0。原Native源、旧runtime和setup失败不改。

[实测评分](D:/workspace/supermimic-rough-prefix-learning-study/outputs/native_a0_v2/evaluation.json) 保留：位置主指标7.366064mm（61点诊断7.401983mm），有向轴32.574641°，实际q超量.269821mrad，穿透2.692671mm，实际目标率11.448526rad/s，负载比1.000000034。仅轴超过20°，其余原门槛通过。全61状态/60solver样本、15区间/15postinterval forward、无中途reset/无外力或状态辅助、0优化器，末sticky0一次。初始fullq/qd/worldv/mocap/u0逐位一致，body-angular API误差1.3411e−7rad/s在预声明1e−6算术容差内。

[372项工程审计](D:/workspace/supermimic-rough-prefix-learning-study/outputs/native_a0_saved_audit_v2/report.json) 和 [独立23项原始数据审计](D:/workspace/supermimic-rough-prefix-learning-study/outputs/native_A0_v2_independent_saved_review_v1/report.json) 均通过。该A0是新的native cadence对照，历史directWarp T0分列；数值差既不归因于唯一forward因素，也不作总体收益或更多独立样本。

下一步以单独训练wrapper锁存每次reset前的容量、保存原始初始权重并在format104原子边界加入checkpoint容量前缀证据，再按新协议执行A1L/P_ONLY/A2各一次10iter工程smoke。当前未启动这些训练；正式长训练仍需后续资格。


## 三臂原生训练 smoke 实际完成

A1L、P_ONLY、A2 按固定顺序各执行一次 seed0／10iter。每臂11次collect、41个控制步/world、5,248条含holdout的真实transition；实际WM/失败头/Actor/Critic更新20/20/8/32，均产出原生format104最终checkpoint，无重训或质量择优。[汇总](D:/workspace/supermimic-rough-prefix-learning-study/outputs/smoke_campaign_summary_v1/report.json)保留三臂独立OS身份、源码归档、容量与成本；worker wall分别34.1931／50.6093／33.4865秒，watchdog是嵌套范围，不相加为GPU busy。

实际checkpoint健康审计A1L89项、P_ONLY108项、A2 108项通过；真实Actor CPU加载各21项通过。原始初始化四模块逐参数字节、actual8、reference和共同训练字段的[三臂31项公平性](D:/workspace/supermimic-rough-prefix-learning-study/outputs/smoke_three_arm_initial_fairness_v1/report.json)通过；P/A2同目标/spec/P数学，仅A2增加support，dt在imag恰一次。每臂checkpoint容量前缀7条（5次reset前、2次checkpoint）与最终8条分开记录，累计OR0／UNKNOWN0；H8终端屏蔽与actualu日志通过。

两类consumer问题均保留：首版健康checker把tuple与JSON list比较为不同，修新checker后重读同artifact通过；首版Actorloader把物理xi46误作context368，修新loader并以实际PenObservations推导后通过。原源、失败收据与checkpoint不改，均未重训。

首次实际物理评估v1在make_env_cfg(play=True)已删除orientation终止项后再次pop，KeyError于Env创建前；0 rollout采样、质量NOT_ASSESSED，GPU初始化成本及失败OS日志保留。当前正以新evaluation_v2及实际factory CPU合同修正，未执行P/A2旧评估配置。后续每臂一次61state/60solver/15nativeforward完整评估，沿用原物理门槛，目标率另覆盖初始化ctrl→首learnedu0；没有据10iter结果宣称CHORD收益。正式长训练尚未启动，Goal active。


## prefix16 三臂 checkpoint 的完整物理评估

新evaluation_v2通过真实factory20项和独立修正16项CPU检查后，A1L/P_ONLY/A2各执行一次完整61state/60solver/15interval/15postinterval forward，0优化器/0中途reset/末capacity0一次。各臂独立saved审计38项全过，[三对输入/初态/时序28项](D:/workspace/supermimic-rough-prefix-learning-study/outputs/policy_saved_comparability_v1/report.json)通过。原v1 Env前失败与耗时保留，旧P/A2v1未执行；训练不重跑。

| 原生同节奏对照 | 位置16点最大/mm | 有向轴16点最大/° | 求解器穿透最大/mm | 含启动目标速率/rad/s | 原门槛判定 |
|---|---:|---:|---:|---:|---|
| A0 零残差 | 7.366 | 32.575 | 2.693 | 11.449 | 仅轴FAIL |
| A1L 10iter | 7.584 | 32.805 | 2.655 | 11.421 | 仅轴FAIL |
| P_ONLY 10iter | 7.474 | 33.352 | 2.653 | 11.421 | 仅轴FAIL |
| A2 10iter | 7.510 | 33.370 | 2.655 | 11.421 | 仅轴FAIL |

[保存数据汇总及四面板图](D:/workspace/supermimic-rough-prefix-learning-study/outputs/native_policy_summary_v1/report.json)仅消费旧raw；Root实际查看PNG，图例/原门槛/坐标清晰。质量不通过不是工程失败，也不因小差异宣称CHORD受益；Nlineage=1、独立ego=0，历史directWarp T0单列。

[新协议4](D:/workspace/supermimic-rough-prefix-learning-study/protocols/rough-prefix-v4.json)预注册新的fromscratch seed0/2000iter同预算prefix16 pilot：128env、warm64、WM-only200、H8、WM每iter2更新、Actor1800/Critic7200、648192含holdout真实transition；每100保存、每iter日志。三臂九项实际工程收据均按字节绑定，不以质量择优。新前端/监督器/CPU-only builder尚待实际默认边界与负例资格；protocol本身不授权GPU，当前没有新的长训练。首次100仅WM200/fail200/Actor0/Critic0，真实40192transition；新健康auditor_v3严格容许原defaultdict缺失zero-expected hookkeys，拒绝缺正数/未知/错误类型，原v2保留。首100健康+分段ETA之后，按用户条件暂停Goal，已授权A1L后台继续；不把此WM-only检查点称策略训练或质量资格。


## A1L 新长训练首次健康检查点与 Goal 暂停

完成前端CPU42项及独立34项验证后，Root另冻单A1L授权配置 `a1l-primary_matched-authorized_v1.json`（SHA1bfdb47b…09fdaa），后台hidden启动2000iter；P_ONLY/A2仅Gfalse配置，无自动队列。实际worker31812，OS创建11:52:32.352996Z；worker launcher37156、supervisor35220（11:52:30.747584Z）、external launcher22972（11:52:30.716253Z），Fresh CIM parent/creation/samefile/config/source闭合，不用status/lock推断liveness。

首model100 SHA `a145bba3…464709` 在worker elapsed82.3914s原子落盘，同冻结v3[健康审计69/69](D:/workspace/supermimic-rough-prefix-learning-study/outputs/health_primary_a1l_100_v1/report.json)通过：101collect／40192真实transition／WM200/fail200/Actor0/Critic0；Actor/Critic原始参数L2=0，policyupdate NOT_RUN符合WM-only阶段。容量仅checkpoint前缀69条read records（68 before-reset调用+1checkpoint）OR0/UNKNOWN0/finalizedfalse，不能写成全程最终容量或68次独立episode；整轮最终容量/健康与物理结果待terminal。

[ETA及已提交metrics快照](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_a1l_seed0_v1_eta_at_pause/report.json)在11:55:02Z／最新committed199时，测得primary WM-only148→198共50区间base .601906s/iter；未来1800 policy updates用同源A1Lsmoke剔前三call后5call稳态 .237596s/update做代理，另20s未测规划量。点估剩余25.53分钟／北京时间20:20:34，粗规划范围20:07:48–20:40:59，非置信区间或保证，不把wall当GPU busy。未额外benchmark或新增物理，当前Actor阶段的长期成本尚未由这份ETA实测。

按用户已有条件，首100健康与ETA满足后暂停Goal，保留A1L后台继续；[暂停收据与Fresh CIM](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_a1l_seed0_v1_goal_pause.json)记录当前真实进程、源码/配置及恢复说明。恢复时先查同PID创建时间/实际终态，勿重复启动/停止；完成后审final2000/最终容量并新冻format104正式物理评估消费合同，再依新授权推进P/A2。整体研究未完成，三臂10iter轴FAIL、独立ego0和完整运动/共享/硬件资格边界不变。


## 2026-11-04 A1L 最终策略验证

A1L原始final2000完整健康148项、独立终态30项、真实Actor消费3081项通过；唯一一次native物理评估及38项raw独审全部通过，shaft16从A0的32.575°降至4.206°，位置3.095mm、穿透2.630mm、目标率6.947rad/s、负载比.728。原13门槛全过，但仅seed0/该固定0.3秒窗口。A1L没有接触prior，不能称CHORD收益。

[详细进展、成本与边界](D:/workspace/supermimic-chord-study/docs/progress-2026-11-04.md) · [保存数据六面板图/CSV](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_native_comparison_v1/report.json)


## P_ONLY 同预算长训练与首次健康检查点

当前P_ONLY原Gfalse配置28076aa3…f61c10经独立24项预检后，Root另冻9dcfac74…bebcb3单次授权配置，2000iter/seed0从头训练，与A1L相同protocol4/input/controller/runtime/WM/采样更新预算；prior仅原-.1P，外部dt=.02一次。A2未授权，无自动队列。后台external launcher6992 OS16:49:33.625226Z、supervisor18372 OS16:49:33.653398Z、workerlauncher6736、actual23944 OS16:49:35.234215Z（UTC），Fresh CIM逐项验证四PID/创建时间/父链/base samefile/source/config，未用status/lock单独推活跃。

[首model100健康88/88](D:/workspace/supermimic-rough-prefix-learning-study/outputs/health_primary_p_only_100_v1/report.json)已同原v3唯一审计：checkpoint0f327d01…de238c，worker elapsed78.6323s落盘；40192真实transition/101collect，WM200/fail200/Actor0/Critic0、Actor/Critic参数未改与policyupdate NOT_RUN符合WM-only阶段。容量只checkpoint前缀，整轮最终容量/任务质量待terminal。Windows唯一审计owner，未停止worker或执行GPU评估。

[同臂分段ETA2与metrics快照](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_p_only_seed0_v1_eta_at_pause/report.json)在16:51:54Z/最新committed186时，以Pprimary WM-only135→185共50间隔base .613488s/iter、P自身smoke稳态5calls .467067s/policyupdate代理未来1800policyupdates，非A1L代理，另20s未测规划量。点估剩余32.893分钟，2026-11-04北京时间01:24:48完成；粗规划范围01:08:21–01:51:07，非CI/保证/实测未来策略阶段或GPU busy。未新增benchmark/physics。

按用户首健康checkpoint+ETA条件，Goal暂停，P_ONLY后台继续。[暂停收据](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_p_only_seed0_v1_goal_pause.json)绑定Fresh CIM、实际配置、checkpoint与ETA。恢复先验证真实进程/终态，勿重复启动/停止；terminal后完整v3最终健康、实际Actor加载、协议3唯一final2000物理评估及原38项saved审计，再按新授权推进A2。原有full151/shared/独立ego/硬件目标不缩小。


## P_ONLY 最终产物、匹配预算与唯一物理评估

P_ONLY原任务已于2026-11-04 01:23:45.875874北京时间exit0/no timeout；旧6992/18372/6736/23944 fresh CIM均为空，未重复启动。final checkpoint c7304411…22bbe，2000iter／648192含holdouttransition／WM4000-fail4000-Actor1800-Critic7200、2001collect／2000finite metrics。原v3[最终健康167/167](D:/workspace/supermimic-rough-prefix-learning-study/outputs/health_primary_p_only_final_v1/report.json)、独立终态32/32、真实Actor2247项通过。容量prefix2233记录+末一次=2234（2212 before-reset调用／21checkpoint／1final）OR0/UNKNOWN0/finalizedtrue；不是2234个episode。worker2050.648077s、watch2051.797s、native2042.71781s嵌套，不相加为GPU busy。原同臂ETA点估晚61.977秒，仅此运行描述。

已补齐暂停时SOURCE_ONLY的初始化核查：旧f500路径斜杠别名误拒保留，新1f9f只规范resolved绝对路径拼写而保持SHA严格，CPU30项通过。实际[A1Lfinal↔Pfinal17项](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_a1l_p_only_initial_fairness_v1/report.json)全过，四模块38个raw namedparameter tensors、actual8三NPZ、参考/init/control/budget/core107源码一致；不比较finalweights。原P评估正向PS汇总仅6consumer计数误报与savedfinalizer首版斜杠误拒完整保留；三builder/Entry2/Sup2实际CPU均exit0，未重跑，保存数据finalizer20项与独审14项通过，消费者和物理源码未改。

Root另冻ad22dc95…ae9d0授权P_ONLY唯一final2000 one-world评估，worker41340/launcher8276 OS身份、exit0、postPID空通过；workerwall30.290616s为该次独立评估setup/rollout成本。完整61state／60solver／15interval／15postinterval forward，无中途reset/物体辅助/Actor normupdate，0优化器，末cap0一次。原470840[保存数据审计38/38](D:/workspace/supermimic-rough-prefix-learning-study/outputs/policy_saved_audit_primary_p_only_v1/report.json)独立重算全部工程与原13质量门槛通过。

| 同原生节奏条件 | 位置16点最大/mm | 轴16点最大/° | 穿透最大/mm | 含启动目标率/rad/s | 负载比 | 质量 |
|---|---:|---:|---:|---:|---:|---|
| A0 零残差 | 7.366 | 32.575 | 2.693 | 11.449 | 1.000 | 仅轴FAIL |
| A1L 2000iter | 3.095 | 4.206 | 2.630 | 6.947 | .728 | 原13门槛PASS |
| P_ONLY 2000iter | 1.989 | 4.329 | 2.578 | 6.291 | .620 | 原13门槛PASS |

P_ONLY全61点诊断位置最大2.538mm／轴4.430°，原16结点主指标不与61点混写；startup差 .1258135rad／20ms=6.290675rad/s，后14次更新最大5.63219rad/s。相较A1L，P位置/穿透/负载指标低，轴指标略高，不能由单源seed0判总体优劣或因果。两者训练wall约34.1775 vs27.8727分钟，虽然real/update预算相同，并非等wall比较。

冻结新plotter71f53d唯一运行生成[匹配预算六面板PNG/SVG与CSV](D:/workspace/supermimic-rough-prefix-learning-study/outputs/matched_primary_comparison_v1/report.json)，Root和agent实际看图，布局/门槛/标签无裁切，0新physics；原A1L10/2000图不覆盖。optionalA2audit分支待第三臂实际结果后用新输出加入，不提前宣称三臂完整比较。

## A2 同预算启动、首次健康检查点与 Goal 暂停

A2原Gfalse35649083…adeac经独立14项当前预检、P完整物理工程资格满足后，Root另冻bb241593…c1f432单次2000iter/seed0从头授权；无自动队列、不按quality筛选方法。相同protocol4/input/controller/native/runtime/WM/实际预算，只在共有-.1P背景上增加冻结H123几何声明support，保留human接触UNKNOWN和旧proxy/gradient资格失败，不能称人类力标签或可靠力预测器。

[首model100唯一健康88/88](D:/workspace/supermimic-rough-prefix-learning-study/outputs/health_primary_a2_100_v1/report.json)通过，checkpoint71145c1e…bc82057e，workerelapsed84.52677s落盘；40192真实transition／101collect、WM200/fail200/Actor0/Critic0/policyupdate NOT_RUN符合WM-only阶段。prior已绑定而无policy bonus更新行是正确预期，容量仅checkpoint前缀。原1f9f工具复用二臂接口做[Pfinal↔A2prefix10018项](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_p_only_a2_initial_fairness_v1/report.json)全过；与已有A1L/P17通过两对字节等同串联共同初始化关系，并非三臂单次审计，不要求A2已结束。

[实际分段ETA2](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_a2_seed0_v1_eta_at_pause/report.json)在18:29:38Z／committed209：WM-only149→199共50间隔base .609996s/iter，A2已测policy202→208的6个稳态调用 .519796s/update，未混用别臂代理；另20s未测规划量。点估剩34.058分钟，北京时间2026-11-04 03:03:42完成，粗规划范围02:46:40–03:30:57，非CI/保证/GPUbusy或最终策略资格。

[暂停收据与Fresh CIM](D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_a2_seed0_v1_goal_pause.json)绑定真实external14432（18:26:52.128072Z）、workerlauncher1728、actual26496（18:26:53.754891Z）及实际supervisor的PID/创建时间/父链/base samefile/source/config。按用户首健康checkpoint+ETA条件，Goal暂停，A2后台继续，不停止worker。恢复时先核实际PID创建时间和终态，审final2000/最终容量/实际Actor加载，再协议3唯一A2物理评估与原38saved审计，之后才补齐本轮三臂完整对照。当前独立ego仍0、仅固定0.3秒source；完整151、多轨迹共享真实训练/transfer/独立targets、接触梯度资格和硬件合同仍未完成，整体Goal不标完成。
