# 非线性可行性恢复与保可行拟合

当前完成了同一受控观察窗口的 R→F 方法、控制导出和实际短动态配对。61 点生成接触深度恢复至 2.998999207 mm，F 没有接受步骤；两条 0.3 s 动态均因轴误差失败。它不是独立 ego 数据，也不是完整 3 s 转笔或 CHORD 收益证明。

## 冻结方法与唯一执行

[协议](D:/workspace/supermimic-observation-restoration-prefix-study/protocols/restoration-filter-prefix-v1.json) SHA `3c4113a60567321fdfc57fa201ad5edc194475d72493f840d1ac8ce71acb387c`，在求解前声明从旧受约束终态 64aa… 初始化，复用同一 5960×340 线性合同，不再求 LP。R 仅最小化非负最大违规 slack；原始几何、线性和系数盒独立核验后才能进入 F。F 使用固定 QP、信赖步长、alpha 序列及原始可行性和严格代价下降过滤。预算、算术容差、终态选择和逐 major 证据均事前固定。

旧物体运动、材料假设、原残差/权重、q0、C2、速率、力限及 3 mm 资格标准不变。新 variant `H123_cubic_minimax_restoration_filtered_prefix16_v1` 仍来自相同 legacy clip007[200,216)，不增加独立来源数，不继承旧静止实验或共享目录资格。缺失 gap 仍为 UNKNOWN。

[正式计划](D:/workspace/supermimic-observation-restoration-prefix-study/configs/restoration-filter-authorized-v1.json) SHA `015ecf45d35abf823d2c20d0dee7de5927bd93d91076a43bcdd24c131f4e24d4` 只运行一次：R 3 iter/4 fun 正常返回，fresh61 通过；F 一个 QP 25 iter/65 fun，虽然下降方向为负，但其线性和线性化深度 primal 残差超出预声明容差，因此拒绝，不进行 alpha trial。F 接受 0 步；按事前规则选择 R 终态，不从历史挑选。

[候选](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_filter_prefix_v1/candidate.npz) SHA `6364a6e262f7fbedac049d2d9e789b2c9077cbd6ef7656ef2e9b394161bab81a` 与 pipeline_return.final_x/last_accepted_x/R_terminal_x 逐位相同。最大生成深度由 3.194716050 降至 2.998999207 mm。线性 raw 残差 +8.055622e−17，exact=false、预声明 1e−12 算术容差 pass，两者同时保留。代价由 312.543411 增至 314.319662（+0.5683%），无拟合改善。

424 次真实 CPU 碰撞调用是一个全局共享计数（fit 所属 61 + constraint 所属 363），包括 R/final 两次强制 fresh61，不能再加嵌套 parent 计数。没有 LP、物理 step、GPU 或训练。实际 worker38560/launcher42620，OS 创建 `2026-11-03T09:02:06.8344550Z`，worker19.004633 s/watchdog19.203 s 为嵌套口径。正常 exit0、无 timeout、终态两个 PID 的 CIM 均为空。

## 控制导出及证据

[控制收据](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restored_prefix_control_export_v1/receipt.json) SHA `5bcd45d8e15e13579fb8d4debe9efcbd1e765bd3d24725acbf6ee38bf09f3afc` 的 18 项 CPU 门槛通过。reference/init/control 分包，解析 qd/qdd 与原生 FD 分开；init 是设计的非零速度，实测字段为 null。最终 FP32 目标 C2 夹量0、更新峰11.448526 rad/s、全曲线系数凸包和参考状态 PD Bernstein/FP32 证书通过，NumPy/Torch/原 C2 逐位一致。这不是实际状态、实际施加目标或实测力保证。

证据：

- [30 项保存数据审计与阶段图](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_saved_review_v2/README.md)，无新查询/求解；v1 图版式问题保留，v2 仅改布局。
- [独立 R/F 原始数据审计](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/RF_saved_independent_review_v1/report.json)。
- [计划控制诊断图](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restored_prefix_control_export_v1/restored_prefix_diagnostics.png) 和独立图 QA；只读保存几何，无新碰撞查询。

实际三 bundle 已通过 [39项独立数学检查](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restored_bundles_independent_review_v2/report.json)，重现全部18门槛，没有新增几何、物理或GPU。物理执行前还通过实际bundle预检、独立worker/driver审查与Windows真实PID/创建时间/可执行文件别名身份资格。完整151帧、学习残差/CHORD比较、共享训练成本、真实ego和硬件合同仍未完成，Goal active。

## 实际短动态：工程完整，两条件轴误差失败

上述预检和独立审阅均通过后，Root 冻结 H0/T0 正式计划并按顺序各运行一次，没有训练、重试、截断或改变参数。H0 的独立工程审计通过后才运行 T0；H0 轴误差失败没有改变 T0 计划。

| 原标准下的实测量 | H0 保持目标 | T0 跟踪目标 | 门槛 |
|---|---:|---:|---:|
| 16 点最大物体位置误差 | 11.072444 mm | 7.209347 mm | ≤20 mm |
| 16 点最大有向轴误差 | **33.282835°** | **32.806787°** | ≤20° |
| 61 状态最大关节硬限超量 | 0.0000505 mrad | 0.268033 mrad | ≤1 mrad |
| 60 求解器 prestate 最大穿透 | 2.692671 mm | 2.692671 mm | ≤3 mm |
| 最大实际 actuator/joint 力限比 | 0.304279 | 1.000000034 | ≤1.01 |
| 最后一次 sticky overflow | 0 | 0 | 0 |

两条均只因轴误差失败；位置、关节、穿透、非手接触、无外加力、负载/charge、时钟、控制读回和覆盖均过。T0 力限比接近1且通过1.01门槛；它与参考状态原始 PD 预测不是同一量。两条初始 fullq/fullv、object twist、mocap、ctrl0，以及八字段双读回逐位相同；计划仅 condition/output 不同。每条61状态/60step/14目标写入、startup forward一次、postinit状态/力写入0、末尾sticky读取一次；未生成已知人类接触标签。

H0 worker7704/launcher13408，OS 创建09:18:17.582717Z；T0 worker22004/launcher26276，OS 创建09:18:51.454398Z。实际源、runtime和OS创建时间绑定均过，退出后两组PID的CIM均空。worker时间和7.718373 s，watchdog槽时间和8.781 s为嵌套口径，不相加，也不是GPU kernel利用率。

- [H0 正式计划](D:/workspace/supermimic-observation-restoration-prefix-study/configs/dynamic-prefix-hold-v1.json) SHA `e9b10f1a…9f6dc6be`，[T0 正式计划](D:/workspace/supermimic-observation-restoration-prefix-study/configs/dynamic-prefix-track-v1.json) SHA `52e7a9a1…6ae45f3d`。
- [H0 独立工程审计](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_hold_saved_audit_v1/report.json) 和 [T0 独立工程审计](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_track_saved_audit_v1/report.json)，各152/152通过；质量分别保留 false。
- [跨条件63项身份核验](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_H0_T0_identity_comparison_v1/report.json)。
- [原始指标复算与图](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_pair_summary_v1/report.json)，Root 已实际查看 [动态诊断图](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_pair_summary_v1/dynamic_pair_diagnostics.png)。

这是同一受控来源、已知历史下每条件N1的0.3s诊断。T0观察到更小位置误差，轴误差仍超限，不能写成总体方法收益；更不能当 CHORD/完整3s/真实ego泛化或硬件结果。参考几何和计划控制通过并未保证旋转跟踪。下一步从保存的实际 q/qd、PD 与接触扳手误差诊断，再冻结新的动态修正实验，不重跑这两条。

[另一份40项独立原始数据审计](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_pair_independent_saved_review_v2/report.json) 确认有向 local+X shaft、16/61采样、控制/力相位、来源、初态和原门槛。首个主采样轴越限分别为 .22 s/.24 s。后续同时审核将此 rough observation reference 接入 fromscratch 残差学习/WM/EMA/reset 的明确研究模式；参考自身开环失败可作为学习修正目标，仍保留 physical_reference_qualified=false，且不继承 shared10/旧151资格。当前没有启动新训练。

## 原生数据链与后续残差学习合同

已复用既有 native BODY21 exporter、resolve_recipe 和真实 SingleHandTrackingReader(device=cpu)，从新 q16/mapped pose16 重算 palm+link1/2/3/4 四键原生格式，未另写 reader。[独立新 manifest](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restored_prefix_native_ABI_v1/native_catalog_manifest.json) 保留 legacy[200,216)/variant/local[0,16)，不迁入旧 shared10。原 adapter 在完成读取并保存数组后，末尾 JSON receipt 因 NumPy bool 类型失败；原进程和源保留为失败。[保存数据20项复核](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/finalization_review_v1/report.json) 和 [独立15项审计](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/rough_prefix_native_ABI_independent_review_v2/report.json) 通过，不重跑 FK/reader。它证明数据 ABI，不伪称原进程完整成功；原未保存的终态 CUDA flag 仍 UNKNOWN。

解析 qd0 与 reader forward-FD0 峰差11.938072rad/s，不能用后者静默覆盖前者。新训练模式采用解析参考速度/物体quintic twist的明确来源，readerFD保留诊断。actual q/qd 与模型学习 x1 仍来自模拟实测，不用解析参考作 dynamics teacher forcing。

[训练接口评审](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/rough_reference_learning_integration_review_v1/report.json)、[终端合同](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/rough_reference_learning_integration_review_v1/terminal_contract_supplement.json) 与 [最小数据适配方案](D:/workspace/supermimic-observation-restoration-prefix-study/docs/prefix16_native_data_adapter_proposal_v1.md) 明确下一步：

- 新独立 rough-reference entry/runtime adapter，保持物理资格 false，以统一工程门进入研究训练。
- 真实与想象使用 baseline_u15 + 原EMA残差，再执行原 hard/control bounds 与一次C2；从原始残差重建，不在已裁剪目标上平移，不把u0−q0存入残差历史。
- 在首reset/collect之前安装解析速度和非零初态；部分reset只触及选定环境，实际8字段、mocap和u0逐项读回。
- phase0…14共15动作、到.3s的terminalobs15，不执行u15；terminal transition保留，imagined dead rows使用显式有限占位及zero survival，禁止NaN乘0或跨episode泄漏。
- 新版本的控制/初始化/参考/速度元数据与checkpoint loader；不迁移旧103 checkpoint。A1L/P_ONLY/A2同新合同、相同采样与更新预算，实际valid mask和成本另报。

现有原生 Manager 在每4个substep后还有forward，与本轮 direct-Warp T0 的数值节奏不同；新 native-cadence 零残差A0作为后续训练/评估共同对照，历史T0单列。当前只有只读评审和数据链证据，runtime adapter、CPU/原生reset/phase/replay/checkpoint资格和三方法smoke仍待实现，未启动训练。

## 保存数据的失败解释

[最终诊断](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/dynamic_saved_failure_diagnosis_v3/README.md) 20/20相位/重建检查通过，Root已实际查看其图。物体实际转动24.032°/25.378°；目标初始坐标中的末端shaft y=+.4064，H0/T0为−.1609/−.1505，说明方向路径偏离。T0关节位置RMS较低，但速度RMS仍1.628rad/s；只有step5、finger3_joint3一个5ms的原始PD计算越力限，实测被限幅，不能归因于持续饱和。

实际COM接触扳手包含direct torsion与力臂矩，按solver步前姿态/COM和actual mass/inertia重建；其参考需求差及离散动量一致性仅为诊断，不证明连续精确或滑动/滚动/映射等唯一原因。预计算目标没有物体姿态误差反馈，后续残差学习将测试这一修正空间。该审阅没有新增查询、求解、物理或参数调整，两条原FAIL保持。
