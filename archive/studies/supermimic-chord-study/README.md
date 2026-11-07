# SuperMimic × CHORD / Wuji Hand 2 研究工作区

2026-11-07 当前研究入口为[方案 v0.31](docs/research-plan-v0.31.md)与[本轮证据及交接](D:/workspace/ego-data-pilot-2026-11-07/outputs/goal_extension_tripod_momentum_milestone_v12.json)。已建立两个真实三指初载条件，但16个无辅助动态案例均失持，尚无两秒持握成功。新几何与动量世界模型已完成250/5000次更新，[首检查点健康15/15](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_momentum_wm_v1/cpu_run1/health_0250.json)，worker正常退出；按用户约定，归档后暂停Goal。训练数据仍来自仿真，真实ego61帧目前仅形成物体运动意图参考。下列较早记录保留各自历史范围，不能替代当前交接或作为真实ego/CHORD收益结论。

当前物理梯度对照已完成：修正版原生仿真取得16个新donor快照、560个分支，48次baseline过原门槛，物理独审194/194及消费保存结果独审30/30通过。128个方向中93稳定且信号足够，两档扰动下A1L/P_ONLY/A2的符号一致率分别83.9%/80.6%/74.2%；Pearson排序不同，不据此判定CHORD整体收益。self15全为负例，不证明正接触能力。原空数组setup失败与路径绑定拒绝保留，未覆盖旧结果。Goal active，无GPU任务；下一步补全完整151帧的reference/初始化/150控制量与接触意图合同。见[结果与范围](D:/workspace/supermimic-rough-prefix-learning-study/docs/current104_physical_gradient_findings_2026-11-04.md)和[六面板图](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_WM_summary_v2/physical_secants_WM_summary.png)。

2026-11-04 当前状态：观察侧粗参考的固定0.3秒窗口已完成 A1L/P_ONLY/A2 三臂同采样、同更新预算的 seed0 正式训练及各一次最终原生评估，三者均通过该窗口原13项物理门槛；A0仍仅笔轴失败。位置最大误差分别3.095/1.989/3.909mm，笔轴最大误差4.206/4.329/4.090°。这是单源短窗口证据，尚无稳定CHORD总体收益。三个世界模型对四条保存轨迹的180次一步预测已完成，独立保存数据审计68/68通过；位置、旋转误差均低于恒速基线，但175/288内部差分因动作边界被跳过，真实物理梯度仍待检验。Goal active，当前没有GPU任务；见[11月4日完整记录](D:/workspace/supermimic-chord-study/docs/progress-2026-11-04.md)、[三臂物理图](D:/workspace/supermimic-rough-prefix-learning-study/outputs/matched_primary_three_arm_comparison_v1/primary_native_diagnostics.png)和[世界模型预测图](D:/workspace/supermimic-rough-prefix-learning-study/outputs/WM_onpolicy_CPU_diagnostic_summary_v1/WM_onpolicy_prediction_heatmaps.png)。

2026-11-03 更新：新种子九次训练与27次确认评估完成，A1L/P_ONLY/A2各1/0/1个训练seed三次全过（各N3），尚无稳定整体收益。观测侧三指H123初始化的单次0.3s静态PD预载保持通过原门槛：位移3.956mm、轴角16.287°、穿透2.183mm、实际q越限0.626mrad。它与先前同q的固定qref保持顺序比较，属于探索性N1，不能据此称完整转笔或方法收益。独立raw重建通过；完整动态跟踪和真实ego资格仍未完成。Goal保持active，见[当前方案](D:/workspace/supermimic-chord-study/docs/research-plan-v0.3.md)、[接触意图](D:/workspace/supermimic-observation-contact-intent-study/README.md)及[原始物理证据](D:/workspace/supermimic-observation-warmseed-grounding-study/README.md)。

此前[首轮16帧连续参考实验](D:/workspace/supermimic-observation-prefix-study/README.md)已完成单次CPU求解和独立复算：首帧固定、16个关节节点合法，但原生手物深度峰5.141mm、节点间连续曲线越限15.627mrad、计划C2目标速率峰28.485rad/s，hold/track参考状态PD预测也超力限。前置门槛失败，未运行新动态GPU或训练；原生reader/代数通过不能覆盖这些失败。

后续[受约束B样条分支](D:/workspace/supermimic-observation-constrained-prefix-study/README.md)已唯一运行LP+SLSQP并独立复算：连续范围、计划速率11.449rad/s、零C2限幅及参考PD证书通过；80轮终态仍有5/61采样重叠超过3mm，峰3.194716mm，尚无几何/物理资格。没有挑历史候选、放宽门槛、追加求解或新GPU/训练。下一步另立有限可行性恢复阶段，再做拟合和实际动力学验证。

共享多轨迹的[机器协议与数据清单](D:/workspace/supermimic-chord-study/protocols/shared-multitrajectory-controlled-v1-bundle.json)保持DRAFT：10条源、3987帧，同一保守lineage，独立ego N0。新严格三字段reader和[10×151帧观测→几何→原生参考ABI](D:/workspace/supermimic-shared-reference-study/outputs/geometry_native_batch_v1/README.md)已实际CPU核验；原生21个BODY原点与IK宏骨架点分别计算，未混用。两帧未收敛和较大拟合误差完整保留，尚无该批次的合格物理初态/接触目标；未启动共享训练或声称提速。

后续[逐轨迹registry CPU阶段](D:/workspace/supermimic-shared-reference-study/docs/shared_cpu_registry_v1.md)已实际加载train8/val009/test010/seen10/audit10五套原生recipe，并通过29项接口/负例检查。训练目录物理排除009/010；旧clip007[200,351)初态不得匹配新[0,151)窗口。十条独立接触bundle均缺失，训练就绪检查明确拒绝；reset/replay/transfer接口尚未接入真实trainer。

本工作区推进原生 Windows、RTX 5090、单只右手固定腕的接触动力学重定向实验。用户已确认：暂无指定 ego 数据，先建立受控实验。目标是检验共享世界模型、参考条件残差策略及 CHORD 接触目标的组合，是否在相同物理质量门槛下减少轨迹修正成本。收益尚未成立。

11月3日新增：共同20mrad目标内收的27条旧模型评估全部完成，A1L/P_ONLY/A2分别有1/2/3个训练seed三次全过（各N3）；A2采样关节超限消失，shaft峰值较5mrad约增0.7°。这是已见单clip的正向工程信号，两剂量分别报告、不选最好结果，尚无独立确认或真实ego泛化。下一步先核验同一控制映射进入真实执行与想象学习，再用新种子4–6做同预算三臂训练。详见[当日记录](docs/progress-2026-11-03.md)和[20mrad报告](D:/workspace/supermimic-c2-020-portfolio-study/outputs/analysis_v1/SUMMARY.txt)。只读关键点的掌根IK分支也已落地，仍未取得物理参考资格。

2026-11-02 的三训练种子复制已完成：同采样/更新预算下 A1L 为1/3通过冻结任务，A2为0/3。A2三对最大笔轴误差均下降，但实际关节越界1.46–3.31mrad超过冻结1mrad容差；目标限幅正确，不能用目标合规代替真实状态合规。1200秒内可用模型口径双方均1/3通过，成功种子不同。当前几何先验没有显示稳定的整体任务收益，仍存在指标改善信号；不能据此作总体有效或无效结论。六次训练的GPU预约窗口约1.882小时，真实物理评估另计，当前A2实现训练耗时约为A1L的1.493倍。

六次运行与严格时间选点的[独立审计](outputs/replication_seeds123_audit_v1/REPORT.md)、[成本记录](outputs/replication_seeds123_cpu_cost_v1/report.json)和[冻结评估清单](configs/replication-seeds123-evaluation-v1.json)保留全部种子。seed0只作为pilot，见[先导对比报告](outputs/paired_pilot_seed0_comparison/REPORT.md)。当前仅一条已见参考、固定初态和腕，不是独立ego视频或跨动作泛化。

后续独立P_ONLY消融也已完成：主/次预算均2/3通过，成功种子分别为[1,2]/[1,3]，不能拼成3/3。P_ONLY三个主预算模型的实际关节门槛均通过；完整A2相对它的最大轴误差更低，而实际关节越界更大。见[三臂历史对照报告](D:/workspace/supermimic-chord-penalty-only-study/outputs/three_arm_seed123_comparison_v1/REPORT.md)。这是结果已知后提出的探索性机制诊断，执行时段/缓存没有完整三臂平衡，不作确定因果或总体推断。

后续有限制动探针未通过快照回放资格：仅1条baseline、0条pulse，恢复/目标/时钟及容量检查通过，但0.2秒内物理回放差异超冻结门槛，故53条未运行、参数仍禁用。新增5ms采集的A1L donor还出现历史关节门槛PASS到本次FAIL的变化；采集流程不同，不直接混池或改原成绩。见[失败与未运行覆盖记录](D:/workspace/supermimic-control-feasibility-draft/outputs/braking_microprobe_analysis_v1/partial_terminal_review.json)。当前优先检查原未加hook评估的fresh-process重复性，尚未据此启动新控制器训练。

原未加hook流程的五次重复现已完成并独立复算：同一A1L检查点F/F/F/F/T，前四次只因实际q越界1.45–1.69mrad失败，第五次0.73mrad；五次物体跟踪及其余门槛通过。见[重复性报告](D:/workspace/supermimic-control-feasibility-draft/outputs/plain_evaluator_repeats_analysis_v1/REPORT.txt)。这揭示冻结1mrad附近判定敏感，训练seed N仍1，不能把本次1/5当方法总体成功率或与原n3混池；后续比较须加入事前定义的重复评估。当前GPU空闲，无训练运行。

2026-11-03新增C2工程信号：同一A1L检查点分别内收目标0/5/20mrad，各3次完整物理评估，容量审计v2通过数为0/3、3/3、3/3；全部overflow0，六条内收轨迹在50Hz记录中无实际关节hard-excess且物体跟踪通过。见[v2报告](D:/workspace/supermimic-target-inset-study/outputs/analysis_capacity_v2/REPORT.txt)与[最新执行记录](docs/progress-2026-11-03.md)。

随后共同5mrad用于三方法全部9个既有模型，每模型3次，共27条完整评估。所有容量和控制合同通过，但A1L/P_ONLY/A2仅各1/3、2/3、1/3训练seed三次全过；A2最大shaft误差三对改善，实际q约束失败仍在。见[组合评估](D:/workspace/supermimic-c2-portfolio-study/outputs/analysis_v1/SUMMARY.txt)。不能将单检查点的成功扩为通用约束保证，也尚无稳定CHORD整体收益。原历史批次均未混池。

输入侧已新增[不读取源机器人关节答案的几何IK原型](D:/workspace/supermimic-observation-retarget-study/README.md)，151帧平均骨方向误差6.13°、归一化目标指尖误差16.2mm；输入/坐标/FK独立复核通过，动力学、接触和真实ego资格仍未成立。

原始方案保持不变；修订与证据保存在这里：

- [证据更新后的研究方案 v0.3](docs/research-plan-v0.3.md)
- [初始修订研究方案 v0.2](docs/research-plan-v0.2.md)
- [SuperMimic 源码审阅](docs/supermimic-audit.md)
- [CHORD 源码审阅](docs/contact-audit.md)
- [实验协议](configs/controlled-study.json)
- [最新执行记录与实际结果](docs/progress-2026-11-02.md)
- [三种子复制的完整配对报告](outputs/replication_seeds123_comparison_v1/REPORT.md)
- [实际关节越界诊断](outputs/replication_joint_limit_diagnostics_v1/README.txt)
- [首轮底座执行记录](docs/progress-2026-11-01.md)
- [首条A0失败分析](docs/a0-findings.md)

## 实现与运行

研究工具使用 `D:\workspace\dex-supermimic-research\.venv` 独立手部环境。不要混入 G1 或 FlashCHORD 的 JAX/Newton 依赖。数据/评估sidecar不改仿真器物理；A2只在策略想象中增加显式几何先验，真实collect与WM拟合保留原生奖励。几何先验实际采用eager计算，原生子网络编译；standalone编译先验的精度对照仍未通过，未用于正式训练。在PowerShell中从此目录运行：

```powershell
$env:PYTHONPATH = 'D:\workspace\supermimic-chord-study\src'
$studyPython = 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe'
& $studyPython -m unittest discover -s tests -v
& $studyPython -m chord_study.cli inspect 'data\reference.npz'
& $studyPython -m chord_study.cli evaluate 'data\reference.npz' 'outputs\rollout.npz' --output 'outputs\evaluation.json'
```

`reference.npz` 与同名 JSON 是待导入数据，不是仓库自带的实验结果。没有物理 rollout，评估器不会输出动力学成功。`screening_success` 使用明确保存的建议筛选阈值，只是工具输出，不能替代最终任务验收、掉落/负载检查或多种子统计。

## 数据约定

统一 SI、右手世界系、`wxyz` 四元数。参考至少含时间、20个具名机器人关节角、物体位姿、腕位姿及阶段。对象物理参数、来源类别、原始演示分组、重定向/坐标变换来源写入 JSON；物体系接触点、朝物体内的力方向、存在/已知掩码独立保存。缺失接触属于 unknown，不能编码成已知无接触。

CHORD support 描述单位摩擦锥接触能提供的扳手方向，不估计视频真实接触力，不证明动力学可行性。扳手为 `[f, (p × f)/radius]`，固定单位六维方向集不再除半径。最终评价回到自由物体、无物体辅助、无逐帧重置的物理仿真。

当前软件边界：上游笔任务的15维接触是自碰撞；其 `task/rot_error` 不是笔参考误差；FailureHead有时序不一致；笔 PPO 尚未实现。这些都在审阅文档中定位。研究侧工具不声称已经解决上述训练端问题。

已导出[不含机器人标签的受控观测视图](outputs/observation_only_fixture_v1/manifest.json)，并通过[禁止标签投毒不影响输出等CPU合同检查](outputs/observation_only_fixture_contract_v1/report.json)。它保留151帧source手关键点、物体/腕位姿和时间，未知置信度明确标记；相机/尺度/语义尚未核验，不称真实ego或已完成重定向。
