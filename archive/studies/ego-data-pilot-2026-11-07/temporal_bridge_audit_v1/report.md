# 手机细步长 → SuperMimic/CHORD 时间接线审查

仅源码与已有配置审查；0新环境、模型加载、NN/FW/AD、Warp、积分、训练或GPU。完整文件身份、逐项行定位和计划算术在 [report.json](report.json)。旧代码及已执行结果不改，也不据本报告授予手机参考/学习资格。

**不能只修改 dt 后沿用旧 full151/105 资格。** 底层 Manager 的子步循环本来就是参数化的，但研究入口、时钟、原始样本索引和所有四子步标签均需新身份和新接线。最小路线是保持控制周期 **H=20ms**，新增一个手机任务/时间合同；同一合同同时服务 SuperMimic 与 CHORD。一个 WM step 仍表示一个20ms控制区间。

| 名义物理步长 h | 每控制区间 n=H/h | 最后 solver PRE 相位 | 150区间/3s的积分步数 | 相对旧积分预算 |
|---|---:|---:|---:|---:|
| 5ms（旧） | 4 | 15ms | 600 | 1× |
| 0.3125ms | 64 | 19.6875ms | 9,600 | 16× |
| 0.15625ms | 128 | 19.84375ms | 19,200 | 32× |

以上是整数 tick 的计划相位；运行必须另存 requested h、实际 WP FP32 h、原生 time 和整数 tick。不能将FP32累计时间除h后取整当索引，也不能把原600步的 `5e-5s` 长时钟资格直接搬来。

## 真正固定在哪里

| 接线点 | 具体源码 | 迁移所需动作 |
|---|---|---|
| Manager通用循环与forward时序 | `mjlab/envs/manager_based_rl_env.py:263–270,395–409,435–490` | `process_action` 每控制步一次，`apply_action/step` 执行n次；reward/termination先读最后solver派生量，最后一次forward后才更新观察。底层不要求n=4。保留并重新证实这套调用次序，勿为“补POST标签”暗加forward。 |
| 旧学习/运行入口硬门 | `full151_native/native.py:65–66`；`learning_v2.py:298–299` | 两处都拒绝非4×5ms。新phone namespace显式接受固定h与n；不能放宽旧入口并冒原已资格运行。 |
| 参考与时钟身份 | `full151_native/bundle.py:43–63,84–93`；`probe.py:16–60,78–87` | 写死151/150、601/600、20ms与5ms参考、顺序FP32累加。按新参考长度N与n生成 `(N*n+1)` 原始状态和新时钟合同；终端观察N无u_N，原151窗口不是手机的天然长度。 |
| 采集与控制值保持 | `serial_probe105_calibration_v1.py:227–238,561–590,805` | tick=`k*4+substep`、总1808/1200、控制区间数`steps//4`均需参数化。逐子步核actual u[n,20]；EMA/C2只在控制边界做一次，不得变成64/128次过滤。`controller.py:136–164`原有每interval恰好一次process守门可作为新合同。 |
| 缓存self15/原生reward相位 | `pen.py:347–375`；`serial_probe…:542–544,632–646` | 旧self15真值是lastsolver约15ms传感器，58state为raw POST20；新应命名lastsolver `(n−1)h`，不能复用旧15ms标签文件或把它称POST20几何真值。同一控制步reward dt仍20ms。 |
| J/K与残差标签 | `boundary_impulse_dataset105_v1.py:89–109,138–142,182–233` | source行`4*g..4*g+3`、`[152,4,…]`、4次actualu、3条内部PRE/POST链接、四项冲量求和全写死。新标签须保全n项：J=Σh_iF_i、K=Σh_iτ_COM,i、ΔT=Σh_i，区分HO与所有非手外边界接触；重新保存r_saved，不能重标旧608/J/K或强令残差0。 |
| 容量/事务/导出与审计 | `serial_probe…:456–465,574–590`；旧`probe.py:262–264` | 原192contacts/4096efc、原inventory/restore及计数仅旧拓扑/调用范围。手机需新容量/溢出/phase与新标签身份；stream逐tick是保存方式，不得仅抽4条替代64/128条。无需复制旧全部审计模板，但不可继承其结果。 |

`range(4)` 不是都表示物理子步：串行计划中的4个joint direction、每path4个snapshot也用4。应分别命名 `substeps_per_control`、`direction_count`、`snapshots_per_path`，不可全文替换常数4。

## 58 / 46 / self15 / “13”可保留什么

- **58state/52delta可保留数值维度，不能凭shape保留语义。** `state.py:6–38` 为20q+20qd+物体p3/R9/v3/ω3，native52用LogR增量；没有h字段。`networks.py:11–61` 每次预测整个控制区间的状态增量，不是求连续时间导数。所以保留20ms时无需把WM rollout或H8的8步扩大成n倍；旧pen权重/normalizer对新phone、材质/求解器/采样相位没有自动资格。
- **当前phone存在root↔COM的实质接口差。** `SimReader.state:95–110` 直接取freejoint的root位置和前三平移速度；旧serial `:508`硬要求`body_ipos==0`。phone common `:45–48,83–85`的惯性中心是asset center，holding也用`p_COM=p_root+R*ipos`。若需要动量一致性，必须在新合同中二选一并保持一致：保root-state并显式推导COM，或新reader统一COM-state。速度相应为 `v_COM=v_root+(Rω_body)×(R ipos)`，不可只改p而不改v；I/τ也要使用同一COM/frame。这里未改实际reader。
- **xi46可保留相同拼接宽度，但不是完整手机物理描述。** `observations.py:14–22`为6个DR比例；`sim.py:185–240`再加20Kp/20Kd比例。它不编码h、solref、solver tolerance、物体mesh、绝对mass/I/COM或所有接触参数。`pen.py:254–284`还硬指定旧`cube/A_bottom/A_top`名。单一固定phone/h可以新默认模型+新xi语义保持46；若混合不同h/几何而同xi，网络看不见这些差异，需另行显式上下文设计，不能说46足够。
- **self15是原Hand2自碰槽，不是HO或2维boundary mode。** `PenSimReader:235–239,347–349`按10个middle与5个distal OR tip组装，形状15可能继续用，但必须按新传感器拓扑验证名称、enabled masks和采样时刻。新mode2/force标签另列，不能互换。
- **此分支里的13是评价cohort数，不是物理mode维。** `native_public_only_common_evaluation105_v1.py:96–99,252`：ALL/A0/A2+8(path,k)+baseline/direction。13分组可复用为报告结构，前提是新样本设计确实相同；旧0/49/99/149、fresh24/74/124/148和未来手机时段均不能仅改标签冒同一cohort。另有13个protocol输入alias，亦不是模型维数。

## 真手机还需同一任务接口，不能把pen外壳只换mesh

现有PenMDP显式50Hz（`pen.py:51–59`），EMA/命令/计时/折扣/奖励应继续按20ms控制tick推进，不随64/128个积分tick推进。`commands.py:345–353`要求参考frame_dt等于控制step_dt；HOT3D原clip为150帧/5s，不能直接每20ms前进一原帧而加速源动作。当前静态候选也不是已经资格化的动态50Hz参考。

旧actor观察155里的object6只表示一条轴的两端点：`pen.py:124–130,175–180,382–387`；旧2+8 taskscore同样只用这条轴，绕轴roll不可辨。手机宽面朝向/接触拓扑不能默认由此完整表达。新phone任务须声明完整刚体姿态/接触目标如何进入policy与公共reward；如改变观察宽度，actor/critic新namespace，不把旧155 checkpoint称可原样resume。

CHORD入口同样不是通用phone adapter：`binding_v3.py:37–49,75–86,221–232`固定旧H123/151时相和pen几何支持目标（512评估/64训练）；`prior_hook_v1.py:57–80`每个imagined control step仅加一次 `.02*bonus`。可复用这个“原生WM预测→几何先验→每20ms加一次”的算法结构；必须换成新phone资产与真实ego目标/未知接触掩码并重新核几何/接线，不能将旧H123设计先验、旧self15或无界Fn支持函数当真实手机接触/扳手能力。

## 最小迁移合同

1. **新phone时间/模型身份**：固定 H=.02、选定h、n∈{64,128}、新参考N、固定腕/重力、实际asset/material/solver/capacity与body/freejoint/COM映射。严格保留原失败和旧namespace。
2. **一次边界，一段积分**：一控制tick计算同一actual u20并保持n子步；原始状态与solver证据逐tick导出，POST20 state和lastsolver sensor分名；terminal不多走一步。若做J/K辅助，全部n项与nativePRE/POST守恒残差重新生成。
3. **新phone Reader/MDP/缓存**：明确root或COM、完整phone姿态观察/目标、50Hz参考、reward dt、self15槽映射；原58/52与46只在明确契约下复用形状。不得复用旧pen的物理标签、norm健康或gradient资格。
4. **真正同预算SM vs CHORD**：相同新scene/h/n、相同初始化/共同warmup及复制后的replay/Actor-Critic起点、同控制采样/WM与policy更新数/实际积分数；唯一方法差是已声明prior。报告control intervals、physics steps、WM/AD/更新、walltime分别计数。细步长使旧相同控制预算的物理成本变16/32倍，不能再称同旧物理预算或先验加速。现有checkpoint仅可作显式未资格初态候选，不冒新任务resume。

这是一份必要接线清单，未实现adapter或启动实验。是否存在可用手机物理参考仍由当前原姿态闭合/无辅助holding分支判断；此审查不阻其并行工作，不宣ego学习、泛化、梯度或policy收益。
