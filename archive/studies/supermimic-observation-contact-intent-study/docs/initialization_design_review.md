# 初始接触意图与固定 PD 预载：有限 CPU 设计审阅

状态：设计建议及局部材料点 CPU 检查；本审阅没有执行候选优化或 GPU。Root 另行执行的v1记录在下文，不能把本文件建议参数冒认为v1/v2的实际 recipe。现有源、baseline、协议和结果均保留。本轮只修 frame0，object pose、default wrist/mocap、时间和观测目标不优化。

## 代码审阅结论

`retarget_palmroot_cpu.py` 提供真实模型 FK、点/骨方向 Jacobian；其归一化骨目标适合保形，却不能替代原始观测点的物体接触位置。`native_filtered_contact_geometry.py` 仅对已生成的原生接触给距离/点 Jacobian，未生成槽保持 NaN，重叠残差/Jacobian 为0。`refine_native_contacts_cpu.py` 因而能压低重叠，不能从未知 gap 自动生成新接触，亦未设计 PD 预载。此前浅重叠候选的单次设计零速 rest 失败，不能把几何接触数当成承力或静力平衡。

新增驱动应是明确、可证伪的**材料点接触意图假设**，而非未知 gap 的距离吸引：固定 object 表面点，与真实 enabled collider 上的固定材料点对齐；最终由未改变过滤规则的原生碰撞独立验收。不能将未生成距离补0、归一化坏 witness、恢复旧 ghost/SDF 代理，或把意图写成观测 known contact/force。

## 建议接口与冻结输入

```text
build_frame0_intent(observations, rigid_Rt, baseline, model, context, hypothesis)
  -> FrozenIntentBundle(source点/表面anchor/法向/material点/全部来源SHA)
solve_q0(bundle, fixed_recipe)
  -> Candidate(q0, u0_control_baseline, convergence, native_witnesses, all_failures)
design_preload(q0, bundle, actual_PD_context, separately_frozen_recipe)
  -> u0 或明确不可行；本轮保持 PENDING
```

只读允许的 observation 字段；不读取原机器人关节标签或 cached wrist。`q_baseline` 是现有 native object+self 推断候选的 frame0，单独保存，不是源关节答案。用原保存的刚性 `R0/t0` 映射 **source 手点**，禁止用 morphology-normalized target 点代替接触观测。归一化 bone target 只进入保形项。object pose 与原 R/t 的一致性、SI/wxyz、未知相机/尺度/解剖与置信度边界继续保留。

Root 当前冻结两个设计：H23 只要求 finger2/3；H123 在同一输入上加入 finger1 的额外设计意图。二者均单独报告，不为满足“三指”凭空引入4/5链远点。H23 是当前近表面 source 证据支持的假设；H123 的新增接触必须标明扩展假设，不能自动称观测证据。两指/三指是各自设计的验收条件，不是普适抓持定理。每个意图保存确切 source landmark index、刚性映射前后点、到表面投影距离、原未知 confidence 和假设来源；优化后不换 finger、landmark 或 anchor。

## Object 与 hand 材料 anchor

1. Object anchor 取实际笔轴**外露侧面**，使用原 MJB 与 actual `geom_size/geom_pos`。在 cylinder 自身坐标中作显式解析表面点投影，不调用 `mj_geomDistance`。本 recipe 可固定轴向区间 `|z_c|<=0.075m`，避开端部盒体/圆柱接缝；半径、轴和变换均读实际模型，不用视觉估计半径。投影本身是意图构造，不是 native gap 测量。保存 `a_i` 与单位外法向 `n_oi`，目标材料点为 `a_i−0.00025*n_oi`。
2. 先按既有 ordinal 对应限制 collider，**不能在该 finger 全部 link 中无条件选最近者**。当前source landmark3/7/11分别对应finger1/2/3的`link4_col`；tip4/8/12则对应各自`tip_sensor_col`。一般 source链的第二点对应link3、第三点对应link4，第一点对应link1；若对应体被原模型禁用，则该假设不可用，不能挪到另一个link“救活”。在这个已声明对应内，才用 baseline pose 的 `−n_oi` 构造朝物体的 support point，按到 object anchor 距离确定一次材料点。当前实际 collider 类型是 CAPSULE 与 MESH：capsule 使用解析表面，mesh 对全部 **compiled vertices** 作线性极值，不另造 hull/proxy。不支持的类型直接失败。
3. Mesh 的世界点应从 compiled geom frame 变换；不得再把 asset 的 scale/refpose/centering 变换重复施加。选中 index 必须属于实际 compiled vertex 表，并位于该方向的支撑平面。`−n_o` 是手朝物体一侧；使用 `+n_o` 会选错手背面。
4. 同时冻结 geom-local support direction 为材料支撑法向。它是该点法向锥中选定的一个方向，不冒充唯一曲面法向。优化时 `p_h(q)=x_g(q)+R_g(q)p_local`、`n_h(q)=R_g(q)n_local`；点 Jacobian 用真实 body 的 `mj_jac`，法向 Jacobian 为 `−skew(n_h)J_rot`。固定点后不对 argmax 或 nearest-geom 求导。
5. Exact/near ties 全部记录；固定 first winner 可用于可重复的材料点定义，不能证明原生 deepest-contact 分支全局光滑。调用器须 copy 数组、`setflags(write=False)`，在优化前保存绑定文件及 SHA；`frozen=True` dataclass 本身不冻结 ndarray 内容。

Caller可用独立 `SimpleNamespace(robot,model,data,hand_ids=对应局部IDs)` 为材料选择器提供局部view。不能修改原生全手geometry实例的`hand_ids/pairs/catalog`或过滤规则：最终overlap和native验收仍使用完整273槽及全部原生生成记录。

Root 已有五条被选 capsule 的局部检查；新增[真实 mesh 独审](../outputs/material_mesh_independent_v1/report.json)覆盖10个 enabled finger mesh，世界支撑平面、独立 body/geom 坐标链和 native rbound 一致。固定材料点后，20DOF中心差分最大误差为 point `1.079e−10m/rad`、normal `2.813e−10/rad`（epsilon1e−6）；forward/step/geomDistance/forces 调用为0。这只核材料点分支，不声称这些 mesh 被观测选择器选中、已有接触或具有全局导数资格。

## 一个固定预算的 q0 recipe

每个已冻结假设仅从同一 `q_baseline` 启动一次，不随机重启、不看后续物理成绩选参数。优化变量为20维 q；bounds 为原 hard limits 各内退1e−6rad与 `q_baseline±0.35rad` 的交集。这是优化可行域，不是对输出/初始物理 q 作投影。

使用解析 Jacobian 的有界 TRF least-squares，固定两阶段 anchor amplitude 为0.25、1.0，`max_nfev=40、80`，各 `ftol/xtol/gtol=1e−8`；每假设总 soft wall60秒、总 nfev120。H23、H123最多两个候选，全部终态保留。建议残差为：

| 残差块 | 固定定义/幅度 |
|---|---|
| 骨方向 | 原 `bone_features(q)−source_unit_direction0`，幅度1 |
| 材料点 | `stage_amplitude*(p_h−(a−0.25mm*n_o))/1mm` |
| 材料法向 | `n_h+n_o`，幅度1 |
| 原生 hand-object overlap | 现有0.5mm allowance /1mm normalization残差，幅度4 |
| 原生 hand-self overlap | 同上，幅度4；全部 enabled hand pair 保留 |
| 保持 baseline | `0.1*(q−q_baseline)/0.35rad` |

幅度是进入平方和之前的残差乘数。这些为已见 frame0 的后验工程设计建议，不是调参范围或独立确认协议。Root 实际执行前应把最终参数落盘；若采用不同参数须记录真实 recipe，不能事后将本建议描述为已执行。

每次 q 查询保持 object pose/default wrist/time 不变；`mj_kinematics`、`mj_comPos`、点 Jacobian 与原生 `mj_collision` 足够，不用 forward/step/solver/contactForce。未知 native gap 仍 NaN、其 overlap 项仍0。新增 point/normal 项是材料意图，不改变这个约定。

## 终态验收与明确失败

下列是本 CPU 几何候选的事前工程 gate，**不改原3秒物理任务阈值**：

- source/模型/context/坐标或冻结绑定漂移、材料数组被改、非有限值、非法四元数、碰撞选项变化、unsupported collider 均失败；不换几何或补造 gap。
- 优化器未正常终止、超过预算，或 q/其FP32 cast 超原 hard limits，均保留候选但不合格；不得把最后迭代静默当已收敛或 clamp 初始 q。
- 每个材料点到其固定目标的误差必须<=1mm。`optimizer_success`、`material_intent_attained`与`native_contact_attained`分别保存；局部收敛或“该finger有某处接触”不能替代材料点意图达成。
- 每个假设要求的不同 finger，必须在原生**实际生成记录**中有其冻结 hand geom 与指定 shaft 的 contact witness：`−0.6mm<=distance<=−0.1mm`、native contact point 距冻结 anchor 不超过2mm。只看材料点残差、contact catalogue 或任意其他手指接触不足以通过。
- 对物体的 native contact 法向按实际 geom 顺序统一：object为geom2时取原normal，object为geom1时取负号；要求其与 `−n_o` 的夹角<=30°。同时材料法向对 `−n_o` <=30°。保留所有接触与 ties，不通过事后丢弃最深/坏法向来制造成功。
- 全部生成 hand-object 与 hand-self 最大深度各<=0.6mm；不得只检查意图对。0.5mm仍是软 allowance，允许的0.6mm是新候选验收上限，不是严格无碰撞声明。非手物/fixture 候选接触另行完整报告，不计作手指成功。
- 同原归一化目标的 fingertip均误差<=15mm、bone-angle均误差<=12°，并完整报告最大误差及 q 变化；超过则失败，不以增加接触覆盖拟合退化。若实际 recipe 改这些上限，必须在运行前单独冻结。

通过只能命名为 `geometry_intent_candidate`，`known_observed_contact=false`、`physical_reference_qualified=false`。原生生成接触没有给出承力、摩擦稳定、静力平衡或运动轨迹成功；法向覆盖/两侧分布可报告，但不包装成 force closure。

## 已执行v1暴露的对应反例

Root的[H23-v1](../outputs/frame0_h23_v1/report.json)和[H123-v1](../outputs/frame0_h123_v1/report.json)均局部正常终止，并报告假设finger均有原生接触，但材料意图并未全部达成。H23材料点误差为20.289847mm、0.043525mm；H123为8.948114mm、20.500070mm、0.043473mm。finger2 source landmark7按原对应是link4，跨link最近选择却落到link2 capsule；该选择不因“同一finger”就成为正确对应。两份原结果保留且未取得动力学候选资格。

下一caller-v2修正应只冻结上述landmark→collider对应，保持已执行recipe的权重、object anchor、材料法向、object/wrist、初始q及native全手过滤不变；不同时用调权或换目标掩盖对应问题。v1/v2实际参数以各自冻结recipe为准，不将前文建议的trust box/预算/验收上限事后套成已执行条件。

## q0 与 u0 分离的后续合同

本轮 q0 优化不使用 u0 代替状态，也不让 PD 目标反写物理 q。为了有合法的对照，`u0_control_baseline=C2_020(q0)` 可单独导出，并记录它与 q0 的差及夹量；这不叫“找到预载”。真正 `u0_preload` 保持 PENDING，在 q0 与意图完全冻结后另行设计，限制在原 hard∩ctrl 的20mrad内收域及原 torque 约束内。

未来实际预载必须核实际 affine position-PD、gear/transmission 与 qd 语义，并记录真实执行 u；不能仅凭几何 contactdist 推断 normal force，也不能把 `u0=q0` 当作受力平衡。新的零速 rest 仍是设计干预；任何有界物理检验须另授权，保留固定 q0、object/wrist、实际 q/qd/object twist/u 和完整失败。这里不启动物理、训练或原参考替换。
