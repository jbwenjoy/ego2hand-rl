# 原152恢复组的分源边界冲量 WM target 数据合同

**SOURCE / METADATA ONLY；尚未实现或计算标签。** 本合同收窄[a81d方法建议](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_WM_method_source_feasibility_v1.md)到一个可执行的数据适配步骤：原 full05 的 **152个20ms restored groups = 24 baseline +128 direction =608个5ms子步**。不含300 donor，不含 fresh calibration test152；不按接触、误差、旧stable9、响应大小或任务成败筛选。本页仅读源码/已有JSON身份和文件hash，未读取NPZ值构造新数组、未作冲量/残差运算、未加载NN/MJB或运行物理。

## 1. 精确 group / 子步 / 原校准输入 join

唯一主键是 `(path, k, branch_id)`，再加 `substep=1..4`。`path_i=0/1` 对应A0/A2；旧相位列表严格为 `[0,49,99,149]`，`phase_i=0..3`，`s=4*path_i+phase_i`。分支 `offset=0..18`，`bid=19*s+offset`；offset0..2为三baseline，之后是原EPS-major方向：`offset=3+8*epsilon_index+2*j+signbit`，`j=0..3`对应控制坐标 `[0,4,8,12]`，signbit0/1对应−/+。不得把j-major64 pair顺序当152 branch顺序。

在 H1 的452组中，**每条path先150 donor，再76 restored**；不是前300 donor接后152 branch。因此 `g=226*path_i+150+19*phase_i+offset`。必须同时核对[原452 identities](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/group_identities452.json)与[原152 metadata](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_CPU_v1/all152_branch_metadata.json)中的 `group_id/path/mode/k/branch_id`，不能仅靠公式猜行。原物理子步行为 `t=4*g+(substep−1)`，firstPRE=`4*g`、lastPOST=`4*g+3`，post20 nativeGPU replay行为 `1808+g`。线/角动量 JSONL 的原 `row_id/path/mode/branch_id/phase/substep/canonical ticks` 逐字段匹配该组；4步内 raw PRE→前步POST 的qpos27/qvel26位链及actual ctrl20四行位恒定在新适配中再核。k149仍为u149到obs150的终态响应，不能生成u150或多一决策。

权威WM输入/目标为[H1 matched_input_arrays](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/matched_input_arrays.npz)中的 `pre_x452_FP32[g] / actual_u452_FP32[g] / target_x452_FP32[g]`，分别来自 firstPRE原生GPU58、真实absolute ctrl（已执行EMA/C2后）、lastPOST20ms原生GPU58；不再做EMA/C2。要逐字节等于旧response `all152_branch_raw.npz` 的 `pre_visible152_FP32 / actual_u152_FP32 / target_y152_FP32` 展平到bid行，并等于已经实际校准使用的 `matched_fit_data152.npz` 的 `pre/u/target`。**原校准 `pre` 是逐组真实native58，不是 fresh eval 的snapshot8×repeat19，也不是用于common-pre chart的统计广播。** 旧shared-state门只对q/qd40位相等、其他18数值相等；不能把它改称全部58位相等。这里只要求每个bid对应的三个保存副本彼此逐字节同源，任何不符整体阻止enrollment，不修复/挑行。

`physical_xi46_FP32`取 H1 `prediction_arrays.npz` 的已冻结原值，逐字节绑定固定context的 `reader_context[0]`；原initial/两component的xi资格继承已存审计，不为本适配读取Torch payload。native58块为 q20、qd20、p3、R9、v3、ω3（v/ω在palm轴；R为object-body→palm）。self15仍是自碰传感器，不是本标签的HO模式。

## 2. 世界冲量与固定palm坐标

对每个bid的四条原行、每源 `c∈{HO,nonhand}`，只用冻结线/角动量输出：

```
h_t = F64(dt_wp32_s_FP32[t])
J_c_world = sum(t=4g..4g+3, h_t * F_c_world_F64[t])       # N s
K_c_world = sum(t=4g..4g+3, h_t * tau_COM,c_world_F64[t])  # N m s
DeltaT = sum(h_t)  # 不取post_time-pre_time，不换成F64(.02)
```

线性来源键是 `FHO_N_F64/Fother_N_F64`；角来源键是 `tau_HO_total_Nm_F64/tau_other_total_Nm_F64`。后者已经由该子步solver-PRE的原world `contact.pos−object_solver_COM` 叉有符号force，再加有符号direct torque **一次**；本适配不重新解码pyramid或重新算wrench。这里的FP32是**实际源timestep**；F/τ是原缓存force6经冻结F64 frame/lever算术得到的source-backed量，不冒称全部为原GPU FP32冲量真值。保留HO/nonhand两通路，不能拿只有HO的旧聚合代总外载；内部object pair净冲量取消而原索引保留。

坐标权威为[qualified GPU packet](D:/workspace/supermimic-full151-observation-study/outputs/full151105_matched_native_GPU_reader_v2/prestate1808_qualified_for_future_scoring.npz)的 `palm_rot[t]`（FP32 body/palm→world）及 `palm_pos[t]`。令每组 `P=F64(palm_rot[4*g])`，`B=P.T`，**不使用matrix inverse、正交化或新四元数转换**；要求4个子步的记录frame按原字节相同，并与该组原reward记录 `state20_reader_palm_rot/pos` 相同，然后 `J_c_palm=B@J_c_world`、`K_c_palm=B@K_c_world`。K仍关于物体即时COM；这里只旋转轴，不把力矩原点搬到palm，不额外加palm杠杆项。A0/A2分别使用各自记录，不按近似相等静默统一frame。

现有frame资格的准确范围是：452个原Reader记录frame、固定 `wrist_pose_w151`、startup/reset mocap读回、以及固定parent拓扑（mocap id0是mocap空间，不是body world id0；palm为无关节identity child）。forensic/replay把每组Reader frame分配至其4个PRE，dense新p/R alignment已通过；**不是1808个直接独立的mocap测量**。本数据合同沿该source-conditioned固定frame，不提升为新的全58或完整隐藏状态资格。Native Reader `state()`明确用 `palm_rot.T` 旋转速度/姿态，见[sim.py:95](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/sim.py:95)。

## 3. 模式与离散缺陷标签

模式 `boundary_constraint_witness[bid,c]` 为四子步中至少一个该源外边界contact具有合格的 `type&1/world0/host address/raw slots/finite force-frame-pos` 证据，**不要求Fn>1e−6，也不要求非零冲量**。类别机械继承完整native catalog及原angular contact长表：HO或`other_external`；terrain等非手接触保留在后者。无该源候选时，0冲量/False只意为本缓存未观察到该源约束，不能称全局无接触/clearance或真实人类无接触。所有608子步若存在未知外接触/无效required endpoint/context/frame，保raw与mask并整体阻止训练dataset enrollment；不补0或删组。此门只检查来源/完整性，不能按残差大小设门。

实际质量 `m` 绑定线动量metadata的actual FP32 readback来源，`I_b`取angular `actual_inertia_observer.npz:I_body_F64`；后者是实际FP32 principal inertia与body_iquat在两path两stage全field位绑定后，**明确F64归一化body_iquat的observer**。不得用nominal惯量、xi中近似值或新MJB替代。重力 `g_world`取线动量metadata的source-effective WP FP32 gravity转F64（保disable语义），`g_p=B@g_world`。令同组native58原始R9转F64为 `A0/A1`（不投影/换成angular归一化q重建R），`I_p0=A0 I_b A0.T`、`I_p1=A1 I_b A1.T`，则标签定义为：

```
r_v_saved = v1_p - v0_p - (J_HO_p + J_nonhand_p + m*g_p*DeltaT)/m
r_w_saved = omega1_p - solve(I_p1, I_p0@omega0_p + K_HO_p + K_nonhand_p)
```

两者单位分别m/s与rad/s；保存每个原操作数、rawR9的det/正交残差及solve有限/非奇异标记，不投影、调容差或让一行失败后继续筛样。若solve不定义，整体停止enrollment。原state R/ω涉及FP32 native旋转，冲量及惯量observer是F64，因此这些是**同源离散observer缺陷**，不是精确连续物理定律的零标签。尤其 `r_w_saved` 必须原样保留，不能置零；可另保同4行冻结angular的 `Σ(delta_L_world_F64−J_contact_Nms_F64)` 作另一observer的世界角冲量缺陷描述，但不能假称它与native58 r_w逐位相同。世界ΔL方程不再添加gyro外力矩，finite implicitgyro/姿态步/solver stationarity与observer算术缺口仍存在。不得据小残差授予完整扳手、human force、force-cap或梯度资格。

## 4. 可执行下一步与缺口

唯一建议下一scope：Root另冻结一个 **saved-only NumPy dataset adapter，一次全152/608**；0Torch/NN/FW/AD/MJB/FK/physics，复用已有source inventory一次并按samefile+SHA去重。输出原 `x/u/target58/xi46`、152×4子步索引和身份、raw固定palm frame、两源 `J_world/J_palm/K_world/K_palm[152,2,3]`、`boundary_constraint_witness[152,2]`、actual m/I、`r_v/r_w[152,3]`、所有guard masks及原缺陷；保存CSV/pins和完整失败资料。无模型训练、无科学准确性阈值、无新force decode。**本页不是该次执行授权或训练recipe。**

仍需实际适配时核验、当前不得补造的缺口：(1) 152 fit/H1/native raw副本及608身份的逐字节join尚未作为新dataset实际运行；(2) 固定frame有source/452记录支持，缺1808直接mocap样本；若研究要求该更强资格则本缓存不足；(3) r_w中的rawR9与归一化惯量observer并非同一GPU计算路径，缺陷必须保留；(4) 隐藏graph scratch、effective qacc/solver grad/generalized passive等未保存，不可由J/K标签复原；(5) 旧152、donor与fresh结果均已见，不再blind，不能把新loss的探索性效果称独立轨迹/ego泛化。新recipe、两臂共享旁支初始化/norm/数据和任何fresh确认仍另待Root决定。

## 5. 固定输入 pins（只hash，不读取数组值）

下列为原文件身份；完整源闭包继承已执行报告，不能把同hash的不同物理copy合并。每条pin比较使用 `Path.samefile + SHA256`。未列出的原physical context/readbacks/catalog/report/sourceinventory精确pins由linear/ angular报告 `input_pins` 继承。

|输入|SHA256|
|---|---|
|[方法来源](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_WM_method_source_feasibility_v1.md)|`a81d1065119a3ef8f3cfa392ad60d96e59f02230bd508ab731f42a8aa7a7c66d`|
|[H1 matched_input_arrays.npz](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/matched_input_arrays.npz)|`18a361bbc6a56139e79df5b6891e3aa80e030abebbbee4fa39e3dbe5129e9cb4`|
|[H1 prediction_arrays.npz（xi）](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/prediction_arrays.npz)|`84fe797f5da806160dc891422961afd606bdcda1bb3e96ffc99cb426761261b1`|
|[H1 identities452](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/group_identities452.json)|`0151e53864bbf26840b4435863c7038a957ba94dd142e8385deadfe18f0d167b`|
|[H1 report](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/report.json) / [saved25](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_saved_independent_v1/report.json)|`202b4e15512491b15fd8fc3fa3efb0f1c9da85a17ce359b372ee619ea535175a` / `b2675cbaaa46c5575d5f9084a64c2bf7b28c7eabe13715b600347e5485cc4d20`|
|[response raw152](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_CPU_v1/all152_branch_raw.npz)|`80a8a89768c95df419f404669e57eeac0a02863777ee85135e1ffcb83e3bc632`|
|[response metadata152](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_CPU_v1/all152_branch_metadata.json) / [saved22](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_saved_independent_v1/report.json)|`5d8baa0218a57de29db5ceea0a7534342672f638d919e03f1b2d9f3977d93553` / `b70a9b4e63b13401a86a282046f19a71fb2c5019c185a851e91a1ce6b86de481`|
|[calibration matched_fit_data152](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_common_pre_calibration_CPU_v1/matched_fit_data152.npz) / [protocol](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_common_pre_calibration_CPU_v1/protocol.json)|`18f3b6e5802df49dfbadbbc56b81a2797135775e02c818c87e734359bb51fd21` / `c4d20d4002c02e4a1cafce105c76ce1f5c43b26d6b62e3e05a410ab79946f65d`|
|[linear arrays](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_linear_momentum_CPU_v1/all1808_momentum_arrays.npz) / [rows](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_linear_momentum_CPU_v1/all1808_rows.jsonl)|`8c710817ed89918a923b83e32100ea018264908ad8b049d3c6843972b9f61306` / `36a8031ea1ae487d62a107430493ee9091808c5503e908a6038eeaef35c620ee`|
|[linear report](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_linear_momentum_CPU_v1/report.json)|`e2a866c71e15357c18bb9e055274df761b0795b036ef0fd3a9c633811a465172`|
|[angular arrays](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_CPU_v1/all1808_angular_momentum_arrays.npz) / [rows](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_CPU_v1/all1808_rows.jsonl)|`570ec3a4bb120ac756929229fff53ec6abae8b0ef35ec9086dc5c76a5644fef8` / `784c3f992f36085940919b6ec11b900f9dc6821f4e146c28ad1225316474a15a`|
|[angular contact长表](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_CPU_v1/all9473_generated_contacts.jsonl) / [actual惯量observer](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_CPU_v1/actual_inertia_observer.npz)|`b9b2901cf2bf0adc736ec123a43438f4160ef232c5c6096d002c73102ea2c1d2` / `beb859b03ba82cefd210a03c9329c5630cd1c523dab71265fd88dc223d3981a9`|
|[angular report](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_CPU_v1/report.json) / [saved9](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_angular_momentum_independent_v1/report.json)|`63f797255bc3f7418135d0cdb5a0d578d116774bead974a734695caa146cbf3f` / `f208f626aea05797aec2337055cfa29a03bbe163db896303ae598276cae542c0`|
|[qualified packet（palm）](D:/workspace/supermimic-full151-observation-study/outputs/full151105_matched_native_GPU_reader_v2/prestate1808_qualified_for_future_scoring.npz) / [replay saved29](D:/workspace/supermimic-full151-observation-study/outputs/full151105_matched_native_GPU_reader_saved_independent_v3/report.json)|`4d205ee2c7cc878745f4575e572622eb3033a81a49408d6cef76dda008b56de5` / `a31cb81a6e513fea013591a9198b86a0ad398a148164f18a0bedfcdb7ec9c3eb`|
|[forensic原frame operands](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_CPU_v1/raw_conversion_all2260.npz) / [dense state qualification](D:/workspace/supermimic-full151-observation-study/outputs/full151105_solver_phase_penalty_GPU_qualified_CPU_v1/state_input_qualification.json)|`86466338c4a6b74512e593d340ab0f9024999d0bcf8be781cbf1552e937ed64d` / `ec2ce46ca3d6bceef7469aa381070012e31e5c050de16c4df62028b2e13ea650`|

关键已执行源码：H1 `state_prediction452105_v1.py` SHA `5cd15a0a3fefd2f17bbb971944fbc7c47fa8faaab473a9164728583598e2e240`（`expected_schedule/assemble_groups`）；旧response `action_response452105_v2.py` `17cb3805ab0e4906f0ca73ff6f289b9bdf525aafd22000af0498387cbac84af3`（`prepare_inputs`）；校准 `response_calibration105_v1.py` `71efe05da17a37df4111158860443433304f1d57762e2377d4dbafc3b57ca27f`（`_prepare_data`）；native Reader `sim.py` `47d0523d8b0d569544f2873a2c7c8a17cad27e570154d6220ec98d1a97aa5d20`。这些都是来源，不是新的调用/训练授权。
