# Boundary14 组件固定 AFTER1000 评价合同 v1

**FROZEN SOURCE / METADATA CONTRACT；不授权执行。** 本次只评价已实际训练的同一批152条20ms条目。比较三角色：`initial_unmodified_A2`、`shared_public_losses`（control）、`shared_public_plus_cross_target_consistency`（method）。最终两角色必须来自各自 `update_1000.pt`，并通过本次 boundary 专用 final1000 saved-health；不是 native105/Actor 检查点。该设计是已知数据上的探索性拟合与风险描述，没有独立轨迹、fresh-test、ego或物理泛化证据，不产生胜负、准确率通过门、CI或新超参数选择。

## 固定身份、预算和运行边界

- 仅加载3份受信本地payload：原 owninitial `ea051…a14bf` 和两份 AFTER1000。`initial_pair_state.pt` 仅由已有 final-health 的metadata绑定，不在本job再次加载，也不把它伪装成旧initial schema。
- 复用冻结 H8 strict initial factory：1次 native WM constructor、2份 RAW deepcopy先于任何hook；保留raw作为initial baseline，对两个wrapper严格加载各自final WM/head。所有role的norm/buffer/host fields、initial有效尺度52、xi46、actual m/I、raw palm P、g_p、DeltaT逐字节绑定共同初态，不能由final重新估计或按第0行猜固定上下文。
- **3次 native FW，全部同一CPU FP32/amp=False/inference_mode、threads4、B152，共456预测；辅助head仅2次FW。** 初态z12/mode2使用已实际证明的直接正零初始化作代数基准，明确标记“未在本job另跑第3个head”。不增加validation FW。无AD、optimizer、训练、119f/Actor加载、MJB、FK、Env、物理或GPU。
- forward仅接收逐条原 `x58/u20/dynamics278=(x58,xi46,zero174)`；实际absolute u已执行EMA/C2，不再处理一次。所有target、witness、冲量和缺陷仅进入forward以外的评价公式。缓存/feature-hook必须每call单独且清空。
- 输入仍是旧full05的24 baseline+128 direction，来自真实逐branch PRE/native58、实际u和POST20 target；不得替换成snapshot8重复19、fresh响应或donor。k149使用u149→obs150，不生成u150。
- **第1000条训练日志的loss是在权重after999下、optimizer1000前计算；本评价是after1000重新FW。** 两者身份/时间相位分开记录，不要求loss相等，不据日志选权重。

## 不筛样的13个cohort

按target的 `group152_identities.json` 原bid顺序机械join，保持全部152与所有原source字段。固定13组：ALL(152)、A0(76)、A2(76)、8个(path,k)单元（A0/A2 × k=0/49/99/149，各19）、baseline(24)、direction(128)。各组是同一总体的重叠描述，不是13份独立实验；不按mode、Fn、误差、旧9个stable、eps或结论筛选。每role输出同样13组与完整152行，不做cohort重权；ALL标准loss依原条目/feature等权。

## 同一公式下分列状态、模式、冲量、缺陷

令 `D52=difference(pred58,target58)` 为**冻结原生函数实际输出FP32**：非旋转是target−pred；旋转3维是原生 `Log(R_pred.T R_target)` 的局部chart，不另做acos、投影、正交化或公共pre变换。使用原生quaternion/log归一化规则与既定 `||D_rot|| < pi-.01` domain检查；保存D，不用独立实现重造Log。尺度s52取共同初态的已绑定实际FP32值，禁止用NumPy sqrt冒原Torch逐位尺度。

|量|唯一主定义/统计|解释边界|
|---|---|---|
|native52 point|原FP32 `(D52/s52).square().mean()`，保存逐行/逐feature平方项和原归约scalar；global152×52等权|平方符号不变，但signed D必须保原target−pred。native归约与F64后算参考分开，不能伪称跨kernel逐位复刻。|
|六个原单位块|D52转F64后，q20 rad、qd20 rad/s、p3 m、rotvec3 rad、v3 m/s、omega3 rad/s。每组/块及52逐feature保存signed mean、MAE、RMSE、maxabs；`componentRMSE=sqrt(mean_i,j e²)`，`vectorRMS=sqrt(mean_i sum_j e²)`|不wrap q、不把六种单位合成58维范数。旋转分量沿各role原native residual chart，不冒称共同固定轴的Cartesian姿态误差。另存raw58 F64(pred−target)六块，其中R9仅dimensionless元素误差/Frobenius RMS，不能替代native rotvec或加入point criterion。|
|mode2|原FP32 BCEWithLogits，两个source为HO、nonhand，target是四子步的boundary constraint witness；非Fn>阈值、非self15、非人类真值。另保存sigmoid_FP32概率，F64 Brier=`mean((p-y)²)`|预测bit严格 `logit>0`；0为tie且预测negative。每source给TP/TN/FP/FN、误分类数、tie数、总数；另给“两bit均正确”行数/152。cohort内空真类仅计数或分母0→null，不能填完美率。mode不门控任何预测、loss或冲量。|
|z12|以同一known PRE rawR9：`I0=R0 I_b R0.T`，`sv=s52[46:49]`、`sw=s52[49:52]`；`zJ*= (J/m)/sv`，`zK*=solve(I0,K)/sw`；原F64 `mean((z-z*)²)`|保留两个source×J/K的dimensionless分量和统计；无mode mask、无“无接触即置零”的另行规则。|
|J/K物理量|`Jhat_c=m*(sv*zJ_c)` (N s)，`Khat_c=I0@(sw*zK_c)` (N m s)，各自减同源saved J/K|每source与两source总和分别报告componentRMSE/vectorRMS/maxabs；J和K不能混单位。总和可能抵消，不能用小总误差遮盖分源错误。冲量已经含四步dt，不再乘DeltaT；K仍关于object COM，只在raw P.T坐标轴下表达。|
|cross-target|`I1hat=Rhat I_b Rhat.T`；`rvhat=vhat-v0-(sum_c Jhat_c+m*g_p*DeltaT)/m`；`rwhat=what-solve(I1hat,I0*w0+sum_c Khat_c)`；`Lcons=mean(concat((rvhat-rv_saved)/sv,(rwhat-rw_saved)/sw)²)`，全F64|使用predicted rawR9，不能用target R1、伪逆/投影或把saved缺陷置0。报告两类差的原单位统计m/s、rad/s，同时保存rhat及r_saved原值；这是同源离散observer缺陷拟合，不是连续零缺陷物理证明。|

三个role都报告相同的 `Lstate/Lmode/Limp/Lpublic/Lcons`：严格保持 `Lpublic = FP32(Lstate+Lmode) + F64(Limp)` 的原加法顺序。可同时列共同诊断 `Lpublic+Lcons`，但不能拿各臂不同训练total排序宣称胜者。真实训练目标仅control=`Lpublic`、method=`Lpublic+Lcons`；initial无“经过新训练”的目标身份。最终展示method−control及每final−initial的逐指标差值与tradeoff，不创建综合分数或事后成功阈值。

## 未监督的原self15风险

本job没有加入可证明相位匹配的native sensor label，**不算self15 label BCE/准确率**。保存全部3×152×15原logits及sigmoid_FP32，描述final相对本次initial的逐feature/全体signed drift、MAE/RMS/maxabs、strict-logit>0 bit变化及tie数。PenSimReader.contact取finger_collision的middle及(distal|tips)槽；不是本新HO/nonhand模式，也不是HO力/人类contact。

另保存源绑定的soft finger-collision代理：`rate_FP32=-sum(sigmoid(logits),axis=-1)`，`reward_component_FP32=rate_FP32*FP32(.02)`（原50Hz PenMDP，dt只乘一次）；报告各final相对initial的差。概率/惩罚向下漂移可能造成低估自碰风险，向上漂移可能造成过度惩罚；两者都不是实际碰撞错误率或完整task reward改善。禁止为本次补跑传感器、构造MDP/goal、计算其他reward/FK或据此加载Actor。

## 完整输出、守门与可独审范围

先保存全部role原FW结果（pred58、self15，final z12/mode2，initial正零代数标记），再保存原生D52、s52、target/x/u/xi、逐行role/source IDs、全部loss操作数与RHS/solve residual、13组统计和152行CSV。保存raw before/after Python/NumPy/TorchCPU RNG及headers、input bytes、norm/context buffers、参数/状态digest关系、hook/call预算、源ZIP/pins和samefile路由。完整state未另序列化时，参数不变digest只能是producer源断言；不能冒独立weight重读。

形状/来源/finite、原chart domain、rawR1 inertia solve的有限/非奇异性是完整性条件，不是新accuracy门。任何required行失败：保原行、mask/NaN/原始输出，整体 `evaluation_complete=False`，不筛到成功子集、不修复或重试。保存所有原冲量/缺陷，不能用残差小来决定enrollment。source/domain失败和科学误差大小分开。成功完成也只表示固定训练条目的可复核描述，不升级完整hidden restore、真实Jacobian、force capacity、物理/任务/轨迹泛化资格。

### 本合同绑定的元数据/源码

所有路径均在 `D:/workspace/supermimic-full151-observation-study/` 下；metadata严格samefile+SHA，下面只是身份，没有加载权重/数组。

|相对路径或角色|SHA256|
|---|---|
|`protocols/full151105_boundary_impulse_matched_training_recipe_v1.json`|`ed5e6cc378f80d94e1771607b8fd4ed1198ce31eec80e17c0465d9933ae4f855`|
|`src/full151_gradient/boundary_impulse_component105_v1.py`|`003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317`|
|`src/full151_gradient/component_H8_donor105_v1.py`|`0c77dc5255ffd2d7294dabdc7bbaf8ed09e80f3f2bc0eb89fee3954d268a6b13`|
|`docs/full151_boundary_impulse_target_dataset_source_contract_v1.md`|`736250610d3d9fd5b3cc39e7c714fe3dae17f9f0498b4811795d8f2ebc487cd4`|
|target `outputs/full151105_saved_boundary_impulse_targets_CPU_v1/all152_boundary_impulse_targets.npz`|`bc22e2b0d6a6ff333f8431eee689134814df1ea7059e8346f27e74f7b0392f80`|
|target独审 `outputs/full151105_saved_boundary_impulse_targets_independent_v1/report.json`|`3703c02f0e72753582ab5e8ff52b7ab79bc5409220e06e7fd7282a0e009851ed`|
|initial `outputs/full151105_WM_common_pre_calibration_CPU_v1/initial_component_state.pt`|`ea051225b7279fc69fbd97725eb15462faaf4b3a6706881f8fe4a05b658a14bf`|
|initial parity独审 `outputs/full151105_boundary_component_initial_parity_saved_independent_v3/report.json`|`da71d009aa7cb08e0caccf0d1d4f45337410d303dfba3fcbb1f3bf5aea5202fe`|
|boundary final1000独审 `outputs/full151105_boundary_impulse_final1000_saved_independent_v1/report.json`|`85267fc2055e09a96fff26917a939308392238d5bc22c48cb9ad9fa1829b9a29`|
|control `outputs/full151105_boundary_impulse_matched_training_CPU_v1/shared_public_losses_update_1000.pt`|`0e8376586e1065f01ff8aa2375ee670f9bd4abc1171f7ba0ef482aa37e89ebe1`|
|method 同目录 `shared_public_plus_cross_target_consistency_update_1000.pt`|`8d2e3281f7b8ea3c74a6010a9d573aa09f2fe17347d1461d9b51b29a02838719`|

原生源码绝对路径 `D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/`：`state.py` difference:30–43 SHA `4a95cec1951d4e54fac9256f757ca0d9a1f3cd81d66e9b97564954e3f55add06`；`pen.py` rewards:143–164/contact:347–349 SHA `c02906c409aedab59aaf68a5ef80f9dbcea52da10e72137091df81858262819a`；`runner.py` imagined sigmoid入口:739 SHA `b19d9ac80e23e4f55b2eb64e458c05516fc5dd4ffb13e1a3050ce9fea2669d46`。实际producer仍须新协议、source review及Root单次执行授权；本页不修改任何已执行源/结果。
