# 共同 native self15 监督的匹配训练合同 v1

**SOURCE_FIXED_PUBLIC15_LOSS_AND_LABEL_CONTRACT_NOT_TRAINING_AUTHORIZATION。** 这是新实验合同。只读源码、提案、已完成JSON报告及label-only身份/schema；未读pt/npz、未算新loss、未构造模型/FW/AD/训练，也未重跑既有审计。旧 `ed5e…` 配方、003 wrapper、两臂旧权重、报告及评价原样冻结。新代码、protocol、artifact和health必须使用新namespace。

## 共同标签与起点

两臂同样加入native15监督；control/method的唯一loss系数差仍是cross-target consistency的0/1。从同一个未修改A2 owninitial `ea051…a14bf` 重建，而非从已训练的旧final1000继续。native WM14个Parameter tensor加aux14头的2个，仍为16个；不新增15维网络头。RAW复制先于hook，独立storage/cache、同一零aux模板、相同FP32初始化/norm/host/error_floor/effective52/xi46/physical context；新Adam状态为空。

监督只读已保存的 **label-only** `joined152_cached_sensor_labels.json`（SHA `8b900a1b63d620aff3a96474b81f47f9dcd92efaaae1bb83dbbc2c364a63f698`）。不能把含三模型概率/误差的metrics NPZ当label，也不能用mode2、预测概率、HO、人类标签或正Fn筛选替代它。

其来源为 cached15 producer `f7d3ddda3a0cde30b2a10fd52b175ffa20a222ede60a951e40185f384721402a`，及独审 `caa1474c84f297f1080432427a5be6665cf21f47a1d680dd8e442a5c30d3d18f`：`schema=full151105_cached_self15_saved_independent:1`、`status=PASS_ALL152_ALL3ROLES_CACHED15_METRICS_NOT_GEOMETRY`、7/7。独审顶层 `producer_report` 绑定前者；它没有直接label pin字段，因此新protocol必须显式pin label文件，并核其与producer同目录 `output_pins.json['joined152_cached_sensor_labels.json']` 的samefile+SHA和上述固定值，不能猜测不存在的报告字段。

新trainer/runtime必须再次逐行核：152行、row_id=0..151、`snapshot_id/path/k/branch_id/group_id/mode/physical_rows/terminal_observation` 与原 `target_identities` 全部一致；`contact15` 长15且0/1可精确表示为FP32；`raw_found25` 长25；source_frame=200+k、terminal=k+1、valid_next_policy_decision=(k<149)、no_u150=True及原phase字符串不变。固定A0/A2×k0/49/99/149×19，含baseline24/direction128，最后u149→obs150完整保留；无donor、fresh、三baseline平均、class重采样或按模型结果删行。任何缺失/重复/错序/未知/phase不符整体阻断两臂，不补0、不退回旧无监督配方。

已完成的标签资格来自原reward JSON中finger_collision found25：每指stride5的offset1/2组成middle10，offset3 OR offset4组成末5；是二值OR，不是计数相加。link1不进入15。原JSON缓存是 **lastsolver约kdt+15ms**，state target是POST20ms；保持原native混合相位监督合同，不能称POST20碰撞几何、HO/人类/力或全局clearance真值。physical_arrays与JSON的关系仍委派原writer/1762及已完成独审，不为本次新增物理或重读该数组来造标签。

## 精确loss、dtype和归约顺序

对每个n=0..151，同一次native FW输出pred58和未经sigmoid的ell15；标签只进loss，forward仍是原逐branch `x58/u20/dynamics278=(x58,xi46,zero174)`。actual absolute u不再EMA/C2。D52必须是冻结native `difference(pred,target)` 的实际FP32输出，非旋转target−pred、旋转保原Log/chart规则；s52使用共同初态实际FP32 frozen effective scale，不用NumPy sqrt冒原kernel逐位真值。

```text
point_row_FP32[n] = mean_52((D52_FP32[n] / s52_FP32)^2)
self_row_FP32[n]  = sum_15(BCEWithLogits(ell15_FP32[n], y15_FP32[n], reduction=none))
L_state_FP32      = mean_152(point_row_FP32)                 # 纯point，不改旧字段含义
L_self15_sum_FP32 = mean_152(self_row_FP32)                  # sum15，绝非mean15
native_pair_row_FP32[n] = point_row_FP32[n] + FP32(.1)*self_row_FP32[n]
L_native_pair_FP32 = mean_152(native_pair_row_FP32)           # 必须逐行加权后归约
L_public = FP32(L_native_pair_FP32 + L_mode_FP32) + L_imp_F64
L_total_control = L_public + 0*L_cons_F64
L_total_method  = L_public + 1*L_cons_F64
```

所有行有效，W1；native pair归约分母152。不能用已归约的两个scalar相加宣称同一FP32结果，不能变成`.1*mean15`、除positive数、乘8、乘reward dt或除四步DeltaT。mode2 BCE保持原FP32 all152×2 mean/权重1，impulse保持原F64全152×2×6 mean/权重1；两臂都计算并记录F64 Lcons，使用预测rawR9/I1 solve和同一非零r_saved，仅优化系数0/1不同。mode或self15标签不门控任何loss/预测，不修正推理输出。

canonical权重依据本机冻结原生源码：`config.py:41–42` 默认 `final_state_weight=1.0/contact_weight=.1`；`runner.py:539–544` 为state mean52+self15 sum15，`:584–588` 实际优化 `sum(mask*loss)/batch_size`，日志component才除mask.sum。本实验W1/all152 valid，故采用上式mean152。这里只证明原源码默认/归约来源，不从默认值臆测所有历史job的runtime flag或把本次离线训练等同原H8/replay训练。0.1不是环境self-collision reward权重、几何P项系数或本次BCE/Brier结果择优得到的新参数。

新helper必须显式命名/保存 `L_state_FP32/L_self15_sum_FP32/L_native_pair_FP32/L_mode_FP32/L_imp_F64/L_cons_F64/L_public/L_total`，不把pair偷塞旧 `native_point_loss` 或 `L_state`。003仍只接原5项impulse labels；若复用它计算原几何/cons算子，新helper仅传原5项，再另算新public/total；不得修改003严格字段集合或伪称它已原生监督15。新label字段、digest、schema显式独立。

## 固定预算、记录与新健康

预算沿原已冻结匹配recipe：CPU4、native FP32/AMP false、fullbatch152固定顺序、1000 lockstep updates/臂、AdamW lr1e-4/wd0/betas(.9,.999)/eps1e-8/foreach=False、clip1/error_if_nonfinite、仅两臂checkpoint100和1000。norm/context不更新，不按类先验重置15 bias，不加pos_weight、smoothing、temperature、阈值训练或参数扫描。每臂每次1 native B152 FW+1 same-call aux FW、1 backward/step；self15 BCE只消费该次输出，不增加NN call。首training FW同时比原saved initial native58/self15及aux正零，无额外validation FW。失效留partial整体停止，不自动retry/resume/放宽门。

输入建议严格16 aliases：旧12 + `cached_sensor_labels/cached_sensor_report/cached_sensor_saved7/target_identities`。新protocol还应hash绑定本合同和最终source review；metadata仅读JSON/hash。新namespace固定为protocol `full151105_boundary_public_self15_matched_training_CPU_protocol:1`、role `matched_native_self15_public_aux_vs_cross_target_consistency`；checkpoint/pair/shared-initial分别 `full151105_boundary_public_self15_WM_component:1`、`full151105_boundary_public_self15_matched_pair:1`、`full151105_boundary_public_self15_matched_initial:1`。不能重标旧ed5e检查点、借旧5健康或旧native105资格。

保持原 `fit_input_digest` 和原5项impulse labels的 `fit_label_digest`（保旧key/含义）；另存 `fit_self15_label_digest` 及 `fit_combined_label_digest`。两臂完全相同；label digest须涵盖真实152×15次序，并绑定label/target_identity来源，不能只靠正负类数相等。全部1000行日志分别记录8项 `state/self15/native_pair/mode/impulse/consistency/public/total`。

checkpoint100/1000须保用于核FP32归约的 **PRE-I** `point_per_row_FP32[152]`、`self15_sum_per_row_FP32[152]` 和native_pair_row，明确权重afterI−1；按新recipe同时保存detached的实际native D52/scale-square、self15 logits/target/BCE elements及8个scalar。scalar日志不能代替逐行pair算术证据。保存原始16项clip前/后梯度、AFTER-I权重/Adam实际step、norm/context/host、initial和双方THIS-run raw Python/NumPy/TorchCPU RNG、完整I行log prefix bytes/SHA。持久RNG比较用raw类型/数值/向量bytes，不用跨torch.save重序列化pickle digest。reader只对保存的原生算子结果/原操作数复算算术，不能声称独立重建nativeLog、NN输出或从AFTER-I权重重算PRE-I梯度。

新reader独立读取本次shared initial+两固定checkpoint；实际16参数/全部Adam states step=I、finite、公共label/系数/norm/init一致、真实FW/backward/step计数、PRE/AFTER相位及prefix范围均必须核。100健康 `final_iteration_scope=False`；仅实际1000完成可为True。两新100健康与本次实际elapsed/剩余900pairs ETA记录完成后，Root按用户条件暂停Goal；后台是否继续仅由本job显式protocol授权，本合同不代替授权或宣布Goal完成。

## 来源及解释界限

- 提案：`docs/full151_boundary_public_self15_supervision_recipe_proposal_v1.md`，SHA `9f4aaa362e6a86023c834b0f9730236ac9cffdbcdb81b40dab7ef7d81759c0da`。
- Root已冻结的新数值recipe：`protocols/full151105_boundary_public_self15_matched_training_recipe_v1.json`，SHA `9e0765fd562c3e24bdb09d52f78219204af4228bb6e8b1d8ffd67104ce094bce`；本合同不替代它或执行protocol。
- native config：`D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/config.py`，SHA `11889b65b5da121ea613d402b3ef9e7bc854d57462f96830af9b644a0b031323`；同目录runner.py SHA `b19d9ac80e23e4f55b2eb64e458c05516fc5dd4ffb13e1a3050ce9fea2669d46`。
- 缓存phase/映射合同：`docs/full151_boundary_final_self15_evaluation_source_contract_v1.md`，SHA `a535112324eeddb9113ebf47b2d68e379c3ddf5f191604eea498e756a30254f5`。本轮只机械核了label-only JSON的152身份/schema及source/copy关系，未计算新class统计或loss。

这是在已见同一缓存结果后提出的一个公共监督修正，不能称blind确认或独立轨迹泛化。后续固定final1000/同initial评价仍全152/既定13重叠cohort，分列state、mode、impulse、cons、self15 BCE/Brier/漏报误报/tie及Pen proxy；不以多数负类accuracy、单一Brier或惩罚变小宣布成功。新健康通过也不保证post20几何、真实接触/力、Actor/CHORD收益、完整hidden restore或物理Jacobian资格。数值recipe已冻结，最终源码和执行protocol仍须另立；当前不执行训练或新评价。
