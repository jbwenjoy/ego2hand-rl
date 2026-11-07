# Public native self15：固定 AFTER1000 评价合同 v1

**SOURCE_FIXED_FINAL_EVALUATION_CONTRACT_NOT_EXECUTION_AUTHORIZATION。** 本合同只读新family源码/recipe及checkpoint manifest身份，没有读pt/npz或新预测/损失，没有NN/FW/AD/物理。旧无self15实验、旧负面结果与原报告全部保留；本次是已见缓存结果后的探索性训练集评价，不是新的blind确认、独立轨迹泛化或Actor收益证据。

## 唯一起点、固定终点与预算

比较三角色：`unwrapped_baseline`（原未修改A2 initial）、`shared_public_losses`（control）、`shared_public_plus_cross_target_consistency`（method）。只取新family两个固定AFTER1000，不看/选择中间权重；PRE1000日志是在after999模型下优化前计算，不能当本次AFTER1000重新前向的loss，也不要求两者相等。

本job恰好 **3个own CPU torch.load**：原initial `ea051…a14bf` + 两个新final。复用冻结H8 initial factory **1次native constructor、2份RAW复制先于hook**，raw作为initial，两个wrapper严格加载各自新WM/head。不能读取新shared-initial再伪装旧schema给H8；该initial_pair仅由新final-health作metadata provenance。final必须满足 `full151105_boundary_public_self15_WM_component:1`、pair `full151105_boundary_public_self15_matched_pair:1`、iteration1000、`AFTER_OPTIMIZER_I`、正确arm及真实training protocol `inputs` ↔ payload `input_pins` 的显式映射，不能接旧boundary family。

**CPU4、native FP32/AMP false、同一inference_mode，3×B152 native FW=456预测，aux head只2次FW。** baseline辅助z12/mode2是已实际证明的初态正零代数，不另跑第3head且必须明示；raw baseline的58与self15确实FW，并逐位等于e427缓存。无额外validation、AD/optimizer、119f/Actor、MJB/FK、Env/physics/GPU。不作推理校正、mode gating或teacher input。norm/host/error_floor、实际初态s52、xi46、m/I/P/g/DeltaT保持共同初态原字节，不重估、不按label调整；三个模型和cache/storage独立。

## 152行、标签与13个cohort

固定17 semantic inputs：此前final-eval的14项（`final_saved_health5` 改名 `final_saved_health`）+ `cached_sensor_labels/cached_sensor_report/cached_sensor_saved7`。最终protocol必须pin新family final1000独立health，要求schema `full151105_boundary_public_self15_checkpoint_saved_review:1`、status `PASS_SAVED_PAIRED_PUBLIC_SELF15_COMPONENT_HEALTH_NOT_ACCURACY`、全部checks通过且final_iteration_scope=True；旧health5/29不能代替本次newfamily final健康。

同一旧target152/608：逐条真实PRE58、actual absolute u20、POST20 target58，不能用snapshot8重复19或fresh/donor替换，不再EMA/C2。target NPZ group_id/substep arrays、原 `group152_identities.json` 与label-only JSON逐行深核 `row_id/snapshot_id/path/k/branch_id/group_id/mode/physical_rows/terminal_observation`。self15只来自固定 `joined152_cached_sensor_labels.json`（`8b900a1b…f698`）的binary contact15；使用f7d3 producer和caa147独审7的来源，不能取metric/probability archive作监督。沿796c `cached_label_rows` 的raw_found25→middle10+distalORtip5及原phase/source_frame/no_u150守门；所有152/15保留，任何missing/duplicate/unknown/mismatch整体拒绝，不能补0或删行。

标签是lastsolver约kdt+15ms finger sensor缓存，state target为postintegration20ms；不是新测POST20自碰几何、HO、人类、force或global-clearance truth。k149→obs150仍保留，没有u150，不因无下一policy decision丢掉末条。physical_arrays↔JSON的关系仍为已存writer/旧独审source资格，本job不重新打开该数组或运行传感器/碰撞。

三个role的固定13组：ALL152、A0/A2各76、A0/A2×k0/49/99/149各19（8组）、baseline24、direction128。保完整bid顺序和全部行；13组是同一总体的重叠描述，不是独立N。没有旧stable9、class/误差/接触/响应大小筛选或cohort重权。

## 新公共loss：唯一主口径

必须调用冻结 `boundary_public_self15_training105_v1.py` **796c…94e7e** 的 `public_self15_losses`，相同context/target15、原003五项boundary labels；不要复用旧无self15 `L_public`。D52为冻结native `difference(pred,target)` 实际FP32输出：非旋转target−pred；旋转为native Log原chart，保`.01` margin，禁止新acos/投影/正交化或独立重造Log。s52使用实际绑定初态FP32值，不以NumPy sqrt作为逐位真值。

```text
point_row = mean52((D52_FP32/s52_FP32)^2)                              FP32
self_row  = sum15(BCEWithLogits(ell15_FP32,target15_FP32,none))          FP32
L_state   = mean152(point_row)                                      # 仍是纯point
L_self15  = mean152(self_row)                                       # sum15，不是mean15
pair_row  = point_row + FP32(.1)*self_row
L_native_pair = mean152(pair_row)                                    FP32
L_mode = mean(saved mode2 BCE-elements)                              FP32
L_public = FP32(L_native_pair+L_mode) + L_imp_F64
L_total = L_public + lambda_cons*L_cons_F64             # initial/control0，method1
```

逐行加权后归约是操作顺序，不得用两个已归约scalar拼成pair。新8项分别是state/self15/native_pair/mode/impulse/consistency/public/total，state不能改名承载pair。0.1来自原Runner/config默认及冻结9e076 recipe，绝非本次调λ；不乘reward dt/window8，不做class weights/smoothing、阈值训练或重采样。796c明确重算mode-none元素并从该元素.mean构造public，避免跨stride BCE bit门；003旧mode kernel仍执行后丢弃，不宣称等walltime/FLOPs或speedup。

每role都保存并比较同一8项与common L_public/L_cons，不能用不同cons系数的训练total排序宣布胜者。保存D52、scale-square、point/self/pair row、self15 logits/target/BCE元素、mode元素和实际FP32 scalar；F64保存后算参考单列，不冒原native归约逐位复刻。

## 全量指标与取舍

每role×13组及完整456行同时输出，不造accuracy通过阈值、综合winner、CI或新的模型选择。

- **状态**：native52六块q20(rad)、qd20(rad/s)、p3(m)、LogR3(rad)、v3(m/s)、omega3(rad/s)，逐feature与块componentRMSE、vectorRMS、MAE、maxabs；各单位分开。native Log只是实际captured residual/chart，不是rawR9的SO3 geodesic真值。另存raw58 F64(pred−target)，R9仅dimensionless元素/Frobenius描述，不投影、不替换criterion。
- **self15**：原FP32 BCE元素sum15/rowmean及captured scalar；另列F64 stable BCE与Brier、逐feature/总混淆TP/FP/FN/TN、strict `logit>0`（0为negative，另记tie）、positive recall `TP/(TP+FN)`。分母0为null且保counts，不能填完美率。三role仍可比initial drift；本次label指标仅cached15资格，不拿多数负类accuracy掩盖漏报。
- **强度与Pen proxy**：predicted intensity=sum(sigmoid15)，target intensity=sum(binary15)，误差的signed mean/MAE/RMS/maxabs；原FP32概率/负和×FP32(.02)捕获，与F64 `-1/50*sum(sigmoid)`单列。proxy差正值代表相对缓存label的惩罚低估，不是完整task reward、真实自碰错误率或Actor改进。
- **mode2、z及J/K**：mode target仍是HO/nonhand四子步constraint witness，不是self15/positive Fn；BCE/Brier/confusion/tie与双bit正确数保留，不能以标签门控预测。z的dimensionless误差与decoded J(N s)/K(N m s)各source及两source合计分列，避免总和抵消掩盖分源错误。冲量不再乘dt，K仍关于object COM，在原raw P.T轴表达，不搬力矩原点。
- **cross-target**：与003相同raw I0/预测raw-R9 I1、原J/K解码、非零r_v_saved/r_w_saved；分别报告rvhat−rv_saved(m/s)、rwhat−rw_saved(rad/s)及normalized L_cons。不得以target R1作oracle、置零缺陷、伪逆/投影或把小残差当物理守恒证明。连同state、self15、mode、impulse同时解释tradeoff，不自动选一项胜出。

## 可独审证据与失败范围

先保存3×152原FW结果，再做loss；记录input x/u/dynamics bytes/digest、norm before/after原buffers和host、共同context、native67与head14状态hash关系、named storage/cache、raw Python/NumPy/TorchCPU RNG before/after、counts、protocol/source/pins/ZIP。完整权重未另保存时“不变”仍是producer digest断言，不冒独立重载证明。所有13组/456行输出原身份和数据精度；BCE clone改变stride时只用预冻machine-eps包络验证算术，不能把另一kernel的bit结果或F64重算当原FP32真值。

dtype/shape/身份/finite、原chart与raw-R9 solve可定义性是完整性门，不是误差大小门。任一required行失败保raw、mask/NaN/阶段，整体evaluation_complete=False；不降到成功子集、改门或自动重试。仅成功完成也不授native105 resume、Actor/ego、独立轨迹泛化、真实force/geometry、physical Jacobian或完整hidden-state资格。旧无self15负面结果不因新实验而撤销；新结果若存在取舍应原样报告。

## 冻结身份（本页只读metadata）

路径相对 `D:/workspace/supermimic-full151-observation-study/`；实际protocol仍须samefile+SHA与新健康深绑：

|角色/文件|SHA256|
|---|---|
|owninitial `outputs/full151105_WM_common_pre_calibration_CPU_v1/initial_component_state.pt`|`ea051225b7279fc69fbd97725eb15462faaf4b3a6706881f8fe4a05b658a14bf`|
|新pair `outputs/full151105_boundary_public_self15_matched_training_CPU_v1/paired_checkpoint_1000.json`|`08f3a341437d0290779fdcba4557fed263289e492cc965e2ff08cbe9d3e924c3`|
|该目录control `shared_public_losses_update_1000.pt`|`6cac9913bc943113b862c0b5cd92e884e904821269a5f4193a37e829324fc547`|
|该目录method `shared_public_plus_cross_target_consistency_update_1000.pt`|`78cd4584adc456dd63cd1bb7013e2f931802b85c1efb29462320f67bf38620bd`|
|新训练protocol `protocols/full151105_boundary_public_self15_matched_training_CPU_run_v1.json`|`b0a3f743aca334636f7dfa05c271a507b494c4fd73555169e311e88428ed59de`|
|新数值recipe `protocols/full151105_boundary_public_self15_matched_training_recipe_v1.json`|`9e0765fd562c3e24bdb09d52f78219204af4228bb6e8b1d8ffd67104ce094bce`|
|helper `src/full151_gradient/boundary_public_self15_training105_v1.py`|`796c360539961143fe1f588fd18eb08e4b168d141b34daee23da802a72194e7e`|
|新训练合同 `docs/full151_boundary_public_self15_matched_training_contract_v1.md`|`5a12bb913b21a45f6cfc3bd6df39a8bbe4074e0588ac47f1791f9561ec3f4ed7`|
|label-only `outputs/full151105_boundary_cached_self15_saved_CPU_v1/joined152_cached_sensor_labels.json`|`8b900a1b63d620aff3a96474b81f47f9dcd92efaaae1bb83dbbc2c364a63f698`|

原目标/label/初态资格的完整pins沿新training protocol、dataset8、cached7及initial-parity10，不把新final健康尚未给出的report pin编造出来。评价producer、Root stats与saved auditor仍须分别SOURCE review、冻结和Root实际执行授权；本页不运行或修改任何既有artifact。
