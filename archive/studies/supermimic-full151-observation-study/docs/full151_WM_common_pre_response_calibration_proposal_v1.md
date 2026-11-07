# 单一候选：共同 pre 坐标的有限响应校准

2026-11-06。**PROPOSAL ONLY；未实现、未授权训练或新物理。** 当前原参考几何及完整三臂任务均 FAIL；即使下述部件校准成功，也不能替代物理可行参考、策略收益或真实 ego 泛化。H8 保持 PARKED。

**可行的最小实现。** 从同一 A2 final2000 WM（checkpoint SHA `70d0a21d5b2f0238ab745264bf24db08515fef65f072cbe00fb6893a33512a0c`）构造两个完全相同的可训练副本，只改 WM；固定同一 `xi46`、input/delta normalizer 的全部 buffer/count、物理上下文、FP32 CPU 精度和数据顺序。两臂均只使用旧 **152 restored transitions / 64 epsilon pairs**，排除已观察的 300 donor transitions；不实例化策略、不接 Env/replay/奖励/先验、不做 H8。使用完整 152 batch、1000 次更新、AdamW `lr=1e-4, weight_decay=0, betas=(.9,.999), eps=1e-8`，沿 native 梯度范数 clip1；每次两臂用同一完整 batch、同样的前向/更新次数，不重采样、不搜参或选最佳 checkpoint。

**唯一变化是有限响应项。** 令 `D(a,b)=native state.difference(a,b)`，它返回 52 维：q/qd/p/v/omega 的目标减输入，以及 `Log(R_a^T R_b)`。冻结 `s52=max(delta_norm.std(), error_floor)`。点式项直接沿用 native state-loss：

```
Lpoint = mean152,52 [ D(Fθ(x,u), y) / s52 ]²
Tx(z) = D(shared_pre_x, z)
h = (actual_u_plus[j] - actual_u_minus[j]) / (1 rad)
Lpair = mean64,52 [ ((Tx(Fplus)-Tx(Fminus))
                  -(Tx(yplus)-Tx(yminus))) / s52 / h ]²
control: Lpoint                  method: Lpoint + 1 * Lpair
```

`h` 是实际执行的 FP32 端点先转 F64 后相减所得的有量纲控制差，再以 **1 rad** 为参考归一化；不是 requested epsilon、无量纲 legacy `d_eff` 或重新 C2。保留全部 clipped/asymmetric pairs。相同 snapshot 的 `shared_pre_x` 须通过原 visible58/control guards，均值不是新物理状态。不能直接相减分别以 model-minus 和 physical-minus 为起点的旋转 log：它们不在同一 chart。共同 pre 的 52 项是**新校准坐标**，不替换已发表的 raw R9 描述，也不保证非线性旋转中的常量偏差严格抵消。

正式执行前须冻结 chart guard：建议固定距 π 的 margin `δ_chart=.01 rad`，检查所有实际参与 native point-loss / common-pre log 的相对旋转 raw trace、finite 输出和 finite 梯度，保存原 clamp 触发描述；不正交化、投影、换 chart、补零或按结果删行。零实际分母、chart/finite/共同状态失败应整体停止并留失败证据，不以更换方向或降低门槛修复。它不是原物理 8eps 门的替代。这里只复用 `fit_world_model` 的 **H1 state-loss 数学**，不称完整 native H8/contact/failure/AMP 训练重放；self15 不添加 HO/力标签或本轮监督。

**确认数据及泄漏。** 先冻结两臂代码、超参、更新数与评价式，再在新 worker fork 中固定测试相位 `[24,74,124,148]`，保留原两 donor、8 snapshot 形状、每点3 baseline、同4 joint/2 epsilon/±及4 substep；共 **24 baseline +128扰动=152 branch /1808 episode substeps**。原 worker 的 `_review_header/validate_review_manifest`（127–184行）绑定 exact worker、8 labels 和各相位 description/topology pins；因此必须先新 **discovery-all1200 steps → 各相位 inventory source review → serialfull1808**，合计两次约 **3008 physical substeps**。可复用已审 opaque 规则做新名单审阅，不能只替换标签继承旧 graph/closure/long-clock 资格。三 baseline/外置 capacity/source/完整 visible state gate 不弱化，隐藏 scratch 仍未知。新增的是 **fresh held-out perturbation responses at preregistered phases**：两条旧150步 donor 的这些相位已在 H1 中被观察，初始 A2 也可能训练于同分布，不能称未见状态、独立轨迹、blind test 或 ego 泛化。新152 raw 及标签由独立过程封存，只做必要工程核验；两臂 final1000 权重冻结后才解封全部64新 ± 响应，禁止以接触类别、旧stable9、当前误差或质量筛选。

**预期判定与最少工程。** 在解封前固定以全部64新 pair 的上述归一化 response error 比较，同时报告全部新152的原 pointwise state-loss、逐块原单位误差和所有失败/undefined；不合成 raw mixed-unit58范数或 CI。建议把“response error 严格下降且 pointwise 不增加”作为这个局部校准候选的确认条件；若只有一项改善，报告 tradeoff。重复/两 epsilon/同谱系不扩大独立 N，不推导真实 Jacobian、HO 力或历史三臂训练因果。新校准只需一个独立 WM-only adapter，复用 native 网络/数学和已验证数组，不建设128world隐藏状态序列化框架。**不得 unfreeze/改写119f的 frozen wrapper 或把改过WM的产物冒作旧105健康通过**：使用新隔离 artifact schema（建议 `full151105_WM_common_pre_component_calibration:1`），绑定 parent105、两臂参数/optimizer/norm/data fingerprints、更新数及源码；旧 checkpoint/health 保持原样。

**首个检查点暂停条件。** 两臂 lockstep，到 checkpoint100 各100次 WM 更新，核共同初始参数/norm/data、实际 optimizer counts、参数/梯度/optimizer finite、normalizer/xi未变、无 Actor/Env/物理/GPU；用实际前100耗时给余900次及保存开销的 ETA，明确启发式范围。健康与 ETA 保存后按用户要求暂停 Goal；后台作业是否继续由 Root 既定独立 job 控制，不靠本建议自动启动。1000结果、fresh 测试与部件收益均 **PENDING**。

源核验（只读，0模型/NPZ转换/前向/AD/训练/物理）：[runner.py](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:511) 的 loss/std 与 optimize，SHA `b19d9ac80e23e4f55b2eb64e458c05516fc5dd4ffb13e1a3050ce9fea2669d46`；[state.py](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py:28) 的 chart，SHA `4a95cec1951d4e54fac9256f757ca0d9a1f3cd81d66e9b97564954e3f55add06`；[networks.py](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/networks.py:50)，SHA `7fbec4de7aa92603565aa24e85d559f4b8c9dc309aae877d3fdd2981402e350a`；[strict119f loader](D:/workspace/supermimic-full151-observation-study/src/full151_native/world_model_checkpoint.py)，SHA `119fa7cebc0e9b0727b0160660b1a47822deee2a0e0e9fa5be69465ac5578052`。原 response22 报告 SHA `b70a9b4e63b13401a86a282046f19a71fb2c5019c185a851e91a1ce6b86de481`；原参考/任务 FAIL、原5/9、precision/Graph限制均保留。以上 recipe 仍待 Root 正式冻结与单独授权，本文不是 run protocol。
