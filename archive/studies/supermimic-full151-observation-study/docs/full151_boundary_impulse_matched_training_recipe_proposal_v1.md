# 分源冲量组件：固定匹配训练与健康合同提案

**SOURCE ONLY / PROPOSED_FIXED_RECIPE_NOT_EXECUTION_AUTHORIZATION。** 本文补充[原loss草稿](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_matched_loss_recipe_draft_v1.md)，不覆盖其 `5926e869…` 原件，不修改plan/state。本轮只读取源码；没有加载权重/标签数组、构造模型、计算真实loss、FW、AD或训练。训练必须另由Root冻结run protocol和实际initial-parity资格；当前尚无新的训练产物。

## 唯一匹配对照及固定数值

两臂沿现模块名称：`shared_public_losses`（control）与 `shared_public_plus_cross_target_consistency`（method）。只采用未修改A2的共同initial WM，不选旧两个final1000，也不继承旧Adam状态。新14头从同一零模板逐字节复制，新优化器从空状态开始。以下系数作为这一次新实验的固定默认，不能用已见fresh确认、H8或task-gradient结果调参：

| 项目 | 两臂共同值 | 唯一差异 |
|---|---|---|
| native52 point loss权重 | 1 | 无 |
| 两源mode BCE权重 | 1 | 无 |
| 无量纲冲量MSE权重 | 1 | 无 |
| cross-target consistency权重 | 相同公式、均计算和记录 | control=0；method=1 |
| 优化 | AdamW，lr=1e-4，wd=0，betas=(.9,.999)，eps=1e-8，foreach=False，clip总参数梯度L2范数1 | 无 |
| 数值/批次 | CPU4，native net/head FP32，AMP=False；全部152固定顺序、fullbatch、不重采样 | 无 |
| 更新/保存 | 每臂1000次；固定100/200/…/1000，不择优、不早停到某个loss阈值 | 无 |

每个loss先按自己的全部维度取mean：`L_state=mean[152,52]`，`L_mode=BCEWithLogits mean[152,2]`，`L_imp=mean[152,2,6]`，`L_cons=mean[152,6]`。不把不同维度直接拼接后重新平均。原caller计算的 **native FP32 D52/LogR point loss** 原样传入，不复制另一套旋转loss；冻结 `effective_s52=max(initial delta_std,error_floor)`，不以新标签重估。

两臂均调用现有 `component_losses` 的全部公式，包括 `L_cons`、预测冲量和solve描述符；control仍计算 `L_public+0*L_cons`，不跳过其metric或物理求解。只有cons权重不同，不引入public-loss、head或监督量的对照混淆。每个更新均为每臂一次B152 native FW、一次同hidden的14头FW、一次backward及一次step：训练计划合计2000个native FW / 304000个预测、2000次backward/step。初始验证及保存成本另实计；相同更新数与调用数**不意味着相同wall compute**。

## 共同起点、单位和数据边界

实际训练前必须绑定新initial-parity报告、实际target dataset报告 `318366de…` 及独立8项报告 `3703c02f…` 的exact pins，并逐值核对旧152 fit的PRE58、absolute-u20、target58、xi46、bid/顺序。只使用旧152恢复组/608子步标签，不能混入300 donor或fresh152；这些均属已见缓存谱系，不是新的blind/state/trajectory/ego证据。

严格先复制hook-free raw WM，再分别构造两个wrapper；不得deepcopy已包装模块/闭包。共同起点的native WM参数与buffer、head零参数、norm raw count/mean/var、host `eps/max_count/_host_count`、error_floor、effective_s52、xi46、mass/I/g/DeltaT/palm frame必须逐字节匹配且存shared-initial。两个wrapper/cache/storage独立；每次只捕获native末层512 hidden，返回None，消费后finally清空，无第二个native FW。实际同batch初始state58和self15须与未包装raw WM bit parity，不能仅凭state_dict或shape宣称函数身份。

14头输出两个无量纲6向量及2 logits。保持 **no gating，mode仅辅助BCE**；mode2不是self15标签。使用同一冻结effective_s52的 `s_v[46:49] / s_omega[49:52]`：`Jhat=m*s_v*zJ`（N·s），`Khat=Ip0@(s_omega*zK)`（N·m·s，物体COM矩）。质量为actual FP32语义；I_body为现有F64 observer原值，raw native R9形成Ip0/预测Ip1，不投影/对称化；DeltaT为四个source FP32 dt转F64之和。物理context必须在全152行核验相同后注册，不能未经核验只取首行。

`L_cons` 比较 `(rhat_v-r_saved_v)/s_v` 与 `(rhat_omega-r_saved_omega)/s_omega`。**r_saved是非零监督observer defect，不是0物理定律、inference correction或native积分器资格。** 预测R1进入Torch F64矩阵乘和`linalg.solve`，梯度同时回到矩阵、rhs、state/head/trunk；禁止NumPy/detach预测、用target R1替代、伪逆/epsilon/投影或删行修复。标签只进入loss，forward仍只有x/u/dynamics，无未来contact/冲量oracle。参数/net梯度存FP32，物理loss算术F64；任一非finite值/梯度或solve失败整体停止并保存partial，不换模式/重抽数据。

公共loss没有native self15监督。即使初始67输出逐位一致，共享trunk更新也可能改变self15 logits；不得宣称训练后自碰准确性、原contact penalty保真或Actor使用资格，不在本次暗加self15 loss/data。

## 新artifact pair与保存语义

建议pair manifest使用 `full151105_boundary_impulse_matched_training_pair:1`；每个初始/周期payload继续采用模块已声明的 `full151105_boundary_impulse_WM_component:1`，另含pair-version/role/arm/iteration。原105/104和旧common-pre component均不能只改schema整数冒充本namespace。

payload保存 **tensor/primitive state，不pickle wrapper/module/hook**：`WM_state`、`aux_head_state`、`physical_context_state`及其reference fingerprints、完整norm host配置、effective_s52/xi、固定有序parameter-name表、raw `optimizer_state`、计数和raw RNG；同时绑定actual共同initial、dataset/source/recipe/protocol、initial-parity pins。shared-initial是iteration0/独立role，不冒final；周期checkpoint是两臂均完成第I次step后的after-I状态。用fresh文件/临时文件完成后发布，禁止覆盖旧checkpoint；实际写盘、attempted/returned与checkpoint pins记录在pair manifest。

为使gradient健康可独立读回，在每个固定保存点额外保存**第I次更新的PREoptimizer raw named gradients（clip前、clip后）**及实际pre/postclip norm；这两个梯度表来自after-(I−1)参数，checkpoint参数来自after-I，不得称checkpoint-I重新前向梯度。保存clone不增加模型FW/AD。参数表覆盖WM与14头，不能将missing gradient补零；norm/context是buffer，不加入优化器。每次更新的finite检查、forward/head/backward/step计数及全部loss分量写固定JSONL行；健康仅使用到I的完整不可变prefix。

## 一个新namespace健康reader

以[旧29项reader](D:/workspace/supermimic-full151-observation-study/scripts/audit_WM_component_checkpoint_cpu_v2.py)的 **saved-value** helper为适配依据，新增一个本namespace reader；不修改旧64c38源、不沿用其旧arm/schema/29PASS当新资格。reader只CPU读取自己的shared-initial和两个固定I checkpoint（3次受信load），map_location=cpu，无模型构造、native Log/FW/AD/MJB/Env/GPU。它应核对：

1. exact schema/role/pair/arm/iteration、Root授权protocol、实际initial/dataset/source pins（absolute+samefile+SHA），本次固定recipe和参数名/顺序；两臂只允许cons权重不同。
2. WM/head/context实际tensor名、dtype、shape、raw bytes及有限性；shared-initial WM/head跨臂身份及零头、source initial-parity来源。不能把训练后WM必须等于initial误作健康门。
3. 所有norm注册raw字段/count/mean/var、host配置、error_floor、effective_s52、xi46、mass/I/g/DeltaT/frame原值与initial逐位相同；不能只验一个汇总bool。
4. actual raw Adam states覆盖全部有序参数且无重复/缺项；每个step有限且等于I，`exp_avg/exp_avg_sq`的shape/dtype匹配对应参数、有限且二阶矩非负；param_groups数值精确符合recipe。不能以adapter调用计数代替真实Adam step。
5. 保存的clip前/后raw gradients逐名与参数shape/dtype匹配并有限，saved norm有限，实际clipflag与记录一致；明确梯度的PREoptimizer相位。仅有source/assertion而没有raw表时，不能称独立核过全部梯度。
6. raw RNG、两臂I次forward/head/backward/step和完整I行prefix；in-memory producer health与独立序列化readback分别标注，禁止producer自报bool充当独立健康。

RNG资格只按**原始状态类型/header/bytes**：Python tuple/version/整数与cached-float bits，NumPy MT19937 header/uint32[624]/position/has-gauss/cached-float bits，Torch CPU uint8向量的dtype/shape/bytes，CUDA始终未初始化。初始化constructor的显式fork scope单列；caller不能末尾restore掩盖变化。训练阶段没有随机层、重采样或随机draw，预期各保存样本与shared-initial raw state相同；比较只证明这些采样点，不证明中间从未变化。**不使用pickle重新序列化后的hash作为RNG等值门**，也不把文件SHA不同当作RNG值不同；旧序列化取证仅作这一限制的依据。

## 首个健康checkpoint、ETA与暂停

只在双方after-100 checkpoint已写盘且**新reader独立通过**后，Root保存实际UTC、pair elapsed、初始化与训练/保存/health耗时的各自scope、剩余更新900/臂以及ETA。以这100个完整pair更新的实测训练加保存耗时估计其余900个pair，不把两臂各900误算为1800个pair；列明health/finalization未测开销和规划范围、不是统计置信区间，也不借旧训练约33秒推断本次F64 backward速度。独立reader自身读盘时间另列，不偷偷混进纯训练速率。

随后按用户条件暂停Goal，不继续研究/挑checkpoint/改系数。1000是固定总计划预算；owned job是否在Goal暂停后后台继续，只能由Root此次实际run protocol显式字段决定，不能继承旧任务的默认值，Goal暂停也不自动等于停止进程。若任一checkpoint健康失败，保留失败/partial，不把另一臂通过称双方健康，不自动重试或启动新训练。本提案本身不启动/暂停任何任务。

源锚点：wrapper `003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317`；原loss草稿 `5926e869f69e1ab6a5ea7aac7d5dee4f0b9adc4b2d2686dfc6b6a6a488f785b2`；旧健康reader `64c38dbed909b98381086cbd0c2cbfad89f8f790108fd07bbe4675ec23b2d47e`（raw_rng_comparison/optimizer_readback）；旧RNG取证源 `475eb682f5d2809796564eb6dd40e7e7d0072dd0fa063a1f8039deb567755fa3`；旧FP32 native point/effective-scale来源 `71efe05da17a37df4111158860443433304f1d57762e2377d4dbafc3b57ca27f`。上述均仅源码参考；没有新的训练、accuracy、physics/HO/force-cap/whole-hidden、Actor/CHORD task benefit或ego资格。
