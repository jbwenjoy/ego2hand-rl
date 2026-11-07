# 分源冲量组件：匹配 loss recipe 草稿

**SOURCE ONLY；loss 系数尚未冻结，未打开权重或标签数组、构造模型、前向、AD或训练。** 本页收口 [a81d方法](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_WM_method_source_feasibility_v1.md) 与 [736250标签合同](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_target_dataset_source_contract_v1.md)。它是 **supervised cross-target consistency**，不是零缺陷物理定律、推断时校正器或原生积分器重放。原确认 tradeoff、physical/HO/whole-hidden/Actor资格边界均不变。

## 一、固定一个匹配对照，不增加输入 oracle

两臂分别为 `shared_public_losses` 和 `shared_public_plus_cross_target_consistency`。共同起点固定为已经验证的 **未修改A2 initial_component** 中的native WM，而非按H8或确认结果选择某个final1000。两臂逐字节克隆同一WM、输入/delta normalizer全部buffer/count/host配置、xi46和新增旁支；实际输入仍只有 `x58,u20,dynamics278=cat(x58,xi46,zero174)`。使用原训练152个恢复组，实际absolute-u、次序和四子步来源保持不变；不纳入300 donor或fresh152选权重。每个bid的PRE/POST是原H1真实native58，与旧校准实际fit输入逐位相同，不能换成fresh snapshot8重复19。

新增标签 `J/K/mode/r_saved`、真实末状态和物理残差只出现在监督loss；不能进入WM features、真实collect奖励或Actor prior。质量、actual惯量、重力和固定frame是已知物理上下文，不是未来接触信息。当前固定上下文以immutable buffers/逐行metadata绑定；不同上下文不能静默使用这套mass/I/frame。human contact与confidence仍unknown。

## 二、14维旁支、明确单位、**不门控**

原 `WorldModel.forward` 的67维网络输出及52维delta解码完全保留，仍返回 `x1_58,self15_logits`。**先deepcopy尚未注册feature hook的原始WM，再逐臂独立构造wrapper并注册hook；不得deepcopy已包装模块。** Python hook closure可能仍绑定原wrapper/self，导致跨臂capture；两臂必须各自持有WM、hook和一次性cache。14头只创建一个零初始化模板/state，再逐臂复制其参数；任何可能消耗随机数的构造仅在隔离的constructor RNG scope内完成。

在 **`wm.net[-1]` 的forward_pre_hook** 捕获其输入hidden `[N,512]`；每次native forward恰好捕获一次，hook返回`None`，不改输入、不detach、不额外调用网络。用捕获hidden计算一个 `F.linear(h,W14,b14)`：W/b全部零初始化，首12维是两源×6，末2维是mode logits。可直接创建零`nn.Parameter`避免随机初始化；若用Linear构造，则只在明确的constructor fork_rng范围内初始化并置零，caller不在末尾恢复RNG掩盖变化。cache每次调用消费/清空，hook最终移除，不序列化图或缓存。

旁支内部输出 **无量纲** `z_J[N,2,3]、z_K[N,2,3]` 和logits。源顺序固定`[HO,nonhand]`。令 `s_v=frozen_s52[46:49]`、`s_ω=frozen_s52[49:52]`，严格取共同初始验证冻结的effective `s52=max(delta_norm.std(),error_floor)`同一份值，不静默改为可能为零的raw std，也不以新标签重估；它们分别对应raw58的v `[52:55]`、ω `[55:58]`。对raw PRE R9形成 `I_p0=R0 I_body R0.T`：

```
J_hat_c = m * (s_v ⊙ z_J,c)                 # N·s
K_hat_c = I_p0 @ (s_ω ⊙ z_K,c)             # N·m·s, about object COM
z_J,target = (J_saved_c / m) / s_v
z_K,target = solve(I_p0, K_saved_c) / s_ω
```

**不乘sigmoid、不用真实mode遮罩、不设Fn阈值或硬零HO。** mode只作辅助BCE；logit0起点表示概率0.5，但它不改变冲量输出。无HO证据时冲量target为0/模式False，仅是该缓存solver边界见证；模型在别处仍可能预测非零HO，不能声称零HO Jacobian保证。self15保留自碰语义且不新增监督，训练后也不能继承原self15准确性或把它替作HO模式。

零旁支不触及原67输出，所以两臂初始state/self15函数都与同一native WM一致，aux输出也相同。未来资格检查应验证参数/旁支字节一致、hook次数和原state/self15初始bit parity；不能只看shape。旁支权重进入两臂相同的优化器，mode目标不反向修改标签。

## 三、公共loss完全相同，method仅加一个项

公共项为原native52点式loss、无类别重加权的两源mode BCE、以及标准化冲量MSE：

```
L_state = mean_N,52 ( D_native(x1_hat, x1_saved) / frozen_s52 )²
L_mode  = mean_N,2 BCEWithLogits(mode_logits, boundary_constraint_witness)
L_imp   = mean_N,2,6 ( z_hat - z_target )²
L_public = L_state + λ_mode*L_mode + λ_imp*L_imp
```

原state项保持FP32、原native旋转log/chart规则；不新增混合单位58范数。公共系数`λ_mode、λ_imp`和method系数`λ_cons`在本草稿均为 **UNFROZEN**；两臂公共值必须相同，控制臂cons系数0、method为一次前置固定的正值。Root应在新run recipe中一次固定，不能依据已见fresh/H8/梯度结果搜索、选择或调整；本页不发执行参数。

按标签合同，`DeltaT`取该组四个实际WP FP32 timestep转F64之和，不替换成F64(.02)。用预测末raw R9形成 `I_p1_hat`，不投影/正交化：

```
r_v_hat = v1_hat - v0 - (J_hat_HO + J_hat_nonhand + m*g_p*DeltaT)/m
r_w_hat = omega1_hat - solve(I_p1_hat,
                            I_p0@omega0 + K_hat_HO + K_hat_nonhand)
L_cons = mean_N,6 concat((r_v_hat-r_v_saved)/s_v,
                        (r_w_hat-r_w_saved)/s_ω)²

control = L_public
method  = L_public + λ_cons*L_cons
```

`r_saved`必须逐行使用producer保存的 **native58/F64 observer离散缺陷**；不是0，不换成另一世界系ΔL observer，也不在inference加回。这个项关联模型状态误差与其分源冲量误差，不能称独立的精确物理标签。真实target只参与loss，预测的`I_p1_hat`必须来自预测R，而非target R。

**F64物理算术保梯度。** native net/head仍FP32；把z、预测v/ω/R、已知输入和物理buffer用Torch `.double()`接入乘加和`torch.linalg.solve`，不用NumPy/SciPy、`.detach()`、`.item()`或重新构造tensor切断预测分支。梯度应经solve的矩阵和rhs回到预测R及旁支，再回同一次hidden/native WM。target/metadata为不求导buffer。求解不定义、非finite值/梯度或标签不完整时整体停止两臂并留partial，不加epsilon、pseudo-inverse、投影或删行救活。

## 四、受信artifact、健康与暂停边界

拟沿现有component预算：CPU4、net FP32/AMP false、fullbatch152无重采样、两臂lockstep1000更新、同AdamW设置/clip1、每100保存固定索引，不择优checkpoint。优化器具体数值仍须新recipe确认。一次共同初始native验证可单列1次B152 forward；训练每臂1000次forward/backward/step，旁支不另算WM forward，所有attempted/returned和validation成本分开。不能从旧约33秒直接承诺本次ETA，F64 solve/backward成本尚未测量。

新schema可用 `full151105_boundary_impulse_WM_component:1`，记录：arm/iteration、原native67 WM state、14head state、optimizer真实state、parent initial/label-dataset/旧152输入pins、规范与sourceframe/mass/I/g/scale fingerprints、norm host配置、计数及原始RNG。checkpoint不可冒充native105、原119f健康或policy-resume文件。健康必须读回实际Adam步数和所有参数/optimizer finite，核WM/head共同rawinitial、冻结norm/xi/物理buffer和数据/source不变、原始Python/NumPy/TorchCPU RNG来源与变化；constructor的fork_rng与caller禁止末尾repair分清。新参数hash和scope单独保存，旧artifact/a81d/736250不修改。

**首次两臂checkpoint100都健康读回、ETA及不确定范围保存后，按用户条件暂停Goal。** 后台是否继续到1000只由Root独立job的新protocol显式字段控制，不能把Goal暂停自动当作停进程或继续研究的授权。此刻无训练发生，不触发新的暂停。

源依据：[networks.py](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/networks.py:50)（67输出及native解码）、[common.py](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/common.py:10)（末层输入512与norm行为）、[state.py](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py:28)（D52）、[既有component源](D:/workspace/supermimic-full151-observation-study/src/full151_gradient/response_calibration105_v1.py:303)（FP32点式loss、初始scale与健康计数）。这只是一个匹配recipe的可实施性草稿，不授予真实积分器/扳手/force-cap/梯度或CHORD任务收益资格，不新增Actor更新、supportloss或诊断候选。
