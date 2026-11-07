# Final boundary component：self15 源合同

**SOURCE/JSON METADATA ONLY。** 未打开pt/npz、加载模型、计算logits/指标、FW/AD或新物理。本次已冻结native evaluation job v1继续只作initial对照drift，不因本合同发现标签而改其输入、代码或评价口径。下述标签路线若使用，必须另立saved-only protocol，由Root在predictions保存后单独授权；不用新增NN/FW。

## self15是什么、如何进入Actor

[networks.py:50](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/networks.py:50) 的native WM把最后67维拆成52维delta与15个未sigmoid logits：`(x_next58, ell15)`。52维state差包含3维旋转log；raw58只有q/qd/p/R9/v/omega，[state.py:11](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py:11) 不含contact。self15也不是新增14头的两个mode logits。

[PenSimReader:235](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:235) 按sensor primary names定位每指link2、link3、link4、tip；[contact():347](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:347) 返回

```
c15 = concat(found[middle10] > 0,
             (found[distal5] > 0) OR (found[tips5] > 0)).float()
```

前10项按finger1…5各link2、link3排列，末5项各自合并link4与tip。sensor的secondary是robot palm subtree，而非object：[env_cfgs.py:340](D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/env_cfgs.py:340)。真实奖励 [finger_self_collision_penalty:169](D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/mdp/rewards.py:169) 做相同合并，RewardTerm weight=-1。它不是HO、human contact、normal-force阈值或全局碰撞几何证书。

在imagination，[Runner:734](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:734) 将 `sigmoid(ell15)` 传给 `reward_step(...,soft=True)`；reward_step绑定PenMDP.rewards（Runner:270）。[PenMDP:161](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:161) 的对应项是

```
r_self,t = -dt * sum_i sigmoid(ell_t,i),   dt=1/50
d r_self / d ell_i = -dt * sigmoid(ell_i)*(1-sigmoid(ell_i))
```

这不是几何P_ONLY的SDF项。Actor优化 [Runner:766](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:766) 的survival加权折扣奖励，加终端critic，最终 `actor_loss=-objective.mean()`（Runner:797）。WM参数在policy_update中被冻结（Runner:699），但forward不处于no_grad，contact没有detach；因此该惩罚仍可沿logits→WM输入/absolute-u→Actor反传。[full151 runtime:146](D:/workspace/supermimic-full151-observation-study/src/full151_native/runtime.py:146) 保留此调用链，phase>=150时随后把整步r清零（:148），不能把inactive的零logit/sigmoid(.5)解释为真实接触预测。

## 原训练标签与本次offline缺口

[PenSimReader.attach_capture:355](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:355) 缓存 `reward_snapshot.physics/contact`；[Runner.collect:394](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:394) 读取该capture，并把 `x1=cap.physics, contact=cap.contact` 写入replay（:447–459）。原WM fit使用 [Runner:540](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:540) 的 `sum_i BCEWithLogits(ell_i,c_i)`，加 `c.contact_weight`（config默认.1，非本次新增系数）。

本次boundary目标构造 [training_join:102](D:/workspace/supermimic-full151-observation-study/src/full151_gradient/boundary_impulse_dataset105_v1.py:102) 只返回x/u/target58/xi（:112）；最终targets在:228–236加入J/K、mode2、r_saved，**没有把self15字段加入新目标archive**。本次public loss也没有self15 BCE；[trainer_v2:186](D:/workspace/supermimic-full151-observation-study/src/full151_gradient/boundary_impulse_training105_v2.py:186) 的self15只在首个training FW与baseline比位，之后未参与loss。共享trunk更新可能改变self15，即使原67结构/initial函数一致；final健康不保证该分支准确或Actor penalty保真。

## 同一152组已有标签，未来可作native-contract accuracy

原full05 [serial_probe105_v2.py:595](D:/workspace/supermimic-full151-observation-study/src/full151_gradient/serial_probe105_v2.py:595) 保存native `contact[15]`，检查二值，并由保存raw_found重建比位（:600–603）；分支结束写 `physical_arrays.npz:branch_self15[bid]`（:744）。同一次capture的JSON也保存contact/raw_found及身份/时相（:616–632）。旧saved审计源 [audit:173](D:/workspace/supermimic-full151-observation-study/scripts/audit_full151105_full_serial_saved_cpu_v1.py:173) 已核branch数组、JSON和raw sensor映射；本轮未重跑该审计或读取NPZ。

这次仅机械核对了原pinned JSON的身份与字段长度：按现 [group152_identities.json](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_boundary_impulse_targets_CPU_v1/group152_identities.json) 的 `(path,k,branch_id)`，152/152都有唯一reward记录，mode一致、contact长度15；没有计算正负类数或accuracy。该plan严格保留A0/A2×原phase0/49/99/149×19组，不能改成donor/fresh，也不能把152当独立样本N。

未来saved-only join必须再核完整bid/group/snapshot/path/k/mode、source_frame=200+k、terminal_observation=k+1及无u150；原target到H1顺序的精确公式见 [group_plan:86](D:/workspace/supermimic-full151-observation-study/src/full151_gradient/boundary_impulse_dataset105_v1.py:86)。k149→observation150的标签保留，但没有下一次policy decision。所有152/15保留；任何缺失/重复/身份或映射不符应整体拒绝accuracy，不补0、不按模型结果删行。

**关键相位限制：** 全部所选JSON的 `label_time` 是 `lastsolver_preintegration~kdt+.015`，`state_time` 是 `postintegration_kdt+.020`。这是原native job的混合时相监督合同，不是post20 pose上重新测得的self-collision几何truth。只可评价这个保存sensor target的预测准确度，不能据此声称post20全局自碰、HO、force/human或物理梯度准确。

原缓存文件/metadata pins：

- [reward_snapshots.jsonl](D:/workspace/supermimic-full151-observation-study/outputs/full151105_gradient_empirical_gradient05_v2/reward_snapshots.jsonl)：`f214ff1b036595d56117363bd87c7b3c876b04af2fb6a1e8f3a2244ad2ee2247`，来自本次dataset协议的 `reward_snapshots` pin。
- [group152_identities.json](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_boundary_impulse_targets_CPU_v1/group152_identities.json)：`c7cf95231922e8b831a532af7bb58b50a4ac48d37297128bf78c345b6b1a8122`。
- A0/A2各自的 [sensor_label_mapping.json](D:/workspace/supermimic-full151-observation-study/outputs/full151105_gradient_empirical_gradient05_v2/donor_A0/sensor_label_mapping.json) 均为 `e4c534d5868f941b03dccc1f19fdfbf17848b9766e5d91e18bfb331d66bbed2a`。raw_found实存[1,25]；middle=`[1,2,6,7,11,12,16,17,21,22]`，distal=`[3,8,13,18,23]`，tips=`[4,9,14,19,24]`。primary的link1不进入15标签。

因此本链**具有标签支持的后继saved评价路线**，不必永久停在initial-relative drift；但当前native job v1仍只有drift资格。Root另行释放后，只用已保存initial/final logits和此152标签做新的saved-only评价，mode2与self15分开；本合同不读取predictions、不预选指标/分类阈值、不授权新增NN、训练或物理。
