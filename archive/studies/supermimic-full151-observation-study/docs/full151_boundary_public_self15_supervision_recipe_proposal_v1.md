# 两臂共同恢复native self15监督：固定配方提案

**SOURCE ONLY / PROPOSAL_NOT_NEW_TRAINING_AUTHORIZATION。** 本轮只重新读取native源码并写本文，没有打开实际pt/npz或训练数据、加载模型、计算真实loss、FW/AD或训练。现有recipe `ed5e6cc3…`、wrapper003、trainer2dd2、旧checkpoint/health及已完成结果全部保持原样。新实验必须另有完整冻结protocol、新代码/新artifact namespace；本提案不启动新job或暂停当前Goal。

## 一个改动，两臂同样加入

新对照保留两臂 `shared_public_losses` / `shared_public_plus_cross_target_consistency`；共同public loss新增 **原native self15标签监督**。method唯一差异仍是cross-target consistency系数0/1；不是让control无self15、method有self15。采用native源码默认 `contact_weight=0.1`，不按此次BCE/Brier/混淆或proxy结果搜索系数、class weights或threshold。

已完成的152训练只监督state52、mode2、impulse及r_saved，未监督原15 logits。共享trunk更新会改变self15，initial native67函数身份或after1000工程健康不能保证contact penalty保真。已有保存值报告显示BCE、Brier、严格>0混淆和概率和proxy并不朝同一方向变化；这使补回原任务标签成为明确的公共合同修正，而非对某个胜出模型追加有利指标。本实验由已见同一缓存结果提出，是探索性实验，不能称新的blind确认。

## 系数与reduction的确切来源

[native config.py:41–42](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/config.py:41) 默认 `final_state_weight=1.0, contact_weight=0.1`。本轮实际重读源码SHA：`11889b65b5da121ea613d402b3ef9e7bc854d57462f96830af9b644a0b031323`。

[Runner.fit_world_model:535–544](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:535) 使用

```
s[n,t] = mean_52((D_native(pred58,target58)/effective_s52)**2)
c[n,t] = sum_15(BCEWithLogits(logits15,target15,reduction='none'))
loss[n,t] = 1*s[n,t] + 0.1*c[n,t]
```

**实际优化loss**在 [Runner:584–588](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:584) 是 `sum(mask*loss)/batch_size`；不是除以valid-mask总数。`state_component/contact_component`的 `/mask.sum()` 是日志分量。源码SHA：`b19d9ac80e23e4f55b2eb64e458c05516fc5dd4ffb13e1a3050ce9fea2669d46`。

本提案保持 **W1、全部152行有效**，所以该部分对应 `mean_152(s[n]+0.1*c[n])`。不把原H8窗口的总尺度搬来乘8，也不把`.1*sum15`误换成`.1*mean15`（会再缩小15倍）。0.1是WM训练loss权重，不是环境finger_collision奖励的−1，也不是几何P_ONLY的0.1；BCE不乘reward dt、不除实际四子步DeltaT。

## 固定的新公共项与精度顺序

同一次B152 native forward返回pred58与原self15 logits，沿原native FP32 `difference`/LogR及冻结effective_s52计算每行s。新15标签只进loss，forward输入仍是x58/u20/dynamics278，不是接触oracle；不得用mode2标签、HO/非手boundary mode、human标签或模型预测概率替代native15。

```
L_state_FP32        = mean_152(s)
L_self15_sum_FP32   = mean_152(c)                  # sum15后mean rows
L_native_pair_FP32 = mean_152(s + 0.1*c)           # 逐行FP32加权后归约
L_public           = (L_native_pair_FP32 + L_mode_FP32) + L_imp_F64
L_total_control    = L_public + 0*L_cons_F64
L_total_method     = L_public + 1*L_cons_F64
```

state/self15 pair及再加mode的subtotal保持FP32，然后提升接入F64 impulse；consistency仍为现有可微F64 raw-R9/inertia solve与非零r_saved目标，两臂均计算/记录，只有系数不同。分别保存上面各loss字段，**不能把combined loss偷塞旧`L_state`/`native_point_loss`字段**，也不能声称F64后处理BCE与原FP32 kernel逐位相同。不要用两个已归约scalar相加冒充原逐行加权归约的bit parity。

同样保留mode BCE权重1、impulse MSE权重1和cons0/1；不增加pos_weight、label smoothing、temperature、阈值训练、mode gating、额外regularizer或正样本重采样。new helper/loss版本必须明确接收native15 logits/target15，原003的严格label字段集合不允许直接塞入新字段后冒同一契约。新15监督使用现有67输出中的15项，不增加网络头或Parameter数量；仍是native WM14个加aux14头的2个，共16个Parameter tensor。

## 标签、起点、预算完全匹配

只加入[同一原152组的cached15标签](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_final_self15_evaluation_source_contract_v1.md)。优先使用label-only `joined152_cached_sensor_labels.json`及其原reward JSON/raw_found25→middle10+distalORtip5来源，不把同时含模型概率的metric archive当监督输入。未来protocol应绑定已完成cached-label producer及独立7项报告（`caa1474c84f297f1080432427a5be6665cf21f47a1d680dd8e442a5c30d3d18f`），并再核row_id/path/k/branch_id/mode/group/order、全部152×15 binary FP32和原映射。当前仅引用已有资格，不在本文重读标签或评价。

标签是 **lastsolver≈kdt+15ms缓存finger sensor**，而state target是post20；保留native原混合时相监督合同。它不是真实post20自碰几何、HO、human、力cap或全hidden状态标签。全部phase0/49/99/149及k149→observation150都保留，无u150；不纳入donor/fresh，不平均三baseline、不补0、不按class或模型结果删行。缺失/重复/phase不符应整体阻断两臂，不能退回drift-only来宣称监督有效。

从**同一未修改A2 initial component**重新建立两臂，不从已训练的两个final1000继续，以免带入不同历史。WM/原15头/norm/hostcfg/error_floor/effective52/xi/context/mass/I/frame逐字节相同；aux14仍由同一零模板独立复制。先raw deepcopy再各自挂hook、独立storage/cache，无额外native FW。只扩展监督label digest，保存原impulse-label digest与新combined-label digest，且两臂一致；fit_input_digest仍对同一152 x/u/dynamics/target58绑定。

数值预算保持CPU4、net FP32/AMP false、fullbatch152固定顺序、1000 lockstep updates/臂、AdamW lr1e-4/wd0/betas(.9,.999)/eps1e-8/foreachFalse、clip1/error_if_nonfinite、仅checkpoint100与1000。原normalizer与物理context不更新，不按label先验重置原15 bias；新Adam状态从空开始。每臂每次仅一次native B152 FW与同hidden aux FW、一次backward/step；self15 BCE只是同次输出上的张量loss，不加NN call。非finite/solve或标签错误留partial整体停，不自动重试、恢复、改权重或扫参。

## 新健康/评价边界与用户暂停条件

推荐新namespace `full151105_boundary_public_self15_WM_component:1`、pair `full151105_boundary_public_self15_matched_pair:1`、shared initial `full151105_boundary_public_self15_matched_initial:1`，并另版trainer/reader；不能给旧checkpoint加一个字段就冒新监督资格，也不改旧5项健康报告。新payload绑定新增15标签/映射pins与digest、全部loss语义、相同rawinitial及两臂public系数、raw16参数/Adam/norm/context/RNG。仍保存PRE-I clip前/后raw梯度与AFTER-I参数/Adam，THIS-run RNG按原值/bytes而非pickle重序列化hash；追加日志prefix按I行/字节/SHA固定，不要求workerterminal才能审100。

双方新的checkpoint100必须经新reader独立健康读回，并保存这次实际elapsed/剩余900pairs ETA及范围后，Root按用户条件暂停Goal。旧checkpoint100/1000通过不代替本次条件；背景是否继续须新job显式protocol授权。本提案没有触发训练或暂停。

未来只评价固定final1000及同一initial、全152/13个既定重叠cohort：同时保留state/impulse/consistency取舍、self15 BCE/Brier、严格logit>0混淆与tie，以及概率和Pen proxy偏差。不能单看accuracy（多数负类/漏报可掩盖）或Brier（有界平方与BCE的错误自信代价不同）；proxy低估惩罚也可能改变想象目标，即使某一分类指标改善。它们都不是实际Actor回报、CHORD收益或物理可行参考的替代品；本次只提出这一个共同监督修正，不自动启动native task或额外诊断/训练。
