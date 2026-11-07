# Native52+self15-only 单臂训练冻结合同 v1

**FROZEN SOURCE / JSON CONTRACT，非训练授权。** 本页不读新pt/NPZ、不运行Torch/NN/Toy或新训练；Root须另冻实际协议、recipe、源审、合成证明和新saved-health reader。依提案 `2b7b885b…c938c`，新增外部arm `native52_self15_only`，与旧public+mode/J/K及+consistency两臂区别明确；不改旧796c/9e源或结果，不选checkpoint/调λ/宣布任务或ego收益。

**一次统一ABI。** 执行protocol schema=`full151105_native_public_only_training_CPU_protocol:1`，role=`native52_self15_only_component_training`；recipe schema=`full151105_native_public_only_training_recipe:1`。own `protocol.inputs` **严格只有3别名** `{public15_eval_protocol,public15_eval_report,public15_eval_saved12}`。先调用冻结81ad `ev.metadata_preflight`、验证实际新family报告/正向独审12与相互pins，从其 `fm.training['inputs']` 派生原16个父路线，另保存为 **`parent_training_inputs`**；新artifact的 `input_pins`仍是own3，不能混成16或把protocol.inputs错当payload.input_pins。原152/ea051/8b9 label-only JSON/e427 baseline/context/四fit digest均从这个父闭包取得并按samefile+SHA绑定，不使用评价预测当训练target。

外部 `arm='native52_self15_only'`；003内部wrapper arm仍 `shared_public_losses`，独立字段 `internal_wrapper_arm`保留实情。原own initial实际加载1次，严格H8 factory1次native ctor、003 factory2次RAW copies后装hook；仅选中的wrapper有optimizer/forward，另一个wrapper **0 FW/0 head/0 backward/0 optimizer**，原state/cache保持。每有效更新恰1次B152 native FW、1次feature hook、1次head FW、1次native difference、1次backward及1次AdamW step；1000更新=152000预测，无额外initial validation FW。首training FW兼e427 native58/self15 bit parity和原zerohead证据。保两个copy和unused wrapper事实，不伪报只造一个或两臂训练。

**完整recipe字典（header另放schema/arm/来源pins，不增隐含数值）。** 它与9e通用字段完全相同；删除 `lambda_mode/lambda_imp/lambda_cons_by_arm/compute_consistency_both_arms`，新增末6项。`background_continue_to_final=True`是执行protocol显式字段，不塞入recipe。

```json
{
  "iterations": 1000,
  "checkpoint_iterations": [100, 1000],
  "fullbatch_rows": 152,
  "cpu_threads": 4,
  "device": "cpu",
  "dtype": "float32",
  "amp": false,
  "optimizer": "AdamW",
  "lr": 0.0001,
  "weight_decay": 0.0,
  "betas": [0.9, 0.999],
  "optimizer_eps": 1e-8,
  "optimizer_foreach": false,
  "clip_norm": 1.0,
  "lambda_state": 1.0,
  "native_log_chart_margin_rad": 0.01,
  "normalizers_frozen": true,
  "physical_context_frozen": true,
  "fixed_fullbatch_order": true,
  "initial_parity_from_first_training_forward": true,
  "save_pre_post_clip_named_gradients": true,
  "log_every_iteration": true,
  "resume_supported": false,
  "lambda_self15": 0.1,
  "self15_reduction": "sum_features_then_mean_rows",
  "native_pair_reduction": "mean_rows_of_FP32_state_row_plus_weighted_self15_sum_row",
  "window_length": 1,
  "all_rows_valid": true,
  "native15_label_only_JSON": true,
  "compute_auxiliary_losses": false,
  "optimized_loss": "L_native_pair_FP32",
  "named_parameter_count": 16,
  "active_gradient_parameter_count": 14,
  "inactive_head_gradient_None": true,
  "optimizer_state_count": 14
}
```

**只有一个优化标量。** 沿796c前半段原FP32次序：`D=difference(pred,target)`，`squared=(D/s52).square()`，`point_row=squared.mean(-1)`，`state=point_row.mean()`；self15一次BCE-none、`selfsum=BCE_elements.sum(-1)`、`self15=selfsum.mean()`；`pair_row=point_row+FP32(.1)*selfsum`、`native_pair=pair_row.mean()`。仅 `L_native_pair_FP32.backward()`。native非旋转D为target−pred，旋转是实际native Log/原π−.01 chart；s52取实际初态captured bytes，不以NumPy sqrt重建。绝不调用旧796c完整loss函数或003 component_losses，不计算mode/J/K/cons losses、不写`0*aux`、不产生`public/total`伪标量。旧五个boundary tensors只可用于原fit/combined来源摘要和context资格，不入forward、不入loss。cached15仍原152逐行label-only JSON及约15ms缓存，state target是POST20；无HO/人类/预测状态几何标签外推。

日志及PRE-I `loss_snapshot.scalars`严格 **3项 `{state,self15,native_pair}`**。raw operands严格9键：`native_D52_FP32,effective_s52_FP32,point_squared52_FP32,point_per_row_FP32,self15_logits_FP32,self15_target_FP32,self15_BCE_elements_FP32,self15_sum_per_row_FP32,native_pair_per_row_FP32`；不含modeBCE。逐row native_pair不是分开的state+.1*self15两个scalar重构；FP32捕获与F64算术校验/机器eps包络分列，不跨stride重跑BCE kernel冒bit真值。

**16个命名Parameter，实际14个更新。** 沿原WM14再aux_head.weight/bias的同序optimizer param-group；16个requires_grad均True。每步 `zero_grad(set_to_none=True)`；WM14梯度存在、FP32有限（不要求非零），head2严格None，不人工填zero。clip_norm1只由实际存在的WM14梯度贡献，但保存ordered16名称，`pre_clip/post_clip`两个dict各含WM14张量和2个head None。head参数全程原positive-zero bytes不变。Adam state只允许14个active参数ID、实际step=I、exp_avg/exp_avg_sq等有限；head ID无state，不伪step0/1000，optimizer param-group仍16。当前796c要求16梯度/16Adamstate的旧守门和reader不能直接调用或放宽；必须新namespace显式分区。normalizers/host/context、输入/labels、unused第二wrapper、原始Python/NumPy/TorchCPU RNG值/向量保持，不能末尾restore掩盖改变。14/16有效容量及全局clip所接梯度不同是方法范围，不是bit-equal收益或同FLOPs/wall承诺。

**单臂序列化合同。** 新artifact=`full151105_native_public_only_WM_component:1`，initial=`full151105_native_public_only_initial:1`，manifest=`full151105_native_public_only_checkpoint_manifest:1`，均不带旧pair_schema/paired-checkpoint伪身份。common必需字段为protocol/recipe/source_pins、own `input_pins`3、`parent_training_inputs`16及父training协议pin、arm/internal_wrapper_arm和原4 fit digest。初态保存source_parent=ea051 pin、WM_state/zero aux_head_state/physical_context_state/normhost、ordered16、active14/head2分区、raw `RNG_this_run`、construction receipt及计数offset。

checkpoint100/1000另外保存 `iteration`、`checkpoint_tag='AFTER_OPTIMIZER_I'`、完整WM/head/context/normhost、optimizer_state、actual counts、raw RNG、追加log-prefix精确bytes/SHA、`gradient_snapshot`与`loss_snapshot`。两snapshot明确 `PRE_OPTIMIZER_I, weights_after_updates=I−1`；checkpoint权重是AFTER-I，PRE1000日志不是AFTER1000评价。manifest仅 `single_checkpoint` pin，并绑定initial_state/log_prefix/protocol/recipe、当次producer-health pin及实际save成本；无two-arm checkpoints map。producer内存健康必须写 `serialized_readback_performed=False`，不能代替独立读回。

新saved reader schema=`full151105_native_public_only_checkpoint_saved_review:1`，成功status=`PASS_SAVED_NATIVE52_SELF15_ONLY_COMPONENT_HEALTH_NOT_ACCURACY`。独立读取3份own artifacts（ea051、新initial、固定I checkpoint）核实际16/14分区、rawAdam steps、未使用head/no-state/zero bytes、norm/xi/context/fit、9ops+3scalars、PRE/AFTER时相、raw RNG与log prefix。`final_iteration_scope`仅实际I=1000且终态满足预算才True；100前缀可健康通过但必须False，不承认native105/Actor/resume或模型准确性。

**执行与暂停边界。** Root另签CPU-only协议，`background_continue_to_final is True`沿已有用户授权。单臂100 checkpoint独立saved-health通过并记录ETA后，Root按用户条件暂停Goal、允许已授权同job后台到固定1000，不擅停进程或另启训练。ETA基于已完成100个**单臂updates**的实际时间/save成本，剩余900 updates及finalization；不得复用“900 pairs/两臂”口径或当CI，若已终态则剩余0。新协议/source/default/证明未完成前不执行。本合同不新增accuracy门或比较胜者；将来固定AFTER1000用同152、13重叠cohort及相同公共指标评价，原任务/ego/独立轨迹/物理梯度缺口原样保留。

固定已读来源：提案 `docs/public_native_only_matched_baseline_source_proposal_v1.md` SHA `2b7b885bbaee4665d490a83f80a28f6c782cb3d602634f2a05b665a20a7c938c`；796c trainer SHA `796c360539961143fe1f588fd18eb08e4b168d141b34daee23da802a72194e7e`；9e recipe SHA `9e0765fd562c3e24bdb09d52f78219204af4228bb6e8b1d8ffd67104ce094bce`；81ad metadata consumer SHA `81ad78161c75b1720a77aead414ccb6c5a447b0f614e5e597a940aaa1ea77419`；003 wrapper SHA `003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317`。所有新recipe/payload/执行hash待Root实际冻结，不在本页捏造。
