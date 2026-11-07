# Public-self15：fresh 端点与 object-score u 梯度冻结合同 v1

状态：**SOURCE / JSON ONLY，科学方法已固定，实际执行尚未授权**。不读取本轮新数组或权重，不计算新指标，不运行 Torch、模型、Log、FW、AD、MJB、Env 或 physics。后续须另有 Root hash-bound 协议、最终源码审阅及少量合成证明；本页不是模型/物理资格。

**固定来源与输入设计。** 三角色为 `unwrapped_baseline`、`shared_public_losses`、`shared_public_plus_cross_target_consistency`：原 own initial `ea051225…a14bf` 加新 public15 两份固定 AFTER1000。严格沿新 family 的健康 `8566053e…b8619`、实际 final evaluation 与独审12 `071cf051…3b205` 载入；不得把旧无 self15 component namespace、PRE1000 日志或其他 checkpoint 代入。数据只取 fresh A0/A2 × **[24,74,124,148]**，每 snapshot 三 baseline 加十六方向分支，共8/152/64 pairs。其工程独审540 `15b7da4e…0ec64` 证明有限 inventory/primitive/控制/缓存标签/三 baseline 范围；**没有直接捕获、重算或资格化逐 branch PRE58**。模型输入明确是 `snapshot_x8` 按各19行重复的 common-visible-state 设计，不称真实 PRE152 或完整 matched-hidden-state 实验。旧 [0,49,99,149] 的本轮训练评价152、旧9稳定组和旧冲量目标均不能换相位或拼入。

**13个语义输入别名固定。** `native_evaluation_protocol, native_evaluation_report, native_evaluation_saved12, physical_context_buffers, fresh_config, fresh_report, fresh_arrays, fresh_branch_index, fresh_snapshot_records, fresh_engineering_saved540, prior_task_gradient_protocol, prior_task_gradient_report, prior_task_gradient_saved27`。按 samefile+SHA 绑定父报告/原路径/实际输出；initial、两final、训练协议/recipe/健康及4个fit digest从已审新 family parent map 派生，不额外加载模型。物理 context 只读原 evaluation 不变 buffer 包；**不读取或伪造 fresh J/K/mode2/r_saved truth**。fresh labels 直接使用 `fresh_arrays.branch_self15` 的全152×15，binary/形状/身份和540 source route必核；raw_found→15的独立源连接委派给540，本job不宣称再次读取 reward JSON 或独立重建该映射。

**最小执行边界。** CPU4，原 native FP32 / amp=False；3个 own trusted component loads，1次 H8 严格 initial native ctor，003 factory 在装 hook 前2次 RAW copies；endpoint 三个 B152 FW，加 center 三个 B8 FW，共 **6次 native FW、480预测、4次 final aux-head FW、3次 uVJP**。initial无headFW，其辅助零模板若保存须称代数模板。全部 raw WM14及每 wrapper16参数冻结、`.grad is None`，normalizer/host/context/cache不更新；中心在可求导范围，**仅 u8 叶子 requires_grad=True**，每角色只 `autograd.grad(score8.sum(),u8)` 一次。不得二次 EMA/C2、优化、Actor、训练、119f替换加载、MJB/几何/新state转换、GPU或新physics。forward 仅 `(x58,u20,cat(x58,xi46,zero174))`，truth/goal/label不进入 WM 输入；goal仅在前向后计分。保存实际预算、原始 RNG前后值/向量、input/norm/context字节及参数摘要，不以事后 restore 掩盖变化；摘要不冒独立原始权重证明。

**端点坐标与 κu。** 同一 B152 prediction复用，不新增FW。point为原 native `difference(pred,target)`52；common-pre 分别计算 `difference(x_design,pred)` 与 `difference(x_design,target)`，不能以 point D相减冒充 common-pre。非旋转 point符号是 target−pred；旋转用实际 native quaternion-normalization/Log坐标与已冻结 π−0.01 chart guard，不换 NumPy acos/新投影。保留native FP32的52坐标后，先 cast F64，再做 plus−minus、除固定原始 captured s52及实际 `h=actual_du_j/κu`，**κu=1 rad**：

`Lresponse = mean_all64×52( ((Cpred_plus−Cpred_minus)−(Ctarget_plus−Ctarget_minus)) / s52 / h )²`。

不在 FP32先相减，不重复除κu；所有64 actual_du_j须有限且严格正，任一无定义即整体停止并留全行/失败，不删行、不加epsilon。原 `c.common_pre_losses` 的 point flat-mean FP32与新 family `mean152(mean52(...))` 可各自捕获并明确命名，二者F64数学描述相同而FP32归约顺序可不同，不能拿几ULP差异推断权重变化。六块原单位 q/qd/p/native LogR/v/omega另列，raw58 的R9差独立描述，无重正交化、角度替换或 mixed-unit58总分。

**主 taskscore 永远只含原 object 2+8。** 调用冻结 `serial_probe105_calibration_v1.controlled_object_score_FP32`，对 POST20 state、同一个 **goal_k**：`r=2 exp(−(||p−gp||/.05)²)+8 exp(−(endpoint_error/.04)²)`；endpoint使用原±0.1m物体轴点及原两端距离平均，terms和sum保持原FP32顺序。物理主值只能是保存 GPU `branch_object_terms_k/branch_object_score_k`；RM derived约15ms、goal_k+1、CPU secondary、self15、mode、impulse/prior/full reward都不能替代或加进来。三次VJP仅该 score.sum，不额外对self15做VJP。self15 logits、sigmoid、强度及 `−.02 Σsigmoid` 仅辅助代理；其BCE/Brier/confusion/严格logit>0与零tie、逐feature/null分母按全缓存标签描述，标签约15ms并非POST20几何、HO或人类接触。

**两个分母、一个共享物理分类。** 对每 `(snapshot,j,epsilon)`，j对应 native joint [0,4,8,12]，requested epsilon [.002,.001] rad，所有FP32端点先转F64：`du=uplus−uminus`、`δr=rGPUplus−rGPUminus`。实际坐标比较为 `δr/du[j]` 对 `(g·du)/du[j]`，单位 score/rad；legacy比较为 `δr/(2epsilon)` 对 `g·d_eff`，其中 `d_eff=du/(2epsilon)` **无量纲**，不是半步。κu=1rad只定义归一化action参考，不在task投影中再乘一次。保存完整du20、两端、半步、center offset、clip和非对称；clip按原writer FP32 numeric array-equal（±0等价），不得以du≠2epsilon重判。g取baseline u0，不声称是弦线中点/有限弦线的真导数。

全部8 snapshot先通过原三baseline门：q/p/R raw9最大差≤2e−5，qd/v/omega≤2e−4，原score误差≤2e−5。`floor=max(1e−5,10*max(三baseline score range,baseline−donor最大score误差))`。旧分类用 nominal `δr/(2epsilon)`：两|δr|严格大于floor、同号、相对差 `|s0−s1|/max(|s0|,|s1|,1e−12)≤.5` 才stable，否则保持weak或unstable。物理数据完全共享、与新模型无关；保留 **32组=6 stable/25 weak/1 unstable、64行** 并按原式复核，不调floor/稳定阈值、不因模型删组。6组两个eps是重复口径，**不是N12**。所有行、两个epsilon、裁剪/弱/饱和状态均报告；没有新增准确性门、winner、CI或method效益结论。

结论上限：已解封、同源两donor、单seed的探索性局部机制检查；无独立轨迹/ego、whole hidden/Graph scratch证书、true physical Jacobian、HO/力梯度、真实reward或Actor任务收益资格。旧ea22失败、旧物理任务FAIL与旧no-self15负面结果原样保留。当前producer代码/协议尚待冻结；不锁未形成的源码hash，不提前宣称实际通过。

固定来源（项目根 `D:/workspace/supermimic-full151-observation-study`；完整父闭包按报告继续绑定）：

|路径|SHA256|
|---|---|
|`docs/full151_public_self15_action_response_gradient_source_plan_v1.md`|`f559743df3c4eabf18b744d09611aba323ef4137f4e7815135d3a1759068a413`|
|`src/full151_gradient/serial_probe105_calibration_v1.py` :304–317|`1418b7d61bea7274da51be23593807472e7edd7a1720a8bbd25c98624cff133a`|
|`src/full151_gradient/component_task_gradient105_v1.py` :71–133|`bc7d310a1b70f84cf257b973596ae5e01ad9e0cd3a1abe1e4e0a32e26b1a0921`|
|`src/full151_gradient/analysis105_v1.py` :106–145|`72b07c31792f768fe8e6d5b60b3dd2c4002a6a3580295d8c7139e71f9aff2e58`|
|`src/full151_gradient/response_calibration105_v1.py` :303–320|`71efe05da17a37df4111158860443433304f1d57762e2377d4dbafc3b57ca27f`|
|`src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py`|`81ad78161c75b1720a77aead414ccb6c5a447b0f614e5e597a940aaa1ea77419`|
|`outputs/full151105_fresh_calibration_full_saved_independent_v1/report.json`|`15b7da4ecbd9a0455108a9e10d5fbed21d3682d1eef42062bb2e8c2df1d0ec64`|
|`outputs/full151105_component_task_gradient_saved_independent_v1/report.json`|`362ea1b688f0374f3653711952b2364e2960638b4a618035d7cc882d9bc3f646`|
|`outputs/full151105_boundary_public_self15_final1000_saved_independent_v1/report.json`|`8566053e9cfdad2024575b0fe9a67a12e3f9c5f2d0d9190fbf95d1c2c25b8619`|
|`outputs/full151105_public_self15_final_prediction_saved_independent_v1/report.json`|`071cf051463565677f9702f1db6664ae6bf36a05ab4f35ff1fec3f55c603b205`|
