# 两份 final1000 的最小任务 score 梯度核验：source-only 可行性

**可实施，所需字段已齐；暂不执行或授权。** 它是当前 ea22 确认评价之后的探索性机制检查，不改变其全64响应 MSE 严格下降且全152 pointwise loss 不增加的主判据，也不构成新 blind 证据。不能仅凭状态 MSE 改善便宣布可供 CHORD Actor 使用。

**同输入与最小成本。** 固定 `pointwise` / `pointwise_plus_common_pre_response` 两份 final1000；沿已冻结 constructor 校验初始身份、norm/host配置、xi46、s52后置 eval、全部参数 requires_grad=False。每臂只 **1次 B8 center WM forward（8个预测）+1次 `autograd.grad(score8.sum(),u8)`**，共2次FW/16预测/2次uVJP；输入 `snapshot_x8, snapshot_u8`，context=`cat(x58,xi46,zero174)`。新 consumer 必须在 enable_grad 下调用，不沿 confirmation 的 inference_mode；不解冻参数、不优化、不更新norm、不创建Actor/Env。两 native WM constructor 在现有 fork_rng 构造范围内恢复初始化 RNG，caller不得在末尾 restore 掩盖变化。可信 component读取至少initial+两final三份；准确 import/读取成本须形成代码后据实记录。无119f替换格式、无MJB/FK/新物理/GPU。

**字段、相位与算术。** 15个必要 NPZ 头部均存在且形状/dtype正确，未读取数组值。`goal_p[8,3]` / `goal_R[8,3,3]` 是原 fresh worker 从其 GPU 计算结果保存的 **preaction k** wrist/palm坐标目标；`branch_x_next[152,58]` 是 post20，`branch_object_score_k` 是同一 post20 state 的 GPU FP32 `2+8` score。因此本次8点不需重建全151 goals；不使用 RewardManager 的 derived15ms score、k+1目标、HO力或几何prior。必须调用原 `serial_probe105_calibration_v1.controlled_object_score_FP32`，CPU FP32 math/原norm距离/原两项顺序和 FP32 subset sum保持不变。保留预测两项、score8、gradient8×20；原保存 GPU terms/scores与CPU-secondary terms的差另列为算术限制，不能把CPU重分数替代物理主值或事后调整floor。

**完整有限弦线比较。** 对所有 `8 snapshots ×4 joints ×2 eps=64` 行，按 `minus=s*19+3+8*ei+2*j`、`plus=minus+1` 机械取端点，保留全部32依赖方向组，不能pool旧9或按arm挑组。令 `δr=F64(rGPU_plus)-F64(rGPU_minus)`，`du=F64(uplus_FP32)-F64(uminus_FP32)`，g为中心uAD：

| 口径 | 物理有限量 | 同分母模型局部量 |
|---|---|---|
| 实际坐标弦线，score/rad | `δr / du[j]` | `(g·du)/du[j]`；已核其他19坐标不变时等于g[j] |
| 原 nominal-eps / effective direction，score/rad | `δr/(2*requested_eps)` | `g·legacy_d_eff`，其中 `legacy_d_eff=du/(2*requested_eps)` 无量纲 |

两个口径分别命名，不能交叉比较。全部actualdu、原FP32端点、中心偏移及clipped bool原样保留；0实际分母留undefined/NaN及原行，绝不加epsilon修复。裁剪后的中心不一定是弦线中点；g在baseline u0求值，非两端均值处的真导数。所有branch controls须与原plus/minus位相同，三baseline与snapshot_u位相同；不二次EMA/C2，不把absolute-u梯度当Actor action/参数/H8梯度。

**若复用旧floor，完整预先固定。** 每snapshot以3个原 GPU baseline score计算 `span=max−min`、`err=max|baseline−donor_GPUscore|`，`floor=max(1e−5,10*max(span,err))`（score单位）。每32组必须保留两个eps；原规则用nominal slopes：两 `|δr|>floor`、同号且 `|s0−s1|/max(|s0|,|s1|,1e−12)≤.5` 才标STABLE_NONFLAT，否则原 below-noise/unstable 标签。它只是该3次经验重复加固定算术floor，不是真噪声界或Jacobian资格。分类只由同一物理数据产生、两arm共享；全部32分类/64行仍输出，稳定子集符号一致率仅次要描述，并同时给全部行误差/幅度/裁剪情况。任何新统计口径必须在看梯度结果前固定，不能以改善一致率调floor。

**风险与结论上限。** snapshot_x是共享可见58输入设计，fresh没有逐branch独立PRE58；540工程/三baseline仅局部经验条件，Graph scratch/全部隐藏状态仍未知。bounded `2+8` score可能指数饱和/物理响应接近FP32分辨率，状态MSE与任务梯度可脱钩；需保留零/微弱信号与score/gradient幅度而不删组。CPU FP32影子与物理GPU算术、有限弦线与局部AD、裁剪非对称和接触状态变化均可能贡献差异，不作唯一归因。旧5/9是其他相位/原模型的独立历史描述，不迁移样本资格。即便本检查改善，也只支持该seen谱系、8点和有限方向的任务score机制，不授予真实Jacobian、HO力梯度、全隐藏恢复、CHORD任务收益或ego泛化。Root若选择此行动，须另冻结小协议和独立saved核验；本页不产生执行器或自动新任务。
