# Public-self15 components：保存响应与任务 u 梯度的下一步源计划

状态：**SOURCE/METADATA ONLY，非执行授权**。本页只读源码和既有 JSON 元数据并记录源 SHA；未读取 pt/npz、拼接实际数值、加载模型、调用 Log/FW/AD/MJB/FK/physics。三个固定角色为 `unwrapped_baseline`、`shared_public_losses`、`shared_public_plus_cross_target_consistency`；两 final 均固定 AFTER1000，新 family 健康为 `8566053e…b8619`（5/5），不是 accuracy 或 Actor 资格。

## 两套 152 必须分开

**本次 native evaluation 是旧相位 A0/A2 × [0,49,99,149]**，每 snapshot 为三 baseline 加十六扰动。新 producer `boundary_public_self15_final_evaluation105_v1.py:144–177` 保存真实旧 152 输入 x58、绝对 u20、物理 POST20 target58、group/substep IDs 及三角色预测；不能把其预测改贴为新相位。

**既有任务梯度的 6 stable / 25 weak / 1 unstable 属另一套 A0/A2 × [24,74,124,148]**，见 `component_task_gradient105_v1.py:13,71–109`。它的 32 个方向组和每组两 epsilon 共 64 行全保留。`component_confirmation105_v1.py:116–165` 明确将该套 `snapshot_x8` 重复 19 次作为输入设计，**不是直接观测的 PRE152**。fresh 工程审计 `15b7da4e…ec64` 仅资格化恢复 inventory/INTEGRATION 和 raw qpos/qvel；没有补出或认证缺失的 branch PRE58。

## 当前保存预测可做什么：新增模型预算为零

在本次三角色预测及其独立 saved review 冻结后，可机械 join 全旧 152/128 扰动/64 对，计算各 snapshot 三 baseline 的预测、target 坐标均值、常量偏差、中心化响应和响应误差；均值仅是 F64 统计参考，不能称新物理状态或投影 raw R9。逐块保留 q、qd、p、R9、v、omega 的原单位，不造混单位总分。实际 x−/x+、u−/u+、target−/target+、FP32 编码零响应及所有 19 个非 j 控制差都保存；不以小响应删行。先核旧数据全部 19 行对原 snapshot 的可见状态守门（q/qd bitexact，其余 18 numeric exact），失败则阻止 fixed-visible-state 解释，不能替换输入或修复。此守门不证明 whole hidden/graph-scratch 相同。

原始 raw58 的有限响应可直接做 NumPy/F64 后处理，**0 load/0 WM FW/0 AD/0 physics**。原生 common-pre 52 响应同样不需要新 WM FW，但需要另行授权的原生 FP32 `state.difference(common_pre, endpoint)` 算术及 chart guard；本次只捕获了 `D(pred,target)`，未捕获 common-pre52。旋转项是 `Log(Rpre.T @ Rend)`，不能从已保存 `D(pred,target)` 相减冒充，也不能把 NumPy 新 Log 当 native bit 真值。这里的 common-pre 是共同坐标图；把模型输入重新设为共同 x0 是另一项输入实验。

## 推荐下一项固定诊断及最小预算

建议在旧 64 对零 FW 描述之后，单独冻结**三模型 fresh 全 152 端点预测 + fresh 八中心任务 uVJP**，共 **3 trusted component reads、1 native constructor + 2 RAW copies、3×B152 FW + 3×B8 FW = 480 个 native next-state 预测、3 次 uVJP**；若沿现 wrapper 路径，两个 final 的 aux14 各执行两批，共 4 次 head FW，透明计数。新加载必须复用本次新 family `load_final_into` 的 schema、source/input map、fit digests、norm/xi/context 检查；旧 confirmation constructor 接受的是旧 component namespace，不能直接冒认新 artifact。RAW copies 必须在装 hook 前完成，不 deepcopy wrapped module。全部参数冻结、CPU4/FP32/amp=False、no normalizer update、outside inference_mode；VJP 只对 u，不产生参数梯度。保存 raw RNG 前后值并校验，不在末尾 restore 掩盖变化。模型只从 x/u/xi46/zero174 前向，任何 truth/label 不进入输入。

fresh B152 的 x 是上述 **common-visible-state 设计**。保留全部实际物理端点与实际 u±，并保留已有 raw PRE 证据；没有逐 branch PRE58 资格前，不把 x± 强制相等的模型输入说成完整 matched-input 物理微分。可同时报告 native52 point/common-pre response（沿原 frozen effective s52、原 `control_reference_rad=1.0` 和 chart guard；requested epsilon 另为 .002/.001 rad）与六块 raw58 误差，以及 cached self15。fresh `branch_self15` 和 reward JSON `contact/raw_found` 的来源见新 worker `:609–632,760`：须按 path/k/branch_id/顺序全 152 join 并独立重建 found25→15。这是最后 solver 约 15ms 的缓存 sensor 标签，**不是 POST20 几何标签**；BCE/Brier/混淆和惩罚强度都不能代替 HO dynamics。

**fresh 没有当前已独立资格化的 J/K/r_saved 目标包。** worker 的数组 schema `:245–253` 及分支赋值 `:760–764` 保存 x/self15/u/score；旧 `318366…` 的 152/608 冲量目标只对应旧相位。即使 fresh raw contacts 可供未来新标签工程，也不能现在把旧 J/K/mode2/r_saved 换 phase、补零或拿 aux14 当 truth。因此本诊断不评价 fresh impulse/consistency loss，不把 mode2 当 self15。

## 任务标量与两种有限差分口径

复用 `serial_probe105_calibration_v1.py:304–318` 的同 POST20、同 goal_k 函数：位置项 `2 exp(-(‖p-gp‖/.05)^2)` 加轴端点项 `8 exp(-(endpoint_error/.04)^2)`。取既有物理 `branch_object_terms_k/branch_object_score_k`、goal_p/R、三 baseline 和 donor；native RewardManager 约 15ms 项、goal_k+1 和 CPU secondary 均不能替代主物理标量。新三模型各在八个 baseline center 只前向一次并对 `score.sum()` 做一次 uVJP。

每个 (snapshot,j,epsilon) 保存 `delta_u=uplus−uminus`（rad）、`actual_den=delta_u[j]`、半步、center offset、原 clip、非对称弦线和全部 20 维实际差。对照分别为 `delta_score/actual_den` vs `g·delta_u/actual_den`，以及 legacy `delta_score/(2 requested_eps)` vs `g·d_eff`，其中 **d_eff=delta_u/(2 requested_eps) 是无量纲**，不偷换成半步。沿旧任务 source，任一 actual_den 非正/非有限整体阻止该诊断，不删行或加 epsilon 救活；clip 继承 writer 的 FP32 endpoint `np.array_equal`（±0 数值等价），不以 delta!=2eps 重判。

物理分组仍完全 model-independent：`floor=max(1e-5,10*max(三 baseline score range, donor score error))`；两 epsilon 的 score 差均严格超过 floor、nominal slope 同号、相对差 `|s1−s2|/max(|s1|,|s2|,1e-12)≤.5` 才标 stable。先核全部八 baseline 原门，不按新模型结果改 floor、挑组或 pool 旧相位组。报告全 32/64，6 stable 仅作为既定子组另列，两 epsilon 不是十二个独立方向。保留 FP32 signal/noise 级和 weak/unstable，不称稳定物理 Jacobian。

## 决策边界与执行前置

这是已见缓存、同源两 donor 的探索性复用，不是新 blind、独立轨迹或 ego 证据。CPU shadow 不冒历史 GPU AMP；同可见状态与经验 baseline 不消除 hidden state/scratch 未知；raw初态相同也不抹去旧 prepolicy100 训练轨迹分歧。旧 rough reference 仍未过物理任务门。有限响应或 uVJP 改善都不能直接宣布 CHORD Actor/task benefit、全 reward、HO force/Jacobian 或可行力上界。

执行前须 Root 冻结新协议/固定三模型健康与 actual eval output pins、新 fresh 工程审计和 saved score 来源、fresh cached15 的全量 join 及失配整体停止规则。新包保存实际两套 cohort 的身份，禁止混并。以这一个固定诊断判断公共 self15 监督后**端点/接触缓存质量是否伴随任务标量局部响应改善**；若并未改善，回到接触驱动物体 dynamics 的已资格化标签/模型结构，不靠增加 Actor 更新补模型偏差。当前不写执行器、不启动该诊断。

### 本页读取的固定源码身份（完整依赖继续沿各父协议，按 samefile+SHA 去重）

| Source | SHA256 |
|---|---|
| `src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py` | `81ad78161c75b1720a77aead414ccb6c5a447b0f614e5e597a940aaa1ea77419` |
| `src/full151_gradient/component_task_gradient105_v1.py` | `bc7d310a1b70f84cf257b973596ae5e01ad9e0cd3a1abe1e4e0a32e26b1a0921` |
| `src/full151_gradient/component_confirmation105_v1.py` | `f75302fa70a894300ceb3a71492b9cffd973244990f13be26cee5b7e5e112659` |
| `src/full151_gradient/response_calibration105_v1.py` | `71efe05da17a37df4111158860443433304f1d57762e2377d4dbafc3b57ca27f` |
| `src/full151_gradient/serial_probe105_calibration_v1.py` | `1418b7d61bea7274da51be23593807472e7edd7a1720a8bbd25c98624cff133a` |
| `src/full151_gradient/analysis105_v1.py` | `72b07c31792f768fe8e6d5b60b3dd2c4002a6a3580295d8c7139e71f9aff2e58` |
| native `src/dex_supermimic/rl/dex_supermimic/state.py` | `4a95cec1951d4e54fac9256f757ca0d9a1f3cd81d66e9b97564954e3f55add06` |
| `protocols/full151105_component_task_gradient_CPU_run_v1.json` | `92e7351a2bd3226a2474f02c239d61a717e4407cc02d002f3f7f64dab67d2a6f` |

项目根为 `D:/workspace/supermimic-full151-observation-study`；native 根为 `D:/workspace/dex-supermimic-research`。本页作者 source assessment，不是独立实际数值审计。
