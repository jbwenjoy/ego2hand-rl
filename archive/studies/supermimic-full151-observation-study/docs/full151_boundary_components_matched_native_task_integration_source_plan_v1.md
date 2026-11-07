# Boundary components → matched native task：最小源集成路线

状态：`SOURCE_PLAN_ONLY_NOT_IMPLEMENTED_NOT_AUTHORIZED`。2026-11-06。本次只读 `.py`、既有 JSON 和说明文档；没有打开 `.pt`/NPZ、导入 NN/backend、构造模型、前向、采集或训练。没有修改冻结 runtime、105 consumer、训练器或结果。本文不是 run protocol，也不把组件健康升级为任务收益。

结论：可以把两个 own AFTER1000 的 **裸 native WM state 与其原配 norm/host 配置**安装到一个新 namespace 的任务实验。现有 format105 接口不能承担这个操作。最小可解释路线是 **一次共同采集/共同 learner 起点，之后安装不同 WM，并在任务阶段固定 WM 与 WM norm，只训练原 Actor/Critic**；两个分支不能各自重新执行 WM-only warmup。是否实际采用和预算仍由 Root 冻结。

## 1. 已确认的限制

- [controller.py:183](D:/workspace/supermimic-full151-observation-study/src/full151_native/controller.py:183) 声明 `exact_resume_supported=False`；[270–277 的入口](D:/workspace/supermimic-full151-observation-study/src/full151_native/controller.py:270) 拒绝 `load` 和 `load_model_only`。[learning_v2.py:40](D:/workspace/supermimic-full151-observation-study/src/full151_native/learning_v2.py:40) 也只接受 from-scratch。不能绕过这些 guard、替换 format_version、伪造四模块初始指纹或把 own artifact 包成105。
- [native Runner.save:1074](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:1074) 保存四模块、优化器、部分计数/curriculum 与 Python/NumPy/Torch RNG；没有保存 replay、当前 obs、完整 Manager/控制器/sensor/物理状态。[load:1112](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:1112) 明确重新填充 replay。它自身也不是精确物理续跑。
- 当前105资格入口是 [checkpoint.py:45](D:/workspace/supermimic-full151-observation-study/src/full151_native/checkpoint.py:45)、[world_model_checkpoint.py:63](D:/workspace/supermimic-full151-observation-study/src/full151_native/world_model_checkpoint.py:63) 与 [health auditor](D:/workspace/supermimic-full151-observation-study/scripts/audit_full151_checkpoint_cpu_v2.py:364)。其 namespace/from-scratch/方法和原始运行绑定不能原封不动用于新安装实验。本轮检索没有发现单独名为 `qualification105.py` 的模块。
- [既有 prepolicy100 比较](D:/workspace/supermimic-full151-observation-study/outputs/full151_schema2_three_arm_prepolicy100_comparison_CPU_v1/report.json) 是39/39保存比较：Actor/Critic 参数尚同原始值、prior 尚未执行，但 WM/failure、WM/Actor norm 及 CUDA RNG 已不同。因此“同 seed、同 rawinit、同更新预算”不能替代共同 warmup。该证据没有唯一定位分歧起因。

## 2. 必须共同的学习起点

下表是未来共同起点要实际保存并跨臂核对的内容，不是本轮已经构造的 artifact。

| 范围 | 共同内容与理由 |
|---|---|
| Actor/Critic | 完整 state_dict，包括 Actor EmpiricalNorm 与 Critic RunningNorm；相同参数顺序、训练模式、学习率进度和相同空优化器状态。原 Actor 是 AdamW，Critic 是 Adam，不能套用组件 AdamW 状态。 |
| failure/torque | 相同 fail_head state/norm、torque_norm 及 host 配置；第一最小实验维持既有 `failure_head=False`，但不能遗漏仍会在 collect 中更新的统计。没有 failure 训练则明确记录0步。 |
| replay | 所有字段和 dtype/shape、有效存储行、ptr/size/capacity/num_envs/train_end/holdout 分区；包括 x/x1/push/u/c/action/m/obs/aux/contact/torque/reward/done/terminated/valid，以及 full151 phase/generation/start/end/source_frame/reference_id/natural_end。未写区域不能当有效数据。failure-pool 缓存应明确为空并以同源方式重建。 |
| 取样/时钟 | 相同起始 iteration、sim_steps、common_step_counter/max_common_steps、LR/curriculum、utd 累加器状态和 warmup 已完成标志；相同 replay start/critic permutation 抽样规则。不要直接调用原 `learn()` 再跑一次 warmup。 |
| RNG | 新实验共同边界的 Python、NumPy 完整值/缓存、Torch CPU 原始字节、所有已初始化 CUDA device 原始字节、CUDA 初始化状态；记录 backend 随机源/噪声配置与实际抽样顺序。旧组件进程的 RNG 不是新任务起点。构造/导入/拷贝需要隔离并比较其前后，分支的随机流如需复用要明确保存/选择，不能只重新 `seed(0)`。 |
| 输入/物理上下文 | 同 input bundle、151 reference、150 baseline、analytic velocity、actual8全字段和全部128×46 xi、joint/actuator 顺序、limits/encoder bias、固定 palm、4×5ms cadence。WM context 是 state58+xi46+zero174，不混 task goal；u 是控制器已经输出的 absolute target，不再 EMA/C2。 |

依据：[Runner 构造与四优化器](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:153)、[collect](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:377)、[ReplayBuffer](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/buffer.py:6)、[policy 起点抽样](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:676)。`collect` 会更新 Actor norm、消耗 action-noise RNG、真实 step、写 replay、更新 torque/WM/failure norm；不是无副作用的“填 buffer”。

## 3. WM norm 坐标系不能和 fresh warmup 混装

两个 own final 的 norm/host 固定来自同一 parent105。安装必须保留 WM `norm.{mean,var,count}`、`delta_norm.{mean,var,count}`、error_floor，以及 `_host_count/eps/max_count`。后者不是完整 state_dict 内容；见 [RunningNorm](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/common.py:22) 和 [host 配置 helper](D:/workspace/supermimic-full151-observation-study/src/full151_gradient/response_calibration105_v1.py:197)。不能把 fresh shared warmup 算出的 WM norm 替换进去，也不能只拷网络权重留下 Runner fresh norm。

最小路线从共同采集开始就使用该 parent norm 的冻结策略；共同 warmup 只为 replay 与策略输入统计服务，**没有独立 WM/failure optimizer warmup**。Actor/torque 等统计可以在一次共同采集中按固定规则产生。若未来选择不同 WM norm，必须另行冻结适配协议并验证整个权重坐标系；本计划不授权这种改动。

仅 `.eval()`、`requires_grad_(False)` 均不足以冻结显式 norm 更新。[collect:412](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:412) 会直接调用 WM norm.update；[policy_update finally](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:800) 又会把 WM/failure/Critic requires_grad 设回 True。未来隔离 adapter 必须明确跳过 WM 统计/优化器写入，并恢复所声明的参数权限；保留对 state/control 的可微路径，不能用 no_grad 包住 imagination。

## 4. 一个最小集成顺序

1. **前置资格先完成**：缓存原 self15 sensor-label 的保存评价、own WM 在所选 GPU 精度下的安装与输入 AD 工程核验、共同 fresh-reset/策略起点证据。当前 fixed152 的初步主要指标并未支持 method 优于 control；不能据组件 loss 或新 head 推定任务收益。self15 的 trunk 随组件训练变动且没有该标签监督，不能当 HO/力头。
2. 新隔离代码生成一次共同 warmup replay 和 Actor/Critic/fail/torque 学习起点。Actor/Critic 在此阶段没有策略更新；其后优化器统一为空。这一份共同 artifact 复制给两臂，逐字段/字节验证，不能各跑一遍64采集+100/200 WM-only 更新。使用新的 shared-start schema，记录 constructor、common-start、installed-WM 三个不同指纹时点。
3. 每个臂构造相同 native learner，再通过 **新 own consumer** strict-load相应 AFTER1000 `WM_state`（含其冻结 norm），核对 `normalizer_host_configuration`、physical context、xi、source/recipe/final health/arm/iteration。参考 [当前 own final loader:70](D:/workspace/supermimic-full151-observation-study/src/full151_gradient/boundary_component_final_evaluation105_v2.py:70) 的 namespace 与输入 map：payload `input_pins` 对 protocol **`inputs`**，source_pins 对 source_pins，均以 samefile+SHA 比较。不能复制旧集成 KeyError 或用伪造 Toy protocol 掩盖 ABI。
4. **只安装裸 WM，不带 aux hook/head/cache**，不 deepcopy 已包裹模块，不 pickle 整模块。原 forward 仍输出 state58+self15；本次边界 head/损失不加入任务 reward，也不额外改变 P_ONLY/A2 prior。两臂使用相同 reward、Actor/Critic/update/terminal 配置，唯一处理变量是已训练 WM 权重；如果再变 prior 就不能把结果单独归因 WM。安装后重新绑定 `step_wm`，禁止保留指向旧对象的 compiled closure；最小阶段保持 compile=False。
5. 第一策略更新使用完全相同的共同 replay、Actor/Critic/optimizers/RNG 起点，先于任何臂特有新采集。这样共同起点可由已保存学习数据核验，不依赖跨环境任意 mid-rollout restore。后续才按新协议进入相同预算的在线任务学习/固定最终评估；不直接复用原 `learn()` 的预热和 WM fit 循环。
6. 在线物理从明确 **fresh phase0 reset 边界**开始。记录全部128 reset readback、obs/history、reader memory、reference phase/generation、action/prev_action、residual EMA、targets/valid、charge、mocap、qpos/qvel/ctrl、sensor/cache 初始化与 common clocks/curriculum。共同 warmup 结束状态不被假称已经精确恢复。若 fresh-reset 输出/接口有未审字段则停止集成解释，不删 guard；这不是 full hidden-state 复制证书。

跨臂共同的是处理施加前的学习状态和所声明的 reset 条件；策略更新后 Actor、Critic、真实轨迹与各自 replay 自然可能分叉。不能要求它们一直相同，也不能把后续同 seed 当作相同噪声 realization。任何跨臂随机流配对要在新协议明确，不边看结果边调整。

## 5. 真实状态与有限参考边界

[runtime.reset:230](D:/workspace/supermimic-full151-observation-study/src/full151_native/runtime.py:230) 会重建所选环境的初始状态并更新观察历史；[reset_readback:93](D:/workspace/supermimic-full151-observation-study/src/full151_native/runtime.py:93) 是部分实际状态证据，不涵盖所有 solver warmstart/graph scratch/foreign state。[PenSimReader.memory:295](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:295) 包含 c1/c2、a1/a2、charge/age/time/start/end、limits、target、mask 与相机缓存；真实采样时 c1/c2 都来自当前 processed_action，imagination 中则由 [advance_memory:340](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:340) 递推，不能用“全时刻 c1=c2”代替状态清单。

保留原 full151 `phase<150` hardmask、generation/reset-suffix replay mask、149→150 terminal/no bootstrap，以及150无新决策。不可为了任务接入无限延长参考。capacity observer/host OR ledger 外置、单调，不随 learner 分支复制或清零；每个新实际作业独立最终读数。旧 snapshot 的 visible baseline/INTEGRATION 结果不能升级任意128world full-resume 或 whole-hidden 资格。

## 6. GPU 与科学待测项

旧 actual A2 配置是 amp=True、compile=False；[learning_v2:268](D:/workspace/supermimic-full151-observation-study/src/full151_native/learning_v2.py:268) 显式 allow_tf32=True。[WorldModel.forward:50](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/networks.py:50) 仅网络使用 CUDA BF16 autocast，输入 norm 和 integrate 仍 FP32。CPU own parity/after1000健康不能替代 GPU 安装资格，也不能把119f CPU wrapper `.to(cuda)` 当 GPU consumer。

未来单独预注册一种相同 GPU precision/TF32/AMP/determinism/reduction/env-override 配置，记录两臂原始 runtime flags、dtype、RNG 与安装前后权重/norm/host/context，明确 CPU shadow 与 native GPU 数值差；不按结果择精度/回退。不要求跨 CPU/BF16 位相等，但验收规则必须先于实际数据；原 unchanged forward/self15 和有限 input-AD 工程事实需与任务收益分开。

最小后续测试仍只是同参考、同谱系/单 seed 的局部任务实验：参考几何与原完整三臂任务均 FAIL，own组件健康、冲量目标可用性、self15评价或共享起点均不能替代物理可行参考。需要实际确认固定WM条件下策略是否改进原完整任务门槛、是否出现 self15 奖励利用、是否产生非手接触/掉落，以及输入分布偏移。没有准确性/成功阈值新增、CI、winner 声明或实际训练授权。

若 Root 采用此路线，下一步只应实现一个新的安装/shared-start adapter 并冻结最小任务协议；之后有界 GPU 工程检查，再启动声明预算。首个配对 checkpoint100 必须分别完成真实保存健康、共同起点/实际更新次数核验和 ETA，按用户约定保存交接并暂停 Goal；后台是否继续到最终 checkpoint 必须由新协议显式声明。这里没有代填迭代总数、自动排队或开始执行。

## 7. 本轮读取的关键源码与元数据 pin

以下 SHA256 是本轮对源码/JSON 的读取；没有重读权重或 NPZ。源码定位见上文绝对链接。own final 两个 pt 的身份仅来自现成 paired manifest JSON，不是本轮 checkpoint 载入。

| 文件 | SHA256 |
|---|---|
| full151_native/runtime.py | `d41057c2f849fe645a52c7ffa8b175bac60eb6aa1ed616d756a9df64a286f8c9` |
| full151_native/controller.py | `b81c5376caaed93941078600983afa6356df60f01be50dcc2b9c259b19a257e0` |
| full151_native/learning_v2.py | `2f3b0269dac7b3983b6ecfb7b4c5201c681ebc2ed3fd3199179c08607d59197d` |
| full151_native/training_capacity.py | `b7137f4824b8375a6c2670deb5f88e75203fa04403731b5115475f80902c97ef` |
| full151_native/checkpoint.py | `abea7d23db082cc98b5ba77fdaa3fafcac87e8241663e9219b11742997376916` |
| full151_native/world_model_checkpoint.py | `119fa7cebc0e9b0727b0160660b1a47822deee2a0e0e9fa5be69465ac5578052` |
| scripts/audit_full151_checkpoint_cpu_v2.py | `5950b9947b00ed0cfce6a8403a65877db6d71be1d7b01b8c3ad1626bdd5de792` |
| native runner.py | `b19d9ac80e23e4f55b2eb64e458c05516fc5dd4ffb13e1a3050ce9fea2669d46` |
| native buffer.py | `3048a42d13aa5142a4c42a29b5f7c77df02c5c628f6276bf9d5256a644494d5f` |
| native networks.py | `7fbec4de7aa92603565aa24e85d559f4b8c9dc309aae877d3fdd2981402e350a` |
| native common.py | `6d6ff994d592c8fec1d46401d7051d0511310281a3d28450db8bd9be8c327f8b` |
| native pen.py | `c02906c409aedab59aaf68a5ef80f9dbcea52da10e72137091df81858262819a` |
| boundary_impulse_component105_v1.py | `003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317` |
| boundary_impulse_training105_v2.py | `2dd2d3c90f5de47736e4ea4ca87b090c3458659f4982c4ecdf102908c216dbb1` |
| boundary_component_final_evaluation105_v2.py | `5eea0973150225a3b7cc659f8fbc11562000d98e8db9d98d9c9eafd6105a75e8` |
| component_H8_donor105_v1.py | `0c77dc5255ffd2d7294dabdc7bbaf8ed09e80f3f2bc0eb89fee3954d268a6b13` |
| response_calibration105_v1.py | `71efe05da17a37df4111158860443433304f1d57762e2377d4dbafc3b57ca27f` |
| prepolicy100 comparison report.json | `c5b0549654a55f3a086c8dea25a5e714170cace1454a1ca0704754988a18f29a` |
| configs/full151-v3-primary2000-A2-seed0-v1.json | `eb7a45da30ff730cf364d5434c26e9f69f424f0b56ff5ea95dbee1c8b36661e8` |

现成 own AFTER1000 manifest：`outputs/full151105_boundary_impulse_matched_training_CPU_v1/paired_checkpoint_1000.json`。control pt SHA `0e8376586e1065f01ff8aa2375ee670f9bd4abc1171f7ba0ef482aa37e89ebe1`；method pt SHA `8d2e3281f7b8ea3c74a6010a9d573aa09f2fe17347d1461d9b51b29a02838719`。这两个摘要标识安装候选，不给它们新增 native105 resume 资格。
