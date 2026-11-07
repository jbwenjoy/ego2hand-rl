# full151 新 mode：身份与时序合同草案 v1

本草案建议把 legacy source `[200,351)` 的观测派生目标接入**独立的新 rough-reference mode**。151 个参考状态对应 150 个真实控制 interval，持续 3 s；现有 Reader 已具备此长度的 CPU ABI。新输入束、运行适配器、checkpoint consumer、物理和训练资格仍待实现。本文件不授权求解、训练或 GPU，不把旧 104 的成功迁移为新轨迹资格。

## 已有证据与不得混淆的身份

[真实 native ABI 报告](D:/workspace/supermimic-shared-reference-study/outputs/native_ABI_cpu_v1/report.json)保存了 151 帧 `SingleHandTrackingReader` CPU 验证，native NPZ SHA 为 `7fe31c191141316fdc12190b1077eb7b7a5a18ec453d2db8eec9e3fb4d79e399`。它来自观察受限重定向的 539d 派生 q_goal，不是原始机器人本体标签，不是 RF-prefix16 的延续。本次仅检查报告及文件字节，不重跑 Reader 或 FK。

[现有差距报告](D:/workspace/supermimic-rough-prefix-learning-study/docs/full151_integration_gap_current104_v1.md)与[本轮坐标/接触来源审阅](D:/workspace/supermimic-full151-observation-study/outputs/full151_coordinate_intent_source_review_v1/report.json)明确：legacy `[200,351)`、RF-prefix `[200,216)`、shared007 `[0,151)` 是三个不同身份。不得只匹配 clip007 或帧数，不得移入旧 shared registry、继承其 split/use，也不得用 16 帧 padding 或新 q0 拼接旧 q1..150。

输入生成器需要新建不可变 variant，绑定观测 SHA、派生 q/物体姿态、时间表、速度权威、初始化、150 条 baseline u、native 四字段与 recipe、MJB/资源/actual8 以及独立 contact-intent bundle。保留 `physical_reference_qualified=false`、`shared10_enrollment=false`；rough-reference 工程资格、训练授权、实际任务成功分别记录。相同来源的 151 帧不是 151 个独立 ego 演示。

native BODY21 是 palm + 每指 link1/2/3/4。几何 macro21 是另一组点，不能改名为 `link_pos_w`。导出必须从最终派生 q 重做准确 BODY21 FK；本轮不执行。证据见 [ABI exporter](D:/workspace/supermimic-shared-reference-study/src/native_reference_abi.py:50) 和 [Reader](D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/mdp/commands.py:707)。

## 一个 horizon authority，禁止全局数字替换

建议在新输入 descriptor 中验证以下只读参数，再由各 adapter 共同消费。不要为接受新数据而放宽旧 104。

| 参数 | 新 mode 值 |
|---|---|
| N / M=N−1 | 151 个 reference states / 150 个 control intervals |
| local reference / source reference | `[0,151)` / `[200,351)` |
| real active phase | 整数 `[0,150)` |
| last real phase / terminal observation | 149 / 150 |
| preaction source frame / terminal source frame | `200+phase`（200..349）/ 350 |
| baseline table / inactive placeholder | FP32 `[150,20]` / `u149` |
| control / integration | 20 ms / 4×5 ms |
| full evaluation | 150 actions、600 solver steps、601 raw states、150 native postinterval forwards |

构造、CUDA graph warmup、reset 的 step/forward 另实计，不能把上表称作 whole-process 精确总数。`N` 与 `M` 必须为严格整数，bool/float、N/M/end/signature 不一致均在构造前拒绝。

网络、contact 和机器人维度保持独立：Actor155→20，state58、xi46、disturbance174、WM298 输入/delta52、自碰 head15。**head15 不是旧 interval15；不能全局 15→150。** [原生 PenObservations](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:381) 与 [WM loader](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/world_model_checkpoint_v2.py:41) 是准确边界。

## 控制与时序

新真实和想象控制都保留：`c=.5*clamp(.5*a,-.6,.6)+.5*c1`；`u=C2_020(clamp(baseline_u[phase]+c−bias0, hard∩ctrl))`。c1/c2 只存残差，baseline/u0−q0 不写进 EMA；`frozen_target` 存实际 absolute u。只由 phase 查 baseline，不增加实时 q、未来 state 或随机数读取。不移动原生已限幅的 `q_ref+c`，而从 processed residual 重建。限幅纯函数、FP32、mask0、scale/clip/EMA、bias0 与 old104 核心数学不变。

新表须在构造前对全部 150 条验证 hard∩ctrl 与 C2 identity，不通过就拒绝，不在 loader 内修复。real process→cache→rawctrl→replay.u→WM input/PD 均应逐位相同；replay.c 继续是 processed residual。[现控制层](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/controller.py:129)、[想象控制](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/controller.py:193)、[collect 取 u/c](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:393) 给出了可复用接线。

Manager 的顺序是 process→4 steps+scene.update→termination/reward→原生 forward→command advance→sense/observation。见 [Manager](D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/envs/manager_based_rl_env.py:434)。因此最后一次 phase149 执行后，在 command 仍为149时标 `done+timeout`，随后提供结点150的 terminal observation；不能再执行 u150。原生 `time+1>=length` 在此 finite-reference 合同下会多执行一个 interval；新 adapter 应按已验证 M 判断 `time>=M−1`，保持 `time_out=True`。不要清 `_manual_reset_pending` 以继续额外 action。[旧 timeout adapter](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/runtime_v2.py:36) 只是 N16 特例。

训练 reward 保持原 native **poststate x(k+1) 对 preaction reference k** 的时相，不把 reference k+1 偷换入训练公式。物理跟踪评估使用 poststate/reference k+1；secondary 的 k/k+1 数值须另标。不重复乘 dt，原 reward、WM loss、failure/selfcollision 监督、几何 prior 公式保持；新目标 bundle 的身份需要重新绑定。

## H8、终端与 replay

新 `phase_active` 由 `0<=phase<M` 导出。H8 从 phase142/148/149/150 开始，活动行数分别是8/2/1/0；第150及以后只允许有限 `u149` 想象占位，实际动作接口则拒绝。第149次 reward 仍参与 actor/critic，之后 survival=0；终端150不可贡献 final bootstrap、critic normalization 或 loss。保持原 H8 迭代、随机抽样和 optimizer 调用顺序，不能以跳过 dead loops 获得不同随机流。对 inactive 行仍验证有限性，不用 `where` 掩盖 NaN。物理软终止与硬参考终端不同，不能把所有软失败改成无效样本。[旧 runtime policy adapter](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/runtime_v2.py:125) 的 active mask、norm eligible、minibatch active denominator、bootstrap 层需参数化。

replay 每条真实 transition 保存 phase0..149、start0/end151、variant/reference identity、source preaction frame200+phase、episode generation 和 natural_end=(phase149)。终端 transition 保留；紧随 reset 的 suffix 被 native alive×valid mask 屏蔽。generation 不能在前一条没有 done 时变化；同一 episode 有效连续 phase 必须+1。timeout 不加入物理 failure label。元数据只供校验，不添加网络 clipID。[现 replay adapter](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/runtime_v2.py:167) 与 [native collect](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:447) 是真实接口。

## 速度、初始化与 selected reset

推荐保留“连续参考解析/明确设计的 qd151、物体 world v/omega 为权威”的结构；**具体插值和求导方法等待 Root 新 inputvariant 冻结，本文不替生成器选择方法**。native Reader FD 仅保留原始 ABI 诊断，不偷偷替代 command/reward/WM context/init 权威。若另选 FD 必须在新 descriptor 中公开其方法、端点、单位和 authority，所有 arm 一致。q_goal/qdot_goal、requested init、measured readback、executed u 四类分开。

在原 fixed-wrist rebase 后、首 reset 前安装新表，并绑定与 BODY21/reference 的相同151结点；不把 world 速度再旋转一次。物体 Entity 写入接口使用 world linear/angular；native freejoint angular 列为 body ω，需 `R0.T@ω_world` 对照实际写回。bodyorigin/COM 偏移假设必须绑定 MJB；不能把当前 COM0 静默泛化。[现安装入口](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/runtime_v2.py:65)、[selected restore](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/runtime_v2.py:92)。

selected reset 只改所选 IDs：new init q/qd/object pose/twist、固定默认 robot palm/mocap、u0、raw/processed/previous action、残差记忆、phase0、时间、generation。非选行和 target/history 不变，不做 q projection、object 助力或额外 PD settling。actual8 在 nominal model upload 后 copy→set_const0 restore→双读后才首 reset。保留已有 ObservationManager `compute(update_history=True, env_ids=ids)` 回填方式，不能再次 reset history/delay 或额外重采样；这条 API 已有真实 CPU 历史证据，不能回退成 False+ids。

初始化实测必须保存 full qpos/qvel、joint地址/dof/ctrl mapping、requested/actual q/qd/object/world-body twist、mocap、cache/rawctrl、EMA/action、phase/generation。缺失或不相符时工程失败，不用 nativeFD/旧 prefix init fallback。

## checkpoint 与 consumer 身份

建议 **105 作为候选版本，尚未预留/实现**。本轮只搜索相关 rough/C2/full151 源，不能声称全局无占用。真正决定前再次核本机版本注册；新 kind/schema 及签名比数字本身更重要。

新保存层只在原 native atomic serialization 边界封装 fresh format3 payload，加入新 mode/controller metadata 和最终新格式；这不是加载旧3 checkpoint。from-scratch-only，resume/modelonly/oldActor/oldWM warmstart 一律拒绝。保留 full modules/norm/optimizers/config/joint/context/reference signature。原3/103/104 loaders 不改，新 loader 严格拒这些旧格式；把 old104 的整数改成105仍因 root/source/horizon/limits/init/velocity/baseline 签名不符而拒绝。[现 save/load 边界](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/controller.py:246)、[严格104 consumer](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/checkpoint_v2.py:68)。

新共同 descriptor 至少绑定：mode/variant；legacy source window/lineage；native recipe/data/signature与 default wrist/BODY21/joint order；N/M/dt；reference/velocity/init/u150 exact source+FP32 raw bytes；hard∩ctrl、C2 .020及 pure source；residual memory/terminal/reset/replay rules；reward/eval phase；MJB/资源/actual8；contact-intent/objective来源；physicalfalse/sharedfalse；input signature。运行附加 method/seed/role/config/source inventory 另签名，mutable latest evidence 不冒充稳定身份。训练更新计数、checkpoint prefix capacity、final capacity、raw initial module fingerprints 独立保存。

## 必须独立实现/参数化的真实函数

| 现有函数或入口 | 新 full151 处理 |
|---|---|
| `bundle_v2.load_bundle` | 新 bundle/schema/variant/window，N151/M150，不接受旧 prefix aliases |
| controller `_phases`, `rough_controller`, `control_target`, `collect`, `save/no_resume` | 由已验证 horizon 推导边界、baseline shape/placeholder/histogram/new signatures；保留核心公式与全 batch 证据 |
| runtime `phase_active`, `reference_timeout`, `_validated_bundle` | N/M/last_phase 同源，严格 real/dead 区分 |
| runtime `install_velocity_authority`, `_restore_initial_state`, `reset_readback` | 新表/actual init/选行缓存历史/frames151，保留固定腕和角速度 API |
| runtime `make_policy_update`, `masked_critic_mse` | H8 hard end、final bootstrap、norm/loss active denominator；源匹配替换数量失败关闭 |
| runtime `validate_replay_window`, `install_replay_metadata` | end151/phase149终末/source offset/generation；终端保留/重置后缀屏蔽 |
| native `PenMDP.reference`, `Reader.read` | 已可使用动态 start/time/end 和实际帧数；不必改其原源 |
| native orchestrator/training/save/capacity | 新 imports/descriptor/format，保留实际8/原 save 与 observer时相；角色授权独立 |
| Actor/WM loader、health auditor、contact binding、probe/policy_probe | 新严格 consumer，审 source graph/新151 target；不放宽旧104 |

## CPU 前置合同与阶段顺序

机器可读取的逐项负例见 [静态报告](D:/workspace/supermimic-full151-observation-study/outputs/full151_mode_static_design_v1/report.json)；**本轮没有执行这些运行合同**。最低顺序：新 input producer freeze→纯数组/source binding 与负例→真实 Reader CPU151/BODY21/velocity diagnostic→真实 ActionTerm/PenMDP core parity/FP32/AMP/梯度→selected reset/history/replay/H8/save/load CPU→另行一次有界 native A0 工程资格→才可独立授权 smoke/学习。A0 的质量失败不自动否定 rough-reference 学习；身份、容量、完整性失败则不可作为学习资格。

必须有的反例包括旧模式/重标格式/window错配、表151或149行、真实phase150、H8跨边界、NaN dead row、dead1000对 norm/梯度影响、generation跳跃/终端行丢弃、selected reset 污染其他 worlds、world/body ω混淆、nativeFD偷换、第二次C2或EMA baseline污染、cache/rawctrl/replay.u不一致、未知/非零overflow、old16 targets padding、sourceq/macroFK注入、质量早停误判完整轨迹、源码/模型/目标SHA漂移。

full horizon评价固定150个控制 interval，原门槛与 criteria source 必须显式绑定；quality failure仅 latch 并走完整3s。数值硬异常可截断但标 censored/未完成，不能以短 prefix 平均代替 fullhorizon。保留 raw5ms601和controlknots151；只在新输入确有 source-bound 601诊断目标时才评估中间点，不能临时补插值。policy rate含 startup ctrl→learned u0 共150个差；baseline 自身只有149个邻接更新，二者不混。

新 A1L/P_ONLY/A2 应共享输入/initialization/controller/native cadence/真实 transition 与 update 预算；旧 prefix2000预算不自动授权新 full151。若将来明确复用128world/2000iter/UTD.4/warm64/wmonly200/2WM，则实际收集 `(64+5000)*128=648192`、WM4000、failure4000、actor1800、critic7200，只是来自原循环的条件算术，**DRAFT_NOT_EXECUTED**。新长轨迹改变每阶段相位覆盖、reset频率、有效 imagined rows和wall，必须实记；同时报告名义 optimizer 次数与 eligible-row/denominator，不以少 reset 当学习提速。teacher/几何输入生成、预训练/新增轨迹适配、A0/各方法评估另计；等 wall 次口径在结果前冻结，超时工程 guard 与研究 wall 截点分离。

当前建议为单源受控 pilot，不创造新独立 N，不并入旧104、历史directWarpT0、shared10或零样本C2 portfolio。多seed/每模型多process及其 all-repeat 主终点若将来采用，必须在新协议预声明；本草案不设训练seed/budget/queue。

## 本轮收口与待决

本轮只完成 source/JSON/AST/hash 静态设计；旧 rough104/shared 源、报告和数据全未改。源SHA、符号行号、静态断言和完整负例清单保存在 report.json，source pre/post字节一致。不加载 NPZ 数组，不 import Torch/Warp/MuJoCo，不执行 Reader/FK/碰撞/优化/WM/Env/GPU。

待 Root 冻结：新 inputvariant及完整151几何选择；q/物体连续曲线与速度authority；150条baseline的构造和FP32证据；新独立151 contact意图来源；105最终版本/kind命名；新模式阶段性工程/训练授权。输入尚未齐备不妨碍先设计合同，但不应实现一个绕过真实 native 接线的“宽松通用 loader”。
