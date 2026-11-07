# 共享多轨迹受控协议 v1（DRAFT，未执行）

本交付是 CPU 数据清单、仅观测字段导出、源码绑定与实现差距；没有启动 GPU、训练或新物理步进。机器协议：[shared-multitrajectory-controlled-v1.json](D:/workspace/supermimic-chord-study/protocols/shared-multitrajectory-controlled-v1.json)；清单：[inventory.json](D:/workspace/supermimic-chord-study/outputs/shared_multitrajectory_inventory_v1/inventory.json)；核验：[checks.json](D:/workspace/supermimic-chord-study/outputs/shared_multitrajectory_inventory_v1/checks.json)。

已有 `configs/controlled-study.json` 与 `docs/research-plan-v0.3.md` 提出1/5/10/20条共享成本曲线，`outputs/source-inventory.json` 已核10条源。本次复用它们，并补上真实源帧、SHA、输入导出、训练接口与预算约束。当前只有10条，N20不能由切片、扰动或换本体凑数。

## 数据与 lineage

全部10条保守归入同一 `upstream_01_single_axis_original_parent_unknown_ALL10` lineage/split_group；原视频父ID缺失，独立ego演示N=0。所有源已被几何/初始化筛查；001工程链路、007训练评估、009静态对照已有历史暴露。以下划分是现在制定的受控同lineage方案，不是假称历史盲测。每条完整源范围见清单；默认待资格窗口是[0,151)，3秒，尚不允许物理启动。旧007[200,351)已见pilot单独保留，不自动替代新观测派生参考。

| 源clip | 帧数 | 时长秒 | SHA前缀 | 同lineage留出轨道用途 |
|---|---:|---:|---|---|
| 01_single_axis/clip_001 | 369 | 7.36 | ca0b9b5dfc5d | train_pool |
| 01_single_axis/clip_002 | 452 | 9.02 | f9d0e4495dc6 | train_pool |
| 01_single_axis/clip_003 | 373 | 7.44 | 148da96fbb61 | train_pool |
| 01_single_axis/clip_004 | 391 | 7.80 | a08449835d64 | train_pool |
| 01_single_axis/clip_005 | 317 | 6.32 | c466d76173d6 | train_pool |
| 01_single_axis/clip_006 | 397 | 7.92 | db00f018be7c | train_pool |
| 01_single_axis/clip_007 | 437 | 8.72 | b7815d663539 | train_pool |
| 01_single_axis/clip_008 | 343 | 6.84 | d8d2439467ce | train_pool |
| 01_single_axis/clip_009 | 459 | 9.16 | 3543e3407ce6 | validation_within_lineage |
| 01_single_axis/clip_010 | 449 | 8.96 | d67d2c803666 | test_within_lineage |

成本轨道N=1/5/10有意在训练过的参考上评估，只回答已条件化参考的成本与质量。另设同lineage轨迹留出轨道：001–008训练、009验证、010测试，训练N=1/5/8；外部留出轨迹不得进入训练env/replay/normalizer。两个轨道不混池，不能称独立视频或真实ego泛化。

## 输入、参考和仿真标签

仅观测导出实际包含 `time_s/hand_landmarks_source_w/object_pose_source_w`，不包含cached robotq、link、wrist或力。机器人标签仅由NPY header登记维度，不参与观测构造。原坐标、尺度、相机和MediaPipe名称均不构成真实ego资格。

原生actor155维分为89维参考命令（q20、物体轴点6、link63）和66维当前机器人/物体/上一动作。无示范者本体输入路线中，q/link必须是观测IK与已知机器人FK的派生目标，不能偷偷使用cached `reference_qpos/link_pos_w`。执行机器人自身q/qd可用，与示范者没有proprio不是同一个条件。物体当前姿态目前由受控仿真提供；真实图像估计仍待资格。参考未来值仅作为预先声明的外生命令，不可读取未来真实物理状态、接触或当前策略的成功轨迹作为目标。

WM保留58维机器人/对象状态和实际执行u20；context46/外生扰动174布局明确。x1和15维手指自碰是机器人仿真监督标签，不是视频观测或手物力标签。物理WM不需要clip one-hot；actor已通过参考内容条件化。整数trajectory/sourceID只用于索引、分层采样和审计，不加入网络输入。即使无ID，参考内容与phase仍可能被记忆，因此只在明确留出轨道报告受控同lineage迁移。

## 已有接口与必须实现的差距

原生MotionCommand已拼接多轨迹并保存start/end，逐轨迹重定位固定腕并计算速度；PenMDP.reference使用start+localtime且在end−1截断，当前memory已保存这些值。本次执行了真实源码方法的CPU合成边界检查（长度2/3/1，负time、首帧、末帧和溢出time），不把它当多轨迹训练通过。

研究entry仍在 `D:/workspace/supermimic-c2-training-study/scripts/train_controlled_pen.py:250` 强制单clip。新实现需隔离新entry并冻结整个catalog/派生参考/初态注册表；保留旧研究源。当前reset目标来自所选motion.joint_pos，并非已证明错位，但多轨迹q/qd/object/EMA/u一致性尚未实际验证。A2目前只有单reference bundle；必须按相同拼接offset构造独立固定多bundle，不能让每条都用localtime查同一目标。unknown/no-contact继续取消吸引，P始终存在，外部dt只乘一次。

ReplayBuffer的holdout只分env，不分轨迹；actor.normalizer更新所有训练env，sample_starts亦包括WM heldout env。这可用于训练环境中的WM诊断，不能充当actor或轨迹留出。外部009/010须单独catalog与env，完全排除训练统计/梯度/采样。新增trajectory元数据和quota必须在replay window跨reset或trajectory变化处掩码，不能跨片拼接监督。

原load_model_only仍严格绑定reference_signature，C2 format103 adapter又明确from-scratch-only；共享预训练迁移不能改format位或禁签名来绕过。必须定义新schema与显式允许的catalog变化，核物理context/控制器/状态布局/normalizer/数据暴露/父checkpoint。旧格式与检查器不被本DRAFT授权复用。原WM loss、真实reward与15维自碰监督不改变；A1L/P_ONLY/A2差异只在原约定imagination reward。

## 公平预算与成本（全部DRAFT）

主比较I0独立WM+actor与S2共享WM+参考条件actor；S1共享WM后逐轨迹actor适配需独立transfer合同后再做。每N、每种子、每目标臂、每共享条件匹配总真实transition和optimizer次数：648192N / WM4000N / fail4000N / actor1800N / critic7200N。提议独立N次各2000iter、warmup64、wm_only200；共享2000Niter、warmup64N、wm_only200N，因此不能只增maxiter而忘记额外actor预算。算法计数基于当前runner learn的128*(64+2000/.4)；仍须新entry实际全批审计，绝不称已训练。

共享batch/H固定，replay提议总128000transition slots（共享1000rows；独立各1000/N rows），峰值内存与历史采样分布另报。工程seed0、N2/10iter/wm_only2只验actor/prior路径，不能充正式结果；正式种子7/8/9仅提议，当前没有执行授权。共享预训练不得在N1之前免费看过全部10条；每新增参考适配、教师、失败尝试、编译、重定向、初态筛选、评估均分列收费。不把历史cache/teacher当免费，不假定共享提速。

次口径提议方法级总墙钟1200N秒，包含预训练/适配/各独立进程启动，从实际OS创建时刻计；只取截止前完整的最近checkpoint，不能按质量挑。未启动/未完成模型标NOT_RUN/PENDING，不能记taskFAIL或删除。driver/child/evaluator时间嵌套，不相加；不同失败尝试成本保留。主口径固定预算不由次口径替换。

## 物理门槛、层级与启动条件

每条先冻结完整原task acceptance字段、input/context/control/参考与接触bundle SHA；3秒151状态、实际u/EMA/t0/capacity/noreset等工程资格和task质量分开。原2cm/20度/3mm/actual-q1mrad、progress/load/rate等字段必须由每参考完整协议定义，旋转progress轴不能把旧clip007轴直接复制给所有轨迹。静态轨迹另标任务，不挑更易任务冒称转笔。物理质量失败锁存且完整继续，只有工程异常/非有限/硬时限停止；无质量停队或自动重试。

每个checkpoint×trajectory固定3个fresh-process重复，以3次全过为该checkpoint/trajectory主终点。方法训练seed N=3，process与轨迹嵌套，帧不作样本N；同lineage轨迹不作独立ego示范N。跨轨迹macro结果与每条完整失败/初始化覆盖率一起报告，不仅显示胜者。教师若未来生成，actualq与targetu分开、控制memory/真实对象初速度保留，成本计入。

目前观测派生rest首帧虽测量工程合格但保持失败，H23/H123 v1还不具静力wrench支持；新映射/候选工作未形成合格multi init registry。本协议允许先完成CPU inventory；不会因此启动物理或大规模训练。新框架、controller real/imag parity、所有目标bundle、checkpoint loader/auditor、逐参考初态与任务门槛实际资格及Root明确授权，都是训练前置条件。
