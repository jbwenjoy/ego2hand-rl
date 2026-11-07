# SuperMimic + CHORD 接触与 ego 输入审阅

- 日期：2026-11-01（Asia/Shanghai）
- 审阅对象：原《Wuji-Hand2-SuperMimic-CHORD-Demo-实施方案》v0.1，以及本地 NVIDIA Video-to-Data / FlashCHORD 源码。
- NVIDIA 本地提交：`0530204f3a085fb9ba9cc41b771fc4c853edeab8`。
- 状态：源码与文件状态审阅已完成；本文不构成 Hand 2 训练、真实视频重建、接触增强收益或真机验证结果。

## 审阅结论

方案的渐进路线合理：先在固定腕 Wuji Hand 2 的已知任务上验证世界模型和残差控制，再检验接触 guidance 是否改善独立物理评估，最后研究跨轨迹成本摊销。共享世界模型有机会降低新增参考的适配成本，但收益仍是待实验的假设。

必须把两个研究问题区分清楚：

1. 已有结构化机器人参考的动力学修正，可用教师轨迹和可控扰动验证。
2. 没有机器人本体状态标签的人类 ego 视频到机器人轨迹，需要额外完成几何/运动重建、尺度与场景对齐、目标手 IK 和固定腕适用性筛选。

教师或 synthetic 数据上的成功不能替代第二项验证。一般浮动手搬运也不能通过删除手腕自由度自动转为固定腕任务。

## 可复查的源码证据

下表路径均相对于 `D:\workspace\video-to-data-research`；行号基于上述固定提交。

| 源码 | 位置与含义 | 对本研究的要求 |
|---|---|---|
| `robotic_grounding/flash_chord/src/flash_chord/data/reference.py:144` | 公共参考接口包含时间、物体 pose、接触几何、资产和尺度 | 原始 RGB 需要先转换为结构化参考 |
| 同文件 `:238` | 浮动手参考包含 wrist pose、finger joint position/name | Wuji 需要专门的 IK/关节映射，不可直接读取 Sharpa 的 22 维手指动作 |
| `robotic_grounding/scripts/retarget/ego_recon_to_sharpa.py:3` | 已有 MANO + object 的 loaded parquet 经 IK 补齐 robot 字段 | 重建与重定向是独立阶段 |
| 同文件 `:93`、`:136`、`:160` | 手尺度、时域平滑、穿透修正是可配置前处理 | 所有处理参数、成本与原始参考均需保留 |
| `robotic_grounding/flash_chord/src/flash_chord/runtime/actions/hand_pose.py:174` | 浮动手动作要求 3 平移 + 3 旋转 wrist DOF | 固定腕必须更换控制契约 |
| 同文件 `:275`、`:297` | wrist reference 与 residual 实际进入控制目标 | 不能把浮动手任务当作固定腕已有结果 |
| `robotic_grounding/flash_chord/src/flash_chord/objectives/wrench.py:36` | 单个接触的摩擦锥 unit rays，扳手为 `[f,(p×f)/rc]`，support 取非负最大值 | 这是几何能力指纹，并非实测力或动力学证明 |
| 同文件 `:164`、`:194`、`:208` | 参考 world→object 变换，basis 用 `rc=1.0`，部件按 1-indexed ID 分桶 | torque 除半径一次；不应再次对 basis 做半径缩放 |
| `robotic_grounding/scripts/retarget/contact_labels.py:115`、`:177` | 接触来自网格邻近与去抖，normal 指向物体内部 | 明确 pseudo contact 的来源与置信度 |
| `robotic_grounding/flash_chord/src/flash_chord/runtime/contact.py:297`、`:315`、`:363` | live contact 聚合的是物体所受力，取接触点平均与总力方向 | 与参考 mesh normal 的语义不同，需校准 |
| `robotic_grounding/flash_chord/src/flash_chord/objectives/contact.py:120` | tolerance-band support 匹配，按 active basis/body/hand 聚合 | 不能将任意 support L1 loss 称作官方奖励复现 |
| 同文件 `:168`、`:206` | 独立的 missed / unintended contact penalty | 释放阶段与异常接触不能被忽略 |
| `robotic_grounding/flash_chord/src/flash_chord/evaluation/harness.py:222` | 无 VOC、无 auto-reset、首帧开始的完整片段评估 | 正式验收不得靠辅助物体力或中途重置 |
| `robotic_grounding/flash_chord/src/flash_chord/evaluation/metrics.py:191` | 完成参考且未跨越误差门槛才计 success | 回报与未终止均不能单独代替成功率 |
| `robotic_grounding/flash_chord/CONVENTIONS.md:28` | Reference/parquet 为 `wxyz`；Warp/Newton 为 `xyzw` | 适配器必须显式转换并校验 |

## ego 数据契约与当前样例状态

“无本体”在本文指人类视频没有机器人关节、力矩或机器人传感器标签。它不表示可以省略人手/物体重建、目标机器人模型或仿真执行状态。

建议 normalized reference 至少保存：

| 类别 | 字段/要求 |
|---|---|
| 来源 | `source_demo_id`、片段范围、原始文件哈希、许可证与数据类型（synthetic / teacher / real-ego-derived / raw-video） |
| 时间 | 时间戳、原始 fps、重采样规则、播放速率；慢放改变动力学，应保留时间缩放 |
| 空间 | 米、弧度、右手系、重力方向、world/robot/object frame 定义、`wxyz`；重建尺度及其来源 |
| 人手 | MANO/关键点、手侧、手腕 pose、可见性与置信度；缺失段 mask |
| 目标手 | 固定的关节名称/顺序、20 维 joint reference、IK 残差、关节限位、腕部安装 pose |
| 物体 | 逐帧位姿、几何/碰撞资产、物体原点与 COM 偏置、质量/惯量/摩擦的来源与不确定范围 |
| 接触 | 点、朝物体内的单位法向、部件 ID、active mask、置信度、生成方法；未知不等于非接触 |
| 场景 | 支撑/障碍物几何、接触材料、场景对齐变换 |
| 初始条件 | 手、物体、控制器历史状态；固定腕适配方法与是否重新设计任务 |

本地 `ego_recon/datacard.md:27`、`:29`、`:73`、`:87`、`:96`、`:108` 明确：

- `tissue_box_simple` 来自一个内部真实 ego 视频，提供 Sharpa 和 Vega 两种重定向表示。
- 原始 MP4 与中间重建包未公开在该样例中，不能声称已复现 raw-video 重建。
- 两种表示来自同一原始演示，不是两个独立演示；没有既定 train/val/test 划分。
- 此样例适合 pipeline/schema/asset 验证，不能单独用于多原始演示泛化结论。
- 审阅时本地 parquet、OBJ、STL 等为约 131–132 字节的 Git LFS pointer，实际样例尚未 hydrate。下载之后仍需再次检查其 schema、资产、坐标与可执行性。

## 固定腕适用性门槛

为每个 ego 片段计算 wrist translation/orientation 范围，以及物体相对手腕的运动；先筛选近固定腕、单手、少量换指的短片段。

普通 tissue-box 搬运通常利用手腕贡献。逐帧使用 `T_wrist(t)^-1` 将物体目标转换到静止手腕系，会改变惯性与重力关系，不能称为保留原演示的接触动力学。若采用此变换，应标记为“固定腕任务重设计”，随后重新验证是否存在物理可行解。

单次全局刚体对齐、逐帧腕相对化、时间缩放、物体缩放是不同操作，应分别保留参数。不得靠前处理把失败片段排除后仅报告剩余成功率；同时报告筛选通过率和排除原因。

## wrench support 移植的可检验定义

每个有效接触点和朝物体内法向，在物体坐标生成 `edges` 个离散摩擦锥 ray，并追加 normal。ray 的 force 被归一化为单位长度。support 为固定 6D basis 上所有接触/ray 扳手投影的最大值，并包含零点。

正式实现必须固定并记录：

- 同一 reference/live 的 basis、seed、摩擦系数、ray 数和相位。
- 接触点相对同一物体原点；若原点不是 COM，显式保存并应用 COM offset。
- `rc` 为同一几何/物理尺度定义且大于零；torque 除半径一次。
- 有效 normal 阈值。官方 kernel 对 `norm > 1e-3` 才计算；更严格的 sidecar 输入校验需记为契约差异。
- live force direction 与 reference 几何法向的区别，以及多 contact 聚合造成的信息损失。

support 只编码 unit-force 几何能力：它不含测得的力幅度、物体质量/惯量、执行力上限或接触力时序。support 相似度高不能证明接触力满足动力学平衡，也不能证明没有掉落、滑移或穿透。

官方 `contact_wrench_support_objective` 对 reference support `cs > 1e-3` 的方向计分；current support `us <= 1e-3` 得零；其余方向以 tolerance band 的平方超差计算 `exp(-loss / var)`。先平均 commanded-active basis，再平均 active body，再平均 active hand。全空参考输出零。missing/unintended contact 是独立项，阶段未知不能伪装为已知无接触。

`scripts/check_wrench_parity.py` 对 sidecar 与原始 FlashCHORD CPU kernel 做数值对照。它验证公式与坐标/尺度处理，不验证已训练 world model 或真实接触 dynamics。

首次 CPU 数值检查使用已有 G1 interpreter 只读导入新研究源码，未向 G1 环境安装研究包或更改依赖。`outputs/wrench-parity-g1-cpu.json` 记录 23 项通过：随机/空/失活接触、极点 normal、摩擦锥边数、零摩擦、COM frame 变换、统一尺度、batch 半径，以及 reward 的多 hand/body 聚合、activity threshold 和固定 reference 梯度契约。运行版本为 Torch 2.14.1+cu130、Warp 1.18.0、NumPy 2.5.4、SciPy 1.18.1；它属于公式独立预检，不替代独立 Dex 环境的复验。

## M0–M3 修订建议

| 阶段 | 任务与进入下一阶段的证据 |
|---|---|
| M0 | 完成版本/关节/时间/坐标契约；记录 ego 原始来源；hydrate 一个真实 ego 派生样例；给出固定腕适用性报告，允许得到不适用结论 |
| M1 | 方块/转笔模型加载、GPU 物理步进、控制目标导出、短训练/保存/恢复；同步记录 EMA、延迟、实际目标等闭环状态 |
| M2 | 教师转笔的可控偏差 A0/A1 以及无辅助全片段评估；独立完成真实 ego 派生样例 adapter；分别报告合成校验与真实来源结果 |
| M3 | support 数值 parity、坐标/尺度/法向校准；接触头独立 calibration；同预算 A1/A2；分析高想象回报/低真实回报反例 |

正式成功标准应同时要求：完整片段、物体任务误差达标、无掉落及非预期支撑、控制/负载限制、合理接触与穿透。对笔记录轴方向与旋转进度；对称性不可观测的自由度单独处理。无物体辅助力、无中途重置，在真实模拟器记录全过程。

多轨迹实验必须按完整原始 `source_demo_id` 分组；同源的不同 embodiment、相邻片段、不同扰动和重建变体留在同一划分。跨轨迹泛化与物体参数/几何泛化分开。成本应包含教师、重建、重定向、训练、编译、校准、评估及重试。

## 已验证与尚未验证

**已验证：**上述固定提交的源码语义、ego datacard 的来源/数量声明、本地样例为 LFS pointer、官方完整序列评估会去除 VOC 与 auto-reset。CPU parity 是否运行及其数值结果以 `outputs` 下 JSON 为准。

**尚未验证：**Hand 2 本机训练收敛、真实 ego 片段对固定腕的可行性、Wuji 目标手 IK、接触增强有效性、learned world model 的物理梯度方向、多轨迹摊销收益、真实硬件接口与执行效果。

## 一手资料与运行平台边界

- [CHORD 原论文](https://arxiv.org/abs/2607.00033)：object-centric wrench guidance 的研究来源。
- [NVIDIA Video-to-Data 官方仓库](https://github.com/nvidia-isaac/video_to_data)：重建、重定向与 grounding 的官方接口和当前平台说明。
- [固定提交 ego datacard](https://github.com/nvidia-isaac/video_to_data/blob/0530204f3a085fb9ba9cc41b771fc4c853edeab8/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/datacard.md)：样例来源与限制。

2026-11-01 在线核查时官方 README 声明，特定 `v2d_cusfm` 多视角重建栈受 TensorRT/cuVSLAM 版本限制，不支持 Blackwell `sm_120`。这不能推广为所有单目 ego 模块均不支持 5090；也不能据 G1 训练可运行推断完整 V2D 重建已兼容。当前最小研究可先从已有重建派生参考与固定腕仿真开始。
