# Dex-SuperMimic：面向 CHORD 接触重定向的源码审阅

审阅日期：2026-11-01。范围：本地源码与原实施方案；只读核查，不把上游作者环境的测试声明计为本机结果。

源码基线：

- `D:\workspace\dex-supermimic-research`：`2becb7380fed86171056f9bf4338b2a6a0c14b01`。
- `D:\workspace\supermimic-manipulation-research`：`eda7da480c6d92a68cc41c9628b7efdae08cae23`，来源 `https://github.com/emoPointer/supermimic_manipulation.git`。
- 原方案：`C:\Users\ASUS\Documents\Codex\2026-10-31\qin\outputs\Wuji-Hand2-SuperMimic-CHORD-Demo-实施方案.md`。

## 1. 结论与首轮适用边界

将 SuperMimic 的可微世界模型与 CHORD 的参考条件/接触目标结合，是能够分阶段检验的研究假设。现有 Dex 仓库已实现固定腕 Hand2 的参考条件残差控制、物理状态世界模型、短想象策略更新与检查点链路，可作为仿真底座。它尚未实现视频重建、无本体人手到机器人重定向或 CHORD 手物接触扳手奖励。

首轮应固定手型、笔、腕姿、控制增益与物理参数，去掉未知扰动，验证已知参考能否回放及修正。无本体视频只提供目标证据，不能单独监督完整机器人动力学；世界模型仍需真实物理仿真中的 `(state, executed_target, next_state)` 数据。

原方案中 A3（pen PPO 对照）目前也是待实现项：`config.py:149-153` 明确将 PPO 限制为 `hand2-reorient`，不能仅把 `--task` 换为 `hand2-pen-spin` 就运行。

## 2. 已实现的数据与控制契约

### 2.1 参考数据边界

`tasks/pen_spin/mdp/commands.py:654-660,707-756` 定义每条 NPZ 的必要字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| `reference_qpos` | `[T,20]` | 机器人关节参考，rad |
| `wrist_pose_w` | `[T,7]` | 原参考腕世界位姿，位置与 wxyz 四元数 |
| `object_pose_w` | `[T,7]` | 笔世界位姿，位置与 wxyz 四元数 |
| `link_pos_w` | `[T,21,3]` | 掌与20个指节 link 原点的世界位置，m |

采样周期来自组目录 `index.json` 的 `frame_dt`，而非 NPZ；命令校验 `frame_dt == control_dt`，并逐控制步推进一帧（`commands.py:345-358`）。现任务为 50 Hz / 0.02 s。

`tooling/clip_store.py` 定义目录 catalog 与 SHA-256；`tooling/recipe.py` 能以 `clips` 选择单条轨迹。下载器 `tooling/download.py:17` 指向官方 v2026.10.22 release，约247 MB归档。审阅开始时本 checkout 内未发现 NPZ 或 PT；后续下载和训练结果应以 bring-up 产物为准。

文件仅提供机器人运动学参考，未携带原视频、相机内外参、分割与遮挡置信度、接触点/法向、真实接触力、原视频 provenance 或训练划分。新 adapter 应补充独立 manifest，不能把这份 NPZ 直接称为无本体 egocentric 数据。

### 2.2 固定腕转换的物理含义

`commands.py:294-343` 先把源数据变为腕局部表示，再将每帧腕位姿替换为同一固定安装位姿；笔目标的世界线/角速度也重新从固定腕后的目标计算。因此转换保留相对手腕的几何目标，删除原始腕的平移/旋转动力学。原始人手靠腕驱动的动作不保证在固定腕 Hand2 上有解，需单独做可行性筛选，不能用坐标变换作为物理可行性证明。

### 2.3 控制闭环

真实控制为 `tasks/pen_spin/mdp/actions.py:62-82`，想象对应 `rl/dex_supermimic/pen.py:81-109`：

```text
residual_raw_t = clamp(0.5 * action_t, -0.6, 0.6)
c_t = alpha * residual_raw_t + (1-alpha) * previous_residual
u_t = reference_qpos_t + c_t - encoder_bias
u_t[frozen_dofs] = previous_executed_target[frozen_dofs]
```

`c_t` 是 EMA residual，`u_t` 是实际绝对位置目标，二者不可混用。WM 输入和 replay 的 `u` 保存后者；策略为20维。默认 `alpha=0.5`。上述代码未对绝对目标施加位置/速度限幅；若加入硬件限幅，应同时修改真实动作执行器与可微控制器，并保存限幅后目标，避免想象学习另一套控制系统。

想象所需记忆在 `pen.py:18-38`：EMA residual、前两步动作、上次绝对目标、冻结 mask/剩余时间、参考 `time/start/end`、过流积分、当前力矩上限、相机延迟/丢帧缓存、编码器与相机偏置。`runner.py:340-352` 显式递推控制/参考记忆，不由 WM 预测。

## 3. 世界模型及接触能力

物理状态为 `state.py:8-19` 的58维：关节位置/速度40维，笔掌心系位置3、旋转矩阵9、线速度3、角速度3。增量52维，旋转由 `R_next=R @ exp(delta_rot)` 积分，避免直接回归无效旋转矩阵。

`networks.py:11-76` 中 WM 从物理状态、仿真 context、外部 wrench 与实际执行目标预测物理增量与接触 logits。参考不直接作为 WM 输入，机器人动作通过绝对控制目标影响动力学；这一复用边界合理。

**现有15维接触是手指自碰撞，不是手与笔接触。** `pen.py:339-341` 读取 `finger_collision`，对应 `config/wuji_hand2/env_cfgs.py:340-352` 中 robot finger geoms 对 robot palm subtree 的 sensor。15维为10个中节标记与5个合并 distal/tip 标记；`pen.py:153` 仅用于自碰撞惩罚。提高 `contact_weight` 或改该奖励并不能检验 CHORD 的物体接触增强。

最小 CHORD 集成应增加独立手物 contact 通道，避免更改自碰撞通道语义：

1. 从仿真手-笔接触提取逐帧接触点、物体向内法向、指节ID与存在掩码，并记录力/穿透等独立评估量。
2. 把固定的参考/教师接触转换到物体坐标系，建立 wrench-support 目标。观测不确定性与未知接触使用有效掩码，未知不能编码成明确无接触。
3. 先在真实仿真评估中验证 support 度量，再增加可微几何代理或独立接触头；自碰撞预测维度保持分开。
4. 接触头单独报告 contact precision/recall、切换时刻误差、support 误差以及高想象回报但物理失败的反例。不能将预测接触自身当作评估真值。

## 4. 会影响结论的漏洞与最小补丁

### 4.1 笔任务 `task/rot_error` 指标引用了错误目标

`runner.py:471-478` 始终计算 `angle_error(object_rotation,memory.goal)`。而笔任务 `pen.py:298` 将兼容字段 `goal` 设置为单位矩阵，真正目标来自 `reference(memory)`。因此日志数值是相对掌心单位姿态的旋转幅度，不能解释为笔轨迹姿态误差。

最小修补：pen 分支使用 `ref_pose = mdp.reference(m)[1]`，报告当前位置误差、笔轴夹角与所定义的旋转进度；cube保留原目标。数值校验应覆盖“物体完全等于非单位参考姿态时误差为0”。A0/A1/A2共用独立 evaluator，避免训练日志自证改善。

此外，`pen.py:116-122,135-144` 的奖励使用沿笔 +X 的两个端点，`observation()`也只暴露端点，因此绕笔本体X轴的 roll 被有意忽略。不能将该奖励表述为完整 SO(3) 姿态跟踪。对笔轴空间翻转，可报告有向轴误差及围绕指定外轴的展开角；若研究笔本体轴向旋转，需要能观测该自由度的物体标记/姿态与相应策略输入。

### 4.2 FailureHead 的真实/预测 context 时序不一致

`runner.py:562-570` 的预测分支先推进 `memory`，再从 `x1,next_memory` 构造 failure features；`runner.py:617-628` 的真实分支却用 `x1,pre_step_memory`。`collect():428-444` 用 next memory 更新 normalization；`policy_update():751-763` 也用 next memory。对笔任务，参考帧与过流 charge会发生一帧错位，同一 failure 标签对应两种不同语义。

最小修补选择统一语义：明确物理终止判断发生于当前参考帧，所有真实/预测/想象分支都构造该同一 postphysics、pre-command-advance 特征，charge使用同一定义；或全部构造 next-memory 并保持标签可解释。为避免以后修改控制器引入重复逻辑，优先存储独立的 failure-phase context（或显式的阶段索引），在三分支复用构造函数。首轮可用 `--no-failure-head` 隔离该问题，解析位置/轴向终止仍保留；修补后必须重训 failure head，不复用旧归一化与权重作严格比较。

### 4.3 DR context 不足以唯一决定接触动力学

`sim.py:185-240` 仅拼接6个平均DR比值与逐关节Kp/Kd。`pen.py:246-278` 复用cube DR读取；`tasks/dex_reorient/mdp/observations.py:222-296` 的几何统计把所有geom尺寸比直接平均。笔训练还随机化COM、惯量、几何、关节摩擦、接触参数、力矩上限等，详见 `pen_spin/config/wuji_hand2/env_cfgs.py:80-218`；这些参数未完整进入WM。不同物理系统可能产生相同context，造成不可约预测误差和错误梯度。当前力矩上限只在 `memory.limits` 用于解析奖励，不进入WM输入。

首轮关闭这些DR，仅保留固定配置；得到可复现基线后按因素添加。多物体/跨参数实验再把质量、物体COM与惯量、几何尺寸、接触摩擦、执行力矩上限以及有效被动参数显式条件化，并记录资产哈希。平均size ratio不是物体尺寸的充分表示。

### 4.4 想象外力与真实扰动调度不匹配

`pen.py:1-6` 和 `runner.py:777-787` 表明 pen 想象保持起点external wrench；真实任务却有启动、衰减、冷却的物体/指尖扰动。监督WM逐步读取replay wrench，但策略不递推调度器。因此从扰动中段抽取起点会把短脉冲当成持久外力。

最小修补：首轮关闭 `object_disturbance`、`fingertip_random_force` 与mask；需要鲁棒性时再把调度器记忆加入闭环、或明确以指定扰动过程采样想象，并在统一无辅助条件验收。接触力与外加wrench不得混为同一输入。

### 4.5 Critic 未包含完整闭环状态

`pen.py:419-421` 的 `critic()` 只取301维当前干净任务帧，不含EMA、负载积分、时间/剩余长度、mask或上下文。物体速度也没有独立传入critic。相同观测可能对应不同未来值，短想象末端bootstrap存在部分可观测性偏差。这不阻止运行，但会限制共享模型研究的解释。

先保持上游critic以建立可对照基线；新增研究分支再将完整控制记忆、笔速度、参考相位/剩余长度、必要物理context加入critic，并为PPO使用同等信息预算。把这一变更作为独立消融，而非与接触奖励同时修改。

### 4.6 上游 holdout 与成功统计不能替代轨迹泛化评估

`buffer.py:7-11,29-35` 的holdout按并行环境编号拆分，全部环境仍可采样相同原始参考。它只评估WM对未训练环境状态/参数的预测，不是保留轨迹或保留视频测试。训练reset使用RSI与adaptive片段采样（`commands.py:573-619`），片段存活或到达reference_end也不等于从首帧完成整条轨迹。

应按原始视频/原始完整轨迹ID分组划分，切片、扰动副本、教师轨迹衍生项保持同split。独立评估从预定初态和首帧开始，禁止RSI与失败后的隐式续段；完整段成功同时要求达到时长、物体目标质量与无掉落/非法接触。每条episode保留起始轨迹、初态、首次失败帧、失败原因与实际完成时长。

## 5. 可独立实现的 adapter / evaluator

建议在研究workspace提供库与CLI，不把视频/接触工具硬塞进上游训练入口：

- **Reference manifest**：原始来源ID与hash、synthetic/robot/egocentric来源标签、时间戳与单位、坐标系、wxyz、关节名、物体尺寸/质量/惯量、接触来源与置信度、腕约束、split及扰动参数。校验时间严格递增、字段有限、四元数有效、关节顺序和单位、接触mask与法向；已生成robot qpos前应有明确重定向算法与输出。
- **Conversion**：对有标定的重建轨迹先统一世界/相机/腕/物体系，再做机器人运动学拟合；由FK生成与机器人qpos一致的link参考。独立保存原视频重建与修正结果，避免覆盖原始证据。固定腕转换前后均保留物体目标，显式记录删除的腕运动。
- **Contact support**：输入物体系contact point/normal、摩擦系数、相位mask、特征长度及参考ID；输出方向support与方向集元数据。单位法向、切线基、normal符号、力矩 `r cross f`、长度归一化均需约定。该表示是作用能力，并非测得的真实接触力或动态可行性证书。
- **Physics evaluator**：输入A0或策略轨迹的仿真日志，分别报告位置、轴角、展开旋转进度、drop/penetration、目标/关节限幅、动作一阶二阶变化、力矩与接触异常。缺失字段应报告unavailable，不能默认通过；没有物理日志的纯几何reference不得输出physics_success。
- **Gradient check**：在真实采样状态对动作作小幅中心差分或投影步，比较世界模型推荐方向在独立仿真中的真实目标变化；报告符号一致率与改善量。评价复用相同初态、物理context和随机流，防止噪声冒充梯度收益。
- **Cost ledger**：把数据获取/重建、教师训练、仿真采样、编译、WM拟合、策略更新、验证、重试分别计时，明确冷/热缓存和GPU资源。共享模型的预训练成本必须计入总成本。

## 6. 第二个 manipulation 仓库可借鉴范围

`supermimic-manipulation-research` 是ARX5机械臂的5个状态输入操作任务，而非灵巧手视频重定向。其OmniReset类别、物理验证reset bank、完整episode evaluator与任务独立的解析reward/状态重建值得借鉴；不能把其README成功率当作本任务证据。

主要参考位置：

- `README.md:103-130`：reset bank与物理状态世界模型的边界；相机主要供展示。
- `src/clab/evaluation/policy.py:18-117`：按独立seed、reset category运行完整episode并保留原始episode记录。
- `src/clab/tasks/*/resets.py`、`collect.py`：将提出初态与物理验证分开。
- `src/clab/tasks/supermimic.py`、`src/clab/observations/state.py`：以预测物理状态重建任务观测，避免在imagination里读取未来真实几何。

建议复用评估设计，不混装其 `mjlab==1.5.3` 环境到Dex的 `mjlab==1.6.0` 环境。

本审阅只确认上述源码结构与风险。模型加载、GPU仿真、训练稳定性、接触增强收益、共享成本下降与真实egocentric数据结果，均需以各自实验日志独立验收。
