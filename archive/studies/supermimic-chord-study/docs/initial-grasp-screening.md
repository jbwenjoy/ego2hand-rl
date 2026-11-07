# 初握筛查与转笔基线

2026-11-01。所有筛查用于训练初态/参考选择，没有使用A1/A2结果筛选，也没有保留视频泛化结论。

## CPU筛查

`scripts/select_initial_grasp.py` 从单轴组10条clip每0.5秒提出一个初态，要求余下至少151帧/3秒。103个初态分别测试零初速与源有限差分初速，共206次0.3秒无辅助hold。初态只初始化一次，之后固定手腕与手指q目标，让物体自由运动。

标准先保存在 `outputs/initial_grasp_scan/substeps_v2/protocol.json`：每5ms检查位置变化≤2cm、有向笔轴变化≤20°、手物/自碰/其他穿透分别≤3mm、关节限位、掉落和非手支撑。所有206条记录保存在同目录 `scan.json`。16次通过，其中源有限差分组9次，零初速组7次。

仅按hold排序的候选clip009_f0250容易保持，但后续运动幅度不足，保留为静态对照。研究主协议随后在9个源有限差分通过者上冻结额外运动门槛：投影旋转进度≥45°、轴最大偏转≥60°、轴路径≥90°、主轴能量比例≥0.60。完整筛选记录在 `outputs/rotating_reference_selection/selection.json`，只有clip007_f0200_source_fd通过。

最终参考为 `outputs/pilot_rotating_reference_v2/reference.npz`，原clip007第200至350帧。其3秒目标投影旋转进度63.94°、轴最大偏转64.33°、轴路径100.85°。投影进度是沿固定参考外轴累计最小笔轴旋转，排除了不可观测的轴向roll，并不是绕轴圈数。原始来源、帧范围、SHA及固定腕任务重设计均保留；未知原视频父ID，继续按整release分组。

## Warp初握转移

`scripts/verify_warp_hold.py` 在实际Warp环境重置到选中源有限差分初态，仅冻结解析command phase，让手保持q0，物体状态不再覆盖。通过每次physics step的真实contact/EFC快照及control结束后的forward端点，覆盖0至0.3秒共61个唯一5ms接触时刻。

选中初态通过：最大手物穿透2.735mm，自碰/其他0；位置最大变化17.278mm，有向轴变化18.725°。无辅助、无reset、无数值censor，手指目标始终等于q0。结果与全部采样见 `outputs/rotating_reference_warp_hold_geomfix/`。

该阈值余量有限；通过短hold仍不等于后续转笔可行。

## 冻结任务协议下的完整3秒A0

`scripts/evaluate_pen_checkpoint.py` 使用同一Warp后端直接执行参考，跟踪失败只锁存而继续记录全部151帧/150个控制interval。输出 `outputs/rotating_reference_warp_a0/`：

| 指标 | A0实测 |
|---|---|
| 任务成功 | false |
| 位置误差均值 / p95 | 73.35 / 144.27mm |
| 有向笔轴误差均值 | 46.17° |
| 参考 / 实际投影旋转进度 | 63.94° / 21.98° |
| 旋转进度终点误差 | 41.95° |
| 首次笔轴方向偏差达到60° | 1.22秒 |
| 穿透、关节限位、掉落、负载筛查 | 均通过 |
| 最大力矩负载比例 / 负载积分 | 0.303 / 0.150 |
| 无辅助、无reset、无censor | 通过 |

该结果把“初握不能保持”和“动态参考跟踪失败”区分开，为后续正式A1/A2提供统一基线。它没有证明世界模型、接触增强或共享预训练已带来改善。

## 接触测量修正与失败尝试

MuJoCo 3.11保留的 `contact.geom1/geom2` 并非当前 `contact.geom[2]` 的别名。`mujoco_warp.get_data_into` 填写后者，旧字段可能未更新。首次Warp hold因旧字段漏计手物接触；修正为读取 `contact.geom` 后重新运行得到上述结果。旧目录 `outputs/rotating_reference_warp_hold/` 的接触分类与初握通过判定无效，原始记录仍保留，不得将其解读为物理失败。

旧clip001 Warp诊断也受旧字段接触计数影响，原始状态轨迹与其跟踪失败事实不受影响；新的正式pilot基线使用修正后的测量。物体对掌/指以外的夹具或mount接触计为非手支撑。

首次启动还因新派生provenance中的project-relative路径失败，未进入GPU物理仿真；重试显式以研究workspace为cwd。失败与重试均有独立cost/log目录，未覆盖原证据。GPU阶段总进程耗时分别记录在 `rotating_reference_warp_hold_geomfix_cost/cost.json` 与 `rotating_reference_warp_a0_cost/cost.json`，不是训练耗时。
