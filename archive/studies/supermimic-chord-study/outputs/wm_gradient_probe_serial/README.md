# WM一步动作方向检验

2026-11-02。有效实验是本目录的单world串行probe。使用原A1第2000轮世界模型和其固定物理context，输入为state58与20维绝对执行目标，不经过raw policy action或EMA优化。

预先固定8个A0时刻（0、0.4、0.8、1.2、1.6、2.0、2.4、2.8秒），各8个seeded单位方向，epsilon分别0.002与0.001rad（20维目标的L2长度）。目标为下一参考帧的原生物体项：位置权重2、笔轴两端点权重8，省略统一dt因子。它不是完整任务回报或多步policy梯度。

所有296个case在同一个Warp world顺序运行。每次恢复319维 `mjSTATE_INTEGRATION`，包括warmstart、ctrl、mocap、time等，再执行4个5ms物理步。每个起点先重复baseline三次，恢复输入误差必须为0，并保持原baseline replay门槛。实际物理结果全部来自Warp，没有用native MuJoCo forward/step当作真值。

恢复与baseline均通过：q、物体位置/旋转元素误差0；关节/物体角速度最大误差约2.38e-7；baseline物体奖励误差0。物理字段采用与训练相同的nominal→运行时actual field复制顺序，固定context逐字段验证。

64个方向中，35个未超过事先规定的可测信号门槛；29个非平坦，24个在两个epsilon下符号一致、斜率差异不超过50%。这24个方向仅来自4个有效非平坦状态，不能把64个方向当作64个独立状态。

| 指标 | 实测 |
|---|---|
| 稳定非平坦方向的符号一致率 | 75.0%，按状态聚类bootstrap 95%区间63.6–84.2% |
| 稳定非平坦方向Pearson相关 | 0.598，聚类区间0.484–0.907 |
| 全部非平坦方向符号一致率 | 69.0% |
| WM建议0.002rad小步的实际目标平均增量 | 4.23e-4，状态bootstrap区间1.10e-4–8.26e-4 |
| 同小步相对8个随机正向平均增量的优势 | 4.40e-4，区间1.05e-4–8.49e-4 |
| 超过信号floor的建议步状态数 | 两个epsilon均3 / 8 |

所有方向均在hard target界内，没有因内部ctrl clipping变成相同输入的方向对，baseline没有关节达到99%力矩饱和。不可测平坦区域没有被包装成方向匹配成功。

这提供该检查点在少量A0状态上的局部、一步、绝对target方向部分可信的证据。它不证明多步想象梯度准确、接触目标梯度可靠、policy更新长期改善，或跨轨迹摊销收益。

`plan.json`是执行前规则；`probe.npz`包含完整integration snapshots、state58、targets、方向、两尺度物理差分、梯度建议步、bounds/force诊断；`branch_index.json`解释case顺序；`baseline_replay_channels.json`拆分各物理通道误差；`summary.json`保存全部统计。

前一次288-world批量版本在1.2秒接触附近的baseline复现未过门槛，尽管保存状态读回完全相同。因此该版本72.7%的表面符号一致率不可作为研究结果。原失败目录与90秒冷编译超时attempt均保留，未放宽门槛来使结果通过。

当前暖缓存进程17.82秒（probe段14.77秒）；此前新CCD专门化的冷编译与失败重试另行记账，不包含在这个暖缓存数值中。对应成本文件是同级 `wm_gradient_probe_serial_cost/cost.json`。
