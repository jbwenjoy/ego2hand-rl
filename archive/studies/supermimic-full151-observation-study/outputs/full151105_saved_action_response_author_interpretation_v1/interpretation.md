有限动作响应：作者结论与阶段边界

这批数据同时显示较大的常量预测偏差与不能忽略的动作响应失配。它们不能互相替代。全部8个snapshot、128个端点和64个actual-coordinate pairs都保留；作者描述11/11，独立数学权威为Contact的22/22。没有旧stable9筛选、noise floor、精度PASS阈值或方法赢家。

全体分量RMS（每个block各自单位）：

| block | baseline bias | physical response | model response | response error | error/physical RMS | secant error/physical RMS |
|---|---:|---:|---:|---:|---:|---:|
| q (mrad) | 23.374686 | 0.0990912 | 0.0933284 | 0.0901081 | 0.90935 | 0.92036 |
| qd (rad/s) | 1.133740 | 0.00540907 | 0.00407395 | 0.00443549 | 0.82001 | 0.82379 |
| p (mm) | 3.653596 | 0.00914856 | 0.00402107 | 0.00917243 | 1.00261 | 1.00052 |
| R (raw9, dimensionless) | 0.0406809 | 0.000338065 | 0.000104332 | 0.000343622 | 1.01644 | 0.97181 |
| v (m/s) | 0.106545 | 0.000545130 | 0.000204875 | 0.000533207 | 0.97813 | 0.97123 |
| omega (rad/s) | 3.097486 | 0.0279249 | 0.00574739 | 0.0281383 | 1.00764 | 0.98629 |

这些ratio只是全量描述，零physical RMS为None，绝不补epsilon。p的响应误差约9.17微米虽然绝对值小，却与约9.15微米的physical响应同量级；不能因相对3mm阈值很小就称局部控制敏感度准确。相反，也不能把本表当真实Jacobian误差证书。当前都是两requested-eps、实际控制宽度分母、CPUshadow及有限可见状态的弦线。

常量偏差不是唯一问题。在当前F64统计定义内，对同一snapshot的所有预测减去同一个、与u无关的常量，会在端点差和secant分子中消掉，因此不能改变这里的响应差异。这不是重新校准模型或训练后的实测结论，也没有把统计baseline均值当新state运行。

全部分组保存在all8_by6_blocks.csv（48行）和原完整分解数组。组间差异大：p的A0 k0/k49 response-error/physical约0.55/0.51，k99/k149约2.77/2.99；A2 k0约1.03。A2 k49、99、149的p/R physical centered responses在保存FP32编码中全部为零，ratio为None；模型在这些组仍有非零object响应。v/omega的一些physical值极小但非零，导致很大的描述比值；这些值不可变成可靠的accuracy倍数或稳定signal资格。

机械zero-control核验保留了全部128行：6个端点的完整20维actualu与原u0逐位相同，位于A2 k49/99/149、声明joint_index4的+方向、两种eps，原branch IDs为101/109/120/128/139/147。它们的模型与physical endpoint对全部3baseline的58维FP32编码也逐位相同，中心响应全0。这里只证明这6个编码重复的结果，不能证明真实noise为0或全hidden状态恢复。三个baseline在本批的perfeature range全0同样不是未来噪声上界。

晚期A2三个snapshot共有48端点，其中6个control未变，另外42个actualdu非零；physical p/R编码平坦并不只由这6个collapsed端点组成。只限本次四个被探测control坐标和两种有限步长，不得推广为所有控制方向的真实object Jacobian为零。12clipped与12asymmetric pairs仍全部保留、64分母均定义；它们反映裁剪后的非对称步长/弦线，不是异步或新增延迟。原四substep控制同步合同未变。

下一最小假设/行动建议（仅建议，本轮不新建协议或实现）：

候选假设是“固定WM在这些状态中仍给出动作到object的作用，而对应的观测有限响应很弱或在编码中平坦；这种耦合是否与现有接触regime不一致”。在决定任何新loss、H8或训练之前，最小可检验行动是对既有全8/152保存数据做一次完整接触regime核对：机械查看同相位的native生成/约束接触、HO力与非手/环境类别、object可见运动，所有snapshot与阴性同时保留。它可以检验这三个晚期A2状态的耦合条件是否支持该假设；不允许只凭本表断言无HO、已经落地或某类接触导致了误差。缺少可核字段则UNKNOWN，不能补标签。这是建议，不是自动授权；本轮没有新prereg、研究代码、模型、AD或物理。

该顺序比继续只消常量bias、只调两个proxy漏检点或立即扩大H8更能保持因果问题可解释。它也不保证随后应训练：Root应先根据阶段证据决定。CPU ampFalse shadow、one-step20ms、known-rough reference、单source/seed、prepolicy100已分歧、wholehidden未知与旧5/9 taskscalar资格限制全部继承。新state-coordinate弦线不替换旧nominalFD/有效方向AD、不能拿新比值与旧5/9混成总成绩。

本交付没有新增模型加载、forward、objective、score/goal、MJB/FK/query、AD、GPU、Env、物理或训练。原始数据、源码、独立报告与失败范围均未改。
