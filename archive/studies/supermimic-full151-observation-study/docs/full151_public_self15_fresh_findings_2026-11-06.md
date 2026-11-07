# 新 public-self15：另一组相位的状态、接触与动作敏感度

2026-11-06。三份固定模型的实际 CPU 评价与保存结果独立复算完成（11/11）。本次没有训练或新物理采样。点预测与接触拟合有所改善，动作敏感度没有一致改善；当前结果不足以支持把新 WM 直接安装到 Actor 训练。

模型依次为原 A2 派生 WM（本轮校准前）、public native52+self15 加 mode/J/K 辅助监督的对照（1000次更新）、相同监督再加一致性项的模型（1000次更新）。对照已经包含辅助监督，不能把它称为纯 native52+self15 trained baseline。

数据为已解封的同谱系 A0/A2 donor、相位24/74/124/148，8个快照各19分支，共152端点、32方向组及64个epsilon比较。输入是 snapshot8 重复19次的共同可见状态设计；没有逐分支实际 PRE58 的直接捕获或资格，不能称 fully matched physical Jacobian 验证。缓存接触标签来自最后 solver 约15ms，状态目标来自 POST20；接触预测准确性不能解释为 POST20 几何或 HO 接触准确性。

|完整人口指标（越低越好）|原 WM|public+aux 对照|再加 consistency|
|---|---:|---:|---:|
|native52 点均方，FP32 row-mean|0.373645|0.099356|0.079481|
|common-pre 归一化有限响应均方，F64，全64|4.997921|4.673220|4.807245|
|物体位置 vector RMS，mm，全152|4.13256|0.73743|0.70629|
|物体角速度 vector RMS，rad/s，全152|2.77937|1.69182|1.97399|
|缓存15 BCE，F64每元素，全152|0.445193|0.228659|0.193842|
|object-score 动作投影比较 RMSE，score/rad，全64实际坐标|0.389315|0.457280|0.496000|
|6个stable方向，epsilon=.002的比较 RMSE|0.642711|0.626655|0.685963|
|同6个stable方向，epsilon=.001的比较 RMSE|0.649444|0.632729|0.692131|

完整缓存混淆 TP/FP/FN/TN 分别为56/96/246/1882、227/20/75/1958、209/19/93/1959；全部有302个positive、1978个negative。consistency模型虽然总体 BCE 更低，positive recall 仍低于公共对照（209/302 对227/302）。逐feature及所有13个cohort均保留，空分母recall/precision输出null。

物理分类按冻结规则保持6 stable、25 weak、1 unstable，全部64行不筛选。每个stable方向的两个epsilon是依赖重复，并非12个独立方向。三模型在6个stable方向的每个epsilon均6/6同号，这不足以证明幅值准确或任务收益。全64实际坐标误差在两份新模型上均高于原模型；consistency相对公共对照的点误差更小，但有限响应、角速度和这里的梯度比较更差，没有统一优胜模型。

主task score只包含原位置2+shaft端点8，使用同一个goal_k与POST20相位。三次 `autograd.grad(score.sum(),u)` 不包含self15或其他reward。有限差分同时保留actual coordinate与legacy requested-epsilon两种口径、真实F32控制端点先转F64再相减；没有新增准确性门、阈值调整或checkpoint选择。

实际预算为3份own checkpoint读取、1次native ctor、2次装hook前raw copies、6次前向（456个端点与24个中心预测）、4次head前向、3次uVJP。独审只复算保存数据，0 Torch/model/native Log/AD/physics；原native Log52及AD值保持captured/source-backed，不能冒独立重放或true physical derivative。FP32均值按预先固定的条件roundoff envelope核验，没有跨kernel bit-replay声明。

后续必须完成真实donor PRE58与实际teacher controls的H8评价；仍需同预算 native52+self15-only trained baseline，才能分离整个mode/J/K/consistency组合的作用。再往任务环节推进前，要核对GPU精度、共同collect/replay/Actor/Critic/RNG起点及normalizer安装契约。原参考与任务FAIL、旧有限响应确认FAIL、force-capacity/hidden-state未知、多轨迹/ego数据缺口均保留。

- [实际评测](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_fresh_gradient_CPU_v1/report.json)
- [独立保存结果复核11/11](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_fresh_gradient_saved_independent_v1/report.json)
- [完整图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_fresh_figure_v1/fixed_public15_fresh_diagnostics.png)
