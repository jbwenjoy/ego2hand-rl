# 新 public-self15：固定 donor 控制的八步预测

2026-11-06。实际评价24次native前向、6864个预测完成；独立保存结果审查v2为24/24 PASS。两份校准模型相对原WM改善了这里的多步误差；一致性项相对公共辅助监督对照有取舍，不能据此授予Actor或物理任务收益。

三角色仍为原A2派生WM（本轮校准前）、native52+self15加mode/J/K辅助监督的1000次更新对照、相同监督再加consistency的1000次更新模型。严格保持同一原own initial、frozen norm/host/context；4个训练fit digest与本次donor输入digest分开，没有假105恢复或新的训练。

H1已资格的native PRE/POST58与真实绝对控制构成A0/A2各150个interval。重新逐字节核验298个完整58维相邻连接及300组四子步控制保持，再按每path起点0..142形成286个重叠窗。每h同时把上一预测写入x及dynamics[:58]，其余为固定xi46+zero174，保留完整278调用轨迹。使用真实外部teacher controls，**没有teacher state forcing**，也没有在预测观测上重新调用Actor。

新的label-only包把全452原始记录按身份连接后取恰300个donor（A2 global gid从226开始，不能取前300行或直接把gid当300数组下标）。独审6/6复核全部4500个raw25→native15 FP32标签值、时刻和终末身份；再按窗口索引展开到2288个依赖引用。标签约15ms缓存，状态是POST20。h>1不同模型到达不同预测状态，对缓存参考的误差包含rollout漂移，并非同输入head遗忘或预测状态的几何接触真值。

|h=8、ALL286完整人口指标|原WM|public+aux 对照|再加consistency|
|---|---:|---:|---:|
|q component RMSE，rad|0.092374|0.049093|0.048358|
|qd component RMSE，rad/s|1.991521|0.756079|0.799311|
|物体p vector RMS，mm|64.5373|35.0171|31.4633|
|物体raw R9 Frobenius RMS，无量纲|0.953228|0.657676|0.617570|
|物体v vector RMS，m/s|0.544160|0.308861|0.347326|
|物体omega vector RMS，rad/s|10.63520|6.78282|6.55788|
|缓存15 BCE，F64每元素|0.513770|0.227752|0.197641|
|缓存positive recall|0.076923|0.742964|0.748593|
|缓存positive precision|0.132686|0.878049|0.841772|

一致性模型在h8位置误差比公共对照低约10.15%，但qd、v误差及缓存precision更差。原参考和任务FAIL不因此改级。每h目标相位人口会变化，BCE曲线下降不能解释为预测步数越长越准确。全部432个原单位块统计、72个接触cohort表、完整逐feature/allh/path摘要和逐元素缓存指标均保存，空分母输出null。R9差不是旋转角，不做混合单位总分或新增accuracy门。

实际预算为3 own loads、1 native ctor、2次装hook前raw copies、24个B286 native FW、16个head FW；worker约11.23秒。没有AD、参数更新、Actor、physics或GPU。保存前后norm/RNG原始值、实际递归输入和参数存储清单，独审在无Torch/模型/native Log的条件下复核数据和算术；权重/host/cache等摘要仍是source-backed关系，没有冒独立模型重放。

独审v1保留一次失败：把仅支持FP32的旧摘要helper误用于FP64 norm.count。v2只将normalizer摘要改为已冻结的dtype-aware原始字节helper并增加FP32/FP64闭式检查；没有修改模型、数据或容差，没有重跑native前向。v1已有中间统计不作PASS依据，最终以完整v2报告为准。

这仍是单谱系两条已见donor的条件描述，286窗不是286条独立轨迹。共同可见fresh端点/梯度实验与本次native donor输入资格不同，不能混合人口；fresh仍缺逐分支PRE58资格，也没有fresh J/K/r_saved目标包。本次没有H8 mode或冲量truth，不算其监督/consistency损失，不能声称HO力、force capacity、whole hidden、physical Jacobian、ego泛化或任务成功。

下一步补同预算native52+self15-only trained baseline，区分整个mode/J/K/consistency组合的作用；其最小源码提案已经形成，尚未建立执行协议或启动训练。之后还需GPU精度与固定norm安装、共同collect/replay/Actor/Critic/RNG起点任务比较，以及多轨迹、真实ego数据和硬件观测契约。旧失败与未知项继续保留。

- [实际H8报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_H8_donor_CPU_v1/report.json)
- [独立复核24/24及逐feature输出](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_H8_donor_saved_independent_v2/report.json)
- [审查程序修正记录](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_H8_auditor_dtype_correction_v2.json)
- [完整八步曲线](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_H8_figure_v1/fixed_public15_H8_diagnostics.png)
- [另一组相位的端点与动作敏感度结果](D:/workspace/supermimic-full151-observation-study/docs/full151_public_self15_fresh_findings_2026-11-06.md)
- [纯native监督同预算对照提案](D:/workspace/supermimic-full151-observation-study/docs/public_native_only_matched_baseline_source_proposal_v1.md)
