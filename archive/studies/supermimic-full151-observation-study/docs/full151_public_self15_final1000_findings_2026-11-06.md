# 公共native15固定1000评价：旧通道拟合恢复，控制响应仍不足

2026-11-06。Goal active。本次恢复后完成新final1000健康5/5、三模型实际3×B152评价和独立12/12；再从同一批保存预测计算全部64对控制扰动响应并独审10/10。本轮没有新训练、Actor、AD或物理采样。

## 同一新公共监督下比较cons0/1

三角色同初始规范、原152输入/顺序/物理context与原norm。以下point/mode是捕获FP32残差/logits的F64重算；FP32 native_pair/八项scalar单独保留。共同公共项逐行 `mean152(point_row+.1*sum15 BCE)` FP32，再+modeFP32+impF64；只有cons0/1不同，不能按异权total选赢家。

|角色|native52 scaled MSE|mode BCE|z impulse MSE|saved-defect cons MSE|p vector RMS mm|
|---|---:|---:|---:|---:|---:|
|Initial|0.92352249|0.69314718|0.69726796|4.1648018|6.32854|
|Public incl. native15|2.0821209e-05|0.00054039558|1.415242e-05|5.6793634e-05|0.0136542|
|Public + consistency|0.00099471979|0.0044056449|0.00087737825|0.0041555516|0.025472|

method/control四项误差比为 47.77、8.15、61.99、73.17。固定1000预算下，当前一致性项没有带来ALL训练集主要拟合收益；两者均显著优于未更新initial。这不证明其他预算/轨迹上无效，也不解释原因。没有以结果调λ或追加更新。

## 缓存15标签不再漏检，但只证明训练内预测

|角色|cached15 mean BCE|Brier|TP/FP/FN/TN|平均Σsigmoid|平均Pen代理误差|
|---|---:|---:|---|---:|---:|
|Initial|0.37995813|0.10579557|0/76/228/1976|0.84323867|0.013135227|
|Public incl. native15|3.7569258e-05|3.558802e-09|228/0/0/2052|1.5001641|-3.2827421e-06|
|Public + consistency|0.00035182022|4.6138945e-07|228/0/0/2052|1.5007826|-1.5652387e-05|

同一152×15中228 positive、2052 negative；两trained role对228 positive全部检出且无误报。对比此前无self15公共监督的全负control及method漏报，新的共同监督实验恢复了这些训练条目上的预测。两者软惩罚偏差接近0；负代理误差表示相对缓存binary强度的轻微过度惩罚，不是完整reward改善。旧无self15实验的BCE/Brier取舍和负面结果保留，跨stage公共目标不同，FP32状态归约也作了逐行澄清，不能把变化拆成严格单因果效应。

标签仍是约15ms缓存finger sensor，state target为POST20。A0全负、A2平均3positive及19个依赖分支，使这些不是2280个独立观测；不得推广成POST20几何、HO/人类接触、独立轨迹或任务成功率。

## 点拟合不等于扰动响应拟合

旧相位0/49/99/149的8snapshot，每个4方向×2epsilon，机械连接64对minus/plus。全部原始响应/误差、actual控制差、actual状态差、selected-slot/nominal两分母和13pair-cohort都保留，没有stable/误差筛样。actual denominator是选中control坐标du_j，原程序未把它当全20维norm；独审另确认本批offaxis=0，故该批数值才恰与向量norm绝对值一致。

|角色|p响应误差 component RMS mm|rawR9响应误差 component RMS|omega响应误差 rad/s|omega actual-secants误差 (rad/s)/rad|
|---|---:|---:|---:|---:|
|Initial|0.0167156|0.0003436915|0.02728078|8.374701|
|Public incl. native15|0.0153216|0.0003434369|0.0282004|8.495279|
|Public + consistency|0.0158924|0.0003232779|0.02758252|8.37328|

对应物理响应本身的component RMS为p=0.0166978mm、rawR9=0.000321183、omega=0.0263597rad/s；control的p响应误差约为信号的0.918倍、omega约1.070倍。method在rawR9/omega有部分相对control改善、在q/qd/p/v没有统一改善；不能由小的绝对点误差推断可用的Actor动作梯度。

64对actual denominator均非零，原undefined机制仍保留；实际du_j范围[0.00100004673,0.004000008106]rad。该批64对PRE58逐位相同、offaxis控制差为0，8组baseline的三次PRE58、actualu与真实endpoint在保存精度下无离散度；这仅是本次可见条件与3次重复的描述，不能证明whole-hidden一致、噪声上界或连续physical Jacobian。R9保持raw dimensionless矩阵，未作Log或投影。

## 数据与执行边界

新AFTER1000评价仅3个own CPU加载、1native constructor、2raw copies、3B152 FW/456预测、2head FW，原norm/context/RNG等守门通过；保存独审12/12核727源、全部152/39组/456表及旧物理项独立scalar-pivot、raw25标签映射。64对分析仅4NPZ读取、零模型；独审10/10再核4original+2saved NPZ，全部raw算术精确、fsum reduction最大差3.55e-15。

64对初次v1因Root传相对protocol路径，在4NPZ读取/统计保存后被最终absolute-pin守门拒绝，未形成成功report。失败、部分数据和诊断原样保留；v2只用新协议/目录和绝对CLI路径，冻结代码、输入、公式、门和mask全部不变，重新一次完成，无自动retry。额外4读取计入费用。

当前old152与另一套fresh152的相位24/74/124/148不同，fresh6stable/25weak/1unstable标签不贴到本次64对。fresh152模型x是8snapshot重复的输入设计，未被独立资格化为逐分支actualPRE58；没有独立fresh J/K/r_saved包，不能套old冲量truth或cons。

下一步应在明确fresh可见输入/恢复限制后做独立响应/uVJP诊断，并评价已资格native donor的H8 teacher-control多步预测；不称policy imagination或trueJac。完整组合收益还需同预算的原native point+self15-only训练对照，区分J/K/mode共同监督与cons贡献，以及共同真实warmup/Actor/Critic/replay/norm/RNG起点的新namespace任务实验。当前public control已经含J/K/mode，不能当原SuperMimic-only对照。

原151参考几何/任务FAIL、旧finite-response确认FAIL、旧H8/warmup取舍、wholehidden未知、force-capacity/接触可行性、多轨迹、ego0及硬件合同仍未解决。完整目标尚未完成，Goal保持active；本轮没有新训练正在等待。

- [机器交接](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_final_research_update_v1.json)
- [预测独审12](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_final_prediction_saved_independent_v1/report.json)
- [64对独审10](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_endpoint_response_saved_independent_v1/report.json)
- [全图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_public_self15_final_metrics_figure_v1/fixed_after1000_public_self15_tradeoffs.png)
- [后继响应/梯度源计划](D:/workspace/supermimic-full151-observation-study/docs/full151_public_self15_action_response_gradient_source_plan_v1.md)
