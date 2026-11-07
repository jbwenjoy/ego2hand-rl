# 固定1000次边界组件评价：当前一致性配方未带来拟合收益

2026-11-06。Goal active。上一轮完成两组各1000次更新并按首100健康+ETA暂停，本次恢复后完成最终saved健康5/5、实际3×B152预测、全部152保存复算，以及旧15通道缓存标签的独立评价。此轮没有新训练、Actor或物理执行。

## 固定预算下的主要结果

三角色使用完全相同的152条训练输入、初始norm/尺度/物理context。以下point/mode是从实际捕获FP32残差/logits用F64重算，非原Torch归约逐位重放；冲量/一致性为F64同公式。公共损失相同，method只多λcons=1。不能比较不同定义的训练total来选赢家。

|角色|native52 scaled MSE|mode BCE|z impulse MSE|saved-defect consistency MSE|p vector RMS mm|
|---|---:|---:|---:|---:|---:|
|Initial|0.92352249|0.69314718|0.69726796|4.1648018|6.32854|
|Public control|6.9514827e-05|0.00053837221|1.9013882e-05|0.00013329453|0.0261165|
|Public + consistency|0.0022209452|0.0034936418|0.0027090312|0.010914031|0.0345898|

在ALL上，method/control四项误差比依次为 31.95、6.49、142.48、81.88。在13个重叠cohort中，method更小的组数依次为0/13、0/13、1/13、0/13。两组都比初态拟合更好，但当前固定1000预算不支持加入一致性项的训练集收益。不能据此断言该约束在其他预算或未见轨迹无效，也不能把原因断言为梯度冲突、某个权重或优化器；本轮没有这些因果消融。

独立保存审计逐行以标量congruence/独立pivot和fsum复算全部3×152辅助量，并核13×3统计、39行cohort CSV、456行CSV、原始norm/context/RNG和706份源归档，11/11通过。native Log本身仍是冻结源码及捕获D52的证据，不独立重放该kernel；矩阵Frobenius差与native LogR残差各自报告，不冒SO3几何证书。PRE1000训练日志来自after999权重，本表来自实际AFTER1000前向。

## 旧self15通道存在实际监督缺口

原WM的self15来自finger_collision与palm-subtree传感器，是Actor soft惩罚的输入；新两个HO/nonhand mode不是其替代。本次公共loss没有旧15 BCE。通过原reward JSON、source/raw25映射和完整(path,k,branch)身份，另立saved-only协议连接全部152×15缓存标签，不增加模型前向或物理。

|角色|cached15 mean BCE|Brier|TP/FP/FN/TN|平均预测强度Σsigmoid|平均Pen代理误差|
|---|---:|---:|---|---:|---:|
|Initial|0.379958|0.105796|0/76/228/1976|0.843239|0.013135|
|Public control|0.490468|0.097317|0/0/228/2052|0.405349|0.021893|
|Public + consistency|0.397659|0.090134|19/38/209/2014|0.470114|0.020598|

共228/2280个正标签（10%），A0平均真实强度0、A2为3、ALL为1.5。strict logit>0分类中，对照组成为全负预测；总体accuracy或单一Brier不能概括风险。两final BCE均高于初态，但Brier下降，method的Brier与强度误差MAE又低于control，必须同时保留这些取舍。原Pen代理为-F64(.02)Σsigmoid；误差相对同一缓存binary强度为正表示惩罚低估，final平均偏差加重，不能称完整reward或任务改善。

这些标签是lastsolver≈15ms缓存sensor，state58是post20ms；这沿用原native混合时相合同，非post20几何碰撞truth，也不是HO/人类contact/force-capacity。关系到physical_arrays.branch_self15依赖既有producer及1762项旧saved审阅，本次不读取该NPZ。标签准确度不能升级物理全局自碰或Actor资格。

## 执行与保留证据

最终checkpoint健康5/5。成功native评估仅3次自有CPU加载、1次原生构造、2份raw复制、3次B152前向/456预测、2次辅助头；全程FP32原模型/F64辅助算术，无AD/优化/Actor/MJB/Env/新物理/GPU。原始权重、输入、norm/context、raw RNG及空hook缓存不变。

原v1评估适配器把checkpoint的input_pins错当成协议同名键，实际协议为inputs，在3load/1ctor/2copy但0FW时失败。Root、owner、静审都漏过该字段问题；失败源/结果和缺陷说明原样保留。v2只修映射与独立routes，并用实际冻结协议keyset校验Toy后执行，没有改训练结果、样本、指标、容差或既有源。费用记入ledger，不能把这次失败当模型预测或物理失败。

## 后继研究

保留本轮负面结果，不自动增加训练次数或事后调λ。优先用原Runner已经定义的native15 BCE reduction及canonical contact权重形成新的共同公共监督实验，两臂均保留旧接触通道监督，唯一方法差异仍为cons0/1；新实验须单独recipe/预算/检查点健康。随后评价响应/H8以及GPU精度安装，按一次共同collect/replay/Actor/Critic/RNG起点开展新namespace任务对照，不独立重跑各臂WM预热、不将own组件冒充105恢复，旧WM配套norm不能换成fresh norm混装。

原参考几何FAIL、原A0及三臂task FAIL、有限响应确认FAIL、旧H8取舍/warmup差异、whole-hidden未知仍保留。当前只是单一缓存谱系内的离线组件研究，真实ego N=0，未完成多轨迹共享/heldout及硬件子版本合同。完整目标尚未完成。

- [可复查研究交接](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_components_final1000_research_update_v1.json)
- [预测独审11](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_component_final_prediction_saved_independent_v1/report.json)
- [缓存15独审](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_cached_self15_saved_independent_v1/report.json)
- [39组CSV](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_component_final_prediction_description_v1/all39_model_cohort_summary.csv)
- [完整图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_component_final_metrics_figure_v1/fixed_after1000_component_tradeoffs.png)
- [匹配native任务源计划](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_components_matched_native_task_integration_source_plan_v1.md)
