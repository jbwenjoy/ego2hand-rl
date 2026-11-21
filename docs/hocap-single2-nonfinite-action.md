# R2 单序列策略的首帧非有限动作

正式单序列第 2 路对象 G16_4 的第 639,744 步检查点，在冻结评测器中从原始源帧 445 重置。原生关节、刚体位置和速度均有限，输入观察 276/276 项有限，确定性策略的输出 26/26 项为 NaN。评测器在发送控制前抛出 `ValueError: Nonfinite actor output`。可报告的运行长度为 0/275 个控制区间、0 个物理子步。失败报告保存原评测器哈希、完整异常栈、检查点与 Parquet 的哈希、实际输入、输出和无效维度掩码。[边界记录](../runs/wuji_r2_single2_action_boundary_v1/failure_boundary.json) · [失败报告](../runs/wuji_r2_single2_action_boundary_v1/report.json)

在 CPU 上用发布版学习器的 PyTree 模板逐张量核查六组留存的策略、学习器及回放文件。到 456,960 步均为有限数值。548,352 步首次观察到已损坏的保存文件。最终策略中，训练参数全部有限，10 组 BatchNorm 运行均值或方差共有 5,276 个 NaN。学习器的 critic、target critic 运行统计以及温度参数和温度优化器状态也含有 NaN。终点回放环的 131,072 行中，97,728 行的动作含 NaN。保存文件的完整性和数值有效性需分别核对。[逐文件与逐张量审计](../runs/wuji_r2_single2_finite_audit_v2/report.json)

首次出现坏数值的已存回放中，最早非有限后继观察早于最早非有限动作 9,443 个已提交环形存储行。训练日志在 537,216 步首次把 actor 与 critic 损失记为 null，在 541,824 步开始出现 null 温度、熵和动作幅值。日志将非有限标量写成 JSON null。这些是最早可核实的记录边界，尚不足以确定最初出错的原始物理子步或训练算子。[日志异常](../runs/wuji_r2_single2_finite_audit_v2/metric_anomalies.json) · [环形存储的时间次序](../runs/wuji_r2_single2_finite_audit_v2/replay_chronology.json)

对同一个首帧输入，只在内存中交叉组合 456,960/639,744 步的 actor 权重及运行统计，做四次确定性前向。两套权重配健康统计时均产生 26 个有限输出，配污染后的统计时均产生 26 个 NaN。这个反事实定位了推理失败的直接原因。没有用拼接的状态产生轨迹、替换正式策略或计入成功。[逐条件输出](../runs/wuji_r2_single2_finite_audit_v2/bn_counterfactual.json)
最早可从坏回放中定位的提交行，当前观察和 26 个动作均有限，动作范围为 −0.829 至 +0.798。该行标记为终止，后继观察有 78 个非有限维度，后继 critic 上下文中物体速度的六个分量全部为 NaN。v0.4.0 的 actor 更新先拼接当前与后继观察，再计算 BatchNorm 统计。Adam 的有限梯度保护覆盖权重及相应优化器槽位，随后仍写入本次运行统计。critic、target critic 也会写入新统计。这解释了为什么终止样本中的非有限后继值仍有传播途径，也与“权重保持有限、统计被污染”的实测相符。[首个保留的非有限后继行](../runs/wuji_r2_single2_finite_audit_v2/earliest_saved_bad_successor.json) · [官方 v0.4.0 学习器](https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/robotic_grounding/flash_chord/src/flash_chord/training/flash_sac/learner.py)
