# Phone 动量约束接口 v1

`momentum_contract.py` 已提供可微的纯 Torch 接口，没有神经网络参数。输入为 phone state58、名义质量与完整 BODY 惯量、world 重力、实际时长及 contact impulse6。

唯一支持的冲量坐标是**冻结 source BODY 轴、力矩绕逐时刻 moving COM**：`source_BODY_axes_moving_COM_impulse6`。其两个三维块均用同一个 source R 转到 world。不能替换为随物体转动的 BODY 分量积分，或绕固定 source COM 的力矩积分。

- `momentum_residual_world` 分别返回线动量 `[N·s]` 和角动量 `[N·m·s]` 残差，不计算混单位总范数。
- `contact_impulse_required_source` 从两个端点算所需冲量，是代数目标，不是测得的接触标签。
- `velocities_from_source_impulse` 在另给 target R 时算条件速度闭合，不预测 p/R，也不证明该 R 可达。碰撞冲量包括所有边界接触；指定支持点的合法性另设分组/损失。

实际修订验证见 `completed_run2/report.json`，14/14，通过原 TRAIN 中684个完整过渡的保存复算和独立解析坐标、惯量、世界旋转、ballistic、冲量输入 Jacobian 夹具。Torch 残差与原 world 导出最大差约 `1.67e-16 N·s / 3.90e-18 N·m·s`。这些是名义仿真动量一致性与接口算术证据，不是接触因果证书、原生动作梯度、真实参数或新模型预测精度。

首次运行已完成数值计算，但因 NumPy bool 的 JSON 序列化失败。`serialization_correction_v1.json` 保留首版源、失败 terminal、部分报告和数值文件；修订仅转换报告标量类型，公式/样本/阈值不变。两次执行累计2次纯输入冲量 Jacobian，0 NN构造/参数更新/原生物理。没有启动新学习训练。

验证读取原 TRAIN state/target 和已经公布的 all960 导出标签；后者包含已曝光 TEST 标签，明确不再称其未见。没有读取原 VAL/TEST state NPZ，也没有拟合损失权重、选择模型或重新划分数据。原36个 TRAIN partial 仍按既有 eligibility 保留在上游，不补造20ms目标。

后续新模型可将此接口作为独立动量项或条件速度参数化；仍须增加接触几何、全H因果路径、接触时序/力约束、姿态与位移过渡、隐藏状态假设，并与真实可执行任务及 Actor/CHORD 对接。当前接口不代替这些要求。
