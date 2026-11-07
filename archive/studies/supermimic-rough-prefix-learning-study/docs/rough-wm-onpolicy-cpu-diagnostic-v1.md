# 新粗参考 WM 一步 CPU 诊断：预声明与下一步顺序

机器协议为 [rough-wm-onpolicy-cpu-diagnostic-v1.json](D:/workspace/supermimic-rough-prefix-learning-study/protocols/rough-wm-onpolicy-cpu-diagnostic-v1.json)，SHA256 `18ef4ad5b39c8ea4ef178665c78d44e9cb3f34f59a56ca2917b9403defd415f7`。这是计算前冻结的分析规则和输入身份；目前只完成协议生成，没有运行 WM 数值诊断。实现 loader、codec、运行器的最终源码与 CPU 证明应由独立 runplan 绑定，不能借补源码 pin 改变样本或指标。

本次读取三个实际 final2000、format104 的 A1L/P_ONLY/A2 WM，在全部 A0/A1L/P_ONLY/A2 已保存轨迹上各预测 15 个 20ms 转移，共 180 个预测。每次起点都是真实保存状态，不是递推预测。四条路径、全部相位和负结果均保留。它们来自同一个已见源窗口 `[200,216)`、同一个训练种子 0，不是 60 个独立演示，也不是跨轨迹泛化检验。

`x[k]` 来自 raw state `4k`，真值 `x[k+1]` 来自 `4k+4`。输入 `u[k]` 用控制事件中的实际执行目标，并逐位核对 state `4k+1..4k+4`；state `4k` 的 ctrl 是上一段目标，不能错接。新状态 codec 必须按 [SimReader.state](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/sim.py:95) 重建 `q20/qd20/p3/R9/v3/ω3`；固定掌系来自机器人根体的真实模型拓扑，不能把任意 mocap pose 当作掌系。ω 的 native freejoint body 坐标先乘物体旋转再转掌系。

比较 identity 和 constant-velocity 两个基线。CV 的旋转为 `Exp(dt[ω_palm]×) R` 左乘；q、p 只按现有速度前推，qd/v/ω 保持，不加重力、不做关节或状态投影。所有模型使用相同 FP32 编码、actual u、实际 xi46 和零 disturbance174；task/critic context368 不可代替物理上下文。

分别报告 q、qd、物体位置、SO(3) 旋转、线速度、角速度的物理单位误差及全部逐相位数据；另保留实际和预测关节极值、1mrad 越界掩码。不给混单位总分，也不预设“够准确”的研究门槛。`error_floor` 是 native loss 的归一化尺度下界，不是 3mrad 的分辨率或死区，更不能据它直接解释越界原因。

物体奖励只复算 [PenMDP.rewards](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:143) 的两项：位置权重 2、尺度 5cm，以及半径 10cm 的两端点平均距离、权重 8、尺度 4cm。主列使用当前 phase k，匹配 native imagination 的 reference memory；另列 k+1，表达下一参考时刻的跟踪。保留未乘 dt 的项和总和，需要时单独列乘 `.02` 的结果。它不包含 torque、imitation、action-rate、selfcollision 或 A2 几何项，不能称为完整 native reward。

全部 180 个目标梯度检查有限性及范数。内部中心差分只用预先固定的 phase 0/7/14、四条路径、三个模型、固定种子 20261104 的四个方向和 eps `.002/.001` rad。越 C2 内域的 FD pair 留原因并跳过，不能夹回域内或换方向。这仅比较同一个冻结 CPU 网络的 autograd 与数值差分，不能获得真实动力学动作方向、物理 Jacobian 或策略改善资格。

native15 维 head 预测的是手指自碰，不是手物接触或力。当前 raw contacts 缺少原 sensor 所依赖的 ContactType/efc/sensordata；候选接触或正法向力不等价于原 native CONSTRAINT sensor。原标签还位于末次 solver 的积分前约 15ms，相对 x1 的 20ms 状态有相位差。没有独立的精确标签重构证明就只保存 logits/probability，BCE/F1/accuracy 标 `UNKNOWN`，不能拿 HO 或独立几何意图替代。

## 下一步优先级及真实缺口

先执行这个有界 CPU 诊断，再推进 full151 的粗观测参考路径，最后接 shared 多轨迹训练。CPU 诊断无需新采样，可及时发现当前模型在已访问分布上的状态或奖励误差；即使结果差也应保留并进入有明确假设的下一步，而非反复调诊断到通过。它无法替代下一阶段独立同 Warp 后端、完整 solver/controller snapshot 的因果动作方向试验。

旧 [FrozenPenWorldModel](D:/workspace/supermimic-chord-study/src/chord_study/world_model.py:21) 明确只认 format3；旧 8-state serial 梯度结果来自旧 151 帧控制模式，不能给当前 format104、baseline_u15、非零解析初速和 native cadence 自动背书。原几何 proxy 的无接触伪 presence 与跨后端梯度失败也只说明旧代理的边界，不能反向断言新 WM 已失败。本轮新一步 CPU 诊断正好填补这个便宜、直接的接口与精度缺口。

当前全长工程不是更换 NPZ 就能完成：[controller](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/controller.py:92) 固定 baseline 15×20，imagined memory 要求 16 knots；[runtime](D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/runtime_v2.py:32) 的 active/end/reset/replay 也专门绑定 15 intervals。full151 应新增整体连续目标/控制/速度、初始化、终端与签名合同，不能把合格 prefix q0 拼入旧 q1..150，不能把本轮 .3s 成功放大为整段资格。粗参考仍可保留 physical_reference_qualified=false 来测试残差学习；并不要求其零残差开环先完美。

shared 数据阶段已有真实进展，但不能绕过剩余接口：[registry READY](D:/workspace/supermimic-shared-reference-study/outputs/shared_registry_cpu_v1/READY.json:66) 记录所有 10 条 measured init 为 null、独立接触目标缺失、native training integration pending；[require_training_ready](D:/workspace/supermimic-shared-reference-study/src/shared_reference/reference_registry.py:107) 仍主动拒绝训练，[target gather](D:/workspace/supermimic-shared-reference-study/src/shared_reference/target_registry.py:64) 也拒绝缺失 bundle。10×151 的 geometry→native ABI 批处理是 CPU 工程通路，含 2 个未收敛帧；不是 measured reset、接触或多轨训练资格。009/010 继续禁止进入 train/replay/normalizer；当前 legacy `[200,216)` 也不属于 shared `[0,151)` 身份。

主 [STUDY_STATE](D:/workspace/supermimic-chord-study/STUDY_STATE.json) 的阶段记录同样列出 full151、shared runtime、接触/动作方向与真实 ego 资格缺口。状态文档可能晚于或早于实际 worker 终态更新，不能拿其中历史 running 字段判断当前进程存活。本次三臂全都通过当前物理验收，却没有一致总体支配；单源、单种子、0.3s 不支持方法优越性或真实 ego 泛化结论。

建议下一实际工作仍有明确上限：完成已冻结 180 次一步诊断并保留结果；据其具体误差选择一次有限机制检查或接 full151 新合同；不追加长训练、阈值放宽或权重搜索。shared 的 sampler/reference/target/replay/normalizer/transfer 成本协议继续作为独立工程任务推进，不能宣称共享 WM 已节约成本。
