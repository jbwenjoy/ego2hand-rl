# Full151/105同钟控制差分提案v2

状态 **proposal_only / NOT_EXECUTED / GPU未授权**。保留v1协议19d8、原文档与source1a65静审/快照；本修正不重写任何已执行训练、评估或旧104物理成绩。

源码审发现真实5ms相位差：SimReader.state直接读取qpos/qvel，reward_snapshot.physics为20ms积分后状态；native object reward调用object_pose_wrist_local，经Entity.root_link_pose_w读取xpos/xquat，是最后solver的15ms derived pose。Manager在最终forward前计算reward，因此不能把实际native GPU reward15ms与从x20重算的score用8eps做等价门。具体installed/source pins在协议source_phase_evidence。

Root采用同钟方案A。主标量是**在捕获的postintegration x20上、以preaction goal_k计算的controlled FP32GPU native-formula 2+8 score**。这是原生数学表达式的受控重算，明确不是原RewardManager实际reward。未来WM输出同为20ms，使用完全相同score/goal/单位，再比较实际执行目标u的自动导数与物理有限区间secant。控制公式为2 exp[-(||p−pg||/.05)^2]+8 exp[-(两轴端点平均误差/.04)^2]，轴半径.1m，两个FP32term及GPU FP32两项和分别保存。若派生乘dt，必须标derived，不能称原始step-value cache。

原RewardManager GPU object_pos/object_rot rate和GPU subset sum另外保存，并标记derived15ms、实际dt/scale/term索引/权重/顺序；self15/torque也保持末solver15ms标签。它们是原reward/接触诊断，不与x20作相等检查，不代替20ms主score。CPU FP32/F64对同x20/goal重算只是主score的同相位算术诊断，8eps32*max(1,|term|)仅算术界，不能吸收5ms运动差。k+1目标score只作诊断，不改命令或追加决策。

其余有限设计不变：fresh A0declared-zero及A2saved-action来源，k0/49/99/149共8snapshots；每个3baseline+4direction×2eps×sign，共152branch/1808episode steps。k149仅终态响应，obs150无u150或policy bootstrap。外部host OR容量账本不可被snapshot恢复；完整inventory/INTEGRATION/actual8/EMA-C2/四子步actualu/三baseline/长clock/foreign source-review仍必须独立通过。首次foreignreview=None只发现inventory并停止，不宣称native restore资格。

FD不是唯一真实梯度；WM自导数不是物理Jacobian。self15不是HO力/人类标签，proxy P不作HO力梯度真值。几何/力/全运动资格不因本探针升级。实施仍需Root独立冻结执行协议与授权；本次没有调用Env、模型、WM或物理。
