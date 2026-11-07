# C2：固定目标内收的零样本受控诊断

本试验在原三臂与A1L同检查点五次重复结果已知后提出，是已见clip上的工程探索。它不替代原实验，也不作为真实ego、多轨迹或训练种子泛化结果。C1速度制动方案仍未通过回放资格，m0/a保持null；C2不使用其制动公式或参数。

## 问题与干预

原A1L检查点的五次原流程复验中，物体跟踪等门槛均通过，但实际关节越界在0.734–1.688mrad之间变化，只有一次通过冻结1mrad门槛。目标角合规不能保证状态合规。先检查一个简单输入干预是否具有继续训练的价值：将原hard-joint与enabled actuator ctrl交集每侧内收固定margin，再投影原绝对目标。

三个条件为0、0.005、0.020rad。5/20mrad是本轮预先固定的工程探测剂量，不是由制动校准得到的鲁棒裕量或硬件标定值。实际状态、速度、物体和接触仍完全由仿真产生；该投影只约束目标输入，不提供实际状态可行性保证。

## 固定设计

- 固定原A1L training seed1的final2000检查点、evaluation seed0、同一参考与物理模型，完整3秒自由物体闭环评估；不训练。
- 每条件3个fresh process，共9次。三block顺序为`0/5/20`、`20/0/5`、`5/20/0`，使每条件在block和位置各出现一次。没有按质量早停、替换、重试或最好一次选择。
- 三条件全部使用同一wrapper、FP32投影函数和50Hz proposed/executed目标采集，包括margin0。margin0不能跳过投影kernel而使用不同的快捷路径。
- 原初始化完整执行，t0物理q/qd、物体位姿/速度、腕及执行目标保持原样。干预从第一条真实动作区间开始，仅在原`process_actions`之后原位更新absolute `_targets`；原EMA `_processed_actions`及其记忆语义不变。
- 后续原`apply_actions`在四个5ms子步沿用缓存目标。没有5ms状态采集、额外forward、物理状态投影、对象辅助力、模型修改或验收放宽。

原未加hook的五次重复只作历史背景，不作为本次0条件样本。新增条件中，策略根据各自真实状态闭环行动；不要求不同条件后续proposed目标序列相同。

## 验证与报告

先用CPU核对投影区间非空、0条件输入逐位恒等、FP32路径、EMA未改、初始化与动作期分离、150个动作/151个状态及原source pins。运行时输入sidecar应满足`executed[i] == rollout.target[i+1]`，另核对缓存及t0目标。触发率按150个tick和150×20个关节分量分别计算，不将5ms保持重复计数。

所有9次报告原task及实际q/target限位、物体跟踪/旋转进度、掉落、穿透、负载、target-rate、接触和成本；保留全部原始记录。固定1mrad关节允许量、20mm/20度跟踪门槛和其余原协议不变。训练seed N仍为1，三次进程重复不是三种独立策略；仅作描述性条件对照，不给总体置信区间或确定因果结论。

GPU串行共用既有锁，每child45秒、整体240秒调度/停止请求预算，工程失败停，任务质量失败继续固定清单。默认入口只做CPU预检。

若后续考虑训练，必须另行实现真实与想象共用的目标映射，并保证replay、WM输入及PD估计均使用实际执行目标；本次zero-shot不宣称已经验证该训练合同。不得因本次已见checkpoint效果直接声称新方法泛化，确认性比较需新种子与事前定义的重复评估。

实现与冻结清单由独立工作区`D:/workspace/supermimic-target-inset-study`保存；运行前固定其具体文件与SHA，原工作区训练与物理记录保持不变。

后续共同控制器评估已完成：5mrad三臂A1L/P_ONLY/A2分别1/2/1个训练seed三次全过，20mrad分别1/2/3个，各方法N=3。20mrad是最初声明的另一剂量，补查在5mrad结果已知后固定，分别保留两轮；不能混池、挑最好剂量或作独立确认。A2的20mrad九次actualq无采样超限，但shaft峰值较5mrad增加约0.7°，A1L seed3仍q超限约12.2mrad。见[执行记录](progress-2026-11-03.md)及[20mrad独立摘要](D:/workspace/supermimic-c2-020-portfolio-study/outputs/analysis_v1/SUMMARY.txt)。

训练合同正在新工作区`D:/workspace/supermimic-c2-training-study`独立实现；上述结果仍是旧模型零样本执行干预。长训练前须核验real/imag同映射、实际u进入replay/WM/PD、EMA及初始化不变和checkpoint控制器身份；工程短测用seed0，未来确认用新seed4/5/6，预算与门槛保持原协议。
