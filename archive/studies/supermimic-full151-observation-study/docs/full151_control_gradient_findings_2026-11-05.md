# 受控转笔：真实分支响应与世界模型控制梯度

完整152分支实验和匹配20ms的世界模型比较均已实际完成。当前结果没有提供足够证据支持A2世界模型的局部控制梯度可靠：32个方向组中，9组通过两个扰动尺度的稳定非平坦条件，模型在其中5组给出相同符号。此结果只描述固定来源、固定seed和所测状态；不能外推CHORD的普遍有效性或无效性。

## 实际执行和独立复核

两条路径分别使用A0声明零残差、A2固定final2000保存动作，进行全新原生仿真重放。在k=0、49、99、149各捕获快照；每个快照先运行3次基线，再执行4个关节方向、2个epsilon、正负扰动。总计8个快照、24个基线、128个扰动分支、1808个episode物理子步。原生进程一次exit0，操作系统身份12项通过，终态原PID已退出；没有训练、重试或容量扩张。

24个基线的q/qd/p/R/v/omega和主评分误差均为零。[物理保存独审1762项](D:/workspace/supermimic-full151-observation-study/outputs/full151105_full_serial_saved_review_v1/report.json)全部通过，覆盖控制、相位、保存方程、逐快照来源、容量及恢复收据。[生命周期与来源60项](D:/workspace/supermimic-full151-observation-study/outputs/full151105_empirical_full05_saved_lifecycle_v2/report.json)另行通过。

世界模型仅在CPU加载一次严格A2 final2000消费者，内部按原加载器读取checkpoint两次；实际只执行一次batch8前向和一次控制输入自动微分。没有Env、Warp、物理子步、GPU或优化器更新。[保存分析独审656项](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_analysis_saved_independent_v2/report.json)全部通过，独立复算64条差分、32组分类、全部有效方向投影及预测指标。权重、normalizer、RNG不变性是冻结源码支持的运行时前后digest断言；独审没有重新加载模型以作第二次证明。

## 比较的量

主量是由积分后20ms状态与动作前goal_k计算的受控2+8物体评分。原RewardManager使用派生15ms位姿，其原始rate单独保留为次要量；本结果不是原15ms reward梯度。

真实分支有限差分先将两个FP32 GPU评分分别转为F64，再相减并除以2epsilon。模型项为同一评分在WM(x58,实际u20)上的输入梯度，沿保存的实际有效方向d_eff投影。没有第二次EMA/C2。实际控制端点的裁剪和中心偏移均保留，不把名义onehot当作实际扰动。

分组判据在执行前固定：两个尺度的评分差均超过max(1e-5,10×max(基线范围,基线与donor误差))、斜率同号且相对差<=0.5，才归为稳定非平坦。裁剪、中心偏移和接触签名变化为独立标记，不用于删除数据。

| 指标 | epsilon=0.002 rad | epsilon=0.001 rad |
|---|---:|---:|
| 稳定非平坦方向组 | 9 | 同一9组 |
| 模型与有限差分同号 | 5/9 | 5/9 |
| 其中端点有裁剪 | 3 | 3 |
| 描述性Pearson | -0.04587 | -0.05297 |
| 平均绝对斜率差，score/rad | 0.77926 | 0.68406 |

其余23组至少一个epsilon低于信号门槛，不能全部称为真实零梯度；两尺度都过门但不稳定的组为0。两个epsilon是同一批依赖方向的两次检查，不是18个独立样本。全部64行、负值、零值、较大响应、裁剪以及k149终端响应均保留，没有质量筛选或新加的准确率验收线。

同一8个中心状态的一步预测：物体位置向量RMS误差6.3282mm、旋转RMS误差4.9473度、评分RMS误差0.24376。它们不是全轨迹误差，也不能替代控制梯度验证。当前数据说明，应分别检验状态预测和优化所需的局部响应。

![全部64条控制响应比较](D:/workspace/supermimic-full151-observation-study/outputs/full151105_A2_WM_physical_gradient_figures_v1/physical_WM_controlled20ms_comparison.png)

[全部64行CSV](D:/workspace/supermimic-full151-observation-study/outputs/full151105_A2_WM_physical_gradient_figures_v1/all64_comparisons.csv) · [分析报告与来源pins](D:/workspace/supermimic-full151-observation-study/outputs/full151105_A2_WM_physical_gradient_CPU_v1/report.json) · [绘图QA](D:/workspace/supermimic-full151-observation-study/outputs/full151105_A2_WM_physical_gradient_figures_v1/visual_qa.json)。图已由绘图agent和Root实际查看。

## 接触、隐藏状态和因果边界

原先发现的接触payload非逐位重复已精确定位：A2/k0三个基线的第一个5ms子步中，一个contact normal分量由0变为约-5.3e-313。接触几何、顺序、约束力、控制、状态、自碰撞标签、评分及物体总受力/总矩均相同；没有观察到宏观力差异。原因未由新实验归因。[细化诊断](D:/workspace/supermimic-full151-observation-study/outputs/full151105_A2_phase0_contact_repeat_diagnostic_v2/report.json)保留原值和旧报告。

Graph隐藏scratch没有逐字节恢复或完整write-before-read证明，继续标记UNKNOWN。经验基线、可见库存及源绑定收据不构成完整隐藏状态证书或唯一真实物理Jacobian。模型self15是自碰撞头，不是手物接触或力头；本实验不授予HO力梯度资格。

此前三方法final2000完整任务均失败的结论不变。三臂在checkpoint100、任何先验前向或Actor更新之前，WM/normalizer已经不同，因此旧同预算比较仍是描述性结果，不能将P_ONLY与A2的差异单独归因于support项。当前受控数据仍只有一个来源、一个seed，独立ego N=0；已知粗糙参考的几何与实际任务失败也未被本诊断修复。

## 下一项研究

先冻结并审阅两个互补实验，再作新的评分或模型执行：一是用本次已保存的152个物理分支，比较训练中实际穿透项、support项及总bonus的有限差分与WM AD；二是在同一固定WM、同一Actor及同一状态批次上隔离加入support对局部计算图的影响。分开报告几何代理、真实solver接触和控制响应；不将梯度能传播写成接触动力学正确，也不以继续扫权重或挑选checkpoint替代诊断。

Goal保持active。真实ego数据接入、多轨迹泛化和真机观测/控制契约仍待后续阶段；本轮没有启动新长训练。
