# 50 Hz原生控制数据与首检查点训练配方

2026-11-08。本文件记录启动新神经训练前已通过的门槛。训练的实际进度、保存
健康和ETA以独立运行目录的`health_0010.json`为准；不能把本页当作训练已成功。

## 已执行的控制与数据工作

新20维残差在每个20 ms间隔只更新一次物理弧度EMA，再保持绝对PD目标至下个
控制时刻。每关节分别计算正/负可用空间`min(0.05, upper-base)`及
`min(0.05, base-lower)`；边界附近仍保留向内动作能力，不裁剪原动作或控制。
EMA系数0.5、初态0。核心数组代数同时用于NumPy/Torch，无GPU标量取回。

初态和前馈PD目标来自已声明的六维合格构型，未改变原模型、重力、惯量或摩擦。
两个dt下的零动作原生轨迹，与历史恒定PD目标的全部16个逐tick数组**字节一致**。
不是重新选择较好基线。6份完整积分状态+EMA恢复的后续间隔也逐字节一致。
恢复不额外`mj_forward`，避免改变已保存的数值求解暖启动；下个step1刷新派生量。

新实际观测为state55：q20、dq20、世界COM位置3、BODY到世界R的前两列6、COM
世界速度3、世界角速度3。POST观测由真实qpos/qvel读出；接触标签来自solver-PRE。
区间力矩逐tick绕移动COM聚合，再统一旋转到区间起始BODY轴，避免把末态坐标
旋转误用到整个区间。原始接触还包含直接力偶，不能重复计入或遗漏力臂。

完整数值状态与暖启动26均保留；state55本身未被假定为有限求解器的精确Markov
状态。源人体接触仍UNKNOWN；已知标签仅是当前名义仿真器的真实接触/力。

|数据角色|来源|完整20 ms行|用途|
|---|---|---:|---|
|TRAIN|16个固定种子，粗dt|378|仅此角色拟合尺度和梯度更新|
|CHECK|4个独立动作种子×双dt|193|只读预测检查|
|CHECK_RESPONSE|3个时刻×4个关节方向×正负动作×双dt|48|只读成对动作响应|

24个不足20 ms的末段保留原始记录但排除固定H拟合；不拉伸时间、不伪造H末状态。
总共54,333次原生积分，含基线、恢复和分支复核。训练/检查输入无完全重复。
所有24个探索/关节反馈episode仍因姿态损失终止，未完成整段2秒。这里的动作
是有界多正弦与关节姿态反馈的混合，不是学到的物体跟踪策略或合格示范。

保存复核7/7通过：控制重建误差0，原始接触重建区间扳手误差4.44e-15、法向力
误差0，独立原生BODY速度/COM读回误差1.60e-14。119项库测试通过。
源对象参考按真实k/50秒记录目标位置、方向、速度及完整H端点跟踪误差；参考
全时段存在，不代表机器人已执行完整源轨迹。仅使用这一锚定任务，不能写成
未见轨迹、对象或RGB重建泛化。

## 有界的新响应模型

`configs/mug_response_first10.toml`显式指定新配方。输入102维为state55、源暖启动
26、真实执行u20和原生dt1；不含未来接触、未来状态或参考目标答案。输出87维为
世界左乘旋转增量的物理delta52、下个暖启动26、区间起始BODY轴平均移动COM
扳手6及三指平均Fn3。Fn头非负，但未校准前不能直接用于接触奖励。

这是一只物体的一步响应基线。没有Actor、Critic、想象优化或A1/A2；没有读取
旧手机检查点或旧328个heldout。尺度只用TRAIN拟合并冻结；CHECK/RESPONSE不参与
优化或选择。它与旧手机state58/366输入模型语义不兼容，不能伪装为其恢复。

CUDA上使用三层512的MLP、batch256、FP32、fused AdamW；固定数据和尺度一次放入
GPU。更新环无几何/原生调用、无逐步`.item()`健康检查或CPU传输，健康检查集中在
首检查点边界。这里只是新小规模基线，不能据此宣称原算法整体训练加速。
优化器/序列化接口依照本地Torch2.14配方使用。
[AdamW接口](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)，
[保存读回接口](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)。

计划1000次更新，但入口**只执行到第10步首检查点**。保存模型/尺度缓冲、完整
Adam状态、Python/NumPy/Torch CPU/CUDA RNG、独立CUDA采样器状态、数据/配置/源码
pin和梯度诊断。读回检查原始dtype/字节、全部Adam步数、RNG、下一采样批次及
CHECK预测一致性；不为了验证而继续优化。ETA仅按前10步末5步估计余990次纯更新，
不含启动、保存、原生采集/校验，也不是整个研究完成时间。

```powershell
# 只检查配置/数据角色，不创建优化器或训练
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/train_mug_response_first10.py --config configs/mug_response_first10.toml --output runs/mug_response_wm_NEW
# 授权的有界CUDA启动；程序在首10健康读回后退出，调用方暂停Goal
& ..\dex-supermimic-research\.venv\Scripts\python.exe -X utf8 tools/train_mug_response_first10.py --config configs/mug_response_first10.toml --output runs/mug_response_wm_NEW --run
```

健康仅表示结构/数值与保存恢复正确，不等于响应、梯度或奖励已可靠。还要报告
独立成对动作的响应误差与零响应先验（NMSE=1）对比；即使保存健康也不隐藏差的
预测检查结果。达到用户要求的首健康新神经检查点后，估计ETA并暂停，不自动
完成1000步或继续Actor训练。

## 保持的研究范围

旧文件字节和配置未改，新显式导入的`learning/compound_response.py`和配方不会
被旧训练自动加载。数据阶段证据在`provenance/mug_H20_data/summary.json`；大型
NPZ、原生轨迹和将来检查点仍在忽略的`runs/`内并由pin定位。

受控方块/转笔、完整无辅助评测、实际物体参考跟踪、WM接触/动作物理梯度、匹配
A1/A2、多轨迹总成本、独立RGB/源重力及真机参数仍需推进。新输入/反馈数据和
一步模型不替代这些要求，不修改驱动/CUDA，也不证明整个Goal完成。
