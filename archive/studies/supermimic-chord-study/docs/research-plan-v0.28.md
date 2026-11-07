# 研究方案 v0.28：原生子步采样与新 phone 世界模型

2026-11-07。完整范围承接 [v0.27](research-plan-v0.27.md)、原实施方案及当前目标；上一回合与本回合均为 progress。原2秒持握、双步长、材料先验、61帧时序、SuperMimic/CHORD同预算与真机观测契约仍未完成。真实ego用于几何源参考，尚不具备手机策略训练或真实接触动力学泛化资格。

## 已实现的实际读取器

`NativePhoneReader` 从同一个冻结原生MuJoCo3.11模型运行20ms绝对位置控制。每episode仅一次初始化forward，随后每tick仅一次mj_step；观察不刷新derived字段、不重置warmstart或物体状态。POST qpos/qvel通过解析quaternion、实际body_ipos得到COM/world state58。接触、能量和实际力矩保留solver PRE时间；raw contact rows包括微弱和零法向力，不凭阈值删原生点。

源状态、赋予当前动作后的源状态及目标状态均保存完整 `mjSTATE_INTEGRATION` 数组和SHA、signature、整数tick与原生clock。RESET_FORWARD的初载与新动作第一步PRE力分开。模型、增益、材料、关节/控制真限位、无外力/插件/回调约束固定；h=.3125/.15625ms分别64/128个子步，h是唯一显式模型选项变化。

唯一一次192步实际黄金对照中，两个步长各14/14、汇总4/4检查通过；所有POST qpos/qvel及PRE力/能量/contact rows/native clock与旧控制循环逐字节一致。实际2次MJB读取、2次初始forward、0观察forward，无训练。[实际对照](D:/workspace/ego-data-pilot-2026-11-07/phone_episode_acquisition_v1/independent_reader_review_v1/actual_golden_v1/report.json)

采样器源审指出初始warning/非有限接触应留存为零区间episode，而不能让后续families消失。独立v2采样器补了此分流、空分区类型和实际ipos/范围元数据；模型/schema/非法控制/外力等结构错误仍终止。Reader及其黄金对照源未改，metadata-only的原v1源/协议逐字节保留。实际采集没有出现初始异常。

## 预冻结分组与实际数据

从exact及1mm两个源映射的seed1/rank0出发，各24个family；每个family的q8初态微扰≤.005rad、其余12q与phone root不动，初始化与控制种子分开。全20控制均改变，主8扰动≤.05rad，其余12≤.025rad，整条动作曲线按真实限位余量缩幅，无逐样本clip。1mm的index某轴正余量约.000628rad，真实激励不对称，不能声称20维均匀辨识。

全部初态、10×20动作和36/6/6 family拆分都在任何新物理执行前冻结。一个family的两步长与全部区间始终同组，未按随机行划分。每episode最多10个20ms区间，采集COM界.15m、角度90°等只用于限定采样范围；原持握成功与50mm/45°早停协议不变。采集允许经过失握状态，用于学习接触过渡。

| 分组 | families | 全部区间记录 | 完整且有限可训练目标 | partial |
|---|---:|---:|---:|---:|
| TRAIN | 36 | 720 | 684 | 36 |
| VALIDATION | 6 | 120 | 114 | 6 |
| TEST | 6 | 120 | 114 | 6 |

唯一采集进程exit0、stderr空，wall约25.25s；96episodes、960记录、89679积分步、2次MJB、96次初始forward、0观察forward。48episode达到0.2s，48episode在COM采集界结束。48条partial保留触发步、实际prefix目标及接触来源，但H目标/delta52为NaN、训练mask=false，未插值或补造20ms目标。所有原始记录约124.75MB。[实际采集](D:/workspace/ego-data-pilot-2026-11-07/phone_episode_acquisition_v1/acquired_v1/report.json)、[预冻结合同](D:/workspace/ego-data-pilot-2026-11-07/phone_episode_acquisition_v1/design_contract_v1.json)

每个区间都保存逐子步载荷、覆盖率、首次阈值tick、最后PRE载荷、完整接触frame/local force及物体COM处世界系接触扳手。训练七个接触量为两pad覆盖率、最小法向力、最后PRE法向力及非pad峰值；前者是整个20ms窗口，最后PRE在H−h，state目标在POST H，语义不混用。正法向聚合与原成功门一致使用sum(max(native decodedFn,0))；原始有符号force6仍完整保留。

完整integration-after-action/target receipt、实际control、模型/h/状态定义组成精确转移别名键，跨分区为0，无去重或重分配。48个初态及控制曲线SHA各不重复。工程分组可用不等于state58已是Markov，也不等于独立ego轨迹：数据仍仅为同一ego源frame两个算法映射附近的局部仿真，质量/惯量、摩擦、功能指腹和源重力注册都未实测校准。

训练前保存数据独立算术复核12/12通过：全部960区间、89679个POST状态、52729个逐步contact及198个RESET contact、2976份完整状态receipt均复算；实际ipos下的COM/world速度、物理world-left delta52、逐步PRE载荷/覆盖/最后PRE标签、冻结动作与连续状态/分组/别名核对一致。912条完整目标、48条partial都保留，0跨组别名。此复核不调用生产数学、原生模型或神经网络；未做TEST模型指标或调参。原始qacc26未逐步导出，不冒称独立检查了未保存字段，其finite守门由Reader来源与NaN/abort传导证据限定。[独立数据复核](D:/workspace/ego-data-pilot-2026-11-07/phone_episode_acquisition_v1/independent_dataset_review_v1/report.json)

## 新世界模型及首个checkpoint约定

新小型MLP使用raw80：完整state58、实际绝对u20、h及子步数；80维schema与旧pen、旧raw78回调明确区分。输出为物理世界系delta52的训练组归一化表示，以及两pad覆盖率和五个log1p法向力量。解码使用冻结的世界系左乘 `Exp(deltaR) @ R`。新建参数，未继承旧pen的权重、chart、尺度或奖励。

固定配方为CPU4线程、128×128 SiLU、AdamW lr=.001、全TRAIN完整H批次、5000updates，checkpoint250/1000/5000。均值与尺度仅由684条TRAIN目标拟合；VALIDATION用于报告，TEST文件只作hash绑定，训练不打开其数组。mask直接使用经审阅的原采集标志，所有排除ID和原理由保持可追溯。

首个checkpoint将原子落盘，并自动校验有限权重/梯度/Adam、原始weights/Adam/RNG序列化读回及验证预测一致性，再给出ETA。该健康检查不自动证明训练续跑恢复、物理梯度或策略收益。遵守用户约定：首个健康checkpoint及ETA确认后暂停目标；进程可在既定5000步预算内继续。

在训练前另计算了不含神经网络的TRAIN均值基线：同114条VALIDATION上，20ms COM向量RMS约8.865mm、R矩阵Frobenius RMS约.08573、覆盖率Brier约.11463。仅用TRAIN均值，TEST数组读取0；后续新模型须与该同数据基线比较，不能仅报训练损失下降。[基线](D:/workspace/ego-data-pilot-2026-11-07/phone_world_model_train_v1/constant_mean_baseline_v1/report.json)

本轮实际启动了新的phone世界模型训练。第250步checkpoint已原子落盘，有限权重/梯度/Adam、严格逐字节保存读回与验证预测一致性检查通过；Root另作一次自有checkpoint加载，核对源/数据/归一化/梯度相位及前250条固定日志前缀。未构造额外网络或优化器。

同114条VALIDATION，COM向量RMS为1.8764mm（常量TRAIN均值基线8.8649mm），R矩阵Frobenius RMS为0.0386134，覆盖率Brier为0.00732121。这是局部仿真family的单步预测结果，TEST模型指标仍未计算，不是持握、策略、独立ego或物理动作梯度收益。

首健康checkpoint时剩余ETA约9.2s，估计结束时间2026-11-07T08:58:50.634562+08:00；该估计不含后续checkpoint I/O，非完成承诺。交接时训练进程已终态，returncode=0。本轮交接后按用户既有约定调用goal暂停；完整目标未完成。[首checkpoint及Root读回](D:/workspace/ego-data-pilot-2026-11-07/phone_world_model_train_v1/first_checkpoint_Root_readback_v1.json)。

## 后续完成条件

恢复目标后先核对实际训练终态，再做未参与训练/选参的TEST family评估、闭环多步误差和同完整积分状态下的动作扰动/接触切换梯度核对。旧112条secant来自另一失败前缀，仅可作具名分布外检查，不能当本次同状态梯度真值。需要为新episode的验证状态生成匹配动作分支，并将克隆保留在其原family分组。

随后将新phone Reader/世界模型接入明确的任务奖励、动作残差和SuperMimic/CHORD同预算协议，检验初始动作预补偿与主动接触恢复。原2秒无辅助持握、双步长、动力学先验鲁棒性、61帧时序、独立ego轨迹、资产校准与真机观测契约继续保留。当前数据和新模型均不能代替这些要求，完整目标未完成。
