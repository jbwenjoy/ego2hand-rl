# 完整151帧原生仿真：首次3秒A0基线

**完整运动的原生仿真链路已实际跑通；A0任务质量失败。** 新模式执行150个控制区间、600次物理步、150次原生Manager末forward并保存601条状态。期间没有重置、质量提前截断、物体辅助、Actor/WM或训练更新。原13项评估中10项通过，位置、轴向和穿透3项失败。它是后续学习修正的完整基线，不是SuperMimic/CHORD收益或真实ego结果。

保持原研究Goal active。当前只是一个已有缓存来源legacy[200,351)的受控实验；shared[0,151)及真实ego仍分开，独立ego N=0。此前[全151连续拟合/SCP失败](D:/workspace/supermimic-full151-observation-study/docs/full151_contact_coupled_findings_2026-11-04.md)完整保留，没有把几何FAIL改判为PASS。

## 从粗糙输入接入实际native链路

以SCP最后接受点major8为明确known-rough参考，重新计算真实native BODY21并执行真实CPU Reader151。没有把旧macro21改名或把16帧padding到151。新[输入束](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_input_v1/bundle.json) SHA `59406c6214a085afba7efcc2b33880d5c82b409ee4f6485ec71c073fec19dc76`，input signature `3599bee61d92152b0c4f7c3bf8b71e99d2c00af460c81d23cecbd76d7e3fdd04`。

实际151次BODY FK/Reader CPU绑定检查16/16、保存数据独审67/67通过；没有碰撞查询或GPU。native FD初速度与解析初速度最大相差7.921499rad/s，故二者必须分列。实际runtime使用冻结解析qd151与物体world v/omega，初始化仍是原P0/P1、q0、非零qd0、pose/twist与u0，未改用Reader FD。

新增隔离包[full151_native](D:/workspace/supermimic-full151-observation-study/src/full151_native)直接包围原生Manager/ActionTerm/PenMDP/Runner接口，旧104源码没有放宽。控制表150×20，真实phase0..149，终端obs150，没有u150；inactive imagination仅u149占位。原EMA残差、单次C2、world/body角速度转换、selected reset/backfill、H8 survival/norm/loss/bootstrap、replay terminal/generation及source offset200保持独立合同。核心真实CPU方法/有限fixture61/61通过；fixture没有代替实际环境reset资格。

新checkpoint身份105与严格Actor/WM消费者实现，65项CPU合同包含旧3/103/104及整数伪改105拒绝。**本轮没有真实105训练checkpoint或健康加载，仍为PENDING。** Actor、Runner和optimizer没有实例化，self head15也没有改成150。151接触意图/扳手目标仍缺失，未用空目标冒充A2。

## 运行与完整性

[实际配置](D:/workspace/supermimic-full151-observation-study/configs/full151-a0-authorized-v1.json) SHA `303d61c55ab62d03d1b8b62fa0f0af8aedc1fa3ee4846d1e6e6b3972d00c30b4`，绑定136个source/input pin；[新协议](D:/workspace/supermimic-full151-observation-study/protocols/full151-native-a0-v1.json)在运行前冻结。默认前端实际CPU preflight没有导入Torch/Warp；显式Root授权后仅运行一次GPU A0，无重启或质量择优。

真实worker PID40120，OS creation UTC `2026-11-04T06:56:08.4947920Z`；launcher41436、supervisor41476。保留的工具session69063已终态exit0，watchdog无超时，精确OS身份12/12通过，终态CIM原PID全部不存在。worker wall15.085858s，watchdog16.203s范围嵌套，不能相加；构造、graph warmup与startup全进程API计数未观测，不能把600称为整个进程总步数。

probe episode实际API attempted/returned为step600/600、forward150/150，另有601状态/600solver行/150control events/150reward snapshots。质量失败只记录，不清pending reset来强行续跑。最后phase149产生timeout/reference_end及obs150。actual8 copy→set_const0 restore→双readback一致，xi46、nominal MJB一致；初始化full qpos/qd/u/mocap精确，bodyω API舍入误差1.3411e-7rad/s在预声明1e-6算术容差内。

容量observer从构造结束后开始，记录after-constructor、explicit reset前和final三次raw flags，均0，未知读0；episode内无reset。构造之前及内部未观测工作不作容量或步数证明。entry旧的“startup separately recorded”泛化措辞不能补造计数，以evaluation/result的`UNKNOWN_NOT_WHOLEPROCESS_OBSERVED`为准。

## 原任务指标

| 指标 | 实际值 | 原标准/判定 |
|---|---:|---|
| 151参考结点物体位置误差最大值 | 38.002625mm | ≤20mm，FAIL |
| 151结点有向shaft误差最大值 | 60.249328° | ≤20°，FAIL |
| 601点shaft诊断最大值 | 60.373322° | 单列，不替换151主判定 |
| 600solver任意类别穿透最大值 | 7.941423mm | ≤3mm，FAIL |
| 601点实际关节硬界越量 | 5.05447e-8rad | ≤.001rad，PASS |
| 150条实际u/缓存/基准表 | 全部逐位一致 | PASS |
| 149次实际目标更新率 | 14.999902rad/s | ≤15，PASS |
| 实际力矩负载比最大值 | 1.000000034 | ≤1.01，PASS |
| 负载累积charge最大值 | .489953 | ≤.8，PASS |
| drop、nonhand物体接触、外部辅助 | 均未发生 | PASS |
| clock、容量 | 通过 | 见独立证据 |

[原始evaluation](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A0_v1/evaluation.json) SHA `4083d833859e04989429053a10d2bed49520932231f5a5d66fc614c5aa053dbe`。原position/shaft/penetration质量阈值没有放宽。

时钟使用预先固定的FP32递推逐bit检查，全部601点一致；末值3.00002121925354s，nominal drift21.219254μs，在原full151 evaluator的50μs预算内。旧0.3s prefix的2μs只保留为诊断且为false；这一full-horizon规则在首run前依据原3秒源码声明，没有改写模拟时钟或在失败后放宽标准。

![完整3秒原生A0结果](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A0_saved_summary_v2/full151_native_A0_summary.png)

## 接触、负载与证据范围

手物最大穿透7.941423mm，30/600步超过3mm，598步生成HO；self最大4.633130mm，12/600步超过3mm，124步生成self。raw contacts共2451条（4 startup、2179 HO、268 self、other0）。这些是原生生成接触，未生成pair不能当作正间隙或已观测human release。

虽然负载门通过，仍有12个joint-step触及近满幅（12/600 solver步），不能称“没有饱和”。actual8 gain/bias、真实u与**solver prestate** q/qd重算并限幅后，对实际force最大差仅2.3277e-7Nm；错误使用poststate会出现.678993Nm差异。它支持本次采样时相与PD接线，不能据此认定任务失败由某一个PD参数唯一导致，也不把旧静态LP witness当作实测接触力。

150个native reward capture中，self15在32行有104个positive components，属于机器人自碰通道，**不是HO或人手接触标签**。传感器found原数组未另存，不能离线宣称完整重放所有sensor逻辑。reward state58、self15、torque遵守原生reward在Manager末forward前的时相；physical tracking使用poststate与reference k+1，二者不混。保留PEN memory和两种state读数供后续WM接口核对，本轮未前向WM。

[独立保存数据审计1893/1893](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A0_saved_independent_v1/report.json)直接复算601/600/150数据、原13门、接触类别/深度/force、初始化、实际控制、FP32时钟、容量与native captures，没有调用生产grader或重跑物理。[生命周期/来源独审67/67](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A0_lifecycle_independent_v1/report.json)核136份ZIP/current源、OS身份、actual8、控制/phase/末端以及证据范围。Root和作图agent均实际查看完整0–3s六面板，全部失败和采样分母保留。

## 后续推进

下一步是把已经真实运行的full151 runtime/controller接入**实际from-scratch训练Runner与format105保存/容量/fingerprint/健康审计链**。先做有明确预算、相位覆盖和partial reset证据的工程smoke，再考虑长训练；A0三项质量失败保留，不能作为工程接口失败借口或改成PASS。A0-only worker仍拒绝训练，后续使用独立入口/协议，不给当前入口偷偷增加role或载入旧104。

A1L基础残差训练可以先完成真实接线；P_ONLY/A2必须等待独立151接触意图/扳手目标及非退化检查，不从unknown观测伪造标签。随后再做同输入、同样本/update预算的完整运动比较和相应WM梯度/接触验证。现有prefix seed0结果、current104局部物理梯度和本次A0均为不同条件，不拼成总体收益或跨轨迹加速结论。

本轮没有新长训练，因此尚未触发“首个健康checkpoint后估时暂停”。shared多轨迹、真实ego数据接入与硬件子版本/观测契约仍按原Goal继续，不因本次基线完成而缩小成功标准。
