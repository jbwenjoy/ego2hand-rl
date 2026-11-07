# 受约束三次B样条参考：CPU结果与未通过的几何资格

2026-11-03：新分支已完成一次L1线性规划及一次SLSQP。连续参考范围、计划控制速率和参考状态PD预测通过；采样手物重叠仍失败，**没有运行动态GPU、物理仿真或训练**。它仍是受控旧clip007的一条0.3秒候选，不是完整转笔或真实ego结果。

| 前置检查 | 本次终态 | 结果 |
| --- | ---: | --- |
| 三次B样条控制点硬限超量 | 0 | 全曲线凸包充分界通过 |
| 最终14次计划C2目标更新速率峰 | 11.448526rad/s | ≤15，通过 |
| 最终15×20计划目标C2限幅分量 | 0 | 提案与C2逐位一致 |
| 最终U32下track/hold全区间参考PD Bernstein证书 | 超量均0 | 充分条件通过，非动力学证明 |
| 61点FP32两种分组参考PD预测 | 均在力限内 | 预测通过，非实测力 |
| 采样手物生成深度峰 | 3.194716mm | 超过3mm资格门槛 |
| 超过3mm的采样时刻 | 5/61 | 几何资格失败 |
| SLSQP终止 | status9，80轮上限 | 不称收敛或全局不可行 |

规划深度上界为2.999mm，是在原3mm资格标准内预先固定的1µm数值裕量。5个采样同时超过两者，不能用裕量或容差覆盖失败。没有实际q/qd、施加u、接触力或startup读回；所有`u_exec`字段均为最终**计划目标**。

## 本次方法和范围

保留[旧失败prefix16](D:/workspace/supermimic-observation-prefix-study/README.md)不变。新[冻结协议](D:/workspace/supermimic-observation-constrained-prefix-study/protocols/constrained-prefix-v1.json) SHA `0798ea7aa6da6f56684067992906bbba3136f9e15b413f743674f54159ad2e6d`，在任何LP/非线性求解前声明：

- 三次clamped B样条18个控制点；P0为source64 H123 q0，340自由变量，初猜全部重复q0。不读取旧539d或旧失败候选的q1..15作为种子/目标。
- 原6,596行拟合目标、尺度、权重和H123材料假设不变；物体仍为冻结rawFD驱动的SO(3)/平移曲线，腕固定。改变的是参考参数化与显式约束。
- 系数盒给全曲线范围充分界；15个提案目标位于C2内域、14次更新速率受限。每个20ms ZOH区间的参考PD力矩用Bernstein系数约束，并在最终FP32 C2目标下重新核验。
- 61个5ms参考采样，每时刻hand-object/hand-self各一项最大**生成重叠**约束。未生成接触的物理gap仍NaN/UNKNOWN；操作指标0不表示测到了零间隙或正净空。并列接触/跨pair并列及选中分支均保存。
- 先一次HiGHS L1最近repeat-q0线性初值，再一次SLSQP；80轮、250次objective fun、220秒软限/240秒owned启动期限，不换初值、权重或模式补跑。

凸包、Bernstein和对hold在moving-qref上的预测都是充分但非必要限制；该hold预测不等于hold实际轨迹。61点约束不证明点间连续净空或实际动力学。控制域2e-6rad、参考力矩2e-6Nm、速率1e-4rad/s规划裕量只作预声明数值收紧，不是模型精度/硬件鲁棒保证。15rad/s是目标更新限制，未偷加为硬件关节速度规格。

源窗口仍为旧clip007[200,216)，与共享目录clip007[0,151)不同。`retarget_variant_id=H123_cubic_bspline_constrained_prefix16_v1`区分新轨迹/控制，raw lineage未增加独立演示N，不复制旧静止保持资格。固定掌映射仍属任务重设计，未知相机/尺度/解剖/接触置信度不因此解决。

## 实际唯一运行与审计

Root[单次计划](D:/workspace/supermimic-observation-constrained-prefix-study/configs/constrained-prefix-authorized-v1.json) SHA `fa490719a1f75f7f0f3ff2bdee589a67f3b2149cf206f132a62790aec38336b2`。默认CLI预检先运行，77次几何调用=16拟合+61约束，LP/SLSQP均0；初猜不满足所有约束是被保存的起点状态，预检通过只表示结构/输入可运行。

正式phase0只有一次LP，HiGHS成功、L1偏移2.95605497e-5rad、618次内部迭代。返回的微小正线性残差6.33e-17原样保存，低于预声明1e-9 LP容差，没有裁剪见证。随后一次SLSQP：80轮、240fun、81objective-Jac、241constraint-fun、81constraint-Jac、80major-callback，240个唯一trial向量。最终cost312.543411；status9迭代上限，原始线性不等式最大残差−9.901e-10、系数边界超量0，但几何不等式未取得可行资格。

只将正常返回的terminal `result.x`导出为[候选](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_prefix_geometry_v1/candidate.npz)，SHA `64aa2cfe44f20fee4546b07c4546916c95d57771404cb5fb4e39156206ad9b43`；未从历史选最优/可行迭代。所有部分检查点标明jac/major/trial来源，不自称accepted或feasible。终止状态和primal条件分开；本次确实是primal几何失败，而不只是状态码失败。

全局18,077次原生碰撞几何调用=fit所属3,616+约束所属14,461；嵌套`fit_parent.actual_collision_calls`是共享同一个g的全局计数，不再次叠加。所有step/forward/inverse/geomDistance/contactForce计数0，Torch/Warp未导入。静态与数学检查包括32项实现、18项计划负例、独立15项basis/线性代数检查；窄检查不转化为全接触Jacobian资格。

实际worker PID24216、launcher30948，OS创建`2026-11-03T08:05:18.676440Z`。worker94.373991s与watchdog94.516s嵌套，不相加。40b owned-tree/159c IO监督器覆盖venv launcher与实际worker，exit0/no timeout，[退出CIM为空](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_prefix_geometry_v1_watchdog/post_terminal_CIM.json)。没有新GPU或长训练，原用户checkpoint健康后暂停条件本轮未触发。

[17项保存产物独审](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_saved_independent_review_v1/report.json)核对LP、terminal系数、16/61参考、122约束、raw contacts、计数、检查点/PID/hash，无新查询或求解。

## 终态接触与拟合取舍

[24项只读终态审阅](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_terminal_review_v1/README.md)给出精确违规时刻：0.140、0.145、0.265、0.295、0.300s，共5个时刻/7个pair-time超过3mm；最大为末端拇指tip，另有食指tip。16个拟合节点中2个也超3mm。手自碰未生成，不当作正间隙证明。

材料位置项占cost68.10%、软重叠24.90%、骨方向6.69%。拇指材料点误差均值4.282mm/峰7.489mm，法向均值61.19°/峰77.76°，不能称接触意图精确达成。强制范围/控制条件后仍存在拟合与几何的取舍，不据本地失败断言动作不可达或某一权重为唯一原因。

240条objective记录包含line-search trial，并非80轮major历史。当前未保存每轮完整最大违规/活跃pair/tie轨迹，不能用trial尖峰捏造分支切换或发散解释；下一算法应补齐这些日志。

## 分包与最终控制核验

Root一次[最终导出](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_prefix_control_export_v1/receipt.json) SHA `7bb31be6bd462fe5f4a9a53e25e9e488c4350a2f575903fbf06796b76d12b541`：reference B样条、设计init、计划control分别有manifest。analytic qd/qdd与原生forwardFD分开，init采用新解析导数；物体世界角速度明确转为freejoint body角速度。init_measured/u_readback仍null，没有实际初始化资格。

最终15目标零C2限幅，14次计划速率峰11.448526rad/s。NumPy/Torch/原C2逐位一致；最终U32的全区间理想实数PD Bernstein证书和61点两种FP32分组预测均通过，二者与未来实际物理/FMA/WM梯度资格分开。实际[22项独立复算](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_control_independent_review_v1/report.json)不调用生产build/export函数，确认上述结论及五个几何违规时刻；track/hold Bernstein距交集力限最小裕量约0.03107/0.03389Nm。

已完成的数学模块/21项检查保留不变。Root新[导出器](D:/workspace/supermimic-observation-constrained-prefix-study/scripts/export_constrained_prefix_cpu.py)在执行前经独立审阅，补齐输入SHA强制检查，随后只执行一次真实导出；不是LP/非线性重试。导出exit2为已保留的几何门槛失败，非工程崩溃。

[诊断图](D:/workspace/supermimic-observation-constrained-prefix-study/outputs/constrained_prefix_control_export_v1/constrained_prefix_diagnostics.png)清楚区分计划与实测，Root已实际查看。图及导出不新增几何查询、LP、优化、物理或GPU。

## 下一步

保持本协议和终态不变。下一方法应另立有限的**非线性可行性恢复→拟合**流程，仍保持原物体运动、线性控制合同和3mm资格标准，并预先声明初值、预算和terminal选择规则；不能仅延长本次80轮、改变软权重或挑历史迭代。需要记录每个major的cost、违规和活跃pair/ties，分别报告局部未证实与真正可行证据。固定材料对应、滑动/滚动/释放假设与观测不确定性仍需审阅。

通过CPU前置检查后仍须新的非零初态/scheduler/实际八参数/时序/容量资格及无辅助动力学验证。完整151帧、学习残差/CHORD收益、共享成本、真实ego与硬件合同仍未完成，Goal保持active。
