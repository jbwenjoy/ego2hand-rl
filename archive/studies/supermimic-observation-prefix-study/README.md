# 连续16帧参考：单次CPU构造与失败资格记录

2026-11-03：完成一条新的H123 prefix16，首帧固定、后15帧联合求解。几何节点、原生BODY21 ABI和控制代数各有独立审计，但连续曲线/控制前置门槛失败。**没有运行新动态物理实验、GPU或训练。** 这不是完整3秒转笔或真实ego结果。

| 检查 | 本次结果 | 资格 |
| --- | ---: | --- |
| 16个q节点的硬限超量 | 0 | 节点通过 |
| 原生生成手物深度峰 | 5.141163mm | 超过3mm；12/16帧超限 |
| 连续q曲线最大硬限超量 | 15.626615mrad | 曲线需在硬范围内，失败 |
| 14次计划目标更新速率峰 | 28.484651rad/s | 超过15rad/s，失败 |
| Tracking参考状态原始PD力矩预测/力限峰比 | 2.673565 | 失败，非实测饱和 |
| Hold参考状态原始PD力矩预测/力限峰比 | 2.508752 | 失败，非实测饱和 |
| NumPy/Torch/原C2的FP32代数 | 逐位一致 | 仅纯代数通过 |
| 原生reader的16帧q/逐clip FD速度 | FP32精确一致 | 仅ABI工程通过 |

连续参考的零余量硬范围检查与未来实际状态的1mrad容差是不同合同。这里没有实际q/qdot、接触力、施加目标或物理startup读回；`u_exec`数组表示冻结的拟执行C2目标，不能称已经执行。

## 输入与唯一求解

来源为受控clip007旧窗口[200,351)的前16帧[200,216)，物体导数使用已冻结全151帧rawFD在这些节点的值。它与共享十条目录中的clip007[0,151)不同。相机、尺度、解剖和接触置信度仍未知，独立ego N=0。

从允许的time/21手点/objectpose重建骨方向及固定掌映射，原539d产物仅提供经独立重建一致的R/t、物体位姿和目标点；未读取其q1..15。Canonical H123 q0在300个自由变量之外硬固定，初猜全部重复q0。三指材料点、物体局部锚点及法向固定为假设，未补接触/力标签；物体与腕不优化。

实际[计划](D:/workspace/supermimic-observation-prefix-study/configs/prefix16-geometry-authorized-v1.json) SHA `52336e48ea0a5e2e74215d930da4a4c15255cbdf9cb8b395cc221f02ee7c5b6f`，绑定[原协议](D:/workspace/supermimic-observation-dynamic-wrench-study/proposals/short_dynamic_tracking_v1/protocol.json)、源码和输入。仅一次TRF/稀疏LSMR，预算80次函数评估、110秒worker软限和120秒owned进程启动期限，不换初值/权重/平滑方法或补跑。

实际46次fun、37次jac、36次callback，`ftol`终止status2；cost224.711397、optimality1.465669，因此不能称梯度收敛或全局最优。全部16节点保留。SciPy只把自由变量的初始边界值向内移动约1.047e-10rad，原q0逐位不变。Formal共706次CPU原生碰撞几何调用（内部初态16、formal_fun690），无step/forward/inverse/力或距离oracle调用、无Torch/Warp导入。外部CLI预检的16次以及owner/独立小FD检查另列，不扩实验N，也不混入formal NFEV。

[候选](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_geometry_v1/candidate.npz) SHA `1e67d99b25b6d991fa8e5209f521ea5c269b3212a45b054fe7353ab5b7277ae4`。原始[报告](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_geometry_v1/report.json)、[完整manifest](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_geometry_v1/manifest.json)、native_contacts、全部迭代/碰撞事件、latest/accepted检查点均保留。稀疏/光滑块与时间块小FD通过，原生碰撞分支、并列项和数值导数误差仍未全面合格，不能把CSR检查当全接触Jacobian证明。

## 连续曲线、初态与控制分包

新q的连续导数使用冻结的中心二阶/端点单边二阶模板，另存原生forward-FD/末行复制的速度。两者不同，设计非零初态使用前者；未以旧零速rest替代。物体平移为quintic Hermite，旋转为SO(3) quintic，保持冻结姿态/世界角速度/角加速度节点。所有曲线端点数值检查通过，未通过的范围检查仍保留。

静态负载先验加参考速度的PD阻尼补偿生成计划绝对目标，再作同一FP32 C2.020。15个拟执行目标覆盖15个20ms区间，第16个仅诊断；14次更新不等于16次命令。300个计划目标分量中22个被限幅，两条件初始u0一致且tracking非恒定。此代数没有求解改变中的惯性/接触力，也不证明Warp、WM或接触梯度资格。

最大曲线越界发生在thumb `right_finger1_joint4`，287.013727ms，q=−1.062626615rad，小于下限−1.047rad；16节点却全部合法。t0设计导数约−8.66e-7rad/s指向下限外，它不是这次15.63mrad最大越界的唯一解释。最大计划目标速率发生在thumb joint3、280ms更新。Tracking/Hold的原始PD力矩超限均是参考q/qd上的预测，模型实际执行时是否限幅及真实状态表现仍未测。

完整三包与数值失败见[导出receipt](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_curves_control_export_v1/receipt.json)。独立不调用生产build/export函数，重算多项式全部实驻点、计划速率、C2和PD，19项检查通过并确认四项数值门槛失败，见[独审](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_curves_control_independent_review_v1/report.json)。

[最终图与证据索引](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_curves_control_export_v1/post_export_interpretation_v1/FINAL_PINS.json)保留原图，另存只改呈现的版本：14个更新只在20…280ms标点，明确为计划目标；未重算任何候选或导出数据。Root与owner均实际查看图。

## 原生ABI与进程证据

另行一次[BODY21导出/真实CPU reader](D:/workspace/supermimic-observation-prefix-study/outputs/prefix_native_ABI_v1/report.json)通过，16帧q和原生FD速度精确匹配，BODY误差约3.81e-8m。它只证明数据接口，init_measured/u_readback仍null，不注册为共享训练就绪。

实际worker PID37020、launcher41752，OS创建`2026-11-03T07:19:14.983023Z`，runtime开始07:19:15.034471Z，二者分列。Worker2.153197s、watchdog2.266s为嵌套口径；GPU时间无此实验，wall不代表方法提速。原[watchdog终态](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_geometry_v1_watchdog/state.json)和[退出CIM](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_geometry_v1_watchdog/post_terminal_CIM.json)确认exit0、无timeout、两个owned PID均已退出。

[保存产物独审](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_raw_independent_review_v2/report.json)20项/43pins通过，直接由保存contact重建逐pair深度，不重跑几何/优化。初版审计的Unicode数组`equal_nan`比较错误保留，v2只修比较dtype分支。

实际使用[监督器v2](D:/workspace/supermimic-observation-prefix-study/scripts/run_prefix16_cpu_v2.py)，复用40b owned-tree watchdog和159c metadata IO。未用于正式任务的v1原型保留：自检发现venv launcher/真实worker分离，单launcher handle不足以覆盖进程树。自检v2又遇路径junction文本比较假失败，v3以samefile验证通过；全部是自身PID检查，未启动求解worker。120秒是子进程启动期限，附加清理有界，不能称整个supervisor wall严格≤120秒。

## 后续范围

[保存数据的失败审阅](D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_failure_review_v1/README.md)26项一致性检查通过。最深重叠是拇指tip_sensor，冻结的拇指材料点却在link4；frame5该点误差仅0.997mm，法向误差59.08°，单点接近不能代表整手无重叠。重叠项已占cost60.53%、材料位置29.68%；保存Jacobian的局部分支还显示二者方向冲突，因此不能简单归结为碰撞权重过小。这些局部代数不构成已合格碰撞梯度或不可达证书。

当前固定q0已生成2.183mm深度，下一版若把全部帧改成硬0.5mm就会与该初态矛盾。0.5mm软allowance和3mm资格门槛必须分开，任何初态/规划裕量变化另立条件。新节点q有限差分峰21.636rad/s也不能替代计划u速率28.485rad/s或未来实际读回。

本协议失败产物不改、不重跑。下一设计应显式处理连续q范围、计划目标速率/参考力限与接触几何约束，以及固定材料锚点假设/观测不确定性之间的冲突；不能只增加软权重，或从本地解推断轨迹不可达。粗几何参考也不自动否定学习残差修正路线，但任何替代控制/学习实验必须另立合同、保留本失败、回到实际无辅助动力学验证。完整151帧、共享训练成本、真实ego和硬件任务均未完成，Goal保持active。
