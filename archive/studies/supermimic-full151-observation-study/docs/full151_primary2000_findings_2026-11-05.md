# 原生 full151 固定 final2000：工程完成，物理任务失败

恢复 Goal 后，首先查询同一个后台 job 的终态及当前 CIM，没有重复启动训练。上一 Goal turn 属于实际 progress：新增真实105训练/评估并完成首100健康与估时。本轮进一步完成该 job 的最终审计、完整物理回放及观测布局疑点定位。

## 结果与范围

固定 final2000 一次完整回放的 **13项原质量标准中9项通过、4项失败**。这条参考来自单一 legacy source[200,351)，151帧/150控制间隔/3秒，几何与A0失败均保留。真实 ego 样本仍为0；本结果不能证明多轨迹共享、泛化或 CHORD 整体收益。

| 原质量指标 | A0 | final40 工程 smoke | 固定 final2000 | 原阈值 |
|---|---:|---:|---:|---:|
| 151点最大物体位置误差 | 38.002625mm | 565.710011mm | **35.834604mm，失败** | ≤20mm |
| 151点最大有向笔轴误差 | 60.249328° | 170.399547° | **74.308207°，失败** | ≤20° |
| 600 solver最大穿透 | 7.941423mm | 16.433617mm | **3.316579mm，失败** | ≤3mm |
| 实际目标最大变化率 | 14.999902rad/s | 15.339008rad/s | **20.508619rad/s，失败** | ≤15rad/s |
| 601点关节最大越界 | 0.0000505mrad | 6.380567mrad | 0.0000505mrad，通过 | ≤1mrad |
| 掉落 / 物体非手接触 | 通过 / 通过 | 失败 / 失败 | 通过 / 通过 | 无 |
| 质量通过数 | 10/13 | 6/13 | 9/13 | 全13项 |

final2000 相较A0的位置和穿透误差下降，但笔轴误差更高、目标速率出现超限，因此不能称整体改善。速率最大值来自后续149次更新，启动差分为11.556205rad/s；完整口径包括启动差分，共150次。A0的启动差分为0，不影响其最大值。没有事后放宽质量标准、截取好区间或挑选中间 checkpoint。

![固定检查点完整曲线](D:/workspace/supermimic-full151-observation-study/outputs/full151_fixed_checkpoint_comparison_v2/full151_fixed_checkpoint_comparison.png)

[PNG](D:/workspace/supermimic-full151-observation-study/outputs/full151_fixed_checkpoint_comparison_v2/full151_fixed_checkpoint_comparison.png) · [SVG](D:/workspace/supermimic-full151-observation-study/outputs/full151_fixed_checkpoint_comparison_v2/full151_fixed_checkpoint_comparison.svg) · [9个逐时CSV与汇总](D:/workspace/supermimic-full151-observation-study/outputs/full151_fixed_checkpoint_comparison_v2/summary.json)

图从保存数据重算27个数值，均吻合。Root实际查看v1发现symlog自动坐标产生无数据的负半轴；新v2仅将位置误差下界设为0，原v1字节及输出保留。所有数据非负，未隐藏数据点或变动阈值。

## 训练终态与最终 checkpoint

训练原配置 SHA `fe2c7bd04a7e27af248c97f71d8aa13a5bbd98e1bfd2023ab7b93b30983adac9`；固定 final2000 SHA `36c90743f678f5f4d02ba3beb72091f1c59ccb3a0958ad16fcb6bbf97b122a21`。实际从零初始化，未读取 smoke 或旧104权重作为训练起点。

2026-11-04北京时间16:07:44结束，worker1708.493894s，watchdog1709.765s。实际648192条转移、WM4000/failure4000/Actor1800/critic7200次更新、2001次collect/2000行metrics。独立105最终健康 **163/163 PASS**，`final_iteration_scope=True`。单独生命周期/来源/上下文审阅 **91/91 PASS**，147 pins/ZIP字节一致，actual8双读、xi128×46及解析初态一致。

150个相位全部实际访问，phase149/natural-end2498，physical termination5242；二者不能简单互斥相加。3346次partialreset涉及7724个world-row。容量共3370条读取：constructor1、before-reset3347、checkpoint21、final1，raw OR0/unknown0，finalized。观察窗从Env构造完成后开始，构造/graph之前的整个进程容量及API次数仍为UNKNOWN。

原四层身份链16444→29188→30948→34468与保存的creation/命令/config/source一致，终态watchdog exit0、无timeout、精确owned worker/launcher不在CIM。恢复时新增CIM也未发现本项目Python worker。不把lock/state文件存在当作活进程证据。

首100估计北京时间16:08:28结束，实际提前44.592秒；从首100锚点算的剩余时长高估约2.74%，落于原16:03–16:17经验范围。这是单次回顾，不构成估时精度保证。

- [最终健康163](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_primary2000_health_v1/report.json)
- [独立生命周期91](D:/workspace/supermimic-full151-observation-study/outputs/primary2000_lifecycle_independent_v1/report.json)
- [ETA复盘](D:/workspace/supermimic-full151-observation-study/outputs/primary2000_lifecycle_independent_v1/eta_vs_actual.json)

## 观测疑点：审计布局假设错误，未发现一步滞后

旧 smoke 保存后审计178/179的额外断言错误地把policy155前20列当作实际q。原生产代码、checkpoint及该失败审计均未修改。新源码与保存数组诊断 **13/13 PASS**，0-based布局为：

| policy155区间 | 内容 | 保存数据对照 |
|---|---|---|
| [0,20) | 当前相位的参考q | 150行逐位相等 |
| [20,26) | 参考物体两轴点 | 独立几何算术核对 |
| [26,89) | 参考BODY21×3 | 150行逐位相等 |
| [89,109) | 当前实际q | 与preaction raw q[4k]逐位相等 |
| [109,129) | .25×当前实际qd | 与同步速度逐位相等 |
| [129,135) | 当前物体两轴点 | 独立几何算术核对 |
| [135,155) | 上一次action | 首行0，其余逐位相等 |

首帧参考q恰与实际q相同，掩盖了旧断言的错误；后续差异不是滞后一拍，也不是策略观测和状态58布局相同。轴点几何的FP32算术差单列，未修改物理质量标准。另有4个用于解释原生观测组装的已安装SDK文件在新诊断中pin，但未在旧152项归档中；不回填旧归档或声称旧执行已冻结这些额外文件。

[观测布局诊断13](D:/workspace/supermimic-full151-observation-study/outputs/full151_policy_observation_layout_diagnostic_v2/report.json)

## 固定 final2000 原生回放证据

新评估配置 SHA `10902fe66b7e1ec04d7f039c2d8528ec33a4735e57e484e65f9a3d7ced47c1b9`，154 pins。默认CPU预检无Torch/Warp；Root显式运行一次，retained session43595已exit0，不重跑。worker23544/launcher1584，OS身份12/12。worker37.591897s，完整150 action、600 solver、601 states、150原Manager forward/reward capture，质量失败仍保留完整3秒，无中途reset/外力/物体辅助。

在CUDA初始化之前严格CPU加载105 Actor155→20。保留原训练root descriptor与controller signature；Actor/norm/checkpoint不变，optimizer0、Runner实例0。601 clock逐位符合FP32递加，终值3.000021219s，漂移21.219µs在预先冻结50µs界内；旧2µs仅diagnostic。构造/reset/graph的整个进程API次数仍为UNKNOWN。

独立新saved auditor **182/182 PASS**，从raw/solver/events重算13项原gate与9个数值，以及150步FP32 EMA/C2/cache/rawctrl、真实155布局、source/ZIP/OS/容量/health/root binding。Actor内存状态是否不变依据保存digest及源码，独审未重新推理/恢复Actor原始内存。

- [完整原始评估](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_policy_eval_primary2000_v1/evaluation.json)
- [保存数据独审182](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A1L_primary2000_eval_saved_independent_v1/report.json)

## 下一步与未完成项

保持Goal active。当前baseline已可评估但任务仍失败，不自动训练更多随机种子来择优。接下来接通独立full151接触意图/支持扳手目标、同预算P_ONLY/A2，并检验世界模型的接触/动作梯度；新目标须独立于任何policy接触，unknown不能当release或人类真值。

正在并行建立“持续H123、逐帧source surface projection”的显式探索假设与固定CPU WM诊断。该假设不等于视频观测到接触；非零support不等于力闭合或能满足运动所需扳手。旧cue仍不可加载为objective，不因本轮baseline结果改mask/阈值。多轨迹shared、独立真实ego和真机观测/子版本契约仍未完成。

## 已完成的新105世界模型CPU诊断

严格119f consumer首次加载真实fixedfinal2000 WM，使用新health163/trusted1fb9；FP32 CPU、AMP关闭、参数冻结。固定A0与final2000两条完整150-step保存路径，state58+actualu20输入，内部加xi46和零174 disturbance，不重复C2。300个一步预测与固定8个状态×4个one-hot动作方向×2个epsilon，共64个模型内部AD–中央FD对照。未创建Env/Warp/optimizer、未添加物理步或GPU。

工程/有限值/来源/normalizer/checkpoint/RNG不变性 **23/23通过**，没有用预测误差或FD容差作为通过标准。下面是保存诊断量，不能把23/23称预测器或动力学梯度合格：

| 一步误差口径 | A0路径 | final2000路径 |
|---|---:|---:|
| q分量RMSE | 19.653mrad | 8.885mrad |
| qd分量RMSE | 0.7671rad/s | 0.4392rad/s |
| 物体位置3D误差范数RMS | 1.8914mm | 1.0511mm |
| 完整SO(3)角误差RMS | 4.2333° | 3.8575° |
| 物体线速度3D误差范数RMS | 0.07853m/s | 0.05782m/s |
| 物体角速度3D误差范数RMS | 4.4834rad/s | 3.1305rad/s |

SO(3)误差不是上表物理任务有向笔轴20°标准。reward snapshot和after-Manager两种保存state本次逐项相同，不当作双倍独立样本。两路径来自同一lineage，后者为训练后策略路径，不能据此证明泛化。

self15只代表原生手自碰撞标签，不是手物接触/力。A0有104/2250正标签，F1≈0.394；final2000有24/2250正标签，但TP=0、F1=0，BCE较小不能解释为正类识别可靠。self15不是新几何意图目标的来源。

WM内部AD–FD相对L2差：epsilon1e-3 median1.01e-4/max4.04e-4，epsilon5e-4 median1.80e-4/max6.08e-4。完整DELTA52向量的范数混合物理单位，仅作数值bookkeeping；没有同Warp物理分支有限差分，因此尚未证明动作/接触的物理梯度正确或有用。

[实际WM CPU诊断](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_WM_CPU_diagnostic_v1/report.json) · [逐组单位与范围](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_WM_CPU_diagnostic_v1/metric_definitions_and_scope.json)

## 新独立full151声明意图数据

固定新假设为“持续H123，逐帧source closest-surface projection”，完全独立于A0或任何learned policy接触；不读取robot q来定义目标。151源帧各自计算，非重复16帧或frame0材料点transport。geometry-valid和declared-H123 mask分列，人类接触/置信度仍全unknown，没有known-release或禁止额外稳定手指接触。H1/H2/H3源点距离表面分别约15.74–21.58、2.05–8.71、5.94–27.96mm，不能称观测到真实接触。

新数据生产 **17/17通过**，另Root独立纯NumPy重建 **15/15通过**，support512最大差4.44e-16。453个声明槽均几何有效；每帧512方向有499–512个大于1e-3，训练64子集有59–64个，因此数值不退化为P-only。H3 BODY法向相对frame0最大变化179.61°，表明不是固定法向复制。

COM点先减COM，力方向取外法向的负号。支持扳手为单位摩擦锥ray的凸支持指纹，mu.1/8edges+normal，moment只除一次nominal MJB COM enclosing radius **0.129045407685m**，没有误用7.45mm截面半径。该mu和unitray不是实际摩擦/力幅值；非零支持不证明力闭合或能支撑解析轨迹所需扳手。

原cue与本次producer manifest的stage权限均未修改；新adapter通过独立spec/CPU enrollment接入，不能回填旧声明为可训练。旧proxy局部与compiled精度失败保持，新的CPU buffer/导数合同也不能消除那些边界。

[新目标与方案索引](D:/workspace/supermimic-full151-observation-study/proposals/full151_observation_H123_support_v1/RESULT_INDEX.json) · [生产17](D:/workspace/supermimic-full151-observation-study/outputs/full151_declared_H123_surface_targets_v1/receipt.json) · [独立NumPy15](D:/workspace/supermimic-full151-observation-study/outputs/full151_declared_targets_numpy_independent_v1/report.json)

## 关键负例：非零支持指纹不等于能满足所需扳手

一次预先冻结的151帧LP诊断使用完整解析加速度/角加速度/角速度/旋转，以及固定实际mass/inertia/iquat和MJB重力，在物体COM BODY坐标计算：

`F_o = R^T m(a-g)`，`tau_o = I_body alpha_o + omega_o × I_body omega_o`。

随后独立NumPy构造H123每帧27条声明摩擦锥ray，求 `[ray_force; ray_moment/radius] lambda = [F_o; tau_o/radius]`、`lambda>=0` 的最小sum(lambda) LP。151次调用，无重试、无力上限、无修改target/mask/阈值。HiGHS primal/dual1e-9和原单位等效N残差1e-7门预先冻结；不是依据失败结果反调参数。

结果 **23 feasible、128 solver-infeasible、0 unknown，frame0不可行**。可行帧的等效N最大残差6.05e-14，raw torque最大残差9.01e-16Nm；非负见证sum(lambda)约0.93–40.59N。这只是无幅值约束的数学分解，不能解释为可用手指力或实测力。实际mass约0.03kg，重力 `[0,0,-9.81]`。

该结论仅针对声明的3个投影点、mu.1、8edge多面锥和当前解析参考；没有Farkas证书，也未证明真实连续摩擦锥、其他手指、不同接触或完整物理系统不可行。尤其不能反向把128帧称为人类release。它足以否定“每帧support非零，因此全段接触动力学可行”的推论。

因此当前H123全151非零weight版本仅继续CPU表示/导数接口验证，不直接升级为合格A2监督或启动新长训练。后继方案需显式区分几何声明与所需扳手有效性；只在23帧赋support权重可作为中间ablation，但不算解决其余128帧，也不缩短完整3秒benchmark。全部未知帧的P项与完整物理评估必须保留，力幅值/执行器/手部可达性仍需解决。

[151帧LP结果与矩阵/见证](D:/workspace/supermimic-full151-observation-study/outputs/full151_declared_H123_dynamic_cone_LP_v1/report.json)

## 独立新adapter的CPU合同已完成，未接训练

唯一CPU-ready接口为 `full151_intent.binding_v2.build_contact_objective(spec, device='cpu')` 与 `build_penalty_only_objective`。新spec schema为 `full151_declared_H123_objective_spec:1`，独立pin生产17、Root纯NumPy15、输入bundle和原始binding/target。原producer的stage flag False保持，另以CPU spec验证此新constructor；没有升级为GPU或训练许可。

constructor直接调用 `nn.Module.__init__`，跳过旧kinematic-reference manifest构造器，保留冻结的forward/score/geometry/P公式。注册151×512/64目标、phase/mask等8个顶层buffer，host模型只拷实际geometry三字段，不上传physics。32/64个保存状态、eager输入autograd、P_ONLY与A2公式分离、0/149/150边界、unknown只关闭吸引而保留深穿透P、CPU autocast、RNG与source不变性、错16帧/旧字段/伪人类truth/篡改拒绝等 **31/31通过**；nativequery/Env/Warp/WM/GPU/optimizer/物理均0。

首版 `binding.py` 实际CPU到达A2构造及saved32 forward/backward后，P_ONLY wrapper因新metadata漏 `scoring` 而KeyError；source/spec/failure全部保留。新 `binding_v2.py` 只补兼容metadata，未改数学、目标、mask或LP结果。不可隐式使用失败v1，也不把CPU静态几何导数有限解释为实际动作动力学正确。terminal150目标确实存在但奖励/bootstrap/critic仍必须由独立runtime的 `phase<150` hardmask排除。

[唯一active API与版本说明](D:/workspace/supermimic-full151-observation-study/src/full151_intent/README.md) · [CPU31](D:/workspace/supermimic-full151-observation-study/outputs/full151_intent_binding_CPU_v2/report.json) · [最终CPU READY与LP负例](D:/workspace/supermimic-full151-observation-study/outputs/READY_full151_intent_binding_v2_final.json)

## 下一项可执行研究路径（尚未运行）

保存LP的独立代数审阅9/9重建demand/矩阵/指纹/23帧非负见证，未新增LP或物理查询。后继一页草案固定H123与H12345两种source-only投影假设，使用同一套实际材料参数，最多151×2=302次CPU LP；完整151帧及物理门禁保持，禁止扫mu/择优/自动改target或训练。

执行前须先从固定actual context/MJB与对应版本MuJoCo/Warp源码核对priority、condim、solver cone及摩擦混合，才能冻结实际生成矩阵与脚本。草案约mu_slide.7、mu_torsion.005m和condim4是待逐字段核实的预期，不能当已验证材料事实。尤其pyramidal/elliptic耦合方式和torsion上界须按源码构造，不能把独立无力约束扭矩ray或任意Cartesian组合当作原生接触锥。即使用实际材料值，source投影的规范切平面仍是声明假设，不是观测到的原生contactframe。

此比较用于区分“三指限制/表示参数”与更完整全手假设，不以更高LP覆盖率自动选择可用监督。即便全151可行，真实接触、力幅/执行器、可达性及连续动力学仍需后续source-bound联合修正与真实物理回放。Goal继续active，本轮没有新增长训练或新的暂停事件。

[下一步一页草案](D:/workspace/supermimic-full151-observation-study/proposals/full151_declared_intent_dynamic_mask_and_all5_v2/NEXT_CPU_ONE_PAGE.json) · [LP保存代数独审9](D:/workspace/supermimic-full151-observation-study/outputs/full151_H123_cone_saved_independent_v1/report.json)
