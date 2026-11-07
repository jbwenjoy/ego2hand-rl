# full151：实际材料对照与三方法学习接入

本轮继续既有完整Goal。上一turn为实际progress，完成fixedfinal2000物理评价、WM CPU诊断和H123表示锥负例；本轮没有重复该训练或物理回放。以下区分已执行结果与尚待运行的学习接入。

## 302项固定对照：两组均151帧有条件可行

先核实同一actual8/MJB与本机MuJoCo Warp源码。21个enabled手部几何与3个object几何的63种手物材料配对，手priority1高于object0；固定模型为PYRAMIDAL、condim4，无explicit-pair或override。五维实际FP32摩擦参数为 `[.699999988079071,.699999988079071,.004999999888241291,.00009999999747378752,.00009999999747378752]`。后两项rolling参数不进入condim4，本次torsion参数单位为m。

原生金字塔每接触6列：所有列法向力系数均为1，分别带滑动±mu_t1、±mu_t2或扭转±mu_spin。扭转列仍携带法向力与接触点的lever moment。`sum(lambda)`是总压缩法向力，不将力矢量归一成unit ray，不添加无力上界的独立纯扭矩ray。此区别不能被“8个摩擦边缘与扭转的任意Cartesian组合”替代。

source-only全5 cue由同一151×21保存矩阵逐帧抽取 `[3,7,11,15,19]`，755槽均几何有效；H123为其前三槽，所有旧H123共享字段与dtype逐位一致。人类接触与置信度仍全部未知，declared-potential mask与几何有效mask分列。这些不是已观测到的五指接触。

冻结授权配置SHA `1fee2990715e6a5a2a7ae351ebf67ee41633b5bf014a10f71f0deb35ce0b17f8`、63pins/58unique preexecution ZIP。Root一次实际302 LP，frame-major `[frame0H123,frame0H12345,...]`，same151 analytic demand/actual mass-I-gravity/material/规范Frisvadframe，HiGHS primal/dual1e-9、原N等效残差1e-7。无force cap、重试、扫mu、替换cue或择优选择。worker44492/launcher32800，3.587827s；watch3.797s、exit0、OS身份12/12，owned PID终态CIM为空，无GPU/Env/物理/WM/optimizer。

| 固定同材料假设 | 有条件可行 | 不可行/未知/未跑 | 求解器法向力目标 min/median/max |
|---|---:|---:|---:|
| H123 | 151/151 | 0/0/0 | .296054 / .545876 / 2.081405 N |
| H12345 | 151/151 | 0/0/0 | .160098 / .382385 / 1.709457 N |

独立保存数值审38/38：原lambda全部非负，最大scaled primal误差H1231.78e-15/H123455.55e-16N；raw torque误差均小于2.83e-17Nm。独立重建A差4.44e-16，analyticF精确、tau差8.67e-19Nm；每帧H12318列都是H12345前18列，字节相同。142帧五指目标值更低、9帧在1e-7N内相等，符合其可行集合包含三指集合的结构；不是五指抓握优越的实测证明。只独立核primal，未保存并独立复核dual最优证书，故将这些数值称求解器给出的目标值。

来源/归档/生命周期独审40/40核对63 current pins、58unique ZIP、实际worker/supervisor副本、OS创建/父子关系与302行覆盖。归档前Root首个路径比较误将`D:/`与`D:\\`的完整pin字典视为不同，实际60个源哈希均未变；失败记录保留，新比较按samefile+SHA，不改科学文件或重跑LP。supervisor v1默认预检的venv子树清理缺口保留；新v2仅修retained Popen+owned-tree cleanup，实际45s科学watch和LPworker数学不变。

旧mu.1、9个unit-force表示ray且无direct torsion的H123诊断仍为23/151，不修改或“升级”旧结果。此次H123在实际参数假设下同样151/151，因此不能把23→151归因于增加手指。当前结论只排查了某个静态所需扳手分解障碍；接触是否存在、手部能否到达、法向力上限、执行器/关节力矩、连续动力学和真实接触frame仍未解决。规范Frisvad切平面也是假设，有限pyramid对该方向敏感，不能冒称重放原生接触。

- [实际302结果](D:/workspace/supermimic-full151-observation-study/outputs/full151_matched_actual_material_cones_CPU_v1/report.json)
- [独立保存数值38](D:/workspace/supermimic-full151-observation-study/outputs/full151_matched_material_cones_saved_numeric_independent_v1/report.json)
- [来源/生命周期40](D:/workspace/supermimic-full151-observation-study/outputs/full151_matched_material_cones_lifecycle_saved_independent_v1/report.json)
- [实际材料合同17](D:/workspace/supermimic-full151-observation-study/outputs/full151_actual_material_cone_contract_v1/material_cone_contract.json)
- [全5 source cue18](D:/workspace/supermimic-full151-observation-study/outputs/full151_all5_source_cues_v1/READY.json)

## Python辅助解码器的维度差异（仿真库未修改）

Root额外的纯代数fixture首次报告写入遇NumPy int序列化错误，source/数值/failure保留；v2只修报告写入后得到3/9，而非据此修改约束数学。本机3.11.0 Python wrapper要求force长度=mu长度+1、pyramid长度=2×mu长度，却将C参数dim设成mu长度；C decoder按contact dim循环2×(dim−1)。两者的一个维度差与最后一对torsion被忽略的实际输出吻合。[3.11 Python源码](https://github.com/google-deepmind/mujoco/blob/3.11.0/python/mujoco/functions.cc#L1463) · [3.11 C decoder源码](https://github.com/google-deepmind/mujoco/blob/3.11.0/src/engine/engine_util_misc.c#L1471)。

新v3使用本机原DLL的显式C ABI dim4，仅7个纯解码调用、无模型/MjData/物理，9/9通过并pin实际DLL/pyd/version。没有修改SDK或驱动，也没有让LP依赖这个Python辅助函数；LP从原生constraint源码与独立NumPy模板构造。历史Warp约束及原生`mj_contactForce`调用不经过该Python mju wrapper，不因这项独立helper失败自动否定历史物理结果。官方matching源已只读归档，不执行下载的源码。

[源与实测边界记录](D:/workspace/supermimic-full151-observation-study/outputs/native_pyramid_binding_source_review_v1/report.json) · [显式C ABI代数9](D:/workspace/supermimic-full151-observation-study/outputs/native_pyramidal4_algebra_CPU_v3/report.json)

## 三方法接入：CPU合同完成，真实GPU待工程smoke

保留在五指对照之前就声明的H123目标，不根据五指更低的求解器目标值选“胜者”。目标/代理的mu.1仍是共同几何表示参数；实际材料cone证据独立保留，不把静态分解当接触/forceclosure/手部可达性或WM物理梯度合格。

新增隔离版本，不修改已执行A1L/controller/runtime/capacity或旧checkpoints：

- `full151_intent.binding_v3`：spec:2、CPU/GPU显式阶段、完整factory/import/device RNG契约，原forward/score/P/151目标不变，CPU48/48。
- `full151_native.learning_v2` + `prior_hook_v1`：schema2三个method、原128world/H8/预算；四模块rawinit/fingerprint之后首collect之前构造objective，并比较整个import/build/to前后CPU及已初始化CUDA RNG。A1L不导入/不实例化目标；P_ONLY只P，A2为同P+support。仅policy想象soft=True奖励加`.02*bonus`一次；native phase<150 hardmask、real collection、WM fit、优化器顺序均保留。CPU68/68，尚无真实schema2 Env/GPU训练证据。
- `audit_full151_checkpoint_cpu_v2.py`：仍消费新105，严格schema2、三method、prior counters/metadata/spec/RNG证据、WM-only prefix边界，CPU59/59；实际schema2检查点审计须等待产物。

Root新worker/builder/supervisor保持默认metadata noTorch/Warp，绑定当前CPU source/report与CPU→GPU spec唯一flag差异。三个method将用同source table、seed0、物理context、controller、预算和独立匹配的四模块原始初始化；未读取旧权重warmstart。不把工程smoke或训练集结果称真实ego收益。目标SCP/A0/final2000失败、旧proxy/compiled精度失败均保留。

下一步先跑固定40iter/WM-only20的三方法有界GPU smoke，独立核对各20992真实转移、80/80/20/80次实际更新、P/A2每policy H8的prior调用和RNG、无real/WM shaping、105保存/恢复与完整3秒回放。只有真实三方法工程和配对初始化验证后再进入相同2000iter预算。若启动新长训练，仍按用户条件在首健康checkpoint和ETA后暂停Goal，并保留已授权后台job。

[binding CPU48](D:/workspace/supermimic-full151-observation-study/outputs/READY_full151_intent_binding_v3.json) · [learning CPU68](D:/workspace/supermimic-full151-observation-study/outputs/full151_prior_learning_preparation_v1/READY.json) · [health CPU59](D:/workspace/supermimic-full151-observation-study/outputs/READY_full151_three_arm_health_v2.json)

## 后续实测：三方法 GPU 工程通过，任务均未通过

上述“待GPU”阶段已完成。冻结同一486源文件表、seed0、128worlds、H8、物理context与控制器后，Root依次完成A1L、P_ONLY、A2三次40iter/WM-only20训练。三组均从零初始化，实际20992真实转移、WM80/failure80/Actor20/critic80；无重试。三组各自OS身份12/12、exit0、终态owned PID不在。固定final40保存健康审计分别118/118、136/136、136/136通过。

独立原始初始化配对75/75：三组四个模块的参数名称、顺序、shape、dtype、逐参数字节完全相同。A1L没有实例化目标且prior计数为零；P_ONLY/A2均20次policy scope、160次H8 forward和160次dt应用，active/inactive分别20412/68与20404/76。完整factory import/build/CUDA迁移前后五组RNG状态与摘要一致，没有把prior写入真实collection或WM fitting。首次配对审计v1因`D:/`与`D:\\`pin整字典比较提前失败、未加载rawinitial；该失败与源保留，v2仅修路径身份语义后完成真实配对。

每组固定final40各执行一次完整3秒、150控制区间、600solver行、601状态回放，优化器更新零；没有按质量提前结束或选checkpoint。原13门槛保持不变。

| 方法 / final40 | 通过门槛 | 位置最大误差 mm | 方向最大误差 ° | 最大穿透 mm | 最大目标速率 rad/s | 掉落门槛 |
|---|---:|---:|---:|---:|---:|---|
| A1L | 10/13 | 39.4291 | 57.5932 | 7.3556 | 14.8884 | 通过 |
| P_ONLY | 9/13 | 41.0186 | 58.5131 | 5.2994 | 15.2125 | 通过 |
| A2 | 7/13 | 619.6770 | 140.2145 | 17.7329 | 15.2245 | 失败 |

三组任务均失败。A2还违反无非手物体接触门槛，不能声称support先验改善操控、提高接触动力学符合程度或带来真实ego泛化；单seed短训练也不足以给出最终方法效应结论。旧schema1 A1L实验单独保留，不将它与新schema2三组混成同源配对。

[三组健康、配对与回放登记](D:/workspace/supermimic-full151-observation-study/outputs/full151_v2_three_arm_smoke_enrollment_v1.json) · [原始初始化配对75](D:/workspace/supermimic-full151-observation-study/outputs/full151_v2_three_arm_smoke_pairing_v2/report.json)

## 长训练预注册及元数据预检优化

先冻结三个相同2000iter/WM-only200/每100保存的primary配置，预期各648192真实转移、WM4000/failure4000/Actor1800/critic7200。仅A2拟在本轮启动，A1L/P_ONLY保留配置、恢复Goal后再执行，没有后台自动队列。此阶段顺序不依据final40质量选胜者。

初版v2 primary元数据预检三组均实际通过，noTorch/Warp/GPU/physics；但其循环对1245个smoke Python源成员逐个线性扫描495pins并反复Windows `Path.resolve`，耗时超过supervisor的20秒预检界限。没有用其启动GPU或放宽时限。新增v3入口仅把成员查询预先索引为(resolved path, SHA)集合，每次仍先重新检查文件SHA；保留旧入口和初版配置。学习模块、目标、controller、runtime、优化器与预算不改。新v3三个supervisor默认CPU预检各约1.5秒通过，498pins，原20秒界限保留。

[v3共同primary协议](D:/workspace/supermimic-full151-observation-study/protocols/full151-v3-three-arm-primary2000-v1.json) · [A2冻结配置](D:/workspace/supermimic-full151-observation-study/configs/full151-v3-primary2000-A2-seed0-v1.json)

## A2 已启动，首检查点健康与 ETA 已实测

Root于UTC2026-11-04 19:44:18.650（北京时间11月5日03:44:18）仅启动一次A2 primary，后台worker33452、父38532、完整链15644→41436→38532→33452。首次CIM和后续独立CIM的创建时间、父子、命令、可执行文件与worker/config/source完全绑定，实时交接检查37/37通过。A1L/P_ONLY两份primary配置尚未运行或排队。

实际`model_100.pt` SHA `6efee26549d967ba32df89599b7f425d74d397be454b7c8d5e189235becaee89`，独立健康116/116通过；固定前100范围40192真实转移、101collect、WM200/failure200，Actor/Critic/prior调用均零，final_scope=False。容量prefix132次观测/130次pre-reset中overflow/unknown均零，仅是该前缀范围。新A2 primary与对应smoke的原始四模块参数另作25/25逐参数字节比较，完全一致；该项不加载训练后权重。三方法回放的独立保存审核亦已实际完成574/574，原13项质量失败判定均吻合。

checkpoint100文件实际mtime为UTC19:45:46.416286，metadata随后17.126ms，SHA/iteration与健康报告绑定。ETA使用本次WM-only末50步平均0.633114s/iter、同A2 smoke去前3次后的17次policy平均0.499047s、保存均值0.060849s，加显式20s余量，估计从checkpoint起2122.4s完成：**北京时间2026-11-05 04:21:09**，启发式范围**04:13:49–04:32:19**。这不是保证或置信区间，后续policy/reset/机器负载可改变耗时；未将native局部timer误加到worker启动时间。

按用户已授权的“新训练首健康checkpoint及ETA后暂停Goal”条件，先保存本地交接，再调用暂停工具。后台原A2进程继续由原7100/7200s watchdog管理，无自动重试；恢复Goal时先核原进程/终态产物，不能因观察中断重复启动。Goal尚未完成：A2最终健康/固定final2000回放、另两臂同预算对照、物理接触梯度、多轨迹与独立ego验证仍待后续。

[checkpoint100健康116](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_primary_A2_ckpt100_health_v1/report.json) · [实时身份与ETA37](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A2_primary100_live_pause_v1/report.json) · [primary/smoke原始初始化配对25](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A2_primary_smoke_initial_pairing_v1/report.json) · [三方法回放保存574](D:/workspace/supermimic-full151-observation-study/outputs/full151_v2_three_arm_smoke_policy_saved_review_v2/report.json)

[完整151帧法向力目标与差值科学图](D:/workspace/supermimic-full151-observation-study/outputs/full151_matched_material_cone_figure_v2/matched_material_cone_objectives.png) 附SVG与逐帧CSV，仅重绘已保存LP数值，未重解LP。图面已实际查看，6项数据检查与视觉QA通过。v1 receipt因NumPy bool序列化失败保留，v2仅修记录写入，PNG/CSV与v1逐字节相同。
