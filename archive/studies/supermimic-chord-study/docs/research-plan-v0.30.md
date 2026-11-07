# 研究方案 v0.30：三指几何可行、原生预载失持与动量接口

2026-11-07，承接[v0.29](research-plan-v0.29.md)及完整 Goal。上一回合和本回合均为 progress。本回合没有新神经网络参数训练，没有触发“新训练首个健康 checkpoint 与 ETA 后暂停”的条件；目标继续 active。

## 本回合改变了哪些判断

三个指定 tip 的名义几何接触可以在当前固定腕、真实关节范围和原碰撞资产下实现。本回合两个候选通过了全部几何门，第三指确实改变了姿态。这不等于已经实现接触承载：随后十二个无物体辅助的原生案例全部触发失持退出，没有一个完成原两秒检验。

当前应解决的是从几何参考进入实际载荷、再维持载荷的控制问题。不能仅增加第三个标签、继续增大法向目标或在这些失败前缀上多训 dense80 模型，便宣称完成 ego 到机器人的接触动力学重定向。

## 三指几何的实际执行

新分支指定 geom15/28/41，真正放开 q0..11，q12..19 保持原值；固定腕、模型尺度、材料、惯量、执行器和 solver 不变。两个旧 source pose 各配两个第三指目标，形成四个具名 entry。物体相对各自 source 的平移范数最多35mm、转角最多30°，这是算法功能映射让步，不是测得的 ego 误差界。所有原二指结果及失败不变。

第一次有限搜索实际执行243.373秒，691次完整几何返回；底层 static evaluator 开始695次，其中4次在阶段 deadline 中途停止。共5562次独立 cheap FK、5490次重凸查询，0 native forward、0积分、0学习。四个 entry 均保存 fresh final、原始候选和明确负 margin；0/4通过，全部 B 阶段为60秒预算停止，不能据此声称三指不可达。[搜索报告](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_support_v1/completed_v1/report.json)、[保存算术诊断](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_support_v1/saved_arithmetic_diagnostics_v1/report.json)

最接近的1mm算法分支仅剩 thumb link4 的非指定碰撞约32.09µm、index gap 上界超出106.77µm。因此新建小步几何修正，没有重复原有限差分搜索：物体 pose 和其余八关节固定，q0..11每步最多0.02rad，以实际 phone 支持平面的局部敏感性提议，再用完整22个 phone/table pair 和189个 self pair 检查每个试探步。非唯一支持集均值只作数值方向，不当作真实最近点、严格距离梯度或接触力点；未建 self/AABB 敏感性只是在提议矩阵中省略，绝不声称其物理导数为零。

新修正唯一执行8.012秒，27次完整几何、206次重凸查询、112次点 Jacobian、8次几何 QP、16次完整 line-search 检查，接受6步；0 forward、0积分、0学习。四个原失败 entry 全部保留，2/4 fresh final 通过原几何门，另外两个仍失败。[修正结果](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_projection_v1/completed_run1/report.json)、[保存独审10/10](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_projection_v1/saved_independent_v1/report.json)

| 几何通过的功能候选 | 相对 source 物体位移 / 转角 | thumb / index / middle 实际凸体间隙 [µm] | 手材料锚相对原两 ego 物点偏差 [mm] |
|---|---:|---:|---:|
| exact / middle-nearest | 18.50mm / 21.49° | +197.04 / −10.44 / −98.69 | 6.88 / 1.53 |
| 1mm算法 / index向COM移30mm | 30.94mm / 15.87° | +34.35 / +198.05 / +13.80 | 9.56 / 4.49 |

几何门仍为三 pad gap ∈[−100,+200]µm，其余 phone/table/self 完整非穿透。正间隙可以通过这个几何邻域门，却不代表 native contact 已经存在。功能材料点让步已显式量化，不能将这两个候选写成原 ego 接触点的精确复现。三点材料锚面积约702/1103mm²也只是几何描述，不是实际承载力臂或力闭合证明。

原姿态及四个搜索候选的[保存可视化](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_visual_v1/completed_v1/saved_pose_comparison.png)已查看：独立1次MJB、6次非积分FK和12次render，0 forward/积分。该图对应第一轮四个失败候选，不能误标成修正通过或持握成功图。

## 原生三档控制：十二例均未完成两秒

新动态 v3 保留原两秒、双 dt、5mm/5°终态、尾速度、穿透、关节、桌面、警告、无外力等门。新尾段要求三pad同时Fn>0.1N的覆盖≥95%，其他刚性 link 的最大每tick总Fn仍≤0.1N；并列旧二指判据，geom41在旧判据中仍属于 nonpad。初始 source 与每次模型、控制、contact/PRE及state/POST相位均明确保存。

两候选各运行预先声明的 ctrl=q、每pad1N、每pad3N三档，再各跑 dt=312.5/156.25µs，共12例。三个法向目标通过当前材料手锚 J 做 inverse PD，手反力为−Fobject，因此 `tau=bias−passive+Jhand.T Fobject`；无clamp，六条路线均在真实ctrl/actuator/joint-actuator界内。1N/3N是工程命令，不是实际接触力保证或硬动态力上限。

唯一实际执行exit0，6845个积分步、2次预载准备forward、12次初始forward、6次锚点Jacobian、6136次contactForce读取，0学习。12例均 `ABORT_LOSS_OF_HOLD`，退出前缀约0.11219–0.12609秒。退出时间不是稳定持握时长，1.5–2秒尾段不存在，尾指标保持null，双dt的短失败前缀相似也不构成两秒一致性通过。[原生报告](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_dynamics_v1/projection_completed_run1/report.json)

exact候选初始拇指Fn始终为零；另两指会因ctrl路线产生不同载荷。1mm候选初始三指Fn均为零，其后发生很强的 nonpad 碰撞，峰值每tick总Fn约29.8–34.3N。exact的ctrl=q/1N前缀nonpad为零，但仍失持；3N路线出现约5.9–6.0N nonpad载荷。三档目标不是单调改善，不能据“3N更大”推导更强持握。

保存曲线显示接触力间歇、COM持续漂移：[COM退出前缀](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_dynamics_v1/saved_description_v1/COM_abort_prefixes.png)、[native PRE载荷](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_dynamics_v1/saved_description_v1/native_PRE_forces.png)。两图已查看；统计仅来自12个保存trace，没有新物理或网络调用。实际载荷、接触时序和扳手分解决定下一步，不能把起始正gap认作唯一失稳原因。

独立保存审14/14复核了6845步、6124个tick contact加12个initial contact、实际PD与限力、三pad/old2pad分类、所有门和双dt前缀。actual PD/限力复算最大差2.22e−16N·m。exact的ctrl=q前缀三pad同时Fn>.1N覆盖为0，1N约4.69–5.21%，3N约8.04–8.08%；1mm所有档均为0。这些是退出前缀比例，不是原1.5–2秒尾段指标。[独立保存审](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_dynamics_v1/projection_saved_review_v1/report_v2.json)

按geom顺序符号和frame.T独立解码的world接触冲量表明：全部接触的向上Jz仅为对应重力冲量大小的约21.0–36.7%；该总量已包含不合格nonpad，不能全部计成合法三指支持。exact的1N/3N中拇指Jz为负，1mm多数向上支持来自nonpad。因此后续必须解决持续的指定pad扳手和COM支持；初载负gap不是唯一缺项。仿真world重力的确定性也不意味着真实ego源重力已注册。

本批所有路线的ctrl静态余量至少0.0440rad；原生实际actuator和joint-actuator力矩界余量分别至少0.1080/0.1783N·m，没有触硬力矩饱和。承载不足不能直接归因于力矩上限，也不能由尚有余量推导增大N就会成功。[保存审作用域与完整pins](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_dynamics_v1/projection_saved_review_v1/final_manifest_v2.json)

首次保存审13/14的唯一偏差来自批量quaternion点积与原逐row点积的末位归约，在acos近1处最大角差1.23e−10rad，超过审计的2e−12比较阈值。旧报告保留；修订仅按原逐row observer重放，12条post_angle逐字节一致，终态角度/COM指标差为零，没有放宽物理门或审计容差。该项是同算术observer复演，其余分类、扳手和门为独立保存复算；没有重做物理。

metadata与动态接线的失败保留：动态v2的JSON-only夹具发现 `candidate` 字段重复造成TypeError，尚未导入backend或读NPZ值；v3将内存内容改为candidate_data，原candidate pin不变，并通过16/16检查。四个失败候选不会因空pad metadata阻断其他候选，也不会被当作可运行状态。旧源和失败现场保留。[v3源码与接线证据](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_dynamics_v1/source_preservation_v3.json)

## 已可调用的动量接口

`phone_momentum_contract_v1/momentum_contract.py` 提供纯Torch可微函数，接收state58起终点、名义质量/完整BODY惯量、world重力、实际H，以及“冻结source BODY轴、绕逐时刻moving COM”的contact impulse6。分别输出world线动量[N·s]、角动量[N·m·s]残差，或在另给target R时计算条件速度。没有神经网络参数，不从冲量擅自决定完整p/R，也不把瞬时旋转BODY分量积分或固定source COM的力矩积分混用。

修订运行14/14：原TRAIN684完整过渡与独立旋转/非球形惯量、世界旋转协变、抛体和冲量输入AD夹具通过。与原导出残差最大差约1.67e−16N·s、3.90e−18N·m·s；TRAIN本身的残差RMS约6.64e−8N·s、1.04e−7N·m·s，没有强制清零。该证据是名义仿真接口和算术一致性，不是现实测力、物理动作Jacobian或新的模型预测精度。[接口与限制](D:/workspace/ego-data-pilot-2026-11-07/phone_momentum_contract_v1/README.md)、[验证结果](D:/workspace/ego-data-pilot-2026-11-07/phone_momentum_contract_v1/completed_run2/report.json)

首次数值计算已完成，但报告因NumPy bool JSON序列化失败；新版本仅转换报告标量，公式、样本和阈值不变，失败源/terminal/部分报告/数值文件全部保留。两次执行累计2次纯输入冲量Jacobian，0NN构造/更新、0原生物理。验证读取TRAIN state和已经公布的all960标签导出，后者包含已曝光TEST标签，明确不声称未见；没有读取原VAL/TEST state NPZ或拟合权重。独立SOURCE审另注明未来变质量batch须用scalar或(...,1)广播。

## 因果数据和完整SuperMimic/CHORD接入

已实作并执行TRAIN-only干预元数据生成器：72个保存源状态、全20轴，72 baseline加2880 side记录，其中2870可运行、10个实际du<1e−6不可用且保留。实际可执行2942分支、最多282432积分步；全计划含不可用记录的上界283392步。先全部72 baseline逐字节匹配，再任何side，避免上回合执行顺序偏差。完整300值source restore、原episode hard-stop参考、分组、真实duF64/FP32、失败/partial mask及全部source pins已生成，但**尚未采集任何新分支或启动训练**。[交付](D:/workspace/ego-data-pilot-2026-11-07/phone_causal_train_design_v1/delivery_v1.json)

6个响应fit family与6个新动作check family均来自原TRAIN，原状态已用于WM5000，因此check只测未训练动作响应，不能称未见状态/ego。另预注册12个真正新的初始化/控制曲线family1000..1011及双dt、最多23040步，尚未采集；仍是同两个已知物体源姿态和名义模型，不是独立真实ego场景。原476个VAL分支不进入TRAIN，原TEST不重新包装为未见。

输入契约只接受当前/想象状态的22对几何与接触相关特征；未来wholeH contact/decoded force/certificate属于监督信息，不能进入部署时forward。所有20轴结构上都有潜在phone接触路径，12个baseline整H无phone接触只是经验描述；严格zero mask全false。当前还没有fullH反事实路径证书或可微FK/几何provider，不能把schema实现写成完整几何模型已接好。[上游逐项接口映射](D:/workspace/ego-data-pilot-2026-11-07/phone_causal_train_design_v1/upstream_interface_map_v1.json)

raw80模型继续作为已暴露因果问题的诊断基线。新phone Reader、world-left52图表、几何/contact输入、Actor、reward、replay和CHORD仍须真正对接并做同预算比较；旧pen图表、旧四子步Reader和旧两pad力量头不可静默继承到三指。动量一致性接口也不能单独约束模型虚构的接触冲量。

## 下一项实际工作

先建立具名native loaded-entry，而非继续把几何邻域当作已承载初态。可把三pad负gap约−75µm作为求解目标，保持原hard几何窗[−100,+200]µm和其他非指定碰撞门；这是搜索策略，不是所有持握都必须满足的新成功门。fresh native要测实际三pad Fn、nonpad/table及真实ctrl/tau，保留所有失败与原始native读数。

两个通过几何的旧材料手锚距其指定phone材料目标仍可达毫米到厘米，因而 `J(A)` 不等于当前真实contact力点的Jacobian。下一步使用实际native `con.pos/frame` 的J和pad-only绕COM扳手来求有界预载/反馈参考，分开真实指定pad、其他接触以及重力贡献。法向目标、理想点LP或静态全qacc=0都不作为软接触动态持握的必要门；不能仅盲增N、放宽µ或靠非pad/桌面供力。

实际承载进入合同通过后，再运行原两秒/双dt/无辅助评估，必要时建立具名原生状态/接触反馈基准，并明确其模拟观测权限，不能混称已有Actor或现实可用策略。在有可执行物理参考后接三pad数据与上述因果/动量模型，再推进完整SuperMimic/CHORD同预算。

61帧时序映射、源重力注册、真实功能指腹、摩擦/质量/COM/惯量校准及鲁棒性、独立ego数据和硬件观测契约继续保留。当前仅有名义物体资产、仿真数据和局部物理证据，真实ego重定向和完整Goal均未完成。今后若启动新的学习训练，仍在首个健康checkpoint校验并给出ETA后暂停目标。
