# 研究方案 v0.31 草稿：实际加载参考、因果动作数据与几何动量 WM

2026-11-07，承接 [v0.30](research-plan-v0.30.md) 和完整 Goal。本文件为待根代理复核、冻结的草稿，不替换 v0.30。训练已实际在首个健康 checkpoint250 后退出；下文采用真实保存结果，不预填后续5000步、Actor或ego成功结论。完整目标仍未完成，最终里程碑与按用户条件暂停由根代理收口。

## 本轮改变了什么

两条具名三指功能分支已从“几何邻域通过但可能没有接触”推进到“原生三pad真实初载”。随后相同模型、无物体辅助的16个双步长案例仍全部失持。这个结果排除了“初始没有三指加载”作为本批失败的统一解释，没有证明持握不可行，也没有证明静力LP或零加速度是动态持握的必要条件。

主线因此转向实际动作干预数据、当前可微几何和结构化动量预测，再进入WM/Actor/CHORD的同预算实验。**手工两秒持握通过不是开始学习的前提。** 手工参考承担模型、坐标、控制、接触和任务接口的诊断作用；其失败数据与真实门均保留，不再靠持续调grasp或放宽门代替学习主线。

## 压缩与真实预载：初始加载存在，持续支持仍不足

三pad的压缩搜索目标为gap∈[−90,−60]µm、中心−75µm，原硬几何窗仍为[−100,+200]µm，其他phone/table及self非穿透门未变。exact分支先通过；1mm算法分支v1仅有−1.4884473e−10m的非pad负余量，仍严格失败。新v2从原投影合格态重新开始，仅选择该分支并以2µm内部余量作搜索目标，未把旧失败归零或当作warmstart。

v2实际4次完整几何、14次点Jacobian、1次QP后通过。三pad gap约−70.12/−74.16/−74.68µm；最小非pad余量为+1.8634µm。它小于2µm搜索目标仍可通过原≥0硬门，两个概念不混用。phone pose及其余八关节字节固定，0 native forward/积分。两条几何参考均不是原ego材料点的精确复现，原功能让步继续保存。[v2结果与保存算术](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_loaded_entry_v1/saved_arithmetic_review_v2/report.json)

原生预载使用实际`con.pos/frame`及对应手body的J，不再把材料锚点当作实际受力点。两条source各保存ctrl=q、每pad总1N、每pad总3N和有限有界GN路线；真实PD/ctrl/actuator/joint-actuator界均核对，无clamp。名义法向命令、三指已加载、扳手误差和动态持握四项分别记录，3N也不是本次动态入口硬上限。

两source×四控制×双dt共16例实际执行9200步。所有初始三pad Fn均>0.1N，并逐字节匹配其原生父probe，但全部在约0.11219–0.125秒触发失持；没有完成两秒，末0.5秒尾指标为null，八个dt pair均不具备完整两秒资格。失败前缀时长不称稳定持握时长。GN exact指定pad向上冲量仅约对应重力冲量的25–26%；GN 1mm仅约3.8–4.4%，另有约30%来自不合格other接触，不能计作指定三pad支持。持续的合法载荷、扳手与状态控制仍未解决。[16例原始结果与独立保存审](D:/workspace/ego-data-pilot-2026-11-07/functional_tripod_loaded_dynamics_v1/saved_review_v1/report.json)

## 因果动作数据已经实际采集，角色没有重命名

原TRAIN的72个完整300值source context先全部重放原action，逐tick/起终点及receipt匹配后才执行任何新side。全20轴计划包含2952条记录：72 baseline、2870个可执行side、10个实际du<1e−6rad的不可用side。实际2942分支全部完成20ms，共282432积分步，2次MJB读取、0 restore/observer forward、0NN训练。不可用记录保留，未clip动作或用epsilon补分母。[实际采集报告](D:/workspace/ego-data-pilot-2026-11-07/phone_causal_train_acquisition_v1/acquired_v1/report.json)

每个分支保留同一before-action完整integration receipt、仅ctrl变化的after-action receipt、原episode hard-stop参考、actual F64 du及FP32表示。POST state58与solver PRE接触相位分开；全H接触冲量和coverage是监督标签，不是当前部署输入。原两指定pad15/28与全部22个phone-boundary slot不变，第三tip41不因另一三指实验而静默变成该数据任务的pad。

| 数据角色 | 本轮实际内容 | 用途与边界 |
|---|---|---|
| 原TRAIN | 684个完整H，36个partial保留 | 完整行用于拟合，不伪造partial的H终点 |
| 因果FIT | 36 baseline、1434 side；另6不可用 | side加入绝对监督；1434对同状态响应监督，baseline映回原TRAIN不重复增权 |
| 已见状态的新动作CHECK | 36 baseline、1436 side；另4不可用 | 不进入本次拟合/normalizer；不是未见状态、独立轨迹或ego验证 |
| 原VAL | 114个完整行 | 固定checkpoint单步监测，不选择checkpoint或系数 |
| 原TEST与新fresh family | 原TEST已有历史曝光；新12 family尚未采集 | 不重新包装成未见，不拿当前CHECK替代真正新初始化/任务验证 |

22对标签覆盖全部2942执行分支，分别保存固定source BODY轴、绕逐tick moving COM的力/力矩冲量，及每对生成接触、正Fn、均值、coverage等量。独立保存复算保留了非零离散动量残差，线/角动量向量RMS约4.60e−8N·s、7.45e−8N·m·s，没有投影为零。这是名义仿真标签算术，不是实测人手力、连续时间真值或已证明的state58 Markov性。[标签与独立保存审](D:/workspace/ego-data-pilot-2026-11-07/phone_causal_impulse_labels_v1/independent_saved_review_v1/report.json)

## 可调用的当前几何与结构化预测

已完成实际编译几何的纯Torch固定腕FK，state58中的R是BODY→world，phone root由COM减去`R·body_ipos`恢复；mesh只应用一次编译geom变换，capsule使用真实中心线/半径，table使用真实box。三份TRAIN当前状态与原生kinematics的F64位置/姿态及q-Jacobian核对通过。该provider没有学习参数，不能把visual面或sensor名称直接解释为真实功能指腹。[几何provider证据](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_features_v1/completed_v1/report.json)

有限方向descriptor在同一phone BODY轴、COM原点中计算`max_n(min_H n·H−max_P n·P)/||n||`，使用1406个实际phone面方向及反向/轴向。F64/F32的22对均可微分片计算，known、pose、exact-gap和normal masks分开；exact/unique-normal/严格浮点证书/full-H零接触mask始终False。两份保存压缩状态与解析夹具通过有限数值合同，但方向不足会很松：1mm middle描述量约−192.94µm，对应保存精确gap约−74.68µm。正的当前descriptor也不能推出未来20ms无接触。FP32 guard只是数值尺度启发式，不是区间误差证书；首checker字段接线失败及新v2修正均保留。[方向descriptor交付与限制](D:/workspace/ego-data-pilot-2026-11-07/phone_directional_gap_features_v1/delivery_v1.json)

新WM输入366维：raw80（state58、absolute u20、原生h/n）+22×12当前相对geom pose+22方向descriptor。输出222维，其中178个物理量为joint delta40、COM-ballistic residual3、world-left dR3和22×6冲量；另有22个meanFn值、22个coverage logits。动量decoder用名义mass/完整BODY惯量/gravity与预测冲量得到v/omega，p/R仍有独立预测头，不从冲量擅自决定完整姿态。冲量轴固定于source BODY，力矩绕逐时刻moving COM；不混成瞬时旋转坐标积分。

这个结构限制输出的代数关系，却不能保证预测的冲量可由真实软接触、摩擦、控制器产生，也不能强制真实标签离散缺陷为零。所有22对包含table/nonpad，其总冲量不能冒称合法三指承载。当前几何导数也不是native动作真Jacobian。

## 首次新模型训练：真实停在250/5000

新模型为fresh 366→128SiLU→128SiLU→222，不加载旧pen/raw80权重或normalizer。固定seed1731、CPU4、全batch、AdamW lr0.001/wd0、clipnorm1，声明预算5000更新，checkpoint250/1000/5000。绝对监督2118行=684原TRAIN+1434 FIT side，另1434对经验有限动作响应；normalizer仅由这些FIT数据计算。模型为FP32，结构化物理decode为F64。

本次训练没有使用刚生成的tripod压缩、预载或16例动态数据作标签。它仍使用原两pad15/28及canonical22身份；三指功能任务的数据、角色与reward要在后续显式接入，当前不能称为已经接好的三指持握模型。

五项loss系数均预先固定为1：物理头、decoded velocity、FIT头secant、meanFn log1p、coverage BCE。物理头/响应按q、qd、COM、dR、力冲量、力矩冲量六个原单位块分别归一再等权；meanFn使用1N参考的log1p，无微小std放大。实际F64 du作响应分母，不把跨接触有限差称真导数。几何可缓存但2118行loss权重不去重；CHECK/TEST目标未进入拟合，VAL仅固定监测。[冻结配方与接口](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_momentum_wm_v1/README.md)

本次CPU训练已真实exit0，保存AFTER_OPTIMIZER_250及PRE_OPTIMIZER_250梯度证据，checkpoint SHA为`a132d171bc938bd17b0d17b617d14317cacba9063d84cdd5fce6a78d5baa5d20`。producer序列化读回health15/15，包括权重/Adam6个step250状态、梯度、normalizer/context/FIT输入目标、log前缀、预测字节与RNG检查；另有保存审9/9和根代理payload核验14/14通过。实际250次训练FW/backward/Adam更新，另2次health FW、1次VAL FW、2个NN构造、1次own checkpoint读取，0 native/GPU。它是训练健康与保存一致性，不能升级为方法准确性或独立策略资格。[真实训练结果](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_momentum_wm_v1/cpu_run1/result.json)、[health与ETA](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_momentum_wm_v1/cpu_run1/health_0250.json)

setup约9.085秒，250更新合计2.7484秒；最近100更新均值0.01070036秒，余4750更新约50.8267秒，仅含训练更新。恢复时重建cache及未来checkpoint开销另计，观测速率场景约60.0–61.4秒，是启发式而非置信区间或保证；它不含全部进程启动、导入和退出开销，terminal与health计时约差1.747秒，不能把该场景当作完整wall-time上界。进程已按首健康点退出，没有后台继续到5000。

VAL114固定监测为COM向量RMS0.26337mm、SO3角RMS1.12679°、vCOM RMS0.021527m/s、omega RMS0.753897rad/s。它们仅为20ms单步描述，没有在此完成ballistic/CV或同预算WM对照、接触梯度检验、长rollout或holding评估，不能据此宣布优于旧模型。剩余4750步尚未完成，Actor/CHORD尚未运行。

## 真实61帧运动意图已接成参考，机器人尚未执行

HOT3D clip1856的42–102帧共61个原始left UmeTrack wrist与phone24 pose，保存真实时间2.00000849秒。接口先形成wrist-relative物体轨迹，再将其相对首帧的SE3增量右乘到具名机器人起点；20ms网格101点止于2秒，不拉伸末端8.49µs、不外推。源相对位移约4.90mm、转角约4.74°，是可供后续任务reward/参考使用的物体运动意图。[61帧参考](D:/workspace/ego-data-pilot-2026-11-07/phone_ego_reference_v1/completed_v1/report.json)

它来自真实标注pose，但没有从RGB重新估计、没有human-left到Hand2-right的完整关节/接触映射，也没有真实ego重定向执行或61帧机器人跟踪。源重力注册、真实功能指腹、质量/COM/惯量/摩擦与接触力真值继续未知。原参考文件保留当时起点的真假几何资格，新的压缩/预载结果不回写旧文件；后续采用新状态须显式绑定。

## 恢复后的主线与完整目标缺口

恢复后先沿冻结预算完成该WM，再按事先固定的状态/动作分区检验状态预测、接触冲量、经验secant与模型自身AD/FD；各物理单位块、跨接触变化和未知量分别报告。当前一个组合模型尚不能分离geometry、momentum与counterfactual监督的贡献。下一比较应匹配数据曝光、初始化/更新数、部署输入权限及原生采样预算，不把不同阶段的旧raw80结果直接当作严格消融结论。

随后真正接入phone Reader、state/world-left52、Actor、reward、replay和CHORD。任务可使用已定义的wrist-relative物体意图，并具名区分原两指源约束与允许的三指功能映射；未来contact/force/全H路径标签不得进入部署forward。原两秒/双dt/无外力/无桌面借力/合法指定pad等门保留作任务评价，不作为允许训练前必须手工通过的条件。任何新branch通过也不得回写成旧二指或旧失败通过。

仍缺真正新初始化/控制family、更多独立ego轨迹、物理参数校准与鲁棒性、长时任务与同预算策略收益、完整SuperMimic/CHORD对接及现实观测契约。当前只完成名义仿真局部数据、几何/动量接口和一个新模型的首健康checkpoint。按用户既定条件，根代理完成保存核验、ETA与交接后暂停Goal；不以草稿声明整个研究或ego重定向完成。
