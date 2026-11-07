# 研究方案 v0.27：源表面区域、完整姿态接口与动态诊断

2026-11-07。完整目标承接 [v0.26](research-plan-v0.26.md) 和原 SuperMimic + CHORD 实施方案。上一回合及本回合均为 progress；未启动新的学习训练，目标保持 active。新训练仍遵守首个健康 checkpoint 落盘、校验及 ETA 后暂停的用户约定。

## 接触映射已经按源表面重建

旧 CAD 点距原始 ego 投影约 4.8 mm，却改变了约 60° 的表面法向。新 v11 不再沿用该点作为源点：`exact_source_original_hand` 精确保留原 phone-body 投影点和真实 triangle367/6，实际 hand triangle2332/1876 不变。原始目标仍是人体扫描与物体间的 **unsigned proximity** 候选，不是测得接触、压力或真实受力方向。

另设独立的 `source1mm_opposed_side_original_hand` 敏感性分支：枚举源点 1 mm 球内所有实际 phone 填充三角面，再按法向到源点对连线的角距、投影距离、triangle ID 排序。该半径是算法假设，不是重建精度或置信区间。拇指选择 triangle511，移动 0.03604 mm、法向变化约 0.04°；食指选择 triangle41，移动 0.49481 mm、法向变化 38.52°。这直接说明小点位误差也可能跨过斜面区域，距离误差不能替代法向/面域约束。

独立实现复核冻结编译几何的全部 696 个 phone 三角面，源球成员分别为 6/4；点、真实重心坐标、外支持法向、候选完整性及排名检查 6/6 通过。这里的编译 convex triangulation 不直接等同于原始视觉资产拓扑。敏感性拇指点在边界；食指41与40的法向排序只差约 3.14e−7°，原始预注册规则决定选41，不能声称其物理上更优。[映射及独审](D:/workspace/ego-data-pilot-2026-11-07/source_surface_mapping_v11/audit_saved_mapping_v1/report.json)

两点内法向到指向另一点连线的角度，在 exact 为约 2.04°/59.18°，在1mm分支为约 1.77°/20.90°；µ=.7 的外圆锥半角约34.99°。这只是二点纯共线力对的几何描述，不是重力支持、原生四射线、多点压力斑块或动态持握证书。所有实际原生点、法向、friction、J 都重新生成。

[几何图](D:/workspace/ego-data-pilot-2026-11-07/source_surface_mapping_v11/source_regions_v1.png) 的箭头为网格外法向，非观测力；点和箭头投影到 phone-body XY，各局部轮廓按源点 z 截面绘制，点有不同 z 时不冒称同一截面上的测量。

## 同模型装配与支持实验

新 helper 接收显式 schema11 锚点与 expected pin，核对真实编译顶点、全部凸体顶点的支持平面及 geom-to-phone-body 变换。原轻量 FK 的查询方程未变，新的完整 FK 使用同一材料 A/B，继续核对 same-x q/A/B；J、bias、passive、PD 均取完整静态返回缓存。原14/87/4检查只保留作父契约来源，不冒称已经测试新锚点。[新接口源派生](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v11/source_derivation_v11.json)

两支均使用 v10 的完整硬关节范围、固定其余12q、相同两个起点、手机范围、阶段算法与预算，原物理模型、腕位、材料、质量/惯量、增益、尺度与solver不变。

| 分支 | 最终记录 | 装配及22/189几何 | rank0最低 native loss | strict支持 | 运行主体时间 |
|---|---:|---:|---:|---:|---:|
| exact source | 3 | 3通过 | 0.254764 | 0通过 | 245.74 s |
| 1 mm敏感性 | 4 | 4通过 | 0.128485 | 0通过 | 183.60 s |

所有 A0 达到200次 fun上限，不能称优化器收敛；四个局部装配 Jacobian 均 rank10。Exact 两个A1分别需几何修复，1mm两个A1直接通过。四个支持阶段均触原90秒预算，不是全局搜索证书。7个最终记录全部通过 max|scaled eq10|≤1e−5、指定pad[-.1,+.2]mm窗口、其余20手机对及189自碰对非穿透；7个 strict 缩界LP均 status2、无支持见证。较低残差不等于合格静力支持或持握。

Exact 第一姿态的实际接触点距原 ego 目标约0.01–0.81mm，明显不同于旧约4.8mm的映射；这是材料参考保持的改善，不是策略收益或动力学通过。选定 rank0 原生初载在 exact 两指均小于3N，1mm的食指约3.46N：3N仍是理想LP设计预算，实际初载标志保持描述性，原动态成功门没有瞬时3N上限。真实控制与力矩界未放宽。

累计这两次搜索：静态FK2207、cheap FK13944、LP1936、native forward7、实际contact Jacobian32；搜索阶段0积分和训练。两进程均exit0、stderr空。原始 attempt/returned 与deadline请求保留，不能把尝试都当返回。[Exact结果](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v11/exact_source_original_hand_completed_v1/report.json)、[1mm结果](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v11/source1mm_opposed_side_original_hand_completed_v1/report.json)

## 原成功门的动态诊断

静力LP是候选生成工具，不是软摩擦动态持握的必要条件。因此新动态诊断允许已装配、碰撞可行、已知且控制合法、两pad已加载且无非pad/table辅助的姿态进入实际2秒测试，原strict失败和零速加速度结果原样保留，不改称通过。

按预先声明的规则，各分支仅比较其两个seed的rank0，以实际native loss选最低者，均为seed1/rank0；rank1未被事后替补。若存在严格见证及重复控制probe则用其实际新控制，否则用原保存的native_probe控制。本次两支均属后者。只运行常量控制，初始qvel=0，之后不重置状态、不加外力或物体辅助。成功仍需原2秒、尾段稳定/双pad载荷、碰撞/关节/无辅助及双步长2mm/2°等全部原门。

四案例实际执行完成，进程exit0、stderr空；1次MJB读取、4次初始化forward、2959实际积分步，0学习/GPU。全部因COM漂移超过原50mm早停；无warning、无table力，关节/穿透门未失败，但最终姿态、漂移及非pad力门失败，1.5–2s尾段均不可用。

| 分支 | dt [ms] | 实际步数 | 退出时间 [s] | 非pad法向力峰值 [N] |
|---|---:|---:|---:|---:|
| exact | 0.3125 | 361 | 0.11281 | 5.435 |
| exact | 0.15625 | 728 | 0.11375 | 21.815 |
| 1 mm | 0.3125 | 622 | 0.19437 | 6.258 |
| 1 mm | 0.15625 | 1248 | 0.19500 | 6.470 |

这些是**退出时刻，不能当作稳定持握时长**。虽然两组早停终点分别相差约0.316/0.248mm和1.920/0.589°，全部缺少2秒完整时长，双步长资格均false；尤其exact非pad力峰值对步长差异明显，不把终点接近写成全接触动力学一致。保留原tail_linear_velocity_RMS的root-world qvel语义，不改称COM速度，也不补造缺失尾段。

本轮几何保持改善后，常量控制仍失败且有非pad干涉；不能声称更长退出时间是学习收益、真实ego接触恢复或可持握的证明。[动态实测报告](D:/workspace/ego-data-pilot-2026-11-07/hand2_source_mapping_dynamics_v11/cpu_nominal_v1/report.json)、[实际进程终态](D:/workspace/ego-data-pilot-2026-11-07/hand2_source_mapping_dynamics_v11/process_run1/terminal.json)

保存轨迹独审11/11进一步定位：exact在2.5ms首次有pad≤.1N，1mm在8.4375ms首次有pad≤.1N，均早于首个20ms控制边界；这是首次阈值事件，不声称之后永久失载。非pad>.1N均出现在首pad失载之后：exact是食指link4（41.25/57.5ms），1mm是拇指link4（103.44/104.06ms）。不能把后续非pad峰值倒置为起始唯一失握原因。该证据要求后续初始控制具有预测预补偿，并保留子步接触事件，不能只训练20ms端点误差。[保存时序与事件](D:/workspace/ego-data-pilot-2026-11-07/hand2_source_mapping_dynamics_v11/saved_review_v1/report.json)

## 可用的 phone 世界模型接口

已实现 [phone_adapter.py](D:/workspace/ego-data-pilot-2026-11-07/phone_world_model_v2/phone_adapter.py) 和保存数据加载器：state58含世界系COM、完整R9、世界线/角速度；action为20ms实际绝对位置控制u20；raw输入78，物理输出delta52。旋转以 `Exp(deltaR_world) @ Rbody_world` 左乘解码，不能接旧 pen 的右乘chart52或权重、缩放和Reader。

Torch解码保留AD，零旋转梯度有限；3个有限契约通过，F64非交换旋转方向AD-FD差约6e−12。该检查验证解码器，不是神经网络、仿真物理Jacobian或预测精度。首次夹具因默认F32字面量与F64参考不一致而失败，保留失败并显式F64修正；阈值未改。另在执行前修复未选中Taylor分支的溢出路径，未改真实旋转向量。[接口交付及实际检查](D:/workspace/ego-data-pilot-2026-11-07/phone_world_model_v2/report.json)

加载器保留322行、161 H2配对及6个跨边界精确别名，7个来源边界仍连成一个prefix，拒绝任何多分区训练/测试拆分。warmstart/HISTORY缺失，state58没有Markov证明。新phone Reader、接触/奖励/Replay、normalizer及SuperMimic/CHORD实际训练接线仍待完成；当前不是训练好的世界模型。

还从冻结规划记录构建112条真正同完整integration输入状态的H1有限动作secant。每次只改变前8/20执行器之一，其余19维控制逐字节匹配，iter1基点按原line-search接受规则重放，不误减previous_control。110条为±.01rad，另外两条实际步长约4.77e−5及.00749rad，按实际分母全部保留，单列较小步长的条件数风险。H2输入状态不同，完全不混入这个动作导数接口。[112条实际步长报告](D:/workspace/ego-data-pilot-2026-11-07/phone_action_secants_v1/actual_step_secants_v2/report.json)

这些是原同一失败前缀上的有限动作响应，不是无限小梯度、完整20维Jacobian、接触/力导数或独立轨迹验证。本项0新增原生物理、Log、模型或训练。

## 下一阶段与完整范围

下一续轮直接实现实际phone 20ms Reader/采样接线：从同一原生模型的POST qpos/qvel得到state58，原生contact/force以最后solver PRE及整数tick保存，不能将它们重命名为POST标签。64/128子步明确入协议；同一初态的两步长以及规划克隆属于同一episode族。先采集具名的不同初态/控制种子和接触过渡，分组后才允许新phone世界模型的新归一化/损失/参数训练，并用已保存的有限动作secant核对。不得随机行拆分旧322条，不从旧pen权重或维数继承资格；新训练仍在首个健康checkpoint+ETA后暂停。

新采样同时保留每个完整20ms区间的逐子步PRE接触来源，派生逐pad最小载荷/覆盖率及首次阈值tick，并将其与端点最后PRE力、POST state58明确区分。旧322条缺少未执行分支的子步缓存，不能回填这些标签。第一控制区间内即有失载，应先检验初始动作预补偿和接触恢复，不能靠20ms后的观测就声称阻止了2.5ms前的事件。

本轮只得到常量控制失握数据，不继续重复同一静态起点作为主要推进方式。完整保留源面域/法向、具名敏感性及实际native manifold；独立仿真episode不能伪装成独立ego示范。后续主动接触恢复与SuperMimic/CHORD必须绑定相同物理初态、观察/奖励和总预算，控制改进、几何变化、材料先验与学习方法收益分别标明，不能用退出更晚替代持握门。

原2秒持握、双步长、动力学先验鲁棒性、61帧时序、手机COM/64或128子步/完整姿态读写、SuperMimic与CHORD同预算训练、世界模型接触与梯度、独立轨迹/ego泛化、资产与动力学校准和硬件观测契约均仍在完整目标内。功能指腹、真实摩擦/质量/COM/惯量及clip重力注册仍UNKNOWN；真实ego当前用于几何参考，尚未用于手机策略训练。本目标未完成。
