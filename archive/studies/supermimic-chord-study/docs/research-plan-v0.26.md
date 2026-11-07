# 研究方案 v0.26：装配已验证，支持受几何与搜索范围共同限制

2026-11-07。完整目标承接 [v0.25](research-plan-v0.25.md) 与原 SuperMimic + CHORD 研究。上一回合及本回合为 progress；目标保持 active。本回合未开始新的学习训练，后续新训练仍遵守首个健康checkpoint、ETA后暂停的用户约定。

## 分阶段求解取得的证据

新的轻量装配查询只执行kinematics与comPos，使用独立MjData并复用原模型；不执行forward、碰撞、RNE或LP。真实三角面材料点及外法向仍来自冻结307e91输入。phone每个真实三角形的固定边给出body切向基，随phone R旋转；位置6维、切向法向4维共10个装配等式，另保留反向分支。原姿态14/14契约通过，位置与保存参考最大误差约5.6e−17m、法向约2.6e−14。[查询契约](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v9/original_pose_contract_v1/report.json)

原起点及预先给定5°倾斜起点均采用相同三阶段：A0只拟合装配；A1在装配等式、反向分支及完整碰撞窗口内验证/修复；B在同一组硬约束内最小化支持LP的缩放L∞平方及微弱姿态项。A1和候选池均要求scaled eq10最大绝对值≤1e−5，位置项尺度为1mm，不能仅凭0.2mm/1°的粗门声称硬等式满足。

两个A0都达到200次fun上限，不能称优化器收敛；但保存的装配等式达到浮点误差量级，局部10×14 Jacobian数值秩均为10，完整几何也通过。因此A1无需再求解。B分别24/23次迭代停止，四个最终候选都重新通过装配和完整22手机对/189自碰对；两指定pad压缩约25µm，使用原[-0.1,+0.2]mm窗口，其余20手机对及189自碰对非穿透。

条件支持统一在真实hand三角点A给物体力臂与手侧J(A)，避免上一版在不同A/B处施加反力的小力偶。phone材料点B仍固定，B−A=−25µm·nH；A是条件预载作用点，不冒称phone表面witness或实际native con.pos。同一个x的完整FK复制anchor/J/bias/passive/PD数组入缓存，不读取另一MjData中可能过期的量。原ego材料目标单独保存，未被算法的CAD材料点覆盖。[算法配置](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v9/completed_v1/algorithm_surface_configuration.json)

## 实际支持仍未通过

四个最终姿态各做一次fresh native forward，实际生成thumb3、index4共7个接触，而非沿用条件2点。全部实际点、frame、friction与J重新进入原生四射线LP；每个finger的法向预算跨点汇总。新接触对选定面法向的角差接近零，实际记录点距固定材料点约十余µm至1.2mm，必须保留实际分布，不能从点数推断已校准压力斑块或独立扭矩能力。

两rank0的条件8ray loss约0.262786、实际native4ray loss约0.218303；rank1实际约0.2196–0.2198。新的force-hard、moment-box LP在四例均未找到缩界内支持witness，未生成新的可执行候选状态或重复控制probe。因此本回合仍没有进入2秒持握，也没有策略收益或faithful ego资格。缩界仅比原1e−4Nm门小1e−10Nm，原数值验收不放宽；其infeasible只属于该缩界问题，不冒称原box或全局不可行。

一次进程exit0、stderr空，wall约133.38秒。实际计数：709次完整搜索FK尝试全部返回，另4次最终完整几何，静态FK共713；cheap FK6581；LP717；native forward4、contact Jacobian28；0积分、训练或GPU更新。[结果](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v9/completed_v1/report.json)、[真实进程终态](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v9/run1_terminal.json)

实际原生初始法向力是否超过3N在新诊断入组中单列描述，不再把它误称原2秒动力学门的一部分；理想支持LP的每指3N设计预算仍为硬条件，原动态门没有瞬态3N上限。这没有放宽真实控制/力矩界或修改材料。仍须分别报告实际力的均值/峰值，不能由理想λ上限推断原生反力合预算。

## 搜索边界与接触来源

四例native LP的真实力矩最小余量约0.0976–0.1003Nm，每指Fn约1.11/2.24N，均未触3N。因此本次残差不能直接归咎于力矩或法向预算不足。四例食指joint1均触原模型硬下界−1.047rad；joint2/joint4均触人为source±0.45rad的下边，两个rank0的joint3还触人为上边。手机平移及rotvec坐标边界未活跃，最小余量约3.68mm/0.154rad。这给出有依据的下一消融：只扩大q8算法范围到原模型硬限位，保持手机范围和全部物理参数。[边界分析](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v9/active_bound_review_v1/report.json)

还有独立的接触面几何限制：在选定两A点模型内，thumb inward normal与朝另一指的连线夹角约58.31°，相应圆锥µ需求约1.62，超过实际µ=.7外圆锥角34.99°；index角约5.87°。整体转动同一材料点对几乎不改变这些固有角，但会改变所需gravity wrench。这只是选定两点的antipodal方向描述，不能否定7点native manifold、其他区域、整个机器人或特定重力下2秒持握，也不能把force closure新增为持握必要门。[几何描述](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v9/active_bound_review_v1/antipodal_geometry_v1.json)

另已生成预注册的真实shared-edge邻接面备用目录，每指仅1/2-ring全9个候选，按与原phone inward normal角距、原native点到实际填充三角距离、triangle ID排序。thumb前3为1436/1438/2334，index为868/2233/1344。六个投影均近顶点，边界重心坐标明确保留；原phone材料点、法向及人体目标未改。本轮分阶段求解尚未使用这些替代面，不能称它们可达、是真实功能pad或可持握。[邻接候选](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v9/adjacent_candidates_v1/adjacent_candidates_v1.json)

## 完整关节范围消融

新v10只移除了q8的人为±0.45rad搜索盒，仍使用原模型关节硬限位内缩1e−9rad；同两个起点、手机平移/rotvec盒、阶段/预算、材料点、碰撞/力学门及全部物理参数不变。六处枚举的路由/schema/bounds编辑可逐字节逆还原v9原源，独立源审确认；旧版本未改。[源派生](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v10/source_derivation_v1.json)

唯一CPU执行正常退出，stderr空，运行主体约183.56秒。两起点仍装配/几何通过，支持阶段均触原90秒搜索时间预算。四个最终候选再次通过完整几何和硬装配，两个rank0的native loss约0.190892/0.190782，但四个缩界strict LP仍无见证，持握入组为0。较小残差不等于合格支持，不称全局不可行或学习方法收益。[完整结果](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v10/completed_v1/report.json)

实际951次完整查询尝试，949次返回；两个deadline请求的x已先写入日志。加4次最终完整几何，静态FK955次；cheap FK6473；LP957=949条件+4原生minimax+4严格box；4forward、28contact Jacobian，0步进/学习。保存读回5/5是执行作者的终态/pin/计数读回，不能冒独立动力学重放。[进程终态](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v10/run1_terminal.json)、[作者读回](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v10/terminal_saved_readback_v1.json)

## 近距匹配遗漏的法向变化

复用既有缓存定位，将**原ego投影目标**和当前固定CAD点的真实phone三角形直接对照：thumb从triangle367对应到当前triangle24，距离4.813mm、外法向差61.03°；index从triangle6到当前triangle139，距离4.738mm、外法向差58.01°，均跨不同平面/近共面组。这不是同一平面内的小位移，不能仅凭毫米距离声称接触语义保持。[面域对照](D:/workspace/ego-data-pilot-2026-11-07/surface_assembly_v9/saved_recheck_v1/original_ego_vs_CAD_phone_faces_v1.json)

这给下一步提供更直接的参考修订依据：显式保存原ego所在object面域及法向，结合不确定接触区域选择实际机器人面域，再求装配/支持。原ego目标仍只是unsigned proximity投影，不能改称测得接触、表面受力方向或压力真值；法向差也不能被写成唯一失败原因。

## 为世界模型准备的完整姿态数据接口

另外从冻结的MPC规划探索保存结果构建全部322条20ms转移，161组H2配对和7个来源边界组各46行完整保留，无随机行切分。state58使用COM/world速度和完整body R；physical world delta52用Log(Rnext Rcurrentᵀ)，以Exp(deltaR)左乘解码。非交换旋转夹具能排除错误右乘，322行最大解码误差约2.55e−15，原始R未投影修复。新physical delta52不是旧WM仅凭维数可互换的chart52。[数据报告](D:/workspace/ego-data-pilot-2026-11-07/phone_transition_dataset_v1/dataset_v1/report.json)、[实际保存读回](D:/workspace/ego-data-pilot-2026-11-07/phone_transition_dataset_v1/saved_readback_v1.json)

样本来自原0–0.12秒来源边界、预测到0.02–0.16秒的**仿真规划/失握前后覆盖**，不是真实ego动力学真值或成功示范。仅六个已执行完整H1可与actual trace逐项交叉核；未执行分支不伪装成actual。端点仅存最后solver的PRE力/时间，H2极值另表，未伪造256子步力缓存。warmstart/HISTORY未入state58，尚无Markov性、预测精度或独立轨迹泛化资格。本项没有新增模型、物理、LP或训练；它推进新phone Reader/完整姿态目标接线，不替代持握研究。

进一步用完整integration输入/输出SHA及control20原字节检测到6组跨cohort精确转移别名：前一所选分支的第二H与下一actual边界的previous-control第一H重复。别名把7个来源cohort全部连成同一个prefix组，**不能把它们当7个独立train/test组**。322行与原schema未删改，未指定拆分；后续独立轨迹验证需新增不重叠的物理初态/episode，不能依靠随机行或边界组切分。[跨组别名审查](D:/workspace/ego-data-pilot-2026-11-07/phone_transition_dataset_v1/overlap_audit_v1.json)

## 官方资产语义的新证据

官方说明区分承力骨骼、接触缓冲软体与提供摩擦的外皮肤；该页限定于Hand2.2 Beta非触觉样机，软体外形在模型中仍作刚体近似。不能由此给当前STL某facet指定实测µ/刚度或功能区。当前官方仓库五个sensor STL与本地固定版本相同，更新裸手资源不会自动补出接触面定义。[官方硬件说明](https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview)、[固定版本XML](https://github.com/wuji-technology/wuji-description/blob/829c9aa27cb12c3a49d5b8d63c9869dd8f656702/hand2/hand2_2/body/mjcf/right.xml)、[版本与文本证据](D:/workspace/ego-data-pilot-2026-11-07/asset_semantics_followup_v10/report.json)

匹配设备版本的FingertipSensorInfo完整positions/point_rpy/base_xyz/base_rpy可补注册点与受力轴；本研究尚无实际数值，轴也不能未经核对直接当几何外法向。没有安装SDK、连接设备或改变仿真模型，真实软体、摩擦和功能pad标定仍UNKNOWN。[官方触觉接口](https://docs.wuji.tech/docs/zh/wuji-sdk/latest/wuji-hand-2-tactile)

## 继续范围

完整q8范围消融已经执行，下一步转向保留源object面域/法向及不确定性范围的接触映射，结合真实hand邻接面域形成具名候选；不继续仅扩大搜索盒或重复同一失败起点。实际接触法向/力学需重新计算，不能通过提高µ或增加虚构点扭矩救旧结果。新phone数据接口可用于后续分组预测与有限action-secant核对，但当前未指定训练/验证拆分或启动训练，不能把同源近邻样本当独立ego泛化。

任何候选仍须原2秒无辅助持握、双步长及动力学先验鲁棒性；随后推进61帧时序、手机COM/20ms子步/完整姿态读写、SuperMimic/CHORD同预算训练与世界模型接触/梯度校验、独立轨迹泛化、真实资产/动力学校准与硬件观测契约。功能指腹、真实摩擦/质量/COM/惯量及clip重力注册仍UNKNOWN。真实ego已用于资产和参考几何，尚未用于手机策略训练；完整目标未缩减或完成。
