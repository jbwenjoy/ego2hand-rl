# 研究方案 v0.29：预留评估、同状态动作响应与功能支持路线

2026-11-07，用户已恢复目标。本轮承接 [v0.28](research-plan-v0.28.md) 与完整目标；上一回合及本回合均为 progress。当前没有新的神经网络参数训练。原先5000步训练的终态已核对为exit0，保留原参数及所有checkpoint；本轮评估和输入动作AD不重复触发“新训练首健康checkpoint后暂停”的条件。

## 预留TEST的固定模型评估

在首次读取TEST模型目标前冻结评估协议，primary固定为final5000（f2e514…1f1f），没有根据TEST挑选250/1000。实际读一个checkpoint、构造一个仅加载已有权重的模型，单步114行及12episode的最多10步递归，共234行预测；0优化器/AD/原生物理。6条partial仍保存mask，不补造H目标。[评估报告](D:/workspace/ego-data-pilot-2026-11-07/phone_world_model_evaluation_v1/completed_v1/report.json)

| TEST单步方法 | COM向量RMS [mm] | 完整SO3角RMS [°] |
|---|---:|---:|
| fixed WM5000 | 2.252 | 1.440 |
| TRAIN均值 | 9.246 | 3.642 |
| 关节/COM/world角速度恒定 | 1.653 | 0.874 |
| COM连续重力抛体，关节/角速度恒定 | 0.766 | 0.874 |

抛体基线g绑定原冻结模型的[0,0,−9.81]，不是从TEST拟合或真实ego重力注册。它与恒速基线均不建模接触，完整R以world-left Exp更新。均值基线较弱，因此此前1.88mm相对8.86mm的改善不能单独支持“物理预测已可用”的结论。

WM对两pad覆盖率的Brier约.03421，优于TRAIN均值.11820；但非pad力的原单位RMSE约30.568N，TRAIN logforce均值经expm1的基线约2.345N。力量头的非负性与log损失并不保证原单位校准或承载一致性。不能由覆盖率改善推导真实接触扳手、稳定持握或CHORD控制收益。

按真实pad coverage>0分层的单步COM：无pad 63行，WM/抛体约2.717/.130mm；一pad 16行约1.827/.666mm；两pad各有覆盖35行约1.306/1.296mm。该分层不是同时双pad接触，也不排除非pad/table，不能将“无指定pad”改称完全无接触。

递归预测使用模型自己的state58及原冻结u/h/n。共同12个真实目标的180ms处，WM误差18.554mm/13.660°，优于均值34.804mm/24.894°和抛体64.590mm/50.002°；长程改善与单步弱项均保留。200ms仅有6个完整真实目标：WM29.550mm/12.880°，均值20.298mm/29.203°，不能把更换群体的H10点接成同一12episode误差曲线。没有projection或joint clipping；WM在H1/H9有2个、H10有3个预测超真实关节范围，最大超出约.03376rad，而对应真实目标为0。[图表与误差口径](D:/workspace/ego-data-pilot-2026-11-07/phone_world_model_evaluation_v1/saved_description_v1/report.md)

250/1000/5000的VALIDATION总损失约1.135/1.915/2.459，覆盖率Brier约.00732/.01258/.02497。完整曲线保留，训练更久并未带来单调验证改善。当前TEST已经用于报告本模型结果；后续模型开发必须注明这一事实并预留新的最终检查数据，不能继续把同一TEST称为未见数据。

## 真正同状态的全20维动作实验

用冻结VAL中每个source最小family（6、30），两dt，各interval0/2/5共12状态。保存完整before-action 300个F64状态量，state signature实际为16383；恢复后不额外forward或Reader.reset，不把当前pose重新设为hardstop初态，仍使用原episode q0/COM/quaternion参考。

先执行唯一四状态实际恢复合同：两source×两dt的interval2，共384步。4例各27/27、汇总4/4：源/after-action/target300、逐步qpos/qvel/state58、PRE力/能量/contact rows/native clock及59观测全部与原轨迹逐字节一致。`setState`不等于全部MjData派生buffer；这里的可重演资格来自该合同与随后的每个matched baseline，不外推所有模型/回调/参数。[恢复合同](D:/workspace/ego-data-pilot-2026-11-07/phone_action_gradient_v2/independent_restore_review_v1/actual_fixture_v1/report.json)

原8轴primary设计保持不变，再在任何新动作物理/模型AD前冻结其余12轴对照，避免仅用2pad主要8轴就声称20D控制梯度可靠。每侧h=min(.01rad,该侧真实控制余量的一半)，实际步长<1e−6的4侧具名不可用；不把另一侧的大可行动作一同缩掉。188个primary side与288个secondary side，加12 baseline，共488 branches，真实积分46848步。所有新动作都完整H，无partial、无新增参数训练，0恢复/观察/初始化forward。

逐分支before-action完整上下文相同；after-action仅唯一20ctrl段改变，其余280个状态量逐字节不变。所有12 baseline完整target及逐步数据也原字节一致。476个新side中126个改变逐PRE的geom/dim/efc-active签名、350个未改变；这只是签名描述，不是光滑性或真实解析Jacobian证明。[原生分支](D:/workspace/ego-data-pilot-2026-11-07/phone_action_gradient_v2/native_completed_v1/report.json)

**预注册顺序偏差：**原文字要求先完成全部12 baseline，再执行任何新side；实际runner按每state先baseline、再本state side。最终12个baseline均通过，所有side都有其自己的匹配对照；但严格顺序遵循为false，不伪装全绿。独审12项数值检查通过，另1项顺序检查明确false，`protocol_conformance=False`，保留所有源/协议与结果，不为排序重复运行或改写数据。[独立保存审](D:/workspace/ego-data-pilot-2026-11-07/phone_action_gradient_v2/native_saved_review_v1/report.json)

## 世界模型动作梯度核对

固定同一5000步模型，输入raw80，与每个原生分支相同state58/u/h/n。59输出为物理world delta52、sigmoid覆盖率2及物理log1p力量5。Primary比较模型与原生在实际F64命令步长下的有限响应；模型AD与自身FD另用实际可表示的FP32动作分离量比较。FP32舍入/步长消失与原硬限位不可用情况单列，不能混用分母。

按q/qd/COM/world rotation/COM速度/world角速度/coverage/logforce的各物理单位分别报告；额外归一化只用冻结TRAIN总体标准差，零方差维mask不可用；零norm的cosine/相对误差保留undefined。原8轴、其余12轴及全20结果同时保留，不能只看混单位全向量norm或事后挑选有利坐标。输入AD不更新任何神经网络参数。

唯一执行exit0、stderr空：1次checkpoint读取、1次构造后仅加载已有权重的模型、500次forward及12个59×20输入Jacobian；0参数更新、0原生积分。保存独审15/15复算了全部差分、分母、Jacobian列、cohort汇总和undefined mask。[比较结果](D:/workspace/ego-data-pilot-2026-11-07/phone_action_gradient_v2/comparison_completed_v1/report.json)、[独立复核](D:/workspace/ego-data-pilot-2026-11-07/phone_action_gradient_v2/independent_comparison_saved_review_v1/report.json)

原8轴188侧中，110侧COM响应非零，其模型FD—原生FD方向cosine中位数约.7970，L2相对误差中位数约.8255；其余78侧零参考响应的cosine/相对误差没有填成0或1。该组模型AD与自身FD的COM方向cosine中位数约.999997，仅支持模型内部线性化诊断。

额外12轴288侧中，233侧原生COM有限响应精确为零；全288侧的原生响应范数中位数0、90分位约2.75e−8m/rad，模型分别约.01813/.03893m/rad。相对同一源状态的原动作，模型预测额外扰动带来的COM位移变化中位数约.181mm、最大.891mm，而很多真实分支的20ms终点COM与原动作完全相同；这不表示物体在绝对世界中静止。233个零真实响应子集也有类似的模型伪响应。这些结论仅限本次12状态，不声称这些手指在所有姿态永远不可能作用物体。

模型内部AD/FD相符并未消除这种因果错误；动作跨接触有限响应也不是原生解析梯度。当前模型可能向20D控制器提供虚假的可控方向，尚不能作为CHORD收益证据。[全476侧的响应与签名图](D:/workspace/ego-data-pilot-2026-11-07/phone_action_gradient_v2/saved_description_v1/COM_response_magnitudes_v1.png)。图显示实际动作引起的变化幅度，方向另在原报告中列出；零值保持零，未用epsilon制造非零点。

## 物体坐标系扳手与下一条物理路线

当前59输出只有两pad的覆盖/力量摘要，不能从min/peak直接推出完整接触扳手。保存数据已转换并导出同tick BODY坐标、冻结source BODY坐标和world坐标，并分别说明力矩原点为瞬时COM或固定source COM；3个解析案例通过，全部960记录与89679个PRE tick只读复算，无新物理或网络调用。

912个完整H的world动量残差为：`mΔv − mgH − Jcontact`向量RMS约6.65e−8N·s，`ΔLworld − Kcontact_about_movingCOM`约1.04e−7N·m·s；除以.02s后的等效力/矩约3.32e−6N和5.18e−6N·m。没有将残差强制归零。该证据说明标签与名义仿真动量基本一致，不是现实测力、支持成功或新物理预测精度。[扳手标签与一致性](D:/workspace/ego-data-pilot-2026-11-07/phone_object_wrench_labels_v1/completed_v1/report.json)

同tick BODY旋转后再平均，与先平均world再用source轴旋转，mean force/torque RMS差约.01849N/.000518N·m；转到真正固定source COM又增加约.000530N·m的力矩差。frame/origin不可互换；BODY分量的积分不被当作惯性系动量守恒。均值/冲量本身也不决定完整p/R过渡。

另增仅指定pad15/28的扳手与静止参考目标标签：`Wreq_world=[−mg,0_about_currentCOM]`，即向上1.962N、绕当前COM零力矩。原运动过渡不被要求瞬时满足它，不新增加速度/入组门；其他phone-boundary贡献单列，防止把非pad撞击的合力误认成合法两指支持。全部960记录保留，912完整H的pad-only静态参考误差：力向量RMS1.629N、力矩向量RMS.02952N·m；other贡献分别约.3068N/.02263N·m，不计算混单位6D norm，也不以这些统计替换2秒成功门。[指定pad目标与分组](D:/workspace/ego-data-pilot-2026-11-07/phone_object_wrench_labels_v1/support_target_v2/completed_v2/report.json)

首次support-target导出因父label identities缺少executed_substeps字段而终止，尚未读取raw contact interval；失败源/协议/输出保留。独立修订源绑定原采集identities的实际子步数并核对derived PRE切片长度，公式、两解析夹具、样本及门不变；修订版本唯一执行exit0，未新增物理/网络调用。[修订记录](D:/workspace/ego-data-pilot-2026-11-07/phone_object_wrench_labels_v1/support_target_v2/execution_correction_receipt_v3.json)

完整Goal并未规定机器人必须使用两个接触；原2pad是受控子假设，而原ego点也只是unsigned proximity。新增具名“机器人三指功能支持”路线用于检验形态差异下实际承载/抗矩能力，原2pad门及所有失败结果保持。第三tip41/right_finger3_tip_sensor_col在两个当前pose的实际保存距离约23.47/44.37mm；这既非可达证明也非不可达证明，不能只把原nonpad改名为support就宣称完成。

下一原生实验需真实放开middle q8..11，重新求完整几何/限位/法向/承载力臂及控制/力矩裕量；新三pad任务单独冻结，保留原2秒、双dt、位置/姿态/速度/碰撞/无辅助门，尾段要求三pad同时Fn>.1N覆盖率≥95%，桌面及非指定刚性link仍不能供力，并同时输出原二指门结果。旧WM把第三tip计入nonpad，不能原样继承成三指支持模型；先有实际可执行物理基准，再采新三pad标签并训练新namespace。[功能支持审阅](D:/workspace/ego-data-pilot-2026-11-07/functional_support_review_v1/memo_v1.json)

功能pad、真实摩擦/质量/COM/惯量、源重力注册、固定腕时序可行性仍未知。任何三指名义通过也不是旧二指通过、真实ego泛化或完整Goal完成。61帧时序、SuperMimic/CHORD同预算、世界模型接触与物理梯度、材料鲁棒性及硬件观测契约均继续保留。

## 接下来的两条实作

物理侧先建立具名三指的可执行参考：增加真实middle关节自由度及材料接触点，完整几何/原生载荷/控制裕量验证后做原门槛动态基准。没有成功持握参考的数据不应被写成成功ego示范。

学习侧保留raw80小模型作为状态/时序/因果诊断基线；它尚未接上游完整几何/接触特征、Actor、奖励和CHORD训练。下一模型需要TRAIN内同状态动作干预数据、物理基线与接触扳手/动量约束，并检查没有接触因果路径的指令是否仍产生伪物体响应。不能仅继续优化这套dense80预测误差就宣称完成原SuperMimic/CHORD迁移；需要重新接入明确的phone几何/接触特征及控制/观测契约。

当前476个VAL分支保持VALIDATION用途，不能回填TRAIN；已公开结果的TEST也不能再宣称未见。后续以新最终检查family保留评估，再按完整协议进行SuperMimic/CHORD同预算比较。新训练仍遵守首健康checkpoint和ETA后暂停目标的约定。本轮所有执行已终态、无新参数训练，目标继续active。
