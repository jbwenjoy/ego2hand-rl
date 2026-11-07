# 研究方案 v0.25：真实表面材料点与支持可行性分阶段求解

2026-11-07。目标保持 active；上一回合与本回合均为 progress。完整范围承接 [v0.24](research-plan-v0.24.md) 及原 SuperMimic + CHORD 目标，不以本轮窄实验替代真实ego时序、同预算学习、世界模型校验或硬件契约。本回合没有学习训练或新的持握积分；下次新训练首个健康checkpoint及ETA后暂停的约定保留。

## 本回合改变了什么

已把当前接触定位到实际凸碰撞体的三角面，并实现同一次原生forward的接触Jacobian/PD导出和按手指汇总预算的支持LP。随后实际测试了三个手机长轴平移参考，再执行两个起点的两指/手机姿态联合搜索。没有取得可进入持握评测的候选，未启动新的2秒持握或训练。

新的证据把下一步指向**先保证所选表面对可装配，再在几何硬约束内求支持**。单一加权目标可以降低条件点力残差，同时增大法向不对置和指垫压缩；本轮最终独立门正确拒绝了这些候选。优化器停止、低条件残差、无非目标碰撞分别都不能代替支持或持握资格。

## 实际表面及材料点

[接触面目录](D:/workspace/ego-data-pilot-2026-11-07/contact_patch_v7/report.json)从两个旧保存姿态恢复出相同的compiled FP32局部顶点：thumb1982、index1848、phone352，跨姿态逐字节一致，无模型加载、FK或物理重跑。保留实际三角形、法向、面积、顶点ID、近共面分组与边界。大平面存在，但没有证据说明最大平面就是功能指腹；凸化外壳也可能包含安装面或填平结构。

v6初态的三个原生点落在sensor小曲面细三角附近。两个index点只相距6.494µm，不能据此赋予两个独立承载区或有限接触斑块。分组边界距小于1µm只对应所用平面/曲率分区，不能称宏观锐边。功能pad、压力分布、真实柔顺及面级摩擦仍UNKNOWN。[简明目录](D:/workspace/ego-data-pilot-2026-11-07/contact_patch_v7/catalog_compact_v1.json)、[解释](D:/workspace/ego-data-pilot-2026-11-07/contact_patch_v7/design_recommendations_v1.json)

另冻结thumb首点和index首点分别在双方**实际填充三角形**上的最近投影，而非质心：保存hand geom-local与phone body-local点、三角形顶点、重心坐标和外法向。重构误差≤7.4e−15m；对应原ego材料候选偏差约4.813/4.738mm。它们只是共享原native记录来源的两个独立表面投影，不是已认证的唯一最近点对；未选的近重复index点仍保留。[固定表面材料点](D:/workspace/ego-data-pilot-2026-11-07/contact_patch_v7/paired_surface_anchors_v1.json)

## 同次原生接触力学接口

新包装器继承冻结v6评估，访问同一fresh MjData，在原生con.pos和其手侧body计算Jp，不再做第二次forward。导出实际friction5、condim、frame、物体受力符号、手指分组及完整gain/bias/gear/ctrl/actuator/joint-force数组。两个重复probe首次87/87通过，共2forward、2COM Jacobian、6contact Jacobian及6接触力读取，0积分。两index点共享finger2预算，不按点数量扩力。[接口](D:/workspace/ego-data-pilot-2026-11-07/native_contact_design_v7/contract_v1/report.json)

新原生四射线LP使用任意数量的实际点，并约束τ=bias−passive+ΣJ_handᵀF_object、真实PD/actuator/joint-force交集、非负系数与每指3N设计预算。四个解析案例首次通过：中心对夹可托重、真实力矩上限阻止托重、交换geom顺序不改变物体受力锥、多点不倍增同指预算。该交集给合法未饱和反解命令的witness，不冒称任意模型下全部饱和平台可达集，也不借joint frictionloss虚构支持。[解析检查](D:/workspace/ego-data-pilot-2026-11-07/native_contact_design_v7/analytic_contract_v1/report.json)

另形成force-hard、moment-box的支持witness接口，避免minimax把力/矩残差折中后漏掉原门内解。原始验收仍为F∞≤1e−8N、M∞≤1e−4Nm；新guarded变体只将求解用moment box内缩1e−10Nm，原始数值验收不放宽。缩界不可行不证明原box不可行。该接口本回合经过源审，但最终六个候选均先败在完整几何，故没有实际执行这一strict LP，不能写成数值验证通过。[guarded源](D:/workspace/ego-data-pilot-2026-11-07/native_contact_design_v7/strict_support_lp_guarded_v7.py)

## 三个真实长轴平移参考

固定原腕、q20、手机R和全部物理参数，将手机沿原body+Z平移0/24.5/49mm，各做一次fresh forward、完整22手机对/189自碰对及实际多接触/真实PD支持LP。三个姿态均通过本轮[-0.1,+0.2]mm指定pad间隙窗口及其他非穿透门；模型没有缩放、改变材料或添加外支撑。

| body-Z位移 | 六维缩放∞残差 | F∞残差 | M∞残差 | 原ego材料点偏差范围 |
| --- | --- | --- | --- | --- |
| 0mm | 0.282215 | 0.553705N | 0.042635Nm | 4.737–4.810mm |
| 24.5mm | 0.180804 | 0.354738N | 0.027315Nm | 28.120–28.152mm |
| 49mm | 0.085071 | 0.166908N | 0.012852Nm | 52.531–52.558mm |

三例均未通过严格支持；力矩余量和3N预算未成为活跃瓶颈。残差减小伴随明显材料重分配，不能称faithful ego重定向改善。一次进程exit0，3forward/3完整几何/3LP，无额外逆控制probe或积分；保存算术9/9通过。[实测](D:/workspace/ego-data-pilot-2026-11-07/native_contact_design_v7/body_axis_scan_v1/report.json)、[保存复核及图](D:/workspace/ego-data-pilot-2026-11-07/native_contact_design_v7/body_axis_scan_saved_recheck_v1/report.json)

保存的三点受力分析也区分了必要条件：没有找到把所需向上合力分离在当前force cone外的平面，不能说完全没有托重力方向。三个点近乎两点，其接触线轴向重力矩约0.007418Nm，而3N预算下两index点微小间距能提供的该轴矩上界仅约1.19e−5Nm。消除这一单轴矩的body-Z位移约59.55mm只是必要条件，不保证六维支持。[分析](D:/workspace/ego-data-pilot-2026-11-07/reference_seed_design_v7/report.json)

据此另锁定并执行一次条件COM轴LP：在硬ΣF=−mg下，COM=C0+aδ使moment=M0λ−δ(a×Ftarget)线性；当前world点/法向/J不动，δ∈[0,75]mm，真实力矩和每指3N不变，要求完整精确零矩。HiGHS返回不可行，δ和u为null；没有伪造新位姿，也没有执行其后验原生验证或持握。结论仅属于固定接触几何、固定轴和精确零矩的条件问题，不能否定其他真实位姿、有限moment box或动态控制。[唯一执行结果](D:/workspace/ego-data-pilot-2026-11-07/reference_seed_design_v7/axis_lever_LP_v1/actual_v1/report.json)

## 固定材料点的两起点姿态搜索

为保留原材料接触位置，新建独立算法实例，将上述实际hand局部点和phone body点设为固定材料点，物理模型不改。变量是thumb/index前8关节与手机palm平移/rotvec共14维，另外12关节、手腕、尺度、质量、摩擦、执行器和求解器保持原样。原起点与预先给出的5°倾斜起点均执行，全部结果保留。

目标包含B−A+25µm·nH的压缩配对、nH+nO法向对置、保守µ/√2八射线6D残差、全部几何负约束与弱姿态项。A是实际手表面材料点、B是手机表面材料点，两者未完全装配前只是条件点力模型；切向反力在A/B之间可产生小力偶，同法向包含关系也不能跨法向误差直接继承。因此短名单必须另算真实native multi-contact/J、完整22/189和新的支持LP。

一次进程正常退出，stderr空，总wall约166.69秒。原起点235次搜索尝试后xtol停止，但optimality约3836，不称已收敛；倾斜起点637次尝试触120秒搜索deadline。六个短名单全部通过新的原生接触导出，但全部不满足面片配对和最终几何，持握入组数为0。

实际计数为878次静态FK评估、2244次重型凸查询、877次LP、6次最终完整几何与6次fresh native forward，0积分/训练。最后一次deadline尝试可能已经消耗FK但未进入LP，不能要求搜索尝试数等于成功LP数，也不能自动重启补做。[完整结果](D:/workspace/ego-data-pilot-2026-11-07/surface_anchor_pose_v8/completed_v1/report.json)、[终态](D:/workspace/ego-data-pilot-2026-11-07/surface_anchor_pose_v8/run1_terminal.json)

来源计数审9/9通过：872次尝试对应871条完整条件LP日志及1次deadline缺口，877 LP=871条件+6原生，strict witness LP为0。失败点的完整x14和内部停止位置未保存，不做补记。短名单使用预定近邻去重，可能保守漏登很近但更优的base；不声称穷尽全部候选，下一接口应在同近邻簇中用更优base替换旧点。[来源计数审](D:/workspace/ego-data-pilot-2026-11-07/surface_anchor_pose_v8/saved_count_review_v1/report.json)

原起点三例的材料点配对位置误差≤81µm，但法向偏差约13.7°/9.0°，且两pad压缩超本轮0.1mm上限约0.9–9.9µm；其余nonpad/self约束未负。条件8ray loss约0.260，实际native4ray loss反而约0.287，未构成真实支持改进。法向1°及pad初始窗口是这次工程候选假设，失败不证明其他法向锥、多面接触或全部持握不可能。

全部六例保存算术9/9通过；旧复核v1把per-contact µ数组当scalar的适配失败另保留，v2只修保存读取，未重跑物理或LP。倾斜起点thumb配对误差约271µm、法向偏差约21.9°，也未过门；六例唯一负几何项都是两pad压缩，总超额约0.3–18.4µm。实际native接触点距固定B材料点约1.27–1.62mm、法向差约6.8–16.1°，故条件/原生loss差异不能仅归于摩擦锥变化。原生index Fn约3.91–4.30N，也不能从理想LP≤3N推断真实预载合预算；原生反力没有裁剪。固定phone材料点和其余12关节均逐字节保留。[完整保存复核](D:/workspace/ego-data-pilot-2026-11-07/surface_anchor_pose_v8/saved_recheck_v2/report.json)

## 下一步与完整研究范围

下一轮先做所选真实表面对的**装配可行性**，暂不以wrench收益交换法向/压缩误差。法向约束可用phone面的两条切向投影各2个独立等式，加反向分支条件，避免把单位法向三个分量当三个独立硬等式。通过配对位置、法向和完整碰撞门后，再在这些硬几何约束内优化支持；保留原失败与完整计数。

若当前三角面对在有限范围内无法装配，沿实际sensor邻接面域选择有记录的替代三角/重心点，核法向和原材料点偏差。不能以最大平面、任意质心或微米重复点冒功能pad/有限压力斑块；不增加源中无依据的接触来源。条件支持仍需真实新接触、合法控制、原生预载以及原2秒无辅助持握门。任何新安装姿态、第三指或材料重分配条件须具名并做匹配对照。

通过名义持握后，继续双步长与动力学先验鲁棒性、61帧时序、手机COM/20ms子步/完整姿态接口、SuperMimic/CHORD同预算训练及世界模型接触/梯度校验、独立轨迹/泛化、真实资产与动力学参数校准及硬件观测契约。当前手机质量/COM/惯量/摩擦与源GT重力注册、功能指腹语义仍未获实证；这些边界不会因窄检查通过而消失。真实ego已用于资产与几何参考，尚未用于手机策略训练。
