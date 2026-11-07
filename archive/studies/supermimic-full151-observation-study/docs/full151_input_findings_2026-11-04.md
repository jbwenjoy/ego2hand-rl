# full151 连续参考与接触几何：首次受控 CPU 结果

本轮已实际构建 legacy clip007[200,351) 的151帧、3秒连续关节参考和150条计划目标，保留当前已测H123的原始P0/P1、q0、非零解析qd0及实际u0。**该候选未通过原数值合同，也未通过全段采样穿透标准；没有READY，没有新训练或物理积分。** Goal继续active。

这是一条已有上游缓存的受控观测视图，不是新增独立ego数据。它与shared十条参考的[0,151)窗口不同。源相机、尺度及置信度仍未独立校准；逐帧palm-root映射是固定腕任务重设计，不能解释为已标定的相机外参。

## 输入与接触意图

允许的观测源为[observation fixture](D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1/manifest.json)。拟合目标使用已保存的539d派生几何关节轨迹；它是观测重定向输出，不是原始机器人答案。原始记录的机器人关节、控制及接触标签没有加载。

[全151帧观测线索](D:/workspace/supermimic-full151-observation-study/outputs/observation_intent_cues_v1/READY.json)保存landmarks 3/7/11到物体表面的最近点、法向、距离、feature/tie和有效性掩码，21项CPU边界检查通过。它们没有被阈值化为接触/释放真值。`human_contact_known`、`declared_robot_intent_known`均为false，`target_ready`与`objective_loading_permitted`均为false。旧16帧机器人几何派生的contact-intent没有扩写为151帧。

源坐标独审显示，旧R/F16与全151派生几何的前16时间/物体映射/归一化手目标一致，但q0相差1.142272 rad。当前拟合使用固定原P0/P1求解全段曲线，没有把H123 q0拼到旧q1，也没有修改qd0或u0来隐藏边界问题。

## 实际拟合与控制结果

[正式协议v2](D:/workspace/supermimic-full151-observation-study/protocols/full151-input-cpu-v2.json)与[生产入口v2](D:/workspace/supermimic-full151-observation-study/scripts/build_full151_input_cpu_v2.py)在执行前冻结；v1未执行，保留供审阅。v2仅修复求解器抛异常时的attempt计数，并记录CUDA可见性，没有改变科学输入或约束。

使用153个三次B样条控制点、固定P0/P1，每关节151个自由系数及302个L1辅助变量，共453变量/2402不等式。目标为平均关节绝对拟合误差加4倍控制系数二阶差分绝对值；后者不是精确物理加速度积分。每个关节仅调用一次HiGHS，单次时间上限5秒；没有重试或结果择优。

| 检查 | 实际结果 |
|---|---|
| 求解器返回 | 20/20 success |
| 原1e-12数值证书 | 15/20关节通过，整体FAIL |
| 最大原始Ax-b | 8.1134987617e-10，joint14的fit辅助行 |
| 真正功能约束的超量 | joint4紧化target上界行1.1025047542e-10；原执行域仍通过 |
| q控制点硬范围 | 最大越界0 |
| 原FP32初始化 | q0、解析qd0、object pose/twist、u0保留 |
| 原始C2域与NumPy/Torch/原clamp | 150条全部通过，0个C2投影改动 |
| 149次目标更新 | 最大14.9999007583 rad/s，低于原15标准 |
| 全区间参考PD Bernstein及601个FP32样本 | 原actuator/joint force caps均通过 |
| 总门禁 | 33/34，唯一失败为全20关节原始数值证书 |

失败关节为0-based `[4,12,13,14,15]`。[实际receipt](D:/workspace/supermimic-full151-observation-study/outputs/full151_cpu_input_v1/receipt.json) SHA `ae32b64e0e66a26fe5657e20e8efcf1d5624d859871cb5751f7c3fb9efd5c7d1` 与所有原始矩阵/解/残差均保留；不因HiGHS success或原控制域通过而重判READY。计划控制表的149次FP32目标更新略高于紧化规划速率14.9999，但低于原15门槛；规划值与原执行标准分别记录。

解析初速度峰值16.055520602 rad/s不是上述目标更新率的越界；两个量不同。参考PD证书也不是实测关节力或接触力。目标索引仅0..149，终点t=3s是状态诊断，没有u150。

相对539d目标，151帧平均绝对关节误差.013219071 rad；前16帧去q0为.0500321 rad、后135帧为.00701510 rad。新曲线前16帧相对原R/F16平均绝对变化.2431604 rad、最大1.1629556 rad。相同初态与物体曲线不等于相同关节路径，旧prefix的几何或物理资格不能继承。

[独立保存数组复算44/44](D:/workspace/supermimic-full151-observation-study/outputs/full151_cpu_input_saved_independent_v1/report.json)重算全部20组Ax-b、边界、目标、FP32控制、Bernstein负载、初始化和物体曲线，确认上述结果；没有重求解或运行模型。[来源/执行语义独审26/26](D:/workspace/supermimic-full151-observation-study/outputs/full151_input_saved_lifecycle_independent_v1/report.json)保留整体FAIL/noREADY。短前台命令的exit1由Root工具观察记录，PID/Creation未采样，没有伪称独立OS生命周期证明。

![full151 CPU候选数值诊断](D:/workspace/supermimic-full151-observation-study/outputs/full151_cpu_input_summary_v2/full151_candidate_summary.png)

## 601点原生过滤几何负例

随后按[独立几何诊断协议](D:/workspace/supermimic-full151-observation-study/protocols/full151-failed-candidate-geometry-v1.json)，对**同一FAILED候选**固定执行601个5ms采样点的几何检查。使用原MJB碰撞设置及actual-context的geom_size/geom_pos、原21个启用手碰撞几何与273个固定pair；没有改CCD精度、重拟合、物理积分或力查询。构造时一次nominal FK，加601次采样FK/COM与601次`mj_collision`。step/forward/inverse/geomDistance/contactForce及三个优化API均被计数拒绝，实际调用0。

| 原生生成接触的穿透 | 601点最大值 | 超原3mm的采样点 | 151个原结点最大值/超标数 |
|---|---:|---:|---:|
| hand-object | 15.158818 mm | 350/601 | 13.760507 mm，89/151 |
| hand-self | 5.884525 mm | 16/601 | 5.836744 mm，4/151 |

最深手物重叠在t=.085s、`object/cube`与`robot/right_finger2_link3_col`，不是仅发生于终点或纯插值中点。首次手物超过3mm在.010s；self最大在1.675s。257个采样点存在tied pairs，15个采样点未生成手物接触。

[几何实际receipt](D:/workspace/supermimic-full151-observation-study/outputs/full151_failed_candidate_geometry_v1/receipt.json) SHA `6d217c95479087896bfd7f70c27f8bb4edb5f58facac49aee55d7057481d6607`。其工程检查7/7通过，采样穿透标准为false；工程完成不代表几何成功。未生成pair的gap保持NaN，不能当作正间隙、实际释放或无接触安全证明。这601点也不证明区间之间的连续几何安全。保存的FK21点属于旧重定向宏点，并非native BODY21参考ABI。

[几何保存数据独审35/35](D:/workspace/supermimic-full151-observation-study/outputs/full151_geometry_saved_independent_v1/report_v2.json)从601行原生接触记录逐位复算273pair的deepest distance、mask、counts、ties与全部统计，原生catalogued contact共3126条、uncatalogued为0。首版审阅器仅有字典迭代的harness错误，旧源/报告保留，v2仅修.values()后消费同一数据，没有重查询或重拟合。后段.305–3s仍有331个手物采样点超过3mm（最大5.050mm），所以问题不只在初态过渡。15个未生成手物接触的点集中.615–.685s，保持UNKNOWN。

![全151候选的601点原生过滤几何诊断](D:/workspace/supermimic-full151-observation-study/outputs/full151_candidate_geometry_summary_v2/full151_sampled_geometry.png)

## 对后续研究的影响

这次负例直接说明：对这一固定初态/映射/权重的候选，关节范围、控制更新率和参考PD负载合法，仍会出现明显手物与自碰撞穿透。既有539d逐帧几何结果没有自动迁移到连续控制参考。该结论不等于原运动全局不可行、SuperMimic/CHORD无效，或新的全段接触方法已经验证。

下一步优先研究在完整连续曲线中加入经过校验的接触几何约束及独立意图假设，并保留tie/未知接触的处理边界。[数值跟进提案v2](D:/workspace/supermimic-full151-observation-study/docs/full151_numerical_followup_proposal_v2.md)中的固定centered变量/row scaling仅为未执行的数值表述提案，不能解决此次实际穿透，也不能替代接触修正。任何后续求解都用新协议与新产物，保留当前失败结果；不通过放宽原1e-12或3mm门槛获得通过。

native151的controller、解析速度/partial reset、H8 survival/replay、终点处理、新checkpoint身份及完整物理评估仍未接入。shared多轨迹及真实ego/硬件验证仍未完成。没有新长训练，因此本轮没有触发“首个健康checkpoint后估时暂停”的条件。

本轮首次fit内部wall4.998725s，其中20次solver合计.256585s；工具前台命令wall5.805594s。几何内部wall.771148s，工具命令1.064602s。各时间范围嵌套，另有审阅、导出及作图成本，不能相加成完整研究开销或与CHORD训练时长比较。新物理积分/WM前向/训练更新均0。
