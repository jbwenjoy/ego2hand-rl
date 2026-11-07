# 缓存关键点到 Hand2 的几何重定向原型

输入仅为已导出的151帧观测视图：时间、21点手骨架和物体位姿。程序不读取原机器人关节轨迹，也不读取缓存腕位姿。机器人MJCF/MJB的已知运动学与静态默认姿态是允许的模型先验，不是演示的关节答案。

这不是原始ego视频重建，也不是已合格的物理参考。源数据仍来自上游运动学数据中的缓存关键点；相机/尺度、解剖对应、右手语义与置信度均未独立校准。

新增掌根v2分支使用固定link1原点代替随关节运动的link2原点，并用当前宏段长度计算方向Jacobian。其新归一化目标的指尖均偏差4.988mm、平均/最大骨方向误差6.359°/17.377°；151帧几何q合规，8项合同与[13项独立复算](outputs/independent_palmroot_review_v1/report.json)通过。原v1目标和产物保留，新旧目标/掌basis不同，不能把数值差当作相同目标的优化改善或解剖标定证据。

掌根v2未修正候选的初始手物筛查：在实际固定geom_size/geom_pos下，26个手碰撞体与3个笔组件的最近负距离，在全部151帧均超过3mm深度，约5.05–15.11mm；原生CPU碰撞查询也复现。11个非最小的非负/零距离返回与segment不一致，完整gap oracle资格保留未知，负距离复核见[查询报告](outputs/palmroot_contact_geometry_screen_v1/reproducible_query_review_v2.json)。这批历史候选与下述修订分别保留，均没有物理推进或接触标签推断。

## 手物与手自碰几何修订（2026-11-03）

严格查询路径保留了明确反例：第105帧的`right_finger1_link4_col`与笔轴一对，在该路径数值设置下返回−5.868961mm距离，却给出7.986848mm的witness线段，误差2.117887mm，guard按约拒绝；交换几何顺序或归一化法向不能使该结果取得资格。原始MJB数值设置下同一对返回约+5.868961mm，原生碰撞也未生成该对接触。见[冻结失败快照独立复现](outputs/contact_geometry_independent_review_v1/failure_snapshot_review.json)。这限制了该查询路径的资格，不能放宽guard、静默修正或声称所有状态都有可信距离与导数。

保留原始MJB数值设置后，先仅加入手物软重叠惩罚，在同一151帧归一化目标和固定mapped object下将成对查询最大深度降至0.515440mm；原生生成手物接触深度为0.515651mm，但手自碰仍达13.551065mm（147帧超过3mm）。指尖目标误差均值从掌根v2的4.987606mm增至9.840386mm；148帧优化器正常终止、3帧未收敛全部保留。见[object-only原报告](outputs/contact_refined_nominal_geometry_v1/report.json)与[20项独立复核](outputs/contact_refined_nominal_independent_review_v1/review_summary.json)。这一步没有取得抓握或物理可行性。

最终分支使用原生过滤后的生成接触：21个启用的手碰撞体，210个手自碰槽及63个手物槽；未生成接触的距离保持NaN、残差为0，不能解释为已知gap或无重叠。掌根目标点及mapped object pose逐位不变。

| 同目标几何候选 | 原生手物最大深度 | 原生手自碰最大深度 | 归一化目标指尖均误差 |
|---|---:|---:|---:|
| 仅手物软惩罚 | 0.515651mm | 13.551065mm | 9.840386mm |
| 原生手物＋自碰软惩罚 | 0.521794mm | 0.505206mm | 11.446535mm |

最终151帧中143帧优化器正常终止，8帧未收敛为54/55/56/82/83/84/85/86；68个不同帧涉及80个并列接触槽，全部保留。几何q无hard-excess，骨方向平均/最大误差8.771468°/31.544343°，q有限差分峰12.171311rad/s。两类最大深度仍略高于0.5mm软allowance，不能写成严格无碰撞；q有限差分也不是实测关节速度。见[最终原报告](outputs/native_contact_refined_geometry_v1/report.json)、[20项独立重建](outputs/native_contact_refined_independent_review_v1/review_summary.json)及[同目标权衡图](outputs/geometry_tradeoff_figure_v1/geometry_tradeoff.png)。

[56项局部合同检查](outputs/native_filtered_contact_local_checks_v2/report.json)只覆盖预声明姿态中唯一、已生成且法向稳定的局部分支；并列、特征切换、残差hinge等反例保留，不能扩成全局光滑或优化收敛结论。全部修订仍为FK/几何碰撞查询，`dynamics_steps=0`、`physical_reference_qualified=false`，没有推断真实known contact、力、phase或PD目标。相机/尺度/掌部对应的不确定性仍在，需要另行无辅助物理grounding。

## 做了什么

用掌部关键点建立每帧局部基，显式去掉掌部位姿变化并映射到MJB默认掌部原点。保留全部R/t和变换后的物体位姿，标明这是固定腕任务重设计。手骨方向按机器人段长归一化，再以解析Jacobian和有界最小二乘求20个关节角；时序与默认姿态只作小权重正则。

原始方向IK使用`mj_kinematics`、`mj_comPos`与Jacobian查询；后续原生过滤分支另外调用`mj_collision`生成几何接触候选，均没有运行`mj_forward`或`mj_step`。几何求解器内部设置q用于FK，不能据此声称物理状态合规。输出不是可直接训练的Reference格式，未补造phase、contact、初始物理速度、PD target或confidence。

## 原始v1结果（保留）

- 151帧优化正常终止，几何q在模型hard范围内。
- 平均/最大骨方向误差6.131°/19.301°。
- 对机器人骨长归一化目标的平均指尖偏差16.173mm；这不是相对真实人手或物体表面的误差。
- 几何q有限差分最大12.517rad/s，不是实测速度或执行目标rate门槛。
- 解析Jacobian与中心差分最大差2.18e-9，手骨特征在输入整体放大10倍后差5.55e-16。物体尺度不因此获得校准。

[原始诊断](outputs/cached_landmarks_ik_v1/report.json)、[8项输入/坐标变换合同检查](outputs/geometry_contract_checks_v2/checks.json)和[18项独立复核](outputs/independent_review_v1/report.json)均已保存。缓存腕位姿全部改为NaN后所有输出逐位一致，额外机器人关节标签及严重非单位物体四元数被拒绝；独立FK与全部151帧点位完全一致。

## 复现

沿用现有独立Hand2环境；没有新增依赖或修改驱动。输出目录必须尚不存在。

```powershell
& 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe' 'D:\workspace\supermimic-observation-retarget-study\scripts\retarget_landmarks_cpu.py' --input-dir 'D:\workspace\supermimic-chord-study\outputs\observation_only_fixture_v1' --robot-model 'D:\workspace\supermimic-chord-study\outputs\a1l_paired_v2_clip007_2000iter_seed1\compiled\nominal_model.mjb' --output 'D:\workspace\supermimic-observation-retarget-study\outputs\new_run'
```

`check_geometry_artifacts_cpu.py`保存输入投毒/拒绝与变换逆还原负例。`render_geometry.py`只读取已生成NPZ重绘，不重新IK或仿真。交付图为[geometry_visual_qa_v4](outputs/geometry_visual_qa_v4/geometry.png)，已检查全部标签、刻度、图例和帧标题；早期布局问题的图仍保留，不作为额外数值实验。

掌根分支使用`retarget_palmroot_cpu.py`及相同参数形式，产物为`outputs/cached_landmarks_palmroot_v2`。`check_palmroot_artifacts_cpu.py`检查新分支。几何筛查命令形式为`screen_geometric_contact_cpu.py --candidate <geometric_retarget.npz> --model <nominal_model.mjb> --context <fixed_physics_context.npz> --output <新目录>`；已有筛查的复核为`review_contact_queries_cpu.py --root D:/workspace/supermimic-observation-retarget-study --compiled D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled --output <新JSON>`。只写新路径，保留原始运行与异常证据。

后续需要校准掌部/解剖锚点与物体尺度，并通过无辅助物理grounding。原v1的16.2mm指尖偏差及后续分支的拟合/软重叠权衡分别保留；上述几何结果仍不支持直接进入物理训练或真实ego结论。
