# 手机首轮无手物理测试 v1

本协议是预先拟定的工程基准，不是实物参数标定。对应机器接口为同名 JSON，schema `phone_physics_stage1_protocol:1`。作者未运行物理、安装软件或训练；旧 Goal/实验状态不改。

**对象与假设。** HOT3D `cellphone`/BOP24，扫描资产约 72.9293×10.3159×153.9552 mm。实际碰撞盒必须绑定资产生成器的未舍入 AABB 与中心，GLB node 毫米→米缩放只烘焙一次。原局部 Y 薄轴经 Rx90 转到世界 Z；root 平移补偿旋转后的 bbox center，使 COM 正确落位。质量为 0.1/0.2/0.3 kg，滑动摩擦为 0.3/0.6/0.9，九个组合全部保留；名义点为 0.2 kg、0.6。本 bench 显式覆写为 condim3/pyramidal；扭转摩擦长度 0.005 m、滚动长度 0.0001 m 留在材质字段中但均不激活。资产原 condim4 默认值只记录、不改原件。两 geom 参数相同。这些都是先验假设。COM 取 bbox center，惯量按各质量的均匀盒公式显式给出，均非实测。

**三项测试，每项 3 秒。** 桌面是 z=0 的固定平面，重力 −9.81 m/s²，初始速度全零，无手、执行器或位姿强制跟随。

1. Drop：最低点离桌 0.1 m，自由落下。
2. Table settling：最低点离桌 0.5 mm，自由静置。
3. Force-push release：同静置初态，先运行至 1.0 s；在 `[1.0,1.2)` 对瞬时 COM 施世界 +X 的 2 N 力，无 COM 附加扭矩；随后完全释放。输入冲量为 0.4 N·s。高摩擦/质量组合不滑动也是合法结果，不作为失败。

每项使用 2.5 ms 与 1.25 ms 两个固定时间步。先测名义点，再覆盖全部九点；每后端 54 条轨迹、97,200 步。CPU 与 Warp 均须完成，才能授予全阶段工程资格。旧 5 ms 测试只可作为历史描述，不能替换细时间步对照。

**执行前绑定。** 保存原 GLB、baked mesh、碰撞模型、变换、资产生成器、runner、两后端版本的 pin；精确记录质量/惯量、有效摩擦混合、contact margin/gap、condim、solref/solimp、cone、solver、integrator、迭代/容差、warmstart、sleeping 和容量。两后端保持同一物理设定。若某选项不支持，明确阻断 parity，不能静默使用不同默认值。各 case 全新初始化，运行中不重置物体位姿。实际 dt 的值/类型与步计数保存，按相同整数 tick 比较，避免累计时钟舍入造成错误连接。

**几何单独验收。** 报告所有扫描顶点到盒面的有符号距离，以及固定间距不大于 2 mm 的盒面样本到扫描三角面的距离：min/max、RMS、p95、数量、坐标/单位和缺陷。预拟界限：扫描顶点在盒外不超过 0.5 mm，盒面采样到扫描 p95≤2 mm、max≤4 mm。此为采样误差，不称完整 Hausdorff 或物理表面真值；缺失计算标 UNKNOWN，不能填零。软件 bench 可保留观察，但 geometry 未过不能给完整阶段资格。

**预拟工程门槛。** 全状态、能量和输入有限，无失稳/容量/截断告警；最大生成接触穿透≤2 mm，最后 `[2.5,3.0]` s≤0.5 mm。该窗口 COM 速度 RMS≤0.01 m/s、max≤0.02 m/s，角速度 RMS≤0.1 rad/s、max≤0.2 rad/s。记录 `T+mgz_COM`、施力功及原生能量；被动窗口正能量增长不超过 `max(0.0005 J,10%*max(|E_start|,0.005 J))`。软接触储能未包含于 `T+U` 的限制必须注明。末窗口有生成桌面接触；不要求每步接触数、力波形完全相同，也不把零候选解释为全局间隙证明。

**时间步与后端门槛。** 每个相同参数/测试的终态位置差≤2 mm、姿态差≤2°、线速度向量差≤0.02 m/s、角速度向量差≤0.2 rad/s；相同 tick 的 COM 轨迹 RMS 差≤3 mm，首次生成接触时间差≤5 ms。先分别检查 CPU/Warp 的时间步对，再检查所有同 dt 的 CPU/Warp 对。保存全轨迹和失败项，不选表现好的参数，不在看到结果后放宽 v1。

**报告范围。** 输出 schema `phone_physics_stage1_result:1`，分别给 geometry/CPU/Warp/parity/full-stage bool、全 case、输入/轨迹/source pins 与失败原因。通过仅表示“声明假设下可重复的手机盒无手工程基准”；`real_parameter_calibrated`、`ego_retargeting_qualified`、`hand_task_qualified` 永远为 false。不能外推杯子凹形/把手、白板笔、旧策略、人手力或真实 HOT3D 接触动力学。阈值是预拟验收口径，不是由实物观察得到，也不是质量/摩擦识别方法。

**执行前已有几何负例。** 拟定上述几何界限后，本作者只读资产报告 `geometry_report.json`（SHA `614d7f60ddda8e4249fa75516137874b36b771b927323b85940fbd5362a0bba2`）：八个盒角到扫描三角面的距离为 4.59–6.01 mm，已超过 max 4 mm 的工程界限。全盒面 p95 尚未产生。因此该资产在 v1 下不能获得 geometry/full-stage 资格；仍可独立运行并报告 CPU/Warp 盒体稳定性。不得因已知结果放宽本版界限，也不能把几何负例说成物理参数已知或 bench 不可执行。

建模依据：[MuJoCo Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html) 的 visual/collision 分离、惯量与单位检查、mesh 凸化及接触参数混合说明；具体运行版本仍须由 execution manifest 绑定。本阶段没有实物校准实验。
