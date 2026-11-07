# full151 观测重定向输入阶段

该目录独立构建 legacy clip007[200,351) 的完整151帧连续参考与150个计划控制量，不覆盖当前rough104或shared[0,151)产物。输入只使用允许的观测及有来源的派生几何；原始机器人关节答案、控制和接触标签不进入输入。

当前阶段为CPU规划，不运行环境或训练。固定当前已测H123 q0、非零qd0与u0初始化边界，使用全151派生几何作拟合目标，避免q0拼接跳变。全段contact-intent需独立定义，原16帧标签不会重复成151帧；human contact保持UNKNOWN。

参考状态的PD负载证书不等于真实动力学、实测力或物理成功。完整native151模式、独立checkpoint身份、真实reset/replay及全运动评估仍需接入。

当前结果：20次固定LP已完成，但原始1e-12数值门槛失败（5关节），33/34门禁通过、无READY。随后601点原生过滤几何诊断发现手物15.159mm/self5.885mm重叠，350/16个采样点超过原3mm标准；没有新物理积分或训练。旧prefix资格不能继承。

[本轮研究记录与后续优先级](D:/workspace/supermimic-full151-observation-study/docs/full151_input_findings_2026-11-04.md) · [数值独审44/44](D:/workspace/supermimic-full151-observation-study/outputs/full151_cpu_input_saved_independent_v1/report.json) · [几何独审35/35](D:/workspace/supermimic-full151-observation-study/outputs/full151_geometry_saved_independent_v1/report_v2.json)

后续全段接触耦合已完成：164次局部probe通过两处指定unique分支，ties仍不合格；8步SCP所有原/新数值与控制门通过，但η和几何失败，HO峰值11.448mm、自碰11.431mm。候选保持FAIL/noREADY，下一重点为明确known-rough身份的native151真实接入，几何研究并行。

[最新接触耦合研究记录](D:/workspace/supermimic-full151-observation-study/docs/full151_contact_coupled_findings_2026-11-04.md) · [数学独审173/173](D:/workspace/supermimic-full151-observation-study/outputs/full151_scp_saved_independent_v1/report.json)

最新：native151真实A0已跑完150interval/600solver/601state，工程/容量/身份通过，任务FAIL三项(38.003mm/60.249°/7.941mm)。新105controller/runtime/消费者已实现，但真实学习Runner/105checkpoint健康仍pending，无新长训练。

[完整3秒原生基线记录](D:/workspace/supermimic-full151-observation-study/docs/full151_native_A0_findings_2026-11-04.md) · [工程enrollment](D:/workspace/supermimic-full151-observation-study/outputs/full151_native_A0_engineering_enrollment_v1.json)
