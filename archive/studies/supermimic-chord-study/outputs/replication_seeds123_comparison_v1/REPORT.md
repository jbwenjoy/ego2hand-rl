# Seeds 1–3 配对复制结果

**等更新主预算：A1L 1/3，A2 0/3；1200 秒内可获得 checkpoint：A1L 1/3，A2 1/3。**

这里的 N=3 是训练随机种子1/2/3；所有评估固定 seed0、同一物理初态。seed0 pilot 不计入复制统计，151帧只是每条轨迹的采样点。仅作配对描述性汇总，不作总体显著性、p值或置信区间推断。

## 逐 seed 全部结果

|预算|方法|训练 seed|评估 seed|iter|task|shaft max °|progress endpoint °|state超限 mrad|target超限 mrad|support512|失败门槛|
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
|主|A1L|1|0|2000|PASS|16.112|11.420|0.8250|0.00010|0.807993|—|
|主|A2|1|0|2000|FAIL|7.695|5.901|3.3052|0.00005|0.778416|task_joint_limits_ok, joint_state_limits_ok|
|主|A1L|2|0|2000|FAIL|20.327|10.436|1.9458|0.00010|0.795522|task_tracking_ok, task_joint_limits_ok, joint_state_limits_ok|
|主|A2|2|0|2000|FAIL|13.115|11.365|1.4561|0.00005|0.770424|task_joint_limits_ok, joint_state_limits_ok|
|主|A2|3|0|2000|FAIL|12.880|7.361|2.0636|0.00010|0.821941|task_joint_limits_ok, joint_state_limits_ok|
|主|A1L|3|0|2000|FAIL|20.532|9.090|15.2026|0.00005|0.799276|task_tracking_ok, task_joint_limits_ok, joint_state_limits_ok|
|次|A2|1|0|1700|FAIL|9.875|5.203|1.7472|0.00010|0.824609|task_joint_limits_ok, joint_state_limits_ok|
|次|A2|2|0|1700|PASS|12.240|6.773|0.2289|0.00005|0.780088|—|
|次|A2|3|0|1700|FAIL|21.483|15.096|2.1351|0.00010|0.826214|task_tracking_ok, task_joint_limits_ok, joint_state_limits_ok|

主预算有 **3/3 对 seed 的 shaft 最大误差降低**，这是描述性跟踪信号；整体任务通过数为上述 1/3 与 0/3，不能把局部指标改善等同整体任务收益。主预算A2有 3/3 条实际 joint state 超过冻结1mrad允许量，而执行 target 限幅 3/3 通过；保持原状态和门槛，没有clamp状态或放宽标准。平均轴误差、位置误差与最大轴误差分开列示，避免把最大值改善泛称所有跟踪指标改善。

次预算中的 A1L 是上述同一 final 的 alias，没有重复评估或新增独立样本。A1L 三次均在1200秒前完成2000迭代；A2三次按 OS CreationDate 规则选1700，未按评估质量选 checkpoint。不能称两臂均耗满1200秒。

主预算两臂每seed均2000 iter、648192真实 transitions、WM/failure-head各4000、actor1800、critic7200更新；次预算A2为552192 transitions、WM/failure-head各3400、actor1500、critic6000。

## 配对差值与描述性范围

差值均为 A2−A1L；误差/越限越低越好，辅助接触分数越高越好。

|预算|指标|seed1差值|seed2差值|seed3差值|均值|中位数|最小 / 最大|
|---|---|---:|---:|---:|---:|---:|---:|
|primary|position_mean_mm|-0.53922|-0.10062|+1.23948|+0.19988|-0.10062|-0.53922 / +1.23948|
|primary|position_max_mm|-0.43114|-0.69684|-1.90139|-1.00979|-0.69684|-1.90139 / -0.43114|
|primary|shaft_mean_deg|-2.64355|+2.81858|-0.21667|-0.01388|-0.21667|-2.64355 / +2.81858|
|primary|shaft_max_deg|-8.41769|-7.21198|-7.65120|-7.76029|-7.65120|-8.41769 / -7.21198|
|primary|progress_endpoint_error_deg|-5.51901|+0.92964|-1.72988|-2.10642|-1.72988|-5.51901 / +0.92964|
|primary|joint_state_violation_mrad|+2.48022|-0.48976|-13.13896|-3.71617|-0.48976|-13.13896 / +2.48022|
|primary|support512|-0.02958|-0.02510|+0.02267|-0.01067|-0.02510|-0.02958 / +0.02267|
|secondary|position_mean_mm|+0.55347|+1.31114|-0.73951|+0.37503|+0.55347|-0.73951 / +1.31114|
|secondary|position_max_mm|+1.58955|-0.22726|-3.93038|-0.85603|-0.22726|-3.93038 / +1.58955|
|secondary|shaft_mean_deg|-2.72655|-0.31281|+4.69291|+0.55118|-0.31281|-2.72655 / +4.69291|
|secondary|shaft_max_deg|-6.23768|-8.08706|+0.95194|-4.45760|-6.23768|-8.08706 / +0.95194|
|secondary|progress_endpoint_error_deg|-6.21725|-3.66244|+6.00579|-1.29130|-3.66244|-6.21725 / +6.00579|
|secondary|joint_state_violation_mrad|+0.92220|-1.71690|-13.06744|-4.62071|-1.71690|-13.06744 / +0.92220|
|secondary|support512|+0.01662|-0.01543|+0.02694|+0.00937|+0.01662|-0.01543 / +0.02694|

所有任务门槛、每组连续指标的mean/median/min/max及逐seed原值均保存在 comparison.json，CSV列出9条唯一物理rollout。没有把缺失项转为0或失败。任务总门槛须全部满足，某个跟踪指标或辅助分数改善不抵消实际关节状态越限。

## 成本口径

|方法|seed|训练记录秒|queue GPU slot秒（含CPU health）|peak allocated GiB|
|---|---:|---:|---:|---:|
|A1L|1|915.506|920.370|0.539|
|A2|1|1348.920|1353.561|7.622|
|A1L|2|902.131|907.269|0.539|
|A2|2|1345.281|1349.611|7.622|
|A2|3|1345.947|1350.493|7.622|
|A1L|3|888.481|892.345|0.539|

等更新不等于等compute。GPU slot是单卡预留墙钟，不是利用率或CUDA计算时间；训练记录与queue含健康检查口径分别列示。9次独立评估成本单列于JSON，secondary alias不重复计费；此前工程smoke、失败编译和seed0 pilot费用不混入此复制实验成本，也不声称为研究总费用。缓存按冻结queue连续复用，warm history随顺序变化。

## 证据与适用边界

九条rollout均核对计划、原训练输入、raw checkpoint config、condition及metadata；training_seed与evaluation_seed分开保存。reference/model/protocol、规范time、raw physics clock和初态q/qd/object pose/twists/wrist/初始target逐值核对；验收从原NPZ重算，512分数由真实solver-active手物接触几何对独立固定意图计算。

A2是soft geometric prior/anticipation，而非合格的接触力预测器。训练64方向与物理评估512方向分开；局部代理/动力学梯度资格失败和standalone compiled-precision失败记录均保留，实际prior走eager，原生子网编译。该辅助score既非受力真值也非任务成功，不能据其变化作物理接触收益结论。

A2同时加入support吸引与penetration prior，没有support-only或penalty-only消融，不能把任何变化只归因support。本实验只有clip007[200:351]固定腕重设计、同一物体、同一初态；无真实无本体ego输入，原视频父ID未知。三训练seed不构成跨视频、跨物体或初态泛化。门槛是50Hz模拟任务标准，不是连续时间/硬件安全认证。

保留的独立失败证据：[compiled precision](D:/workspace/supermimic-chord-study/outputs/a2_cuda_pointmax_final_20261102/result.json)、[geometry/dynamics proxy qualification](D:/workspace/supermimic-chord-study/outputs/contact_geometry_cpu_exterior_v3/report.json)。

![全部跟踪轨迹](tracking.png)

![逐seed配对图](paired_effects.png)

## 复现

```powershell
& 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe' 'D:\workspace\supermimic-chord-study\scripts\summarize_replication_seeds.py'
```
