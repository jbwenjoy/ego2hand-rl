# 三臂历史对照后续探索：A1L / P_ONLY / A2

本报告仅使用训练seeds1/2/3；evaluation seed统一0、同一物理初态。N=3训练seed对，不是151帧，也不包含seed0 pilot。仅报告描述性结果，不计算总体显著性或置信区间。

|预算|A1L通过|P_ONLY通过|A2通过|
|---|---:|---:|---:|
|primary|1/3|2/3|0/3|
|secondary|1/3|2/3|1/3|

## primary 逐seed全部结果

|方法|训练seed|评估seed|iter|task|位置mean/max mm|轴mean/max °|进度终点 °|qstate超限 mrad|target超限 mrad|support512|实际接触帧|失败门槛|
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|A1L|1|0|2000|PASS|5.61/8.78|7.220/16.112|11.420|0.8250|0.00010|0.807993|151/151|—|
|P_ONLY|1|0|2000|PASS|7.95/15.89|7.263/14.910|7.959|0.6500|0.00005|0.818758|151/151|—|
|A2|1|0|2000|FAIL|5.07/8.35|4.576/7.695|5.901|3.3052|0.00005|0.778416|151/151|task_joint_limits_ok, joint_state_limits_ok|
|A1L|2|0|2000|FAIL|4.74/9.35|6.218/20.327|10.436|1.9458|0.00010|0.795522|151/151|task_tracking_ok, task_joint_limits_ok, joint_state_limits_ok|
|P_ONLY|2|0|2000|PASS|4.01/14.21|7.172/13.722|11.040|0.3393|0.00005|0.814429|151/151|—|
|A2|2|0|2000|FAIL|4.64/8.66|9.036/13.115|11.365|1.4561|0.00005|0.770424|151/151|task_joint_limits_ok, joint_state_limits_ok|
|A1L|3|0|2000|FAIL|5.91/14.60|7.186/20.532|9.090|15.2026|0.00005|0.799276|151/151|task_tracking_ok, task_joint_limits_ok, joint_state_limits_ok|
|P_ONLY|3|0|2000|FAIL|7.91/12.93|12.454/24.135|8.333|0.0000|0.00000|0.808270|151/151|task_tracking_ok|
|A2|3|0|2000|FAIL|7.15/12.70|6.970/12.880|7.361|2.0636|0.00010|0.821941|151/151|task_joint_limits_ok, joint_state_limits_ok|

P_ONLY 的 primary 成功seed为 [1, 2]；按本预算独立计数。没有跨预算取每seed最好结果，也没有按评估结果选择checkpoint。

## secondary 逐seed全部结果

|方法|训练seed|评估seed|iter|task|位置mean/max mm|轴mean/max °|进度终点 °|qstate超限 mrad|target超限 mrad|support512|实际接触帧|失败门槛|
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|A1L (alias)|1|0|2000|PASS|5.61/8.78|7.220/16.112|11.420|0.8250|0.00010|0.807993|151/151|—|
|P_ONLY|1|0|1800|PASS|9.04/19.49|8.331/17.186|6.130|0.4807|0.00005|0.829711|151/151|—|
|A2|1|0|1700|FAIL|6.17/10.37|4.493/9.875|5.203|1.7472|0.00010|0.824609|151/151|task_joint_limits_ok, joint_state_limits_ok|
|A1L (alias)|2|0|2000|FAIL|4.74/9.35|6.218/20.327|10.436|1.9458|0.00010|0.795522|151/151|task_tracking_ok, task_joint_limits_ok, joint_state_limits_ok|
|P_ONLY|2|0|1800|FAIL|7.96/14.85|16.635/34.586|31.924|0.2041|0.00005|0.846523|151/151|task_tracking_ok, progress_ok|
|A2|2|0|1700|PASS|6.05/9.13|5.905/12.240|6.773|0.2289|0.00005|0.780088|151/151|—|
|A1L (alias)|3|0|2000|FAIL|5.91/14.60|7.186/20.532|9.090|15.2026|0.00005|0.799276|151/151|task_tracking_ok, task_joint_limits_ok, joint_state_limits_ok|
|P_ONLY|3|0|1800|PASS|6.97/12.57|8.047/14.613|0.269|0.0000|0.00000|0.793245|151/151|—|
|A2|3|0|1700|FAIL|5.17/10.67|11.879/21.483|15.096|2.1351|0.00010|0.826214|151/151|task_tracking_ok, task_joint_limits_ok, joint_state_limits_ok|

P_ONLY 的 secondary 成功seed为 [1, 3]；按本预算独立计数。没有跨预算取每seed最好结果，也没有按评估结果选择checkpoint。

## 目标对照与配对描述

P_ONLY−A1L 是不含support吸引时新增penalty的目标对照；A2−P_ONLY 是在同一penalty系数下增加support的条件对照。下表差值为右臂减左臂。误差/超限降低与分数提高分别解读，不能只看峰值而忽略时间均值或整体门槛。

|预算|对照|指标|seed1|seed2|seed3|均值|中位数|最小/最大|
|---|---|---|---:|---:|---:|---:|---:|---:|
|primary|P_ONLY_minus_A1L|position_mean_mm|+2.33431|-0.72850|+1.99932|+1.20171|+1.99932|-0.72850/+2.33431|
|primary|P_ONLY_minus_A1L|position_max_mm|+7.10172|+4.86080|-1.67195|+3.43019|+4.86080|-1.67195/+7.10172|
|primary|P_ONLY_minus_A1L|shaft_mean_deg|+0.04323|+0.95408|+5.26739|+2.08823|+0.95408|+0.04323/+5.26739|
|primary|P_ONLY_minus_A1L|shaft_max_deg|-1.20200|-6.60439|+3.60363|-1.40092|-1.20200|-6.60439/+3.60363|
|primary|P_ONLY_minus_A1L|progress_endpoint_error_deg|-3.46160|+0.60473|-0.75723|-1.20470|-0.75723|-3.46160/+0.60473|
|primary|P_ONLY_minus_A1L|joint_state_violation_mrad|-0.17505|-1.60651|-15.20257|-5.66138|-1.60651|-15.20257/-0.17505|
|primary|P_ONLY_minus_A1L|support512|+0.01077|+0.01891|+0.00899|+0.01289|+0.01077|+0.00899/+0.01891|
|primary|A2_minus_P_ONLY|position_mean_mm|-2.87353|+0.62788|-0.75983|-1.00183|-0.75983|-2.87353/+0.62788|
|primary|A2_minus_P_ONLY|position_max_mm|-7.53286|-5.55763|-0.22944|-4.43998|-5.55763|-7.53286/-0.22944|
|primary|A2_minus_P_ONLY|shaft_mean_deg|-2.68678|+1.86450|-5.48406|-2.10211|-2.68678|-5.48406/+1.86450|
|primary|A2_minus_P_ONLY|shaft_max_deg|-7.21569|-0.60759|-11.25483|-6.35937|-7.21569|-11.25483/-0.60759|
|primary|A2_minus_P_ONLY|progress_endpoint_error_deg|-2.05741|+0.32491|-0.97265|-0.90172|-0.97265|-2.05741/+0.32491|
|primary|A2_minus_P_ONLY|joint_state_violation_mrad|+2.65527|+1.11675|+2.06361|+1.94521|+2.06361|+1.11675/+2.65527|
|primary|A2_minus_P_ONLY|support512|-0.04034|-0.04400|+0.01367|-0.02356|-0.04034|-0.04400/+0.01367|
|secondary|P_ONLY_minus_A1L|position_mean_mm|+3.42878|+3.21985|+1.06819|+2.57228|+3.21985|+1.06819/+3.42878|
|secondary|P_ONLY_minus_A1L|position_max_mm|+10.70923|+5.49385|-2.03204|+4.72368|+5.49385|-2.03204/+10.70923|
|secondary|P_ONLY_minus_A1L|shaft_mean_deg|+1.11082|+10.41695|+0.86077|+4.12951|+1.11082|+0.86077/+10.41695|
|secondary|P_ONLY_minus_A1L|shaft_max_deg|+1.07409|+14.25927|-5.91820|+3.13839|+1.07409|-5.91820/+14.25927|
|secondary|P_ONLY_minus_A1L|progress_endpoint_error_deg|-5.29030|+21.48873|-8.82131|+2.45904|-5.29030|-8.82131/+21.48873|
|secondary|P_ONLY_minus_A1L|joint_state_violation_mrad|-0.34433|-1.74170|-15.20257|-5.76287|-1.74170|-15.20257/-0.34433|
|secondary|P_ONLY_minus_A1L|support512|+0.02172|+0.05100|-0.00603|+0.02223|+0.02172|-0.00603/+0.05100|
|secondary|A2_minus_P_ONLY|position_mean_mm|-2.87531|-1.90871|-1.80770|-2.19724|-1.90871|-2.87531/-1.80770|
|secondary|A2_minus_P_ONLY|position_max_mm|-9.11968|-5.72111|-1.89834|-5.57971|-5.72111|-9.11968/-1.89834|
|secondary|A2_minus_P_ONLY|shaft_mean_deg|-3.83737|-10.72976|+3.83214|-3.57833|-3.83737|-10.72976/+3.83214|
|secondary|A2_minus_P_ONLY|shaft_max_deg|-7.31178|-22.34634|+6.87013|-7.59599|-7.31178|-22.34634/+6.87013|
|secondary|A2_minus_P_ONLY|progress_endpoint_error_deg|-0.92695|-25.15117|+14.82710|-3.75034|-0.92695|-25.15117/+14.82710|
|secondary|A2_minus_P_ONLY|joint_state_violation_mrad|+1.26653|+0.02480|+2.13514|+1.14215|+1.26653|+0.02480/+2.13514|
|secondary|A2_minus_P_ONLY|support512|-0.00510|-0.06644|+0.03297|-0.01286|-0.00510|-0.06644/+0.03297|

本次主预算中，A2相对P_ONLY有 3/3 对seed的最大轴误差降低，同时 3/3 对实际关节状态超限增大。这里是观测到的配对信号，不宣称support有无条件因果主效应；每条失败门槛仍如表保留。
P_ONLY / A2 / A1L 的不同预算结果不可拼成一组更高成功率。没有support-only(1,0)臂，无法识别完整2×2交互或无条件support效应。

## 预算、历史顺序与成本

主预算各臂每seed均2000iter、648192真实transitions、WM/failure-head各4000、actor1800、critic7200更新；这不等于等compute。次预算是2000上限实验中strictpre1200可获得的checkpoint：A1L2000(648192/WM4000/actor1800/critic7200)、P_ONLY1800(584192/3600/1600/6400)、A21700(552192/3400/1500/6000)。A1L提前完成，不称耗满1200秒。次预算训练量不同，仅作时间可用性比较，不能作为纯系数机制分离。

P_ONLY是在既有A1L/A2及seed0结果已知后冻结的探索性follow-up，不是原复制实验预注册的一臂。A1L/A2是不可变历史控制，P_ONLY三run全部发生在之后；三臂未同期随机排序，时间、系统/缓存历史与batch混杂仍在，不能把这些目标对照当作无偏因果估计。不同臂会产生不同on-policy训练数据，并非逐样本相同。

|方法|seed|训练记录秒|GPU slot秒(含CPU health)|peak allocated GiB|
|---|---:|---:|---:|---:|
|A1L|1|915.506|920.370|0.539|
|A1L|2|902.131|907.269|0.539|
|A1L|3|888.481|892.345|0.539|
|P_ONLY|1|1282.151|1287.199|2.195|
|P_ONLY|2|1281.178|1286.170|2.195|
|P_ONLY|3|1275.433|1280.351|2.195|
|A2|1|1348.920|1353.561|7.622|
|A2|2|1345.281|1349.611|7.622|
|A2|3|1345.947|1350.493|7.622|

15条唯一物理评估的独立预留成本合计 0.053365 GPUh；A1L次预算alias不重复计费。GPU slot不是利用率或训练计算时间；此前smoke、工程/编译失败、seed0及CPU报告费用未混入上表，不声称为研究总费用。P_ONLY保留完整A2前向用于最小共享P，但反向可能更省；没有填充计算来制造等成本。

## 输入、原始接触与审计边界

15条唯一rollout均绑定各自计划和completed manager状态，核对原训练输入、raw checkpoint训练seed、evaluation seed0、condition和checkpoint SHA；所有reference/model/任务协议、规范time/raw physics clock、t0 q/qd/object pose/twists/wrist/初始target一致。所有验收从原NPZ重算，与保存报告相符。P_ONLY三个final和三个1800各有独立52项健康审计；A1L/A2历史审计原样保留。

512分数逐条从真实solver-active手物接触对固定运动学意图独立计算，并从raw contact位置/法向/active重新验证support512、COM移位、单位法向及无mount混入。该支持函数使用单位摩擦锥几何，不是测量力幅值拟合，也不是整体任务成功标准。actual joint state与bounded command分别报告，冻结1mrad允许量不变，没有clamp记录的状态。

P_ONLY复用的P仅是采样手物几何penetration惩罚，不含显式actual-q joint-limit barrier。关节合规是本次不同学习轨迹的观测，不能据此证明P直接约束关节或已识别其力学/因果作用链。

A2/P_ONLY的几何先验没有通过既有contact predictor/dynamics-gradient资格；standalone compiled-precision失败仍保留，实际prior走eager，原生子网编译。旧CPU精确penalty tensor检查是计算合同验证，不是动力学有效性。只有一个clip007[200:351]固定腕重设计、同一物体/初态、每checkpoint一次物理回放；无真实无本体ego输入、未知原视频父ID，训练seed并非跨动作/视频泛化。50Hz采样门槛也不是连续时间或硬件安全认证。

保留证据：[P_ONLY独立审计](D:/workspace/supermimic-chord-penalty-only-study/outputs/penalty_only_seeds123_audit_v2/manifest.json)、[strictpre1200选择](D:/workspace/supermimic-chord-penalty-only-study/outputs/penalty_only_seeds123_audit_v2/secondary1200_manifest.json)、[compiled precision失败](D:/workspace/supermimic-chord-study/outputs/a2_cuda_pointmax_final_20261102/result.json)、[proxy qualification失败](D:/workspace/supermimic-chord-study/outputs/contact_geometry_cpu_exterior_v3/report.json)。

![主预算原始轨迹](tracking_primary.png)

![次预算原始轨迹](tracking_secondary.png)

![三臂配对图](paired_effects.png)

## 复现

```powershell
& 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe' 'D:\workspace\supermimic-chord-penalty-only-study\scripts\summarize_three_arm_followup.py'
```
