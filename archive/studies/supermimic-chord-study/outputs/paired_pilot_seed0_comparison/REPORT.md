# Seed0 A1L / A2 配对 pilot 对比

**本次单 seed 结果不支持 A2 改善冻结物理任务验收：A1L 通过；A2 在等更新预算和等时可用 checkpoint 两种口径下均失败。**

## 等采样、等更新主预算

| 方法 | iter | task | 位置均值 / 最大 mm | shaft 最大 ° | progress 终点误差 ° | support512 ↑ |
|---|---:|---|---:|---:|---:|---:|
| A1L 2000 | 2000 | PASS | 5.58 / 13.02 | 10.124 | 3.896 | 0.822422 |
| A2 2000 | 2000 | FAIL | 6.76 / 10.06 | 24.295 | 21.432 | 0.828956 |

两臂均为 2000 iter、648192 个真实 transition、WM/failure-head 各 4000、actor 1800、critic 7200 次更新。记录的进程总耗时分别为 **923.15s** 和 **1355.16s**（A2 为 A1L 的 1.468 倍）；这项比较不是等计算成本。

A2 的位置最大误差较低，但 shaft 最大误差 24.295° 超过冻结 20°，progress 终点误差 21.432° 超过冻结 20°。三次 rollout 的无掉落、位置、穿透、执行目标与实际关节限位、负载及日志合同均通过；主预算 A2 的失败项如实保留。

## 1200 秒内可获得 checkpoint 的次预算

| 方法 | iter | task | 位置均值 / 最大 mm | shaft 最大 ° | progress 终点误差 ° | support512 ↑ |
|---|---:|---|---:|---:|---:|---:|
| A1L 2000 | 2000 | PASS | 5.58 / 13.02 | 10.124 | 3.896 | 0.822422 |
| A2 1700 (<1200s) | 1700 | FAIL | 7.33 / 11.38 | 21.665 | 2.305 | 0.818899 |

该次预算来自各自 **2000 iter 上限实验** 中截止前已完整保存的 checkpoint，未延长 A1L 训练来耗满 1200 秒。A1L 在约 923 秒已达 2000 iter，沿用同一 final；A2 选择 model_1700。选择仅使用完成时间和完整性，不使用评估质量。

A2 model_1700 在 OS 创建后 **1148.633531s** 完成，具备 552192 transitions、WM/failure-head 各 3400、actor 1500、critic 6000 次更新；1800 checkpoint 已超时。该模型 progress 终点误差 2.305° 通过，但 shaft 最大误差 21.665° 仍超过 20°，因此 task 仍失败。

## 接触诊断与适用范围

主预算 A1L / A2 的辅助 support512 得分分别为 0.8224216 / 0.8289563，差值 +0.0065347。分数使用真实 Warp solver-active 手物接触几何、物体 COM、同一固定 512 方向 basis，与独立冻结的运动学接触意图比较。这不是接触力真值匹配，也不是新的任务成功条件；分数略升同时出现任务失败，不构成物理接触收益证据。

A2 是 imagination-only 的 soft geometric prior / anticipation shaping，目标独立冻结。训练采用 64 个固定 stride8 方向、每碰撞 geom 64 点共 1344 点；本报告接触评估采用完整 512 方向。先验实际执行为 eager，原生 actor/WM 子网编译；此前 standalone Inductor compiled-precision 门槛失败的记录仍保留，不能把训练完成改写为该精度门槛通过。旧局部 contact predictor / dynamics-gradient 资格失败同样保留。

这是一条 upstream_kinematic clip007[200:351] 的固定腕任务重设计，原始视频父 ID 未知，不是新 ego 视频、独立新动作或跨物体泛化。仅 seed0，没有置信区间或总体有效性结论。A2 同时增加 support 吸引和 penetration prior；即使未来有收益，也需 penalty-only 消融才能区分来源。

## 输入和审计

三次评估 reference NPZ/JSON、model、任务协议、评估器 SHA 完全相同；151 帧规范时间、实际初态 q/qd、物体姿态与线/角速度、腕、初始执行目标均逐值相同。本脚本从原 NPZ 重新计算验收指标并与已保存 evaluation.json 核对；完整性和哈希核查保存在 comparison.json。

成本仅统计对应完整正式训练进程，不含此前中断、工程 smoke、失败编译尝试及独立评估。记录的 GPU peak allocated：A1L 0.539 GiB，A2 7.622 GiB。该值不是峰值 reserved 或全卡显存占用。

![原始跟踪曲线](tracking.png)

## 复现

```powershell
& 'D:\workspace\dex-supermimic-research\.venv\Scripts\python.exe' scripts/summarize_paired_pilot.py
```

产物：comparison.json、三个独立 contact_score_*.json、tracking.png 与 tracking.svg。图中不做平滑，仅接触诊断对 unknown/no-contact 阶段不画吸引分数；所有原始逐帧分数和 phase 权重保留于 JSON。
