# 三个固定组件的 H8：校准降低累计误差，响应项仍有取舍

2026-11-06。三模型沿已执行 donor 控制的全部286个重叠窗口滚动8步，实际完成24次B286前向、6864个状态预测。独立保存复算24/24通过；这验证了本次保存算术与递归契约，不授予模型准确性、物理可行性或策略收益资格。

每条A0/A2 donor有143个起点k=0..142，每窗使用u[k..k+7]和观测[k+1..k+8]，不存在u150。运行先核298条完整58维相邻位链、300个四子步held控制及xi46，再构造窗口。每步同时更新独立x和dynamics278的前58维；不teacher-force状态，不投影/clamp输出。其余174维为零、xi固定，normalizer不更新。初始A2来自其独立artifact命名空间，两校准臂来自此前冻结的final1000。

## 全窗口第8步结果

控制间隔20ms，h8对应从各起点向前160ms。q/qd为逐分量RMSE；p/v/omega为向量RMS；R为原9矩阵元素的Frobenius RMS，不能解释为旋转角。六块单位分别保留，不合成混合单位总分。

| ALL，N=286 | 初始A2 | 点值校准 | 点值+响应校准 |
|---|---:|---:|---:|
| q / rad | 0.092374 | 0.043919 | 0.039444 |
| qd / rad·s^-1 | 1.991521 | 0.658683 | 0.649281 |
| p向量 / mm | 64.537 | 34.663 | 38.059 |
| R矩阵 / 1 | 0.953228 | 0.691851 | 0.779826 |
| v向量 / m·s^-1 | 0.544160 | 0.318382 | 0.309192 |
| omega向量 / rad·s^-1 | 10.635204 | 6.021769 | 6.718811 |

在ALL的h8上，两种校准相对初始模型的六块误差均降低；响应臂相对点值臂的q、qd、v较低，p、R、omega较高。没有单一总体赢家，不能凭这六个终点数值覆盖原确认条件失败。

路径差异也保留：h8位置向量RMS在A0为8.713/7.117/6.621mm，在A2为90.853/48.502/53.414mm（三数依次初始/点值/响应）。A0响应臂相对初始模型的qd（0.449998对0.424477rad/s）和v（0.102096对0.093955m/s）反增，因此ALL改善不是每条路径都改善。A0响应臂相对点值臂仅位置更低，另外五块更高；A2为q/qd/v更低、p/R/omega更高。各h并不都单调，例如A0的速度/角速度曲线会回落。全部3×8×3×6=432项统计、MAE、最大绝对分量误差和原数组均保留。

## 解释与下一步

结果支持“这次单步校准没有使ALL多步误差相对其初始模型全面退化”，但不足以支持将响应臂升级为Actor使用的赢家。它不确定遗忘或其他差异的唯一来源。两个校准模型仍在固定已记录控制下累计数十毫米的位置误差；这不能替代物理参考、真实闭环控制或任务成功证据。

286窗口相互重叠且来自同一已见参考缓存谱系，不能视为286条独立轨迹。固定teacher控制与Actor imagination不同；本次CPU FP32路径也不等同历史AMP训练。H1旧确认的152 restored状态与这里的286 donor窗口不是同一总体，禁止直接拼池比较。已知粗糙参考、原确认tradeoff、任务失败、warmup不同、Graph scratch未知、旧局部梯度范围和ego N=0继续保留。

后续回到同一native151受控输入的接触支持和物体动力学一致性。先查保存的全部物体接触力能否解释同一子步的线动量变化，并核重力、实际质量、自由关节坐标、质心偏移和积分器项。仅有手物接触力不足以解释落到地面的物体；不能把terrain支持写成手的支撑，也不能从机器人仿真恢复未知的人类接触力。该检查需另行冻结源码/相位/预算，不延长Actor训练来掩盖当前误差。

## 成本与证据

实际worker 9.5972163秒；3个可信component读取、3次native WM构造、24次前向/6864预测，0AD、optimizer、Actor、原119f加载、MJB、环境/物理或GPU。源审、合成测试、独审和绘图另计。独审验证718个源文件闭包、输入和输出pins、全部递归x/dynamics/u轨迹、mask、误差和432项汇总、原始RNG；没有重新运行模型，权重/normalizer不变仍保留源和摘要证据的范围。

- [实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_three_component_H8_donor_CPU_v1/report.json)，SHA `c676c80037a056f7f51d7877bcbdaeffa5575fafb360fae3c2148ba01087deba`
- [独立24项复核](D:/workspace/supermimic-full151-observation-study/outputs/full151105_three_component_H8_donor_saved_independent_v1/report.json)，SHA `ab43a6e5efb150eccccc792215f3f6c4e4d22bd795b5f2bae15e83d749dfc4b8`
- [ALL全部六单位曲线](D:/workspace/supermimic-full151-observation-study/outputs/full151105_three_component_H8_figures_v1/ALL_all6_units_H1_to_H8.png)
- [A0全部六单位曲线](D:/workspace/supermimic-full151-observation-study/outputs/full151105_three_component_H8_figures_v1/A0_all6_units_H1_to_H8.png)
- [A2全部六单位曲线](D:/workspace/supermimic-full151-observation-study/outputs/full151105_three_component_H8_figures_v1/A2_all6_units_H1_to_H8.png)
- [完整绘图指标CSV](D:/workspace/supermimic-full151-observation-study/outputs/full151105_three_component_H8_figures_v1/all432_plotted_metrics.csv)，三幅完整图已逐一目视检查。
- [冻结运行协议](D:/workspace/supermimic-full151-observation-study/protocols/full151105_three_component_H8_donor_CPU_run_v1.json)，SHA `d6b244e76f440da96401a353fa7e759e8f4789141ba216e2df0e9f6dec4ab436`
