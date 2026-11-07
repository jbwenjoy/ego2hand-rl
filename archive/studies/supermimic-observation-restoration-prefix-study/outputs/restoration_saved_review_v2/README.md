# R/F 已保存结果独立复核

唯一 canonical 是正常结束时按预先声明规则保留的 **R terminal / F0**，SHA `6364a6e2…61bab81a`。它与 `pipeline_return.final_x`、`last_accepted_x`、R 正常终态的 340 个控制变量逐位相同，没有用 QP 步向量或历史状态替代。30/30 保存数据一致性检查通过；本审阅没有新增几何查询、优化或物理。

R 用 3 次迭代、4 次函数调用把人工 slack 降至 0；随后独立 fresh61 与最终 fresh61 均保存了实际 61 次原生碰撞查询。最大生成重叠为 **2.998999207 mm**，严格低于 2.999 mm 规划上界，也低于原 3 mm 门槛。距规划上界约 **0.793 nm** 的差是计算值，不能作为碰撞模型精度或物理安全裕量。线性原始最大残差为 **+8.0556e−17 native units**，所以原始 exact 布尔仍是 false；它按本轮求解前已固定的每行 1e−12 算术容差通过，两项事实均保留。

相对冻结 seed，最大控制点变化为 **0.005951097 rad**（P17、joint index3），低于 R 的 ±0.05 rad trust box；原系数界限超量为零。q0、mapped object pose、骨方向目标及归一化观测目标逐位不变。原拟合 cost 从 **312.543411 增至 314.319662（+0.5683%）**；骨方向均值从 16.730734° 变为 16.732297°，指尖均值从 16.413427 mm 变为 16.414331 mm。R 的目标只有 slack，因此恢复可行性不等于改善拟合，F0 也没有拟合改善。

F 的单个 QP 在 25 次迭代、65 次函数调用后返回 status9。方向满足 `g·step = −8.80448 < 0`，最大步长 0.01670266 rad，步界限通过；但独立重算发现：

| 检查 | 原始残差 | 求解前固定容差 | 结论 |
|---|---:|---:|---|
| target_lower row363，interval3/joint3 | 7.97795e−11 rad | 1e−12 rad | 超过约 79.8 倍 |
| target_lower row403，interval5/joint3 | 7.17264e−11 rad | 1e−12 rad | 超过约 71.7 倍 |
| 线性化 HO row58，0.145 s | 1.22807e−8 mm | 1e−9 mm | 超过约 12.3 倍 |

因此它按原规则在提议门禁停止，**0 alpha、0 actual F trial、0 F accepted step**。这不是仅因为 status9，也没有测试该 QP 方向的真实几何效果。预测目标 306.891079 只属于未执行的固定线性化 QP，不能称作实际拟合改善。本轮没有调整容差或再运行 QP。

所有 6 个实际状态 archive 的数据和接触 SHA、phase trace、source plan、冻结源码、R/F 选择链及最终原始接触最大值均已重核。实际共享几何计数为 **424 = fit16 的 61 + constraint61 的 363**；其中 R terminal 和 final 各包含一次 61-query 刷新，不能再加 parent 的同一个共享计数。每次 state creation 的实际查询增量保存在 [report.json](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_saved_review_v2/report.json)，相同 x 缓存复用没有被计成新测量。

这里获得的是固定映射、未知接触意图假设下，61 个名义采样的生成重叠可行性。未生成接触的距离保持 NaN/UNKNOWN；结果不证明连续间隙、动态抓持、受力正确、真实 ego 数据有效或完整 151 帧运动资格。三 bundle 与最终 FP32 控制导出另行独立复核，不能由本报告的几何通过推断。

[phase_trace.csv](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_saved_review_v2/phase_trace.csv) · [QP 拒绝精确值](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_saved_review_v2/QP_rejection_exact.json) · [可复现只读脚本](D:/workspace/supermimic-observation-restoration-prefix-study/scripts/audit_restoration_saved_cpu.py)

![R/F 保存结果审阅](D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_saved_review_v2/RF_saved_audit.png)

v1 的图检发现图例覆盖末端 cost 标记，已保留该版本和布局问题记录；v2 仅修正图例与注释位置，所有数据、判断和容差不变。
