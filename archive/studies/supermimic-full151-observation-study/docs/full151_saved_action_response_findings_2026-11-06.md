# 保存数据中的 baseline 偏差与有限动作响应

分解已完成，独立 saved-only 审计 **22/22 通过**。全部 8 个 snapshot、152 个恢复分支、128 个扰动
端点、64 对 ±epsilon 均保留；64 对的实际控制分母均非零。没有重新加载/前向模型、AD、MJB、
原生几何查询、物理或 GPU。唯一实际 NumPy run 用时 0.313 秒。

## 比较对象与守门

每个 snapshot 的 19 条记录均对齐原始 `snapshot_x`：q/qd 要求 FP32 位一致，其他 18 维要求精确数值
一致，并保存原始位掩码。三条 baseline 的 u20 与原 u0 位一致；扰动除声明的一个坐标外，其他
19 个控制分量保持原位值。全部 14 类守门通过。它固定的是可见 state58，未取得隐藏缓存状态资格。

以三条 baseline 的预测、物理端点和控制分别形成 F64 坐标均值。这些是统计参考，不作为新物理
状态或动作，不投影 R、不重做 EMA/C2，也不送入模型。记 baseline 误差为 B，扰动预测/物理端点
相对各自 baseline 的变化为 M/P，则响应残差为 M−P。完整 F64 关联舍入残差保留，没有据此设置
物理准确性门槛。本批两个写法的最大关联舍入残差为 0。

## 结果

下表全部使用各块的分量 RMS。baseline 偏差统计覆盖 8 个中心，响应统计覆盖全部 128 个小扰动
端点；两者的幅值比较依赖本次选定的 1/2 mrad 控制变化，不能当成总体误差贡献率。

| 块/单位 | baseline 偏差 | 物理有限响应 | WM 有限响应 | 响应残差 |
|---|---:|---:|---:|---:|
| q / mrad | 23.374686 | 0.099091 | 0.093328 | 0.090108 |
| qd / rad/s | 1.133740 | 0.005409 | 0.004074 | 0.004435 |
| p / mm | 3.653596 | 0.009149 | 0.004021 | 0.009172 |
| R9 / 无量纲 | 0.040680918 | 0.000338065 | 0.000104332 | 0.000343622 |
| v / m/s | 0.106544652 | 0.000545130 | 0.000204875 | 0.000533207 |
| omega / rad/s | 3.097486460 | 0.027924881 | 0.005747394 | 0.028138296 |

baseline 偏差与有限动作响应失配同时存在。例如，p 响应残差约 9.17 µm，与物理有限响应约
9.15 µm 同量级；绝对值小不等于动作敏感度已经准确。固定的同 snapshot 坐标常量在两个端点之差
中抵消，不改变本实验保存的有限响应。模型重新训练或结构改变的效果尚未实验。

三条 baseline 的物理及 WM 可见端点，在本批所有 feature 上 range 均为 0；这只是三次重复的
编码描述，不是噪声上界。全部 64×58 对称/非对称端点差中，物理的 1642 个分量编码为数值零，
WM 的 1 个分量编码为数值零；保留数值零与位相同两种标志，不把它们解释为真实物理导数恰为零。

## 实际控制步长与裁剪

所有 secant 使用实际 `(u_plus[j]−u_minus[j])`，范围为
`0.001000046730041504–0.0040000081062316895 rad`。12 对带原始 clipped 标志且相对 baseline
非对称；全部仍保留。它们是有限弦线，不冒充 baseline 处中心导数。

原 `direction__d_eff=(u_plus−u_minus)/(2*requested_epsilon)` 保持无量纲含义；
`actual_delta_u_rad`、`actual_halfstep_u_rad` 与 `center_offset_rad` 另存。clip 按原 writer 的 FP32
名义端点数值比较验证，正负零等价，不凭 `delta != 2*epsilon` 判断。首次准备 v1/40 项使用了位比较，
已保留；唯一实际分解使用修正后的 v2/42 项源码，不存在实际 v1 数值 run。

128 个端点中有 6 个完整 u20 与 baseline 逐位相同，其预测/物理端点也对三条 baseline 逐位相同。
它们与“12 对 clipped”是不同计数单位。A2 k49/99/149 的全部 16 个扰动端点在物体 p/R 上编码为
baseline 值；这 48 个端点中有 42 个实际控制改变，WM 仍预测了非零物体响应。它是后续核验接触
regime 的候选现象，没有据此宣布无 HO、落地、真实零 Jacobian 或唯一误差原因。

## 资格与后续

没有按旧 stable9、符号、一致性或噪声门槛选样。逐 feature58、六个单位块、全部 64 pair 与
零分母策略均独立复算一致；本批 undefined=0，策略仍保留 NaN/mask，不以填零修复。
原来的 5/9 物理梯度方向结果、参考几何失败、prepolicy100 的 WM/normalizer 分歧与 graph scratch
未知均保留。预测来自固定 A2 FP32 CPU shadow，不能作为历史 GPU AMP 策略梯度重放。

下一步先对全部 152×4=608 条保存的分支 solver PRE 记录机械核对接触类别与已有力证据，
区分 generated HO/self/object-with-nonhand/other，以及有效约束上的原 decoded Fn 描述。
不从无候选推全局间隙、不把零力推为无接触，也不将 self15 传感器标签等同 generated self pair。
现有字段足够做这项保存数据描述，完整原始 friction-pyramid 重解码的资格并不由此取得。
当前未执行这项新汇总、H8 或训练；目标保持 active，真实 ego 数量仍为 0。

## 证据入口

- [实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_CPU_v1/report.json)，SHA `b7e951f25dd0d45228fd26c80a2b5eac9c56b4cb9248a52cad6209a2cf154689`
- [独立 22 项审计](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_saved_independent_v1/report.json)，SHA `b70a9b4e63b13401a86a282046f19a71fb2c5019c185a851e91a1ce6b86de481`
- [全部 baseline/响应/64 对数组](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_CPU_v1/decomposition_arrays.npz)与[全部 58 分量统计](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_CPU_v1/per_feature58_statistics.json)
- [偏差/响应图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_figures_v1/baseline_bias_and_all128_responses.png)与[全部 3712 个 secant 分量图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_figures_v1/all64_secants_all3712_components.png)，Root 已视检，图的原始长表另存。
- [作者全量描述与解释](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_action_response_author_interpretation_v1/interpretation.md)，通过 11 项描述性核对，属于作者复读，不作为独立审计。
