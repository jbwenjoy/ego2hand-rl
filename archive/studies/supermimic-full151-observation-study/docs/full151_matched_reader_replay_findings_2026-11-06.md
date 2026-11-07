# 原生 CUDA、批量大小 1 的状态重放

记录日期：2026-11-06（Asia/Shanghai；实际运行 UTC 日期为 2026-11-05）。

本轮完成 2260 次原生 `SimReader.state` 的 CUDA FP32、B1 调用。
全部 452 个 post20 终点与 160 个首状态锚点的 58 维结果，均与对应原 GPU 记录数值相同。
手关节位置/速度仍要求 FP32 位一致；其余 18 维仍使用原来的逐分量
`8 * eps32 * max(1, abs(actual), abs(reference))` 规则。
没有放宽门槛、删减状态分量、搜索精度条件或重新运行物理。

独立 saved-only 审计 **29/29 通过**，复算了全部原门槛，核对原始输入、上传回读、
逐次 B1 调用证据、RNG、后端标志、源归档和终态进程身份。它没有调用 Torch、原生 Reader、
几何工厂或物理。继承的 1808 个 solver PRE 位姿关系也通过原规则；这是已保存位姿对的核验，
并非 GPU 输出 1808 行 p/R 与 solver 位姿的新直接比较。

## 已取得的资格与限制

- 可将特定的 1808 行 GPU PRE 状态，交给另行固定的 CPU 几何计分实验。
- 本轮几何工厂、计分、WM、Actor、梯度、优化器、Env 与物理调用均为 0。
- `dense_scoring_eligibility` 和 `proxy_scoring_authorized` 在重放产物中仍为 false。
- 尚未取得几何正确性、接触动力学、真实人手接触、物理梯度或任务成功资格。
- 只有单一来源、单种子、两条受控动作路径与相依恢复分支；真实 ego 数据数量仍为 0。

原 CPU 批量转换的失败及未计分记录完整保留。其后新 CPU forensic 记录的 452 个终点中，
5 个角速度分量超过原门槛；这份 forensic 不是原失败进程未保存中间值的恢复。
新 CUDA/B1 对照同时改变后端与调用形状，不能据此唯一归因 GPU、批量大小、TF32 或特定归约。
原生 mm/mv 使用逐元素乘法再求和，也不能把差异直接解释为 GEMM/TF32。

全部 2260 行的新 GPU 与 forensic CPU 结果之间，q、qd、p、线速度完全相同；
旋转矩阵元素最大绝对差为 `4.76837158203125e-7`，角速度为
`5.9604644775390625e-6 rad/s`。这些数值仅描述两次算术实现的差异，未作为新合格门槛。

## 工程记录

唯一实际运行使用配置 SHA `9573b5ad074904775c07c54f49ffcd43bb354a178fc386a0537827df95210078`。
worker 用时 26.395 秒，正常 exit0；实际 PID 24292、launcher PID 34260，OS 身份检查 12/12。
Root exec session 40231 已关闭，无重放重试或后台物理作业。

原始 MJB 地址索引为整数，转为原生 CUDA 要求的 int64 时保持值、范围和互斥关系。
四组 FP32 输入要求原始字节一致。独立审计 v1 错把索引转换也要求为同 dtype 原始字节一致，
因此产生 27/28 的 harness 误报；该报告保留。最终 v3 的显式 typed/range/unique/disjoint 检查修正了
这一审计规则，未修改数据、浮点算术或原数值门槛。

`output_pins_core.json` 内的 `runtime.json` 是 core 返回前的中间状态 pin，entry 终态更新后确实过期。
审计明确保留两份 hash，单独以终态 runtime、原始 OS 记录、命令、创建时间和源/配置身份校验。
所有不可变 core 产物仍严格检查 hash；不能把本结果写成“全部旧清单 pin 与终态一致”。

## 下一步

新建 CPU consumer，直接消费已独立确认的 GPU packet，不再执行 CPU 状态转换。
计分前新增全部 1808 行的直接 p/R 对齐检查：以记录的 palm 和 forensic solver world p/R，
用 NumPy F64 显式乘加形成派生 palm 参考，再按同一 8eps32 规则检查；保存参考、差、界与 mask。
它是新增的保守对齐门，不被描述为原 GPU FP32 真值；失败则工厂和计分调用为 0。

通过后才执行原定的一次 P_ONLY 工厂、15 批（128×14+16）的 1808 行分析，保持原采样、
strict 3 mm 原生 HO 标签、全部精确并列最深 pair 主覆盖与全部超阈值 pair 副覆盖。
显式静态 MJB 分类读取与工厂内部的被动 MJB 读取分别记账。
这将检验稀疏/时相覆盖与同位姿代理漏检；不重新评定旧三臂任务结论。

## 证据入口

- [重放配置](D:/workspace/supermimic-full151-observation-study/configs/full151105-matched-native-GPU-reader-v2.json)
- [实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_matched_native_GPU_reader_v2/report.json)，SHA `0ac1bf85d41c3e6136f820786fab8dd60ff23470bfb8e7182eeb9b9e4bbee8f9`
- [独立 29 项审计](D:/workspace/supermimic-full151-observation-study/outputs/full151105_matched_native_GPU_reader_saved_independent_v3/report.json)，SHA `a31cb81a6e513fea013591a9198b86a0ad398a148164f18a0bedfcdb7ec9c3eb`
- [452/160 数值补表](D:/workspace/supermimic-full151-observation-study/outputs/full151105_matched_native_GPU_reader_saved_independent_v3/difference_summary.json)，SHA `64dac8681f5a137e111858d8f810c3207857fbc06d22ff87329162d6f1dff2cd`
- [此前 CPU forensic 与失败解释](D:/workspace/supermimic-full151-observation-study/docs/full151_converter_forensic_findings_2026-11-05.md)
