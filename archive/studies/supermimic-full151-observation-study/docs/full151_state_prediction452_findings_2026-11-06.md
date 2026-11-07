# 固定 A2 世界模型的全部 452 个单步转移误差

本轮完成一次严格 A2 final2000 模型加载、一次 batch452 前向，独立 saved-only 审计 **25/25 通过**。
输入、实际绝对控制、对应 GPU post20 目标和统计全部逐行核验。完整 452 行与失败阶段均保留。
这是同一个模型预测两条不同动作来源路径，不能把 A0/A2 路径误差差异当作两个方法的对照。
两条 donor 为 fresh saved-action-source 重放；不将它们描述为旧闭环评估档案的轨迹复现。

## 全量结果与单位

全部值为 `predicted_FP32.astype(F64) - observed_post20_FP32.astype(F64)` 的描述性统计。
没有把不同量纲合成一个准确率，也没有设置通过/失败阈值。

| 分块 | 单位 | 全 452 行的分量 RMSE | 另列的向量/Frobenius RMS |
|---|---|---:|---:|
| q20 | mrad | 21.244937 | — |
| qd20 | rad/s | 0.922717 | — |
| 物体 p3 | mm | 4.734269 | 8.199995 |
| 物体 R9 | 无量纲 | 0.042359924 | 0.127079771 |
| 物体 v3 | m/s | 0.118979207 | 0.206078032 |
| 物体 omega3 | rad/s | 2.894185990 | 5.012877182 |

分量 RMSE 对行和该块所有分量共同求平方均值；向量 RMS 先对每行分量平方求和，再对行求均值。
R 只比较保存的九个矩阵元素，不投影、不换成角度。完整每分量的 RMSE/MAE/max 与带符号差在报告/NPZ。

| 路径/模式 | 行数 | q 分量 RMSE mrad | p 向量 RMS mm | R Frobenius RMS | v 向量 RMS m/s | omega 向量 RMS rad/s |
|---|---:|---:|---:|---:|---:|---:|
| A0 donor | 150 | 15.714 | 2.094 | 0.11020 | 0.10005 | 5.01939 |
| A0 baseline | 12 | 18.795 | 1.925 | 0.07473 | 0.10484 | 3.35592 |
| A0 perturbation | 64 | 18.794 | 1.925 | 0.07471 | 0.10484 | 3.35094 |
| A2 donor | 150 | 23.654 | 12.556 | 0.14666 | 0.28886 | 4.64884 |
| A2 baseline | 12 | 27.194 | 8.740 | 0.15558 | 0.23900 | 6.80473 |
| A2 perturbation | 64 | 27.194 | 8.741 | 0.15497 | 0.23906 | 6.76382 |

这些平均值按保存行加权；恢复分支和同一 donor 内时刻相依，不提供独立 N、置信区间或泛化成功率。

## 完整相位覆盖与峰值

全六块均按预先固定的每行 component RMS 找最大值，保留全部精确并列；本批各有一个主峰。
此规则与最大单分量、向量 norm 的峰值分别保存，不能混称。

| 块 | group_id | 来源 | 相位 k | 每行 component RMS 主峰 |
|---|---:|---|---:|---:|
| q | 227 | A2 donor | 1 | 53.053922 mrad |
| qd | 379 | A2 perturbation | 0 | 2.116039 rad/s |
| p | 285 | A2 donor | 59 | 34.651959 mm |
| R | 3 | A0 donor | 3 | 0.275137 无量纲 |
| v | 286 | A2 donor | 60 | 1.237080 m/s |
| omega | 3 | A0 donor | 3 | 20.743487 rad/s |

位置峰的 34.65 mm 是三分量 RMS，不是位置向量 norm。相位 1/3/59/60 等峰值也不全在旧局部
物理差分选取的 k0/49/99/149，因此新的全曲线增加了覆盖，不能反过来用旧八个 snapshot 解释全部峰。
机械按每区间四个 PRE 记录汇总 HO depth 后，共 14 个区间含 44 个 HO>3mm 子步；六个主峰均未超过
这个 HO 阈值。它不排除较浅接触、非手物接触、环境或其他机制，本轮没有做全类别的峰值因果分析。

## 本轮执行合同

每组首 PRE state58 来自已独立确认的 GPU/B1 packet；目标为同组最后 post20 native GPU state。
四个实际绝对控制逐位相同，所有 group/path/k/branch 与端点身份一致；保留 k149→观测150，不生成 u150。
物理 xi46 与原训练 actual8/bundle 一致，zero174 沿用已有 wrapper；不重做 EMA/C2，不补 privileged368。

使用原 strict105 CPU wrapper，amp=False、四线程、参数/normalizer 冻结。CPU FP32 前向不被描述为
历史训练 GPU AMP/TF32 的精确重放。self15 仅保存原始有限 logits，不评作 HO 或力标签。
没有 goal 重建、task score、objective、AD、状态转换、Actor 前向、新 Env/物理/GPU 或训练。

worker 用时 9.170 秒。实际插桩为 loader1、checkpoint 读取2、WM 前向1；三次被动 MJB 读取是成功
路径源码推导（caller bundle1 + strict 内部2），没有冒充原生 API 全局计数。源归档按实际文件身份与
SHA 去重为 698 个文件，语义输入 pin 保持独立。原始 Python/NumPy/TorchCPU RNG 字节前后一致并经独审；
权重/normalizer 的不变性仍由执行源的前后 digest 断言支持，未额外重新运行模型。

## 研究含义与下一步

本结果扩大了单步状态误差的相位覆盖，显示当前误差不能仅用少数局部状态的数值概括。
它没有改善或撤销此前 5/9 物理梯度方向同号的结果：单步状态误差与动作响应/梯度是不同问题。
已知参考几何失败、prepolicy100 各 arm WM/normalizer 已不同、graph scratch/完整隐藏状态未知的限制均保留。

下一步优先利用现有保存数据，在旧八个 snapshot 的三条 baseline 和全部 128 条扰动之间分解
“baseline 预测偏差”与“有限动作响应误差”，另行冻结配对、实际裁剪/非对称控制、零分母和精度规则。
该诊断不再加载或前向模型，不选旧 stable9 子集，也不把有限 secant 称为真实物理 Jacobian。
它只覆盖已有八个 snapshot，不能补出上述未采集的峰值扰动数据。

H8 固定动作序列预测的输入连续性另已完成可行性检查，当前 parked，没有 H8 前向或训练。
任务收益、接触动力学与真实 ego 泛化仍未取得资格，独立真实 ego 数量仍为 0。

## 证据入口

- [实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/report.json)，SHA `202b4e15512491b15fd8fc3fa3efb0f1c9da85a17ce359b372ee619ea535175a`
- [独立 25 项审计](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_saved_independent_v1/report.json)，SHA `b2675cbaaa46c5575d5f9084a64c2bf7b28c7eabe13715b600347e5485cc4d20`
- [全 452 行 CSV](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/all452_errors.csv)与[原始输入/预测/误差](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_CPU_v1/prediction_arrays.npz)
- [全部六块误差曲线](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_figures_v1/all452_state_prediction_error_curves.png)，Root 已整体视检。
- [作者完整峰值与解释](D:/workspace/supermimic-full151-observation-study/outputs/full151105_state_prediction452_author_interpretation_v1/interpretation.md)，另有 12 项描述性核对，不冒充独立审计。
- [被动模型读取成本更正](D:/workspace/supermimic-full151-observation-study/docs/full151_passive_model_read_cost_addendum_2026-11-06.md)
