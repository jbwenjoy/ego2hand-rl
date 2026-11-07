# 2026-11-07：固定 offline1000 的研究结果

**训练已完成；本次四个原生 Actor 案例均失持，尚无2秒持握或策略收益资格。** WM 对新三指分布的端点拟合改善，但旧分布有退化；有限动作响应仍暴露物理误差。以下保留全部案例，未按表现选 checkpoint、动作轴或起点。

范围是同一固定腕 Hand2、原模型/材料/增益、名义手机质量0.2kg等假设下的局部已知谱系。MON/CHECK/fresh 都不等于独立 ego 场景泛化。本文仅读取保存的 JSON 与两份320行响应 NPZ；未导入 Torch、重跑模型、原生积分、训练或性能优化。

## 训练与进程终态

固定 checkpoint 为 paired1000 / shared-WM1100（共同WM适配100 + 配对更新1000），序列化健康15/15；两臂各1000次 Actor/Critic 更新。训练过程 native calls=0。GPU恢复进程28104 exit0，终止于2026-11-07 12:50:06 UTC；Actor评价与诊断监督器也已exit0。12:54:06 UTC 的 OS 查询中，28104、21664、48212、18456及相关训练/等待Python命令均不存在，相关存活进程数为0。此为当时快照，不是常驻监控。

## 四个 Actor 原生案例

|策略|dt (ms)|实际步数|早停时刻 (s)|末端COM漂移 (mm)|末端姿态变化 (°)|2秒物理门|
|---|---:|---:|---:|---:|---:|---|
|A1 tracking|0.31250|388|0.12125000|50.2449|41.1549|FAIL|
|A1 tracking|0.15625|775|0.12109375|50.1258|40.9481|FAIL|
|A2 engineered proxy|0.31250|388|0.12125000|50.2156|41.1102|FAIL|
|A2 engineered proxy|0.15625|775|0.12109375|50.0655|40.9122|FAIL|

四例全部 `ABORT_LOSS_OF_HOLD`，合计2326步、28次 Actor forward；没有 WM/Critic forward、辅助外力或新优化更新。穿透、非指定接触、桌面、关节限位和警告门通过（最大穿透0.164–0.188mm，非指定/table力为0，警告为0）；持续时长、末端5mm/5°及尾段门未通过。尾速度与三指尾覆盖率均为 **null**，不是0。约0.121秒是失持退出时刻，不能称为稳定持握时长。

每例只有6个可用POST控制参考帧，其后94帧缺失。A1两dt的参考COM RMS为26.952/26.958mm、姿态RMS约23.54/23.48°；A2为26.934/26.929mm、约23.51/23.44°。同6帧的固定物体**纯指令先验**为2.605mm/3.366°，它不是另一条原生持握轨迹。没有缺失尾段填充或2秒tracking成功判断。

双dt的早停前缀端点差：A1为0.12568mm/0.34654°，A2为0.18038mm/0.22697°。虽然分别小于2mm/2°，两例都未完成2秒，因此两策略的 `dt_pair_consistency_pass=False`，不能认证2秒一致性。

[四例完整报告及每例trace链接](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_policy_evaluation_v1/learned_fixed1000_run2/report.json>)；[双dt结果](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_policy_evaluation_v1/learned_fixed1000_run2/native_dt_pair_consistency.json>)。

## WM 新旧分布：固定 final 与 warm5000

下表是全部完整20ms端点，COM为向量RMS、姿态为SO(3)测地角RMS。保留旧CHECK的4个未知与fresh的12个partial掩码，不将它们补成完整目标。

|分布|完整行|warm5000 COM (mm)|final COM (mm)|warm姿态 (°)|final姿态 (°)|
|---|---:|---:|---:|---:|---:|
|MON|52|4.861647|0.331681|17.941712|1.243925|
|CHECK|1472|0.125835|0.151828|0.387725|0.553361|
|fresh|228|0.699749|0.549968|1.534079|1.460302|

新MON端点明显改善；旧CHECK端点退化；fresh的COM/姿态改善，但原两指覆盖 Brier 从0.006302升至0.007627。fresh没有J22/meanFn22真值。不能据这些混合结果声称整体泛化改善或策略可持握。

旧CHECK全部1436条有限动作响应：六个物理头块的分量误差RMS如下（量纲各自独立，非混合总分）。

|头块|单位|warm5000|final|
|---|---|---:|---:|
|joint q|rad|0.000478186|0.000492825|
|joint qd|rad/s|0.0355382|0.0360591|
|COM ballistic residual|m|1.72326e-05|1.62072e-05|
|source world-left rotation|rad|0.000602084|0.000606322|
|pair force impulse|N s|9.69855e-05|9.7962e-05|
|pair moment impulse|N m s|5.81865e-06|5.86383e-06|

旧CHECK中836条真实COM精确零响应，模型仍产生非零“phantom”COM响应：向量RMS warm=17.622µm，final=16.739µm。旧响应指标有改善也有退化；不把它们合成无单位赢家。

[分布诊断报告](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_final_WM_diagnostics_v1/completed_v1/report.json>)；[全部端点/cohort统计](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_final_WM_diagnostics_v1/completed_v1/endpoint_statistics.json>)；[全部旧响应/cohort统计](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_final_WM_diagnostics_v1/completed_v1/response_statistics.json>)。

## 新三指：全部320条真实动作响应

8个MON完整integration状态，各20轴±约0.01rad；328次原生分支全部完整，8条 baseline 全部逐bit复现且先于任何侧分支。所有320侧保留：执行器合法320，当前 Actor-EMA 可达290、范围外30；范围描述不作筛选。真实COM精确零86条，末端旋转精确零80条。轴16–19仅有约4e−16到3e−14m的COM向量RMS，不能把任意F64非零视为强信号。

同实际动作端点的**分量**响应误差RMS（旋转为 Log(R_side R_baseline^T)，不是 source-relative Log 的简单差）：

|块|单位|warm5000误差|final误差|零响应先验误差|final/先验|
|---|---|---:|---:|---:|---:|
|q|rad|0.000863167|0.000762474|0.000711276|1.0720|
|COM|µm|26.3497|21.6017|21.4117|1.0089|
|端点world旋转|mrad|1.12608|0.834592|0.659967|1.2646|
|vCOM|m/s|0.00231699|0.00222383|0.00230053|0.9667|
|world omega|rad/s|0.0805312|0.0750932|0.0768843|0.9767|
|J22 force|N s|0.000154043|0.000147301|0.00015219|0.9679|
|J22 moment|N m s|9.63918e-06|9.16359e-06|9.52733e-06|0.9618|

final相对warm的平均响应误差下降，但COM和旋转仍未优于该批“预测零响应”先验。86条真实COM精确零响应上的phantom向量RMS：warm=23.861µm、final=17.037µm（中位数19.026/14.636µm）。真实响应为零时，不定义相对误差为0。没有将边际coverage相乘伪造三指同时覆盖预测。

模型内部 AD 与 ±1e−4rad FD 的160个context-axis对全部在界内，无FP32 encoded-du坍缩。各单位块相对L2误差范围：warm约0.166–0.355%，final约0.147–0.251%；COM为0.2481/0.2270%，局部旋转为0.3543/0.2513%。这是**同模型的数值自洽性**，不证明对真实原生动力学的无限小梯度准确；上面的有限动作误差和phantom响应仍然存在。共2次checkpoint读取、2个WM、20次forward、16次Jacobian API（274输出的批VJP，4384个等价反向方向），0参数更新/原生调用。

[新动作响应评价报告](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/completed_v1/report.json>)；[全部20轴有限响应统计](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/completed_v1/finite_response_statistics.json>)；[final模型AD/FD自检](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/completed_v1/offline1000_WM1100_model_only_AD_FD_statistics.json>)；[原生局部响应与离散动量描述](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_v1/saved_response_statistics_v1/report.json>)。

## 原件 SHA256

链接均指原始 artifacts；本文件未改写它们。以下摘要直接对当前文件字节计算，checkpoint只核文件hash，未重新反序列化模型。

- [最终训练结果](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/gpu_resume1/result.json>) — `da9445f4dc3011eb307ca9fb1dd2198681b6c94db6b9b63e820b8c62971a67a6` (1090 B)
- [最终健康15/15](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/gpu_resume1/health_1000.json>) — `749e65e57c3954c2d3b206e067fcf0f9e766866ac12bccd5d37fa3af6aa5ce90` (2579 B)
- [训练终态](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/process_resume1/terminal.json>) — `dea1df8db694de1be02a3b037b64c05be2f1182fc3f5d741af7cbffc5523303d` (2331 B)
- [Actor报告](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_policy_evaluation_v1/learned_fixed1000_run2/report.json>) — `8406b22b3cfe4fe2f00056bc9a36773c6bb1bb06aeaec4cfc770ddf3302b1a83` (155535 B)
- [Actor终态](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_policy_evaluation_v1/learned_process_v2_run1/terminal.json>) — `8a416afe3c92f6ab659f5a63992516d6f6d20017cb317d00022d5d8aa0dfb975` (2209 B)
- [分布报告](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_final_WM_diagnostics_v1/completed_v1/report.json>) — `05c056a6c5527e8b9057e3cc00239912069c8a112c9a4720b1d686c4d2492393` (7258 B)
- [新响应报告](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/completed_v1/report.json>) — `481e3ffbbf949f09141c38d13ae3358305dfcb3ae0ec98db46d417f88fd95ca4` (2943 B)
- [两诊断监督器终态](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_posttraining_v1/process_run1/terminal.json>) — `821496113c72a99f76991d419e438ada9d9d76e83374d1093a8f3c519eac594d` (3962 B)
- [warm AD/FD](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/completed_v1/warm5000_model_only_AD_FD_statistics.json>) — `2265d323370fe454bfa531f91e8e10a053e2a775e523bd95e537455ace9b67e6` (8100 B)
- [final AD/FD](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/completed_v1/offline1000_WM1100_model_only_AD_FD_statistics.json>) — `1743be10124128229a19414cfd466099f7a0621bc11be6a867c4f5d5a9c2796b` (8111 B)
- [最终checkpoint](<D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/gpu_resume1/paired_checkpoint_1000.pt>) — `ab82105330ed381421cc2ceacde856d209529f8df3d4539696743dc5ff5d3383` (39211209 B)
