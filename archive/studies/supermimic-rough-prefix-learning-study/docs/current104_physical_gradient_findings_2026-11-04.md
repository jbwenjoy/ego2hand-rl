# 当前104的一步物理方向对照

2026-11-04：新的同实例原生物理探针已完整完成，随后三个冻结世界模型在独立CPU进程上消费同一批物理响应。结果支持在此短窗口继续研究世界模型优化，但没有建立完整Jacobian、接触力预测或CHORD整体收益结论。

## 实际执行与门禁

四条新donor分别重放A0/A1L/P_ONLY/A2的固定保存动作，使用同一来源、seed0、固定腕和相同模型实际八字段。每条取phase0/4/8/12，得到16个快照；每个快照先执行3次baseline，再执行8方向×2步长×正负扰动，共560个分支。原生每区间四个5ms子步和一个Manager末forward保留，机器人绝对目标是唯一分支干预，物体无外力或状态辅助。

[物理报告](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_FD_v2/physical_report.json)记录16/16、560/560及2480个scheduled积分子步全部完成，exit0/no timeout。新执行配置SHA为`02753b7defccc30a5b8bf9a1ee32c5e2bbc17f1dba8280765af6b96e1fd81c15`；实际worker6656、launcher9072、supervisor35076，OS身份12项及终态清理通过。

[物理保存数据独审194/194](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_saved_independent_v1/report_v2.json)重算415项current/ZIP绑定、16个快照数组库存hash、560次恢复的begin/end收据、3612条外置容量记录、全部实际控制及时相。48个baseline中，q误差最大5.96e−8rad、qd9.54e−7rad/s、物体位置和R矩阵误差0、v1.86e−8m/s、ω7.15e−7rad/s，原生FP32物体奖励误差0，均满足预定门槛。重复求解的contact force6/efc力仍有最大3.28e−7/6.93e−7浮点差异，作为诊断保留，没有新增或放宽力值门槛。

scheduled工作量之外，另外记录了8个reset forward、构造中的8个Warp step API/8个forward API和4个CPU forward调用；嵌套API不与public graph replay相加。成本为worker84.007373s、watch85.188000s、native段70.283916s，范围嵌套，不相加作GPU busy。原空Warp数组setup失败版本及其24.156s watch成本单列保留，没有重启或覆盖它。

## 世界模型方向结果

[CPU消费报告](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_WM_CPU_v1/report.json)为`COMPLETED_CPU_SHARED_PHYSICAL_SECANT_COMPARISON`。A1L/P_ONLY/A2各一次batch16前向，总48个名义预测、6个梯度batch；256对物理central差分由三个模型共用，形成768行对照。没有重新运行物理、Actor或训练；输入、权重/归一化及RNG保持不变是绑定源码的运行期断言，文件bytes另行核验。

主目标保留当前参考phase k的原生FP32 object2+8两项奖励；ref k+1只作次列。AD主投影使用实际量化方向`d_eff=(u_plus-u_minus)/(2eps)`，nominal方向另列。内域free坐标只依已冻结动作边界定义，不依模型或物理响应挑选。每状态8方向的秩为8，free坐标维数18–20；边界法向、单边方向及完整20维Jacobian未测。

按预声明的两epsilon稳定/非平坦规则，128方向中93稳定且信号足够，30非平坦但不稳定，5平坦或低于噪声。两步长均为有效合法目标，无丢弃案例。

| WM | eps=.002 符号一致 | eps=.001 符号一致 | Pearson r，.002/.001 |
|---|---:|---:|---:|
| A1L | 78/93，83.87% | 78/93，83.87% | .6797 / .6754 |
| P_ONLY | 75/93，80.65% | 75/93，80.65% | .6765 / .6832 |
| A2 | 69/93，74.19% | 69/93，74.19% | .7841 / .7702 |

符号和相关性排序不同，各path/phase结果也不同。这些是同一组新状态上的描述结果，不能从一个训练seed推断方法因果或总体优劣。93个方向来自16个相依快照；两epsilon重复同一方向，768行不是768个独立实验。三模型标签下重复保存的1680行physical response deltas也只是同一560分支的重复标注，不是新增物理样本或WM预测误差。

[六面板图、SVG与CSV](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_WM_summary_v2/report.json)保留全部有限散点、离群点和不稳定/平坦方向。新版仅更正BCE/Brier科学计数显示及fresh-donor说明，四份CSV与原图版本逐字节相同；Root实际查看PNG，布局和单位清晰。

[消费保存结果独审30/30](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_v2_WM_CPU_saved_independent_v2/report.json)重算256个physical secant、噪声、全部768个AD投影/相位对照、150个分组及self15校准，没有重载或前向任何WM。首版审阅器在最终JSON写入时遇NumPy布尔标量序列化错误，原源码和partial结果保留；新版本仅归一化JSON scalar并另存输出，五份已重算数学JSON逐字节一致，没有更改数学或物理结果。

## 接触、归档漂移与解释范围

620个原生self15向量全为零。该标签是原生自碰撞found，不是手物接触或人类力标签。16个名义状态的240个二值位置全部为负，三个WM在.5阈值也全预测为负；BCE分别9.554e−6/1.035e−5/1.842e−5，Brier2.394e−10/4.185e−10/1.213e−9，F1未定义。不能把100%负例准确率写成正接触识别能力。

本次物理输入取新donor的实际state58，未直接套用旧归档状态。A0与归档一致，其他动作源出现非零漂移，包括A1L全61点body ω最大差2.9581rad/s。它们按协议单独报告，不挑donor、不重评旧Actor任务成绩。分支资格由同实例fresh donor与三次baseline比较确定；固定动作重放也不是重新执行Actor闭环。

初次快照的合法空Warp数组`ptr=None`错误、首次独审的schema/quat/force错误断言、CPU证据配置副本路径绑定拒绝、以及图表小数格式更正均有独立保留版本。路径绑定现使用真实OS command中的配置路径，并同时验证与保存副本的SHA相等；没有更改原物理数据、数值门槛或冻结consumer。

## 接下来推进完整运动

本段仍只有controlled legacy clip007[200,216)、0.3秒、一个保守lineage、独立ego N0。完整151帧和共享训练没有因此完成。当前合理结论是：在这些采样内域方向上，多数稳定方向的模型导数与物理同号，但错误方向和epsilon不稳定性仍明显；状态预测误差小不能替代动力学梯度检查。

[full151差距审阅](D:/workspace/supermimic-rough-prefix-learning-study/docs/full151_integration_gap_current104_v1.md)已定位真实native Reader可复用接口。下一步先建立独立的完整reference/连续速度/非零初始化/150控制基准及contact-intent束，再接真实controller/runtime、末端H8/replay与严格checkpoint身份；不padding当前16帧，不把旧q1接到新H123 q0，也不迁移旧104检查点。shared007[0,151)与legacy[200,351)保持不同身份，shared10缺少独立targets与真实reset/replay/transfer接入的问题继续保留。Goal active，无GPU任务，本轮没有新长训练。
