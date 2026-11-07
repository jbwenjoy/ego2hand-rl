# 研究方案 v0.18：多步状态改善，继续分离辅助监督作用与任务收益

2026-11-06。Goal active。新public15两项评价完成：fresh共同可见设计实际6FW/480rows/3uVJP及独审11/11；native donor H8实际24FW/6864rows/16head及独审v2 24/24。新donor300缓存标签独审6/6完成。没有新训练或物理采样；当前没有训练/评测进程，本轮不触发checkpoint暂停。

两份校准模型改善了这里的端点/缓存接触与H8状态预测，但没有一致改善局部动作敏感度。H8位置RMS为64.54/35.02/31.46mm（原WM/public+aux/+consistency）；consistency相对公共对照改善部分多步指标，但qd/v和缓存precision更差。不能推断Actor收益或独立轨迹泛化。

接下来按完整研究目标推进：

1. 以已形成的native52+self15-only最小提案建立新schema/recipe/训练与14有效Adam状态健康reader，补同1000更新预算的纯native监督对照。保持原initial、152样本、self15逐row定义、norm/xi/CPU精度与优化器设置，不调λ或根据结果筛数据。新训练若启动，按用户约定首checkpoint正常落盘并校验后保存ETA再暂停Goal；不重复把历史checkpoint条件应用到本轮评价。
2. 保留fresh共同可见输入诊断的边界。后续物理敏感度资格须补直接/独立资格的逐branch PRE58与相位身份，不能把snapshot8 repeat19当已测PRE152，也不能借旧J/K/r_saved冒fresh目标。保全部32方向/64行、原6/25/1分类与两epsilon口径。
3. 对需要进入任务的模型先资格化GPU精度、固定norm与组件安装；再以共同collect/replay/Actor/Critic/optimizer/RNG起点在新namespace比较。own component不能伪装105完整resume，`.eval/requires_gradFalse`不能替代对外部norm.update/重新启用参数梯度的控制。
4. 继续接触可行性、力容量、同预算任务、多轨迹及ego数据接入工作。原参考/SCP与物理任务FAIL、旧有限响应确认FAIL、角动量observer缺陷、whole hidden未知、H123人类接触未知、ego N=0与硬件版本/观测契约均保留。当前指标不授予这些资格。

- [fresh端点/接触/梯度结果](D:/workspace/supermimic-full151-observation-study/docs/full151_public_self15_fresh_findings_2026-11-06.md)
- [H8结果与完整证据](D:/workspace/supermimic-full151-observation-study/docs/full151_public_self15_H8_findings_2026-11-06.md)
- [纯native监督对照源码提案](D:/workspace/supermimic-full151-observation-study/docs/public_native_only_matched_baseline_source_proposal_v1.md)
- [上一版](D:/workspace/supermimic-chord-study/docs/research-plan-v0.17.md)
