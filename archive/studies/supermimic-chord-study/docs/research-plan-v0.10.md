# 研究方案 v0.10：由多步误差回到接触动力学约束

2026-11-06。Goal active。已完成原生加载、保存恢复、受控训练和多项模型检查，但原完整参考/任务仍未通过，CHORD任务收益、共享多轨迹、真实ego及硬件契约继续开放。本版保留v0.9与此前所有失败证据，不增加Actor训练。

三固定模型H8已实际完成：24次B286前向、6864预测，独立保存复核24/24。初始A2、点值final1000、点值+响应final1000在全部286重叠窗口的h8位置向量RMS分别64.537、34.663、38.059mm。两校准臂在ALL h8的六块原单位误差均低于初始模型；响应臂对点值臂为q/qd/v较好、p/R9/omega较差。A0位置方向改善、A2变差，不能选总体赢家。全部h1..8、A0/A2/ALL、六单位曲线和432统计保留。固定已记录控制不等同Actor闭环或新轨迹；ego N=0不变。

原确认条件仍失败：normalized有限响应MSE降低22.603%，点式损失增加10.180%。探索性任务score梯度在共享6个stable方向符号一致，但响应臂幅值误差更高。H8结果不能覆盖这两项取舍，也不唯一解释为遗忘。

下一步只做一个有界的离散线动量检查，服务于后续物体动力学结构和接触支撑目标：

1. 固定原full05全部1808个子步（1200 donor、608 restored），不按接触或残差筛选。沿原PRE/POST自由关节速度、solver-PRE接触frame及原有序geom对计算，含terrain等nonhand对物体的作用，区分手物与其他物体边界力。
2. 允许一次被动MJB常量读取，不创建MjData/FK/forward/step；实际质量取已存WP actual8 readback，不能取nominal host质量。核COM0、独立freebody、地址、重力/补偿、damping/armature/passive/fluid/equality/actuator路径与实际FP32步长。
3. 在可识别范围保存 `m*(v_post-v_pre)`、各已知力的离散冲量及未分配残差，单位N·s和N。使用F64(FP32速度)差与native有效FP32 dt；不用累计时钟差代替dt，不把直接接触力矩加入线性力，不按Fn阈值删弱力，不重解金字塔EFC。
4. 不预设残差精度通过阈值。遗漏项无法排除时写UNKNOWN；有限solver收敛、implicitfast有效加速度和未存广义力分解仍可能进入残差。即便残差很小，也只形成同一模拟缓存的条件化一致性证据，不是人类手力、完整六维扳手、单指容量或物理可行参考资格。

该步骤之后依据物理记录修正参考/接触支持或物体动力学结构，再回到共同真实学习初态下的CHORD同预算任务对照；不靠更换窗口、综合单位或追加Actor迭代掩盖当前模型问题。真实ego等待可用数据，硬件部署前核子版本/观测契约，不修改驱动或CUDA。未来新的训练仍在首个健康checkpoint和已存ETA后按用户约定暂停；历史训练的暂停已履行并由用户恢复，本次检查不触发重复暂停。

- [H8完整结果与路径差异](D:/workspace/supermimic-full151-observation-study/docs/full151_component_H8_findings_2026-11-06.md)
- [线动量来源与限界](D:/workspace/supermimic-full151-observation-study/docs/full151_object_linear_momentum_source_feasibility_v1.md)
- [来源合同审阅文本](D:/workspace/supermimic-full151-observation-study/docs/full151_object_momentum_source_contract_independent_v1.md)，其承诺的source_pins文件未完成；[Root另核10份原执行源码](D:/workspace/supermimic-full151-observation-study/outputs/full151105_object_momentum_source_root_binding_v1/report.json)，不冒称独立审计已完成。
- [上一版](D:/workspace/supermimic-chord-study/docs/research-plan-v0.9.md)
