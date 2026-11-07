# 研究方案 v0.11：线动量证据已建立，补齐旋转动力学合同

2026-11-06。Goal active，无新训练或后台物理。v0.10的全部1808子步线动量检查已实际完成；原完整目标和失败证据继续保留。

一次被动MJB读取核实了独立COM0自由物体、速度地址和静态受力通路，实际质量由两path两阶段WP readback绑定。所有物体边界接触均有原力证据，计算包含手物与terrain，不按Fn或残差筛选。全部1808子步的等效力残差向量RMS为3.3890e-5N，中位范数8.6845e-8N，最大9.6614e-4N。没有预设残差准确性通过阈值。

Root另一套逐分量求和复算一致，但不冒称另一研究者的独立实际审计。静态元数据仍是一次读取/源码绑定范围；未保存的有限solver、implicitfast有效加速度和完整广义力分解继续限制残差归因。单模拟谱系与ego N=0不变。

接触来源是本轮对方案的实际约束：3589条HO与4149条terrain–object接触必须分开。事后描述中，仅保留HO与重力、遗漏已知nonhand接触会得到0.2883N的等效力残差RMS；这不是物理因果消融。后续从物体运动构造手部支撑需求时必须处理环境接触，不能把地面承载归给手，也不能把模拟力当未知人类力。

下一项限定为一次角动量/COM力矩一致性实验的源码和观测合同。复用已保存被动元数据与实际inertia/iquat readbacks，使用每行pre/post姿态与body-frame角速度、solver-PRE接触位置和全部边界force6，直接力矩只计一次。先明确端点姿态转换、惯性坐标和时间离散误差，不额外读取MJB、不补采物理、不训练Actor；Root冻结后才允许实际算术。线动量检查不能替代旋转或完整六维扳手资格。

三模型H8结果保持：initial/point/response的ALL h8位置RMS为64.537/34.663/38.059mm，响应臂相对点值臂存在六块和路径取舍。原有限响应确认条件仍失败，taskscore梯度共享子集幅值误差未改善。当前不足以选响应组件为Actor赢家；先据物体动力学与接触支持证据修改参考或模型结构，再回到共同真实学习初态的同预算CHORD任务对照。

原参考/三臂任务FAIL、几何/控制与warmup限制、多轨迹共享与heldout对照、真实ego接入、硬件子版本/观测契约仍未完成。未来新的训练在首个健康checkpoint与保存ETA后按用户约定暂停；历史暂停已由用户恢复，本轮没有新训练，不重复暂停。

- [线动量完整结果与限制](D:/workspace/supermimic-full151-observation-study/docs/full151_object_linear_momentum_findings_2026-11-06.md)
- [H8完整结果](D:/workspace/supermimic-full151-observation-study/docs/full151_component_H8_findings_2026-11-06.md)
- [上一版方案](D:/workspace/supermimic-chord-study/docs/research-plan-v0.10.md)
