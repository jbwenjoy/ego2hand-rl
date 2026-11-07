# 研究方案 v0.12：角动量检查完成，进入分源冲量监督目标

2026-11-06。Goal active，无新训练或后台物理。原完整目标保留，原参考/任务失败、多轨迹、ego及硬件要求未完成。

全部1808子步/9473条contact的角动量实际算术已完成，独立保存复算9/9通过，0新MJB、模型或物理。F64端点观察、实际惯量、body/world坐标和PRE接触力矩的定义均已冻结；全部行的力矩残差向量RMS为1.129187mN·m，最大14.518062mN·m。接触总力矩RMS为18.978215mN·m，事后同单位量级比为5.9499%，不是准确率或新通过门。两端姿态必须分别参与角动量，direct torque只计一次。

线动量与角动量的单位不同，不直接比较两个残差数值。有限solver、implicitfast/gyro、有限姿态步和观察约定的影响未能分离；不能把1808行eligibility或9/9实现一致性升级成完整6D物理证书。原HO/nonhand分开、terrain不能归给手、human力未知的约束继续成立。

下一项进入一个WM方法的目标生产准备，不追加Actor迭代，也不改旧几何support奖励或无Fn容量的LP：

1. 仅用原152个restored的20ms训练组及其608子步，形成HO/nonhand各自的线/角冲量和solver边界约束见证。严格join已有native58 pre/target、实际绝对u与固定palm坐标；四个native FP32 dt逐项汇聚，mode不是self15、positive-force、人类意图或几何potential。
2. 不将donor300或fresh152混入该训练、归一化或调参。已有全部缓存/旧fresh的状态已被观察，后续评价只作探索性评价，不能重新宣称blind或独立ego泛化。
3. 两组件共用同一14维旁支、公共模式/冲量监督、数据顺序及真实初始WM/normalizer/xi。唯一method差异是状态与冲量之间的监督一致性项；旧native67输出初始同函数，原self15语义保持。
4. 若使用`r_saved`，它是同一观测端点与实测冲量形成的监督目标，保留原离散缺陷；不是零缺陷Newton–Euler定律或推理时可用的修正。目标不进入dynamics278或rollout输入，不使用真实contact mask teacher-force，也不保证未见输入上的零HO梯度。

当前只有source方法建议，目标dataset、坐标/组身份合同、损失精确尺度/系数、recipe及新组件训练尚未完成。先完成目标来源绑定和实际生产，再冻结matched recipe；不用角动量残差事后挑样或调整权重。后续仍须回到物理可行参考、接触/执行器范围以及共同真实初态下的CHORD同预算任务对照。

原有限响应确认FAIL、任务score梯度幅值取舍、三模型H8路径/状态取舍、原三臂任务FAIL、warmup差异、whole-hidden未知和ego N=0均保留。未来新训练在首个健康checkpoint与已存ETA后按用户约定暂停；历史暂停已由用户恢复，本轮无新训练。

- [角动量实测与完整限制](D:/workspace/supermimic-full151-observation-study/docs/full151_object_angular_momentum_findings_2026-11-06.md)
- [单一WM方法源码建议](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_WM_method_source_feasibility_v1.md)
- [线动量结果](D:/workspace/supermimic-full151-observation-study/docs/full151_object_linear_momentum_findings_2026-11-06.md)
- [上一版](D:/workspace/supermimic-chord-study/docs/research-plan-v0.11.md)
