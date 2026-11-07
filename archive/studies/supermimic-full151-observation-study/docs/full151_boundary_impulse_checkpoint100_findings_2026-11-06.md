# 分源边界冲量组件：实际初始验证与首个配对检查点

2026-11-06。首个双方100次检查点已实际保存，独立健康读取 5/5 通过，ETA已保存，达到用户约定的Goal暂停条件。本轮1000次已在记录时完成，实际剩余0；首100次ETA保留作历史测量。

实际初始验证只读自有初始组件，1次native构造、2次raw复制、3次152批前向；原WM与两旁支的native58/self15逐位一致，14个新通道为正零，原始RNG、输入、normalizer、context及参数不变。独立保存审计v3为10/10，零Torch/模型/前向。v1 8/10和v2 9/10完整保留：前者错误假定NumPy sqrt复刻Torch位级值，后者错写非旋转差分符号；均为审计器错误，未改变producer、数据或科学容差。新审计绑定两份此前逐位审核过的native scale52原字节，使用原生target−pred非旋转差分。

训练前固定recipe：两臂同一实际WM初始、相同零14头、152条顺序、有效s52和m/I/g/DeltaT，公共native点/mode/无量纲冲量损失系数均1。两臂均计算一致性，只有一致性系数control0、method1不同。CPU4线程FP32原网络、F64冲量/求解，AdamW lr1e-4、betas(.9,.999)、eps1e-8、weight_decay0，clip1，固定1000次，仅100/1000保存。首次训练前向复用为actual初始位级验证，没有附加验证前向。v1合成训练器因可见CUDA被优化器初始化而失败已留存；v2在importTorch前隔离CUDA并完成3项Toy检查，原数学/recipe不变。

独立健康进程只加载这次自己的initial_pair与两份AFTER100字典，共3次CPU读取，检查16个参数、原始Adam状态/step100、PRE100的命名裁剪前后梯度、冻结norm/context、THIS RUN原始RNG及100条日志前缀身份；没有模型重建、前向、AD或物理。PRE100梯度/日志来自已更新99次的权重，不能当作AFTER100模型复测。

首100次测得配对均时 0.024208s；按剩900对和保存/显式20s收尾估算剩余 41.9s，预计UTC 2026-11-06T05:38:40.795081+00:00。范围为规划启发式，不是置信区间或加速证据。日志两臂摘要及进程快照见handoff。

本轮只获得构造/优化与序列化健康证据。原151参考几何FAIL、旧三臂任务FAIL、有限响应确认FAIL、H8取舍及warmup差异均保留。152条来自单一已见机器人模拟谱系，r_saved保留observer/solver非零缺陷；不是成功人类力监督。旧self15没有公共标签损失，fit后可改变；尚未完成任务、Actor、物理梯度、force-capacity、多轨迹、ego或真机资格。

- [可复查handoff](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_impulse_checkpoint100_pause_handoff_v1.json)
- [首100独立健康](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_impulse_checkpoint100_saved_independent_v1/report.json)
- [ETA](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_impulse_matched_training_CPU_v1/checkpoint100_ETA.json)
- [冻结配方](D:/workspace/supermimic-full151-observation-study/protocols/full151105_boundary_impulse_matched_training_recipe_v1.json)
