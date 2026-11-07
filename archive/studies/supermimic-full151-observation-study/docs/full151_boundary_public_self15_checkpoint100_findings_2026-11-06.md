# 两臂共同native self15监督：首个配对检查点与暂停交接

2026-11-06。新namespace实验已实际启动。双方100次检查点已保存，独立健康5/5通过，ETA已保存，达到用户约定的本次Goal暂停条件。记录时两组已完成固定1000更新，实际剩余0；最终1000独审/效果评价仍待恢复后进行。

本轮从同一未改A2 initial重新开始，原WM/15通道、零14头、有效s52、norm/host/context、152输入顺序相同。新标签仅来自已独核7项的label-only JSON，按原plan逐152身份/raw25映射及15ms缓存时相核对，不使用带模型概率的metric archive作监督、不进入forward。

新公共项严格逐行FP32：s=mean52((D/s52)^2)，c=sum15 BCE，native_pair=mean152(s+.1*c)，再+mode FP32、+impulse F64；两组唯一方法差异为cons0/1。0.1来自原native源码默认，非结果调权；mean15不能替代sum15，不能乘rewarddt或H8。原003物理项和旧5标签字典保留，state仍独立纯point，新native_pair/self15字段不冒旧state。新自15监督没有新Parameter，仍16个tensor。

每组CPU4/FP32、AdamW lr1e-4、betas(.9,.999)、eps1e-8、wd0、clip1，固定1000更新，仅100/1000保存。首次训练前向复用为e427 native58/self15位级检查与正零14，无额外验证FW；norm/context/输入及三种label digest不变。新checkpoint为AFTER-I，原始grad与loss snapshot为PRE-I、权重after(I-1)，不能用PRE100日志冒AFTER100模型复测。

新reader仅3次自有CPU加载（本次initial_pair+双方fixed100），核16参数/Adamsteps/clip前后原始命名梯度、冻结norm/context/host、THIS RUN rawRNG、100行/bytes/SHA前缀，以及PRE-I nativeD52、FP32平方/pointrows、self15 logits/targets/BCE元素与sum15/native_pair/public归约。BCE跨stride重算仅用预锁FP32数值包络，后续精确归约从实际saved BCE元素起；nativeLog与旧impulse/cons的捕获来源不冒本次模型重放。无模型构造/FW/AD/MJB/Env/物理/GPU。

首100测得配对均时0.023938s；剩900对+保存/显式20s收尾估计41.6s，预计UTC 2026-11-06T07:26:02.263428+00:00。范围为规划启发式，非CI或加速证据；若此处已完成1000，该ETA仅为历史测量。

这只证明新监督/优化/序列化健康，尚未证明拟合、接触惩罚或任务收益。旧无self15实验的固定1000负面结果原样保留；原参考/任务FAIL、有限响应确认FAIL、旧H8取舍/共同warmup缺口、wholehidden未知、单谱系/ego0、多轨迹与硬件合同仍未解决。完整研究目标尚未完成。

- [机器交接](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_public_self15_checkpoint100_pause_handoff_v1.json)
- [新100健康](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_public_self15_checkpoint100_saved_independent_v1/report.json)
- [本次冻结配方](D:/workspace/supermimic-full151-observation-study/protocols/full151105_boundary_public_self15_matched_training_recipe_v1.json)
- [ETA](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_public_self15_matched_training_CPU_v1/checkpoint100_ETA.json)
