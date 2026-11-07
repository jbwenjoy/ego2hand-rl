# 研究方案 v0.13：目标数据与旁支就绪，先做真实组件验证

2026-11-06。Goal active。分源边界冲量目标已实际生产，原152组/608子步全部入库，独立保存复算8/8通过。9项来源/身份/控制/坐标/求解门通过，未混入donor300或fresh152，没有模型、MJB或新物理。原完整目标与既有失败证据保持。

每组保留HO/nonhand两源J/K及约束见证、实际native58 pre/target/u/xi和原始操作数；95组有HO见证、38组有环境见证、19组两者均未记录，mode不要求正Fn或非零冲量。native rawR9定义下的r_v向量RMS为5.8593e-6m/s、r_w为0.91644rad/s；非零缺陷原样保留，不视为严格零缺陷物理定律。这些是机器人模拟结果，不是期望成功演示、人类力、force-cap或真实ego标签。

共享14维零旁支与辅助损失已完成3/3 Toy检查：两臂先克隆raw WM再各自挂hook，原forward保留，读取同次hidden，无额外WM forward、无contact oracle输入或mode门控。公共native52点损失、mode BCE和无量纲冲量监督相同，method仅多带r_saved的监督一致性项；F64 solve到预测R/head/trunk梯度通路通过合成检查。真实初始权重、native WM、实际152数据和优化器尚未执行，不能把Toy身份升级为实际功能资格。

下一项收口为真实组件的受控CPU验证和匹配训练准备：

1. 固定已验证的原A2 initial_component及其源/normalizer/context，用实际152输入检查原WM和两包装臂的同batch state58/self15初始函数身份、单次hook/forward、零旁支、独立storage/cache与原始RNG/normalizer不变。严格自有artifact命名空间，不调用或冒充旧105健康/全Runner恢复。
2. 在同一有效s52、实际m/I/g/DeltaT、相同14头、公共损失和数据顺序下，明确固定loss系数、优化器/更新预算、保存/健康与ETA协议。r_saved只在loss作监督目标，不进入rollout输入；不以fresh/donor表现事后调权或挑样。
3. 只有上述实际验证及新recipe就绪才启动两组件训练；首个双方正常checkpoint落盘、独立健康检查与保存ETA后按用户约定暂停Goal。本轮没有新训练，不重复对历史checkpoint暂停。

公共loss当前未包含旧self15监督，trunk训练后可能改变该通道；初始67函数同一不保证fit后接触惩罚保真或Actor可用。须在后续评价中明确此限制，不把2个新mode当15个旧标签，也不把离线组件改善当CHORD任务收益。旧fresh/缓存已被观察，只能作探索性评价，不重新宣称blind或独立轨迹泛化。

原参考/三臂任务FAIL、有限响应确认FAIL、H8取舍、warmup差异、whole-hidden未知、真实多轨迹与ego N=0、硬件子版本/观测契约仍未完成。后续必须回到物理参考/接触支持和共同真实初态的同预算任务对照，不以离线数据检查结束完整研究目标。

- [目标生产与旁支检查完整结果](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_targets_findings_2026-11-06.md)
- [实际目标数据](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_boundary_impulse_targets_CPU_v1/all152_boundary_impulse_targets.npz)
- [旁支Toy READY](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_impulse_component_preparation_CPU_v1/READY.json)
- [尚未冻结的精确loss配方草稿](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_matched_loss_recipe_draft_v1.md)
- [上一版](D:/workspace/supermimic-chord-study/docs/research-plan-v0.12.md)
