# 分源边界冲量目标已生产，WM旁支通过合成检查

2026-11-06。原152个恢复组、608个物理子步的监督目标一次实际构建完成，独立保存复算8/8通过。没有Torch模型、MJB、环境或物理执行；没有混入300 donor或fresh152。模型旁支和辅助损失另完成3/3自写ToyWM检查，尚未加载真实初始权重、运行真实native WM或启动训练。

## 目标数据与来源

全部24 baseline和128 direction组均入库。每组从旧H1的452组身份机械定位四个子步，逐字段匹配path/k/branch/substep/ticks/time；三条内部qpos27/qvel26位链、四次实际绝对控制位恒定均通过。152组的native58 pre/target及u与旧response和实际校准fit数据逐字节一致；没有改用snapshot8广播，也没有将原18维numeric shared-state门升级成58维位一致。

已保存每源HO/nonhand的世界系及palm系线冲量J（N·s）、即时物体COM角冲量K（N·m·s）、约束见证、原输入/目标/xi、152×4索引、所有操作数及求解掩码。J/K只汇聚冻结线/角动量输出的四个native FP32 h乘F/τ，不重算接触力矩或解码pyramid。使用原palm矩阵转置，不取inverse、正交化、移动力矩原点或再做EMA/C2。

全部152记录的raw palm旋转/位置、palm重力及DeltaT字节相同，满足后续单一固定物理context的表示条件；这是452个Reader记录与固定wrist来源所支持的范围，不是1808次直接独立mocap观测。

| 来源描述 | 全152组 |
|---|---:|
| 有HO约束见证 | 95 |
| 有nonhand约束见证 | 38 |
| 两源同时有见证 | 0 |
| 两源均未记录见证 | 19 |
| 选中子步中保留的原contact | 2831 |

见证取该源四步中任何合格边界约束，不要求正Fn或非零冲量，不是self15、人类意图或全局clearance。A2/k49的19组均无物体外边界见证；A2/k99、k149的环境承载留在nonhand。该数据来自原机器人模拟结果，包括失败轨迹，不能用作“期望成功抓取”或人类接触标签。

## 非零缺陷完整保留

用原native58的raw R9形成`I_p0/I_p1`惯量变换，保存以下监督观察值，不以归一化q观察值替换：

`r_v = v1-v0-(J_HO+J_nonhand+m*g_p*DeltaT)/m`

`r_w = omega1-solve(I_p1, I_p0*omega0+K_HO+K_nonhand)`

全152组的向量RMS：r_v=5.8592773e-6m/s，r_w=0.91644324rad/s；最大范数分别7.2234037e-5m/s和2.4930942rad/s。全13个总体/路径/相位/模式表及全部行均保留，未按缺陷大小筛选。r_w较大的A2/k49也没有删去。两种单位不合成总分，不能从这些数值推导准确率。

原始矩阵均可求解，I_p1条件数约166.24480–166.24489仅作数值描述；求解方程残差RMS约7.15e-20，不等同物理残差接近0。独立实现用fsum、标量P.T旋转及独立3×3消元，最大异实现差为J 8.67e-19N·s、K 5.42e-20N·m·s、r_v 2.89e-17m/s、r_w 2.18e-13rad/s。比较容差事前固定，仅核实现一致性。

r_saved描述当前模拟/观察路径的离散缺陷，不是连续物理定律的零值，也不是推理时已知修正。有限solver/implicitgyro/姿态步、FP32与observer等来源仍未分离；不唯一归因，不授予完整扳手、force-cap、人类力、真实梯度或任务资格。

## 旁支与损失的当前状态

新包装器保留原WM forward，使用同一次512维hidden的pre-hook接14维零初始化旁支：12个无量纲分源冲量坐标与2个模式logit。无模式门控，mode仅作公共BCE。两臂先分别deepcopy raw WM，再安装各自hook和head；禁止copy已包裹模块或pickle带hook/cache的整个模块，后续保存独立state_dict命名空间。

SI解码为`J=m*s_v*z_J`、`K=I_p0*(s_omega*z_K)`。尺度须与同一真实初始WM的effective max(delta_std,error_floor)相符；公共state/mode/impulse损失一致，method只增加带r_saved的监督目标一致性。预测R1、head与trunk到F64 solve的梯度保留，不经NumPy或detach；native FP32点损失由原调用者提供，不重复实现LogR。

3个Toy检查验证了合成67输出/state/self15函数位一致、两臂storage/cache及RNG隔离、SI单位和尺度、无真实mode门控、非零缺陷保留，以及head/trunk/predR的有限非零梯度。1个ToyWM、4次toy forward、1次autograd.grad和1次backward，0optimizer。它不是实际152数据或真实native WM的功能验证。

下一步必须在真实初始component上独立核验同batch函数身份、normalizer/context/RNG，再冻结精确loss系数、匹配优化预算、独立artifact与健康检查。公共loss当前没有self15监督，trunk训练后可能改变self15输出；初始67函数身份不能保证训练后接触惩罚保真或Actor可用，不将2个新mode冒充15个旧标签。首次双方健康checkpoint和已存ETA后仍按用户约定暂停Goal。

本轮没有新训练，原确认FAIL、H8取舍、原参考/三臂任务FAIL、whole-hidden未知、多轨迹/真实ego与硬件要求保持。

## 成本与证据

目标构建worker4.2798337秒，152组/608子步，652个来源；源审、合成、独审和描述绘图另计。完整四单位图已目视检查。新数据的enrollment只证明来源与定义检查通过，不是物理或学习收益资格。

- [目标构建报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_boundary_impulse_targets_CPU_v1/report.json)，SHA `318366deb991c1a875593c41156bdbbbf1266c39b772d41ecf0bebb54e6c1d81`
- [目标NPZ](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_boundary_impulse_targets_CPU_v1/all152_boundary_impulse_targets.npz)，SHA `bc22e2b0d6a6ff333f8431eee689134814df1ea7059e8346f27e74f7b0392f80`
- [独立8项报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_boundary_impulse_targets_independent_v1/report.json)，SHA `3703c02f0e72753582ab5e8ff52b7ab79bc5409220e06e7fd7282a0e009851ed`
- [完整目标描述图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_impulse_targets_description_v1/all152_impulse_and_observed_defect_targets.png)
- [全部13项描述表](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_impulse_targets_description_v1/report.json)
- [旁支/损失Toy READY](D:/workspace/supermimic-full151-observation-study/outputs/full151105_boundary_impulse_component_preparation_CPU_v1/READY.json)，SHA `707f09cca3ca92ee743bc82eb3212db29b68b1e29517f79351c22f45b43ed291`
- [配方源码草稿](D:/workspace/supermimic-full151-observation-study/docs/full151_boundary_impulse_matched_loss_recipe_draft_v1.md)，系数和实际训练尚未冻结/执行。
