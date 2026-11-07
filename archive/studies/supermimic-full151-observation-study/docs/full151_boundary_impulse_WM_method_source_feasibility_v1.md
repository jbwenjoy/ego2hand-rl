# 单一下一方法：分源边界冲量监督的 WM 动量一致性正则

**只改WM学习，不再加Actor更新或support奖励。** v0.11已说明总物体力必须包含terrain，现有几何support是单位力cone包络，不是实际可执行载荷。建议在角动量惯量/符号/相位合同完成后，做一个隔离的WM-only组件：**学习HO/nonhand分源冲量，并约束它与物体v/ω预测一致**。它是软物理一致性正则，不是假装精确Newton–Euler硬积分或新的动力学资格；当前不产数组、模型或执行协议。

**最小接口。** 原 `WorldModel.forward(x58,u20,dynamics278)` 的 `x1_58,self15_logits` 完全保留；新组件额外返回 `boundary_impulse_hat[N,2,6]`（两源为HO/nonhand；前三线冲量N·s，后三关于即时COM的角冲量N·m·s）及 `boundary_constraint_logits[N,2]`。用原 `wm.net[-1]` 的512维输入features挂一个14维旁支，捕获同一次forward中的hidden、不detach、不重复WM调用；原native67输出/52维delta归一化不改。self15仍是手部自碰传感器语义，不能拿来冒充HO。新artifact/loader命名空间记录旁支与物理buffers，不伪造旧105健康或修改冻结native源。

**唯一标签生产口径。** 每个原训练20ms组的四个5ms solver-PRE，沿原顺序解码证据及COM求和：`J_c=Σ dt*F_c`、`K_c=Σ dt*τ_COM,c`，c分别HO/nonhand。保留terrain等nonhand；不按Fn阈值、残差或成败筛选，不将实际机器人接触当人类意图。模式标签是该组是否存在合格的solver boundary constraint（允许力为0），不是H123声明、self15、几何potential或human contact。确认无HO时HO冲量标签为0；terrain承载留nonhand。标签完整性不足就阻止dataset enrollment，不能补0或按结果删组。COM矩必须等角动量合同完成；线动量通过不能代替6D标签资格。世界系label与WM腕系之间只用已绑定固定palm旋转做显式转换，mass/inertia/iquat取actual8/readback而非nominal模型；不加几何support半径或单位cone缩放。

**可直接实现的单一损失改变。** 两个匹配组件都使用原native52 state loss，并具有完全相同的14维旁支、mode BCE和冲量监督；**method只多一个动量一致性项**。预状态和预测末状态均在同一腕系，I(R)由actual inertia及R形成：

```
r_v = v1 - v0 - (J_HO + J_nonhand + m*g*ΔT)/m
r_w = ω1 - solve(I(R1), I(R0)*ω0 + K_HO + K_nonhand)
L_cons = mean(((r_v_hat-r_v_saved)/s_v)^2,
              ((r_w_hat-r_w_saved)/s_ω)^2)
```

`s_v/s_ω`只取共同初始已冻结delta_norm的46:49/49:52；冲量监督相应除m、I(R0)转换成速度单位再归一化。`r_saved`来自同一真实端点与实测分源冲量，保留已量化的有限solver/积分缺陷，**不把真实残差改成0**。mode概率可软门控冲量旁支；实测模式只作target，rollout只能用预测概率，不能以真实contact遮罩teacher-force。系数、数据split与更新数须Root在新recipe前置固定，不根据fresh结果调权；本页不设置新准确性门。

**保持可解释对照与不泄漏。** 第一实现只沿原校准训练152/64及其608个源子步构造labels，不混入已用于H8的300 donor或fresh152响应挑超参。两组件克隆同一A2初始WM/norm/xi，旁支只初始化一次再逐字节复制；原state forward因此初始完全同函数。两臂相同输入顺序、固定actual absolute-u、loss公共部分与优化预算，记录全部参数/normalizer/RNG身份。新监督不能进入dynamics278、真实collect reward或Actor prior；旧fresh、H8与全部缓存已看过，只能做探索性检验，不能重新包装成blind泛化。未来CHORD对照必须让全部先验臂共用同一明确声明的WM初始/物理上下文并重核Actor/Critic等rawinitial；若用组件warm-start必须另立身份，不能沿旧from-scratch声明偷偷resume。

**为什么不先改supportloss/force LP。** `prior_hook.policy_scope`仅在imagined soft reward加`.02*bonus`，不训练WM；binding_v3复用的`support_match-.1P`及mu.1单位cone/soft距离权重不含N容量，不能教给WM缺失的物体动力学。matched LP的`bounds=(0,None)`允许无限Fn；原151可行不证明PD执行器可实现。`PenMDP.pd_torque`的实际20个forcerange仍保持，不能把LP λ或观测到的最大力当hand force-cap，也不能把环境载荷分配给H123。单一新方法先使模型学习**实际有界执行器运行下的分源动力学**，不授予超出数据的force-cap或contact reachability证书；原human unknown/H123目标及所有任务FAIL不改。

该改动成功与否都须回到同一native151物理参考/接触支持及共同真实初态的CHORD任务，而不是用更长Actor训练掩盖动力学偏差。当前只有来源和接口可实施性；角动量、完整labels、recipe、paired组件训练和任何新任务执行均待Root明确决定。
