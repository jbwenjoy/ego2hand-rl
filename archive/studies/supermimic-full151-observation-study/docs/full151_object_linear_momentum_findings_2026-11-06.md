# 物体线动量：全部接触与速度增量相容，手物力不能代替总力

2026-11-06。冻结协议后一次实际执行，保留原full05全部1808个5ms子步（1200 donor、96 baseline、512 direction），没有新物理采样或训练。读取一次被动MJB常量，没有创建MjData、FK、forward、step、Torch或GPU。全部9473条原generated contact均保留，未按Fn、接触深度或残差筛选。

本轮得到条件化的离散线动量证据：物体接触力加已知重力与原速度增量相容。它提供后续接触动力学结构的数据依据，不是完整六维扳手、手部容量、人类接触力或任务成功证书。

## 绑定与定义

物体为body2的独立freebody，无子体，body_ipos恰为0；free joint0的qposadr/dofadr均为0。因此使用原完整26维qvel的前三维世界线速度，不能误取手关节后的速度或WM的palm-frame速度。

实际质量为WP FP32 readback的0.029999999329447746kg，A0/A2、after_copy/after_set_const_0均与固定context相符。结构和其他静态受力路径检查未留下unsupported项：相关armature、damping/poly、frictionloss、spring、gravcomp、fluid、tendon、equality、actuator coupling及plugin/sleep均满足本次限界。该结论限于保存的静态元数据及绑定源码，不代表逐步未保存的力分解也被独立观测。

使用源支持的有效WP步长0.004999999888241291s和重力[0,0,-9.8100004196167]m/s²。MJB nominal timestep=.005、gravity=-9.81；WP转换为FP32。nominal solver tolerance=1e-8，经固定io源码floor至1e-6、再FP32化；积分器implicitfast、Newton 10 iterations。options没有新增运行时WP readback，不把nominal字节冒充实际设备字节。

逐行计算：

`delta_p = m * (F64(v_post_FP32) - F64(v_pre_FP32))`

`residual_impulse = delta_p - dt * F_all_object_contacts - dt * F_gravity`

`residual_force = residual_impulse / dt`

接触力来自本子步solver-PRE的已解码force6与frame：物体在geom2取正、在geom1取负，`frame.T @ force6[:3]`；不把直接力矩加进线性力。不使用累计时钟差作dt，不跨restore拼差分，不重解金字塔EFC，也不用重排host地址索引raw WP efc_force。静态COM0和全部物体外接触的证据完整，1808行均满足本轮解释前提；该eligibility不是准确性通过门。

## 全量结果

| 原总体 | 子步数 | 等效力残差向量RMS / N |
|---|---:|---:|
| ALL | 1808 | 3.3890115e-5 |
| donor | 1200 | 3.9671818e-5 |
| baseline | 96 | 1.7569502e-7 |
| direction | 512 | 1.9158745e-5 |
| A0全部 | 904 | 4.1640479e-5 |
| A2全部 | 904 | 2.3730789e-5 |

事后描述的全1808残差向量范数中位数为8.6845034e-8N，p99为8.0700830e-7N，最大9.6613544e-4N（A0 donor，phase58/substep1，原row232）。峰值完整保留；这些分位数不是新阈值、误差上界或置信区间。全部路径/模式表和逐行数组可查看。

接触分解为3589 HO、4149 other object boundary、1735非物体接触，内部物体对0；本次所有9473条均具有原约束力证据，物体外部不合格条数0。other外部有序pair为(0,2):2028、(0,1):53、(0,3):2068；原catalog的geom0是terrain，1/2/3是三个物体geom。A2 donor曲线显示后段由terrain接触承载，而手物力为零；它不能解释为手仍在支撑物体。

额外事后描述中，若从已保存的全量分解中省去nonhand物体接触，仅保留HO与重力，等效力残差向量RMS为0.2883156N。它是省略已知受力项的算术演示，不是新物理消融或因果试验，未用于选择样本或设置门。用于从ego物体运动推断手需求时，也必须明确环境接触；不能把地面承载归给灵巧手。

## 验证范围与限制

Root另用逐分量math.fsum重写接触求和，从原1808样本复算11组向量数组，全部满足浮点算术一致性；最大力求和差1.7763568e-15N、冲量差8.6736174e-18N·s。重算HO与原生产者保存的HO聚合最大差同为1.7763568e-15N。检查没有重新加载MJB、模型或物理。

这明确是Root的另一套算术复算，不是另一位研究者完成的独立实际审计。两个代理的source-only文本已保存，但其中承诺的完整source_pins报告未完成；Root另核10份关键原执行源码字节与原清单相同。实际静态元数据只读取一次，不虚称独立重载。3/3合成检查覆盖frame/符号/内部抵消/弱力及未知地址、FP32先转F64和dt、两path两阶段质量与完整速度/时钟输入。

即便数值残差较小，有限solver stationarity、implicitfast有效加速度/速度导数和完整passive/constraint广义力分解并未逐步保存，不能唯一解释剩余误差来源。原外施力零来自每步get_data_into后的源码断言及保存max0，未保存完整原始外力vector。没有真实人类力、全局clearance、HO力梯度、whole-hidden、物理参考或CHORD任务资格。仍为单缓存谱系、相依donor/restored记录、ego N=0。

线性检查也没有验证角动量、COM接触力矩或完整6D扳手。下一项只准备一次使用已存实际惯量/姿态/角速度/接触位置的角动量合同；不再读取MJB或增加训练。完成6D动力学与接触支撑的限界后，再决定如何修改参考或模型结构和任务对照。

## 证据与成本

实际worker4.1618856秒，1被动MJB、1808保存子步；Root交叉检查、源码/合成/绘图另计。两幅完整科学图已目视检查；v1首图页脚遮挡已保留失败记录，v2仅修排版，未改数据或统计。

- [冻结运行协议](D:/workspace/supermimic-full151-observation-study/protocols/full151105_saved_object_linear_momentum_CPU_run_v1.json)，SHA `177ffb8f629758498d1524c8cab7738431552b287a3fd7828557f16323d36067`
- [实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_linear_momentum_CPU_v1/report.json)，SHA `e2a866c71e15357c18bb9e055274df761b0795b036ef0fd3a9c633811a465172`
- [Root另式保存复算](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_momentum_root_crosscheck_v1/report.json)，SHA `7a806fe6bf08791d6ba6aaffc4c55a4328e56eaffe68908ae6225aaa8e3cac79`
- [全部1808残差图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_momentum_figures_v2/all1808_linear_momentum_consistency.png)
- [两donor接触来源图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_momentum_figures_v2/both_donors_force_source_decomposition.png)
- [全部路径/模式和事后描述](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_object_momentum_figures_v2/report.json)
