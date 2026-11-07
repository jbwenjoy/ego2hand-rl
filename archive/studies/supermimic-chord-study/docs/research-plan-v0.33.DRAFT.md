# 研究方案 v0.33：三指原生replay与首次GPU配对Actor学习

2026-11-07，承接[v0.32](research-plan-v0.32.md)及完整Goal。上一回合和本回合均为progress。本回合首次将新phone任务、原生replay、共享几何/动量WM、Actor、Critic及工程CHORD支持代理连成真实优化循环。完整目标仍未完成。

## 从实际三指初载进入新任务数据

两个dt分别实测一次reset和20ms控制，Reader桥25/25通过：原始state58与probe030完全一致，Float32零action保留真实Float64 GN ctrl；初始三pad Fn约1.485/1.081/1.435N，非pad与table为零。实际192个积分步、2次reset forward、0observer forward、0NN。20ms三pad联合载荷覆盖仅约29.7%/30.5%，这是接口及初载证据，不是持续持握。[实际桥报告](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_bridge_native_v1/completed_run1/report.json)

随后固定16组控制曲线、双dt、同一exact/probe030起点，每episode最多20个20ms区间。F0为零action，F1–15用预声明seed/phase产生有界曲线，实际EMA/绝对ctrl逐字节核对，没有clamp或额外物体辅助。F0–11 TRAIN、F12–15 MONITOR，两个dt始终同family。相同初态是已知lineage，不称未见场景。

实际32个episode、205个完整H、19712个积分步，153 TRAIN、52 MONITOR；32条轨迹全部在第6–7个控制区间因POST-H失持停止。无partial、无精确input/action跨分区别名。TRAIN的nonpad/table峰值均0；MONITOR有2行nonpad>0.1N，峰2.389N，table为0。所有失败均保留，不据MONITOR修改reward或筛family。原始PRE contact、POST state、全部300值receipt、decision、memory及observation在进入通用replay前已经持久化。[采集保存核验12/12](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_task_acquisition_v1/saved_review_v1/report.json)

MDP终止以控制边界POST-H的COM>50mm或角度>45°定义，Reader每native-tick安全停止另列；采集0.4s预算截断可bootstrap，失持吸收终止不能bootstrap。32个canonical absorbing标签与旧桥原始`replay__done=0`分开保存，trainer采用canonical标签。它们都不替代最终原逐native-tick的两秒物理评价。

## 实际标签与数据角色

从全部205条实际raw trace导出22对source-BODY、moving-COM冲量，meanFn、边际coverage、独立三指联合coverage和178物理监督量。按实际geom顺序解码frame.T和符号、力臂项与direct torque各一次；观察到的线/角动量离散缺陷约1.97e−7N·s/3.29e−7N·m·s RMS，仍保留非零。三pad联合coverage均值4.58%，不能由边际coverage乘积代替。

新的153条TRAIN和旧原TRAIN684参与WM拟合；52条MONITOR仅固定WM损失监测。历史新12族、原CHECK/VAL/TEST和旧CF side目标均不进入此次拟合。新数据H2有真实连续后继的TRAIN窗口129条；本次Actor从全部153个已知、完整、prealive当前状态开始想象H2，不将这些起点冒称153个两步原生真值窗口。[标签与训练就绪核验10/10](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_replay_dataset_v1/saved_review_v1/report.json)

## 新学习实验的固定配方

seed1733，RTX5090/cuda:0、CPU4线程，NN Float32、几何与动量Float64，AMP/TF32关闭。WM以固定5000模型权重启动，但新建AdamW，不恢复旧优化器；旧input/output normalizer冻结。WM损失尺度按新TRAIN与旧684的50/50总体分布计算，仍用原单位floors。没有静默将上轮5000配方与本次适配算成同一实验，也没有继续旧CF监督。

先进行100次shared-WM适配；之后每个paired round进行1次共同WM监督更新，两臂各1次Critic真实TD、1次Actor想象优化。WM batch8=4新+4旧；两臂共享同一WM函数、相同真实batch、初始网络和TRAIN norm曝光。Actor batch8/H2，每臂4个microbatch2累计梯度，明确计4次backward；不是单次大batch显存/性能结论。Actor/WM/Critic学习率分别1e−4/1e−4/3e−4，clipnorm1，gamma0.99、targetCritic EMA0.01。

A1使用物体跟踪及原动作/关节项；A2增加同材料点、同模板BODY法向、1mm proximity权重的512方向工程接触支持代理。它仍不是视频测力、实际接触/最近法向或严格摩擦可行性证书。Native原4边pyramidal contact与代理8边设计锥的区别继续保留。

这是共享WM的offline matched pilot：真实replay已经采集，首次训练过程不做online native采样。完整policy-native采集闭环、多轨迹摊销和方法收益仍待验证。原模型在新状态的旋转弱项及虚假动作敏感性没有因接好Actor而自动解决。[冻结配方与源码说明](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/README.md)

{{FIRST100_RESULT}}

## 后续策略物理评测的循环已准备

新评测kernel在冻结v3两秒native循环上只增加显式50Hz合法ctrl回调、回调前后300值state只读检查和逐PRE实际ctrl日志；原力读取、POST观察、逐native-tick失持终止、全部物理门及双dt判据不变。零action回调的两dt实际1171步与原GN固定控制所有保存trace数组逐字节相同，资格12/12；两条仍分别在0.12156/0.12219秒失持，不能写成两秒通过。[回调原生等价检查](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_policy_evaluation_v1/zero_qualified_run2/report.json)

第一次回调接线误将命名state转换模块当作Reader的`actual_state`实现，AttributeError发生在首次reset forward之后、首次积分之前。v1源/协议/失败/terminal保留；v2仅改用Reader相同的transition codec及新输出路径，未改模型、门或控制映射。失败时没有序列化native计数，其零积分范围依据保存的调用栈和代码阶段，不冒计数receipt。

尚未执行学习策略的原生两秒评测，原16个三指动态失败和原两指结果均不改。未来需绑定完整训练checkpoint与Actor norm，分别报告真实指定pad/other/table、自碰撞、穿透、关节、尾速度、对象跟踪和双dt一致性。不能把zero-action kernel资格、训练健康或想象return作为策略收益。

## 恢复后的工作与完整Goal

恢复本次配对checkpoint，沿相同固定预算完成1000 paired rounds，再进行统一无辅助原生评价和旧/新分布WM响应检查；保留所有模式、种子、失败及成本，不根据此次训练损失选择方法。继续接通每臂同native预算的policy采集→共享池→WM/Actor更新，形成真正online闭环，而非止于offline pilot。

源重力注册、名义碰撞资产及质量/COM/惯量/摩擦的校准和鲁棒性、更多独立ego、多轨迹共享适配成本、完整61帧执行及真机观测/运控契约继续保留。真实ego输入目前仅标注pose运动意图；未完成RGB重建→完整手部重定向，也未证明稳定CHORD收益。按用户既定条件，新实验首健康checkpoint与ETA具备并归档后暂停Goal。
