# A2 完整2000实测与 A1L 首检查点交接（2026-11-05）

上一Goal turn属于实际progress：完成302 LP、三方法短训/配对/回放、新A2首100与ETA。本轮恢复后先核原worker/watchdog和fresh四PID CIM，无重启A2；A2已于北京时间04:20:44完成，随后对预声明固定final2000执行一次完整3秒物理评价。

## A2：完整训练工程合格，物理任务失败

A2 fixedfinal2000健康196/196、保存生命周期30/30通过，source498+ZIP、原实际8field、controller与固定预算一致；648192真实转移、WM4000/failure4000/Actor1800/critic7200。Prior1800个policy scope、14400次forward/dt应用，active1821808/inactive21392。最终容量3326次观测/3303次pre-reset，overflow/unknown为零。固定100..2000及final共21检查点按声明次数保存，未按质量择优。四级进程均已独立CIM核终态不在。

单次完整评价150controls/600solver/601states，Actor与normalizer不变、无优化器更新、无物体辅助。保存独审188/188复核全部原始13门槛、source505+ZIP及OS身份。任务仅7/13通过，失败项为位置、轴向、穿透、非手接触、目标速率、掉落。

| A2 检查点 | 门槛通过 | 位置151最大误差 mm | 轴向151最大误差 ° | 最大穿透 mm | 目标最大速率 rad/s | 掉落 |
|---|---:|---:|---:|---:|---:|---|
| final40 | 7/13 | 619.6770 | 140.2145 | 17.7329 | 15.2245 | 是 |
| fixedfinal2000 | 7/13 | 553.9911 | 150.4940 | 28.4029 | 17.0143 | 是 |

这不是三方法完整同预算效应对照：schema2 A1L/P_ONLY完整2000尚未同时完成，旧schema1 A1L不能代替新配对基线。当前A2没有任务合格或先验收益证据。静态材料锥H123/H12345的151/151条件分解也不能替代真实接触、连续动力学、force/actuator或物理梯度资格。独立ego N仍为0，固定参考仍knownrough。

A2实际worker wall 2183.829s；首100 ETA点预测比实际完成晚24.751566s，此项只描述本次估时，不宣称泛化准确度。

## A1L：按既有同预算配置启动，首100通过

Root只启动一次A1L primary，原四PID链为29736→34456→36500→16096。首次启动记录脚本中我将A2文本全局替换为A1L，意外把SHA256改成SHA1L56；错误发生在Start-Process之后、写launch record之前。原脚本与错误保留，随后按fresh OS命令、创建时间和父子链恢复原launch记录，未重复启动或改动训练进程。新独立实时报告再次严格核四PID与冻结worker/config/source，37/37通过。后续新增参数化launcher只静态解析通过，方法作为参数、预先计算源哈希，尚未用于启动任何进程。

固定model100健康98/98，40192真实转移/101collect/100metrics，WM200/failure200，Actor、Critic、prior更新均零，final_scope=False。A1L无prior实例/JSONL；原始四模块参数与schema2三臂smoke和A2 primary逐参数配对一致。首100只证明WM-only工程前缀，不代表政策/最终容量或任务合格。

以model100实际mtime 2026-11-05T05:45:03.638098+00:00 为锚，预计北京时间 **2026-11-05T14:12:24.986017+08:00** 完成，启发式范围 **2026-11-05T14:06:40.894895+08:00–2026-11-05T14:21:11.122700+08:00**。使用本次末50步平均0.631115s/iter与同A1L smoke policy去前三次平均0.234076s，无native局部timer起点混用。范围不是置信区间或保证。

按用户已授权条件，保存本地交接后调用Goal暂停工具；状态以工具返回为准。原A1L后台训练继续由7100/7200s watchdog管理；P_ONLY尚未启动或排队。恢复后先核同一原进程/已有终态，固定final2000健康+完整3秒评价，然后执行冻结P_ONLY预算与配对；不得因观察断开而重复启动。

[A2 final健康196](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_primary_A2_final2000_health_v1/report.json) · [A2 保存物理独审188](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A2_primary2000_policy_saved_review_v1/report.json) · [A2 生命周期30](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A2_primary2000_saved_lifecycle_v1/report.json) · [A1L checkpoint100健康98](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_primary_A1L_ckpt100_health_v1/report.json) · [A1L 实时身份与ETA](D:/workspace/supermimic-full151-observation-study/outputs/full151_v3_A1L_primary100_live_pause_v1/report.json)

研究仍未完成：同预算三方法比较、同Warp物理控制/接触梯度诊断、可实现意图/参考的几何-力幅-执行器-连续动态资格、多轨迹共享及真实ego验证待完成；真机子版本及观测/控制契约在部署前核对。

## 同Warp梯度诊断：本轮只完成接口审阅与草案

只读ABI检查25/25，未加载Torch/WM/Env/物理。原105 loader源码采用显式method/health/config/source绑定，未发现A1L或schema1硬编码；新A2 WM实际strict load仍未执行。新105事务inventory、solver warmstart与长时钟恢复、三次baseline重复以及物理容量资格均PENDING。

草案拟定A0零残差与A2归档action的两个fresh donor，k=0/49/99/149共8状态，3baseline+4方向×2eps×2sign，共152分支；连同donor为1808步预算提案。直接扰动经验证的actual绝对控制目标，保持原EMA/C2路径一次，比较同一native物体score的有限区间斜率与WM自身AD；不等于Actor全链梯度。k149只比较终态响应，不增加u150或policy bootstrap。HO force/wrench与self15分开标时相；self15不是HO力head，该物体score控制诊断也不能替代接触扳手目标或真实HO力梯度资格。Root已阅读草案，尚未冻结或运行此实验。

[同Warp草案](D:/workspace/supermimic-full151-observation-study/protocols/full151_same_warp_control_gradient_proposal_v1.json) · [契约与待补接口](D:/workspace/supermimic-full151-observation-study/docs/full151_same_warp_control_gradient_proposal_v1.md) · [只读ABI准备25](D:/workspace/supermimic-full151-observation-study/outputs/full151105_gradient_probe_readonly_preparation_v1/report.json)
