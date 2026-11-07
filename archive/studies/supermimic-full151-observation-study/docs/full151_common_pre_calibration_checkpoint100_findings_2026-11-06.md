# 共同PRE有限响应校准：首个检查点健康与按约定暂停

2026-11-06。用户要求的新训练首个检查点健康核验及ETA条件已满足，Goal按约定暂停。两臂校准较快，独立首检查点核验期间已各完成1000更新；当前没有后台作业，剩余时间为0。确认响应仍封存，未进行部件收益比较。

## 已执行

- 新相位 `[24,74,124,148]`：discovery1200 episode步/8 captures/0恢复分支，独审77/77；新exact8 source enrollment后，full1808 episode步/152分支/24 baseline全部完成，独审540/540。两次OS身份各12/12，retained session7497与65745均exit0。构造与graph capture API计数另列，不冒1200/1808全局物理调用总数。
- 校准recipe及全部代码在新增受控响应采集前冻结；新数据仅工程核验后封存77个文件。训练allowlist只有旧152 restored transitions/64pairs，排除旧300 donor；两个副本共享固定A2参数、normalizer/host count、xi及数据，无随机采样或新warmup。
- 对照为native52点式损失，方法添加lambda1共同PRE52实际控制差响应项。CPU FP32与F64后处理的边界、native quaternion-log chart严格margin、AdamW和更新数均保持冻结定义。新component artifact独立，不改旧105 checkpoint/health或119f wrapper。
- 两臂各1000次完整152-row前向、反向及AdamW更新，另有1次固定模型同批验证前向。实际worker总时间33.1690023秒，至checkpoint1000的fit计时24.0483295秒。strict loader读取原checkpoint2次，完整正常路径被动MJB3次为源码计数；独立own checkpoint读回另计，不充相同预算或学习加速证据。

## 首个100更新检查点

独立序列化读回 **29/29 PASS**：实际读回initial及两臂after100文件，核每臂14个Adam参数状态的step均为100、参数/矩有限、全部norm buffer/count/host配置和xi/s52未变、数据及64实际控制差一致、原始RNG一致，以及iteration1/100原生52中间量的损失后处理算术。iteration100损失证据是after99/preupdate；checkpoint100参数是after100，没有据此假称重新前向checkpoint100。

首次reader27/29已保留：仅两条RNG检查错误地要求loaded对象重新pickle摘要等于原内存对象摘要。Root原值诊断8/8和独立forensic均确认Python状态、NumPy MT header/dtype/vector及TorchCPU向量相同。新reader只改为逐字段/字节比较，原数据、训练、其余27门及阈值均未改。旧pickle摘要另作来源诊断，不确定其内部表示差异的唯一成因。

原实际梯度finite和native Log本身仍部分依赖源码/日志证据；本次不重跑模型或Log、不授予真实物理Jacobian、HO力梯度或旧native105健康资格。独立审计只覆盖100更新；final1000已完成保存但其独立序列化复核留待恢复Goal后执行。

## ETA与暂停

首个100更新时，fit计时2.5061631秒，剩余900更新/臂的线性估计为22.5554679秒，规划范围11.2777–40.5998秒，另保留20秒未测终结开销；这是启发式范围，不是CI或GPU策略训练ETA。随后retained session30345正常exit0，总worker时间约33秒，故当前ETA剩余0秒。暂停时无活跃训练或物理作业。

## 恢复后待办及限制

先独立读回两臂final1000，再按原先锁定的评价式解封全部64新pair与152端点，统一比较normalized响应MSE、pointwise loss及原单位块误差，不筛contact/stable/error、不调参或选checkpoint。当前没有确认数据的预测或收益指标。

新增的是同缓存谱系中预定相位的受控扰动响应，nominal状态已见，seed0指父模型来源；此完整batch校准无新随机采样，不伪称原始起始RNG重新seed0。新分支没有独立直接捕获PRE58，不重算或授予其全58资格；保持源恢复、原始primitive和baseline门所支持的条件范围，whole graph scratch仍UNKNOWN。

原参考几何及三臂完整任务FAIL、旧5/9方向结果、代理漏检、warmup分歧、单谱系/真实egoN0均保留。物理可行参考修复、CHORD任务对照、多轨迹共享及真实ego/硬件契约仍待推进。

## 证据入口

- [首个检查点独立29项报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_common_pre_checkpoint100_saved_independent_v2/report.json)
- [按用户条件暂停的核验记录](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_common_pre_user_pause_receipt_v1.json)
- [两臂校准终态](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_common_pre_calibration_CPU_v1/result.json)
- [封存确认数据](D:/workspace/supermimic-full151-observation-study/outputs/full151105_fresh_calibration_response_seal_v1/sealed_manifest.json)
- [冻结recipe](D:/workspace/supermimic-full151-observation-study/protocols/full151105_WM_common_pre_calibration_recipe_v1.json)与[实际run协议](D:/workspace/supermimic-full151-observation-study/protocols/full151105_WM_common_pre_calibration_CPU_run_v1.json)
- [更新成本账v6](D:/workspace/supermimic-full151-observation-study/outputs/full151105_matched_replay_dense_compute_ledger_v6.json)
