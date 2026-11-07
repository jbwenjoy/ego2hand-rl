# 当前 104 的同 Warp 一步物理 FD 提案

当前有效草案：[protocol.json](D:/workspace/supermimic-rough-prefix-learning-study/proposals/current104_samewarp_physical_fd_v2/protocol.json)，SHA256 `ce9a9575b6aa7dc9b5b535bfe1cf7ff5be4db6065dc122ba4860561d40096e09`。状态为 `DRAFT_NOT_AUTHORIZED_PHYSICS_PENDING_CPU_DIRECTIONS_PREPARED`，GPU 授权为 false。物理 worker、监督器和 CPU 合同尚待独立实现审阅及 Root 另行授权；本提案生成没有调用 WM、Env、物理、碰撞或优化器。

V1 和 [mandatory correction](D:/workspace/supermimic-rough-prefix-learning-study/proposals/current104_samewarp_physical_fd_v1/CORRECTION_v1.json) 保留原字节。未来执行配置应绑定整合 V2，不能单独消费 V1 的 `startup_forwards=4`。该值误把逻辑初始化次数当作全部 forward 次数；四次 donor reset 中，原 Manager.reset 和粗参考初态写入后各有一次 forward，因此至少 8 次 reset startup forward。构造、graph warmup 的任何 step/forward 另须实际计数。

提案绑定现有三个 final2000 format104 WM，以及 native A0/A1L/P_ONLY/A2 四条已保存轨迹。每条都用 phase 0/4/8/12，共 16 个预定快照。新 donor 在同一 native Manager 实例重放原始 15 个 action；A0 原日志没有 action 字段，明确采用原 A0 的零动作设计，不能伪称已有该观测。新 donor 是新增 probe 成本，其状态与旧记录的漂移独立报告，不替换旧物理成绩，也不反复重放到“更像原轨迹”。

## 预算上限

| 阶段 | 控制转移 | 5ms 积分子步 | 原生 interval 后 forward |
|---|---:|---:|---:|
| 四条 donor，各完整 15 intervals | 60 | 240 | 60 |
| 16 快照 × 3 baseline | 48 | 192 | 48 |
| 16 × 8 方向 × 2 eps × 正负 | 512 | 2048 | 512 |
| 计划 donor + branch 合计 | 620 | 2480 | 620 |

这是检查初态后的 scheduled 工作量上限，合计模拟时间 12.4s；不是 whole-process 的精确 step/forward 总数。构造、reset、编译/graph warmup 和异常清理均单列时间及调用数。物理分支对三个 WM 共用，不乘以三。任一工程门禁失败时保留 partial/NOT_RUN，实际数可能较低；不自动重试，不按任务质量挑选样本。拟议 soft220/hard240 秒只是新的工程上限，尚未授权。

## 复现与干预

模型按 nominal MJB → 实际 8 字段 → `set_const_0(restore=True)` → 双读回 → 初始化的顺序建立。快照取在 phase k 的 action processing 之前，含 `mjSTATE_INTEGRATION` 与原生 `WarpTaskSnapshot` 的完整同实例设备阵列。INTEGRATION 含 HISTORY 和 warmstart，但不含所有 derived/contact/sensor/overflow 状态。原生 snapshot walker 也不自动覆盖函数 closure、NumPy 数组和任意 rough_prefix 自定义类，必须给实际用到的 controller/PEN memory、obs/cache 和外部状态补明确 sidecar、恢复及清单核验；不能用 serializer exact0 代替动力学复现。

每次恢复后以三个预定 baseline 对新 donor 的下一状态核验。q/p/R 分量阈值 2e−5、qd/v/ω 阈值 2e−4、未乘 dt 的物体奖励阈值 2e−5，均明确来自旧 serial296 的数值规则，但旧实验资格不会迁移。完整状态/控制器/RNG 恢复的 bitexact 扩展是新提议。全部 raw 误差保留，不因失败调阈值。旧 Manager microprobe 曾在恢复 exact0 后仍未通过 baseline，因此这道门不能略去。

每个 branch 仍走原 `Env.step(saved_action)`：保留原 action 处理与 residual EMA，只在 processing 之后把 cached executed target 替换为已验证的绝对 u，四个子步持有，然后执行原生 forward/command/sense/obs。没有额外 forward、状态投影、物体辅助或外力。原 policy_probe 的 baseline+EMA/C2 断言会正确拒绝这种干预，未来需新的独立 observer 分开验证“原 EMA 侧车”和“声明的 target override”，不能修改已冻结原 probe 或静默跳过实际 ctrl 核验。

## 方向与分析边界

现有纯 NumPy [准备结果](D:/workspace/supermimic-rough-prefix-learning-study/outputs/current104_physical_directions_CPU_v1/report.json) 已被 V2 精确绑定，不重新生成。种子 2026110401；每个状态八个方向供三个 WM 共用。只有距 C2 内域边界至少 `eps_max + 8*eps32*max(1,|u|,|lo|,|hi|)` 的坐标参与，其他坐标置零后在 FP64 中归一化。选择只依输入边界；不依 WM 梯度或物理响应。空子空间记 NOT_RUN；非空却产生零范数按固定模块 fail closed，不重抽。

这里测的是对称的内部坐标子空间，不是完整 tangent cone；靠边界的单边/face-normal 方向未测。正负目标转 FP32 后逐位保存，出域不 clamp。主 AD 投影使用实际 `d_eff=(u_plus-u_minus)/(2eps)`；nominal direction 的结果同步保存为次列，不能看哪种更吻合再选择。方向秩和有效位移也保留，八行不构成八个独立实验。

主目标是当前 phase k 的原生物体两项奖励 2+8；k+1 只作单列 tracking 解释。平坦、噪声和 epsilon 不稳定结果必须保留，不能计作符号正确。需要的 sign/correlation 仅是局部描述，不设“足够正确”的方法通过门槛，不把单源种子 0 的 16 个依赖快照当成独立演示。

权威 selfcollision15 标签取原 `reward_snapshot.contact`：四步后、Manager forward 前，保留原 scene.update 缓存失效过程。forward 后的 `reader.contact()` 仍可能沿用 lazy cache，不能自动称作新 20ms 真值。保存新 donor 标签及三次 baseline 的一致性；不多数投票、不拿 HO/力阈值替代标签，也不把二值接触的有限差分当连续物理导数。

完整 snapshot 恢复可能把 overflow 数组回滚，容量 OR ledger 必须在 snapshot 遍历外的 host 上维护，在每次 restore 前、branch 后、donor 末和关闭前记录。只看到最终 sticky0 不足以覆盖先前分支。容量 UNKNOWN、overflow、非有限值或恢复失败按工程失败停止并留证。

GPU worker 只采集物理输入与真值，结束并释放 GPU 后，由独立、CUDA 未初始化的 CPU 进程用严格 104 loader 计算三个 WM 的预测和梯度。不能绕过 loader 的 CPU/初始化保护。本次若完成，也只提供当前控制模式的一步、局部、受采样约束的物理方向证据；不直接授予手物力预测、完整接触动力学、full151/shared、跨本体 ego 或硬件资格。
