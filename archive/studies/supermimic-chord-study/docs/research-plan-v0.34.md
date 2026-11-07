# 研究方案 v0.34：同一 GPU 配对实验恢复与最终策略评测准备

2026-11-07，承接 [v0.33](research-plan-v0.33.md) 和完整 Goal。用户恢复后，沿相同数据、源码、配方及 paired100 检查点继续固定 1000 轮实验。完整目标仍未完成。本轮属于 progress；训练尚未结束。

## 已实际恢复的 GPU 训练

实际 worker PID 28104，supervisor PID 24420，启动于北京时间 19:17:48。设备为 RTX 5090 / `cuda:0`。这是同一 offline shared-WM A1/A2 实验的 100→1000 恢复，跳过已完成的 common100，完整恢复两臂、WM、五个优化器、采样器和 CPU/CUDA RNG。原训练源码 SHA `a1bcf9f820e00d7da4ac515dd81d45d39a2c59abd07c85319d1346db384b2439` 未修改。恢复源码审阅 14/14。

19:33 左右已到 paired265 / shared-WM365，最近一轮约 5.55 秒；整机 GPU 瞬时利用率 43%，显存 4529 / 32607 MiB（13.9%）。该显存包含桌面和 CUDA 上下文，不等于训练 Tensor allocated。按当时速率剩余约 69 分钟，预计约 20:42，后续检查点健康及 IO 可能增加耗时。即时进度及不可变日志前缀见本轮里程碑快照。

`CPUFalse` 表示未执行后端的协议/源码检查，不表示 actual worker 在 CPU 训练。训练实际协议是 `protocol_CPU_GPU_True_resume1.json`，设备为 `cuda:0`。本次新优化器实验不是旧 CPU WM5000 训练的继续；旧 WM 仅是固定 warm start。

本轮沿原固定 batch8、H2、microbatch2×4、Float32 NN / Float64 几何与物理量、AMP/TF32 关闭继续。保持实验定义，不为提高 GPU 百分比临时更改 batch、精度或优化器。低占用的具体瓶颈仍需后续 profiling，不能从单次利用率确定。

本次恢复已经获得用户授权；不在历史 paired100 或中途 paired500 再次触发同一首检查点暂停。未来启动另一个新学习实验，仍遵守首健康检查点与 ETA 后暂停的既定规则。

## 最终 Actor 原生评测已准备，尚未执行

[评测 v2](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_policy_evaluation_v1/evaluate_learned_actors_v2.py) 与 CPUFalse 协议已准备，独立源码审阅 14/14。执行前四项最终输入仍为空，必须等待 paired1000、WM1100、health1000 通过及 supervisor terminal exit0，再绑定实际 resume protocol；不能以训练日志、paired100 或 paired500 代替。

两臂 × 两 dt 共四个原生案例，最多 38400 native steps / 400 Actor forward。恢复 Actor 及其冻结 normalizer，使用相同 F64 控制历史和 50Hz 控制回调；实际读取物理接触并保留原两秒逐积分步门、旧两 pad 并行门及无辅助条件。直接调用冻结 `dt_pair_summary`，要求两 dt 都完成两秒，最终 COM 差 ≤2mm、姿态差 ≤2°。这些都是实际运行时判据，尚无策略通过结果。

物体跟踪读取整数 `k*n` 的实际 POST 位姿，与 101 点命令比较 COM / 完整 SO(3)。缺失尾段保留 NaN，初始 k0 不计入执行误差。固定不动物体仅作命令空间的描述性参照，不冒称仿真基线。纯 NumPy 合成数据核验 26/26，不包含 Torch、模型或原生执行。未执行的 v1、修订和早期测试接线错误均保留。

## 下一阶段采集到 replay 的接口

[在线采集源码交付](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_online_collection_v1/SOURCE_DELIVERY_v1.json) 已冻结：Actor→真实 Bridge→持久化 raw→canonical labels→common replay API。纯源码/元数据契约 8/8，独立窄源码复核未发现阻断；本轮未实际采集、加载最终 Actor 或优化。

预先固定两臂 × 四个控制扰动 profile × 两 dt，每条最多 20H，共最多 320H / 30720 native steps。两臂预算上限一致；失持提前停止后的实际样本数可以不同，不补采或筛选。保留原吸收终止、采集截断、Reader 安全停止、partial 和 episode-safe H2 区分。final binder 必须绑定真正完成该模型的 resume 协议，不能遗留 initial run1 协议。

这些接口尚不构成完整 online SuperMimic 学习循环。最终固定模型评测后，仍须实现独立版本的 collection/replay→WM/Actor/Critic 更新消费者，明确新阶段配方和暂停规则。

## 后续执行与完整范围

先观察现有 worker，确认 paired500/1000 及健康产物，完成 fixed1000 原生评测。若工具 session 失效，先查 OS PID、创建时间和日志，不重复启动同一训练。不因 loss 下降宣称持握或 CHORD 收益。

随后继续在线更新闭环、旧/新分布 WM 动作响应、真实资产/质量/COM/惯量/摩擦的校准及鲁棒性、源重力注册、更多独立 ego、多轨迹共享成本、完整61帧机器人执行以及硬件观测与运控契约。目前真实 ego 来源仍主要是标注物体 pose 运动意图，尚非 RGB 手部重建或完整人手重定向；未有成功两秒持握或稳定 CHORD 收益的新证据。

本轮不可变状态入口：[里程碑 v15](D:/workspace/ego-data-pilot-2026-11-07/outputs/goal_extension_tripod_GPU_resume_milestone_v15.json)。Goal 保持 active；历史 v14 的暂停记录不覆盖本次用户恢复。
