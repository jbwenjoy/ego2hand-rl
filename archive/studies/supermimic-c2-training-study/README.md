# C2_020 learning-controller extension

本目录把固定 20 mrad 输入内收同时接入 SuperMimic 的真实采样与想象控制。原始仓库、既有训练结果、参考、物理参数、奖励和 WM loss 保持原样。当前完成工程验证；正式种子 4/5/6 尚未运行。

真实路径在原 `process_actions` 完成后投影绝对 `_targets`；初始化不投影，随后四个 5 ms 子步沿用同一个目标。想象路径先调用原 `PenMDP.control_target`，保留首返回值 EMA residual，仅投影第二返回值 actual u 并更新 `frozen_target`。两条路径使用同一 FP32 函数和冻结的 hard-limit/actuator-ctrl 交集。原 collector 的 replay.u、WM 输入及 PD torque 继续使用 actual u；c/c1/c2 保持 residual 含义，没有新增 rate memory。

三臂为 A1L、P_ONLY、A2。A1L 不构造 prior；后两者复用冻结的几何模块与目标，分别采用 `-.1*P` 和 `phase_weight*support-.1*P`，只在 policy imagination 中外乘 dt=.02 一次。原 15 维 contact head 仍是手指自碰监督。几何 prior 保持 eager，原 native 子模块保持既有 compile/AMP 设置；这不是整个 prior 的 Inductor 精度资格。

seed0 三次实际短训练均完成：每臂 10 iterations、128 env、WM 预训 2 iterations、WM/failure-head 各 20 updates、actor 8、critic 32、11,392 real transitions，实际报告 horizon 8。每臂 11 次 collect 的全 batch 投影、cache、实际 ctrl、replay.u、residual 前后及 replay.c 检查全部一致，且 finite/位于内收域。原生 CPU fixture、CUDA 2 env ×16 组命令处理及 FP32/AMP 检查也通过；CUDA 命令处理检查本身没有物理 rollout，真实采样与学习路径由三次短训练覆盖。

短训练只验证工程链路，不证明新策略的任务质量、动力学梯度、连续状态安全或 ego/hardware 泛化。正式设计是每方法三个新训练种子，并对各 checkpoint 重复评估；当前没有正式执行授权。checkpoint 使用独立格式 103，必须由显式 C2 工具读取。旧 load、model-only 和 play 对三个真实文件的九路调用均拒绝加载。当前入口仅支持从头训练，不能声称精确 resume。

`scripts/train_controlled_pen.py --config ...` 默认仅 CPU 配置预检；添加 `--run` 才执行有界 GPU 训练。`scripts/run_engineering_smokes.py` 默认也是 CPU 预检，其已执行的唯一三臂工程队列使用共享 GPU 锁、native 600 s / child 720 s / whole 2300 s 调度预算、无重试。不要把工程配置用于正式预算。

- [第一阶段结论](D:/workspace/supermimic-c2-training-study/outputs/engineering_phase1_conclusion.json)
- [成本记录](D:/workspace/supermimic-c2-training-study/outputs/engineering_cost_ledger_v1.json)：whole watcher 156.546 s；child wall 合计 156.233 s。各 native entry 报告的墙钟与 phase 时间包含重叠范围，不应相加成 GPU compute。
- [源码与最小差分](D:/workspace/supermimic-c2-training-study/outputs/engineering_cpu_preparation/source_contract.json)
- [实际旧 CLI 拒载](D:/workspace/supermimic-c2-training-study/outputs/legacy_checkpoint_rejection_cpu_v1/checks.json)
- [三次独立 103 健康审计](D:/workspace/supermimic-c2-training-study/outputs/engineering_health_v2)

每次运行归档 `source/c2_controller` 中的 adapter、pure projection、P_ONLY wrapper、实际入口与原 8237 入口，并在 inventory/checkpoint 中绑定身份。actualu JSONL 保留每 collect 的全 batch 摘要；trace 固定取第 1/2/3 次及每 100 次 collect 的末尾最多四步、前四个环境。模块初始参数 fingerprint 已保存，三臂一致。Torch peak allocated 分别约 .539 / 2.194 / 7.622 GiB；完整设备 VRAM、活动 CUDA kernel 时间和未记录的准备时间不作推断。
