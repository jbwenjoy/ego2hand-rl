# 下一步：先构造连续 prefix16，再资格化短动态跟踪

**当前只是一份下一轮 CPU 任务草案，READY=PENDING；不授权优化、GPU 或训练。**

下一轮最小工作是生成一条新的 16 帧关节参考，并把初态、参考和实际控制目标分别导出。它通过全部 CPU 前置合同后，才考虑另行审阅两条各一次的 0.3 秒动态实验。不是继续用静止保持替代运动目标。

## 为什么需要新轨迹

已完成的 H123 预载 rest 只证明：在该几何 q0、设计零速度和固定 u 下，本次 0.3 秒静止门槛通过。它不资格化非零速度初态，也不证明视频运动或完整轨迹可执行。H123 选择时已有几何和 QREF 结果可见；后续比较仍是顺序探索，不称盲选或统计复制。

不能把 warm q0 接到旧 539d 的后续 q。只读反例见 `splice_counterexample.json`：warm q0 到旧 frame1 最大跳变 **1.10785854 rad**，20ms 隐含 q-goal 速度 **55.39292695 rad/s**。这是参考不连续的反例，不是实测 actual-u 速度；q 和 u 仍须分开。

复算反例：

```powershell
& D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe D:/workspace/supermimic-observation-dynamic-wrench-study/scripts/diagnose_forbidden_q_splice_cpu.py
```

## 下一轮唯一 CPU 候选

1. 固定 canonical H123 baseline 的几何 q0；**16 帧初始猜测全部重复 q0**。不读取旧 539d q1..15 作为输出、暖启动或正则目标。
2. 从允许的观测和既有固定掌根映射重建前 16 帧骨方向目标。物体参考仍为原 mapped pose，腕为原 MJB 默认 mocap；不优化物体或腕。
3. 冻结 H123 的手部材料点及物体局部表面锚点。把物体局部锚点随目标物体位姿传输，形成一个事前固定的接触意图假设。它不是 contact label，不按后续状态重选手指或最近顶点，也不禁止额外稳定接触。
4. 对 q1..15 的 300 个变量作一次整段联合最小二乘；使用稀疏解析块 Jacobian。保留骨方向、原生手物/手自碰浅穿透、位置/法向锚点、时间差分与曲率正则。数值、权重和归一化在 `protocol.json` 中明确。
5. 一次求解最多 80 次函数评估，软上限 110 秒、进程上限 120 秒；不换初值、权重、平滑方法或失败分支。不收敛、难接触分支和部分结果均保存，不自动补跑。

这是一条**全新的 prefix16**，不是修改旧 151q。之后若扩展到 3 秒，也需重新构造全段，而不是把新 prefix 接回旧 tail。

## q、qdot、u 和初态分开

| 字段 | 含义与来源 |
|---|---|
| `q_goal` | 新 prefix 的运动学目标，不是 actual q 或控制命令 |
| `native_FD_goal_qd` | 从新 q_goal 导出的设计速度，不是视频/机器人测量 |
| `q_init_requested`, `qd_init_requested` | 为新实验设计并写入的初态；各自明确 FP32 转换 |
| 物体初始 world v、world ω | 已冻结 rawFD 设计；不是视频测量。native 自由关节角速度须用 `R_init.T @ ω_world` |
| `u_proposed`, `u_executed` | 已知 PD 模型下生成的绝对目标，以及经同一 C2.020 后的实际目标 |
| `init_measured` | 真实物理 startup 之后的读回；现在必须为空/PENDING |

物体 body_ipos 当前为 0，线速度等于 COM world 速度。通用转换仍需显式处理 `v_body_origin = v_COM − ω × (R body_ipos)`。

参考应有连续的 q/qdot 边界。物体位置采用以冻结 rawFD v/a 为端点约束的分段 quintic Hermite；旋转采用 SO(3) quintic Hermite，在 proper rotation 上匹配姿态、world ω 和 world α，不能逐 quaternion 分量滤波。新 q-goal 的导数仅来自它自己的 16 帧，再导出连续曲线。**这些新曲线/导出器尚未实现，必须独立数值检验后才能使用。**

旧 window9/poly3 只保留为敏感性分析，不在失败后换作主设计。0.3 秒末不伪设零速度；停止记录不等于物体停止。

## 实际 PD 阻尼与控制合同

实际模型为 `τ = kp*u + bias0 + bias1*q_actual + bias2*qd_actual`。非零 qdot 会改变初始力矩，不能把旧静态 u0 的见证直接当动态力矩。

最小控制基线只携带一个固定静态负载先验，并补偿**参考速度**对应的 PD 阻尼：

```text
tau_ff_static0 = kp*u_PDU + bias0 + bias1*q0
u_pre[k] = (tau_ff_static0 - bias0 - bias1*q_goal[k] - bias2*qd_goal[k]) / kp
u_exec[k] = C2_020_FP32(u_pre[k])
```

这不是动态逆解，不求解改变中的惯性/接触受力，也不使用后续 actual q 来重算目标。真实 PD 对 actual q/qd 的反馈仍由原 actuator 执行。必须冻结 FP32 分组运算、记录 C2 夹量，并在参考状态上检查实际 actuator/joint force caps。

CPU 还需证明 NumPy 导出路径与纯 Torch 想象控制代数一致，连同 C2、时间索引、初始记忆和实际 u。此证明**不等于 WM 动力学或接触梯度合格**。本次不加载 actor/WM、没有学习残差或 EMA；旧 format3 WM wrapper 也不是新 format103/registry 的即插即用接口。

## 后续最多两条物理条件——当前均未授权

| 顺序 | 条件 | 控制 |
|---|---|---|
| 1 | H0：matched dynamic init hold | 保持新 `u_exec[0]` |
| 2 | T0：matched dynamic init track | 使用新 15 个 20ms 区间目标 |

两条使用**同一新非零初态**：q0、qd0、物体位姿/world v/world ω，以及同一新 u0。旧 PDU rest 的速度为零，不能替代 H0。每条件最多一次、45 秒 owned 上限；不新增零速条件、不按结果挑选或补跑。

15 个目标覆盖 `[0,.02),...,[.28,.30)`；每目标保持四个 5ms 子步。除初始化 u0 外，只在 step5、9、…、57 前作 14 次 ctrl 更新。不得写 q/qvel/物体/mocap 或补外力。旧 037 rest worker 的恒定目标/零速度断言不能原样冒用；新 scheduler 必须有新源哈希和 CPU 负例合同。

测量继续区分 solver prestate 与 postintegration state。第 j 步力/接触对应 prestate j−1、控制区间 `floor((j−1)/4)`；q/qvel/time 对应 poststate j。命令更新事件单独记录，不用额外 forward 刷新力。

## GPU 前置 gate

现在以下新接口全部 **PENDING**：

1. 新 prefix 候选求解成功，全部帧/连续曲线 q 范围、端点、导数、接触诊断完整；缺失距离为 UNKNOWN，不作零间隙证明。
2. 参考曲线和 source/window/hash 一致；初始接触点相对速度有完整诊断，不伪造无滑动/实测接触置信度。
3. u_schedule 与 C2 real/imag 代数、实际 PD/力限、每 20ms actual target-rate ≤15rad/s 合同通过；两条件新 u0 位级相同且 T0 目标确实不同于 hold。
4. 初始化、参考、控制三类 bundle 独立；初始化来源与实际读回分开。新 worker 的非零速度、target scheduler、forcephase、实际八字段、容量及生命周期合同通过。
5. 所有新源/数据/协议哈希冻结，由 Root 另行审阅授权。任何 gate 失败均保存并停止，不改参数凑通过。

现有 shared registry 已有真实 API，但仅收录新 source `[0,151)` 窗口。本方案属于旧 clip007 `[200,351)` 的派生 prefix，不能迁移到同名 clip 的 registry 身份。需另建 prefix 身份/适配；当前 entry=null。registry 的 `resolve_init_requested` 返回设计值，`init_measured=None`，也不表示 native training/reset 已接通。

## 新短跟踪判定与证据

数值保留：moving-reference position ≤20mm、directed shaft error ≤20°、actual joint excess ≤1mrad、所有类别穿透 ≤3mm、无非手支撑/外力；源时钟容差、61状态/60 solver、实际八字段与末步唯一 sticky overflow=0 不变。actual-u 从“恒定”改为“精确符合冻结 schedule”；明确这是新合同，不是原 rest grade。

同时记录原目标速率15rad/s、load ratio1.01、charge0.8/decay2s、drop阈值，以及完整 SO3/参考进展诊断。0.3秒不能满足或替代原3秒旋转任务的45°/60°/90°运动资格；**只可能资格化这个新短跟踪任务**，不声称完整 motion/spin 成功。

原始 nominal MJB → actual8 copy → put_data → set_const_0(restore=True) → 一次 startup forward；之后仅60次 step/readback。仍采用192 contact、4096 constraint、world1、40s worker/45s watchdog、shared GPU lock、159c bounded IO、精确 CIM 身份。工程/容量不合格为 UNKNOWN，质量失败保留完整60步。

## 路线取舍

- **现在做整段 prefix CPU 构造。** 它处理 warm q0 后的连续目标、接触意图与速度边界。
- 逐帧 previous-q 暖启动只能是一个求解策略；已有时间正则并不证明速度连续，不作当前失败后的自动备用路线。
- PD+学习残差在新参考、初始化和 actual-u 接口物理资格化后再讨论；不复用不合格接触代理作为已验证力预测，也不把旧 actor 的同受控 clip 结果当观测重定向泛化。

当前产物只是 protocol、源码复用清单及可复算的拼接反例。没有新 solver query、优化、物理或训练运行。
