# 状态可行性与后续控制器条件：独立只读审阅

2026-11-02。仅审阅本地源代码及已有诊断；未修改冻结实验源、运行 GPU 或重新步进物理。以下后续设计尚未实现或验证。

## 已核实事实

| 项目 | 实际代码与含义 |
|---|---|
| WM 的 q 分量 `error_floor` | [networks.py:38](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/networks.py:38) 的前20项为 `0.003`。前20维对应关节角：见 [state.py:11](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py:11) 与 [state.py:26](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py:26)。单位为 rad。 |
| 该数值的用途 | [runner.py:535](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:535) 使用 `std=max(delta_norm.std(),error_floor)`，随后计算 `(difference/std)^2`。它是 **loss 归一化分母的下界**；实际分母可能更大。 |
| 不是模型精度下界 | 前向预测使用 `delta_norm.std()`，见 [networks.py:58](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/networks.py:58)。`error_floor` 没有把预测量化为3mrad，也没有形成误差死区。floor 生效时，1mrad残差的单分量归一化平方误差仍为 `1/9`，之后才做维度平均和权重缩放；小于3mrad的非零残差仍有连续非零损失梯度。不能写“模型无法区分1mrad”。 |
| 冻结验收 margin | 实际 [reference.json:121](D:/workspace/supermimic-chord-study/outputs/pilot_rotating_reference_v2/reference.json:121) 为 `joint_limit_margin_rad=0.001`，协议 SHA 为 `41e7bdd97d07a0f4cac06cefc84d59548407d29332f6836fc5094e5e2f4fc2a3`。验收对实际 q 与执行 target 都检查 MJB hard range 加此 margin，见 [acceptance.py:184](D:/workspace/supermimic-chord-study/src/chord_study/acceptance.py:184)。 |
| 原生 actor 奖励 | [PenMDP.rewards:143](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:143) 包含物体/连杆/关节模仿、动作变化、torque、持续负载及自碰项。`joint_pos` 依赖 q 实际/预测值，但惩罚的是相对参考误差；joint range 仅作归一化。没有直接针对 `q<lower` 或 `q>upper` 的独立 barrier/越界项。真实环境配置也对应这些项，见 [env_cfgs.py:274](D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/env_cfgs.py:274)。 |
| 现有保护的位置 | 真实执行 target 在 [actions.py:76](D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/mdp/actions.py:76) 及147行限幅；imagination 在 [pen.py:115](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:115) 限幅。它们约束绝对控制目标，不是对物理 q 的投影。WM 的 q 更新同样是 `q+dq`，见 [state.py:20](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py:20)。 |

因此，**训练损失的尺度与最终状态验收尺度不同**，但这不是模型精度不足、奖励缺项或 overshoot 因果关系的证明。已有 [九条诊断](D:/workspace/supermimic-chord-study/outputs/replication_joint_limit_diagnostics_v1/report.json) 证明了“target 在浮点容差内 bounded，而观测 q 仍可越界”的事实。首次采样越界处 qd 指向上界外、记录的 actuator torque 指向内侧；这些日志不能分离惯性、接触反力、软约束求解、控制时序及模型误差各自的贡献。20ms日志也不提供未记录子步极值或硬件安全结论。

若要判断 WM 误差是否相关，需要独立 holdout 的逐关节预测残差、近边界 extrema、跨接触模式/速度条件的尾部误差，以及“预测可行但实际超过1mrad”的漏检率。单个平均 loss 或此归一化常数都不足以替代这些测量；当前九条日志用于提出后续假设，不再冒称未见数据。

## 未来 velocity-aware target filter 的最小新条件契约

1. **独立命名、配置和源快照。** 在新 controller condition 下比较；冻结原任务参考、1mrad验收、完整3s评估及采样/更新预算。先保持原奖励和 WM loss 不变，以隔离 filter；若另加状态 barrier 或改 `error_floor`，再分别命名条件。
2. **只改控制输入。** 从当前 q/qd、同一 phase 的参考、原 EMA residual、encoder-bias映射及已知 actuator 参数，生成候选绝对 target，再过滤得到 executed target。输出仍满足 joint-hard/actuator-ctrl交集、预声明 target-rate 和力矩条件；不得 clamp、重置或覆盖真实/预测 q、qd、物体状态来使验收通过。接触不确定性下的可行域必须校准，不能直接把启发式速度裕量称为安全保证。
3. **real / imagined 共用同一函数。** 建议显式接口为 `filter(q, qd, proposed_u, previous_executed_u, gains, limits, dt, filter_memory)`。现有 [control_target:86](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/pen.py:86) 没有 q/qd 参数，新条件应明确扩展接口。真实路径输入当前测量，imagination 输入当前预测 x；不能在 imagination 内读取 live simulator 的 q/qd 或未来接触作为答案。同输入输出一致，并不等于预测状态与真实状态一致。
4. **保持 controller memory 的语义。** 当前 `control_target` 返回 `(EMA residual, absolute target)`；[runner.py:723](D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/runner.py:723) 的 `target` 变量实际是 residual，`u` 才是送 WM/PD 的绝对控制。`c1` 继续保存 residual；记录/缓存的 executed/frozen target 必须为过滤后的 u。若增加 anti-windup 或历史速度状态，显式定义初始化、每步更新、reset、replay和imagination递推，并冻结其布局，不能悄悄把 c1 改成绝对 target。
5. **同一动作贯穿物理与学习。** 真实 actuator、记录的 replay `u`、WM输入、PD torque估计及下一步 filter memory 都使用实际过滤后的 u。新控制状态必须进入 replay/memory 契约；不得只在真实路径过滤而让 imagination 优化未过滤动作。若使用不同的反向 surrogate/不可导投影，单独披露并检验，不能把它称为精确梯度。
6. **不可行时诚实处理。** 向边界运动或接触负载可能使当前状态已不满足候选内域。预声明确定性的降级/制动规则及 `infeasible` 标记，不伪造状态、不把 episode 提前重置为成功。独立记录 proposed/filtered u、q/qd、激活约束、力矩与首次/峰值越界；`target[k]` 属于 `[k-1,k]` 区间，不能误作下一步动作。
7. **先验证再比较。** CPU同输入批量/边界/重置与无额外RNG检查；再做实际后端的 real/imagination target及memory一致性、AMP/数值与梯度检查。最后按预声明种子保留所有真实3s结果，分别报告 task、actual-state constraints、接触、负载和成本。鲁棒裕量/动态假设必须来自独立校准，不能依据本轮通过数倒调验收门槛。

## 与当前 P_ONLY 的隔离

当前三臂识别保持：A1L=`(support=0, penetration=0)`，P_ONLY=`(0,0.1)`，A2=`(1,0.1)`。P_ONLY 的唯一干预仍是把 support 系数设为0，保持同一 penetration 公式、3mm allowance、1mm scale、dt、phase规则、控制器与训练预算。现有 penalty 是手表面相对物体的采样几何深度项，见 [contact_objective.py:186](D:/workspace/supermimic-chord-study/src/chord_study/contact_objective.py:186)，不是关节状态可行性 barrier。

**不把 velocity filter、关节 barrier、anti-windup、WM精度权重变化或新 margin 混入本轮 P_ONLY。** 当前 A2−P_ONLY 只能识别原 penetration 背景下 support 的增量；未来 controller condition 必须另行比较。

## 本次源码指纹

- `networks.py`: `7fbec4de7aa92603565aa24e85d559f4b8c9dc309aae877d3fdd2981402e350a`
- `runner.py`: `b19d9ac80e23e4f55b2eb64e458c05516fc5dd4ffb13e1a3050ce9fea2669d46`
- `pen.py`: `c02906c409aedab59aaf68a5ef80f9dbcea52da10e72137091df81858262819a`
- `acceptance.py`: `b7afba11a1bc0b97425b2d1987615bcbe21acb74467759a0b822f932868a9bb4`
