# HO-Cap 纸盒的首次全段物理重放

**2026-11-21，开发序列，名义物理参数。** 在 HO-Cap 的右手拿放示范 `subject_1/20231025_170650` 第 0–140 帧上，Wuji Hand 2 通过接触把自由纸盒从桌面举起、放回并离手。纸盒位置全程 RMS 为 **7.59 mm**，完整旋转 RMS 为 **5.10°**。独立进程从首帧初态仅重放实际发送的 5,600 组手指指令和腕部广义力，141 个端点的机器状态与记录一致。轨迹属于逐条优化的名义仿真结果，真实物品质量和摩擦仍未知。[全段并排视频](../runs/hocap_carton_wrist_peak_visual_v1/source_vs_physics.mp4) · [可独立执行的导出包](../runs/hocap_carton_export_v1/Wuji-HOCap-carton-replay.zip) · [解包重放记录](../runs/hocap_carton_export_v1/report.json)

## 验收结果

在首次通过的轨迹上进一步将二十路手指目标限在 **30 rad/s**。该版本也通过全部门限，位置和完整旋转 RMS 改善至 **6.21 mm、4.27°**，最差帧为 13.70 mm 和 9.94°。抬高 241.2 mm，末帧位置误差 4.54 mm，51 个主要腾空区间均有手部支撑，末尾三个区间均离手。对八个限速档执行了相同腕部指令，4–20 rad/s 的档位出现明显偏离。30 rad/s 是目标指令限速，真机驱动器可执行的速度和力矩仍需实测。下表保留首次通过时的数值。[限速后完整验收](../runs/hocap_carton_peak_slew_v1/phase_evaluation.json) · [原片与限速后回放](../runs/hocap_carton_slew30_visual_v1/source_vs_physics.mp4) · [限速版独立重放包](../runs/hocap_carton_export_slew30_v1/Wuji-HOCap-carton-replay.zip)

| 可观察量 | 原动作或门限 | 仿真结果 |
| --- | --- | --- |
| 物体中心最高抬升 | 244.2 mm | 238.3 mm |
| 物体位置 RMS / 最大 | ≤20 / 40 mm | 7.59 / 15.89 mm，最大误差在放回途中的第 106 帧 |
| 物体完整旋转 RMS / 最大 | ≤8 / 15° | 5.10 / 10.79°，最大误差在第 51 帧 |
| 末帧物体位置误差 | ≤15 mm | 8.39 mm |
| 源物体高于起点 100 mm 的区间 | 51 个时间区间 | 51/51 有手部支撑且无显著桌面载荷 |
| 末尾三个控制区间 | 离手，桌面承重 | 手物接触数均为 0，区间平均桌面支撑最小为 0.960 倍物重 |
| 实际五指尖对源手对应点的全程 RMS | ≤20 mm | 17.28 mm |
| 最大物体穿透 / 手部自穿透 | ≤2 / 0.5 mm | 0.55 / 0.00 mm |
| 关节软限位最大超出量 | ≤2° | 1.34° |

门限是这一开发片段的工程选择，初版直接 IK 失败后、最终控制器搜索前确定。源世界坐标、物体位姿、真实手部标注和帧序始终作为评分目标。固定纸盒在首帧姿态的参考误差为 138.42 mm / 7.76° RMS。旋转误差使用完整 SO(3)，盒体朝向没有按对称物体折叠。[评价配置](../configs/hocap_carton_evaluation.json) · [逐段指标与原生复验](../runs/hocap_carton_wrist_peak_v1/phase_evaluation.json)

## 哪些修正起了作用

直接 IK 把五指尖拟合到源手，运动学 RMS 为 3.10 mm，实际纸盒最高抬升为 0。把指垫而非关键点纳入接触位置约束，并在抓持阶段给相对两面的手指增加闭合预载后，模拟手初次举起纸盒，完整旋转误差仍大。10 mm 是指令上的虚拟预载偏移，实际穿透在重放中另测。修正拇指末节在接近阶段的目标跳变，保住了抓持并减小关节限位偏差。[直接 IK](../runs/hocap_carton_baseline_v1/report.json) · [指垫预载实验](../runs/hocap_carton_pad_10mm_v1/report.json) · [拇指修正](../runs/hocap_carton_acquisition_v1/report.json)

在同一个自由物体模型上，用模拟所得物体位姿误差迭代修正腕部指令，调整拿放期间的转腕和离手时间，然后对第 48 帧附近的局部姿态误差做短时腕部修正。物体仅由接触和重力驱动。导出的实际指令经独立积分与源视频并排检查。阶段 RMS 中，位置误差最大的是下降段（12.11 mm），旋转误差最大的是放下和离手段（7.35°）。最后两次 CPU 参数搜索分别检查 120、48 组控制，墙钟分别为 29.1、15.4 秒，均使用六个 CPU 工作进程。这些数只覆盖相应搜索和复验，冷启动数据处理、初始求解、此前失败尝试及方法开发另计，不能用来推算相对 FlashCHORD 的训练加速比。[腕部搜索](../runs/hocap_carton_wrist_fit_v1/cost.json) · [局部修正与成本](../runs/hocap_carton_wrist_peak_v1/cost.json)

## 参数和数据的适用范围

纸盒使用扫描网格的凸包，假设质量 150 g、均匀惯量、手盒摩擦系数 0.7。桌面取深度拟合的法向，平移至源首帧盒底，和共识深度平面相差约 7.2 mm。物体和手保持同一源世界坐标。RGB 末帧的源扫描投影与可见盒边有偏差，故 7.59 mm 描述相对源标注的误差。作者同序列视频核实为 30 fps。[源坐标、时间与桌面审核](hocap-carton-source-support.md) · [独立 RGB 轮廓核验](../runs/hocap_carton_rgb_outline_v1/report.json)

冻结腕部参考、手指命令和反馈增益，扫描 5 个质量与 3 个手盒摩擦值。150 g 下把摩擦降为 0.5，纸盒一度抬升 233 mm，但全程位置 RMS 升至 104.6 mm。480 g、摩擦 0.7 时最高抬升仅 6.1 mm。作者清单将 G04_2 标为 Pillsbury Chocolate Fudge，链接商品标称内装 16 盎司，捕获时的装填状态没有记录。这样大的质量差异会改变任务是否可完成。[15 组物理敏感性结果](../runs/hocap_carton_wrist_peak_v1/parameter_sensitivity.json) · [HO-Cap 作者物品清单](https://utdallas.box.com/s/jrj1s2wn46cv4gvpcstbhuus15sjrw7y)

本次形成了一条真实 ego 来源、名义物理条件下合格的开发轨迹。多参考训练仅使用事前声明的开发集，独立参与者 subject_2 的纸盒片段封存。后续三组学习器共享未经本次逐条优化的运动学输入和评估器，记录每个模型从初始输入达到同等物理验收所需的墙钟与 GPU 小时。[当前研究目标](current-research-goal.md) · [封存留出数据记录](../runs/hocap_carton_reserved_subject2_v1/reservation.json)

在 5090 上，FlashCHORD v0.4.0 已加载同一源片段的右手 Wuji 原始 IK、20 个关节、6 个腕动作维度及自由纸盒，完成 140 个控制周期的无辅助物理回放。零策略残差下物体没有抬起，位置 RMS 为 146.07 mm。该运行验证了新训练栈的输入与步进。比较算法训练成本时，应在同一物理场景和逐段标准上重新计时。[FlashCHORD 原生 Wuji 回放](../runs/flashchord_wuji_replay_v1/report.json)
