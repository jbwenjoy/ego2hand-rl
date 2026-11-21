# FlashCHORD v0.4.0 的 Wuji Hand 2 右手适配

2026-11-21。用隔离的 WSL 运行环境，在同一张 RTX 5090 上调用 NVIDIA v0.4.0 的 `setup_scene`、`build_robot_reference_binding`、`ReplayRunner` 和原生动作缓冲。适配器读取实际 Wuji Hand 2 校准几何和执行器参数，接受带 20 个命名关节、腕部世界位姿的 `motion_v1/single_robot` Parquet。这里的物理回放使用零策略残差、零虚拟物体控制，不包含 RL 更新。[NVIDIA 实现](https://github.com/nvidia-isaac/video_to_data/tree/v0.4.0/robotic_grounding/flash_chord) · [本机运行环境](../runs/flashchord_wsl_v040_preflight_v1/cuda_interop.json)

## 手与场景

以三个移动关节和一个球关节表示浮动腕。机器人有 27 个位置坐标、26 个自由度，包含 20 个原模型指关节。保留原模型的掌、指节、指垫共 21 个碰撞体及 5 个真实指尖 site 的局部坐标。删除不参与接触的装饰网格。传感器固定体单独保留。标记掌、五根远指节及五个指尖，按实际关节名称绑定，避免依赖 Parquet 列顺序。三个腕部移动体各附加 0.1 g 数值质量。[导出器](../tools/export_wuji_flashchord.py) · [Newton 实施](../src/supermimic_research/runtime/flash_wuji.py)

原生物理基准在腕上施加有限的反馈广义力。FlashCHORD 使用移动关节 PD、球关节转矩控制和被动阻尼。移动关节单轴努力上限为 55/√3 N，球关节单轴力矩上限为 4/√3 N·m，三轴执行命令的向量上限分别为 55 N 与 4 N·m。此处遵循官方控制器的逐轴裁剪方式。位置目标残差写入后，在 GPU 上按真实指关节的 `ctrlrange` 逐项裁剪。原 MuJoCo 执行器的总指关节力矩上限映射至 Newton 的关节级双向努力上限。世界重力为 −Z，保留源物体 URDF 的质量、质心与惯量，场景仅启用一个上表面 z=0 的桌体。当前纸盒质量假设为 0.15 kg。原生基准的手盒/桌盒摩擦分别为 0.7/0.6，v0.4.0 的标准场景对所有碰撞体应用同一摩擦，本次统一为 0.7。

## 模型核查

对 `subject1_G04_2`、`subject3_G10_3`、`subject4_G16_4` 的全部 141、221、276 帧，分别核对官方绑定后 FK 与独立参考。掌坐标、五个远指节与指尖的世界位置误差在三条轨迹上均小于 0.00019 mm。在纸盒的三个姿态、每个碰撞体 102 个空间方向上比较 Newton 求解器实际编译的支撑距离与 MuJoCo 原模型，21 个碰撞体最大差约 7.2 × 10⁻⁸ m。所有 210 对手部碰撞体的候选碰撞资格一致。20 个关节的有效刚度、阻尼、努力上限、范围、摩擦与反射惯量逐项吻合至 10⁻⁷ 绝对误差内。对极端残差及超范围目标执行实测，GPU 输出距校准关节目标上下限的浮点误差小于 1.1 × 10⁻⁷ rad。[纸盒绑定](../runs/flashchord_wuji_replay_v1/binding.json) · [第二条绑定](../runs/flashchord_wuji_bind_subject3_G10_3_v1/binding.json) · [第三条绑定](../runs/flashchord_wuji_bind_subject4_G16_4_v1/binding.json) · [求解器审计](../runs/flashchord_wuji_solver_audit_v1/report.json)

纸盒轨迹首帧重置后执行连续 140 个 30 Hz 控制区间，每区间 40 个子步，共 5,600 个子步。所有状态有限，直接 IK 的零残差回放抬升 0 mm，位置 RMS 146.07 mm、姿态 RMS 6.27°。这与原生直接 IK 同样未能拿起纸盒。WSL 记录的场景与求解器初始化为 2.68 秒，首控制步（含新核编译）27.74 秒，后续单世界同步读回的每控制步中位数 0.166 秒。该时间只对应这个调试回放。训练效率以实际学习器更新和达标成本独立计算。[官方路径回放脚本](../tools/replay_flashchord_wuji.py) · [物理结果](../runs/flashchord_wuji_replay_v1/report.json) · [原生 IK 对照](../runs/hocap_carton_baseline_v1/report.json)

## 复算入口

Windows 侧用已有 Wuji 研究解释器运行 `tools/export_wuji_flashchord.py OUT`，得到可在 Linux 读取的 MJCF、网格与参数清单。WSL 侧使用隔离的 v0.4.0 解释器，令 `PYTHONPATH` 指向本仓库的 `src`，`V2D_ASSETS_DIR` 指向本次 Parquet 所附的 assets 根目录。启动 `tools/replay_flashchord_wuji.py --asset ASSET_DIR --parquet PARQUET --warp-cache CACHE_DIR --out NEW_DIR`。三个输入共用同一份手部模型，求解器核查入口为 `tools/audit_flashchord_wuji_solver.py`，参数相同。[求解器核查代码](../tools/audit_flashchord_wuji_solver.py)
