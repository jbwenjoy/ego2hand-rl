## 留出动作上的世界模型迁移

这一补充比较使用相同的最终共享 actor。原 v1 实验在新动作上重新拟合世界模型。补充组同时迁移开发集的动力学权重，以及输入状态和输出增量的中心与尺度。在目标动作的第一次历史缓冲区填满时也保留源尺度，随后继续用目标仿真转移更新模型和策略。两组的目标 critic、优化器、经验回放、世界模型历史和计数均从零开始。不同留出动作之间不传递状态。[事前规则](../configs/hocap_heldout_warm_dynamics_v2.json) · [原有对照](hocap-heldout-comparison.md)

开发集最终选择锁定后，在原比较的 `stage` 之前执行 `run_wuji_heldout_dynamics_supplement.py register-final --original V1_OUT --out V2_OUT --bulk V2_BULK --final-selection-sha256 HASH`。注册器持有原比较的互斥锁，检查原研究尚未打开任何目标源，核对同一训练步的 actor、原生优化器状态、检查点组 ID 和辅助文件哈希。注册器仅把模型参数与尺度写入专用迁移文件。迁移文件附带特征、动作、critic 上下文、奖励列、物体几何和物理规则的指纹。[导出工具](../tools/export_wuji_heldout_dynamics.py)

原比较完成后，用 `run --out V2_OUT --supplement-sha256 HASH --registered-stages-sha256 HASH` 执行。补充组使用相同目标阶段、并行数、四个真实步数预算、课程和评测规则。相同 actor 的零样本结果按检查点、源参考、源阶段哈希复用。最终端点达到干预时刻时才额外做独立扰动实验。每个目标独立保存迁移指纹和目标世界模型、优化器及缓冲区，恢复会继续已有目标适配。[补充比较驱动](../tools/run_wuji_heldout_dynamics_supplement.py) · [目标训练入口](../tools/train_wuji_heldout_warm_dynamics.py)

报告同时给出从头部署该方法的成本，以及运行补充实验所增加的成本。前者为同一个共享教师计入一次预训练，后者复用已经支付的预训练、源准备、零样本和力校验费用。首个通过时间限于真正评估的检查点。CPU 开发样例验证了迁移前后模型预测一致、初次目标缓冲区填满时尺度不变、目标更新和成对恢复。[开发样例验证记录](../runs/hocap_warm_dynamics_fixture_v1/test_report.json)
