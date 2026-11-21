# 无法部署的最终策略如何做独立补充比较

原始评测仍使用各共享方法在最后固定预算处的指定 actor。若该文件含有 NaN 或无穷值，保留其原有结论、三名参与者名额及完整开发成本。补充组在同一方法的实际训练谱系内，按保存步数从大到小检查较早的 actor，只读取推理张量、检查点身份和训练配置。策略分数、损失和留出视频不会参与选择。[事前登记规则](../configs/hocap_heldout_deployment_recovery_v1.json)

谱系由各段真实载入的原生训练状态和回放文件建立，根节点须与共同第零步张量见证吻合。物理和动作语义、参考与重置种子必须一致。第一个全部推理张量有限的 actor 成为恢复候选。若找不到候选，保留该方法的全部失败成本和未执行分配。固定终点本来就有限时，不产生恢复策略或重复评分。若需要迁移恢复策略的世界模型，模型和归一化必须属于该 actor 的同一步数、同一检查点组。缺失、未训练或数值无效的配对模型单列为不可用。[选择实现](../tools/heldout_recovery_selection.py)

在 WSL 的 FlashCHORD 环境中按下列顺序执行。符号变量分别指原 v1 输出目录、恢复输出和大文件目录。终点选择与两个补充方案的登记都应在解封任何源窗口之前完成。

```bash
python tools/run_wuji_heldout_deployment_recovery.py register-final \
  --original "$V1_OUT" --out "$RECOVERY_OUT" --bulk "$RECOVERY_BULK" \
  --final-selection-sha256 "$FINAL_SHA"
```

该命令输出 `recovery_selection.json` 的 SHA-256。原评测全部结束后，向恢复评测传入该 SHA 和原评测已登记的源阶段 SHA。

```bash
python tools/run_wuji_heldout_deployment_recovery.py run \
  --out "$RECOVERY_OUT" --recovery-sha256 "$RECOVERY_SHA" \
  --registered-stages-sha256 "$STAGES_SHA"
```

恢复候选先在所有可评分参与者上做零样本测试，再按原有四档目标步数独立适配。每段沿用相同的世界数量、目标回放、初始化、物理规则、全程和独立重复验收。终点或中途不再可恢复时，记录实际消费和后续无法达到的预算。原比较已经完成的源提取、刚体校验、冷启动实验及有限终点策略不会重跑。实际实验账本把共享预训练只计一次；模拟单独部署某一恢复方法时，向它计入该方法包含失败段的全部开发开销及检查耗时。[补充驱动](../tools/run_wuji_heldout_deployment_recovery.py) · [开发样例验证](../runs/heldout_deployment_recovery_dev_v1/report.json)
