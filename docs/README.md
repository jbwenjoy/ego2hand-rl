# 研究笔记

如果只读两篇，可以从[开发纸盒](hocap-carton-first-source-replay.md)和[留出实验](hocap-heldout-results.md)开始。前一篇跟着一条示范看手怎样拿稳纸盒，后一篇解释换人、换物体后，四种训练方案各自走到了哪里。

## 真实人手示范

| 问题 | 记录 |
| --- | --- |
| 手、物体和摄像机怎样放在同一个世界里？ | [时间、支撑平面和接触阶段](hocap-carton-source-support.md) |
| 为什么指尖拟合得很准，盒子仍拿不起来？ | [直接 IK 与指垫预载](hocap-carton-control-baseline.md) · [学习策略的抓取诊断](hocap-first-policy-grasp-diagnosis.md) |
| 桌面高度会怎样改变结果？ | [首帧碰撞和深度观测](hocap-heldout-initial-support.md) |
| 怎样比较新动作上的训练开销？ | [留出划分和四档训练预算](hocap-heldout-comparison.md) · [动力学迁移对照](hocap-heldout-dynamics-transfer.md) · [同卡开发实验](hocap-flash-training-comparison.md) |

## 代码与实验环境

[脚本索引](../tools/README.md)按实验顺序列出入口。[研究目标](current-research-goal.md)记录验收条件，[代码结构](architecture.md)解释各包的分工。CPU 纸盒回放与 GPU 上的 FlashCHORD 各有一套运行环境，具体配置见[环境说明](environment.md)和[求解器校准](flashchord-contact-solver-calibration.md)。

## 早期实验

固定腕转笔用来验证密集接触、并行采样和策略重放。[冻结策略的长程评测](pen-frozen-cost-long-results-2026-11-20.md)保留了笔轴偏差和提前终止的结果。持杯实验暴露了固定腕与源腕位移不相容的问题，[持杯动作复盘](mug-source-fidelity-correction.md)说明了为什么仿真中的杯子会出现不同于原片的朝向。两条线怎样促成现在的研究设计，见[研究复盘](ego-retargeting-goal-review-2026-11-21.md)。

早期按日期和版本写下的进度仍可在本目录查阅。需要对照整理前的首页时，可在 Git 中查看 `git show 3e44b6e:README.md`。
