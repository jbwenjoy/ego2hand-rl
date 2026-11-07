# 研究方案 v0.5：先核验世界模型的接触相关响应

2026-11-06。用户确认尚无指定 ego 数据，继续受控实验，Goal 保持 active。
原方案的组合可作为研究路线；目前尚未形成完整接触动力学重定向或 CHORD 任务收益证据。
本修订保留 v0.4 与所有历史结果，更新下一步优先级。

## 本轮已完成

| 检查 | 实测结果 | 取得的范围 |
|---|---|---|
| 原生 GPU/B1 状态重放 | 2260 行；452 终点和 160 锚点完整58维数值差0，独审29项 | 特定保存状态的算术/身份资格，旧CPU失败保留 |
| 同 solver PRE 的 P 代理 | 1808 行；44 行原生HO>3mm、42行P>0、2行已覆盖几何上的漏检，独审35项 | 同位姿的代理检出描述，不等全局几何/接触证书 |
| 固定 A2 WM 单步预测 | 全452区间；位置向量RMS8.20mm，独审25项 | FP32CPU shadow 的单步状态误差覆盖，不等梯度资格 |
| baseline/动作响应分解 | 全8中心/128端点/64对，独审22项 | 大baseline偏差与有限响应失配同时存在；不升级真实Jacobian |

这些结果使用同一缓存谱系、单种子和相依恢复分支。A0/A2 是动作来源路径，本轮的预测模型始终
是同一个固定 A2。旧三臂整体任务失败、不同 warmup 以及已有 5/9 梯度方向同号结果均不重判。
v0.3 其他窗口/控制器/三种子的结果也不与本批混池。

## 当前判断

接触几何先验已经实现，局部 P=0 不足以否定它在完整路径或训练想象状态中活跃。
但代理仍有同几何低敏感，且世界模型在可见状态预测与有限动作响应上都有偏差。
下一步需把模型的接触相关响应作为单独问题验证，随后再评估先验与学习收益。
现有观察不能以增加先验权重、只改常量偏差或选取较好 seed 代替。

## 下一项具体工作

只使用既有保存数据，核对全部 152 个恢复分支的 608 个 solver PRE 子步。
先固定字段、分类、valid/unknown 和时相规则，再执行一次机械汇总与独立核对。

- 按完整 native catalog 的 geom/body IDs 和原 enabled-hand 规则分类；记录全部候选，不先按力过滤。
- 区分原 generated HO/self/other 与明确命名的 object-with-nonhand 描述；不把未知非手对象自动称为静态环境。
- 原 decoded force6 及 active 条件作为保存的来源证据。正法向力描述另需 type 的 CONSTRAINT 位、world0、
  原始 efc 槽有效性；不使用已重排的 host 地址去索引 raw Warp efc force。
- 保留 sample608 表与原 contact index 长表，零候选记录也保留；valid-empty 的 observed candidate depth0
  与全局间隙未知分别表示，不把空类别的最小距离补成已测间隙。
- 全部以 solver PRE pose/COM 为时相，避免拿15ms solver力与20ms poststate混配。
- 将全8/152的接触描述与现成零控制差、编码零响应相联，不按旧9组、当前误差或某个 contact regime 筛样。

此步骤不新建 MJB、原生 FK/碰撞查询、模型前向、AD、物理或训练，不声称全量原始 pyramid 力已重解码。
它检验“物理有限耦合弱或编码平坦处仍有 WM 动作—物体响应”这一候选描述，不能单独确定原因。

## 随后如何推进

依据接触描述事前固定一个最小模型修正假设与对照，再写预算、数据隔离与确认规则；
可以研究有限响应匹配等校准，具体新训练尚未安排或执行。
正式方法比较仍需清楚的共同学习初态/normalizer/replay与控制边界、固定种子与终点、
失败成本账及独立评估，不能把 equal updates 写成相同 warmup 的因果对照。

物理可行参考、无机器人标签输入契约、多轨迹共享/留出隔离、真实视频谱系与硬件契约保持原目标。
固定腕重设计、接触未知、相机/尺度不确定性不会因本轮数值审计而变成已知。
单条已见受控轨迹不作为真实 ego 泛化或新增轨迹加速的证据。

如果启动新的长训练，仍按用户要求在首个检查点健康验证并给出 ETA 后暂停 Goal。
当前无新训练或后台物理作业；H8 已确认数据连续性但保持 parked。

## 结果与复现入口

- [1808行几何惩罚](D:/workspace/supermimic-full151-observation-study/docs/full151_dense_penalty_observability_findings_2026-11-06.md)
- [452区间状态预测](D:/workspace/supermimic-full151-observation-study/docs/full151_state_prediction452_findings_2026-11-06.md)
- [128响应与64对动作分解](D:/workspace/supermimic-full151-observation-study/docs/full151_saved_action_response_findings_2026-11-06.md)
- [成本读取补充](D:/workspace/supermimic-full151-observation-study/docs/full151_passive_model_read_cost_addendum_2026-11-06.md)
- [上一版方案](D:/workspace/supermimic-chord-study/docs/research-plan-v0.4.md)
