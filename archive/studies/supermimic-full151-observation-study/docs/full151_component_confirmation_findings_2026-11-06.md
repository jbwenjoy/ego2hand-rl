# 共同PRE响应校准的确认结果：响应改善，点式状态退化

2026-11-06。用户已恢复Goal。两臂final1000独立读回29/29通过后，原77文件封存哈希全部核对一致，按原条件解封。唯一实际确认评价已完成，独立保存数据复算24/24通过。**原局部确认条件未满足**；没有改权重、挑checkpoint、筛样或调整判据。

## 冻结比较与结果

两个模型均从同一A2参数/normalizer/physical xi开始，使用相同旧152恢复分支/64对，分别进行1000次全batch更新。新相位 `[24,74,124,148]` 的全部152端点/64对用于本次确认。其nominal状态已经观察过；新增的是受控扰动响应，不是未见轨迹或真实ego数据。

确认判据在新增响应采集前已固定：全64对的归一化共同PRE52响应MSE严格降低，且全152行native52平均点式损失不增加。结果如下：

| 冻结指标 | pointwise only | pointwise + response | 变化 |
|---|---:|---:|---:|
| 全64对 normalized response MSE | 4.367903148 | 3.380625014 | −22.603% |
| 全152行 native pointwise loss | 0.077275984 | 0.085142404 | +10.180% |

第一项通过，第二项未通过，`local_component_confirmation=False`。训练数据上的loss改善不能代替这一确认结果。

## 原单位的状态误差

下表保留六块的原单位；q/qd为分量RMSE，p/v/omega为向量RMS，R为原始9矩阵元素误差的Frobenius RMS。没有合成跨单位58维总范数，也没有把R9解释为角度。

| 状态块 | pointwise only | pointwise + response |
|---|---:|---:|
| q / mrad | 9.831946 | 8.657872 |
| qd / rad/s | 0.322274 | 0.343166 |
| p向量 / mm | 0.675140 | 1.059959 |
| R9 Frobenius / 无量纲 | 0.03035729 | 0.02989074 |
| v向量 / m/s | 0.02129786 | 0.02987332 |
| omega向量 / rad/s | 1.729411 | 2.089311 |

q与R9略有改善；qd、p、v、omega退化。全部8个snapshot的响应MSE均降低，但平均pointwise loss只有A0/k74和A0/k124降低，其余6组增加，A2四组全部增加。全部8组和8×6块表已保留，不以局部改善替代全量判据。

共同PRE52响应损失有40个手部q/qd坐标和12个物体坐标。按原52维权重分解，总MSE下降约94%来自q/qd贡献的下降。物体p块的normalized响应MSE还略有增加。normalized与原单位有限响应残差也不必同向：s52逐坐标不同，实际控制差逐pair不同；例如logR3/v块的normalized指标降低，原单位有限响应残差RMS略增。两类表分别保存，不合成“整体物体响应都改善”的结论。

## 覆盖、时相与实际成本

每臂仅一次B152前向；共3个可信新component文件读取、2次native WM构造、2次前向/304预测行。没有119f原模型加载、Actor、AD、optimizer、MJB、状态转换、碰撞查询、物理或GPU。两个新组件均strict加载、eval/冻结参数、AMP关闭；原始RNG和输入张量、权重/normalizer前后关系均核验。完整worker用时7.0727787秒；这不包括审阅、合成检查、独审或作图成本，不构成训练加速证据。

输入是原生直接捕获的snapshot_x8各重复19次。新分支没有直接保存PRE58；这里只保留engineering540/原始primitive、INTEGRATION与三baseline所支持的条件范围，不补全该资格。目标是原生直接保存的post20 branch_x_next152。原native15ms reward/self15与手物力不混用；本次self15仅保留有限raw logits，不用作HO标签。

独审从保存的native FP32 point/common52中间量重做后续算术，并核对全部152/64、原单位raw58、角色/顺序、实际控制差、clipped/asymmetric记录、release/final29/recipe/pins、源码归档与raw RNG。它不重新运行模型或native Log；权重不变的前后digest关系仍是源码保存证据，未伪称另一次模型读回。

Root已视检两份完整图：全64响应及冻结判据、全152六块原单位误差。图与长表完整覆盖全部样本，没有图中筛点或人为噪声下限。

## 研究判断与后续

本次有限响应监督取得局部取舍，不能据此将该校准组件作为已确认升级接入CHORD Actor。冻结确认失败保留。

下一项是固定这两个final1000的探索性机制检查：在同8个中心，用匹配post20物体任务score的uVJP对照原物理有限差分，全部32方向/64步长保留；实际控制差与legacy d_eff口径分列。它直接检验状态响应MSE降低是否伴随任务score梯度更贴近物理有限响应，不改变本次确认结果，也不作为新的blind检验。当前仅源码/字段可行性确认，未执行新FW或AD。

物理参考及三臂完整任务失败、原5/9方向结果、warmup分歧、graph scratch未知、代理漏检、单谱系/egoN0等保持。原目标中的可行参考修复、接触扳手支持、共同真实初态任务对照、多轨迹与真实ego/硬件验证仍未完成；Goal保持active。

## 证据入口

- [实际确认报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_component_confirmation_CPU_v1/report.json)，SHA `82b88d40d26aa0a940b09777a40e741c62dfecf2c2e14ba5bbbe51040b100908`
- [独立24项报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_component_confirmation_saved_independent_v1/report.json)，SHA `2ca69685e2bc63e95e3dde1edefa4829858c47b7a4d62e3275185eda56cc3b9f`
- [全64对与冻结判据图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_component_confirmation_figures_v1/all64_response_and_frozen_criterion.png)、[全152原单位图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_component_confirmation_figures_v1/all152_state_errors_original_units.png)
- [作者全8/6块/8×6描述](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_component_confirmation_author_description_v1/description.md)，作者解释不充独立审计
- [固定recipe](D:/workspace/supermimic-full151-observation-study/protocols/full151105_WM_common_pre_calibration_recipe_v1.json)、[解封记录](D:/workspace/supermimic-full151-observation-study/outputs/full151105_common_pre_confirmation_release_v1/release.json)、[实际评价协议](D:/workspace/supermimic-full151-observation-study/protocols/full151105_WM_component_confirmation_CPU_run_v1.json)
- [任务梯度字段可行性建议](D:/workspace/supermimic-full151-observation-study/docs/full151_final_components_task_gradient_source_feasibility_v1.md)
