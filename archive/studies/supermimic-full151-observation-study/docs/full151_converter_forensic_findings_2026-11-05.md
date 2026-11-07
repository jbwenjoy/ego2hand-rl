# 同solver时相检查：转换误差门失败与全量算术诊断

密集穿透检查尚未产生几何评分。首次CPU转换在452个端点的原8eps32误差门处停止；原失败、配置和源码均保留。另一个明确禁止几何评分的算术诊断条件已完成，并独立复现：新CPU复算中只有5个端点角速度分量超出旧界，最大差约5.2452e-6 rad/s。位置、线速度、关节位置和关节速度一致，旋转矩阵差异均在旧界内。

## 两个独立运行条件

第一条件实际完成一次被动MJB常量读取、一次2260行原生unbound SimReader.state转换，随后触发原门失败。工厂构造、目标前向、评分均为0。失败前没有保存转换数组，因此不能恢复或声称已独立复算原进程中的逐元素残差。[失败保存审计15/15](D:/workspace/supermimic-full151-observation-study/outputs/full151105_solver_phase_converter_failure_saved_review_v2/report.json)核对了源归档、真实Root配置、计数和缺失数据；没有改判原失败。

第二条件是新配置、新输出与新源码的算术诊断。它复用相同输入准备和冻结的原生状态方法，实际只读取一个被动MJB并转换一次2260行。所有原始转换结果先保存，再记录452个端点、160个首步anchor及1808个solver位姿比较，不因为误差门失败而丢失残差。该诊断函数没有可达的工厂或评分路径。

[新诊断报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_CPU_v1/report.json) · [新原始数组](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_CPU_v1/raw_conversion_all2260.npz) · [独立保存复核23/23](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_independent_v1/report.json)。这是对同一保存输入的另一次算术复算，未恢复原失败进程的中间数组。

## 全量结果与单位

| 452个post20端点的分量组 | 最大绝对差 | 原门超限分量数 |
|---|---:|---:|
| 关节q，rad | 0 | 0，逐位一致 |
| 关节qd，rad/s | 0 | 0，逐位一致 |
| 物体位置，m | 0 | 0 |
| 物体旋转矩阵元素 | 4.768371582e-7 | 0 |
| 物体线速度，m/s | 0 | 0 |
| 物体角速度，reader/wrist frame rad/s | 5.245208740e-6 | 5，分布于5行 |

最大角速度差出现在A2 donor k10、端点row236、feature57。全部5个超限均为未扰动donor端点，未只保留或绘制这些行。角速度由原始BODY坐标自由关节角速度，经物体旋转与palm逆旋转转换得到；表中的最后3维不是原始BODY角速度。

160个首步anchor均通过原规则，角速度最大差2.384185791e-7 rad/s。1808个solver位置/旋转与对应积分前自由关节位姿比较均通过原规则。q/qd使用原逐位要求，其余分量仍按8*eps32*max(1,abs(actual),abs(reference))逐元素判断；没有替代容差或删行。

![全部452端点的原规则](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v2/all452_endpoint_old_rule.png)

[160个anchor](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v2/all160_anchor_old_rule.png) · [1808个solver位姿](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v2/all1808_solver_pose_old_rule.png) · [2260行条件量图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v2/all2260_conditioning_secondary.png)。四图已由作者与Root实际查看；全量CSV和逐元素差、旧界与flags仍保存。

## 当前证据能解释什么

源代码中的mm/mv是逐元素乘法与求和，不是GEMM；TF32设置本身不能直接解释这一步转换差异。原始物理数据来自GPU、单环境，而本次CPU转换一次处理2260行，后端与运算调度均不同。相同数学定义不保证两条FP32求值路径逐位相同。

在看到新残差之前，另行冻结了按输入量级、quaternion归一化和嵌套旋转传播的条件算术解释。[只读条件模型结果](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_conditioning_saved_v1/report.json)显示，全部CPU/F64及保存GPU端点比较落在该条件envelope内；5个超限分量的pair residual/envelope比例约0.047–0.112。这只是F64求值的条件模型，不是严格向外舍入的区间证明、新资格门或唯一原因判定。GPU中间旋转量未保存，F64 shadow也不是替代状态真值。

这些结果与舍入、归一化和嵌套旋转的条件敏感性相容；仍不把差异唯一归因于消减、TF32或某个kernel，也不把“数值很小”直接解释为无害。

## 下一步保持完整58维与原门限

下一项条件将使用原生CUDA、与原单环境相同的batch1方式，按保存的q/v/index/palm FP32字节逐行调用同一个unbound SimReader.state，共2260次。它不读取权重、不构造环境、不运行物理或几何目标，只将GPU结果按字节复制回CPU，按原规则重新检查全部452端点、160个anchor及1808个位姿。

这项选择旨在匹配原数据的生成实现；不将检查缩减为几何所需的部分分量，不放宽旧界，也不改用F64状态。新条件尚未执行，不预设一定通过；任一门失败仍不允许后续密集评分。它同时改变后端与batch调度，因此也不能单独区分二者的因果作用。

原CPU条件仍为FAIL，密集穿透评分资格仍为False。此前局部穿透项全零与完整训练P均值为正的结论保留；是否存在代理漏检，仍需通过新的坐标检查后才能判断。Goal保持active，本轮未启动新训练。
