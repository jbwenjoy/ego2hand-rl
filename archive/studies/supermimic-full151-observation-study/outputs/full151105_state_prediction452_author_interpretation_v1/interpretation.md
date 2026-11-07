全452单步WM误差：作者解释与下一步边界

本说明基于完整452组，未重新加载模型或运行forward。作者保存数据描述12/12通过；Contact的25/25是独立数值权威。全部452曲线、六path/mode统计、每feature误差和所有exactties均保留，没有按误差或HO阳性筛选。

全体分量RMSE：q=21.244937mrad，qd=0.922717rad/s，p=4.734269mm/分量，R=0.0423599（raw9、无量纲），v=0.118979m/s/分量，ω=2.894186rad/s/分量。相应p向量RMS=8.199995mm、R Frobenius RMS=0.1270798、v向量RMS=0.206078m/s、ω向量RMS=5.012877rad/s。这些保留各自单位，不能相加成总分；R没有转角或投影，q没有wrap，ω是Reader/wrist frame分量。

预先固定的主峰是每行各block的分量RMS，在全部452行上argmax，并保所有exactties。此次各block均只有一个主峰：

| block | 峰值（逐行分量RMS） | 原group ID | path/mode/k | source frame |
|---|---:|---:|---|---:|
| q | 53.053922 mrad | 227 | A2/donor/k1 | 201 |
| qd | 2.11603876 rad/s | 379 | A2/direction/k0 | 200 |
| p | 34.651959 mm | 285 | A2/donor/k59 | 259 |
| R | 0.275136964 dimensionless | 3 | A0/donor/k3 | 203 |
| v | 1.23708017 m/s | 286 | A2/donor/k60 | 260 |
| omega | 20.7434872 rad/s | 3 | A0/donor/k3 | 203 |

最大绝对单分量是另一种明确保存的次要峰：q在group281/A2 donor k55、feature19，为0.120481rad；qd在group4/A0 donor k4、feature26，为6.709242rad/s；p在group285/A2 donor k59、feature41，为58.195621mm；R在group3/A0 donor k3、feature47，为0.501556；v在group286/A2 donor k60、feature53，为2.055109m/s；ω在group4/A0 donor k4、feature55，为30.667265rad/s。不能把单分量峰与分量RMS峰混为一个事件。原始row/feature exactties完整保存在all_six_block_peaks.json。

分层提示：A0 donor150组的q分量RMSE=15.71367mrad、p分量RMSE=1.208901mm；A2 donor150组分别为23.653717mrad和7.249081mm。12baseline和64direction在每条path内的聚合绝对误差很接近，但这些重复/近邻分支不是独立样本，且聚合近似不能证明控制敏感度正确。它们使用同一个A2 WM，路径输入分布与动作历史不同，因此也不是两个方法训练收益的比较。

HO机械join保留全部452组：每组四个native solver PREintegration深度取max，14组有>3mm，共44个阳性substep；全部阴性组也保留。六个主RMS峰的该HO max均未>3mm，而次要qd/ω单分量峰group4的HO max为7.106532mm。这只限制了“全部大误差都来自>3mm HO”的说法；它不能排除≤3mm接触、非手环境接触、自碰、落地/掉落、未覆盖状态分布或其它机制。这里没有完整类别join，也不把区间四PRE的最大值冒充post20几何标签。

下一最小行动建议：先对现有保存数组做同snapshot的“常量偏差 vs 有限动作响应”分解，而不是再跑WM/训练、修两个proxy点或改prior。已有8snapshot×19分支包含3次baseline与128个扰动端点；对每个snapshot保留全部3baseline，先声明固定基准规则，再比较模型与物理的中心化next-state增量；可保留全部64个两eps状态有限差分向量以及原实际u、clip/asymmetry标记。它只使用现有predicted/target/actualu数组，0新forward/AD/目标/goal/物理。要开始仍须Root先冻结新saved-only分析规则，不能在本说明中偷偷计算或按这六个误差峰选择子集。

这样能更直接区分：目前看到的是较大的预测offset，还是连同同状态下的动作响应也失真。中心化响应仍是有限区间、有限epsilon、带接触切换与量化的描述，不是真实Jacobian；此前9个稳定依赖方向组的5/9同号不因此升级，也不能拿两个epsilon当独立N。若需要state单位的signal/noise资格，应事先另定规则，不能挪用taskscore floor或看结果调阈值。

限制继续有效：本次是固定A2 WM的CPU FP32/ampFalse shadow、单步20ms，不是原GPU AMP策略梯度或多步H8滚动。仅从p向量RMS8.2mm不能推断1mm SDF带宽失效或P/支持梯度为零，这次没有计算objective。known-rough参考及原任务失败均未改判；原checkpoint100在prior forward之前WM/failure已分歧，保留相同初始参数/预算的描述性配对，不补造共同训练轨迹因果证据。未来方法收益与共同warmup/共同模型控制问题应分开设计，不自动重训。

本说明没有增加模型、FK/MJB/query、forward、AD、GPU、Env、物理或训练调用。原报告、源与数据均未修改。
