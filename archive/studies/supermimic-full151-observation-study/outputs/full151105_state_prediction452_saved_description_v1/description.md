全452状态误差：作者保存数据说明（不是独立审计）

峰值规则在读取预测值前固化：对全部452行、全部6个block取逐行分量RMS最大值，保留所有精确并列；次要表保留最大绝对分量的所有(row,feature)精确并列。没有isclose、挑组或混单位总分。

| block | units | all452 component RMSE | MAE | max absolute component |
|---|---|---:|---:|---:|
| q | rad | 0.0212449369 | 0.0156112452 | 0.120480955 |
| qd | rad/s | 0.92271679 | 0.651665717 | 6.70924234 |
| p | m | 0.00473426927 | 0.00236983557 | 0.0581956208 |
| R | dimensionless | 0.0423599237 | 0.0302763524 | 0.501556292 |
| v | m/s | 0.118979207 | 0.0752864463 | 2.05510864 |
| omega | rad/s | 2.89418599 | 1.93019594 | 30.6672649 |

完整六path/mode统计在report.json；全部452曲线在all452_descriptive_curves.csv。以下只是一页峰值索引，所有精确并列完整保存在all_six_block_peaks.json。

| block | primary peak(component RMS) | original group ID/path/mode/k |
|---|---:|---|
| q | 53.053922 mrad | 227/A2/donor/k1 |
| qd | 2.11603876 rad/s | 379/A2/direction/k0 |
| p | 34.651959 mm | 285/A2/donor/k59 |
| R | 0.275136964 dimensionless | 3/A0/donor/k3 |
| v | 1.23708017 m/s | 286/A2/donor/k60 |
| omega | 20.7434872 rad/s | 3/A0/donor/k3 |

机械HO join保留全部452组，其中14组的4个solver PRE量最大值>3mm，合计44个positive solver substep；这不是post20几何标签。全部阴性组同时保留。

这里只读既存预测、target、身份和标量数组；没有运行模型、目标、MJB/FK/query、AD、GPU或物理。R不变成角度，q不wrap，不对预测state投影/修复。
CPU ampFalse one-step forward error、20ms末state与区间solver PRE量的时相要分开；该报告没有梯度资格或方法收益结论。
