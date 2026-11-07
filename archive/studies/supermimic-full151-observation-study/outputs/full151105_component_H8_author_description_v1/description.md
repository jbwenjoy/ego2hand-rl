# 三模型 H8：全量保存统计的作者描述

**探索性、非独立审计；没有新的科学通过门或总体赢家。** 本文只读已完成报告的432条统计，不打开rollout数组，不导入Torch/NumPy、不加载权重、不前向或运行物理。

286个相互重叠的固定donor控制windows来自A0/A2各143起点；每个h的ALL为286，路径各143。三模型使用同一cohort；ALL不与其两个子路径重复累加。8个h共6864条模型输出相依，不作为独立N或CI。

## 全部模型的 ALL h8（分量RMSE，原单位）

|块/单位|initial|point final1000|response final1000|response相对point变化|
|---|---:|---:|---:|---:|
|q / rad|0.092373535|0.0439187436|0.0394440566|-10.189%|
|qd / rad/s|1.99152084|0.658682593|0.649280576|-1.427%|
|p / m|0.0372606533|0.0200127264|0.0219731517|9.796%|
|R_raw9 / dimensionless|0.317742592|0.230617081|0.259942055|12.716%|
|v / m/s|0.314170721|0.18381783|0.178512064|-2.886%|
|omega / rad/s|6.14023805|3.47666986|3.87910713|11.575%|

ALL h8位置**向量**RMS为 initial 64.537345、point 34.663059、response 38.058615 mm；它与上表p分量RMSE相差sqrt(3)。Rraw9是矩阵元素/无量纲Frobenius误差，绝不解释为旋转角。

在ALL h8，两个校准模型的六个分量RMSE均低于initial。response相对point在q/qd/v较低、p/Rraw9/omega较高；这是逐块的tradeoff，不能通过跨单位投票或求和形成总赢家，也不推唯一遗忘原因。

## 全h的位置向量RMS（mm，仅本H8 cohort）

|h|initial|point|response|
|---|---:|---:|---:|
|1|9.158803|6.04202|6.240906|
|2|18.0154|11.81862|11.85019|
|3|26.57475|17.13664|16.95789|
|4|34.83071|21.85327|21.68468|
|5|42.76888|25.91791|26.12549|
|6|50.3784|29.3455|30.3223|
|7|57.64095|32.21547|34.30235|
|8|64.53734|34.66306|38.05862|

## 所有模型/单位的 ALL h1→h8 分量RMSE变化

|role|块/单位|h1|h8|h8/h1|相邻h增加/减少/相等次数|
|---|---|---:|---:|---:|---:|
|initial|q / rad|0.02015323|0.09237354|4.58356|7/0/0|
|initial|qd / rad/s|0.7940685|1.991521|2.508|7/0/0|
|initial|p / m|0.005287837|0.03726065|7.04648|7/0/0|
|initial|R_raw9 / dimensionless|0.04389963|0.3177426|7.23793|7/0/0|
|initial|v / m/s|0.1262028|0.3141707|2.48941|6/1/0|
|initial|omega / rad/s|2.841818|6.140238|2.16067|7/0/0|
|point|q / rad|0.01484597|0.04391874|2.95829|7/0/0|
|point|qd / rad/s|0.5386084|0.6586826|1.22293|4/3/0|
|point|p / m|0.003488362|0.02001273|5.737|7/0/0|
|point|R_raw9 / dimensionless|0.03784604|0.2306171|6.09356|7/0/0|
|point|v / m/s|0.1011131|0.1838178|1.81794|5/2/0|
|point|omega / rad/s|2.52911|3.47667|1.37466|6/1/0|
|response|q / rad|0.01510024|0.03944406|2.61215|7/0/0|
|response|qd / rad/s|0.6075776|0.6492806|1.06864|4/3/0|
|response|p / m|0.003603189|0.02197315|6.09825|7/0/0|
|response|R_raw9 / dimensionless|0.03914775|0.2599421|6.64003|7/0/0|
|response|v / m/s|0.09875323|0.1785121|1.80766|5/2/0|
|response|omega / rad/s|2.604944|3.879107|1.48913|6/1/0|

不要求增长单调；固定起点的目标时相随h推进，目标观测范围从h1的1..143变为h8的8..150。逐h比较描述自回归积累及其目标时相，不是控制了时间分布的因果证明。完整ALL/A0/A2所有h增长及相对误差保存在CSV，零基准分母只标undefined、不填0。

## h8全部路径异质性（分量RMSE；同单位内比较）

|role|块/单位|A0|A2|A2/A0|
|---|---|---:|---:|---:|
|initial|q / rad|0.0355918|0.1256939|3.53154|
|initial|qd / rad/s|0.4244769|2.784265|6.55928|
|initial|p / m|0.005030639|0.05245384|10.4269|
|initial|R_raw9 / dimensionless|0.2344457|0.3833483|1.63513|
|initial|v / m/s|0.0542449|0.4409807|8.12944|
|initial|omega / rad/s|3.126475|8.101247|2.59118|
|point|q / rad|0.02926819|0.05478216|1.87173|
|point|qd / rad/s|0.40274|0.8399559|2.0856|
|point|p / m|0.00410908|0.02800239|6.81476|
|point|R_raw9 / dimensionless|0.1199612|0.3032784|2.52814|
|point|v / m/s|0.03922153|0.2569818|6.55206|
|point|omega / rad/s|2.007143|4.488412|2.23622|
|response|q / rad|0.03208996|0.04562786|1.42187|
|response|qd / rad/s|0.4499978|0.8003952|1.77866|
|response|p / m|0.003822407|0.03083874|8.06789|
|response|R_raw9 / dimensionless|0.1870999|0.3164386|1.69128|
|response|v / m/s|0.05894522|0.2454762|4.16448|
|response|omega / rad/s|3.098929|4.526763|1.46075|

## 解释限度与文件

三个角色分别是未修改的initial A2组件，以及相同起点/norm/数据的两种final1000校准；initial不是伪造final1000或重新加载旧105wrapper。固定外部actualu的open-loop状态诊断不等于Actor重选动作、EMA/记忆、prior/reward或survival的native H8 imagination，更不是policy-gradient replay。zero174是既有consumer假设，CPU FP32不等于原GPU AMP。

数据是已见单谱系/单seed donor路径；不能与旧all452 H1、新fresh152确认或任务梯度stable组混cohort。原确认tradeoff FAIL、物理参考/任务FAIL与hidden state UNKNOWN不变。这里的误差变化不另立accuracy pass阈值、泛化或梯度资格。

- `all432_report_statistics.csv`：完整原报告统计；`all54_h8_statistics.csv`保留h8的ALL/A0/A2全部角色/单位。
- `all432_horizon_growth_from_h1.csv`：每个h/path/单位的同模型变化；`all432_within_cohort_model_comparisons.csv`：三种明确配对对照，每次只比较同h/path/单位。
- `all18_h8_path_heterogeneity.csv`：全部h8路径对照。没有按误差、成功组或flat筛行。

源报告SHA `c676c80037a056f7f51d7877bcbdaeffa5575fafb360fae3c2148ba01087deba`；Contact独立24项审计 `ab43a6e5efb150eccccc792215f3f6c4e4d22bd795b5f2bae15e83d749dfc4b8`另列，本文为作者描述而非独立复算。
