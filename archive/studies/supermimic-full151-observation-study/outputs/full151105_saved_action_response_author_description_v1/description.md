保存数据响应分解：作者描述（非独立审计）

全部8snapshot、152branches、128端点、64pairs保留。新指标只描述actual-coordinate finite secants；不替换旧nominalFD或5/9 taskscalar结果。

| block | display units | baseline bias RMS | physical centered response RMS | model centered response RMS | response error RMS | error/physical RMS |
|---|---|---:|---:|---:|---:|---:|
| q | mrad | 23.3746864 | 0.0990911846 | 0.0933283703 | 0.0901081284 | 0.9093455558223618 |
| qd | rad/s | 1.13373952 | 0.00540906954 | 0.00407394923 | 0.0044354853 | 0.8200089255412393 |
| p | mm | 3.65359635 | 0.009148559 | 0.00402107174 | 0.00917242981 | 1.0026092429421853 |
| R | dimensionless | 0.0406809181 | 0.000338064977 | 0.00010433247 | 0.000343622322 | 1.0164386877280995 |
| v | m/s | 0.106544652 | 0.000545129989 | 0.000204875196 | 0.000533206686 | 0.978127595284366 |
| omega | rad/s | 3.09748646 | 0.0279248813 | 0.00574739415 | 0.0281382963 | 1.0076424703195848 |

| block | physical coordinate secant RMS | model coordinate secant RMS | secant error RMS | error/physical RMS |
|---|---:|---:|---:|---:|
| q (rad/control rad) | 0.0629490307 | 0.0606385403 | 0.057935737 | 0.9203594773243093 |
| qd (rad/s/control rad) | 3.437241 | 2.62606286 | 2.83157471 | 0.8237928939290224 |
| p (m/control rad) | 0.00498797849 | 0.00271372745 | 0.0049905797 | 1.00052149635052 |
| R (dimensionless/control rad) | 0.11051875 | 0.0715205658 | 0.107403353 | 0.9718111439265853 |
| v (m/s/control rad) | 0.285900234 | 0.134974828 | 0.277673985 | 0.971226856912448 |
| omega (rad/s/control rad) | 8.49110446 | 3.8541539 | 8.37470104 | 0.9862911330939345 |

全128端点机械标记：6个完整20维actualdu数值全0，6个actualu原始bits与u0全相同。所有非零控制行也保留，零控制行和三个baseline的完整58维原始编码差分别存于NPZ；不得把这些有限重复的范围宣称为真实noise界。

all8_by6_blocks.csv保留全部snapshot分组，all128_endpoints.csv保留每一个端点。ratio没有epsilon/floor，physical RMS=0时为None。
没有新模型、forward、AD、score、MJB/FK/query、GPU、Env、物理或训练；后续假设应由Root决定，本描述不创建新prereg/实现。
