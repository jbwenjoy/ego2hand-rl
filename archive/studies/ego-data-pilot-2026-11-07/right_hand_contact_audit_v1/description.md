# 右手接触分配审查（官方人体FK）

帧42–102保留全部61行，实际可用61、missing0。官方人体FK一次batch调用、61行；原scan最近triangle点查询49288。0机器人FK/MjData/NN权重/学习/积分/GPU。

右手hand_side=1；原始proper wrist pose按官方API输入，官方代码内部处理第一列反射及face winding翻转。没有自行镜像、copy左手pose或22→20截断；22角完整传入，官方skinning内部仍只用前20。

|frame42粗区域|顶点数|min mm|≤1/2/5mm顶点数|
|---|---:|---:|---:|
|root_wrist_influenced|274|51.72920|0 / 0 / 0|
|thumb|98|36.48617|0 / 0 / 0|
|index|108|24.67411|0 / 0 / 0|
|middle|104|15.39395|0 / 0 / 0|
|ring|106|35.00762|0 / 0 / 0|
|pinky|97|48.34679|0 / 0 / 0|
|unknown_cross_region_tie|1|76.67974|0 / 0 / 0|

全61帧逐区域最小值、分位数、1/2/5mm数量与逐段持续性见all61_region_temporal_summary.json；frame42的全部≤5mm右手点/phone材料点见frame42_all_near5mm_material_points.json。

这只是可见/重建人体与扫描表面的几何近距；不是传感接触、力或独立支持点证书。原scan不封闭、距离无符号，可能穿入或受tracking/skin/profile/scan误差影响；region仅dominant蒙皮影响。缺帧未插值。连续近距样本只报告时间戳跨度，不能证明采样间实际接触。即使右手也近距，仍不能推定其负担了多少重量；右手较远也不能证明unimanual受力。旧左手参考、机器人目标与当前优化均未改。
