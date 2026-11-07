# 已保存人体全网格近距区域审查

全部61帧×788顶点均保留；仅统计原unsigned扫描距离，0新FK/最近点query/模型/物理。Dominant skinning仅提供可核影响区域，A/B/C不硬命名为指腹；并列unknown和所有混合权重边界均保存。

|frame42区域|顶点数|min mm|p05 mm|median mm|≤1mm|≤2mm|≤5mm|
|---|---:|---:|---:|---:|---:|---:|---:|
|root_wrist_influenced|274|45.5941|53.8119|86.3445|0|0|0|
|thumb|98|0.0318|1.8497|23.7898|4|6|10|
|index|108|0.4252|3.1254|30.5903|2|3|7|
|middle|104|28.8388|31.2889|52.7210|0|0|0|
|ring|106|48.8233|52.2280|63.2897|0|0|0|
|pinky|97|55.7177|56.9060|72.7006|0|0|0|
|unknown_cross_region_tie|1|80.4011|80.4011|80.4011|0|0|0|

frame42非thumb/index近距区域（预定阈值，完整列出）：{"1mm": [], "2mm": [], "5mm": []}

所有≤5mm顶点的hand点、原scan最近材料点与triangleID：`frame42_all_vertices_within5mm_material_points.json`；每区域min的所有exact ties：`frame42_all_region_minima_and_ties.json`。这些不是新接触目标，也不是可任意取代表点的有限接触patch。

其他tip很远不能排除同指链A/B/C或掌根表面的近距。反之，一/二/五毫米以内的皮肤采样点可能来自tracking/skin/profile/scan标定误差或穿入；原scan非watertight，unsigned距离无内外符号。convex_inside是独立凸包代理，不是真实scan signed深度。网格顶点与阈值诱导连通分量不代表连续面接触面积、独立contact数或力；61帧同谱系、非独立样本。不得直接给机器人增第三接触或宣称静态支持。

完整61帧/全部17frame+coarse区域统计、阈值顶点连通分量及所有失败范围见report.json及所列文件。旧参考和优化未改。
