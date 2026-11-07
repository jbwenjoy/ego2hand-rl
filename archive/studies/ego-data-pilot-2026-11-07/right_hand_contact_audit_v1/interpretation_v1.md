# 右手是否也接近手机

全部61帧（42–102）都有right UmeTrack pose，没有缺帧插值。原始22角完整保留，官方内部skinning使用前20；hand_side=1的镜像列和面绕序处理由冻结官方API执行，调用者没有自行反射。实际只调用一次官方人体FK batch61，原手机扫描nearest-triangle点查询49,288个，0机器人FK/MjData/NN权重/学习/积分/GPU。

**Frame42没有≤5mm的右手网格顶点。** 最小值是middle影响区域15.393948mm；其他区域最小值为index24.674113mm、ring35.007621mm、thumb36.486169mm、pinky48.346789mm、root/wrist51.729197mm。

全部61帧的右手最小值为 **2.618969mm**，来自right index。≤1mm和≤2mm的采样帧数均为0；≤5mm只有6帧，且均为index影响区域：

|frame|最小unsigned距离mm|≤5mm顶点数|
|---|---:|---:|
|46|3.439109|4|
|51|2.618969|8|
|52|4.189382|2|
|57|4.788581|2|
|58|3.428015|4|
|64|3.741161|4|

按相邻观测帧的预定规则，四段为46、51–52、57–58、64，最长仅两个样本，首尾时间戳跨度约33.3275ms。这不是持续接触时长：采样间接触没有被观测。全部其余区域在全窗口仍>5mm（middle最小11.712450mm、thumb24.003044mm、ring32.368002mm、root/wrist41.290729mm、pinky46.822055mm）。

因此，现有右手网格**未提供frame42接触或全窗口持续贴合/承重的几何证据**；也不能据此认证unimanual。unsigned扫描距离无穿透符号；scan不封闭、连续三角形之间的接近未穷尽，tracking/skin/profile/scan误差和未建模表面仍有限制。2–5mm候选更不能自动当作真实支持点或力。

完整逐帧/逐region分位数、持续性、所有22角和61×788最近材料点在 [report.json](report.json) 与 `right_geometry_and_proximity61.npz`；frame42≤5mm材料点表为空是该阈值下的观测结果，非missing。报告SHA `1322d1f30f029dea63db1c3362a0ebf425c15ab381965a81d2426a4023286f99`。旧左手参考及机器人目标均未改。
