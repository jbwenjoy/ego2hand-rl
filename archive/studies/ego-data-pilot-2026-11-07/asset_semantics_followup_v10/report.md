# Hand2 functional-pad semantics: official source follow-up

结论：**官方资料能确认组件/分层角色，但尚不能把当前凸包的某个面标为已验证的功能接触面。** 172/108 mm²大平面究竟是安装面还是工作面仍为 UNKNOWN，不能按面积、照片或`material="sensor_pad"`颜色名决定。

| 固定来源 | 版本与时间 |
|---|---|
| 本地vendor声明 | `wuji-description`本地封装0.1.0，资源pin `829c9aa27cb12c3a49d5b8d63c9869dd8f656702`，上游release2026.10.12 |
| 当前官方仓库 | `5c4dfa16917355d073e2e1d7e3d03897b6cf9508`，release2026.10.26 |
| 官方SDK示例 | `b6be39b1c923b822f627ac8564dd91ceb38b303b`，commit2026.10.26 |

本次逐blob核对五个右手`*_tip_sensor_frame.STL`，本地、829c与当前5c4完全一致。官方两版`right.xml`也字节相同；本地副本仅CRLF换行不同。因而更新到当前公开裸手资源不会自动补出新碰撞表面。原XML把整件sensor-frame mesh用于视觉与碰撞，只有腕部IMU传感器定义，没有功能facet mask或可变形垫模型。[固定MJCF](https://github.com/wuji-technology/wuji-description/blob/829c9aa27cb12c3a49d5b8d63c9869dd8f656702/hand2/hand2_2/body/mjcf/right.xml)

官方把Hand2.2的sensor-frame描述为焊接在末节的指尖触觉垫组件。当前README的外观变体共享碰撞几何，历史完整STEP也不等同当前仿真外形，不能混用版本来赋予工作面语义。[官方README](https://github.com/wuji-technology/wuji-description/blob/5c4dfa16917355d073e2e1d7e3d03897b6cf9508/README.md)

硬件说明明确区分承力骨骼、接触缓冲软体、提供摩擦的外皮肤，并要求运行时佩戴皮肤；但该页限定于非触觉2.2 Beta样机，指尖硬度和皮肤方案尚未定型。模型仍用刚体近似软体外形。这些说明不能提供当前STL具体facet、实测μ或压缩刚度。[产品说明](https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview)、[模型合同](https://docs.wuji.tech/docs/zh/wuji-hand/latest/model-integration)

可补语义的明确接口是匹配实机的`FingertipSensorInfo.format`：完整`positions/point_rpy/base_xyz/base_rpy`可注册触点及受力轴到`tip_sensor_frame`，须绑定设备版本和digest。本次没有对应数值；轴方向也不能未经验证直接当凸包外法向。合力位置是固定质心，不是动态压力中心或实测接触力矩。[官方触觉合同](https://docs.wuji.tech/docs/zh/wuji-sdk/latest/wuji-hand-2-tactile)

因此保留当前冻结模型；任何暂选接触面应标为engineering hypothesis。取得同版本设备的完整安装元数据或明确分面CAD后，才能进一步注册功能区域。未下载STL/CAD，未安装SDK或连接设备；仅保存约1.09 MB文本/版本元数据，另有657 B二进制USD sensor子层未语义解码、不用于否定证据。没有模型/FK/LP/物理调用或外部消息。全部URL、哈希、版本比较与未找到项在`report.json`及`source_manifest.json`。
