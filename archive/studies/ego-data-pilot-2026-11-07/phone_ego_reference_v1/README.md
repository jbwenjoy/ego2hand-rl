# 61帧物体相对运动参考

`reference.py` 是可调用的纯NumPy接口，不创建网络或仿真器。输入为HOT3D clip1856、42–102帧的原始left UmeTrack wrist与phone24 pose。原始近单位quaternion明确归一化为F64 proper旋转；不从FP32 mesh变换投影R、不镜像物体或改几何尺度。

设 `T_wo=inv(T_world_wrist)@T_world_phone`，保存 `D[k]=inv(T_wo[0])@T_wo[k]`。给定具名机器人起始物体pose `T0`，输出 `T_robot[k]=T0@D[k]`；物体COM另加实际编译模型的BODY ipos。这只重定位物体相对运动意图，human left到robot right的手指对应、接触保持与关节重定向仍需独立求解。

原61个整数`ref_timestamp_ns`和frame ID全部保留，真实时长为2.00000849s。20ms网格有101点，覆盖0–2s；末端8.49µs仍保留在原61帧轨迹，不被拉伸/外推或冒充完整20ms目标。插值采用初始phone坐标中的平移线性插值和SO3短弧插值，拒绝区间外查询。有限区间COM速度与geodesic角速度只是参考描述，不是瞬时速度/力真值。

`build_reference_v1.py`实际运行11/11检查通过：相对pose往返、首帧锚定、世界坐标变化不变性、纯wrist transport消除、61个源采样插值、越界拒绝、proper handedness检查。导出两个具名压缩候选的参考，并保留各自初始几何资格；失败起始状态不会因参考导出而获得物理资格。

这份结果使用真实ego数据的标注pose，尚未从RGB估计重建，也未执行61帧机器人跟踪。它不是完整hand state58/关节参考、接触标注或成功示范。源重力注册、材料与动力学参数仍未知；机器人固定腕仅追踪wrist-relative物体意图，不复现原世界中的整手运输。当前神经训练数据仍是仿真采集数据。
