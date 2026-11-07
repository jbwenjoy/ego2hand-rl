# 小规模真实 egocentric 数据筛选与首批下载

日期：2026-11-07。用途：为 Wuji Hand 2 的 SuperMimic + CHORD 探索准备真实手物观测。结论是先接入 HOT3D，再用 EPIC-Contact 补充接触几何。原研究 Goal 保持暂停，本轮仅筛选、下载和检查数据。

## 来源比较

| 来源 | 与本研究相关的内容 | 小样本取得方式与限制 | 本轮决定 |
|---|---|---|---|
| [HOT3D-Clips 官方说明](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md) / [官方数据](https://huggingface.co/datasets/bop-benchmark/hot3d/tree/main/train_aria) | 真实头戴视角；Aria RGB 与两路单色；训练片段提供双手、物体位姿和相机标定，另有物体网格 | 每 clip 150 帧、约5秒，独立 TAR；选择 train_aria 完整标签，避开隐藏标签的测试集 | 首选，已下载3条 |
| [EPIC-Contact 作者数据说明](https://huggingface.co/datasets/Sid2697/epic-contact/blob/main/DATASET.md) | EPIC-Kitchens 真实日常视角，手物网格及接触对应点 | 可逐 clip 取 central_frames；中心帧接触人工标注，其他帧主要为传播伪标签。相机/尺度约定须单独处理；CC BY-NC 4.0，手模型另受 MANO 条款约束 | 第二来源，先用于几何接触检查；本轮未下载 |
| [ARCTIC 官方下载文档](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) | ego/外部视角、双手、物体6D与关节、网格及相机 | 按序列获取图像，需账户；MANO/SMPL-X 另有许可。未取得认证前无法核实所选序列实际下载字节 | 后续扩展双手及关节物体 |
| [H2O 官方](https://h2odataset.ethz.ch/) / [下载脚本](https://github.com/taeinkwon/h2odataset/blob/main/download_script.py) | ego RGB-D、双手与物体姿态、模型、标定 | 官方脚本按 subject/camera/modalities 取包；需申请凭据，未核实匿名逐 clip 下载路径 | 备选，暂不拉包 |
| [HOI4D 官方说明](https://github.com/leolyliu/HOI4D-Instructions) | ego RGB-D、手物几何与时序标注 | 官方分模态存储；尚未核实可直接取单 clip 的官方小包；不以许可不一致的第三方镜像替代 | 备选，暂不拉包 |
| [EgoDex 官方](https://github.com/apple-aiml-research/ml-egodex) | 大规模真实 ego 视频与头、手、身体3D轨迹 | 官方最小 test.zip 为16GB；公开 HDF5 字段未提供本轮所需的配套物体6D/网格接口；数据 CC BY-NC-ND | 暂缓，优先解决有物体几何的小样本 |
| [OakInk2 作者数据](https://huggingface.co/datasets/kelvin34501/OakInk-v2/tree/main) | ego及外部视角、手物/相机标注 | 已查到按序列 TAR；示例RGB包约2.13GB，另需标注与模型。并非最小下载方案 | 后续扩充复杂任务 |
| [AssemblyHands 官方](https://github.com/facebookresearch/assemblyhands-toolkit) | ego 手部3D与相机 | 缺少本轮直接使用的物体6D/网格接口；不优先用于接触动力学参考 | 手部视觉前端备选 |

上表是研究适用性判断，不是数据质量排名。没有完整标注的纯视频来源也可研究，但需先增加重建和不确定性估计，现阶段会掩盖重定向环节的问题。

## 本地首批数据

数据目录：`D:\workspace\ego-data-pilot-2026-11-07`。

HOT3D 数据固定版本：`30fe9674782f32e1e5edba98476b6ff4300132c5`。下载清单分别见 [视频包清单](download_manifest_v1.json) 和 [原始网格清单](selected_models_manifest_v1.json)，记录来源、字节数和校验值。

| clip | 视频包字节数 | 源序列 | 9张抽帧中观察到的主要内容 |
|---|---:|---|---|
| 001852 | 69,744,640 | P0001_23fa0ee8 | 手机及桌面附近手部交互 |
| 001856 | 89,088,000 | P0001_23fa0ee8 | 手机、键盘及桌面附近交互 |
| 001884 | 78,817,280 | P0001_8d136980 | 白色杯子拿持/移动与桌面交互 |

三包合计237,649,920字节，含450个RGB时刻及900张单色图像。它们是同一参与者 P0001 的两条源序列切片，不能算三个独立演示者或据此报告跨人泛化；抽帧观察也不等于逐帧动作标注。选择的是下载前已固定的接入样本，没有按方法效果筛选。

额外下载 object_models 中原始 GLB：9 `mug_white`、24 `cellphone`、32 `whiteboard_marker`，连同 models_info 共16,124,189字节。仅这三种物体有本地网格；场景共出现9种ID，其他6种未下载，环境碰撞建模尚不完整。网格与视频包合计253,774,109字节（约254 MB，另有少量说明/许可/索引文件）。原始网格也需进一步检查表面、尺度和碰撞近似，下载完成不等于接触可用。

![三个片段各取首/中/末RGB](outputs/hot3d_three_clip_preview.png)

预览取每条第0、74、149帧，保留原始图像方向和鱼眼畸变。9张RGB已解码目视检查；未逐帧解码全部1350张图像。

## 已完成的检查与接入约束

[独立只读检查报告](outputs/data_source_independent_v1.json) 重核了3包的LFS大小/SHA256，逐一检查450帧与共享手形的2253份JSON、1350张JPEG头部、源序列身份和时间戳。数值均有限、双手姿态完整、时间严格递增；JPEG头部与声明尺寸一致。001852右手虽无图像框，仍有全部姿态，不能以框缺失判定手数据缺失。

后续解析须按 [固定源码证据清单](sources/independent_parser_source_v1/source_manifest.json) 执行：

1. 实际 UmeTrack `joint_angles` 为22项，`T_world_from_wrist` 为 quaternion_wxyz / translation_xyz 字典，与 README 简写不同；保留原数组，通过官方手形拓扑/FK生成几何后再做机器人优化，禁止截为20项直接对应关节。
2. `T_world_from_object` 为物体到世界，`T_world_from_camera` 为相机到世界。物体相对手腕应由 `inverse(T_world_from_wrist) @ T_world_from_object` 构造，同时核对手腕框架/镜像约定。
3. 世界位姿平移使用米；UmeTrack profile 的静态几何使用毫米并在官方解析中转换。转换只作用于相应几何字段，不能重复缩放世界平移。物体尺寸元数据也需与网格实测核对。
4. 保留原始纳秒时间戳、各路相机偏移和 FISHEYE624 标定；实测帧间隔约31.726–35.026ms，不以固定30Hz替换。若旋转/去畸变图像，必须同步变更投影约定。

官方 [HOT3D 数据许可](https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/hot3d_dataset_license_agreement.pdf?download=true) 区分序列、手部标注及物体模型：序列/非手标注为 CC BY-SA，手标注为 CC BY-NC-SA；物体模型另含禁止销售或并入待售产品的修改条款。许可原件已随下载保存。本轮未下载需另行注册的 MANO 模型文件。

## 下一轮接入的具体顺序

1. 先从手机/杯子片段各选一段可见、接触相对稳定的窗口，按官方 UmeTrack 做 FK，核对多视图重投影、手物单位与腕相对轨迹；保留可见性和源序列标识。
2. 生成 Hand2 几何重定向参考，测量指尖/手物距离、穿透、关节/速度约束。先判断固定腕是否足以表达所选任务；不能把人手腕运动直接冻结后称轨迹可行。
3. 通过几何检查后再进入有碰撞的物理回放，显式设置并扫描物体质量、惯量、摩擦，记录失稳/滑移/穿透，再决定小预算 SuperMimic + CHORD 试验。
4. 接入确认后再扩到10–20条，按 participant + source sequence 分组采样/划分，覆盖不同物体与运动，避免把邻接切片当独立训练/测试演示。

这批数据是真实 ego 原始观测及其标注；尚未运行 FK、Hand2 重定向、物理回放或新训练，没有真实 ego 方法效果可报告。已有的人工/受控数据实验结论不因此改变，旧实验的独立 ego 评测数也未更新。数据中的手物位姿、网格或接触对应点不提供真实接触力、摩擦、质量/惯量和机器人动作；只有经过物理可行性与鲁棒性检验后，才能讨论 contact dynamics 是否满足。
