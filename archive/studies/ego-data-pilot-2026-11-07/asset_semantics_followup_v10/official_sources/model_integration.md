> ## Documentation Index
> Fetch the complete documentation index at: https://docs.wuji.tech/llms.txt
> Use this file to discover all available pages before exploring further.

# 模型资产与加载

本页介绍 Wuji Hand 2.2 的模型资产、根连杆定义，以及在 MuJoCo、ROS 2 和 Isaac Sim 中加载模型的方法。

* **根连杆**：URDF / MJCF / USD 均以臂法兰安装座 `{l,r}_mount`（69 g）为根，腕壳通过 `{l,r}_wrist_fixed` 固定连接到安装座
* **执行器命名**：`{l,r}_{THJ|FFJ|MFJ|RFJ|LFJ}{0-3}`，解剖学命名对照见[关节表](https://docs.wuji.tech/docs/zh/wuji-hand/latest/control-guide.md#关节表)

腕部安装坐标系与根连杆的相对关系以 URDF 定义为准。模型格式与可用 CAD 资产随 [wuji-description 仓库](https://github.com/wuji-technology/wuji-description)提供。掌心安装接口的 STEP 与 A3 标注图纸见[机械集成](https://docs.wuji.tech/docs/zh/wuji-hand/latest/hardware-integration.md#掌心安装接口)。

## 模型资产

本页使用 wuji-description v2026.10.12 起的模型目录与包名。

| 类别 | 内容 |
| - | - |
| 模型根连杆 | 臂法兰安装座 `{l,r}_mount` |
| 仿真模型 | 左右手 URDF（含 ROS 版）、MJCF、USD、Meshes |
| ROS 2 包 | `wuji_hand2_2_description`，包根 `hand2/hand2_2/body/` |

已提供包含软体外形的刚体模型，暂不支持软体变形仿真。

模型已包含臂法兰安装座，不再提供独立的 `_with_mount` 变体。

模型命名为 `wujihand2_2-{side}`，USD 入口为 `wujihand2_2.usd`。

## 环境要求

| 用途 | 要求 |
| - | - |
| MuJoCo 可视化 | Python 3.8 及以上，MuJoCo 2.3 及以上 |
| ROS 2 可视化（RViz） | ROS 2 Lyrical，Ubuntu 26.04 |
| Isaac Sim USD | Isaac Sim 2024.x 及以上 |
| STEP / CAD | 支持 STEP 格式的 CAD 工具，例如 FreeCAD、SolidWorks |

## 获取与加载

```bash theme={null}
git clone https://github.com/wuji-technology/wuji-description.git
cd wuji-description
```

MuJoCo：

```bash theme={null}
# 右手
python -m mujoco.viewer --mjcf=hand2/hand2_2/body/mjcf/right.xml

# 左手
python -m mujoco.viewer --mjcf=hand2/hand2_2/body/mjcf/left.xml

```

MJCF 的碰撞几何取各链接 mesh 的凸包，27 个带网格的链接参与碰撞，按模型配置排除 15 对碰撞。指间接触与指尖到手掌的接触保持有效。显示分组为组 1 视觉外观、组 2 碰撞几何、组 3 指尖 site。

ROS 2 与 RViz 的依赖安装、描述包安装和启动方法见 [ROS 2 集成](https://docs.wuji.tech/docs/zh/wuji-hand/latest/ros2-integration.md)。

Isaac Sim USD：

| 变体 | 路径 |
| - | - |
| 左手 | `hand2/hand2_2/body/usd/left/wujihand2_2.usd` |
| 右手 | `hand2/hand2_2/body/usd/right/wujihand2_2.usd` |

USD 为分层资产，包含 base、physics、robot、sensor 子层与 Logo 纹理。保留每个 `usd/{left,right}/` 目录的完整内容。根 prim 为 `wujihand2_2_{side}`。点击 **Play** 时驱动增益保持当前位姿。

URDF 预览需要 Rust/Cargo，安装要求见 [urdf-viz 官方说明](https://github.com/openrr/urdf-viz#install)。

URDF 预览：

```bash theme={null}
cargo install urdf-viz
urdf-viz hand2/hand2_2/body/urdf/right.urdf
```

## 验证

1. 确认 MuJoCo 或 URDF viewer 正常打开，没有缺失网格
2. 确认模型显示 20 个 revolute 关节
3. 确认所用目录与 ROS 2 包名对应 2.2

模型加载成功说明资产、路径与可视化链路可用。固件、网络、传感器与真实设备控制需要接入设备后单独验证。


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.