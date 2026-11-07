> ## Documentation Index
> Fetch the complete documentation index at: https://docs.wuji.tech/llms.txt
> Use this file to discover all available pages before exploring further.

# 产品介绍

> [!NOTE]
>
> 本手册对应 Wuji Hand 2.2 (Beta) 非触觉版样机，相关参数与表现不代表最终产品状态。设备为 2.1 时，请查看 [Wuji Hand 2.1 文档](https://docs.wuji.tech/docs/zh/wuji-hand/beta1/overview.md)，版本判断方法见[版本识别与兼容性](https://docs.wuji.tech/docs/zh/wuji-hand/latest/version-compatibility.md)。

## 产品概述

Wuji Hand 2 是舞肌科技（WUJI TECH）研发的具有 20 个主动自由度的仿生灵巧手，面向科研实验、机器人集成、人机交互等多种应用场景。

2.2 阶段，产品提供：

* 20 DoF 仿生五指机械本体，各主动关节可独立驱动，拇指支持对指动作
* MIT 力位混合控制，控制参数由用户自行调节
* 腕部官方标配法兰，并随产品提供腕部安装图纸
* Wuji SDK 关节级接口（Python 与 C），支持基础设备发现、状态读取与控制下发
* URDF / MJCF / USD 仿真模型，已包含软体外形，采用刚体建模，暂不支持软体变形仿真

还提供基于 Wuji Glove 的[遥操作（teleop）演示链路](https://docs.wuji.tech/docs/zh/wuji-hand/latest/quick-start.md#跨产品遥操作)。仿真模型无需实机即可在 MuJoCo、Isaac Sim 等仿真器中加载预览与集成，获取与加载方式见[模型资产与加载](https://docs.wuji.tech/docs/zh/wuji-hand/latest/model-integration.md)。

## 产品参数

Wuji Hand 2 核心技术参数表涵盖结构、尺寸与重量、驱动控制和电气特性等参数，供产品选型、仿真建模和应用集成参考。

<table>
  <thead>
    <tr>
      <th>参数分类</th>
      <th>参数项</th>
      <th>典型数值或描述</th>
    </tr>
  </thead>

  <tbody>
    <tr><td rowSpan="5">结构与自由度</td><td>主动自由度</td><td>20，每指 4 个</td></tr>
    <tr><td>关节数量</td><td>20，每指 4 个</td></tr>
    <tr><td>抓指构型</td><td>可全对指，支持侧摆</td></tr>
    <tr><td>运动学构型</td><td>旋转关节串联直驱，无非线性运动学关系</td></tr>
    <tr><td>最小抓握直径</td><td>0 mm</td></tr>
    <tr><td rowSpan="2">尺寸与重量</td><td>自重\
    （整备质量，含软体与输出线缆，不含底座）</td><td>800 g（裸手 650 g，底座 70 g 另计）</td></tr>
    <tr><td>尺寸</td><td>180 mm × 80 mm × 40 mm（随手姿态变化，为参考值）</td></tr>
    <tr><td rowSpan="2">驱动方式</td><td>驱动方式</td><td>可反驱旋转直驱关节</td></tr>
    <tr><td>控制算法</td><td>FOC 矢量控制驱动</td></tr>
    <tr><td rowSpan="3">控制与通信</td><td>控制频率</td><td>1000 Hz × 20 轴</td></tr>
    <tr><td>控制模式</td><td>MIT 力位混合控制</td></tr>
    <tr><td>通信接口</td><td>RJ45 以太网，100BASE-TX</td></tr>
    <tr><td rowSpan="3">电气性能</td><td>工作电压</td><td>11–13 V DC（标称 12 V）</td></tr>
    <tr><td>电源适配器</td><td>标配 12 V 20 A（额定功率不低于 200 W）</td></tr>
    <tr><td>对外接口</td><td>XT30（电源）+ RJ45（通信）</td></tr>
  </tbody>
</table>

## 产品结构特性

本节介绍 Wuji Hand 2 整体结构设计、通信与电源接口布局，以及各自由度的命名规则与运动参数，帮助用户了解产品的机械结构与使用规范。

### 外形尺寸

Wuji Hand 2 五指完全张开并拢时的整体外形尺寸如下图所示，可用于工作空间规划与安装结构设计。

![Wuji Hand 2 外形尺寸](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/dimensions-white.png?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=54f4606d403d9375cb69c847cd2ea87e)

图中尺寸单位为 mm：长 180、宽 80、厚 40。

### 结构示意图

![结构示意图](https://mintcdn.com/wuji-0bbde051/AKOYhODgPYmD1XQo/shared/assets/source/images/zh/wuji-hand/latest/overview/structure-white.png?fit=max\&auto=format\&n=AKOYhODgPYmD1XQo\&q=85\&s=479b13995e150ee0778e075431408812)

图中 A\~E 分别对应：

| 编号 | 名称 | 英文 |
| - | - | - |
| A | 拇指 | Thumb |
| B | 食指 | Index |
| C | 中指 | Middle |
| D | 无名指 | Ring |
| E | 小指 | Little (pinky) |

图中 1\~5 对应手部常用的解剖学区域标识：

| 编号 | 名称 | 英文 |
| - | - | - |
| 1 | 远节指骨 | Distal phalanges |
| 2 | 中节指骨 | Intermediate phalanges |
| 3 | 近节指骨 | Proximal phalanges |
| 4 | 掌骨 | Metacarpals |
| 5 | 腕骨 | Carpals |

### 自由度配置

![Wuji Hand 2 自由度配置](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/dof-configuration-white.png?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=7bafb227d048429ea75f675d3c7479c9)

图中 1\~8 分别对应：

| 编号 | 关节 |
| - | - |
| 1 | thumb\_cmc\_flex |
| 2 | thumb\_cmc\_abd |
| 3 | thumb\_mcp |
| 4 | thumb\_ip |
| 5 | mcp\_flex |
| 6 | mcp\_abd |
| 7 | pip |
| 8 | dip |

### 各关节运动方向与范围

食指、中指和无名指结构一致，单指全长约 137 mm。小指略短，约 134 mm。拇指模块结构略有差异，单指全长约 140 mm。下图以拇指和小指为例，进一步说明各自由度的运动方向及角度范围：

![各关节运动方向与范围（以拇指、小指为例）](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/joint-motion-range.png?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=799bada36c25d47bbcc9a2d9bd7b1c54)

**拇指**

| 关节 | 运动方向 | 范围 |
| - | - | - |
| 1 | 掌根展收（`cmc_flex`） | -68° \~ 74° |
| 2 | 掌根侧摆（`cmc_abd`） | -85° \~ 40° |
| 3 | MCP 屈伸（`mcp`） | -60° \~ 90° |
| 4 | IP 屈伸（`ip`） | -60° \~ 90° |

**四指通用（食指、中指、无名指、小指）**

| 关节 | 运动方向 | 范围 |
| - | - | - |
| a | MCP 屈伸（`mcp_flex`） | -60° \~ 90° |
| b | MCP 侧摆（`mcp_abd`） | -40° \~ 40° |
| c | PIP 屈伸（`pip`） | -60° \~ 120° |
| d | DIP 屈伸（`dip`） | -60° \~ 90° |

四指各关节的运动范围一致，与模型文件（URDF / MJCF / USD）中的关节限位定义相同，左右手对称。

### 通信与电源接口示意图

Wuji Hand 2 通过 RJ45 接口实现与上位机的数据通信，同时支持外部电源供电。下图展示了 Wuji Hand 2 的通信和电源接口位置及连接方式：

![通信与电源接口示意图](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/connector-layout.png?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=d241ff4b4807e5d64f6e1dabffc3dc7c)

> [!NOTE]
>
> 为保障通信稳定性与供电安全，建议使用原装配套线材。

> [!WARNING]
>
> 2.2 阶段后续版本的对外接口形式会变化（不再是 XT30 + RJ45 两线直引方案），集成方案请预留改动空间。

### 三层结构：骨骼、软体、皮肤

Wuji Hand 2 由三层结构组成，分别承担承力、接触柔顺与表面防护三种角色：

| 层级 | 名称 | 作用 | 用户可操作性 |
| - | - | - | - |
| 内层 | 骨骼 | 灵巧手核心机械结构，承载关节、传动与电机，决定整手运动学与负载能力 | 整根手指可下电拆装更换 |
| 中层 | 软体 | 包覆于骨架外的仿生软组织，模拟人手软组织的硬度、压缩行程，承担拟人接触与缓冲 | 可更换 |
| 外层 | 皮肤 | 包覆软体外的被动防护层，提供表面摩擦、耐磨与外观防护，无传感能力 | 可自行拆装更换 不可脱离皮肤使用灵巧手，运行时须始终佩戴皮肤 |

> [!WARNING]
>
> 2.2 阶段指尖软体硬度仍在调整，皮肤层方案未完全锁定。当前样机的触感、外观、软体寿命表现不作为最终产品状态参考。

### 手背状态灯

手背 Logo 灯（RGB）是唯一对外状态灯，关节驱动板上的绿色单色灯在完整形态下基本不可见，仅供维修或裸机调试参考。状态灯由设备自身驱动，无需连接上位机即可生效。

> [!NOTE]
>
> 状态灯需要配备手背状态灯的硬件批次，以及固件 v2.2.0 及以上版本。

<table>
  <thead>
    <tr>
      <th>等级</th>
      <th>图示</th>
      <th>颜色</th>
      <th>模式</th>
      <th>含义</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>L0</td>

      <td>
        ![白色呼吸](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/indicator-white-breathing.svg?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=deeec2fa9f9723e63e03101bc4de7f49)
      </td>

      <td>白</td>
      <td>呼吸（1Hz）</td>
      <td>就绪：全部在线关节就绪、未使能，或未连接上位机</td>
    </tr>

    <tr>
      <td>L0</td>

      <td>
        ![蓝色呼吸](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/indicator-blue-breathing.svg?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=5dcda571c835102125f959ba80a0f95c)
      </td>

      <td>蓝</td>
      <td>呼吸（1Hz）</td>
      <td>上电 / 自检 / 预充</td>
    </tr>

    <tr>
      <td>L0</td>

      <td>
        ![白色常亮](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/indicator-white-solid.svg?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=33fb93d390ebee83c4c2b26720a26468)
      </td>

      <td>白</td>
      <td>常亮</td>
      <td>运行中：至少一个关节已使能</td>
    </tr>

    <tr>
      <td>L0</td>

      <td>
        ![蓝色慢闪](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/indicator-blue-slow-flash.svg?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=27195c7ea1bade7e3dd0dfa3284f8ca3)
      </td>

      <td>蓝</td>
      <td>慢闪 1Hz</td>
      <td>固件升级中</td>
    </tr>

    <tr>
      <td>L1</td>

      <td>
        ![黄色慢闪](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/indicator-yellow-slow-flash.svg?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=3d98282a678c55c8be27e4171ea8bf7d)
      </td>

      <td>黄</td>
      <td>慢闪 1Hz</td>
      <td>告警：过温预警、编码器标定、通信降级、节点掉线等非关键异常</td>
    </tr>

    <tr>
      <td>L2</td>

      <td>
        ![红色慢闪](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/indicator-red-slow-flash.svg?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=a87eb39f64e606e90512f57ae88716ce)
      </td>

      <td>红</td>
      <td>慢闪 1Hz</td>
      <td>故障：过流、过压欠压、驱动故障等可恢复故障</td>
    </tr>

    <tr>
      <td>L3</td>

      <td>
        ![红色常亮](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/indicator-red-solid.svg?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=cf8231306884443b48103bf8d3b8abbf)
      </td>

      <td>红</td>
      <td>常亮</td>
      <td>致命：多次过流、硬件损坏、初始化失败等不可恢复故障，错误码的 S 位为 3</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> 与 2.1 的行为差异：2.1 中固件升级的蓝色慢闪优先级高于告警的黄色慢闪，升级期间保持蓝色慢闪。2.2 中固件升级属 L0，低于 L1 告警，行为相反。

状态灯进入告警（L1）、故障（L2）或致命（L3）时的处理动作：

| 灯效 | 等级 | 处理 |
| - | - | - |
| 黄色慢闪 | L1 告警 | 解码错误码，排除散热、接线或负载原因后可继续使用，见[故障排查](https://docs.wuji.tech/docs/zh/wuji-hand/latest/troubleshooting.md#状态灯显示告警或故障) |
| 红色慢闪 | L2 故障 | 按错误码排除原因后清除故障并重新使能，见[故障排查](https://docs.wuji.tech/docs/zh/wuji-hand/latest/troubleshooting.md#状态灯显示告警或故障) |
| 红色常亮 | L3 致命 | 断电重启，复现则停止使用并联系技术支持，见[故障排查](https://docs.wuji.tech/docs/zh/wuji-hand/latest/troubleshooting.md#状态灯显示告警或故障) |

## 产品电气特性

引脚采用标准 RJ45 网口，支持 100BASE-TX。

## 内容物清单

灵巧手本体的结构、参数与接口信息见前述各章节。除本体外，包装内还随附以下内容物：

<table>
  <thead>
    <tr>
      <th>图片</th>
      <th>名称</th>
      <th>数量</th>
      <th>说明</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>
        ![电源适配器](https://mintcdn.com/wuji-0bbde051/AKOYhODgPYmD1XQo/shared/assets/source/images/zh/wuji-hand/latest/overview/power-adapter.png?fit=max\&auto=format\&n=AKOYhODgPYmD1XQo\&q=85\&s=ec24890fc2c5726315eb56124ba0b737)
      </td>

      <td>电源适配器</td><td>1</td><td>额定 12 V 20 A</td>
    </tr>

    <tr>
      <td>
        ![电源适配器 AC 线材](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/power-adapter-ac-cable.png?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=c9b2267513d4f95e06bb9bdc1f25f7ad)
      </td>

      <td>电源适配器 AC 线材</td><td>1</td><td>国标三孔插头</td>
    </tr>

    <tr>
      <td>
        ![RJ45 网线](https://mintcdn.com/wuji-0bbde051/AKOYhODgPYmD1XQo/shared/assets/source/images/zh/wuji-hand/latest/overview/rj45-cable.png?fit=max\&auto=format\&n=AKOYhODgPYmD1XQo\&q=85\&s=ec4ce2604113cb731c39ce49de383f3d)
      </td>

      <td>RJ45 网线</td><td>1</td><td>超五类网线带屏蔽，2 m</td>
    </tr>

    <tr>
      <td>
        ![XT30 公对母延长线](https://mintcdn.com/wuji-0bbde051/AKOYhODgPYmD1XQo/shared/assets/source/images/zh/wuji-hand/latest/overview/xt30-extension-cable.png?fit=max\&auto=format\&n=AKOYhODgPYmD1XQo\&q=85\&s=fd59b6e899a0b18d9ef9fdb702823185)
      </td>

      <td>XT30 公对母延长线</td><td>1</td><td>14 AWG，2 m</td>
    </tr>

    <tr>
      <td>
        ![转接座](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/adapter-mount.png?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=69d90e09aa1bd564906294436a1e492e)
      </td>

      <td>转接座</td><td>1</td><td>—</td>
    </tr>

    <tr>
      <td>
        ![转接法兰](https://mintcdn.com/wuji-0bbde051/3uhnC3cY-ox8e3b-/shared/assets/source/images/zh/wuji-hand/latest/overview/adapter-flange.png?fit=max\&auto=format\&n=3uhnC3cY-ox8e3b-\&q=85\&s=68b91728d408bd5ecaf7c941800f5f56)
      </td>

      <td>转接法兰</td><td>1</td><td>—</td>
    </tr>
  </tbody>
</table>


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.