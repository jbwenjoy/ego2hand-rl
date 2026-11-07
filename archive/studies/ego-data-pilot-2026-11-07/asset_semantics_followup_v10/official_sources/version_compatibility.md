> ## Documentation Index
> Fetch the complete documentation index at: https://docs.wuji.tech/llms.txt
> Use this file to discover all available pages before exploring further.

# 版本识别与兼容性

> [!NOTE]
>
> 本页两版通用，Wuji Hand 2.2 (Beta) 与 Wuji Hand 2.1 (Beta) 各有一份。Wuji Hand 2.1 (Beta) 的文档见 [Wuji Hand 2.1](https://docs.wuji.tech/docs/zh/wuji-hand/beta1/overview.md)。

## 判断设备版本

推荐先查看序列号。序列号第 5–7 位为 `A00` 时，设备为 Wuji Hand 2.1 (Beta)。第 5–7 位为 `A01` 时，设备为 Wuji Hand 2.2 (Beta)。

## 硬件与软件兼容矩阵

| 组件 | 2.2 (Beta) | 2.1 (Beta) |
| - | - | - |
| 供电与接口 | 11–13 V DC 输入（标称 12 V）。XT30 + RJ45 两线，详见[用户须知](https://docs.wuji.tech/docs/zh/wuji-hand/latest/user-notice.md#电源使用安全) | 仅 12 V DC，其他电压不兼容。接口与 Wuji Hand 2.2 (Beta) 相同，详见[用户须知](https://docs.wuji.tech/docs/zh/wuji-hand/beta1/user-notice.md#电源使用安全) |
| 固件 | v2.0.0 起，当前 v2.5.1 | 停留在 v1.2.4，不再接收更新 |
| Wuji SDK | 可升级至最新版 | 可升级至最新版，部分能力不可用（见下文） |
| Wuji CLI | 可用，按硬件型号推送匹配的固件包 | 与 2.2 (Beta) 相同 |
| Wuji Studio | 已归档，停止维护。固件升级改用 Wuji CLI | 与 2.2 (Beta) 相同 |
| 仿真模型 | `hand2/hand2_2/body/`，ROS2 包 `wuji_hand2_2_description`，模型命名 `wujihand2_2-{side}`，USD 入口 `wujihand2_2.usd`，指尖新增软体连杆。两个包名不同，可与 2.1 在同一工作空间共存。加载方式同 Wuji Hand 2.1 (Beta) 各链接，资产清单见[模型资产](https://docs.wuji.tech/docs/zh/wuji-hand/latest/model-integration.md#模型资产) | `hand2/hand2_1/body/`，ROS2 包 `wuji_hand2_1_description`，模型命名 `wujihand2_1-{side}`。加载方式见[获取与加载](https://docs.wuji.tech/docs/zh/wuji-hand/beta1/hardware-integration#获取与加载) |
| 手背状态灯 | 支持，L0–L3 等级制，需固件 v2.2.0 及以上（见[产品介绍](https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview.md#手背状态灯)） | 支持，优先级序号制（需相应硬件批次） |

> [!NOTE]
>
> Wuji Hand 2.2 (Beta) 的固件与 Wuji SDK 需按各自发布记录配套升级，版本要求见对应发布记录。Wuji SDK 的接口在 Wuji Hand 2.2 (Beta) 与 Wuji Hand 2.1 (Beta) 设备上通用，以下能力受硬件条件限制，2.1 设备升级固件与 SDK 后仍不可用或行为不同。

## 手背状态灯

| 项 | 2.2 (Beta) | 2.1 (Beta) |
| - | - | - |
| 灯效体系 | L0–L3 等级制 | 优先级序号制 |
| 硬件前提 | 标配 | 需相应硬件批次 |
| 未配备状态灯硬件时 | — | 无灯效输出，接口调用不报错 |

Wuji Hand 2.2 (Beta) 的等级制灯效需固件 v2.2.0 及以上，灯效含义见[产品介绍的手背状态灯](https://docs.wuji.tech/docs/zh/wuji-hand/latest/overview.md#手背状态灯)。

## 内部字段差异

`Hand2CommSummary` 的设备内部字段在 Wuji Hand 2.1 (Beta) 上依赖固件版本，具体版本门槛暂不提供，将在后续更新。字段定义见 [SDK 接口的数据类型](https://docs.wuji.tech/docs/zh/wuji-hand/latest/sdk-reference/diagnostics.md#hand2commsummary)。

## Wuji Hand 2.1 (Beta) 用户升级 SDK 后的能力限制

* 手背状态灯：未配备状态灯硬件的设备无灯效输出


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.