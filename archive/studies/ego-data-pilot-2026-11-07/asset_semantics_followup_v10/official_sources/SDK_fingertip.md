> ## Documentation Index
> Fetch the complete documentation index at: https://docs.wuji.tech/llms.txt
> Use this file to discover all available pages before exploring further.

# Wuji Hand 2 指尖触觉

Wuji Hand 2 通过五路按指数据流提供指尖触觉数据。本文集中说明硬件与版本要求、自描述数据格式、Python 与 C 接口、零基线标定、状态查询、通信诊断和完整示例。

## 使用要求

本页接口适用于配备指尖触觉传感器的设备。接入前读取传感器元数据，确认设备提供相应接口，再订阅数据。

本页示例使用 Wuji SDK v2026.10.12 及以上版本。

各项能力需要下表所列固件版本或更高版本。

| 能力 | 最低固件版本 |
| - | - |
| 指尖触觉数据 | v2.1.0 |
| 调节数据流输出率 | v2.3.0 |
| 按满量程归一化分点力 | v2.4.0 |
| 触点安装位姿：`point_rpy`、`base_xyz`、`base_rpy` | v2.5.0 |
| 合力作用点与姿态：`aggregate_xyz`、`aggregate_rpy` | v2.6.0 |
| 节点离线错误：`NodeOfflineError`、`WUJI_STATUS_ERR_NODE_OFFLINE` | v2.6.0 |

下表以当前文档所述传感器规格为例，原生输出率为 100 Hz。连接设备的触点数量以 `FingertipSensorInfo.format` JSON 中的 `point_count` 为准，原生输出率以 `FingertipSensorInfo.rate_hz` 为准。

| 手指 | 触点数量 | 数据流 |
| - | -: | - |
| 拇指 | 40 | `fingertip/thumb/data` |
| 食指 | 34 | `fingertip/index/data` |
| 中指 | 34 | `fingertip/middle/data` |
| 无名指 | 34 | `fingertip/ring/data` |
| 小指 | 34 | `fingertip/pinky/data` |

订阅后可通过 `Subscription.set_rate()` 调低输出率，传入 `0` 恢复设备默认值。C SDK 使用 `wuji_sub_set_rate()`。

## 接口概览

| 操作 | Python | C |
| - | - | - |
| 读取传感器元数据 | `hand.fingertip_<finger>_info().get()` | `wuji_hand_2_get_fingertip_<finger>_info()` |
| 订阅指尖数据 | `hand.fingertip_<finger>_data().subscribe()` | `wuji_hand_2_subscribe_fingertip_<finger>_data()` |
| 整手零基线标定 | `hand.calibrate_fingertip()` | `wuji_hand_2_calibrate_fingertip()` |
| 查询单指状态 | `hand.fingertip_<finger>_status().get()`，返回 `FingertipSensorStatus` | `wuji_hand_2_get_fingertip_<finger>_status()`，填充 `WujiFingertipSensorStatus` |
| 查询指尖模块在线状态 | `hand.fingertip_online()` | `wuji_hand_2_fingertip_online()` |
| 调低数据流输出率 | `sub.set_rate(frequency_hz)` | `wuji_sub_set_rate()` |

Python 的 `fingertip_<finger>_info()`、`fingertip_<finger>_data()` 和 `fingertip_<finger>_status()` 在方法名里指定手指，`<finger>` 为 `thumb`、`index`、`middle`、`ring` 或 `pinky`。`fingertip_online()` 返回按拇指到小指排列的 5 个布尔值，表示此刻各指尖模块是否响应手部心跳，不表示模块是否安装。

> [!NOTE]
>
> `tactile_calibrate()`、`tactile_status(finger)`、`TactileType`、`TactileState` 和 `TactileStatus` 仍作为兼容别名提供，但已弃用，将在 2026.11 移除。C SDK 中对应的 `wuji_hand_2_tactile_calibrate()`、`wuji_hand_2_tactile_status()`、`WujiTactileType` 和 `WujiTactileState` 也已弃用。新代码使用上表中的 `fingertip` 名称。

## 自描述数据格式

先读取该手指的 `FingertipSensorInfo`，再按其 `format` JSON 解码 `FingertipSensorData.data`。不要固定数据长度、字段偏移、比例或单位。

```python theme={null}
import json

from wuji_sdk import WujiException

try:
    info = hand.fingertip_thumb_info().get()
except WujiException as exc:
    # 上电后立即读取、未安装传感器或传感器固件较早时属于预期情况。
    # 跳过该手指，继续解码其余手指。
    print(f"thumb: no sensor info yet ({exc})")
else:
    fmt = json.loads(info.format)
    sub = hand.fingertip_thumb_data().subscribe()
    frame = await sub.recv_async()
    if frame.info_digest != info.digest:
        info = hand.fingertip_thumb_info().get()
        fmt = json.loads(info.format)
        # 按更新后的 fmt 重建解码器
```

手指可能暂时没有元数据：未安装传感器、传感器固件较早，或上电后立即读取时手部仍在收集。此时读取元数据会抛出 `WujiException`。跳过该手指并继续解码其余手指即可。这是预期状态，不是故障。

### 元数据与数据帧

| 类型 | 字段 | 说明 |
| - | - | - |
| `FingertipSensorInfo` | `header` | 帧序号、时间戳和坐标系 |
| `FingertipSensorInfo` | `digest` | 元数据内容摘要，用于绑定数据帧 |
| `FingertipSensorInfo` | `model` | 传感器型号字符串，可能为空 |
| `FingertipSensorInfo` | `device_type` | 传感器类型 |
| `FingertipSensorInfo` | `rate_hz` | 原生输出率，当前为 100 Hz |
| `FingertipSensorInfo` | `format` | 描述数据布局与安装位置的 JSON |
| `FingertipSensorData` | `header` | 数据帧序号、时间戳和坐标系 |
| `FingertipSensorData` | `info_digest` | 生成该数据帧时使用的元数据摘要 |
| `FingertipSensorData` | `data` | 按 `format` 解码的原始字节。Python 以 `list[int]` 交付，调用 `struct.unpack_from()` 前先用 `bytes(frame.data)` 转换。C 以 `uint8_t *data` 与 `data_len` 交付 |

`format` 中的 `v`、`encoding`、`point_count`、`point_stride`、`point_fields`、`aggregate_stride`、`aggregate_fields` 和 `unit` 描述数据布局。解码前先确认 `v` 为 `1` 且 `encoding` 为 `point_array`。数据长度等于 `point_count * point_stride + aggregate_stride`，可用于校验每一帧。每个触点包含 `fx`、`fy` 和 `fz`。聚合字段提供三轴合力与温度。

归一化分点力按传感器满量程上报，法向轴范围为 0 到 1，两个切向轴范围为 -1 到 1，超量程截断。未归一化的分点力以牛顿上报。三轴合力保持以牛顿为单位，温度保持以摄氏度为单位。读取 `format` 中的 `scale` 与 `unit` 可兼容两种格式。

### 触点在手模型中的位置

触点安装位姿由 `format` 中的 `point_rpy`、`base_xyz` 和 `base_rpy` 字段描述。`format` 可以只包含 `positions`，而不包含其他安装位姿字段。仅当 `point_rpy`、`base_xyz` 和 `base_rpy` 全部存在时，才表示固件提供完整的安装位姿。`positions` 与 `point_rpy` 都必须包含 `point_count` 行数据，每行包含 3 个数字。`base_xyz` 与 `base_rpy` 都必须包含 3 个数字。如果只存在部分字段或数组形状不匹配，将格式视为无效，不要使用不完整的元数据放置触点。

`positions` 与 `point_rpy` 位于传感器模块坐标系，`base_xyz` 与 `base_rpy` 描述该坐标系相对手指 `tip_sensor_frame` 的位姿。按以下公式换算每个触点及其受力坐标轴：

```text theme={null}
point_link = R(base_rpy) * positions[i] + base_xyz
force_link = R(base_rpy) * R(point_rpy[i]) * [fx, fy, fz]
```

`R(rpy)` 按 URDF 固定轴顺序使用 roll、pitch 和 yaw：`Rz(yaw) * Ry(pitch) * Rx(roll)`。`format` 上报的位置单位为米，角度单位为弧度。两个公式都将传感器模块坐标系中的数值换算到手指 `tip_sensor_frame`。

设备上报与左右手类型匹配的安装位姿，因此左手和右手可使用相同的坐标变换。如果 3 个安装位姿字段都不存在，仍可读取力数据，但无法将触点放置到手模型中。

### 定位合力作用点

合力作用点与姿态由 `format` 中的 `aggregate_xyz` 与 `aggregate_rpy` 字段描述。`aggregate_xyz` 是传感器垫在传感器模块坐标系中的固定几何质心，`aggregate_fields` 中的合力作用于该点。该点不随每帧的压力中心变化。按以下公式换算到手指连杆坐标系：

```text theme={null}
point_link = R(base_rpy) * aggregate_xyz + base_xyz
```

`aggregate_rpy` 描述合力三轴在同一传感器模块坐标系中的姿态。按以下公式换算合力及其相对手指连杆原点的力矩：

```text theme={null}
force_link = R(base_rpy) * R(aggregate_rpy) * [fx, fy, fz]
moment_link = point_link × force_link
```

由于 `aggregate_xyz` 固定不变，`moment_link` 是合力在固定力臂下的力矩，不是实测的接触力矩。

固件未提供 `aggregate_xyz` 和 `aggregate_rpy` 时，仍可读取合力，但无法计算其作用点和力矩。缺少 `aggregate_rpy` 时，无论是否存在 `aggregate_xyz`，都只用传感器安装姿态 `R(base_rpy)` 表示合力三轴。如果 `aggregate_xyz` 或 `aggregate_rpy` 存在，但值不是 3 个数，就把整个 `format` 视为无效，不要当作该字段缺失。

## 读取与显示示例

Python 与 C 示例读取五指元数据，构建与当前 `digest` 匹配的解码器。Python 只订阅成功读到元数据的手指，C 订阅五路数据流并在显示时跳过没有元数据的手指。终端以 100 Hz 刷新同一显示区域，展示以下信息：

* 每个手指的合力、温度、接触点数量和受力最大的触点
* 每个触点的三轴力，其中 `fx` 和 `fy` 表示局部 x、y 轴切向力，`fz` 表示局部 z 轴法向力
* 受力最大的触点在手模型中的位置与力方向
* 作用点位置及合力相对手指连杆原点产生的力矩

### 帧处理与状态

Python 示例在每次刷新时清空各路订阅队列，仅显示每个手指收到的最新帧。C 示例保存每个手指的最新帧，再由主线程统一刷新显示。`info_digest` 与当前元数据不一致时，对应状态行会提示重新读取 `FingertipSensorInfo` 并重建解码器。数据长度与 `format` 不一致时，示例跳过该手指的当前帧，其他手指继续更新。

C 示例在每个手指状态行显示等待数据、数据滞后、数据流结束、数据流错误、元数据变化和数据长度异常。Python 示例遇到单指数据长度异常时也会继续运行。

### 格式校验

两个示例都要求 `aggregate_fields` 包含 `fx`、`fy`、`fz` 和 `temperature`。C 示例在订阅前校验点数、stride、字段偏移和比例值，并拒绝非有限值、非整数布局值、越界字段和超过 2048 字节的单指数据布局。

力值超出预设列宽时，表格会扩展该行，不会截断数值。

示例按 `format` 声明的单位选择接触阈值。归一化数据使用 0.02，牛顿数据使用 0.2 N。这些数值用于示例显示，不属于 SDK 数据契约。

* Python：[`3.fingertip_typed.py`](https://github.com/wuji-technology/wuji-sdk/blob/main/examples/python/wuji_hand_2/3.fingertip_typed.py)
* C：[`3_fingertip.c`](https://github.com/wuji-technology/wuji-sdk/blob/main/examples/c/wuji_hand_2/3_fingertip.c)

> [!NOTE]
>
> C 订阅回调在 SDK 工作线程上运行。回调收到的帧及其 `data` 只在本次回调期间有效。示例在回调返回前将数据复制到互斥锁保护的缓冲区，再由主线程渲染五指快照。通过示例提供的 `CMakeLists.txt` 构建，构建配置会链接数学库和 Threads/pthread。结束运行时，通过 `wuji_sub_close()` 关闭订阅。

## 零基线标定与状态查询

标定前保持五个传感器表面无负载。一次调用会按拇指到小指的顺序触发全部传感器。遇到离线手指时会立即报错，后续手指不会继续触发。返回的错误类型见 [处理传感器离线](#处理传感器离线)。

```python theme={null}
import time

from wuji_sdk import FingertipSensorState, NodeOfflineError

hand.calibrate_fingertip()

pending = ["thumb", "index", "middle", "ring", "pinky"]
deadline = time.monotonic() + 30.0
while pending and time.monotonic() < deadline:
    not_ready = []
    for finger in pending:
        try:
            status = getattr(hand, f"fingertip_{finger}_status")().get()
            if status.state != FingertipSensorState.Ready:
                not_ready.append(finger)
        except NodeOfflineError:
            # 该手指的触觉模块离线，继续等待
            not_ready.append(finger)
    pending = not_ready
    if pending:
        time.sleep(0.5)

if pending:
    raise RuntimeError(f"tactile calibration timed out: {pending}")
print("all fingertip sensors are ready")
```

调用成功表示标定命令已发送到全部手指，不表示标定已经完成。轮询五个手指的状态，直到全部为 `FingertipSensorState.Ready`，并为整个等待过程设置总超时。单次状态查询为阻塞调用，最长约 1 秒，五指轮询一轮约需 5 秒，超时只在两轮之间检查。示例将抛出 `NodeOfflineError` 的手指视为未就绪，并继续等待直到超时。

`FingertipSensorModel.Thumb` 表示 40 触点拇指传感器，`FingertipSensorModel.Standard` 表示 34 触点标准传感器。运行状态包括 `FingertipSensorState.Calibrating` 和 `FingertipSensorState.Ready`。

## 处理传感器离线

传感器离线时仍可创建对应的指尖数据订阅。如果订阅未收到数据帧，调用对应手指的 `fingertip_<finger>_status().get()`，确认传感器是否离线。

满足 [节点离线错误的固件要求](#使用要求) 时：

* Python 的 `fingertip_<finger>_info().get()`、`fingertip_<finger>_status().get()` 和 `calibrate_fingertip` 会抛出 `NodeOfflineError`。
* 对应的 C 接口会返回 `WUJI_STATUS_ERR_NODE_OFFLINE (-12)`。
* 通用 GET、SET 和 EXEC 请求的目标离线时，也会返回相同类型的结果。

不满足该固件要求时，可能报告通用内部错误。`NodeOfflineError` 继承自 `WujiException`，不继承 `RuntimeError`。已捕获 `WujiException` 的代码无需调整。为这些操作捕获 `RuntimeError` 的代码需改为捕获 `NodeOfflineError` 或 `WujiException`。

```python theme={null}
from wuji_sdk import NodeOfflineError, WujiException

try:
    hand.fingertip_ring_status().get()
except NodeOfflineError:
    print("无名指指尖传感器离线")
except WujiException as error:
    print(f"SDK 错误：{error}")
```

该结果只表示设备无法与传感器通信，不能据此判断具体原因。先确认模组连接。如果设备上电后传感器仍离线，请联系技术支持，模组可能存在硬件损伤。重复执行相同请求无法恢复通信。

## 触觉通信诊断

通过 `hand.joint_diagnostics().subscribe()` 订阅后，从 `recv()` 或 `recv_async()` 返回的每一帧读取 `JointDiagnosticsFrame.comm`：

```python theme={null}
sub = hand.joint_diagnostics().subscribe()
frame = await sub.recv_async()
comm = frame.comm
```

`comm` 包含触觉通信状态：

| 字段 | 说明 |
| - | - |
| `age_ms` | 设备内部快照距今的毫秒数。`65535` 表示从未成功采样，`65534` 表示快照已有 65.5 秒或更久 |
| `comm_get_failures` | 快照刷新失败的累计次数 |
| `tactile_online_mask` | 五指触觉在线位图，bit 0 到 bit 4 依次表示拇指到小指 |
| `tactile_response_rate_pct` | 5 元数组，每个手指的总线响应率，范围为 0 到 100。索引 0 到 4 依次表示拇指到小指，与 `tactile_online_mask` 位序一致 |
| `tactile_timeout_total` | 5 元数组，每个手指累计总线超时次数，索引顺序相同。计数达到 4294967295 后饱和不回绕，应按两次采样的差值判读 |

先用 `tactile_online_mask` 定位离线手指，再结合响应率和超时次数判断通信是否持续退化。

位图与两个按指数组都来自设备内部快照。快照不可用时，三者均为 0。把全 0 位图判定为五指离线之前，先检查 `age_ms`。`age_ms` 持续增长而 `comm_get_failures` 不变，表示 SDK 有意跳过刷新，触觉标定和固件升级期间都会如此。`age_ms` 增长且 `comm_get_failures` 同步增加，表示刷新本身失败。

`Hand2CommSummary` 完整字段见 [Wuji Hand 2 SDK 接口参考 — Hand2CommSummary](https://docs.wuji.tech/docs/zh/wuji-hand/latest/sdk-reference/diagnostics.md#hand2commsummary)。

实跑记录、能力判定方法与 SDK 接口版本兼容性见 [Wuji Hand 2 指尖触觉实践](https://docs.wuji.tech/docs/zh/wuji-sdk/latest/sdk-practice/hand2-fingertip.md)。


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.