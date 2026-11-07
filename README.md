# SuperMimic + CHORD 统一研究仓库

这里是 Wuji Hand2 研究的后续开发入口。共享实现只维护 `src` 中的一份，
实验差异通过配置表达。GPU 性能优化从这里继续，不逐份修改历史 study 副本。

|目录|用途|
|---|---|
|`src/supermimic_research/runtime`|state58、SO(3)、动量、结构化解码、几何、任务、原生读取与评测|
|`src/supermimic_research/learning`|世界模型、Actor 想象、共享更新逻辑、数据读取和离线入口|
|`src/chord_study`、`src/dex_supermimic`|唯一维护的 CHORD 与 Dex 源码包，保留原 namespace/NOTICE|
|`configs`|实验参数、数据角色和外部资产索引|
|`archive/studies`|24 个旧目录的源码、配置与文档快照；只作档案|
|`provenance`|逐文件来源/SHA、原 Git 状态及未提交补丁、迁移验证记录|

本次归档包含 6,002 个文件，约94.16MiB，逐文件与原件字节一致。环境、缓存、
数据、权重、编译模型和视频继续在外部保存，未复制数十GB内容进 Git。
旧目录仍保留原路径，因为历史协议把绝对路径和源码哈希也计入证据。
本次没有删除、移动或以 junction 替换旧树；后续活跃维护以本仓库为准。

## 使用

在仓库根目录，用现有 Hand2 解释器启动独立进程，避免改写仍被旧实验依赖的 editable 安装：

```powershell
& ..\dex-supermimic-research\.venv\Scripts\python.exe scripts\research.py doctor
& ..\dex-supermimic-research\.venv\Scripts\python.exe scripts\research.py artifacts --verify
& ..\dex-supermimic-research\.venv\Scripts\python.exe scripts\research.py train --config configs\phone_offline.toml --mode inspect
& ..\dex-supermimic-research\.venv\Scripts\python.exe scripts\research.py train --config configs\phone_offline.toml --mode smoke
```

`inspect` 不加载学习或仿真后端；`smoke` 是有界 CPU 前向，0优化器、0训练更新。
示例保留三指4条+原TRAIN4条的WM混合批次，Actor/Critic只用指定三指TRAIN。
真实训练配置默认关闭，不会因检查仓库而启动。新离线入口具有自身 schema 的恢复路由；
它不会把旧冻结检查点伪装成新代码下的同一次实验恢复。

完整原生 online 采集编排目前仍在历史档案中；共享任务、读取器、评测和更新核心已抽出，
后续编排应薄适配到这些公共接口。没有重跑全部旧实验或宣称它们都已完成重新验收。

## 已验证的迁移范围

- 全部6,002份归档的字节、SHA及4份Git工作树补丁通过核查，重复导入幂等。
- 两个固定真实上下文的73项状态/几何/任务输出比较最大差值0，输入AD一致。
- 同权重WM raw222/head178/decoded58及动作输入AD对照差值0；Actor/Critic来源与保存状态一致。
- 原生零控制循环的两dt、共1171步，每例31个trace数组逐字节一致，原失败门保持一致。
- 数据路径重定位、内容哈希及越界检查通过；迁移检查未修改任何研究数据角色。

验证入口：[runtime](provenance/runtime_validation_final.json)、
[learning](provenance/learning_parity.json)、[native](provenance/native_validation.json)。
详见 [架构](docs/architecture.md)、[环境边界](docs/environment.md)、
[性能工作清单](docs/performance-work.md)及 `provenance` 中的记录。
迁移验证只支持其明确覆盖的接口，不是新模型、物理梯度或策略收益证书。

原固定1000训练及原生评测已结束，[研究结果](docs/research-results-2026-11-07.md)
单独保留：四个策略案例均未通过两秒持握。合仓与后续性能工作不改变该结论。
