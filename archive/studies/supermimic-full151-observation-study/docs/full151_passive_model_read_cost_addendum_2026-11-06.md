# 被动 MJB 读取成本补充

本补充更正成功路径中的嵌套读取计数。原 1808 行结果、35 项数值审计、源码、配置和账本均保留，
不重新执行或更改几何数值。两份独立源码审阅得到相同调用图。

`full151_native.bundle.load_bundle()` 每次都会调用冻结 `prefix_control.load_actual_pd()`，后者新建
一个 `MjModel.from_binary_path()`。`sys.modules` 缓存的是辅助模块，没有缓存模型返回值。

| 路径 | 来源 | 正常完成路径的被动 MJB 读取 |
|---|---|---:|
| 已执行 denseP | caller 的 bundle/PD 验证 | 1 |
| 已执行 denseP | 显式 geometry catalog/model 常量读取 | 1 |
| 已执行 denseP | factory 内的 bundle/PD 验证 | 1 |
| 已执行 denseP | factory 内的显式 model 常量读取 | 1 |
| **denseP 合计** | 工厂子计为 2；原显式 counter 仍只记录其中一项 | **4** |
| 下一 452 WM | caller 的 bundle/PD 验证 | 1 |
| 下一 452 WM | strict loader → Actor → bundle 重验 | 1 |
| 下一 452 WM | Actor 验证的显式 model 常量读取 | 1 |
| **452 WM 计划合计** | strict loader 内部子计为 2 | **3** |

这些总数是完整成功路径的源码推导，未冒充全局原生 API 插桩观测。原 denseP 的
`passive_mapping_MJB attempted=returned=1`、factory API=1、前向=15 的局部插桩计数仍正确；
旧文字“factory 内部被动 MJB 1 次”漏掉了 bundle/PD 重验，旧成本账本也未覆盖两次间接读取。
原 62.002 秒 worker 时间已经包含全部调用，没有每次读取的独立耗时可供分摊。

本补充没有新增 MJB 加载、物理步或模型执行。1808 行、44/42/2 统计、全部原算术门槛和梯度/
任务资格不变。下一 452 协议已另存 v2，明列 caller1 + strict 内部2 = 3，不省略任何校验。

- [独立源码成本补充（9 项）](D:/workspace/supermimic-full151-observation-study/outputs/full151105_passive_MJB_cost_source_addendum_v1/report.json)，SHA `d53b3d27c5e930238bdb580a592dab66f991eb1e77d49cd2551bbfca952cb5b6`
- [另一份源码调用图复核](D:/workspace/supermimic-full151-observation-study/outputs/full151105_passive_MJB_callgraph_source_review_v1/report.json)，SHA `98c3d3917acac0942a3d031f4a8a6d294da090dc33abb7068349a3c30198ee6e`
- [新 452 prereg v2](D:/workspace/supermimic-full151-observation-study/protocols/full151105_state_prediction452_CPU_prereg_v2.json)，SHA `37652718ed150421ed84d25bcafc14b27a1e7390a7dee7e729a860b3d91ef411`
