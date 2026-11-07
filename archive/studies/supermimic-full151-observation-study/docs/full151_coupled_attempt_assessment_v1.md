# full151 有限耦合尝试：结果边界与下一步

本轮8次SCP保持 **FAIL**，没有READY、物理资格或训练授权。7次接受、major6拒绝后信赖域从.15缩为.075rad，最后来源是major8，不是历史择优。全部LP原始线性数值门通过；最终仅η门与真实采样几何门失败。[独立来源/终态审计32/32](D:/workspace/supermimic-full151-observation-study/outputs/full151_SCP_saved_lifecycle_independent_v1/report.json)、[独立原始数学审计173/173](D:/workspace/supermimic-full151-observation-study/outputs/full151_scp_saved_independent_v1/report.json)。

103份原不可变artifact/source精确匹配。唯一`live_state.json`在receipt收集hash后由LP_RUNNING改成TERMINAL，旧pin已过期；[说明sidecar](D:/workspace/supermimic-full151-observation-study/outputs/full151_contact_coupled_scp_v1/terminal_artifact_inventory_v1.json)保留两hash，不能声称原104项pin全匹配。Root工具观察记录once/closed/exit1，但PID/Creation未测、完整外层wall未知。内部wall10.811131s包含solver3.689296s，不能相加。

## 目标下降允许了类别间退化

最大手物穿透15.158818→11.447693mm，超3mm样本350→338；最大自碰撞5.884525→11.431469mm，超标16→38。最终η=8.421356mm，远大于附加1nm门；实际全局planning violation为8.448693mm。Φ从122.009731降到85.103095，但直接fit/smooth cost从.411554升到.616162。

[driver](D:/workspace/supermimic-full151-observation-study/scripts/run_contact_coupled_scp_cpu_v2.py)用`F+1e4*max(0,全类别最坏穿透−2.999mm)`，单一η同时覆盖手物和自碰撞。它允许在更大的手物峰下降时，自碰撞明显增加但仍低于新最坏值；这与已观测轨迹一致。**不能称总体几何提升，更不能称动力学或CHORD收益。** 它也不证明单一max是自碰撞恶化的唯一原因；局部feature、系数范围及控制约束等尚未独立隔离。

8步预算耗尽不支持“多迭代就会通过”。下一次几何研究若进行，应冻结独立新条件：分开类别可行性/退化约束、检查trust局部模型失配及固定初态/系数范围的结构限制，保存受限不可行与数值失败。不要自动增加iteration、调权重搜索或选择历史最好点；不要把局部两branch FD资格升级为所有601点或全局Jacobian资格。

## 首要并行重点：真实 native151 known-rough 接入

原目标是SuperMimic/CHORD通过动力学修正粗观测目标，**不应无限等待几何参考本身完美openloop**。当前计划控制/原始数值门通过但几何不合格，可在一个明确、独立授权的`known-rough full151`输入合同中继续研究。必须保留本次FAIL及穿透证据，不把未出CPU_SAMPLE_READY偷改成通过。

最小真实接线仍是：[新mode草案](D:/workspace/supermimic-full151-observation-study/docs/full151_mode_contract_draft_v1.md)中151state/150action、phase149→terminalobs150、无u150、finite u149 dead placeholder；新baseline150+原EMA残差/hard∩ctrl/单C2；解析qd151/物体world twist权威与nativeFD诊断分开。新BODY21必须由新派生q准确FK导出再经过真实Reader CPU链，不能拿宏21点改名；不能直接加载旧104/103/3 checkpoint或继承prefix物理资格。

随后在新native151实现中验证selected reset/obs history、H8 survival与dead critic norm/loss、replay terminal保留与generation、全batch cache/rawctrl/replay.u/WM/PD一致、新checkpoint签名与原actual8。先做一次同native cadence、同初态、无辅助的完整3s A0工程与独立保存审计；质量可以失败，但身份/容量/完整性/来源不能伪通过。之后才另行决定fromscratch小smoke及同预算学习，不能由本短评授权。

A2的独立151接触意图仍缺。观测近表面cue与unknown human contact不能补成真值；旧16意图不能tiling。若采用显式robot-intent工程假设，必须先独立冻结来源/phase/mask，再由所有对应条件共同使用。没有有效support目标的消融不能冒称非退化A2。此项可与native151工程接线并行，不必让缺意图阻断A0接口验证。

## RF16前段为何不能直接复制，以及一个未来结构选项

[独立限制映射诊断](D:/workspace/supermimic-full151-observation-study/outputs/RF16_full151_basis_embedding_readonly_v1/report.json)只读取已有派生控制点/knots并做BSpline线代。P0/P1相同不等于路径相同：prefix在.3s是4重端点，full在.3s是单重内部knot；knots首差index19为.3/.32。直接复制18控制点，basis从>.26s开始不同，首个不同参考结点为.28s。

存在精确restriction映射：F0..15=RF0..15，F16=1.5RF16−.5RF15，F17=6RF17−6RF16+RF15。它重现[0,.3]的q/qd/qdd到浮点舍入误差，但F17的joint3/joint7超当前full系数hardbox47.065/110.639mrad。因此**相同RF前缀与当前充分系数证书不兼容，不等于q(t)越物理关节界或运动不可行**。没有构造P18..152或新的U150，也没有导出warmstart候选。

后续若研究此结构，可用逐段cubic q的4个Bernstein系数范围，或经独立认证的严格极值根审计，替代过强的全局controlpoint box，**保持原物理joint界**。已有[纯线代补充](D:/workspace/supermimic-full151-observation-study/outputs/RF16_full151_basis_embedding_readonly_v1/Bernstein_range_supplement.json)显示映射前15段与旧RF的Bezier值差2.22e−16rad、各段原joint界超量0，而全局CP仍超110.639mrad。这只是充分非必要证书的结构证据，不是新full151可行轨迹。

改变范围证书需要新source/solver/postaudit协议；不retro豁免旧gate，不clip映射系数后声称仍是同一前缀，不直接splice q samples或未知尾段，也不继承prefix物理成功。它是未来单一方法研究选项，不能取代眼下native151 known-rough实际接入的优先级。

本文只读取既存来源与保存数组并做小规模basis线代；没有新增LP/优化/FK/碰撞/Env/WM/物理/GPU执行，没有修改旧源码或失败数据。
