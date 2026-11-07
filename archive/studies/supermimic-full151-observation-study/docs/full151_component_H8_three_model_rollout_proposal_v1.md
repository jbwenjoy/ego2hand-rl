# 探索性三模型 H8：固定 donor 控制下的状态积累误差

2026-11-06。**PROPOSAL ONLY；未实现/拼接 windows/运行模型，非执行授权。** 保留[两模型源核验](D:/workspace/supermimic-full151-observation-study/docs/full151_component_H8_donor_source_feasibility_v1.md)（SHA `38ba68b53df526791773599b480be93e90701e4341848f2a8da782aa4e76028d`）。Root 拟在正式冻结后解除 H8 parked；confirmation FAIL 与新 task-gradient 27项结果不因本提案改变。

**为何加入一个初始模型。** 最小两模型方案可比较 pointwise 与 +response final1000；新增未修改的 initial A2 WM，能同时描述“离线校准整体相对起点的多步代价”以及两种校准损失的差别，避免只在两个已变化模型间比较。它检查可能的多步退化/遗忘现象，不以三条曲线唯一归因。确认测试已显示 response MSE 改善伴随 point loss 增加；新物理 task-gradient 仅共享6个 stable组、两model均6/6符号而 response 幅值误差更大，不能当作 H8 已准确或与旧9组混合。

|明确 artifact role|同一训练输出目录中的文件|文件 SHA256|
|---|---|---|
|`initial_A2_unmodified_reference`|`initial_component_state.pt`|`ea051225b7279fc69fbd97725eb15462faaf4b3a6706881f8fe4a05b658a14bf`|
|`pointwise_final1000`|`pointwise_update_1000.pt`|`54d81c0e1d50e24510a985c90944af328e40f4a69d1c2e1987130af4ea3a719e`|
|`pointwise_plus_response_final1000`|`pointwise_plus_common_pre_response_update_1000.pt`|`8e27eaec850d7d0adfd8e50cf426ba8f3414ae58379cefc2af43b4387a59e3e3`|

目录为 [full151105_WM_common_pre_calibration_CPU_v1](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_common_pre_calibration_CPU_v1)。三个文件已由[final1000独审29](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_common_pre_final1000_saved_independent_v1/report.json)绑定（SHA `daff6c0b4f1d165b054b2431f4cb62476237f65869b9f3e4fff2846b861b36fa`）。初始对象使用自己的 `WM` 字段，无 arm/final1000身份；[initial_validation.json](D:/workspace/supermimic-full151-observation-study/outputs/full151105_WM_common_pre_calibration_CPU_v1/initial_validation.json)（SHA `36c43a9ab4b03edd7f45fe8760fc2743821ecfe2599ea5795b7ade44a04b4186`）记录其 WM state digest **`d28551ced4b939f59dfc26ee2a2f2ff0146977d7e9277e6afa9c32128171c4ec`**。此来源仅本轮读JSON，未打开权重。

**输入、边界与 recurrence 不变。** 采用旧 H1 `matched_input_arrays.npz`（SHA `18a361bbc6a56139e79df5b6891e3aa80e030abebbbee4fa39e3dbe5129e9cb4`）及 `group_identities452.json`（SHA `0151e53864bbf26840b4435863c7038a957ba94dd142e8385deadfe18f0d167b`）：它们直接来自已合格 native GPU/B1 PRE/POST58及四子步 actual absoluteu。按 `(path,mode='donor',k)` 取两path各150行，不取raw前300；既有 contiguity `6512f824…9d0ff` 已证298个相邻全58 links与300组4ctrl位一致，本轮不重算。起点每path0..142，共286重叠窗；h1..8使用u[k+h−1]，最后u149→obs150，无u150。每步以先前 **预测58** 重建 `dynamics278=cat(pred58,shared_xi46,zero174)`，仅初始取真实 PRE；无 teacher forcing、Actor、EMA/C2、output clamp/旋转修复。152恢复分支没有后续160ms，按availability排除，不看误差筛窗。不得改用 fresh152 中 snapshot8重复19假作完整 donor PRE300。

**最小新接口与精确预算。** 两final仍用已执行 `component_confirmation105_v1.make_eval_component`；初始需一个小型独立 `make_initial_eval_component(initial, initial_validation)`：核自己的 schema/WM digest、完整keys/shape/dtype/finite、running input/delta norm配置与buffer、xi46/s52，再同样 `fork_rng` 构造278/15/512×3 native WM、strict load/eval/冻结参数/ampFalse。不能为通过旧函数伪造 arm/iteration1000，也不借旧105 wrapper资格。实际运行前将三model xi 与旧 H1已保存 physical_xi46做位绑定；来源一致已有metadata支持，新增数组比较尚未执行。

三模型仍 **3 own component文件 loads**（initial一次、two finals各一次），**3 native WM ctors、3×8×B286=24FW /6864 dependent prediction rows**；相对两模型只多1ctor/8FW/2288输出。0旧119f实际load、Actor、AD、optimizer、objective、MJB、state转换、FK/query、Env、physics、GPU。保存原始RNG/输入/参数与running norm前后证据，沿成熟component机制，不扩隐藏状态框架；无新wall/内存benchmark，不把24FW当整run耗时的线性保证。

**只描述全量原单位误差。** 全三模型×两path×143起点×h1..8保留raw58预测/目标/完成mask及六块 q/qd/p/Rraw9/v/omega原单位 component RMSE，p/v/omega另vector RMS、R9另Frobenius RMS。无 angle投影、mixed-unit58 scalar、新准确性门、CI、成功组选择或总体赢家；不要求误差单调。任何nonfinite保partial和全部索引，不补truth/重试/改clip/仅汇报成功窗。h1的286起点与旧all452人口不同，不能直接归因跨表差值。

这是**已见路径、同谱系单seed、FP32 CPU shadow、固定外部控制**的探索性多步动力学核验；与 native H8 imagination 的递归state有关，但不含预测obs上的Actor重选控制、memory/prior/reward、aux/push变化、survival/critic。zero174沿既有consumer假设，self15只可留finite辅助输出，不变成HO/力标签。它不升级 unseen state/trajectory/ego、独立样本N、原物理参考/任务、wholehidden或真实梯度资格。待 Root 正式冻结 metadata/pins/预算与执行范围后才可实现/运行。
