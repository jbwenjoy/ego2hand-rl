# Prior介入前的差异：固定checkpoint100诊断

对三份新schema2 primary的固定model100做一次CPU保存比较，39/39比较合同检查完成。没有构造模型、前向、环境、物理或GPU，也没有按结果挑选其他checkpoint或重训。

Actor和Critic的可训练参数仍与原始初始化逐字节相同，对应optimizer均空；WM/failure各200step，prior forward仍0。但三对WM参数均不逐字节相同，L2差约3.299–3.479、最大绝对差约.0571–.0691；WM/Actor normalizer也已不同，Critic完整module相同。保存的CPU Torch、NumPy、Python RNG相同，CUDA RNG不同，各实际转移数均40192，prefix容量读取记录数为145/130/132。

这证明在尚未执行policy/prior更新的前缀里，训练数值与采集历史已经不完全相同。它不唯一定位差异起因，不直接证明factory RNG合同、物理源码或优化器实现有错误；同样也不能把同seed、共同初始参数和共同预算当成完全共享的训练轨迹。原始参数配对、各自完整预算和fixedfinal物理结果仍是真实的，原13门槛失败均保留。

因此三方法表只能作一个seed/一个来源的描述性结果。P_ONLY保留物体而A2掉落的观察不能据此作support因果效应估计，也没有总体收益或置信区间。后续应先定位预热分歧及验证WM的物理响应；在参考本身资格不足时继续扫权重、追加一批seed或选择checkpoint，不能替代这些诊断。

下一项同Warp探针使用一个fresh物理输入/绝对控制目标和明确定义的标量，对同一输入做恢复基线和方向差分；它在固定输入层面检验WM自身导数与物理有限区间响应。该探针仍不能自动解释三组在线训练结果的差异起因。

[原始39项比较报告](D:/workspace/supermimic-full151-observation-study/outputs/full151_schema2_three_arm_prepolicy100_comparison_CPU_v1/report.json) · [三方法完整结果](D:/workspace/supermimic-full151-observation-study/docs/full151_three_arm_primary_results_2026-11-05.md)
