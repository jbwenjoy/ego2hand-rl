# Full151 / 105 的完整152分支经验探针

本次为固定受控转笔数据上的局部诊断。A0使用声明的零残差，A2使用固定final2000保存动作的全新重放；二者都不是保存状态恢复或新的闭环Actor评价。每条路径在k=0/49/99/149捕获快照，分别先做三次基线，再执行4个控制方向、2个epsilon、正负扰动。执行前方案、源码、逐时刻foreign清单和Root边界决策已冻结。

实际运行一次并正常exit0：300个donor控制区间，加24个基线与128个扰动分支，共452个控制区间、1808个episode子步，152次begin/finish/原生快照恢复均返回。8个快照均完成；24个基线的q/qd/p/R/v/omega/主评分最大差异全部为零。固定容量仍为192/4096，无自动重试、容量扩张、WM/Actor前向或优化器更新。独立操作系统进程资格12/12，终态原两个PID均不在。

这些是生产者保存结果及进程证据；独立逐行保存审计尚在进行，当前不能用进程正常或基线零差异替代全部物理方程/控制/时钟/容量审阅。

主评分为直接由积分后20ms物理状态与动作前goal_k计算的受控FP32 GPU 2+8函数。原RewardManager使用派生15ms物体位姿，其rate单独保留为次要记录。接下来WM比较使用相同20ms状态与相同主函数；不会将15/20ms相位差当作数值舍入。

隐藏CUDA Graph scratch既未逐字节恢复，也没有完整write-before-read证明，继续标记UNKNOWN。逐份源码审阅、可见数组字节/指针、INTEGRATION和三次经验基线只支持有边界的局部有限区间响应研究，不支持完整隐藏状态证书、唯一真实物理Jacobian、HO力梯度或灵巧操作成功结论。

下一步在独立保存审计通过后，按执行前proposal2规定进行一次A2严格CPU WM加载、batch8 forward及一次u AD。64条epsilon差分记录、32组方向全部保留；两个epsilon都超过噪声门槛、同号且相对斜率差<=0.5才称稳定非平坦。裁剪、不对称、低信号、不稳定、接触签名切换均单独保留，不从结果中挑选方向。WM AD沿实际d_eff投影，而非假定名义onehot。

当前仍只有一个来源、一个seed、独立ego N=0。此前三方法final2000全任务失败的结论不变；本探针不能升级参考的物理可行性或证明CHORD收益。

- [实际生产者报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_gradient_empirical_gradient05_v2/physical_report.json)
- [24个基线](D:/workspace/supermimic-full151-observation-study/outputs/full151105_gradient_empirical_gradient05_v2/baseline_gates.json)
- [运行前决策](D:/workspace/supermimic-full151-observation-study/outputs/full151105_full05_empirical_exploration_decision_v1.json)
- [固定方案2](D:/workspace/supermimic-full151-observation-study/protocols/full151_same_warp_control_gradient_proposal_v2.json)
- [进程身份审阅](D:/workspace/supermimic-full151-observation-study/outputs/full151105_gradient_empirical_gradient05_v2_watchdog/engineering_identity.json)
