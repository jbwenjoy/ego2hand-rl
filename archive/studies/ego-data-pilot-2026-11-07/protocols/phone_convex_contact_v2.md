# 手机凸包接触配置 v2：有限筛选与工程确认

这是新条件，保留 stage1 原资产、协议、18项 drop 穿透失败与所有旧结果。原 CPU tol=1e-10、Warp put_model 下限约1e-6的算法设定差异也保留，不冒称旧软件对照已完全同设定。本轮统一请求 tolerance=1e-6、ls_tolerance=.01，实际设备值须在首积分前读取并验证其 FP32 表示。

**固定筛选。** 新碰撞体是已保存的单凸包（352顶点/700面），视觉不改；COM仍为原bbox中心，惯量仍按均匀bbox假设。先只在 mass=.2 kg、sliding µ=.6 上比较 solref=(τ,1)，τ固定为 [.02,.005,.0025] s，dt固定为 [1.25,.625] ms；drop最低点.1 m、settle最低点.5 mm、push在[1,1.2) s施COM +X 2 N，全部3 s。共18条 CPU轨迹/64,800步，全部保存，无结果过滤。

**选择规则。** 完整18例保存且独立 saved 审阅输入/算术成立后，对每个τ要求三个测试在两个dt的G1–G6全部通过，并且三对时间步G7全部通过。取合格集合中最大的τ，即最软的通过配置。集合为空就不确认；后续确认失败也不在v2内退而改选更硬τ。不得按跟踪误差、速度或某个喜欢的轨迹择优。

**同族工程确认。** 选中后新跑原九点 mass=[.1,.2,.3]×µ=[.3,.6,.9]，三个测试、两个dt，共54条 CPU/194,400步；筛选的名义六例不得改名充数。其后相同54条 Warp，由同一 selection receipt 和 CPU确认证据严绑。这里只是同一合成模型族的重复/参数范围确认，不叫独立科学确认或轨迹泛化。

**边界与验收不放宽。** 原G3穿透2 mm/末窗.5 mm、G4停止速度、G5被动能量、G6接触与施力、G7时间步和G8后端差异的数值界限逐值保留；最后rest仍[2.5,3] s。G1只变阶段数量。G2明示观测范围：CPU原生warning表；Warp每tick有限值、九位sticky overflow、nacon/ncollision/nefc/solver_niter、完整stdout/stderr与Python warning记录。不存在的CPU式GPU warning数组写NOT_AVAILABLE，不能填零或说两后端警报语义相同。任何有限值/溢出故障保留原行和后续NOT_RUN，不静默筛除。

**初态必须用真实凸包支持点。** 对所有实际碰撞顶点计算 smin=min_z(R(vi−COM))，令世界COM高度=指定clearance−smin，再补 root=COMworld−R*COMbody。Rx90下可能与旧盒厚度给出相同高度，但必须保存顶点/变换/min值证实，不能直接假设。考虑编译引入的mesh/geom位姿。惯量不改成凸包体积推断。

**官方solref约束。** [MuJoCo Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters) 说明正值是(timeconst,dampratio)，较大τ更软；通常要求τ≥2dt。[refsafe官方条目](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag-refsafe) 说明启用时标准安全机制取max(τ,2dt)。本轮保留refsafe，最小τ恰等于粗dt的两倍；实际dtype与有效配置仍需记录。这不是低穿透/稳定性保证，也不是材料参数标定。

新凸包有限采样在原.5/2/4 mm几何界限下3/3通过，但采样非面积均匀、非连续Hausdorff或真实表面证明。几何、CPU、Warp、parity分别报告。即使全过，真实参数、手任务、人手力与ego重定向资格仍为false。JSON schema为phone_convex_contact_protocol:2；筛选、选择、确认报告字段见output_ABI，源/资产/运行绑定必须先于每阶段积分固定。本作者未运行物理或训练。
