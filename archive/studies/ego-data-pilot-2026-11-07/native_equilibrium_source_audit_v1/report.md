# MuJoCo 3.11零速原生平衡接口：源码审查

**接口成立，但要称原生软约束模型的瞬时静止候选。** `mj_forward`完整计算位置、速度、驱动、无约束加速度和约束求解，不积分时间；源码没有`mj_step`或学习。它不是一轮训练，也不能单独认证持握、稳定性或真实人体受力。[3.11 API](https://mujoco.readthedocs.io/en/3.11.0/APIreference/APIfunctions.html#mj-forward)

本审查0模型/MjData/FK/forward/积分/NN/GPU。Windows独占至多4次实际forward：两个指定seed各一次及相同输入的fresh重复；数值由其输出报告，不在本页虚构或重复计算。所有本地源与官方3.11.0提交`b85fdca54f0e0038b804af146a0b4e94199e00d0`的文件身份见`report.json`。

## 最小正确提取

1. 每个候选新建`MjData(model)`，写完整有限qpos与真实ctrl；全部qvel、act、qacc_warmstart、qfrc_applied、xfrc_applied清零，time=0。不从上一优化点继承data、接触力或warmstart。确认无control/passive/contactfilter/sensor/actuation callbacks、plugins、equalities或mocap辅助；保原模型、真实ctrl/actuator/joint限位与摩擦，不为通过改warmstart/solver标志。
2. 一次`mj_forward`后，读取完整`qacc26`、qfrc_bias/passive/actuator/smooth/constraint、warning及solver统计，并核qpos/qvel/time/ctrl/applied与固定腕未变。`qfrc_constraint=J_efcᵀ f_efc`含**equality、DOF/tendon frictionloss、joint/tendon limits及contact**，不是纯HO扳手。记录efc type/id/force及ne/nf/nl以界定贡献；零速不意味着frictionloss或弹簧/限位约束为零。
3. 取`C=data.xipos[phone_body]`，调用`mj_jac(model,data,Jp,Jr,C,phone_body)`；它与`mj_jacBodyCom`相同。两矩阵均为世界坐标3×nv。全部qvel=0时，`Jdot*qvel=0`，因此

   `a_COM_world = Jp @ qacc`，`alpha_world = Jr @ qacc`。

   不能直接把freejoint后6个qacc当作世界COM线/角加速度：body原点与COM不重合，角坐标也不是世界轴。[Jacobian源175–235](https://github.com/google-deepmind/mujoco/blob/b85fdca54f0e0038b804af146a0b4e94199e00d0/src/engine/engine_core_util.c#L175)
4. 对单一自由刚体，若无另一个有质量的附属body、phone六DOF armature为0且采用普通刚体质量度量，零ω下可用

   `F_net = m*a_COM`，`tau_net_about_COM = I_world*alpha_world`，

   `I_world = data.ximat[phone] @ diag(model.body_inertia[phone]) @ data.ximat[phone].T`。

   这是**所有作用项的净惯性扳手/加速度残差**，不是接触力本身，也不是已经抵消的重力。与全部物体边界接触+重力+实际已知其他力矩分开保存并交叉核算；重力补偿、applied等仅在实际证明为0后删去。有限solver容差下不假定两种表达逐bit闭合。3.11存在flex的有效`M+K`度量分支，应记录nflex与`data.efm_active`并要求本刚体路线inactive，不能把该等式无条件推广到flex/虚拟armature。[forward源911–956](https://github.com/google-deepmind/mujoco/blob/b85fdca54f0e0038b804af146a0b4e94199e00d0/src/engine/engine_forward.c#L911)

## 接触力方向与法向

`contact.frame`的三行是normal、tangent1、tangent2的世界轴；normal从`geom[0]`指向`geom[1]`。`mj_contactForce`返回接触坐标的`[force3, torque3]`，并由原生API解码pyramidal约束。令`S=+1`当phone为geom1，`S=−1`当phone为geom0，则

`F_phone=S*frame.T@force6[:3]`，`T_direct_phone=S*frame.T@force6[3:]`；

`tau_COM=(contact.pos−C)×F_phone+T_direct_phone`。

两部分同号，直接矩只加一次。若两geom均属同一物体，属于内部作用，不当外边界力重复相加。使用**实际contact行的geom顺序**，不假定pair声明/名称的排列恒定。`efc_address<0`时API给零结果，但generated contact仍应保留为无力约束候选；不能把没有generated contact当作全局间隙证明。不得用Fn>阈值删小力行。

3.11的`mj_contactForce`还会从法向分量减去adhesion；无adhesion时Fn通常为压缩正值，存在adhesion时不能硬拒负Fn。当前condim3设计没有spin/rolling力矩自由度，不能自行补torsional矩；仍保存原生force6和dim/adhesion/排除状态。[contactForce源1074–1094](https://github.com/google-deepmind/mujoco/blob/b85fdca54f0e0038b804af146a0b4e94199e00d0/src/engine/engine_core_util.c#L1074)、[受力符号源2438–2460](https://github.com/google-deepmind/mujoco/blob/b85fdca54f0e0038b804af146a0b4e94199e00d0/src/engine/engine_core_smooth.c#L2438)

## 候选守门与解释

- **手也必须静止。** 同时保存/约束20个关节qacc，不能仅手机6维平衡而允许手开始运动。各单位分别报告，不混成未经声明的无量纲门；实际PD、ctrl clamp、actuator与joint force clamp均以原生读回为准。
- 两个seed各fresh重复可检查确定性：同qpos/ctrl/model、zero warmstart/applied下比较qacc、J、contacts/efc与force。warmstart启用时原生会在固定零warmstart和smooth解之间选较低cost；这不是复用历史。重复通过只证明该次同输入路径，非广义确定性或优化可微性。
- 当前配置为implicitfast/Newton、100 iterations/20 line-search、tol1e−6/ls_tol.01、pyramidal condim3、μ_hand=.7/μ_table=.6、solref[.0025,1]、solimp[.9,.95,.001,.5,2]、margin/gap0；phone gravcomp0。两细步长下τ均大于2h，标准solref的REFSAFE下界不应主动抬高τ，但以实际选定模型/efc读回为准，不能据源码假装已经验证运行值。
- 原生soft contact在零速与指定压缩/间隙下产生解，接触切换、frictionloss/limit激活及有限迭代会使目标分段、不光滑。小qacc仅是此模型、此ctrl和此速度的瞬时平衡，不保证受扰稳定、可达预载、持续双pad接触或2秒持握。最终仍须独立22/189几何与以后另授权的无辅助动力学门，不能用优化器返回success代替。

以上仅修正参考优化的物理模型接口；不拟合模型参数、不更新NN、不启动学习，也不授予ego/策略或硬件收益。
