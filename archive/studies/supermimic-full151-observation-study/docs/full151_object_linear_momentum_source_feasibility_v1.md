# 物体 COM 线动量一致性：最小 source-only 建议

**可做一个有界的保存数据残差实验，尚不能直接授予动力学 label 资格。** 此页只读 full05 的原执行源码及元数据；10份关键源码当前字节均与原source清单一致，未加载MJB、读取新力数值或计算残差。依据本地固定版本，而非可能已变化的最新积分器文档。

**唯一拟议实验。** Root另冻结后，CPU读原全部1808个5ms PRE/POST样本（1200 donor+608 restored，不按接触/响应筛选），加 **1次被动 MjModel.from_binary_path** 读取原已pin nominal MJB，仅查询常量/地址；0 MjData/Reader/FK/forward/step/model/AD/新物理。先验证下面的闭合条件，再保存每行完整原始量、各力项及残差：

`e_impulse = m_actual*(vCOM_post - vCOM_pre) - dt_wp32*(F_all_object_contacts_at_solverPRE + m_actual*g_enabled + F_other_known)`，单位N·s；另列 `e_force=e_impulse/dt_wp32`（N）。原FP32速度先转F64再相减，**dt取固定WP float32 timestep来源，不用post_time-pre_time**（累积时钟有舍入），同时保存canonical ticks/原时间。只报告全部行/路径/模式原单位残差，不预设精度PASS门或以结果调阈值。

| 来源及必须前置核实 | 约束/缺口 |
|---|---|
| 唯一 `object/cube_freejoint`、object刚体/子树、`jnt_qposadr/jnt_dofadr`、parent/type/关节数/几何完整集合 | 必须从同hash MJB命名地址取得，不猜qvel[20:]或BODY ID。首个最小契约要求独立单一free刚体、无额外有质量子体/关节。 |
| `body_ipos` 与自由关节速度 | pinned forward.py 的自由平移增量是world qvel；角速度是body-local。一般 `vCOM=v_origin_world+R*(omega_body×body_ipos)`。最小闭合先要求COM偏移恰为0，才直接取3个平移DOF；非零不静默忽略，也不引入有限步旋转修复。 |
| 质量、gravity/gravcomp、被动与约束常量 | **质量取WP actual8 body_mass的已存copy/set_const/readback，不能取仍nominal的host MJB mass**。MJB用于gravity+disableflags、gravcomp/jnt_actgravcomp、freeDOF armature/damping(含poly)/spring/frictionloss、fluid density/viscosity/wind、tendon/equality/actuator作用对象、plugin/sleep等。actual8未改这些字段仍须source绑定。首契约要求相关额外项可证零/无对象耦合；否则missing-term UNKNOWN/停止闭合，不补0。gravity disabled则重力0；gravcomp非零须另列路由，不能直接称mg。 |

**接触总力不能只用HO缓存。** 遍历原contacts全部object侧配对，包括terrain等nonhand；用原有序geom及native完整object geom集合，`F_on_object = +frame.T@force6[:3]`（object为geom2），为geom1取负。两侧均属于所选物体系统时作为内部力对取消并保原pair。直接接触力矩 `force6[3:]` 不加进线性力。以原HO worldF仅检查HO子集的方向/次序一致性，不能用它代替总物体外接触力，更不能把nonhand自动认作静态环境。力和frame都来自**该5ms solver PRE**；qvel是该步积分前/后，保存的solver pose/COM为PRE，不用POST qpos重建接触力或用20ms奖励state拼15ms force。

保留所有原contact index、type/world/raw WP slots和重排host地址；CONSTRAINT用`type&1`，含type3。使用已解码force6是source-backed测量，不是重新解pyramidal约束；缺完整dim，不能冒称独立pyramid重解码。host efc_address不能索引raw efc_force。对存在constraint但地址/slot/世界或finite证据不足的object行，标力证据不完整而不是0；**总力不能按Fn>1e-6筛掉弱接触或负值**。sensor-only与未生成候选不提供全球无接触/间隙证书。

**相位可对齐，但严格闭合仍有已知缺口。** pinned step先forward/solve，再implicitfast/_advance；原日志逐步用get_data_into读回POST qvel，而contacts/xpos/xipos留在刚求解PRE。io.py:2230–2231确实拷贝qfrc_applied/xfrc_applied，worker raw()逐次断言两者全0，故样本external_force_max=0有源支持；没有逐行原始外力vector，须标source-backed，不当成独立字节重算。逐步qfrc_passive/smooth/constraint、qacc、efc.Ma、solver gradient、implicitfast有效qacc未保存。solver.py:2147有 `grad=Ma-smooth-constraint`，有限iterations的stationarity residual未必0；implicitfast又用efc.Ma及速度导数修正积分，不保证原d.qacc等于速度实际增量。若object平移块存在damping/armature/非接触约束等，单凭contact+gravity不能闭合。即使静态遗漏项全证零，残差仍混合FP32减法、积分/有限solver收敛及source解码误差，不唯一归因于真实力或模型。

因此首轮只产**条件化完整残差和可识别性报告**。只有完整作用力来源与误差范围有独立证据后，才讨论将其用于physically structured dynamics；目前不直接改训练、H123目标或LP eligibility。线性动量不证明角动量/完整六维扳手、单指分配或手部力cap；若后续要推手部需求，必须明确扣除已测nonhand作用，不能把terrain承载算成手需求。实际human力始终未知；这是同一模拟缓存谱系，不是人类力观测。原参考/CHORD任务资格与whole-hidden/HO-gradient边界均不变。
