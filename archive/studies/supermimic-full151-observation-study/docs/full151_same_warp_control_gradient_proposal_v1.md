# Full151/105同Warp控制差分诊断草案

状态：**proposal_only / NOT_EXECUTED**。本文件不授权物理、模型前向、训练或GPU。本次只读源码和已保存JSON/NPZ，未导入Torch/Warp/MuJoCo/Env。参数为下一次实施前供Root审阅的一套有限方案，不是已冻结执行计划。

## 已有证据与缺口

实际WM文件为`src/full151_native/world_model_checkpoint.py`，不是`world_model105_loader.py`。原119f加载器可严格消费105、显式方法/最终2000健康/训练配置/同bundle与actual8；物理xi为46维，不能用critic context368代替。`forward(x58,u20)`附加xi46和zero174，返回state58与self15 logits；不含EMA或C2，冻结参数仍允许u输入自动微分。源码`validate_health_metadata`使用显式expected_method/expected_health_auditor，后续绑定primary_matched/实际config/2000，没有schema1、A1L或旧e9f194 learning硬编码；Actor105检查root五个extra和动态execution_source_pins（逐pin核hash），只固定仍沿用的controller/pure源。因此源码接口可接schema2/A2，但不会独立重审prior/capacity，须传新5950健康producer及实际final健康。当前schema2 A2的实际WM加载/前向仍PENDING，本次没有加载checkpoint。若未来实际guard拒绝，须另新隔离消费者修复，不能绕过或猴补旧loader。

旧105诊断只测历史schema1 A1L WM、A0/A1L各150条保存转移及WM内部FD。23PASS是工程/有限/不变性检查；不是预测精度门或物理梯度资格。旧104同Warp结果亦不能转移到151/150、105、解析速度及新控制签名。旧104事务`_native_profile`硬编码`_rough_*`，新上下文是`_full151_*`，必须另版本适配并审完整inventory；不能直接把旧成功当新恢复证明。

当前schema2 A2 final2000健康196/196、完整原生评估saved审188/188工程通过，任务7/13通过，位置553.991mm/笔轴150.494°/穿透28.4029mm且掉落、非手接触、速率超量。其任务失败原样保留，不因此换checkpoint、换状态或缩短窗口。当前新A1L长训练正在进行；本草案不预设其结果，也不声称已有三臂新WM公平对照。

## 单一有限设计

先用**一个已固定A2 final2000 WM**，两个fresh donor：A0零残差、A2最终评估保存的150个clamped action按原顺序重放。A0原日志没有action字段，必须显式`declared_zero`；A2有action和policy_input155。fresh donor是动作来源类别，不是旧闭环轨迹复现，不重新运行Actor，不加载WM到物理进程。旧轨迹和fresh漂移独立列报，WM只消费fresh snapshot。

每条完整150 interval donor在k=[0,49,99,149]截取4个pre-action状态：前一原生4×5ms、Manager forward、sense/command/obs已完成，当前process_actions尚未发生。共8 snapshots，每个先3次baseline，再4个onehot关节[0,4,8,12]×两eps[.002,.001]rad×正负；19分支/状态、152分支/608步，连同两donor1200步共1808集成步。无随机方向、状态挑选、重试或history择优。k149只产生obs150/timeout，不查u150、不自动reset。该最后一次物理state/scalar响应可比较，但不是有效后继policy决策或bootstrap资格；phase150必须survival0，无第151个控制决策。

捕获后的分支调用原`env.step(source_action)`，原process_actions恰一次并保持EMA残差c1/c2；只在它之后改cached absolute `_targets`。u±以**执行目标空间**构造：由fresh中心u32加减eps32，在原硬∩ctrl且内收20mrad的FP32域预验证。中心及端点均须C2 identity，真正分支不再投影、不改变残差，不额外forward或写q/qd/物体。原apply_actions每5ms硬限须identity，四步实际ctrl逐位等u±。保存clip、不对称中心偏移与实际d_eff；这不等于Actor action→EMA→C2的全链导数。

## 相同标量、相同时间和单位

主标量为原生物体两项未乘dt的FP32 score：s=2 exp[-(||p−pg||/.05)^2]+8 exp[-(e_axis/.04)^2]；e_axis为±r轴端点平均距离，r从冻结command axis_point_radius_m读取并断言实际.1m，不另猜半径。目标采用preaction k，与原native reward一致；k+1仅并列次口径。step物体reward=.02*s另列，不能混分母。

物理x1来自`PenSimReader.attach_capture`的`reward_snapshot.physics`（20ms postintegration、Manager最终forward前）。self15和torque来自末次solver的15ms preintegration时相，必须单独标时钟；x1_after_Manager只作次记录。物理和WM同输入x58、actualu20、xi46、相同goal，WM不再次EMA/C2。保存两个FP32 term和其FP32 sum；独立F64几何核验只用预声明8eps32算术界。FD为两个F32 score先转F64再相减/2eps，单位score/rad；WM AD主比较为∇u s(WM(x,u))·d_eff，nominal onehot次列。不得把F64重算换成native主标量。

## 恢复、实际context与容量资格

可复用API见source_pins及source_symbols.json：`full151_runtime`外层、`full151_controller`内层，构造Manager+PenSimReader；`install_actual_context`保持CPU nominal MJB，actual8复制后`set_const_0(restore=True)`，两次真实readback，xi1×46与bundle逐位一致；analytic非零init和defaultmocap沿现有reset。实际8字段/shape必须由原源取，不能把nominal MJB当实际随机化context。

`WarpTaskSnapshot(SimpleNamespace(env,reader)).restore()`恢复原WP model/data数组、Torch张量/manager容器与RNG，但NumPy、函数closure、foreign object及新105属性必须由新事务sidecar显式cover。snapshot需包括solver warmstart、sensor/cache/obs histories、termination、motion phase、action `_raw/_processed/_targets/_targets_valid`、controller pending/active、runtime generation、PEN memory、mocap/exogenous/model实际参数和RNG。`mjSTATE_INTEGRATION` size/signature从实际模型算，set/get bitexact只是必要条件；`get_data_into`是WP→host读回，原raw_states qpos/qvel也不含完整warmstart，不能反构完整snapshot。

overflow会随snapshot恢复，故容量observer不能挂env/reader或进入sidecar。host OR/UNKNOWN账本在每次restore前、每substep后、branch finish、donor terminal和全末读，记录实际dtype/shape/nacon/nefc与source；OR非0、UNKNOWN或读取缺失工程FAIL。不把末次0覆盖早前overflow。固定原容量，不调大容量补跑。

每snapshot三baseline须先全部工程/actualu/clock/integration/inventory合格，并对fresh donor x1按原旧门q2e−5rad、qd2e−4rad/s、p/R元素2e−5、v/ω2e−4和score2e−5比较；fresh baseline失败不得进入方向分支或WM。原始实际误差照存。不同clock来源区分canonical k*.02与全程顺序FP32时钟；不要继承旧.3秒2µs门。

## 非光滑、标签与失败解释

所有计划行必须落表：NOT_RUN、clip/asymmetric、flat/below noise、nonflat unstable、stable nonflat，不能删outlier。score差阈max(1e−5,10×三baseline噪声)，两eps同号且相对差≤.5才stable；边界不对称单列，不冒称中心局部导数。接触geom/efc/类型/normal/active-set与self15改变作为独立非光滑flag；传感阈值跳变不强行求连续真实导数。每eps指同32方向，不能当64独立样本，8状态仍同源/同seed相依。

self15为10中节+5远端(link4 OR tip)的native hand self-collision。它不是HO、力、穿透或人类接触真值。新worker应在权威reward capture同时保存raw finger_collision found和原sensordata以便独立标签重建；现旧full151日志只有self15，不能假称可从HO几何反推。测得HO contact force6/COM wrench、distance和接触signature只作原native测量诊断（必须包含direct torque/侧别与preintegration相位）；WM没有HO力head，不能拿proxy P的梯度当真实HO力梯度。实际力可很大而标量差平，二者不互替。

全151几何SCP候选仍FAIL（HO11.448mm/self11.431mm），声明surface cone302的可行见证是条件假设、无真实contact/wholehand/constitutive资格；本探针不能补这些缺口。FD是有限区间secant，不是真实唯一梯度；WM AD是模型自身FP32CPU导数，二者吻合也不证明物理Jacobian、GPU AMP、训练收益、真实ego或全运动可行。

## 实施边界与待补接口

需要新增105事务profile/新物理worker/入口及独立saved-only finalizer和CPUconsumer，不改旧冻结源。物理进程零WM/optimizer；CPUconsumer用原119f strict loader、显式schema2 A2 final健康producer和2000，保持checkpoint/权重/norm/RNG不变，actual8/xi sourcebound，先独立重算本方案**24个baseline**及全部工程门，再允许WM。consumer不得继承旧104固定16/560/903字段。

建议预先soft180/hard240秒、共同GPU锁、精确PID/PPID/CreationDate/actualcommand base samefile与终态CIM空、原159c有界metadata I/O、无retry。此为预算提案，Root须在实施前确认并单独授权；超时/未知退出/容量/restore/baseline失败完整保partial，CPU WM状态NOT_RUN。工程门不看任务quality，质量fail保留且不缩短完整horizon。没有新训练或自动后续run。

本次被动数组检查只证明现有150转移/601状态/动作字段与源ABI可接，不能证明未来105事务恢复、GPU容量/长时钟、A2 WM加载或物理分支已合格。详见同目录proposal JSON及只读准备report。
