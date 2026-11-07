# 原生软接触：零速度不等于零切向力，但严格零加速度有额外限制

**结论：在当前这类标准soft约束、R>0、无外加surface velocity且未启用NoSlip后处理的模型中，全qvel=0且全qacc=0时，contact Ft为0；condim3 pyramidal的四射线耦合不会例外。** 这不意味着“qvel=0就Ft=0”，也不意味着所有零速平衡都不可能：接触法向的合成仍可能托重。它说明依靠切向静摩擦承重的pinch不宜把理想硬Coulomb的完全静止条件直接当作此软模型的可行性必要门。

## 3.11.0源码推导

固定官方提交`b85fdca54f0e0038b804af146a0b4e94199e00d0`，没有以最新版本或理想摩擦模型替代。

`engine_core_constraint.c:1677–1692`的pyramidal contact为每个切向k建立两个ray，Jacobian为`J_n ± μ_k J_tk`，所有ray使用相同normal的dist/margin。`:2156–2206`给同一个contact的ray相同K/B/imp；`:2239–2247`更明确地将所有ray的R都置为同一个R_py。`:3238–3247`计算

`aref_e = -B*v_e - K*imp*(dist-margin)`；

`:3281,3330–3343`给每个非负ray

`λ_e = max(0, (aref_e - J_e*qacc)/R_py)`。

令`b=-K*imp*(dist-margin)`，则正负ray为

`λ_k± = max(0, [b - B*v_n - a_n ∓ μ_k*(a_tk+B*v_tk)]/R_py)`。

当所有v、a为0，所有ray同值`max(0,b/R_py)`；`engine_util_misc.c:1591–1599`解码为`Fn=Σλ`、`Ft_k=μ_k*(λ_k+−λ_k−)=0`。因此即使normal压缩产生Fn，四射线也不自动产生非零静态Ft。这个结论不要求µ两方向相同；R_py公共且正负成对即可。[固定版源码](https://github.com/google-deepmind/mujoco/blob/b85fdca54f0e0038b804af146a0b4e94199e00d0/src/engine/engine_core_constraint.c#L1677)

若正负ray都活跃，局部有`Ft_k=−2μ_k²*(a_tk+B*v_tk)/R_py`。所以**非零相对切向加速度，或很小的持续相对滑动速度，均可通过软阻抗产生有限Ft**。这只是模型关系，不是必须人为植入滑动速度的建议。有限接触切换/迭代误差及cone边界使它不能当所有状态的全局线性式。Elliptic cone的切向无位置残差、K=0，在同样严格v=a=0时也无Ft。

DOF/tendon `frictionloss`是另一个有界、load-independent约束，不是contact tangent：其K=0，力为`clip(−(a+Bv)/R,−η,η)`。严格v=a=0仍给0；“preemptive”是可以在速度尚未出现、但存在加速度需求时抵抗，而不是在此有限R的唯一解中任意选择静态力。限位/接触normal有位置残差，与它不同。

官方文档也将soft模型固有的滑动与NoSlip区分：后者是另一个后处理求解，将摩擦方向R置0。启用它会改变上述模型关系，不能在旧v6协议里暗中打开后称原模型通过。[3.11 soft constraints / NoSlip](https://mujoco.readthedocs.io/en/3.11.0/computation/index.html#noslip)、[原生NoSlip选项](https://mujoco.readthedocs.io/en/3.11.0/XMLreference.html#option-noslip-iterations)

当前cold保存的options子集没有`noslip_iterations`，所以本次不冒称已从raw证明该值0；官方默认0和原构建源码不主动开启，仅属source范围。相同条件还应排除施加的surface velocity及adhesion路径；本页不以其他默认替代未导出的模型字段。

## 已保存cold/preload证据，0新forward

四个cold raw（两个seed各fresh重复）均没有generated contact。phone的`qfrc_constraint/passive/actuator/applied`自由关节6项严格为0；COM加速度为`[约0,约0,−9.81]`，α约1e−15 rad/s²。20个efc frictionloss行都属于hand DOF0–19，未落到phone20–25。这个结果排除**这些cold状态**中的虚假外支撑。

原始`dof_frictionloss/damping[20:26]`模型数组没有导出，不能据零速下passive=0独立证明damping参数为0；本次没有为了补字段重建模型。全4个状态的数值/byte-zero检查和各pin在`report.json`。

已有preload raw则直接显示：全qvel=0时，seed0首contact的`[Fn,Ft1,Ft2]≈[1.102272,−0.771590,0]N`，但完整qacc并非0（最大约274.4rad/s²）。这验证“v=0”本身不排除Ft，不能把上面的严格v=a=0条件遗漏。preload仍不是平衡或holding成功。

## 对v6及下一步的界限

v6的`m*a`/`I_world*alpha`无量纲化、同时保留20joint qacc以及独立最终22/189几何的分层，在实现语义上没有把NN学习混入；它仅优化姿态/ctrl，0训练/step。但若把全速度0且极小瞬时加速度当靠摩擦承重的唯一入组门，会排除该软模型允许少量滑移建立Ft的持握行为。0proposal只是这次有限静态目标失败，不能变成机器人不可行或摩擦不足证明。

**一个最小后继建议：**在新协议中用同一原生软模型、qvel=0的真实初始化和合法ctrl，直接优化/筛选一个有明确预算的短时无辅助rollout的漂移/速度/非目标碰撞表现；让摩擦在原生动力学中建立，不强求每个零速probe的所有qacc精确归零。原始零速qacc保留为诊断，旧v6结果/门不修改，最终仍必须通过原2秒、5mm/5°等全部holding门。此处未实现或执行该建议，也不建议通过改µ、R/NoSlip或虚假外力救旧结果。
