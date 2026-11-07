# Public native52+self15-only：同预算最小消融提案 v1

**SOURCE / JSON ONLY；非训练授权或待通过结果。** 本页只读冻结796c训练源、9e recipe与本机Adam源码；未读pt/NPZ、计算新指标、运行Torch/NN/Toy或改旧源。待Root看完H8后决定是否创建新namespace、recipe、健康reader及独立执行协议。旧物理任务/ego/独立轨迹缺口与全部负面结果保留。

现有 `shared_public_losses` **不是native-only**：它优化native52+self15，还包括mode BCE及J/K无量纲监督；另一臂另加跨目标consistency。建议只增一个 `native52_self15_only` 训练臂，与这两份已固定AFTER1000按下表比较，不能把旧control重新命名或拿PRE1000日志当AFTER1000评价。

|训练臂|实际优化标量|
|---|---|
|新native-only|`Lnative_pair`|
|现有public control|`FP32(Lnative_pair+Lmode)+Limp_F64`|
|现有public+consistency|上项 `+Lcons_F64`|

**数据、原生损失及预算不变。** 同原own initial `ea051225…a14bf`、原152真实旧相位[0,49,99,149] x/u/target与label-only cached15 JSON `8b900a1b…3f698`；全部行/顺序、xi46、captured effective s52、frozen norm/host/context及原π−.01 chart规则不变。令native FP32 `D=difference(pred,target)`，`p_i=mean52((D_i/s52)^2)`、`c_i=sum15(BCEWithLogits(self15_i,label15_i))`，则 **`Lnative_pair=mean152(p_i+FP32(.1)*c_i)`（FP32逐row后mean）**。不能以两个已归约scalar相加代替、把15改mean、增加reward dt/window/classweight/mask，也不向forward注入labels。标签仍是约15ms缓存而state为POST20，不是HO或预测状态几何truth。

固定1000次、每次同序B152/单forward/单backward/单AdamW step，共152000预测；first training FW承担原initial-function parity，不额外validation FW。沿9e：CPU4/FP32/ampFalse，lr1e−4、betas(.9,.999)、eps1e−8、weight_decay0、foreachFalse、全局clip_norm1、zero_grad(set_to_none=True)，无shuffle/resume/早停/新超参数搜索；100及1000保存PRE-I损失/梯度和AFTER-I权重+真实Adam步，沿原用户暂停/后台条件另行明示。对比是同原生FW/更新次数和数据预算，**不宣称相同FLOPs或wall time**：新臂不算mode/J/K/cons损失，loss/backward成本自然不同。

**推荐保留同一zero unused14head的wrapper，优于改成raw14模型。** 仍使用003的同架构、16个命名Parameter、RAW copy-before-hook、一次head FW、零head初值及一致storage/norm/context检查；forward返回head但优化scalar完全不引用z/mode。将同16参数按原顺序交给AdamW/clip API，明确只有WM14能从本loss得到梯度；head两Parameter每次必须 **`.grad is None`**、原始positive-zero bytes保持、optimizer中没有其lazy state。**不**用`0*Lmode/Limp/Lcons`或人工zero gradient“补齐16步”：这会建立不同梯度/Adam语义。当前Adam `_init_group`只收`p.grad is not None`；因此新臂应有14份有效Adam state、step=I，head无state（不是step0或伪1000）。当前796c的“16梯度全存在/16 Adam state”守门须在**新版本/新schema**改成这个显式分区，旧reader不得放宽或借作新健康。

这里三臂安装的native WM14与aux head2架构相同；native-only有效更新14，其他两臂16（head含14×512+14=7182个scalar）。这不是完全相同的有效可学习容量：aux监督及其额外输出是所比较方法的一部分，global clipping也会接收不同梯度。保留unused head只能排除“删除模块/换constructor”的额外实现差异，不能据此声称纯loss唯一因果。raw14直接训练也合理，但会改变installed参数/optimizer-group/hook路径，所需新健康与接口改动更多，本次不优先选。不能要求训练后WM bit-equal或制造bit-equal“收益”；初始原生输出/参数bytes相同只是工程前置。

**最小评价/证据。** 新初态实际比位，保存WM14/head2分区、PRE梯度、14 Adam状态、head无state/不变、raw RNG值及所有原pins；全固定1000终态须独立serialized health，不能拿in-memory摘要替代。三者评价都用同实际152的native_pair/point52六物理单位、cached15 BCE/Brier/混淆/强度及已固定公共诊断，保13重叠cohort；方法各自优化total不作跨臂winner分数。若追加fresh/H8/task梯度，需同一数据/控制/精度/预算和先冻结协议，不据结果调λ/筛组；不是blind、独立轨迹或Actor/物理任务资格。

固定源（项目根 `D:/workspace/supermimic-full151-observation-study`；只引用既有hash，不制造未来artifact hash）：

|源|SHA256|
|---|---|
|`src/full151_gradient/boundary_public_self15_training105_v1.py` :130–154,184–198,256–286|`796c360539961143fe1f588fd18eb08e4b168d141b34daee23da802a72194e7e`|
|`protocols/full151105_boundary_public_self15_matched_training_recipe_v1.json`|`9e0765fd562c3e24bdb09d52f78219204af4228bb6e8b1d8ffd67104ce094bce`|
|`src/full151_gradient/boundary_impulse_component105_v1.py` :82–89,113–153|`003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317`|
|本机 `.venv/Lib/site-packages/torch/optim/adam.py` :139–161（根 `D:/workspace/dex-supermimic-research`）|`e333d196bd24f6a44146839139ad0222bfaab0f71dcdb341585a4e3f9f4c2d87`|
