全1808行同solver相位结果说明（作者事后解读，非独立审计）

固定原3mm标准：44行native生成HO深度超界，42行P>0；42行共同阳性、2行native阳性/P=0、0行proxy阳性但无观测nativeHO超界。全部1808行保留在all1808_description.csv。独立数值权威为Contact的35/35报告，本文件不替代它。

1200行fresh donor中，A0为30/29/1（native阳性/P阳性/漏检），A2为14/13/1。96行baseline和512行direction全部native≤3mm且P=0。因此旧8个局部名义状态、160个20ms末状态的P=0是稀疏时相覆盖的事实，不能外推全段proxy不响应，更不能推成训练期间P未激活。训练日志里P_ONLY和A2各1800次policy update的active P均值都>0；active加权P均值约1.08115/1.07391。日志只保存imagined active均值，不能恢复每行阳性比例、真实接触或Actor/u梯度；它与真实donor分布不同，不直接比较均值。

两次漏检都不是缺失geom：

| 路径/solver预时刻 | native最深pair | native深度mm | 同一geom采样深度mm | 同geom差mm |
|---|---|---:|---:|---:|
| A0 k11/sub4，0.235s | finger2_link3 geom27 / object geom1 | 3.030756 | 2.964228 | 0.066528 |
| A2 k5/sub3，0.110s | finger2_link3 geom27 / object geom1 | 3.573335 | 2.814320 | 0.759015 |

每个geom都有原64采样点。A0 native超界30.756μm，proxy低于3mm为35.772μm。A2最深geom27并非proxy全局最深geom：全局最深来自geom31 finger2_tip_sensor，为2.971762mm；其native次要超界pair为3.072037mm，同geom差0.100275mm。A2最深pair的native超界达0.573335mm，不能把两次都叫同一种微小边界现象。两行都不在FP32 allowance的精确等值边界；float32(0.003)与0.003只差2.60770320268e-11m，不能据该scalar转换解释这些深度差。

这些是“已覆盖geom的proxy欠敏感候选”，不是唯一采样原因。native候选pair距离、有限表面点SDF/几何表示和聚合是不同量；没有新native查询或加密采样对照，不能唯一归因某一环节。全体非阳性行也不证明全局无穿透。原threshold、全部exact deepest tie-set与secondary全部超界pair均未更改。

下一最小研究行动建议（仅可行性，未执行）

优先Root提出的固定现有A2 WM、全部452个匹配20ms组的预测误差诊断，先于针对这两点修proxy或直接换prior/加训练。数据已经有300个donor组和152个baseline/direction组；每组从GPU-qualified prestate取首行，目标从同次native GPU replay的452个post20行取，actualu要求四substep逐位一致，保持phase/顺序，绝不再次EMA/C2。冻结119f CPU loader支持任意正batch的FP32 CPU state58/u20，内部固定xi46与zero174→278，再加u20→298；输出next58和self15。一次strict加载（内部2次checkpoint读取与Actor身份构造，0Actor forward）+一次batch452 WM forward即可；不需新physics、训练或AD。执行前仍须新明确协议/配置、完整原始身份与输入gates、只读normalizer/RNG记录，当前没有运行。

主结果按q/qd/p/R/v/ω原单位和path/phase/donor-versus-replayed分层保留全部452行，不按42个P阳性挑样本。这个诊断更直接回答稀疏4相位是否代表WM全段误差；它本身不把误差曲线变成物理梯度证书。现有physical/WM比较仅32个依赖方向组中9组稳定、两eps各5/9同号（含3组clip），不能当18个独立样本或HO梯度资格。下一步若要解释AD与finite-interval差异，需另预先固定分析，不能靠这次预测误差自动下结论。

当前三方法的WM/failure在尚无Actor更新和prior forward的checkpoint100就已非逐字节一致；Actor/Critic仍原始相同。因此旧单seed结果保持“相同初始参数与预算”的描述性配对，不是共同训练轨迹的严格因果对比。未来方法归因需要另设计共同WM/共同输入的控制或共同warmup，不能用本dense表补造因果证据。已知rough reference和三方法原task conjunction FAIL全部保留；不用等待完美kinematic reference才能研究动力学，也不能用proxy大多检出替代reference/动力学/任务资格。本交付不新建H1实验、不训练、不重物理。

证据与范围：report.json逐一pin dense产物、独立35、旧bonus523、WM656、全1800日志、prepolicy100与fixed-final三臂总结；source和输入哈希前后不变。没有模型、目标、MJB、FK/query、AD、GPU、Env或物理调用。
