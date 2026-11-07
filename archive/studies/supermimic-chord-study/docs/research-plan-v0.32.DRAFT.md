# 研究方案 v0.32：WM5000完成、响应局限与真实Actor任务接线

2026-11-07，承接[v0.31](research-plan-v0.31.md)及完整Goal。上一目标回合与本回合均为progress。用户已恢复目标；本回合沿同一冻结实验从250续训至5000，原“首健康点与ETA后暂停”已在250履行，没有重新开启同一暂停点。完整研究仍未完成。

## 当前结论及其依据

新几何/动量世界模型完成了预定5000次更新，保存恢复正常。它在已见状态的新动作上有较小的20ms一步预测误差，但该结果没有解决物理动作敏感性问题。在预登记的新初始化/控制族上，旋转和关节预测劣于简单先验；不能据端点误差小就宣布模型已经可以可靠地提供Actor物理梯度。

因此主线继续接通任务、原生采样、replay和Actor/CHORD，并用新三指任务的实际数据进行适配。手工两秒持握成功仍不是允许学习的前提；所有方法最终仍接受原两秒、双dt、无辅助的原生评价。已有16个三指初载动态案例全部失败的结论不变，不被本次WM结果覆盖。

## 同一实验的恢复训练已经完成

沿用源`224a0c9f…ca539`、seed1731、2118条绝对监督和1434对FIT响应、366→128→128→222网络及全部冻结loss/normalizer规则。CPU恢复worker实际新增4750个训练FW/backward/Adam更新，训练谱系累计5000；不能把worker计数中恢复带入的250再算成新增更新。恢复进程外部时间63.065秒，其中更新合计51.875秒、setup约9.147秒。

最终检查点固定为5000，SHA `344bac4568f57b8d97973bda06e5568b97fe10e13864f8b3d5a326bcf98dc9a7`。1000和5000两个producer读回均15/15，保存payload/log/读回证据另核15/15。没有根据CHECK或fresh指标回选250/1000。恢复目录的`fresh_initial_state.pt`是载入亲代前的重建模板；真实训练起点是已验证的250亲代及其Adam/RNG。[恢复结果](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_momentum_wm_v1/cpu_resume1/result.json)、[保存核验](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_momentum_wm_v1/resume_saved_review_v1/report.json)

新模型仍训练于原两pad15/28任务的仿真数据和其原TRAIN状态干预。没有用本轮三指初载/动态16例、新12族或真实ego视频作为拟合标签。它不是已经适配三指任务的WM，更不是完整SuperMimic策略。

## 新动作CHECK与新初始化族分别评价

已见状态CHECK保留1476条计划记录，1472个完整端点、1436个有效side、4个不可用side及720个正负中心配对。原状态已用于TRAIN，只有新动作响应未参与拟合。测试仅固定5000模型，1次own加载、1个构造、2个FW共1586行（CHECK1472、固定VAL114），0优化/AD/新原生调用。F64实际du、原始FP32动作表示差、归一后FP32实际NN编码差分别保存；没有epsilon救援或best轴选择。[CHECK结果与保存摘要](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_momentum_evaluation_v1/saved_summary_v1/report.json)

预登记family1000–1011已实际采集：两已知source pose、24个双dt episode、240条记录，22543个积分步、24次reset forward、0observer forward。228个完整H目标用于新族一步评价，12个partial保留NaN而不伪造20ms终点；它们不是任务成功样本。对原960行完整receipt/action及新族内的精确别名检查均为零。新族未参与WM拟合或选点，且仍是相同物体/模型/两源姿态下的受控验证，不是独立ego泛化。[采集与保存核验](D:/workspace/ego-data-pilot-2026-11-07/phone_causal_fresh_acquisition_v1/saved_review_v1/report.json)

| 分区与指标（20ms，一步） | WM5000 | 恒速先验 | 抛体先验 |
|---|---:|---:|---:|
| CHECK新动作1436：COM向量RMS [mm] | 0.126 | 1.315 | 1.152 |
| CHECK新动作1436：旋转RMS [°] | 0.388 | 1.414 | 1.414 |
| 新族完整228：COM向量RMS [mm] | 0.700 | 1.627 | 0.766 |
| 新族完整228：旋转RMS [°] | 1.534 | 0.839 | 0.839 |
| 新族完整228：关节角分量RMS [rad] | 0.00974 | 0.00553 | 0.00553 |

CV与抛体先验均按常world角速度外推R、常关节速度外推q；抛体增加重力平动项。它们不是原生求解，不是非球形惯量下完整无力矩积分。表中转移行在family内相关，不能当独立样本或把两个分区混池。新族的COM略优于抛体、旋转与关节更差，属于分项tradeoff，不能表述为整体泛化改善。[新族评价与保存核验](D:/workspace/ego-data-pilot-2026-11-07/phone_causal_fresh_evaluation_v1/saved_review_v1/report.json)

## 动作响应仍是主要短板

CHECK有效side中，物理头响应误差RMS除以原生响应RMS，q/qd/COM/world-dR/逐pair力冲量/力矩冲量分别为0.693/0.806/0.916/1.216/0.793/0.804。同状态下动作无关先验预测零响应，其该比值为1；world-dR响应因此比该先验更差。这里比较的是有界有限动作响应，不是真连续动力学Jacobian。

1436个side中，836个保存的原生COM端点响应逐分量为零；模型在这些记录上的COM有限响应范数除以实际du，中位数1.176mm/rad、最大7.671mm/rad，仍存在虚假敏感性。整个物体delta52的观测响应分类为：374个F64零、609个非零但FP32端点塌缩、453个FP32可分辨。没有把表示尺度当原生噪声/准确性下界，也没有据观测零签发full-H结构零证书。新数据与历史12状态诊断的cohort不同，不直接把两者中位数包装为严格改善倍数。

[四面板结果图](D:/workspace/ego-data-pilot-2026-11-07/outputs/geometry_WM5000_description_v1/WM5000_predictions_and_response_limits.png)已查看。端点预测、有限响应和剩余虚假敏感性分别展示，不将任何一项提升为持握或CHORD收益。

## 同模型导数与GPU路径已有局部工程证据

四个事先固定的原TRAIN状态、全20轴、两个中心差分尺度1e−3/5e−4rad，各有78个可双向扰动轴，2个越界轴保留unknown。模型自身action AD与world-left旋转中心FD的六块相对L2误差在细尺度约1.45e−4至2.71e−4。仅7次NN FW、4个input Jacobian、0更新/原生调用；当前几何与动作无关，使用当前特征缓存不改变action导数。它验证实现自身可微，不证明与原生物理的梯度一致。[同模型AD/FD](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_action_AD_v1/completed_run1/report.json)

相同四状态、同权重/normalizer的CPU与RTX5090对照亦通过：完整F64当前几何、F32 NN、F64动量解码以及q/u输入AD均在预声明工程阈值内。几何最大差4.44e−16，raw222最大差7.63e−6，q/u梯度相对L2约3.97e−7/5.04e−7。GPU该进程峰值allocated约1.26GiB、reserved约1.37GiB；2次FW、2次输入AD、0学习。这不是CPU/GPU训练性能比较，也不是大batch多步Actor显存上界。[CPU/GPU对照](D:/workspace/ego-data-pilot-2026-11-07/phone_geometry_cuda_parity_v1/completed_run1/report.json)

## Actor与CHORD接线的具体边界

新任务绑定HOT3D1856的61帧wrist-relative物体运动意图，以及exact压缩/实际native probe030的具名初始状态和ctrl。物体参考101个20ms节点，当前手物state58、任务memory83、Actor观察222、20维tanh动作映射到合法绝对ctrl。EMA残差与过去动作外置于WM输入，通过任务memory精确更新；WM仍只接当前/想象state、实际u、h/n及当前几何。未来物体reference是任务命令，未来接触、力、真实next state不能进入WM forward。

已实现reference/observation/control/reward/memory、Reader/replay转换、fresh上游Actor/Critic构造入口，以及Actor→合法ctrl→新WM→world-left解码→任务reward→memory→critic的可调用想象目标。没有把旧PenMDP、right-multiply姿态图表或self15标签静默当作phone语义，也没有完整Actor训练器、Actor检查点或策略结果。

A1为物体跟踪；A2增加明确的engineered hypothetical material-contact prior：以probe030实际接触点和F_on_object法向为模板，当前可微FK输运手侧材料点，按其与冻结phone材料点的距离用1mm带宽soft权重；BODY法向按模板保持，再计算512方向的单位摩擦锥扳手支持并与工程reference比较。它假设材料对应和法向保持，不是当前最近接触点、实测法向/接触力、ego力GT或接触动力学可行证书。8边设计锥不等于native condim3的4边pyramidal求解，mu0.7是名义设计。proxy/objective-known与实际contact/force-known分别保存，后者仍unknown。

三指任务相对当前两pad拟合数据是新分布。训练采集桥沿用Reader已有安全停止，任务POST20ms停止描述也不等于原逐native-tick两秒判据；不能把该桥生成的done/coverage当完整原生评价。后续评价仍须统一真实指定三pad、nonpad/table、自碰撞、穿透、关节、尾速度、无外力及双dt门。

{{TASK_PARAMETERFREE_RESULT}}

## 下一项执行与完整目标保留

先在已加载的新三指任务中做具名原生采样，保留全部失败和partial，从原始PRE contact/POSТ state派生监督并送入replay，再适配共享WM。无需先让手工控制器持握两秒；也不将新族验证数据无声合并进TRAIN。随后实现并运行A1/A2完整Actor训练器，在相同采样/更新预算、初始化和观察权限下比较，并用原生双dt无辅助评价检查是否利用了模型虚假梯度。GPU大batch/多步想象的实际峰值、性能与同设备一致性要在真实训练中报告，不用这次4状态检查作替代。

物体碰撞资产仍是名义仿真版本；真实质量/COM/惯量/摩擦、source重力注册、更多独立ego轨迹、完整61帧机器人执行、多轨迹摊销成本和真机观测/运控合同继续保留。当前未证明完整SuperMimic+CHORD收益、真实ego重定向或整项Goal完成。若启动新的学习实验，仍按用户既定条件在首健康checkpoint和ETA后暂停；本次同一5000实验的历史暂停点不重复触发。
