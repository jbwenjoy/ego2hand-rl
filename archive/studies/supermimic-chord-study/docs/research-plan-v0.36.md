# 研究方案 v0.36：三指原生动作响应与固定最终模型诊断

2026-11-07，承接[v0.35](research-plan-v0.35.md)与完整Goal。本回合取得实际 progress：同一GPU配对训练的500检查点通过保存读回；新增永久heldout三指动作干预数据并完成保存及数值核查。最终1000轮、策略成功与完整目标尚未完成。

## 同一训练继续，500检查点健康

GPU worker28104、supervisor24420继续原100→1000恢复实验。paired500时shared WM新实验累计更新600次，两臂各Actor500/Critic500。producer保存读回15/15，检查点SHA `a43c51aa181d56e887dd7ea2b41b6a5a1382ab35369a03512fa8063abc060b3d`。Root另核当前文件字节/hash、健康头、实际cuda协议和readback文件，不新增checkpoint对象加载、NN或物理调用。[500产物核验收据](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/checkpoint500_artifact_receipt_v1.json)

该点是同一已恢复实验的中途检查，不据MONITOR或loss选模型，不重复历史首次健康点暂停。训练仍以固定1000结束，之后才评价学习策略。

## 真实三指动作响应已采集

机械固定MON family12/15 × 两dt × step0/3，共8个已知上下文。source qpos/qvel来自原reset或前一POST，原episode安全原点保留；用同一冻结NativeSavedActionProbe直接恢复完整300值receipt，不增加恢复/观察forward。

先重放全部8个原控制，逐native-tick trace、source/after-action/target300及canonical标签字节全部相同后，才跨越全局屏障执行任何新侧。每个上下文测试20轴的正负名义0.01rad绝对控制变化，采用真实ctrlrange剩余空间，无clip/rescue。实际320侧均可用且未触发bound缩小；290侧位于当前Actor残差+EMA映射的归一化范围，30侧在范围外，两者只作描述，均保留。这是绝对控制干预，不冒称实际Actor输出。

[实际采集报告](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_v1/native_completed_v1/report.json)：328条全部完整、标签已知，0partial；31488积分步、48072次力读取、1312次receipt读取、2次MJB、328个MjData，initial/restore/observer forward均0，NN与学习更新0。实际worker17996 exit0，约15.59秒。各20ms假设分支不是一条连续持握轨迹，也不证明任意状态恢复或完整MjData/Markov可观测性。

原始trace先持久化，再从实际PRE contact/frame/符号/逐PRE移动COM产生22对source-BODY冲量、meanFn、边际coverage及真实三指同时coverage。旧probe59量明确另列为原两pad辅助，不混作三指真值。全部数据角色固定为 `ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE`，永久不参与任何拟合或检查点选择。已知MON及共享起点lineage不冒称独立ego/未见状态。

[保存完整性核查](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_v1/saved_integrity_review_v1/report.json)10/10：全328 raw/labeled、动作与来源、原安全初态、8baseline屏障、预算及partial/alias保持一致。这是源码作者保存完整性范围，不重复宣称独立物理标签证明。

## 原生响应的数值事实

[全部320侧的数值报告](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_v1/saved_response_statistics_v1/report.json)保留所有轴、dt、step和22对，无bestcase筛选。

COM末端位移响应有86/320精确F64零；全部侧的向量norm RMS为37.086µm，非零234侧为43.369µm。端点 `Log(Rside Rbase.T)` 测地旋转有80条精确零，全部RMS为0.06550°，非零240侧为0.07563°；原delta52中的两个source-Log之差另存，不混为同一个旋转量。

完整22对的force/moment冲量响应各只有1条整体精确零，但每对稀疏性另存；Fn响应15条整体零、coverage152条整体零、真实三指联合coverage响应212条零。step3的160侧三指coverage变化均0；这个变化量本身不说明coverage绝对值。轴16–19的COM效应约4e−16..3e−14m，即使F64非零也不能据此宣称强物理信号；没有发明测量噪声或物理精度阈值。

独立scalar/fsum在328分支、31851个phone contact行上核查world wrench及动量：线/角冲量离散缺陷RMS约6.429e−8N·s / 1.123e−7N·m·s，总world wrench相对Reader最大差约1.11e−15。保留这些非零及其离散口径，不设置物理证书或无限小Jacobian容差。

[科研图](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_v1/saved_response_statistics_v1/native_finite_action_COM_and_pair_impulse_v1.png)展示双dt×20轴COM响应和20×22 force-impulse响应，moment不混单位。图已视觉检查。两dt响应不同仍不构成步长收敛或精度证明。

## 固定最终模型与AD/FD的比较入口

[新动作响应评测源码](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_action_response_evaluation_v1/SOURCE_DELIVERY_v1.json)已冻结、元数据预检和独立窄源码审阅通过，实际数据报告/review/terminal已绑定，仅final1000四文件待绑定。

固定比较warm5000与offline final1000/WM1100，全部328行、320侧、160个双侧记录及所有mask保留。模型在相同实际u端点的有限响应是原生比较主项；实际F64 du、原控制FP32分离度和真正归一化NN FP32分离度另列。模型只有边际22coverage头，绝不以乘积伪造三指联合预测。

另在8个baseline上每模型计算20动作轴、274输出的Jacobian及±1e−4rad模型FD自一致，上限20次NN前向、16个Jacobian API。每Jacobian为一次forward加一个含274 cotangent方向的batched reverse VJP，16次等价4384标量反向方向，不能写作16次标量AD。selfcheck按各状态单位及物理178的六个单位块分别报告；不混合rad/m/Ns/Nms给出赢家。

H1当前state/geometry对u为常量，动作导数只经过u槽、NN和decoder；它不验证geometry对q的梯度、H2全链或原生无限小接触Jacobian。模型±1e−4自检与原生±0.01有限动作是不同实验。weights/norm/RNG冻结，0拟合、0native、0GPU。尚未实际执行上述NN或AD。

## 训练完成后的实际队列与下一阶段

现有原生策略评测等待进程41220先等原训练terminal0及final1000/500健康，然后执行原四组无辅助两秒案例。新增[固定诊断supervisor](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_posttraining_v1/execute_fixed_diagnostics_after_training_v1.py)已启动，PID49908 / session2506，等待四案例被成功记录；物理门失败仍作为结果，不筛掉。它先让两项boundFalse元数据全部通过，再依次执行旧/新分布4FW诊断和本次20FW/16Jac动作响应诊断。任何软件失败均保留，不自动retry或重启训练；没有启动online新学习阶段。

后续先依据固定1000的真实物理与响应证据评估限制，再验证已实现的三轮online采集/共享池/WM与两臂更新循环。若启动新学习实验，仍在首cycle1/update10健康和ETA后按用户规则暂停。

校准与鲁棒性、源重力、独立ego/RGB手部重建及机器人输入契约、完整61帧执行、多轨迹共享成本/匹配对照与硬件契约全部继续保留。当前仍未证明两秒持握或CHORD收益；完整Goal保持active、未完成。本轮具体产物与实时入口见[里程碑v17](D:/workspace/ego-data-pilot-2026-11-07/outputs/goal_extension_tripod_native_action_response_milestone_v17.json)。
