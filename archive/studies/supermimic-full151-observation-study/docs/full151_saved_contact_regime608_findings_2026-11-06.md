# 全部608条分支子步的接触记录与有限动作响应

2026-11-06。全部保存数据汇总完成，独立复算 **22/22 PASS**。本轮只处理原数据，未新增模型加载、前向、AD、MJB、FK、碰撞查询、物理、GPU或训练。

## 覆盖与规则

从原1808条物理记录按 `branch_id is not None` 机械提取全部152×4=608条 solver PRE；保留全部2831条生成接触的原contact index、顺序geom/body IDs、距离、frame、原decoded force6、raw type/world/efc槽。完整catalog含73 geom、31 body、21个enabled手geom和3个物体geom。没有按力、误差、旧stable9或接触类别筛样。

主类为手物1140条、手部自接触475条、原other1216条。全部2831条通过本次保存字段的约束证据资格检查；其中2717条原decoded Fn严格大于1e-6 N。这只验证CONSTRAINT位（含type3）、world0、host地址与raw槽范围等保存证据，不等完整摩擦锥重解码或真实力标签。114条未达正Fn条件的生成候选仍保留。

38条样本的全部生成候选为空。空类别最小距离为null、observed candidate depth为0；所有样本的全局间隙仍UNKNOWN。时相为刚完成求解的PRE pose/COM，不能与postintegration q/qvel/time混为同一时刻。原世界合力及COM扳手逐字段照抄，未重新旋转、求和或计算力矩。

## 全8组结果

每组均有19个独立恢复区间、76条PRE记录。下表“有HO/正Fn的样本”是样本计数；“候选条数”是contact长表行数，不是持续接触轨迹。

| 路径 / k | 有HO样本 | 有合格原正Fn的HO样本 | HO候选条数 | 自接触候选条数 | object-with-nonhand候选条数 |
|---|---:|---:|---:|---:|---:|
| A0 / 0 | 76 | 76 | 190 | 0 | 0 |
| A0 / 49 | 76 | 76 | 152 | 0 | 0 |
| A0 / 99 | 76 | 76 | 380 | 0 | 0 |
| A0 / 149 | 76 | 76 | 304 | 0 | 0 |
| A2 / 0 | 38 | 38 | 114 | 0 | 0 |
| A2 / 49 | 0 | 0 | 0 | 171 | 0 |
| A2 / 99 | 0 | 0 | 0 | 152 | 608 |
| A2 / 149 | 0 | 0 | 0 | 152 | 608 |

A2/k0的每条分支在后两个子步没有保存候选，合计38个valid-empty样本。A2/k49的全部76条PRE只记录手部自接触。A2/k99和149的物体非手候选明确为geom `(0,2)` / `(0,3)`，名字 `terrain`–`object/A_bottom` / `object/A_top`，body IDs为1–2、名字为`terrain`–`object/cube`；每组608条均具备本次合格的原正Fn证据。这里依据实际名字描述，未将所有“非手”自动称为静态环境。

此前已知A2/k49、99、149的全部48个扰动端点，其物体p/R相对baseline编码为零响应；本轮三组全部228条PRE也没有生成HO候选。其中只有6个端点的完整u20未改变，另外42个端点控制改变而WM仍预测非零物体响应。全部24个±pair的实际控制差非零。

因此，“后三处记录的手物接触缺失与有限物体响应平坦并存”得到全量保存证据支持。k49的物体未出现在候选中，k99/149有物体–地形候选，二者不同。不能把这些描述升级为全局无接触、真实Jacobian为零、完整隐藏状态相同或唯一误差成因。A0的全部304条PRE有HO候选，也不能据此推导任意动作都能产生所需物体扳手。

## 实施与验证

Root唯一实际reader运行exit0，用时8.0656668秒，归档723个按实际文件身份+SHA去重的来源文件。focused consumer合同9/9、独立数学7/7通过。实际独审v1为19/22，失败是adapter漏映射最终ABI中的原HO active字段及snapshot全19区间汇总字段；原失败与源码保留。新adapter v2只补映射，独立数学与阈值不变，同一实际产物复算22/22通过，未重跑producer或修数据。

完整四面板图覆盖全部608样本，Root已查看整图；图中空HO标记明确保留clearance UNKNOWN。作者全8组描述独立标为作者解释，不充独立审计。

## 下一步与界限

选择一个部件修正假设：共同PRE坐标中的有限响应监督，比较原点式状态损失与加入该响应项的WM校准。两个副本必须共享初始参数、normalizer、数据和更新预算。旋转使用同一PRE的native52坐标，不能把两侧各自minus坐标中的旋转log直接相减。

拟采集的 `[24,74,124,148]` 是新增受控扰动响应相位；它们的nominal donor状态已经观察过，不能称未见状态或独立轨迹。正式recipe和评价式冻结后才采集、封存响应，两臂final1000冻结后解封。新worker与WM-only adapter目前仅源码草案，尚无新物理或训练。若启动训练，仍在首个checkpoint100健康核验及ETA保存后按用户要求暂停Goal。

物理参考几何失败、完整三臂任务失败、旧5/9方向结果、warmup分歧、graph scratch未知及单缓存谱系/seed0限制保留。真实ego数据数量为0；本轮不构成CHORD任务收益或ego泛化证据。

## 证据

- [实际报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_contact_regime608_CPU_v1/report.json)，SHA `fa62879353cebdbbeee8032fe1f3fed601da76b6b9907725698020790c181932`
- [独立22项报告](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_contact_regime608_saved_independent_v2/report.json)，SHA `f8fe6845e117b8c3921a6b9b912671c0a83b4d5cdd264527c6a107d9bdd5c663`
- [全部608样本](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_contact_regime608_CPU_v1/sample608.json)、[全部contact长表](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_contact_regime608_CPU_v1/contact_long.jsonl)、[全部64对](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_contact_regime608_CPU_v1/response_pairs64.json)
- [完整图](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_contact_regime608_figures_v1/all608_saved_contact_regime.png)与[全8组CSV](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_contact_regime608_figures_v1/all8_snapshot_descriptors.csv)，Root视觉QA SHA `0e52da0a9851296a290a992c47fede615738fc3da900b708f51586bc846083d0`
- [作者解释](D:/workspace/supermimic-full151-observation-study/outputs/full151105_saved_contact_regime608_author_description_v1/description.md)与[下一候选proposal](D:/workspace/supermimic-full151-observation-study/docs/full151_WM_common_pre_response_calibration_proposal_v1.md)
