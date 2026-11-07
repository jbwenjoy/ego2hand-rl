"""Metadata-only research handoff after independent saved-data reviews."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OTHER=Path('D:/workspace/supermimic-chord-study')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).absolute()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,v,replace=False):
    with Path(p).open('w' if replace else 'x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def checks(v):
    assert v['checks'] and all(x is True for x in v['checks'].values()) and v['passed']==v['total']==len(v['checks'])


def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    pred_dir=ROOT/'outputs/full151105_boundary_component_final_evaluation_CPU_v2'
    desc_dir=ROOT/'outputs/full151105_boundary_component_final_prediction_description_v1'
    label_dir=ROOT/'outputs/full151105_boundary_cached_self15_saved_CPU_v1'
    paths={
        'final1000_saved_health':ROOT/'outputs/full151105_boundary_impulse_final1000_saved_independent_v1/report.json',
        'native_prediction_report':pred_dir/'report.json',
        'prediction_description':desc_dir/'report.json',
        'prediction_saved_review':ROOT/'outputs/full151105_boundary_component_final_prediction_saved_independent_v1/report.json',
        'cached_self15_report':label_dir/'report.json',
        'cached_self15_saved_review':ROOT/'outputs/full151105_boundary_cached_self15_saved_independent_v1/report.json',
        'figure':ROOT/'outputs/full151105_boundary_component_final_metrics_figure_v1/fixed_after1000_component_tradeoffs.png',
        'figure_QA':ROOT/'outputs/full151105_boundary_component_final_metrics_figure_v1/visual_QA.json',
        'native_task_integration_source_plan':ROOT/'docs/full151_boundary_components_matched_native_task_integration_source_plan_v1.md',
        'failed_eval_v1':ROOT/'outputs/full151105_boundary_component_final_evaluation_CPU_v1/failure.json',
        'failed_eval_v1_diagnosis':ROOT/'outputs/full151105_boundary_final_eval_adapter_failure_diagnosis_v1.json',
        'native_eval_protocol':ROOT/'protocols/full151105_boundary_component_final_evaluation_CPU_run_v2.json',
        'cached_label_protocol':ROOT/'protocols/full151105_boundary_cached_self15_saved_CPU_run_v1.json',
    }
    for k in ('final1000_saved_health','native_prediction_report','prediction_saved_review','cached_self15_saved_review'):checks(read(paths[k]))
    d=read(paths['prediction_description']);labels=read(label_dir/'three_roles_13cohorts.json')
    roles=d['roles'];c,m=roles[1:];names=['Initial','Public control','Public + consistency']
    scalar_keys=['native52_scaled_MSE_F64_of_saved_FP32_delta','mode_BCE_F64_of_saved_FP32_logits','impulse_z_MSE_F64','saved_defect_consistency_MSE_F64']
    ratios={k:d['statistics'][m]['ALL'][k]/d['statistics'][c]['ALL'][k] for k in scalar_keys}
    no_benefit={k:sum(d['statistics'][m][g][k]<d['statistics'][c][g][k] for g in d['cohorts']) for k in scalar_keys}
    assert all(x>1 for x in ratios.values())
    pins={k:pin(p) for k,p in paths.items()}
    proposal=ROOT/'docs/full151_boundary_public_self15_supervision_recipe_proposal_v1.md'
    if proposal.exists():pins['public_self15_recipe_proposal']=pin(proposal)
    update={'schema':'full151105_boundary_components_final1000_research_update:1','created_UTC':now,'goal_status':'active',
        'status':'FIXED1000_CONTROL_FITS_BETTER_MAIN_FOUR_METRICS_SELF15_CACHED_RISK_FOUND',
        'pins':pins,'method_to_control_ALL_ratios':ratios,'method_lower_cohorts_of13':no_benefit,
        'cached15_ALL':{r:labels[r]['ALL'] for r in roles},
        'scientific_scope':'Dependent old152 training rows/one cached robot lineage. Fixed1000 exploration, no causal explanation, statistical winner, task/Actor/CHORD, forcecapacity/physicalgradient/ego/generalization qualification.',
        'next_actions':['Preserve fixed1000 negative result; no automatic extra iterations or outcome-tuned consistency coefficient.',
                        'Form new matched public native-self15 supervision recipe from original Runner reduction and canonical contact weight, retaining exact cached sensor timing. Do not count this as already trained.',
                        'Require component response/H8 and GPU precision installation qualification before task experiments; one common replay/Actor/Critic/norm/RNG starting point, own namespace and fresh-reset conditional visible state, no105 fake resume.',
                        'Return to physically feasible full151 reference/contact support and matched physical task evaluation; multi-trajectory and real ego/hardware contracts remain open.'],
        'new_training_launched_this_evaluation_turn':False,'running_training_or_evaluation_jobs':[],
        'goal_complete':False,'historical_checkpoint100_pause_already_fulfilled':True}
    up=ROOT/'outputs/full151105_boundary_components_final1000_research_update_v1.json';write(up,update)
    rows=[]
    for title,r in zip(names,roles):
        s=d['statistics'][r]['ALL'];rows.append(f"|{title}|{s[scalar_keys[0]]:.8g}|{s[scalar_keys[1]]:.8g}|{s[scalar_keys[2]]:.8g}|{s[scalar_keys[3]]:.8g}|{s['state_native_delta']['p']['vector_RMS']*1000:.6g}|")
    sensor_rows=[]
    for title,r in zip(names,roles):
        s=labels[r]['ALL'];q=s['confusion_total'];sensor_rows.append(f"|{title}|{s['BCE_mean_entry_F64']:.6f}|{s['Brier_mean_entry_F64']:.6f}|{q['TP']}/{q['FP']}/{q['FN']}/{q['TN']}|{s['pred_intensity_mean']:.6f}|{s['Pen_proxy_penalty_error_F64']['mean']:.6f}|")
    doc=ROOT/'docs/full151_boundary_components_final1000_findings_2026-11-06.md'
    text=f'''# 固定1000次边界组件评价：当前一致性配方未带来拟合收益

2026-11-06。Goal active。上一轮完成两组各1000次更新并按首100健康+ETA暂停，本次恢复后完成最终saved健康5/5、实际3×B152预测、全部152保存复算，以及旧15通道缓存标签的独立评价。此轮没有新训练、Actor或物理执行。

## 固定预算下的主要结果

三角色使用完全相同的152条训练输入、初始norm/尺度/物理context。以下point/mode是从实际捕获FP32残差/logits用F64重算，非原Torch归约逐位重放；冲量/一致性为F64同公式。公共损失相同，method只多λcons=1。不能比较不同定义的训练total来选赢家。

|角色|native52 scaled MSE|mode BCE|z impulse MSE|saved-defect consistency MSE|p vector RMS mm|
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

在ALL上，method/control四项误差比依次为 {ratios[scalar_keys[0]]:.2f}、{ratios[scalar_keys[1]]:.2f}、{ratios[scalar_keys[2]]:.2f}、{ratios[scalar_keys[3]]:.2f}。在13个重叠cohort中，method更小的组数依次为{no_benefit[scalar_keys[0]]}/13、{no_benefit[scalar_keys[1]]}/13、{no_benefit[scalar_keys[2]]}/13、{no_benefit[scalar_keys[3]]}/13。两组都比初态拟合更好，但当前固定1000预算不支持加入一致性项的训练集收益。不能据此断言该约束在其他预算或未见轨迹无效，也不能把原因断言为梯度冲突、某个权重或优化器；本轮没有这些因果消融。

独立保存审计逐行以标量congruence/独立pivot和fsum复算全部3×152辅助量，并核13×3统计、39行cohort CSV、456行CSV、原始norm/context/RNG和706份源归档，11/11通过。native Log本身仍是冻结源码及捕获D52的证据，不独立重放该kernel；矩阵Frobenius差与native LogR残差各自报告，不冒SO3几何证书。PRE1000训练日志来自after999权重，本表来自实际AFTER1000前向。

## 旧self15通道存在实际监督缺口

原WM的self15来自finger_collision与palm-subtree传感器，是Actor soft惩罚的输入；新两个HO/nonhand mode不是其替代。本次公共loss没有旧15 BCE。通过原reward JSON、source/raw25映射和完整(path,k,branch)身份，另立saved-only协议连接全部152×15缓存标签，不增加模型前向或物理。

|角色|cached15 mean BCE|Brier|TP/FP/FN/TN|平均预测强度Σsigmoid|平均Pen代理误差|
|---|---:|---:|---|---:|---:|
{chr(10).join(sensor_rows)}

共228/2280个正标签（10%），A0平均真实强度0、A2为3、ALL为1.5。strict logit>0分类中，对照组成为全负预测；总体accuracy或单一Brier不能概括风险。两final BCE均高于初态，但Brier下降，method的Brier与强度误差MAE又低于control，必须同时保留这些取舍。原Pen代理为-F64(.02)Σsigmoid；误差相对同一缓存binary强度为正表示惩罚低估，final平均偏差加重，不能称完整reward或任务改善。

这些标签是lastsolver≈15ms缓存sensor，state58是post20ms；这沿用原native混合时相合同，非post20几何碰撞truth，也不是HO/人类contact/force-capacity。关系到physical_arrays.branch_self15依赖既有producer及1762项旧saved审阅，本次不读取该NPZ。标签准确度不能升级物理全局自碰或Actor资格。

## 执行与保留证据

最终checkpoint健康5/5。成功native评估仅3次自有CPU加载、1次原生构造、2份raw复制、3次B152前向/456预测、2次辅助头；全程FP32原模型/F64辅助算术，无AD/优化/Actor/MJB/Env/新物理/GPU。原始权重、输入、norm/context、raw RNG及空hook缓存不变。

原v1评估适配器把checkpoint的input_pins错当成协议同名键，实际协议为inputs，在3load/1ctor/2copy但0FW时失败。Root、owner、静审都漏过该字段问题；失败源/结果和缺陷说明原样保留。v2只修映射与独立routes，并用实际冻结协议keyset校验Toy后执行，没有改训练结果、样本、指标、容差或既有源。费用记入ledger，不能把这次失败当模型预测或物理失败。

## 后继研究

保留本轮负面结果，不自动增加训练次数或事后调λ。优先用原Runner已经定义的native15 BCE reduction及canonical contact权重形成新的共同公共监督实验，两臂均保留旧接触通道监督，唯一方法差异仍为cons0/1；新实验须单独recipe/预算/检查点健康。随后评价响应/H8以及GPU精度安装，按一次共同collect/replay/Actor/Critic/RNG起点开展新namespace任务对照，不独立重跑各臂WM预热、不将own组件冒充105恢复，旧WM配套norm不能换成fresh norm混装。

原参考几何FAIL、原A0及三臂task FAIL、有限响应确认FAIL、旧H8取舍/warmup差异、whole-hidden未知仍保留。当前只是单一缓存谱系内的离线组件研究，真实ego N=0，未完成多轨迹共享/heldout及硬件子版本合同。完整目标尚未完成。

- [可复查研究交接]({up.as_posix()})
- [预测独审11]({paths['prediction_saved_review'].as_posix()})
- [缓存15独审]({paths['cached_self15_saved_review'].as_posix()})
- [39组CSV]({(desc_dir/'all39_model_cohort_summary.csv').as_posix()})
- [完整图]({paths['figure'].as_posix()})
- [匹配native任务源计划]({paths['native_task_integration_source_plan'].as_posix()})
'''
    with doc.open('x',encoding='utf-8') as f:f.write(text)
    plan=OTHER/'docs/research-plan-v0.15.md'
    with plan.open('x',encoding='utf-8') as f:f.write(f'''# 研究方案 v0.15：保留一致性负面结果，修补旧接触监督后再接任务

2026-11-06。Goal active。两组固定1000更新的最终健康5/5、实际3×152预测与独立11项保存复算已完成；另已完成缓存self15标签评价及独立审阅。不是任务、CHORD或真实ego收益资格。

当前cons1方法在ALL的point/mode/impulse/consistency四项误差均高于公共损失对照，完整13组均保留，不自动增迭代或调权。旧15未入公共loss，两final BCE较初态恶化、Brier改善并出现低估软惩罚偏差；对照成为全负分类。新的2mode不能替代旧15。标签是原15ms缓存传感器，不能当post20几何truth。

下一项应固定新的共同公共native15监督配方（canonical原loss定义/权重，source明确），再做同initial/context/norm/order/budget的cons0/1对照，首健康checkpoint+ETA按用户约定暂停。尚未启动该新训练。反事实/响应与H8、GPU精度安装和共同fresh-reset/策略起点仍须验证；之后才开展共同warmup/replay/Actor/Critic/RNG的新namespace任务实验，不能假105恢复或换norm坐标系。

原151参考/任务FAIL、finite-response确认FAIL、H8取舍、whole-hidden未知、单谱系/ego0、多轨迹与硬件合同均未解决。完整目标保持，不以离线拟合结束研究。

- [本轮结果/证据/限制]({doc.as_posix()})
- [机器可查交接]({up.as_posix()})
- [上一版]({(OTHER/'docs/research-plan-v0.14.md').as_posix()})
''')
    oldledger=ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v13.json';ledger=read(oldledger)
    ledger['schema']='full151105_matched_replay_dense_compute_ledger:14';ledger['previous_preserved']=pin(oldledger)
    training=read(ROOT/'outputs/full151105_boundary_impulse_matched_training_CPU_v1/result.json')
    ledger['boundary_matched_training_complete1000_CPU']={'report':pin(ROOT/'outputs/full151105_boundary_impulse_matched_training_CPU_v1/result.json'),'training_elapsed_seconds':training['training_elapsed_seconds'],'total_elapsed_seconds':training['total_elapsed_seconds'],'training_counts':training['training_counts'],'process_counts':training['process_counts'],'scope':'Total1000 includes previously reported first100; do not add subset counters again.'}
    ledger['boundary_evaluation_cost_added_since_v13']={'final1000_saved_health_CPU_loads':3,'failed_eval_v1_counts':read(paths['failed_eval_v1'])['counts'],'failed_eval_v1_observed_process_wall_seconds':5.8607458,'successful_eval_v2_counts':read(paths['native_prediction_report'])['counts'],'successful_eval_v2_worker_elapsed_seconds':read(paths['native_prediction_report'])['elapsed_seconds'],'cached_self15_saved_counts':read(paths['cached_self15_report'])['counts'],'independent_saved_reviews_models_FW_AD_physics':0,'new_training_in_this_evaluation_turn':0,'reports':{k:pins[k] for k in ('final1000_saved_health','native_prediction_report','prediction_saved_review','cached_self15_report','cached_self15_saved_review')}}
    lp=ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v14.json';write(lp,ledger)
    for parent in (ROOT,OTHER):
        sp=parent/'STUDY_STATE.json';s=read(sp);s.update(updated_UTC=now,goal_status='active',phase='PROGRESS final1000health5 +actual3x152 predictions +savedindependent11 +cached15 independent reviewed. Cons1 worse thancontrol4primaryfitmetrics; oldself15 supervision gap. Next newmatchedpublicnative15loss recipe, no newtrainingyet.',running_jobs=[])
        s['latest_research_update']=pin(doc);s['latest_research_plan']=pin(plan);s['latest_controlled_compute_ledger']=pin(lp);s['boundary_final1000_research_update']=pin(up)
        b=s.setdefault('full151_native105_boundary_impulse_WM_next',{});b.update(status='FIXED1000_EVALUATED_NO_CURRENT_CONSISTENCY_FIT_BENEFIT_CACHED15_RISK',final1000_evaluated=True,final1000_saved_health=pins['final1000_saved_health'],final_prediction_review=pins['prediction_saved_review'],cached15_review=pins['cached_self15_saved_review'],accuracy_or_benefit_qualified=False,goal_paused=False,new_public_self15_training_started=False,latest_result=pin(up))
        write(sp,s,True)
    with (OTHER/'docs/progress-2026-11-05.md').open('a',encoding='utf-8') as f:f.write(f'\n\n{now}: Goalactive; final1000health5, actual3x152+indep11, cached15labels+indep completed. Fixed1000cons1 worse4primaryfitmetrics; retained self15BCE/Brier/recall/proxy tradeoffs. Failedv1 ABI3load/1ctor/0FW preserved, v2 scientificsame successful. Newpublicnative15 recipe needed, no newtraining launched. {doc.as_posix()}\n')
    print(json.dumps({'update':pin(up),'findings':pin(doc),'plan':pin(plan),'ledger':pin(lp),'goal_status':'active','new_training_launched':False,'method_to_control_ALL_ratios':ratios}))


if __name__=='__main__':main()
