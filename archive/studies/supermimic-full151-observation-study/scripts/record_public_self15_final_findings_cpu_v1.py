"""Metadata-only verified public15 fit and endpoint-response research handoff."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OTHER=Path('D:/workspace/supermimic-chord-study')
ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
PRIMARY=('native52_scaled_MSE_F64_of_saved_FP32_delta','mode_BCE_F64_of_saved_FP32_logits','impulse_z_MSE_F64','saved_defect_consistency_MSE_F64')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).absolute()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,v,replace=False):
    with Path(p).open('w' if replace else 'x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def checks(r):
    assert r['checks'] and all(v is True for v in r['checks'].values()) and r['passed']==r['total']==len(r['checks'])


def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    paths={
      'new_final1000_health':ROOT/'outputs/full151105_boundary_public_self15_final1000_saved_independent_v1/report.json',
      'new_native_eval':ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_CPU_v1/report.json',
      'new_description':ROOT/'outputs/full151105_public_self15_final_prediction_description_v1/report.json',
      'new_prediction_review':ROOT/'outputs/full151105_public_self15_final_prediction_saved_independent_v1/report.json',
      'endpoint_response':ROOT/'outputs/full151105_public_self15_endpoint_response_CPU_v2/report.json',
      'endpoint_review':ROOT/'outputs/full151105_public_self15_endpoint_response_saved_independent_v1/report.json',
      'endpoint_failed_v1':ROOT/'outputs/full151105_public_self15_endpoint_response_CPU_v1/failure.json',
      'endpoint_invocation_diagnosis':ROOT/'outputs/full151105_public_self15_endpoint_response_invocation_diagnosis_v1.json',
      'figure':ROOT/'outputs/full151105_public_self15_final_metrics_figure_v1/fixed_after1000_public_self15_tradeoffs.png',
      'figure_QA':ROOT/'outputs/full151105_public_self15_final_metrics_figure_v1/visual_QA.json',
      'prior_negative_findings':ROOT/'docs/full151_boundary_components_final1000_findings_2026-11-06.md',
      'next_response_gradient_source_plan':ROOT/'docs/full151_public_self15_action_response_gradient_source_plan_v1.md',
      'native_task_integration_source_plan':ROOT/'docs/full151_boundary_components_matched_native_task_integration_source_plan_v1.md',
    }
    for k in ('new_final1000_health','new_native_eval','new_prediction_review','endpoint_review'):checks(read(paths[k]))
    d=read(paths['new_description']);e=read(paths['endpoint_response'])
    assert e['counts']['pairs']==64 and e['counts']['actual_defined_pairs']==64 and e['counts']['actual_undefined_pairs']==0
    rows=read(paths['endpoint_response'].parent/'all64_pair_records.json');baseline=read(paths['endpoint_response'].parent/'baseline_repeat_descriptors8.json')
    assert len(rows)==64 and len(baseline)==8
    c,m=ROLES[1:];cs=d['statistics'][c]['ALL'];ms=d['statistics'][m]['ALL']
    ratios={k:ms[k]/cs[k] for k in PRIMARY};assert all(v>1 for v in ratios.values())
    for role in (c,m):
        q=d['statistics'][role]['ALL']['cached_self15']['confusion_total'];assert (q['TP'],q['FP'],q['FN'],q['TN'])==(228,0,0,2052)
    relative={role:{block:s['ALL']['response_error']['blocks'][block]['component_RMS']/s['ALL']['physical_response']['blocks'][block]['component_RMS']
                    for block in ('q','qd','p','R_raw9','v','omega')} for role,s in e['summary'].items()}
    descriptors={'same_visible_x58_bits_pairs':sum(r['same_visible_x58_bits'] for r in rows),'max_abs_offaxis_du':max(abs(v) for r in rows for v in r['offaxis_delta_u20_rad_F64']),
      'actual_selected_du_range_rad':[min(r['actual_coordinate_denominator_rad'] for r in rows),max(r['actual_coordinate_denominator_rad'] for r in rows)],
      'same_baseline_u_bits_snapshots':sum(r['baseline_u_three_bits_equal'] for r in baseline),
      'largest_baseline_physical_range58':max(v for r in baseline for v in r['descriptors']['physical']['range58']),
      'scope':'Serialized visible values and three repeated baseline samples only; no wholehidden/noise bound or pure physical Jacobian proof.'}
    pins={k:pin(v) for k,v in paths.items()}
    update={'schema':'full151105_public_self15_final_evaluation_research_update:1','created_UTC':now,'goal_status':'active',
      'status':'PUBLIC15_CACHED_TRAINING_CLASSIFICATION_RESTORED_CONS1_NO_PRIMARY_FIT_GAIN_ENDPOINT_RESPONSE_WEAK',
      'pins':pins,'method_to_control_ALL_primary_MSE_ratios':ratios,
      'cached15_ALL':{r:d['statistics'][r]['ALL']['cached_self15'] for r in ROLES},
      'endpoint_response_error_component_RMS_div_physical_response_component_RMS':relative,'endpoint_visible_descriptors':descriptors,
      'scientific_scope':'Known old152 robot training rows, one cached lineage,13 overlapping statecohorts; endpoint64/13paircohorts are same training cases. Cached15≈15ms vsstatePOST20, not globalgeometry. Differentcons-weighted total not a winner criterion. No task/Actor/CHORD/forcecapacity/trueJac/wholehidden/ego/generalization qualification.',
      'next_actions':['Keep current fixed1000 result and all prior negative evidence; no automatic iteration or lambda search.',
        'Prepare action-response/u-gradient diagnostics with distinct fresh24/74/124/148 source contracts. Freshx152 currently repeats snapshot8 by input design and is not observedPRE152; qualify or explicitly retain this conditional scope before physical matching claims. No freshJ/K/r_saved labels exist.',
        'Evaluate multi-step prediction using qualified native donor inputs/teacher controls with H8, without calling it policy imagination or independent trajectory generalization.',
        'Before full task comparisons, add a proper same-budget native point+self15-only trained baseline to distinguish auxiliary J/K/mode supervision from consistency, and enforce one real common warmup/Actor/Critic/replay/norm/RNG starting point with own install namespace.',
        'Original reference/task failures, contact force-capacity/geometry feasibility, multi-trajectory/ego/hardware contracts remain open.'],
      'new_training_in_this_turn':False,'running_training_or_eval_jobs':[],'goal_complete':False}
    up=ROOT/'outputs/full151105_public_self15_final_research_update_v1.json';write(up,update)
    names=('Initial','Public incl. native15','Public + consistency')
    table=[];sensor=[];response=[]
    for name,role in zip(names,ROLES):
        s=d['statistics'][role]['ALL'];q=s['cached_self15'];z=q['confusion_total'];p=e['summary'][role]['ALL']
        table.append(f"|{name}|{s[PRIMARY[0]]:.8g}|{s[PRIMARY[1]]:.8g}|{s[PRIMARY[2]]:.8g}|{s[PRIMARY[3]]:.8g}|{s['state_native_delta']['p']['vector_RMS']*1000:.6g}|")
        sensor.append(f"|{name}|{q['BCE_mean_entry_F64']:.8g}|{q['Brier_mean_entry_F64']:.8g}|{z['TP']}/{z['FP']}/{z['FN']}/{z['TN']}|{q['pred_intensity_mean']:.8g}|{q['Pen_proxy_penalty_error_F64']['mean']:.8g}|")
        response.append(f"|{name}|{p['response_error']['blocks']['p']['component_RMS']*1000:.6g}|{p['response_error']['blocks']['R_raw9']['component_RMS']:.7g}|{p['response_error']['blocks']['omega']['component_RMS']:.7g}|{p['actual_secant_error']['blocks']['omega']['component_RMS']:.7g}|")
    doc=ROOT/'docs/full151_public_self15_final1000_findings_2026-11-06.md'
    text=f'''# 公共native15固定1000评价：旧通道拟合恢复，控制响应仍不足

2026-11-06。Goal active。本次恢复后完成新final1000健康5/5、三模型实际3×B152评价和独立12/12；再从同一批保存预测计算全部64对控制扰动响应并独审10/10。本轮没有新训练、Actor、AD或物理采样。

## 同一新公共监督下比较cons0/1

三角色同初始规范、原152输入/顺序/物理context与原norm。以下point/mode是捕获FP32残差/logits的F64重算；FP32 native_pair/八项scalar单独保留。共同公共项逐行 `mean152(point_row+.1*sum15 BCE)` FP32，再+modeFP32+impF64；只有cons0/1不同，不能按异权total选赢家。

|角色|native52 scaled MSE|mode BCE|z impulse MSE|saved-defect cons MSE|p vector RMS mm|
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

method/control四项误差比为 {ratios[PRIMARY[0]]:.2f}、{ratios[PRIMARY[1]]:.2f}、{ratios[PRIMARY[2]]:.2f}、{ratios[PRIMARY[3]]:.2f}。固定1000预算下，当前一致性项没有带来ALL训练集主要拟合收益；两者均显著优于未更新initial。这不证明其他预算/轨迹上无效，也不解释原因。没有以结果调λ或追加更新。

## 缓存15标签不再漏检，但只证明训练内预测

|角色|cached15 mean BCE|Brier|TP/FP/FN/TN|平均Σsigmoid|平均Pen代理误差|
|---|---:|---:|---|---:|---:|
{chr(10).join(sensor)}

同一152×15中228 positive、2052 negative；两trained role对228 positive全部检出且无误报。对比此前无self15公共监督的全负control及method漏报，新的共同监督实验恢复了这些训练条目上的预测。两者软惩罚偏差接近0；负代理误差表示相对缓存binary强度的轻微过度惩罚，不是完整reward改善。旧无self15实验的BCE/Brier取舍和负面结果保留，跨stage公共目标不同，FP32状态归约也作了逐行澄清，不能把变化拆成严格单因果效应。

标签仍是约15ms缓存finger sensor，state target为POST20。A0全负、A2平均3positive及19个依赖分支，使这些不是2280个独立观测；不得推广成POST20几何、HO/人类接触、独立轨迹或任务成功率。

## 点拟合不等于扰动响应拟合

旧相位0/49/99/149的8snapshot，每个4方向×2epsilon，机械连接64对minus/plus。全部原始响应/误差、actual控制差、actual状态差、selected-slot/nominal两分母和13pair-cohort都保留，没有stable/误差筛样。actual denominator是选中control坐标du_j，原程序未把它当全20维norm；独审另确认本批offaxis=0，故该批数值才恰与向量norm绝对值一致。

|角色|p响应误差 component RMS mm|rawR9响应误差 component RMS|omega响应误差 rad/s|omega actual-secants误差 (rad/s)/rad|
|---|---:|---:|---:|---:|
{chr(10).join(response)}

对应物理响应本身的component RMS为p=0.0166978mm、rawR9=0.000321183、omega=0.0263597rad/s；control的p响应误差约为信号的{relative[c]['p']:.3f}倍、omega约{relative[c]['omega']:.3f}倍。method在rawR9/omega有部分相对control改善、在q/qd/p/v没有统一改善；不能由小的绝对点误差推断可用的Actor动作梯度。

64对actual denominator均非零，原undefined机制仍保留；实际du_j范围[{descriptors['actual_selected_du_range_rad'][0]:.10g},{descriptors['actual_selected_du_range_rad'][1]:.10g}]rad。该批64对PRE58逐位相同、offaxis控制差为0，8组baseline的三次PRE58、actualu与真实endpoint在保存精度下无离散度；这仅是本次可见条件与3次重复的描述，不能证明whole-hidden一致、噪声上界或连续physical Jacobian。R9保持raw dimensionless矩阵，未作Log或投影。

## 数据与执行边界

新AFTER1000评价仅3个own CPU加载、1native constructor、2raw copies、3B152 FW/456预测、2head FW，原norm/context/RNG等守门通过；保存独审12/12核727源、全部152/39组/456表及旧物理项独立scalar-pivot、raw25标签映射。64对分析仅4NPZ读取、零模型；独审10/10再核4original+2saved NPZ，全部raw算术精确、fsum reduction最大差3.55e-15。

64对初次v1因Root传相对protocol路径，在4NPZ读取/统计保存后被最终absolute-pin守门拒绝，未形成成功report。失败、部分数据和诊断原样保留；v2只用新协议/目录和绝对CLI路径，冻结代码、输入、公式、门和mask全部不变，重新一次完成，无自动retry。额外4读取计入费用。

当前old152与另一套fresh152的相位24/74/124/148不同，fresh6stable/25weak/1unstable标签不贴到本次64对。fresh152模型x是8snapshot重复的输入设计，未被独立资格化为逐分支actualPRE58；没有独立fresh J/K/r_saved包，不能套old冲量truth或cons。

下一步应在明确fresh可见输入/恢复限制后做独立响应/uVJP诊断，并评价已资格native donor的H8 teacher-control多步预测；不称policy imagination或trueJac。完整组合收益还需同预算的原native point+self15-only训练对照，区分J/K/mode共同监督与cons贡献，以及共同真实warmup/Actor/Critic/replay/norm/RNG起点的新namespace任务实验。当前public control已经含J/K/mode，不能当原SuperMimic-only对照。

原151参考几何/任务FAIL、旧finite-response确认FAIL、旧H8/warmup取舍、wholehidden未知、force-capacity/接触可行性、多轨迹、ego0及硬件合同仍未解决。完整目标尚未完成，Goal保持active；本轮没有新训练正在等待。

- [机器交接]({up.as_posix()})
- [预测独审12]({paths['new_prediction_review'].as_posix()})
- [64对独审10]({paths['endpoint_review'].as_posix()})
- [全图]({paths['figure'].as_posix()})
- [后继响应/梯度源计划]({paths['next_response_gradient_source_plan'].as_posix()})
'''
    with doc.open('x',encoding='utf-8') as f:f.write(text)
    plan=OTHER/'docs/research-plan-v0.17.md'
    with plan.open('x',encoding='utf-8') as f:f.write(f'''# 研究方案 v0.17：缓存接触拟合恢复，继续验证控制响应与任务可用性

2026-11-06。Goal active。新public-native15 final1000健康5/5、实际3×152及保存独审12/12、旧64训练对有限endpoint响应独审10/10完成。两个trained模型缓存15均TP228/FP0/FN0/TN2052，仍是单缓存谱系的训练内结果；cons1在ALL四主要拟合误差仍高于cons0。小点误差没有带来同等精度的物体扰动响应，不能直接推Actor梯度或任务收益。

下一阶段保留固定1000结果，不自动增迭代/调λ。分别推进：

1. 依据fresh24/74/124/148源合同资格化可见输入与恢复条件；当前fresh152是snapshot8重复设计，不是已证actualPRE152。若执行common-visible task uVJP/finite响应，必须明确条件限制，保全部32方向/64对及既定6/25/1子组，不能冒truephysicalJac或套old J/K/r_saved。
2. 对已资格native donor输入/实际teacher controls做H8多步预测，分别报告状态与缓存15随horizon误差；不称Actor imagination、独立轨迹泛化或新物理。
3. 完整组合收益须native point+self15-only同预算trained baseline，当前公共对照已含J/K/mode。只有响应/多步与GPU精度安装/norm契约足够，才进入一次共同collect/replay/Actor/Critic/RNG起点、fresh-reset条件状态的新namespace任务比较；不能假105完整恢复或换norm坐标系。

原参考/任务FAIL、有限响应确认FAIL、旧H8/warmup取舍、wholehidden未知、物理接触支持/forcecapacity、多轨迹/ego0/硬件合同均保留。完整研究目标未完成；当前没有运行的训练或评测，本轮没有新训练触发暂停。

- [本轮结果与限制]({doc.as_posix()})
- [机器交接]({up.as_posix()})
- [上一版]({(OTHER/'docs/research-plan-v0.16.md').as_posix()})
''')
    oldledger=ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v15.json';ledger=read(oldledger)
    ledger['schema']='full151105_matched_replay_dense_compute_ledger:16';ledger['previous_preserved']=pin(oldledger)
    tr=ROOT/'outputs/full151105_boundary_public_self15_matched_training_CPU_v1/result.json';train=read(tr)
    ledger['public15_complete1000_total']={'report':pin(tr),'training_counts':train['training_counts'],'process_counts':train['process_counts'],'training_elapsed_seconds':train['training_elapsed_seconds'],'total_elapsed_seconds':train['total_elapsed_seconds'],'scope':'Full1000 includes priorfirst100 subset; do not sum both.'}
    ledger['public15_evaluation_added_since_v15']={'final1000_saved_health_CPUcheckpoint_loads':3,'final1000_saved_health_cached_JSON_reads':1,
      'native_eval_counts':read(paths['new_native_eval'])['counts'],'native_eval_worker_seconds':read(paths['new_native_eval'])['elapsed_seconds'],
      'saved_prediction_review_no_models_FW_AD_physics':True,'endpoint_failed_v1_counts':read(paths['endpoint_failed_v1'])['counts'],'endpoint_success_v2_counts':e['counts'],
      'endpoint_success_v2_worker_seconds':e['elapsed_seconds'],'endpoint_independent_NPZ_reads':6,'endpoint_saved_review_model_FW_AD_physics':0,
      'new_training_in_this_turn':0,'reports':{k:pins[k] for k in ('new_final1000_health','new_native_eval','new_prediction_review','endpoint_response','endpoint_review','endpoint_invocation_diagnosis')}}
    lp=ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v16.json';write(lp,ledger)
    for parent in (ROOT,OTHER):
        sp=parent/'STUDY_STATE.json';s=read(sp);s.update(updated_UTC=now,goal_status='active',goal_pause_condition_met=False,
          phase='PROGRESS newpublic15final1000health5+actual3x152+saved12;cached15trainingclassificationrestored;cons1fourfitmetricsworse. Old64endpointresponse+saved10showsweakobjectresponse. Next freshinput/u-gradient qualification and qualifieddonorH8; fulltask/ego unresolved.',running_jobs=[])
        s['latest_research_update']=pin(doc);s['latest_research_plan']=pin(plan);s['latest_controlled_compute_ledger']=pin(lp);s['public15_final_research_update']=pin(up)
        b=s.setdefault('full151_native105_boundary_public_self15_experiment',{});b.update(status='FINAL1000_EVALUATED_CACHED15_FIT_RESTORED_RESPONSE_NOT_QUALIFIED_FOR_POLICY',
          final1000_saved_health=pins['new_final1000_health'],final_prediction_review=pins['new_prediction_review'],endpoint_response_review=pins['endpoint_review'],
          training_set_cached15_metrics_verified=True,task_or_generalization_benefit_qualified=False,physical_gradient_qualified=False,goal_paused=False,latest_result=pin(up))
        write(sp,s,True)
    with (OTHER/'docs/progress-2026-11-05.md').open('a',encoding='utf-8') as f:f.write(f'\n\n{now}: Goalactive. Public15finalhealth5+3x152 actual+saved12; cached15 TP228 FP0 FN0 TN2052 bothnewarms. Cons1fourmainALLfitmetrics worse. Old64response+saved10, actualinputvisiblebits64/offaxis0 but no hidden/Jac certificate. Endpointv1relativepathfailure4reads preserved; v2absolutepath4readscompleted, sourceunchanged. No newtraining; next freshqualifiedresponse/uVJP andnative donorH8. {doc.as_posix()}\n')
    print(json.dumps({'update':pin(up),'findings':pin(doc),'plan':pin(plan),'ledger':pin(lp),'goal_status':'active','primary_method_to_control_ratios':ratios,'response_error_to_signal_component_RMS':relative}))


if __name__=='__main__':main()
