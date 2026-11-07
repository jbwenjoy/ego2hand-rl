"""Metadata-only first healthy public-native15 checkpoint and pause handoff."""
from pathlib import Path
import datetime
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parents[1]
OTHER=Path('D:/workspace/supermimic-chord-study')
RUN=ROOT/'outputs/full151105_boundary_public_self15_matched_training_CPU_v1'


def read(p):
    # A live runtime file may be between two writes. Re-observe that same file;
    # a transient JSON decode is not a stopped worker or a retry of training.
    for attempt in range(3):
        try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
        except json.JSONDecodeError:
            if attempt==2:raise
            time.sleep(.03)
def pin(p):return {'path':str(Path(p).absolute()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,v,replace=False):
    with Path(p).open('w' if replace else 'x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    hp=ROOT/'outputs/full151105_boundary_public_self15_checkpoint100_saved_independent_v1/report.json'
    h=read(hp)
    assert h['schema']=='full151105_boundary_public_self15_checkpoint_saved_review:1'
    assert h['status']=='PASS_SAVED_PAIRED_PUBLIC_SELF15_COMPONENT_HEALTH_NOT_ACCURACY' and h['engineering_qualified'] is True
    assert h['iteration']==100 and h['passed']==h['total']==len(h['checks']) and all(v is True for v in h['checks'].values()) and h['own_CPU_checkpoint_loads']==3
    eta=read(RUN/'checkpoint100_ETA.json');runtime=read(RUN/'runtime.json')
    assert runtime['status']!='FAILED_PARTIAL_NO_RETRY' and not (RUN/'failure.json').exists() and runtime['completed_pair_iteration']>=100
    assert eta['completed_pairs']==100 and eta['remaining_pairs']==900
    pp=ROOT/'protocols/full151105_boundary_public_self15_matched_training_CPU_run_v1.json';p=read(pp)
    assert p['background_continue_to_final'] is True
    terminal=(RUN/'result.json').exists();result=read(RUN/'result.json') if terminal else None
    if terminal:assert result['status']=='COMPLETED_FIXED1000_MATCHED_PUBLIC_SELF15_COMPONENT_TRAINING' and result['iteration']==1000
    with (RUN/'updates.jsonl').open('rb') as f:rows=[json.loads(f.readline()) for _ in range(100)]
    assert [r['iteration'] for r in rows]==list(range(1,101))
    arms=tuple(rows[0]['arms']);assert len(arms)==2
    snapshots={a:{'PRE1_losses':rows[0]['arms'][a]['losses'],'PRE100_losses':rows[-1]['arms'][a]['losses'],'counts_AFTER100':rows[-1]['arms'][a]['counts']} for a in arms}
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    launch=ROOT/'outputs/full151105_boundary_public_self15_matched_training_launch_v1/launch.json'
    pins={'protocol':pin(pp),'preregistration':p['preregistration'],'recipe':p['recipe'],'source_contract':p['source_contract'],'source_review':p['source_review'],
          'paired100':pin(RUN/'paired_checkpoint_0100.json'),'saved_health100':pin(hp),'ETA100':pin(RUN/'checkpoint100_ETA.json'),'hidden_worker_launch':pin(launch),
          'prior_no_self15_findings':pin(ROOT/'docs/full151_boundary_components_final1000_findings_2026-11-06.md')}
    handoff={'schema':'full151105_boundary_public_self15_checkpoint100_goal_pause_handoff:1','created_UTC':now,
        'status':'NEW_PAIRED100_SAVED_HEALTH_AND_ETA_VERIFIED_GOAL_PAUSE_DUE','pins':pins,
        'saved_health_checks':{'passed':h['passed'],'total':h['total']},'PRE_saved_log_snapshots':snapshots,
        'loss_scope':'PRE100 uses weightsafter99; checkpoint100 is AFTER100. No independent model prediction/accuracy evaluation in this handoff.',
        'worker_runtime_snapshot':runtime,'worker_terminal_at_snapshot':terminal,'background_continue_to_final':True,'ETA_at_first100':eta,
        'fit_and_label_scope':'Original152 source-dependent robot W1 rows; cachednative15 lastsolver~15ms vsPOST20 target. Common point+.1*sum15BCE pairFP32, modeFP32, impulseF64; methodonlycons1/control0.',
        'next_after_resume':['Observe existing worker/final result without restarting.', 'New fixed1000 saved-health and matched prediction/self15 cached-label evaluation, retaining all13 cohorts and previous negative result.', 'Response/H8 and GPU install/norm/commonwarmup task qualification remain pending; no current task/CHORD/ego benefit.'],
        'full_goal_complete':False,'accuracy_task_CHORD_physicalgradient_ego_qualified':False}
    if terminal:handoff.update(terminal_result=pin(RUN/'result.json'),remaining_seconds_at_snapshot=0)
    out=ROOT/'outputs/full151105_boundary_public_self15_checkpoint100_pause_handoff_v1.json';write(out,handoff)
    worker_text='记录时两组已完成固定1000更新，实际剩余0；最终1000独审/效果评价仍待恢复后进行。' if terminal else '训练进程获本次协议明确授权在后台继续固定1000，暂停Goal不终止此进程。'
    doc=ROOT/'docs/full151_boundary_public_self15_checkpoint100_findings_2026-11-06.md'
    text=f'''# 两臂共同native self15监督：首个配对检查点与暂停交接

2026-11-06。新namespace实验已实际启动。双方100次检查点已保存，独立健康{h['passed']}/{h['total']}通过，ETA已保存，达到用户约定的本次Goal暂停条件。{worker_text}

本轮从同一未改A2 initial重新开始，原WM/15通道、零14头、有效s52、norm/host/context、152输入顺序相同。新标签仅来自已独核7项的label-only JSON，按原plan逐152身份/raw25映射及15ms缓存时相核对，不使用带模型概率的metric archive作监督、不进入forward。

新公共项严格逐行FP32：s=mean52((D/s52)^2)，c=sum15 BCE，native_pair=mean152(s+.1*c)，再+mode FP32、+impulse F64；两组唯一方法差异为cons0/1。0.1来自原native源码默认，非结果调权；mean15不能替代sum15，不能乘rewarddt或H8。原003物理项和旧5标签字典保留，state仍独立纯point，新native_pair/self15字段不冒旧state。新自15监督没有新Parameter，仍16个tensor。

每组CPU4/FP32、AdamW lr1e-4、betas(.9,.999)、eps1e-8、wd0、clip1，固定1000更新，仅100/1000保存。首次训练前向复用为e427 native58/self15位级检查与正零14，无额外验证FW；norm/context/输入及三种label digest不变。新checkpoint为AFTER-I，原始grad与loss snapshot为PRE-I、权重after(I-1)，不能用PRE100日志冒AFTER100模型复测。

新reader仅3次自有CPU加载（本次initial_pair+双方fixed100），核16参数/Adamsteps/clip前后原始命名梯度、冻结norm/context/host、THIS RUN rawRNG、100行/bytes/SHA前缀，以及PRE-I nativeD52、FP32平方/pointrows、self15 logits/targets/BCE元素与sum15/native_pair/public归约。BCE跨stride重算仅用预锁FP32数值包络，后续精确归约从实际saved BCE元素起；nativeLog与旧impulse/cons的捕获来源不冒本次模型重放。无模型构造/FW/AD/MJB/Env/物理/GPU。

首100测得配对均时{eta['observed_mean_pair_seconds_after_trim']:.6f}s；剩900对+保存/显式20s收尾估计{eta['estimated_remaining_seconds']:.1f}s，预计UTC {eta['estimated_finish_UTC']}。范围为规划启发式，非CI或加速证据；若此处已完成1000，该ETA仅为历史测量。

这只证明新监督/优化/序列化健康，尚未证明拟合、接触惩罚或任务收益。旧无self15实验的固定1000负面结果原样保留；原参考/任务FAIL、有限响应确认FAIL、旧H8取舍/共同warmup缺口、wholehidden未知、单谱系/ego0、多轨迹与硬件合同仍未解决。完整研究目标尚未完成。

- [机器交接]({out.as_posix()})
- [新100健康]({hp.as_posix()})
- [本次冻结配方]({Path(p['recipe']['path']).as_posix()})
- [ETA]({(RUN/'checkpoint100_ETA.json').as_posix()})
'''
    with doc.open('x',encoding='utf-8') as f:f.write(text)
    plan=OTHER/'docs/research-plan-v0.16.md'
    with plan.open('x',encoding='utf-8') as f:f.write(f'''# 研究方案 v0.16：共同native15监督已训练，首100健康后交接

2026-11-06。新的两臂共同native15监督实验已实际启动，配对100 checkpoint与新独立健康{h['passed']}/{h['total']}、ETA已就绪，按用户约定暂停Goal。{worker_text}

两臂同初始WM/零14头/context/norm/152顺序/1000预算；公共native_pair=mean152(point_row+.1*sum15 BCE)FP32，再+modeFP32+impF64，唯一方法差异仍cons0/1。标签仅为旧native15ms缓存sensor，不能视为post20几何或人类contact。checkpoint健康不能当准确性或CHORD收益。

恢复后先观察既有worker/终态，禁止重启消耗同一协议；对固定1000检查点做新saved独审和全152/13cohort预测/自15评价，保留既有无自15实验的负面结果。之后需response/H8、GPU精度/原norm安装、一次共同collect/replay/Actor/Critic/RNG及fresh-reset条件起点，使用新namespace，不冒105完整恢复。

原参考/任务FAIL、有限响应确认FAIL、旧H8/warmup差异、wholehidden未知、多轨迹/ego0/硬件合同仍未解决，完整目标保持。

- [本轮证据与限制]({doc.as_posix()})
- [交接]({out.as_posix()})
- [上一版]({(OTHER/'docs/research-plan-v0.15.md').as_posix()})
''')
    previous=ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v14.json';ledger=read(previous)
    ledger['schema']='full151105_matched_replay_dense_compute_ledger:15';ledger['previous_preserved']=pin(previous)
    ledger['public_native15_matched_training_first100_CPU']={'pair':pins['paired100'],'saved_health':pins['saved_health100'],'ETA':pins['ETA100'],
        'each_arm_updates':100,'training_native_FW_through100':200,'predicted_rows_through100':30400,'own_initial_loads':1,'native_constructor':1,'raw_deepcopies':2,
        'extra_validation_FW':0,'Actor_MJB_Env_physics_GPU':0,'saved_health_own_CPU_checkpoint_loads':3,'saved_health_model_FW_AD_physics':0,
        'worker_terminal_at_handoff':terminal,'scope':'First100 is a subset of authorized fixed1000; final report is authoritative for full job totals. No same-walltime/speedup claim.'}
    lp=ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v15.json';write(lp,ledger)
    for parent in (ROOT,OTHER):
        sp=parent/'STUDY_STATE.json';s=read(sp)
        s.update(updated_UTC=now,goal_status='active',goal_pause_condition_met=True,phase='ACTUAL matchedpublicnative15 firstpaired100 SAVED newindependenthealth+ETA PASS; Goalpause due peruser. Priorno-self15negative result retained, finalqualificationpending.')
        s['latest_research_update']=pin(doc);s['latest_research_plan']=pin(plan);s['latest_controlled_compute_ledger']=pin(lp);s['boundary_public_self15_checkpoint100_pause_handoff']=pin(out)
        s['running_jobs']=[] if terminal else [{'kind':'explicit_background_fixed1000_publicnative15_CPU','run_dir':str(RUN),'pid':read(launch)['pid'],'launch':pin(launch),'protocol':pin(pp),'last_saved_qualified_iteration':100}]
        s['full151_native105_boundary_public_self15_experiment']={'status':'NEW_PAIRED100_SAVED_HEALTH_AND_ETA_PASS','recipe':p['recipe'],'source_contract':p['source_contract'],'protocol':pin(pp),'handoff':pin(out),'training_started':True,'first100_saved_health_pass':True,'goal_pause_due':True,'background_continue_to_final':True,'worker_terminal_at_snapshot':terminal,'accuracy_or_task_benefit_qualified':False,'self15_label_scope':'Native cached15ms vsPOST20, not geometry/HO/human.'}
        write(sp,s,True)
    with (OTHER/'docs/progress-2026-11-05.md').open('a',encoding='utf-8') as f:f.write(f'\n\n{now}: Newpublicnative15matched trainingfirst100 saved, newhealth{h["passed"]}/{h["total"]}+ETA verified; userGoalpause due. Backgroundfixed1000 explicitlyauthorized, terminal_at_snapshot={terminal}. Priornegative results unchanged; no newaccuracy/task/egoqualification. {doc.as_posix()}\n')
    print(json.dumps({'handoff':pin(out),'findings':pin(doc),'plan':pin(plan),'ledger':pin(lp),'health_passed':h['passed'],'health_total':h['total'],'terminal_at_snapshot':terminal,'ETA':eta}))


if __name__=='__main__':main()
