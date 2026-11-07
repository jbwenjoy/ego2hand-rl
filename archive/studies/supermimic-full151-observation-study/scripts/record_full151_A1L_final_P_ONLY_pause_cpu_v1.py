"""Record fixed A1L/A2 failures, P_ONLY first100 and bounded CPU preparation."""
from pathlib import Path
import csv,datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1];MAIN=Path('D:/workspace/supermimic-chord-study')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def positive(r):assert r['checks'] and all(v is True for v in r['checks'].values()) and r['passed']==r['total']==len(r['checks'])
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def link(label,p):return f'[{label}]({Path(p).as_posix()})'
def main():
    paths={'A1L_final2000_health178':ROOT/'outputs/full151_v3_primary_A1L_final2000_health_v1/report.json',
      'A1L_final2000_lifecycle32':ROOT/'outputs/full151_v3_A1L_primary2000_saved_lifecycle_v1/report.json',
      'A1L_final2000_policy_saved188':ROOT/'outputs/full151_v3_A1L_primary2000_policy_saved_review_v1/report.json',
      'A1L_terminal_fourPID_CIM':ROOT/'outputs/full151_v3_A1L_terminal_fourPID_CIM_v1.json',
      'A2_final2000_health196':ROOT/'outputs/full151_v3_primary_A2_final2000_health_v1/report.json',
      'A2_final2000_policy_saved188':ROOT/'outputs/full151_v3_A2_primary2000_policy_saved_review_v1/report.json',
      'P_ONLY_primary100_health116':ROOT/'outputs/full151_v3_primary_P_ONLY_ckpt100_health_v1/report.json',
      'P_ONLY_primary100_live_pause37':ROOT/'outputs/full151_v3_P_ONLY_primary100_live_pause_v1/report.json',
      'P_ONLY_primary100_ETA':ROOT/'outputs/full151_v3_P_ONLY_primary100_live_pause_v1/ETA.json',
      'P_ONLY_owned_launch':ROOT/'outputs/full151_v3_primary2000_P_ONLY_launch_v1/launch.json',
      'P_ONLY_primary_smoke_initial_pairing':ROOT/'outputs/full151_v3_P_ONLY_primary_smoke_initial_pairing_v1/report.json'}
    for name in ('A1L_final2000_health178','A1L_final2000_lifecycle32','A1L_final2000_policy_saved188','A2_final2000_health196','A2_final2000_policy_saved188','P_ONLY_primary100_health116','P_ONLY_primary100_live_pause37','P_ONLY_primary_smoke_initial_pairing'):positive(read(paths[name]))
    pair=read(paths['P_ONLY_primary_smoke_initial_pairing']);assert pair['same_initial_four_parameter_fingerprints'] is True
    h=read(paths['P_ONLY_primary100_health116']);p=read(paths['P_ONLY_primary100_live_pause37']);e=read(paths['P_ONLY_primary100_ETA'])
    assert h['method']=='P_ONLY' and h['iteration']==100 and h['final_iteration_scope'] is False and p['prefix100_policy_updates']==0
    cfg=read(h['config']['path']);run=Path(cfg['output']);live=read(run/'runtime.json');assert live['status']=='running' and live['pid']==37396
    assert p['details']['four_PID_chain']==[30280,33496,22700,37396]
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();data={};rows=[]
    for method in ('A1L','A2'):
        ep=ROOT/f'outputs/full151_v2_policy_eval_primary2000_{method}_seed0_v1/evaluation.json';ev=read(ep)
        assert ev['complete'] is True and ev['physical_steps']==600 and ev['sampled_state_rows']==601 and ev['sampled_solver_rows']==600
        assert sum(ev['gates'].values())==7 and ev['native_policy_pass'] is False
        paths[method+'_fixed2000_evaluation']=ep;data[method]=ev
        rows.append({'method':method,'fixed_checkpoint_iteration':2000,'quality_gates_passed':7,'quality_gates_total':13,'task_pass':False,'position151_max_mm':1000*ev['max_object_position_error_151_m'],'directed_shaft151_max_deg':ev['max_shaft_angle_151_deg'],'allcategory_penetration_max_mm':1000*ev['max_penetration_m'],'target_rate_max_rad_s':ev['max_actual_target_rate_rad_s'],'drop_gate_pass':ev['gates']['drop'],'nonhand_gate_pass':ev['gates']['no_nonhand_object_contacts']})
    optional={'schema2_pair_WM_actual_CPU_check':ROOT/'outputs/full151_schema2_pair_WM_loader_CPU_v1/report.json','transaction105_CPU_preparation':ROOT/'outputs/full151105_transaction_preparation_CPU_v1/report.json'}
    for name,path in optional.items():
        if path.exists():positive(read(path));paths[name]=path
    paths['transaction105_component_source']=ROOT/'src/full151_gradient/transaction105.py'
    paths['transaction105_READY_API']=ROOT/'outputs/full151105_transaction_preparation_CPU_v1/READY.json'
    previous_eta=read(ROOT/'outputs/full151_v3_A1L_primary100_live_pause_v1/ETA.json');end=read(ROOT/'outputs/full151_v3_primary2000_A1L_seed0_v1/runtime.json')
    eta_error=(datetime.datetime.fromisoformat(previous_eta['estimated_finish_Asia_Shanghai'])-datetime.datetime.fromisoformat(end['finished_utc'])).total_seconds()
    notes=ROOT/'docs/full151_A1L_final2000_P_ONLY_checkpoint100_2026-11-05.md'
    if notes.exists():raise FileExistsError(notes)
    table='\n'.join(f"| {r['method']} | 7/13 | {r['position151_max_mm']:.4f} | {r['directed_shaft151_max_deg']:.4f} | {r['allcategory_penetration_max_mm']:.4f} | {r['target_rate_max_rad_s']:.4f} |" for r in rows)
    text=f'''# A1L 完整2000、P_ONLY 首检查点与梯度诊断准备（2026-11-05）

上一Goal turn是实际progress：A2完整健康与物理评价、新A1L首100、同Warp草案与只读ABI25。本轮先核原A1L终态与fresh四PID CIM，没有重启A1L。A1L于北京时间14:11:54正常完成2000iter，worker wall {end['wall_seconds']:.3f}s，原ETA点预测比实际晚{eta_error:.6f}s；只描述本次估时。

## 新schema2固定final2000结果

A1L健康178/178、生命周期32/32、完整3秒保存独审188/188通过；498源pin与ZIP、actual8/controller/固定21个检查点、完整648192真实转移及WM4000/failure4000/Actor1800/critic7200均核对。A1L始终prior=None、无objective或prior JSONL、计数0。最终容量3222次观测/3199pre-reset，overflow/unknown均0；原启动记录错误与恢复来源仍保留，四PID终态独立核不在。

Root对预声明fixedfinal2000仅做一次150controls/600solver/601states回放，无Actor/norm更新、无物体辅助、无质量提前结束。A1L与A2均任务失败，失败项一致：位置、方向、穿透、非手接触、目标速率和掉落。

| 方法 / fixedfinal2000 | 通过门槛 | 位置151最大误差 mm | 方向151最大误差 ° | 所有类别最大穿透 mm | 目标最大速率 rad/s |
|---|---:|---:|---:|---:|---:|
{table}

这两组为同源、同seed0、同原始四模块参数、同预算的完整训练；P_ONLY final2000仍未完成，故三方法完整对照尚未闭合。二者都未达标，不能因某项误差略低就称先验有效，也不能从单条训练参考、单seed外推泛化。旧schema1实验没有进入新三方法配对，不替代本轮基线。

## P_ONLY：第三个冻结预算已启动

Root通过参数化launcher一次启动P_ONLY，UTC09:44:55.472（北京时间17:44:55）。本次预先计算固定SHA256与source pins，再Start-Process；launch.json与initial_CIM原生记录完成，没有记录恢复或重复启动。四PID为30280→33496→22700→37396。三组primary和对应smoke原始四模块逐name/order/shape/dtype/bytes配对一致。

固定model100健康116/116，40192真实转移、101collect/100metrics、WM200/failure200，Actor/Critic/prior调用均0；P_ONLY objective已构造且metadata/spec/RNG一致，无policy JSONL，final_scope=False。这只证明WM-only工程前缀；最终政策更新、容量和任务仍需终态审计。

实时身份/估时37/37通过，预计北京时间 **{e['estimated_finish_Asia_Shanghai']}** 完成，启发式范围 **{e['finish_range_Asia_Shanghai'][0]}–{e['finish_range_Asia_Shanghai'][1]}**。以model100实际mtime {e['anchor_UTC']} 为锚，使用本次末50步平均{e['primary_baseline_last50_differences_s_per_iteration']:.6f}s与同P_ONLY smoke去前三次policy均值{e['smoke_policy_trimfirst3_17calls_mean_s']:.6f}s。范围不是置信区间或保证。

按用户已授权条件，保存本地交接后调用暂停工具，Goal状态以工具返回为准。P_ONLY原后台训练继续由7100/7200s watchdog管理，无后续队列或自动重试。恢复时先核37396原worker/watchdog/已有终态，禁止因观察超时重复启动；固定final2000健康、完整3秒评价与独立保存核验后，才汇总三方法最终表。

{link('A1L final健康178',paths['A1L_final2000_health178'])} · {link('A1L物理保存审188',paths['A1L_final2000_policy_saved188'])} · {link('A1L生命周期32',paths['A1L_final2000_lifecycle32'])} · {link('P_ONLY首100健康116',paths['P_ONLY_primary100_health116'])} · {link('P_ONLY实时身份与ETA',paths['P_ONLY_primary100_live_pause37'])}

## 梯度验证仍按物理范围分层

本轮已完成两项边界明确的工程工作。新schema2 A1L/A2 fixedfinal各用冻结119f loader实际严格加载一次，每个只在同一保存A0 phase0 state58/actualu20上做一次forward和一次u AD，15/15工程检查通过。内部共4次torch.load、2次CPU Actor构造但无Actorforward、2次被动MjModel读；无Env/Warp/CUDA初始化/物理API。权重、normalizer、输入和RNG不变，state58/self15/u梯度均有限。AD标量是混单位输出sum，仅用于连通性检查，不能充当native任务标量或精度证据。

新增隔离transaction105组件与CPU surrogate契约42/42通过，覆盖缺字段、旧104、phase149/terminal150、恢复不完整、外部容量账本及别名、closure、RNG、失败分支不重试等拒绝场景。Root已阅读源码：真实PenSimReader.memory将c1/c2都复制当前processed_action，故相等guard只用于真实捕获边界；想象advance_memory之后两者可不同，禁止套用该profile。opaque foreign仅记录身份，内部状态资格仍PENDING。上述42项不是实际Native恢复试验。

冻结old observed_capacity将observer挂在env上，不适合可恢复donor graph；future worker必须把容量host OR/UNKNOWN账本放在事务外，任何restore都不能清除此账本，旧source不改。真实105完整inventory、warmstart、三baseline、长时钟、sameWarp控制差分及HO测量均需实际资格。

WM的self15仅是手自碰撞，不是HO接触/力head。CPU WM内部u AD有限只能证明模型自身可微；不是物理Jacobian、HO力梯度、GPU AMP或任务收益证据。152分支/1808步的sameWarp方案仍为proposal，未执行。几何候选SCP、真实接触/力幅/执行器/连续动力学资格、多轨迹共享和独立ego数据仍未解决，独立ego N=0；真机子版本与观测/控制契约在部署前核对。
'''
    for name in optional:
        if name in paths:text+='\n'+link(name,paths[name])+'\n'
    notes.write_text(text,encoding='utf-8');paths['A1L_results_P_ONLY_handoff']=notes
    summary_dir=ROOT/'outputs/full151_schema2_primary_interim_pair_v1';summary_dir.mkdir(exist_ok=False)
    with (summary_dir/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(summary_dir/'summary.json',{'schema':'full151_schema2_fixed2000_interim_pair:1','status':'TWO_METHODS_COMPLETE_BOTH_TASK_FAIL_THIRD_PENDING','methods_complete':['A1L','A2'],'third_method':'P_ONLY','third_status':'RUNNING_NOT_EVALUATED','rows':rows,'no_method_benefit_claim':True,'independent_ego_N':0,'writer':pin(__file__)})
    paths['interim_two_method_metrics']=summary_dir/'metrics.csv';paths['interim_two_method_summary']=summary_dir/'summary.json'
    artifacts={k:pin(v) for k,v in paths.items()}
    job={'method':'P_ONLY','status_at_observation':'running','worker_pid':37396,'parent_pid':22700,'four_PID_chain':[30280,33496,22700,37396],'config':h['config'],'output':str(run),'launch_record':artifacts['P_ONLY_owned_launch'],'checkpoint100':h['checkpoint'],'health':artifacts['P_ONLY_primary100_health116'],'ETA':artifacts['P_ONLY_primary100_ETA'],'continues_after_Goal_pause':True,'automatic_retry':False,'hard_seconds':7200,'other_primary_methods_queued':False,'observed_UTC':now,'observed_iteration':live.get('iteration')}
    pause={'satisfied':True,'user_requested':True,'goal_pause_tool_pending':True,'background_job_stop_requested':False,'intended_status':'paused','action':'Submit user-authorized update_goal paused after saving evidence. Original P_ONLY job continues; no automatic queue/retry.'}
    for path in (ROOT/'STUDY_STATE.json',MAIN/'STUDY_STATE.json'):
        state=read(path);state.update(updated_UTC=now,goal_status='pause_requested_pending_tool',phase='PROGRESS: A1L fixed2000 healthy178/lifecycle32/full3ssaved188 taskFAIL7/13; A2 alsoTASKFAIL7/13. P_ONLY samebudget live checkpoint100healthy116+ETA. CPUgradient/interface preparation evidence separately scoped. Authorized Goalpause pending; original P_ONLY continues.',last_goal_turn_classification='progress',previous_goal_turn_classification='progress',running_jobs=[job],conditional_goal_pause=pause,independent_ego_N=0,full151_schema2_primary_status={'A2':'COMPLETED_HEALTHY196_TASKFAIL7OF13_SAVED188_LIFECYCLE30','A1L':'COMPLETED_HEALTHY178_TASKFAIL7OF13_SAVED188_LIFECYCLE32','P_ONLY':'RUNNING_CHECKPOINT100_HEALTHY116_WM_ONLY'},full151_schema2_A1L_final2000_task_success=False,full151_schema2_A1L_final2000_evaluation_summary={k:data['A1L'][k] for k in ('gates','max_object_position_error_151_m','max_shaft_angle_151_deg','max_penetration_m','max_actual_target_rate_rad_s')},next_safe_action={'action':'After user resumes inspect original P_ONLY worker/watchdog or terminal artifacts, no relaunch; fixedfinal2000 health+3seval then fullthree matched saved summary. Implement/qualify105transaction externalcapacity and actual sameWarp control/contact diagnostics; CPU WM interface/self15 not physicalHO gradient proof.', 'requires_user_permission_now':False},remaining_work='P_ONLY fixed2000 completion/evaluation+raw final three-arm comparison; actual105 snapshot/baseline/longclock/capacity qualification then sameWarp physical control/contact diagnostic; feasible observation intent/reference geometry/force/actuator/continuous dynamics; shared multitrajectory transfer; independent ego when supplied; hardware observer/control contract before deployment.')
        state.setdefault('artifacts',{}).update(artifacts);write(path,state)
    with (MAIN/'docs/progress-2026-11-05.md').open('a',encoding='utf-8') as f:f.write('\n\n## A1L完整2000与P_ONLY第三预算\n\nA1L原训练终态已核，健康178/生命周期32/完整3秒保存审188通过，任务7/13失败且掉落；与A2完整结果均未达标。一次启动冻结P_ONLY，首100健康116与实时身份/ETA37通过；预计北京18:20(约18:13–18:31)，保存后按用户条件暂停Goal保留原后台job。三方法final表尚待P_ONLY，不先作收益结论。\n\n'+link('完整结果与恢复交接',notes)+'\n')
    out=ROOT/'outputs/full151_v3_A1L_final_P_ONLY_primary100_pause_preparation_v1.json'
    if out.exists():raise FileExistsError(out)
    write(out,{'schema':'full151_A1L_final_P_ONLY_primary100_pause_preparation:1','status':'PROGRESS_USER_AUTHORIZED_PAUSE_PENDING_TOOL','created_UTC':now,'artifacts':artifacts,'job':job,'pause':pause,'source':pin(__file__),'no_process_modified':True,'Goal_status_changed_by_this_script':False})
    print(json.dumps({'saved':pin(out),'handoff':pin(notes),'ETA_Asia_Shanghai':e['estimated_finish_Asia_Shanghai'],'range':e['finish_range_Asia_Shanghai'],'P_ONLY_live_pid':37396,'two_finished_tasks_pass':False},ensure_ascii=False))
if __name__=='__main__':main()
