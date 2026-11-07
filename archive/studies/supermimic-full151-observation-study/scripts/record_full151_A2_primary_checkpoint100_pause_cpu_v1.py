"""Save completed research evidence and the authorized conditional pause handoff."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('D:/workspace/supermimic-chord-study')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def healthy(r):assert r['checks'] and all(v is True for v in r['checks'].values()) and r['passed']==r['total']==len(r['checks'])
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def link(label,p):return f'[{label}]({Path(p).as_posix()})'
def main():
    hp=ROOT/'outputs/full151_v3_primary_A2_ckpt100_health_v1/report.json'
    pp=ROOT/'outputs/full151_v3_A2_primary100_live_pause_v1/report.json'
    ep=pp.with_name('ETA.json');sp=ROOT/'outputs/full151_v2_three_arm_smoke_policy_saved_review_v2/report.json'
    h=read(hp);p=read(pp);e=read(ep);s=read(sp)
    for r in (h,p,s):healthy(r)
    assert h['iteration']==100 and h['final_iteration_scope'] is False and h['method']=='A2'
    assert p['status']=='HEALTHY_SCHEMA2_PREFIX100_LIVE_OWNED_TRAINING' and p['prefix100_policy_updates']==0
    assert e['method']=='A2' and e['anchor_checkpoint_iteration']==100
    cfg=read(h['config']['path']);run=Path(cfg['output']);runtime=read(run/'runtime.json')
    assert runtime['status']=='running' and runtime['pid']==p['details']['four_PID_chain'][-1]
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    notes=ROOT/'docs/full151_three_arm_A2_checkpoint100_pause_2026-11-05.md'
    text=f'''# 三方法工程验证与 A2 首检查点交接（2026-11-05）

A1L、P_ONLY、A2 的新 schema2 同源短训练均完成，保存健康检查118/118、136/136、136/136，原始四模块初始化配对75/75。三次固定 final40 全3秒物理回放的保存独审574/574通过；任务门槛分别10/13、9/13、7/13，三组任务均失败，A2掉落且出现非手物体接触。目前没有接触先验收益证据。

实际材料的固定302项LP中，H123和H12345均151/151有条件可行；这是理想接触点与框架下的所需扳手分解，未证明实际接触、手部可达性、力幅上限或连续动力学可行。旧mu.1表示锥23/151结果仍保留，不能把其差异归因于增加手指。

Root一次启动 A2 的2000iter从零训练；共同协议也冻结A1L/P_ONLY配置，但未启动或加入后台队列。新v3入口仅优化元数据pin查找，训练仍使用实测三方法smoke中的 learning_v2/prior_hook/controller/runtime。完整预算为648192真实转移、WM4000/failure4000/Actor1800/critic7200。

首个 `model_100.pt` 健康审计{h['passed']}/{h['total']}通过：40192真实转移、WM200/failure200，Actor/critic/prior政策更新仍为零；这是WM-only前缀，不能据此声称完整A2政策训练或任务已合格。最终容量与最终checkpoint尚待完成后检查。

四PID链 `{p['details']['four_PID_chain']}` 通过创建时间、父子关系、命令、解释器和冻结source身份复核，后台worker为 `{runtime['pid']}`。首次记录与当前CIM身份一致；运行仍由原有soft7100s/hard7200s watchdog管理，无自动重试。

以实际checkpoint100文件mtime `{e['anchor_UTC']}` 为锚，预计北京时间 **{e['estimated_finish_Asia_Shanghai']}** 完成；启发式范围 **{e['finish_range_Asia_Shanghai'][0]} 至 {e['finish_range_Asia_Shanghai'][1]}**。估时使用本次前100步末50差分与同方法A2 smoke去前3次后的policy开销，不把native局部计时加到worker启动时间；范围不是置信区间或保证，后续策略、reset和机器负载可能改变速度。

用户已授权首健康checkpoint及ETA之后暂停Goal、保留后台job。此文与状态文件在调用暂停工具之前写入，实际Goal状态以随后工具返回为准；暂停不等于研究完成或停止此A2进程。

{link('首检查点健康',hp)} · {link('实时身份与ETA交接',pp)} · {link('完整ETA',ep)} · {link('三方法回放保存独审',sp)} · {link('冻结A2配置',h['config']['path'])}

恢复Goal时先读取原worker/watchdog终态与已有最终产物，禁止因观察超时重启。对固定final2000做独立健康/身份/完整3秒回放；随后执行已冻结的A1L/P_ONLY同预算实验并核原始初始化，再比较完整2000结果。仍需接触/物理梯度校验、多轨迹共享和独立ego验证。无指定ego数据，独立ego N=0；真机子版本及观测/控制契约在硬件部署前核对。
'''
    if notes.exists():raise FileExistsError(notes)
    notes.write_text(text,encoding='utf-8')
    artifact_paths={'three_arm_smoke_enrollment':ROOT/'outputs/full151_v2_three_arm_smoke_enrollment_v1.json','three_arm_smoke_pairing75':ROOT/'outputs/full151_v2_three_arm_smoke_pairing_v2/report.json','three_arm_smoke_policy_saved574':sp,'A2_primary100_health':hp,'A2_primary100_live_pause':pp,'A2_primary100_ETA':ep,'A2_primary_hidden_launch':ROOT/'outputs/full151_v3_primary2000_A2_launch_v1/launch.json','three_arm_primary_protocol':ROOT/'protocols/full151-v3-three-arm-primary2000-v1.json','three_arm_checkpoint100_handoff':notes,'material_prior_integration_findings':ROOT/'docs/full151_matched_material_prior_integration_2026-11-05.md'}
    artifact_paths.update(A2_primary_smoke_rawinitial_pair25=ROOT/'outputs/full151_v3_A2_primary_smoke_initial_pairing_v1/report.json',metadata_front_index_static_review=ROOT/'outputs/full151_v3_preflight_index_static_independent_v1/report.json')
    figure=ROOT/'outputs/full151_matched_material_cone_figure_v2/summary.json'
    if figure.exists():artifact_paths['matched_material_scientific_figure']=figure
    artifacts={k:pin(v) for k,v in artifact_paths.items()}
    pause={'satisfied':True,'user_requested':True,'goal_pause_tool_pending':True,'background_job_stop_requested':False,'intended_status':'paused','action':'Call update_goal paused after these saved local writes; tool return is authoritative. Existing owned A2 training continues; no autoqueue/retry.'}
    job={'method':'A2','status_at_observation':'running','worker_pid':runtime['pid'],'parent_pid':runtime['parent_pid'],'four_PID_chain':p['details']['four_PID_chain'],'config':h['config'],'output':str(run),'launch_record':artifacts['A2_primary_hidden_launch'],'checkpoint100':h['checkpoint'],'health':artifacts['A2_primary100_health'],'ETA':artifacts['A2_primary100_ETA'],'continues_after_Goal_pause':True,'automatic_retry':False,'hard_seconds':7200,'other_primary_methods_queued':False,'observed_UTC':now,'observed_iteration':runtime.get('iteration')}
    for path in (ROOT/'STUDY_STATE.json',MAIN/'STUDY_STATE.json'):
        state=read(path);state.update(updated_UTC=now,goal_status='pause_requested_pending_tool',phase='PROGRESS: material302 H3/H5 both151conditional; schema2three GPU smoke healthy/rawinit75/full3ssaved574; all3taskFAIL. NewA2primary running, checkpoint100healthy+ETA. User-authorized Goalpause pending tool; owned background job continues.',last_goal_turn_classification='progress',running_jobs=[job],conditional_goal_pause=pause,independent_ego_N=0,full151_contact_prior_training_ready=True,full151_contact_prior_training_scope='Exploratory CUDA eager engineering only; physical/reference/contact-gradient qualification remains false.',full151_three_arm_GPU_smoke_status='COMPLETED_ENGINEERING_NOT_TASK_QUALITY',full151_three_arm_smoke={'health_checks':{'A1L':[118,118],'P_ONLY':[136,136],'A2':[136,136]},'same_initial_four_parameter_fingerprints':True,'pair_checks':[75,75],'saved_full3s_checks':[574,574],'task_gates_passed':{'A1L':10,'P_ONLY':9,'A2':7},'task_gates_total':13,'all_task_pass':False,'prior_benefit_qualified':False},next_safe_action={'action':'After user resumes: inspect same owned A2 watchdog/runtime/terminal artifacts, no relaunch; fixedfinal2000 health+3seval, then preregistered A1L/P_ONLY primary budgets and pairing before method comparison.','requires_user_permission_now':False},remaining_work='Finish and evaluate A2 fixed2000, run preregistered matching A1L/P_ONLY primary budgets; physical/contact gradient qualification; independent intent geometry/force/actuator/continuous feasibility; shared multi-trajectory transfer; independent ego when supplied; hardware observation/control contract before deployment.')
        state.setdefault('artifacts',{}).update(artifacts);write(path,state)
    progress=MAIN/'docs/progress-2026-11-05.md'
    with progress.open('a',encoding='utf-8') as f:f.write('\n\n## 本轮更新：同材料302与三方法实测\n\nH123/H12345同实际材料均151/151条件可行，独立数值38/来源生命周期40；不代表真实手可行。新schema2三臂GPU smoke及固定final40完整3秒回放均完成，健康118/136/136、初始参数配对75、保存物理独审574全部通过，任务10/9/7项共13均失败。新A2 primary已一次后台启动，首100健康与ETA已完成；按用户条件随后暂停Goal并保留后台job，其余两臂未排队。\n\n'+link('首检查点与恢复交接',notes)+' · '+link('完整实测与限制',ROOT/'docs/full151_matched_material_prior_integration_2026-11-05.md')+'\n')
    stage={'schema':'full151_A2_primary100_pause_local_preparation:1','status':'USER_AUTHORIZED_PAUSE_CONDITION_SATISFIED_TOOL_PENDING','created_UTC':now,'artifacts':artifacts,'job':job,'pause':pause,'writer':pin(__file__),'no_process_modified':True,'Goal_status_changed_by_this_script':False}
    out=ROOT/'outputs/full151_v3_A2_primary100_pause_preparation_v1.json'
    if out.exists():raise FileExistsError(out)
    write(out,stage);print(json.dumps({'saved':pin(out),'handoff':pin(notes),'ETA_Asia_Shanghai':e['estimated_finish_Asia_Shanghai'],'checks_health':[h['passed'],h['total']],'live_worker_pid':runtime['pid']},ensure_ascii=False))
if __name__=='__main__':main()
