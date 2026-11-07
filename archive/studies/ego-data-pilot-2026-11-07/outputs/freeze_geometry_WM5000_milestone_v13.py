"""Archive this completed resume/evaluation turn without completing the Goal."""
from pathlib import Path
import argparse,datetime,hashlib,json,re
P=Path(r'D:\workspace\ego-data-pilot-2026-11-07');M=Path(r'D:\workspace\supermimic-chord-study')
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
a=argparse.ArgumentParser();a.add_argument('--task-report',type=Path,required=True);a.add_argument('--task-protocol',type=Path,required=True);a.add_argument('--task-terminal',type=Path,required=True);args=a.parse_args()
task=read(args.task_report);assert task['passed']==task['total'] and all(task['checks'].values())
assert read(args.task_terminal)['returncode']==0
draft=M/'docs/research-plan-v0.32.DRAFT.md';plan=M/'docs/research-plan-v0.32.md';text=draft.read_text(encoding='utf8')
assert text.count('{{TASK_PARAMETERFREE_RESULT}}')==1
para=f"参数free CPU合同实际{task['passed']}/{task['total']}通过，验证参考初始对齐、k→k+1奖励、合法EMA控制、记忆/replay已知掩码及工程支持代理的输入微分；没有Actor/WM/Critic构造、NN forward、优化或原生积分。v1源接线中发现的20ms/整数记忆校验、运行时模块SHA绑定、在线单环境Actor批次问题已在保留旧版的v2中修正。GPU工厂入口已形成源码，尚无该入口的实际Actor运行。该合同不授予完整训练器或物理策略资格。[实际合同报告]({str(args.task_report).replace(chr(92),'/')})"
text=text.replace('{{TASK_PARAMETERFREE_RESULT}}',para).replace('PRE contact/POSТ state','PRE contact/POST state')
text+='\n\n完整新训练、评价、数据与接线证据见[里程碑v13](D:/workspace/ego-data-pilot-2026-11-07/outputs/goal_extension_geometry_WM5000_milestone_v13.json)。Goal继续active，下一项是新三指任务的原生采样/replay、WM适配和完整A1/A2 Actor训练。\n'
for link in re.findall(r'\]\(([^)]+)\)',text):
    if link.endswith('goal_extension_geometry_WM5000_milestone_v13.json'):continue
    if link.startswith('D:/'):assert Path(link).exists(),link
    elif not link.startswith(('https://','http://')):assert (plan.parent/link).exists(),link
assert not plan.exists()
with plan.open('x',encoding='utf8',newline='\n') as f:f.write(text)
W=P/'phone_geometry_momentum_wm_v1';E=P/'phone_geometry_momentum_evaluation_v1';A=P/'phone_causal_fresh_acquisition_v1';F=P/'phone_causal_fresh_evaluation_v1';T=P/'phone_geometry_actor_integration_v1'
refs=dict(previous_handoff=P/'outputs/goal_extension_tripod_momentum_milestone_v12.json',plan=plan,
    resume_protocol=W/'protocol_CPUTrue_resume1.json',final_checkpoint=W/'cpu_resume1/checkpoint_5000.pt',final_health=W/'cpu_resume1/health_5000.json',resume_terminal=W/'process_resume1/terminal.json',resume_saved_review=W/'resume_saved_review_v1/report.json',resume_source_review=W/'independent_resume_source_review_v1/report.json',
    CHECK_result=E/'completed_v1/report.json',CHECK_protocol=E/'protocol_CPUTrue_run1.json',CHECK_terminal=E/'process_run1/terminal.json',CHECK_saved_summary=E/'saved_summary_v1/report.json',CHECK_SOURCE_review=E/'independent_source_review_v1/report.json',
    fresh_acquisition=A/'acquired_v1/report.json',fresh_acquisition_protocol=A/'protocol_CPUTrue_run1.json',fresh_acquisition_terminal=A/'process_run1/terminal.json',fresh_acquisition_saved_review=A/'saved_review_v1/report.json',
    fresh_evaluation=F/'completed_v1/report.json',fresh_evaluation_protocol=F/'protocol_CPUTrue_run1.json',fresh_evaluation_terminal=F/'process_run1/terminal.json',fresh_evaluation_saved_review=F/'saved_review_v1/report.json',
    CUDA_parity=P/'phone_geometry_cuda_parity_v1/completed_run1/report.json',CUDA_terminal=P/'phone_geometry_cuda_parity_v1/process_run1/terminal.json',
    own_action_AD_FD=P/'phone_geometry_action_AD_v1/completed_run1/report.json',own_action_AD_FD_terminal=P/'phone_geometry_action_AD_v1/process_run1/terminal.json',
    scientific_description=P/'outputs/geometry_WM5000_description_v1/report.json',figure_PNG=P/'outputs/geometry_WM5000_description_v1/WM5000_predictions_and_response_limits.png',
    task_bundle=T/'task_bundle_v2.json',task_SOURCE_revision=T/'source_revision_v2.json',task_SOURCE_peer=P/'phone_geometry_actor_integration_source_review_v1/addendum_v2.json',
    task_parameterfree_report=args.task_report,task_parameterfree_protocol=args.task_protocol,task_parameterfree_terminal=args.task_terminal,
    freezer_source=Path(__file__))
pins={k:pin(v) for k,v in refs.items()};previous=read(refs['previous_handoff'])
assert all(pin(v['path'])==v for v in previous['evidence_pins'].values()),'Prior v12 artifacts remain unchanged'
for k in ['resume_saved_review','CHECK_saved_summary','fresh_acquisition_saved_review','fresh_evaluation_saved_review']:
    r=read(refs[k]);assert r['passed']==r['total'] and all(r['checks'].values()),k
for k in ['resume_terminal','CHECK_terminal','fresh_acquisition_terminal','fresh_evaluation_terminal','CUDA_terminal','own_action_AD_FD_terminal']:assert read(refs[k])['returncode']==0,k
fresh=read(refs['fresh_acquisition']);description=read(refs['scientific_description'])
v=dict(schema='geometry_WM5000_fresh_controlled_response_and_Actor_bridge_goal_milestone:13',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    previous_goal_turn_classification='progress',this_goal_turn_classification='progress',native_goal_status_last_verified='active',original_full_goal_complete=False,
    same_learning_lineage_resumed=True,new_learning_experiment_started=False,actual_added_optimizer_updates_this_turn=4750,resumed_lineage_total_updates=5000,
    first250_pause_previously_fulfilled=True,goal_pause_due=False,Actor_optimizer_updates=0,GPU_training_updates=0,
    actual_resume_worker_counts=read(refs['resume_saved_review'])['actual_this_resume_worker'],
    CHECK_inference_counts=read(refs['CHECK_result'])['counts'],fresh_inference_counts=read(refs['fresh_evaluation'])['counts'],
    same_weight_CUDA_parity_counts=read(refs['CUDA_parity'])['operation_counts'],own_model_AD_FD_counts=read(refs['own_action_AD_FD'])['operation_counts'],
    new_native_acquisition_counts=fresh['counts'],fresh_planned_rows=240,fresh_complete_rows=228,fresh_partial_rows=12,fresh_independent_ego=False,
    remaining_model_phantom_response_proven_observationally=True,strict_allH_structural_zero_certificates=0,
    CHECK_response_error_over_zero_prior=[.6928529150,.8063340367,.9156939892,1.2164621247,.7927748050,.8041954841],
    response_blocks=['q','qd','COM','worlddR','pair_force_impulse','pair_moment_impulse'],
    exactzero_native_COM_side_records=836,model_COM_secant_on_native_zero_mm_per_rad=description['model_COM_secant_on_native_zero_mm_per_rad'],
    fresh_state_tradeoff='COM0.69975mm vsballistic0.76610mm; orientation1.53408deg vsprior0.838624deg and qRMS0.0097413rad vsprior0.0055309rad. No overall generalization gain claim.',
    scientific_figure_PNG_visually_inspected=True,scientific_figure_SVG_exported_not_separately_rendered=True,
    actual_new_Actor_training=False,complete_Actor_trainer_ready=False,engineered_support_proxy_is_real_contact_or_force=False,
    original_two_second_holding_success_newly_proven=False,real_ego_robot_execution_or_CHORD_benefit_proven=False,
    no_background_training_at_handoff=True,
    next_action='Execute named native collection for new loaded tripod task, persist raw PREcontacts/POSTstate/receipts and complete/partial masks to replay; adapt WM with explicit new task data rather than fitting fresh final validation. Implement full A1/A2 same-budget Actor/critic/replay runner, GPU execution and native double-dt noaid evaluation. Do not require manual2s success before learning or let proxy/endpoint accuracy stand in for physical gradient quality.',
    full_goal_remaining='Object asset/dynamics uncertainty and calibration, source gravity, multi-trajectory shared amortization, independent ego, full61-frame robot execution and hardware observability/control contract remain; original full objective is not narrowed.',evidence_pins=pins)
out=P/'outputs/goal_extension_geometry_WM5000_milestone_v13.json'
with out.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
assert all(pin(v['path'])==v for v in pins.values())
print(json.dumps(dict(plan=pin(plan),milestone=pin(out),evidence_pins=len(pins),goal_remains='active',original_full_goal_complete=False)))
