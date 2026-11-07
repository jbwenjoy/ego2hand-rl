"""Freeze the first paired Actor checkpoint handoff before the requested pause."""
from pathlib import Path
import datetime,hashlib,json,re
P=Path(r'D:\workspace\ego-data-pilot-2026-11-07');M=Path(r'D:\workspace\supermimic-chord-study');W=P/'phone_tripod_supermimic_v1';R=W/'gpu_run1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
h=read(R/'health_0100.json');result=read(R/'result.json');review=read(W/'root_saved_first100_review_v2/report.json');term=read(W/'process_run1/terminal.json');gpu=read(W/'process_run1/GPU_summary.json')
assert h['passed']==h['total'] and all(h['checks'].values()) and review['passed']==review['total'] and all(review['checks'].values()) and term['returncode']==0
assert result['paired_round']==100 and result['shared_WM_updates']==200 and h['process_stops_here'] is True
eta=h['estimated_remaining_training_s'];peak=h['GPU_peak_allocated_bytes']/1024**3;util=gpu['GPU_util_pct'];mem=gpu['memory_used_MiB']
para=(f"本次实际在首paired100健康点退出：shared WM新更新200次，两臂各Actor100/Critic100，固定目标1000 paired rounds尚余900。"
      f"producer保存读回{h['passed']}/{h['total']}、Root保存payload/log核验{review['passed']}/{review['total']}通过。"
      f"检查点SHA `{h['checkpoint']['sha256']}`。按最近25轮实测速率，剩余更新约{eta/60:.1f}分钟，恢复setup及未来health/IO另计；这是速率外推，不是完工时刻保证。"
      f"GPU worker峰值allocated约{peak:.2f}GiB；含setup/health的整机GPU采样中位数{util['median']:.0f}%、范围{util['min']:.0f}%–{util['max']:.0f}%，"
      f"整机显存采样峰值{mem['max']/1024:.2f}GiB（含桌面及CUDA上下文，不等于训练Tensor allocated）。"
      "A2有额外proxy计算，同更新/采样预算不表示同wall compute。没有在此执行学习策略原生评测，想象loss/TD健康不提升为持握或CHORD收益。"
      "[检查点健康与ETA](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/gpu_run1/health_0100.json)、"
      "[保存核验](D:/workspace/ego-data-pilot-2026-11-07/phone_tripod_supermimic_v1/root_saved_first100_review_v2/report.json)")
draft=M/'docs/research-plan-v0.33.DRAFT.md';plan=M/'docs/research-plan-v0.33.md';s=draft.read_text(encoding='utf8');assert s.count('{{FIRST100_RESULT}}')==1
s=s.replace('{{FIRST100_RESULT}}',para)+'\n\n本回合完整证据与恢复入口见[里程碑v14](D:/workspace/ego-data-pilot-2026-11-07/outputs/goal_extension_tripod_matched_learning_milestone_v14.json)。首健康点和ETA已具备，归档后请求原生Goal暂停；完整目标继续未完成。\n'
for link in re.findall(r'\]\(([^)]+)\)',s):
    if link.endswith('goal_extension_tripod_matched_learning_milestone_v14.json'):continue
    if link.startswith('D:/'):assert Path(link).exists(),link
    elif not link.startswith(('http://','https://')):assert (plan.parent/link).exists(),link
with plan.open('x',encoding='utf8',newline='\n') as f:f.write(s)
A=P/'phone_tripod_task_acquisition_v1';D=P/'phone_tripod_replay_dataset_v1';E=P/'phone_tripod_policy_evaluation_v1'
refs=dict(previous_handoff=P/'outputs/goal_extension_geometry_WM5000_milestone_v13.json',plan=plan,
    actual_native_bridge=P/'phone_tripod_bridge_native_v1/completed_run1/report.json',native_bridge_terminal=P/'phone_tripod_bridge_native_v1/process_run1/terminal.json',
    native_task_data=A/'acquired_v1/report.json',native_data_terminal=A/'process_run1/terminal.json',native_data_protocol=A/'protocol_CPUTrue_run1.json',native_data_saved_review=A/'saved_review_v1/report.json',
    dataset=D/'completed_v1/report.json',dataset_saved_review=D/'saved_review_v1/report.json',dataset_export_terminal=D/'process_run1/terminal.json',
    policy_callback_kernel=E/'controlled_native_kernel_v1.py',policy_kernel_derivation=E/'kernel_derivation_v1.json',policy_zero_qualification=E/'zero_qualified_run2/report.json',policy_zero_terminal=E/'zero_process_v2_run1/terminal.json',policy_failed_binding_terminal=E/'zero_process_run1/terminal.json',policy_binding_correction=E/'codec_binding_revision_v2.json',
    trainer=W/'train_shared_WM_A1_A2.py',recipe=W/'recipe_v1.json',trainer_SOURCE_review=W/'source_review_windows_v1/report.json',training_protocol=W/'protocol_CPU_GPU_True_run1.json',training_terminal=W/'process_run1/terminal.json',
    initial_modules=R/'initial_modules_not_learning_checkpoint.pt',learning_checkpoint=R/'paired_checkpoint_0100.pt',learning_health=R/'health_0100.json',learning_readback=R/'health_readback_0100.npz',learning_result=R/'result.json',learning_log=R/'paired_updates.jsonl',root_payload_review=W/'root_saved_first100_review_v2/report.json',saved_review_ABI_correction=W/'saved_review_ABI_revision_v2.json',hardware_samples=W/'process_run1/GPU_samples.jsonl',hardware_summary=W/'process_run1/GPU_summary.json',freezer_source=Path(__file__))
pins={k:pin(v) for k,v in refs.items()};previous=read(refs['previous_handoff']);assert all(pin(v['path'])==v for v in previous['evidence_pins'].values())
assert pins['learning_checkpoint']==h['checkpoint']==result['checkpoint']
for key in ['actual_native_bridge','native_data_saved_review','dataset_saved_review','policy_zero_qualification','trainer_SOURCE_review','root_payload_review']:
    q=read(refs[key]);assert q['passed']==q['total'] and all(q['checks'].values()),key
record=dict(schema='tripod_native_replay_first_GPU_paired_Actor_learning_goal_milestone:14',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn_classification='progress',this_goal_turn_classification='progress',native_goal_status_last_verified='active',original_full_goal_complete=False,
    native_tripod_data_rows=205,TRIPOD_TRAIN_rows=153,TRIPOD_MONITOR_rows=52,all_native_episodes_POST_H_loss=32,new_successful_2s_holding_cases=0,
    task_data_native_steps=19712,bridge_qualification_native_steps=192,zero_callback_qualification_native_steps=1171,zero_callback_prior_trace_arrays_all_raw_equal=True,
    first_GPU_Actor_learning_started=True,offline_shared_WM_pilot=True,online_policy_native_collection_started=False,
    paired_rounds_completed=100,paired_rounds_target=1000,shared_WM_updates=200,Actor_updates_each_arm=100,Critic_updates_each_arm=100,
    training_worker_counts=h['counts'],checkpoint_health=dict(passed=h['passed'],total=h['total']),root_saved_review=dict(passed=review['passed'],total=review['total']),extra_Root_CPU_payload_reads=4,extra_Root_NN_or_native_calls=0,
    remaining_rounds_update_only_ETA_s=eta,GPU_peak_Torch_allocated_bytes=h['GPU_peak_allocated_bytes'],machine_GPU_samples_summary=gpu,
    background_learning_continues=False,standing_first_new_training_health_ETA_pause_triggered=True,goal_pause_due=True,
    pause_scope='Archive prepared before native update_goal paused request; this file does not claim the API has already returned paused.',
    pretrained_WM5000_is_distinct_from_new_optimizer_experiment=True,actual_Actor_native_2s_or_ego_CHORD_benefit_qualified=False,engineered_support_proxy_is_real_force_or_contact=False,
    implementation_failure_preserved='Policy callback v1 bound wrong actual_state module; failure after first RESET_FORWARD before first native step, no serialized counter receipt. v2 changes dependency/paths only, then exact old-trace equivalence passed.',
    next_action='Resume identical own paired100 checkpoint/source/recipe/data to fixed1000; evaluate both Actors with original2s/double-dt/noaid native gates and object tracking descriptors. Continue policy-native collection/replay online loop and multi-trajectory sharing; no model selection from monitor.',
    full_goal_remaining='Independent ego/RGB hand reconstruction and input contract, calibrated/robust geometry and dynamics/source gravity, full61frame robot execution, shared multi-trajectory costs/matched methods, hardware observability/control contract and stable task/CHORD benefit remain; no scope narrowing.',evidence_pins=pins)
out=P/'outputs/goal_extension_tripod_matched_learning_milestone_v14.json'
with out.open('x',encoding='utf8') as f:json.dump(record,f,indent=2,allow_nan=False)
assert all(pin(v['path'])==v for v in pins.values())
print(json.dumps(dict(plan=pin(plan),milestone=pin(out),evidence_pins=len(pins),pause_due=True,original_full_goal_complete=False)))
