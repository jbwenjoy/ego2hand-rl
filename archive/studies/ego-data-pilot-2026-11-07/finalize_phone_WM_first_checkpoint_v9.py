"""Verify first real phone checkpoint, document ETA, prepare requested goal pause."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os
import numpy as np

P=Path(__file__).resolve().parent;T=P/'phone_world_model_train_v1';run=T/'cpu_run1'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):assert pin(r['path'])==r,'Changed evidence '+r['path'];return Path(r['path'])
def write(p,j):
    with Path(p).open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False)

ap=argparse.ArgumentParser();ap.add_argument('--exec-session-id',type=int);a=ap.parse_args()
health_path=run/'checkpoint_0250_health.json';h=read(health_path)
assert h['iteration']==250 and h['status']=='HEALTHY_SAVED_READBACK_NOT_DYNAMICS_QUALIFICATION'
assert h['finite_all_parameters_gradients_Adam'] and h['exact_save_load_predictions']
assert h['raw_RNG_unchanged_from_pre_training'] and h['raw_Adam_and_all_state_readback_equal']
assert h['TEST_arrays_read']==0 and h['first_checkpoint_handoff_required'] is True
for k in ('training_FW','backward','optimizer_steps'):assert h['counts_after_readback'][k]==250
assert h['counts_after_readback']['validation_FW']==h['counts_after_readback']['checkpoint_readback_FW']==1
assert h['counts_after_readback']['own_checkpoint_loads']==1
ckpt=checked(h['checkpoint'])
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
payload=torch.load(ckpt,map_location='cpu',weights_only=False)
assert payload['schema']=='phone_physical_world_model_checkpoint:1' and payload['iteration']==250
assert payload['input_schema']=='phone_COM_world58_absolute_u20_h_s_nsubsteps80:1'
assert payload['output_schema']=='phone_normalized_world_delta52_pad_coverage2_logforce5:1'
assert payload['normalizers_fit_partition']=='train_only'
assert payload['gradient_phase']=='PRE_OPTIMIZER_250' and payload['gradient_weights_after_updates']==249
assert payload['weights_phase']=='AFTER_OPTIMIZER_250'
assert all(torch.isfinite(v).all().item() for v in payload['model_state'].values())
assert all(torch.isfinite(v).all().item() for v in payload['gradients_postclip'].values())
for state in payload['optimizer_state']['state'].values():
    assert float(state['step'])==250
    assert torch.isfinite(state['exp_avg']).all().item() and torch.isfinite(state['exp_avg_sq']).all().item()
assert not torch.cuda.is_initialized()
protocol=read(checked(payload['protocol_pin']))
assert protocol['recipe']['updates']==5000 and protocol['recipe']['background_continue_to_final'] is True
assert payload['dataset_input_pins']==protocol['inputs']
for r in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:checked(r)
plan=read(P/'phone_episode_acquisition_v1/family_plan_v1.json')
assert payload['TRAIN_family_ids']==[f['family_id'] for f in plan['families'] if f['split']=='train']
prefix=payload['log_prefix']
with Path(prefix['path']).open('rb') as f:data=f.read(prefix['bytes'])
assert len(data)==prefix['bytes'] and hashlib.sha256(data).hexdigest()==prefix['sha256'] and prefix['records']==250
logs=[json.loads(row) for row in data.splitlines()]
assert len(logs)==250 and [v['iteration'] for v in logs]==list(range(1,251))
assert all(math.isfinite(v['preclip_norm']) and all(math.isfinite(x) for x in v['losses'].values()) for v in logs)
metrics=h['validation'];baseline=read(T/'constant_mean_baseline_v1/report.json')
assert len(metrics['physical_delta_RMSE_per_feature'])==52
assert all(math.isfinite(x) for x in metrics['physical_delta_RMSE_per_feature'])
for key in ['COM_vector_RMS_m','Rbody_raw9_Frobenius_RMS','coverage_Brier_mean']:
    assert math.isfinite(metrics[key]) and math.isfinite(baseline[key])
eta=float(h['ETA_remaining_seconds']);assert math.isfinite(eta) and eta>=0
estimated_end=datetime.datetime.fromtimestamp(health_path.stat().st_mtime,datetime.timezone.utc)+datetime.timedelta(seconds=eta)
end_shanghai=estimated_end.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
term_path=T/'process_training_run1/terminal.json'
terminal=read(term_path) if term_path.exists() else None
terminal_info=dict(observed_terminal=terminal is not None,exec_session_id=a.exec_session_id,
    scope='If no terminal here, last live exec handle must be rechecked on resume. Launch receipt alone is not liveness.')
if terminal:
    terminal_info.update(returncode=terminal['returncode'],terminal=pin(term_path))
    for k in ['source','runner','protocol','stdout','stderr']:checked(terminal[k])

verification=dict(schema='phone_WM_first_real_checkpoint_Root_readback:1',status='PASS_HEALTHY250_PREPARED_USER_REQUESTED_GOAL_PAUSE',
    checkpoint=pin(ckpt),health=pin(health_path),protocol=payload['protocol_pin'],
    checkpoint_iteration=250,first_checkpoint_remaining_ETA_s=eta,estimated_end_at_first_health_AsiaShanghai=end_shanghai,
    validation=metrics,constant_TRAIN_mean_validation={k:baseline[k] for k in ['COM_vector_RMS_m','Rbody_raw9_Frobenius_RMS','coverage_Brier_mean']},
    better_than_constant_mean_validation={k:metrics[k]<baseline[k] for k in ['COM_vector_RMS_m','Rbody_raw9_Frobenius_RMS','coverage_Brier_mean']},
    immutable_log_prefix=prefix,root_extra_checkpoint_loads=1,root_model_constructors_optimizer_FW_AD_GPU=0,
    finite_saved_weights_gradients_Adam=True,metadata_phase_and_train_family_binding=True,
    TEST_model_metrics_computed=False,optimizer_resume_continuation_qualified=False,Markov_or_real_ego_policy_qualified=False,
    process_at_handoff=terminal_info,goal_pause_tool_still_to_call=True)
write(T/'first_checkpoint_Root_readback_v1.json',verification)

doc=Path(r'D:\workspace\supermimic-chord-study\docs\research-plan-v0.28.md')
text=doc.read_text(encoding='utf8')
placeholder='实际训练状态与首健康checkpoint/ETA在执行后补记；当前源码准备与数据采集不冒称已训练或已恢复手机持握。'
assert text.count(placeholder)==1
observed=('交接时训练进程已终态，returncode='+str(terminal['returncode'])+'。') if terminal else '交接时尚未取得进程终态，后台按既定5000步预算继续，恢复后须先核实原进程。'
insert=(f"本轮实际启动了新的phone世界模型训练。第250步checkpoint已原子落盘，有限权重/梯度/Adam、严格逐字节保存读回与验证预测一致性检查通过；Root另作一次自有checkpoint加载，核对源/数据/归一化/梯度相位及前250条固定日志前缀。未构造额外网络或优化器。\n\n"
    f"同114条VALIDATION，COM向量RMS为{metrics['COM_vector_RMS_m']*1000:.4f}mm（常量TRAIN均值基线{baseline['COM_vector_RMS_m']*1000:.4f}mm），R矩阵Frobenius RMS为{metrics['Rbody_raw9_Frobenius_RMS']:.6g}，覆盖率Brier为{metrics['coverage_Brier_mean']:.6g}。这是局部仿真family的单步预测结果，TEST模型指标仍未计算，不是持握、策略、独立ego或物理动作梯度收益。\n\n"
    f"首健康checkpoint时剩余ETA约{eta:.1f}s，估计结束时间{end_shanghai}；该估计不含后续checkpoint I/O，非完成承诺。{observed}本轮交接后按用户既有约定调用goal暂停；完整目标未完成。[首checkpoint及Root读回](D:/workspace/ego-data-pilot-2026-11-07/phone_world_model_train_v1/first_checkpoint_Root_readback_v1.json)。")
doc.write_text(text.replace(placeholder,insert),encoding='utf8')
oldpath=P/'outputs/goal_extension_source_mapping_milestone_v8.json';old=read(oldpath)
for r in old['evidence_pins'].values():checked(r)
paths=dict(plan=doc,previous_handoff=oldpath,reader=P/'phone_episode_acquisition_v1/native_phone_reader.py',
    reader_golden=P/'phone_episode_acquisition_v1/independent_reader_review_v1/actual_golden_v1/report.json',
    acquisition_design=P/'phone_episode_acquisition_v1/design_contract_v1.json',family_plan=P/'phone_episode_acquisition_v1/family_plan_v1.json',
    acquisition=P/'phone_episode_acquisition_v1/acquired_v1/report.json',
    acquisition_terminal=P/'phone_episode_acquisition_v1/process_run1/terminal.json',
    acquisition_saved_review=P/'phone_episode_acquisition_v1/independent_dataset_review_v1/report.json',
    training_source_review=T/'independent_source_review_v1/report.json',training_revision=T/'source_revision_v2.json',
    training_protocol=T/'protocol_CPU_run1.json',training_source=T/'train_phone_wm_v2.py',training_CLI=T/'train_phone_wm_cpu_v2.py',
    constant_mean_baseline=T/'constant_mean_baseline_v1/report.json',first_checkpoint=ckpt,
    first_checkpoint_health=health_path,root_checkpoint_readback=T/'first_checkpoint_Root_readback_v1.json')
if terminal:paths['training_terminal']=term_path
record=dict(schema='phone_H20ms_episode_WM_first_checkpoint_goal_extension:9',
    created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn_classification='progress',
    this_goal_turn_classification='progress',original_full_goal_complete=False,
    status='FIRST_REAL_PHONE_WM_HEALTHY_CHECKPOINT_USER_REQUESTED_GOAL_PAUSE_PENDING',
    native_goal_status_before_pause='active',pause_condition_explicitly_user_requested=True,goal_pause_tool_still_to_call=True,
    new_training_started=True,training_kind='fresh simulated phone dynamics/contact world model; CPU4; no Actor/CHORD training',
    new_physics_steps=89679+192,new_physics_MJB_reads=4,new_initial_forward_calls=98,observer_forward_calls=0,
    acquired_families=48,episodes=96,retained_interval_records=960,complete_eligible_rows=912,partial_rows=48,
    training_rows=684,validation_rows=114,reserved_test_rows=114,cross_split_exact_aliases=0,
    first_healthy_checkpoint_iteration=250,first_checkpoint_eta_remaining_s=eta,
    process_at_handoff=terminal_info,fixed_training_update_budget=5000,
    new_ego_or_policy_or_holding_or_WM_physics_gradient_benefit_qualified=False,
    state58_Markov_and_real_material_calibration_qualified=False,
    next_action='After user resumes, inspect original training handle/terminal before any restart. Evaluate reserved TEST families, multi-step rollout and matched full-state action/contact secants; verify optimizer continuation separately. Then implement task/reward/residual and matched SuperMimic/CHORD budget. Preserve original2s/two-dt/material/61frame/independentego/hardware scope.',
    evidence_pins={k:pin(v) for k,v in paths.items()},finalizer=pin(__file__))
write(P/'outputs/goal_extension_phone_WM_first_checkpoint_v9.json',record)
print(json.dumps(dict(checkpoint_iteration=250,ETA_remaining_s=eta,estimated_end_AsiaShanghai=end_shanghai,
    COM_validation_RMS_mm=metrics['COM_vector_RMS_m']*1000,coverage_Brier=metrics['coverage_Brier_mean'],
    terminal=terminal_info,prepared_goal_pause=True,
    handoff=pin(P/'outputs/goal_extension_phone_WM_first_checkpoint_v9.json'))))
