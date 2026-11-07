"""Independent CPU-only actual-u and wall-cost audit, after terminal27 only."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT=Path('D:/workspace/supermimic-c2-confirmation-eval-study')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def error(a,b):return float(np.max(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda x:x.item())+'\n')
started=time.perf_counter();planpath=ROOT/'configs/c2-confirmation-evaluation-v3.json';p=read(planpath)
statepath=Path(p['output_root'])/'state.json';state=read(statepath)
assert state['status']=='completed_27_jobs' and len(state['jobs'])==27 and state['plan_sha256']==sha(planpath)
rows=[];initial=None
for expected,job in zip(p['jobs'],state['jobs']):
    assert expected['job_id']==job['job_id'] and job['status']=='completed'
    folder=Path(job['output_dir']);a=load(folder/'control_targets_50hz.npz');raw=load(folder/'rollout.npz')
    meta=read(folder/'rollout.json');runtime=read(Path(job['wrapper_metadata_dir'])/'runtime.json');cap=read(folder/'capacity_audit.json');condition=read(folder/'condition.json')
    proposed=a['u_proposed_original_bounded_rad'];executed=a['u_executed_filtered_rad']
    native=np.clip(proposed,a['lower_bound_rad']+np.float32(.020),a['upper_bound_rad']-np.float32(.020))
    errors={'projection':error(executed,native),'cache_vs_executed':error(a['u_cached_after_step_rad'],executed),
      'actual_vs_executed':error(a['u_actual_ctrl_readback_rad'],executed),'raw_target_vs_actual':error(raw['joint_target_rad'][1:],executed),
      'EMA_after_vs_before':error(a['ema_residual_after_filter_rad'],a['ema_residual_before_filter_rad'])}
    if initial is None:initial={k:a[k] for k in ('initial_qpos','initial_qvel','initial_ctrl','initial_cached_target_rad','u_prev_rad')}
    checks={'full151_150':meta['recorded_frames']==151 and meta['recorded_control_intervals']==150 and not meta['rollout_censored'],
      'all150x20finite':proposed.shape==executed.shape==(150,20) and all(np.isfinite(a[k]).all() for k in a if np.issubdtype(a[k].dtype,np.number)),
      'all_actual_u_errors_zero':all(x==0 for x in errors.values()),
      'all_initial_full_state_ctrl_exact':all(np.array_equal(a[k],v) for k,v in initial.items()),
      'u_prev_original_t0':np.array_equal(a['u_prev_rad'],a['initial_ctrl'][a['position_ctrl_ids']]) and error(a['u_prev_rad'],raw['joint_target_rad'][0])==0,
      'runtime_complete':runtime['status']=='completed' and runtime['control_intervals']==150,
      'capacity0_once150_noreset':cap['status']=='OK' and cap['overflow_bitmask']==0 and cap['flags_by_world']==[0] and cap['read_count']==1 and cap['read_at_control_intervals']==150 and cap['actual_world_count']==1 and cap['resets_during_rollout']==meta['resets_during_rollout']==0,
      'capacity_bindings':runtime['capacity_audit']==cap and condition['capacity_audit_sha256']==sha(folder/'capacity_audit.json') and cap['control_npz_sha256']==sha(folder/'control_targets_50hz.npz') and cap['rollout_npz_sha256']==sha(folder/'rollout.npz'),
      'source_identity':meta['script_sha256']==p['source_pins']['evaluator']['sha256'] and cap['worker_sha256']==p['source_pins']['worker']['sha256'] and condition['source_hashes']=={k:v['sha256'] for k,v in p['source_pins'].items()},
      'format103':meta['checkpoint_format_version']==condition['training_checkpoint_format']==runtime['training_checkpoint_format']==cap['training_checkpoint_format']==103}
    evaluated=read(folder/'evaluation.json')
    rows.append({'job_id':job['job_id'],'method':job['trained_method'],'training_seed':job['training_seed'],'evaluation_repeat':job['evaluation_repeat'],
      'checks':checks,'max_abs_errors_rad':errors,'clamped_components':int(np.count_nonzero(executed!=proposed)),
      'cost':{'child_wall_seconds':job['process']['elapsed_s'],'wrapper_wall_seconds':runtime['wall_seconds'],
        'evaluator_report_wall_seconds':evaluated['wall_seconds'],'rollout_report_wall_seconds':meta['wall_seconds']},
      'actual_pid':job['process']['actual_pid'],'actual_OS_creation_utc':job['process']['actual_process_OS_observation']['creation_date_utc']})
sourcechecks={k:sha(v['path'])==v['sha256'] for k,v in p['source_pins'].items()}
permethod={}
for method in ('A1L','P_ONLY','A2'):
    selected=[r for r in rows if r['method']==method]
    permethod[method]={'N_training_seeds':3,'N_fresh_processes':9,'child_wall_sum_seconds':sum(r['cost']['child_wall_seconds'] for r in selected),
      'per_training_seed_child_sum_seconds':{str(s):sum(r['cost']['child_wall_seconds'] for r in selected if r['training_seed']==s) for s in (4,5,6)}}
qualification=read(ROOT/'outputs/format103_qualification_v1/state.json');failed=read(ROOT/'outputs/confirmation_evaluation_v2/state.json')
cost={'v3_whole_driver_wall_seconds':state['whole_queue_wall_seconds'],'v3_sum_child_process_wall_seconds':sum(r['cost']['child_wall_seconds'] for r in rows),
  'v3_sum_wrapper_wall_seconds':sum(r['cost']['wrapper_wall_seconds'] for r in rows),
  'v3_sum_evaluator_report_wall_seconds':sum(r['cost']['evaluator_report_wall_seconds'] for r in rows),
  'engineering_extra_qualification_driver_wall_seconds':qualification['whole_wall_seconds'],
  'engineering_extra_failed_v2_driver_wall_seconds':failed['whole_queue_wall_seconds'],
  'all_disjoint_attempt_driver_walls_sum_seconds':state['whole_queue_wall_seconds']+qualification['whole_wall_seconds']+failed['whole_queue_wall_seconds'],
  'GPU_kernel_busy_seconds':None,'exact_GPU_mutex_held_seconds':None,'metadata_IO_retry_counts':None,
  'interpretation':'Whole-driver, child, wrapper and evaluator scopes nest and must not be summed. Different attempt-driver wall durations are disjoint but exclude human/review gaps. No wall scope is GPU compute time; per-commit retry counts were not instrumented.'}
report={'status':'passed' if all(sourcechecks.values()) and all(all(r['checks'].values()) for r in rows) else 'failed',
  'scope':'Independent full27 CPU control/t0/capacity/source/cost audit only; task gates/COM512 are independently recomputed by frozen analysis',
  'plan_sha256':sha(planpath),'manager_sha256':sha(statepath),'source_checks':sourcechecks,'jobs':rows,'method_costs':permethod,'cost':cost,
  'historical_qualification_and_failed_V2_pooled':False,'GPU_used':False,'audit_CPU_wall_seconds':time.perf_counter()-started}
write(Path(__file__).parent/'report.json',report);print(json.dumps({'status':report['status'],'jobs':len(rows),'cost':cost},indent=2));raise SystemExit(0 if report['status']=='passed' else 1)
