"""Files/JSON/readback-array review only; no Torch, target arrays or NN."""
from pathlib import Path
import json,hashlib,math
import numpy as np
H=Path(__file__).resolve().parent;RUN=H/'cpu_run1';OUT=H/'saved_run_review_v1'

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checkpin(v):
    if pin(v['path'])!=v:raise ValueError('Current file pin mismatch: '+v['path'])
def raw_equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def near(a,b):return abs(float(a)-float(b))<=256*np.finfo(np.float64).eps*max(1.,abs(float(a)),abs(float(b)))

def main():
    if OUT.exists():raise ValueError('Fresh review output only')
    OUT.mkdir()
    r=read(RUN/'result.json');health=read(RUN/'health_0250.json');terminal=read(H/'process_run1/terminal.json')
    protocol=read(r['protocol']['path']);recipe=read(protocol['recipe_pin']['path'])
    logs=[json.loads(line) for line in (RUN/'updates.jsonl').read_text(encoding='utf8').splitlines()]
    selection=read(RUN/'data_selection.json');normalizer=read(RUN/'normalizer_scope.json')
    for v in [r['protocol'],r['health'],r['checkpoint'],health['checkpoint'],health['saved_readback_evidence'],terminal['source'],terminal['protocol'],terminal['stdout'],terminal['stderr'],terminal['launcher'],protocol['recipe_pin']]:checkpin(v)
    need_ckpt='a132d171bc938bd17b0d17b617d14317cacba9063d84cdd5fce6a78d5baa5d20'
    pins=(r['checkpoint']==health['checkpoint'] and r['checkpoint']['sha256']==need_ckpt and terminal['protocol']==r['protocol'] and terminal['source']==protocol['sources']['trainer'] and protocol['sources']['trainer']['sha256']=='224a0c9f47e196c626e862aa52d46402a454a6d16f0b497d3720ae74e25ca539' and protocol['recipe']==recipe['recipe'])
    phase=(r['schema']=='phone_geometry_momentum_training_result:1' and r['status']=='STOPPED_AT_FIRST_HEALTHY_250_RESUMABLE_TARGET5000_NOT_COMPLETE' and r['iteration']==health['iteration']==250 and health['gradients_phase']=='PRE_OPTIMIZER_250; weights AFTER_OPTIMIZER_250' and health['first_checkpoint_pausepoint'] is True and health['process_stops_here'] is True and protocol['recipe']['updates']==5000 and protocol['recipe']['stop_after_first_healthy_checkpoint'] is True)
    health_header=(health['schema']=='phone_geometry_momentum_checkpoint_health:1' and health['passed']==health['total']==15 and len(health['checks'])==15 and all(v is True for v in health['checks'].values()))
    expected_losses={'absolute_heads','decoded_velocity','FIT_head_secants','meanFn_log1p','coverage_BCE','total'}
    logok=(len(logs)==250 and [v['iteration'] for v in logs]==list(range(1,251)) and all(set(v['losses'])==expected_losses and all(math.isfinite(x) for x in v['losses'].values()) and math.isfinite(v['clip_pre_norm']) and v['clip_pre_norm']>=0 and math.isfinite(v['update_seconds']) and v['update_seconds']>0 for v in logs) and not (RUN/'checkpoint_1000.pt').exists() and not (RUN/'checkpoint_5000.pt').exists() and not (RUN/'failure.json').exists())
    with np.load(health['saved_readback_evidence']['path'],allow_pickle=False) as z:e={k:z[k].copy() for k in z.files}
    evidence=(set(e)=={'before_reload_raw222_FP32','after_reload_raw222_FP32','Torch_RNG_before','Torch_RNG_after'} and e['before_reload_raw222_FP32'].dtype==np.float32 and e['before_reload_raw222_FP32'].shape==(8,222) and np.isfinite(e['before_reload_raw222_FP32']).all() and raw_equal(e['before_reload_raw222_FP32'],e['after_reload_raw222_FP32']) and e['Torch_RNG_before'].dtype==np.uint8 and raw_equal(e['Torch_RNG_before'],e['Torch_RNG_after']))
    expected_counts=dict(training_FW=250,backward=250,optimizer_steps=250,validation_FW=1,health_FW=2,NN_constructors=2,own_checkpoint_loads=1,native_FK_forward_steps=0)
    counts=(r['counts']==health['counts']==expected_counts and sum(health['counts'][k] for k in ['training_FW','validation_FW','health_FW'])==253 and health['own_CPU_loads']==1 and health['extra_NN_forward_calls']==2 and health['extra_NN_constructors']==1)
    seconds=np.array([v['update_seconds'] for v in logs]);rate_all=float(np.mean(seconds));rate_recent=float(np.mean(seconds[-100:]));remaining=4750
    eta=rate_recent*remaining;setup=health['setup_s'];overhead=health['checkpoint_health_validation_IO_s']
    scenarios=[setup+min(rate_all,rate_recent)*remaining+2*overhead,setup+max(rate_all,rate_recent)*remaining+2*overhead]
    etaok=all([near(rate_all,health['update_seconds_mean_all']),near(rate_recent,health['update_seconds_mean_recent100']),near(math.fsum(seconds),health['training_update_seconds_sum']),near(eta,health['estimated_remaining_s']),near(eta,r['ETA_s']),near(setup,health['estimated_resume_setup_s']),all(near(x,y) for x,y in zip(scenarios,health['estimated_resume_end_to_end_s_rate_scenarios']))])
    rows=(selection['absolute_rows']==2118 and selection['original_complete_rows']==684 and selection['FIT_side_rows']==1434 and selection['baseline_references_not_duplicated']==36 and len(selection['CF_side_absolute_index'])==len(selection['CF_baseline_absolute_index'])==len(selection['CF_actual_selected_du_rad'])==1434 and selection['CF_side_absolute_index']==list(range(684,2118)) and max(selection['CF_baseline_absolute_index'])<684 and len(selection['excluded_original_partial_row_ids'])==36 and len(selection['excluded_FIT_unavailable_branch_ids'])==6 and selection['TEST_CHECK_NPZ_opens']==0)
    norm=(normalizer['rows']==2118 and normalizer['CF_pairs']==1434 and normalizer['fit_only'] is True and normalizer['TEST_CHECK_np_load']==0 and normalizer['geometry_cache']['rows']==2118 and normalizer['validation_geometry_cache']['rows']==114)
    exitok=(terminal['returncode']==0 and terminal['automatic_retry'] is False and terminal['stderr']['bytes']==0 and r['no_Actor_CHORD_or_ego_qualification'] is True)
    checks=dict(current_pins_source_recipe_and_fixed250_artifact=pins,checkpoint_result_phase_and_stop_semantics=phase,producer_health15of15_header_preserved=health_header,
        exactly250_finite_update_records_no251_or_final5000=logok,saved_B8_prediction_and_Torch_RNG_rawbytes=evidence,recorded253NNFW_and_update_counts_consistent=counts,
        last100_training_ETA_and_resume_scenarios_recomputed=etaok,FIT_only_selection_normalizer_scope_receipts=rows and norm,terminal_exit0_no_retry_or_stderr=exitok)
    result=dict(schema='phone_geometry_momentum_WM_saved_files_review:1',status='PASS_SAVED_250_PAUSEPOINT_FILES_NOT_WEIGHT_PAYLOAD_REAUDIT' if all(checks.values()) else 'FAIL_SAVED_FILES_REVIEW_PRESERVED',
        source_author_review=True,auditor=pin(__file__),checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),
        bindings={q.name:pin(q) for q in [RUN/'result.json',RUN/'health_0250.json',RUN/'updates.jsonl',RUN/'data_selection.json',RUN/'normalizer_scope.json',H/'process_run1/terminal.json']},
        checkpoint=r['checkpoint'],readback=health['saved_readback_evidence'],protocol=r['protocol'],
        observed_counts=expected_counts,total_recorded_NN_FW=253,independent_readback_evidence_shape=[8,222],
        ETA=dict(remaining_updates=4750,last100_mean_update_s=rate_recent,remaining_training_s=eta,resume_setup_estimate_s=setup,resume_inside_process_scenarios_s=scenarios,
            external_terminal_minus_health_timer_s=terminal['elapsed_s']-health['elapsed_s'],scope='Training ETA excludes setup/health/IO. Resume scenarios include observed setup and future checkpoint overhead but not all launch/import/exit overhead outside the health timer; heuristic, not precise wall time or confidence bounds.'),
        review_counts=dict(readback_NPZ_reads=1,target_NPZ_reads=0,checkpoint_Torch_loads=0,NN_constructors_forward_AD_updates=0,native_calls=0),
        limitations=['Serialized weights/Adam/grad/normalizer/RNG payload finite/identity tests remain producer health claims; not reloaded here. Root separately owns its CPU payload review.',
          'Recorded training/FW counts source-supported and mutually consistent, not an external call profiler.',
          'The5000 budget is not complete. No TEST/action-check/ego/holding or Actor benefit evaluation by this review.'])
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(OUT/'report.json'),passed=result['passed'],total=result['total'],status=result['status'])))
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
