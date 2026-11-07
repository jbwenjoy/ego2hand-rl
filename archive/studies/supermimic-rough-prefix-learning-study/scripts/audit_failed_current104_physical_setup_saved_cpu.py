"""Saved-only audit of the first failed physical-probe setup, never rerun it."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/current104_physical_FD_v1'
LOG=RUN.with_name(RUN.name+'_watchdog')
OUT=ROOT/'outputs/current104_physical_failed_setup_audit_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return {'path':str(p.resolve()),'sha256':h.hexdigest()}


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    config=read(RUN/'config.json');runtime=read(RUN/'runtime.json');report=read(RUN/'physical_report.json')
    state=read(LOG/'state.json');identity=read(LOG/'engineering_identity.json');post=read(LOG/'post_terminal_CIM.json')
    failure=read(RUN/'failure.json');inventory=read(RUN/'source/inventory.json')
    checks={};current=[];archive=[]
    checks['authorized_config_exact_b098']=pin(RUN/'config.json')['sha256']=='b0982350c0f1d761dd7421a69b1b4620b86fb3319159511a84d2de5ef209f0f5'
    checks['current_authorized_config_equal_saved_copy']=pin(ROOT/'configs/current104-physical-FD-authorized_v1.json')['sha256']==pin(RUN/'config.json')['sha256']
    checks['source_inventory_equals_actual_config_411']=inventory==config['source_pins'] and len(inventory)==411
    for key,value in inventory.items():
        got=pin(value['path'])
        if got['sha256']!=value['sha256']:current.append({'key':key,'wanted':value,'current':got})
    with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
        expected={f'{i:04d}_{Path(v["path"]).name}':v for i,v in enumerate(inventory.values())}
        checks['archive_exact_membership_411']=set(z.namelist())==set(expected)
        for name,value in expected.items():
            got=hashlib.sha256(z.read(name)).hexdigest()
            if got!=value['sha256']:archive.append({'name':name,'wanted':value['sha256'],'saved':got})
    checks['all411_current_source_data_pins_preserved']=not current
    checks['all411_archived_source_bytes_preserved']=not archive
    auth=read(config['source_pins']['Root_authorization']['path']);base=read(auth['base_config']['path'])
    expected_base={**config,'GPU_run_authorized':False,'source_pins':{k:v for k,v in config['source_pins'].items() if k!='Root_authorization'}}
    checks['authorization_changes_only_flag_plus_receipt']=base==expected_base and config['GPU_run_authorized'] is True and auth['automatic_retry'] is False
    checks['failed_exit1_without_timeout']=state['status']=='failed' and state['exit_code']==1 and state['timeout'] is False and state['automatic_retry'] is False
    checks['actual_OS_identity_observed_10412_21968']=state['actual_pid']==runtime['pid']==identity['actual_pid']==10412 and state['launcher_pid']==runtime['parent_pid']==21968 and state['supervisor_pid']==2480
    observed=state['actual_process_OS_observation']
    checks['exact_creation_and_parent_saved']=observed['status']=='exact_OS_identity_observed' and observed['pid']==10412 and observed['parent_pid']==21968 and observed['creation_date_utc']=='2026-11-04T04:22:24.2135100Z'
    checks['identity11true_only_exit_gate_false']=len(identity['checks'])==12 and identity['checks']['watchdog_completed_exit0_no_timeout'] is False and all(v is True for k,v in identity['checks'].items() if k!='watchdog_completed_exit0_no_timeout') and identity['parse_error'] is None and identity['engineering_identity_qualified'] is False
    checks['saved_postterminal_owned_PIDs_empty']=post['status']=='NO_OWNED_PID_REMAINING' and post['queried_pids']==[10412,21968] and post['actual_CIM_rows']==[]
    checks['runtime_plan_source_and_failed_state']=runtime['status']=='failed_engineering_partial' and runtime['plan_sha256']==pin(RUN/'config.json')['sha256'] and runtime['source_sha256']==config['source_pins']['entry']['sha256']
    checks['failure_trace_is_snapshot_pointer_metadata_before_interval']=all(s in failure['traceback'] for s in ('ProbeTransaction','probe_transaction_v2.py','int(value.ptr)','NoneType')) and failure['no_retry'] is True
    checks['no_completed_scheduled_snapshots_branches_or_steps']=report['status']=='FAILED_ENGINEERING_PARTIAL' and report['scheduled_substeps']==report['completed_branch_count']==report['completed_snapshot_count']==0
    checks['runtime_null_is_unknown_not_fabricated_total0']=runtime['physics_steps'] is None
    calls=report['backend_call_counts']
    checks['setup_API_and_reset_calls_preserved']=calls=={'A0/constructor':{'CPU_mj_forward_API':1,'mjwarp.step_API':2,'mjwarp.forward_API':2},'A0/checked_reset':{'Simulation.forward':2}}
    checks['no_actor_WM_or_optimization']=all(type(report[k]) is int and report[k]==0 for k in ('Actor_forwards','WM_forwards','optimizer_updates'))
    checks['partial_protocol_not_inference_qualified']=all(report[k] is False for k in ('engineering_qualified','baseline_all_passed','capacity_finalized','capacity_qualified'))
    rows=read(RUN/'branch_index.json');a=load(RUN/'physical_arrays.npz')
    checks['all560_rows_NOT_RUN_null_outcomes']=len(rows)==560 and all(row['row_id']==i and row['status']=='NOT_RUN' and row['actualu'] is None and row['r_k'] is None and row['r_kplus1'] is None and row['restorequalified'] is False for i,row in enumerate(rows))
    checks['all_completion_and_restore_masks_false']=all(a[k].dtype==np.bool_ and not a[k].any() for k in ('snapshot_completed','completed','restore_qualified'))
    outcomes=('donor_next_x','donor_self15','donor_object_terms_k','donor_object_terms_kplus1','branch_x_next','branch_self15','branch_u','branch_object_terms_k','branch_object_terms_kplus1')
    checks['no_fabricated_outcome_all_nan']=all(np.isnan(a[k]).all() for k in outcomes)
    partial_inputs=('snapshot_x','snapshot_u','goal_p','goal_R','goal_p_next','goal_R_next')
    checks['only_unqualified_snapshot0_input_fragment_is_finite']=all(np.isfinite(a[k][0]).all() and np.isnan(a[k][1:]).all() for k in partial_inputs)
    checks['no_completed_snapshot_donor_restore_receipts']=all(read(RUN/f)==[] for f in ('snapshot_records.json','donor_records.json','restore_receipts.json'))
    checks['no_physics_or_reward_sample_records']=(RUN/'physical_samples.jsonl').stat().st_size==(RUN/'reward_snapshots.jsonl').stat().st_size==0
    ledgers=read(RUN/'capacity_ledgers.json');ledger=ledgers[0]['ledger'];records=ledger['records']
    checks['only_A0_capacity_setup_two_reads']=len(ledgers)==1 and ledgers[0]['path']=='A0' and [v['event'] for v in records]==['before_reset','final'] and ledger['record_count']==2 and ledger['before_reset_read_count']==ledger['final_read_count']==1
    checks['observed_setup_capacity_raw0_no_unknown']=ledger['finalized'] is True and ledger['status']=='OK' and ledger['cumulative_OR_bitmask']==ledger['unknown_read_count']==0 and all(v['raw_flags_dtype']=='int32' and v['raw_flags_shape']==[1] and v['raw_flags_values']==[0] and v['read_status']=='OK' and v['raw_bitmask']==v['cumulative_OR_bitmask']==v['cumulative_unknown_reads']==0 and v['flags_cleared_by_observer'] is False for v in records)
    bundle=read(config['input_bundle']['path']);context=load(bundle['context']['path'])
    got=load(RUN/'donor_A0/actual_context_readbacks.npz');ctxchecks={}
    fields=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
    for stage in ('after_copy','after_set_const_0'):
        for field in fields:
            x=got[stage+'__'+field];y=np.broadcast_to(context[field][0].astype(x.dtype),x.shape)
            ctxchecks[stage+'__'+field]=np.array_equal(x.view(np.uint8),y.copy().view(np.uint8))
    checks['actual8_both_readbacks_preserved_16']=all(ctxchecks.values())
    xi=load(RUN/'donor_A0/reader_context.npz')['reader_context']
    checks['reader_physical_xi46_exact']=xi.shape==(1,46) and np.array_equal(xi.view(np.uint32),context['reader_context'][:1].view(np.uint32))
    checks['nominal_MJB_bound']=pin(RUN/'donor_A0/nominal_model.mjb')['sha256']==bundle['model']['sha256']
    init=read(RUN/'donor_A0/initial_readback.json');old=load(config['datasets'][0]['input_pins']['raw_states.npz']['path'])
    checks['initial_full_state_control_mocap_matches_archive0']=all(np.array_equal(np.asarray(init[k],np.float32).view(np.uint32),old[v][0].astype(np.float32).view(np.uint32)) for k,v in [('qpos','full_qpos'),('qvel','full_qvel'),('ctrl','ctrl_rad'),('mocap_pos','mocap_pos_w'),('mocap_quat','mocap_quat_wxyz')])
    checks['missing_entry_postsource_and_manifest_not_fabricated']=not (RUN/'source_and_input_preservation.json').exists() and not (RUN/'physical_manifest.json').exists() and not (RUN/'worker_completion.json').exists()
    arrays={k:{'shape':list(v.shape),'dtype':str(v.dtype),'true_count':int(v.sum())} if v.dtype==np.bool_ else {'shape':list(v.shape),'dtype':str(v.dtype),'finite_count':int(np.isfinite(v).sum()),'NaN_count':int(np.isnan(v).sum())} for k,v in a.items()}
    result={'schema':'current104_failed_physical_setup_independent_saved_audit:1','status':'PASS_FAILED_SETUP_PROVENANCE_AUDIT' if all(checks.values()) else 'AUDIT_DISCREPANCY',
      'checks':checks,'passed':sum(checks.values()),'total':len(checks),'run':str(RUN),'config':pin(RUN/'config.json'),
      'auditor':pin(__file__),'source_current_mismatches':current,'source_archive_mismatches':archive,'source_count':411,
      'array_partial_summary':arrays,'actual8_readback_checks':ctxchecks,
      'process_evidence':{'watchdog':pin(LOG/'state.json'),'identity':pin(LOG/'engineering_identity.json'),'postterminal':pin(LOG/'post_terminal_CIM.json'),'actual_PID':10412,'launcher_PID':21968,'supervisor_PID':2480,'actual_OS_creation_utc':observed['creation_date_utc'],'OS_identity_observed':True,'completed_exit_gate':False,'Root_reported_external_launcher_PID':20280,'external_PID_scope':'Root separately reported freshCIM empty; not inferred from saved two-owned-PID query'},
      'cost_seconds':{'worker_whole':runtime['wall_seconds'],'watchdog_nested_worker_wall':state['elapsed_s'],'native_excluding_imports_nested':report['native_wall_seconds_excluding_imports'],'do_not_sum_nested_intervals':True},
      'source_postcheck_scope':'Entry did not reach postsource commit. This saved audit independently hashes current and ZIP411; it does not invent missing entry receipt.',
      'physical_scope':{'scheduled_donor_branch_substeps':0,'completed_snapshots':0,'completed_baselines':0,'completed_FD_branches':0,'wholeprocess_physics_zero_claim':False,'setup_call_counts':calls,'API_calls_not_sum_with_public_graph_replays':True,'WM_status':'NOT_RUN','physical_quality':'NOT_ASSESSED','physical_gradient_qualified':False,'retry_performed':False},
      'failure_scope':'Trace identifies int(None) in _meta(pointer) during snapshot setup. Specific offending native array name/shape was not saved in v1; legalempty-Warp fix proof is a separate newhelper record, not rerun here.',
      'partial_input_scope':'snapshot0 preaction x/u/goals were observed before transaction construction failed, but no snapshot completion/nextstate/label/FD result exists; no partial input is promoted to teacher or transition.',
      'Torch_imported':'torch' in sys.modules,'Env_created_by_audit':False,'GPU_used_by_audit':False,'new_physics_steps':0,'new_WM_forwards':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'report':pin(OUT/'report.json')}))
    if result['status']!='PASS_FAILED_SETUP_PROVENANCE_AUDIT':raise SystemExit(1)


if __name__=='__main__':main()
