"""Bind terminal owned-process evidence to saved physical data; no model load."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin,checked,sha
from rough_prefix.metadata import write


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-directory',type=Path,required=True)
    ap.add_argument('--run',action='store_true');args=ap.parse_args()
    out=args.run_directory.resolve();logs=out.with_name(out.name+'_watchdog')
    if out!=(ROOT/'outputs/current104_physical_FD_v2').resolve():raise ValueError('Exact owned attempt required')
    required=[out/f for f in ('config.json','runtime.json','physical_report.json','physical_arrays.npz','branch_index.json','capacity_ledgers.json','source_and_input_preservation.json')]
    required += [logs/f for f in ('state.json','engineering_identity.json','post_terminal_CIM.json')]
    missing=[str(f) for f in required if not f.is_file()]
    if missing:
        print(json.dumps(dict(status='NOT_READY_MISSING_TERMINAL_DATA',missing=missing,WM_forwards=0,GPU_used=False)));return
    if not args.run:
        print(json.dumps(dict(status='SAVED_ONLY_FINALIZER_PREFLIGHT',files=len(required),WM_forwards=0,GPU_used=False)));return
    if any((out/name).exists() for name in ('lifecycle_evidence_v2.json','physical_manifest_v2.json','finalization_report_v2.json')):raise ValueError('Finalization v2 artifact already exists; no overwrite')
    config=read(out/'config.json');runtime=read(out/'runtime.json');report=read(out/'physical_report.json')
    state=read(logs/'state.json');identity=read(logs/'engineering_identity.json');post=read(logs/'post_terminal_CIM.json')
    preservation=read(out/'source_and_input_preservation.json');ledgers=read(out/'capacity_ledgers.json')
    checks={}
    def need(name,value):checks[name]=bool(value)
    need('owned_supervisor_exit0_no_timeout',state.get('status')=='completed' and state.get('exit_code')==0 and state.get('timeout') is False)
    need('exact_OS_identity_all_checks',identity.get('engineering_identity_qualified') is True and identity.get('status')=='ENGINEERING_OS_IDENTITY_QUALIFIED' and bool(identity.get('checks')) and all(v is True for v in identity['checks'].values()))
    pids=sorted(set(v for v in (state.get('actual_pid'),state.get('launcher_pid')) if type(v) is int and v>0))
    need('postterminal_exact_owned_PIDs_absent',bool(pids) and post.get('status')=='NO_OWNED_PID_REMAINING' and post.get('queried_pids')==pids and post.get('actual_CIM_rows')==[])
    need('runtime_PID_and_source_bound',runtime.get('pid')==state.get('actual_pid')==identity.get('actual_pid') and runtime.get('plan_sha256')==sha(out/'config.json') and runtime.get('source_sha256')==config['source_pins']['entry']['sha256'])
    command=state.get('command')
    if type(command) is not list or len(command)!=8 or command[3]!='--config':raise ValueError('Exact executed command ABI required')
    actual_config=Path(command[4]).resolve(strict=True)
    need('actual_config_bytes_match_saved_copy',sha(actual_config)==sha(out/'config.json')==runtime.get('plan_sha256'))
    need('actual_config_owned_route',actual_config==(ROOT/'configs/current104-physical-FD-authorized_v2.json').resolve())
    need('worker_terminal_completed',runtime.get('status')=='completed_physical_probe')
    need('source_input_and_config_preserved',preservation.get('all_pins_unchanged') is True and preservation.get('config_sha256')==sha(out/'config.json'))
    need('all_physical_engineering_gates',report.get('status')=='COMPLETED_PHYSICAL_PROBE' and all(report.get(k) is True for k in ('engineering_qualified','baseline_all_passed','capacity_finalized','capacity_qualified')))
    need('full_scheduled_coverage',report.get('completed_branch_count')==560 and report.get('completed_snapshot_count')==16 and report.get('scheduled_substeps')==2480)
    need('controlled_scope_and_no_learning',report.get('source_lineage_N')==1 and report.get('independent_ego_N')==0 and report.get('optimizer_updates')==0 and report.get('WM_forwards')==0 and report.get('Actor_forwards')==0)
    need('four_external_capacity_ledgers',isinstance(ledgers,list) and [v.get('path') for v in ledgers]==['A0','A1L','P_ONLY','A2'])
    for i,entry in enumerate(ledgers):
        ledger=entry['ledger'];records=ledger.get('records',[])
        need('capacity_finalized_'+str(i),ledger.get('finalized') is True and ledger.get('engineering_capacity_qualified') is True and ledger.get('status')=='OK' and ledger.get('cumulative_OR_bitmask')==0 and ledger.get('unknown_read_count')==0 and ledger.get('final_read_count')==1)
        need('all_capacity_raw_flags_zero_'+str(i),bool(records) and len(records)==ledger.get('record_count') and all(r.get('read_status')=='OK' and r.get('raw_bitmask')==0 and r.get('cumulative_OR_bitmask')==0 and r.get('cumulative_unknown_reads')==0 and r.get('flags_cleared_by_observer') is False and np.asarray(r.get('raw_flags_values')).size>0 and (np.asarray(r.get('raw_flags_values'))==0).all() for r in records))
    with np.load(out/'physical_arrays.npz',allow_pickle=False) as a:
        need('complete_masks',a['completed'].shape==(560,) and a['completed'].dtype==np.bool_ and a['completed'].all() and a['snapshot_completed'].shape==(16,) and a['snapshot_completed'].dtype==np.bool_ and a['snapshot_completed'].all() and a['restore_qualified'].shape==(560,) and a['restore_qualified'].dtype==np.bool_ and a['restore_qualified'].all())
        need('all_completed_array_values_finite',all(np.isfinite(a[k]).all() for k in a.files))
    inventory=read(out/'source/inventory.json')
    need('source_inventory_equals_actual_config',inventory==config['source_pins'])
    try:
        for v in config['source_pins'].values():checked(v)
        source_current=True
    except Exception:source_current=False
    need('all_registered_source_input_pins_current',source_current)
    lifecycle=dict(schema='current104_physical_probe_lifecycle:1',watchdog_state=pin(logs/'state.json'),
        engineering_identity=pin(logs/'engineering_identity.json'),post_terminal_CIM=pin(logs/'post_terminal_CIM.json'),
        runtime=pin(out/'runtime.json'),config=pin(actual_config))
    write(out/'lifecycle_evidence_v2.json',lifecycle)
    qualified=all(checks.values())
    manifest=dict(schema='current104_physical_probe_result:1',proposal=config['parent_proposal'],
        physical_report=pin(out/'physical_report.json'),physical_arrays=pin(out/'physical_arrays.npz'),
        branch_index=pin(out/'branch_index.json'),source_pins={**config['source_pins'],'saved_only_finalizer':pin(__file__)},
        capacity_ledgers=pin(out/'capacity_ledgers.json'),lifecycle_evidence=pin(out/'lifecycle_evidence_v2.json'),
        postterminal_identity_qualified=qualified)
    write(out/'physical_manifest_v2.json',manifest)
    final=dict(schema='current104_physical_probe_saved_finalization:1',status='QUALIFIED_FOR_CPU_CONSUMER_RECHECK' if qualified else 'UNQUALIFIED_NO_WM_PERMITTED',
               checks=checks,manifest=pin(out/'physical_manifest_v2.json'),finalizer=pin(__file__),new_physics_steps=0,WM_forwards=0,GPU_used=False,
               scope='Saved-only qualification; CPU consumer independently rechecks 48baseline numbers and raw capacity/identity semantics before any WM load.')
    write(out/'finalization_report_v2.json',final);print(json.dumps(dict(status=final['status'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
    if not qualified:raise SystemExit(1)


if __name__=='__main__':main()
