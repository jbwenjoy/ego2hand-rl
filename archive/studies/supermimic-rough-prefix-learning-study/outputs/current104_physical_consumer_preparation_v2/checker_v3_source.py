"""Only new semantic guard negatives and no-framework CLI contracts.

Synthetic saved records are explicitly fixtures, not OS/physics observations.
This does not rerun the original30 mathematical checks or load any model.
"""
from __future__ import annotations
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import pin,read
from rough_prefix.physical_probe_manifest_guard_v2 import PATHS,PHASES,IDENTITY_CHECKS,capacity_semantics,lifecycle_semantics


def ledgers():
    result=[]
    for pi,path in enumerate(PATHS):
        timeline=[('before_reset',path)]+[('after_substep',path)]*60+[('donor_terminal',path)]
        for sid in range(pi*4,pi*4+4):
            for bid in range(sid*35,(sid+1)*35):
                stage=f'{path}/{PHASES[sid%4]}/branch_{bid}'
                timeline += [('before_restore',stage)]+[('after_substep',stage)]*4+[('after_branch',stage)]
        timeline += [('final',f'{path}/12/branch_{pi*140+139}')]
        records=[dict(sequence=i,event=event,stage=stage,read_attempts_for_record=1,backend_field='env.sim.wp_data.overflow',flags_cleared_by_observer=False,
            raw_flags_dtype='int32',raw_flags_shape=[1],raw_flags_values=[0],raw_bitmask=0,read_status='OK',overall_status='OK',cumulative_OR_bitmask=0,cumulative_unknown_reads=0) for i,(event,stage) in enumerate(timeline,1)]
        ledger=dict(schema='rough_prefix_reset_aware_capacity_observation:1',status='OK',engineering_capacity_qualified=True,finalized=True,
            cumulative_OR_bitmask=0,unknown_read_count=0,final_read_count=1,before_reset_read_count=1,backend_field='env.sim.wp_data.overflow',
            configured_fixed_capacities={'nconmax':192,'njmax':4096},record_count=len(records),records=records)
        result.append(dict(path=path,ledger=ledger))
    return result


def lifecycle_fixture(directory,entry):
    directory.mkdir();physical=directory/'physical';physical.mkdir();watch=directory/'watchdog';watch.mkdir()
    config=directory/'config.json';config.write_text(json.dumps(dict(GPU_run_authorized=True,automatic_retry=False,hard_process_seconds=240,
        output=str(physical),source_pins={'entry':pin(entry)})))
    runtime=physical/'runtime.json';runtime_data=dict(status='completed_physical_probe',pid=1001,parent_pid=1002,
        started_utc='2026-11-04T01:00:02+00:00',finished_utc='2026-11-04T01:00:04+00:00',plan_sha256=pin(config)['sha256'],
        source_sha256=pin(entry)['sha256'],physics_steps=2480,optimizer_updates=0,automatic_retry=False)
    runtime.write_text(json.dumps(runtime_data));python=Path(sys.executable);base=Path(sys._base_executable)
    python_id=dict(chosen_python=str(python),supervisor_sys_executable=str(python),base_executable_raw=str(base),base_executable_resolved=str(base.resolve()),
        base_executable_sha256=pin(base)['sha256'],venv_selected_samefile=True,base_alias_resolved_samefile=True)
    command=[str(python),'-u',str(entry),'--config',str(config),'--output',str(physical),'--run']
    observed=dict(status='exact_OS_identity_observed',pid=1001,parent_pid=1002,creation_date_utc='2026-11-04T01:00:01Z',observed_utc='2026-11-04T01:00:03Z',
        executable_path=str(base),command_line='"'+str(base)+'" -u "'+str(entry)+'" --config "'+str(config)+'" --output "'+str(physical)+'" --run')
    state_data=dict(status='completed',exit_code=0,timeout=False,actual_pid=1001,launcher_pid=1002,supervisor_pid=1003,worker_output=str(physical),
        Popen_prelaunch_utc='2026-11-04T01:00:00Z',finished_utc='2026-11-04T01:00:05Z',actual_process_OS_observation=observed,
        runtime_metadata={k:runtime_data[k] for k in ('pid','parent_pid','started_utc','plan_sha256','source_sha256')},command=command,hard_seconds=240)
    identity_data=dict(status='ENGINEERING_OS_IDENTITY_QUALIFIED',engineering_identity_qualified=True,checks={k:True for k in IDENTITY_CHECKS},parse_error=None,
        next_condition_permitted_by_identity=True,actual_pid=1001,launcher_pid=1002,selected_python_identity=python_id,
        identity_source_sha256='f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475',supervisor_sha256=pin(entry)['sha256'])
    post_data=dict(status='NO_OWNED_PID_REMAINING',actual_CIM_rows=[],queried_pids=[1001,1002])
    data={'watchdog_state':state_data,'engineering_identity':identity_data,'post_terminal_CIM':post_data}
    evidence={'schema':'current104_physical_probe_lifecycle:1','runtime':pin(runtime),'config':pin(config)}
    for key,value in data.items():
        p=watch/(key+'.json');p.write_text(json.dumps(value));evidence[key]=pin(p)
    return evidence,physical,data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    files={'consumer':ROOT/'scripts/consume_current104_physical_probe_cpu_v2.py','qualification_guard':ROOT/'src/rough_prefix/physical_probe_manifest_guard_v2.py','checker':Path(__file__)}
    for k,p in files.items():(out/(k+'__'+p.name)).write_bytes(p.read_bytes())
    checks={}
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError,OSError):checks[name]=True
        else:checks[name]=False
    good=ledgers();cap=capacity_semantics(good);checks['four903_total3612_positive']=cap['qualified'] and cap['records_recomputed']==3612
    cases=[('rawflag1_despite_forged_true',lambda x:x[0]['ledger']['records'][2].update(raw_flags_values=[1])),
           ('rawdtype_float_reject',lambda x:x[0]['ledger']['records'][2].update(raw_flags_dtype='float32')),
           ('rawbool_not_integer',lambda x:x[0]['ledger']['records'][2].update(raw_flags_values=[False])),
           ('unknown_record_reject',lambda x:x[0]['ledger']['records'][2].update(read_status='UNKNOWN')),
           ('missing_before_restore_reject',lambda x:x[0]['ledger']['records'][62].update(event='after_substep')),
           ('missing_after_branch_reject',lambda x:x[0]['ledger']['records'][67].update(event='after_substep')),
           ('reordered_branchstage_reject',lambda x:x[0]['ledger']['records'][62].update(stage='A0/4/branch_0')),
           ('missing_donor_terminal_reject',lambda x:x[0]['ledger']['records'][61].update(event='after_substep')),
           ('nonfinalized_despite_true_report',lambda x:x[0]['ledger'].update(finalized=False)),
           ('final_twice_reject',lambda x:x[0]['ledger'].update(final_read_count=2)),
           ('unknown_aggregate_reject',lambda x:x[0]['ledger'].update(unknown_read_count=1))]
    for name,mutate in cases:
        bad=copy.deepcopy(good);mutate(bad);reject(name,lambda:capacity_semantics(bad))
    evidence,physical,raw=lifecycle_fixture(out/'synthetic_lifecycle',files['checker'])
    checks['saved_lifecycle_semantics_positive']=lifecycle_semantics(evidence,physical)['qualified']
    def lifecycle_bad(name,key,mutate):
        e=copy.deepcopy(evidence);value=read(e[key]['path']);mutate(value);p=out/(name+'.json');p.write_text(json.dumps(value));e[key]=pin(p)
        reject(name,lambda:lifecycle_semantics(e,physical))
    lifecycle_bad('identity_false_even_report_true','engineering_identity',lambda x:x.update(engineering_identity_qualified=False))
    lifecycle_bad('unknown_missing_identity_check','engineering_identity',lambda x:x['checks'].pop('exact_OS_identity_observed'))
    lifecycle_bad('watchdog_exit1_despite_identitytrue','watchdog_state',lambda x:x.update(exit_code=1))
    lifecycle_bad('watchdog_timeout_despite_identitytrue','watchdog_state',lambda x:x.update(timeout=True))
    lifecycle_bad('PIDbool_rejected','watchdog_state',lambda x:x.update(actual_pid=True))
    lifecycle_bad('raw_CIM_nonempty_despite_identitytrue','post_terminal_CIM',lambda x:x.update(actual_CIM_rows=[{'pid':1001}]))
    lifecycle_bad('unobserved_actualOS_rejected','watchdog_state',lambda x:x['actual_process_OS_observation'].update(status='unobserved'))
    lifecycle_bad('sourceSHA_mismatch_reject','runtime',lambda x:x.update(source_sha256='0'*64))
    lifecycle_bad('creation_outside_interval_reject','watchdog_state',lambda x:x['actual_process_OS_observation'].update(creation_date_utc='2026-11-04T00:59:00Z'))
    # Actual metadata-only CLI and blocked explicit CPUrun. All booleans true,
    # but one raw flag is nonzero. No WM or physics is imported/executed.
    capacity_path=physical/'capacity_ledgers.json';bad=copy.deepcopy(good);bad[0]['ledger']['records'][2]['raw_flags_values']=[1];capacity_path.write_text(json.dumps(bad))
    lifecycle_path=out/'synthetic_lifecycle_evidence.json';lifecycle_path.write_text(json.dumps(evidence))
    report_path=out/'synthetic_physical_report.json';report_path.write_text(json.dumps(dict(status='COMPLETED_PHYSICAL_PROBE',engineering_qualified=True,
        baseline_all_passed=True,capacity_finalized=True,capacity_qualified=True,source_lineage_N=1,independent_ego_N=0)))
    placeholder=physical/'physical_arrays.npz';placeholder.write_text('SYNTHETIC_NOT_ARRAYS_SHOULD_NOT_BE_OPENED')
    index=physical/'branch_index.json';index.write_text('[]')
    proposal=pin(ROOT/'proposals/current104_samewarp_physical_fd_v2/protocol.json')
    manifest=out/'synthetic_manifest.json';manifest.write_text(json.dumps(dict(schema='current104_physical_probe_result:1',proposal=proposal,
        physical_report=pin(report_path),physical_arrays=pin(placeholder),branch_index=pin(index),source_pins={'fixture':pin(files['checker'])},
        capacity_ledgers=pin(capacity_path),lifecycle_evidence=pin(lifecycle_path),postterminal_identity_qualified=True)))
    source_names={'consumer':files['consumer'],'qualification_guard':files['qualification_guard'],'math':ROOT/'src/rough_prefix/physical_probe_consumer_math.py',
        'WM_loader':ROOT/'src/rough_prefix/world_model_checkpoint_v2.py','Actor_loader':ROOT/'src/rough_prefix/checkpoint_v2.py','bundle_loader':ROOT/'src/rough_prefix/bundle_v2.py','io':ROOT/'src/rough_prefix/io.py'}
    plan=dict(schema='current104_physical_probe_CPU_consumer_plan:1',proposal=proposal,physical_manifest=pin(manifest),output=str(out/'blocked_explicit_run'),
        source_pins={k:pin(v) for k,v in source_names.items()},CPU_run_authorized=True,soft_worker_seconds=180,automatic_retry=False)
    pp=out/'synthetic_consumer_plan.json';pp.write_text(json.dumps(plan));cmd=[sys.executable,str(files['consumer']),'--run-plan',str(pp)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'}
    default=subprocess.run(cmd,capture_output=True,text=True,timeout=30,env=env);(out/'default_stdout.txt').write_text(default.stdout+default.stderr)
    proof=json.loads(default.stdout) if default.returncode==0 else {}
    checks['actual_default_rejects_rawledger_semantically_noTorch']=default.returncode==0 and proof.get('Torch_imported') is False and proof.get('physical_gates_reported_and_semantically_pass') is False and proof.get('saved_evidence',{}).get('qualified') is False
    blocked=subprocess.run(cmd+['--run'],capture_output=True,text=True,timeout=30,env=env);(out/'blocked_stdout.txt').write_text(blocked.stdout+blocked.stderr)
    saved=read(out/'blocked_explicit_run/report.json') if (out/'blocked_explicit_run/report.json').exists() else {}
    checks['actual_all_booltrue_rawbad_NOT_RUN_before_WM']=blocked.returncode==2 and saved.get('WM_forwards')==0 and saved.get('status')=='NOT_RUN_PHYSICAL_UNQUALIFIED' and not saved.get('saved_evidence_semantic_qualification',{}).get('qualified',True)
    # A second targeted trap: capacity is now genuinely clean, but the pinned
    # supervisor identity receipt is explicitly unqualified despite final booleans.
    identity_bad,physical2,_=lifecycle_fixture(out/'synthetic_identity_attempt',files['checker'])
    capacity2=physical2/'capacity_ledgers.json';capacity2.write_text(json.dumps(good))
    placeholder2=physical2/'physical_arrays.npz';placeholder2.write_text('SYNTHETIC_NOT_ARRAYS_SHOULD_NOT_BE_OPENED')
    index2=physical2/'branch_index.json';index2.write_text('[]')
    raw_identity=read(identity_bad['engineering_identity']['path']);raw_identity['engineering_identity_qualified']=False
    bad_identity_file=out/'synthetic_identity_false.json';bad_identity_file.write_text(json.dumps(raw_identity));identity_bad['engineering_identity']=pin(bad_identity_file)
    lifecycle_bad_file=out/'synthetic_identity_false_evidence.json';lifecycle_bad_file.write_text(json.dumps(identity_bad))
    bad_manifest=read(manifest);bad_manifest.update(capacity_ledgers=pin(capacity2),lifecycle_evidence=pin(lifecycle_bad_file),physical_arrays=pin(placeholder2),branch_index=pin(index2))
    bad_manifest_file=out/'synthetic_identity_false_manifest.json';bad_manifest_file.write_text(json.dumps(bad_manifest))
    bad_plan={**plan,'physical_manifest':pin(bad_manifest_file),'output':str(out/'blocked_identity_run')};bad_plan_file=out/'synthetic_identity_false_plan.json';bad_plan_file.write_text(json.dumps(bad_plan))
    res=subprocess.run([sys.executable,str(files['consumer']),'--run-plan',str(bad_plan_file),'--run'],capture_output=True,text=True,timeout=30,env=env)
    (out/'identity_blocked_stdout.txt').write_text(res.stdout+res.stderr)
    sr=read(out/'blocked_identity_run/report.json') if (out/'blocked_identity_run/report.json').exists() else {}
    checks['actual_booltrue_identityfalse_NOT_RUN_before_WM']=res.returncode==2 and sr.get('WM_forwards')==0 and sr.get('saved_evidence_semantic_qualification',{}).get('capacity_semantics',{}).get('qualified') is True and not sr.get('saved_evidence_semantic_qualification',{}).get('qualified',True)
    checks['no_framework_import']=not any(k in sys.modules for k in ('torch','warp','mujoco_warp'))
    result={'schema':'physical_probe_consumer_semantic_guard_CPU:2','status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,
        'source_pins':{k:pin(v) for k,v in files.items()},'WM_loads':0,'WM_forwards':0,'Env_physics_GPU':False,'scope':'Synthetic saved capacity/lifecycle/CLI only; no new live identity/physics query, no repetition of original30 mathematical proof.'}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if all(checks.values()) else 1


if __name__=='__main__':sys.exit(main())
