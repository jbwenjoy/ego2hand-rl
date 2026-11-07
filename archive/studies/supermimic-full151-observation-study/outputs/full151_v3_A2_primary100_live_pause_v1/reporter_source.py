"""Schema2 primary100 saved health + exact live four-PID identity and ETA.

Standard-library JSON/filesystem/CIM only. Does not import Torch, native code,
an environment, or a model. Run only after Root supplies actual primary100.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os,re,shlex,statistics,subprocess,sys,traceback

ROOT=Path(__file__).resolve().parents[1]
UTC=datetime.timezone.utc
SHANGHAI=datetime.timezone(datetime.timedelta(hours=8))
HEALTH_AUDITOR_SHA='5950b9947b00ed0cfce6a8403a65877db6d71be1d7b01b8c3ad1626bdd5de792'
CORE_SOURCES=('learning_v2.py','prior_hook_v1.py','training_capacity.py','controller.py','runtime.py','native.py')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):path=Path(path).resolve(strict=True);return {'path':str(path),'sha256':sha(path)}
def checked(value):
    if type(value) is not dict or set(value)!={'path','sha256'} or not Path(value['path']).is_absolute() or sha(value['path'])!=value['sha256']:raise ValueError('Missing/changed exact file pin')
    return Path(value['path']).resolve(strict=True)
def same_pin(a,b):return a['sha256']==b['sha256'] and os.path.samefile(checked(a),checked(b))
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stamp(value):
    d=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None:raise ValueError('UTC offset is required')
    return d.astimezone(UTC)
def file_stamp(path):
    st=Path(path).stat()
    return {'path':str(Path(path).resolve()),'mtime_ns':st.st_mtime_ns,
        'mtime_UTC':datetime.datetime.fromtimestamp(st.st_mtime_ns/1e9,UTC).isoformat(),
        'size_bytes':st.st_size}
def complete_rows(path):
    data=Path(path).read_bytes();end=data.rfind(b'\n')+1
    return [json.loads(line) for line in data[:end].splitlines() if line.strip()],data[:end],len(data)-end
def healthy(value):
    c=value.get('checks')
    return (type(c) is dict and bool(c) and all(type(k) is str and v is True for k,v in c.items())
        and type(value.get('passed')) is int and type(value.get('total')) is int
        and value['passed']==value['total']==len(c))
def normalized_counts(value):
    allowed={'wm','fail_head','actor','critic'}
    if type(value) is not dict or not set(value)<=allowed or any(type(v) is not int or v<0 for v in value.values()):raise ValueError('Malformed optimizer counters')
    if 'wm' not in value or 'fail_head' not in value:raise ValueError('Positive WM/failure counts missing')
    return {k:value.get(k,0) for k in sorted(allowed)}
def argv(line):
    # This fixed project uses ordinary quoted absolute Windows paths. Reject
    # ambiguous embedded quotes rather than guessing or invoking another shell.
    values=[v[1:-1] if len(v)>=2 and v[0]==v[-1]=='"' else v for v in shlex.split(line,posix=False)]
    if any('"' in v for v in values):raise ValueError('Ambiguous quoted process command')
    return values
def command_tail_equal(actual,expected):
    a=argv(actual)
    if len(a)!=len(expected):return False
    for left,right in zip(a[1:],expected[1:]):
        if right.startswith('--') or right=='-u':
            if left!=right:return False
        elif str(Path(left).resolve()).lower()!=str(Path(right).resolve()).lower():return False
    return True
def core_pin(config,name):
    path=ROOT/'src/full151_native'/name
    values=[v for v in config['source_pins'].values() if Path(v['path']).resolve()==path.resolve()]
    if not values or not all(v['sha256']==sha(path) for v in values):raise ValueError('Core executed source missing/changed '+name)
    return values[0]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('health','smoke-run','launch-record','output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'reporter_source.py').write_bytes(Path(__file__).read_bytes())
    checks={};snapshots={};detail={};observed=[];eta=None
    def need(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise ValueError('Failed live-prefix contract: '+name)
    try:
        health=read(a.health);cfgpath=checked(health['config']);cfg=read(cfgpath);run=Path(cfg['output']).resolve();method=cfg['method']
        need('schema2_primary100_health_complete',health['schema']=='full151105_checkpoint_CPU_health:1' and health['status']=='PASS_SAVED_ENGINEERING_HEALTH' and healthy(health)
             and type(health['iteration']) is int and health['iteration']==100 and health['final_iteration_scope'] is False and health['role']=='primary_matched')
        need('trusted_new_auditor',checked(health['auditor']).name=='audit_full151_checkpoint_cpu_v2.py' and health['auditor']['sha256']==HEALTH_AUDITOR_SHA)
        need('new_schema2_method_seed_source',cfg['schema']=='full151_knownrough_native_training_run:2' and cfg['role']=='primary_matched' and method in ('A1L','P_ONLY','A2') and health['method']==method and type(cfg['seed']) is int and cfg['seed']==health['seed']==0 and same_pin(cfg['input_bundle'],health['input_bundle']))
        need('primary_declared2000_WM200',cfg['training']['max_iterations']==2000 and cfg['training']['wm_only_iters']==200 and cfg['training']['warmup_steps']==64 and cfg['training']['num_envs']==128 and cfg['training']['save_interval']==100 and cfg['training']['log_interval']==1)
        need('prefix100_counts_WM_only_not_policy',health['budget_counts']=={'real_transitions':40192,'optimizer_steps':{'wm':200,'fail_head':200,'actor':0,'critic':0},'collect_calls':101,'metric_rows':100})
        need('prefix_optimizer_count_types',normalized_counts(health['budget_counts']['optimizer_steps'])=={'actor':0,'critic':0,'fail_head':200,'wm':200})
        checkpoint=checked(health['checkpoint']);meta_path=checkpoint.with_suffix('.meta.json');meta=read(meta_path)
        need('indexed100_not_final_file',checkpoint==run/'train/model_100.pt' and Path(health['run_dir']).resolve()==run and meta['iteration']==100 and meta['sim_steps']==40192 and meta['sha256']==health['checkpoint']['sha256'] and Path(meta['checkpoint']).resolve()==checkpoint)
        need('meta100_sparse_zero_optimizer_counts',normalized_counts(meta['actual_optimizer_steps'])=={'actor':0,'critic':0,'fail_head':200,'wm':200})
        checkpoint_stat=file_stamp(checkpoint);meta_stat=file_stamp(meta_path)
        need('mtime_sidecar_follows_checkpoint',meta_stat['mtime_ns']>=checkpoint_stat['mtime_ns'])
        for v in cfg['source_pins'].values():checked(v)
        need('config_copy_hash',sha(run/'config.json')==health['config']['sha256'])
        smoke=a.smoke_run.resolve(strict=True);smokecfg=read(smoke/'config.json');smokeresult=read(smoke/'result.json')
        need('same_method_schema2_complete_smoke40',smokecfg['schema']==cfg['schema'] and smokecfg['method']==smokeresult['method']==method and smokecfg['seed']==cfg['seed'] and smokecfg['role']=='engineering_smoke' and smokeresult['schema']=='full151_native_training_completion:2' and smokeresult['status']=='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY' and smokeresult['final_iteration']==40 and smokeresult['full151_capacity_qualified_final'] is True and smokeresult['cleanup_errors']==[] and same_pin(smokecfg['input_bundle'],cfg['input_bundle']))
        exclusions={'max_iterations','wm_only_iters','save_interval'}
        need('smoke_primary_common_hyperparameters', {k:v for k,v in smokecfg['training'].items() if k not in exclusions}=={k:v for k,v in cfg['training'].items() if k not in exclusions} and smokecfg['training']['wm_only_iters']==20 and smokecfg['training']['save_interval']==40)
        need('same_executed_core_source_as_smoke',all(same_pin(core_pin(cfg,n),core_pin(smokecfg,n)) for n in CORE_SOURCES))
        enrollment=read(checked(cfg['readiness']['paired_smoke_enrollment']));smokehealth_pin=enrollment['health'][method];smokehealth=read(checked(smokehealth_pin))
        need('actual_same_method_smoke_health_bound',healthy(smokehealth) and smokehealth['status']=='PASS_SAVED_ENGINEERING_HEALTH' and smokehealth['method']==method and smokehealth['role']=='engineering_smoke' and smokehealth['iteration']==40 and smokehealth['final_iteration_scope'] is True and Path(smokehealth['run_dir']).resolve()==smoke and sha(smoke/'config.json')==smokehealth['config']['sha256'])
        watchdir=run.with_name(run.name+'_watchdog');runtime=read(run/'runtime.json');watch=read(watchdir/'state.json');preflight=read(watchdir/'preflight.json');launch=read(a.launch_record)
        need('launch_schema2_same_method_config_hidden_noretry',launch['schema']=='full151_owned_hidden_supervisor_launch:2' and launch['method']==method and Path(launch['config']).resolve()==cfgpath and launch['config_sha256']==health['config']['sha256'] and launch['hidden'] is True and launch['automatic_retry'] is False)
        if 'initial_CIM' not in launch:raise ValueError('Launch record must bind an initial_CIM pin for exact four CreationDate checks')
        initial=read(checked(launch['initial_CIM']))
        if type(initial) is not list:raise ValueError('Initial four-process CIM rows required')
        ids=[launch['retained_launcher_pid'],watch['supervisor_pid'],watch['launcher_pid'],watch['actual_pid']]
        need('four_distinct_positive_owned_PIDs',len(set(ids))==4 and all(type(v) is int and v>0 for v in ids))
        initial_by={row['pid']:row for row in initial};need('initial_CIM_exact_four_PIDs',set(initial_by)==set(ids))
        query='Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @('+','.join(map(str,ids))+') } | ForEach-Object { [pscustomobject]@{pid=$_.ProcessId;parent_pid=$_.ParentProcessId;creation_utc=$_.CreationDate.ToUniversalTime().ToString("o");command_line=$_.CommandLine;executable_path=$_.ExecutablePath} } | ConvertTo-Json -Compress'
        process=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,timeout=15,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        (out/'live_CIM_stdout.json').write_text(process.stdout,encoding='utf-8');(out/'live_CIM_stderr.log').write_text(process.stderr,encoding='utf-8')
        parsed=json.loads(process.stdout) if process.stdout.strip() else [];observed=parsed if isinstance(parsed,list) else [parsed];by={row['pid']:row for row in observed}
        need('current_CIM_exact_four_owned_PIDs',set(by)==set(ids))
        for pid in ids:
            old=initial_by[pid];now=by[pid]
            need('PID_'+str(pid)+'_same_parent_creation_command_exe',now['parent_pid']==old['parent_pid'] and stamp(now['creation_utc'])==stamp(old['creation_utc']) and now['command_line']==old['command_line'] and os.path.samefile(now['executable_path'],old['executable_path']))
        need('four_PID_parent_chain',by[ids[1]]['parent_pid']==ids[0] and by[ids[2]]['parent_pid']==ids[1] and by[ids[3]]['parent_pid']==ids[2])
        need('four_PID_creation_order',all(stamp(by[left]['creation_utc'])<=stamp(by[right]['creation_utc']) for left,right in zip(ids,ids[1:])))
        command=watch['command'];worker=Path(command[2]).resolve();supervisor=Path(launch['supervisor']).resolve();py=preflight['selected_python_identity']
        need('worker_supervisor_project_namespace',worker.parent==ROOT/'scripts' and supervisor.parent==ROOT/'scripts' and re.fullmatch(r'execute_full151_learning_v\d+\.py',worker.name) is not None and re.fullmatch(r'run_full151_learning_v\d+\.py',supervisor.name) is not None)
        expected_worker=[py['chosen_python'],'-u',str(worker),'--config',str(cfgpath),'--output',str(run),'--run'];expected_sup=[py['chosen_python'],'-u',str(supervisor),'--config',str(cfgpath),'--run']
        need('exact_worker_supervisor_command_routes',command==expected_worker and all(command_tail_equal(by[pid]['command_line'],expected_worker if i>=2 else expected_sup) for i,pid in enumerate(ids)))
        need('venv_redirectors_base_alias_identity',os.path.samefile(by[ids[0]]['executable_path'],py['chosen_python']) and os.path.samefile(by[ids[2]]['executable_path'],py['chosen_python']) and os.path.samefile(by[ids[1]]['executable_path'],py['base_executable_raw']) and os.path.samefile(by[ids[3]]['executable_path'],py['base_executable_resolved']) and os.path.samefile(py['base_executable_raw'],py['base_executable_resolved']) and sha(py['base_executable_resolved'])==py['base_executable_sha256'])
        need('runtime_live_same_worker_source_config',runtime['status']=='running' and runtime['pid']==ids[3] and runtime['parent_pid']==ids[2] and runtime['method']==method and runtime['role']=='primary_matched' and runtime['plan_sha256']==health['config']['sha256'] and runtime['source_sha256']==sha(worker) and runtime['automatic_retry'] is False)
        need('runtime_executed_worker_and_supervisor_archived',all(any(Path(v['path']).resolve()==source and v['sha256']==sha(source) for v in cfg['source_pins'].values()) for source in (worker,supervisor)))
        if 'worker_sha256' in launch:need('launch_optional_worker_source_hash',launch['worker_sha256']==sha(worker))
        if 'supervisor_sha256' in launch:need('launch_optional_supervisor_source_hash',launch['supervisor_sha256']==sha(supervisor))
        initial_worker=watch['actual_process_OS_observation']
        need('watchdog_live_bound_exact_initial_worker',watch['status']=='running' and watch['automatic_retry'] is False and watch['worker_output']==str(run) and initial_worker['status']=='exact_OS_identity_observed' and initial_worker['pid']==ids[3] and initial_worker['parent_pid']==ids[2] and stamp(initial_worker['creation_date_utc'])==stamp(by[ids[3]]['creation_utc']) and initial_worker['command_line']==by[ids[3]]['command_line'])
        live_rows,live_raw,partial_bytes=complete_rows(run/'metrics.jsonl');prefix=live_rows[:100]
        need('exact100_contiguous_finite_metric_prefix',len(prefix)==100 and [r['iteration'] for r in prefix]==list(range(100)) and all(type(r['training/time']) in (int,float) and math.isfinite(r['training/time']) for r in prefix))
        smoke_rows,_,smokepartial=complete_rows(smoke/'metrics.jsonl')
        need('complete40_smoke_profile_rows',len(smoke_rows)==40 and smokepartial==0 and [r['iteration'] for r in smoke_rows]==list(range(40)) and smokeresult['phase_calls']['policy']==20 and smokeresult['phase_calls']['checkpoint']==2)
        base=[prefix[i]['training/time']-prefix[i-1]['training/time'] for i in range(50,100)]
        policy=[smoke_rows[i]['profile/seconds/policy']-smoke_rows[i-1]['profile/seconds/policy'] for i in range(23,40)]
        need('same_method_policy_profile_calls_after_first3',all(smoke_rows[i]['profile/calls/policy']==i-19 for i in range(20,40)))
        save_cost=smokeresult['phase_seconds']['checkpoint']/smokeresult['phase_calls']['checkpoint']
        need('positive_finite_measured_cost_inputs',all(math.isfinite(v) and v>0 for v in base+policy+[save_cost]))
        b=statistics.mean(base);pol=statistics.mean(policy);remaining=1900*b+1800*pol+20*save_cost+20.;lower=.8*(1900*b+1800*pol)+20*save_cost;upper=1.3*(1900*b+1800*pol)+20*save_cost+60.
        anchor=stamp(checkpoint_stat['mtime_UTC']);now=datetime.datetime.now(UTC)
        need('checkpoint_and_sidecar_pins_still_unchanged',pin(checkpoint)==health['checkpoint'] and file_stamp(checkpoint)==checkpoint_stat and file_stamp(meta_path)==meta_stat)
        eta={'schema':'full151_measured_training_ETA:2','method':method,'anchor_checkpoint_iteration':100,'anchor_kind':'model_100.pt stat.st_mtime_ns filesystem timestamp; not startedUTC+native_local_elapsed','anchor_UTC':anchor.isoformat(),'anchor_checkpoint_stat':checkpoint_stat,'crosscheck_meta_stat':meta_stat,'observed_checkpoint_to_meta_mtime_s':(meta_stat['mtime_ns']-checkpoint_stat['mtime_ns'])/1e9,'mtime_scope':'Observed filesystem write timestamps; not a nanosecond proof of atomicrename/commit instant. Sidecar SHA/config/iteration crossbound separately.','meta_worker_elapsed_seconds_native_local':meta['worker_elapsed_seconds'],'meta_elapsed_origin':'learning_v2._run_native local perf start after framework imports; not worker module STARTED, and never added to runtime.startedUTC here.','runtime_started_UTC':runtime['started_utc'],'future_baseline_collect_WM_holdout_log_iterations':1900,'future_H8_policy_updates':1800,'future_checkpoint_writes':20,'same_method_smoke_run':str(smoke),'primary_baseline_last50_differences_s_per_iteration':b,'smoke_policy_trimfirst3_17calls_mean_s':pol,'smoke_save_mean_s':save_cost,'unmeasured_overhead_allowance_s':20.,'remaining_from_checkpoint_s':remaining,'remaining_range_s':[lower,upper],'observed_UTC':now.isoformat(),'estimated_finish_Asia_Shanghai':(anchor+datetime.timedelta(seconds=remaining)).astimezone(SHANGHAI).isoformat(),'finish_range_Asia_Shanghai':[(anchor+datetime.timedelta(seconds=v)).astimezone(SHANGHAI).isoformat() for v in (lower,upper)],'remaining_from_report_time_s':max(0.,remaining-(now-anchor).total_seconds()),'range_scope':'Heuristic .8x..1.3x measured recurring costs with explicit overhead; not CI, guarantee, GPUbusy or future taskquality. Source/thermal/reset/geometry-policy costs may change.','formula':'1900*primary100_last50_training/time_differences +1800*same_method_smoke_policy_trim3 +20*smoke_save +20s','raw_primary_baseline50_slopes_s':base,'raw_smoke_policy17_slopes_s':policy,'policy_at100':'NOT_RUN; WM-only health is not Actor/critic/prior-update qualification'}
        write(out/'health_receipt.json',health);write(out/'runtime_snapshot.json',runtime);write(out/'watchdog_snapshot.json',watch);write(out/'launch_snapshot.json',launch);write(out/'initial_CIM_snapshot.json',initial);write(out/'live_CIM.json',observed);write(out/'checkpoint100_meta_snapshot.json',meta);write(out/'ETA.json',eta)
        (out/'primary_metric_prefix100.jsonl').write_bytes(b'\n'.join(live_raw.splitlines()[:100])+b'\n')
        snapshots={p.name:pin(p) for p in out.iterdir() if p.is_file()}
        detail={'live_iteration_at_observation':runtime.get('iteration'),'live_optimizer_counts_at_observation':runtime.get('actual_optimizer_steps'),'additional_complete_metric_rows_after100_not_used':len(live_rows)-100,'partial_trailing_metric_bytes_ignored':partial_bytes,'four_PID_chain':ids,'initial_CIM_authority':launch['initial_CIM'],'health_config':health['config'],'health_checkpoint':health['checkpoint'],'smoke_config':pin(smoke/'config.json'),'smoke_health':smokehealth_pin,'smoke_result':pin(smoke/'result.json'),'smoke_metrics':pin(smoke/'metrics.jsonl')}
        status='HEALTHY_SCHEMA2_PREFIX100_LIVE_OWNED_TRAINING'
    except BaseException as exc:
        status='FAILED_LIVE_PAUSE_REPORT_CONTRACT';detail.update(error=repr(exc),traceback=traceback.format_exc())
        if observed:write(out/'live_CIM_unqualified.json',observed)
        write(out/'report.json',{'schema':'full151_first_healthy_checkpoint_live_pause_receipt:2','status':status,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'details':detail,'reporter':pin(__file__),'Torch_Warp_GPU_Env_used':False,'training_or_process_modified':False})
        raise
    report={'schema':'full151_first_healthy_checkpoint_live_pause_receipt:2','status':status,'created_UTC':datetime.datetime.now(UTC).isoformat(),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'method':method,'health':pin(a.health),'config':health['config'],'checkpoint':health['checkpoint'],'launch_record':pin(a.launch_record),'ETA':pin(out/'ETA.json'),'details':detail,'snapshot_pins':snapshots,'reporter':pin(__file__),'training_continues_after_goal_pause':True,'Goal_status_changed_by_reporter':False,'terminal_OS_identity':'NOT_YET_TERMINAL','final_capacity_scope':False,'physical_task_success':'NOT_ASSESSED_BY_PREFIX_HEALTH','prefix100_policy_updates':0,'WM_only_prefix_does_not_qualify_Actor_critic_prior':True,'Torch_Warp_GPU_Env_used':False,'training_or_process_modified':False}
    write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'ETA':eta,'live_worker_pid':runtime['pid']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
