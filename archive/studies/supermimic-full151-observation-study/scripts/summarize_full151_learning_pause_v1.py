"""Saved CPU prefix health plus live owned PID identity and measured ETA."""
from pathlib import Path
import argparse,datetime,json,math,os,statistics,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import checked,pin,read,write

def rows(path):
    data=path.read_bytes();end=data.rfind(b'\n')+1
    return [json.loads(x) for x in data[:end].splitlines()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--health',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Fresh report only')
    health=read(a.health);cfg=read(checked(health['config']));run=Path(cfg['output'])
    if health['status']!='PASS_SAVED_ENGINEERING_HEALTH' or health['iteration']!=100 or health['role']!='primary_matched' or health['final_iteration_scope'] is not False or not health['checks'] or not all(v is True for v in health['checks'].values()):raise ValueError('Healthy exact100 primary prefix required')
    checked(health['checkpoint']);checked(health['auditor'])
    runtime=read(run/'runtime.json');watch=read(run.with_name(run.name+'_watchdog')/'state.json')
    launchdir=ROOT/'outputs/full151_native_A1L_primary2000_launch_v1';launch=read(launchdir/'launch.json')
    ids=[launch['retained_launcher_pid'],watch['supervisor_pid'],watch['launcher_pid'],watch['actual_pid']]
    query='Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @('+','.join(map(str,ids))+') } | ForEach-Object { [pscustomobject]@{pid=$_.ProcessId;parent_pid=$_.ParentProcessId;creation_utc=$_.CreationDate.ToUniversalTime().ToString("o");command_line=$_.CommandLine;executable_path=$_.ExecutablePath} } | ConvertTo-Json -Compress'
    proc=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,timeout=15,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    observed=json.loads(proc.stdout);observed=observed if isinstance(observed,list) else [observed]
    actual=next((x for x in observed if x['pid']==runtime['pid']),None)
    initial=watch['actual_process_OS_observation']
    checks={'live_worker_same_PID_parent_creation_command':actual is not None and actual['pid']==watch['actual_pid']==initial['pid'] and actual['parent_pid']==runtime['parent_pid']==initial['parent_pid'] and datetime.datetime.fromisoformat(actual['creation_utc'].replace('Z','+00:00'))==datetime.datetime.fromisoformat(initial['creation_date_utc'].replace('Z','+00:00')) and actual['command_line']==initial['command_line'],
        'worker_runtime_running_config_source_bound':runtime['status']=='running' and runtime['plan_sha256']==health['config']['sha256'] and runtime['source_sha256']==pin(ROOT/'scripts/execute_full151_learning_v1.py')['sha256'],
        'watchdog_running_exact_plan':watch['status']=='running' and watch['automatic_retry'] is False and watch['command'][watch['command'].index('--config')+1]==health['config']['path'],
        'hidden_detached_supervisor_alive':launch['hidden'] is True and all(any(x['pid']==pid for x in observed) for pid in ids),
        'raw_health_complete':health['passed']==health['total']==len(health['checks']) and all(v is True for v in health['checks'].values())}
    if not all(checks.values()):raise ValueError('Live owned identity/health not qualified: '+repr(checks))
    live_rows=rows(run/'metrics.jsonl');prefix=live_rows[:100]
    if len(prefix)!=100 or [r['iteration'] for r in prefix]!=list(range(100)):raise ValueError('Exact100 metrics prefix required')
    base=[prefix[i]['training/time']-prefix[i-1]['training/time'] for i in range(50,100)]
    smoke=ROOT/'outputs/full151_native_A1L_smoke40_v1';sm=rows(smoke/'metrics.jsonl')
    policy=[sm[i]['profile/seconds/policy']-sm[i-1]['profile/seconds/policy'] for i in range(23,40)]
    if not all(math.isfinite(x) and x>0 for x in base+policy):raise ValueError('Invalid measured timing slopes')
    save=read(smoke/'result.json')['phase_seconds']['checkpoint']/2
    meta=read(checked(health['checkpoint']).with_suffix('.meta.json'))
    anchor=datetime.datetime.fromisoformat(runtime['started_utc'])+datetime.timedelta(seconds=meta['worker_elapsed_seconds'])
    b=statistics.mean(base);p=statistics.mean(policy)
    # The first100 are WM-only; future1800 policy iterations are charged
    # separately using measured full151 smoke H8 cost, with startup3 excluded.
    remaining=1900*b+1800*p+20*save+20.
    lower=.8*(1900*b+1800*p)+20*save
    upper=1.3*(1900*b+1800*p)+20*save+60.
    tz=datetime.timezone(datetime.timedelta(hours=8))
    eta={'schema':'full151_measured_training_ETA:1','anchor_checkpoint_iteration':100,'anchor_UTC':anchor.isoformat(),
        'future_WM_only_and_collect_iterations':1900,'future_H8_policy_iterations':1800,'future_checkpoint_writes':20,
        'primary_baseline_last50_s_per_iteration_mean':b,'smoke_policy_s_per_iteration_trim_first3_mean':p,'smoke_checkpoint_s_per_write':save,
        'unmeasured_overhead_allowance_s':20.,'remaining_from_checkpoint_s':remaining,'remaining_range_s':[lower,upper],
        'estimated_finish_Asia_Shanghai':(anchor+datetime.timedelta(seconds=remaining)).astimezone(tz).isoformat(),
        'finish_range_Asia_Shanghai':[(anchor+datetime.timedelta(seconds=x)).astimezone(tz).isoformat() for x in (lower,upper)],
        'range_scope':'Heuristic .8x–1.3x measured steady costs plus explicit overhead, not a confidence interval or guarantee. Thermal/load/reset/policy costs may change.',
        'formula':'1900*live_WM_only_baseline(last50)+1800*smoke_H8_policy(trim3)+20*save+20s',
        'raw_primary_baseline_slopes_s':base,'raw_smoke_policy_slopes_s':policy}
    a.output.mkdir(parents=True);write(a.output/'health_receipt.json',health);write(a.output/'runtime_snapshot.json',runtime);write(a.output/'watchdog_snapshot.json',watch);write(a.output/'live_CIM.json',observed);write(a.output/'ETA.json',eta)
    report={'schema':'full151_first_healthy_checkpoint_live_pause_receipt:1','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'HEALTHY_SAVED_PREFIX_LIVE_OWNED_TRAINING','checks':checks,'passed':sum(checks.values()),'total':len(checks),
        'health':pin(a.health),'checkpoint':health['checkpoint'],'config':health['config'],'input_bundle':cfg['input_bundle'],'live_worker_pid':actual['pid'],
        'launch':pin(launchdir/'launch.json'),'ETA':pin(a.output/'ETA.json'),'reporter':pin(__file__),
        'training_continues_after_goal_pause':True,'terminal_OS_identity':'NOT_YET_TERMINAL','final_capacity_scope':False,
        'physical_task_success':'NOT_ASSESSED_BY_PREFIX_HEALTH','current_policy_updates_at100':0,'WM_only_prefix_does_not_qualify_Actor_or_critic':True}
    write(a.output/'report.json',report)
    print(json.dumps({'report':pin(a.output/'report.json'),'ETA':eta,'live_worker_pid':actual['pid']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
