"""Semantic saved capacity/lifecycle gates, without frameworks or live OS query.

These gates inspect pinned raw metadata rather than accepting producer/finalizer
booleans as sufficient proof. They do not assert physical numerical accuracy.
"""
from __future__ import annotations
from datetime import datetime
import hashlib,json,os,re
from pathlib import Path

PATHS=('A0','A1L','P_ONLY','A2')
PHASES=(0,4,8,12)
LIFECYCLE_KEYS={'schema','watchdog_state','engineering_identity','post_terminal_CIM','runtime','config'}
IDENTITY_CHECKS={'watchdog_completed_exit0_no_timeout','exact_OS_identity_observed','owned_PID_fields_present',
 'runtime_actual_PID_parent_match','watch_command_equals_launched_command','chosen_venv_supervisor_identity_samefile',
 'OS_command_exe_expected_base_samefile','OS_command_worker_plan_output_bound','runtime_source_plan_bound',
 'watched_runtime_matches_terminal_identity','exact_creation_time_order_bound','postterminal_CIM_exact_owned_PIDs_absent'}


def need(ok,message):
    if not ok:raise ValueError(message)


def checked(item):
    need(type(item) is dict and set(item)=={'path','sha256'},'Exact path/SHA pin required')
    p=Path(item['path']);need(p.is_absolute() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Saved evidence pin mismatch')
    return p


def read(item):
    return json.loads(checked(item).read_text(encoding='utf-8-sig'),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def isint(x,minimum=0):return type(x) is int and x>=minimum


def integer_shape(value):
    if type(value) is int:return [],[value]
    need(type(value) is list and len(value)>0,'Nonempty integer flag array required')
    children=[integer_shape(v) for v in value]
    need(all(v[0]==children[0][0] for v in children),'Ragged raw flag array')
    return [len(value)]+children[0][0],[x for _,xs in children for x in xs]


def capacity_semantics(items):
    need(type(items) is list and len(items)==4,'Four complete capacity observers required')
    counts=[]
    for path_index,(path,item) in enumerate(zip(PATHS,items)):
        need(type(item) is dict and set(item)=={'path','ledger'} and item['path']==path,'Capacity path/order mismatch')
        ledger=item['ledger']
        need(type(ledger) is dict and ledger.get('schema')=='rough_prefix_reset_aware_capacity_observation:1','Wrong frozen7380 capacity schema')
        need(ledger.get('status')=='OK' and ledger.get('engineering_capacity_qualified') is True and ledger.get('finalized') is True,'Capacity not finalized/qualified/OK')
        for key,expected in [('cumulative_OR_bitmask',0),('unknown_read_count',0),('final_read_count',1),('before_reset_read_count',1)]:
            need(type(ledger.get(key)) is int and ledger[key]==expected,'Bad capacity aggregate '+key)
        need(ledger.get('backend_field')=='env.sim.wp_data.overflow' and ledger.get('configured_fixed_capacities')=={'nconmax':192,'njmax':4096},'Capacity backend/limits differ')
        expected=[('before_reset',path)]+[('after_substep',path)]*60+[('donor_terminal',path)]
        for sid in range(path_index*4,path_index*4+4):
            for bid in range(sid*35,(sid+1)*35):
                stage=f'{path}/{PHASES[sid%4]}/branch_{bid}'
                expected += [('before_restore',stage)]+[('after_substep',stage)]*4+[('after_branch',stage)]
        expected += [('final',f'{path}/12/branch_{path_index*140+139}')]
        records=ledger.get('records')
        need(type(records) is list and len(records)==len(expected)==903,'Incomplete capacity readpoint coverage')
        need(type(ledger.get('record_count')) is int and ledger['record_count']==len(records),'Record count differs')
        cumulative=0;unknown=0
        for i,(record,(event,stage)) in enumerate(zip(records,expected),1):
            need(type(record) is dict and record.get('sequence')==i and type(record.get('sequence')) is int,'Bad capacity record sequence')
            need(record.get('event')==event and record.get('stage')==stage,'Missing/reordered restore/substep/branch/donor/final readpoint')
            need(record.get('read_attempts_for_record')==1 and type(record.get('read_attempts_for_record')) is int,'Capacity observation repeated/unknown')
            need(record.get('backend_field')=='env.sim.wp_data.overflow' and record.get('flags_cleared_by_observer') is False,'Observer changed flags or field')
            need(type(record.get('raw_flags_dtype')) is str and re.fullmatch(r'u?int(8|16|32|64)',record['raw_flags_dtype']),'Raw flags not integer dtype')
            shape,values=integer_shape(record.get('raw_flags_values'))
            need(record.get('raw_flags_shape')==shape and all(isint(v) for v in record.get('raw_flags_shape',[])),'Raw flag shape/value mismatch')
            need(all(v>=0 for v in values),'Negative/unknown raw overflow bits')
            mask=0
            for value in values:mask|=value
            cumulative |= mask
            need(record.get('read_status')=='OK' and record.get('overall_status')=='OK','Unknown/overflow record')
            need(record.get('error') is None,'Capacity record contains read/selection error')
            need(type(record.get('raw_bitmask')) is int and record['raw_bitmask']==mask==0,'Nonzero/forged raw bitmask')
            need(type(record.get('cumulative_OR_bitmask')) is int and record['cumulative_OR_bitmask']==cumulative==0,'Cumulative capacity OR inconsistent')
            need(type(record.get('cumulative_unknown_reads')) is int and record['cumulative_unknown_reads']==unknown==0,'Unknown capacity reads')
        counts.append({'path':path,'records_recomputed':len(records),'after_substep_reads':620,
                       'before_restore_reads':140,'after_branch_reads':140,'final_reads':1,'OR_recomputed':cumulative,'unknown_recomputed':unknown})
    return {'qualified':True,'observers':counts,'records_recomputed':sum(c['records_recomputed'] for c in counts),
            'scope':'Raw sticky flags and frozen observer readpoint coverage only; raw nacon/nefc/restore/numerical gates remain separately required.'}


def timestamp(value):
    need(type(value) is str,'Missing timestamp')
    stamp=datetime.fromisoformat(value.replace('Z','+00:00'));need(stamp.utcoffset() is not None,'Naive timestamp')
    return stamp


def normalized(path):return os.path.normcase(str(Path(path).resolve()))


def lifecycle_semantics(evidence,output):
    need(type(evidence) is dict and set(evidence)==LIFECYCLE_KEYS and evidence['schema']=='current104_physical_probe_lifecycle:1','Lifecycle exact schema/keyset differs')
    state=read(evidence['watchdog_state']);identity=read(evidence['engineering_identity']);post=read(evidence['post_terminal_CIM']);runtime=read(evidence['runtime']);config=read(evidence['config'])
    need(state.get('status')=='completed' and type(state.get('exit_code')) is int and state['exit_code']==0 and state.get('timeout') is False,'Watchdog incomplete/timeout/nonzero')
    checks=identity.get('checks')
    need(identity.get('status')=='ENGINEERING_OS_IDENTITY_QUALIFIED' and identity.get('engineering_identity_qualified') is True and type(checks) is dict and set(checks)==IDENTITY_CHECKS and all(v is True for v in checks.values()),'Actual identity unqualified/unknown')
    need(identity.get('parse_error') is None and identity.get('next_condition_permitted_by_identity') is True,'Identity parse failure')
    pid,launcher,supervisor=(state.get(k) for k in ('actual_pid','launcher_pid','supervisor_pid'))
    need(all(isint(p,1) for p in (pid,launcher,supervisor)),'Positive actual/launcher/supervisor PID required')
    observed=state.get('actual_process_OS_observation',{})
    need(observed.get('status')=='exact_OS_identity_observed','No exact actual OS observation')
    parent=supervisor if pid==launcher else launcher
    need(runtime.get('pid')==pid and type(runtime.get('pid')) is int and runtime.get('parent_pid')==parent and type(runtime.get('parent_pid')) is int,'Runtime PID/PPID differs')
    need(observed.get('pid')==pid and type(observed.get('pid')) is int and observed.get('parent_pid')==parent and type(observed.get('parent_pid')) is int,'Observed PID/PPID differs')
    need(identity.get('actual_pid')==pid and type(identity.get('actual_pid')) is int and identity.get('launcher_pid')==launcher and type(identity.get('launcher_pid')) is int,'Identity receipt PID differs')
    need(config.get('GPU_run_authorized') is True and config.get('automatic_retry') is False,'Actual physical config not authorized/no retry')
    need(type(config.get('hard_process_seconds')) is int and config['hard_process_seconds']==240 and type(state.get('hard_seconds')) is int and state['hard_seconds']==240,'Physical watchdog budget differs')
    entry=config.get('source_pins',{}).get('entry');checked(entry)
    need(identity.get('identity_source_sha256')=='f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475','Identity classifier source differs')
    source_pins=config.get('source_pins',{})
    need(any(v.get('sha256')==identity.get('supervisor_sha256') for v in source_pins.values() if type(v) is dict),'Identity supervisor source not config-bound')
    need(runtime.get('plan_sha256')==evidence['config']['sha256'] and runtime.get('source_sha256')==entry['sha256'],'Runtime config/entry bytes differ')
    need(normalized(config.get('output'))==normalized(output) and normalized(state.get('worker_output'))==normalized(output),'Saved output routing differs')
    need(Path(evidence['runtime']['path']).resolve()==Path(output).resolve()/'runtime.json','Runtime not actual physical output')
    captured=state.get('runtime_metadata',{})
    for key in ('pid','parent_pid','started_utc','plan_sha256','source_sha256'):
        need(type(captured.get(key)) is type(runtime.get(key)) and captured.get(key)==runtime.get(key),'Watched runtime identity differs: '+key)
    command=state.get('command')
    need(type(command) is list and len(command)==8 and all(type(x) is str for x in command) and command[1]=='-u' and command[3]=='--config' and command[5]=='--output' and command[7]=='--run','Unknown launched command ABI')
    need(normalized(command[2])==normalized(entry['path']) and normalized(command[4])==normalized(evidence['config']['path']) and normalized(command[6])==normalized(output),'Launched source/config/output differs')
    line=observed.get('command_line');need(type(line) is str,'Missing actual OS command')
    match=re.match(r'\s*(?:"([^"]+)"|(\S+))',line);need(match is not None,'Missing OS command executable')
    first=match.group(1) or match.group(2);python=identity.get('selected_python_identity',{})
    need(os.path.samefile(first,observed['executable_path']) and os.path.samefile(first,python['base_executable_raw']) and os.path.samefile(first,python['base_executable_resolved']),'OS executable/base aliases differ')
    need(hashlib.sha256(Path(first).read_bytes()).hexdigest()==python['base_executable_sha256'],'OS base executable changed')
    need(os.path.samefile(command[0],python['chosen_python']) and os.path.samefile(command[0],python['supervisor_sys_executable']),'Chosen venv/supervisor differs')
    need(python.get('venv_selected_samefile') is True and python.get('base_alias_resolved_samefile') is True,'Venv/base identity assertions missing')
    def normline(s):return s.replace('/','\\').lower()
    need(all(normline(str(Path(v).resolve())) in normline(line) for v in (entry['path'],evidence['config']['path'],output)) and '--run' in line,'OS command routing differs')
    need(timestamp(state['Popen_prelaunch_utc'])<=timestamp(observed['creation_date_utc'])<=timestamp(runtime['started_utc'])<=timestamp(observed['observed_utc']),'Creation/start/observation order differs')
    need(timestamp(runtime['finished_utc'])>=timestamp(runtime['started_utc']) and timestamp(state['finished_utc'])>=timestamp(runtime['finished_utc']),'Terminal clocks inconsistent')
    need(runtime.get('status')=='completed_physical_probe' and type(runtime.get('physics_steps')) is int and runtime['physics_steps']==2480,'Runtime not full completed physical probe')
    need(runtime.get('optimizer_updates')==0 and type(runtime.get('optimizer_updates')) is int and runtime.get('automatic_retry') is False,'Unexpected optimizer/retry')
    need(post.get('status')=='NO_OWNED_PID_REMAINING' and post.get('actual_CIM_rows')==[] and post.get('queried_pids')==sorted(set((pid,launcher))),'Actual terminal owned PIDs not proven absent')
    need(all(type(v) is int for v in post['queried_pids']),'Invalid terminal PID types')
    return {'qualified':True,'actual_pid':pid,'launcher_pid':launcher,'supervisor_pid':supervisor,
            'raw_completed_exit0_no_timeout':True,'raw_postterminal_owned_PIDs_absent':True,
            'scope':'Saved exact OS chain/source/config/command/timestamp evidence only; no new CIM query or GPU-idleness claim.'}


def qualify_saved_evidence(manifest):
    """Return false plus reasons for semantically bad but readable pinned evidence."""
    capacity=read(manifest['capacity_ledgers']);lifecycle=read(manifest['lifecycle_evidence'])
    output=Path(manifest['physical_arrays']['path']).parent
    need(checked(manifest['capacity_ledgers']).resolve()==output.resolve()/'capacity_ledgers.json','Capacity evidence not actual physical output')
    result={'qualified':False,'capacity_semantics':None,'lifecycle_semantics':None,'errors':[]}
    for key,fn in [('capacity_semantics',lambda:capacity_semantics(capacity)),
                   ('lifecycle_semantics',lambda:lifecycle_semantics(lifecycle,output))]:
        try:result[key]=fn()
        except (ValueError,TypeError,KeyError,FileNotFoundError,OSError) as exc:result['errors'].append({'gate':key,'error':repr(exc)})
    result['qualified']=not result['errors']
    return result
