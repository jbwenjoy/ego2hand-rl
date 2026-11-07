"""Independent saved-only audit. No production imports, Torch, Warp, MuJoCo or WM.

This audit reads raw arrays/JSON and source bytes only. Runtime inventory identity
and Python-container equality remain source-bound in-process receipts; their raw
object graphs were intentionally not serialized. Every assertion is scoped below.
"""
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import traceback
import zipfile

import numpy as np

ROOT = Path('D:/workspace/supermimic-rough-prefix-learning-study')
RUN = ROOT / 'outputs/current104_physical_FD_v2'
OUT = Path(__file__).parent
PATHS = ['A0', 'A1L', 'P_ONLY', 'A2']
PHASES = [0, 4, 8, 12]
EXPECTED_CONFIG = '02753b7defccc30a5b8bf9a1ee32c5e2bbc17f1dba8280765af6b96e1fd81c15'
checks, details, exceptions = {}, {}, {}
hash_cache = {}

def J(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    key = str(Path(p).resolve())
    if key not in hash_cache:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        hash_cache[key] = h.hexdigest()
    return hash_cache[key]

def C(name, value):
    if name in checks:
        raise ValueError('Duplicate check '+name)
    checks[name] = bool(value)

def section(name):
    def decorate(fn):
        try:
            fn()
        except Exception:
            checks[name+'_exception_free'] = False
            exceptions[name] = traceback.format_exc()
    return decorate

def eq32(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return a.shape == b.shape and a.tobytes() == b.tobytes()

def eqarr(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def readnp(p):
    with np.load(p, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}

def pins(tree):
    if isinstance(tree, dict):
        if {'path', 'sha256'} <= set(tree):
            yield tree
        for v in tree.values():
            yield from pins(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from pins(v)

def timeutc(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00'))

config = J(RUN/'config.json')
report = J(RUN/'physical_report.json')
runtime = J(RUN/'runtime.json')
arrays = readnp(RUN/'physical_arrays.npz')
branches = J(RUN/'branch_index.json')
snapshots = J(RUN/'snapshot_records.json')
restores = J(RUN/'restore_receipts.json')
donors = J(RUN/'donor_records.json')
ledgers = J(RUN/'capacity_ledgers.json')
samples = [json.loads(s) for s in (RUN/'physical_samples.jsonl').read_text().splitlines()]
rewards = [json.loads(s) for s in (RUN/'reward_snapshots.jsonl').read_text().splitlines()]
directions = readnp(config['directions']['arrays']['path'])
snap_arrays = {}
inventory_digests = {}
by_interval = collections.defaultdict(list)
reward_index = {}
for s in samples:
    by_interval[(s['path'], s['mode'], s['branch_id'], s['phase'])].append(s)
for r in rewards:
    key = (r['path'], r['mode'], r['branch_id'], r['k'])
    if key in reward_index:
        raise ValueError('Duplicate reward capture '+str(key))
    reward_index[key] = r

@section('source')
def source():
    sourcepins = config['source_pins']
    inventory = J(RUN/'source/inventory.json')
    C('config_saved_and_actual_authorized_bytes', sha(RUN/'config.json') == EXPECTED_CONFIG)
    C('source_inventory_equals_config', inventory == sourcepins)
    C('current_415_source_pins_exact', len(sourcepins) == 415 and all(sha(p['path']) == p['sha256'] for p in sourcepins.values()))
    allpins = list(pins(config))
    C('all_config_485_pin_occurrences_exact', len(allpins) == 485 and all(sha(p['path']) == p['sha256'] for p in allpins))
    with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
        expected = [f'{i:04d}_{Path(p["path"]).name}' for i,p in enumerate(sourcepins.values())]
        C('ZIP_415_names_order_exact', z.namelist() == expected)
        C('ZIP_415_content_SHA_exact', all(hashlib.sha256(z.read(n)).hexdigest() == p['sha256'] for n,p in zip(expected,sourcepins.values())))
    preserved = J(RUN/'source_and_input_preservation.json')
    C('postrun_preservation_claim_matches_independent_hashes', preserved == {'all_pins_unchanged': True,'config_sha256': EXPECTED_CONFIG,'pin_occurrences': 485})
    C('oneworld_no_retry_no_compile_fallback', config['worlds']==1 and config['automatic_retry'] is False and config['compile_fallback'] is False)
    C('transaction_v3_actual_pin', sourcepins['transaction_helper']['sha256']=='43378a259abf3a38c9f06bca98cf50e755de1f99089ddfe28229d86ac1012851')
    details['source'] = {'current_source_pins':len(sourcepins),'config_pin_occurrences':len(allpins), 'config_sha256':EXPECTED_CONFIG,
                         'ZIP_sha256':sha(RUN/'source/execution_sources.zip')}

@section('lifecycle')
def lifecycle():
    w = RUN.with_name(RUN.name+'_watchdog')
    state = J(w/'state.json'); identity = J(w/'engineering_identity.json')
    post = J(w/'post_terminal_CIM.json'); rootpost = J(RUN/'root_post_terminal_CIM.json')
    obs = state['actual_process_OS_observation']; cmd = state['command']
    C('watchdog_normal_exit_no_timeout', state['status']=='completed' and state['exit_code']==0 and state['timeout'] is False)
    C('OS_12_identity_checks_true', len(identity['checks'])==12 and all(v is True for v in identity['checks'].values()) and identity['engineering_identity_qualified'] is True)
    C('independent_PID_parent_binding', runtime['pid']==obs['pid']==state['actual_pid']==6656 and runtime['parent_pid']==obs['parent_pid']==state['launcher_pid']==9072 and state['supervisor_pid']==35076)
    C('actual_command_plan_source_output_bound', len(cmd)==8 and cmd[3]=='--config' and cmd[5]=='--output' and cmd[7]=='--run' and Path(cmd[6]).resolve()==RUN.resolve() and sha(cmd[4])==EXPECTED_CONFIG and runtime['plan_sha256']==EXPECTED_CONFIG and sha(cmd[2])==runtime['source_sha256'])
    C('OS_command_contains_exact_launched_paths', all(str(x).lower() in obs['command_line'].lower() for x in (cmd[2],cmd[4],cmd[6])) and obs['pid']==runtime['pid'])
    C('chosen_and_actual_Python_identity', os.path.samefile(cmd[0],identity['selected_python_identity']['supervisor_sys_executable']) and os.path.samefile(obs['executable_path'],identity['selected_python_identity']['base_executable_resolved']) and sha(obs['executable_path'])==identity['selected_python_identity']['base_executable_sha256'])
    C('OS_creation_within_launch_runtime_bounds', timeutc(state['Popen_prelaunch_utc']) <= timeutc(obs['creation_date_utc']) <= timeutc(runtime['started_utc']) <= timeutc(obs['observed_utc']) < timeutc(runtime['finished_utc']))
    C('saved_postterminal_owned_CIM_empty', post['actual_CIM_rows']==[] and set(post['queried_pids'])=={6656,9072} and rootpost['rows']==[] and set(rootpost['queried_pids'])=={28428,35076,9072,6656} and rootpost['session_closed'] is True and rootpost['exec_exit_code']==0)
    C('runtime_completed_full_scheduled_work', runtime['status']=='completed_physical_probe' and runtime['physics_steps']==runtime['scheduled_substeps']==2480 and runtime['completed_branches']==560 and runtime['automatic_retry'] is False)
    C('WM_Actor_optimizer_NOT_RUN', report['WM_forwards']==report['Actor_forwards']==report['optimizer_updates']==runtime['optimizer_updates']==0)
    C('scoped_cost_within_owned_budget', 0 < report['native_wall_seconds_excluding_imports'] < runtime['wall_seconds'] < state['elapsed_s'] < state['hard_seconds']==240)
    details['cost']={'native_nested_seconds':report['native_wall_seconds_excluding_imports'],'worker_total_seconds':runtime['wall_seconds'],'watchdog_seconds':state['elapsed_s'],'do_not_sum_nested_costs':True,'observed_worker_PID':obs['pid'],'OS_creation_UTC':obs['creation_date_utc']}
    details['manifest_scope']='Raw qualification here does not inherit old physical_manifest lifecycle.config path. Actual launched config is state.command[4], its SHA equals the saved copy.'

@section('counts_schedule')
def counts_schedule():
    C('all_expected_arrays_finite_and_complete', all(np.isfinite(a).all() for a in arrays.values()) and arrays['completed'].all() and arrays['snapshot_completed'].all() and arrays['restore_qualified'].all())
    C('unique_560_branch_ids_order', [b['row_id'] for b in branches]==list(range(560)) and len(restores)==560 and [r['branch_id'] for r in restores]==list(range(560)))
    C('unique_16_snapshot_order', [(s['snapshot_id'],s['path'],s['k']) for s in snapshots]==[(i,p,k) for i,(p,k) in enumerate((p,k) for p in PATHS for k in PHASES)])
    schedule_ok=True
    for sid in range(16):
        rows=branches[sid*35:(sid+1)*35]
        expected=[('baseline',b,None,None,0) for b in range(3)]+[('direction',None,j,e,sgn) for e in range(2) for j in range(8) for sgn in (-1,1)]
        schedule_ok &= [(b['type'],b['baseline_repeat'],b['j'],b['epsilon_index'],b['sign']) for b in rows]==expected
        schedule_ok &= all(b['snapshot_id']==sid and b['path']==PATHS[sid//4] and b['k']==PHASES[sid%4] and b['status']=='COMPLETED' and b['restorequalified'] is True for b in rows)
    C('frozen_35branch_schedule_exact_no_selection',schedule_ok)
    C('2480_raw_substeps_620_reward_intervals',len(samples)==2480 and len(by_interval)==len(rewards)==len(reward_index)==620)
    C('exact_interval_coverage', set(by_interval)==set(reward_index) and sum(k[1]=='donor' for k in by_interval)==60 and sum(k[1]=='baseline' for k in by_interval)==48 and sum(k[1]=='direction' for k in by_interval)==512)
    C('four_substeps_in_order_per_interval', all([s['substep'] for s in rows]==[1,2,3,4] for rows in by_interval.values()))
    counts=report['backend_call_counts']; public=collections.Counter();construct=collections.Counter()
    for p in PATHS:
        for mode in ['donor','baseline','direction']:
            public.update(counts[p+'/'+mode])
        C(p+'_public_scheduled_counts', counts[p+'/donor']=={'Simulation.forward':15,'Simulation.step':60} and counts[p+'/baseline']=={'Simulation.forward':12,'Simulation.step':48} and counts[p+'/direction']=={'Simulation.forward':128,'Simulation.step':512})
        construct.update(counts[p+'/constructor'])
    C('620_public_interval_forward_plus_8_reset_forwards',public=={'Simulation.forward':620,'Simulation.step':2480} and sum(counts[p+'/checked_reset']['Simulation.forward'] for p in PATHS)==8)
    C('constructor_nested_calls_separate_not_zero',construct=={'CPU_mj_forward_API':4,'mjwarp.step_API':8,'mjwarp.forward_API':8})
    details['counts']={'scheduled_substeps':2480,'interval_rewards':620,'baseline_branches':48,'perturbation_branches':512,'snapshots':16,'restore_event_records':1120,'public_interval_forward':620,'reset_forward':8,'constructor_nested_API':dict(construct),'not_a_global_kernel_count':True}

@section('snapshots_restore')
def snapshots_restore():
    empty_count=0; native_counts=[]; records=[]; pen_keys=None
    for rec in snapshots:
        sid=rec['snapshot_id']; inv=rec['inventory']; a=readnp(rec['state_archive']['path']);snap_arrays[sid]=a
        C(f'snapshot{sid}_archives_SHA_exact',sha(rec['state_archive']['path'])==rec['state_archive']['sha256'] and sha(rec['rng_archive']['path'])==rec['rng_archive']['sha256'])
        prefixes=[('native_Warp_',inv['native_Warp_arrays']),('native_Torch_',inv['native_Torch_tensors']),('explicit_array_',inv['explicit_arrays'])]
        keys=[pref+str(i) for pref,n in prefixes for i in range(n)]
        C(f'snapshot{sid}_numeric_inventory_count', all(k in a for k in keys) and sum(k.startswith(('native_Warp_','native_Torch_','explicit_array_')) for k in a)==len(keys))
        h=hashlib.sha256()
        for k in keys:h.update(a[k].tobytes())
        inventory_digests[sid]=h.hexdigest()
        empty_count += sum(a[k].size==0 for k in a if k.startswith('native_Warp_'))
        C(f'snapshot{sid}_integration_PEN_RNG_source_inventory',a['integration'].shape==(1,rec['integration_size']) and rec['integration_sig']==16383 and rec['integration_roundtrip_bitexact'] is True and a['source_action'].shape==(1,20) and hashlib.sha256(a['source_action'].tobytes()).hexdigest()==inv['source_action_sha256'] and a['RNG_Torch_CPU'].size>0 and a['RNG_Torch_CUDA_0'].size>0 and a['RNG_NumPy_MT'].shape==(624,))
        C(f'snapshot{sid}_strict_real_native_profile',inv['strict_native_profile'] is True and inv['CPU_fixture_is_not_native_qualification'] is False and inv['native_entries']>0 and inv['closure_cells']>0 and inv['capacity_ledger_inside_snapshot'] is False and inv['failed'] is False and inv['native_physics_qualification']=='NOT_GRANTED_BY_HELPER')
        pk=sorted(k for k in a if k.startswith('PEN__'))
        if pen_keys is None:pen_keys=pk
        C(f'snapshot{sid}_PEN_memory_fields_complete',pk==pen_keys and len(pk)>15 and np.isfinite(np.concatenate([a[k].ravel() for k in pk])).all())
        rng=J(rec['rng_archive']['path']); C(f'snapshot{sid}_RNG_sidecar_present',set(rng)=={'python','NumPy_header'} and rng['NumPy_header'][0]=='MT19937')
        native_counts.append({k:inv[k] for k in ['native_entries','native_Warp_arrays','native_Torch_tensors','explicit_containers','explicit_arrays','closure_cells']})
    beginok=finishok=digestok=ledgerlink=True
    for r in restores:
        bid=r['branch_id']; b=branches[bid];sid=b['snapshot_id']; begin=r['begin']; events=r['transaction_events']; ledger=ledgers[sid//4]['ledger']['records']
        beginok &= len(events)==2 and events[0]==begin and begin['branch_id']=='branch_'+str(bid) and begin['snapshot_label']==b['path']+'/'+str(b['k']) and begin['restore_qualified'] is True and begin['inventory']['bitexact'] is True and begin['inventory']['mismatches']==[]
        digestok &= begin['inventory']['value_inventory_sha256']==inventory_digests[sid] and begin['source_action_sha256']==snapshots[sid]['inventory']['source_action_sha256']
        end=events[1];finishok &= end['branch_id']==begin['branch_id'] and end['body_completed'] is True and end['capacity_qualified'] is True
        before=begin['capacity_before_restore'];after=end['capacity_after_branch']
        ledgerlink &= ledger[before['sequence']-1]==before and ledger[after['sequence']-1]==after and after['sequence']-before['sequence']==5
    C('560_begin_restore_inventory_receipts_exact',beginok)
    C('560_live_inventory_hashes_match_saved_native_arrays',digestok)
    C('560_finish_receipts_body_completed',finishok)
    C('1120_transaction_capacity_events_exact_ledger_links',ledgerlink)
    C('actual_real_empty_Warp_arrays_saved',empty_count>0)
    details['snapshot_inventory']={'counts_by_snapshot':native_counts,'empty_Warp_array_occurrences':empty_count,'PEN_keys':pen_keys,'value_inventory_sha256':inventory_digests,
        'independent_scope':'Digest independently rebuilt from all saved Warp/Torch/explicit arrays. Live object/container/RNG equality is recorded by source-pinned in-process assertions, not independently re-executed; complete post-restore object graphs are not serialized.'}

@section('capacity')
def capacity():
    capacity_summary=[];all_ok=True
    for pi,item in enumerate(ledgers):
        p=item['path'];l=item['ledger'];rows=l['records'];or_value=0;unknown=0
        C(p+'_capacity_903_sequence', p==PATHS[pi] and len(rows)==l['record_count']==903 and [x['sequence'] for x in rows]==list(range(1,904)))
        schedule=[('before_reset',p)]+[('after_substep',p)]*60+[('donor_terminal',p)]
        for b in branches[pi*140:(pi+1)*140]:
            stage=p+'/'+str(b['k'])+'/branch_'+str(b['row_id'])
            schedule += [('before_restore',stage)]+[('after_substep',stage)]*4+[('after_branch',stage)]
        schedule += [('final',p+'/12/branch_'+str((pi+1)*140-1))]
        C(p+'_capacity_every_reset_substep_restore_finish_final',[(x['event'],x['stage']) for x in rows]==schedule)
        for x in rows:
            a=np.asarray(x['raw_flags_values'],dtype=np.dtype(x['raw_flags_dtype']))
            bitmask=int(np.bitwise_or.reduce(a.ravel(),initial=0));or_value |= bitmask
            unknown += x['read_status']!='OK'
            all_ok &= list(a.shape)==x['raw_flags_shape'] and bitmask==x['raw_bitmask'] and x['cumulative_OR_bitmask']==or_value and x['cumulative_unknown_reads']==unknown
            all_ok &= x['read_attempts_for_record']==1 and x['flags_cleared_by_observer'] is False and x['backend_field']=='env.sim.wp_data.overflow' and x['flag_shape_is_world_count'] is False
        C(p+'_sticky_OR_zero_unknown_zero_finalized',or_value==l['cumulative_OR_bitmask']==0 and unknown==l['unknown_read_count']==0 and l['finalized'] is True and l['final_read_count']==l['before_reset_read_count']==1 and l['engineering_capacity_qualified'] is True)
        capacity_summary.append({'path':p,'records':len(rows),'raw_OR':or_value,'unknown':unknown,'final_read_once':l['final_read_count']})
    C('3612_raw_flags_OR_and_cumulative_values_recomputed',all_ok)
    details['capacity']=capacity_summary

@section('controls_phases_raw')
def controls_phases_raw():
    baseline_state_errors=[];controls=True;continuity=True;clock=True;noassist=True;rawshape=True;start_restore=True;solverpose=True
    for key,ss in by_interval.items():
        p,mode,bid,k=key
        expected=donors[PATHS.index(p)]['events'][k]['actualu'] if mode=='donor' else arrays['branch_u'][bid]
        for i,s in enumerate(ss):
            controls &= eq32(s['ctrl'],expected)
            clock &= abs(s['pre_time']-(.02*k+.005*i))<=2e-6 and abs(s['post_time']-(.02*k+.005*(i+1)))<=2e-6 and abs((s['post_time']-s['pre_time'])-.005)<=2e-6
            noassist &= s['external_force_max']==0
            rawshape &= len(s['pre_qpos'])==len(s['post_qpos'])==27 and len(s['pre_qvel'])==len(s['post_qvel'])==26 and len(s['actuator_force'])==20 and 0<=s['nacon']<192 and 0<=s['nefc']<4096
            rawshape &= len(s['raw_wp_contact']['geom'])==len(s['raw_wp_contact']['type'])==len(s['contacts'])==s['nacon'] and len(s['raw_efc_force'])==len(s['raw_efc_type'])==s['nefc']
            rawshape &= s['phase_contract']=='solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward'
            solverpose &= eq32(s['object_solver_pose'],s['pre_qpos'][:7]) and eq32(s['object_solver_COM'],s['pre_qpos'][:3])
            if i:continuity &= eq32(s['pre_qpos'],ss[i-1]['post_qpos']) and eq32(s['pre_qvel'],ss[i-1]['post_qvel']) and s['pre_time']==ss[i-1]['post_time']
        if mode!='donor':
            original=by_interval[(p,'donor',None,k)][0]
            start_restore &= eq32(ss[0]['pre_qpos'],original['pre_qpos']) and eq32(ss[0]['pre_qvel'],original['pre_qvel']) and ss[0]['pre_time']==original['pre_time']
        reward=reward_index[key]
        rawshape &= len(reward['physics'])==58 and len(reward['contact'])==15 and len(reward['torque'])==20
        rawshape &= reward['capture']=='after_original_reward_compute_before_Manager_forward' and reward['label_time']=='lastsolver_preintegration~kdt+.015' and reward['state_time']=='postintegration_kdt+.020'
        rawshape &= eq32(reward['torque'],ss[-1]['actuator_force'])
        expected_x=donors[PATHS.index(p)]['events'][k]['x'] if mode=='donor' else arrays['branch_x_next'][bid]
        rawshape &= eq32(reward['physics'],expected_x)
    C('all2480_actual_ctrl_matches_held_intervention_bitexact',controls)
    C('all2480_clock_pre_post_and_4step_phase',clock)
    C('within_interval_q_qd_continuity_exact',continuity)
    C('all560_branch_prestate_restores_match_fresh_donor',start_restore)
    C('all2480_zero_exogenous_force',noassist)
    C('raw_contact_efc_shapes_capacity_and_620_reward_phase',rawshape)
    C('all2480_solver_object_pose_COM_is_preintegration',solverpose)
    requested=True;nativeu=True
    for b in branches:
        sid=b['snapshot_id'];bid=b['row_id']
        want=directions['centers'][sid] if b['type']=='baseline' else directions['plus' if b['sign']==1 else 'minus'][sid,b['j'],b['epsilon_index']]
        requested &= eq32(arrays['branch_u'][bid],want) and eq32(b['actualu'],want)
        nativeu &= np.all(arrays['branch_u'][bid]>=directions['lower']) and np.all(arrays['branch_u'][bid]<=directions['upper'])
    C('560_control_assignments_match_frozen_F32_targets',requested)
    C('560_controls_inside_original_inner_domain',nativeu)
    C('16_fresh_centers_match_predeclared_centers_bitexact',eq32(arrays['snapshot_u'],directions['centers']))
    details['substep_counts_by_path_mode']={p:{mode:sum(s['path']==p and s['mode']==mode for s in samples) for mode in ['donor','baseline','direction']} for p in PATHS}
    details['clock_max_abs_error_s']=max(abs(s['post_time']-(s['phase']*.02+s['substep']*.005)) for s in samples)

@section('baseline_reward_sensor')
def baseline_reward_sensor():
    groups={'q':(0,20,2e-5),'qd':(20,40,2e-4),'p':(40,43,2e-5),'R':(43,52,2e-5),'v':(52,55,2e-4),'omega':(55,58,2e-4)}
    baseline_rows=[];baseline_ok=True;receipt_ok=True;raw3=True
    for sid in range(16):
        ids=list(range(sid*35,sid*35+3)); truth=arrays['donor_next_x'][sid]
        ref_r=float(np.sum(arrays['donor_object_terms_k'][sid],dtype=np.float32))
        basekeys=[]
        for bid in ids:
            b=branches[bid];errors={n:float(np.max(np.abs(arrays['branch_x_next'][bid,a:z].astype(np.float64)-truth[a:z].astype(np.float64)))) for n,(a,z,tol) in groups.items()}
            errors['reward']=abs(b['r_k']-ref_r)
            valid=all(errors[n]<=v[2] for n,v in groups.items()) and errors['reward']<=2e-5
            baseline_ok &= valid;receipt_ok &= restores[bid]['baseline_gate']=={'errors':errors,'passed':valid}
            key=(b['path'],'baseline',bid,b['k']);basekeys.append(key)
            baseline_rows.append({'branch_id':bid,'snapshot_id':sid,'errors':errors,'passed':bool(valid)})
        rr=[reward_index[k] for k in basekeys]
        for r in rr[1:]:raw3 &= all(eq32(r[f],rr[0][f]) for f in ['contact','raw_found','raw_sensordata','torque','physics'])
        for i in range(1,3):
            x,y=by_interval[basekeys[i]],by_interval[basekeys[0]]
            raw3 &= all(a['raw_wp_contact']==b['raw_wp_contact'] and a['contacts']==b['contacts'] and eq32(a['raw_efc_force'],b['raw_efc_force']) and a['raw_efc_type']==b['raw_efc_type'] for a,b in zip(x,y))
    C('all48_sixchannel_and_primary_reward_original_baseline_gates',baseline_ok)
    C('48_recorded_baseline_errors_independently_exact',receipt_ok)
    C('each_snapshot_3baseline_raw_contacts_found_sensor_and_forces_identical',raw3)
    termsok=True;max_term_err=0.;sumok=True
    for bid,b in enumerate(branches):
        sid=b['snapshot_id'];x=arrays['branch_x_next'][bid].astype(np.float64)
        for suffix,gp,gR,rkey in [('k','goal_p','goal_R','r_k'),('kplus1','goal_p_next','goal_R_next','r_kplus1')]:
            p=x[40:43];R=x[43:52].reshape(3,3);goalp=arrays[gp][sid].astype(np.float64);goalR=arrays[gR][sid].astype(np.float64)
            ep=(np.linalg.norm(p+.1*R[:,0]-goalp-.1*goalR[:,0])+np.linalg.norm(p-.1*R[:,0]-goalp+.1*goalR[:,0]))*.5
            target=np.array([2*np.exp(-(np.linalg.norm(p-goalp)/.05)**2),8*np.exp(-(ep/.04)**2)])
            got=arrays['branch_object_terms_'+suffix][bid]
            err=np.abs(target-got);max_term_err=max(max_term_err,float(err.max()))
            termsok &= np.all(err<=8*np.finfo(np.float32).eps*np.maximum(1,np.abs(got)))
            sumok &= b[rkey]==float(got.sum(dtype=np.float32))
    C('1120_object_reward_terms_independent_F64_roundoff_bound',termsok)
    C('1120_recorded_rewards_exact_sum_of_saved_FP32_terms',sumok)
    metadata=J(ROOT/'outputs/native104_WM_raw_interface_review_v1/static_model_metadata.json')
    names=[s['primary_geom_name'].removeprefix('robot/') for s in metadata['finger_collision_sensors']]
    middle=[names.index(f'right_finger{f}_link{l}_col') for f in range(1,6) for l in (2,3)]
    distal=[names.index(f'right_finger{f}_link4_col') for f in range(1,6)];tips=[names.index(f'right_finger{f}_tip_sensor_col') for f in range(1,6)]
    sensor_ok=True;data_equal=True;ones=0;variation=0
    for r in rewards:
        f=np.asarray(r['raw_found'],np.float32);d=np.asarray(r['raw_sensordata'],np.float32);active=f[0]>0
        expected=np.r_[active[middle],active[distal]|active[tips]].astype(np.float32)
        sensor_ok &= f.shape==(1,25) and eq32(expected,r['contact']) and np.isin(r['contact'],[0,1]).all()
        data_equal &= eq32(f,d);ones += int(expected.sum())
    for b in branches:
        bid=b['row_id'];r=reward_index[(b['path'],b['type'],bid,b['k'])]
        sensor_ok &= eq32(r['contact'],arrays['branch_self15'][bid])
    for sid in range(16):
        variation += sum(not eq32(arrays['branch_self15'][sid*35+i],arrays['donor_self15'][sid]) for i in range(3))
    C('620_native_self15_rebuilt_from_raw_found_exact',sensor_ok)
    C('620_raw_found_equals_saved_sensor25_values',data_equal)
    details['baselines']=baseline_rows
    details['baseline_max_errors']={k:max(r['errors'][k] for r in baseline_rows) for k in [*groups,'reward']}
    details['reward_max_F64_vs_native_F32_term_error']=max_term_err
    details['self15']={'native_reward_snapshot_available':True,'all_rebuilt_from_found':bool(sensor_ok),'positive_components_all620':ones,'baseline_vs_donor_different_rows':variation,
        'semantic_scope':'10 middle(link2/link3) plus5 distal(link4 OR tip) self-contact labels, not hand-object contact, not force and not fresh post-forward geometry. Raw sensor25 and raw found equality is observed. Cache phase is original native reward capture.'}

@section('donors_initial_actual_context')
def donors_initial_actual_context():
    initial=True;actionsok=True;archive_u=True;context=True;model=True;drift=[];donor_cont=True;state_fields=True;centerx=True;goalnext=True
    expected_context=None;reader_context=None
    for pi,p in enumerate(PATHS):
        d=donors[pi];pathdir=RUN/('donor_'+p);ds=config['datasets'][pi];archdir=Path(ds['directory']);ar=readnp(archdir/'raw_states.npz');events=J(archdir/'control_events.json');init=J(pathdir/'initial_readback.json')
        initial &= eq32(init['qpos'],ar['full_qpos'][0]) and eq32(init['qvel'],ar['full_qvel'][0]) and eq32(init['ctrl'],ar['ctrl_rad'][0]) and eq32(init['mocap_pos'],ar['mocap_pos_w'][0]) and eq32(init['mocap_quat'],ar['mocap_quat_wxyz'][0]) and init['time']==0
        q=[init['qpos']];qd=[init['qvel']]
        for k in range(15):
            ss=by_interval[(p,'donor',None,k)];e=d['events'][k]
            q.extend(s['post_qpos'] for s in ss);qd.extend(s['post_qvel'] for s in ss)
            if k:donor_cont &= eq32(ss[0]['pre_qpos'],d['events'][k-1]['rawstate']['qpos']) and eq32(ss[0]['pre_qvel'],d['events'][k-1]['rawstate']['qvel'])
            archive_u &= eq32(e['actualu'],np.asarray(events[k]['actualu'],np.float32).reshape(20)) and eq32(e['actualu'],ss[-1]['ctrl'])
            wanted=np.zeros((1,20),np.float32) if p=='A0' else events[k]['action']
            actionsok &= eq32(d['actions'][k],wanted)
            state_fields &= eq32(e['rawstate']['qpos'],ss[-1]['post_qpos']) and eq32(e['rawstate']['qvel'],ss[-1]['post_qvel'])
            state_fields &= eq32(e['x'][:20],ss[-1]['post_qpos'][7:]) and eq32(e['x'][20:40],ss[-1]['post_qvel'][6:])
            if k in PHASES:
                sid=pi*4+PHASES.index(k)
                centerx &= eq32(arrays['snapshot_u'][sid],e['actualu']) and eq32(arrays['donor_next_x'][sid],e['x']) and eq32(arrays['donor_self15'][sid],e['contact'])
                centerx &= eq32(snap_arrays[sid]['source_action'],d['actions'][k])
                before=ss[0]
                centerx &= eq32(arrays['snapshot_x'][sid,:20],before['pre_qpos'][7:]) and eq32(arrays['snapshot_x'][sid,20:40],before['pre_qvel'][6:])
                goalnext &= eq32(arrays['donor_object_terms_k'][sid],e['terms_k']) and eq32(arrays['donor_object_terms_kplus1'][sid],e['terms_next'])
        q=np.asarray(q);qd=np.asarray(qd)
        eq=float(np.abs(q[:,7:]-ar['joint_rad']).max());eqd=float(np.abs(qd[:,6:]-ar['joint_velocity_rad_s']).max())
        flag=eq>2e-5 or eqd>2e-4
        C(p+'_archive_drift_saved_values_recomputed',eq==d['archive_q_maxabs'] and eqd==d['archive_qd_maxabs'] and flag==d['archival_drift_flag'] and d['archive_quality_not_regraded'] is True)
        drift.append({'path':p,'archive_q_maxabs_rad':eq,'archive_qd_maxabs_rad_s':eqd,'flag_only_not_regraded':flag})
        ex=readnp(pathdir/'expected_context_fields.npz');got=readnp(pathdir/'actual_context_readbacks.npz')
        if expected_context is None:expected_context=ex
        context &= len(ex)==8 and set(got)=={stage+'__'+k for stage in ['after_copy','after_set_const_0'] for k in ex}
        for k,a in ex.items():
            context &= eqarr(a,expected_context[k])
            for stage in ['after_copy','after_set_const_0']:
                context &= eqarr(got[stage+'__'+k],np.broadcast_to(a.astype(np.float32),got[stage+'__'+k].shape).copy())
        rc=readnp(pathdir/'reader_context.npz')['reader_context']
        if reader_context is None:reader_context=rc
        context &= eqarr(rc,reader_context)
        model &= sha(pathdir/'nominal_model.mjb')=='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd'
    C('four_initial_fullstate_q_qd_ctrl_mocap_equals_archives',initial)
    C('all60_original_donor_actions_exact_A0_declared_zero',actionsok)
    C('all60_fresh_nominal_u_equals_archived_centers_bitexact',archive_u)
    C('donor_15interval_q_qd_continuity_no_reset',donor_cont)
    C('all60_reward_x_joint_fields_match_postintegration_raw',state_fields)
    C('16_snapshot_source_action_state_and_donor_targets_match',centerx)
    C('16_two_phase_donor_reward_terms_match',goalnext)
    C('actual8_both_readbacks_all4paths_and_reader46_exact',context)
    C('all4_nominal_MJB_SHA_same_frozen_model',model)
    details['archive_drift']=drift
    details['initialization_scope']='Archive full initial qpos/qvel/ctrl/mocap exact. Source state codec coordinate convention is inherited from frozen native SimReader. This audit independently validates raw joint blocks and temporal links, without rerunning native feature transforms or physics.'

passed=sum(checks.values());total=len(checks)
result={'schema':'current104_physical_v2_saved_independent_CPU_audit:1','status':'PASS_SAVED_PHYSICAL_ENGINEERING_ONLY' if passed==total and not exceptions else 'FAILED_REVIEW_HARNESS_OR_EVIDENCE',
        'engineering_qualified':passed==total and not exceptions,'passed':passed,'total':total,'checks':checks,'exceptions':exceptions,
        'audit_source':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'config':{'path':str(RUN/'config.json'),'sha256':sha(RUN/'config.json')},
        'physical_report':{'path':str(RUN/'physical_report.json'),'sha256':sha(RUN/'physical_report.json')},'details':details,
        'boundaries':['Saved-only NumPy/stdlib; no Torch/Warp/MuJoCo/Env/WM imports, physics or optimizer execution.',
                      'No gradient comparison, predictive label accuracy or full action-space Jacobian qualification follows from complete physical sampling.',
                      'Raw arrays and in-process runtime assertion receipts are distinguished. Python object graph identity and RNG restore are not independently replayed.',
                      'Archive trajectory drift does not regrade or replace prior physical outcomes. Native self15 is not hand-object supervision.',
                      'Old manifest config-path rejection remains separate. Only actual OS-launched config path is used here.']}
(OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':passed,'total':total,'failed':[k for k,v in checks.items() if not v],'exceptions':exceptions},indent=2))
sys.exit(0 if result['engineering_qualified'] else 1)
