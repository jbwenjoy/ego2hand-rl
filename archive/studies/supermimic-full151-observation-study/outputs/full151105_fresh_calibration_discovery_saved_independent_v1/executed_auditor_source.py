"""Fresh24/74/124/148 discovery saved audit; NumPy/hash/JSON only.

No production import, model, geometry, score, force decode, world wrench or
physics calculation. Generic mature itemsize helpers are reused by exact pin.
Saved clone coverage and donor engineering never qualify opaque graph restore.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'scripts/audit_full151105_phase0_empirical_saved_cpu_v1.py'
HELPER_SHA='f2df7dbda2f2a8ff0826a612e881786550db0df0037602350c116207514d045f'
WORKER_SHA='1418b7d61bea7274da51be23593807472e7edd7a1720a8bbd25c98624cff133a'
ENTRY_SHA='52352b749318d662f08ed7fadef18fbd7e4d46e7c05e9d4f6afe4266218274bb'
PHASES=(24,74,124,148)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}


def checked(p):
    q=Path(p['path']).resolve(strict=True)
    if sha(q)!=p['sha256']:raise ValueError('Changed pin '+str(q))
    return q


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def lines(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]


def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def f32same(a,b):return bits(np.asarray(a,dtype='f4'),np.asarray(b,dtype='f4'))


def dump(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)


def helper():
    if sha(HELPER)!=HELPER_SHA:raise ValueError('Mature pure helper changed')
    s=importlib.util.spec_from_file_location('pure_inventory_bytes',HELPER);h=importlib.util.module_from_spec(s);s.loader.exec_module(h);return h


def byte_inventory(top,values,desc,h):
    keys=[];counts=collections.Counter();total=0;empty_null=0;valid=[]
    for item in top['array_values']:
        key=item['key'];z=values[key];meta=item['target_meta'];n=int(np.prod(meta[2]));keys.append(key);counts[meta[0]]+=1;total+=z.size
        ok=z.dtype==np.uint8 and z.ndim==1 and z.size==(n*h.itemsize(meta) if n else 0) and hashlib.sha256(z.tobytes()).hexdigest()==item['value_sha256']
        if meta[0]=='warp' and meta[-1] is None:empty_null+=1;ok=ok and n==0 and z.size==0
        valid.append(ok)
    expected=desc['native_Warp_arrays']+desc['native_Torch_tensors']+desc['explicit_arrays']
    return {'all_valid':all(valid) and len(keys)==len(set(keys))==expected and counts['warp']==desc['native_Warp_arrays'] and counts['torch']==desc['native_Torch_tensors'] and counts['numpy']==desc['explicit_arrays'],
      'keys':keys,'counts':dict(counts),'array_bytes_total':total,'empty_Warp_ptr_None':empty_null}


def audit(run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_saved_data.json',{'auditor':auditor,'pure_helper':pin(HELPER),'phases':list(PHASES),'actual_discovery_read':False,
      'scope':'1200 donor steps/8 captures/0restores only; no force/score/worldwrench arithmetic or opaque internal restore certificate.'})
    h=helper();checks={};stats=[];snapshotpins={};evidence={}
    def gate(k,v):checks[k]=bool(v)
    def keep(p):evidence[str(Path(p).resolve())]=pin(p)
    watchdir=run.with_name(run.name+'_watchdog');c=read(run/'config.json');r=read(run/'physical_report.json');rt=read(run/'runtime.json');inv=read(run/'source/inventory.json')
    watch=read(watchdir/'state.json');identity=read(watchdir/'engineering_identity.json');post=read(watchdir/'post_terminal_CIM.json');rootconfig=pin(watch['command'][4])
    a=load(run/'physical_arrays.npz');rows=read(run/'branch_index.json');snapshots=read(run/'snapshot_records.json');donors=read(run/'donor_records.json');ledgers=read(run/'capacity_ledgers.json')
    samples=lines(run/'physical_samples.jsonl');rewards=lines(run/'reward_snapshots.jsonl');bundle=read(checked(c['input_bundle']));context=load(checked(bundle['context']))
    gate('new_namespace_phases_discovery_only_no_recipe_or_old_review',c['schema']=='full151105_fresh_calibration_response_physical_run:1' and c['role']=='fresh_response_calibration'
       and c['probe_mode']=='inventory_discovery_all' and c['foreign_inventory_review'] is c['inventory_discovery_health'] is c['calibration_recipe'] is None
       and c['GPU_run_authorized'] is True and c['training_authorized'] is c['automatic_retry'] is c['compile_fallback'] is False
       and r['schema']=='full151105_calibration_response_worker_report:1' and r['role']==c['role'] and r['snapshot_phases']==list(PHASES))
    gate('root_actual_config_bytes_runtime_command_savedcopies',rootconfig['sha256']==pin(run/'config.json')['sha256']==pin(watchdir/'config_snapshot.json')['sha256']==rt['plan_sha256'] and Path(c['output']).samefile(run))
    gate('all_current_sourcepins_exact_inventory_entry_worker',inv==c['source_pins'] and all(checked(p) for p in inv.values()) and any(p['sha256']==WORKER_SHA for p in inv.values()) and rt['source_sha256']==ENTRY_SHA==sha(watch['command'][2]))
    with zipfile.ZipFile(run/'source/execution_sources.zip') as z:
        names=['%04d_%s'%(i,Path(p['path']).name) for i,p in enumerate(inv.values())]
        zipok=z.namelist()==names and all(hashlib.sha256(z.read(n)).hexdigest()==p['sha256'] for n,p in zip(names,inv.values()))
    gate('all_archived_source_bytes_exact_no_fixed_sourcecount',zipok)
    gate('actual_terminal_exit0_no_timeout_no_failure',watch['status']=='completed' and watch['exit_code']==0 and watch['timeout'] is False and rt['status']=='completed_full151_calibration_response_probe' and (run/'completion.json').is_file() and not(run/'failure.json').exists())
    gate('OS12_original_raw_identity_checks',identity['engineering_identity_qualified'] is True and len(identity['checks'])==12 and all(v is True for v in identity['checks'].values()))
    osr=watch['actual_process_OS_observation'];py=identity['selected_python_identity'];cmd=watch['command']
    gate('raw_PID_parent_executable_command_not_hardcoded',osr['pid']==rt['pid']==watch['actual_pid'] and osr['parent_pid']==rt['parent_pid']==watch['launcher_pid'] and os.path.samefile(osr['executable_path'],py['base_executable_raw']) and osr['command_line']=='"'+py['base_executable_raw']+'" '+' '.join(cmd[1:]))
    gate('raw_creation_bounds_postterminal_owned_CIM_empty',h.stamp(watch['actual_creation_conservative_bounds']['lower_utc'])<=h.stamp(osr['creation_date_utc'])<=h.stamp(watch['actual_creation_conservative_bounds']['upper_utc']) and post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[])
    flags=('baseline_all_passed','full_serial_qualified','foreign_inventory_qualified','restore_inventory_qualified','longclock_qualified','gradient_qualified','whole_graph_hidden_state_qualified','physical_reference_qualified')
    gate('completed_inventory_donor_only_not_restore_baseline_gradient',r['status']=='COMPLETED_INVENTORY_DISCOVERY' and r['engineering_qualified'] is r['inventory_discovery_only'] is True and all(r[k] is False for k in flags) and r['empirical_visible_state_baseline_scope']=='NOT_RUN_INVENTORY_ONLY')
    gate('zero_begin_finish_native_restore_actual_counters',all(r[k]==0 for k in ('transaction_begin_attempted','transaction_begin_returned','transaction_finish_attempted','transaction_finish_returned','native_snapshot_restore_attempted','native_snapshot_restore_returned','completed_branch_count','baseline_gate_count')) and read(run/'baseline_gates.json')==read(run/'restore_receipts.json')==[])
    gate('16integration_get_8samevalue_sets_not_restore',r['integration_get_attempted']==r['integration_get_returned']==16 and r['integration_set_attempted']==r['integration_set_returned']==8 and 'same-value' in r['state_write_scope'])
    gate('all152_slots_NOT_RUN_NaNs_and_false_restore_masks',len(rows)==152 and [v['row_id'] for v in rows]==list(range(152)) and all(v['scheduled'] is v['restorequalified'] is False and v['status']=='NOT_RUN' and v['actualu'] is v['r_k'] is None for v in rows) and all(not a[k].any() for k in ('completed','restore_inventory_bitexact','restore_qualified')) and all(np.isnan(v).all() for k,v in a.items() if k.startswith('branch_')))
    gate('1200actual_episode_300intervals_no_WM_Actor_Runner_optimizer',len(samples)==1200 and len(rewards)==300 and r['scheduled_substeps']==r['episode_step_attempted']==r['episode_step_returned']==1200 and r['control_interval_attempted']==r['control_interval_returned']==300 and all(r[k]==0 for k in ('optimizer_updates','WM_forwards','Actor_forwards','Runner_instances','model_loads')))
    gate('all8_capture_statuses_pending_no_foreign_qualification',len(snapshots)==8 and a['snapshot_completed'].dtype==bool and a['snapshot_completed'].shape==(8,) and a['snapshot_completed'].all() and r['completed_snapshot_count']==8 and all(s['status']=='REVIEW_PENDING' and s['scheduled'] is s['inventory_construction_completed'] is s['integration_roundtrip_bitexact'] is True and s['foreign_review'] is None and all(s[k] is False for k in ('foreign_review_qualified','full_native_restore_qualified','baseline3_qualified','longclock_qualified')) for s in snapshots))
    for di,name in enumerate(('A0','A2')):
        d=run/('donor_'+name);oldpins=c['donor_inputs'][name]['input_pins'];old=load(checked(oldpins['raw_states.npz']));old_events=read(checked(oldpins['control_events.json']));initial=read(d/'initial_readback.json');reset=read(d/'runtime_reset_readback.json');dr=donors[di]
        for p in oldpins.values():checked(p)
        ss=[x for x in samples if x['path']==name];rr=[x for x in rewards if x['path']==name];prefix=name+'__'
        expected=load(d/'expected_context_fields.npz');actual=load(d/'actual_context_readbacks.npz')
        gate(prefix+'actual8_twomeasuredcopies_and_physicalxi46',set(expected)==set(h.FIELDS) and len(actual)==16 and all(bits(expected[k],context[k][0]) and all(bits(actual[stage+'__'+k],np.broadcast_to(expected[k],actual[stage+'__'+k].shape)) for stage in ('after_copy','after_set_const_0')) for k in h.FIELDS) and bits(load(d/'reader_context.npz')['reader_context'],context['reader_context'][:1]) and sha(d/'nominal_model.mjb')==bundle['model']['sha256'])
        gate(prefix+'initial_fullstate_ctrl_mocap_and_resetexact',initial['time']==0. and all(f32same(initial[k],old[q][0]) for k,q in [('qpos','full_qpos'),('qvel','full_qvel'),('ctrl','ctrl_rad'),('mocap_pos','mocap_pos_w'),('mocap_quat','mocap_quat_wxyz')]) and reset['phase']==[0] and reset['time']==[0.] and not np.asarray(reset['EMA_residual']).any() and not np.asarray(reset['action']).any())
        q=np.vstack((initial['qpos'],[s['post_qpos'] for s in ss]));qd=np.vstack((initial['qvel'],[s['post_qvel'] for s in ss]));clock=[np.float32(0.)]
        for _ in range(600):clock.append(np.float32(clock[-1]+np.float32(.005)))
        gate(prefix+'601state_150events_600samples_150rewards_fullcadence',dr['path']==name and dr['fresh_state_count']==601 and len(dr['events'])==150 and len(ss)==600 and len(rr)==150 and [e['phase'] for e in dr['events']]==[e['k'] for e in rr]==list(range(150)))
        gate(prefix+'600contiguous_prepost_clockbits_phase_noassist',np.isfinite(q).all() and np.isfinite(qd).all() and all(f32same(s['pre_qpos'],q[i]) and f32same(s['pre_qvel'],qd[i]) and s['phase']==i//4 and s['substep']==i%4+1 and s['canonical_pre_tick']==i and s['canonical_post_tick']==i+1 and s['source_frame']==200+i//4 and f32same(s['pre_time'],clock[i]) and f32same(s['post_time'],clock[i+1]) and abs(s['post_time']-(i+1)*.005)<=5e-5 and s['mode']=='donor' and s['branch_id'] is None and s['external_force_max']==0. for i,s in enumerate(ss)))
        actions=np.zeros((150,1,20),dtype='f4') if name=='A0' else np.asarray([e['action'] for e in old_events],dtype='f4').reshape(150,1,20)
        actualu=np.asarray([e['actualu'] for e in dr['events']],dtype='f4');originalu=np.asarray([e['actualu'] for e in old_events],dtype='f4')
        if originalu.shape!=(150,1,20):raise ValueError('Original donor target ABI150x1x20')
        gate(prefix+'fixed_source_action_150targets_held600_mocap_no_teacherforce',f32same(dr['actions'],actions) and actualu.shape==(150,20) and bits(actualu,originalu[:,0]) and all(f32same(s['ctrl'],actualu[i//4]) for i,s in enumerate(ss)) and all(f32same(e['rawstate']['mocap_pos'],initial['mocap_pos']) and f32same(e['rawstate']['mocap_quat'],initial['mocap_quat']) for e in dr['events']))
        gate(prefix+'terminal149_to150_no_extra_u150',all(e['terminal_observation']==e['k']+1 and e['no_u150'] is True for e in rr) and rr[-1]['k']==149 and rr[-1]['terminal_observation']==150 and rr[-1]['valid_next_policy_decision'] is False)
        ledger=ledgers[di]['ledger'];records=ledger['records'];events=['after_constructor_before_actual8','after_actual8_before_reset','before_reset']+['after_substep']*600+['donor_terminal','final']
        gate(prefix+'external_hostledger605_reads_OR0_unknown0_finalonce',ledgers[di]['path']==name and ledger['record_count']==len(records)==605 and [v['event'] for v in records]==events and [v['sequence'] for v in records]==list(range(1,606)) and all(v['read_status']=='OK' and v['flags_cleared_by_observer'] is False and v['raw_flags_values']==[0] and v['cumulative_OR_bitmask']==v['cumulative_unknown_reads']==0 for v in records) and ledger['engineering_capacity_qualified'] is True and ledger['finalized'] is True and ledger['final_read_count']==ledger['before_reset_read_count']==1 and ledger['cumulative_OR_bitmask']==ledger['unknown_read_count']==0)
        previous=np.zeros((1,20),dtype='f4');previous_rows=[]
        for k in range(150):previous_rows.append(previous.copy());previous=np.float32(.5)*np.clip(np.float32(.5)*np.clip(actions[k],-1.,1.),np.float32(-.6),np.float32(.6))+np.float32(.5)*previous
        for pi,k in enumerate(PHASES):
            sid=di*4+pi;s=snapshots[sid];label=name+'/'+str(k);base=d/('snapshot_'+str(k)+'_inventory');desc=read(base/'transaction_description.json');top=read(base/'inventory_topology.json');values=load(base/'inventory_values.npz');profile=read(d/('profile_'+str(k)+'.json'));v=byte_inventory(top,values,desc,h)
            snapshotpins[label]={'description':pin(base/'transaction_description.json'),'topology':pin(base/'inventory_topology.json')}
            gate(label+'__exactlabel_profile_newworker_tx_and_archivepins',s['snapshot_id']==sid and s['path']==name and s['k']==k and s['source_frame']==200+k and desc['label']==label and desc['profile']==profile and profile['phase']==k and profile['boundary']=='pre_action' and profile['format_version']==105 and profile['strict_native_types_checked'] is True and top['worker_source']['sha256']==WORKER_SHA and all(checked(x) for x in s['inventory_archive'].values()) and checked(top['worker_source']) and checked(top['transaction_source']))
            gate(label+'__all_array_SHA_dtype_shape_emptyNone_payloadcoverage',v['all_valid'] and len(top['native_entries'])==desc['native_entries'] and len(top['closure_cells'])==desc['closure_cells'] and len(top['explicit_container_entries'])==desc['explicit_containers'])
            memory={n:np.frombuffer(values[item['key']].tobytes(),np.float32).reshape(item['meta'][2]) for n,item in top['PEN_memory'].items()}
            gate(label+'__PEN31_phase_end_previouscontroller_sourceaction',len(memory)==31 and all(np.isfinite(x).all() for x in memory.values()) and np.array_equal(memory['time'],[[k]]) and np.array_equal(memory['end'],[[151]]) and f32same(memory['c1'],previous_rows[k]) and f32same(values['source_action_FP32'],actions[k]) and hashlib.sha256(values['source_action_FP32'].tobytes()).hexdigest()==desc['source_action_sha256'])
            extra={g['state_bytes_key'] for g in top['local_generators'] if 'state_bytes_key' in g};keys=set(v['keys'])|{x['key'] for x in top['PEN_memory'].values()}|extra|{'INTEGRATION_FP32','source_action_FP32','RNG_numpy_MT','torch_cpu_0','torch_cuda_0'};rng=read(base/'RNG_python_numpy.json')
            gate(label+'__all_RNG_keys_integration_samevalue_bytes_not_live_restore',set(values)==keys and values['RNG_numpy_MT'].dtype==np.uint32 and values['RNG_numpy_MT'].shape==(624,) and rng['numpy_header'][0]=='MT19937' and len(rng['python'])==3 and values['torch_cpu_0'].dtype==values['torch_cuda_0'].dtype==np.uint8 and s['integration_sig']==16383 and s['integration_size']==319 and values['INTEGRATION_FP32'].shape==(1,319) and values['INTEGRATION_FP32'].dtype==np.float32 and np.isfinite(values['INTEGRATION_FP32']).all())
            gate(label+'__opaque_unknown_source_review_pending_no_qualification',len({x['path'] for x in desc['opaque_identity_only']})==len(desc['opaque_identity_only']) and all(x=='PENDING' for x in desc['qualification'].values()) and desc['native_physics_qualification']=='NOT_GRANTED_BY_HELPER' and desc['capacity_ledger_inside_snapshot'] is False and s['foreign_review'] is None and s['foreign_review_status']=='REVIEW_PENDING')
            gate(label+'__snapshotx_u_exact_sourcephase_q_qd_control',f32same(a['snapshot_x'][sid,:20],q[k*4,7:]) and f32same(a['snapshot_x'][sid,20:40],qd[k*4,6:]) and f32same(a['snapshot_u'][sid],actualu[k]) and f32same(a['donor_next_x'][sid],rr[k]['physics']))
            stats.append({'label':label,'native_entries':desc['native_entries'],'Warp_arrays':desc['native_Warp_arrays'],'Torch_tensors':desc['native_Torch_tensors'],'sidecar_arrays':desc['explicit_arrays'],'opaque':len(desc['opaque_identity_only']),'closure_cells':desc['closure_cells'],**{key:v[key] for key in ('array_bytes_total','empty_Warp_ptr_None')}})
            for namefile in ('transaction_description.json','inventory_topology.json','inventory_values.npz','RNG_python_numpy.json'):keep(base/namefile)
    gate('source_postchecks_all_current_and_no_heavy_reviewer_import',all(checked(p) for p in inv.values()) and not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic','scipy')))
    for filename in ('config.json','physical_report.json','physical_arrays.npz','snapshot_records.json','branch_index.json','donor_records.json','capacity_ledgers.json','runtime.json'):keep(run/filename)
    for filename in ('state.json','engineering_identity.json','post_terminal_CIM.json'):keep(watchdir/filename)
    result={'schema':'full151105_fresh_calibration_discovery_saved_independent:1','status':'PASS_FRESH_DONOR_AND_VISIBLE_INVENTORY_ONLY' if all(checks.values()) else 'FAILED_FRESH_DISCOVERY_SAVED_REVIEW',
      'engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'pure_saved_helper':pin(HELPER),
      'probe_mode':'inventory_discovery_all','completed_inventory_snapshots':int(a['snapshot_completed'].sum()),'episode_step_returned':r['episode_step_returned'],'restored_branches':int(a['restore_qualified'].sum()),
      'config':rootconfig,'producer_report':pin(run/'physical_report.json'),'run_dir':str(run.resolve()),'input_bundle':c['input_bundle'],'source_inventory':pin(run/'source/inventory.json'),'inventory_snapshot_pins':snapshotpins,
      'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'restored_baseline_qualified':False,'source_count':len(inv),'inventory_statistics':stats,'evidence':evidence,
      'limits':['Visible clone byte hashes/types/shapes, donor controls/clock/actual8 and capture scope only. Live pointer comparisons and INTEGRATION set/get are pinned-runtime assertions, not independently re-executed.',
       'Opaque/closure foreign internal state remains UNKNOWN. Exact per-new-label source review is required; old phase inventory/qualification is not inherited.',
       'No restored baseline or directions exist. Every152 branch slot remains NOT_RUN. This is not held-out WM accuracy, unseen states/trajectories, native-gradient, or physical-task success.',
       'Constructor/reset/graph API work is separate from1200 episode returns. No new score/contact-force/pyramid/world-force/COM-moment calculation is performed here.'],
      'new_models_Env_MJB_FK_WM_FW_AD_physics_GPU':0,'new_world_force_COMmoment_or_score_calculations':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v]},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:raise SystemExit(audit(a.run_dir,a.output))
    except Exception as e:
        if a.output.exists() and not(a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'new_native_execution':False})
        raise
