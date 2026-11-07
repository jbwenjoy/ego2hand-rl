"""Saved discovery02 NumPy/JSON/hash audit; no native/framework imports."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
import zipfile
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151105_gradient_restore_inventory_discovery02_v1'
WATCH=RUN.with_name(RUN.name+'_watchdog')
OUT=ROOT/'outputs/full151105_native_inventory_discovery_saved_review_v1'
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==np.ascontiguousarray(b).tobytes()
def fp32same(a,b):return np.array_equal(np.asarray(a,np.float32).view(np.uint32),np.asarray(b,np.float32).view(np.uint32))
def stamp(v):return datetime.datetime.fromisoformat(v.replace('Z','+00:00'))


def itemsize(meta):
    kind,dtype=meta[:2]
    if kind=='numpy':return np.dtype(dtype).itemsize
    if kind=='torch':return {'torch.float32':4,'torch.int32':4,'torch.int64':8,'torch.bool':1}[dtype]
    name=dtype.split('.')[-1].split("'")[0]
    if name in ('int32','float32'):return 4
    if name=='bool':return 1
    if name=='spatial_vectorf':return 24
    if name=='quatf':return 16
    if name=='mat33f':return 36
    match=re.fullmatch(r'vec(\d+)[fi]',name)
    if match:return 4*int(match[1])
    raise ValueError('Unknown nonempty dtype: '+dtype)


def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter();checks={};details={};failures={}
    def check(name,value):checks[name]=bool(value)
    cfg=read(RUN/'config.json');inventory=read(RUN/'source/inventory.json');physical=read(RUN/'physical_report.json')
    arrays=load(RUN/'physical_arrays.npz');rows=read(RUN/'branch_index.json');snapshots=read(RUN/'snapshot_records.json')
    failure=read(RUN/'failure.json');runtime=read(RUN/'runtime.json');watch=read(WATCH/'state.json');identity=read(WATCH/'engineering_identity.json');post=read(WATCH/'post_terminal_CIM.json')
    donor=RUN/'donor_A0';directory=donor/'snapshot_0_inventory';top=read(directory/'inventory_topology.json');desc=read(directory/'transaction_description.json');values=load(directory/'inventory_values.npz')
    check('source583_config_inventory_current',len(inventory)==583 and inventory==cfg['source_pins'] and all(pin(v['path'])['sha256']==v['sha256'] for v in inventory.values()))
    with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
        names=[f'{i:04d}_{Path(v["path"]).name}' for i,v in enumerate(inventory.values())]
        check('source583_ZIP_members_bytes_exact',z.namelist()==names and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in zip(names,inventory.values())))
    check('executed_worker792_and_entrybe406_bound',pin(top['worker_source']['path'])==top['worker_source'] and top['worker_source']['sha256']=='79207ad27252157036915360696442924176b22692464A3ea855c879656c104d'.lower() and runtime['source_sha256']=='be406f3b928711ba6f8f31f4ab727563269fdbf2b3b62215437e783a1b455b0b')
    check('actual_config_command_snapshot_SHA',pin(watch['command'][4])['sha256']==pin(RUN/'config.json')['sha256']==pin(WATCH/'config_snapshot.json')['sha256']==runtime['plan_sha256']=='e5a8a6df5c5db9f480dc961b31b1f75a0a679bcbbf74b423db976372960ba383')
    check('expected_discovery_foreignNone_explicit_no_retry',cfg['foreign_inventory_review'] is None and cfg['probe_mode']=='engineering_restore_smoke' and cfg['automatic_retry'] is False and cfg['training_authorized'] is False and failure['no_retry'] is True)
    check('terminal_expected_exit1_no_timeout_not_success',watch['status']=='failed' and watch['exit_code']==1 and watch['timeout'] is False and runtime['status']=='failed_full151_gradient_probe' and physical['status']=='INVENTORY_REVIEW_REQUIRED' and 'InventoryReviewRequired' in failure['error'])
    bad=[k for k,v in identity['checks'].items() if not v]
    check('OS_identity11_true_exit0_gate_false_retained',len(identity['checks'])==12 and bad==['watchdog_completed_exit0_no_timeout'] and identity['engineering_identity_qualified'] is False)
    osrec=watch['actual_process_OS_observation'];py=identity['selected_python_identity'];cmd=watch['command']
    check('raw_actual_PID_parent_executable_command_binding',osrec['pid']==runtime['pid']==watch['actual_pid']==33788 and osrec['parent_pid']==runtime['parent_pid']==watch['launcher_pid']==35308 and osrec['command_line']==f'"{py["base_executable_raw"]}" '+' '.join(cmd[1:]) and os.path.samefile(osrec['executable_path'],py['base_executable_raw']) and os.path.samefile(py['base_executable_raw'],py['base_executable_resolved']) and pin(cmd[2])['sha256']==runtime['source_sha256'])
    check('OS_creation_bounds_and_postterminal_owned_absence',stamp(watch['actual_creation_conservative_bounds']['lower_utc'])<=stamp(osrec['creation_date_utc'])<=stamp(watch['actual_creation_conservative_bounds']['upper_utc']) and post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[])
    check('all_episode_and_completed_counts_zero',all(physical[k]==0 for k in ('scheduled_substeps','episode_step_attempted','episode_step_returned','control_interval_attempted','control_interval_returned','completed_branch_count','completed_snapshot_count','optimizer_updates','WM_forwards','Actor_forwards','Runner_instances')))
    check('all_global_qualification_flags_false',all(physical[k] is False for k in ('engineering_qualified','baseline_all_passed','capacity_finalized','capacity_qualified','full_serial_qualified','foreign_inventory_qualified','restore_inventory_qualified','longclock_qualified','physical_reference_qualified')) and not (RUN/'completion.json').exists())
    check('all_response_completed_restore_masks_false',all(arrays[k].dtype==np.bool_ and not arrays[k].any() for k in ('snapshot_completed','completed','restore_inventory_bitexact','restore_qualified')))
    response=[k for k in arrays if k.startswith('branch_') or k.startswith('donor_')]
    check('unexecuted_response_arrays_all_NaN',all(np.isnan(arrays[k]).all() for k in response))
    check('scheduled152slots_retained_no_branch_claim',len(rows)==152 and [v['row_id'] for v in rows]==list(range(152)) and sum(v['scheduled'] for v in rows)==3 and all(v['status']=='NOT_RUN' and v['restorequalified'] is False and v['actualu'] is None and v['r_k'] is None for v in rows))
    check('snapshot0_only_inventory_not_response_qualified',len(snapshots)==8 and snapshots[0]['snapshot_id']==0 and snapshots[0]['path']=='A0' and snapshots[0]['k']==0 and snapshots[0]['source_frame']==200 and snapshots[0]['inventory_construction_completed'] is True and snapshots[0]['integration_roundtrip_bitexact'] is True and snapshots[0]['status']=='INVENTORY_REVIEW_REQUIRED' and all(snapshots[0][k] is False for k in ('foreign_review_qualified','full_native_restore_qualified','baseline3_qualified','longclock_qualified')) and all(v['status']=='NOT_RUN' and v['scheduled'] is False for v in snapshots[1:]))
    check('no_baseline_restore_or_donor_results',all(read(RUN/name)==[] for name in ('baseline_gates.json','restore_receipts.json','donor_records.json')) and (RUN/'physical_samples.jsonl').stat().st_size==(RUN/'reward_snapshots.jsonl').stat().st_size==0)
    calls=physical['backend_call_counts'];construct=calls['A0/constructor'];reset=calls['A0/checked_reset']
    check('constructor_graph_APIs_not_global_zero_physics',construct=={'CPU_mj_forward_API':1,'CPU_mj_forward_API_returned':1,'mjwarp.step_API':2,'mjwarp.step_API_returned':2,'mjwarp.forward_API':2,'mjwarp.forward_API_returned':2} and reset=={'Simulation.forward':2,'Simulation.forward_returned':2} and set(calls)=={'A0/constructor','A0/checked_reset'})
    check('native_inventory_counts_and_schema',desc['schema']=='full151105_probe_transaction_inventory:1' and len(top['native_entries'])==desc['native_entries']==865 and desc['native_Warp_arrays']==574 and desc['native_Torch_tensors']==459 and desc['explicit_arrays']==8 and desc['explicit_containers']==len(top['explicit_container_entries'])==184 and desc['closure_cells']==len(top['closure_cells'])==44 and desc['local_generators']==len(top['local_generators'])==0)
    check('native_topology_unique_entries_and_positive_identities',[e['entry'] for e in top['native_entries']]==list(range(865)) and all(type(e['identity']) is int and e['identity']>0 and e['kind'] in ('object','dict','list') for e in top['native_entries']))
    check('opaque198_explicit_unique_paths_not_serialized',len(desc['opaque_identity_only'])==198 and len({v['path'] for v in desc['opaque_identity_only']})==198 and all(type(v['id']) is int and v['id']>0 for v in desc['opaque_identity_only']) and desc['native_physics_qualification']=='NOT_GRANTED_BY_HELPER' and all(v=='PENDING' for v in desc['qualification'].values()))
    archived=[];bytechecks=[];sizes=[];empty_null=0;unknown=[]
    for item in top['array_values']:
        key=item['key'];v=values[key];meta=item['target_meta'];shape=meta[2];n=math.prod(shape)
        valid=v.dtype==np.uint8 and v.ndim==1 and hashlib.sha256(v.tobytes()).hexdigest()==item['value_sha256']
        try:expected=0 if n==0 else n*itemsize(meta)
        except ValueError as e:unknown.append(str(e));expected=-1
        valid=valid and len(v)==expected
        if meta[0]=='warp' and meta[-1] is None:
            empty_null+=1;valid=valid and n==0 and len(v)==0
        bytechecks.append(valid);archived.append(key);sizes.append(len(v))
    check('all1041_WP_Torch_sidecar_bytes_SHA_dtype_shape',len(archived)==1041 and len(set(archived))==1041 and all(bytechecks) and not unknown)
    check('574WP459Torch8sidecar_key_coverage',sum(k.startswith('Warp_') for k in archived)==574 and sum(k.startswith('Torch_') for k in archived)==459 and sum(k.startswith('sidecar_') for k in archived)==8)
    check('empty_Warp_pointer_None_only_zero_elements',empty_null>0 and all(meta['target_meta'][-1] is not None or math.prod(meta['target_meta'][2])==0 for meta in top['array_values'] if meta['target_meta'][0]=='warp'))
    memory={}
    for key,item in top['PEN_memory'].items():
        meta=item['meta'];v=values[item['key']];dtype={'torch.float32':np.float32}[meta[1]]
        memory[key]=np.frombuffer(v.tobytes(),dtype=dtype).reshape(meta[2]).copy()
    check('31_PEN_memory_fields_finite_and_phase0',len(memory)==31 and all(np.isfinite(v).all() for v in memory.values()) and np.array_equal(memory['time'],[[0.]]) and np.array_equal(memory['start'],[[0.]]) and np.array_equal(memory['end'],[[151.]]) and all(not memory[k].any() for k in ('c1','c2','a1','a2','encoder_bias')))
    rng=read(directory/'RNG_python_numpy.json')
    check('RNG_archive_CPU_CUDA_numpy_python_present',rng['numpy_header'][0]=='MT19937' and values['RNG_numpy_MT'].shape==(624,) and values['RNG_numpy_MT'].dtype==np.uint32 and values['torch_cpu_0'].dtype==np.uint8 and values['torch_cuda_0'].dtype==np.uint8 and len(rng['python'])==3)
    check('integration_saved_sig_size_and_source_action',snapshots[0]['integration_sig']==16383 and snapshots[0]['integration_size']==319 and values['INTEGRATION_FP32'].shape==(1,319) and values['INTEGRATION_FP32'].dtype==np.float32 and np.isfinite(values['INTEGRATION_FP32']).all() and values['source_action_FP32'].shape==(1,20) and values['source_action_FP32'].dtype==np.float32 and not values['source_action_FP32'].any() and hashlib.sha256(values['source_action_FP32'].tobytes()).hexdigest()==desc['source_action_sha256'])
    expected_keys=set(archived)|{v['key'] for v in top['PEN_memory'].values()}|{'INTEGRATION_FP32','source_action_FP32','RNG_numpy_MT','torch_cpu_0','torch_cuda_0'}
    check('all1077_inventory_NPZ_members_accounted',set(values)==expected_keys and len(values)==1077)
    check('snapshot_inventory_archive_pins_exact',all(pin(v['path'])==v for v in snapshots[0]['inventory_archive'].values()) and top['worker_source']==pin(top['worker_source']['path']) and top['transaction_source']==pin(top['transaction_source']['path']))
    profile=read(donor/'profile_0.json');initial=read(donor/'initial_readback.json');reset_read=read(donor/'runtime_reset_readback.json')
    check('phase0_profile_native_shapes_not_surrogate',profile['phase']==0 and profile['boundary']=='pre_action' and profile['strict_native_types_checked'] is True and profile['CPU_surrogate_is_not_native_qualification'] is False and desc['profile']==profile)
    inputs=cfg['donor_inputs']['A0']['input_pins'];old=load(inputs['raw_states.npz']['path']);events=read(inputs['control_events.json']['path'])
    check('A0_input7_pins_current_exact',len(inputs)==7 and all(pin(v['path'])==v for v in inputs.values()))
    check('initial_q_qd_ctrl_mocap_full_archive_FP32_exact',initial['time']==0. and all(fp32same(initial[a],old[b][0]) for a,b in (('qpos','full_qpos'),('qvel','full_qvel'),('ctrl','ctrl_rad'),('mocap_pos','mocap_pos_w'),('mocap_quat','mocap_quat_wxyz'))))
    check('reset_phase0_no_residual_and_initial_sourcecenter',reset_read['phase']==[0] and reset_read['time']==[0.] and not np.asarray(reset_read['EMA_residual']).any() and not np.asarray(reset_read['action']).any() and fp32same(arrays['snapshot_u'][0],old['ctrl_rad'][0]) and fp32same(memory['frozen_target'],arrays['snapshot_u'][0][None]) and fp32same(arrays['direction__centers'][0],arrays['snapshot_u'][0]) and np.isfinite(arrays['snapshot_x'][0]).all() and np.isnan(arrays['snapshot_x'][1:]).all())
    bundle=read(cfg['input_bundle']['path']);context=load(bundle['context']['path']);expected=load(donor/'expected_context_fields.npz');actual=load(donor/'actual_context_readbacks.npz')
    check('actual8_expected_and_two_measured_readbacks_exact',set(expected)==set(FIELDS) and len(actual)==16 and all(same(expected[k],context[k][0]) and all(same(actual[stage+'__'+k],np.broadcast_to(expected[k],actual[stage+'__'+k].shape)) for stage in ('after_copy','after_set_const_0')) for k in FIELDS))
    xi=load(donor/'reader_context.npz')['reader_context'];check('actual_xi46_exact',same(xi,context['reader_context'][:1]))
    check('nominal_MJB_bytes_and_descriptor_exact_A0_source',pin(donor/'nominal_model.mjb')['sha256']==bundle['model']['sha256']=='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd' and read(donor/'controller_descriptor.json')==read(inputs['controller_descriptor.json']['path']))
    ledgers=read(RUN/'capacity_ledgers.json');ledger=ledgers[0]['ledger']
    check('external_capacity4records_finalized_OK_observation_only',len(ledgers)==1 and ledgers[0]['path']=='A0' and [v['event'] for v in ledger['records']]==['after_constructor_before_actual8','after_actual8_before_reset','before_reset','final'] and ledger['status']=='OK' and ledger['finalized'] is True and ledger['cumulative_OR_bitmask']==ledger['unknown_read_count']==0 and ledger['final_read_count']==1 and all(v['raw_flags_values']==[0] and v['read_status']=='OK' and v['read_attempts_for_record']==1 and v['flags_cleared_by_observer'] is False for v in ledger['records']))
    check('external_ledger_not_native_inventory_qualification',desc['capacity_ledger_inside_snapshot'] is False and desc['native_physics_qualification']=='NOT_GRANTED_BY_HELPER' and ledger['engineering_capacity_qualified'] is True and physical['capacity_qualified'] is False)
    check('no_native_framework_or_solver_import',not any(n.split('.')[0] in ('mujoco','mujoco_warp','mjlab','warp','torch','scipy','dex_supermimic') for n in sys.modules))
    details.update(inventory_entries=865,WP_arrays=574,Torch_tensors=459,sidecar_arrays=8,archive_byte_arrays=1041,archive_bytes_total=sum(sizes),empty_Warp_ptr_None=empty_null,opaque_identity_only=198,closures=44,
        original_OS_checks_true=11,original_OS_exit0_check_false=True,episode_steps=0,branches=0,constructor_graph_calls=physical['backend_call_counts'],
        native_wall_s=physical['native_wall_seconds_excluding_imports'],worker_wall_s=runtime['wall_seconds'],watch_wall_s=watch['elapsed_s'])
    report={'schema':'full151105_native_inventory_discovery_saved_independent:1','status':'PASS_SAVED_DISCOVERY_AUDIT_NO_RESTORE_OR_GRADIENT_QUALIFICATION' if all(checks.values()) else 'SAVED_AUDIT_FAILURE_RETAINED',
        'checks':checks,'passed':sum(checks.values()),'total':len(checks),'auditor':pin(__file__),'details':details,
        'evidence':{str(p.relative_to(ROOT)):pin(p) for p in (RUN/'config.json',RUN/'source/inventory.json',RUN/'source/execution_sources.zip',RUN/'physical_report.json',RUN/'physical_arrays.npz',RUN/'snapshot_records.json',directory/'inventory_values.npz',directory/'inventory_topology.json',directory/'transaction_description.json',WATCH/'state.json',WATCH/'engineering_identity.json',WATCH/'post_terminal_CIM.json')},
        'limits':['The discovery intentionally stops before any baseline/restore/perturbed branch. Original process engineering exit gate remains failed/exit1;11 otherOSidentity checks are not upgraded to12/12.',
        'Archived bytes/hashes/shapes are independently recomputed. Live device values andset/getroundtrip were checked insideexecutedsource; separatebefore/afterintegration arrays andliveinventorybytes are notboth serialized here, so thoseactualcomparisons are source-backedassertions, notindependentlyrerun.',
        'Opaque198 entries remainidentity-only, notserialized foreignCstate. Actualforeign/sourceclosure review andthreefreshnativebaselines remainPENDING. Parentplanmustmakeanewcondition, neverauto-retrythisdiscovery.',
        '0episode/0branches doesnotmean0globalbackendphysics: constructorCUDAgraph APIcaptures include2step/2forward calls, CPUforward1, resetpublicforward2, separatelyreported andnotaddedasdevicekernelcounts.',
        'No donor response, scalar reward, self15 label or derivative hasbeen measuredbythisstage; finitepreactionsnapshotanddirectionendpoints are inventory/input dataonly.'],
        'new_Env_model_WM_Actor_GPU':False,'new_native_queries_or_physics':0,'wall_seconds':time.perf_counter()-started}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')},indent=2))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
