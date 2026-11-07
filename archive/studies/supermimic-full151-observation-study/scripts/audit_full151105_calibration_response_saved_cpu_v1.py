"""SOURCE DRAFT: fresh-response ENGINEERING audit; no response/model assessment.

Default: JSON/hash-only terminal preflight. Explicit --run reads saved arrays to
check new-phase capture/restore/cadence/baseline/source evidence. Does NOT load a
model, convert full58, score perturbation response or decode forces. In particular,
branch PRE full58 was not captured: its qualification remains NOT_ESTABLISHED.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
WORKER=ROOT/'src/full151_gradient/serial_probe105_calibration_v1.py'
WORKER_SHA='1418b7d61bea7274da51be23593807472e7edd7a1720a8bbd25c98624cff133a'
ENTRY=ROOT/'scripts/execute_full151_calibration_response_v1.py'
ENTRY_SHA='52352b749318d662f08ed7fadef18fbd7e4d46e7c05e9d4f6afe4266218274bb'
HELPER=ROOT/'scripts/audit_full151105_phase0_empirical_saved_cpu_v1.py'
HELPER_SHA='f2df7dbda2f2a8ff0826a612e881786550db0df0037602350c116207514d045f'
SUPPORT=Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco_warp/_src/support.py')
SUPPORT_SHA='86ab21d4662c0e5ac1094585e4841e994f928187326af31e54ff9ecf76f4abfb'
PHASES=(24,74,124,148)
LABELS=tuple(name+'/'+str(k) for name in ('A0','A2') for k in PHASES)
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
GROUPS={'q':(0,20,2e-5),'qd':(20,40,2e-4),'p':(40,43,2e-5),
        'R':(43,52,2e-5),'v':(52,55,2e-4),'omega':(55,58,2e-4)}


def require(value,message):
    if not value:raise ValueError(message)


def pin(path):
    path=Path(path).resolve(strict=True);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(path),'sha256':h.hexdigest()}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked(value):
    require(type(value) is dict and set(value)=={'path','sha256'},'Exact file pin')
    p=Path(value['path']).resolve(strict=True)
    require(pin(p)['sha256']==value['sha256'],'Changed file: '+str(p));return p


def same_pin(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def no_backend():
    require(not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules),'Saved-only process imported backend')


def declaration():
    no_backend()
    return {'schema':'full151105_fresh_response_engineering_auditor_declaration:1',
            'status':'SOURCE_DRAFT_NOT_EXECUTED','phases':list(PHASES),'counts':{'snapshots':8,'branches':152,'baseline':24,'episode_steps':1808},
            'source_pins':{'auditor':pin(__file__),'pure_IO_byte_helper':pin(HELPER),'worker':pin(WORKER),'entry':pin(ENTRY),'serializer_layout':pin(SUPPORT)},
            'saved_arrays_opened':0,'new_models_converters_FK_physics_GPU':0,
            'branch_PRE_full58_qualification':'NOT_ESTABLISHED_NO_DIRECT_CAPTURE',
            'perturbation_response_or_model_metrics_computed':False}


def metadata_preflight(run_dir,output=None):
    no_backend();run=Path(run_dir).resolve(strict=True);watchdir=run.with_name(run.name+'_watchdog')
    c=read(run/'config.json');rt=read(run/'runtime.json');r=read(run/'physical_report.json')
    watch=read(watchdir/'state.json');identity=read(watchdir/'engineering_identity.json');post=read(watchdir/'post_terminal_CIM.json')
    require(c['schema']=='full151105_fresh_calibration_response_physical_run:1' and c['role']=='fresh_response_calibration' and c['probe_mode']=='serial_full','Only actual NEW serial_full responses')
    require(r['schema']=='full151105_calibration_response_worker_report:1' and r['role']=='fresh_response_calibration' and r['snapshot_phases']==list(PHASES),'New report/phase binding')
    require(pin(WORKER)['sha256']==WORKER_SHA and pin(ENTRY)['sha256']==ENTRY_SHA and pin(HELPER)['sha256']==HELPER_SHA and pin(SUPPORT)['sha256']==SUPPORT_SHA,'Frozen source drift')
    inv=read(run/'source/inventory.json');require(inv==c['source_pins'] and inv,'Actual source inventory/config')
    for v in inv.values():checked(v)
    def member(v):return any(same_pin(v,x) for x in inv.values())
    require(member(pin(WORKER)) and member(pin(ENTRY)) and member(pin(SUPPORT)),'New worker/entry/layout absent from archive')
    command=watch['command'];rootconfig=pin(command[4])
    require(rootconfig['sha256']==pin(run/'config.json')['sha256']==pin(watchdir/'config_snapshot.json')['sha256']==rt['plan_sha256'],'Actual OS config route/copies')
    require(Path(command[2]).samefile(ENTRY) and rt['source_sha256']==ENTRY_SHA,'Actual new entry identity')
    require(watch['status']=='completed' and watch['exit_code']==0 and watch['timeout'] is False and rt['status']=='completed_full151_calibration_response_probe','Terminal exit0 required')
    require(identity['engineering_identity_qualified'] is True and identity['checks'] and all(v is True for v in identity['checks'].values()),'Saved OS identity checks')
    require(post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[],'Saved terminal owned PID absence')
    completion=read(run/'completion.json')
    require(completion['schema']=='full151105_calibration_response_worker_completion:1' and completion['config_sha256']==rootconfig['sha256'] and completion['result']==r,'Completion/report identity')
    require(not (run/'failure.json').exists(),'Failure receipt present')
    protocol=read(checked(c['protocol']));recipe=checked(c['calibration_recipe'])
    require(protocol['snapshot_phases']==list(PHASES) and same_pin(protocol['calibration_recipe'],c['calibration_recipe']) and member(pin(recipe)),'Recipe frozen into physical run')
    for v in protocol['calibration_source_pins'].values():require(member(v),'Calibration code not frozen before response acquisition')
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and not out.exists(),'Fresh independent output')
    no_backend()
    return {'run':run,'watchdir':watchdir,'config':c,'runtime':rt,'producer':r,'inventory':inv,
            'watch':watch,'identity':identity,'post':post,'config_pin':rootconfig,'output':out,
            'status':'PASS_NEW_FRESH_RESPONSE_METADATA_ONLY','saved_arrays_opened':0}


def run_audit(run_dir,output):
    m=metadata_preflight(run_dir,output);out=m['output'];require(out is not None,'Output required');out.mkdir()
    start=time.perf_counter();run=m['run'];c=m['config'];r=m['producer'];checks={};details={};evidence={}
    def check(k,v):checks[k]=bool(v)
    def keep(p):evidence[str(Path(p))]=pin(p)
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        spec=importlib.util.spec_from_file_location('_frozen_saved_IO_only',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
        load=h.load;bits=h.fp32same;f32=h.f32
        a=load(run/'physical_arrays.npz');rows=read(run/'branch_index.json');snaps=read(run/'snapshot_records.json')
        donors=read(run/'donor_records.json');baselines=read(run/'baseline_gates.json');restores=read(run/'restore_receipts.json');ledgers=read(run/'capacity_ledgers.json')
        samples=h.lines(run/'physical_samples.jsonl');rewards=h.lines(run/'reward_snapshots.jsonl')
        with zipfile.ZipFile(run/'source/execution_sources.zip') as z:
            names=[f'{idx:04d}_{Path(v["path"]).name}' for idx,v in enumerate(m['inventory'].values())]
            check('current_archive_unique_members_bytes',z.namelist()==names and all(hashlib.sha256(z.read(name)).hexdigest()==v['sha256'] for name,v in zip(names,m['inventory'].values())))
        osr=m['watch']['actual_process_OS_observation'];py=m['identity']['selected_python_identity'];rt=m['runtime'];cmd=m['watch']['command']
        check('raw_OS_PID_parent_exe_command_creation',osr['pid']==rt['pid']==m['watch']['actual_pid']>0 and osr['parent_pid']==rt['parent_pid']==m['watch']['launcher_pid']>0
              and os.path.samefile(osr['executable_path'],py['base_executable_raw'])
              and osr['command_line']==f'"{py["base_executable_raw"]}" '+' '.join(cmd[1:])
              and h.stamp(m['watch']['actual_creation_conservative_bounds']['lower_utc'])<=h.stamp(osr['creation_date_utc'])<=h.stamp(m['watch']['actual_creation_conservative_bounds']['upper_utc']))
        check('new_complete_empirical_ceiling',r['status']=='COMPLETED_PHYSICAL_PROBE' and all(r[k] is True for k in ('engineering_qualified','baseline_all_passed','full_serial_qualified','foreign_inventory_qualified','restore_inventory_qualified','longclock_qualified','capacity_finalized','capacity_qualified'))
              and all(r[k] is False for k in ('gradient_qualified','whole_graph_hidden_state_qualified','physical_reference_qualified')))
        check('exact_episode_capture_restore_counts',r['episode_step_attempted']==r['episode_step_returned']==r['scheduled_substeps']==len(samples)==1808
              and r['control_interval_attempted']==r['control_interval_returned']==len(rewards)==452
              and all(r[k]==152 for k in ('transaction_begin_attempted','transaction_begin_returned','transaction_finish_attempted','transaction_finish_returned','native_snapshot_restore_attempted','native_snapshot_restore_returned','completed_branch_count'))
              and r['integration_get_attempted']==r['integration_get_returned']==168 and r['integration_set_attempted']==r['integration_set_returned']==8
              and all(r[k]==0 for k in ('optimizer_updates','WM_forwards','Actor_forwards','Runner_instances','model_loads')))
        check('all_original_slots_and_capture_masks',len(rows)==len(restores)==152 and len(snaps)==8 and len(baselines)==24 and len(donors)==len(ledgers)==2
              and [x['row_id'] for x in rows]==list(range(152)) and all(a[k].dtype==np.bool_ and a[k].all() for k in ('completed','snapshot_completed','restore_inventory_bitexact','restore_qualified')))
        check('saved_native_state58_control_and_self15_finite',all(a[k].dtype==np.float32 and a[k].shape==shape and np.isfinite(a[k]).all()
              for k,shape in (('snapshot_x',(8,58)),('snapshot_u',(8,20)),('donor_next_x',(8,58)),('branch_x_next',(152,58)),('branch_u',(152,20)))))
        check('saved_native_self15_binary',all(a[k].dtype==np.float32 and a[k].shape==shape and np.isin(a[k],[0,1]).all()
              for k,shape in (('donor_self15',(8,15)),('branch_self15',(152,15)))))
        bundle=read(checked(c['input_bundle']));context=load(checked(bundle['context']));control=load(checked(bundle['source_pins']['control_arrays']))
        foreign=read(checked(c['foreign_inventory_review']));discovery=read(checked(c['inventory_discovery_health']))
        check('fresh_exact8_foreign_discovery_provenance',foreign['schema']=='full151105_foreign_inventory_per_snapshot_source_review:2' and set(foreign['per_snapshot'])==set(LABELS)
              and foreign['allowed_snapshot_labels']==list(LABELS) and foreign['approved_probe_modes']==['serial_full']
              and same_pin(foreign['worker'],pin(WORKER)) and same_pin(foreign['input_bundle'],c['input_bundle'])
              and same_pin(foreign['source_inventory_saved_review'],c['inventory_discovery_health']) and discovery['episode_step_returned']==1200
              and discovery['completed_inventory_snapshots']==8 and discovery['restored_branches']==0
              and same_pin(foreign['source_inventory_config'],discovery['config']) and same_pin(foreign['source_inventory_report'],discovery['producer_report'])
              and Path(foreign['source_inventory_run']).samefile(discovery['run_dir']) and foreign['whole_graph_hidden_state_qualified'] is False)
        dc=read(checked(discovery['config']));dr=read(checked(discovery['producer_report']))
        check('new_discovery_positive_not_old1762_relabel',discovery['engineering_qualified'] is True and discovery['checks'] and all(v is True for v in discovery['checks'].values())
              and dc['schema']==c['schema'] and dc['role']==c['role'] and dc['probe_mode']=='inventory_discovery_all'
              and dr['schema']==r['schema'] and dr['snapshot_phases']==list(PHASES) and same_pin(dc['proposal'],c['proposal'])
              and any(same_pin(v,pin(WORKER)) for v in dc['source_pins'].values()))
        digests={};primitive={};donor_rewards={};inventory_pins={};ledger_by_path={x['path']:x['ledger'] for x in ledgers}
        for path_i,name in enumerate(('A0','A2')):
            d=run/('donor_'+name);initial=read(d/'initial_readback.json');reset=read(d/'runtime_reset_readback.json')
            src=c['donor_inputs'][name]['input_pins'];old=load(checked(src['raw_states.npz']));old_events=read(checked(src['control_events.json']))
            expected=load(d/'expected_context_fields.npz');actual=load(d/'actual_context_readbacks.npz');xi=load(d/'reader_context.npz')['reader_context'];meta=read(d/'control_metadata.json')
            check(name+'_actual8_double_read_xi_model',set(expected)==set(FIELDS) and len(actual)==2*len(FIELDS) and all(h.same(expected[k],context[k][0]) and all(h.same(actual[p+'__'+k],np.broadcast_to(expected[k],actual[p+'__'+k].shape)) for p in ('after_copy','after_set_const_0')) for k in FIELDS)
                  and h.same(xi,context['reader_context'][:1]) and pin(d/'nominal_model.mjb')['sha256']==bundle['model']['sha256'])
            check(name+'_full_nonzero_init_and_reset',initial['time']==0. and all(bits(initial[k],old[q][0]) for k,q in (('qpos','full_qpos'),('qvel','full_qvel'),('ctrl','ctrl_rad'),('mocap_pos','mocap_pos_w'),('mocap_quat','mocap_quat_wxyz')))
                  and reset['phase']==[0] and reset['time']==[0.] and not np.asarray(reset['EMA_residual']).any() and not np.asarray(reset['action']).any())
            ss=[s for s in samples if s['path']==name and s['mode']=='donor'];rr=[v for v in rewards if v['path']==name and v['mode']=='donor'];record=donors[path_i]
            require(len(ss)==600 and len(rr)==150,'Complete source donor primitives')
            check(name+'_150_donor_interval_phase_terminal_boundary',[z['k'] for z in rr]==list(range(150))
                  and all(z['terminal_observation']==k+1 and z['valid_next_policy_decision']==(k+1<150) and z['no_u150'] is True for k,z in enumerate(rr)))
            q=np.vstack((initial['qpos'],[s['post_qpos'] for s in ss]));v=np.vstack((initial['qvel'],[s['post_qvel'] for s in ss]));clock=[np.float32(0.)]
            for _ in range(600):clock.append(np.float32(clock[-1]+np.float32(.005)))
            primitive[name]=(q,v,np.asarray(clock));donor_rewards[name]=rr
            check(name+'_donor_primitive_contiguity_clock',all(bits(s['pre_qpos'],q[z]) and bits(s['pre_qvel'],v[z]) and bits(s['pre_time'],clock[z]) and bits(s['post_time'],clock[z+1])
                  and s['phase']==z//4 and s['substep']==z%4+1 and s['branch_id'] is None and s['canonical_pre_tick']==z and s['canonical_post_tick']==z+1 for z,s in enumerate(ss)))
            actions=np.zeros((150,1,20),np.float32) if name=='A0' else f32([e['action'] for e in old_events]).reshape(150,1,20)
            check(name+'_fixed_source_actions',bits(record['actions'],actions) and len(record['events'])==150 and len(record['actions'])==150)
            previous=np.zeros((1,20),np.float32);u=[];previous_rows=[];lo=f32(meta['lower']);hi=f32(meta['upper']);bias=f32(meta['encoder_bias'])
            for k in range(150):
                previous_rows.append(previous.copy());previous=np.float32(.5)*np.clip(np.float32(.5)*np.clip(actions[k],-1.,1.),np.float32(-.6),np.float32(.6))+np.float32(.5)*previous
                target=np.minimum(np.maximum(control['u_postC2_150_FP32_rad'][k][None]+previous-bias,lo),hi)
                u.append(np.minimum(np.maximum(target,lo+np.float32(.02)),hi-np.float32(.02))[0])
            u=np.asarray(u)
            check(name+'_native_EMA_C2_held_ctrl_and_events',all(bits(e['actualu'],u[k]) and bits(e['x'],rr[k]['physics']) and bits(e['contact'],rr[k]['contact']) for k,e in enumerate(record['events']))
                  and all(bits(s['ctrl'],u[z//4]) for z,s in enumerate(ss)))
            smap=read(d/'sensor_label_mapping.json');native_labels=[]
            for k,z in enumerate(rr):
                found=np.asarray(z['raw_found'])>0
                contact=np.r_[found[0,smap['middle']],found[0,smap['distal']]|found[0,smap['tips']]].astype(np.float32)
                native=f32(z['native15_GPU_object_rate_terms_secondary']);primary=f32(z['matched_post20_GPU_terms_FP32']);last=ss[4*k+3]
                native_labels.append(bits(contact,z['contact']) and bits(z['physics'][:20],last['post_qpos'][7:]) and bits(z['physics'][20:40],last['post_qvel'][6:])
                    and np.isfinite(f32(z['physics'])).all() and bits(z['torque'],last['actuator_force'])
                    and bits(np.sum(native,dtype=np.float32),z['native15_GPU_object_rate_score_secondary']) and bits(np.sum(primary,dtype=np.float32),z['matched_post20_GPU_score_FP32'])
                    and z['label_time']=='lastsolver_preintegration~kdt+.015' and z['state_time']=='postintegration_kdt+.020'
                    and z['native_GPU_capture_metadata']['compute_dt_s']==.02 and z['native_GPU_capture_metadata']['scale_by_dt'] is True)
            check(name+'_all_donor_GPU_capture_primitive_and_native_label_bind',all(native_labels))
            ledger=ledger_by_path[name];records=ledger['records'];local_branches=[x for x in rows if x['path']==name]
            events=['after_constructor_before_actual8','after_actual8_before_reset','before_reset']+['after_substep']*600+['donor_terminal']
            for _ in local_branches:events+=['before_restore']+['after_substep']*4+['after_branch']
            events+=['final']
            check(name+'_external_host_capacity_raw_OR_unknown_final',len(records)==ledger['record_count']==len(events) and [z['event'] for z in records]==events
                  and [z['sequence'] for z in records]==list(range(1,len(records)+1)) and ledger['finalized'] is True and ledger['status']=='OK'
                  and ledger['engineering_capacity_qualified'] is True and ledger['cumulative_OR_bitmask']==ledger['unknown_read_count']==0 and ledger['final_read_count']==ledger['before_reset_read_count']==1
                  and all(z['read_status']=='OK' and z['raw_flags_values']==[0] and z['raw_bitmask']==z['cumulative_OR_bitmask']==z['cumulative_unknown_reads']==0 and z['flags_cleared_by_observer'] is False for z in records))
            for phase_i,k in enumerate(PHASES):
                sid=path_i*4+phase_i;label=name+'/'+str(k);s=snaps[sid];base=d/('snapshot_'+str(k)+'_inventory')
                desc=read(base/'transaction_description.json');top=read(base/'inventory_topology.json');values=load(base/'inventory_values.npz');profile=read(d/('profile_'+str(k)+'.json'))
                inventory_pins[label]={z:pin(base/f) for z,f in (('description','transaction_description.json'),('topology','inventory_topology.json'),('values','inventory_values.npz'))}
                allowed=foreign['per_snapshot'][label]['allowed_identity_only']
                check(label+'_actual_profile_review_and_source',s['snapshot_id']==sid and s['k']==k and s['source_frame']==200+k and s['status']=='BASELINE3_QUALIFIED'
                      and s['integration_roundtrip_bitexact'] is True and s['baseline3_qualified'] is True and s['full_native_restore_qualified'] is False
                      and desc['label']==label and desc['profile']==profile and profile['phase']==k and profile['boundary']=='pre_action' and profile['format_version']==105 and profile['strict_native_types_checked'] is True
                      and same_pin(top['worker_source'],pin(WORKER)) and all(pin(z['path'])['sha256']==z['sha256'] for z in s['inventory_archive'].values())
                      and len(allowed)==len(desc['opaque_identity_only']) and {(z['path'],z['type']) for z in allowed}=={(z['path'],z['type']) for z in desc['opaque_identity_only']})
                keys=[];valid=[]
                for item in top['array_values']:
                    z=values[item['key']];md=item['target_meta'];n=math.prod(md[2]);keys.append(item['key'])
                    valid.append(z.dtype==np.uint8 and z.ndim==1 and z.size==(n*h.itemsize(md) if n else 0) and hashlib.sha256(z.tobytes()).hexdigest()==item['value_sha256'])
                check(label+'_visited_array_bytes_dynamic_counts',all(valid) and len(keys)==len(set(keys))==desc['native_Warp_arrays']+desc['native_Torch_tensors']+desc['explicit_arrays']
                      and len(top['native_entries'])==desc['native_entries'] and len(top['closure_cells'])==desc['closure_cells'] and len(top['explicit_container_entries'])==desc['explicit_containers'])
                digests[label]=hashlib.sha256(b''.join(values[z].tobytes() for z in keys)).hexdigest()
                mem={z:np.frombuffer(values[item['key']].tobytes(),np.float32).reshape(item['meta'][2]) for z,item in top['PEN_memory'].items()}
                check(label+'_PEN_phase_residual_sourceaction',np.array_equal(mem['time'],[[k]]) and np.array_equal(mem['end'],[[151]]) and bits(mem['c1'],previous_rows[k])
                      and bits(values['source_action_FP32'],actions[k]) and hashlib.sha256(values['source_action_FP32'].tobytes()).hexdigest()==desc['source_action_sha256'])
                integ=values['INTEGRATION_FP32'];nq=q.shape[1];nv=v.shape[1]
                # Source-pinned support.get_state: TIME then QPOS then QVEL.
                check(label+'_INTEGRATION_primitive_prefix_snapshot_native40',s['integration_sig']==16383 and integ.dtype==np.float32 and integ.shape==(1,s['integration_size']) and np.isfinite(integ).all()
                      and bits(integ[0,0],clock[4*k]) and bits(integ[0,1:1+nq],q[4*k]) and bits(integ[0,1+nq:1+nq+nv],v[4*k])
                      and bits(a['snapshot_x'][sid,:20],q[4*k,7:]) and bits(a['snapshot_x'][sid,20:40],v[4*k,6:])
                      and bits(a['snapshot_u'][sid],u[k]) and bits(a['donor_next_x'][sid],rr[k]['physics']) and desc['capacity_ledger_inside_snapshot'] is False)
                for f in ('transaction_description.json','inventory_topology.json','inventory_values.npz','RNG_python_numpy.json'):keep(base/f)
            for f in ('initial_readback.json','runtime_reset_readback.json','actual_context_readbacks.npz','expected_context_fields.npz','reader_context.npz'):keep(d/f)
        bybranch={v['branch_id']:v for v in rewards if v['branch_id'] is not None};base_errors=[]
        check('152_unique_branch_rewards',set(bybranch)==set(range(152)))
        for bid,row in enumerate(rows):
            sid=bid//19;off=bid%19;name=('A0','A2')[sid//4];k=PHASES[sid%4];label=name+'/'+str(k);q,v,clock=primitive[name]
            ss=[s for s in samples if s['branch_id']==bid];reward=bybranch[bid];receipt=restores[bid];baseline=off<3
            if baseline:expected_u=a['snapshot_u'][sid];schedule=row['type']=='baseline' and row['baseline_repeat']==off and row['sign']==0
            else:
                z=off-3;ei=z//8;j=(z%8)//2;sign=(-1,1)[z%2]
                expected_u=a['direction__plus' if sign==1 else 'direction__minus'][sid,j,ei]
                schedule=row['type']=='direction' and row['j']==j and row['joint_index']==(0,4,8,12)[j] and row['epsilon_index']==ei and row['epsilon_rad']==(.002,.001)[ei] and row['sign']==sign
            check('branch%d_schedule_ctrl_clock_primitive'%bid,schedule and row['row_id']==bid and row['snapshot_id']==sid and row['path']==name and row['k']==k and row['status']=='COMPLETED' and row['restorequalified'] is True
                  and bits(row['actualu'],expected_u) and bits(a['branch_u'][bid],expected_u) and len(ss)==4 and [s['substep'] for s in ss]==[1,2,3,4]
                  and all(s['path']==name and s['mode']==row['type'] and s['phase']==k and s['canonical_pre_tick']==4*k+z and s['canonical_post_tick']==4*k+z+1 and bits(s['ctrl'],expected_u)
                          and bits(s['pre_time'],clock[4*k+z]) and bits(s['post_time'],clock[4*k+z+1]) for z,s in enumerate(ss))
                  and bits(ss[0]['pre_qpos'],q[4*k]) and bits(ss[0]['pre_qvel'],v[4*k])
                  and all(bits(ss[z]['post_qpos'],ss[z+1]['pre_qpos']) and bits(ss[z]['post_qvel'],ss[z+1]['pre_qvel']) for z in range(3)))
            smap=read(run/('donor_'+name)/'sensor_label_mapping.json');found=np.asarray(reward['raw_found'])>0
            self15=np.r_[found[0,smap['middle']],found[0,smap['distal']]|found[0,smap['tips']]].astype(np.float32)
            terms=f32(reward['matched_post20_GPU_terms_FP32']);native=f32(reward['native15_GPU_object_rate_terms_secondary'])
            check('branch%d_direct_GPU_capture_labels_native15'%bid,bits(a['branch_x_next'][bid],reward['physics']) and bits(a['branch_x_next'][bid,:20],ss[-1]['post_qpos'][7:]) and bits(a['branch_x_next'][bid,20:40],ss[-1]['post_qvel'][6:])
                  and bits(self15,reward['contact']) and bits(self15,a['branch_self15'][bid]) and bits(reward['torque'],ss[-1]['actuator_force'])
                  and bits(terms,a['branch_object_terms_k'][bid]) and bits(np.sum(terms,dtype=np.float32),a['branch_object_score_k'][bid]) and float(a['branch_object_score_k'][bid])==row['r_k']==reward['matched_post20_GPU_score_FP32']
                  and bits(native,a['branch_native_reward_terms_15ms_secondary'][bid]) and bits(np.sum(native,dtype=np.float32),a['branch_native_reward_score_15ms_secondary'][bid])
                  and reward['label_time']=='lastsolver_preintegration~kdt+.015' and reward['state_time']=='postintegration_kdt+.020' and reward['terminal_observation']==k+1 and reward['valid_next_policy_decision']==(k+1<150) and reward['no_u150'] is True)
            begin=receipt['begin'];finish=receipt['transaction_events'][1];local=bid-(sid//4)*76;records=ledger_by_path[name]['records']
            check('branch%d_restore_receipt_inventory_externalcapacity'%bid,receipt['branch_id']==bid and begin['inventory_restored_bitexact'] is True and begin['inventory']['bitexact'] is True and begin['inventory']['mismatches']==[]
                  and begin['inventory']['value_inventory_sha256']==digests[label] and begin['capacity_before_restore']==records[604+local*6]
                  and finish['body_completed'] is True and finish['capacity_qualified'] is True and finish['capacity_after_branch']==records[609+local*6])
            if baseline:
                error=np.abs(a['branch_x_next'][bid].astype(float)-a['donor_next_x'][sid].astype(float))
                errs={z:float(error[lo:hi].max()) for z,(lo,hi,tol) in GROUPS.items()};errs['reward']=abs(float(a['branch_object_score_k'][bid])-float(a['donor_object_score_k'][sid]))
                passed=all(errs[z]<=tol for z,(_,_,tol) in GROUPS.items()) and errs['reward']<=2e-5
                old=baselines[len(base_errors)]
                check('baseline%d_original7_empirical_restore_gates'%bid,old['row_id']==bid and old['errors']==errs and old['passed'] is True and passed and receipt['baseline_gate']['errors']==errs)
                base_errors.append(errs)
        check('24_baselines_before_all_directions',len(base_errors)==24 and [z['row_id'] for z in baselines]==[s*19+i for s in range(8) for i in range(3)])
        check('all_saved_primitive_capture_fields_finite_noassist',all(np.isfinite(np.asarray(s[k])).all() for s in samples for k in ('pre_qpos','pre_qvel','post_qpos','post_qvel','ctrl','actuator_force')) and all(s['external_force_max']==0. for s in samples))
        no_backend()
        for f in ('config.json','source/inventory.json','source/execution_sources.zip','physical_report.json','physical_arrays.npz','snapshot_records.json','branch_index.json','donor_records.json','physical_samples.jsonl','reward_snapshots.jsonl','capacity_ledgers.json','restore_receipts.json','baseline_gates.json','runtime.json','completion.json'):keep(run/f)
        for f in ('state.json','engineering_identity.json','post_terminal_CIM.json'):keep(m['watchdir']/f)
        passed=all(checks.values())
        report={'schema':'full151105_fresh_response_engineering_saved_independent:1','status':'PASS_FRESH_RESPONSE_ENGINEERING_SEALED_NO_MODEL_METRICS' if passed else 'FAIL_FRESH_RESPONSE_ENGINEERING_RETAINED',
                'engineering_qualified':passed,'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':pin(__file__),'pure_IO_helper':pin(HELPER),'serializer_layout_source':pin(SUPPORT),
                'run_dir':str(run),'config':m['config_pin'],'producer_report':pin(run/'physical_report.json'),'input_bundle':c['input_bundle'],'calibration_recipe':c['calibration_recipe'],
                'probe_mode':'serial_full','snapshot_phases':list(PHASES),'completed_inventory_snapshots':8,'completed_branches':152,'baseline_count':24,'episode_step_returned':1808,
                'inventory_snapshot_pins':inventory_pins,'evidence':evidence,'baseline_all_passed':passed,'capacity_qualified':passed,
                'whole_hidden_state_qualified':False,'physical_gradient_qualified':False,'WM_accuracy_or_response_benefit_evaluated':False,
                'branch_PRE_full58_direct_capture':False,'branch_PRE_full58_recomputed':False,'branch_PRE_full58_qualified':False,
                'branch_PRE_scope':'Source-backed restored inventory/INTEGRATION plus exact full-qpos/qvel PRE primitives; snapshot native58 direct capture q/qd bound. No missing PRE58 reconstructed or certified.',
                'sealing_scope':'Engineering report and immutable hashes only, not access control. Raw branch endpoints remain in producer files for Root to seal externally; no perturbation response/secant/sign/model metrics computed here.',
                'source_lineage_N':1,'independent_ego_N':0,'new_models_MJB_FK_converters_physics_GPU':0,'elapsed_seconds':time.perf_counter()-start}
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return report
    except BaseException as error:
        failure={'schema':'full151105_fresh_response_engineering_audit_failure:1','error':repr(error),'auditor':pin(__file__),'checks_so_far':checks,'response_metrics_computed':False}
        try:(out/'audit_harness_failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
        except BaseException as save_error:
            try:error.add_note('Failure receipt write failed: '+repr(save_error))
            except BaseException:pass
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--run',action='store_true');a=parser.parse_args(argv)
    if a.run:
        if a.run_dir is None or a.output is None:parser.error('--run needs --run-dir and fresh --output')
        r=run_audit(a.run_dir,a.output);print(json.dumps({k:r[k] for k in ('status','passed','total','engineering_qualified','WM_accuracy_or_response_benefit_evaluated')}));return 0 if r['engineering_qualified'] else 1
    if a.run_dir is None:
        if a.output is not None:parser.error('--output needs --run-dir')
        print(json.dumps(declaration()));return 0
    m=metadata_preflight(a.run_dir,a.output);print(json.dumps({'status':m['status'],'config':m['config_pin'],'saved_arrays_opened':0,'models_physics_GPU':0}));return 0


if __name__=='__main__':raise SystemExit(main())
