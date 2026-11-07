"""Independent saved-only phase0 condition03 audit. NumPy/JSON/hash only."""
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
RUN=ROOT/'outputs/full151105_gradient_phase0_empirical_baseline03_v1'
WATCH=RUN.with_name(RUN.name+'_watchdog')
OUT=ROOT/'outputs/full151105_phase0_empirical_baseline_saved_review_v1'
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
def f32(v):return np.asarray(v,dtype=np.float32)
def fp32same(a,b):return same(f32(a),f32(b))
def stamp(v):return datetime.datetime.fromisoformat(v.replace('Z','+00:00'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def maxabs(a):return float(np.max(np.abs(a))) if np.size(a) else 0.

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

def terms64(x,p,R):
    x=np.asarray(x,float);p=np.asarray(p,float);R=np.asarray(R,float)
    a=x[40:43];r=x[43:52].reshape(3,3)
    endpoint=(np.linalg.norm(a+.1*r[:,0]-p-.1*R[:,0])+np.linalg.norm(a-.1*r[:,0]-p+.1*R[:,0]))*.5
    return np.array([2*np.exp(-(np.linalg.norm(a-p)/.05)**2),8*np.exp(-(endpoint/.04)**2)])

def quatmat(q):
    w,x,y,z=np.asarray(q,float)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def pose_terms64(pose,goal):
    p,R=np.asarray(pose,float)[:3],quatmat(np.asarray(pose,float)[3:])
    gp,gR=np.asarray(goal,float)[:3],quatmat(np.asarray(goal,float)[3:])
    x=np.r_[np.zeros(40),p,R.reshape(-1),np.zeros(6)]
    return terms64(x,gp,gR)

def main():
    OUT.mkdir(exist_ok=False);started=time.perf_counter();checks={};d={}
    def check(k,v):checks[k]=bool(v)
    c=read(RUN/'config.json');inv=read(RUN/'source/inventory.json');r=read(RUN/'physical_report.json')
    a=load(RUN/'physical_arrays.npz');rows=read(RUN/'branch_index.json');snaps=read(RUN/'snapshot_records.json')
    restores=read(RUN/'restore_receipts.json');gates=read(RUN/'baseline_gates.json');donors=read(RUN/'donor_records.json')
    rt=read(RUN/'runtime.json');watch=read(WATCH/'state.json');ident=read(WATCH/'engineering_identity.json');post=read(WATCH/'post_terminal_CIM.json')
    samples=lines(RUN/'physical_samples.jsonl');rewards=lines(RUN/'reward_snapshots.jsonl')
    directory=RUN/'donor_A0/snapshot_0_inventory';top=read(directory/'inventory_topology.json');desc=read(directory/'transaction_description.json');v=load(directory/'inventory_values.npz')
    check('source652_current_config_inventory_exact',len(inv)==652 and inv==c['source_pins'] and all(pin(p['path'])['sha256']==p['sha256'] for p in inv.values()))
    with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
        names=[f'{i:04d}_{Path(p["path"]).name}' for i,p in enumerate(inv.values())]
        check('source652_ZIP_members_bytes_exact',z.namelist()==names and all(hashlib.sha256(z.read(n)).hexdigest()==p['sha256'] for n,p in zip(names,inv.values())))
    check('worker792_transaction_f1bd_current_bound',top['worker_source']==pin(top['worker_source']['path']) and top['worker_source']['sha256']=='79207ad27252157036915360696442924176b22692464a3ea855c879656c104d' and top['transaction_source']==pin(top['transaction_source']['path']) and top['transaction_source']['sha256']=='f1bd0eafed58783a6efbb78a0ba40da086b36851119eddd00e29fcddb972a55c')
    check('actual_config_command_snapshot_SHA86151',pin(watch['command'][4])['sha256']==pin(RUN/'config.json')['sha256']==pin(WATCH/'config_snapshot.json')['sha256']==rt['plan_sha256']=='86151e82cd9f7007dc8607fda0d46d7bd6185247df21de0776aa71691e6ce0dd')
    check('terminal_completed_exit0_no_timeout',watch['status']=='completed' and watch['exit_code']==0 and watch['timeout'] is False and rt['status']=='completed_full151_gradient_probe' and (RUN/'completion.json').exists() and not (RUN/'failure.json').exists())
    check('OS12_actual_identity_all_true',ident['engineering_identity_qualified'] is True and len(ident['checks'])==12 and all(x is True for x in ident['checks'].values()))
    osr=watch['actual_process_OS_observation'];py=ident['selected_python_identity'];cmd=watch['command']
    check('OS_PID_parent_command_source_exact',osr['pid']==rt['pid']==watch['actual_pid']==31684 and osr['parent_pid']==rt['parent_pid']==watch['launcher_pid']==16656 and osr['command_line']==f'"{py["base_executable_raw"]}" '+' '.join(cmd[1:]) and os.path.samefile(osr['executable_path'],py['base_executable_raw']) and pin(cmd[2])['sha256']==rt['source_sha256'])
    check('OS_creation_bounds_owned_terminal_absence',stamp(watch['actual_creation_conservative_bounds']['lower_utc'])<=stamp(osr['creation_date_utc'])<=stamp(watch['actual_creation_conservative_bounds']['upper_utc']) and post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[])
    check('phase0_no_training_retry_or_FD_mode',c['probe_mode']=='engineering_restore_smoke' and c['automatic_retry'] is False and c['training_authorized'] is False and r['WM_forwards']==r['Actor_forwards']==r['Runner_instances']==r['model_loads']==r['optimizer_updates']==0)
    check('scope_fullserial_longclock_physical_false',r['full_serial_qualified'] is False and r['longclock_qualified'] is False and r['physical_reference_qualified'] is False and r['longclock_scope']=='PENDING_ENGINEERING_SMOKE_ONLY')
    check('producer_flags_are_declared_empirical_stage_only',r['status']=='COMPLETED_ENGINEERING_RESTORE_SMOKE' and all(r[k] is True for k in ('engineering_qualified','baseline_all_passed','capacity_finalized','capacity_qualified','foreign_inventory_qualified','restore_inventory_qualified')) and 'opaque' in r['fullstate_scope'])
    decision=read(ROOT/'outputs/full151105_phase0_empirical_exploration_decision_v1.json')
    check('decision_ceiling_hidden_state_and_gradient_false',decision['config']==pin(watch['command'][4]) and decision['foreign_review']==c['foreign_inventory_review'] and decision['planned_episode_substeps']==16 and decision['planned_baselines']==3 and decision['perturbation_branches']==0 and all(decision[k] is False for k in ('hidden_graph_workspace_all_bits_restored','exhaustive_hidden_workspace_write_before_read_proof','complete_native_state_certificate_granted','physical_gradient_qualified','full152_authorized_by_this_decision')))
    foreign=read(c['foreign_inventory_review']['path'])
    check('foreign_review_pin_198_exact_path_type_set',pin(c['foreign_inventory_review']['path'])==c['foreign_inventory_review'] and len(foreign['allowed_identity_only'])==198 and {(i['path'],i['type']) for i in foreign['allowed_identity_only']}=={(i['path'],i['type']) for i in desc['opaque_identity_only']} and len({i['path'] for i in foreign['allowed_identity_only']})==198)
    check('foreign68_current_pins_scope_and_no_full_qualification',len(foreign['source_pins'])==68 and all(pin(p['path'])==p for p in foreign['source_pins'].values()) and foreign['worker']==top['worker_source'] and foreign['transaction']==top['transaction_source'] and foreign['input_bundle']==c['input_bundle'] and foreign['approved_probe_modes']==['engineering_restore_smoke'] and foreign['allowed_snapshot_labels']==['A0/0'] and foreign['phase0_only'] is True and all(foreign[k] is False for k in ('whole_graph_hidden_state_qualified','native_restore_qualified','physical_gradient_qualified','GPU_run_authorized')))
    check('three_baseline_only152slots_retained',len(rows)==152 and [x['row_id'] for x in rows]==list(range(152)) and [x['row_id'] for x in rows if x['scheduled']]==[0,1,2] and all(x['type']=='baseline' and x['k']==0 and x['status']=='COMPLETED' for x in rows[:3]) and all(x['status']=='NOT_RUN' and x['restorequalified'] is False and x['actualu'] is None for x in rows[3:]))
    check('arrays_completed_inventory_masks_only0to2',all(np.array_equal(a[k],np.arange(152)<3) for k in ('completed','restore_inventory_bitexact','restore_qualified')) and np.array_equal(a['snapshot_completed'],np.arange(8)==0))
    check('unexecuted_branch_responses_NaN',all(np.isnan(a[k][3:]).all() for k in a if k.startswith('branch_')))
    check('unvisited_snapshot_donor_goals_NaN',all(np.isnan(a[k][1:]).all() for k in a if k.startswith(('snapshot_','donor_','goal_')) and k!='snapshot_completed'))
    check('snapshot0_and_remaining7_scopes',len(snaps)==8 and snaps[0]['k']==0 and snaps[0]['source_frame']==200 and snaps[0]['status']=='BASELINE3_QUALIFIED' and snaps[0]['longclock_qualified'] is False and all(x['status']=='NOT_RUN' and x['scheduled'] is False for x in snaps[1:]))
    check('native_inventory865_574WP_459Torch_8sidecars',len(top['native_entries'])==desc['native_entries']==865 and desc['native_Warp_arrays']==574 and desc['native_Torch_tensors']==459 and desc['explicit_arrays']==8 and len(top['explicit_container_entries'])==desc['explicit_containers']==184 and len(top['closure_cells'])==desc['closure_cells']==44 and desc['local_generators']==0)
    archive_keys=[];byte_ok=[];empty_null=0;inventory_hasher=hashlib.sha256()
    for item in top['array_values']:
        key=item['key'];raw=v[key];meta=item['target_meta'];n=math.prod(meta[2]);archive_keys.append(key)
        valid=raw.dtype==np.uint8 and raw.ndim==1 and len(raw)==(n*itemsize(meta) if n else 0) and hashlib.sha256(raw.tobytes()).hexdigest()==item['value_sha256']
        if meta[0]=='warp' and meta[-1] is None:empty_null+=1;valid=valid and n==0 and len(raw)==0
        byte_ok.append(valid);inventory_hasher.update(raw.tobytes())
    check('all1041_array_bytes_dtype_shape_SHA_and_emptyNull',len(archive_keys)==len(set(archive_keys))==1041 and all(byte_ok) and empty_null==212)
    check('array_byte_category5744598_exact',sum(k.startswith('Warp_') for k in archive_keys)==574 and sum(k.startswith('Torch_') for k in archive_keys)==459 and sum(k.startswith('sidecar_') for k in archive_keys)==8)
    mem={k:np.frombuffer(v[item['key']].tobytes(),np.float32).reshape(item['meta'][2]) for k,item in top['PEN_memory'].items()}
    check('31_PEN_memory_phase0_EMA_zero',len(mem)==31 and all(np.isfinite(x).all() for x in mem.values()) and np.array_equal(mem['time'],[[0.]]) and np.array_equal(mem['end'],[[151.]]) and all(not mem[k].any() for k in ('c1','c2','a1','a2','encoder_bias')))
    check('integration319_FP32_sig16383_sourceaction_zero',v['INTEGRATION_FP32'].shape==(1,319) and v['INTEGRATION_FP32'].dtype==np.float32 and np.isfinite(v['INTEGRATION_FP32']).all() and snaps[0]['integration_size']==319 and snaps[0]['integration_sig']==16383 and not v['source_action_FP32'].any() and hashlib.sha256(v['source_action_FP32'].tobytes()).hexdigest()==desc['source_action_sha256'])
    rng=read(directory/'RNG_python_numpy.json')
    check('RNG_and_1077_inventory_members_accounted',rng['numpy_header'][0]=='MT19937' and v['RNG_numpy_MT'].shape==(624,) and len(rng['python'])==3 and set(v)==set(archive_keys)|{x['key'] for x in top['PEN_memory'].values()}|{'INTEGRATION_FP32','source_action_FP32','RNG_numpy_MT','torch_cpu_0','torch_cuda_0'} and len(v)==1077)
    check('snapshot_archive_pins_current_exact',all(pin(p['path'])==p for p in snaps[0]['inventory_archive'].values()))
    check('three_restore_receipts_source_assertions_and_archive_digest',len(restores)==3 and [x['branch_id'] for x in restores]==[0,1,2] and all(x['begin']['inventory']['bitexact'] is True and x['begin']['inventory']['mismatches']==[] and x['begin']['inventory']['value_inventory_sha256']==inventory_hasher.hexdigest() and x['begin']['source_action_sha256']==desc['source_action_sha256'] and x['begin']['native_restore_qualification']=='PENDING_WORKER_GATE' and x['transaction_events'][1]['body_completed'] is True for x in restores))
    donor=RUN/'donor_A0';initial=read(donor/'initial_readback.json');reset=read(donor/'runtime_reset_readback.json');inputs=c['donor_inputs']['A0']['input_pins'];old=load(inputs['raw_states.npz']['path'])
    check('donorA0_seven_input_pins_exact',len(inputs)==7 and all(pin(p['path'])==p for p in inputs.values()))
    check('initial_full_q_qd_ctrl_mocap_byte_exact_old0',initial['time']==0. and all(fp32same(initial[k],old[q][0]) for k,q in (('qpos','full_qpos'),('qvel','full_qvel'),('ctrl','ctrl_rad'),('mocap_pos','mocap_pos_w'),('mocap_quat','mocap_quat_wxyz'))))
    check('reset_prephase0_EMA_and_action_zero',reset['phase']==[0] and reset['time']==[0.] and not np.asarray(reset['EMA_residual']).any() and not np.asarray(reset['action']).any() and fp32same(mem['frozen_target'].reshape(20),initial['ctrl']))
    bundle=read(c['input_bundle']['path']);context=load(bundle['context']['path']);expected=load(donor/'expected_context_fields.npz');actual=load(donor/'actual_context_readbacks.npz')
    check('actual8_expected_two_readbacks_exact',set(expected)==set(FIELDS) and len(actual)==16 and all(same(expected[k],context[k][0]) and all(same(actual[s+'__'+k],np.broadcast_to(expected[k],actual[s+'__'+k].shape)) for s in ('after_copy','after_set_const_0')) for k in FIELDS))
    check('xi46_nominal_model_descriptor_exact',same(load(donor/'reader_context.npz')['reader_context'],context['reader_context'][:1]) and pin(donor/'nominal_model.mjb')['sha256']==bundle['model']['sha256']=='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd' and read(donor/'controller_descriptor.json')==read(inputs['controller_descriptor.json']['path']))
    check('16samples_4rewards_3restore_3gates_actual_counts',len(samples)==16 and len(rewards)==4 and len(gates)==3 and len(restores)==3 and r['scheduled_substeps']==r['episode_step_attempted']==r['episode_step_returned']==16 and r['control_interval_attempted']==r['control_interval_returned']==4 and r['completed_branch_count']==3 and r['completed_snapshot_count']==1)
    blocks=[samples[4*i:4*i+4] for i in range(4)];controlmeta=read(donor/'control_metadata.json');lo=f32(controlmeta['lower']);hi=f32(controlmeta['upper']);innerlo=lo+np.float32(.02);innerhi=hi-np.float32(.02)
    check('all_interval_phase0_4steps_fixed_held_center_C2domain',all([s['substep'] for s in block]==[1,2,3,4] and all(s['phase']==0 and s['source_frame']==200 and fp32same(s['ctrl'],a['snapshot_u'][0]) for s in block) for block in blocks) and np.all(a['snapshot_u'][0]>=innerlo) and np.all(a['snapshot_u'][0]<=innerhi) and all(fp32same(a['branch_u'][i],a['snapshot_u'][0]) for i in range(3)))
    check('donor_and_branch_rowIDs_correspond_raw_blocks',all(s['mode']=='donor' and s['branch_id'] is None for s in blocks[0]) and all(s['mode']=='baseline' and s['branch_id']==i for i,block in enumerate(blocks[1:]) for s in block) and [q['branch_id'] for q in rewards]==[None,0,1,2])
    clock=[np.float32(0.)]
    for _ in range(4):clock.append(np.float32(clock[-1]+np.float32(.005)))
    check('FP32clock0to20_each_interval_no_unplanned_progress',all(s['canonical_pre_tick']==j and s['canonical_post_tick']==j+1 and fp32same(s['pre_time'],clock[j]) and fp32same(s['post_time'],clock[j+1]) for block in blocks for j,s in enumerate(block)))
    check('prepost_raw_chain_and_initial_restore_observed',all(fp32same(block[0]['pre_qpos'],initial['qpos']) and fp32same(block[0]['pre_qvel'],initial['qvel']) and all(fp32same(block[j]['post_qpos'],block[j+1]['pre_qpos']) and fp32same(block[j]['post_qvel'],block[j+1]['pre_qvel']) for j in range(3)) for block in blocks))
    check('all3_complete_substep_states_and_controls_equal_truth',all(all(fp32same(block[j][k],blocks[0][j][k]) for j in range(4) for k in ('pre_qpos','pre_qvel','post_qpos','post_qvel','ctrl','actuator_force')) for block in blocks[1:]))
    errors=[]
    for i in range(3):
        delta=abs(a['branch_x_next'][i].astype(float)-a['donor_next_x'][0].astype(float));groups={'q':(0,20),'qd':(20,40),'p':(40,43),'R':(43,52),'v':(52,55),'omega':(55,58)}
        e={k:maxabs(delta[x:y]) for k,(x,y) in groups.items()};e['reward']=abs(float(a['branch_object_score_k'][i])-float(a['donor_object_score_k'][0]));errors.append(e)
    check('three_baseline7errors_independently_all_zero',all(all(x==0. for x in e.values()) for e in errors) and all(gates[i]['errors']==errors[i] and gates[i]['passed'] is True and restores[i]['baseline_gate']['errors']==errors[i] for i in range(3)))
    check('saved58states_and_rewardrows_match_arrays_exact',all(fp32same(q['physics'],a['donor_next_x'][0] if i==0 else a['branch_x_next'][i-1]) and fp32same(q['contact'],a['donor_self15'][0] if i==0 else a['branch_self15'][i-1]) for i,q in enumerate(rewards)))
    check('rewardstate20_joint_q_qd_equals_postintegration',all(fp32same(q['physics'][:20],blocks[i][-1]['post_qpos'][7:]) and fp32same(q['physics'][20:40],blocks[i][-1]['post_qvel'][6:]) for i,q in enumerate(rewards)))
    smap=read(donor/'sensor_label_mapping.json');label=[]
    for q in rewards:
        found=np.asarray(q['raw_found'])>0;label.append(np.r_[found[0,smap['middle']],found[0,smap['distal']]|found[0,smap['tips']]].astype(np.float32))
    check('self15_from_saved_found_not_HO',all(fp32same(q['contact'],x) and np.isin(x,[0.,1.]).all() for q,x in zip(rewards,label)) and all(q['label_time']=='lastsolver_preintegration~kdt+.015' and q['state_time']=='postintegration_kdt+.020' for q in rewards))
    recomputed=[];score_err=[];native_diag=[]
    for i,q in enumerate(rewards):
        t=terms64(q['physics'],a['goal_p'][0],a['goal_R'][0]);bound=8*np.finfo(np.float32).eps*np.maximum(1.,abs(t));primary=f32(q['matched_post20_GPU_terms_FP32']);recomputed.append(t);score_err.append(abs(primary.astype(float)-t));native_diag.append(pose_terms64(q['native15_current_pose_wristlocal'][0],q['native15_goal_pose_wristlocal'][0]))
        check('same_x20_F64_primary_term_bound_row'+str(i),np.all(abs(primary.astype(float)-t)<=bound) and np.array_equal(np.asarray(q['F64_object_rate_terms_diagnostic']),t) and np.array_equal(np.asarray(q['F64_arithmetic_bound']),bound))
        check('primary_FP32_subset_sum_row'+str(i),fp32same(np.sum(primary,dtype=np.float32),q['matched_post20_GPU_score_FP32']) and q['scalar_contract']==r['scalar_contract'] and q['scalar_contract']['primary']=='matched_post20_gpu_score')
    check('GPU_primary_and_CPU_secondary_kept_distinct',all(fp32same(q['matched_post20_GPU_terms_FP32'],a['donor_object_terms_k'][0] if i==0 else a['branch_object_terms_k'][i-1]) and fp32same(q['CPU_FP32_object_rate_terms_secondary'],a['donor_object_terms_k_CPU_secondary'][0] if i==0 else a['branch_object_terms_k_CPU_secondary'][i-1]) for i,q in enumerate(rewards)))
    check('native15_secondary_rate_original_capture_metadata',all(q['native_GPU_capture_metadata']['compute_dt_s']==.02 and q['native_GPU_capture_metadata']['scale_by_dt'] is True and q['native_GPU_capture_metadata']['term_names']==['object_pos','object_rot'] and q['native_GPU_capture_metadata']['weights']==[2.,8.] and fp32same(np.sum(f32(q['native15_GPU_object_rate_terms_secondary']),dtype=np.float32),q['native15_GPU_object_rate_score_secondary']) for q in rewards))
    check('native15_pose_uses_lastsolver_prestate_not_x20',all(maxabs(np.asarray(q['native15_object_pose_derived_world'])[0,:3]-np.asarray(blocks[i][-1]['pre_qpos'])[:3])==0. and fp32same(q['native15_object_pose_derived_world'][0][3:],blocks[i][-1]['pre_qpos'][3:7]) for i,q in enumerate(rewards)))
    native_err=np.abs(np.asarray(native_diag)-np.asarray([q['native15_GPU_object_rate_terms_secondary'] for q in rewards]))
    check('native15_same_phase_secondary_formula_arithmetic_only',np.all(native_err<=8*np.finfo(np.float32).eps*np.maximum(1.,abs(np.asarray(native_diag)))))
    check('primary20_not_crosscompared_to_native15',maxabs(np.asarray([q['matched_post20_GPU_terms_FP32'] for q in rewards])-np.asarray([q['native15_GPU_object_rate_terms_secondary'] for q in rewards]))>1e-3 and all(q['capture']=='after_original_reward_compute_before_Manager_forward' and q['k']==0 and q['source_frame']==200 and q['terminal_observation']==1 for q in rewards))
    material=read(ROOT/'outputs/full151_actual_material_cone_contract_v1/material_cone_contract.json');topology=read(material['material_table']['path']);objects={g['geom_id'] for g in topology['rows'] if not g['name'].startswith('robot/')};hand={g['geom_id'] for g in topology['rows'] if g['name'].startswith('robot/') and g['enabled']}
    check('saved_passive_topology_pinned_21hand3object_no_query',pin(material['material_table']['path'])==material['material_table'] and len(objects)==3 and len(hand)==21 and material['inputs']['model']['sha256']==pin(donor/'nominal_model.mjb')['sha256'])
    forceerr=[];momenterr=[];decodeerr=[];torqueerr=[];phaseerr=[];contactcounts=[]
    netforces=[];netmoments=[]
    for s in samples:
        ho=[q for q in s['contacts'] if (q['geom'][0] in objects or q['geom'][1] in objects) and (q['geom'][0] in hand or q['geom'][1] in hand)]
        m=s['measured_hand_object'];F=[];T=[]
        for j,q in enumerate(ho):
            frame=np.asarray(q['frame']);f=np.asarray(q['force6']);sign=1 if q['geom'][1] in objects else -1
            F.append(sign*(f[:3]@frame));T.append(sign*(f[3:]@frame))
            address=q['efc_address'];lam=np.asarray(s['raw_efc_force'])[address:address+6];mu=np.asarray(m['friction'][j])
            decoded=np.r_[lam.sum(),mu[0]*(lam[0]-lam[1]),mu[1]*(lam[2]-lam[3]),mu[2]*(lam[4]-lam[5]),0.,0.]
            decodeerr.append(maxabs(f-decoded))
        F=np.asarray(F).reshape(-1,3);T=np.asarray(T).reshape(-1,3);pos=np.asarray([q['pos'] for q in ho]).reshape(-1,3);com=np.asarray(s['object_solver_COM']);netF=F.sum(0);netT=(np.cross(pos-com,F)+T).sum(0);netforces.append(netF);netmoments.append(netT)
        forceerr.append(maxabs(F-np.asarray(m['force_w_N']).reshape(-1,3)));torqueerr.append(maxabs(T-np.asarray(m['torque_at_contact_w_Nm']).reshape(-1,3)));momenterr.append(maxabs(netT-s['hand_object_moment_about_COM_w_Nm']))
        phaseerr.append(maxabs(com-np.asarray(s['pre_qpos'])[:3]));contactcounts.append(len(ho))
        check('raw_contact_WP_count_order_row'+str(len(contactcounts)-1),len(s['contacts'])==s['nacon']==len(s['raw_wp_contact']['geom']) and [q['geom'] for q in s['contacts']]==s['raw_wp_contact']['geom'] and [q['geom'] for q in ho]==m['geom_pair'] and len(s['raw_efc_force'])==s['nefc'] and s['direct_torque_included_once'] is True and fp32same(netF,s['hand_object_force_on_object_w_N']))
    check('all_HO_force_world_sign_and_direct_torque_exact',max(forceerr+[0.])==0. and max(torqueerr+[0.])==0.)
    check('all_COM_cross_moment_plus_direct_exact',max(momenterr+[0.])==0.)
    check('six_pyramid_lambdas_decode_localforce_F64_roundoff',max(decodeerr+[0.])<=1e-12)
    check('solver_COM_matches_preintegration_position_exact',max(phaseerr+[0.])==0. and all(s['phase_contract']=='solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward' for s in samples))
    check('no_exogenous_and_capacity_counts_below_allocations',all(s['external_force_max']==0. and s['nacon']<192 and s['nefc']<4096 for s in samples))
    check('reward_torque_matches_final_solver_force_FP32',all(fp32same(q['torque'],blocks[i][-1]['actuator_force']) for i,q in enumerate(rewards)))
    contact_replay_exact=all(block[j]['contacts']==blocks[0][j]['contacts'] and block[j]['measured_hand_object']==blocks[0][j]['measured_hand_object'] and block[j]['raw_efc_force']==blocks[0][j]['raw_efc_force'] for block in blocks[1:] for j in range(4))
    # Contact/force replay is diagnostic, never an added bit-exact baseline gate.
    ledger=read(RUN/'capacity_ledgers.json')[0]['ledger'];records=ledger['records'];events=[x['event'] for x in records]
    expected_events=['after_constructor_before_actual8','after_actual8_before_reset','before_reset']+['after_substep']*4+['donor_terminal']+sum((['before_restore']+['after_substep']*4+['after_branch'] for _ in range(3)),[])+['final']
    check('hostledger27events_read_before_each_restore_no_rollback',len(records)==ledger['record_count']==27 and events==expected_events and [x['sequence'] for x in records]==list(range(1,28)))
    check('capacity_flags27allzero_final_once_no_clear_unknown',ledger['status']=='OK' and ledger['engineering_capacity_qualified'] is True and ledger['cumulative_OR_bitmask']==ledger['unknown_read_count']==0 and ledger['final_read_count']==1 and ledger['before_reset_read_count']==1 and ledger['finalized'] is True and all(x['raw_flags_values']==[0] and x['raw_bitmask']==x['cumulative_OR_bitmask']==x['cumulative_unknown_reads']==0 and x['read_status']=='OK' and x['flags_cleared_by_observer'] is False for x in records))
    check('before_after_branch_capacity_receipts_link_external_ledger',all(x['begin']['capacity_before_restore']==records[8+i*6] and x['transaction_events'][1]['capacity_after_branch']==records[13+i*6] for i,x in enumerate(restores)) and desc['capacity_ledger_inside_snapshot'] is False)
    calls=r['backend_call_counts']
    check('episode_step16_forward4_public_returned_closed',calls['A0/donor']=={'Simulation.forward':1,'Simulation.step':4,'Simulation.step_returned':4,'Simulation.forward_returned':1} and calls['A0/baseline']=={'Simulation.forward':3,'Simulation.step':12,'Simulation.step_returned':12,'Simulation.forward_returned':3})
    check('constructor_graph_and_reset_calls_separate_not_episode',calls['A0/constructor']=={'CPU_mj_forward_API':1,'CPU_mj_forward_API_returned':1,'mjwarp.step_API':2,'mjwarp.step_API_returned':2,'mjwarp.forward_API':2,'mjwarp.forward_API_returned':2} and calls['A0/checked_reset']=={'Simulation.forward':2,'Simulation.forward_returned':2} and set(calls)=={'A0/constructor','A0/checked_reset','A0/donor','A0/baseline'})
    check('archive_drift_descriptive_not_old_quality_regrade',len(donors)==1 and donors[0]['fresh_state_count']==5 and donors[0]['archive_quality_not_regraded'] is True and all(all(v==0 for v in donors[0][name].values()) for name in ('archive_observed_horizon_drift','archive_selected_snapshot_drift','archive_selected_next_drift')))
    check('no_native_framework_model_solver_imports',not any(n.split('.')[0] in ('mujoco','mujoco_warp','mjlab','warp','torch','scipy','dex_supermimic') for n in sys.modules))
    d.update(source_count=len(inv),opaque_count=198,inventory_entries=865,WP_arrays=574,Torch_tensors=459,sidecar_arrays=8,empty_Warp_ptr_None=empty_null,archived_array_bytes=sum(len(v[k]) for k in archive_keys),inventory_digest=inventory_hasher.hexdigest(),baseline_errors=errors,primary_score=float(a['donor_object_score_k'][0]),native15_secondary_score=float(a['donor_native_reward_score_15ms_secondary'][0]),primary_F64_term_maxerror=maxabs(score_err),native15_F64_term_maxerror=maxabs(native_err),force_world_maxerror=max(forceerr),direct_torque_world_maxerror=max(torqueerr),COM_moment_maxerror=max(momenterr),pyramid_decode_maxerror=max(decodeerr),solver_COM_phase_maxerror=max(phaseerr),HO_contacts_per_substep=contactcounts,raw_contact_force_replay_exact_diagnostic_only=contact_replay_exact,capacity_records=len(records),episode_substeps=16,branch_count=3,constructor_reset_counts={k:calls[k] for k in ('A0/constructor','A0/checked_reset')},native_wall_s=r['native_wall_seconds_excluding_imports'],worker_wall_s=rt['wall_seconds'],watch_wall_s=watch['elapsed_s'])
    np.savez(OUT/'independent_numeric_arrays.npz',baseline_error_matrix=np.array([list(e.values()) for e in errors]),primary_terms_F64=np.asarray(recomputed),primary_terms_GPU=np.asarray([q['matched_post20_GPU_terms_FP32'] for q in rewards]),native15_terms_F64=np.asarray(native_diag),native15_terms_GPU=np.asarray([q['native15_GPU_object_rate_terms_secondary'] for q in rewards]),HO_netforce_w_N=np.asarray(netforces),HO_COMmoment_w_Nm=np.asarray(netmoments))
    evidence_paths=[RUN/n for n in ('config.json','source/inventory.json','source/execution_sources.zip','physical_report.json','physical_arrays.npz','branch_index.json','snapshot_records.json','restore_receipts.json','baseline_gates.json','physical_samples.jsonl','reward_snapshots.jsonl','capacity_ledgers.json')]+[directory/n for n in ('inventory_values.npz','inventory_topology.json','transaction_description.json')]+[WATCH/n for n in ('state.json','engineering_identity.json','post_terminal_CIM.json')]
    report={'schema':'full151105_phase0_empirical_baseline_saved_independent:1','status':'PASS_VISIBLE_INVENTORY_AND_PHASE0_SINGLE_INPUT_EMPIRICAL_REPEAT_ONLY' if all(checks.values()) else 'SAVED_AUDIT_FAILURE_RETAINED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':pin(__file__),'details':d,'inputs':{str(p.relative_to(ROOT)):pin(p) for p in evidence_paths},'foreign_review':c['foreign_inventory_review'],'decision':pin(ROOT/'outputs/full151105_phase0_empirical_exploration_decision_v1.json'),'numeric_arrays':pin(OUT/'independent_numeric_arrays.npz'),'qualification':{'visible_inventory_archive_checked':True,'single_input_phase0_empirical_baseline_repeated':all(all(x==0 for x in e.values()) for e in errors),'complete_hidden_graphstate_certificate':False,'physical_gradient_qualified':False,'full8snapshot_qualified':False,'longclock_qualified':False,'WM_or_derivative_comparison_performed':False},'limits':['Three repetitions share one phase0 input; they are empirical replay evidence, not independent states or universal native-state/hidden graph scratch restoration. The source-reviewed foreign198 handles include mutable graph execution caches and unknown hidden scratch.','Pointer identities, live-target inventory equality, PEN/RNG restoration and INTEGRATION set/get comparisons were performed by the pinned runtime. Saved clone bytes, metadata and receipt digests are independently rebound here; separate live before/after arrays were not saved, so this audit does not independently repeat those native comparisons.','Primary is the captured post20 GPU native-formula controlled2+8 score with fixed goal0. Actual RewardManager derived15 rate is a separate secondary; their nonzero difference is physical phase, not arithmetic noise.','Raw HO contact forces and COM moments refer to each preintegration solver; saved q/qvel/time refer to its postintegration state. Self15 is reconstructed from native raw_found at lastsolver15ms, not HO or human truth.','Archive agreement is descriptive and does not regrade previous A0/policy outcomes. No perturbation, new WM forward, actual derivative, model/Env/query/physics/GPU was invoked.','The producer stage flags remain subordinate to the separate Root decision ceiling; no full152/8snapshot, hidden-state certificate or gradient qualification is granted.'],'new_native_Env_model_WM_Actor_GPU':False,'wall_seconds':time.perf_counter()-started}
    report.update(empirical_phase0_baseline_passed=all(checks.values()) and all(all(x==0 for x in e.values()) for e in errors),probe_mode=c['probe_mode'],input_bundle=c['input_bundle'],episode_step_returned=r['episode_step_returned'],baseline_count=len(gates),physical_gradient_qualified=False,whole_hidden_state_qualified=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,x in checks.items() if not x],'report':pin(OUT/'report.json')},indent=2))
    return 0 if all(checks.values()) else 1

if __name__=='__main__':raise SystemExit(main())
