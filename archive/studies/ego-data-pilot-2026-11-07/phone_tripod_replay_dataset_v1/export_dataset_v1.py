"""Configured saved-only export. Default JSON/hash only; never native/NN/FK."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('tripod_saved_adapter',H/'dataset_adapter_v1.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D)
need,read,pin,checked,write,bits,close=D.need,D.read,D.pin,D.checked,D.write,D.bits,D.close
FIELD_MAP=dict(x58='x58_F64',u20='u20_F64',y58='y58_F64',delta52='target_delta52_F64',memory83='memory83_F64',next_memory83='next_memory83_F64',
    obs222='obs222_FP32',next_obs222='next_obs222_FP32',action20='normalized_action20_F64',h='native_dt_s_F64',n='n_substeps_i64',executed_n='executed_substeps_i64',
    complete_H='complete_H_bool',WM_eligible='WM_eligible_bool',Actor_start_eligible='Actor_start_eligible_bool',actor_absorbing_done='actor_absorbing_done_bool',
    reference_timeout='reference_timeout_bool',acquisition_done='acquisition_done_bool',collection_truncated='acquisition_truncated_bool',
    row_id='row_id_i64',family_id='family_id_i64',episode_id='episode_id_i64',step_k='step_k_i64',impulse22x6='impulse_source_BODY_moving_COM22x6_F64',
    meanFn22='mean_positive_Fn22_N_F64',coverage22='Fn_gt_0p1_coverage22_F64',triple_coverage='simultaneous_triple_coverage_F64')

def metadata(path):
    p=read(path);need(p['schema']=='phone_tripod_replay_dataset_CPU_protocol:1' and type(p['saved_only_run_authorized']) is bool and p['native_NN_FK_authorized'] is False,'Saved dataset contract')
    for v in [*p['inputs'].values(),*p['sources'].values()]:checked(v)
    need(p['sources']['exporter']==pin(__file__) and p['sources']['adapter']==pin(H/'dataset_adapter_v1.py'),'Own source binding')
    r=read(checked(p['inputs']['collector_report']));need(r['schema']=='phone_tripod_task_acquisition_result:1' and r['status']=='COMPLETE_NEW_TRIPOD_TASK_CONTROLLED_COLLECTION','Complete native producer')
    need(r['families']==16 and r['episodes']==32 and r['rows']<=640 and r['partition_qualified'] is True and r['cross_partition_alias_groups']==[],'Complete family plan/no crosssplit alias')
    task=read(checked(r['task_bundle']));plan=read(checked(r['family_plan']));need(r['task_bundle']==p['inputs']['task_bundle'] and r['family_plan']==p['inputs']['family_plan'],'Task/plan lineage')
    need(task['task_designated_geom_ids']==[15,28,41] and r['tripod_designated_geom_ids']==[15,28,41] and r['canonical_pair_geom_ids']==list(D.PAIRS),'Explicit new3pad roles and canonical22')
    need([v['family_id'] for v in plan['families']]==list(range(16)) and [v['split'] for v in plan['families']]==['TRIPOD_TRAIN']*12+['TRIPOD_MONITOR']*4,'Prespecified family roles')
    term=read(checked(p['inputs']['collector_terminal']));need(term['returncode']==0,'True producer terminal exit0')
    need(term['protocol']==r['protocol'],'Terminal/producer protocol binding')
    parent=read(checked(r['protocol']));need(parent['inputs']['task_bundle']==r['task_bundle'],'Native source protocol current task')
    meta=read(checked(p['inputs']['geometry_metadata']));physical=read(checked(p['inputs']['physical_parameters']))
    need([v['other_geom_id'] for v in meta['pairs']]==list(D.PAIRS) and meta['model']==task['model'],'Compiled pair/model match')
    need(physical['mass_kg']==.2,'Explicit nominal context')
    close(np.array(physical['body_ipos_body_m']),np.array(meta['phone_body_ipos']),'Saved physical and compiled BODY COM offset')
    for audit in r['model_audits'].values():need(audit['passed'] is True and audit['model_pin']==task['model'],'Actual same fixed model across native dt')
    need(p['cross_split_alias_policy']=='BLOCK_QUALIFICATION_PRESERVE_ALL_ROWS','No silent alias split reassignment/filtering')
    return p,r,task,plan,meta,physical

def load_npz(v):
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(path,a):
    with Path(path).open('xb') as f:np.savez_compressed(f,**a)
def stack_rows(rows):
    need(rows,'Cannot invent shape-qualified arrays with no observed rows')
    return {k:np.asarray([r[k] for r in rows],dtype=np.bool_ if k.endswith('_bool') else np.int64 if k.endswith('_i64') else np.float32 if k.endswith('_FP32') else np.float64) for k in rows[0]}
def union_components(groups):
    parent=list(range(16))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for g in groups:
        ids=g['family_ids']
        for a in ids[1:]:parent[find(a)]=find(ids[0])
    components=defaultdict(list)
    for f in range(16):components[find(f)].append(f)
    return sorted(components.values(),key=lambda g:g[0])

def run(protocol_path):
    p,r,task,plan,meta,physical=metadata(protocol_path);need(p['saved_only_run_authorized'] is True,'Explicit saved arithmetic authorization')
    O=Path(p['output']);need(not O.exists(),'Fresh export output');O.mkdir();started=time.perf_counter()
    maths=D.load_math();fixture=D.analytic_tripod_semantic_fixture(maths);write(O/'semantic_fixture.json',fixture)
    identities=read(checked(r['sample_identities']));episodes=read(checked(r['episode_records']));need(len(identities)==r['rows'] and len(episodes)==32,'Full producer row/episode manifests')
    need([v['sample_id'] for v in identities]==list(range(len(identities))),'All original row indices/order')
    byid={v['sample_id']:v for v in identities};rows={};notes={};aliases=defaultdict(list);control_offsets=set();read_count=0;tick_count=0;contact_count=0
    cmds={f['family_id']:f for f in plan['families']};seen_episode=set();paired_controls={}
    for ep in episodes:
        eid=ep['episode_id'];fid=ep['family_id'];h=float(ep['dt_s']);n=int(ep['n_substeps'])
        need(eid not in seen_episode and eid==2*fid+int(h==.00015625) and ep['split']==D.SPLITS[int(fid>=12)],'Exact family/dt episode identity');seen_episode.add(eid)
        initial=load_npz(ep['initial']['arrays']);initial_task=load_npz(ep['initial_task']);read_count+=2
        need(bits(initial['qpos27'],np.array(task['initial']['qpos27_F64'])) and bits(initial['qvel26'],np.array(task['initial']['qvel26_F64'])) and bits(initial['ctrl20'],np.array(task['initial']['ctrl20_F64'])),'Named loaded initial q/v/u bytes')
        close(initial['state58'],maths['codec'].actual_state(initial['qpos27'],initial['qvel26'],np.array(meta['phone_body_ipos'])),'Raw reset COM/world state58')
        need(bits(initial_task['state58_F64'],initial['state58']),'Initial task state source');D.receipt(initial['integration_state'],ep['initial']['receipt'],0,h)
        prev=initial['integration_state'];prealive=True
        for k,sid in enumerate(ep['interval_row_ids']):
            ident=byid[sid];need(ident['episode_id']==eid and ident['family_id']==fid and ident['step_k']==k,'Chronological within-episode intervals')
            a=load_npz(ident['trace']);decision=load_npz(ident['decision']);read_count+=2
            for key in set(a)&set(decision):need(bits(a[key],decision[key]),'Raw/decision duplicate byte agreement '+key)
            need(bits(a['source_integration_state'],prev),'Full300 target->next source continuity; no observer reset')
            need(bits(a['actual_control20'],decision['actual_u20_F64']) and bits(a['action_normalized20_F64'],decision['action20_F64']),'Raw action/control decision binding')
            a.update(decision);a['complete_H_bool']=np.asarray(ident['complete_H20ms']);a['reward_scalar_F64']=a['reward_F64'];a['task_done']=a['task_done_bool'];a['reader_done']=a['reader_done_bool']
            out,note,key=D.canonical_row(a,ident,ep,task,physical,maths,pre_hold_alive=prealive)
            need(out['physical_labels_known_bool']==ident['physical_training_eligible'],'Full-H finite/no-warning label validity is unchanged')
            need(out['actor_absorbing_done_bool']==bool(ident['terminated'] or ident['reference_timeout']),'Boundary task/Reader absorption distinct from collection truncation')
            out['acquisition_done_bool']=bool(ident['done']);out['collector_valid_bool']=bool(ident['valid'])
            need(out['acquisition_truncated_bool']==ident['collect_truncated'] and out['reference_timeout_bool']==ident['reference_timeout'],'Truncation/timeout exact roles')
            need(bits(out['normalized_action20_F64'],np.array(cmds[fid]['action20x20_F64'])[k]) and bits(out['u20_F64'],np.array(cmds[fid]['ctrl20x20_F64'])[k]),'Frozen action/EMA curve actual bytes')
            paired=(fid,k)
            if paired in paired_controls:need(bits(out['u20_F64'],paired_controls[paired]),'Paired dt held controls identical')
            paired_controls[paired]=out['u20_F64'].copy()
            close(a['post_state58'],np.array([maths['codec'].actual_state(q,v,np.array(meta['phone_body_ipos'])) for q,v in zip(a['post_qpos27'],a['post_qvel26'])]),'Every raw POST COM/world state58')
            if out['complete_H_bool'] and out['finite_bool']:
                need(bits(out['obs222_FP32'].astype(np.float64),a['obs222_F64']) and bits(out['next_obs222_FP32'].astype(np.float64),a['next_obs222_F64']),'F64 observation aliases are casts only')
            source_meta=read(checked(ident['raw_metadata']));need(source_meta['original_two_pad_summary'] is not None,'Original two-pad summary retained')
            note['original_two_pad_summary']=source_meta['original_two_pad_summary'];note['source_task_hard_terminal']=bool(ident['task_done']);note['source_decision']=ident['decision']
            shared=r['model_audits'][str(h)]['shared_physics_digest_without_dt'];aliases[key+(shared,'phone_COM_world_state58:1')].append(sid)
            control_offsets.add(note['source_control_segment_offset']);rows[sid]=out;notes[sid]=note;tick_count+=int(out['executed_substeps_i64']);contact_count+=len(a['contact_rows24'])
            prev=a['target_integration_state'];prealive=bool(prealive and not out['actor_absorbing_done_bool'])
        need(len(ep['interval_row_ids'])<=20,'Episode finite collection budget')
    need(seen_episode==set(range(32)) and len(rows)==r['rows'] and len(control_offsets)==1,'All32 episodes/rows and common discovered ctrl segment')
    groups=[dict(sample_ids=v,family_ids=sorted({identities[i]['family_id'] for i in v}),splits=sorted({identities[i]['split'] for i in v})) for v in aliases.values() if len(v)>1]
    cross=[g for g in groups if len(g['splits'])>1];need(groups==r['alias_groups'] and cross==r['cross_partition_alias_groups'],'Independent fullcontext/action alias membership')
    ordered=[rows[i] for i in range(len(rows))];components=union_components(groups)
    for i,row in enumerate(ordered):
        row['alias_family_component_i64']=next(j for j,g in enumerate(components) if row['family_id_i64'] in g)
        following=rows.get(i+1);row['H2_start_eligible_bool']=bool(following is not None and row['Actor_start_eligible_bool'] and following['Actor_start_eligible_bool'] and not row['actor_absorbing_done_bool'] and row['episode_id_i64']==following['episode_id_i64'] and following['step_k_i64']==row['step_k_i64']+1)
    arrays=stack_rows(ordered);save(O/'all_rows_with_partial.npz',arrays)
    partitions={};stats={}
    for split,name in zip(D.SPLITS,['train','monitor']):
        ids=np.array([i for i,v in enumerate(identities) if v['split']==split],dtype=np.int64)
        dest=O/(name+'.npz');save(dest,{k:v[ids] for k,v in arrays.items()});partitions[name]=pin(dest)
        stats[name]=dict(rows=len(ids),complete_H=int(arrays['complete_H_bool'][ids].sum()),WM_eligible=int(arrays['WM_eligible_bool'][ids].sum()),Actor_start_eligible=int(arrays['Actor_start_eligible_bool'][ids].sum()),H2_start_eligible=int(arrays['H2_start_eligible_bool'][ids].sum()),partial=int((~arrays['complete_H_bool'][ids]).sum()))
    write(O/'row_identities.json',[notes[i] for i in range(len(notes))]);write(O/'alias_groups.json',dict(exact_input_action_groups=groups,cross_split_groups=cross,alias_connected_family_components=components,policy=p['cross_split_alias_policy']))
    need(tick_count==r['counts']['integration_steps'] and contact_count==r['counts']['step_contact_force_reads'],'All native tick/contact rows accounted, no extra physics')
    checks={k:True for k in ['all32episodes16families_complete_source_manifests','allrawPOST_COM_world_state58_and_delta52','native_before_afteraction_target300_receipts_and_continuity','actual_action_EMA_u_memory83_obs222_phase','canonical22_signed_impulses_sourceBODY_movingCOM_directonce','triple_joint_coverage_not_marginal_product','partial_H_labels_NaN_and_prefix_preserved','boundary_taskterminal_distinct_anyPOST_and_collectiontruncation','exact_alias_groups_and_fixed_family_partitions','original_twopad_summary_and_no_future_labels_as_inputs']}
    checks['no_crosssplit_alias']=not cross;checks['both_partition_WM_rows_and_TRAIN_H2_rows_available']=bool(stats['train']['WM_eligible'] and stats['monitor']['WM_eligible'] and stats['train']['H2_start_eligible'])
    qualified=all(checks.values());report=dict(schema='phone_tripod_replay_training_dataset:1',status='COMPLETED_SAVED_TRIPOD_REPLAY22_LABEL_DATASET' if qualified else 'BLOCKED_DATASET_QUALIFICATION_ALL_ROWS_RETAINED',engineering_qualified=qualified,checks=checks,passed=sum(checks.values()),total=len(checks),
        protocol=pin(protocol_path),collector_report=p['inputs']['collector_report'],collector_terminal=p['inputs']['collector_terminal'],family_plan=r['family_plan'],task_bundle=r['task_bundle'],
        physical_parameters=p['inputs']['physical_parameters'],geometry_metadata=p['inputs']['geometry_metadata'],partition_arrays=partitions,family_split=dict(train=list(range(12)),monitor=list(range(12,16))),
        all_rows=pin(O/'all_rows_with_partial.npz'),sample_identities=pin(O/'row_identities.json'),alias_review=pin(O/'alias_groups.json'),rows=len(ordered),field_map=FIELD_MAP,split_counts=stats,
        canonical_pair_geom_ids=list(D.PAIRS),designated_pad_geom_ids=list(D.PADS),input_alias_control_segment_discovered_offset=next(iter(control_offsets)),
        frames='Signed frame.T decoded native force; positive on geom2, negative on geom1. Moment about actual moving PRE COM, then both blocks rotate into frozen source BODY axes. H labels only complete/finite/no-warning.',
        termination_scope='Actor/TD absorption uses source boundary POST-H task/Reader done. Any-native-tick threshold event is separate diagnostic.0.4s collection truncation may bootstrap; not2sreference timeout.',
        observation_scope='Canonical FP32 observations; F64 aliases are stored casts only. Geometry descriptor slices source-backed; exporter independently checks current state/history/clock slices, not new FK.',
        numeric_scope='Reused pinned saved decoder and pure NumPy structured encoder; observed nonzero momentum defects are retained, not forced to0.',
        data_scope='New tripod local known-initial-state simulation only. No historical fresh-final12, oldCHECK/TEST or old TRAIN array payload opened by exporter; no unseen ego/policy/holding claim.',
        counts=dict(saved_NPZ_reads=read_count,native_PRE_ticks_reprocessed=tick_count,raw_contact_rows=contact_count,new_native_model_FK_forward_steps_NN_calls=0),
        sources=p['sources'],elapsed_s=time.perf_counter()-started)
    write(O/'report.json',report);need(qualified,'Crosssplit aliases/insufficient rows: complete artifact retained, training blocked');return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:
        p,r,*_=metadata(a.protocol);print(json.dumps(dict(status='METADATA_ONLY_NO_LABEL_NPZ_TORCH_NATIVE',rows=r['rows'],run_authorized=p['saved_only_run_authorized'],protocol=pin(a.protocol))))
    else:
        try:
            r=run(a.protocol);print(json.dumps(dict(report=pin(Path(read(a.protocol)['output'])/'report.json'),rows=r['rows'],split_counts=r['split_counts'],passed=r['passed'],total=r['total'])))
        except BaseException as e:
            out=Path(read(a.protocol)['output']);out.mkdir(exist_ok=True);write(out/'failure.json',dict(error=repr(e),source=pin(__file__),automatic_retry=False,new_native_calls=0));raise
