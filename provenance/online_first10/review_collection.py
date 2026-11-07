"""Saved NumPy/JSON arithmetic only; no runtime/model/backend imports."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,math,sys,traceback,tomllib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'runs/phone_online_v1/collection_cycle01'
OUTPUT=Path(__file__).with_name('collection_review.json')
PAIRS=(0,2,8,11,13,15,21,24,26,28,34,37,39,41,47,50,52,54,60,63,65,67)
PADS=(15,28,41)
checks={};errors={};reads=Counter();maxerr=defaultdict(float)

def need(v,label):
    if not bool(v):raise ValueError(label)
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):
    p=Path(v['path']).resolve(strict=True)
    need(pin(p)==v,'Saved artifact SHA/bytes/path '+str(p));return p
def read(path):
    reads['JSON']+=1;return json.loads(Path(path).read_text(encoding='utf8'))
def load(path):
    reads['NPZ']+=1
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def close(a,b,label,tol=2e-12):
    a,b=np.asarray(a),np.asarray(b)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' finite shape')
    e=float(np.max(np.abs(a-b))) if a.size else 0.;maxerr[label]=max(maxerr[label],e)
    need(e<=tol,label+' arithmetic tolerance')
def receipt(v,m,tick,h):
    need(v.dtype==np.float64 and v.shape==(300,) and m['signature']==16383 and m['size']==300,'Typed integration300')
    need(hashlib.sha256(v.tobytes()).hexdigest()==m['sha256'],'Integration300 raw SHA')
    need(m['global_tick']==tick,'Receipt integer tick')
    close(np.array(m['native_time_s']),np.array(tick*h),'receipt_clock',1e-12)
def rotations(q):
    q=np.asarray(q);w,x,y,z=q.T
    need(np.max(abs(np.sum(q*q,axis=1)-1))<=1e-12,'Raw unit quaternion, no repair')
    return np.array([[w*w+x*x-y*y-z*z,2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),w*w-x*x+y*y-z*z,2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),w*w-x*x-y*y+z*z]]).transpose(2,0,1)
def state58(q,v,ipos):
    R=rotations(q[:,23:27]);offset=np.einsum('bij,j->bi',R,ipos)
    omega=np.einsum('bij,bj->bi',R,v[:,23:26])
    return np.c_[q[:,:20],v[:,:20],q[:,20:23]+offset,R.reshape(-1,9),v[:,20:23]+np.cross(omega,offset),omega]
def logR(R):
    v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2
    s=np.linalg.norm(v);theta=math.atan2(s,(np.trace(R)-1)/2)
    need(theta<math.pi-1e-6,'Observed world Log branch')
    return v if s<1e-12 else theta/s*v
def norm_digest(arrays,mode):
    h=hashlib.sha256();prefix=mode+'__'
    for key in sorted(k for k in arrays if k.startswith(prefix)):
        a=arrays[key];name=key[len(prefix):]
        h.update(name.encode());h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def run():
    report=read(INPUT/'report.json');dataset=read(checked(report['dataset_report']));worker=read(INPUT/'worker_receipt.json')
    protocol=read(checked(report['protocol']));need(worker['report']==pin(INPUT/'report.json') and worker['protocol']==report['protocol'],'Worker/report/protocol pins')
    need(report['schema']=='phone_tripod_shared_native_collection_result:1' and dataset['schema']=='phone_tripod_online_common_replay_dataset:1','Own actual collection schemas')
    for value in report['runtime_sources'].values():checked(value)
    need(dataset['protocol']==report['protocol'] and dataset['actor_input']==report['actor_input'],'Dataset same collection Actor provenance')
    # Metadata only: no checkpoint/model payload is opened in this reviewer.
    need(report['checkpoint']==report['actor_input']['checkpoint']==dataset['paired_checkpoint'],'Actor metadata joins')
    manifest=checked(protocol['artifact_manifest']);m=tomllib.loads(manifest.read_text())
    physical_entry=m['artifacts'][protocol['configuration']['artifacts']['physical']]
    physical_path=(manifest.parent/Path(m['roots'][physical_entry['root']])/physical_entry['path']).resolve()
    if physical_entry.get('sha256'):need(pin(physical_path)['sha256']==physical_entry['sha256'],'Physical JSON registered SHA')
    physical=read(physical_path);ipos=np.asarray(physical['body_ipos_body_m']);g=np.asarray(physical['gravity_world_m_s2'])
    bundle=read(checked(report['task_bundle']));initial=np.asarray(bundle['initial']['native_PRE_state58_F64'])
    profile=read(checked(report['profile_plan']));curves={p['profile_id']:np.asarray(p['action20x20_F64']) for p in profile['profiles']}
    pool=load(checked(dataset['common_pool']));notes=read(checked(dataset['sample_identities']));episodes=read(checked(report['episode_records']))
    need(len(notes)==len(pool['row_id_i64'])==112 and len(episodes)==16,'All112 rows/16 episodes')
    need(np.array_equal(pool['row_id_i64'],np.arange(112)),'No row filtering/reindex')
    checks['pinned_sources_metadata_full_row_order']=True
    alias=defaultdict(list);initial_alias=defaultdict(list);source_alias=defaultdict(list);offsets=set();episode_stats=[]
    total_steps=total_contacts=initial_contacts=0
    for ep in episodes:
        eid=ep['global_episode_id'];h=ep['dt_s'];n=ep['n_substeps'];ids=ep['interval_row_ids'];arm=int(eid>=8)
        need(eid==8*arm+2*ep['family_id']+int(h==.00015625) and n in (64,128) and h*n==.02,'Paired profile/dt global episode identity')
        need(len(ids)==7,'Every episode actual7H, not assumed full20H')
        init=load(checked(ep['initial']['arrays']));imeta=read(checked(ep['initial']['metadata']));it=load(checked(ep['initial_task']))
        receipt(init['integration_state'],ep['initial']['receipt'],0,h)
        need(imeta['receipt300']==ep['initial']['receipt'],'Reset metadata receipt')
        for key,expected in [('qpos27',bundle['initial']['qpos27_F64']),('qvel26',bundle['initial']['qvel26_F64']),('ctrl20',bundle['initial']['ctrl20_F64'])]:
            need(bits(init[key],np.asarray(expected,dtype=np.float64)),'Initial raw q/v/u source bytes')
        close(state58(init['qpos27'][None],init['qvel26'][None],ipos)[0],init['state58'],'initial_state58')
        need(bits(it['state58_F64'],init['state58']),'Initial task state')
        initial_contacts+=len(init['initial_contact_rows24']);previous=init['integration_state'];previous_x=init['state58'];previous_m=it['memory83_F64']
        initial_alias[(ep['initial']['receipt']['sha256'],h)].append(eid)
        losses=[]
        for k,sid in enumerate(ids):
            note=notes[sid];identity=note['source_identity'];raw=load(checked(identity['trace']));decision=load(checked(identity['decision']));canon=load(checked(note['canonical_row']));meta=read(checked(identity['raw_metadata']))
            need(identity['sample_id']==sid and identity['global_episode_id']==eid and identity['step_k']==k and identity['data_role']=='ONLINE_TRAIN_ONLY_NEW_STAGE','Actual row/source identity')
            need(identity['family_id']==ep['family_id'] and identity['arm']==ep['arm'] and identity['dt_s']==h,'Matched episode source joins')
            need(meta['raw']==identity['trace'] and meta['episode_id']==eid and meta['step_k']==k,'Durable raw metadata link')
            for key,value in canon.items():need(bits(value,pool[key][sid]),'Per-row canonical/raw pool exact field '+key)
            N=len(raw['pre_tick']);total_steps+=N;total_contacts+=len(raw['contact_rows24'])
            need(N==n==identity['executed_substeps'] and np.array_equal(raw['pre_tick'],np.arange(k*n,(k+1)*n)) and np.array_equal(raw['post_tick'],raw['pre_tick']+1),'Full H integer PRE/POST phase')
            close(raw['pre_native_time_s'],raw['pre_tick']*h,'PRE_clock',1e-12);close(raw['post_native_time_s'],raw['post_tick']*h,'POST_clock',1e-12)
            for mk,ak,t in [('source_receipt','source_integration_state',k*n),('source_after_action_receipt','source_after_action_integration_state',k*n),('target_receipt','target_integration_state',(k+1)*n)]:
                need(identity[mk]==meta[mk],'Receipt raw metadata/identity equality');receipt(raw[ak],identity[mk],t,h)
            need(bits(raw['source_integration_state'],previous) and bits(raw['source_state58'],previous_x),'Complete target->next before-action context chain')
            before,after=raw['source_integration_state'],raw['source_after_action_integration_state'];u=raw['actual_control20'];memory=decision['memory_before83_F64']
            possible=[j for j in range(281) if bits(after[j:j+20],u) and bits(before[j:j+20],memory[61:81]) and bits(before[:j],after[:j]) and bits(before[j+20:],after[j+20:])]
            need(len(possible)==1,'Measured after-action control-only mutation, no scratch reset');offsets.add(possible[0])
            need(bits(memory,previous_m) and bits(memory,raw['memory_before83_F64']),'Memory83 source continuity')
            a=decision['action20_F64'];expected_action=.95*decision['actor_action20_FP32'].astype(np.float64)+.05*curves[ep['family_id']][k]
            need(bits(a,expected_action) and bits(decision['exploration_profile20_F64'],curves[ep['family_id']][k]),'Actual95/5 action and fixed profile')
            need(np.all(abs(a)<=1) and bits(a,raw['action_normalized20_F64']),'Normalized action source and bound')
            residual=.5*np.asarray(bundle['action']['symmetric_residual_scale20_rad'])*a+.5*memory[41:61]
            need(bits(residual,decision['residual20_F64']) and bits(u,np.asarray(bundle['initial']['ctrl20_F64'])+residual),'Exact EMA/control, no clipping')
            need(bits(u,decision['actual_u20_F64']) and bits(u,pool['u20_F64'][sid]) and hashlib.sha256(u.tobytes()).hexdigest()==identity['actual_control20_raw_sha256'],'Raw/decision/pool control')
            need(np.all(u>=bundle['action']['target_lower20']) and np.all(u<=bundle['action']['target_upper20']),'True control bounds')
            expected_m=np.r_[k+1,a,memory[1:21],residual,u,h,n]
            need(bits(decision['memory_after83_F64'],expected_m),'Memory exact action phase advancement')
            close(state58(raw['post_qpos27'],raw['post_qvel26'],ipos),raw['post_state58'],'all_POST_state58')
            x=raw['source_state58'];y=raw['post_state58'][-1];R=x[43:52].reshape(3,3);R1=y[43:52].reshape(3,3)
            need(bits(y,raw['target_at_H_state58']) and bits(y,pool['y58_F64'][sid]),'Complete actual H endpoint')
            delta=np.r_[y[:43]-x[:43],logR(R1@R.T),y[52:]-x[52:]];close(delta,pool['target_delta52_F64'][sid],'physical_world_delta52')
            relative=raw['post_state58'][:,43:52].reshape(-1,3,3)@initial[43:52].reshape(3,3).T
            vee=np.stack([relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]],axis=1)/2
            angles=np.arctan2(np.linalg.norm(vee,axis=1),(np.trace(relative,axis1=1,axis2=2)-1)/2)
            drifts=np.linalg.norm(raw['post_state58'][:,40:43]-initial[40:43],axis=1);loss=(drifts>.05)|(angles>math.pi/4);boundary=bool(loss[-1]);losses.append(boundary)
            reader_done=meta['termination']!='COMPLETE_INTERVAL';no_warning=not any(identity['warning_numbers']);finite=all(np.isfinite(raw[z]).all() for z in ['post_state58','post_qpos27','post_qvel26','contact_rows24'])
            wm=bool(N==n and finite and no_warning);actor=bool(wm and not any(losses[:-1]) and np.isfinite(decision['reward_F64']))
            absorbing=bool(boundary or reader_done);trunc=bool(k==19 and not absorbing);done=bool(absorbing or trunc)
            for key,expected in [('complete_H_bool',True),('WM_eligible_bool',wm),('Actor_start_eligible_bool',actor),('physical_labels_known_bool',wm),('pre_hold_alive_bool',not any(losses[:-1])),('task_hold_failed_boundary_POST_bool',boundary),('task_hold_failed_any_native_POST_bool',bool(loss.any())),('actor_absorbing_done_bool',absorbing),('reader_done_bool',reader_done),('acquisition_truncated_bool',trunc),('reference_timeout_bool',False),('acquisition_done_bool',done)]:need(bool(pool[key][sid])==expected,'Independent canonical mask '+key)
            for key,expected in [('task_done_bool',boundary),('terminated_bool',absorbing),('done_bool',done),('reader_done_bool',reader_done),('truncated_bool',trunc),('collect_truncated_bool',trunc),('reference_timeout_bool',False)]:need(bool(decision[key])==expected,'Decision stop/mask '+key)
            need(bool(identity['task_done'])==boundary and bool(identity['done'])==done and bool(identity['terminated'])==absorbing,'Identity stop evidence')
            Fn=np.zeros((N,22));generated=np.zeros((N,22),dtype=np.int64);J=np.zeros((22,6));world=np.zeros((N,6));COM_pre=np.vstack([x[None,40:43],raw['post_state58'][:-1,40:43]])
            for cr in raw['contact_rows24']:
                tick,g1,g2=map(int,cr[:3]);ti=tick-k*n;need(0<=ti<N,'Every contact PRE tick')
                if 71 not in (g1,g2):continue
                gid=g2 if g1==71 else g1;need(gid in PAIRS,'Canonical phone boundary membership');j=PAIRS.index(gid);sign=1. if g2==71 else -1.;frame=cr[9:18].reshape(3,3)
                F=sign*(frame.T@cr[18:21]);M=np.cross(cr[6:9]-COM_pre[ti],F)+sign*(frame.T@cr[21:24])
                J[j]+=h*np.r_[R.T@F,R.T@M];world[ti]+=np.r_[F,M];Fn[ti,j]+=max(0.,cr[18]);generated[ti,j]+=1
            close(J,pool['impulse_source_BODY_moving_COM22x6_F64'][sid],'signed_J22_sourceBODY_movingCOM')
            close(world,raw['pre_contact_wrench_about_COM_world6'],'raw_world_wrench',1e-11)
            close(Fn,raw['tripod_pair_positive_Fn22_N'],'raw_pair_Fn')
            need(np.array_equal(generated,raw['tripod_pair_generated22']),'Raw generated rows')
            close(Fn.mean(0),pool['mean_positive_Fn22_N_F64'][sid],'meanFn22')
            need(bits((Fn>.1).mean(0),pool['Fn_gt_0p1_coverage22_F64'][sid]),'Pair actualPRE coverage')
            need(bits((generated>0).mean(0),pool['generated_contact_coverage22_F64'][sid]),'Generated coverage')
            triple=float(np.all(Fn[:,[PAIRS.index(p) for p in PADS]]>.1,axis=1).mean())
            need(triple==pool['simultaneous_triple_coverage_F64'][sid]==meta['triple_simultaneous_coverage'],'Joint triple coverage, not marginal product')
            head=np.r_[y[:40]-x[:40],y[40:43]-x[40:43]-x[52:55]*.02-.5*g*.02**2,delta[43:46],J.ravel()]
            close(head,pool['target_physical_heads178_F64'][sid],'physical_heads178')
            obs=decision['obs222_FP32'];need(bits(obs[:138],np.r_[x,memory[1:81]].astype(np.float32)) and bits(obs[174:176],np.array([k/100,(100-k)/100],dtype=np.float32)),'Current-only observation state/history/phase')
            need(bits(obs, pool['obs222_FP32'][sid]) and bits(decision['next_obs222_FP32'],pool['next_obs222_FP32'][sid]),'Saved obs222 pool/decision joins')
            need(np.isfinite(pool['reward_A1_tracking_F64'][sid]) and np.isfinite(pool['reward_A2_engineered_material_proxy_F64'][sid]),'Both reward columns finite, future force not recomputed as reward')
            alias[(identity['source_after_action_receipt']['sha256'],hashlib.sha256(u.tobytes()).hexdigest(),h,n,16383,report['model_audits'][str(h)]['shared_physics_digest_without_dt'])].append(sid)
            source_alias[(identity['source_receipt']['sha256'],h)].append(sid)
            previous=raw['target_integration_state'];previous_x=y;previous_m=decision['memory_after83_F64']
        need(losses==[False]*6+[True] and ep['termination']=='TASK_POST_H_LOSS','First boundary loss at last recorded row, no postfail samples')
        episode_stats.append(dict(global_episode_id=eid,arm=ep['arm'],profile=ep['family_id'],dt_s=h,H_intervals=len(ids),native_steps=n*len(ids),duration_s=.02*len(ids),POST_H_taskloss=True))
    checks['raw_durable_pins_canonical_pool_and_decision_links']=True
    checks['all300_receipts_chain_and_only_control_changed_after_action']=True
    checks['all112_actual_H_actions_EMA_memory_and_POST_state58']=True
    checks['all112_canonical22_impulses_Fn_coverage_and_joint_tripod_recomputed']=True
    checks['independent_boundary_termination_WM_Actor_TD_and_partial_masks']=True
    need(offsets=={80},'Discovered common control slice, never guessed before check')
    h2=np.zeros(112,dtype=np.bool_)
    for i in range(111):h2[i]=bool(pool['Actor_start_eligible_bool'][i] and pool['Actor_start_eligible_bool'][i+1] and not pool['actor_absorbing_done_bool'][i] and pool['online_global_episode_id_i64'][i]==pool['online_global_episode_id_i64'][i+1] and pool['step_k_i64'][i+1]==pool['step_k_i64'][i]+1)
    need(bits(h2,pool['H2_start_eligible_bool']),'H2 same-episode/next-step mask')
    alias_groups=[v for v in alias.values() if len(v)>1];declared=read(checked(dataset['alias_review']))
    need(alias_groups==[] and declared['exact_full_afteraction_action_groups']==[],'Independent exact context/action aliases')
    initial_groups=[dict(global_episode_ids=v,receipt_sha256=k[0],dt_s=k[1]) for k,v in initial_alias.items() if len(v)>1]
    source_groups=[dict(sample_ids=v,receipt_sha256=k[0],dt_s=k[1]) for k,v in source_alias.items() if len(v)>1]
    need(initial_groups==report['initial_receipt_alias_groups'] and source_groups==report['source_state_alias_groups'],'Independent known initial/source alias cohorts')
    checks['global_episode_H2_masks_and_all_alias_groups']=True
    arm_counts={}
    for ai,mode in enumerate(('A1_tracking','A2_engineered_material_proxy')):
        ii=pool['arm_i64']==ai;part=load(checked(dataset['arm_partitions'][mode]));need(set(part)==set(pool),'All arm fields retained')
        for key in pool:need(bits(part[key],pool[key][ii]),'Exact arm partition, no filtering '+key)
        arm_counts[mode]=dict(rows=int(ii.sum()),WM_rows=int(pool['WM_eligible_bool'][ii].sum()),Actor_rows=int(pool['Actor_start_eligible_bool'][ii].sum()),H2_rows=int(h2[ii].sum()),absorbing_task_failures=int(pool['actor_absorbing_done_bool'][ii].sum()),native_steps=int(pool['executed_substeps_i64'][ii].sum()))
        need(arm_counts[mode]['rows']==56 and arm_counts[mode]['native_steps']==5376,'Equal actual source budgets')
        need({k:arm_counts[mode][k] for k in ('WM_rows','Actor_rows','H2_rows')}==dataset['eligibility_counts'][mode],'Independent dataset eligibility counts')
    need(total_steps==10752==report['counts']['integration_steps'] and total_contacts==report['counts']['step_contact_force_reads']==17247 and initial_contacts==report['counts']['initial_contact_force_reads']==48,'Every native step and contact count from saved data')
    need(report['counts']['contact_force_reads']==total_contacts+initial_contacts and report['counts']['receipt_reads']==16+3*112 and report['counts']['initial_forward']==16 and report['counts']['observer_forward']==0,'Source-accounted reset/receipt/observer counts')
    need(report['Actor_FW']==112 and report['WM_FW']==report['Critic_FW']==report['training_updates']==0,'Collector source call counts, not new reviewer calls')
    checks['all16_episodes_bothdt_full_112_rows_and_native_budget']=True
    nb=load(INPUT/'actor_normalizers_before.npz');na=load(INPUT/'actor_normalizers_after.npz');need(set(nb)==set(na),'Norm raw keyset')
    for key in nb:need(bits(nb[key],na[key]),'Raw norm dtype/bytes unchanged '+key)
    evidence=read(checked(report['read_only_actor_evidence']));need(evidence['before']==evidence['after'],'Weight/host equality as reported; no PT load')
    for mode in arm_counts:need(norm_digest(nb,mode)==evidence['before'][mode]['norm_digest']==norm_digest(na,mode),'Independent raw normalizer digest')
    rb=load(INPUT/'RNG_before.npz');ra=load(INPUT/'RNG_after.npz');need(set(rb)==set(ra),'Raw RNG keys')
    for key in rb:need(bits(rb[key],ra[key]),'Exported raw RNG arrays unchanged')
    checks['raw_normalizer_and_exported_RNG_bytes_unchanged']=True
    checks['review_backend_free_NumPy_JSON_only']=not any(k in sys.modules for k in ('torch','mujoco','warp'))
    need(checks['review_backend_free_NumPy_JSON_only'],'No backend imported by review')
    return dict(schema='phone_tripod_online_collection_saved_review:1',status='PASS_SAVED_COLLECTION_INTEGRITY_NOT_POLICY_QUALIFICATION',engineering_qualified=True,
        producer_report=pin(INPUT/'report.json'),dataset_report=report['dataset_report'],protocol=report['protocol'],worker_receipt=pin(INPUT/'worker_receipt.json'),
        checks=checks,passed=sum(checks.values()),total=len(checks),rows=112,episodes=16,partial_rows=0,WM_eligible_rows=112,Actor_start_eligible_rows=112,H2_eligible_rows=int(h2.sum()),
        actor_absorbing_done_rows=int(pool['actor_absorbing_done_bool'].sum()),collection_truncations=int(pool['acquisition_truncated_bool'].sum()),reference_timeouts=int(pool['reference_timeout_bool'].sum()),Reader_done_rows=int(pool['reader_done_bool'].sum()),
        TD_nonabsorbing_complete_rows=int((pool['complete_H_bool']&~pool['actor_absorbing_done_bool']).sum()),all_episode_durations_s=[s['duration_s'] for s in episode_stats],
        per_arm=arm_counts,episode_summary=episode_stats,actual_native_steps_from_traces=total_steps,step_contact_rows=total_contacts,initial_contact_rows=initial_contacts,
        exact_afteraction_action_alias_groups=alias_groups,known_initial_alias_groups=initial_groups,known_source_alias_groups=source_groups,discovered_control_offset=80,max_abs_recomputed_error=dict(maxerr),
        scope=dict(reviewer_is_shared_collector_labels_source_author=True,independent_saved_arithmetic=True,independent_native_experiment=False,
            raw_pin_existence_and_bytes_verified=True,fsync_execution_order='source-backed; saved artifacts alone do not reconstruct write-time ordering',
            full_weights_and_norm_host_readonly='source-backed state/host digest equality; no checkpoint payload opened',raw_norm_readonly_independently_checked=True,
            RNG_scope='TorchCPU and NumPy key arrays independently compared; Python/NumPy scalar header after-state equality is producer/source-backed because only before-header JSON is exported',
            collection_scope='0.14s POST-H terminated known-initial local simulation, not2s tickwise holding or policy benefit/independent ego qualification',
            old_CHECK_TEST_fresh_final_used=False,training_results_or_checkpoint_payloads_read=False),
        calls=dict(saved_files_read=dict(reads),NN_native_forward_step_AD_training_GPU_ptloads=0),review_source=pin(__file__))

if __name__=='__main__':
    try:r=run()
    except BaseException as e:
        r=dict(schema='phone_tripod_online_collection_saved_review:1',status='FAILED_SAVED_REVIEW_NO_RETRY',engineering_qualified=False,checks=checks,passed=sum(checks.values()),total=len(checks),error=repr(e),traceback=traceback.format_exc(),calls=dict(NN_native_forward_step_AD_training_GPU_ptloads=0),review_source=pin(__file__))
    with OUTPUT.open('x',encoding='utf8') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(OUTPUT),status=r['status'],passed=r['passed'],total=r['total'])))
    if not r['engineering_qualified']:raise SystemExit(1)
