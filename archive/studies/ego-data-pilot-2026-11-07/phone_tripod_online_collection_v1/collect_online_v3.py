"""New-stage online Actor -> native Reader -> durable raw -> canonical replay.

Default path is JSON/hash-only. No top-level Torch/native import. Actual CPU
collection is separate from frozen offline1000 training and performs no fit.
"""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,importlib.util,json,os,random,sys,time,traceback
import numpy as np

H=Path(__file__).resolve().parent
SCHEMA='phone_tripod_matched_online_collection_CPU_protocol:3'
RESULT='phone_tripod_matched_online_collection_result:3'
POOL='phone_tripod_online_common_replay_dataset:1'
MODES=('A1_tracking','A2_engineered_material_proxy')
RULES=dict(profiles_per_arm=4,native_dt_s=[.0003125,.00015625],substeps=[64,128],
    maximum_intervals_per_episode=20,episodes_per_arm=8,maximum_H_per_arm=160,
    maximum_native_steps_per_arm=15360,maximum_rows=320,maximum_native_steps=30720,
    H_s=.02,exploration_mixture=.05,action_rule='a=.95*Actor_FP32_as_F64+.05*fixed_profile_F64; no clip',
    device='cpu',NN_dtype='float32',physical_memory_action_dtype='float64',cpu_threads=4,
    offline_root_paired_round=1000,offline_root_shared_WM_updates=1100,collection_cycles=3,updates_per_completed_source_cycle=100,
    training_updates=0,optimizer_constructors=0,WM_forward=0,Critic_forward=0,
    terminal='boundaryPOSTH task loss or Reader physical loss; partial invalid; collection20H truncation bootstraps',
    data_role='ONLINE_TRAIN_ONLY_NEW_STAGE; no independent ego or heldout claim',
    automatic_retry=False)

def need(x,msg):
    if not x:raise ValueError(msg)

def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def checked(v):
    need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Changed absolute pin')
    return Path(v['path'])

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def profile_plan():
    """Four named deterministic open-loop exploration signals; shared by arms/dt."""
    t=(np.arange(20,dtype=np.float64)+.5)*.02;j=np.arange(20,dtype=np.float64)
    a=np.sin(2*np.pi*(.5+j/20)[None,:]*t[:,None]+j[None,:]*.37)
    b=np.cos(2*np.pi*(.75+j/40)[None,:]*t[:,None]+j[None,:]*.19)
    return dict(schema='phone_tripod_online_fixed_profile_plan:1',profiles=[
        dict(profile_id=i,name=name,action20x20_F64=v.tolist(),raw_sha256=sha(v))
        for i,(name,v) in enumerate(zip(['sin_plus','sin_minus','cos_plus','cos_minus'],[a,-a,b,-b]))],
        actor_action_not_known_before_execution=True,matching='same profiles/initial/nativeH budgets; actions may differ by arm and closed-loop state',
        predeclared_before_new_native=True)

def metadata_preflight(path):
    p=read(path);need(p['schema']==SCHEMA and p['rules']==RULES,'Versioned online collection recipe')
    need(type(p['CPU_run_authorized']) is bool and p['training_authorized'] is False and p['GPU_authorized'] is False,'CPU collection only')
    for v in [*p['sources'].values(),*[v for v in p['inputs'].values() if v is not None]]:checked(v)
    need(p['sources']['collector']==pin(__file__),'Own v3 source binding')
    need(read(checked(p['inputs']['profile_plan']))==profile_plan(),'Unchanged fixed profile bytes')
    seed=read(checked(p['inputs']['offline_seed_protocol']))
    for key in ['task','reader_replay','directional','imagination','Actor_Critic','Actor_common']:
        need(seed['sources'][key]==p['sources'][key],'Frozen root interface source '+key)
    need(seed['sources']['trainer']==p['sources']['paired_trainer'],'Frozen canonical F64 memory wrapper')
    bundle=read(checked(p['inputs']['task_bundle']))
    need(bundle['schema']=='phone_fixed_wrist_ego61_tripod_task:1' and bundle['task_designated_geom_ids']==[15,28,41],'Same tripod task')
    need(seed['inputs']['task_bundle']==p['inputs']['task_bundle'] and bundle['model']==p['inputs']['model_mjb'],'Same initial/task/model')
    qual=read(checked(p['inputs']['bridge_qualification']))
    need(qual['status']=='PASS_TWO_ACTUAL_ONE_H_INTERFACES' and all(v is True for v in qual['checks'].values()),'Actual bridge qualification')
    qp=read(checked(qual['protocol']));need(qp['inputs']['bundle']==p['inputs']['task_bundle'] and qp['inputs']['model']==p['inputs']['model_mjb'],'Qualified bridge lineage')
    for k in ['task','reader_replay','native_reader','directional']:need(qp['sources'][k]==p['sources'][k],'Frozen bridge implementation '+k)
    dataset=read(checked(p['inputs']['offline_dataset']));review=read(checked(p['inputs']['offline_dataset_review']))
    need(dataset['task_bundle']==p['inputs']['task_bundle'] and review['dataset_report']==p['inputs']['offline_dataset'] and review['engineering_qualified'] is True,'Frozen offline provenance; no array read')
    L=mod('online_actor_lineage_v3',checked(p['sources']['actor_lineage']))
    context=L.validate_actor_metadata(p)
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==H and not out.exists(),'Every collection round gets fresh local output')
    return p,bundle,out,context['pending']

def tensor_tree_digest(state):
    """Raw typed header+bytes; mixed F32/F64/int normalizers are not cast."""
    h=hashlib.sha256()
    for name,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def load_actors_for_round(p,torch):
    L=mod('online_actor_lineage_payload',checked(p['sources']['actor_lineage']))
    context=L.validate_actor_metadata(p);need(not context['pending'],'Actor lineage fully bound before one own load')
    factory=mod('online_actual_factory',checked(p['sources']['imagination']))
    sys.path.insert(0,str(Path(p['sources']['Actor_Critic']['path']).parents[3]))
    ck=torch.load(checked(p['actor_input']['checkpoint']),map_location='cpu',weights_only=False)
    L.validate_actor_payload(ck,p,context)
    actors={};evidence={}
    for mode in MODES:
        pair=factory.make_fresh_actor_critic(seed=1733,source_pins=p['sources'],device='cpu')
        actor=pair['actor'];actor.load_state_dict(ck['arms'][mode]['stateactor'],strict=True)
        need(tensor_tree_digest(actor.state_dict())==tensor_tree_digest(ck['arms'][mode]['stateactor']) and all(torch.isfinite(v).all().item() for v in actor.state_dict().values()),'Exact finite serialized Actor')
        need(tensor_tree_digest(actor.norm.state_dict())==tensor_tree_digest(ck['arms'][mode]['norm']['actor']),'Loaded raw Actor normalizer; never refit')
        actor.eval();actor.requires_grad_(False);actors[mode]=actor
        evidence[mode]=dict(state_digest=tensor_tree_digest(actor.state_dict()),norm_digest=tensor_tree_digest(actor.norm.state_dict()),
            norm_host={k:v for k,v in vars(actor.norm).items() if isinstance(v,(bool,int,float,str))})
        del pair
    return actors,evidence

def canonical_online_row(D,raw,identity,episode,bundle,physical,maths,prealive):
    """Reuse exact frozen arithmetic, not original family identities or old data."""
    need(identity['data_role']=='ONLINE_TRAIN_ONLY_NEW_STAGE' and 0<=identity['family_id']<4,'Own online family namespace')
    delegated=dict(identity,split='TRIPOD_TRAIN')
    result,note,alias=D.canonical_row(raw,delegated,episode,bundle,physical,maths,pre_hold_alive=prealive)
    note['split']='ONLINE_TRAIN';note['source_identity']=identity
    result['acquisition_done_bool']=bool(identity['done'])
    return result,note,alias

def collect_round(protocol_path):
    """Execute one new collection round. No learning/callback/optimizer side effect."""
    p,bundle,out,pending=metadata_preflight(protocol_path)
    need(p['CPU_run_authorized'] is True and not pending,'Root-controlled actual online collection')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    need(not torch.cuda.is_initialized(),'CPU inference only')
    C=mod('online_frozen_collector',checked(p['sources']['raw_collector']))
    D=mod('online_frozen_labels',checked(p['sources']['label_adapter']))
    E=mod('online_field_map',checked(p['sources']['dataset_exporter']))
    L=mod('online_round_lineage',checked(p['sources']['actor_lineage']))
    out.mkdir();C.write(out/'protocol.json',p);start=time.perf_counter();readers={};episodes=[];notes=[];rows=[];aliases=defaultdict(list)
    phase='load_actors_for_round';actor_fw=0;raw_calls=0
    try:
        actors,before=load_actors_for_round(p,torch)
        C.save(out/'actor_normalizers_before.npz',{mode+'__'+k:v.detach().cpu().numpy() for mode,a in actors.items() for k,v in a.norm.state_dict().items()})
        rng_before=dict(python=random.getstate(),numpy=np.random.get_state(),torchCPU=torch.get_rng_state().numpy())
        C.save(out/'RNG_before.npz',dict(torchCPU=rng_before['torchCPU'],numpy_keys=rng_before['numpy'][1]))
        C.write(out/'RNG_before.json',dict(python=rng_before['python'],numpy_header=[rng_before['numpy'][0],*rng_before['numpy'][2:]]))
        taskmod=mod('online_task',checked(p['sources']['task']));bridge=mod('online_bridge',checked(p['sources']['reader_replay']))
        reader=mod('online_reader',checked(p['sources']['native_reader']));gap=mod('online_gap',checked(p['sources']['directional']))
        geometry=gap.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']))
        trainer=mod('online_canonical_memory',checked(p['sources']['paired_trainer']))
        tasks={mode:trainer.CanonicalMemoryTask(taskmod.PhoneTask(checked(p['inputs']['task_bundle']),mode=mode,geometry=geometry)) for mode in MODES}
        physical=read(checked(p['inputs']['physical_parameters']));maths=D.load_math();plan=read(checked(p['inputs']['profile_plan']))
        for h in RULES['native_dt_s']:readers[h]=reader.NativePhoneReader(checked(p['inputs']['model_mjb']),dt_s=h)
        need(len({r.shared_physics_digest for r in readers.values()})==1,'Same model except dt')
        with torch.inference_mode():
            for ai,mode in enumerate(MODES):
                for profile in plan['profiles']:
                    fid=profile['profile_id'];curve=np.array(profile['action20x20_F64'],dtype=np.float64)
                    for hi,h in enumerate(RULES['native_dt_s']):
                        local_ep=2*fid+hi;global_ep=8*ai+local_ep;phase=f'{mode}/profile{fid}/dt{hi}'
                        folder=out/f'episode_{global_ep:02d}';folder.mkdir();sink=C.PersistentRawSink(folder,global_ep)
                        task=tasks[mode];r=readers[h];adapter=bridge.PhoneReaderTaskBridge(r,task,geometry,sink);adapter.episode_id=global_ep
                        x,obs0=adapter.reset();C.save(folder/'initial_task.npz',dict(state58_F64=x.numpy(),memory83_F64=task.flatten_memory(adapter.memory).numpy(),obs222_FP32=obs0.numpy()))
                        ep=dict(episode_id=local_ep,global_episode_id=global_ep,family_id=fid,arm=mode,dt_s=h,n_substeps=r.n,
                            initial=sink.initial,initial_task=C.pin(folder/'initial_task.npz'),interval_row_ids=[],first_action=None,termination=None)
                        prealive=True
                        for k in range(20):
                            sink.k=k;m0=adapter.memory;memory0=task.flatten_memory(m0).clone();obs=task.observation(adapter.x,m0,geometry)
                            actor_action=bridge.online_actor_action(actors[mode],obs,device='cpu');actor_fw+=1
                            a=.95*actor_action.double()+.05*torch.tensor(curve[k],dtype=torch.float64)
                            need(a.shape==(20,) and torch.isfinite(a).all() and bool((a.abs()<=1).all()),'Convex bounded action; never clip')
                            v=adapter.step(a);raw=v['raw_interval'];rep=v['replay_transition'];need(k in sink.intervals,'fsynced raw before replay/labels')
                            complete=bool(raw['complete_H20ms']);finite=bool(raw['finite']);physical_done=raw['termination']!='COMPLETE_INTERVAL' or not complete
                            stop=task.termination(v['state58'],m0) if complete and finite else None
                            task_done=bool(stop['predicted_loss_of_hold'].item()) if stop is not None else False
                            timeout=bool(stop['reference_timeout'].item()) if stop is not None else False
                            trunc=bool(k==19 and not physical_done and not task_done and not timeout);terminated=bool(physical_done or task_done);done=bool(terminated or trunc or timeout)
                            mem1=task.flatten_memory(adapter.memory).numpy();nextobs=task.observation(adapter.x,adapter.memory,geometry).numpy() if complete and finite else np.full(222,np.nan,dtype=np.float32)
                            reward=float(v['reward']['total'].item());meta=sink.intervals[k];sid=len(rows)
                            decisions=dict(memory_before83_F64=memory0.numpy(),memory_after83_F64=mem1,obs222_FP32=obs.numpy(),next_obs222_FP32=nextobs,
                                actor_action20_FP32=actor_action.numpy(),action20_F64=a.numpy(),residual20_F64=rep['c'][0].numpy(),actual_u20_F64=rep['u'][0].numpy(),
                                exploration_profile20_F64=curve[k],reward_F64=np.array(reward),done_bool=np.array(done),terminated_bool=np.array(terminated),
                                truncated_bool=np.array(trunc or timeout),reader_done_bool=np.array(physical_done),task_done_bool=np.array(task_done),collect_truncated_bool=np.array(trunc),reference_timeout_bool=np.array(timeout))
                            C.save(folder/f'decision_{k:02d}.npz',decisions);C.save(folder/f'reward_{k:02d}.npz',C.tensor_arrays(v['reward'],'reward__'))
                            ident=dict(sample_id=sid,family_id=fid,episode_id=local_ep,global_episode_id=global_ep,arm=mode,profile_id=fid,step_k=k,data_role='ONLINE_TRAIN_ONLY_NEW_STAGE',
                                split='ONLINE_TRAIN',dt_s=h,requested_substeps=r.n,executed_substeps=raw['returned_substeps'],complete_H20ms=complete,
                                trace=meta['raw'],raw_metadata=C.pin(folder/f'interval_{k:02d}.json'),decision=C.pin(folder/f'decision_{k:02d}.npz'),
                                source_receipt=meta['source_receipt'],source_after_action_receipt=meta['source_after_action_receipt'],target_receipt=meta['target_receipt'],
                                actual_control20_raw_sha256=C.sha(raw['actual_control20']),warning_numbers=raw['warning_numbers'],termination=raw['termination'],
                                task_done=task_done,reader_done=physical_done,terminated=terminated,done=done,collect_truncated=trunc,reference_timeout=timeout)
                            # Labels read only after durable raw. Original future-H decoder/codec; no native calls.
                            arrays=dict(raw['trace'],source_state58=raw['source_state58'],actual_control20=raw['actual_control20'],
                                observed_prefix_target58=np.asarray(raw['target_state58']),target_at_H_state58=np.asarray(raw['target_state58']) if complete else np.full(58,np.nan),
                                physical_world_delta52=raw['physical_world_delta52'],source_integration_state=raw['source_receipt']['integration_state'],
                                source_after_action_integration_state=raw['source_after_action_receipt']['integration_state'],target_integration_state=raw['target_receipt']['integration_state'],
                                action_normalized20_F64=a.numpy(),complete_H_bool=np.array(complete),reward_scalar_F64=np.array(reward),task_done=task_done,reader_done=physical_done)
                            arrays.update(decisions)
                            row,note,key=canonical_online_row(D,arrays,ident,ep,bundle,physical,maths,prealive)
                            row.update(arm_i64=ai,online_global_episode_id_i64=global_ep,profile_id_i64=fid,collection_cycle_i64=p['collection_cycle'],actor_source_stage_id_i64=p['actor_input']['stage_id'],actor_action20_FP32=actor_action.numpy())
                            for rewardmode,rewardtask in tasks.items():
                                row['reward_'+rewardmode+'_F64']=float(rewardtask.rewards(v['state58'],m0,a)['total'].item()) if complete and finite else float('nan')
                            C.save(folder/f'canonical_{k:02d}.npz',{n:np.asarray(w) for n,w in row.items()});note['canonical_row']=C.pin(folder/f'canonical_{k:02d}.npz')
                            note['source_identity']=ident;rows.append(row);notes.append(note);ep['interval_row_ids'].append(sid)
                            aliases[key+(r.shared_physics_digest,)].append(sid)
                            if ep['first_action'] is None:ep['first_action']=dict(actor_FP32_sha256=C.sha(actor_action.numpy()),mixed_F64_sha256=C.sha(a.numpy()),actual_u_F64_sha256=C.sha(raw['actual_control20']),source_receipt=meta['source_receipt'])
                            prealive=bool(prealive and not row['actor_absorbing_done_bool'])
                            if done:
                                ep['termination']='TASK_POST_H_LOSS' if task_done else raw['termination'] if physical_done else 'REFERENCE_TIMEOUT' if timeout else 'COLLECTION20H_TRUNCATION'
                                adapter.closed=True;break
                        episodes.append(ep);raw_calls+=sink.calls
                        print(json.dumps(dict(arm=mode,profile=fid,dt_s=h,rows=len(ep['interval_row_ids']),termination=ep['termination'])),flush=True)
        phase='final_raw_pool'
        after={mode:dict(state_digest=tensor_tree_digest(a.state_dict()),norm_digest=tensor_tree_digest(a.norm.state_dict()),norm_host={k:v for k,v in vars(a.norm).items() if isinstance(v,(bool,int,float,str))}) for mode,a in actors.items()}
        need(before==after,'Read-only Actor weights/buffers/host norm throughout collection')
        C.save(out/'actor_normalizers_after.npz',{mode+'__'+k:v.detach().cpu().numpy() for mode,a in actors.items() for k,v in a.norm.state_dict().items()})
        rng_after=dict(python=random.getstate(),numpy=np.random.get_state(),torchCPU=torch.get_rng_state().numpy())
        need(rng_before['python']==rng_after['python'] and rng_before['numpy'][0]==rng_after['numpy'][0] and np.array_equal(rng_before['numpy'][1],rng_after['numpy'][1]) and rng_before['numpy'][2:]==rng_after['numpy'][2:] and C.same(rng_before['torchCPU'],rng_after['torchCPU']),'No stochastic inference/exploration RNG mutation')
        C.save(out/'RNG_after.npz',dict(torchCPU=rng_after['torchCPU'],numpy_keys=rng_after['numpy'][1]))
        for i,row in enumerate(rows):
            nxt=rows[i+1] if i+1<len(rows) else None
            row['H2_start_eligible_bool']=bool(nxt is not None and row['Actor_start_eligible_bool'] and nxt['Actor_start_eligible_bool'] and not row['actor_absorbing_done_bool'] and row['online_global_episode_id_i64']==nxt['online_global_episode_id_i64'] and nxt['step_k_i64']==row['step_k_i64']+1)
        allarrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};C.save(out/'common_pool.npz',allarrays)
        partitions={}
        for ai,mode in enumerate(MODES):
            mask=allarrays['arm_i64']==ai;C.save(out/(mode+'.npz'),{k:v[mask] for k,v in allarrays.items()});partitions[mode]=C.pin(out/(mode+'.npz'))
        groups=[dict(sample_ids=ss,arms=sorted({notes[i]['source_identity']['arm'] for i in ss}),profiles=sorted({notes[i]['source_identity']['profile_id'] for i in ss})) for ss in aliases.values() if len(ss)>1]
        C.write(out/'sample_identities.json',notes);C.write(out/'episodes.json',episodes);C.write(out/'alias_groups.json',dict(exact_full_afteraction_action_groups=groups,no_silent_deduplication=True,cohort_key='protocolSHA/profile_id; both arms and dts remain grouped; no holdout claim'))
        counts={k:sum(r.counts[k] for r in readers.values()) for k in next(iter(readers.values())).counts};audits={str(h):r.audit() for h,r in readers.items()}
        need(len(episodes)==16 and len(rows)<=320 and counts['integration_steps']<=30720 and counts['observer_forward']==0,'Matched finite native budget')
        need(all(a['passed'] is True and a['model_pin']==p['inputs']['model_mjb'] for a in audits.values()),'Actual shared model audits pass before replay qualification')
        perarm={mode:dict(episodes=sum(e['arm']==mode for e in episodes),rows=sum(r['arm_i64']==ai for r in rows),native_steps=sum(int(r['executed_substeps_i64']) for r in rows if r['arm_i64']==ai),planned_maximum_H=160,unused_H_budget=160-sum(r['arm_i64']==ai for r in rows)) for ai,mode in enumerate(MODES)}
        need(all(v['episodes']==8 and v['native_steps']<=15360 for v in perarm.values()),'Each arm same maximum budget, no resampling after failure')
        C.write(out/'read_only_actor_evidence.json',dict(before=before,after=after,own_checkpoint=p['actor_input']['checkpoint'],no_parameter_or_norm_update=True))
        eligible_counts={mode:dict(WM_rows=sum(bool(r['WM_eligible_bool']) for r in rows if r['arm_i64']==ai),Actor_rows=sum(bool(r['Actor_start_eligible_bool']) for r in rows if r['arm_i64']==ai),H2_rows=sum(bool(r['H2_start_eligible_bool']) for r in rows if r['arm_i64']==ai)) for ai,mode in enumerate(MODES)}
        ready=bool(sum(v['WM_rows'] for v in eligible_counts.values()) and all(v['Actor_rows'] for v in eligible_counts.values()))
        dataset=dict(schema=POOL,status='COMPLETE_ONLINE_COMMON_POOL_NOT_LEARNING',data_training_ready=ready,eligibility_counts=eligible_counts,protocol=C.pin(protocol_path),paired_checkpoint=p['actor_input']['checkpoint'],actor_input=p['actor_input'],collection_cycle=p['collection_cycle'],offline1000_root=L.root_metadata(p)['checkpoint'],task_bundle=p['inputs']['task_bundle'],
            common_pool=C.pin(out/'common_pool.npz'),arm_partitions=partitions,field_map=dict(E.FIELD_MAP,episode_id='online_global_episode_id_i64'),mode_reward_keys={m:'reward_'+m+'_F64' for m in MODES},
            sample_identities=C.pin(out/'sample_identities.json'),alias_review=C.pin(out/'alias_groups.json'),rows=len(rows),partial_rows=sum(not r['complete_H_bool'] for r in rows),
            eligibility='WM complete finite no-warning; Actor additionally pre-boundary alive. Terminal endpoint retained, H2 mask excludes crossing episode/taskfailure. Newstage train-only, all arms pooled; no heldout split.',
            future_labels_only=['impulse22x6','meanFn22','coverage22','triple_coverage'],offline1000_training_budget_unchanged=True,new_training_authorized=False)
        C.write(out/'dataset_report.json',dataset)
        initial_alias=defaultdict(list);state_alias=defaultdict(list)
        for ep in episodes:initial_alias[(ep['initial']['receipt']['sha256'],ep['dt_s'])].append(ep['global_episode_id'])
        for note in notes:
            ident=note['source_identity'];state_alias[(ident['source_receipt']['sha256'],ident['dt_s'])].append(ident['sample_id'])
        initial_groups=[dict(global_episode_ids=v,receipt_sha256=k[0],dt_s=k[1]) for k,v in initial_alias.items() if len(v)>1]
        state_groups=[dict(sample_ids=v,receipt_sha256=k[0],dt_s=k[1]) for k,v in state_alias.items() if len(v)>1]
        report=dict(schema=RESULT,status='COMPLETED_MATCHED_ONLINE_CPU_COLLECTION_NOT_POLICY_QUALIFICATION',protocol=C.pin(protocol_path),dataset_report=C.pin(out/'dataset_report.json'),
            task_bundle=p['inputs']['task_bundle'],checkpoint=p['actor_input']['checkpoint'],health=p['actor_input']['health'],actor_input=p['actor_input'],collection_cycle=p['collection_cycle'],actor_checkpoint_kind=p['actor_input']['kind'],actor_training_protocol=p['actor_input']['training_protocol'],offline1000_root=L.root_metadata(p)['checkpoint'],collector_sources=L.subset(p),profile_plan=p['inputs']['profile_plan'],episodes=16,rows=len(rows),
            per_arm=perarm,counts=counts,Actor_FW=actor_fw,NN_constructors=4,unused_Critic_constructors=2,Critic_FW=0,WM_FW=0,own_checkpoint_loads=1,training_updates=0,
            raw_sink_calls=raw_calls,episode_records=C.pin(out/'episodes.json'),model_audits=audits,initial_receipt_alias_groups=initial_groups,source_state_alias_groups=state_groups,read_only_actor_evidence=C.pin(out/'read_only_actor_evidence.json'),
            source_scope='Exact reused Reader and raw sink/label arithmetic; no mutation of offline protocols/data. Model uses current observation only. Future contacts are labels, never Actor inputs.',
            evaluation_scope='0.4s local policy exploration with POST20ms task termination; not full2s tickwise holding evaluation, no independent ego or benefit claim.',elapsed_s=time.perf_counter()-start)
        C.write(out/'report.json',report);return report
    except BaseException as e:
        forensic={}
        for h,r in readers.items():
            if getattr(r,'d',None) is not None:
                d=r.d;a={k:np.array(getattr(d,k),copy=True) for k in ['qpos','qvel','ctrl','qacc','qacc_warmstart','qfrc_applied','xfrc_applied']};rr=r.receipt();a['integration_state']=rr['integration_state'];a['warning_numbers']=np.array(d.warning.number,copy=True)
                path=out/f'failure_native_{h}.npz';C.save(path,a);forensic[str(h)]=dict(raw=C.pin(path),receipt=C.receipt_meta(rr))
        C.write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),completed_rows=len(rows),episodes=episodes,counts={str(h):r.counts for h,r in readers.items()},raw_failure_snapshots=forensic,automatic_retry=False));raise

def load_common_pool(dataset_report_pin,*,reward_mode):
    """NumPy data-loop API for a separately authorized next-stage trainer."""
    need(reward_mode in MODES,'Named A1/A2 reward');d=read(checked(dataset_report_pin))
    need(d['schema']==POOL and d['data_training_ready'] is True,'Completed online replay only')
    with np.load(checked(d['common_pool']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return dict(arrays=a,canonical={k:a[v] for k,v in d['field_map'].items()},field_map=d['field_map'],reward_key=d['mode_reward_keys'][reward_mode],
        WM_rows=np.flatnonzero(a['WM_eligible_bool']),Actor_rows=np.flatnonzero(a['Actor_start_eligible_bool']),
        H2_rows=np.flatnonzero(a['H2_start_eligible_bool']),raw_source=d,no_fitting_performed=True)


def make_common_replay(dataset_report_pin,*,reward_mode,ReplayBuffer,device,torch):
    """Populate upstream replay from durable canonical arrays; performs no fit.

    Raw partial NaNs remain in the pinned NPZ. Buffer.nan_to_num cannot change
    validity masks. Global episode IDs avoid crossing arm/profile boundaries.
    """
    view=load_common_pool(dataset_report_pin,reward_mode=reward_mode);d=view['canonical'];raw=view['arrays']
    buffer=ReplayBuffer(max(4096,len(d['row_id'])),1,device,holdout_fraction=0.)
    for i in range(len(d['row_id'])):
        values=dict(x=d['x58'][i],x1=d['y58'][i],u=d['u20'][i],m=d['memory83'][i],m1=d['next_memory83'][i],
            obs=d['obs222'][i],obs1=d['next_obs222'][i],action=d['action20'][i],reward=raw[view['reward_key']][i],
            valid=d['WM_eligible'][i],actor_start=d['Actor_start_eligible'][i],
            done=bool(d['actor_absorbing_done'][i] or d['acquisition_done'][i] or not d['complete_H'][i]),
            terminated=d['actor_absorbing_done'][i],reference_timeout=d['reference_timeout'][i],
            episode_id=d['episode_id'][i],row_id=d['row_id'][i],arm=raw['arm_i64'][i],profile=raw['profile_id_i64'][i])
        fields={}
        for key,value in values.items():
            a=np.asarray(value);dtype=torch.bool if a.dtype.kind=='b' else torch.int64 if a.dtype.kind in 'iu' else torch.float32 if key in ['obs','obs1'] else torch.float64
            fields[key]=torch.as_tensor(value,device=device,dtype=dtype).unsqueeze(0)
        buffer.add(fields)
    return dict(buffer=buffer,WM_row_indices=view['WM_rows'],Actor_row_indices=view['Actor_rows'],H2_row_indices=view['H2_rows'],source=view['raw_source'],training_updates=0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if args.run:r=collect_round(args.protocol);print(json.dumps(dict(report=pin(Path(read(args.protocol)['output'])/'report.json'),rows=r['rows'])))
    else:
        p,b,out,pending=metadata_preflight(args.protocol)
        print(json.dumps(dict(status='SOURCE_METADATA_ONLY_NO_TORCH_NATIVE_NPZ_VALUES',pending_final_checkpoint=pending,maximum_rows=320,maximum_native_steps=30720,output=str(out))))
