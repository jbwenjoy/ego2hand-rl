"""Loaded-tripod real PhoneReaderTaskBridge collection. Default JSON/hash only.

No Actor/WM/optimizer is constructed. Torch is used only for frozen task/FK/
memory APIs after CPU authorization. The original Reader owns every native tick.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,importlib.util,json,math,os,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent
SCHEMA='phone_tripod_task_acquisition_CPU_protocol:1'
RESULT='phone_tripod_task_acquisition_result:1'
TRAIN='TRIPOD_TRAIN';MONITOR='TRIPOD_MONITOR'
INPUTS={'task_bundle','task_protocol','family_plan','model_mjb','geometry_catalog','bridge_qualification'}
RULES=dict(families=16,episodes=32,intervals_per_episode=20,H_s=.02,native_dt_s=[.0003125,.00015625],substeps=[64,128],maximum_native_steps=61440,maximum_rows=640,task_mode='A1_tracking',first_seed=4302000,TRAIN_family_ids=list(range(12)),MONITOR_family_ids=list(range(12,16)),normalized_action='F0zero;other15 PCG64seed4302000+i fixed random phase lowfrequency sin plus half-frequency sin; tanh no clipping',EMA_alpha=.5,POST_H_task_COM_loss_m=.05,POST_H_task_angle_loss_rad=math.pi/4,no_NN_training_GPU=True)
EXPECTED=dict(task_bundle='b61c4e1f6488a016516199a30dea7ebbf70cc0b7321dfdc86cf4e6c876f166a5',task='1d069f19760cee674abd94e76166cff4aee89b4eab2a3f42b54ce43308d7acaf',reader_replay='266c4b0609193601e1ec8fbdecf43b04ac8dfc37a817fea6a1bdc6a80462c6cd',native_reader='877aa4c5d51bff09e0a6824804eef3f2b335a8dbda26dcc3d0235bb3af57b841',initial_probe='8396acc6099dc3a812f9a20a2a2bf8df8702c802dc3bb2d7a71eada752e5ad15')

def need(v,msg):
    if not v:raise ValueError(msg)
def pin(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes();return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):
    p=Path(v['path']);need(p.is_absolute() and pin(p)==v,'Changed absolute pin '+str(p));return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def same(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [clean(a) for a in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(clean(value),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def save(path,arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
def tensor_arrays(value,prefix=''):
    if isinstance(value,dict):
        result={}
        for k,v in value.items():result.update(tensor_arrays(v,prefix+str(k)+'__'))
        return result
    if hasattr(value,'detach'):return {prefix[:-2]:value.detach().cpu().numpy()}
    if isinstance(value,np.ndarray):return {prefix[:-2]:value}
    return {}
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def receipt_meta(v):return {k:a for k,a in v.items() if k!='integration_state'}|dict(dtype=v['integration_state'].dtype.str,shape=list(v['integration_state'].shape))
def action_curve(fid):
    if fid==0:return np.zeros((20,20),dtype=np.float64)
    rng=np.random.Generator(np.random.PCG64(4302000+fid));freq=rng.uniform(.25,1.25,20);amp=rng.uniform(.25,1.25,20)
    phase=rng.uniform(-np.pi,np.pi,20);phase2=rng.uniform(-np.pi,np.pi,20);t=(np.arange(20,dtype=np.float64)+.5)*.02
    return np.tanh(amp*np.sin(2*np.pi*t[:,None]*freq+phase)+.15*np.sin(np.pi*t[:,None]*freq+phase2))
def planned_controls(actions,bundle):
    scale=np.asarray(bundle['action']['symmetric_residual_scale20_rad'],dtype=np.float64);base=np.asarray(bundle['initial']['ctrl20_F64'],dtype=np.float64)
    c=np.zeros(20,dtype=np.float64);res=[];commands=[]
    for a in actions:c=.5*scale*a+.5*c;res.append(c.copy());commands.append(base+c)
    return np.asarray(res),np.asarray(commands)
def make_family_plan(bundle):
    families=[]
    for fid in range(16):
        a=action_curve(fid);c,u=planned_controls(a,bundle)
        families.append(dict(family_id=fid,split=TRAIN if fid<12 else MONITOR,seed=4302000+fid,action20x20_F64=a.tolist(),action_raw_sha256=sha(a),residual20x20_F64=c.tolist(),ctrl20x20_F64=u.tolist(),ctrl_raw_sha256=sha(u),first_ctrl_raw_sha256=sha(u[0])))
    return dict(schema='phone_loaded_tripod16_control_family_plan:1',rules=RULES,families=families,same_initial_lineage='exact_source_probe030_fixed_loaded_q',same_initial_is_known_state_not_unseen=True,WM_outputs_used_for_selection=False)

def metadata_preflight(path):
    p=read(path);need(p['schema']==SCHEMA and p['rules']==RULES and set(p['inputs'])==INPUTS,'Exact new-task collection contract')
    need(type(p['CPU_run_authorized']) is bool and p['NN_training_GPU_authorized'] is False and p['automatic_retry'] is False,'Explicit CPU-only collection')
    for v in [*p['inputs'].values(),*p['sources'].values()]:checked(v)
    need(p['sources']['collector']==pin(__file__),'Own collector source')
    for name in ['task','reader_replay','native_reader']:need(p['sources'][name]['sha256']==EXPECTED[name],'Frozen bridge sources')
    need(p['inputs']['task_bundle']['sha256']==EXPECTED['task_bundle'],'Frozen task bundle2')
    b=read(checked(p['inputs']['task_bundle']));need(b['initial']['source_probe']['sha256']==EXPECTED['initial_probe'],'Exact/probe030 only')
    checked(b['initial']['source_probe']);need(b['model']==p['inputs']['model_mjb'],'Original model only')
    for key,receipt_key in [('qpos27_F64','qpos27'),('qvel26_F64','qvel26'),('ctrl20_F64','ctrl20')]:need(sha(np.asarray(b['initial'][key],dtype=np.float64))==b['initial']['raw_qpos_qvel_ctrl_receipt'][receipt_key]['sha256'],'Exact named initial raw bytes')
    need(b['action']['alpha']==.5 and b['action']['clamp'] is False and b['task_designated_geom_ids']==[15,28,41],'Actual EMA and designated tripod roles')
    parent=read(checked(p['inputs']['task_protocol']));need(parent['task_bundle']==p['inputs']['task_bundle'],'Task protocol bundle lineage')
    for k in ['task','reader_replay','native_reader','directional']:need(parent['sources'][k]==p['sources'][k],'Task protocol source lineage')
    qual=read(checked(p['inputs']['bridge_qualification']));need(qual['schema']=='phone_loaded_tripod_native_reader_bridge_result:1' and qual['status']=='PASS_TWO_ACTUAL_ONE_H_INTERFACES' and qual['passed']==qual['total']==25 and all(v is True for v in qual['checks'].values()),'Existing actual2dt bridge qualification')
    qp=read(checked(qual['protocol']));need(qp['inputs']['bundle']==p['inputs']['task_bundle'] and qp['inputs']['model']==p['inputs']['model_mjb'],'Bridge qualification same bundle/model')
    for key in ['task','reader_replay','native_reader','directional']:need(qp['sources'][key]==p['sources'][key],'Actually qualified bridge implementations')
    plan=read(checked(p['inputs']['family_plan']));expected=make_family_plan(b);need(plan==expected,'Literal fixed seed families, no WM selection')
    lo=np.asarray(b['action']['target_lower20']);hi=np.asarray(b['action']['target_upper20']);first=[]
    for f in plan['families']:
        a=np.asarray(f['action20x20_F64']);u=np.asarray(f['ctrl20x20_F64']);need(a.shape==u.shape==(20,20) and np.isfinite(a).all() and np.all(abs(a)<1),'Finite bounded tanh20 no clipping')
        need(np.all(u>=lo)&np.all(u<=hi),'Exact EMA curves remain inside true control intersection')
        need(sha(a)==f['action_raw_sha256'] and sha(u)==f['ctrl_raw_sha256'],'Planned bytes')
        first.append(f['first_ctrl_raw_sha256'])
    need(len(set(first))==16 and np.all(np.asarray(plan['families'][0]['action20x20_F64'])==0),'F0 zero and first actions distinct across partitions')
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==H and not out.exists(),'Fresh local acquisition output')
    return p,b,plan,out

class PersistentRawSink:
    def __init__(self,folder,episode_id):self.folder=folder;self.episode_id=episode_id;self.k=None;self.initial=None;self.intervals={};self.calls=0
    def __call__(self,v):
        self.calls+=1
        if v['schema']=='phone_actor_native_reset_receipt:1':
            need(self.initial is None,'One raw reset sink per episode');r=v['reset'];path=self.folder/'initial.npz'
            arrays={k:np.asarray(a) for k,a in r.items() if isinstance(a,np.ndarray)};arrays['integration_state']=r['receipt']['integration_state']
            save(path,arrays);meta=dict(schema=v['schema'],episode_id=self.episode_id,source_initial=v['source_initial'],receipt300=receipt_meta(r['receipt']),contact_phase=r['contact_phase'],raw_arrays=pin(path))
            write(self.folder/'initial.json',meta);self.initial=dict(arrays=pin(path),metadata=pin(self.folder/'initial.json'),receipt=meta['receipt300']);return
        need(v['schema']=='phone_actor_raw_transition_sidecar:1' and self.k is not None and self.k not in self.intervals,'Known step raw sink exactly once')
        r=v['interval'];contacts=v['tripod_contacts'];path=self.folder/f'interval_{self.k:02d}.npz'
        arrays=dict(r['trace'])|dict(source_state58=r['source_state58'],observed_prefix_target58=v['observed_prefix_state58'],target_at_H_state58=v['target_H58'],physical_world_delta52=r['physical_world_delta52'],actual_control20=r['actual_control20'],source_integration_state=r['source_receipt']['integration_state'],source_after_action_integration_state=r['source_after_action_receipt']['integration_state'],target_integration_state=r['target_receipt']['integration_state'],memory_before83_F64=v['memory_before'],action_normalized20_F64=v['action_normalized20'],residual20_F64=v['residual20'],tripod_pair_positive_Fn22_N=contacts['per_tick_pair_positive_Fn22_N'],tripod_pair_generated22=contacts['per_tick_pair_generated22'],tripod_pair_coverage22=contacts['pair_coverage22'],tripod_pair_mean_positive_Fn22_N=contacts['pair_mean_positive_Fn22_N'])
        save(path,arrays)
        meta=dict(schema=v['schema'],episode_id=self.episode_id,step_k=self.k,raw=pin(path),source_receipt=receipt_meta(v['source_receipt']),source_after_action_receipt=receipt_meta(v['after_action_receipt']),target_receipt=receipt_meta(v['target_receipt']),complete_H=v['complete_H'],valid=v['valid'],termination=r['termination'],force_phase=r['force_phase'],original_two_pad_summary=contacts['original_two_pad_summary'],triple_simultaneous_coverage=contacts['triple_simultaneous_coverage'],actual_contact_known=True,proxy_is_observed_contact=False,warning_numbers=r['warning_numbers'])
        write(self.folder/f'interval_{self.k:02d}.json',meta);self.intervals[self.k]=meta

def execute(protocol_path):
    protocol_path=Path(protocol_path).resolve(strict=True);p,b,plan,out=metadata_preflight(protocol_path);need(p['CPU_run_authorized'] is True,'Root-only actual native collection')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CPU only')
    out.mkdir();write(out/'protocol.json',p);started=time.perf_counter();readers={};rows=[];identities=[];episodes=[];phase='construct_parameterfree_bridge'
    try:
        taskmod=mod('tripod_collect_task',checked(p['sources']['task']));bridge=mod('tripod_collect_bridge',checked(p['sources']['reader_replay']))
        reader=mod('tripod_collect_reader',checked(p['sources']['native_reader']));gap=mod('tripod_collect_gap',checked(p['sources']['directional']))
        geometry=gap.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']));task=taskmod.PhoneTask(checked(p['inputs']['task_bundle']),mode='A1_tracking')
        for h in RULES['native_dt_s']:readers[h]=reader.NativePhoneReader(checked(p['inputs']['model_mjb']),dt_s=h)
        need(len({r.shared_physics_digest for r in readers.values()})==1,'Same original physical parameters acrossdt')
        aliases=defaultdict(list);initial_aliases=defaultdict(list);raw_sink_calls=0
        with torch.inference_mode():
            for family in plan['families']:
                fid=family['family_id'];actions=np.asarray(family['action20x20_F64']);planned_u=np.asarray(family['ctrl20x20_F64'])
                for hi,h in enumerate(RULES['native_dt_s']):
                    ep=2*fid+hi;phase=f'family{fid}_dt{hi}';folder=out/f'episode_{ep:03d}';folder.mkdir()
                    sink=PersistentRawSink(folder,ep);r=readers[h];adapter=bridge.PhoneReaderTaskBridge(r,task,geometry,sink);adapter.episode_id=ep
                    x,obs0=adapter.reset();save(folder/'initial_task.npz',dict(state58_F64=x.numpy(),memory83_F64=task.flatten_memory(adapter.memory).numpy(),obs222_FP32=obs0.numpy()))
                    initial_aliases[(sink.initial['receipt']['sha256'],h,r.shared_physics_digest)].append(ep)
                    e=dict(episode_id=ep,family_id=fid,split=family['split'],seed=family['seed'],dt_s=h,n_substeps=r.n,initial=sink.initial,initial_task=pin(folder/'initial_task.npz'),interval_row_ids=[],termination='ACQUISITION_BUDGET_20_H_NOT_REFERENCE_TIMEOUT')
                    for k in range(20):
                        sink.k=k;m0=adapter.memory;memory0=task.flatten_memory(m0).clone();a=torch.tensor(actions[k],dtype=torch.float64)
                        v=adapter.step(a);raw=v['raw_interval'];rep=v['replay_transition'];need(k in sink.intervals,'Raw arrays fsynced before replay use')
                        need(same(raw['actual_control20'],planned_u[k]),'Actual frozen EMA control bytes equal planned command; no clamp')
                        complete=bool(raw['complete_H20ms']);finite=bool(raw['finite']);physical_done=raw['termination']!='COMPLETE_INTERVAL' or not complete
                        term=task.termination(v['state58'],m0) if finite else None
                        task_done=bool(term['predicted_loss_of_hold'].item()) if complete and finite else False
                        reference_timeout=bool(term['reference_timeout'].item()) if complete and finite else False
                        collect_truncated=bool(k==19 and not physical_done and not task_done and not reference_timeout)
                        terminated=bool(physical_done or task_done);truncated=bool(collect_truncated or reference_timeout);done=terminated or truncated
                        valid=bool(rep['valid'][0].item());sid=len(rows);mem1=task.flatten_memory(adapter.memory).numpy()
                        obs=rep['obs'][0].numpy();nextobs=task.observation(adapter.x,adapter.memory,geometry).numpy() if complete and finite else np.full(222,np.nan,dtype=np.float32)
                        reward=float(v['reward']['total'].item());trip=bridge.regroup_interval_contacts(raw);fn=trip['per_tick_pair_positive_Fn22_N'];padcols=[bridge.PAIRS.index(g) for g in bridge.PADS]
                        noncols=[j for j,g in enumerate(bridge.PAIRS) if g not in (*bridge.PADS,0)]
                        decision=dict(memory_before83_F64=memory0.numpy(),memory_after83_F64=mem1,obs222_FP32=obs,next_obs222_FP32=nextobs,obs222_F64=obs.astype(np.float64),next_obs222_F64=nextobs.astype(np.float64),action20_F64=a.numpy(),residual20_F64=rep['c'][0].numpy(),actual_u20_F64=rep['u'][0].numpy(),reward_F64=np.asarray(reward),memory_after_H_known_bool=np.asarray(complete and finite),done_bool=np.asarray(done),terminated_bool=np.asarray(terminated),truncated_bool=np.asarray(truncated),reader_done_bool=np.asarray(physical_done),task_done_bool=np.asarray(task_done),collect_truncated_bool=np.asarray(collect_truncated),reference_timeout_bool=np.asarray(reference_timeout))
                        decision.update({'replay__'+key:val.numpy() for key,val in rep.items()});save(folder/f'decision_{k:02d}.npz',decision)
                        reward_arrays=tensor_arrays(v['reward'],'reward__');save(folder/f'reward_{k:02d}.npz',reward_arrays)
                        row=dict(x58_F64=raw['source_state58'],u20_F64=raw['actual_control20'],y58_F64=np.asarray(raw['target_state58']) if complete else np.full(58,np.nan),target_delta52_F64=raw['physical_world_delta52'],native_dt_s_F64=h,n_substeps_i64=r.n,executed_substeps_i64=raw['returned_substeps'],complete_H_bool=complete,physical_training_eligible_bool=bool(raw['training_eligible']),valid_bool=valid,training_eligible_bool=valid,row_id_i64=sid,family_id_i64=fid,episode_id_i64=ep,step_k_i64=k,memory_before83_F64=memory0.numpy(),memory_after83_F64=mem1,obs222_FP32=obs,next_obs222_FP32=nextobs,obs222_F64=obs.astype(np.float64),next_obs222_F64=nextobs.astype(np.float64),action20_F64=a.numpy(),residual20_F64=rep['c'][0].numpy(),reward_F64=reward,done_bool=done,terminated_bool=terminated,truncated_bool=truncated,reader_done_bool=physical_done,task_done_bool=task_done,collect_truncated_bool=collect_truncated,reference_timeout_bool=reference_timeout,memory_after_H_known_bool=complete and finite,pair_coverage22_F64=trip['pair_coverage22'],pair_mean_positive_Fn22_F64=trip['pair_mean_positive_Fn22_N'],triple_coverage_F64=trip['triple_simultaneous_coverage'],tripod_nonpad_max_positive_Fn_F64=float(fn[:,noncols].sum(1).max()),table_max_positive_Fn_F64=float(fn[:,bridge.PAIRS.index(0)].max()),split=family['split'])
                        rows.append(row);e['interval_row_ids'].append(sid)
                        meta=sink.intervals[k]
                        identity=dict(sample_id=sid,family_id=fid,episode_id=ep,step_k=k,split=family['split'],dt_s=h,requested_substeps=r.n,executed_substeps=raw['returned_substeps'],complete_H20ms=complete,valid=valid,physical_training_eligible=raw['training_eligible'],trace=meta['raw'],raw_metadata=pin(folder/f'interval_{k:02d}.json'),decision=pin(folder/f'decision_{k:02d}.npz'),source_receipt=meta['source_receipt'],source_after_action_receipt=meta['source_after_action_receipt'],target_receipt=meta['target_receipt'],actual_control20_raw_sha256=sha(raw['actual_control20']),reader_done=physical_done,task_done=task_done,collect_truncated=collect_truncated,reference_timeout=reference_timeout,terminated=terminated,truncated=truncated,done=done,reader_termination=raw['termination'],warning_numbers=raw['warning_numbers'],force_phase=raw['force_phase'],reward_phase='POST_H state vs reference k+1 with preadvance memory; unknown for partial',obs_operator_dtype='FP32; F64 fields are exact casts, not F64 observation arithmetic',memory_after_H_known=complete and finite)
                        identity['reward_packet_arrays']=pin(folder/f'reward_{k:02d}.npz')
                        identity['initial_raw']=sink.initial
                        identities.append(identity);key=(meta['source_after_action_receipt']['sha256'],sha(raw['actual_control20']),h,r.n,r.shared_physics_digest,meta['source_after_action_receipt']['signature']);aliases[key].append(sid)
                        if done:
                            e['termination']='TASK_POST_H_LOSS' if task_done else raw['termination'] if physical_done else 'REFERENCE_TIMEOUT' if reference_timeout else 'ACQUISITION_BUDGET_20_H_NOT_REFERENCE_TIMEOUT'
                            adapter.closed=True;break
                    raw_sink_calls+=sink.calls;episodes.append(e)
                    print(json.dumps(dict(episode=ep,family=fid,split=family['split'],rows=len(e['interval_row_ids']),steps=r.tick,termination=e['termination'])),flush=True)
        phase='write_aggregates';outputs={}
        widths=dict(x58_F64=58,u20_F64=20,y58_F64=58,target_delta52_F64=52,memory_before83_F64=83,memory_after83_F64=83,obs222_FP32=222,next_obs222_FP32=222,obs222_F64=222,next_obs222_F64=222,action20_F64=20,residual20_F64=20,pair_coverage22_F64=22,pair_mean_positive_Fn22_F64=22)
        for split,name in [(TRAIN,'train'),(MONITOR,'monitor')]:
            chosen=[v for v in rows if v['split']==split];arrays={}
            for key in rows[0] if rows else []:
                if key=='split':continue
                dtype=np.bool_ if key.endswith('_bool') else np.int64 if key.endswith('_i64') else np.float32 if key.endswith('_FP32') else np.float64
                arrays[key]=np.asarray([v[key] for v in chosen],dtype=dtype)
                if key in widths:arrays[key]=arrays[key].reshape(-1,widths[key])
            save(out/(name+'.npz'),arrays);outputs[name]=pin(out/(name+'.npz'))
        write(out/'sample_identities.json',identities);write(out/'episodes.json',episodes)
        aa=[dict(sample_ids=v,family_ids=sorted({identities[i]['family_id'] for i in v}),splits=sorted({identities[i]['split'] for i in v})) for v in aliases.values() if len(v)>1];cross=[v for v in aa if len(v['splits'])>1]
        audits={str(h):r.audit() for h,r in readers.items()};counts={k:sum(r.counts[k] for r in readers.values()) for k in next(iter(readers.values())).counts}
        need(counts['integration_steps']<=61440 and counts['observer_forward']==0 and len(episodes)==32 and len(rows)<=640,'Declared native budget')
        result=dict(schema=RESULT,status='COMPLETE_NEW_TRIPOD_TASK_CONTROLLED_COLLECTION',protocol=pin(protocol_path),task_bundle=p['inputs']['task_bundle'],family_plan=p['inputs']['family_plan'],counts=counts,raw_sink_calls=raw_sink_calls,rows=len(rows),families=16,episodes=32,planned_maximum_rows=640,unexecuted_planned_intervals=640-len(rows),outputs=outputs,sample_identities=pin(out/'sample_identities.json'),episode_records=pin(out/'episodes.json'),model_audits=audits,split_counts={name:dict(families=12 if name=='train' else 4,rows=sum(v['split']==split for v in rows),valid_rows=sum(v['split']==split and v['valid_bool'] for v in rows)) for split,name in [(TRAIN,'train'),(MONITOR,'monitor')]},alias_groups=aa,cross_partition_alias_groups=cross,partition_qualified=not cross,initial_receipt_alias_groups=[dict(episode_ids=v,dt_s=k[1]) for k,v in initial_aliases.items() if len(v)>1],same_initial_lineage_known=True,no_deduplication_or_reassignment=True,no_NN_training_GPU=True,geometry_counts=geometry.counts,partial_semantics='Raw observed prefix retained, H target/nextobs NaN, validFalse. Bridge mechanically advances post-memory even on partial; memory_after_H_knownFalse prevents use as a real H-boundary state.',termination_semantics='Task loss is POST20ms COM>.05m or angle>45deg from fixed initial. Reader tighter-tick acquisition safety remains separate. Collect0.4s truncation is not2sMDPtimeout and may bootstrap; terminated task/Reader losses cannot bootstrap.',old_two_pad_summary_preserved=True,tripod_designated_geom_ids=[15,28,41],canonical_pair_geom_ids=list(bridge.PAIRS),holding_or_independent_ego_or_policy_qualified=False,elapsed_s=time.perf_counter()-started)
        result.update(state_schema='phone_COM_world_state58:1',delta_schema='phone_physical_world_delta52:2',array_schema='phone_tripod_task_H20ms_replay_arrays:1',observation_precision='Frozen task producesFP32; F64 obs aliases are exact casts only, not F64 arithmetic.')
        write(out/'report.json',result)
        if cross:raise ValueError('Cross-partition fullinput/action aliases retained; partition qualification blocked')
        print(json.dumps(dict(report=pin(out/'report.json'),counts=counts,rows=len(rows),partition_qualified=not cross)));return 0
    except BaseException as e:
        snapshots={}
        for h,r in readers.items():
            if getattr(r,'d',None) is not None:
                d=r.d;arrays={name:np.array(getattr(d,name),copy=True) for name in ['qpos','qvel','ctrl','qacc','qacc_warmstart','qfrc_applied','xfrc_applied']}
                arrays['warning_numbers']=np.array(d.warning.number,copy=True)
                # One counted getState forensic read, no forward/contactForce/step.
                rr=r.receipt();arrays['integration_state']=rr['integration_state']
                dest=out/('failure_native_'+str(h)+'.npz');save(dest,arrays);snapshots[str(h)]=dict(arrays=pin(dest),receipt=receipt_meta(rr))
        write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),returned_rows=len(rows),episodes=episodes,counts={str(h):r.counts for h,r in readers.items()},raw_failure_snapshots=snapshots,automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if args.run:raise SystemExit(execute(args.protocol))
    p,b,plan,out=metadata_preflight(args.protocol);print(json.dumps(dict(status='METADATA_ONLY_NO_TORCH_NATIVE_NPZ',families=16,episodes=32,maximum_steps=61440,output=str(out))))
