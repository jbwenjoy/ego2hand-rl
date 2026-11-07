"""Three real collection/update cycles for matched phone A1/A2 learning.

UNFROZEN SOURCE DRAFT. Default is JSON/hash only, no Torch/native import.
Native collection runs in an isolated CPU subprocess, never the GPU process.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,random,subprocess,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
SCHEMA='phone_tripod_online_matched_learning_protocol:1'
ARTIFACT='phone_tripod_online_matched_checkpoint:1'
HEALTH='phone_tripod_online_matched_checkpoint_health:1'
MODES=['A1_tracking','A2_engineered_material_proxy']
COLLECTOR_KEYS=['task','reader_replay','directional','imagination','Actor_Critic','Actor_common','paired_trainer']
RECIPE=dict(seed=1734,cpu_threads=4,NN_dtype='float32',physical_dtype='float64',amp=False,TF32=False,
 collection_cycles=3,updates_per_cycle=100,first_healthy_global_update=10,stop_after_first_healthy=True,
 collection_profiles_per_arm=4,collection_episodes_per_cycle=16,maximum_H_per_cycle=320,maximum_native_steps_per_cycle=30720,maximum_native_steps=92160,
 WM_batch=8,WM_online_rows=4,WM_old_original_rows=2,WM_old_tripod_rows=2,Actor_batch=8,imagination_horizon=2,Actor_microbatch=2,
 WM_lr=.0001,Actor_lr=.0001,Critic_lr=.0003,optimizer='AdamW',weight_decay=0.,betas=[.9,.999],epsilon=1e-8,foreach=False,capturable=False,clip_norm=1.,
 gamma=.99,target_critic_tau=.01,normalizers='frozen_offline1000_WM_actor_critic_target_norms_no_refit',WM_loss_scales='frozen_offline1000_no_refit',
 WM_supervision='equal6physical_headblocks+equal2decoded_velocityblocks+log1p_meanFn1N+coverageBCE;absolute_only_no_CF',
 WM_sampling='4_cumulative_online+2_old_original684+2_old_tripod153_uniform_with_replacement',
 Actor_sampling='common_cumulative_online_complete_prealive_B8_with_replacement;imaginedH2_not_groundtruth_window',
 TD='per_mode_saved_real_reward+gamma*not_actor_absorbing_done*not_reference_timeout*targetCritic(nextobs);collection_truncation_bootstraps',
 shared_WM=True,shared_real_batch_indices=True,online_arms_pooled=True,checkpoint_every_cycle_end=True,
 Adam_start='fresh5_at_new_stage_only;resume_all5_no_reset',monitor_or_CHECK_training=False,old_CF_continued=False,
 stage_counter='global_paired_updates_online_only;lineage_paired=1000+global;sharedWMlineage=1100+global',
 collection_time='full_cycle100_checkpoint_only_for_next_cycle;firststep10_resume_reuses_saved_cycle1_pool',automatic_retry=False)

def need(ok,msg):
    if not ok:raise ValueError(msg)
def pin(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes();return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Current absolute source/input pin');return Path(v['path'])
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
    if isinstance(v,(tuple,list)):return [clean(a) for a in v]
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return float(v)
    return v
def write(path,v):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x',encoding='utf8') as f:json.dump(clean(v),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def metadata_preflight(path):
    p=read(path);need(p['schema']==SCHEMA and p['recipe']==RECIPE,'Fixed new online experiment')
    need(read(checked(p['recipe_pin']))['recipe']==RECIPE,'Frozen exact recipe bytes')
    for k in ['CPU_run_authorized','GPU_run_authorized','native_collection_authorized']:need(type(p[k]) is bool,'Typed actual authority')
    need(p['device']=='cpu' or (p['device'].startswith('cuda:') and p['device'][5:].isdigit()),'Explicit training device')
    for v in p['sources'].values():
        if v is not None:checked(v)
    for v in p['inputs'].values():
        if v is not None:checked(v)
    need(p['sources']['trainer']==pin(__file__),'Own online source')
    need(p['sources']['paired_trainer']['sha256']=='a1bcf9f820e00d7da4ac515dd81d45d39a2c59abd07c85319d1346db384b2439','Frozen offline helper source; never mutate its RECIPE globals')
    seed=read(checked(p['inputs']['training_seed_protocol']));need(seed['inputs']['task_bundle']==p['inputs']['task_bundle'] and seed['inputs']['geometry_catalog']==p['inputs']['geometry_catalog'],'Same physical task/geometry')
    for k in ['task','reader_replay','directional','imagination','Actor_Critic','Actor_common']:need(seed['sources'][k]==p['sources'][k],'Actual source '+k)
    missing=any(p['inputs'][k] is None for k in ['root_training_protocol','root_result','root_health','root_terminal','collector_template']) or p['sources']['collector'] is None
    if missing:need(p['CPU_run_authorized'] is False and p['GPU_run_authorized'] is False and p['native_collection_authorized'] is False,'Pending final/source is metadata only')
    else:
        tp=read(checked(p['inputs']['root_training_protocol']));r=read(checked(p['inputs']['root_result']));h=read(checked(p['inputs']['root_health']));terminal=read(checked(p['inputs']['root_terminal']))
        need(r['schema']=='phone_tripod_matched_supermimic_training_result:1' and r['paired_round']==1000 and r['shared_WM_updates']==1100 and r['status']=='COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION','Fixed completed offline root')
        need(h['checkpoint']==r['checkpoint'] and h['paired_round']==1000 and h['shared_WM_updates']==1100 and h['checks'] and all(q is True for q in h['checks'].values()),'Positive offline root health')
        checked(r['checkpoint']);need(terminal['returncode']==0 and terminal['protocol']==p['inputs']['root_training_protocol'] and terminal['source']==p['sources']['paired_trainer'],'Successful actual offline terminal/source')
        need(tp['CPU_run_authorized'] is True and (tp['device']=='cpu' and tp['GPU_run_authorized'] is False or tp['device'].startswith('cuda:') and tp['device'][5:].isdigit() and tp['GPU_run_authorized'] is True) and tp['native_online_authorized'] is False,'Actual offline root device/authority contract')
        need(tp['sources']['trainer']==p['sources']['paired_trainer'] and tp['inputs']==seed['inputs'] and tp['recipe']==seed['recipe'],'Root exact fixed lineage')
        template=read(checked(p['inputs']['collector_template']));need(template['sources']['collector']==p['sources']['collector'],'Actual v3 collector template/source')
    if p.get('resume_checkpoint') is not None:
        checked(p['resume_checkpoint']);hh=read(checked(p['resume_health']))
        need(hh['schema']==HEALTH and hh['checkpoint']==p['resume_checkpoint'] and hh['checks'] and all(q is True for q in hh['checks'].values()),'Own healthy online resume')
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==H and not out.exists(),'Fresh local online output; no retry')
    return p,out,missing

def merge_pool_reports(reports,collector):
    """Stable (pool pin, original row/episode) identities; no crosscycle aliases.

    All raw rows are retained; NaNs stay in raw source and eligibility decides
    fitting. Re-indexing changes identity only, never masks or physical values.
    """
    parts=[];rewards={m:[] for m in MODES};mapping=[];epmap={};next_ep=0;next_row=0
    for pool_index,rp in enumerate(reports):
        view=collector.load_common_pool(rp,reward_mode=MODES[0]);d={k:np.array(v,copy=True) for k,v in view['canonical'].items()};a=view['arrays'];n=len(d['row_id'])
        need(n>0 and all(len(v)==n for v in d.values()),'Complete canonical pool rows')
        origin_rows=d['row_id'].copy();origin_eps=d['episode_id'].copy()
        for i,(rid,eid) in enumerate(zip(origin_rows,origin_eps)):
            key=(rp['sha256'],int(eid))
            if key not in epmap:epmap[key]=next_ep;next_ep+=1
            d['episode_id'][i]=epmap[key];d['row_id'][i]=next_row
            mapping.append(dict(global_row_id=next_row,global_episode_id=epmap[key],pool_index=pool_index,pool=rp,original_row_id=int(rid),original_episode_id=int(eid)));next_row+=1
        for mode in MODES:
            key=view['raw_source']['mode_reward_keys'][mode];rewards[mode].append(a[key].copy())
        parts.append(d)
    merged={k:np.concatenate([a[k] for a in parts]) for k in parts[0]}
    return merged,{m:np.concatenate(v) for m,v in rewards.items()},mapping

def draw(rows,k,g,torch,device):return torch.randint(rows,(k,),generator=g,device='cpu').to(device)
def validate_online_data(d):
    n=len(d['row_id']);wm=d['WM_eligible'];actor=d['Actor_start_eligible'];complete=d['complete_H']
    need(wm.dtype==actor.dtype==complete.dtype==np.bool_ and np.all(~wm|complete) and np.all(~actor|wm),'Explicit complete/known/Actor masks, never finite-filter promotion')
    for k,s in [('x58',58),('y58',58),('u20',20),('memory83',83),('next_memory83',83),('obs222',222),('next_obs222',222),('action20',20)]:need(d[k].shape==(n,s) and np.isfinite(d[k][wm]).all(),'Eligible online shape/finite '+k)
    for k in ['impulse22x6','meanFn22','coverage22']:need(np.isfinite(d[k][wm]).all(),'Actual known supervised target '+k)
    need(np.all(d['h'][wm]*d['n'][wm]==.02) and np.all(d['executed_n'][wm]==d['n'][wm]),'Known complete20ms only')
    need(np.isnan(d['y58'][~complete]).all(),'Partial H target remains NaN')
def mixed_WM_batch(online,old,tripod,g,torch):
    ii=[draw(len(d['x']),n,g,torch,d['x'].device) for d,n in [(online,4),(old,2),(tripod,2)]]
    return {k:torch.cat([d[k][i] for d,i in zip([online,old,tripod],ii)]) for k in online}
def save_collection_protocol(p,cycle,actor_input,output):
    """Contact v3 canonical actor_input binding, populated at runtime only."""
    q=read(checked(p['inputs']['collector_template']));q['CPU_run_authorized']=True;q['output']=str(output);q['collection_cycle']=cycle;q['actor_input']=actor_input
    for k in ['root_training_protocol','root_result','root_health','root_terminal']:q['inputs'][k]=p['inputs'][k]
    need(not any(k in q['inputs'] for k in ['paired_training_protocol','paired_checkpoint','paired_health']),'v3 protocol removes obsolete paired input names')
    return q
def collect_cycle(p,out,cycle,actor_input):
    need(p['native_collection_authorized'] is True,'Native subprocess authorization')
    collector_file=checked(p['sources']['collector']);target=collector_file.parent/(Path(p['output']).name+f'_cycle{cycle:02d}')
    protocol=out/f'collection_cycle{cycle:02d}.json';q=save_collection_protocol(p,cycle,actor_input,target);write(protocol,q)
    collector=mod(f'online_collection_preflight_cycle{cycle}',collector_file)
    _,_,_,pending=collector.metadata_preflight(protocol);need(pending is False,'Actual complete v3 metadata/authority gates before CPU child')
    stdout=out/f'collection_cycle{cycle:02d}.stdout.log';stderr=out/f'collection_cycle{cycle:02d}.stderr.log'
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1';cmd=[sys.executable,'-X','utf8',str(collector_file),'--protocol',str(protocol),'--run'];start=time.perf_counter()
    with stdout.open('xb') as so,stderr.open('xb') as se:
        proc=subprocess.Popen(cmd,stdout=so,stderr=se,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));code=proc.wait()
    terminal=dict(schema='phone_tripod_online_collection_child_terminal:1',cycle=cycle,returncode=code,command=cmd,protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),elapsed_s=time.perf_counter()-start,automatic_retry=False)
    write(out/f'collection_cycle{cycle:02d}.terminal.json',terminal);need(code==0,'CPU collector failed, preserve and stop; no automatic retry')
    report=read(target/'report.json');dataset=read(checked(report['dataset_report']))
    need(report['protocol']==pin(protocol) and report['episodes']==16 and report['counts']['integration_steps']<=30720 and report['counts']['observer_forward']==0,'Actual bounded collection/source')
    need(dataset['data_training_ready'] is True and dataset['schema']=='phone_tripod_online_common_replay_dataset:1','Actual complete eligible online pool')
    return dict(cycle=cycle,protocol=pin(protocol),terminal=pin(out/f'collection_cycle{cycle:02d}.terminal.json'),report=pin(target/'report.json'),dataset_report=report['dataset_report'],actor_input=actor_input,native_steps=report['counts']['integration_steps'],Actor_FW=report['Actor_FW'])

def verify_log_segment(lp):
    with Path(lp['path']).open('rb') as f:b=f.read(lp['bytes'])
    rows=[json.loads(v) for v in b.splitlines()]
    need(len(b)==lp['bytes'] and hashlib.sha256(b).hexdigest()==lp['sha256'] and len(rows)==lp['records'] and [r['global_paired_updates'] for r in rows]==list(range(lp['global_range'][0],lp['global_range'][1]+1)),'Saved prior log prefix bytes/SHA/global range')
def verify_collections(receipts):
    need([r['cycle'] for r in receipts]==list(range(1,len(receipts)+1)) and len(receipts)<=3,'Exact cumulative collection order')
    for r in receipts:
        for k in ['protocol','terminal','report','dataset_report']:checked(r[k])
        terminal=read(checked(r['terminal']));report=read(checked(r['report']))
        need(terminal['returncode']==0 and terminal['protocol']==r['protocol'] and report['protocol']==r['protocol'] and report['dataset_report']==r['dataset_report'],'Collection raw/protocol/dataset/terminal join')
    need(sum(r['native_steps'] for r in receipts)<=92160,'Cumulative native budget, no refill')

def checkpoint_health(path,p,protocol,wm,arms,opts,grads,norm,scales,physical,generator,probeF,probeO,helper,factory,device,cycle,step,global_step,torch):
    before=helper.rng(torch);cuda=helper.cuda_rng(device,torch)
    with torch.no_grad():
        expected={'WM':wm(probeF)}
        for mode,a in arms.items():expected[mode+'_Actor']=a['actor'](probeO);expected[mode+'_Critic']=a['critic'](probeO)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=[] if device.type=='cpu' else [device.index]):clone=helper_WM.make_model(torch).to(device)
    clone.load_state_dict(ck['shared_WM']);restored={}
    for mode in MODES:
        pair=factory.make_fresh_actor_critic(seed=RECIPE['seed'],source_pins=p['sources'],device=str(device));pair['actor'].load_state_dict(ck['arms'][mode]['stateactor']);pair['critic'].load_state_dict(ck['arms'][mode]['statecritic']);restored[mode]=pair
    with torch.no_grad():
        actual={'WM':clone(probeF)}
        for mode,a in restored.items():actual[mode+'_Actor']=a['actor'](probeO);actual[mode+'_Critic']=a['critic'](probeO)
    lp=ck['log_prefix']
    with Path(lp['path']).open('rb') as f:prefix=f.read(lp['bytes'])
    logs=[json.loads(a) for a in prefix.splitlines()]
    checks=dict(schema_phase=ck['schema']==ARTIFACT and ck['stage_id']==ck['stage_cycle']==cycle and ck['update_in_cycle']==step and ck['global_paired_updates']==global_step and ck['total_lineage_paired_updates']==1000+global_step and ck['shared_WM_updates']==1100+global_step,
        all_serialized_predictions_raw=all(helper_WM.tree_equal(expected[k],actual[k],torch) for k in expected),
        raw_WM_actor_critic_target_norm=helper_WM.tree_equal(ck['shared_WM'],wm.state_dict(),torch) and helper_WM.tree_equal(ck['arms'],helper.arm_states(arms),torch),
        frozen_norm_scale_context=helper_WM.tree_equal(ck['WM_normalizers'],norm,torch) and helper_WM.tree_equal(ck['WM_loss_scales'],scales,torch) and helper_WM.tree_equal(ck['physical_context'],physical,torch),
        five_serialized_Adam_raw=all(helper_WM.tree_equal(ck['optimizers'][k],opt.state_dict(),torch) for k,opt in opts.items()),
        all_Adam_steps=all(len(opt.state)==(6 if k=='WM' else 8 if k.endswith('Actor') else 10) and all(float(s['step'])==global_step and torch.isfinite(s['exp_avg']).all().item() and torch.isfinite(s['exp_avg_sq']).all().item() for s in opt.state.values()) for k,opt in opts.items()),
        saved_named_PRE_POST_gradients_raw=helper_WM.tree_equal(ck['gradient_snapshots'],grads,torch) and all(torch.isfinite(v).all().item() for gr in grads.values() for side in ['pre_clip','post_clip'] for v in gr[side].values()),
        all_weights_finite=all(torch.isfinite(v).all().item() for module in [wm,*[a[k] for a in arms.values() for k in ['actor','critic','target_critic']]] for v in module.parameters()),
        sampler_raw=helper_WM.tree_equal(ck['sample_generator'],generator.get_state(),torch),
        raw_saved_and_current_RNG=helper_WM.tree_equal(ck['RNG'],before,torch) and helper_WM.tree_equal(before,helper_WM.rng(torch),torch) and (device.type=='cpu' or helper_WM.tree_equal(ck['CUDA_RNG'],cuda,torch) and helper_WM.tree_equal(cuda,helper.cuda_rng(device,torch),torch)),
        protocol_recipe_source_inputs=ck['protocol']==pin(protocol) and ck['recipe']==RECIPE and ck['recipe_pin']==p['recipe_pin'] and ck['sources']==p['sources'] and ck['inputs']==p['inputs'],
        immutable_log_prefix=len(prefix)==lp['bytes'] and hashlib.sha256(prefix).hexdigest()==lp['sha256'] and len(logs)==lp['records'] and [r['global_paired_updates'] for r in logs]==list(range(lp['global_range'][0],global_step+1)))
    for segment in ck['lineage_log_prefixes']:verify_log_segment(segment)
    verify_collections(ck['collections']);checked(ck['pool_identity_map'])
    checks['prior_process_log_prefixes_and_collection_pins_preserved']=True
    need(all(checks.values()),'Online serialized health failure, preserve and stop')
    health={k:ck[k] for k in ['stage_id','stage_cycle','update_in_cycle','global_paired_updates','total_lineage_paired_updates','shared_WM_updates','protocol','recipe_pin','parent_checkpoint','parent_health','offline1000_root','norm_origin_checkpoint','collector_sources','dataset_pins','lineage_log_prefixes','log_prefix']}
    health.update(schema=HEALTH,checkpoint=pin(path),checks=checks,passed=sum(checks.values()),total=len(checks),extra_NN_FW=10,extra_NN_ctors=5,own_CPU_checkpoint_reads=1)
    evidence={k+'_before':v.detach().cpu().numpy() for k,v in expected.items()}|{k+'_after':v.detach().cpu().numpy() for k,v in actual.items()}
    return health,evidence

# Bound only inside --run after explicit authorization; not a top-level import.
helper_WM=None
def run(protocol):
    global helper_WM
    p,out,missing=metadata_preflight(protocol);need(not missing and p['CPU_run_authorized'] is True and p['native_collection_authorized'] is True,'Actual Root authorization/data required')
    if p['device'].startswith('cuda:'):need(p['GPU_run_authorized'] is True,'Explicit GPU authorization')
    if p['device']=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(p['device'])
    if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    random.seed(RECIPE['seed']);np.random.seed(RECIPE['seed']);torch.random.default_generator.manual_seed(RECIPE['seed'])
    if device.type=='cuda':torch.cuda.manual_seed(RECIPE['seed'])
    out.mkdir();write(out/'protocol.json',p);start=time.perf_counter();phase='load';cycle=0;step=0;global_step=0;collections=[];grads={};timings=[]
    counts=dict(WM_FW=0,WM_backward=0,WM_optimizer_steps=0,Actor_FW=0,WM_imagination_FW=0,Critic_real_TD_FW=0,Critic_imagination_FW=0,Actor_backward_microbatches=0,Actor_optimizer_steps=0,Critic_backward=0,Critic_optimizer_steps=0)
    diagnostic=dict(own_CPU_checkpoint_loads=0,NN_constructors=0,target_critic_deepcopies=0,optimizer_constructors=0,health_NN_FW=0,health_NN_ctors=0)
    try:
        helper=mod('online_frozen_training_helpers',checked(p['sources']['paired_trainer']));helper_WM=mod('online_frozen_physical_WM',checked(p['sources']['WM_trainer']))
        codec=mod('online_fixed_structured',checked(p['sources']['structured_codec']));taskmod=mod('online_fixed_task',checked(p['sources']['task']));factory=mod('online_fixed_Actor_factory',checked(p['sources']['imagination']));gap=mod('online_fixed_geometry',checked(p['sources']['directional']));collector=mod('online_collection_source',checked(p['sources']['collector']));replaymodule=mod('online_upstream_replay',checked(p['sources']['replay']))
        sys.path.insert(0,str(Path(p['sources']['Actor_Critic']['path']).parents[3]));geometry=gap.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']))
        root_result=read(checked(p['inputs']['root_result']));root_protocol=read(checked(p['inputs']['root_training_protocol']));root_pin=root_result['checkpoint']
        root=torch.load(checked(root_pin),map_location='cpu',weights_only=False);diagnostic['own_CPU_checkpoint_loads']+=1
        need(root['schema']==helper.ARTIFACT and root['paired_round']==1000 and root['shared_WM_updates']==1100 and root['protocol']==p['inputs']['root_training_protocol'] and root['recipe']==root_protocol['recipe'] and root['inputs']==root_protocol['inputs'] and root['sources']==root_protocol['sources'],'Complete exact offline root payload')
        wm=helper_WM.make_model(torch).to(device);wm.load_state_dict(root['shared_WM']);wm.eval();diagnostic['NN_constructors']+=1
        norm=helper.move(root['WM_normalizers'],device,torch);scales=helper.move(root['WM_loss_scales'],device,torch);physical=helper.move(root['physical_context'],device,torch)
        arms={};opts={'WM':helper.make_opt(wm,RECIPE['WM_lr'],torch)}
        for mode in MODES:
            a=factory.make_fresh_actor_critic(seed=RECIPE['seed'],source_pins=p['sources'],device=str(device));target=copy.deepcopy(a['critic']).eval();target.requires_grad_(False)
            arms[mode]=dict(actor=a['actor'],critic=a['critic'],target_critic=target);diagnostic['NN_constructors']+=2;diagnostic['target_critic_deepcopies']+=1
            opts[mode+'_Actor']=helper.make_opt(a['actor'],RECIPE['Actor_lr'],torch);opts[mode+'_Critic']=helper.make_opt(a['critic'],RECIPE['Critic_lr'],torch)
        helper.restore_arm_states(arms,root['arms']);diagnostic['optimizer_constructors']=5
        tasks={mode:helper.CanonicalMemoryTask(taskmod.PhoneTask(checked(p['inputs']['task_bundle']),mode=mode,geometry=geometry)) for mode in MODES}
        sampler=torch.Generator(device='cpu');sampler.manual_seed(RECIPE['seed']+1)
        old=helper.old_rows(dict(inputs=root_protocol['inputs']),codec);old_t,oldcache=helper.encode_and_cache(old,geometry,codec,helper_WM,physical,device,torch)
        offline_report=read(checked(root_protocol['inputs']['dataset']));tripod_raw=helper.map_partition(offline_report,'train');tripod_t,tripodcache=helper.encode_and_cache(helper.physical_data(tripod_raw),geometry,codec,helper_WM,physical,device,torch)
        need(len(old_t['x'])==684 and len(tripod_t['x'])==153,'Fixed offline replay only, no monitor/CHECK/final')
        parent_cp=root_pin;parent_health=p['inputs']['root_health'];start_cycle=1;start_update=1;log_segments=[];actor_input=dict(kind='offline1000',stage_id=0,checkpoint=root_pin,health=p['inputs']['root_health'],training_protocol=p['inputs']['root_training_protocol'])
        if p.get('resume_checkpoint') is not None:
            resume=torch.load(checked(p['resume_checkpoint']),map_location='cpu',weights_only=False);diagnostic['own_CPU_checkpoint_loads']+=1
            need(resume['schema']==ARTIFACT and resume['recipe']==RECIPE and resume['recipe_pin']==p['recipe_pin'] and resume['sources']==p['sources'] and resume['inputs']==p['inputs'] and resume['offline1000_root']==root_pin,'Exact newstage resume lineage')
            need(helper_WM.tree_equal(resume['WM_normalizers'],norm,torch) and helper_WM.tree_equal(resume['WM_loss_scales'],scales,torch) and helper_WM.tree_equal(resume['physical_context'],physical,torch),'No norm/context refit on resume')
            wm.load_state_dict(resume['shared_WM']);helper.restore_arm_states(arms,resume['arms'])
            for k,opt in opts.items():opt.load_state_dict(resume['optimizers'][k])
            checked(resume['protocol']);prior_protocol=read(checked(resume['protocol']));need(prior_protocol['recipe_pin']==p['recipe_pin'] and prior_protocol['inputs']==p['inputs'] and prior_protocol['sources']==p['sources'],'Resume protocol source/input/recipe lineage')
            global_step=resume['global_paired_updates'];counts=resume['counts'];collections=resume['collections'];verify_collections(collections)
            log_segments=resume['lineage_log_prefixes']+[resume['log_prefix']]
            for segment in log_segments:verify_log_segment(segment)
            sampler.set_state(resume['sample_generator']);parent_cp=p['resume_checkpoint'];parent_health=p['resume_health']
            start_cycle=resume['stage_cycle'];start_update=resume['update_in_cycle']+1
            if start_update>100:start_cycle+=1;start_update=1
            random.setstate(resume['RNG']['python']);np.random.set_state(resume['RNG']['numpy']);torch.set_rng_state(resume['RNG']['torch'])
            if device.type=='cuda':torch.cuda.set_rng_state(resume['CUDA_RNG'],device)
            actor_input=dict(kind='online_stage',stage_id=resume['stage_cycle'],checkpoint=parent_cp,health=parent_health,training_protocol=resume['protocol'])
        setup_s=time.perf_counter()-start;logpath=out/'updates.jsonl';start_global=global_step+1
        with logpath.open('x',encoding='utf8') as log:
            for cycle in range(start_cycle,4):
                if len(collections)<cycle:
                    need(len(collections)==cycle-1 and start_update==1,'No skipped or duplicate collection')
                    phase=f'collection_cycle{cycle}';receipt=collect_cycle(p,out,cycle,actor_input);collections.append(receipt);verify_collections(collections);write(out/f'collection_manifest_cycle{cycle}.json',dict(collections=collections,no_recollection_on_resume=True))
                else:
                    need(collections[cycle-1]['cycle']==cycle,'Exact existing saved cycle data')
                    for v in [collections[cycle-1]['dataset_report'],collections[cycle-1]['terminal']]:checked(v)
                phase=f'cache_cycle{cycle}';cache_start=time.perf_counter();pool_pins=[c['dataset_report'] for c in collections];data,rewards,mapping=merge_pool_reports(pool_pins,collector)
                validate_online_data(data)
                need(all(np.isfinite(v[data['Actor_start_eligible']&data['WM_eligible']]).all() for v in rewards.values()),'Both per-mode real rewards known at live complete starts')
                replay,eligible=helper.make_replay(data,replaymodule.ReplayBuffer,device,torch);reward_t={m:torch.tensor(v,dtype=torch.float64,device=device) for m,v in rewards.items()}
                online_t,onlinecache=helper.encode_and_cache(helper.physical_data(data),geometry,codec,helper_WM,physical,device,torch)
                need(len(online_t['x'])>0 and len(eligible)>0,'Online complete fit and liveActor rows; explicit replacement sampling')
                write(out/f'pool_identity_map_cycle{cycle}.json',mapping);map_pin=pin(out/f'pool_identity_map_cycle{cycle}.json');cycle_cache_s=time.perf_counter()-cache_start
                for step in range(start_update,101):
                    global_step+=1;need(global_step==100*(cycle-1)+step,'No skipped or repeated update');phase=f'cycle{cycle}_update{step}';tick=time.perf_counter()
                    wm.train();wm.requires_grad_(True);opts['WM'].zero_grad(set_to_none=True);batch=mixed_WM_batch(online_t,old_t,tripod_t,sampler,torch)
                    loss,wparts=helper.wm_loss(wm,batch,norm,scales,physical,codec,helper_WM,torch);grads['WM']=helper.grad_step(loss,wm,opts['WM'],f'PRE_ONLINE_WM_{global_step}',torch);counts['WM_FW']+=1;counts['WM_backward']+=1;counts['WM_optimizer_steps']+=1
                    wm.eval();wm.requires_grad_(False);real,indices=helper.sample_replay(replay,eligible,sampler,device,torch);mode_losses={}
                    for mode in MODES:
                        a=arms[mode];task=tasks[mode];optc=opts[mode+'_Critic'];opta=opts[mode+'_Actor'];optc.zero_grad(set_to_none=True)
                        with torch.no_grad():target=reward_t[mode][indices]+.99*(~real['terminated']).double()*(~real['reference_timeout']).double()*a['target_critic'](real['obs1']).double()
                        value=a['critic'](real['obs']);closs=(value-target.float()).square().mean();counts['Critic_real_TD_FW']+=2
                        grads[mode+'_Critic']=helper.grad_step(closs,a['critic'],optc,f'PRE_ONLINE_{mode}_Critic_{global_step}',torch);counts['Critic_backward']+=1;counts['Critic_optimizer_steps']+=1
                        with torch.no_grad():
                            for q,tq in zip(a['critic'].parameters(),a['target_critic'].parameters()):tq.lerp_(q,.01)
                        opta.zero_grad(set_to_none=True);aloss=0.
                        for j in range(0,8,2):
                            objective=factory.imagined_actor_objective(actor=a['actor'],critic=a['critic'],world_model=wm,task=task,geometry=geometry,codec=codec,norms=norm,physical=physical,initial_state58=real['x'][j:j+2],memory=task.unflatten_memory(real['m'][j:j+2]),horizon=2,gamma=.99,wm_step=taskmod.wm_step_current_only)
                            need(torch.isfinite(objective['actor_loss']).item(),'Finite actor loss, no retry');(objective['actor_loss']/4).backward();aloss+=float(objective['actor_loss'].detach())/4
                            counts['Actor_backward_microbatches']+=1;counts['Actor_FW']+=2;counts['WM_imagination_FW']+=2;counts['Critic_imagination_FW']+=1
                        grads[mode+'_Actor']=helper.finish_step(a['actor'],opta,f'PRE_ONLINE_{mode}_Actor_{global_step}',torch);counts['Actor_optimizer_steps']+=1;mode_losses[mode]=dict(real_TD=float(closs.detach()),imagined_Actor=aloss)
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    seconds=time.perf_counter()-tick;timings.append(seconds);row=dict(stage_cycle=cycle,update_in_cycle=step,global_paired_updates=global_step,WM_loss=float(loss.detach()),modes=mode_losses,common_online_global_row_ids=real['row_id'].detach().cpu().tolist(),update_seconds=seconds)
                    log.write(json.dumps(row)+'\n');log.flush()
                    if global_step==10 or step==100:
                        phase=f'health_cycle{cycle}_update{step}';healthstart=time.perf_counter();os.fsync(log.fileno());prefix=logpath.read_bytes();cp=out/f'checkpoint_cycle{cycle:02d}_update{step:04d}.pt'
                        payload=dict(schema=ARTIFACT,stage_id=cycle,stage_cycle=cycle,update_in_cycle=step,global_paired_updates=global_step,total_lineage_paired_updates=1000+global_step,shared_WM_updates=1100+global_step,
                            recipe=RECIPE,recipe_pin=p['recipe_pin'],protocol=pin(protocol),inputs=p['inputs'],sources=p['sources'],collector_sources={k:p['sources'][k] for k in COLLECTOR_KEYS},
                            parent_checkpoint=parent_cp,parent_health=parent_health,offline1000_root=root_pin,norm_origin_checkpoint=root_pin,task_bundle=p['inputs']['task_bundle'],dataset_pins=pool_pins,collections=collections,pool_identity_map=map_pin,
                            shared_WM=wm.state_dict(),arms=helper.arm_states(arms),WM_normalizers=norm,WM_loss_scales=scales,physical_context=physical,optimizers={k:o.state_dict() for k,o in opts.items()},gradient_snapshots=grads,
                            sample_generator=sampler.get_state(),RNG=helper_WM.rng(torch),CUDA_RNG=helper.cuda_rng(device,torch),counts=counts.copy(),lineage_log_prefixes=log_segments,log_prefix=dict(path=str(logpath),bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),records=global_step-start_global+1,global_range=[start_global,global_step]),
                            policy_interface=dict(obs_dim=222,memory_dim=83,action_dim=20,arm_fields=['stateactor','statecritic','statecritic_target','norm']),phase_contract='allnamed gradients PRE online update; all weights AFTER online update; optimizer step count online-only')
                        helper_WM.atomic_checkpoint(cp,payload,torch)
                        F=((online_t['feature_F64'][:8]-norm['input_mean'])/norm['input_std']).float();O=torch.tensor(data['obs222'][data['Actor_start_eligible']&data['WM_eligible']][:8],dtype=torch.float32,device=device)
                        health,evidence=checkpoint_health(cp,p,protocol,wm,arms,opts,grads,norm,scales,physical,sampler,F,O,helper,factory,device,cycle,step,global_step,torch)
                        diagnostic['own_CPU_checkpoint_loads']+=1;diagnostic['health_NN_FW']+=10;diagnostic['health_NN_ctors']+=5;np.savez_compressed(out/f'health_cycle{cycle:02d}_update{step:04d}.npz',**evidence)
                        mean=float(np.mean(timings[-min(25,len(timings)):]))
                        health.update(counts=counts.copy(),process_diagnostics=diagnostic.copy(),setup_s=setup_s,current_cycle_geometry_cache_s=cycle_cache_s,elapsed_s=time.perf_counter()-start,measured_update_seconds_mean_last25=mean,ETA_measured_update_count=min(25,len(timings)),remaining_learning_updates=300-global_step,ETA_learning_only_s=(300-global_step)*mean,remaining_collection_cycles=3-len(collections),ETA_collection_scope='Unmeasured remaining policy-dependent CPU episodes; prior child elapsed is descriptor, no false exact ETA; future cache/health/IO and resume_setup separately unknown',observed_collection_elapsed_s=[read(checked(c['terminal']))['elapsed_s'] for c in collections],checkpoint_health_IO_s=time.perf_counter()-healthstart,GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,device=str(device),first_healthy_pausepoint=global_step==10,stops_here=global_step==10 and RECIPE['stop_after_first_healthy'])
                        hp=out/f'health_cycle{cycle:02d}_update{step:04d}.json';write(hp,health);print(json.dumps(dict(cycle=cycle,update=step,global_online=global_step,health=health['passed'],total=health['total'],ETA_learning_s=health['ETA_learning_only_s'])),flush=True)
                        parent_cp=pin(cp);parent_health=pin(hp)
                        if global_step==10 and RECIPE['stop_after_first_healthy']:
                            write(out/'result.json',dict(schema='phone_tripod_online_matched_learning_result:1',status='STOPPED_FIRST_HEALTHY_CYCLE1_UPDATE10_RESUMABLE_TARGET3x100',checkpoint=parent_cp,health=parent_health,stage_cycle=cycle,update_in_cycle=step,global_paired_updates=global_step,counts=counts,collections=collections,policy_benefit_qualified=False));return 0
                actor_input=dict(kind='online_stage',stage_id=cycle,checkpoint=parent_cp,health=parent_health,training_protocol=pin(protocol));start_update=1
        write(out/'result.json',dict(schema='phone_tripod_online_matched_learning_result:1',status='COMPLETED_FIXED3_COLLECTION_UPDATE_CYCLES_NOT_POLICY_QUALIFICATION',checkpoint=parent_cp,health=parent_health,stage_cycle=3,update_in_cycle=100,global_paired_updates=300,counts=counts,collections=collections,policy_benefit_qualified=False));return 0
    except BaseException as e:
        write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),stage_cycle=cycle,update_in_cycle=step,global_paired_updates=global_step,collections=collections,counts=counts,automatic_retry=False));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,required=True);a.add_argument('--run',action='store_true');args=a.parse_args()
    if args.run:raise SystemExit(run(args.protocol))
    p,out,pending=metadata_preflight(args.protocol);print(json.dumps(dict(status='METADATA_ONLY_PENDING_PARENT_OR_COLLECTOR' if pending else 'METADATA_ONLY_SOURCE_READY_FOR_REVIEW',Torch_imported='torch' in sys.modules,native_NN_calls=0,output=str(out))))
