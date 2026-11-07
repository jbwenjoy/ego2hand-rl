"""Executable matched shared-WM offline SuperMimic phone training.

No top-level Torch/native import. CPUFalse is metadata only. This is a new
phone trainer, not the old PenMDP/Runner. Native online collection requires
a separately reviewed extension; this fixed recipe is offline only.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,math,os,random,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
SCHEMA='phone_tripod_sharedWM_matched_supermimic_protocol:1'
ARTIFACT='phone_tripod_sharedWM_A1A2_training_checkpoint:1'
MODES=['A1_tracking','A2_engineered_material_proxy']
FIELDS={'x58','u20','y58','delta52','memory83','next_memory83','obs222','next_obs222','action20','h','n','executed_n','complete_H','WM_eligible','Actor_start_eligible','actor_absorbing_done','reference_timeout','acquisition_done','collection_truncated','row_id','family_id','episode_id','step_k','impulse22x6','meanFn22','coverage22','triple_coverage'}
RECIPE=dict(seed=1733,cpu_threads=4,NN_dtype='float32',physical_dtype='float64',amp=False,TF32=False,
 common_WM_warmup_updates=100,paired_rounds=1000,first_healthy_round=100,checkpoint_rounds=[100,500,1000],stop_after_first_healthy=True,
 WM_batch=8,WM_new_rows=4,WM_old_rows=4,Actor_batch=8,imagination_horizon=2,Actor_microbatch=2,
 WM_lr=.0001,Actor_lr=.0001,Critic_lr=.0003,optimizer='AdamW',weight_decay=0.,betas=[.9,.999],epsilon=1e-8,foreach=False,capturable=False,clip_norm=1.,
 gamma=.99,target_critic_tau=.01,WM_losses=['equal6block_physical_heads','equal2block_decoded_velocity','log1p_meanFn_1N','coverage_BCE'],WM_coefficients=[1.,1.,1.,1.],
 old_CF_reused=False,WM_input_output_normalizers='frozen_from_warmstart5000',WM_loss_scale='TRAIN_only_equal_distribution_newTRIPOD_and_old684_population_std_with_original_unit_floors',
 Actor_Critic_norm='fit_once_common_TRIPOD_TRAIN_current_and_next_obs_only_then_frozen',
 real_TD='r_mode_POST_H_kplus1 +gamma*not_actor_absorbing_done*not_reference_timeout*targetCritic(nextobs);collection_truncation_bootstraps',
 Actor_start='datasetActor_start_eligible &completeWMeligible &k+H<=100',shared_real_batches=True,shared_WM=True,
 memory_arithmetic='canonicalF64state_and_allfloatmemory;FP32Actor_output_cast_toF64history',replay_capacity=4096,
 monitor='fixedTRIPOD_MONITOR_WM_only_not_norm_or_selection',online_collection_rounds=[],online_H_per_arm=0)

def need(x,msg):
    if not x:raise ValueError(msg)
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Current absolute pin');return Path(v['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
    if isinstance(v,(tuple,list)):return [clean(a) for a in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return float(v)
    return v
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('x',encoding='utf8') as f:json.dump(clean(v),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def loadmod(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def npz(v):
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def positive(v):return v.get('engineering_qualified') is True and v.get('checks') and all(q is True for q in v['checks'].values())

def metadata_preflight(path):
    p=read(path);need(p['schema']==SCHEMA and p['recipe']==RECIPE,'New fixed phone recipe')
    need(type(p['CPU_run_authorized']) is bool and type(p['GPU_run_authorized']) is bool and type(p['native_online_authorized']) is bool,'Explicit execution flags')
    for v in p['sources'].values():checked(v)
    need(p['sources']['trainer']==pin(__file__),'Current trainer')
    for k,v in p['inputs'].items():
        if v is not None:checked(v)
    warm=read(checked(p['inputs']['WM_result']));health=read(checked(p['inputs']['WM_health']))
    need(warm['iteration']==5000 and warm['checkpoint']==health['checkpoint'] and warm['health']==p['inputs']['WM_health'] and all(v is True for v in health['checks'].values()),'Healthy fixed5000 WM warmstart, no old pen weights')
    need(warm['checkpoint']['sha256']=='344bac4568f57b8d97973bda06e5568b97fe10e13864f8b3d5a326bcf98dc9a7','Fixed physical component')
    checked(warm['checkpoint']);bundle=read(checked(p['inputs']['task_bundle']))
    need(bundle['schema']=='phone_fixed_wrist_ego61_tripod_task:1' and bundle['task_designated_geom_ids']==[15,28,41] and bundle['model_training_original_designated_ids']==[15,28],'Explicit new tripod task/OOD')
    need(read(checked(p['recipe_pin']))['recipe']==RECIPE,'Frozen recipe bytes')
    missing=p['inputs']['dataset'] is None or p['inputs']['dataset_review'] is None
    dataset=None
    if not missing:
        dataset=read(checked(p['inputs']['dataset']));review=read(checked(p['inputs']['dataset_review']))
        need(dataset['schema']=='phone_tripod_replay_training_dataset:1' and dataset['status']=='COMPLETED_SAVED_TRIPOD_REPLAY22_LABEL_DATASET' and positive(dataset),'Real qualified new task replay')
        need(review['schema']=='phone_tripod_replay_dataset_saved_review:1' and positive(review) and review['dataset_report']==p['inputs']['dataset'],'Positive dataset saved review')
        need(dataset['family_split']==dict(train=list(range(12)),monitor=list(range(12,16))) and set(dataset['field_map'])>=FIELDS,'Frozen family split and canonical fields')
        need(dataset['task_bundle']==p['inputs']['task_bundle'],'Same actual task/baseline/reference/control')
        for v in dataset['partition_arrays'].values():checked(v)
    if missing:need(p['CPU_run_authorized'] is False and p['GPU_run_authorized'] is False,'Actual training requires completed data/review pins')
    need(p['device']=='cpu' or (p['device'].startswith('cuda:') and p['device'][5:].isdigit()),'Explicit CPU or cuda:index')
    need(not RECIPE['online_collection_rounds'] and RECIPE['online_H_per_arm']==0 and p['native_online_authorized'] is False,'This first recipe is offline; online extension needs a new reviewed recipe/protocol')
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==H and not out.exists(),'Fresh local output')
    if p.get('resume_checkpoint') is not None:
        checked(p['resume_checkpoint']);rh=read(checked(p['resume_health']));need(rh['checkpoint']==p['resume_checkpoint'] and all(v is True for v in rh['checks'].values()),'Resume own healthy paired checkpoint')
    return p,dataset,bundle,warm,out,missing

def map_partition(report,role):
    raw=npz(report['partition_arrays'][role]);m=report['field_map'];d={k:raw[m[k]] for k in FIELDS}
    n=len(d['row_id']);need(all(len(v)==n for v in d.values()),'All raw rows/masks retained')
    for key,size in [('x58',58),('y58',58),('u20',20),('memory83',83),('next_memory83',83),('obs222',222),('next_obs222',222),('action20',20),('delta52',52)]:need(d[key].shape==(n,size),'Shape: '+key)
    need(d['impulse22x6'].shape==(n,22,6) and d['meanFn22'].shape==d['coverage22'].shape==(n,22),'Canonical22 labels')
    for key in ['complete_H','WM_eligible','Actor_start_eligible','actor_absorbing_done','reference_timeout','acquisition_done','collection_truncated']:need(d[key].dtype==np.bool_,'Explicit mask: '+key)
    wm=d['WM_eligible'];actor=d['Actor_start_eligible']
    need(np.all(~wm|d['complete_H']) and np.all(~actor|wm),'No partial/unknown eligibility promotion')
    need(all(np.isfinite(d[k][wm]).all() for k in ['x58','y58','u20','memory83','next_memory83','impulse22x6','meanFn22','coverage22']),'Known complete labels must be finite')
    need(np.all(d['h'][wm]*d['n'][wm]==.02) and np.all(d['executed_n'][wm]==d['n'][wm]),'Actual complete20ms')
    need(np.all(d['n'][wm]==d['memory83'][wm,82]) and np.all(d['step_k'][wm]==d['memory83'][wm,0]),'Memory time corresponds to physical transition')
    need(np.isnan(d['y58'][~d['complete_H']]).all(),'Partial target not a fake H endpoint')
    return d

def old_rows(p,codec):
    r=read(checked(p['inputs']['old_original_TRAIN']));labels=read(checked(p['inputs']['old_pair_labels']))
    old=npz(r['outputs']['train']);lab=npz(labels['outputs']['labels']);lookup={int(v):i for i,v in enumerate(lab['sample_id_i64'])}
    mask=old['training_eligible_bool'];need(int(mask.sum())==684,'Only old originalTRAIN684, no CHECK/freshfinal')
    ii=np.array([lookup[int(v)] for v in old['row_id_i64'][mask]])
    return dict(x=old['x58_F64'][mask],u=old['u20_F64'][mask],y=old['y58_F64'][mask],h=old['native_dt_s_F64'][mask],n=old['n_substeps_i64'][mask],J=lab['impulse_source_BODY_moving_COM22x6'][ii],Fn=lab['mean_positive_Fn22_N'][ii],cov=lab['Fn_gt_0p1_coverage22'][ii])

def physical_data(d):
    m=d['WM_eligible'];return dict(x=d['x58'][m],u=d['u20'][m],y=d['y58'][m],h=d['h'][m],n=d['n'][m],J=d['impulse22x6'][m],Fn=d['meanFn22'][m],cov=d['coverage22'][m])
def encode_and_cache(a,geometry,codec,helper,physical,device,torch):
    T=a['h']*a['n'];heads=codec.encode_observed_numpy(a['x'],a['y'],a['J'],T,physical['g'].detach().cpu().numpy())
    head=np.c_[heads['joint_delta40'],heads['COM_ballistic_residual3_m'],heads['world_rotation_delta3_rad'],a['J'].reshape(-1,132)]
    F,cache=helper.precompute_features(dict(x=a['x'],u=a['u'],h=a['h'],n=a['n']),geometry,torch)
    return dict(x=torch.tensor(a['x'],dtype=torch.float64,device=device),y=torch.tensor(a['y'],dtype=torch.float64,device=device),T=torch.tensor(T,dtype=torch.float64,device=device),head=torch.tensor(head,dtype=torch.float64,device=device),Fn=torch.tensor(a['Fn'],dtype=torch.float32,device=device),cov=torch.tensor(a['cov'],dtype=torch.float32,device=device),feature_F64=torch.tensor(F,dtype=torch.float64,device=device)),cache
def mixed_scales(old,new,helper,torch):
    def scale(a,b,floor):
        mean=(a.mean(0)+b.mean(0))*.5;var=((a.var(0,unbiased=False)+(a.mean(0)-mean).square())+(b.var(0,unbiased=False)+(b.mean(0)-mean).square()))*.5
        return torch.maximum(torch.sqrt(var),torch.as_tensor(floor,dtype=a.dtype,device=a.device))
    return dict(head=scale(old['head'],new['head'],helper.physical_floors()),velocity=scale(old['y'][:,52:58]-old['x'][:,52:58],new['y'][:,52:58]-new['x'][:,52:58],[.01,.01,.01,.1,.1,.1]))
def select_mix(new,old,generator,torch):
    a=torch.randint(len(new['x']),(4,),generator=generator,device='cpu').to(new['x'].device);b=torch.randint(len(old['x']),(4,),generator=generator,device='cpu').to(old['x'].device)
    return {k:torch.cat([new[k][a],old[k][b]]) for k in new}
def wm_loss(wm,batch,norm,scales,physical,codec,helper,torch):
    raw=wm(((batch['feature_F64']-norm['input_mean'])/norm['input_std']).float());head=raw[:,:178].double()*norm['head_std']+norm['head_mean']
    y=codec.decode_physical_heads(batch['x'],helper.heads_from_physical(head),batch['T'],physical['mass'],physical['I'],physical['g'])
    lh=helper.block_mse((head-batch['head'])/scales['head'],helper.physical_blocks(),torch)
    v=(y[:,52:58]-batch['y'][:,52:58])/scales['velocity'];lv=(v[:,:3].square().mean()+v[:,3:].square().mean())*.5
    lf=(torch.log1p(torch.nn.functional.softplus(raw[:,178:200]))-torch.log1p(batch['Fn'])).square().mean()
    lc=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,200:222],batch['cov'])
    return lh+lv+lf+lc,dict(physical_heads=lh,decoded_velocity=lv,meanFn=lf,coverage=lc)

class CanonicalMemoryTask:
    """Newtrainer precision contract; frozenv2 scientific task remains intact."""
    def __init__(self,task):self.task=task
    def __getattr__(self,name):return getattr(self.task,name)
    def advance_memory(self,m,a,c,u):
        result=self.task.advance_memory(m,a,c,u)
        for key in ['a1','a2','c1','last_u']:result[key]=result[key].to(dtype=u.dtype,device=u.device)
        return result

def grad_step(loss,module,opt,tag,torch):
    need(torch.isfinite(loss).item(),'Finite '+tag);loss.backward()
    return finish_step(module,opt,tag,torch)
def finish_step(module,opt,tag,torch):
    need(all(q.grad is not None and torch.isfinite(q.grad).all().item() for q in module.parameters()),'All finite named grads '+tag)
    pre={n:q.grad.detach().clone() for n,q in module.named_parameters()};n=torch.nn.utils.clip_grad_norm_(module.parameters(),1.,error_if_nonfinite=True,foreach=False)
    post={k:q.grad.detach().clone() for k,q in module.named_parameters()};opt.step();need(all(torch.isfinite(q).all().item() for q in module.parameters()),'Finite updated weights '+tag)
    return dict(tag=tag,pre_clip=pre,post_clip=post,clip_returned_norm=n.detach())
def make_opt(module,lr,torch):return torch.optim.AdamW(module.parameters(),lr=lr,weight_decay=0.,betas=(.9,.999),eps=1e-8,foreach=False,capturable=False)
def move(value,device,torch):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device,torch) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device,torch) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device,torch) for v in value)
    return value
def cuda_rng(device,torch):return torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
def arm_states(arms):
    return {mode:dict(stateactor=a['actor'].state_dict(),statecritic=a['critic'].state_dict(),statecritic_target=a['target_critic'].state_dict(),
        norm=dict(actor=a['actor'].norm.state_dict(),critic=a['critic'].norm.state_dict(),critic_target=a['target_critic'].norm.state_dict())) for mode,a in arms.items()}
def restore_arm_states(arms,saved):
    for mode,a in arms.items():
        for key,field in [('actor','stateactor'),('critic','statecritic'),('target_critic','statecritic_target')]:a[key].load_state_dict(saved[mode][field])

def make_replay(d,ReplayBuffer,device,torch):
    """Generic upstreambuffer + explicit eligibleindex; raw source stays pinned."""
    buffer=ReplayBuffer(max(RECIPE['replay_capacity'],len(d['row_id'])),1,device,holdout_fraction=0.)
    for i in range(len(d['row_id'])):
        tensors=dict(x=d['x58'][i],x1=d['y58'][i],u=d['u20'][i],m=d['memory83'][i],m1=d['next_memory83'][i],obs=d['obs222'][i],obs1=d['next_obs222'][i],action=d['action20'][i],valid=d['WM_eligible'][i],actor_start=d['Actor_start_eligible'][i],done=bool(d['actor_absorbing_done'][i] or d['acquisition_done'][i] or d['collection_truncated'][i] or not d['complete_H'][i]),terminated=d['actor_absorbing_done'][i],reference_timeout=d['reference_timeout'][i],episode_id=d['episode_id'][i],row_id=d['row_id'][i])
        fields={}
        for k,v in tensors.items():
            a=np.asarray(v);dt=torch.bool if a.dtype.kind=='b' else torch.int64 if a.dtype.kind in 'iu' else torch.float32 if k in ['obs','obs1'] else torch.float64
            fields[k]=torch.as_tensor(v,device=device,dtype=dt).unsqueeze(0)
        buffer.add(fields)
    eligible=np.flatnonzero(d['Actor_start_eligible']&d['WM_eligible']&(d['step_k']+2<=100));need(len(eligible)>0,'Real complete live Actor starts required')
    return buffer,torch.tensor(eligible,dtype=torch.int64,device='cpu')
def sample_replay(buffer,eligible,generator,device,torch):
    chosen=eligible[torch.randint(len(eligible),(8,),generator=generator,device='cpu')].to(device)
    return {k:v[chosen,0] for k,v in buffer.data.items()},chosen

def real_rewards(data,tasks,device,torch):
    result={};indices=np.flatnonzero(data['Actor_start_eligible']&data['WM_eligible'])
    for mode,task in tasks.items():
        values=torch.full((len(data['row_id']),),float('nan'),dtype=torch.float64,device=device)
        with torch.no_grad():
            for offset in range(0,len(indices),8):
                ii=indices[offset:offset+8];x=torch.tensor(data['y58'][ii],dtype=torch.float64,device=device);m=task.unflatten_memory(torch.tensor(data['memory83'][ii],dtype=torch.float64,device=device));a=torch.tensor(data['action20'][ii],dtype=torch.float64,device=device)
                r=task.rewards(x,m,a);need(r['objective_known'].all().item(),'Known engineered/actualstate task objective');values[torch.tensor(ii,device=device)]=r['total']
        result[mode]=values
    return result

def health_checkpoint(path,wm,arms,opts,grads,norm,physical,scales,generator,p,protocol,probe_feature,probe_obs,helper,actor_module,sourcepins,device,seed,round_id,wm_updates,torch):
    before=helper.rng(torch);before_cuda=cuda_rng(device,torch)
    with torch.no_grad():
        expected={'WM':wm(probe_feature).detach()}
        for mode,a in arms.items():expected[mode+'_Actor']=a['actor'](probe_obs).detach();expected[mode+'_Critic']=a['critic'](probe_obs).detach()
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    need(loaded['schema']==ARTIFACT and loaded['paired_round']==round_id and loaded['shared_WM_updates']==wm_updates,'New paired checkpoint phase')
    with torch.random.fork_rng(devices=[] if device.type=='cpu' else [device.index]):clone=helper.make_model(torch).to(device)
    clone.load_state_dict(loaded['shared_WM']);restored={}
    for mode in MODES:
        q=actor_module.make_fresh_actor_critic(seed=seed,source_pins=sourcepins,device=str(device));q['actor'].load_state_dict(loaded['arms'][mode]['stateactor']);q['critic'].load_state_dict(loaded['arms'][mode]['statecritic']);restored[mode]=q
    with torch.no_grad():
        actual={'WM':clone(probe_feature)}
        for mode,a in restored.items():actual[mode+'_Actor']=a['actor'](probe_obs);actual[mode+'_Critic']=a['critic'](probe_obs)
    checks=dict(all_serialized_predictions_raw_equal=all(helper.tree_equal(expected[k],actual[k],torch) for k in expected),
        all_serialized_WM_actor_critic_target_and_norm_raw=helper.tree_equal(loaded['shared_WM'],wm.state_dict(),torch) and helper.tree_equal(loaded['arms'],arm_states(arms),torch),
        normalizers_raw=helper.tree_equal(loaded['WM_normalizers'],norm,torch),physical_context_raw=helper.tree_equal(loaded['physical_context'],physical,torch),loss_scales_TRAIN_only_raw=helper.tree_equal(loaded['WM_loss_scales'],scales,torch),
        serialized_optimizers_raw=all(helper.tree_equal(loaded['optimizers'][k],opt.state_dict(),torch) for k,opt in opts.items()),
        serialized_gradients_raw=helper.tree_equal(loaded['gradient_snapshots'],grads,torch),
        finite_all_weights=all(torch.isfinite(v).all().item() for m in [wm,*[a[k] for a in arms.values() for k in ['actor','critic','target_critic']]] for v in m.parameters()),
        finite_saved_gradients=all(torch.isfinite(v).all().item() for g in grads.values() for side in ['pre_clip','post_clip'] for v in g[side].values()),
        actual_Adam_steps_correct=all(len(opt.state)==(6 if key=='WM' else 8 if key.endswith('Actor') else 10) and all(float(s['step'])==(wm_updates if key=='WM' else round_id) and torch.isfinite(s['exp_avg']).all().item() and torch.isfinite(s['exp_avg_sq']).all().item() for s in opt.state.values()) for key,opt in opts.items()),
        saved_sample_generator_raw=helper.tree_equal(loaded['sample_generator'],generator.get_state(),torch),
        saved_RNG_matches_current=helper.tree_equal(loaded['RNG'],before,torch) and (device.type=='cpu' or helper.tree_equal(loaded['CUDA_RNG'],before_cuda,torch)),
        raw_RNG_unchanged=helper.tree_equal(before,helper.rng(torch),torch) and (device.type=='cpu' or helper.tree_equal(before_cuda,cuda_rng(device,torch),torch)),
        protocol_recipe_source_inputs_bound=loaded['protocol']==pin(protocol) and loaded['recipe']==RECIPE and loaded['inputs']==p['inputs'] and loaded['sources']==p['sources'])
    lp=loaded['log_prefix']
    with Path(lp['path']).open('rb') as f:prefix=f.read(lp['bytes'])
    lines=[json.loads(v) for v in prefix.splitlines()]
    checks['log_prefix_bytes_SHA_round_range']=(len(prefix)==lp['bytes'] and hashlib.sha256(prefix).hexdigest()==lp['sha256'] and len(lines)==lp['records'] and [v['paired_round'] for v in lines]==list(range(lp['round_range'][0],lp['round_range'][1]+1)) and lp['round_range'][1]==round_id)
    need(all(checks.values()),'Serialized paired training health failed, no retry')
    return dict(schema='phone_tripod_matched_learning_checkpoint_health:1',paired_round=round_id,shared_WM_updates=wm_updates,checkpoint=pin(path),checks=checks,passed=sum(checks.values()),total=len(checks),own_checkpoint_loads=1,extra_NN_ctors=5,extra_NN_FW=10),{k+'_before':v.detach().cpu().numpy() for k,v in expected.items()}|{k+'_after':v.detach().cpu().numpy() for k,v in actual.items()}

def run_training(protocol):
    p,report,bundle,warm,out,missing=metadata_preflight(protocol);need(not missing and p['CPU_run_authorized'] is True,'Root needs actual data qualification and CPU authorization')
    if p['device'].startswith('cuda:'):need(p['GPU_run_authorized'] is True,'Explicit Root GPU authorization')
    need(not p['native_online_authorized'],'This fixed offline recipe makes no native calls')
    if p['device']=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(p['device'])
    if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    random.seed(RECIPE['seed']);np.random.seed(RECIPE['seed']);torch.random.default_generator.manual_seed(RECIPE['seed'])
    if device.type=='cuda':torch.cuda.manual_seed(RECIPE['seed'])
    out.mkdir();write(out/'protocol.json',p);started=time.perf_counter();phase='load';round_id=0;wm_updates=0
    counts=dict(WM_supervised_FW=0,WM_supervised_backward=0,WM_optimizer_steps=0,WM_imagination_FW=0,Actor_FW=0,Critic_TD_FW=0,Critic_bootstrap_FW=0,Actor_backward_microbatches=0,Critic_backward=0,Actor_optimizer_steps=0,Critic_optimizer_steps=0,monitor_WM_FW=0,NN_fresh_constructors=0,target_critic_deepcopies=0,health_FW=0,health_ctors=0,own_checkpoint_loads=0,native_calls=0)
    process_counts=dict(NN_fresh_constructors=0,target_critic_deepcopies=0,health_FW=0,health_ctors=0,own_checkpoint_loads=0,monitor_WM_FW=0)
    try:
        helper=loadmod('phone_tripod_frozen_WM',checked(p['sources']['WM_trainer']));codec=loadmod('phone_tripod_structured',checked(p['sources']['structured_codec']))
        gap=loadmod('phone_tripod_geometry',checked(p['sources']['directional']));geometry=gap.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']))
        taskmodule=loadmod('phone_tripod_task',checked(p['sources']['task']));actor_module=loadmod('phone_tripod_actor_bridge',checked(p['sources']['imagination']))
        upstream_root=Path(p['sources']['Actor_Critic']['path']).parents[3];sys.path.insert(0,str(upstream_root))
        replaymodule=loadmod('phone_tripod_upstream_replay',checked(p['sources']['replay']))
        d=map_partition(report,'train');monitor=map_partition(report,'monitor')
        need(set(d['family_id'])<=set(range(12)) and set(monitor['family_id'])<=set(range(12,16)),'No MONITOR family in TRAIN')
        ck=torch.load(checked(warm['checkpoint']),map_location='cpu',weights_only=False);counts['own_checkpoint_loads']+=1;process_counts['own_checkpoint_loads']+=1
        need(ck['schema']==helper.ARTIFACT and ck['iteration']==5000,'Only physical5000 weight warmstart; old optimizers not resumed')
        norm=move(ck['normalizers'],device,torch);physical=move(ck['physical_context'],device,torch)
        wm=helper.make_model(torch).to(device);counts['NN_fresh_constructors']+=1;process_counts['NN_fresh_constructors']+=1;wm.load_state_dict(ck['model_state'],strict=True);wm.eval()
        tasks={mode:CanonicalMemoryTask(taskmodule.PhoneTask(checked(p['inputs']['task_bundle']),mode=mode,geometry=geometry)) for mode in MODES}
        old=old_rows(p,codec);new=physical_data(d)
        old_t,oldcache=encode_and_cache(old,geometry,codec,helper,physical,device,torch);new_t,newcache=encode_and_cache(new,geometry,codec,helper,physical,device,torch)
        mon_t,moncache=encode_and_cache(physical_data(monitor),geometry,codec,helper,physical,device,torch)
        scales=mixed_scales(old_t,new_t,helper,torch)
        replay,eligible=make_replay(d,replaymodule.ReplayBuffer,device,torch);rewards=real_rewards(d,tasks,device,torch)
        arms={};opts={'WM':make_opt(wm,RECIPE['WM_lr'],torch)}
        fit=np.flatnonzero(d['Actor_start_eligible']&d['WM_eligible'])
        commonobs=torch.tensor(np.concatenate([d['obs222'][fit],d['next_obs222'][fit]]),dtype=torch.float32,device=device)
        for mode in MODES:
            modules=actor_module.make_fresh_actor_critic(seed=RECIPE['seed'],source_pins=p['sources'],device=str(device));actor=modules['actor'];critic=modules['critic']
            actor.norm.update(commonobs);critic.norm.update(commonobs);target=copy.deepcopy(critic).eval();target.requires_grad_(False)
            counts['NN_fresh_constructors']+=2;process_counts['NN_fresh_constructors']+=2;counts['target_critic_deepcopies']+=1;process_counts['target_critic_deepcopies']+=1
            arms[mode]=dict(actor=actor,critic=critic,target_critic=target);opts[mode+'_Actor']=make_opt(actor,RECIPE['Actor_lr'],torch);opts[mode+'_Critic']=make_opt(critic,RECIPE['Critic_lr'],torch)
        need(helper.tensor_digest(arms[MODES[0]]['actor'].state_dict(),torch)==helper.tensor_digest(arms[MODES[1]]['actor'].state_dict(),torch) and helper.tensor_digest(arms[MODES[0]]['critic'].state_dict(),torch)==helper.tensor_digest(arms[MODES[1]]['critic'].state_dict(),torch),'Matched initial policies/norm exposure')
        generator=torch.Generator(device='cpu');generator.manual_seed(RECIPE['seed']+1);grads={};start_round=1
        write(out/'raw_replay_source_manifest.json',dict(dataset=p['inputs']['dataset'],partitions=report['partition_arrays'],all_TRAIN_rows=d['row_id'],WM_included_row_ids=d['row_id'][d['WM_eligible']],Actor_start_row_ids=d['row_id'][d['Actor_start_eligible']&d['WM_eligible']],excluded_partial_row_ids=d['row_id'][~d['complete_H']],no_MONITOR_normalization=True,raw_source_preserved_before_generic_replay_nan_to_num=True))
        if p.get('resume_checkpoint') is None:
            helper.atomic_checkpoint(out/'initial_modules_not_learning_checkpoint.pt',dict(schema='phone_tripod_matched_initial_modules:1',shared_WM=wm.state_dict(),arms=arm_states(arms),WM_normalizers=norm,WM_loss_scales=scales,physical_context=physical,RNG=helper.rng(torch),CUDA_RNG=cuda_rng(device,torch),healthy_learning_checkpoint=False),torch)
            phase='common_WM_adaptation'
            for j in range(100):
                wm.train();wm.requires_grad_(True);opts['WM'].zero_grad(set_to_none=True);batch=select_mix(new_t,old_t,generator,torch);loss,_=wm_loss(wm,batch,norm,scales,physical,codec,helper,torch)
                grads['WM']=grad_step(loss,wm,opts['WM'],f'PRE_SHARED_WM_UPDATE_{wm_updates+1}',torch);wm_updates+=1;counts['WM_supervised_FW']+=1;counts['WM_supervised_backward']+=1;counts['WM_optimizer_steps']+=1
            write(out/'common_adaptation_counters.json',dict(common_WM_updates=100,optimizer_steps=100,last_loss=float(loss.detach()),counts=counts.copy(),updated_WM_or_Adam_checkpoint_written=False,first_formal_learning_checkpoint='paired100_after_common100'))
        else:
            resume=torch.load(checked(p['resume_checkpoint']),map_location='cpu',weights_only=False);counts['own_checkpoint_loads']+=1;process_counts['own_checkpoint_loads']+=1
            need(resume['schema']==ARTIFACT and resume['recipe']==RECIPE and resume['inputs']==p['inputs'] and resume['sources']==p['sources'],'Exact own resume lineage')
            wm.load_state_dict(resume['shared_WM']);wm_updates=resume['shared_WM_updates'];start_round=resume['paired_round']+1
            restore_arm_states(arms,resume['arms'])
            for key,opt in opts.items():opt.load_state_dict(resume['optimizers'][key])
            need(helper.tree_equal(resume['WM_normalizers'],norm,torch) and helper.tree_equal(resume['WM_loss_scales'],scales,torch),'Same frozen modelnorms/TRAIN-only loss scales')
            generator.set_state(resume['sample_generator']);random.setstate(resume['RNG']['python']);np.random.set_state(resume['RNG']['numpy']);torch.set_rng_state(resume['RNG']['torch'])
            if device.type=='cuda':torch.cuda.set_rng_state(resume['CUDA_RNG'],device)
            parent_health=read(checked(p['resume_health']))
            for key,value in parent_health['counts'].items():counts[key]=counts.get(key,0)+value
        setup_s=time.perf_counter()-started;times=[];logpath=out/'paired_updates.jsonl'
        with logpath.open('x',encoding='utf8') as log:
            for round_id in range(start_round,1001):
                phase=f'paired_round_{round_id}';tick=time.perf_counter();wm.train();wm.requires_grad_(True);opts['WM'].zero_grad(set_to_none=True)
                batch=select_mix(new_t,old_t,generator,torch);loss,wparts=wm_loss(wm,batch,norm,scales,physical,codec,helper,torch)
                grads['WM']=grad_step(loss,wm,opts['WM'],f'PRE_SHARED_WM_UPDATE_{wm_updates+1}',torch);wm_updates+=1;counts['WM_supervised_FW']+=1;counts['WM_supervised_backward']+=1;counts['WM_optimizer_steps']+=1
                wm.eval();wm.requires_grad_(False);real,indices=sample_replay(replay,eligible,generator,device,torch);mode_losses={}
                for mode in MODES:
                    a=arms[mode];task=tasks[mode];optc=opts[mode+'_Critic'];opta=opts[mode+'_Actor'];optc.zero_grad(set_to_none=True)
                    with torch.no_grad():target=rewards[mode][indices]+.99*(~real['terminated']).double()*(~real['reference_timeout']).double()*a['target_critic'](real['obs1']).double()
                    value=a['critic'](real['obs']);closs=(value-target.float()).square().mean();counts['Critic_TD_FW']+=2
                    grads[mode+'_Critic']=grad_step(closs,a['critic'],optc,f'PRE_{mode}_CRITIC_ROUND_{round_id}',torch);counts['Critic_backward']+=1;counts['Critic_optimizer_steps']+=1
                    with torch.no_grad():
                        for q,tq in zip(a['critic'].parameters(),a['target_critic'].parameters()):tq.lerp_(q,.01)
                    opta.zero_grad(set_to_none=True);aloss=0.
                    for start in range(0,8,2):
                        m=task.unflatten_memory(real['m'][start:start+2]);obj=actor_module.imagined_actor_objective(actor=a['actor'],critic=a['critic'],world_model=wm,task=task,geometry=geometry,codec=codec,norms=norm,physical=physical,initial_state58=real['x'][start:start+2],memory=m,horizon=2,gamma=.99,wm_step=taskmodule.wm_step_current_only)
                        need(torch.isfinite(obj['actor_loss']).item(),'Finite imagined loss, no retry');(obj['actor_loss']/4).backward();aloss+=float(obj['actor_loss'].detach())/4
                        counts['Actor_backward_microbatches']+=1;counts['Actor_FW']+=2;counts['WM_imagination_FW']+=2;counts['Critic_bootstrap_FW']+=1
                    grads[mode+'_Actor']=finish_step(a['actor'],opta,f'PRE_{mode}_ACTOR_ROUND_{round_id}',torch);counts['Actor_optimizer_steps']+=1
                    mode_losses[mode]=dict(real_TD_loss=float(closs.detach()),imagined_Actor_loss=aloss)
                if device.type=='cuda':torch.cuda.synchronize(device)
                duration=time.perf_counter()-tick;times.append(duration)
                row=dict(paired_round=round_id,shared_WM_updates=wm_updates,WM_loss=float(loss.detach()),WM_parts={k:float(v.detach()) for k,v in wparts.items()},modes=mode_losses,round_seconds=duration,common_real_row_ids=real['row_id'].detach().cpu().tolist())
                log.write(json.dumps(row)+'\n');log.flush()
                if round_id in RECIPE['checkpoint_rounds']:
                    phase=f'health_round_{round_id}';healthstart=time.perf_counter();os.fsync(log.fileno());prefix=logpath.read_bytes();path=out/f'paired_checkpoint_{round_id:04d}.pt'
                    payload=dict(schema=ARTIFACT,paired_round=round_id,shared_WM_updates=wm_updates,recipe=RECIPE,protocol=pin(protocol),inputs=p['inputs'],sources=p['sources'],shared_WM={k:v.detach().clone() for k,v in wm.state_dict().items()},arms=arm_states(arms),optimizers={k:v.state_dict() for k,v in opts.items()},gradient_snapshots=grads,WM_normalizers=norm,WM_loss_scales=scales,physical_context=physical,
                        sample_generator=generator.get_state(),RNG=helper.rng(torch),CUDA_RNG=cuda_rng(device,torch),counts=counts.copy(),common_adaptation_completed=True,
                        log_prefix=dict(path=str(logpath),bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),records=round_id-start_round+1,round_range=[start_round,round_id]),offline_replay_source=p['inputs']['dataset'],native_online_live_state=None,resume_parent_checkpoint=p.get('resume_checkpoint'),task_bundle=p['inputs']['task_bundle'],policy_interface=dict(obs_dim=222,memory_dim=83,action_dim=20,arm_fields=['stateactor','statecritic','statecritic_target','norm']))
                    helper.atomic_checkpoint(path,payload,torch)
                    probe=((new_t['feature_F64'][:8]-norm['input_mean'])/norm['input_std']).float();obs=commonobs[:8]
                    health,evidence=health_checkpoint(path,wm,arms,opts,grads,norm,physical,scales,generator,p,protocol,probe,obs,helper,actor_module,p['sources'],device,RECIPE['seed'],round_id,wm_updates,torch)
                    counts['own_checkpoint_loads']+=1;counts['health_ctors']+=5;counts['health_FW']+=10
                    process_counts['own_checkpoint_loads']+=1;process_counts['health_ctors']+=5;process_counts['health_FW']+=10
                    np.savez_compressed(out/f'health_readback_{round_id:04d}.npz',**evidence)
                    with torch.no_grad():monitor_loss,_=wm_loss(wm,mon_t,norm,scales,physical,codec,helper,torch)
                    counts['monitor_WM_FW']+=1;process_counts['monitor_WM_FW']+=1
                    overhead=time.perf_counter()-healthstart;remaining=1000-round_id;rate=float(np.mean(times[-min(25,len(times)):]))
                    health.update(counts=counts.copy(),process_setup_diagnostic_counts=process_counts.copy(),monitor_WM_loss=float(monitor_loss),monitor_not_model_selection=True,setup_s=setup_s,elapsed_s=time.perf_counter()-started,last25_mean_round_seconds=rate,estimated_remaining_training_s=remaining*rate,estimated_resume_setup_s=setup_s,checkpoint_health_IO_s=overhead,
                        GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,device=str(device),TF32=False,microbatch2_H2=True,phase_contract='WM gradient PRE_SHARED_UPDATE; Actor/Critic gradients PRE_ROUND; checkpoint weights AFTER pairedround',first_learning_checkpoint=round_id==100,process_stops_here=round_id==100 and RECIPE['stop_after_first_healthy'])
                    write(out/f'health_{round_id:04d}.json',health);print(json.dumps(dict(round=round_id,WMupdates=wm_updates,health_passed=health['passed'],health_total=health['total'],ETA_s=health['estimated_remaining_training_s'],GPUpeak=health['GPU_peak_allocated_bytes'])),flush=True)
                    if round_id==100 and RECIPE['stop_after_first_healthy']:
                        write(out/'result.json',dict(schema='phone_tripod_matched_supermimic_training_result:1',status='STOPPED_AT_FIRST_HEALTHY_PAIRED100_RESUMABLE_TARGET1000',paired_round=round_id,shared_WM_updates=wm_updates,checkpoint=pin(path),health=pin(out/f'health_{round_id:04d}.json'),counts=counts,ETA_s=health['estimated_remaining_training_s'],Actor_native_benefit_qualified=False));return 0
        write(out/'result.json',dict(schema='phone_tripod_matched_supermimic_training_result:1',status='COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION',paired_round=1000,shared_WM_updates=1100,checkpoint=pin(out/'paired_checkpoint_1000.pt'),counts=counts,Actor_native_benefit_qualified=False));return 0
    except BaseException as e:
        write(out/'failure.json',dict(status='FAILED_PRESERVED_NO_RETRY',phase=phase,paired_round=round_id,shared_WM_updates=wm_updates,error=repr(e),traceback=traceback.format_exc(),counts=counts));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.run:raise SystemExit(run_training(a.protocol))
    p,d,b,w,out,missing=metadata_preflight(a.protocol);print(json.dumps(dict(status='CPUFALSE_METADATA_ONLY_PENDING_DATA' if missing else 'CPUFALSE_METADATA_ONLY_READY_FOR_REVIEW',missing_dataset=missing,NN_native_training_calls=0,output=str(out))))
