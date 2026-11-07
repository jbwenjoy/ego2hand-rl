"""Configuration-driven offline driver using the shared implementation.

inspect: metadata only. smoke: bounded CPU inference, zero optimizer/normfit.
train: explicit authorization, real shared-WM/A1/A2 updates and first-health
stop. Own unified checkpoints have a resume route; no legacy checkpoint is
misrepresented as resume. Online native orchestration is not included yet.
"""
from pathlib import Path
import copy,hashlib,json,os,random,time,tomllib
import numpy as np
from supermimic_research.artifacts import ArtifactRegistry,file_pin
from . import data,world_model as WM,imagination as IM,update_step as U

MODES=['A1_tracking','A2_engineered_material_proxy']
ARTIFACT_KEYS=['task','geometry','tripod_dataset','old_original','old_pair_labels','warmstart']

def source_pins():
    """Own implementation recovery binding, not imports of legacy sources."""
    package=Path(__file__).resolve().parent.parent;src=package.parent
    files={
        **{sub+'/'+path.name:path for sub in ['learning','runtime'] for path in sorted((package/sub).glob('*.py'))},
        **{'dex/'+name:src/'dex_supermimic'/'rl'/'dex_supermimic'/name for name in ['networks.py','common.py','state.py','buffer.py','rotations.py']},
        **{'chord/'+name:src/'chord_study'/name for name in ['contacts.py','contact_geometry.py']},
        'artifacts.py':package/'artifacts.py',
    }
    return {name:file_pin(path) for name,path in files.items()}

def _content_bindings(pins):return {k:(v['bytes'],v['sha256']) for k,v in pins.items()}
def _artifact_pins(registry,c):return {key:registry.pin(c['artifacts'][key]) for key in ARTIFACT_KEYS}
def _counts_valid(counts,update,c):
    return (type(update) is int and 0<update<=c['training']['updates'] and
            counts==dict(WM_updates=c['training']['common_WM_updates']+update,paired_updates=update))

def configuration(path):
    path=Path(path).resolve(strict=True);c=tomllib.loads(path.read_text(encoding='utf8'))
    if c.get('schema')!=1:raise ValueError('Offline driver config schema1')
    registry=ArtifactRegistry((path.parent/c['artifacts']['manifest']).resolve(),local=(path.parent/c['artifacts']['local']).resolve() if c['artifacts'].get('local') else None)
    if bool(c['training'].get('resume_artifact'))!=bool(c['training'].get('resume_health')):raise ValueError('Own resume requires checkpoint and health registry keys')
    if c['sampling']['tripod_WM_rows']+c['sampling']['old_WM_rows']!=c['sampling']['WM_batch']:raise ValueError('Explicit WM source mixture')
    for key in ['updates','first_checkpoint','horizon','microbatch','cpu_threads']:
        if type(c['training'][key]) is not int or c['training'][key]<1:raise ValueError('Positive integer training '+key)
    if type(c['training']['common_WM_updates']) is not int or c['training']['common_WM_updates']<0 or c['training']['first_checkpoint']>c['training']['updates']:raise ValueError('Common/first/total update budget')
    return path,c,registry

def inspect(config_path):
    path,c,r=configuration(config_path);names=[c['artifacts'][key] for key in ARTIFACT_KEYS]
    pins={name:r.pin(name) for name in names}
    report=json.loads(r.resolve(c['artifacts']['tripod_dataset']).read_text(encoding='utf8'))
    return dict(status='OFFLINE_CONFIG_METADATA_ONLY',config=file_pin(path),artifact_manifest=file_pin(r.manifest),artifacts=pins,sources=source_pins(),partition_counts=report['split_counts'],device=c['training']['device'],training_authorized=c['training']['authorized'],updates=c['training']['updates'],common_WM_updates=c['training']['common_WM_updates'],resume_supported='own_unified_schema_only',NN_native_calls=0)

def _science_config(c):
    x=copy.deepcopy(c)
    for key in ['authorized','output','resume_artifact','resume_health']:x['training'].pop(key,None)
    return x
def _log_prefix_valid(p):
    with Path(p['path']).open('rb') as f:b=f.read(p['bytes'])
    rows=[json.loads(v) for v in b.splitlines()]
    return len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256'] and len(rows)==p['records'] and [v['update'] for v in rows]==list(range(p['range'][0],p['range'][1]+1))

def _subset(rows,indices):return {k:v[indices] for k,v in rows.items()}

def _real_bank(rows,*,device,torch):
    """One setup copy in canonical row order, including original masks/NaNs.

    Do not reuse the compressed WM-eligible cache as an Actor row bank: its
    row indices can differ. Only the existing eligible sampler selects rows.
    """
    fields=[('x','x58',torch.float64),('m','memory83',torch.float64),
            ('obs','obs222',torch.float32),('obs1','next_obs222',torch.float32),
            ('terminated','actor_absorbing_done',torch.bool),
            ('reference_timeout','reference_timeout',torch.bool)]
    return {key:torch.tensor(rows[source],dtype=dtype,device=device) for key,source,dtype in fields}

def _real_batch(bank,eligible,generator,batch_size,*,device,torch):
    """Preserve the CPU RNG draw/order; transfer the selected index only."""
    pick=torch.randint(len(eligible),(batch_size,),generator=generator,device='cpu').numpy()
    ii=eligible[pick];idx=torch.tensor(ii,device=device)
    return {key:value.index_select(0,idx) for key,value in bank.items()},ii,idx

def _tripod_rows(dataset):
    d=dataset.canonical;i=dataset.wm_indices()
    return dict(x=d['x58'][i],u=d['u20'][i],y=d['y58'][i],h=d['h'][i],n=d['n'][i],J=d['impulse22x6'][i],Fn=d['meanFn22'][i],cov=d['coverage22'][i])
def _sample(new,old,g,c,torch):
    a=torch.randint(len(new['x']),(c['sampling']['tripod_WM_rows'],),generator=g,device='cpu').to(new['x'].device);b=torch.randint(len(old['x']),(c['sampling']['old_WM_rows'],),generator=g,device='cpu').to(old['x'].device)
    return {k:torch.cat([new[k][a],old[k][b]]) for k in new}

def _geometry(registry,c):
    from supermimic_research.runtime.directional import DirectionalGapProvider
    path=registry.resolve(c['artifacts']['geometry']);catalog=json.loads(path.read_text(encoding='utf8'));meta=registry.resolve_pin(catalog['provider_metadata']);m=json.loads(meta.read_text(encoding='utf8'))
    return DirectionalGapProvider(path,metadata_path=meta,geometry_arrays_path=registry.resolve_pin(m['geometry_arrays']),direction_arrays_path=registry.resolve_pin(catalog['direction_arrays']))
def _task(registry,c,mode,geometry):
    from supermimic_research.runtime.phone_task import PhoneTask
    path=registry.resolve(c['artifacts']['task']);bundle=json.loads(path.read_text(encoding='utf8'))
    return IM.CanonicalMemoryTask(PhoneTask(path,mode=mode,geometry=geometry,reference_arrays_path=registry.resolve_pin(bundle['reference_arrays']),support_basis_path=registry.resolve_pin(bundle['basis'])))

def _setup(config_path,*,smoke_only):
    path,c,r=configuration(config_path);device='cpu' if smoke_only else c['training']['device']
    if device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(c['training']['cpu_threads']);torch.set_default_dtype(torch.float32);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.device(device).type=='cuda':torch.cuda.set_device(device)
    random.seed(c['training']['seed']);np.random.seed(c['training']['seed']);torch.random.default_generator.manual_seed(c['training']['seed'])
    if torch.device(device).type=='cuda':torch.cuda.manual_seed(c['training']['seed'])
    dataset=data.from_registry(r,c['artifacts']['tripod_dataset'],role='train',for_training=True);old,oldreceipt=data.read_original_train(r,c['artifacts']['old_original'],c['artifacts']['old_pair_labels']);new=_tripod_rows(dataset)
    if oldreceipt['pair_ids']!=dataset.report['canonical_pair_geom_ids']:raise ValueError('Old/TRIPOD source-BODY pair order differs')
    warmname=c['artifacts']['warmstart']
    if 'parity' in r.entries[warmname].role.lower() and not smoke_only:raise ValueError('Parity checkpoint is not a training recipe choice')
    warm_manifest=r.resolve(warmname);record=json.loads(warm_manifest.read_text(encoding='utf8'));ckpath=r.resolve_pin(record['checkpoint']);ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    if ck['schema']=='phone_geometry_momentum_counterfactual_WM_checkpoint:1':
        if ck['iteration']!=5000:raise ValueError('Warmstart must use declared fixed5000')
        weights=ck['model_state'];norm=ck['normalizers'];physical=ck['physical_context'];policy_states=None
    elif ck['schema']=='phone_tripod_sharedWM_A1A2_training_checkpoint:1':
        weights=ck['shared_WM'];norm=ck['WM_normalizers'];physical=ck['physical_context'];policy_states=ck['arms']
    else:raise ValueError('Explicit supported phone artifact, never old Pen checkpoint by shape')
    cfg=WM.WorldModelConfig(hidden=tuple(c['model']['hidden']),geometry_chunk_rows=c['model']['geometry_chunk_rows']);model=WM.make_model(cfg,device=device);model.load_state_dict(weights,strict=True);model.eval();model.requires_grad_(False)
    norm=WM.move_tensors(norm,device);physical=WM.move_tensors(physical,device);geometry=_geometry(r,c);arms={}
    for mode in MODES:
        modules=IM.make_fresh_actor_critic(seed=c['training']['seed'],device=device,config=IM.PolicyConfig(obs_dim=222));target=copy.deepcopy(modules['critic']).eval();target.requires_grad_(False)
        arms[mode]=dict(actor=modules['actor'],critic=modules['critic'],target_critic=target)
    if policy_states is not None:U.restore_arm_states(arms,policy_states)
    tasks={mode:_task(r,c,mode,geometry) for mode in MODES};generator=torch.Generator(device='cpu');generator.manual_seed(c['training']['seed']+1)
    if smoke_only:
        # Exactly4+4 examples; no expensive whole-dataset cache or fitting.
        ni=torch.randint(len(new['x']),(c['sampling']['tripod_WM_rows'],),generator=generator,device='cpu').numpy();oi=torch.randint(len(old['x']),(c['sampling']['old_WM_rows'],),generator=generator,device='cpu').numpy()
        new=_subset(new,ni);old=_subset(old,oi)
    new_t,newscope=data.prepare_physical_rows(new,geometry,physical,device=device,chunk_rows=cfg.geometry_chunk_rows);old_t,oldscope=data.prepare_physical_rows(old,geometry,physical,device=device,chunk_rows=cfg.geometry_chunk_rows)
    return dict(path=path,c=c,registry=r,torch=torch,device=device,dataset=dataset,new=new_t,old=old_t,model=model,norm=norm,physical=physical,arms=arms,tasks=tasks,generator=generator,geometry=geometry,starting_checkpoint=file_pin(ckpath),policy_warmstart=policy_states is not None,cache=dict(new=newscope,old=oldscope),old_selection=oldreceipt)

def smoke(config_path):
    s=_setup(config_path,smoke_only=True);torch=s['torch'];model=s['model'];dataset=s['dataset'];d=dataset.canonical
    batch={k:torch.cat([s['new'][k],s['old'][k]]) for k in s['new']};features=WM.normalize_features(batch['feature_F64'],s['norm']);known=dataset.actor_indices();B=min(2,len(known));obs=torch.tensor(d['obs222'][known[:B]],dtype=torch.float32)
    before={mode:WM.tensor_digest(a['actor'].state_dict()) for mode,a in s['arms'].items()}
    with torch.no_grad():
        raw=model(features);head=raw[:,:178].double()*s['norm']['head_std']+s['norm']['head_mean']
        from supermimic_research.runtime import structured
        y=structured.decode_physical_heads(batch['x'],WM.heads_from_physical(head),batch['T'],s['physical']['mass'],s['physical']['I'],s['physical']['g'])
        outputs={mode:(a['actor'](obs),a['critic'](obs)) for mode,a in s['arms'].items()}
    if raw.shape!=(8,222) or not torch.isfinite(y).all() or not all(torch.isfinite(v).all() for pair in outputs.values() for v in pair):raise ValueError('CPU shared model/Actor/Critic smoke failed')
    if not all(before[m]==WM.tensor_digest(a['actor'].state_dict()) for m,a in s['arms'].items()) or torch.cuda.is_initialized():raise ValueError('Smoke must preserve Actor/norm and remainCPU')
    return dict(status='PASS_ZERO_UPDATE_CPU_SMOKE',WM_mix_rows=dict(tripod=len(s['new']['x']),old=len(s['old']['x'])),WM_output=list(raw.shape),decoded_state=list(y.shape),Actor_shapes={m:list(pair[0].shape) for m,pair in outputs.items()},counts=dict(own_CPU_loads=1,NN_constructors=5,target_deepcopies=2,NN_FW=5,physical_decodes=1,optimizer_constructors=0,backward=0,updates=0,normalizer_fit=0,native=0,GPU=0),starting_checkpoint=s['starting_checkpoint'],selection=dataset.selection_receipt(),geometry_cache=s['cache'],scope='Warmstart-only zero-update smoke, not own-resume execution')

def _mixed_scales(new,old):
    import torch
    def scale(a,b,floor):
        mean=(a.mean(0)+b.mean(0))*.5;var=((a.var(0,unbiased=False)+(a.mean(0)-mean).square())+(b.var(0,unbiased=False)+(b.mean(0)-mean).square()))*.5
        return torch.maximum(torch.sqrt(var),torch.as_tensor(floor,dtype=a.dtype,device=a.device))
    return dict(head=scale(new['head'],old['head'],WM.physical_floors()),velocity=scale(new['y'][:,52:58]-new['x'][:,52:58],old['y'][:,52:58]-old['x'][:,52:58],[.01,.01,.01,.1,.1,.1]))

def _health(path,context,optimizers,gradients,counts,probeF,probeO):
    s=context;torch=s['torch'];rng=WM.capture_rng(device=s['device']);model=s['model'];arms=s['arms']
    with torch.no_grad():expected={'WM':model(probeF)}|{m+key:a[key](probeO) for m,a in arms.items() for key in ['actor','critic']}
    ck=torch.load(path,map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=[] if s['device']=='cpu' else [torch.device(s['device']).index]):clone=WM.make_model(WM.WorldModelConfig(hidden=tuple(s['c']['model']['hidden'])),device=s['device'])
    clone.load_state_dict(ck['shared_WM']);restored={}
    for mode in MODES:
        a=IM.make_fresh_actor_critic(seed=s['c']['training']['seed'],device=s['device']);a['actor'].load_state_dict(ck['arms'][mode]['stateactor']);a['critic'].load_state_dict(ck['arms'][mode]['statecritic']);restored[mode]=a
    with torch.no_grad():actual={'WM':clone(probeF)}|{m+key:a[key](probeO) for m,a in restored.items() for key in ['actor','critic']}
    checks=dict(restored_predictions=all(WM.raw_tree_equal(expected[k],actual[k]) and torch.isfinite(actual[k]).all() for k in expected),all_networks_and_norms=WM.raw_tree_equal(ck['shared_WM'],model.state_dict()) and WM.raw_tree_equal(ck['arms'],U.arm_states(arms)),all_optimizers=all(WM.raw_tree_equal(ck['optimizers'][k],o.state_dict()) for k,o in optimizers.items()),saved_gradients=WM.raw_tree_equal(ck['gradient_snapshots'],gradients),frozen_normalizers=WM.raw_tree_equal(ck['normalizers'],s['norm']),loss_scales=WM.raw_tree_equal(ck['loss_scales'],s['scales']),physical_context=WM.raw_tree_equal(ck['physical_context'],s['physical']),sample_generator=WM.raw_tree_equal(ck['sample_generator'],s['generator'].get_state()),log_prefix=_log_prefix_valid(ck['log_prefix']) and all(_log_prefix_valid(v) for v in ck['parent_log_prefixes']),saved_RNG=WM.raw_tree_equal(ck['RNG'],rng),raw_RNG_unchanged=WM.raw_tree_equal(rng,WM.capture_rng(device=s['device'])),configuration_and_bindings=ck['configuration']==s['c'] and ck['provenance']==s['provenance'],counters=ck['counts']==counts and _counts_valid(counts,ck['update'],s['c']),all_Adam_steps=all(len(o.state)==(6 if k=='WM' else 8 if k.endswith('Actor') else 10) and all(float(v['step'])==(counts['WM_updates'] if k=='WM' else counts['paired_updates']) and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in o.state.values()) for k,o in optimizers.items()))
    if not all(checks.values()):raise ValueError('Checkpoint health failed; preserve files, no retry')
    return dict(schema='supermimic_research.offline_checkpoint_health:1',checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),counts=counts.copy(),extra_NN_FW=10,extra_NN_constructors=5,own_CPU_checkpoint_reads=1)

def train(config_path):
    path,c,r=configuration(config_path)
    if c['training']['authorized'] is not True:raise PermissionError('Training remains explicitly disabled in the consolidation config')
    out=(path.parent/c['training']['output']).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.perf_counter();phase='setup';counts=dict(WM_updates=0,paired_updates=0)
    def output(name,value):
        with (out/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
    try:
        s=_setup(path,smoke_only=False);torch=s['torch'];d=s['dataset'].canonical;model=s['model'];arms=s['arms'];norm=s['norm'];physical=s['physical'];eligible=s['dataset'].actor_indices(c['training']['horizon']);resume=None;start_update=1;parent_logs=[];current_sources=source_pins();current_inputs=_artifact_pins(r,c)
        if not len(eligible):raise ValueError('No live complete TRAIN starts')
        if c['training'].get('resume_artifact'):
            rp=r.resolve(c['training']['resume_artifact']);rh=json.loads(r.resolve(c['training']['resume_health']).read_text(encoding='utf8'))
            if rh['schema']!='supermimic_research.offline_checkpoint_health:1' or rh['checkpoint']!=file_pin(rp) or not rh['checks'] or not all(v is True for v in rh['checks'].values()):raise ValueError('Own positive unified-schema resume health')
            resume=torch.load(rp,map_location='cpu',weights_only=False)
            if resume['schema']!='supermimic_research.offline_checkpoint:1' or _science_config(resume['configuration'])!=_science_config(c):raise ValueError('Own scientific configuration must match; old frozen checkpoints are warmstart only')
            if not _counts_valid(resume['counts'],resume['update'],c) or rh['counts']!=resume['counts'] or resume['update']>=c['training']['updates']:raise ValueError('Positive incomplete own update cursor and exact counters required')
            if _content_bindings(resume['provenance']['sources'])!=_content_bindings(current_sources) or _content_bindings(resume['provenance']['inputs'])!=_content_bindings(current_inputs):raise ValueError('Resume implementation or selected task/geometry/data inputs changed; use an explicit new experiment warmstart or migration')
            for key,current in [('starting_checkpoint',s['starting_checkpoint']),('train_report',r.pin(c['artifacts']['tripod_dataset'])),('old_report',r.pin(c['artifacts']['old_original'])),('old_labels',r.pin(c['artifacts']['old_pair_labels']))]:
                if resume['provenance'][key]['sha256']!=current['sha256']:raise ValueError('Resume data/initial lineage changed')
            if not WM.raw_tree_equal(resume['normalizers'],norm) or not WM.raw_tree_equal(resume['physical_context'],physical):raise ValueError('Resume must retain frozen warmstart normalizers/physical context')
            model.load_state_dict(resume['shared_WM']);U.restore_arm_states(arms,resume['arms']);norm=WM.move_tensors(resume['normalizers'],s['device']);physical=WM.move_tensors(resume['physical_context'],s['device']);scales=WM.move_tensors(resume['loss_scales'],s['device']);counts=resume['counts'];start_update=resume['update']+1
            parent_logs=resume['parent_log_prefixes']+[resume['log_prefix']]
            if not all(_log_prefix_valid(v) for v in parent_logs):raise ValueError('Parent log byte prefixes must remain intact')
        else:scales=_mixed_scales(s['new'],s['old'])
        s['norm']=norm;s['physical']=physical;s['scales']=scales
        if resume is None and not s['policy_warmstart']:
            fit=s['dataset'].actor_indices(1);obs=torch.tensor(np.concatenate([d['obs222'][fit],d['next_obs222'][fit]]),dtype=torch.float32,device=s['device'])
            for a in arms.values():a['actor'].norm.update(obs);a['critic'].norm.update(obs);a['target_critic'].load_state_dict(a['critic'].state_dict())
        optimizers={'WM':U.make_optimizer(model,learning_rate=c['optimizer']['WM_lr'])};arm_opts={}
        for mode,a in arms.items():
            arm_opts[mode]={key:U.make_optimizer(a[key],learning_rate=c['optimizer'][key+'_lr']) for key in ['actor','critic']}
            optimizers[mode+'_Actor']=arm_opts[mode]['actor'];optimizers[mode+'_Critic']=arm_opts[mode]['critic']
        if resume is not None:
            for key,opt in optimizers.items():opt.load_state_dict(resume['optimizers'][key])
        gradients={};update=U.UpdateConfig(imagination_horizon=c['training']['horizon'],microbatch=c['training']['microbatch']);loss_config=WM.LossConfig();rewards={}
        with torch.no_grad():
            for mode,task in s['tasks'].items():
                rr=torch.full((len(d['row_id']),),float('nan'),dtype=torch.float64,device=s['device'])
                for start in range(0,len(eligible),8):
                    i=eligible[start:start+8];reward=task.rewards(torch.tensor(d['y58'][i],dtype=torch.float64,device=s['device']),task.unflatten_memory(torch.tensor(d['memory83'][i],dtype=torch.float64,device=s['device'])),torch.tensor(d['action20'][i],dtype=torch.float64,device=s['device']))
                    if not reward['objective_known'].all() or not torch.isfinite(reward['total']).all():raise ValueError('Known finite cached real rewards required')
                    rr[torch.tensor(i,device=s['device'])]=reward['total']
                rewards[mode]=rr
        real_bank=_real_bank(d,device=s['device'],torch=torch)
        from supermimic_research.runtime import structured,phone_task
        setup_s=time.perf_counter()-started;times=[];provenance=dict(config=file_pin(path),artifacts=file_pin(r.manifest),starting_checkpoint=s['starting_checkpoint'],train_report=r.pin(c['artifacts']['tripod_dataset']),old_report=r.pin(c['artifacts']['old_original']),old_labels=r.pin(c['artifacts']['old_pair_labels']),inputs=current_inputs,sources=current_sources,source_fingerprint={key:value['sha256'] for key,value in current_sources.items()});s['provenance']=provenance
        if resume is not None:s['generator'].set_state(resume['sample_generator']);WM.restore_rng(resume['RNG'],device=s['device'])
        for common in range(0 if resume is not None else c['training']['common_WM_updates']):
            step=U.supervised_WM_step(model,optimizers['WM'],_sample(s['new'],s['old'],s['generator'],c,torch),norm,scales,physical,loss_config=loss_config,codec=structured,phase='PRE_COMMON_WM_'+str(common+1),materialize_metrics=False);gradients['WM']=step['gradient_snapshot'];counts['WM_updates']+=1
        output('common_WM_counters.json',dict(counts=counts.copy(),no_updated_pt_before_first_paired_checkpoint=True))
        logpath=out/'updates.jsonl'
        with logpath.open('x',encoding='utf8') as log:
            for i in range(start_update,c['training']['updates']+1):
                phase='update_'+str(i);tick=time.perf_counter();step=U.supervised_WM_step(model,optimizers['WM'],_sample(s['new'],s['old'],s['generator'],c,torch),norm,scales,physical,codec=structured,phase='PRE_WM_'+str(counts['WM_updates']+1),materialize_metrics=False);gradients['WM']=step['gradient_snapshot'];counts['WM_updates']+=1
                real,ii,idx=_real_batch(real_bank,eligible,s['generator'],c['sampling']['Actor_batch'],device=s['device'],torch=torch)
                ac=U.matched_actor_critic_step(world_model=model,arms=arms,optimizers=arm_opts,real_batch=real,mode_rewards={m:v[idx] for m,v in rewards.items()},tasks=s['tasks'],geometry=s['geometry'],codec=structured,norms=norm,physical=physical,wm_step=phone_task.wm_step_current_only,config=update,phase='PRE_PAIRED_'+str(i),materialize_metrics=False)
                for mode,result in ac.items():gradients[mode+'_Actor']=result['gradient_snapshots']['actor'];gradients[mode+'_Critic']=result['gradient_snapshots']['critic']
                counts['paired_updates']=i
                metrics=U.materialize_metrics_tree(dict(WM_loss=step['loss'],modes={m:{k:v for k,v in a.items() if k in ['real_TD_loss','imagined_Actor_loss']} for m,a in ac.items()}))
                if s['device'].startswith('cuda:'):torch.cuda.synchronize(s['device'])
                seconds=time.perf_counter()-tick;times.append(seconds);row=dict(update=i,**metrics,update_seconds=seconds,common_real_row_ids=d['row_id'][ii].tolist());log.write(json.dumps(row)+'\n');log.flush()
                if i==c['training']['first_checkpoint'] or i==c['training']['updates']:
                    os.fsync(log.fileno());prefix=logpath.read_bytes();checkpoint=out/f'checkpoint_{i:04d}.pt';payload=dict(schema='supermimic_research.offline_checkpoint:1',update=i,counts=counts.copy(),provenance=provenance,configuration=c,shared_WM=model.state_dict(),arms=U.arm_states(arms),normalizers=norm,loss_scales=scales,physical_context=physical,optimizers={k:o.state_dict() for k,o in optimizers.items()},gradient_snapshots=gradients,RNG=WM.capture_rng(device=s['device']),sample_generator=s['generator'].get_state(),parent_log_prefixes=parent_logs,parent_checkpoint=file_pin(r.resolve(c['training']['resume_artifact'])) if resume is not None else None,log_prefix=dict(path=str(logpath),bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),records=i-start_update+1,range=[start_update,i]))
                    WM.atomic_checkpoint(checkpoint,payload);probeF=WM.normalize_features(s['new']['feature_F64'][:8],norm);probeO=real['obs'];health=_health(checkpoint,s,optimizers,gradients,counts,probeF,probeO);rate=float(np.mean(times[-min(25,len(times)):]));health.update(checkpoint=file_pin(checkpoint),setup_s=setup_s,measured_update_seconds=rate,remaining_updates=c['training']['updates']-i,ETA_learning_only_s=(c['training']['updates']-i)*rate,resume_supported='own_unified_schema_only')
                    output(f'health_{i:04d}.json',health)
                    if i==c['training']['first_checkpoint'] and c['training']['stop_after_first_healthy']:
                        output('result.json',dict(status='STOPPED_FIRST_HEALTHY_CHECKPOINT_WITH_ETA',checkpoint=file_pin(checkpoint),health=file_pin(out/f'health_{i:04d}.json'),target_updates=c['training']['updates'],completed_updates=i,resume_supported='own_unified_schema_only',policy_benefit_qualified=False));return
        output('result.json',dict(status='COMPLETED_CONFIGURED_OFFLINE_UPDATES_NOT_POLICY_QUALIFICATION',checkpoint=file_pin(checkpoint),health=file_pin(out/f'health_{i:04d}.json'),completed_updates=counts['paired_updates'],policy_benefit_qualified=False))
    except BaseException as e:output('failure.json',dict(phase=phase,error=repr(e),counts=counts,automatic_retry=False));raise
